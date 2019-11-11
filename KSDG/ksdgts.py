"""Time-steppers for solution of the Keller-Segel PDEs."""

import sys, os, traceback, gc
import numpy as np
from datetime import datetime
import h5py
import pickle
import dill
from petsc4py import PETSc
# As a general rule, I import only fenics classes by name.
import fenics as fe
from fenics import (UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh,
                    FiniteElement, MixedElement, VectorElement,
                    TestFunctions, FunctionAssigner,
                    Expression, FunctionSpace, VectorFunctionSpace,
                    TrialFunction, TestFunction, Function, Constant,
                    Measure, FacetNormal, CellDiameter, PETScVector,
                    PETScMatrix, HDF5File, XDMFFile, File, MeshEditor,
                    Mesh)
# from timeseries import TimeSeries
from mpi4py import MPI
MPIINT = MPI.INT64_T
from .ksdgdebug import log
from .ksdgmakesolver import makeKSDGSolver
from .ksdgtimeseries import KSDGTimeSeries
from .ksdggather import gather_mesh, dofremap

def logTS(*args, **kwargs):
    log(*args, system='TS', **kwargs)

def dumpTS(obj):
    for key in dir(obj):
        if key[0:2] != '__':
            try:
                logTS(key, getattr(obj, key))
            except:
                pass
    
def logFSINFO(*args, **kwargs):
    log(*args, system='FSINFO', **kwargs)


class KSDGTS(PETSc.TS):
    """Base class for KSDG timesteppers."""

    default_rollback_factor = 0.25
    default_hmin = 1e-20

    def __init__(
        self,
        ksdg,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        restart=True,
        tstype = PETSc.TS.Type.ROSW,
        finaltime = PETSc.TS.ExactFinalTime.STEPOVER,
        rollback_factor = None,
        hmin = None,
        comm = PETSc.COMM_WORLD
    ):
        """
        Create a timestepper

        Required positional argument:
        ksdg: the KSDGSolver in which the problem has been set up.

        Keyword arguments:
        t0=0.0: the initial time.
        dt=0.001: the initial time step.
        maxdt = 0.5: the maximum time step
        tmax=20: the final time.
        maxsteps=100: maximum number of steps to take.
        rtol=1e-5: relative error tolerance.
        atol=1e-5: absolute error tolerance.
        restart=True: whether to set the initial condition to rho0, U0
        tstype=PETSc.TS.Type.ROSW: implicit solver to use.
        finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
            final time step.
        fpe_factor=0.1: scale time step by this factor on floating
            point error.
        hmin=1e-20: minimum step size
        comm = PETSc.COMM_WORLD

        You can set other options by calling TS member functions
        directly. If you want to pay attention to PETSc command-line
        options, you should call ts.setFromOptions(). Call ts.solve()
        to run the timestepper, call ts.cleanup() to destroy its
        data when you're finished. 

        """
#        print("KSDGTS __init__ entered", flush=True)
        super().__init__()
        self.create()
#        self.create(comm=comm)
#        self._comm = comm
        self.mpi_comm = self.comm
        if (not isinstance(self.comm, type(MPI.COMM_SELF))):
            self.mpi_comm = self.comm.tompi4py()
        self.ksdg = ksdg
        popts = PETSc.Options()
        if rollback_factor is None:
            try:
                self.rollback_factor = popts.getReal(
                    'ts_adapt_scale_solve_failed')
            except KeyError:
                self.rollback_factor = self.default_rollback_factor
        else:
            self.rollback_factor = rollback_factor
        if hmin:
            self.hmin = hmin
        else:
            self.hmin = self.default_hmin
        self.setProblemType(self.ProblemType.NONLINEAR)
        self.setMaxSNESFailures(1)
        self.setTolerances(atol=atol, rtol=rtol)
        self.setType(tstype)
        logTS('KSDGTS __init__, calling setup_problem')
        try:
            self.ksdg.setup_problem(t0)
        except TypeError:
            self.ksdg.setup_problem()
        logTS('KSDGTS __init__, setup_problem returned')
        self.history = []
        J = fe.as_backend_type(self.ksdg.A).mat().duplicate()
        J.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        J.setUp()
        if restart: self.ksdg.restart()
        u = fe.as_backend_type(self.ksdg.sol.vector()).vec().duplicate()
        u.setUp()
        f = u.duplicate()
        f.setUp()
        self.params = dict(
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,    
            tstype = tstype,
            finaltime = finaltime,
            restart=restart,
            J = J,
            u = u,
            f = f
        )
        fe.as_backend_type(self.ksdg.sol.vector()).vec().copy(u)
        # logTS('init u.array', u.array)
        self.setSolution(u)
        self.setTime(t0)
        self.setTimeStep(dt)
        self.setMaxSteps(max_steps=maxsteps)
        self.setMaxTime(max_time=tmax)
        self.setExactFinalTime(finaltime)
        self.setMaxTime(tmax)
        self.setMaxSteps(maxsteps)
        fe.parameters['krylov_solver']['nonzero_initial_guess'] = True

    def solve(self, u=None):
        """Run the timestepper.

        Calls ts.setFromOptions before solving. Ideally this would be
        left to the user's control, but solve crashes without.
        """
        if u:
            u.copy(self.params['u'])
        u = self.params['u']
        t0 = self.params['t0']
        self.setSolution(u)
        # logTS('solve u.array', u.array)
        self.setTime(t0)
        self.setFromOptions()
        tmax = self.getMaxTime()
        kmax = self.getMaxSteps()
        k = self.getStepNumber()
        h = self.getTimeStep()
        t = self.getTime()
        lastu = u.duplicate()
        lastu.setUp()
        self.monitor(k, t, u)
        while (
                (not self.diverged) and
                k <= kmax and t <= tmax and
                h >= self.hmin
        ):
            self.rollback = False
            try:
                lastk, lasth, lastt = k, h, t
                u.copy(lastu)
                lastu.assemble()
                super().step()
                gc.collect()
                k = self.getStepNumber()
                h = self.getTimeStep()
                t = self.getTime()
                u = self.getSolution()
            except (FloatingPointError, ValueError) as e:
                tbstr = traceback.format_exc()
                logTS('ts.step() threw Error:', str(type(e)), str(e), tbstr)
#                self.rollback = True
            except (PETSc.Error) as e:
                tbstr = traceback.format_exc()
                logTS('ts.step() threw Error:', str(type(e)), str(e), tbstr)
#                self.rollback = True
                raise
            #
            # Not sure if this is the right thing to do:
            # Force BCs on solution vector.
            #
            if hasattr(self.ksdg, 'bcs'):
                for bc in self.ksdg.bcs:
                    bc.apply(self.ksdg.sol.vector())
                self.ksdg.sol.vector().apply('insert')
            try:
                solvec = self.ksdg.sol.vector().array()
            except AttributeError:
                solvec = self.ksdg.sol.vector()[:]
            if not np.min(solvec) > 0.0:   # also catches nan
                # skip this check, since can get negatives with periodic BCs
                pass
#                self.rollback = True
#                logTS('min, max sol', np.min(solvec), np.max(solvec))
            #
            # The rollback code has effectively been disabled by
            # commenting out all the lines that might set
            # self.rollback. Rolling back doesn't really work any
            # better than PETSc basic TSAdapt.
            #
            rollback = self.mpi_comm.allreduce(self.rollback, MPI.LOR)
            if rollback:
                logTS('rolling back')
                lasth *= self.rollback_factor
                lasth = self.mpi_comm.allreduce(lasth, MPI.MIN)
                h = lasth
                k = lastk
                t = lastt
                self.setTimeStep(h)
                self.setTime(t)
                self.setStepNumber(k)
                lastu.copy(u)
                u.assemble()
                self.setSolution(u)
                logTS('np.min(u.array)', np.min(u.array))
                logTS('np.min(lastu.array)', np.min(lastu.array))
            else:
                self.monitor(k, t, u)

    def cleanup(self):
        """Should be called when finished with a TS

        Leaves history unchanged.
        """
        J = self.params['J']
        u = self.params['u']
        f = self.params['f']
        del J, u, f, self.params

    def printMonitor(self, ts, k, t, u):
        """For use as TS monitor. Prints status of solution."""
#        print("printMonitor", flush=True)
        if self.comm.rank == 0:
            h = ts.getTimeStep()
            if hasattr(self, 'lastt'):
                dt = t - self.lastt
                out = "clock: %s, step %3d t=%8.3g dt=%8.3g h=%8.3g" % (
                          datetime.now().strftime('%H:%M:%S'), k, t, dt, h
                      )
            else:
                out = "clock: %s, step %3d t=%8.3g h=%8.3g" % (
                          datetime.now().strftime('%H:%M:%S'), k, t, h
                       )
            print(out, flush=True)
            self.lastt = t

    def historyMonitor(self, ts, k, t, u):
        """For use as TS monitor. Stores results in history"""
#        logTS("historyMonitor", flush=True)
        h = ts.getTimeStep()
        if not hasattr(self, 'history'):
            self.history = []
        #
        # make a local copy of the dof vector
        #
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        lu = self.ksdg.sol.vector().gather_on_zero()
        self.history.append(dict(
            step = k,
            h = h,
            t = t,
            u = lu.array.copy()
        ))

    def checkpointMonitor(self, ts, k, t, u, prefix):
        """For use as TS monitor. Checkpoints results"""
        h = ts.getTimeStep()
        #
        # make a local copy of the dof vector
        #
        if not hasattr(self, 'remap'):
            self.lmesh = gather_mesh(self.ksdg.mesh)
            logTS('checkpointMonitor: self.lmesh', self.lmesh)
            self.remap = dofremap(self.ksdg)
            logTS('checkpointMonitor: self.remap', self.remap)
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        lu = self.ksdg.sol.vector().gather_on_zero()
        cpname = prefix + '_' + str(k) + '.h5'
        if self.comm.rank == 0:
            lu = lu[self.remap]
            cpf = KSDGTimeSeries(cpname, 'w')
            cpf.store(lu, t, k=k)
            cpf.close()

    def makeSaveMonitor(self, prefix=None):
        """Make a saveMonitor for use as a TS monitor

        Note that makesaveMonitor is not itself a monitor
        function. It returns a callable that can be used as a save
        monitor. Typical usage:

        (saveMonitor, closer) = ts.makeSaveMonitor(prefix='solution')
        ts.setMonitor(save_Monitor)
        ts.solve()
        ...
        closer()

        Optional keyword argument:
        prefix=None: The filename prefix to use for the saved
            files (may begin with a file path. If not provided,
            the prefix is "solution' prepended to a string
            produced by the uuid function uuid4.
        """
        if not prefix:
            prefix = 'solution' + str(uuid4()) + '_'
        #
        # save basic info about FunctionSpace
        #
        fsname = prefix + 'rank' + str(self.comm.rank) + '_fsinfo.h5'
        if not hasattr(self, 'remap'):
            self.lmesh = gather_mesh(self.ksdg.mesh)
            logTS('makeSaveMonitor: self.lmesh', self.lmesh)
            self.remap = dofremap(self.ksdg, gathered_mesh=self.lmesh)
            logTS('makeSaveMonitor: self.remap', self.remap)
        if hasattr(self.ksdg, 'omesh'):
            self.omesh = gather_mesh(self.ksdg.omesh)
        if self.comm.rank == 0:
            logTS('self.lmesh.mpi_comm().size', self.lmesh.mpi_comm().size)
            lmeshf = File(MPI.COMM_SELF, prefix + '_mesh.xml.gz')
            lmeshf << self.lmesh
            logTS('self.lmesh saved')
            if hasattr(self, 'omesh'):
                logTS('self.omesh.mpi_comm().size', self.omesh.mpi_comm().size)
                omeshf = File(MPI.COMM_SELF, prefix + '_omesh.xml.gz')
                omeshf << self.omesh
                logTS('self.omesh saved')
        try:
            scomm = fe.mpi_comm_self()
        except AttributeError:
            scomm = MPI.COMM_SELF
        fsf = HDF5File(scomm, fsname, 'w')
        fs = self.ksdg.sol.function_space()
        fsf.write(self.ksdg.sol, 'sol')
        fsf.write(self.ksdg.mesh, 'mesh')
        if self.comm.rank == 0:
            fsf.write(self.lmesh, 'lmesh')
        fsf.close()
        fsf = h5py.File(fsname, 'r+')
        fsf['/mpi/rank'] = self.comm.rank
        fsf['/mpi/size'] = self.comm.size
        if self.comm.rank == 0:
            fsf['remap'] = self.remap
        if hasattr(self.ksdg, 'srhos'):
            rhofss = [rho.function_space() for rho in self.ksdg.srhos]
        else:
            rhofss = [self.ksdg.srho.function_space()]
        if hasattr(self.ksdg, 'sUs'):
            Ufss = [U.function_space() for U in self.ksdg.sUs]
        else:
            Ufss = [self.ksdg.sU.function_space()]
        dofmap = fs.dofmap()
        fsinfo = {}
        fsf['degree'] = self.ksdg.degree
        if hasattr(self.ksdg, 'nelements'):
            fsf['nelements'] = self.ksdg.nelements
        if hasattr(self.ksdg, 'periodic'):
            fsf['periodic'] = self.ksdg.periodic
        else:
            fsf['periodic'] = False
        if hasattr(self.ksdg, 'ligands'):
            fsf['ligands'] = pickle.dumps(self.ksdg.ligands, protocol=0)
        else:
            fsf['ligands'] = False
        if hasattr(self.ksdg, 'param_names'):
            fsf['param_names'] = pickle.dumps(self.ksdg.param_names, protocol=0)
        else:
            fsf['param_names'] = pickle.dumps([], protocol=0)
        if hasattr(self.ksdg, 'param_funcs'):
            try:
                dillfunc = dill.dumps(self.ksdg.param_funcs, protocol=0)
                fsf['param_funcs'] = np.void(dillfunc)
                #
                # retrieve using fsf['param_funcs'].tobytes()
                #
            except ValueError as e:
                traceback.print_exc()
                fsf['param_funcs'] = dill.dumps([], protocol=0)
        else:
            fsf['param_funcs'] = dill.dumps([], protocol=0)
        if hasattr(self.ksdg, 'params0'):
            fsf['params0'] = pickle.dumps(self.ksdg.params0, protocol=0)
        else:
            fsf['params0'] = pickle.dumps({}, protocol=0)
        dim = self.ksdg.dim
        fsf['dim'] = dim
        try:
            fsf['ghost_mode'] = self.ksdg.mesh.ghost_mode()
        except AttributeError:
            pass
        fsf['off_process_owner'] = dofmap.off_process_owner()
        owneddofs = dofmap.ownership_range()
        fsf['ownership_range'] = owneddofs
        logFSINFO('owneddofs', owneddofs)
        ltgu = np.zeros(len(dofmap.local_to_global_unowned()), dtype=int)
        ltgu[:] = dofmap.local_to_global_unowned()
        fsf['local_to_global_unowned'] = ltgu
        ltgi = np.zeros(owneddofs[1] - owneddofs[0], dtype=int)
        logFSINFO('np.array(dofmap.dofs())', np.array(dofmap.dofs()))
        logFSINFO('dofmap local_to_global_indexes', *owneddofs,
                  np.array([dofmap.local_to_global_index(d) for d in range(*owneddofs)]))
        logFSINFO('dofmap local_to_global_indexes', 0, owneddofs[1]-owneddofs[0],
                  np.array([dofmap.local_to_global_index(d) for d in range(owneddofs[1]-owneddofs[0])]))
        ltgi = np.array(
            np.array([dofmap.local_to_global_index(d) for d in range(*owneddofs)])
        )
        fsf['local_to_global_index'] = ltgi
        fsf['tabulate_local_to_global_dofs'] = dofmap.tabulate_local_to_global_dofs()
        logFSINFO('tabulate_local_to_global_dofs', dofmap.tabulate_local_to_global_dofs())
#        fsf['shared_nodes'] = dofmap.shared_nodes()
        try:
            fsf['neighbours'] = repr(dofmap.neighbours())
        except (AttributeError, TypeError):
            pass
        try:
            fsf['dofmap_str'] = dofmap.str(True)
        except (AttributeError, TypeError):
            fsf['dofmap_str'] = repr(dofmap)
        try:
            fsf['dofmap_id'] = dofmap.id()
        except (AttributeError, TypeError):
            pass
        try:
            fsf['dofmap_label'] = dofmap.label()
        except (AttributeError, TypeError):
            pass
        try:
            fsf['dofmap_name'] = dofmap.name()
        except (AttributeError, TypeError):
            pass
        try:
            fsf['max_cell_dimension'] = dofmap.max_cell_dimension()
        except (AttributeError, TypeError):
            pass
        try:
            fsf['max_element_dofs'] = dofmap.max_element_dofs()
        except (AttributeError, TypeError):
            pass
        try:
            fsf['dofmap_parameters'] = list(dofmap.parameters.iteritems())
        except (AttributeError, TypeError):
            try:
                fsf['dofmap_parameters'] = list(dofmap.parameters.items())
            except (AttributeError, TypeError):
                pass
        try:
            fsf['dofmap_thisown'] = dofmap.thisown
        except (AttributeError, TypeError):
            pass
        try:
            fsf['is_view'] = dofmap.is_view()
        except (AttributeError, TypeError):
            pass
        for d in range(dim+1):
            fsf['entity_closure_dofs' + str(d)] = \
              dofmap.entity_closure_dofs(self.ksdg.mesh, d)
            fsf['entity_dofs' + str(d)] = \
              dofmap.entity_dofs(self.ksdg.mesh, d)
        fsf['dofcoords'] = np.reshape(
            fs.tabulate_dof_coordinates(), (-1, dim)
        )
        try:
            fsf['block_size'] = dofmap.block_size()
        except (AttributeError, TypeError):
            pass
        try:
            fsf['dofs'] = dofmap.dofs()
        except (AttributeError, TypeError):
            pass
        for i in range(len(rhofss)):
            fsf['rho_dofs/'+str(i)] = rhofss[i].dofmap().dofs()
        for i in range(len(Ufss)):
            fsf['U_dofs/'+str(i)] = Ufss[i].dofmap().dofs()
        dofspercell = dofmap.cell_dofs(0).size
        ncells = self.ksdg.mesh.cells().shape[0]
        celldofs = np.zeros((ncells, dofspercell), dtype=np.int)
        for cell in range(ncells):
            celldofs[cell] = dofmap.cell_dofs(cell)
        fsf['cell_dofs'] = celldofs
        fsf.close()
        logTS('creating KSDGTimeSeries')
        tsname = prefix + '_ts.h5'
        if self.comm.rank == 0:
            tsf = KSDGTimeSeries(tsname, 'w')
            tsf.close()

        def closeSaveMonitor():
            if self.comm.rank == 0:
                tsf.close()
        

        def saveMonitor(ts, k, t, u):
            #
            # make a local copy of the dof vector
            #
#            logTS('saveMonitor entered')
            psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
            u.copy(psol)
            self.ksdg.sol.vector().apply('insert')
            lu = self.ksdg.sol.vector().gather_on_zero()
            if ts.comm.rank == 0:
                lu = lu[self.remap]
                #
                # reopen and close every time, so file valid after abort
                #
                tsf = KSDGTimeSeries(tsname, 'r+')
                tsf.store(lu, t, k=k)
                tsf.close()

        return (saveMonitor, closeSaveMonitor)


class implicitTS(KSDGTS):
    """Fully implicit timestepper."""
    
    def __init__(
        self,
        ksdg,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        restart=True,
        tstype = PETSc.TS.Type.ROSW,
        finaltime = PETSc.TS.ExactFinalTime.STEPOVER,
        comm = PETSc.COMM_WORLD
    ):
        """
        Create an implicit timestepper
        
        Required positional argument:
        ksdg: the KSDGSolver in which the problem has been set up.

        Keyword arguments:
        t0=0.0: the initial time.
        dt=0.001: the initial time step.
        tmax=20: the final time.
        maxsteps=100: maximum number of steps to take.
        rtol=1e-5: relative error tolerance.
        atol=1e-5: absolute error tolerance.
        restart=True: whether to set the initial condition to rho0, U0
        tstype=PETSc.TS.Type.ROSW: implicit solver to use.
        finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
            final time step.
        comm = PETSc.COMM_WORLD.

        Other options can be set by modifying the PETSc Options
        database.
        """
        logTS("implicitTS __init__ entered")
        super().__init__(
            ksdg,
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,
            restart=restart,
            tstype = tstype,
            finaltime = finaltime,
            comm = comm
        )
        f = self.params['f']
        J = self.params['J']
        self.setEquationType(self.EquationType.IMPLICIT)
        self.setIFunction(self.implicitIF, f=f)
        self.setIJacobian(self.implicitIJ, J=J, P=J)

    def implicitIF(self, ts, t, u, udot, f):
        """Fully implicit IFunction for PETSc TS

        This is designed for use as the LHS in a fully implicit PETSc
        time-stepper. The DE to be solved is always A.u' = b(u). u
        corresponds to self.ksdg.sol. A is a constant (i.e., u-independent)
        matrix calculated by assembling the u'-dependent term of the
        weak form, and b a u-dependent vector calculated by assembling
        the remaining terms. This equation may be solved in any of
        three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: implicitIF and implicitIJ calculate the
        LHS A.u' - b and its Jacobian for the fully implicit
        form. Arguments: 

        t: the time
        u: a petsc4py.PETSc.Vec containing the state vector
        udot: a petsc4py.PETSc.Vec containing the time derivative u'
        f: a petsc4py.PETSc.Vec in which A.u' - b will be left.
        """
        try:
#            if self.rollback:
#                raise ValueError('rolling back')
            logTS('implicitIF entered')
            try:
                self.ksdg.setup_problem(t)
            except TypeError:
                self.ksdg.setup_problem()
            # logTS('self.ksdg.A.array()', self.ksdg.A.array())
            pA = fe.as_backend_type(self.ksdg.A).mat()
            # logTS('u.array', u.array)
            psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
            u.copy(psol)
            # logTS('psol.array', psol.array)
            self.ksdg.sol.vector().apply('insert')
            self.ksdg.b = fe.assemble(self.ksdg.all_terms)
            # logTS('self.ksdg.b[:]', self.ksdg.b[:])
            if hasattr(self.ksdg, 'bcs'):
                for bc in self.ksdg.bcs:
                    bc.apply(self.ksdg.b)
                self.ksdg.b.apply('insert')
            pb = fe.as_backend_type(self.ksdg.b).vec()
            # logTS('pb.array', pb.array)
            pb.copy(f)
            f.scale(-1.0)
            pA.multAdd(udot, f, f)
            f.assemble()
            logTS('implicitIF, f assembled')
#            if not np.min(u.array) > 0:
#                logTS('np.min(u.array)', np.min(u.array))
#                raise(ValueError('nonpositive function values'))
            # logTS('udot.array', udot.array)
            if np.isnan(np.min(udot.array)):
                logTS('np.min(udot.array)', np.min(udot.array))
#                raise(ValueError('nans in udot'))
            # logTS('f.array', f.array)
            if np.isnan(np.min(f.array)):
                logTS('np.min(f.array)', np.min(f.array))
#                raise(ValueError('nans in computed f'))
        except (ValueError, FloatingPointError) as e:
            logTS('implicitIF got FloatingPointError', str(type(e)), str(e))
            einfo = sys.exc_info()
            tbstr = traceback.format_exc()
            logTS('traceback', tbstr)
#            self.rollback = True          # signal trouble upstairs
            f.set(0.0)
            f.assemble()
        except Exception as e:
            logTS('implicitIF got exception', str(type(e)), str(e))
            tbstr = traceback.format_exc()
            logTS('Exception:', tbstr)
            logTS('str(ts)', str(self))
            einfo = sys.exc_info()
            raise einfo[1].with_traceback(einfo[2])

    def implicitIJ(self, ts, t, u, udot, shift, J, B):
        """Fully implicit IJacobian for PETSc TS

        This is designed for use as the LHS in a fully implicit PETSc
        time-stepper. The DE to be solved is always A.u' = b(u). u
        corresponds to self.ksdg.sol. A is a constant (i.e., u-independent)
        matrix calculated by assembling the u'-dependent term of the
        weak form, and b a u-dependent vector calculated by assembling
        the remaining terms. This equation may be solved in any of
        three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: implicitIF and implicitIJ calculate the
        LHS A.u' - b and its Jacobian for the fully implicit
        form. Arguments:

        t: the time.
        u: a petsc4py.PETSc.Vec containing the state vector
        udot: a petsc4py.PETSc.Vec containing the u' (not used)
        shift: a real number -- see PETSc Ts documentation.
        J, B: matrices in which the Jacobian shift*A - Jacobian(b(u)) are left. 
        """
        logTS('implicitIJ entered')
        try:
            try:
                self.ksdg.setup_problem(t)
            except TypeError:
                self.ksdg.setup_problem()
            pA = fe.as_backend_type(self.ksdg.A).mat()
            Adiag = pA.getDiagonal()
            logTS('Adiag.array', Adiag.array)
            psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
            u.copy(psol)
            self.ksdg.sol.vector().apply('insert')
            # if isinstance(self.ksdg.JU_terms, list):
            #     JUs = [fe.assemble(JU_term)
            #            for JU_term in self.ksdg.JU_terms]
            # else:
            #     JUs = [fe.assemble(self.ksdg.JU_terms)]
            # pJUs = [fe.as_backend_type(JU).mat()
            #         for JU in JUs]
            # if isinstance(self.ksdg.Jrho_terms, list):
            #     # Jrhos = [fe.assemble(Jrho_term,
            #     #          form_compiler_parameters = {"quadrature_degree": 3})
            #     #        for Jrho_term in self.ksdg.Jrho_terms]
            #     Jrhos = [fe.assemble(Jrho_term)
            #            for Jrho_term in self.ksdg.Jrho_terms]
            # else:
            #     Jrhos = [fe.assemble(self.ksdg.Jrho_terms)]
            # pJrhos = [fe.as_backend_type(Jrho).mat()
            #           for Jrho in Jrhos]
            Jsol = fe.assemble(self.ksdg.J_terms)
            #
            # Don't know if this is really what I should be doing:
            #
            for bc in self.ksdg.bcs:
                bc.apply(Jsol)
            pJsol = fe.as_backend_type(Jsol).mat()
            Jdiag = pJsol.getDiagonal()
            logTS('Jdiag.array', Jdiag.array)
            logTS('Adiag.array/Jdiag.array', Adiag.array/Jdiag.array)
            logTS('shift', shift)
            pA.copy(J)
            J.scale(shift)
            J.axpy(-1.0, pJsol)
            J.assemble()
            logTS('implicitIJ, J assembled')
            if J != B:
                J.copy(B)
                B.assemble()
        except FloatingPointError as e:
            logTS('implicitIF got FloatingPointError', str(type(e), str(e)))
            einfo = sys.exc_info()
            sys.excepthook(*einfo)
        except Exception as e:
            logTS('implicitIJ got exception', str(type(e)), str(e))
            tbstr = traceback.format_exc()
            logTS('Exception:', tbstr)
            logTS('str(ts)', str(self))
            einfo = sys.exc_info()
            raise einfo[1].with_traceback(einfo[2])
        return True

class imExTS(KSDGTS):
    """Implicit/Explicit timstepper."""
    
    def __init__(
        self,
        ksdg,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        restart=True,
        tstype = PETSc.TS.Type.ARKIMEX,
        finaltime = PETSc.TS.ExactFinalTime.STEPOVER,
        comm = PETSc.COMM_WORLD
    ):
        """
        Create an implicit/explicit timestepper

        Required positional argument:
        ksdg: the KSDGSolver in which the problem has been set up.

        Keyword arguments:
        t0=0.0: the initial time.
        dt=0.001: the initial time step.
        tmax=20: the final time.
        maxsteps=100: maximum number of steps to take.
        rtol=1e-5: relative error tolerance.
        atol=1e-5: absolute error tolerance.
        restart=True: whether to set the initial condition to rho0, U0
        tstype=PETSc.TS.Type.ARKIMEX: implicit solver to use.
        finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
            final time step.
        comm = PETSc.COMM_WORLD

        Other options can be set by modifying the PETSc options
        database.
        """
        super().__init__(
            ksdg,
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,
            restart=restart,
            tstype = tstype,
            finaltime = finaltime,
            comm = comm
        )
        f = self.params['f']
        J = self.params['J']
        self.setEquationType(self.EquationType.EXPLICIT)
        self.setIFunction(self.imExIF, f=f)
        self.setIJacobian(self.imExIJ, J=J, P=J)
        self.setRHSFunction(self.imExRHSF, f=f)
        self.setRHSJacobian(self.imExRHSJ, J=J, P=J)
            
    def imExIF(self, ts, t, u, udot, f):
        """Implicit/Explicit IFunction for PETSc TS

        This is designed for use as the LHS in an implicit/explicit
        PETSc time-stepper. The DE to be solved is always A.u' =
        b(u). u corresponds to self.sol. A is a constant (i.e.,
        u-independent) matrix calculated by assembling the
        u'-dependent term of the weak form, and b a u-dependent vector
        calculated by assembling the remaining terms. This equation
        may be solved in any of three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: imExIF, imExIJ, imExRHSF, and imExRHSJ
        calculate the LHS function and Jacobian and the RHS function
        and Jacobian for the implicit/explicit case. Arguments:

        t: the time.
        u: a petsc4py.PETSc.Vec containing the state vector (not used).
        udot: a petsc4py.PETSc.Vec containing the time derivative u'
        f: a petsc4py.PETSc.Vec in which A.u' will be left.
        """
        try:
            self.ksdg.setup_problem(t)
        except TypeError:
            self.ksdg.setup_problem()
        pA = fe.as_backend_type(self.ksdg.A).mat()
        pA.mult(udot, f)
        f.assemble()

    def imExIJ(self, ts, t, u, udot, shift, J, B):
        """Implicit/Explicit IJacobian for PETSc TS

        This is designed for use as the LHS in an implicit/explicit
        PETSc time-stepper. The DE to be solved is always A.u' =
        b(u). u corresponds to self.sol. A is a constant (i.e.,
        u-independent) matrix calculated by assembling the
        u'-dependent term of the weak form, and b a u-dependent vector
        calculated by assembling the remaining terms. This equation
        may be solved in any of three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: imExIF, imExIJ, imExRHSF, and imExRHSJ
        calculate the LHS function and Jacobian and the RHS function
        and Jacobian. for the implicit/explicit case. Arguments:

        t: the time.
        u: a petsc4py.PETSc.Vec containing the state vector (not used).
        udot: a petsc4py.PETSc.Vec containing u' (not used)
        shift: see PETSc Ts documentation.
        J, B: petsc4py.PETSc.Mat's in which shift*A will be left.
        """
        try:
            self.ksdg.setup_problem(t)
        except TypeError:
            self.ksdg.setup_problem()
        pA = fe.as_backend_type(self.ksdg.A).mat()
        pA.copy(J)
        J.scale(shift)
        J.assemble()
        if J != B:
            J.copy(B)
            B.assemble()
        return True

    def imExRHSF(self, ts, t, u, g, *args, **kwargs):
        """Implicit/Explicit RHSFunction for PETSc TS

        This is designed for use as the RHS in an implicit/explicit
        PETSc time-stepper. The DE to be solved is always A.u' =
        b(u). u corresponds to self.ksdg.sol. A is a constant (i.e.,
        u-independent) matrix calculated by assembling the
        u'-dependent term of the weak form, and b a u-dependent vector
        calculated by assembling the remaining terms. This equation
        may be solved in any of three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: imExIF, imExIJ, imExRHSF, and imExRHSJ
        calculate the LHS function and Jacobian and the RHS function
        and Jacobian for the implicit/explicit case. Arguments:

        t: the time.
        u: a petsc4py.PETSc.Vec containing the state vector.
        g: a petsc4py.PETSc.Vec in which b will be left.
        """
        #
        # PETSc apparently calls RHSF when it shoud call RHSJ. We
        # detect this by the extra argument and reroute the call.
        #
        if (args):
            return self.imExRHSJ(t, u, g, *args, **kwargs)
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        try:
            self.ksdg.setup_problem(t)
        except TypeError:
            self.ksdg.setup_problem()
        self.ksdg.b = fe.assemble(self.ksdg.all_terms)
        if hasattr(self.ksdg, 'bcs'):
            for bc in self.ksdg.bcs:
                bc.apply(self.ksdg.b)
        pb = fe.as_backend_type(self.ksdg.b).vec()
        pb.copy(g)
        g.assemble()

    def imExRHSJ(self, ts, t, u, J, B):
        """Implicit/Explicit RHSFunction for PETSc TS

        This is designed for use as the RHS in an implicit/explicit
        PETSc time-stepper. The DE to be solved is always A.u' =
        b(u). u corresponds to self.sol. A is a constant (i.e.,
        u-independent) matrix calculated by assembling the
        u'-dependent term of the weak form, and b a u-dependent vector
        calculated by assembling the remaining terms. This equation
        may be solved in any of three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: imExIF, imExIJ, imExRHSF, and imExRHSJ
        calculate the LHS function and Jacobian and the RHS function
        and Jacobian for the implicit/explicit case. Arguments:

        t: the time.
        u: a petsc4py.PETSc.Vec containing the state vector
        J, B: matrices in which the Jacobian of b(u) is left. 
        """
        try:
            self.ksdg.setup_problem(t)
        except TypeError:
            self.ksdg.setup_problem()
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        Jsol = fe.assemble(self.ksdg.J_terms)
        pJsol = fe.as_backend_type(Jsol).mat()
        pA.copy(J)
        J.scale(shift)
        J.axpy(1.0, pJsol)
        pJsol.copy(J)
        J.assemble()
        if J != B:
            J.copy(B)
            B.assemble()
        return True


class explicitTS(KSDGTS):
    """Explicit timstepper."""
    
    def __init__(
        self,
        ksdg,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        restart=True,
        tstype = PETSc.TS.Type.RK,
        finaltime = PETSc.TS.ExactFinalTime.STEPOVER,
        comm = PETSc.COMM_WORLD
    ):
        """
        Create an explicit timestepper and solve the DE

        Keyword arguments:
        t0=0.0: the initial time.
        dt=0.001: the initial time step.
        tmax=20: the final time.
        maxsteps=100: maximum number of steps to take.
        rtol=1e-5: relative error tolerance.
        atol=1e-5: absolute error tolerance.
        restart=True: whether to set the initial condition to rho0, U0
        tstype=PETSc.TS.Type.RK: explicit solver to use.
        finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
            final time step.
        comm = PETSc.COMM_WORLD

        Other options can be set by modifying the PETSc options
        database.
        """
        super().__init__(
            ksdg,
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,
            restart=restart,
            tstype = tstype,
            finaltime = finaltime,
            comm = comm
        )
        f = self.params['f']
        self.setEquationType(self.EquationType.EXPLICIT)
        self.setRHSFunction(self.explicitRHSF, f=f)

    def explicitRHSF(self, ts, t, u, g):
        """Explicit RHSFunction for PETSc TS

        This is designed for use as the RHS in an explicit PETSc
        time-stepper. The DE to be solved is always A.u' = b(u). u
        corresponds to self.ksdg.sol. A is a constant (i.e., u-independent)
        matrix calculated by assembling the u'-dependent term of the
        weak form, and b a u-dependent vector calculated by assembling
        the remaining terms. This equation may be solved in any of
        three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: explicitRHSF calculates
        the RHS function for the explicit case. Arguments:

        t: the time.
        u: a petsc4py.PETSc.Vec containing the state vector.
        g: a petsc4py.PETSc.Vec in which A^(-1).b(u) will be left.
        """
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        try:
            self.ksdg.setup_problem(t)
        except TypeError:
            self.ksdg.setup_problem()
        self.ksdg.b = fe.assemble(self.ksdg.all_terms)
        if hasattr(self.ksdg, 'bcs'):
            for bc in self.ksdg.bcs:
                bc.apply(self.ksdg.b)
        ret = self.ksdg.solver.solve(self.ksdg.dsol.vector(), self.ksdg.b)
        pdu = fe.as_backend_type(self.ksdg.dsol.vector()).vec()
        pdu.copy(g)
        g.assemble()

