"""Time-steppers for solution of the Keller-Segel PDEs."""

import sys, os, traceback
import numpy as np
from datetime import datetime
import h5py
from petsc4py import PETSc
import ufl
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
from KSDG.ksdgdebug import log
from KSDG.KSDG import KSDGSolver
from .ksdgtimeseries import KSDGTimeSeries
from .ksdggather import gather_mesh

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
        self.mpi_comm = mesh.comm
        if (not isinstance(self.comm, type(MPI.COMM_SELF))):
            self.mpi_comm = self.comm.tompi4py()
        self.ksdg = ksdg
        popts = PETSc.Options()
        if rollback_factor is None:
            try:
                self.rollback_factor = popts.getReal(
                    'ts_adapt_scale_solve_failed')
            except:
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
        self.ksdg.setup_problem()
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
        self.setSolution(u)
        self.setTime(t0)
        self.setInitialTimeStep(t0, dt)
        self.setDuration(max_time=tmax, max_steps=maxsteps)
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
        self.setTime(t0)
        self.setFromOptions()
        tmax, kmax = self.getDuration()
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
            solvec = self.ksdg.sol.vector().array()
            if not np.min(solvec) > 0.0:   # also catches nan
#                self.rollback = True
                logTS('min, max sol', np.min(solvec), np.max(solvec))
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
#        logTS("historyMonitor", flush=True)
        h = ts.getTimeStep()
        #
        # make a local copy of the dof vector
        #
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        lu = self.ksdg.sol.vector().gather_on_zero()
        cpname = prefix + '_' + str(k) + '.h5'
        if self.comm.rank == 0:
            cpf = KSDGTimeSeries(cpname, 'w')
            cpf.store(lu, t, k=k)
            cpf.close()

    def makeSaveMonitor(self, prefix=None):
        """Make a saveMonitor for use as a TS monitor

        Note that make saveMonitor is not itself a monitor
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
        lmesh = gather_mesh(self.ksdg.mesh)
        if self.comm.rank == 0:
            lmeshf = File(prefix + '_mesh.xml.gz')
            lmeshf << lmesh
        try:
            scomm = fe.mpi_comm_self()
        except AttributeError:
            scomm = MPI.COMM_SELF
        fsf = HDF5File(scomm, fsname, 'w')
        fs = self.ksdg.sol.function_space()
        fsf.write(self.ksdg.sol, 'sol')
        fsf.write(self.ksdg.mesh, 'mesh')
        if self.comm.rank == 0:
            fsf.write(lmesh, 'lmesh')
        fsf.close()
        fsf = h5py.File(fsname, 'r+')
        fsf['/mpi/rank'] = self.comm.rank
        fsf['/mpi/size'] = self.comm.size
        rhofs = self.ksdg.rho.function_space()
        Ufs = self.ksdg.U.function_space()
        dofmap = fs.dofmap()
        fsinfo = {}
        fsf['degree'] = self.ksdg.degree
        if hasattr(self.ksdg, 'nelements'):
            fsf['nelements'] = self.ksdg.nelements
        dim = self.ksdg.dim
        fsf['dim'] = dim
        fsf['ghost_mode'] = self.ksdg.mesh.ghost_mode()
        fsf['off_process_owner'] = dofmap.off_process_owner()
        owneddofs = dofmap.ownership_range()
        fsf['ownership_range'] = owneddofs
        logFSINFO('owneddofs', owneddofs)
        ltgu = np.zeros(len(dofmap.local_to_global_unowned()), dtype=int)
        ltgu[:] = dofmap.local_to_global_unowned()
        fsf['local_to_global_unowned'] = ltgu
        ltgi = np.zeros(owneddofs[1] - owneddofs[0], dtype=int)
        logFSINFO('dofmap.dofs()', dofmap.dofs())
        logFSINFO('dofmap local_to_global_indexes', *owneddofs,
                  [dofmap.local_to_global_index(d) for d in range(*owneddofs)])
        logFSINFO('dofmap local_to_global_indexes', 0, owneddofs[1]-owneddofs[0],
                  [dofmap.local_to_global_index(d) for d in range(owneddofs[1]-owneddofs[0])])
        ltgi = np.array(
            [dofmap.local_to_global_index(d) for d in range(*owneddofs)]
        )
        fsf['local_to_global_index'] = ltgi
        fsf['tabulate_local_to_global_dofs'] = dofmap.tabulate_local_to_global_dofs()
        logFSINFO('tabulate_local_to_global_dofs', dofmap.tabulate_local_to_global_dofs())
#        fsf['shared_nodes'] = dofmap.shared_nodes()
        fsf['neighbours'] = dofmap.neighbours()
        fsf['dofmap_str'] = dofmap.str(True)
        fsf['dofmap_id'] = dofmap.id()
        fsf['dofmap_label'] = dofmap.label()
        fsf['dofmap_name'] = dofmap.name()
        fsf['max_cell_dimension'] = dofmap.max_cell_dimension()
        fsf['max_element_dofs'] = dofmap.max_element_dofs()
        fsf['dofmap_parameters'] = list(dofmap.parameters.iteritems())
        fsf['dofmap_thisown'] = dofmap.thisown
        fsf['is_view'] = dofmap.is_view()
        for d in range(dim+1):
            fsf['entity_closure_dofs' + str(d)] = \
              dofmap.entity_closure_dofs(self.ksdg.mesh, d)
            fsf['entity_dofs' + str(d)] = \
              dofmap.entity_dofs(self.ksdg.mesh, d)
        fsf['dofcoords'] = np.reshape(
            fs.tabulate_dof_coordinates(), (-1, dim)
        ),
        fsf['block_size'] = dofmap.block_size()
        fsf['dofs'] = dofmap.dofs()
        fsf['rho_dofs'] = rhofs.dofmap().dofs()
        fsf['U_dofs'] = Ufs.dofmap().dofs()
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
                #
                # reopen and close every time, so that valid if aborted
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
#        print("implicitTS __init__ entered", flush=True)
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
#        print(f, flush=True)
        self.setEquationType(self.EquationType.IMPLICIT)
        self.setIFunction(self.implicitIF, f=f)
        self.setIJacobian(self.implicitIJ, J=J, P=J)

    def implicitIF(self, ts, t, u, udot, f):
        """Fully implicit IFunction for PETSc TS

        This is designed for use as the LHS in a fully implicit PETSc
        time-stepper. The DE to be solved is always A.u' = b(u). u
        corresponds to self.sol. A is a constant (i.e., u-independent)
        matrix calculated by assembling the u'-dependent term of the
        weak form, and b a u-dependent vector calculated by assembling
        the remaining terms. This equation may be solved in any of
        three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: implicitIF and implicitIJ calculate the
        LHS A.u' - b and its Jacobian for the fully explicit
        form. Arguments: 

        t: the time (not used).
        u: a petsc4py.PETSc.Vec containing the state vector
        udot: a petsc4py.PETSc.Vec containing the time derivative u'
        f: a petsc4py.PETSc.Vec in which A.u' - b will be left.
        """
        try:
#            if self.rollback:
#                raise ValueError('rolling back')
            self.ksdg.setup_problem()
            pA = fe.as_backend_type(self.ksdg.A).mat()
            psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
            u.copy(psol)
            self.ksdg.sol.vector().apply('insert')
            self.ksdg.b = fe.assemble(self.ksdg.all_terms)
            pb = fe.as_backend_type(self.ksdg.b).vec()
            pb.copy(f)
            f.scale(-1.0)
            pA.multAdd(udot, f, f)
            f.assemble()
            if not np.min(u.array) > 0:
                logTS('np.min(u.array)', np.min(u.array))
#                raise(ValueError('nonpositive function values'))
            if np.isnan(np.min(udot.array)):
                logTS('np.min(udot.array)', np.min(udot.array))
#                raise(ValueError('nans in udot'))
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
        corresponds to self.sol. A is a constant (i.e., u-independent)
        matrix calculated by assembling the u'-dependent term of the
        weak form, and b a u-dependent vector calculated by assembling
        the remaining terms. This equation may be solved in any of
        three forms:

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: implicitIF and implicitIJ calculate the
        LHS A.u' - b and its Jacobian for the fully explicit
        form. Arguments:

        t: the time (not used).
        u: a petsc4py.PETSc.Vec containing the state vector
        udot: a petsc4py.PETSc.Vec containing the u' (not used)
        shift: a real number -- see PETSc Ts documentation.
        J, B: matrices in which the Jacobian shift*A - Jacobian(b(u)) are left. 
        """
        try:
            self.ksdg.setup_problem()
            pA = fe.as_backend_type(self.ksdg.A).mat()
            psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
            u.copy(psol)
            self.ksdg.sol.vector().apply('insert')
            JU = fe.assemble(self.ksdg.JU_terms)
            pJU = fe.as_backend_type(JU).mat()
            Jrho = fe.assemble(self.ksdg.Jrho_terms)
            pJrho = fe.as_backend_type(Jrho).mat()
            pA.copy(J)
            J.scale(shift)
            J.axpy(-1.0, pJU)
            J.axpy(-1.0, pJrho)
            J.assemble()
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

        t: the time (not used).
        u: a petsc4py.PETSc.Vec containing the state vector (not used).
        udot: a petsc4py.PETSc.Vec containing the time derivative u'
        f: a petsc4py.PETSc.Vec in which A.u' will be left.
        """
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

        t: the time (not used).
        u: a petsc4py.PETSc.Vec containing the state vector (not used).
        udot: a petsc4py.PETSc.Vec containing u' (not used)
        shift: see PETSc Ts documentation.
        J, B: petsc4py.PETSc.Mat's in which shift*A will be left.
        """
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

        t: the time (not used).
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
        self.ksdg.setup_problem()
        self.ksdg.b = fe.assemble(self.ksdg.all_terms)
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

        t: the time (not used).
        u: a petsc4py.PETSc.Vec containing the state vector
        J, B: matrices in which the Jacobian of b(u) is left. 
        """
        self.ksdg.setup_problem()
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        JU = fe.assemble(self.ksdg.JU_terms)
        pJU = fe.as_backend_type(JU).mat()
        Jrho = fe.assemble(self.ksdg.Jrho_terms)
        pJrho = fe.as_backend_type(Jrho).mat()
        pJU.copy(J)
        J.axpy(1.0, pJrho)
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
        corresponds to self.sol. A is a constant (i.e., u-independent)
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

        t: the time (not used).
        u: a petsc4py.PETSc.Vec containing the state vector.
        g: a petsc4py.PETSc.Vec in which A^(-1).b(u) will be left.
        """
        psol = fe.as_backend_type(self.ksdg.sol.vector()).vec()
        u.copy(psol)
        self.ksdg.sol.vector().apply('insert')
        self.ksdg.setup_problem()
        self.ksdg.b = fe.assemble(self.ksdg.all_terms)
        ret = self.ksdg.solver.solve(self.ksdg.dsol.vector(), self.ksdg.b)
        pdu = fe.as_backend_type(self.ksdg.dsol.vector()).vec()
        pdu.copy(g)
        g.assemble()

