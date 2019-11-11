"""Functions for discontinuous Galerkin solution of the Keller-Segel PDEs."""

import sys
import numpy as np
from datetime import datetime
from petsc4py import PETSc
from mpi4py import MPI
import ufl
# As a general rule, I import only fenics classes by name.
import fenics as fe
from fenics import (UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh,
                    IntervalMesh, RectangleMesh, BoxMesh, Point,
                    FiniteElement, MixedElement, VectorElement,
                    TestFunctions,
                    Expression, FunctionSpace, VectorFunctionSpace,
                    TrialFunction, TestFunction, Function, Constant,
                    Measure, FacetNormal, CellDiameter, PETScVector,
                    PETScMatrix)
import ufl
from .ksdgdebug import log
from .ksdgexception import KSDGException
from .ksdggather import gather_dof_coords, function_interpolate

def logSOLVER(*args, **kwargs):
    log(*args, system='SOLVER', **kwargs)


meshMakers = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
boxMeshMakers = [IntervalMesh, RectangleMesh, BoxMesh]
cellShapes = [fe.interval, fe.triangle, fe.tetrahedron]

def unit_mesh(nelements=8, dim=1):
    """Make a Unit*Mesh of dimension 1-3

    Parameters:
    nelements: # elements into which each dimension is divided
    dim: Dimension of the space, should be 1, 2, or 3.
    """
    mesh = meshMakers[dim-1](*[nelements]*dim)
    return mesh

def box_mesh(width=1.0, dim=1, nelements=8):
    """Make a rectangular mesh of dimension 1-3

    Parameters:
    width=1.0: width (also length and height) of space
    dim=1: 1, 2, or 3, dimension of space
    nelements=8: division of space into elements
    """
    if dim == 1:
        return IntervalMesh(nelements, 0.0, width)
    elif dim == 2:
        return RectangleMesh(
            Point(np.array([0.0, 0.0], dtype=float)),
            Point(np.array([width, width], dtype=float)),
            nelements, nelements
        )
    elif dim == 3:
        return BoxMesh(
            Point(np.array([0.0, 0.0, 0.0], dtype=float)),
            Point(np.array([width, width, width], dtype=float)),
            nelements, nelements, nelements
        )
    else:
        raise KSDGException("Only dimensions 1, 2, 3 supported.")

def shapes(dim):
    return(cell_shapes[dim-1])

class KSDGSolver:
    default_params = dict(
        rho_min = 1e-7,
        U_min = 1e-7,
        width = 1.0,
        rhopen = 10,
        Upen = 1,
        grhopen = 1,
        gUpen = 1,
    )

    def __init__(
            self,
            mesh=None,
            width=1.0,
            dim=1,
            nelements=8,
            degree=2,
            parameters={},
            V=(lambda U: U),
            U0=None,
            rho0=None,
            t0=0.0,
            debug=False,
            solver_type = 'gmres',
            preconditioner_type = 'default',
            periodic=False,
            ligands=None
            ):
        """Discontinuous Galerkin solver for the Keller-Segel PDE system

        Keyword parameters:
        mesh=None: the mesh on which to solve the problem
        width=1.0: the width of the domain
        dim=1: # of spatial dimensions.
        nelements=8: If mesh is not supplied, one will be
        contructed using UnitIntervalMesh, UnitSquareMesh, or
        UnitCubeMesh (depending on dim). dim and nelements are not
        needed if mesh is supplied.
        degree=2: degree of the polynomial approximation
        parameters={}: a dict giving the values of scalar parameters of
            .V, U0, and rho0 Expressions. This dict needs to also
            define numerical parameters that appear in the PDE. Some
            of these have defaults:
            dim = dim: # of spatial dimensions
            sigma: organism movement rate
            s: attractant secretion rate
            gamma: attractant decay rate
            D: attractant diffusion constant
            rho_min=10.0**-7: minimum feasible worm density
            U_min=10.0**-7: minimum feasible attractant concentration
            rhopen=10: penalty for discontinuities in rho
            Upen=1: penalty for discontinuities in U
            grhopen=1, gUpen=1: penalties for discontinuities in gradients
        V=(lambda U: U): a callable taking two numerical arguments, U
            and rho, or a single argument, U, and returning a single
            number, V, the potential corresponding to U. Use fenics
            versions of mathematical functions, e.g. ufl.ln, abs,
            ufl.exp.
        U0, rho0: Expressions, Functions, or strs specifying the
            initial condition.
        t0=0.0: initial time
        solver_type='gmres'
        preconditioner_type='default'
        periodic, ligands: ignored for caompatibility
        """
        logSOLVER('creating KSDGSolver')
        self.args = dict(
            mesh=mesh,
            width=width,
            dim=dim,
            nelements=nelements,
            degree=degree,
            parameters=parameters,
            V=V,
            U0=U0,
            rho0=rho0,
            t0=t0,
            debug=debug,
            solver_type = solver_type,
            preconditioner_type = preconditioner_type,
            periodic=periodic,
            ligands=ligands
        )
        self.debug = debug
        self.solver_type = solver_type
        self.preconditioner_type = preconditioner_type
        self.periodic = False
        self.ligands = ligands
        self.params = self.default_params.copy()
        if (mesh):
            self.omesh = self.mesh = mesh
        else:
            self.omesh = self.mesh = box_mesh(width=width, dim=dim,
                                              nelements=nelements)
            self.nelements = nelements
        logSOLVER('self.mesh', self.mesh)
        logSOLVER('self.mesh.mpi_comm().size', self.mesh.mpi_comm().size)
        self.nelements = nelements
        self.degree = degree
        self.dim = self.mesh.geometry().dim()
        self.params['dim'] = self.dim
        self.params.update(parameters)
        # 
        # Solution spaces and Functions
        # 
        fss = self.make_function_space()
        (self.SE, self.SS, self.VE, self.VS) = [
            fss[fs] for fs in ('SE', 'SS', 'VE', 'VS')
        ]
        logSOLVER('self.VS', self.VS)
        self.sol = Function(self.VS)                  # sol, current soln
        logSOLVER('self.sol', self.sol)
        self.srho, self.sU = self.sol.sub(0), self.sol.sub(1)
        self.irho, self.iU = fe.split(self.sol)
        self.wrho, self.wU = TestFunctions(self.VS)
        self.tdsol = TrialFunction(self.VS)
        self.tdrho, self.tdU = fe.split(self.tdsol)
        self.n = FacetNormal(self.mesh)
        self.h = CellDiameter(self.mesh)
        self.havg = fe.avg(self.h)
        self.dx = fe.dx
#        self.dx = fe.dx(metadata={'quadrature_degree': min(degree, 10)})
        self.dS = fe.dS
#        self.dS = fe.dS(metadata={'quadrature_degree': min(degree, 10)})
        #
        # record initial state
        #
        try:
            V(self.iU, self.irho)
            def realV(U, rho):
                return V(U, rho)
        except TypeError:
            def realV(U, rho):
                return V(U)
        self.V = realV
        if not U0:
            U0 = Constant(0.0)
        if isinstance(U0, ufl.coefficient.Coefficient):
            self.U0 = U0
        else:
            self.U0 = Expression(U0, **self.params,
                                 degree=self.degree, domain=self.mesh)
        if not rho0:
            rho0 = Constant(0.0)
        if isinstance(rho0, ufl.coefficient.Coefficient):
            self.rho0 = rho0
        else:
            self.rho0 = Expression(rho0, **self.params,
                                   degree=self.degree, domain=self.mesh)
        self.t0 = t0
        #
        # initialize state
        #
        # cache assigners
        logSOLVER('restarting')
        self.restart()
        logSOLVER('restart returned')
        return(None)

    def make_function_space(self,
                            mesh=None,
                            dim=None,
                            degree=None
                            ):
        if not mesh: mesh = self.mesh
        if not dim: dim = self.dim
        if not degree: degree = self.degree
        SE = FiniteElement('DG', cellShapes[dim-1], degree)
        SS = FunctionSpace(mesh, SE)   # scalar space
        VE = MixedElement(SE, SE)
        VS = FunctionSpace(mesh, VE)   # vector space
        return dict(SE=SE, SS=SS, VE=VE, VS=VS)

    def restart(self):
        logSOLVER('restart')
        self.t = self.t0
        CE = FiniteElement('CG', cellShapes[self.dim-1], self.degree)
        CS = FunctionSpace(self.mesh, CE)   # scalar space
        coords = gather_dof_coords(CS)
        logSOLVER('function_interpolate(self.U0, self.SS, coords=coords)',
                  function_interpolate(self.U0, self.SS,
                                       coords=coords))
        fe.assign(self.sol.sub(1),
                  function_interpolate(self.U0, self.SS,
                                       coords=coords))
        logSOLVER('U0 assign returned')
        fe.assign(self.sol.sub(0),
                  function_interpolate(self.rho0, self.SS,
                                       coords=coords))

    def set_time(t):
        """Stub for derived classes to override"""
        self.t = t

    def setup_problem(self, debug=False):
        #
        # assemble the matrix, if necessary (once for all time points)
        #
        if not hasattr(self, 'A'):
            self.drho_integral = self.tdrho*self.wrho*self.dx
            self.dU_integral = self.tdU*self.wU*self.dx
            self.A = fe.assemble(self.drho_integral + self.dU_integral)
            # if self.solver_type == 'lu':
            #     self.solver = fe.LUSolver(
            #         self.A,
            #     )
            #     self.solver.parameters['reuse_factorization'] = True
            # else:
            #     self.solver = fe.KrylovSolver(
            #         self.A,
            #         self.solver_type,
            #         self.preconditioner_type
            #     )
            # self.solver.parameters.add('linear_solver', self.solver_type)
            # kparams = fe.Parameters('krylov_solver')
            # kparams.add('report', True)
            # kparams.add('nonzero_initial_guess', True)
            # self.solver.parameters.add(kparams)
            # lparams = fe.Parameters('lu_solver')
            # lparams.add('report', True)
            # lparams.add('reuse_factorization', True)
            # lparams.add('verbose', True)
            # self.solver.parameters.add(lparams)
            self.dsol = Function(self.VS)
            self.drho, self.dU = self.dsol.sub(0), self.dsol.sub(1)
        #
        # assemble RHS (for each time point, but compile only once)
        #
        if not hasattr(self, 'rho_terms'):
            self.sigma = self.params['sigma']
            self.s2 = self.sigma * self.sigma / 2
            self.rho_min = self.params['rho_min']
            self.rhopen = self.params['rhopen']
            self.grhopen = self.params['grhopen']
            self.v = -ufl.grad(self.V(self.iU, self.irho)) - (
                self.s2*ufl.grad(self.irho)/ufl.max_value(self.irho,
                                                          self.rho_min) 
            )
            self.flux = self.v * self.irho
            self.vn = ufl.max_value(ufl.dot(self.v, self.n), 0)
            self.facet_flux = (
                self.vn('+')*ufl.max_value(self.irho('+'), 0.0) -
                self.vn('-')*ufl.max_value(self.irho('-'), 0.0)
            )
            self.rho_flux_jump = -self.facet_flux*ufl.jump(self.wrho)*self.dS
            self.rho_grad_move = ufl.dot(self.flux,
                                        ufl.grad(self.wrho))*self.dx
            self.rho_penalty = -(
                (self.rhopen * self.degree**2 / self.havg) *
                ufl.dot(ufl.jump(self.irho, self.n),
                       ufl.jump(self.wrho, self.n)) * self.dS
            )
            self.grho_penalty = -(
                self.grhopen * self.degree**2 *
                (ufl.jump(ufl.grad(self.irho), self.n) *
                ufl.jump(ufl.grad(self.wrho), self.n)) * self.dS
            )
            self.rho_terms = (
                self.rho_flux_jump + self.rho_grad_move +
                self.rho_penalty + self.grho_penalty
            )
        if not hasattr(self, 'U_terms'):
            self.U_min = self.params['U_min']
            self.gamma = self.params['gamma']
            self.s = self.params['s']
            self.D = self.params['D']
            self.Upen = self.params['Upen']
            self.gUpen = self.params['gUpen']
            self.U_decay = -self.gamma * self.iU * self.wU * self.dx
            self.U_secretion = self.s * self.irho * self.wU * self.dx
            self.jump_gUw = (
                self.D * ufl.jump(self.wU * ufl.grad(self.iU), self.n)
                * self.dS 
            )
            self.U_diffusion = - self.D * ufl.dot(ufl.grad(self.iU),
                                                 ufl.grad(self.wU))*self.dx
            self.U_penalty = -(
                (self.Upen * self.degree**2 / self.havg) *
                ufl.dot(ufl.jump(self.iU, self.n), ufl.jump(self.wU, self.n))*self.dS
            )
            self.gU_penalty = -(
                self.gUpen * self.degree**2 *
                (ufl.jump(ufl.grad(self.iU), self.n) *
                 ufl.jump(ufl.grad(self.wU), self.n)) * self.dS
            )
            self.U_terms = (
                # decay and secretion
                self.U_decay + self.U_secretion +
                # diffusion
                self.jump_gUw + self.U_diffusion +
                # penalties (to enforce continuity)
                self.U_penalty + self.gU_penalty
            )
        if not hasattr(self, 'all_terms'):
            self.all_terms = self.rho_terms + self.U_terms
        if not hasattr(self, 'J_terms'):
            self.J_terms = fe.derivative(self.all_terms, self.sol)
        # if not hasattr(self, 'JU_terms'):
        #     self.JU_terms = fe.derivative(self.all_terms, self.sU)
        # if not hasattr(self, 'Jrho_terms'):
        #     self.Jrho_terms = fe.derivative(self.all_terms, self.srho)
            

    def ddt(self, debug=False):
        """Calculate time derivative of rho and U

        Results are left in self.dsol as a two-component vector function.
        """
        self.setup_problem(debug)
        self.b = fe.assemble(self.all_terms)
        return fe.solve(self.A, self.dsol.vector(), self.b,
                        self.solver_type)

    #
    # The following member functions should not really exist -- use
    # the TS classes in ts instead. These are here only to allow some
    # old code to work.
            
    def implicitTS(
        self,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        prt=True,
        restart=True,
        tstype = PETSc.TS.Type.ROSW,
        finaltime = PETSc.TS.ExactFinalTime.STEPOVER
    ):
        """
        Create an implicit timestepper and solve the DE

        Keyword arguments:
        t0=0.0: the initial time.
        dt=0.001: the initial time step.
        tmax=20: the final time.
        maxsteps=100: maximum number of steps to take.
        rtol=1e-5: relative error tolerance.
        atol=1e-5: absolute error tolerance.
        prt=True: whether to print results as solution progresses
        restart=True: whether to set the initial condition to rho0, U0
        tstype=PETSc.TS.Type.ROSW: implicit solver to use.
        finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
            final time step.

        Other options can be set by modifying the PETSc Options
        database.
        """
#        print("KSDGSolver.implicitTS __init__ entered")
        from .ts import implicitTS # done here to avoid circular imports
        self.ts = implicitTS(
            self,
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,
            restart=restart,
            tstype = tstype,
            finaltime = finaltime,
        )            
        self.ts.setMonitor(self.ts.historyMonitor)
        if prt:
            self.ts.setMonitor(self.ts.printMonitor)
        self.ts.solve()
        self.ts.cleanup()

    def cleanupTS():
        """Should be called when finished with a TS

        Leaves history unchanged.
        """
        del self.ts, self.tsparams

    def imExTS(
        self,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        prt=True,
        restart=True,
        tstype = PETSc.TS.Type.ARKIMEX,
        finaltime = PETSc.TS.ExactFinalTime.STEPOVER
    ):
        """
        Create an implicit/explicit timestepper and solve the DE

        Keyword arguments:
        t0=0.0: the initial time.
        dt=0.001: the initial time step.
        tmax=20: the final time.
        maxsteps=100: maximum number of steps to take.
        rtol=1e-5: relative error tolerance.
        atol=1e-5: absolute error tolerance.
        prt=True: whether to print results as solution progresses
        restart=True: whether to set the initial condition to rho0, U0
        tstype=PETSc.TS.Type.ARKIMEX: implicit solver to use.
        finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
            final time step.

        Other options can be set by modifying the PETSc options
        database.
        """
        from .ts import imExTS   # done here to avoid circular imports
        self.ts = imExTS(
            self,
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,
            restart=restart,
            tstype = tstype,
            finaltime = finaltime,
        )            
        self.ts.setMonitor(self.ts.historyMonitor)
        if prt:
            self.ts.setMonitor(self.ts.printMonitor)
        self.ts.solve()
        self.ts.cleanup()
            
    def explicitTS(
        self,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        prt=True,
        restart=True,
        tstype = PETSc.TS.Type.RK,
        finaltime = PETSc.TS.ExactFinalTime.STEPOVER
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
        prt=True: whether to print results as solution progresses
        restart=True: whether to set the initial condition to rho0, U0
        tstype=PETSc.TS.Type.RK: explicit solver to use.
        finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
            final time step.

        Other options can be set by modifyign the PETSc options
        database.
        """
        from .ts import explicitTS # done here to avoid circular imports
        self.ts = explicitTS(
            self,
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,
            restart=restart,
            tstype = tstype,
            finaltime = finaltime,
        )            
        self.ts.setMonitor(self.ts.historyMonitor)
        if prt:
            self.ts.setMonitor(self.ts.printMonitor)
        self.ts.solve()
        self.ts.cleanup()

class EKKSDGSolver(KSDGSolver):
    """KSDSolver that uses the Epshteyn and Kurganov scheme for rho.

    Overrides the setup_problem method of KSDGSolver.
    """

    def setup_problem(self, debug=False):
        #
        # assemble the matrix, if necessary (once for all time points)
        #
        if not hasattr(self, 'A'):
            drho_integral = self.tdrho*self.wrho*self.dx
            dU_integral = self.tdU*self.wU*self.dx
            self.A = fe.assemble(drho_integral + dU_integral)
            # if self.solver_type == 'lu':
            #     self.solver = fe.LUSolver(
            #         self.A,
            #         method=self.solver_type
            #     )
            #     self.solver.parameters['reuse_factorization'] = True
            # else:
            #     self.solver = fe.KrylovSolver(
            #         self.A,
            #         self.solver_type,
            #         self.preconditioner_type
            #     )
            self.dsol = Function(self.VS)
            self.drho, self.dU = self.dsol.sub(0), self.dsol.sub(1)
        #
        # assemble RHS (has to be done for each time point)
        #
        if not hasattr(self, 'rho_terms'):
            self.sigma = self.params['sigma']
            self.s2 = self.sigma * self.sigma / 2
            self.rho_min = self.params['rho_min']
            self.rhopen = self.params['rhopen']
            self.grhopen = self.params['grhopen']
            self.v = -ufl.grad(self.V(self.iU, self.irho))
            self.flux = self.v * self.irho
            self.vn = ufl.max_value(ufl.dot(self.v, self.n), 0)
            self.facet_flux = (
                self.vn('+')*self.irho('+') -
                self.vn('-')*self.irho('-')
            )
            self.rho_flux_jump = -self.facet_flux*ufl.jump(self.wrho)*self.dS
            self.rho_grad_move = ufl.dot(self.flux,
                                        ufl.grad(self.wrho))*self.dx
            self.rho_penalty = -(
                (self.rhopen * self.degree**2 / self.havg) *
                ufl.dot(ufl.jump(self.irho, self.n),
                       ufl.jump(self.wrho, self.n)) * self.dS
            )
            # self.facet_flux = (
            #     self.vn('+')*self.rho('+') - self.vn('-')*self.rho('-')
            # )
            # self.rho_flux_jump = -self.facet_flux*ufl.jump(self.wrho)*self.dS
            # self.rho_grad_move = ufl.dot(self.flux, ufl.grad(self.wrho))*self.dx
            self.jump_grhow = (
                self.s2 * ufl.jump(self.wrho * ufl.grad(self.irho),
                                   self.n) * self.dS
            )
            self.rho_diffusion = -self.s2 * ufl.dot(ufl.grad(self.irho),
                                                 ufl.grad(self.wrho))*self.dx
            # self.rho_penalty = -(
            #     (self.rhopen * self.degree**2 / self.havg) *
            #     ufl.dot(ufl.jump(self.rho, self.n),
            #            ufl.jump(self.wrho, self.n)) * self.dS
            # )
            self.grho_penalty = -(
                self.grhopen * self.degree**2 *
                (ufl.jump(ufl.grad(self.irho), self.n) *
                ufl.jump(ufl.grad(self.wrho), self.n)) * self.dS
            )
            self.rho_terms = (
                # advection terms
                self.rho_flux_jump + self.rho_grad_move +
                # diffusive terms
                self.rho_diffusion + self.jump_grhow +
                # penalty terms (to enforce continuity)
                self.rho_penalty + self.grho_penalty
            )
        if not hasattr(self, 'U_terms'):
            self.U_min = self.params['U_min']
            self.gamma = self.params['gamma']
            self.s = self.params['s']
            self.D = self.params['D']
            self.Upen = self.params['Upen']
            self.gUpen = self.params['gUpen']
            self.U_decay = -self.gamma * self.iU * self.wU * self.dx
            self.U_secretion = self.s * self.irho * self.wU * self.dx
            self.jump_gUw = (
                self.D * ufl.jump(self.wU * ufl.grad(self.iU), self.n)
                * self.dS 
            )
            self.U_diffusion = - self.D * ufl.dot(ufl.grad(self.iU),
                                                 ufl.grad(self.wU))*self.dx
            self.U_penalty = -(
                (self.Upen * self.degree**2 / self.havg) *
                ufl.dot(ufl.jump(self.iU, self.n), ufl.jump(self.wU, self.n))*self.dS
            )
            self.gU_penalty = -(
                self.gUpen * self.degree**2 *
                (ufl.jump(ufl.grad(self.iU), self.n) *
                 ufl.jump(ufl.grad(self.wU), self.n)) * self.dS
            )
            self.U_terms = (
                # decay and secretion
                self.U_decay + self.U_secretion +
                # diffusion
                self.jump_gUw + self.U_diffusion +
                # penalties (to enforce continuity)
                self.U_penalty + self.gU_penalty
            )
        if not hasattr(self, 'all_terms'):
            self.all_terms = self.rho_terms + self.U_terms
        if not hasattr(self, 'all_terms'):
            self.all_terms = self.rho_terms + self.U_terms
        if not hasattr(self, 'J_terms'):
            self.J_terms = fe.derivative(self.all_terms, self.sol)
        # if not hasattr(self, 'JU_terms'):
        #     self.JU_terms = fe.derivative(self.all_terms, self.sU)
        # if not hasattr(self, 'Jrho_terms'):
        #     self.Jrho_terms = fe.derivative(self.all_terms, self.srho)
                
                
def main():
    nelements = 8
    dim = 1
    degree = 2
    params = {
        'alpha': 1,
        'beta': 1,
        'mu': 0.4,
        'Umax': 1,
        'Ufac': 4,
        'sU': 1,
        'sigma': 1,
        'N': 1,
        'M': 1,
        's': 10,
        'gamma': 1,
        'D': 0.01,
        'srho0': 0.01,
        'grhopen': 10,
    }
    U0str = 'Ufac/dim * (0.25 - pow(x[0] - mu, 2)/(2*sU*sU))'
    rho0str = 'N*exp(-beta*log((%s)+alpha)/(sigma*sigma/2))' % U0str
    def Vfunc(U):
        return -params['beta']*ufl.ln(U + params['alpha'])
    mesh = unit_mesh(nelements, dim)
    fe.parameters["form_compiler"]["representation"] = "uflacs"
    fe.parameters['linear_algebra_backend'] = 'PETSc'
    solver = makeKSDGSolver(dim=dim, degree=degree, nelements=nelements,
                        parameters=params,
                        V=Vfunc, U0=U0str, rho0=rho0str,
                        debug=True)
    print(str(solver))
    print(solver.ddt(debug=True))
    print("dsol/dt components:", solver.dsol.vector()[:])

    eksolver = EKKSDGSolver(dim=dim, degree=degree, nelements=nelements,
                        parameters=params,
                        V=Vfunc, U0=U0str, rho0=rho0str,
                        debug=True)
    print(str(eksolver))
    print(eksolver.ddt(debug=True))
    print("dsol/dt components:", eksolver.dsol.vector()[:])
    #
    # try out the time-stepper
    #
    np.random.seed(793817931)
    murho0 = params['N']/params['M']
    Cd1 = fe.FunctionSpace(mesh, 'CG', 1)
    rho0 = Function(Cd1)
    rho0.vector()[:] = np.random.normal(murho0, params['srho0'],
                                        rho0.vector()[:].shape)
    U0 = Function(Cd1)
    U0.vector()[:] = (params['s']/params['gamma'])*rho0.vector()
    ksdg = makeKSDGSolver(degree=degree, mesh=mesh,
                          parameters=params,
                          V=Vfunc, U0=U0, rho0=rho0)
    ksdg.implicitTS()
    print('Final time point')
    print(ksdg.history[-1])
    ksdg.imExTS()
    print('Final time point')
    print(ksdg.history[-1])
    ksdg.explicitTS()
    print('Final time point')
    print(ksdg.history[-1])
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
