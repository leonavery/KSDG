"""Functions for discontinuous Galerkin solution of the Keller-Segel PDEs."""

import sys
import numpy as np
import itertools
import copy
from datetime import datetime
from petsc4py import PETSc
from mpi4py import MPI
import ufl
# As a general rule, I import only fenics classes by name.
import fenics as fe
from fenics import (UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh,
                    IntervalMesh, RectangleMesh, BoxMesh, Point,
                    FiniteElement, MixedElement, VectorElement,
                    TestFunctions, Mesh, File,
                    Expression, FunctionSpace, VectorFunctionSpace,
                    TrialFunction, TestFunction, Function, Constant,
                    Measure, FacetNormal, CellDiameter, PETScVector,
                    PETScMatrix, SubDomain, MeshFunction,
                    SubMesh, DirichletBC)
import ufl

from .ksdgsolver import (KSDGSolver, meshMakers, boxMeshMakers,
                         cellShapes, unit_mesh, box_mesh, shapes)
from .ksdgdebug import log
from .ksdgexception import KSDGException
from .ksdggather import gather_dof_coords, function_interpolate

def logMULTIPLE(*args, **kwargs):
    log(*args, system='MULTIPLE', **kwargs)


class KSDGSolverMultiple(KSDGSolver):
    default_params = dict(
        rho_min = 1e-7,
        U_min = 1e-7,
        width = 1.0,
        rhopen = 10,
        Upen = 1,
        grhopen = 1,
        gUpen = 1,
        ligands = None,
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
            U0=[],
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
            rho_min=10.0**-7: minimum feasible worm density
            U_min=10.0**-7: minimum feasible attractant concentration
            rhopen=10: penalty for discontinuities in rho
            Upen=1: penalty for discontinuities in U
            grhopen=1, gUpen=1: penalties for discontinuities in gradients
            nligands=1, number of ligands
        V=(lambda Us: Us): a callable taking two arguments, Us and
            rho, or a single argument, Us. Us is a list of length
            nligands. rho is a single expression. V returns a single
            number, V, the potential corresponding to Us (and
            rho). Use ufl versions of mathematical functions,
            e.g. ufl.ln, abs, ufl.exp.
        rho0: Expressions, Functions, or strs specifying the
            initial condition for rho.
        U0: a list of nligands Expressions, Functions or strs
            specifying the initial conditions for the ligands.
        t0=0.0: initial time
        solver_type='gmres'
        preconditioner_type='default'
        ligands=LigandGroups(): ligand list
        periodic=False: ignored for compatibility
        """
        logMULTIPLE('creating KSDGSolverMultiple')
        if not ligands:
            ligands = LigandGroups()
        else:
            ligands = copy.deepcopy(ligands)
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
        self.nligands = ligands.nligands()
        self.params = self.default_params.copy()
        if (mesh):
            self.omesh = self.mesh = mesh
        else:
            self.omesh = self.mesh = box_mesh(width=width, dim=dim,
                                              nelements=nelements)
            self.nelements = nelements
        logMULTIPLE('self.mesh', self.mesh)
        logMULTIPLE('self.mesh.mpi_comm().size', self.mesh.mpi_comm().size)
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
        logMULTIPLE('self.VS', self.VS)
        self.sol = Function(self.VS)                  # sol, current soln
        logMULTIPLE('self.sol', self.sol)
        self.srho, self.sUs = self.sol.sub(0), self.sol.split()[1:]
        splitsol = fe.split(self.sol)
        self.irho, self.iUs = splitsol[0], splitsol[1:]
        tfs = TestFunctions(self.VS)
        self.wrho, self.wUs = tfs[0], tfs[1:]
        self.tdsol = TrialFunction(self.VS)
        splittdsol = fe.split(self.tdsol)
        self.tdrho, self.tdUs = splittdsol[0], splittdsol[1:]
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
            V(self.iUs, self.irho)
            def realV(Us, rho):
                return V(Us, rho)
        except TypeError:
            def realV(Us, rho):
                return V(Us)
        self.V = realV
        if not U0:
            U0 = [Constant(0.0)] * self.nligands
        self.U0s = [Constant(0.0)] * self.nligands
        for i,U0i in enumerate(U0):
            if isinstance(U0i, ufl.coefficient.Coefficient):
                self.U0s[i] = U0i
            else:
                self.U0s[i] = Expression(U0i, **self.params,
                                         degree=self.degree,
                                         domain=self.mesh)
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
        logMULTIPLE('restarting')
        self.restart()
        logMULTIPLE('restart returned')
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
        elements = [SE] * (self.nligands + 1)
        VE = MixedElement(elements)
        VS = FunctionSpace(mesh, VE)   # vector space
        return dict(SE=SE, SS=SS, VE=VE, VS=VS)

    def restart(self):
        logMULTIPLE('restart')
        self.t = self.t0
        CE = FiniteElement('CG', cellShapes[self.dim-1], self.degree)
        CS = FunctionSpace(self.mesh, CE)   # scalar space
        coords = gather_dof_coords(CS)
        fe.assign(self.sol.sub(0),
                  function_interpolate(self.rho0, self.SS,
                                       coords=coords))
        for i,U0i in enumerate(self.U0s):
            fe.assign(self.sol.sub(i+1),
                      function_interpolate(U0i, self.SS, coords=coords)) 
        logMULTIPLE('U0s assign returned')
        
    def setup_problem(self, debug=False):
        #
        # assemble the matrix, if necessary (once for all time points)
        #
        if not hasattr(self, 'A'):
            self.drho_integral = self.tdrho*self.wrho*self.dx
            self.dU_integral = sum(
                [tdUi*wUi*self.dx for tdUi,wUi in zip(self.tdUs, self.wUs)]
            )
            self.A = fe.assemble(self.drho_integral + self.dU_integral)
            self.dsol = Function(self.VS)
            dsolsplit = self.dsol.split()
            self.drho, self.dUs = dsolsplit[0], dsolsplit[1:]
        #
        # assemble RHS (for each time point, but compile only once)
        #
        if not hasattr(self, 'rho_terms'):
            self.sigma = self.params['sigma']
            self.s2 = self.sigma * self.sigma / 2
            self.rho_min = self.params['rho_min']
            self.rhopen = self.params['rhopen']
            self.grhopen = self.params['grhopen']
            self.v = -ufl.grad(self.V(self.iUs, self.irho)) - (
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
            self.Upen = self.params['Upen']
            self.gUpen = self.params['gUpen']
            self.U_decay = sum(
                [-lig.gamma * iUi * wUi * self.dx for
                 lig,iUi,wUi in
                 zip(self.ligands.ligands(), self.iUs, self.wUs)]
            )
            self.U_secretion = sum(
                [lig.s * self.irho * wUi * self.dx for
                 lig,wUi in zip(self.ligands.ligands(), self.wUs)]
            )
            self.jump_gUw = sum(
                [lig.D * ufl.jump(wUi * ufl.grad(iUi), self.n) * self.dS
                for lig,wUi,iUi in
                zip(self.ligands.ligands(), self.wUs, self.iUs)]
            )
            self.U_diffusion = sum(
                [-lig.D * ufl.dot(ufl.grad(iUi), ufl.grad(wUi))*self.dx for
                 lig,iUi,wUi in
                 zip(self.ligands.ligands(), self.iUs, self.wUs)]
            )
            self.U_penalty = sum(
                [-(self.Upen*self.degree**2/self.havg) *
                 ufl.dot(ufl.jump(iUi, self.n),
                         ufl.jump(wUi, self.n))*self.dS for
                 iUi,wUi in zip(self.iUs, self.wUs)]
            )
            self.gU_penalty = -self.gUpen * self.degree**2 * sum(
                [ufl.jump(ufl.grad(iUi), self.n) *
                 ufl.jump(ufl.grad(wUi), self.n) * self.dS for
                 iUi,wUi in zip(self.iUs, self.wUs)]
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

    def ddt(self, debug=False):
        """Calculate time derivative of rho and U

        Results are left in self.dsol as a two-component vector function.
        """
        self.setup_problem(debug)
        self.b = fe.assemble(self.all_terms)
        return fe.solve(self.A, self.dsol.vector(), self.b,
                        self.solver_type)

                
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
