"""DG solution of Keller-Segel with multiple ligands and periodic domain
"""

import sys
import numpy as np
import copy
import itertools
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
from .ksdggather import (gather_dof_coords, gather_vertex_coords,
                         gather_mesh, coord_remap, bcast_function,
                         distribute_mesh, function_interpolate)
from .ksdgperiodic import (_cstats, mesh_stats, CornerDomain,
                         corner_submesh, evenodd_symmetries,
                         evenodd_matrix, matmul, vectotal,
                         evenodd_functions, FacesDomain,
                         KSDGSolverPeriodic)
from .ksdgmultiple import (KSDGSolverMultiple)

def logPERIODIC(*args, **kwargs):
    log(*args, system='PERIODIC', **kwargs)

def logMULTIPLE(*args, **kwargs):
    log(*args, system='MULTIPLE', **kwargs)

def transpose(nestedlist, fillvalue=None):
    itertools.zip_longest(*nestedlist)
    
class KSDGSolverMultiPeriodic(KSDGSolverMultiple, KSDGSolverPeriodic):
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
            nelements=None,
            degree=2,
            parameters={},
            V=(lambda U: U),
            U0=None,
            rho0=None,
            t0=0.0,
            debug=False,
            solver_type = 'lu',
            preconditioner_type = 'default',
            periodic=True,
            ligands=None
            ):
        """DG solver for the periodic Keller-Segel PDE system

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
            versions of mathematical functions, e.g. fe.ln, abs,
            fe.exp.
        U0, rho0: Expressions, Functions, or strs specifying the
            initial condition.
        t0=0.0: initial time
        solver_type='lu'
        preconditioner_type='default'
        periodic=True: Allowed for compatibility, but ignored
        ligands=None: ignored for compatibility
        """
        logPERIODIC('creating KSDGSolverMultiPeriodic')
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
            periodic=True,
            ligands=ligands
        )
        self.debug = debug
        self.solver_type = solver_type
        self.preconditioner_type = preconditioner_type
        self.periodic = True
        self.ligands = ligands
        self.nligands = ligands.nligands()
        self.params = self.default_params.copy()
        #
        # Store the original mesh in self.omesh. self.mesh will be the
        # corner mesh.
        #
        if nelements is None:
            self.nelements = 8
        else:
            self.nelements = nelements
        if (mesh):
            self.omesh = mesh
        else:
            self.omesh = box_mesh(width=width, dim=dim,
                                  nelements=self.nelements)
        omeshstats = mesh_stats(self.omesh)
        try:
            comm = self.omesh.mpi_comm().tompi4py()
        except AttributeError:
            comm = self.omesh.mpi_comm()
        self.lmesh = gather_mesh(self.omesh)
        logPERIODIC('omeshstats', omeshstats)
        self.xmin = omeshstats['xmin']
        self.xmax = omeshstats['xmax']
        self.xmid = omeshstats['xmid']
        self.delta_ = omeshstats['dx']
        if nelements is None:
            self.nelements = (self.xmax - self.xmin) / self.delta_
        self.mesh = corner_submesh(self.lmesh)
        meshstats = mesh_stats(self.mesh)
        logPERIODIC('meshstats', meshstats)
        logPERIODIC('self.omesh', self.omesh)
        logPERIODIC('self.mesh', self.mesh)
        logPERIODIC('self.mesh.mpi_comm().size', self.mesh.mpi_comm().size)
        self.degree = degree
        self.dim = self.mesh.geometry().dim()
        self.params['dim'] = self.dim
        self.params.update(parameters)
        # 
        # Solution spaces and Functions
        #
        # The solution function space is a vector space with
        # (nligands+1)*(2**dim) elements. The first 2**dim components
        # are even and odd parts of rho; These are followed by even
        # and odd parts of each ligand. The array self.evenodd
        # identifies even and odd components. Each row is a length dim
        # sequence 0s and 1s and represents one component. For
        # instance, if evenodd[i] is [0, 1, 0], then component i of
        # the vector space is even in dimensions 0 and 2 (x and z
        # conventionally) and off in dimension 1 (y).
        #
        self.symmetries = evenodd_symmetries(self.dim)
        self.signs = [fe.as_matrix(np.diagflat(1.0 - 2.0*eo))
                      for eo in self.symmetries]
        self.eomat = evenodd_matrix(self.symmetries)
        fss = self.make_function_space()
        (self.SE, self.SS, self.VE, self.VS) = [
            fss[fs] for fs in ('SE', 'SS', 'VE', 'VS')
        ]
        self.sol = Function(self.VS)                  # sol, current soln
        logPERIODIC('self.sol', self.sol)
        # srhos and sUs are functions defined on subspaces
        self.srhos = self.sol.split()[:2**self.dim]
        self.sUs = self.sol.split()[2**self.dim:]
        # irhos and iUs are Indexed UFL expressions
        self.irhos = fe.split(self.sol)[:2**self.dim]
        self.iUs = fe.split(self.sol)[2**self.dim:]
        self.wrhos = TestFunctions(self.VS)[:2**self.dim]
        self.wUs = TestFunctions(self.VS)[2**self.dim:]
        self.tdsol = TrialFunction(self.VS) # time derivatives
        self.tdrhos = fe.split(self.tdsol)[:2**self.dim]
        self.tdUs = fe.split(self.tdsol)[2**self.dim:]
        bc_method = 'geometric' if self.dim > 1 else 'pointwise'
        rhobcs = [DirichletBC(
            self.VS.sub(i),
            Constant(0),
            FacesDomain(self.mesh, self.symmetries[i]),
            method=bc_method
        ) for i in range(2**self.dim) if np.any(self.symmetries[i] != 0.0)]
        Ubcs = list(itertools.chain(*[
            [
                DirichletBC(
                    self.VS.sub(i + (lig+1)*2**self.dim),
                    Constant(0),
                    FacesDomain(self.mesh, self.symmetries[i]),
                    method=bc_method
                ) for i in range(2**self.dim)
                if np.any(self.symmetries[i] != 0.0)
            ] for lig in range(self.nligands)
        ]))
        self.bcs = rhobcs + Ubcs
        self.n = FacetNormal(self.mesh)
        self.h = CellDiameter(self.mesh)
        self.havg = fe.avg(self.h)
        self.dx = fe.dx
        self.dS = fe.dS
        #
        # record initial state
        #
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
        try:
            V(self.U0s, self.rho0)
            def realV(U, rho):
                return V(U, rho)
        except TypeError:
            def realV(U, rho):
                return V(U)
        self.V = realV
        self.t0 = t0
        #
        # initialize state
        #
        # cache assigners
        logPERIODIC('restarting')
        self.restart()
        logPERIODIC('restart returned')
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
        SS = FunctionSpace(mesh, SE) # scalar space
        elements = [SE] * ((self.nligands+1)*2**self.dim)
        VE = MixedElement(elements)
        VS = FunctionSpace(mesh, VE)   # vector space
        logPERIODIC('VS', VS)
        return dict(SE=SE, SS=SS, VE=VE, VS=VS)


    def restart(self):
        logPERIODIC('restart')
        self.t = self.t0
        U0comps = [None] * self.nligands * 2**self.dim
        for i,U0i in enumerate(self.U0s):
            eofuncs = evenodd_functions(
                omesh=self.omesh,
                degree=self.degree,
                func=U0i,
                evenodd=self.symmetries,
                width=self.xmax
            )
            U0comps[i*2**self.dim:(i+1)*2**self.dim] = eofuncs
        rho0comps = evenodd_functions(
            omesh=self.omesh,
            degree=self.degree,
            func=self.rho0,
            evenodd=self.symmetries,
            width=self.xmax
        )
        coords = gather_dof_coords(rho0comps[0].function_space())
        for i in range(2**self.dim):
            fe.assign(self.sol.sub(i),
                      function_interpolate(rho0comps[i],
                                           self.SS,
                                           coords=coords))
        for i in range(self.nligands*2**self.dim):
            fe.assign(self.sol.sub(i + 2**self.dim),
                      function_interpolate(U0comps[i],
                                           self.SS,
                                           coords=coords))
        
    def setup_problem(self, debug=False):
        #
        # assemble the matrix, if necessary (once for all time points)
        #
        if not hasattr(self, 'A'):
            drho_integral = vectotal(
                [tdrho*wrho*self.dx for tdrho,wrho in
                 zip(self.tdrhos, self.wrhos)]
            )
            dU_integral = vectotal(
                [tdU*wU*self.dx
                 for tdU,wU in zip(self.tdUs, self.wUs)
                ]
            )
            self.A = fe.assemble(drho_integral + dU_integral)
            for bc in self.bcs:
                bc.apply(self.A)
            self.dsol = Function(self.VS)
            self.drhos = self.dsol.split()[:2**self.dim]
            self.dUs = self.dsol.split()[2**self.dim:]
        #
        # These are the values of rho and U themselves (not their
        # symmetrized versions) on all subdomains of the original
        # domain.
        #
        if not hasattr(self, 'rhosds'):
            self.rhosds = matmul(self.eomat, self.irhos)
        # self.Usds is a list of nligands lists. Sublist i is of
        # length 2**dim and lists the value of ligand i on each of the
        # 2**dim subdomains.
        #
        if not hasattr(self, 'Usds'):
            self.Usds = [
                matmul(self.eomat,
                       self.iUs[i*2**self.dim:(i+1)*2**self.dim])
                for i in range(self.nligands)
            ]
        #
        # assemble RHS (for each time point, but compile only once)
        #
        if not hasattr(self, 'rho_terms'):
            self.sigma = self.params['sigma']
            self.s2 = self.sigma * self.sigma / 2
            self.rho_min = self.params['rho_min']
            self.rhopen = self.params['rhopen']
            self.grhopen = self.params['grhopen']
            #
            # Compute fluxes on subdomains.
            # Vsds is a list of length 2**dim, the value of V on each
            # subdomain.
            #
            self.Vsds = []
            for Usd,rhosd in zip(zip(*self.Usds), self.rhosds):
                self.Vsds.append(self.V(Usd, rhosd))
            #
            # I may need to adjust the signs of the subdomain vs by
            # the symmetries of the combinations
            #
            self.vsds = [-ufl.grad(Vsd) - (
                self.s2*ufl.grad(rhosd)/ufl.max_value(rhosd, self.rho_min)
            ) for Vsd,rhosd in zip(self.Vsds, self.rhosds)]
            self.fluxsds = [vsd * rhosd for vsd,rhosd in
                            zip(self.vsds, self.rhosds)]
            self.vnsds = [ufl.max_value(ufl.dot(vsd, self.n), 0)
                          for vsd in self.vsds]
            self.facet_fluxsds = [(
                vnsd('+')*ufl.max_value(rhosd('+'), 0.0) -
                vnsd('-')*ufl.max_value(rhosd('-'), 0.0)
            ) for vnsd,rhosd in zip(self.vnsds, self.rhosds)]
            #
            # Now combine the subdomain fluxes to get the fluxes for
            # the symmetrized functions
            #
            self.fluxs = matmul((2.0**-self.dim)*self.eomat,
                                self.fluxsds)
            self.facet_fluxs = matmul((2.0**-self.dim)*self.eomat,
                                      self.facet_fluxsds)
            self.rho_flux_jump = vectotal(
                [-facet_flux*ufl.jump(wrho)*self.dS
                 for facet_flux,wrho in
                 zip(self.facet_fluxs, self.wrhos)]
            )
            self.rho_grad_move = vectotal(
                [ufl.dot(flux, ufl.grad(wrho))*self.dx
                 for flux,wrho in
                 zip(self.fluxs, self.wrhos)]
            )
            self.rho_penalty = vectotal(
                [-(self.rhopen * self.degree**2 / self.havg) *
                 ufl.dot(ufl.jump(rho, self.n),
                        ufl.jump(wrho, self.n)) * self.dS
                 for rho,wrho in zip(self.irhos, self.wrhos)]
            )
            self.grho_penalty = vectotal(
                [-self.grhopen * self.degree**2 *
                 (ufl.jump(ufl.grad(rho), self.n) *
                  ufl.jump(ufl.grad(wrho), self.n)) * self.dS
                 for rho,wrho in zip(self.irhos, self.wrhos)]
            )
            self.rho_terms = (
                self.rho_flux_jump + self.rho_grad_move +
                self.rho_penalty + self.grho_penalty
            )
        if not hasattr(self, 'U_terms'):
            self.U_min = self.params['U_min']
            self.Upen = self.params['Upen']
            self.gUpen = self.params['gUpen']
            self.U_decay = 0.0
            self.U_secretion = 0.0
            self.jump_gUw = 0.0
            self.U_diffusion = 0.0
            self.U_penalty = 0.0
            self.gU_penalty = 0.0
            for j,lig in enumerate(self.ligands.ligands()):
                sl = slice(j*2**self.dim, (j+1)*2**self.dim)
                self.U_decay += -lig.gamma * sum(
                    [iUi * wUi * self.dx for
                     iUi,wUi in
                     zip(self.iUs[sl], self.wUs[sl])]
                )
                self.U_secretion += lig.s * sum(
                    [rho * wU * self.dx
                     for rho, wU in zip(self.irhos, self.wUs[sl])]
                )
                self.jump_gUw += lig.D * sum(
                    [ufl.jump(wU * ufl.grad(U), self.n) * self.dS
                     for wU,U in zip(self.wUs[sl], self.iUs[sl])
                    ]
                )
                self.U_diffusion += -lig.D * sum(
                    [ufl.dot(ufl.grad(U), ufl.grad(wU))*self.dx
                     for U,wU in zip(self.iUs[sl], self.wUs[sl])]
                )
                self.U_penalty += -self.Upen * self.degree**2 * sum(
                    [(1.0 / self.havg) *
                    ufl.dot(ufl.jump(U, self.n), ufl.jump(wU, self.n))*self.dS
                    for U,wU in zip(self.iUs[sl], self.wUs[sl])]
                )
                self.gU_penalty += -self.gUpen * self.degree**2 * sum(
                    [ufl.jump(ufl.grad(U), self.n) *
                     ufl.jump(ufl.grad(wU), self.n) * self.dS
                     for U,wU in zip(self.iUs[sl], self.wUs[sl])]
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
        #     self.JU_terms = [fe.derivative(self.all_terms, U)
        #                      for U in self.Us]
        # if not hasattr(self, 'Jrho_terms'):
        #     self.Jrho_terms = [fe.derivative(self.all_terms, rho)
        #                        for rho in self.rhos]


    def ddt(self, debug=False):
        """Calculate time derivative of rho and U

        Results are left in self.dsol as a two-component vector function.
        """
        self.setup_problem(debug)
        self.b = fe.assemble(self.all_terms)
        for bc in self.bcs:
            bc.apply(self.b)
        return fe.solve(self.A, self.dsol.vector(), self.b,
                        self.solver_type)
