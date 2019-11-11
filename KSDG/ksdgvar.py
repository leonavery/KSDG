"""Functions for discontinuous Galerkin solution of the Keller-Segel PDEs.

This module defines KSDGSolverVariable, an extension of
KSDGSolverMultiple that allows for time-varying parameters. Most of
the real model parameters are allowed to vary. The parameters argument
to the constructor now supplies just the initial values of the
parameters. Any parameter in this mappable whose value is of type
float will be treated as a model variable that can vary with
time. Most of these parameters have to do with the computation of the
potential V, a function that the caller also supplies. Therefore, it
is mostly up to the caller to determine what the parameters are. There
are a few parameters that KSDGSolverVariable itself uses directly:
sigma, rhomin, Umin, width, rhopen, Upen, grhopen, and gUpen. If not
supplied by the caller, KSDGSolverVariable has defaults for these
parameters. Although KSDGSolverVariable will treat all of them as
time-varying, this probably makes sense only for sigma. (Note also
that a time-varying width will not actually result in a time-varying
domain size. The domain geometry is fixed at initialization.) 

To accomodate the time variation, the constructor takes a
new argument, param_funcs. (It is the presence of this argument that
signals to makeKSDGSolver that it is to create a KSDGSolverVariable
object.)

    param_funcs={}:
    This is a mappable that contains functions to compute the values
    of the parameters as a function of time. Each such function is
    called with two arguments, the time and a mappable whose keys are
    the names of the other parameters. For instance, it could be
    param_funcs={
        'sigma': lambda t, params={}: sigma0 * ufl.exp(-dsigma * t),
        'beta,2': lambda t, params={}: beta0 * ufl.exp(-dbeta * t)
    }
    The first argument to the function is the time. It will also be
    passed a mappable giving the values of the other parameters. The
    time will also be passed as params['t']. As in the example, you
    may ignore the params argument if you don't need it (However, your
    function definition must accept this argument.)

    The values of the parameter and the values in the mappable params
    will not always be floats, even though they represent real
    numbers. In fact, they may be Indexed UFL objects that
    reference the parameter subspace. You should therefore use UFl
    functions (e.g., ufl.exp in the example).

    The KSDGSolverVariable class supplies a default function for any
    parameters whose time derivatives are not specified in param_funcs
    that will always return the initial value of the parameter. To
    force all parameters to be fixed in time, pass param_funcs={}.

The function V passed as an argument to the constructor must also
accept a keyword argument params whose value is a mappable specifying
the parameter values (generally UFL objects). Typically you will not
be able to get away with ignoring this argument, since some of these
parameters affect the computation of the potential.

Functions and FunctionsSpaces:
As always, the solution is held in a fenics Function ksdg.sol (where
ksdg is the KSDGSolverVariable object). The FunctionSpace is
ksdg.VS. 

Parameter values are held in a distinct UFL object, ksdg.PSf, which is
created using:
    self.PSf = fe.Constant(params)
where params is an array of (float) parameter values. This object
looks like a FEniCS Function. For instance, repr(ksdg.PSf) is
    "Coefficient(FunctionSpace(None,
                 VectorElement(FiniteElement('Real', None, 0),
                 dim=32)), 0)"

This object, however, mostly doesn't work in the way a FEniCS Function
would. It has no vector or function_space attributes. You can create a
FunctionSpace that seems to be of the type that the object is built
on, using
    fe.FunctionSpace(None. fe.VectorElement(fe.FiniteElement('R',
                     None, 0), dim=32)
but the FunctionSpace so created is not functional. You can't create a
Function on it, nor do most of the other things you expect to do with
a FunctionSpace. 

Neverthless, the Constant object created as above can be indexed (the
easiest way to do this is to use fe.split(ksdg.PSf), which returns a
tuple of Indexed objects, one for each parameter. These objects can be
used in UFL expressions. Furthermore, their avleus can be changed
using
    ksdg.PSf.assign(fe.Constant(newvalues))
where newvalues is an array of the same dimension as the original
parameter list. These new values will then be used in any form that
has been created referencing the Indexed values, without need for
recomplication. The values (which can be retrieved using
ksdg.PSf.values()) are held locally, so that they can be refern=enced
inexpensively by all processes in MPI execuation. 
"""

import sys
import numpy as np
import itertools
import collections
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
                    TrialFunctions, TestFunctions, Mesh, File,
                    Expression, FunctionSpace, VectorFunctionSpace,
                    TrialFunction, TestFunction, Function, Constant,
                    Measure, FacetNormal, CellDiameter, PETScVector,
                    PETScMatrix, SubDomain, MeshFunction,
                    SubMesh, DirichletBC, Assembler)

from .ksdgsolver import (KSDGSolver, meshMakers, boxMeshMakers,
                         cellShapes, unit_mesh, box_mesh, shapes)
from .ksdgmultiple import KSDGSolverMultiple
from .ksdgdebug import log
from .ksdgexception import KSDGException
from .ksdggather import (gather_dof_coords, gather_vertex_coords,
                         gather_mesh, coord_remap, bcast_function,
                         distribute_mesh, function_interpolate,
                         local_Rdofs)
from .ksdgligand import ParameterList
from .ksdgperiodic import (_cstats, mesh_stats, CornerDomain,
                         corner_submesh, evenodd_symmetries,
                         evenodd_matrix, matmul, vectotal,
                         evenodd_functions, FacesDomain,
                         KSDGSolverPeriodic)

def logVARIABLE(*args, **kwargs):
    log(*args, system='VARIABLE', **kwargs)

def logPERIODIC(*args, **kwargs):
    log(*args, system='PERIODIC', **kwargs)

class KSDGSolverVariable(KSDGSolverMultiple):
    default_params = collections.OrderedDict(
        sigma = 1.0,
        rhomin = 1e-7,
        Umin = 1e-7,
        width = 1.0,
        rhopen = 10.0,
        Upen = 1.0,
        grhopen = 1.0,
        gUpen = 1.0,
    )

    def __init__(
            self,
            mesh=None,
            width=1.0,
            dim=1,
            nelements=8,
            degree=2,
            parameters={},
            param_funcs={},
            V=(lambda U, params={}: sum(U)),
            U0=[],
            rho0=None,
            t0=0.0,
            debug=False,
            solver_type = 'petsc',
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
        parameters={}: a dict giving the initial values of scalar
            parameters of .V, U0, and rho0 Expressions. This dict
            needs to also define numerical parameters that appear in
            the PDE. Some of these have defaults: dim = dim: # of
            spatial dimensions sigma: organism movement rate
            rhomin=10.0**-7: minimum feasible worm density
            Umin=10.0**-7: minimum feasible attractant concentration
            rhopen=10: penalty for discontinuities in rho Upen=1:
            penalty for discontinuities in U grhopen=1, gUpen=1:
            penalties for discontinuities in gradients nligands=1,
            number of ligands.
        V=(lambda Us, params={}: sum(Us)): a callable taking two
            arguments, Us and rho, or a single argument, Us. Us is a
            list of length nligands. rho is a single expression. V
            returns a single number, V, the potential corresponding to
            Us (and rho). Use ufl versions of mathematical functions,
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
        logVARIABLE('creating KSDGSolverVariable')
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
            param_funcs=param_funcs,
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
        self.t0 = t0
        self.debug = debug
        self.solver_type = solver_type
        self.preconditioner_type = preconditioner_type
        self.periodic = False
        self.ligands = ligands
        self.nligands = ligands.nligands()
        self.init_params(parameters, param_funcs)
        if (mesh):
            self.omesh = self.mesh = mesh
        else:
            self.omesh = self.mesh = box_mesh(width=width, dim=dim,
                                              nelements=nelements)
            self.nelements = nelements
        logVARIABLE('self.mesh', self.mesh)
        logVARIABLE('self.mesh.mpi_comm().size', self.mesh.mpi_comm().size)
        self.nelements = nelements
        self.degree = degree
        self.dim = self.mesh.geometry().dim()
        # 
        # Solution spaces and Functions
        # 
        fss = self.make_function_space()
        (self.SE, self.SS, self.VE, self.VS) = [
            fss[fs] for fs in ('SE', 'SS', 'VE', 'VS')
        ]
        logVARIABLE('self.VS', self.VS)
        self.sol = Function(self.VS)              # sol, current soln
        logVARIABLE('self.sol', self.sol)
        splitsol = self.sol.split()
        self.srho, self.sUs = splitsol[0], splitsol[1:]
        splitsol = list(fe.split(self.sol))
        self.irho, self.iUs = splitsol[0], splitsol[1:]
        self.iPs = list(fe.split(self.PSf))
        self.iparams = collections.OrderedDict(
            zip(self.param_names, self.iPs)
        )
        self.iligands = copy.deepcopy(self.ligands)
        self.iligand_params = ParameterList([
            p for p in self.iligands.params()
            if p[0] in self.param_numbers
        ])
        for k in self.iligand_params.keys():
            i = self.param_numbers[k]
            self.iligand_params[k] = self.iPs[i]
        tfs = list(TestFunctions(self.VS))
        self.wrho, self.wUs = tfs[0], tfs[1:]
        tfs = list(TrialFunctions(self.VS))
        self.tdrho, self.tdUs = tfs[0], tfs[1:]
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
            V(self.iUs, self.irho, params=self.iparams)
            def realV(Us, rho):
                return V(Us, rho, params=self.iparams)
        except TypeError:
            def realV(Us, rho):
                return V(Us, self.iparams)
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
        self.set_time(t0)
        #
        # initialize state
        #
        self.restart()
        return None

    def init_params(self, parameters, param_funcs):
        """Initialize parameter attributes from __init__ arguments

        The attributes initialized are:
        self.params0: a dict giving initial values of all parameters
        (not just floats). This is basically a copy of the parameters
        argument to __init__, with the insertion of 't' as a new
        parameter (always param_names[-1]).
        self.param_names: a list of the names of the time-varying
        parameters. This is the keys of params0 whose corrsponding
        values are of type float. The order is the order of the
        parameters in self.PSf.
        self.nparams: len(self.param_names)
        self.param_numbers: a dict mapping param names to numbers
        (ints) in the list param_names and the parameters subspace of
        the solution FunctionSpace.
        self.param_funcs: a dict whose keys are the param_names and
        whose values are functions to determine their values as a
        function of time, as explained above. These are copied from
        the param_funcs argument of __init__, except that the default
        initial value function is filled in for parameters not present
        in the argument. Also, the function defined for 't' always
        returns t.
        self.PSf: a Constant object of dimension self.nparams, holding
        the initial values of the parameters.
        """
        self.param_names = [
            n for n,v in parameters.items() if
            (type(v) is float and n != 't')
        ]
        self.param_names.append('t')
        self.nparams = len(self.param_names)
        logVARIABLE('self.param_names', self.param_names)
        logVARIABLE('self.nparams', self.nparams)
        self.param_numbers = collections.OrderedDict(
            zip(self.param_names, itertools.count())
        )
        self.params0 = collections.OrderedDict(parameters)
        self.params0['t'] = 0.0
        self.param_funcs = param_funcs.copy()
        def identity(t, params={}):
            return t

        self.param_funcs['t'] = identity
        for n in self.param_names:
            if n not in self.param_funcs:
                def value0(t, params={}, v0=self.params0[n]):
                    return v0

                self.param_funcs[n] = value0
        self.PSf = Constant([
            self.params0[n] for n in self.param_names
        ])
        return

    def set_time(self, t):
        self.t = t
        params = collections.OrderedDict(
            zip(self.param_names, self.PSf.values())
        )
        self.PSf.assign(Constant([
                self.param_funcs[n](t, params=params)
                for n in self.param_names
        ]))
        logVARIABLE('self.t', self.t)
        logVARIABLE(
            'collections.OrderedDict(zip(self.param_names, self.PSf.values()))',
            collections.OrderedDict(zip(self.param_names, self.PSf.values()))
        )

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
        logVARIABLE('restart')
        self.set_time(self.t0)
        CE = FiniteElement('CG', cellShapes[self.dim-1], self.degree)
        CS = FunctionSpace(self.mesh, CE)   # scalar space
        coords = gather_dof_coords(CS)
        fe.assign(self.sol.sub(0),
                  function_interpolate(self.rho0, self.SS,
                                       coords=coords))
        for i,U0i in enumerate(self.U0s):
            fe.assign(self.sol.sub(i+1),
                      function_interpolate(U0i, self.SS,
                                           coords=coords))

        
    def setup_problem(self, t, debug=False):
        self.set_time(t)
        #
        # assemble the matrix, if necessary (once for all time points)
        #
        if not hasattr(self, 'A'):
            self.drho_integral = self.tdrho*self.wrho*self.dx
            self.dU_integral = sum(
                [tdUi*wUi*self.dx for tdUi,wUi in zip(self.tdUs, self.wUs)]
            )
            logVARIABLE('assembling A')
            self.A = PETScMatrix()
            logVARIABLE('self.A', self.A)
            fe.assemble(self.drho_integral + self.dU_integral,
                        tensor=self.A)
            logVARIABLE('A assembled. Applying BCs')
            self.dsol = Function(self.VS)
            dsolsplit = self.dsol.split()
            self.drho, self.dUs = dsolsplit[0], dsolsplit[1:]
        #
        # assemble RHS (for each time point, but compile only once)
        #
        if not hasattr(self, 'rho_terms'):
            self.sigma = self.iparams['sigma']
            self.s2 = self.sigma * self.sigma / 2
            self.rhomin = self.iparams['rhomin']
            self.rhopen = self.iparams['rhopen']
            self.grhopen = self.iparams['grhopen']
            self.v = -ufl.grad(self.V(self.iUs, 
                                      ufl.max_value(self.irho,
                                                    self.rhomin)) - (
                self.s2*ufl.grad(self.irho)/ufl.max_value(self.irho,
                                                          self.rhomin) 
            ))
            self.flux = self.v * self.irho
            self.vn = ufl.max_value(ufl.dot(self.v, self.n), 0)
            self.facet_flux = ufl.jump(self.vn*ufl.max_value(self.irho, 0.0))
            self.rho_flux_jump = -self.facet_flux*ufl.jump(self.wrho)*self.dS
            self.rho_grad_move = ufl.dot(self.flux,
                                         ufl.grad(self.wrho))*self.dx
            self.rho_penalty = -(
                (self.degree**2 / self.havg) *
                ufl.dot(ufl.jump(self.irho, self.n),
                        ufl.jump(self.rhopen*self.wrho, self.n)) * self.dS
            )
            self.grho_penalty = -(
                self.degree**2 *
                (ufl.jump(ufl.grad(self.irho), self.n) *
                 ufl.jump(ufl.grad(self.grhopen*self.wrho), self.n)) *
                self.dS
            )
            self.rho_terms = (
                self.rho_flux_jump + self.rho_grad_move +
                self.rho_penalty + self.grho_penalty
            )
        if not hasattr(self, 'U_terms'):
            self.Umin = self.iparams['Umin']
            self.Upen = self.iparams['Upen']
            self.gUpen = self.iparams['gUpen']
            self.U_decay = sum(
                [-lig.gamma * iUi * wUi * self.dx for
                 lig,iUi,wUi in
                 zip(self.iligands.ligands(), self.iUs, self.wUs)]
            )
            self.U_secretion = sum(
                [lig.s * self.irho * wUi * self.dx for
                 lig,wUi in zip(self.iligands.ligands(), self.wUs)]
            )
            self.jump_gUw = sum(
                [ufl.jump(lig.D * wUi * ufl.grad(iUi), self.n) * self.dS
                for lig,wUi,iUi in
                zip(self.iligands.ligands(), self.wUs, self.iUs)]
            )
            self.U_diffusion = sum(
                [-lig.D * ufl.dot(ufl.grad(iUi), ufl.grad(wUi))*self.dx for
                 lig,iUi,wUi in
                 zip(self.iligands.ligands(), self.iUs, self.wUs)]
            )
            self.U_penalty = sum(
                [-(self.degree**2/self.havg) *
                 ufl.dot(ufl.jump(iUi, self.n),
                         ufl.jump(self.Upen*wUi, self.n))*self.dS for
                 iUi,wUi in zip(self.iUs, self.wUs)]
            )
            self.gU_penalty = sum(
                [-self.degree**2 * 
                 ufl.jump(ufl.grad(iUi), self.n) *
                 ufl.jump(ufl.grad(self.gUpen*wUi), self.n) * self.dS for
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

    def ddt(self, t, debug=False):
        """Calculate time derivative of rho and U

        Results are left in self.dsol as a two-component vector function.
        """
        self.setup_problem(t, debug=debug)
        self.b = fe.assemble(self.all_terms)
        return fe.solve(self.A, self.dsol.vector(), self.b,
                        self.solver_type)

class KSDGSolverVariablePeriodic(KSDGSolverVariable, KSDGSolverPeriodic):
    default_params = collections.OrderedDict(
        sigma = 1.0,
        rhomin = 1e-7,
        Umin = 1e-7,
        width = 1.0,
        rhopen = 10.0,
        Upen = 1.0,
        grhopen = 1.0,
        gUpen = 1.0,
    )

    def __init__(
            self,
            mesh=None,
            width=1.0,
            dim=1,
            nelements=8,
            degree=2,
            parameters={},
            param_funcs={},
            V=(lambda U, params={}: sum(U)),
            U0=[],
            rho0=None,
            t0=0.0,
            debug=False,
            solver_type = 'petsc',
            preconditioner_type = 'default',
            periodic=True,
            ligands=None
            ):
        """Discontinuous Galerkin solver for the Keller-Segel PDE system

        Like KSDGSolverVariable, but with periodic boundary conditions.
        """
        logVARIABLE('creating KSDGSolverVariablePeriodic')
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
            param_funcs=param_funcs,
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
        self.t0 = t0
        self.debug = debug
        self.solver_type = solver_type
        self.preconditioner_type = preconditioner_type
        self.periodic = True
        self.ligands = ligands
        self.nligands = ligands.nligands()
        self.init_params(parameters, param_funcs)
        if nelements is None:
            self.nelements = 8
        else:
            self.nelements = nelements
        if (mesh):
            self.omesh = self.mesh = mesh
        else:
            self.omesh = self.mesh = box_mesh(width=width, dim=dim,
                                              nelements=self.nelements)
            self.nelements = nelements
        omeshstats = mesh_stats(self.omesh)
        try:
            comm = self.omesh.mpi_comm().tompi4py()
        except AttributeError:
            comm = self.omesh.mpi_comm()
        self.lmesh = gather_mesh(self.omesh)
        logVARIABLE('omeshstats', omeshstats)
        self.xmin = omeshstats['xmin']
        self.xmax = omeshstats['xmax']
        self.xmid = omeshstats['xmid']
        self.delta_ = omeshstats['dx']
        if nelements is None:
            self.nelements = (self.xmax - self.xmin) / self.delta_
        self.mesh = corner_submesh(self.lmesh)
        meshstats = mesh_stats(self.mesh)
        self.degree = degree
        self.dim = self.mesh.geometry().dim()
        # 
        # Solution spaces and Functions
        # 
        self.symmetries = evenodd_symmetries(self.dim)
        self.signs = [fe.as_matrix(np.diagflat(1.0 - 2.0*eo))
                      for eo in self.symmetries]
        self.eomat = evenodd_matrix(self.symmetries)
        fss = self.make_function_space()
        (self.SE, self.SS, self.VE, self.VS) = [
            fss[fs] for fs in ('SE', 'SS', 'VE', 'VS')
        ]
        logVARIABLE('self.VS', self.VS)
        self.sol = Function(self.VS)              # sol, current soln
        logVARIABLE('self.sol', self.sol)
        splitsol = self.sol.split()
        self.srhos = splitsol[:2**self.dim]
        self.sUs = splitsol[2**self.dim:]
        splitsol = list(fe.split(self.sol))
        self.irhos = splitsol[:2**self.dim]
        self.iUs = splitsol[2**self.dim:]
        self.iPs = list(fe.split(self.PSf))
        self.iparams = collections.OrderedDict(
            zip(self.param_names, self.iPs)
        )
        self.iligands = copy.deepcopy(self.ligands)
        self.iligand_params = ParameterList([
            p for p in self.iligands.params()
            if p[0] in self.param_numbers
        ])
        for k in self.iligand_params.keys():
            i = self.param_numbers[k]
            self.iligand_params[k] = self.iPs[i]
        tfs = list(TestFunctions(self.VS))
        self.wrhos, self.wUs = tfs[:2**self.dim], tfs[2**self.dim:]
        tfs = list(TrialFunctions(self.VS))
        self.tdrhos, self.tdUs = tfs[:2**self.dim], tfs[2**self.dim:]
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
        self.set_time(t0)
        #
        # work out how to call V
        #
        try:
            V(self.U0s, self.rho0, params=self.iparams)
            def realV(Us, rho):
                return V(Us, rho, params=self.iparams)
        except TypeError:
            def realV(Us, rho):
                return V(Us, self.iparams)
        self.V = realV
        #
        # initialize state
        #
        self.restart()
        return None

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
        elements = [SE] * ((self.nligands+1)*2**self.dim)
        VE = MixedElement(elements)
        VS = FunctionSpace(mesh, VE)   # vector space
        return dict(SE=SE, SS=SS, VE=VE, VS=VS)

    def restart(self):
        logVARIABLE('restart')
        self.set_time(self.t0)
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
        
    def setup_problem(self, t, debug=False):
        self.set_time(t)
        #
        # assemble the matrix, if necessary (once for all time points)
        #
        if not hasattr(self, 'A'):
            logVARIABLE('making matrix A')
            self.drho_integral = sum(
                [tdrho*wrho*self.dx for tdrho,wrho in
                 zip(self.tdrhos, self.wrhos)]
            )
            self.dU_integral = sum(
                [tdU*wU*self.dx
                 for tdU,wU in zip(self.tdUs, self.wUs)
                ]
            )
            logVARIABLE('assembling A')
            self.A = fe.PETScMatrix()
            logVARIABLE('self.A', self.A)
            fe.assemble(self.drho_integral + self.dU_integral,
                        tensor=self.A)
            logVARIABLE('A assembled. Applying BCs')
            pA = fe.as_backend_type(self.A).mat()
            Adiag = pA.getDiagonal()
            logVARIABLE('Adiag.array', Adiag.array)
            # self.A = fe.assemble(self.drho_integral + self.dU_integral +
            #                      self.dP_integral)
            for bc in self.bcs:
                bc.apply(self.A)
            Adiag = pA.getDiagonal()
            logVARIABLE('Adiag.array', Adiag.array)
            self.dsol = Function(self.VS)
            dsolsplit = self.dsol.split()
            self.drhos, self.dUs = (dsolsplit[:2**self.dim],
                                   dsolsplit[2**self.dim:])
        #
        # assemble RHS (for each time point, but compile only once)
        #
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
        if not hasattr(self, 'rho_terms'):
            logVARIABLE('making rho_terms')
            self.sigma = self.iparams['sigma']
            self.s2 = self.sigma * self.sigma / 2
            self.rhomin = self.iparams['rhomin']
            self.rhopen = self.iparams['rhopen']
            self.grhopen = self.iparams['grhopen']
            #
            # Compute fluxes on subdomains.
            # Vsds is a list of length 2**dim, the value of V on each
            # subdomain.
            #
            self.Vsds = []
            for Usd,rhosd in zip(zip(*self.Usds), self.rhosds):
                self.Vsds.append(self.V(Usd, ufl.max_value(
                    rhosd, self.rhomin)))
            self.vsds = [-ufl.grad(Vsd) - (
                self.s2*ufl.grad(rhosd)/ufl.max_value(rhosd, self.rhomin)
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
            self.rho_flux_jump = sum(
                [-facet_flux*ufl.jump(wrho)*self.dS
                 for facet_flux,wrho in
                 zip(self.facet_fluxs, self.wrhos)]
            )
            self.rho_grad_move = sum(
                [ufl.dot(flux, ufl.grad(wrho))*self.dx
                 for flux,wrho in
                 zip(self.fluxs, self.wrhos)]
            )
            self.rho_penalty = sum(
                [-(self.degree**2 / self.havg) *
                 ufl.dot(ufl.jump(rho, self.n),
                        ufl.jump(self.rhopen*wrho, self.n)) * self.dS
                 for rho,wrho in zip(self.irhos, self.wrhos)]
            )
            self.grho_penalty = sum(
                [self.degree**2 *
                 (ufl.jump(ufl.grad(rho), self.n) *
                  ufl.jump(ufl.grad(-self.grhopen*wrho), self.n)) * self.dS
                 for rho,wrho in zip(self.irhos, self.wrhos)]
            )
            self.rho_terms = (
                self.rho_flux_jump + self.rho_grad_move +
                self.rho_penalty + self.grho_penalty
            )
            logVARIABLE('rho_terms made')
        if not hasattr(self, 'U_terms'):
            logVARIABLE('making U_terms')
            self.Umin = self.iparams['Umin']
            self.Upen = self.iparams['Upen']
            self.gUpen = self.iparams['gUpen']
            self.U_decay = 0.0
            self.U_secretion = 0.0
            self.jump_gUw = 0.0
            self.U_diffusion = 0.0
            self.U_penalty = 0.0
            self.gU_penalty = 0.0
            for j,lig in enumerate(self.iligands.ligands()):
                sl = slice(j*2**self.dim, (j+1)*2**self.dim)
                self.U_decay += sum(
                    [-lig.gamma * iUi * wUi * self.dx for
                     iUi,wUi in
                     zip(self.iUs[sl], self.wUs[sl])]
                )
                self.U_secretion += sum(
                    [lig.s * rho * wU * self.dx
                     for rho, wU in zip(self.irhos, self.wUs[sl])]
                )
                self.jump_gUw += sum(
                    [ufl.jump(lig.D * wU * ufl.grad(U), self.n) * self.dS
                     for wU,U in zip(self.wUs[sl], self.iUs[sl])
                    ]
                )
                self.U_diffusion += sum(
                    [-lig.D * ufl.dot(ufl.grad(U), ufl.grad(wU))*self.dx
                     for U,wU in zip(self.iUs[sl], self.wUs[sl])]
                )
                self.U_penalty += sum(
                    [(-self.degree**2 / self.havg) *
                    ufl.dot(ufl.jump(U, self.n),
                            ufl.jump(self.Upen*wU, self.n))*self.dS
                    for U,wU in zip(self.iUs[sl], self.wUs[sl])]
                )
                self.gU_penalty += sum(
                    [-self.degree**2 *
                     ufl.jump(ufl.grad(U), self.n) *
                     ufl.jump(ufl.grad(self.gUpen*wU), self.n) * self.dS
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
            logVARIABLE('U_terms made')
        if not hasattr(self, 'all_terms'):
            logVARIABLE('making all_terms')
            self.all_terms = self.rho_terms + self.U_terms
        if not hasattr(self, 'J_terms'):
            logVARIABLE('making J_terms')
            self.J_terms = fe.derivative(self.all_terms, self.sol)

    def ddt(self, t, debug=False):
        """Calculate time derivative of rho and U

        Results are left in self.dsol as a two-component vector function.
        """
        self.setup_problem(t, debug=debug)
        self.b = fe.assemble(self.all_terms)
        for bc in self.bcs:
            bc.apply(self.b)
        return fe.solve(self.A, self.dsol.vector(), self.b,
                        self.solver_type)

