"""DG solution of the Keller-Segel PDEs with periodic boundary conditions.

This module defines a class, KSDGSolverPeriodic, that handles the
"periodic=True" option of KSDGSolver.  My original intention was to do
this just by using the constrained_domain argument of FunctionSpace. I
then learned that periodic boundary conditions don't work with DG
FunctionsSpaces in the current version of FeniCS
(https://bitbucket.org/fenics-project/dolfin/issues/558/periodic-boundary-conditions-for-dg).

I therefore implemented periodic boundary conditions in a more
labor-intensive way. This is most easily explained first in one
dimension. Suppose we have a PDE system in rho and U on [0, w], an dI
want solutions that are periodic. The straightforward interpretation
of that requirement is to impose boundary conditions U(0) = U(w) and
U'(0) = U'(w) (with U' standing for the x-derivative), and likewise
for rho. But I impose a slightly stronger condition: that U is the
restriction to [0, w] of a function with period w, with as many
continuous derivatives as convenient. I then define odd and even
components of U:

Ue(x) = (U(x) + U(-x))/2
Uo(x) = (U(x) - U(-x))/2

whence:

U(x) = Ue(x) + Uo(x)
U(-x) = Ue(x) - Uo(x)

Now, it is easy to see from these definitions and the periodicity of U
(and hence of Ue and Uo) that Uo satisfies Dirichlet boundary
conditions on [0, w/2], and Ue Neumann boundary conditions on [0,
w/2]. 

Uo(0) = Uo(w/2) = 0
Ue'(0) = Ue'(w/2) = 0

In fact, although it is a little more work to prove it, Uo is odd not
just around 0, but also around w/2. Likewise, Ue is even about w/2. Of
course, I can write PDEs for Ue and Uo based on that for U. It turns
out to be essentially the same equation, since all the terms in this
PDe are linear and first-order. The PDEs for rho_e and rho_o are more
complicated, since the advection term is nonlinear.

So I end up with a system of PDEs in four functions, Ue, Uo, rho_e,
rho_o. However, I need only solve this on [0, w/2], i.e. a domain half
the size of the original. The effect is that, for the same spatial
resolution, the number of degrees of freedom is essentially
unchanged. The Neumann boundary condition above is imposed on Ue (at 0
and w/2) and the Dirichlet conditions on Uo (again at 0 and
w/2). Thus, this separation of the function into even and odd parts
replaces the periodic boundary conditions that FeniCS currently can't
handle with Neumann and Dirichlet boundary conditions that it can. The
function on [0, w] is then reconstructed from the half-functions using

Ue(x) = Ue(w - x)     for x in [w/2, w]
Uo(x) = -Uo(w - x)    for x in (w/2, w)
U(x) = Ue(x) + Uo(x)  for x in [0, w]

Ue and Uo satisfy PDEs that can be derived from the PDE for U. In
fact, since that PDE is linear and homogeneous, the PDEs for Ue and Uo
end up being identical to that for U. For rho the situation is more
complicated. The nonlinear advection term splits up into a sum of two
terms, one for x and one for -x. By the TANSTAAFL principle, it
becomes necessary to compute an upwind flux for each of these. The
weak form for rho_e an drho_o are sums adn differences of these upwind
fluxes. 

All of this works similarly in two dimensions. There is, however, one
important change. U needs to be split into not just 2, but 4
functions, Uee, Ueo, Uoe, and Uoo, where the first subscript
designates even or odd in the x dimension, and the second even or odd
in the y dimension. The PDEs are solved on [0, w/2]^2. So, once again,
we conserve the number of DOFs: having four functions on a domain 1/4
the size. The boundary cinditions match the symmetry of the
function. For instance, Ueo satisfies Neumann boundary conditions at x
= 0 and x = w/2, and Dirichlet boundary conditions at y = 0 and y =
w/2. Once again all four of these functions satisfy the same PDE as U
itself, but four different upwind fluxes have to be computed and
combined for the four components of rho. 

In three dimension, of course, each function splits into 8 component
functions, and the PDEs are solved on [0, w/2]^3.
"""

import sys
import numpy as np
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
                    TestFunctions, Mesh,
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
from .ksdggather import gather_dof_coords, coord_remap, bcast_function

def logPERIODIC(*args, **kwargs):
    log(*args, system='PERIODIC', **kwargs)


def _cstats(coords, d, tol=fe.DOLFIN_EPS):
    """Find statistics along one dimension"""
    xs = np.unique(coords[:, d])
    xmin = np.min(xs)
    xmax = np.max(xs)
    xtol = tol*(xmax - xmin)
    if abs(xmin) > xtol:
        raise KSDGException(
            "The origin is not a corner of the mesh."
        )
    dxs = np.unique(np.diff(xs))
    dx = np.mean(dxs)
    if np.max(np.abs(dxs - dx)) > xtol:
        raise KSDGException(
            "vertexes not equally spaced in dimension %d"%d
            )
    xmid = (xmin + xmax)/2
    if np.min(np.abs(xs - xmid)) > xtol:
        raise KSDGException(
            "midpoint not present in dimension %d"%d
            )
    return dict(
        xmin = xmin,
        xmax = xmax,
        xmid = xmid,
        dx = dxs[0]
        )


def mesh_stats(mesh, tol=fe.DOLFIN_EPS):
    """Find the dimensions and midpoints of a mesh.

    Required argument: mesh

    The mesh should be on an interval, square, or cube. Vertex points
    should be equally spaced in each dimension, and the spacing should
    be the same in each. Finally, there should be a vertex at the
    midpoint of the domain. 

    meshstats returns a dict with keys xmin, xmax, xmid, dx. Each
    value is a single float number. 

    Exceptions:
    mesh_stats raises a KSDGException if it detects that the
    conditions on the mesh are not met.
    """
    dim = mesh.geometry().dim()
    if dim < 1 or dim > 3:
        raise KSDGException(
            "dimension of mesh = %d, must be 1, 2, or 3"%dim
            )
    cs = mesh.coordinates()
    stats = _cstats(cs, 0)
    for d in range(1, dim):
        dstats = _cstats(cs, d)
        if (not fe.near(stats['xmin'], dstats['xmin'], tol) or
            not fe.near(stats['xmax'], dstats['xmax'], tol) or
            not fe.near(stats['xmid'], dstats['xmid'], tol) or
            not fe.near(stats['dx'], dstats['dx'], tol)):
            raise KSDGException(
                "dimension %d spacing doesn't match dimension 0"%d
                )
    return(stats)


class CornerDomain(SubDomain):
    def __init__(
        self,
        mesh,
        xmin=None,
        xmid=None,
        tol=fe.DOLFIN_EPS
        ):
        super().__init__()
        self.dim = mesh.geometry().dim()
        self.tol = tol
        if (xmin is None and xmid is None):
            stats = mesh_stats(mesh, tol)
            self.xmin = stats['xmin']
            self.xmid = stats['xmid']
        else:
            self.xmin = xmin
            self.xmid = xmid
        self.lb = self.xmin - 2*self.tol*(self.xmid - self.xmin)
        self.ub = self.xmid + 2*self.tol*(self.xmid - self.xmin)

    def inside(self, x, on_boundary):
        for d in range(self.dim):
            if (x[d] < self.lb or x[d] > self.ub):
                return(False)
        return(True)


def corner_submesh(mesh):
    """create a submesh extending from the origin to the midpoint"""
    cf = MeshFunction("size_t", mesh, mesh.geometric_dimension())
    corner = CornerDomain(mesh)
    cf.set_all(0)
    corner.mark(cf, 1)
    submesh = SubMesh(mesh, cf, 1)
    return(submesh)

def evenodd_symmetries(dim):
    """Classify functions by even and odd symmetris
    
    Required parameter:
    dim: number of spatial dimensions
    
    Returns:
    A 2**dim x dim matrix of 0s and 1s specifying which dimensions of
    each component function are to be even and odd. (For instance,
    the row [0, 1, 0] requests a function that is even in
    dimensions 0 and 2 and odd in dimension 1. One function is
    returned for each row of evenodd.
    """
    return(np.array(list(
        itertools.product([0, 1], repeat=dim))
    ))

def evenodd_matrix(
    evenodd=None,
    dim=None
):
    """Find matrix to extract even and odd components
    Required parameter:
    evenodd=None: the matrix of 0s and 1s specifying which dimensions
        of each returned function are to be even and odd.
    dim=None: number of spatial dimensions

    At least one of evenodd or dim must be provided. If dim is not
    given evenodd.shape[1] is used. If evenodd is not provided, the
    matrix evenodd_symmetries(dim) is used.
    
    Returns:
    A 2**dim x 2**dim matrix whose componets are all either 1 or -1.
    This matrix E, multiplied by a vector, for instance
    [f(x, y), f(x, -y), f(-x, y), f(-x,-y)] produce functions that are even
    and odd in the dimensions as indiciated by evenodd.
    This matrix is essentially self-inverse, i.e., E.E = 2**dim I.

    """
    if (evenodd is None) and (not dim):
        raise KSDGException(
            "evenodd_matrix requires either evenodd or dim"
        )
    if evenodd is not None and dim:
        if evenodd.shape != (2**dim, dim):
            raise KSDGException(
                "dim doesn't match shape of evenodd"
            )
    elif evenodd is not None:
        dim = evenodd.shape[1]
    else:
        evenodd = evenodd_symmetries(dim)
    flips = evenodd_symmetries(dim)
    E = np.ones((2**dim, 2**dim), dtype=float)
    for r,eo in enumerate(evenodd):
        for c,flip in enumerate(flips):
            for d,s in enumerate(eo):
                if s != 0 and flip[d] != 0:
                    E[r, c] = -E[r, c]
    #
    # confirm that it's self-inverse
    #
    assert np.all((np.matmul(E, E) == 2** dim * np.identity(2**dim)))
    return(E)

def matmul(mat, vec):
    """Multiply a matrix by a vector
    
    This function does a boring multiplication of the matrix that is
    its first argument by the vector that is its second.  This really
    only makes sense when one or both of these is nonnumeric (if both
    are numeric, you should probably be using numpy). For instance,
    vec may be a list of FEniCS functions. This function requires only
    that the elements of mat be accessible as mat[row][col], those of
    vec of vec[col], and that mat elements can be multiplied by vec
    elements and those products added to each other.
    """
    m = len(mat)
    n = len(mat[0])
    assert len(vec) == n
    if m <= 0 or n <= 0:
        return(0.0)             # may not be the right type
    out = []
    for row in range(m):
        matrc = mat[row][0]
        if isinstance(matrc, np.generic):
            matrc = np.asscalar(matrc)
        vecc = vec[0]
        if isinstance(vecc, np.generic):
            vecc = np.asscalar(vecc)
        s = matrc * vecc
        # print(
        #     type(mat[row][0]),
        #     type(vec[0]),
        #     type(s)
        # )
        for col in range(1, n):
            matrc = mat[row][col]
            if isinstance(matrc, np.generic):
                matrc = np.asscalar(matrc)
                vecc = vec[col]
                if isinstance(vecc, np.generic):
                    vecc = np.asscalar(vecc)
            s += matrc * vecc
            # print(
            #     type(mat[row][col]),
            #     type(vec[col]),
            #     type(s)
            # )
        out.append(s)
    return(out)

def vectotal(vec):
    """Total a (possibly nonnumeric) vector"""
    sum = 0
    for c in vec:
        sum += c
    return(sum)

def evenodd_functions_old(
    omesh,
    degree,
    func,
    width=None,
    evenodd=None
):
    """Break a function into even and odd components

    Required parameters:
    omesh: the mesh on which the function is defined
    degree: the degree of the FunctionSpace 
    func: the Function. This has to be something that fe.interpolate
        can interpolate onto a FunctionSpace or that fe.project can
        project onto a FunctionSpace.
    width: the width of the domain on which func is defined. (If not
        provided, this will be determined from omesh.
    evenodd: the symmetries of the functions to be constructed
        evenodd_symmetries(dim) is used if this is not provided
    """
    SS = FunctionSpace(omesh, 'CG', degree)
    dim = omesh.geometry().dim()
    if width is None:
        stats = mesh_stats(omesh)
        width = stats['xmax']
    if evenodd is None:
        evenodd = evenodd_symmetries(dim)
    try:
        f0 = fe.interpolate(func, SS)
    except TypeError:
        f0 = fe.project(func, SS)
    ffuncs = []
    flips = evenodd_symmetries(dim)
    for flip in (flips):
        fmesh = Mesh(omesh)
        SSf = FunctionSpace(fmesh, 'CG', degree)
        ffunc = fe.interpolate(f0, SSf)
        fmesh.coordinates()[:, :] = (width*flip
                                     + (1 - 2*flip)*fmesh.coordinates())
        fmesh.bounding_box_tree().build(fmesh)
        ffuncs.append(ffunc)
    E = evenodd_matrix(evenodd)
    components = matmul(2**(-dim)*E, ffuncs)
    cs = []
    for c in components:
        try:
            cs.append(fe.interpolate(c, SS))
        except TypeError:
            cs.append(fe.project(c, SS, solver_type='lu'))
    return(cs)


def evenodd_functions(
    omesh,
    degree,
    func,
    width=None,
    evenodd=None
):
    """Break a function into even and odd components

    Required parameters:
    omesh: the mesh on which the function is defined
    degree: the degree of the FunctionSpace 
    func: the Function. This has to be something that fe.interpolate
        can interpolate onto a FunctionSpace or that fe.project can
        project onto a FunctionSpace.
    width: the width of the domain on which func is defined. (If not
        provided, this will be determined from omesh.
    evenodd: the symmetries of the functions to be constructed
        evenodd_symmetries(dim) is used if this is not provided
    """
    SS = FunctionSpace(omesh, 'CG', degree)
    dim = omesh.geometry().dim()
    comm = omesh.mpi_comm()
    rank = comm.rank
    if width is None:
        stats = mesh_stats(omesh)
        width = stats['xmax']
    if evenodd is None:
        evenodd = evenodd_symmetries(dim)
    try:
        f0 = fe.interpolate(func, SS)
    except TypeError:
        f0 = fe.project(func, SS)
    vec0 = f0.vector().gather_on_zero()
    dofcoords = gather_dof_coords(SS)
    if rank == 0:
        fvecs = np.empty((2**dim, len(vec0)), float)
        flips = evenodd_symmetries(dim)
        for row,flip in enumerate(flips):
            newcoords = (width*flip
                        + (1 - 2*flip)*dofcoords)
            remap = coord_remap(SS, newcoords)
            fvecs[row, :] = vec0[remap]
        E = evenodd_matrix(evenodd)
        components = np.matmul(2**(-dim)*E, fvecs)
    else:
        components = np.zeros((2**dim, len(vec0)), float)
    fs = []
    for c in components:
        f = bcast_function(SS, c)
        fs.append(f)
    return(fs)


class FacesDomain(SubDomain):
    def __init__(
        self,
        mesh,
        evenodd,
        tol=fe.DOLFIN_EPS
    ):
        """A domain that includes certain faces of a mesh.

        Required parameters:
        mesh: the mesh on which the domain is to be defined evenodd:
            the symmetries of the function. This is a list of 0's and
            1's. For each 1, the domain will include the faces at the
            ends of that dimension.
        """
        super().__init__()
        self.dim = mesh.geometry().dim()
        self.evenodd = evenodd
        assert len(self.evenodd) == self.dim
        self.tol = tol
        stats = mesh_stats(mesh, tol)
        self.xmin = stats['xmin']
        self.xmax = stats['xmax']

    def inside(self, x, on_boundary):
        for d,eo in enumerate(self.evenodd):
            if ((self.dim <= 1 or on_boundary) and eo 
                and (fe.near(x[d], self.xmin, self.tol) 
                     or fe.near(x[d], self.xmax, self.tol))):
                return(True)
        return(False)

class KSDGSolverPeriodic(KSDGSolver):
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
            solver_type = 'lu',
            preconditioner_type = 'default',
            periodic=True
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
        """
        logPERIODIC('creating KSDGSolverPeriodic')
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
            periodic=True
        )
        self.debug = debug
        self.solver_type = solver_type
        self.preconditioner_type = preconditioner_type
        self.periodic = True
        self.params = self.default_params.copy()
        #
        # Store the original mesh in self.omesh. self.mesh will be the
        # corner mesh.
        #
        if (mesh):
            self.omesh = mesh
        else:
            self.omesh = box_mesh(width=width, dim=dim, nelements=nelements)
            self.nelements = nelements
        meshstats = mesh_stats(mesh)
        self.xmin = meshstats['xmin']
        self.xmax = meshstats['xmax']
        self.xmid = meshstats['xmid']
        self.delta_ = meshstats['dx']
        self.mesh = corner_submesh(self.omesh)
        logPERIODIC('self.omesh', self.omesh)
        logPERIODIC('meshstats', meshstats)
        logPERIODIC('self.mesh', self.mesh)
        logPERIODIC('self.mesh.mpi_comm().size', self.mesh.mpi_comm().size)
        self.nelements = nelements
        self.degree = degree
        self.dim = self.mesh.geometry().dim()
        self.params['dim'] = self.dim
        self.params.update(parameters)
        # 
        # Solution spaces and Functions
        #
        # The solution function space is a vector space with
        # 2*(2**dim) elements. The first 2**dim components are even
        # and odd parts of rho; These are followed by even and
        # odd parts of U. The array self.evenodd identifies even
        # and odd components. Each row is a length dim sequence 0s and
        # 1s and represnts one component. For instance, if evenodd[i]
        # is [0, 1, 0], then component i of the vector space is even
        # in dimensions 0 and 2 (x and z conventionally) and off in
        # dimension 1 (y).
        #
        self.symmetries = evenodd_symmetries(self.dim)
        self.signs = [fe.as_matrix(np.diagflat(1.0 - 2.0*eo))
                      for eo in self.symmetries]
        self.eomat = evenodd_matrix(self.symmetries)
        self.SE = FiniteElement('DG', cellShapes[self.dim-1], self.degree)
        self.SS = FunctionSpace(self.mesh, self.SE)   # scalar space
        elements = [self.SE] * (2*2**self.dim)
        self.VE = MixedElement(elements)
        self.VS = FunctionSpace(self.mesh, self.VE)   # vector space
        logPERIODIC('self.VS', self.VS)
        self.sol = Function(self.VS)                  # sol, current soln
        logPERIODIC('self.sol', self.sol)
        # srhos and sUs are fcuntions defiend on subspaces
        self.srhos = self.sol.split()[:2**self.dim]
        self.sUs = self.sol.split()[2**self.dim:]
        # irhos and iUs are Indexed UFL expressions
        self.irhos = fe.split(self.sol)[:2**self.dim]
        self.iUs = fe.split(self.sol)[2**self.dim:]
        self.wrhos = TestFunctions(self.VS)[: 2**self.dim]
        self.wUs = TestFunctions(self.VS)[2**self.dim :]
        self.tdsol = TrialFunction(self.VS) # time derivatives
        self.tdrhos = fe.split(self.tdsol)[: 2**self.dim]
        self.tdUs = fe.split(self.tdsol)[2**self.dim :]
        bc_method = 'geometric' if self.dim > 1 else 'pointwise'
        rhobcs = [DirichletBC(
            self.VS.sub(i),
            Constant(0),
            FacesDomain(self.mesh, self.symmetries[i]),
            method=bc_method
        ) for i in range(2**self.dim) if np.any(self.symmetries[i] != 0.0)]
        Ubcs = [DirichletBC(
            self.VS.sub(i + 2**self.dim),
            Constant(0),
            FacesDomain(self.mesh, self.symmetries[i]),
            method=bc_method
        ) for i in range(2**self.dim)  if np.any(self.symmetries[i] != 0.0)]
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
        try:
            V(self.U0, self.rho0)
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
        logPERIODIC('restarting')
        self.restart()
        logPERIODIC('restart returned')
        return(None)

    def restart(self):
        logPERIODIC('restart')
        self.t = self.t0
        U0comps = evenodd_functions(
            omesh=self.omesh,
            degree=self.degree,
            func=self.U0,
            evenodd=self.symmetries,
            width=self.xmax
        )
        rho0comps = evenodd_functions(
            omesh=self.omesh,
            degree=self.degree,
            func=self.rho0,
            evenodd=self.symmetries,
            width=self.xmax
        )
        for i in range(2**self.dim):
            fe.assign(self.sol.sub(i),
                      fe.interpolate(rho0comps[i], self.SS))
            fe.assign(self.sol.sub(i + 2**self.dim),
                      fe.interpolate(U0comps[i], self.SS))
        
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
            self.dsol = Function(self.VS)
            self.drhos = self.dsol.split()[: 2**self.dim]
            self.dUs = self.dsol.split()[2**self.dim :]
        #
        # These are the values of rho and U themselves (not their
        # symmetrized versions) on all subdomains of the original
        # domain.
        #
        if not hasattr(self, 'rhosds'):
            self.rhosds = matmul(self.eomat, self.irhos)
        if not hasattr(self, 'Usds'):
            self.Usds = matmul(self.eomat, self.iUs)
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
            #
            self.Vsds = [self.V(Usd, rhosd) for Usd,rhosd in
                         zip(self.Usds, self.rhosds)]
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
            self.gamma = self.params['gamma']
            self.s = self.params['s']
            self.D = self.params['D']
            self.Upen = self.params['Upen']
            self.gUpen = self.params['gUpen']
            self.U_decay = vectotal(
                [-self.gamma * U * wU * self.dx
                 for U,wU in zip(self.iUs, self.wUs)]
            )
            self.U_secretion = vectotal(
                [self.s * rho * wU * self.dx
                 for rho, wU in zip(self.irhos, self.wUs)]
            )
            self.jump_gUw = vectotal(
                [self.D * ufl.jump(wU * ufl.grad(U), self.n) * self.dS
                 for wU, U in zip(self.wUs, self.iUs)
                ]
            )
            self.U_diffusion = vectotal(
                [-self.D
                 * ufl.dot(ufl.grad(U), ufl.grad(wU))*self.dx
                 for U,wU in zip(self.iUs, self.wUs)
                ]
            )
            self.U_penalty = vectotal(
                [-(self.Upen * self.degree**2 / self.havg)
                 * ufl.dot(ufl.jump(U, self.n), ufl.jump(wU, self.n))*self.dS
                 for U,wU in zip(self.iUs, self.wUs)
                ]
            )
            self.gU_penalty = vectotal(
                [-self.gUpen * self.degree**2 *
                 ufl.jump(ufl.grad(U), self.n) *
                 ufl.jump(ufl.grad(wU), self.n) * self.dS
                 for U,wU in zip(self.iUs, self.wUs)
                ]
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
