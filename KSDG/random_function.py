import fenics as fe
from fenics import (Function, FunctionSpace, UnitSquareMesh)
from mpi4py import MPI
import mshr
import numpy as np
from scipy.spatial import KDTree

def gather(mesh):
    """gather a list of vertex coordinates from all parallel processes.

    This is an Allgather -- i.e., the list is assembled on each
    process. Thus ksdggather.gather_array can't be used. But the logic
    is similar.
    """
    comm = mesh.mpi_comm()
    if (not isinstance(comm, type(MPI.COMM_SELF))):
        comm = comm.tompi4py()
    rank = comm.rank
    size = comm.size
    dim = mesh.geometry().dim()
    local_coordinates = mesh.coordinates().flatten()
    count = np.array(len(local_coordinates), dtype=int)
    counts = np.zeros(size, dtype=int)
    comm.Allgather(count, counts)
    offsets = np.zeros_like(counts)
    total = 0
    for r, nv in enumerate(counts):
        offsets[r] = total
        total += nv
    vcoords = np.zeros(total, dtype=float)
    comm.Allgatherv(local_coordinates,
                    [vcoords, counts, offsets, MPI.DOUBLE])
    vcoords = vcoords.reshape(-1, dim)
    vcoords = np.unique(vcoords, axis=0)
    return vcoords

def bcast(mesh, vals):
    comm = mesh.mpi_comm()
    if (not isinstance(comm, type(MPI.COMM_SELF))):
        comm = comm.tompi4py()
    comm.Bcast(vals, root=0)
    return(vals)

def random_function(
        function,
        mesh=None,
        fs=None,
        vals=None,
        mu=1.0,
        sigma=0.01,
        h = None, 
        tol=1e-10,
        periodic=False,
        f=(lambda x: 2*x**3 - 3*x**2 + 1)
):
    """define a pseudorandom Function

    random_function defines a scalar-valued FEniCS Function that has
    psedorandom values at points on a rectangular grid, and that is
    continuous and differentiable everywhere.

    Required positional argument:
    function: a FEniCS scalar-valued function, in which will be
        returned the desired function.

    Keyword arguments:
    mesh: the mesh on whose vertices the random values are to be
        imposed. (If not provided, this will be extracted from
        function.) The vertexes of this mesh must be a regular square
        array with equal spacing in all dimensions. The mesh need not
        be that on which function is defined, but its vertexes should
        be a subset of those of the mesh on which function is defined.
    fs: The FunctionSpace on which function is defined. (If not
        provided, this will be extracted from function.) This
        FunctionSpace should have order at least 3.
    vals: The values the function is to take on at each vertex. (The
        ordering is that returned by mesh.coordinates().) If not
        provided, random normal variates are drawn.
    mu=1.0: the mean of the normal distribution
    sigma=0.01: the standard deviation of the normal distribution.
    tol=1e-10: a tolerance for floating-point arithmetic errors.
    periodic=False: whether periodic boundary conditions are to be
        applied. The effect of periodic-True is to map values from the
        left and bottom edges to the right and top.
    f=(lambda x: 2*x**3 - 3*x**2 + 1): the function used to fill in the
        space between the vertexes. 
    """
    if not fs:
        fs = function.function_space()
    if not mesh:
        mesh = fs.mesh()
    dofmap = fs.dofmap()
    gdim = mesh.geometry().dim()
    dcoords = fs.tabulate_dof_coordinates().reshape(-1, gdim)
    tree = KDTree(dcoords)
    dvec = np.zeros(len(dcoords), dtype=float)
    vcoords = gather(mesh)
    vertexes = vcoords.shape[0]
    #
    # h will be an estimate of the the grid spacing (found by
    # computing for each pair of vertexes in cell 0 the disatnce
    # between, then finding the minimum positive value)
    # 
    if not h:
        hs = np.array([ 
            np.linalg.norm(mesh.coordinates()[u] - mesh.coordinates()[v]) 
            for u in mesh.cells()[0] for v in mesh.cells()[0]
        ])
        h = np.min(hs[hs > 0])        
    if not vals:
        vals = np.random.normal(mu, sigma, vertexes)
    vals = bcast(mesh, vals)
    vf = lambda x: np.product(f(x))
    mcoords = vcoords.copy()
    mapped = []
    if periodic:                # map right/top to left/bottom
        for d in range(gdim):
            sd = np.argsort(mcoords[:, d])
            mn = mcoords[sd[0], d]
            mx = mcoords[sd[-1], d]
            i = len(sd) - 1
            while i >= 0 and ((mx - mcoords[sd[i], d])/(mx - mn) <= tol):
                mcoords[sd[i], d] = mn
                mapped.append(sd[i])
                i -= 1
        mapped = np.unique(mapped)
        vtree = KDTree(vcoords)
        for i in mapped:
            opp = vtree.query_ball_point(mcoords[i], tol*(mx - mn))
            vals[i] = vals[opp[0]]
    for v, vc in enumerate(vcoords):
        touched = np.array(tree.query_ball_point(vc, h, float('inf')))
        if len(touched) > 0:                    # there may be none
            x = np.abs(dcoords[touched] - vc) / h
            touched2 = np.where(np.amax(x, 1) < 1 - tol)[0]
            touched = touched[touched2]
            dvec[touched] += vals[v] * np.fromiter(map(vf, x[touched2]), float)
    function.vector()[:] = dvec
    function.vector().apply('insert')
    return function


def main():
    np.random.seed(793817931)
    degree = 3
    mesh = UnitSquareMesh(2, 2)
    gdim = mesh.geometry().dim()
    fs = FunctionSpace(mesh, 'CG', degree)
    f = Function(fs)
    random_function(f)
    print(f.vector()[:])

if __name__ == '__main__':
    main()
