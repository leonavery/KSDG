"""Functions for gathering info from multiple processes"""

import h5py
import os
import sys
import tempfile
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from mpi4py import MPI
import fenics as fe
from fenics import (Mesh, MeshEditor, FunctionSpace, Function, File)
from .ksdgdebug import log
from .ksdgtimeseries import KSDGTimeSeries
from .ksdgexception import KSDGException

def logGATHER(*args, **kwargs):
    log(*args, system='GATHER', **kwargs)

cellShapes = [fe.interval, fe.triangle, fe.tetrahedron]

def dtype2mpitype(dtype):
    """convert a numpy dtype to an mpi4py type

    This is based on an undocumented private MPI element, so it is
    fragile.
    """
    try:
        mpitype = MPI.__TypeDict__[dtype.char]
    except AttributeError:
        mpitype = MPI._typedict[dtype.char]
    return(mpitype)

def gather_array(array, comm=MPI.COMM_WORLD):
    """Concatenate ndarrays from different processes

    Required parameter:
    array -- the local segment of the array

    array is a one-dimensional ndarray, that differs from process to
    process. gather_array gathers all the arrays and returns an single
    one-dimensional array whose contents are the process arrays
    concatenated in rank order. The gather is actually an allgather,
    so the gathered array is available in all processes.
    """
    size = comm.size
    dtype = dtype2mpitype(array.dtype)
    outshape = (-1,) + array.shape[1:]
    farray = array.flatten()
    n = farray.size
    total = comm.allreduce(n)
    logGATHER('n,total', (n, total))
    ns = np.empty(size, dtype=int)
    comm.Allgather(np.array(n), ns)
    logGATHER('ns', ns)
    displs = np.zeros(size + 1, dtype=int);
    displs[1:] = ns
    displs = np.cumsum(displs)
    displs = tuple(displs[:-1])
    farrays = np.zeros(total, dtype=array.dtype)
    recvbuf = [farrays, tuple(ns), displs, dtype]
    # sendbuf = [farray, n]
    sendbuf = farray
    logGATHER('sendbuf', sendbuf)
    logGATHER('recvbuf', recvbuf)
    logGATHER('len(farrays)', len(farrays))
    request = comm.Iallgatherv(sendbuf, recvbuf)
    logGATHER('request.Get_status()', request.Get_status())
    request.Wait()
    logGATHER('request.Get_status()', request.Get_status())
    logGATHER('farrays', farrays)
    logGATHER('outshape', outshape)
    arrays = farrays.reshape(outshape)
    logGATHER('arrays', arrays)
    return(arrays)

def distribute_mesh(mesh):
    """Distribute a local copy of a mesh

    Required argument:
    mesh: a sequential copy of the mesh to be distributed. Only the
    copy on the rank 0 process is used. This mesh will be distribuetd
    among all processes connected to MPI.COMM_WORLD so that it can be
    used for FEniCS computations.

    Return value: the distributed mesh
    """
    scomm = MPI.COMM_SELF
    wcomm = MPI.COMM_WORLD
    gmesh = gather_mesh(mesh)
    if wcomm.rank == 0:
        assert(gmesh.mpi_comm().size == 1) # check it's sequential
        with tempfile.TemporaryDirectory() as td:
            meshfile = os.path.join(td, 'mesh.xml')
            logGATHER('writing ' + meshfile)
            File(MPI.COMM_SELF, meshfile) << gmesh
            logGATHER('broadcasting meshfile')
            meshfile = wcomm.bcast(meshfile, root=0)
            try:
                with open(meshfile, 'r') as mf: # rank 0
                    dmesh = fe.Mesh(wcomm, meshfile)
                    logGATHER('dmesh', dmesh)
            except FileNotFoundError:
                logGATHER('meshfile', meshfile, 'not found')
            wcomm.Barrier()      # wait for everyone to read it
            logGATHER('destroying temp directory')
        # context manager destroyed, temp dir gets deleted
    else:
        meshfile = None
        logGATHER('receiving broadcast meshfile')
        meshfile = wcomm.bcast(meshfile, root=0)
        logGATHER('meshfile', meshfile)
        try:
            with open(meshfile, 'r') as mf: # rank != 0
                dmesh = fe.Mesh(wcomm, meshfile)
                logGATHER('dmesh', dmesh)
        except FileNotFoundError:
            logGATHER('meshfile', meshfile, 'not found')
        wcomm.Barrier()
    return(dmesh)

gather_mesh_cache = {}

def gather_mesh(mesh):
    """Gather a local copy of a distributed mesh"""
    # comm = MPI.COMM_WORLD
    if mesh in gather_mesh_cache:
        logGATHER('gather_mesh cache hit')
        return gather_mesh_cache[mesh]
    comm = mesh.mpi_comm()
    size = comm.size
    if size == 1:
        gather_mesh_cache[mesh] = mesh
        return(mesh)            # sequential: nothing to do
    dim = mesh.geometry().dim()
    topology = mesh.topology()
    logGATHER('topology.global_indices(0)', topology.global_indices(0))
    logGATHER('topology.size(0)', topology.size(0))
    #
    # To define a mesh, we need two things: a list of all the
    # vertices, each with an index and coordinates, and a list of all
    # cells, each specified by a list of vertices.
    #
    vcoord = mesh.coordinates()
    vcoords = gather_array(vcoord, comm) # vertexes from all processes
    vindex = topology.global_indices(0)
    vindexes = gather_array(vindex, comm) # global vertex indexes
    try:
        gnv = topology.size_global(0)
    except AttributeError:
        gnv = np.max(vindexes) + 1
    logGATHER('gnv', gnv)
    gvcs = np.zeros((gnv, dim), dtype=vcoords.dtype)
    for v, vc in enumerate(vcoords): # coords indexed by global indices
        gvcs[vindexes[v], :] = vc
    logGATHER('gvcs', gvcs)
    #
    # We now have in gvcs a list of global vertex coordinates. (The
    # indices are just 0...len(gvcs)-1.) Now for the cells:
    #
    nc = mesh.num_cells()
    logGATHER('nc', nc)
    total = comm.allreduce(nc)
    logGATHER('total', total)
    cell = np.zeros(nc*(dim+1), dtype=int)
    cell[:] = vindex[mesh.cells().flatten()]   # vindex to get global indices
    logGATHER('cell', cell)
    cells = gather_array(cell, comm)
    cells = cells.reshape(-1, dim+1)
    logGATHER('cells', cells)
    cindex = topology.global_indices(dim)
    logGATHER('cindex', cindex)
    cindexes = gather_array(cindex, comm)
    logGATHER('cindexes', cindexes)
    try:
        gnc = topology.size_global(dim)
    except AttributeError:
        gnc = np.max(cindexes) + 1
    logGATHER('gnc', gnc)
    gcells = np.zeros((gnc, dim+1), dtype=int)
    for v, cell in enumerate(cells):
        gcells[cindexes[v], :] = cell
    logGATHER('gcells', gcells)
    #
    # Now use this collected info to construct a mesh
    #
    try:
        scomm = fe.mpi_comm_self()
    except AttributeError:
        scomm = MPI.COMM_SELF
    mesh = Mesh(scomm) # new mesh is sequential
    logGATHER('scomm', scomm)
    logGATHER('MPI.COMM_SELF', MPI.COMM_SELF)
    # if comm.rank == 0:
    if True:                    # construct a mesh in all processes
        editor = MeshEditor()
        editor.open(mesh, str(cellShapes[dim-1]), dim, dim)
        editor.init_vertices_global(len(gvcs), len(gvcs))
        editor.init_cells_global(len(gcells), len(gcells))
        for v, vc in enumerate(gvcs):
            editor.add_vertex_global(v, v, vc)
        for c, cell in enumerate(gcells):
            logGATHER('c, cell', c, cell)
            editor.add_cell(c, cell)
    mesh.order()
    logGATHER('mesh.mpi_comm()', mesh.mpi_comm())
    logGATHER('mesh.mpi_comm().size', mesh.mpi_comm().size)
    gather_mesh_cache[mesh] = mesh
    logGATHER('caching gathered mesh')
    return(mesh)

def integerify_transform(array, tol=None):
    """Find spacing, base, and range for transformation to integer.
    
    Required parameter:
    array: This is a numpy array of floats of any shape. It is
        flattened and the numebrs used to find the transform
    tol=1e-7: Absolute tolerance for meaningful difference betweem two
        floats. Two numbers that differ by an absolute value <= tol
        are considered to be the same.

    Return: a tuple (spacing, base, imin, imax)
    If x is one of the floats in array, then x/spacing - base is close
    to an integer. imin and imax are the smallest and the largest
    integers that will result from applying this transform to array. 
    """
    if tol is None:
        tol = 1e-7
    xs = np.unique(array.flatten())
    dxs = np.unique(np.diff(xs))
    s1 = dxs.searchsorted(tol)
    s2 = dxs.searchsorted(dxs[s1] + tol, side='right')
    spacing = np.mean(dxs[s1:s2])
    bases = np.unique(tol + xs/spacing - np.floor(tol + xs/spacing)) - tol
    base = np.mean(bases)
    imin, imax = integerify(np.array([xs[0], xs[-1]]),
                            (spacing, base, 0, 0))
    if (np.max(np.abs(bases - base)) > tol):
        raise KSDGException(
            "couldn't transform coordinates to integer"
        )
    return((spacing, base, imin, imax))

def integerify(
    array,
    transform=None,
    tol=None
):
    """Map elements of a numpy array to integers

    Required argument:
    array: the numpy array. This will usually have elements of some
    floating point type.

    Optional keyword argument:
    transform: This is a list (or tuple or array) of at least two
    elements. The first element is the minimum spcaing, and the second
    the base. Elements of array are transformed by array/spacing -
    base, then rounded to integers.
    If not provided, integerify_transform(array) is used.
    """
    if transform is None:
        transform = integerify_transform(array, tol=tol)
    spacing = transform[0]
    base = transform[1]
    iarray = np.empty_like(array, dtype=int)
    return np.rint(array.astype(float)/spacing - base, iarray,
                   casting='unsafe')

def dofremap(ksdg, gathered_mesh=None, tol=None):
    """remap dofs from multiprocess FunctionSpace to single process

    Required parameter:
    ksdg -- the KSDGSolver object whose VS FunctionSpace to map.
    
    Optional keyword parameter
    gathered_mesh: the global mesh computed by gathering ksdg.mesh on
    rank 0. If not provided, gather_mesh is called to compute
    this. This parameter is ignored except on rank 0. 
    tol: the tolerance for intergerifying the coordinates.

    In the rank 0 processes, dofremap returns a vector of integers
    that can be used to remap degrees of freedom with the current
    global numbering to the numbering that would fit a single-process
    FunctionSpace. 

    remap = dofremap(ksdg, local_mesh=lmesh)
    sequential _function.vector()[:] = global_function.vector().array()[remap]
    
    How this works:
    ksdg.VS is a DG FunctionSpace on
    ksdg.mesh. gathered_mesh is a sequential (i.e., mesh.mpi_comm() ==
    fe.mpi_comm_self()) mesh constructed to have the same vertices and
    topology as ksdg.mesh, and whose vertex indexes correspond to the
    global vertex indexes of ksdg.mesh. dofremap begins by defining a
    vector FunctionSpace on gathered_mesh in a way that corresponds
    exactly to the way that ksdg.VS is built on ksdg.mesh.

    Now, a dof on a DG FunctionSpace is uniquely defined by three
    pieces of information: the cell it belongs to, the vector
    component (rho or U) it expresses, and its coordinates. (All three
    pieces of info are needed. In a DG space, there will typically be
    multiple dofs with the same coordinates, but belonging to different
    cells.) dofremap figures out how to reorder the dofs in ksdg so that
    they correspond in global cell index, component, and coordinates to
    those in the FunctionSpace defined on lmesh.
    """
    if not gathered_mesh:
        gmesh = gather_mesh(ksdg.mesh)
    else:
        gmesh = gathered_mesh
    comm = ksdg.mesh.mpi_comm()
    rank = comm.rank
    size = comm.size
    fs = ksdg.sol.function_space()
    pdofs = global_dofs(ksdg)
    pdofs = pdofs[:,:-1]        # delete final column
    logGATHER('pdofs', pdofs)
    logGATHER('pdofs.shape', pdofs.shape)
    # if rank == 0:
    if True:
        transform = integerify_transform(pdofs[:, 2:])
        spacing = transform[0]
        base = transform[1]
        logGATHER('transform', transform)
        logGATHER('creating FunctionSpaces')
        fss = ksdg.make_function_space(gmesh)
        VS = fss['PVS'] if 'PVS' in fss else fss['VS']
        logGATHER('gathering gdofs')
        gdofs = local_dofs(VS)
        gdofs = gdofs[np.argsort(gdofs[:,-1])] # should be unnecessary
        gdofs = gdofs[:,:-1]
        #
        # DOFs belonging to R subspaces don't have cells or coordinates
        #
        Rdofs = local_Rdofs(VS)[0] # VS is local, so finds all R dofs
        logGATHER('Rdofs', Rdofs)
        igdofs = np.empty_like(gdofs, dtype=int)
        np.rint(gdofs[:, :2], out=igdofs[:, :2], casting='unsafe')
        np.rint(gdofs[:, 2:]/spacing - base, out=igdofs[:, 2:],
                casting='unsafe')
        igdofs[Rdofs, 0] = -2
        igdofs[Rdofs, 2:] = -2
        gdofs[Rdofs, 0] = -2.0
        gdofs[Rdofs, 2:] = -2.0
        logGATHER('igdofs', igdofs)
        logGATHER('igdofs.shape', igdofs.shape)
        ipdofs = np.empty_like(pdofs, dtype=int)
        np.rint(pdofs[:, :2], out=ipdofs[:, :2], casting='unsafe')
        np.rint(pdofs[:, 2:]/spacing - base, out=ipdofs[:, 2:],
                casting='unsafe')
        ipdofs[Rdofs, 0] = -2
        ipdofs[Rdofs, 2:] = -2
        pdofs[Rdofs, 0] = -2.0
        pdofs[Rdofs, 2:] = -2.0
        logGATHER('ipdofs', ipdofs)
        logGATHER('ipdofs.shape', ipdofs.shape)
        remap = remap_list(ipdofs, igdofs, pdofs, gdofs)
    else:
        remap = np.zeros((0,0), dtype=int)
    logGATHER('remap', remap)
    return(remap)

def subspaces(fs):
    """ list (recursively) all subspaces of a FunctionSpace."""
    fssplit = fs.split()
    if not fssplit:                    # bottom out when split() == []
        return [fs]
    sss = []
    for ss in fssplit:
        sss.extend(subspaces(ss))
    return sss

def isRsubspace(fs):
    return fs.ufl_element().family() in set(['Real', 'R'])

def local_Rdofs(fs):
    """Find DOFs local to this process belonging to 'R' subspaces

    Required parameter:
    fs: the FunctionSpace to explore

    local_Rdofs expands the FunctionSpace fs to its individual
    subspaces (by calling subspaces). It identifies those that are of
    type 'R' (or 'Real') using isRsubspace, 

    Return:
    (Rdofs, Rlocal): Both of these are index arrays, intended to be
    used something like this:

    vec[Rdofs] = params[Rlocal]

    Rlocal is a list of numbers of the 'R' subspaces that have local
    DOFs, i.e., DOFs in the current process. Rdofs is a list of
    corresponding indexes of these DOFs in the local DOF vector. 
    """
    Rdofs = [
        np.array(ss.dofmap().dofs(), dtype=int)
        for ss in subspaces(fs) if isRsubspace(ss)
    ]
    present = np.array([ len(row) > 0 for row in Rdofs ], dtype=bool)
    present = np.nonzero(present)[0]
    if (Rdofs):
        Rdofs = np.concatenate(Rdofs)
    else:
        Rdofs = np.array(Rdofs, dtype=int)
    Rdofs -= fs.dofmap().ownership_range()[0]
    logGATHER('local_Rdofs Rdofs', Rdofs)
    logGATHER('local_Rdofs present', present)
    assert len(Rdofs) == len(present)
    return (Rdofs, present)

global_dofs_cache = {}

def global_dofs(fs):
    """gather local_dofs from all processes"""
    if not isinstance(fs, FunctionSpace):
        ksdg = fs
        fs = ksdg.sol.function_space()
    mesh = fs.mesh()
    comm = mesh.mpi_comm()
    dim = mesh.geometry().dim()
    dofmap = fs.dofmap()
    if dofmap in global_dofs_cache: # dofmap is key, since fs not hashable
        logGATHER('global_dofs cache hit')
        return global_dofs_cache[dofmap]
    ldofs = local_dofs(fs);
    logGATHER('ldofs', ldofs)
    logGATHER('ldofs.shape', ldofs.shape)
    pdofs = gather_array(ldofs, comm)
    pdofs = pdofs[np.argsort(pdofs[:,-1])]
    global_dofs_cache[dofmap] = pdofs
    logGATHER('global_dofs caching result')
    return pdofs

local_dofs_cache = {}

def local_dofs(fs):
    """List the dofs local to this process
    Required parameter:
    fs: the FunctionSpace to remap
        OR
    ksdg: the KSDGSolver object whose VS FunctionSpace to map.
    
    Returns an ndofs x dim+3 float array with one row for each local
    dof. The last element of the row is the global dof number. The
    first (i.e., 0th) is the global cell number in which this dof is
    located. The second is the number of the vector component
    associated with this dof. The remaining elements (2 through -2)
    are the coordinates of the dof.
    """
    if not isinstance(fs, FunctionSpace):
        ksdg = fs
        fs = ksdg.sol.function_space()
    mesh = fs.mesh()
    logGATHER('local_dofs: mesh', mesh)
    logGATHER('local_dofs: mesh.mpi_comm().size', mesh.mpi_comm().size)
    dim = mesh.geometry().dim()
    fss = subspaces(fs)
    if not fss:                 # scalar FunctionSpace
        fss = [fs]
    dofmap = fs.dofmap()
    if dofmap in local_dofs_cache: # dofmap is key, since fs not hashable
        logGATHER('local_dofs cache hit')
        return local_dofs_cache[dofmap]
    dofmaps = [fsi.dofmap() for fsi in fss]
    #
    # Now build a numpy array with the required info
    #
    owneddofs = dofmap.ownership_range()
    logGATHER('owneddofs', owneddofs)
    nlocal = owneddofs[1] - owneddofs[0]
    doflist = np.zeros((nlocal, dim + 3), dtype=float) # float, for simplicity
    ltg = dofmap.tabulate_local_to_global_dofs() # local to global map
    logGATHER('ltg', ltg)
    logGATHER('ltg.shape', ltg.shape)
    logGATHER('(np.sort(ltg[:nlocal]) == range(*owneddofs)).all()',
              (np.sort(ltg[:nlocal]) == range(*owneddofs)).all())
    doflist[:, -1] = ltg[:nlocal] # record global dof #s in last column
    gtl = np.argsort(ltg[:nlocal]) # global to local map
    logGATHER('gtl', gtl)
    logGATHER('gtl.shape', gtl.shape)
    for n,dofmapn in enumerate(dofmaps):
        try:
            dofs = np.array(dofmapn.dofs(), dtype=int)-owneddofs[0]
            doflist[gtl[dofs], 1] = float(n)
        except IndexError:
            logGATHER('dofs', dofs)
            logGATHER('dofs.dtype', dofs.dtype)
    logGATHER('mesh.num_cells()', mesh.num_cells())
    cindex = mesh.topology().global_indices(dim)
    logGATHER('cindex', cindex)
    logGATHER('cindex.shape', cindex.shape)
    dofspercell = dofmap.cell_dofs(0).size
    logGATHER('dofmap.cell_dofs(0)', dofmap.cell_dofs(0))
    ncells = len(cindex)
    #
    # Here we have a little problem. Some of the cells in our list may
    # contain dofs not owned by this process. I haven't found any list
    # in the dofmap or the mesh data structures that contains only
    # cells whose dofs are exclusive to this process. So we go with
    # EAFP and just try it and catch the IndexError if we get a dof we
    # don't own. This of course leaves open the possibility that some
    # dofs don't get a cell assigned to them, which would be a Very
    # Bad Thing. So we set everyone's cell identity to -1 first,
    # allowing us to catch such failures.
    #
    doflist[:, 0] = -1          # to catch failure to find a cell
    for cell in range(ncells):
        for dof in dofmap.cell_dofs(cell):
            try:
                doflist[dof, 0] = cindex[cell]
            except IndexError as ie:
                pass            # produces too much logging...
                # logGATHER('IndexError: cell,np.array(dofmap.cell_dofs(cell))',
                #           cell, np.array(dofmap.cell_dofs(cell)))
    logGATHER('(doflist[:, 0] == -1).any()',
              (doflist[:, 0] == -1).any())
    coords = np.reshape(fs.tabulate_dof_coordinates(), (-1, dim))
    doflist[:, 2:-1] = coords
    logGATHER('doflist.shape', doflist.shape)
    logGATHER('local_dofs caching result')
    local_dofs_cache[dofmap] = doflist
    return(doflist)

gather_vertex_coords_cache = {}

def gather_vertex_coords(mesh):
    """Gather the vertexes of a mesh."""
    if mesh in gather_vertex_coords_cache:
        logGATHER('gather_vertex_coords cache hit')
        return gather_vertex_coords_cache[mesh]
    comm = mesh.mpi_comm()
    dim = mesh.geometry().dim()
    lcoords = mesh.coordinates()
    logGATHER('lcoords', lcoords)
    gcoords = gather_array(lcoords, comm)
    # comm = mesh.mpi_comm()
    # comm.Bcast(gcoords, root=0)
    logGATHER('gcoords', gcoords)
    logGATHER('gather_vertex_coords result cached')
    gather_vertex_coords_cache[mesh] = gcoords
    return(gcoords)

gather_dof_coords_cache = {}

def gather_dof_coords(fs):
    """Gather the coordinates of the DOFs in the FunctionSpace."""
    if fs.dofmap() in gather_dof_coords_cache:
        logGATHER('gather_dof_coords cache hit')
        return gather_dof_coords_cache[fs.dofmap()]
    comm = fs.mesh().mpi_comm()
    dim = fs.mesh().geometry().dim()
    lcoords = np.reshape(fs.tabulate_dof_coordinates(), (-1, dim))
    gcoords = gather_array(lcoords, comm)
    logGATHER('gather_dof_coords result cached')
    gather_dof_coords_cache[fs.dofmap()] = gcoords
    return(gcoords)

def remap_list_sort(ipdofs, igdofs, pdofs=None, gdofs=None, tol=1e-7):
    """Utility function to make remapping from lists"""
    psort = np.lexsort(np.rot90(ipdofs))
    gsort = np.lexsort(np.rot90(igdofs))
    if (pdofs is not None and gdofs is not None):
        error = np.abs(pdofs[psort] - gdofs[gsort])
        logGATHER('error', error)
        if (error > tol).any():
            raise KSDGException("discrepancy in DOF remapping")
    remap = psort[np.argsort(gsort)]
    return(remap)

remap_list = remap_list_sort

def fsinfo_filename(prefix, rank=0):
    fsname = prefix + 'rank' + str(rank) + '_fsinfo.h5'
    return(fsname)

def remap_from_files(prefix, ksdg=None):
    """remap dofs from multiprocess FunctionSpace to single process

    Required parameter:
    prefix -- the filename prefix for the fsinfo files

    Optional parameter:
    ksdg -- the KSDGSolver that defines the single process
            FunctionsSpace

    Returns a vector of integers that can be used to remap degrees of
    freedom on a distributed mesh to those on the local mesh, using
    numpy indirection, i.e.

    remap = remap_from_files(ksdg, local_mesh=lmesh)
    local_function.vector()[:] = global_function.vector()[:][remap]
    
    This function does the same thing as dofremap, but at a different
    time and a different way. Unlike dofremap, remap_from_files is
    called after a solution is finished, not while it is in
    progress. The information about the degrees of freedom is
    retrieved from h5 files created at the time of solution. If the
    solution was run as an MPI group on S processes, there
    remap_from_files consults S+1 files:

    prefix_mesh.xml.gz
    prefixrank0_fsinfo.h5
    prefixrank1_fsinfo.h5
    ...
    prefixrank<S-1>_fsinfo.h5

    The mesh file contains the total mesh for the problem, as
    assembled by gather_mesh. Of the fsinfo files (which contain
    information about the function space local to each MPI process),
    at least the rank0 file must exist. Its 'size' entry is read to
    determine the number of fsinfo files, and its degree and dim
    entries are used to create a FunctionSpace on this mesh (by
    creating a KSDGSolver with those arguments). It then goes through
    the fsinfo files sequentially, determining the mapping from each
    one's DOFs to the DOFs pof the global FunctionSpace. 

    Note: this works only for solutions with a single rho and a single
    U.
    """
    if (ksdg):
        mesh = ksdg.mesh
    else:
        meshfile = prefix + '_mesh.xml.gz'
        mesh = Mesh(meshfile)    
    dim = mesh.geometry().dim()
    r0name = fsinfo_filename(prefix, 0)
    with h5py.File(r0name, 'r') as fsf:
        size = fsf['/mpi/size'].value
        if (dim != fsf['dim'].value):
            raise KSDGException(
                "mesh dimension = " + dim +
                ", FunctionSpace dim = " + fsf['dim']
            )
        try:
            degree = fsf['degree'].value
        except KeyError:
            degree = 3
            try:
                periodic = fsf['periodic'].value
            except KeyError:
                periodic = False
    if not ksdg:
        from .ksdgmakesolver import makeKSDGSolver # circular import
        ksdg = makeKSDGSolver(
            mesh=mesh,
            degree=degree,
            project_initial_condition=False,
            periodic=periodic
        )
    owneddofs = []
    ltg = []
    dofcoords = []
    dofs = []
    rho_dofs = []
    U_dofs = []
    cell_dofs = []
    cell_indices = []
    #
    # collect info about each processes dofs from files
    #
    for rank in range(size):
        rname = fsinfo_filename(prefix, rank)
        with h5py.File(rname, 'r') as fsf:
            owneddofs.append(fsf['ownership_range'].value.copy())
            ltg.append(fsf['tabulate_local_to_global_dofs'].value.copy())
            dofcoords.append(fsf['dofcoords'].value[0].copy())
            dofs.append(fsf['dofs'].value.copy())
            U_dofs.append(fsf['U_dofs'].value.copy())
            rho_dofs.append(fsf['rho_dofs'].value.copy())
            cell_dofs.append(fsf['cell_dofs'].value.copy())
            cell_indices.append(fsf['/mesh/cell_indices'].value.copy())
    transform = integerify_transform(dofcoords)
    spacing = transform[0]
    base = transform[1]
    #
    # check against global info
    #
    owneddofs = np.array(owneddofs)
    gdofmap = ksdg.VS.dofmap()
    gdofrange = gdofmap.ownership_range()
    if gdofrange[0] != 0:
        raise KSDGException(
            "First dof %d, expected 0", gdofrange[0]
        )
    fdofmin = owneddofs[:, 0].min()
    fdofmax = owneddofs[:, 1].max()
    if (fdofmin != gdofrange[0] or fdofmax != gdofrange[1]):
        raise KSDGException(
            "dof mismatch: global %d - %d, files %d - %d" %
            (gdofrange[0], gdofrange[1], fdofmin, fdofmax)
        )
    fdofs = np.zeros((gdofrange[1],dim + 2))
    #
    # combine file info into one list of file dofs
    #
    for rank in range(size):
        fdofs[rho_dofs[rank], 1] = 0.0
        fdofs[U_dofs[rank], 1] = 1.0
        ncells = cell_indices[rank].shape[0]
        dofspercell = cell_dofs[rank].shape[1]
        for cell in range(ncells):
            fdofs[ltg[rank][cell_dofs[rank][cell]],0] = cell_indices[rank][cell]
        nlocal = owneddofs[rank, 1] - owneddofs[rank, 0]
        fdofs[ltg[rank][:nlocal], 2:] = dofcoords[rank]
    #
    # assemble global dof list
    #
    ifdofs = np.empty_like(fdofs,dtype=int)
    np.rint(fdofs[:, :2], out=ifdofs[:, :2], casting='unsafe')
    np.rint(fdofs[:, 2:]/spacing - base, out=ifdofs[:, 2:],
            casting='unsafe')
    gdofs = local_dofs(ksdg)
    gdofs = gdofs[:, :-1]
    igdofs = np.empty_like(gdofs,dtype=int)
    np.rint(gdofs[:, :2], out=igdofs[:, :2], casting='unsafe')
    np.rint(gdofs[:, 2:]/spacing - base, out=igdofs[:, 2:],
            casting='unsafe')
    #
    # DOFs belonging to R subspaces don't have cells or coordinates
    #
    Rdofs = np.array([], dtype=int)
    for ss in subspaces(fs):
        if isRsubspace(ss):
            Rdofs = np.append(Rdofs, ss.dofmap().dofs())
    logGATHER('Rdofs', Rdofs)
    try:
        doflist[Rdofs-owneddofs[0], 1] = -2.0
        doflist[Rdofs-owneddofs[0], 2:-1] = -float('inf')
    except IndexError:
        logGATHER('IndexError: Rdofs-owneddofs[0]', Rdofs-owneddofs[0])
    return(remap_list(ifdofs, igdofs, fdofs, gdofs))

def coord_remap(
        fs,
        new_coords
):
    """Find a DOF remapping that effects a coordinate transformation.

    Required parameteters:
    fs: The FunctionSpace to be remapped
    new_coords: The coordinates of the remapped DOFs. This is a numpy
    matrix of shape (ndofs, dim), where ndofs is the number of degrees
    of freedom. 

    Returns:
    remap: an array of global dofs indexes such that dof i in the
    original FunctionSpace should be mapped to remap[i]. The remapping
    doesn't respect cell identities.
    """
    comm = fs.mesh().mpi_comm()
    old_coords = gather_dof_coords(fs)
    transform = integerify_transform(old_coords)
    spacing = transform[0]
    base = transform[1]
    pdofs = global_dofs(fs);
    pdofs = pdofs[np.argsort(pdofs[:,-1])]
    pdofs = pdofs[:, 1:-1]      # delete cell no and global index cols
    ndofs = pdofs.copy()
    ndofs[:, 1:] = new_coords
    ipdofs = np.empty_like(pdofs,dtype=int)
    np.rint(pdofs[:, :1], out=ipdofs[:, :2], casting='unsafe')
    np.rint(pdofs[:, 1:]/spacing - base, out=ipdofs[:, 1:],
            casting='unsafe')
    indofs = np.empty_like(ndofs,dtype=int)
    np.rint(ndofs[:, :1], out=indofs[:, :1], casting='unsafe')
    np.rint(ndofs[:, 1:]/spacing - base, out=indofs[:, 1:],
            casting='unsafe')
    remap = remap_list(ipdofs, indofs, pdofs, ndofs)
    logGATHER('coord_remap: remap', remap)
    return remap

def bcast_function(
        fs,
        vec0,
        f
):
    """Broadcast DOFs from process 0 to all processes.

    fs: the FunctionSpace on which the Function is to be assembled.
    vec0: the vector of DOF values (as a numpy array), on process 0
    f: The Function to which the broadcast vector is to be assigned.

    bcast_vector returns a Function defined on fs whose DOFs have been
    set to the values in vec0.
    """
    logGATHER('vec0', vec0)
    vec = vec0.copy()
    mesh = fs.mesh()
    comm = mesh.mpi_comm()
    if (not isinstance(comm, type(MPI.COMM_SELF))):
        comm = comm.tompi4py()
    logGATHER('comm.rank', comm.rank)
    logGATHER('broadcasting vec')
    comm.Bcast(vec, root=0)
    logGATHER('vec', vec)
    # f = Function(fs)
    dofmap = fs.dofmap()
    owneddofs = dofmap.ownership_range()
    logGATHER('owneddofs', owneddofs)
    gtl = np.array(dofmap.dofs())
    logGATHER('gtl', gtl)
    f.vector()[:] = vec[gtl]
    logGATHER('f.vector()[:]', f.vector()[:])
    f.vector().apply('insert')
    return(f)

def function_interpolate(fin, fsout, coords=None, method='nearest'):
    """Copy a fenics Function

    Required arguments:
    fin: the input Function to copy
    fsout: the output FunctionSpace onto which to copy it

    Optional arguments:
    coords: the global coordinates of the DOFs of the FunctionSpace on
    which fin is defined. If not provided, gather_dof_coords will be
    called to get them. If you intend to do several interpolations
    from functions defined on the same FunctionSpace, you can avoid
    multiple calls to gather_dof_coords by supplying this argument.
    method='nearest': the method argument of
    scipy.interpolate.griddata.

    Returns:
    The values from fin are interpolated into fout, a Function defined
    on fsout. fout is returned.
    """
    comm = fsout.mesh().mpi_comm()
    logGATHER('comm.size', comm.size)
    try:
        fsin = fin.function_space()
    except AttributeError:  # fallback for Constant
        fout = fe.interpolate(fin, fsout)
        return(fout)
    vlen = fsin.dim()          # # dofs
    logGATHER('vlen', vlen)
    if coords is None:
        coords = gather_dof_coords(fsin)
    logGATHER('fsin.mesh().mpi_comm().size',
              fsin.mesh().mpi_comm().size)
    try:
        vec0 = fin.vector().gather_on_zero()
    except AttributeError:      # fallback for Constant
        fout = fe.interpolate(fin, fsout)
        return(fout)
    if comm.rank == 0:
        vec = vec0.copy()
    else:
        vec = np.empty(vlen)
    comm.Bcast(vec, root=0)
    logGATHER('vec', vec)
    fout = Function(fsout)
    fout.vector()[:] = griddata(coords, vec,
                                fsout.tabulate_dof_coordinates(),
                                method=method).flatten()
    fout.vector().apply('insert')
    return(fout)
