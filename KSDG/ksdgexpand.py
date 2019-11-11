"""Mesh expansion of periodic boundary condition submeshes.

The KSDGSolverPeriodic class solves a Keller-Segel system with
periodic boundary conditions by expressing the functions as sums and
differences of odd and even periodic functions. These odd and even
functions are defined only on a coner of the original mesh. For
instance, a problem to be solved on a 1 cm x 1 cm space with a 128 x
128 mesh, it is reduced to an 64 x 64 mesh occupying the lower left
0.5 cm x 0.5 cm of the original mesh. The function on the entire 128 x
128 mesh can then be reconstructed from the odd and even functions
defined only on the corner submesh. That reconstruction is the purpsoe
of the ExpandedMesh class. 

This class and all its methods are designed to operate sequentially,
i.e., in a single process environment. 
"""

from mpi4py import MPI
import fenics as fe
from fenics import (Function, Mesh, FunctionSpace, MixedElement)
import pandas as pd
import numpy as np
import ufl
from ufl.log import UFLValueError
from .ksdggather import (integerify_transform, integerify)
from .ksdgperiodic import (evenodd_symmetries, evenodd_matrix,
                           mesh_stats)
from .ksdgsolver import (cellShapes)

def vertex_list(mesh, transform=None):
    """Get a list of the vertexes of a mesh as a DataFrame.

    Required argument:
    mesh: This may be either the Mesh istelf, or a list of the vertex
        coordinates.

    Optional Argument:
    transform=None: The transform to be used with integerify to put
        the coordinates in integer form. If not provided,
        integerify_transform is called on the vertex coordinates to
        get one.

    Return:
    vertex_list returns a pandas DataFrame with columns 0, 1,
        holding the vertex coordinates and index columns i0, i1,
        holding the integerified coorindates. 
    """
    if isinstance(mesh, fe.Mesh):
        coords = mesh.coordinates()
    else:
        coords = mesh
    dim = coords.shape[1]
    if transform is None:
        transform = integerify_transform(coords)
    icols = [ 'i'+str(i) for i in range(dim) ]
    df = pd.DataFrame(coords)
    ics = pd.DataFrame(integerify(coords, transform))
    df[icols] = ics
    df.set_index(icols, inplace=True)
    return df

def expand_mesh(mesh, transform=None):
    """expand_mesh -- expand a corner submesh to a full mesh.

    Required parameter:
    mesh -- the corner submesh to be expanded

    Optional parameter:
    transform -- the transform to integerfiy the vertex
        coordinates of the submesh.

    Returns:
    A list of four values: emsh, maps, cellmaps, and vertexmaps.
    emesh is the desired expanded mesh.
    maps is a list of 2**dim functions. Each, applied to a pair of
        coordinates in the submesh, maps them to a corresponding point
        in the appropriate sector of the expanded mesh.
    cellmaps is a 2**dim x nc integer ndarray (where nc is the number
        of cells in mesh). Each rowlists the cells in one of the
        2**dim sectors of emesh to which cells in the original submesh
        correspond. These lists are all disjoint.
    vertexmaps is a 2**dim x nv integer ndarray (where nv is the
        number of vertexes in mesh. Each row lists the vertexes of
        emesh to which vertexes of the original submesh map. These are
        not disjoint -- they overlap on the border of the submesh.
    """
    nvs = mesh.num_vertices()
    ncs = mesh.num_cells()
    cells = mesh.cells()
    dim = mesh.geometry().dim()
    cols = np.arange(dim)
    icols = [ 'i'+str(i) for i in range(dim) ]
    if transform is None:
        mtrans = integerify_transform(mesh.coordinates())
    else:
        mtrans = transform
    imax = 2*mtrans[-1]
    mtrans = list(mtrans[:-1]) + [imax]
    meshstats = mesh_stats(mesh)
    width = meshstats['xmax']
    symmetries = evenodd_symmetries(dim)
    maps = [
        lambda x, eo=eo, width=width: 2*width*eo + (1 - 2*eo)*x
        for eo in symmetries
    ]
    vertexes = vertex_list(mesh)
    vertexes['mesh'] = np.arange(nvs)
    ecoords = np.concatenate([ map(mesh.coordinates()) for map in maps ])
    evertexes = vertex_list(ecoords, transform=mtrans)
    evertexes.reset_index(inplace=True)
    evertexes = evertexes.drop_duplicates(subset=icols)
    evertexes.set_index(icols, inplace=True)
    evertexes.sort_index(inplace=True)
    envs = len(evertexes)
    encs = len(maps) * ncs
    emesh = fe.Mesh(MPI.COMM_SELF)
    editor = fe.MeshEditor()
    editor.open(emesh, str(cellShapes[dim-1]), dim, dim)
    editor.init_vertices_global(envs, envs)
    editor.init_cells_global(encs, encs)
    for v in range(envs):
        coords = evertexes.iloc[v, cols].values
        editor.add_vertex_global(v, v, coords)
    emesh.order()
    evertexes = vertex_list(emesh)
    evertexes['global'] = np.arange(envs)
    base = 0
    cellmaps = []
    vertexmaps = []
    for map in maps:
        scoords = map(mesh.coordinates())
        svertexes = vertex_list(scoords)
        svertexes = svertexes.merge(evertexes, how='inner')
        vertexmap = svertexes['global'].values
        vertexmaps.append(vertexmap)
        for c,cell in enumerate(cells):
            editor.add_cell(base + c, vertexmap[cell])
        cellmap = base + np.arange(ncs)
        cellmaps.append(cellmap)
        base += ncs
    editor.close()
    cellmaps = np.array(cellmaps)
    vertexmaps = np.array(vertexmaps)
    return emesh, maps, cellmaps, vertexmaps

def dof_list(fs, transform=None):
    """Get a list of the DOFs of a FunctionSpace as a DataFrame.

    Required parameter:
    fs: the fenics FunctionSpace


    Optional Argument:
    transform=None: The transform to be used with integerify to put
        the coordinates in integer form. If not provided,
        integerify_transform is called on the DOF coordinates to
        get one.

    Return:
    dof_list returns a pandas DataFrame listing all the DOFs. The
        index is the global DOF number. The data columns are 'dof',
        (also the global dof number), 0, 1, ... the coordinates of the
        DOF, 'sub', the subspace to which this DOF belongs (an
        integer), cell (the number of the cell in which this DOF is
        located (an integer), and 'i0', 'i1', ..., the integerified
        coordinates. Before returning, dof_list also asserts that the
        subspace number of each dof is the global dof number mod the
        number of subspaces, (i.e., that the DOFs are ordered by
        subspace in the most obvious and expected way). 
    """
    mesh = fs.mesh()
    nss = fs.num_sub_spaces()
    dim = mesh.geometric_dimension()
    dofmap = fs.dofmap()
    dcoords = np.reshape(fs.tabulate_dof_coordinates(), (-1,dim))
    if transform is None:
        transform = integerify_transform(dcoords)
    dofsdf = pd.DataFrame(dcoords)
    dofsdf.insert(loc=dim, column='sub', value=-1)
    ncs = mesh.num_cells()
    sdofs = np.full(len(dofsdf), -1, dtype=int)
    for s in range(nss):
        ssdofs = fs.sub(s).dofmap().dofs()
        sdofs[ssdofs] = s
    dofsdf['sub'] = sdofs
    dofsdf.insert(loc=dim+1, column='cell', value=-1)
    cells = np.full(len(dofsdf), -1, dtype=int)
    for c in range(ncs):
        cells[fs.dofmap().cell_dofs(c)] = c
    dofsdf['cell'] = cells
    dofsdf.reset_index(inplace=True)
    cols = np.arange(dim)
    icols = [ 'i'+str(i) for i in range(dim) ]
    dofsdf[icols] = pd.DataFrame(integerify(dofsdf[cols].values, transform=transform))
    dofsdf.rename(columns={'index': 'dof'}, inplace=True)
    assert(
        np.all(dofsdf['dof']%nss == dofsdf['sub']) 
    )
    return dofsdf

def scalar_element(VS):
    """Return a fencis FiniteElement from a vector FunctionSpace"""
    # from fenics import (FiniteElement, interval, triangle,
    #                     tetrahedron)
    # SE = VS.sub(0).element()
    # return eval(SE.signature())
    return VS.sub(0).ufl_element()

class ExpandedMesh:
    def __init__(
        self,
        function
    ):
        """Expand a Function defined on a corner mesh to a full mesh.

        Required parameter:
        funcion: the Function to be expanded.
        """
        self.VS = function.function_space()
        self.submesh = self.VS.mesh()
        self.dim = self.submesh.geometric_dimension()
        self.nss = self.VS.num_sub_spaces()
        self.nfields = self.nss // (2**self.dim)
        self.subcoords = self.submesh.coordinates()
        self.vtrans = integerify_transform(self.subcoords)
        self.icols = [ 'i'+str(i) for i in range(self.dim) ]
        self.emesh, self.maps, self.cellmaps, self.vertexmaps = (
            expand_mesh(self.submesh, self.vtrans)
        )
        self.dcoords  = np.reshape(self.VS.tabulate_dof_coordinates(),
                                   (-1, self.dim))
        self.dtrans = integerify_transform(self.dcoords)
        self.eSE = scalar_element(self.VS)
        self.degree = self.eSE.degree()
        self.eVE = MixedElement([ self.eSE ] * self.nfields)
        self.eVS = FunctionSpace(self.emesh, self.eVE)
        self.sub_dof_list = dof_list(self.VS, self.dtrans)
        subs = self.sub_dof_list['sub'].values
        self.sub_dof_list['submesh'] = subs % (2**self.dim)
        self.sub_dof_list['field'] = subs // (2**self.dim)
        sc = self.sub_dof_list[['submesh', 'cell']].values
        self.sub_dof_list['ecell'] = self.cellmaps[
            (sc[:, 0], sc[:, 1])
        ]
        self.e_dof_list = dof_list(self.eVS, self.dtrans)
        self.remap = self.sub2e_map()
        self.symmetries = evenodd_symmetries(self.dim)
        self.eomat = evenodd_matrix(self.symmetries)
        self.sub_function = Function(self.VS)
        self.expanded_function = Function(self.eVS)

    def expand(self):
        fvec = self.sub_function.vector()[:]
        fvec = np.reshape(fvec, (-1, 2**self.dim))
        fvec = np.matmul(fvec, self.eomat)
        fvec = fvec.flatten()
        fvec = fvec[self.remap]
        self.expanded_function.vector()[:] = fvec

    def sub2e_map(self):
        cs = list(range(self.dim))
        sdlcols = ['dof', 'submesh', 'field', 'ecell'] + self.icols
        sdls = pd.DataFrame(columns=sdlcols + cs)
        sdl = self.sub_dof_list.copy()
        for submesh in range(2**self.dim):
            sdlsm = sdl[sdl['submesh'] == submesh].copy()
            sdlsm[cs] = self.maps[submesh](sdlsm[cs])
            sdls = sdls.append(sdlsm[sdlcols + cs], sort=False)
        sdls[self.icols] = integerify(sdls[cs].values,
                                      transform=self.dtrans)
        sdls.sort_values('dof', inplace=True)
        sdls[sdlcols] = np.array(sdls[sdlcols].values, dtype='int')
        remapdf = self.e_dof_list.merge(
            right=sdls,
            left_on=['sub', 'cell'] + self.icols,
            right_on=['field', 'ecell'] + self.icols
        )
        assert(
            self.e_dof_list.shape[0] == sdls.shape[0] and
            remapdf.shape[0] == sdls.shape[0]
        )
        remapdf.sort_values(['dof_x'], inplace=True)
        return remapdf['dof_y'].values

            
