import petsc4py, sys
#
# this needs to be done before importing PETSc
#
petsc4py.init(sys.argv)
#
# I haven't worked out how to make the following peacefully coexist
# with PETSc command-line parsing.
# import fenics as fe
# fe.parameters.parse(sys.argv)

from .ksdgsolver import *
from .ksdgexception import *
from .random_function import *
from .ksdgts import *
from .ksdgtimeseries import *
from .ksdggather import *
from .ksdgperiodic import *
from .ksdgmakesolver import *
from .ksdgflotsam import *
from .ksdgexpand import *
from .ksdgligand import *
from .ksdgmultiple import *
from .ksdgmultper import *
from .ksdgvar import *
from .ksdgsoln import *

__all__ = [
    "KSDGException",
    "meshMakers",
    "cellShapes",
    "project_gradient_neumann",
    "project_gradient",
    "fplot",
    "unit_mesh",
    "box_mesh",
    "shapes",
    "KSDGSolver",
    "EKKSDGSolver",
    "random_function",
    "KSDGTS",
    "implicitTS",
    "imExTS",
    "explicitTS",
    "KSDGTimeSeries",
    "gather_array",
    "distribute_mesh",
    "gather_mesh",
    "gather_dof_coords",
    "gather_vertex_coords",
    "integerify_transform",
    "integerify",
    "dofremap",
    "isRsubspace",
    "local_Rdofs",
    "local_dofs",
    "remap_list",
    "function_interpolate",
    "fsinfo_filename",
    "remap_from_files",
    "mesh_stats",
    "CornerDomain",
    "corner_submesh",
    "evenodd_symmetries",
    "evenodd_matrix",
    "matmul",
    "vectotal",
    "evenodd_functions",
    "FacesDomain",
    "KSDGSolverPeriodic",
    "makeKSDGSolver",
    "vertex_list",
    "expand_mesh",
    "dof_list",
    "scalar_element",
    "ExpandedMesh",
    "Parameter",
    "ParameterList",
    "Ligand",
    "LigandGroup",
    "LigandGroups",
    'KSDGSolverMultiple',
    'KSDGSolverMultiPeriodic',
    'KSDGSolverVariable',
    'KSDGSolverVariablePeriodic',
    'Solution',
    'default_parameters',
]
