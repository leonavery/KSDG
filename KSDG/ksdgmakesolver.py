"""Factory function for making KSDGSolver, handles multiligand and periodic."""

from .ksdgexception import KSDGException

def makeKSDGSolver(*args, **kwargs):
    if 'periodic' in kwargs:
        periodic = kwargs['periodic']
    else:
        kwargs['periodic'] = periodic = False
    if 'ligands' in kwargs:
        ligands = kwargs['ligands']
    else:
        ligands = False
    variable = ('param_funcs' in kwargs and
                kwargs['param_funcs'] is not None)
    if variable:
        if periodic:
            return(KSDGSolverVariablePeriodic(*args, **kwargs))
        else:
            return(KSDGSolverVariable(*args, **kwargs))
    if 'dparamsdt' in kwargs: del(kwargs['dparamsdt'])
    if ligands and periodic:
        return(KSDGSolverMultiPeriodic(*args, **kwargs))
    elif ligands:
        return(KSDGSolverMultiple(*args, **kwargs))
    elif periodic:
        return(KSDGSolverPeriodic(*args, **kwargs))
    else:
        return(KSDGSolver(*args, **kwargs))

            
from .ksdgsolver import KSDGSolver
from .ksdgperiodic import KSDGSolverPeriodic
from .ksdgmultiple import KSDGSolverMultiple
from .ksdgmultper import KSDGSolverMultiPeriodic
from .ksdgvar import KSDGSolverVariable, KSDGSolverVariablePeriodic
