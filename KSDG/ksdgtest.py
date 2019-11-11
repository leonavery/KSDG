from argparse import Namespace
from ksdgargparse import KSDGParser

commandlineArguments = Namespace()

def parse_commandline():
    global commandlineArguments
    parser = KSDGParser()
    parser.add_argument('--implicit')
    parser.add_argument('--imex')
    parser.add_argument('--explicit')
    parser.add_argument('--solver', default='gmres')
    parser.add_argument('--seed', type=int, default=793817931)
    commandlineArguments = parser.parse_args(namespace=commandlineArguments)

parse_commandline()
import petsc4py, sys
from KSDG import *
import fenics as fe
import numpy as np
#
# this needs to be done before importing PETSc
#
petsc4py.init(sys.argv[0:1] + commandlineArguments.petsc)
from petsc4py import PETSc
fe.parameters.parse(sys.argv[0:1] + commandlineArguments.fenics)
fe.parameters['ghost_mode'] = 'shared_facet'
                
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
        return -params['beta']*fe.ln(U + params['alpha'])
    mesh = unit_mesh(nelements, dim)
    fe.parameters["form_compiler"]["representation"] = "uflacs"
    fe.parameters['linear_algebra_backend'] = 'PETSc'
    solver = KSDGSolver(dim=dim, degree=degree, nelements=nelements,
                        parameters=params,
                        V=Vfunc, U0=U0str, rho0=rho0str,
                        debug=True)
    print(str(solver))
    print(solver.ddt(debug=True))
    print("dsol/dt components:", solver.dsol.vector().array())

    eksolver = EKKSDGSolver(dim=dim, degree=degree, nelements=nelements,
                        parameters=params,
                        V=Vfunc, U0=U0str, rho0=rho0str,
                        debug=True)
    print(str(eksolver))
    print(eksolver.ddt(debug=True))
    print("dsol/dt components:", eksolver.dsol.vector().array())
    #
    # try out the time-stepper
    #
    np.random.seed(commandlineArguments.seed)
    murho0 = params['N']/params['M']
    Cd1 = fe.FunctionSpace(mesh, 'CG', 1)
    rho0 = fe.Function(Cd1)
    random_function(rho0, mu=murho0, sigma=params['srho0'])
#    rho0.vector()[:] = np.random.normal(murho0, params['srho0'],
#                                        rho0.vector().array().shape)
    U0 = fe.Function(Cd1)
    U0.vector()[:] = (params['s']/params['gamma'])*rho0.vector()
    ksdg = KSDGSolver(degree=degree, mesh=mesh,
                  parameters=params, project_initial_condition=False,
                  solver_type=commandlineArguments.solver,
                  V=Vfunc, U0=U0, rho0=rho0)
    options = PETSc.Options()
    options.setValue('ts_max_snes_failures', 100)
    ts = implicitTS(ksdg, maxsteps=200)
    ts.setMonitor(ts.historyMonitor)
    ts.setMonitor(ts.printMonitor)
    if commandlineArguments.implicit:
        saveMonitor, closeMonitor = ts.makeSaveMonitor(
            prefix=commandlineArguments.implicit
        )
        ts.setMonitor(saveMonitor)
    ts.solve()
    if commandlineArguments.implicit:
        closeMonitor()
    ts.cleanup()
    print("SNES failures = ", ts.getSNESFailures())
    print('Final time point')
    print(ts.history[-1])
    ts = imExTS(ksdg, maxsteps=200)
    ts.setMonitor(ts.historyMonitor)
    ts.setMonitor(ts.printMonitor)
    ts.solve()
    ts.cleanup()
    print("SNES failures = ", ts.getSNESFailures())
    print('Final time point')
    print(ts.history[-1])
    ts = explicitTS(ksdg, maxsteps=200)
    ts.setMonitor(ts.historyMonitor)
    ts.setMonitor(ts.printMonitor)
    ts.solve()
    ts.cleanup()
    print("SNES failures = ", ts.getSNESFailures())
    print('Final time point')
    print(ts.history[-1])
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

