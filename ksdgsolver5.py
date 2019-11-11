import IPython
# The preceding line is there for an odd reason. I don't use anything
# in IPython, as far as I know. However, when I first tried running
# this, I got the following warning when I imported fenics.
#
# --------------------------------------------------------------------------
# A process has executed an operation involving a call to the
# "fork()" system call to create a child process.  Open MPI is currently
# operating in a condition that could result in memory corruption or
# other system errors; your job may hang, crash, or produce silent
# data corruption.  The use of fork() (or system() or other calls that
# create child processes) is strongly discouraged.

# The process that invoked fork was:

#   Local host:          [[59055,1],0] (PID 14564)

# If you are *absolutely sure* that your application will successfully
# and correctly survive a call to fork(), you may disable this warning
# by setting the mpi_warn_on_fork MCA parameter to 0.
# --------------------------------------------------------------------------
#
# I eventually tracked this down. dolfin, it turns out, uses the
# pkgconfig module to get information about its own configuration. The
# pkgconfig moduel runs the pkg-config command. I kludged aroudn this
# by putting my own modified version of the pkgconfig module in the
# PYTHONPATH ahead of the oen instaleld in the usual lcoation. I
# modified my version to cache the results of a lookup in a dbm
# file. Thus, after importing fencis once, values are obtained from
# the cache and pkg-config is not run.
#
# That took care of the fenics problem, but then I found that h5py
# also produced a fork warning. Oddly, though, this didn't happen in
# ipython. Therefore I tried importing the IPython module before
# importing h5py, and that suppressed the fork warning on importing
# h5py. IPyhton has to be imported before fenics, or it produces its
# own fork warning.
#
from argparse import Namespace, RawDescriptionHelpFormatter
from ksdgargparse import KSDGParser

lig_help = """
Use --showparams to see ligand parameters
"""

# In principle, I should not need this definition here -- I could
# import default_parameters from KSDG. However, there is a weird
# conflict with petsc4py. It is essential to initialize petsc4py with
# the command line arguments before importing anything from
# KSDG. Otherwise PETSc snarfs up all the command line arguments, and
# since many of them are not PETSc arguments, gets confused. Ugly.
#
default_parameters = [
    ('ngroups', 1, 'number of ligand groups'),
    ('nligands,1', 1, 'number of ligands in group 1'),
    ('degree', 3, 'degree of polynomial approximations'),
    ('dim', 1, 'spatial dimensions'),
    ('nelements', 8, 'number finite elements in width'),
    ('width', 1.0, 'width of spatial domain'),
    ('Umin', 1e-7, 'minimum allowed value of U'),
    ('rhomin', 1e-7, 'minimum allowed value of rho'),
    ('rhomax', 10.0, 'approximate max value of rho'),
    ('cushion', 0.5, 'cushion on rho'),
    ('maxscale', 2.0, 'scale of cap potential'),
    ('sigma', 1.0, 'random worm movement'),
    ('Nworms', 1.0, 'total number of worms'),
    ('srho0', 0.01, 'standard deviation of rho(0)'),
    ('rho0', '0.0', 'C++ string function for rho0, added to random rho0'),
    ('rhopen', 1.0, 'rho discontinuity penalty'),
    ('grhopen', 10.0, 'grad(rho) discontinuity penalty'),
    ('Upen', 1.0, 'U discontinuity penalty'),
    ('gUpen', 1.0, 'grad(U) discontinuity penalty'),
    ('maxsteps', 200, 'maximum number of time steps'),
    ('t0', 0.0, 'initial time'),
    ('dt', 0.001, 'first time step'),
    ('tmax', 20.0, 'time to simulate'),
    ('rtol', 1e-5, 'relative tolerance for step size adaptation'),
    ('atol', 1e-5, 'absolute tolerance for step size adaptation'),
]

def parameter_help(param_list=default_parameters, add_help=lig_help):
    help = 'Parameters:'
    for t,d,h in param_list:
        help += t + '=' + str(d) + ' -- ' + h + '\n'
    help += lig_help
    return help    

def parse_commandline(args=None):
    commandlineArguments = Namespace()
    parser = KSDGParser(
        description='Solve Keller-Segel PDEs',
        epilog=parameter_help(),
        formatter_class=RawDescriptionHelpFormatter
    )
    # parser.add_argument('--ngroups', type=int, default=1,
    #                     help='# ligand groups')
    # parser.add_argument('--nligands', type=int, nargs=2, action='append',
    #                     default=[], help='# ligands in each group')
    parser.add_argument('--randgrid', type=int,
                        help='# divisions in grid for random rho0')
    parser.add_argument('--cappotential', choices=['tophat', 'witch'],
                        default='tophat',
                        help='potential function for capping rho')
    parser.add_argument('--penalties', type=float, default=None,
                        help='discontinuity penalties')
    parser.add_argument('--save',
                        help='filename prefix in which to save results')
    parser.add_argument('--check',
                        help='filename prefix for checkpoints')
    parser.add_argument('--resume',
                        help='resume from last point of a TimeSeries')
    parser.add_argument('--restart',
                        help='restart (i.e., with t=t0) from last point of a TimeSeries')
    parser.add_argument('--showparams', action='store_true',
                        help='print all parameters')
    parser.add_argument('--periodic', action='store_true',
                        help='use periodic boundary conditions')
    parser.add_argument('--solver', default='petsc',
                        help='linear solver')
    parser.add_argument('--seed', type=int, default=793817931,
                        help='random number generator seed')
    parser.add_argument('--decay', type=str, action='append',
                        default=[], help='decay rate of parameter')
    parser.add_argument('--slope', type=str, action='append',
                        default=[], help='slope of parameter increase')
    parser.add_argument('params', type=str, nargs='*',
                        help='parameter values')
    commandlineArguments = parser.parse_args(
        args=args,
        namespace=commandlineArguments
    )
    return commandlineArguments

import petsc4py
import sys
import os
import copy
import numpy as np
import fenics as fe
import h5py
from mpi4py import MPI

def in_notebook():
    try:
        cfg = get_ipython()
        if cfg.config['IPKernelApp']:
            return(True)
    except NameError:
        return(False)
    return(False)

from signal import (signal, NSIG, SIGHUP, SIGINT, SIGPIPE, SIGALRM,
                    SIGTERM, SIGXCPU, SIGXFSZ, SIGVTALRM, SIGPROF,
                    SIGUSR1, SIGUSR2, SIGQUIT, SIGILL, SIGTRAP,
                    SIGABRT, SIGFPE, SIGBUS, SIGSEGV,
                    SIGSYS)

def signal_exception(signum, frame):
    raise KeyboardInterrupt('Caught signal ' + str(signum))

def catch_signals(signals=None):
    """catch all possible signals and trurn them into exceptions"""
    terminators = set([
        SIGHUP,
        SIGINT,
        SIGPIPE,
        SIGALRM,
        SIGTERM,
        SIGXCPU,
        SIGXFSZ,
        SIGVTALRM,
        SIGPROF,
        SIGUSR1,
        SIGUSR2,
        SIGQUIT,
        SIGILL,
        SIGTRAP,
        SIGABRT,
        SIGFPE,
        SIGBUS,
        SIGSEGV,
        SIGSYS,
    ])
    if not signals:
        signals = terminators
    for sig in signals:
        try:
            signal(sig, signal_exception)
        except:
            pass

def pfuncs(clargs, params0, t0=0.0):
    """Create functions for parameters

    Required arguments:
    clargs: Command line arguments.
    params0: a mappable of initial numerical values of parameters

    Optional parameter:
    t0=0.0: initial time (time at which params have valuve params0

    Returns a dict mapping param names (from params0) to
    functions. Each function has the call signature func(t, params={})
    and returns the value of the parameter at time t. 
    """
    import ufl
    from KSDG import ParameterList, KSDGException
    decays = ParameterList()
    decays.decode(clargs.decay, allow_new=True)
    slopes = ParameterList()
    slopes.decode(clargs.slope, allow_new=True)
    keys = set(decays.keys()) | set(slopes.keys())
    extras = keys - set(params0.keys())
    if extras:
        raise KSDGException(
            ', '.join([k for k in extras]) +
            ': no such parameter'
        )
    funcs = {}
    for k in params0.keys():
        d = decays[k] if k in decays else 0.0
        s = slopes[k] if k in slopes else 0.0
        pt0 = params0[k] if k in params0 else 1.0
        if d == 0 and s == 0:
            def func(t, params={}, p0=pt0):
                return p0
        elif d == 0:
            p0 = pt0 - s*t0
            def func(t, params={}, p0=pt0, s=s):
                return p0 + s*t
        else:
            a = (ufl.exp(d*t0)*(d*pt0 - s) - s)/d
            pinf = s/d
            def func(t, params={}, d=d, pinf=pinf, a=a):
                return pinf + a*ufl.exp(-d*t)
        
        funcs[k] = func
    return funcs
        
    
def main(*args):
    #
    # I don't understand why this has to be done as follows, but PETSc
    # doesn't get commandline arguments if things are done in different
    # order.
    #
    if args:
        args = list(args)
    else:
        args = sys.argv
    commandlineArguments = parse_commandline(args[1:])
    #
    # this needs to be done before importing PETSc
    #
    petsc4py.init(args=(args[0:1] + commandlineArguments.petsc))
    from KSDG import (box_mesh, makeKSDGSolver, dofremap,
                      KSDGException, KSDGTimeSeries, random_function,
                      implicitTS, LigandGroups, ParameterList,
                      default_parameters)
    from KSDG.ksdgdebug import log
    def logMAIN(*args, **kwargs):
        log(*args, system='MAIN', **kwargs)
    logMAIN('commandlineArguments.petsc', commandlineArguments.petsc)
    # import fenics as fe
    import ufl
    from petsc4py import PETSc
    fe.parameters.parse(sys.argv[0:1] + commandlineArguments.fenics)
    fe.parameters['ghost_mode'] = 'shared_facet'
    fe.parameters["form_compiler"]["optimize"]     = True
    fe.parameters["form_compiler"]["cpp_optimize"] = True
    catch_signals()
    comm = MPI.COMM_WORLD
    groups = LigandGroups(commandlineArguments)
    params0 = ParameterList(default_parameters)   # parameter initial values
    params0.add(groups.params())
    params0.decode(commandlineArguments.params)
    groups.fourier_series()
    params0.add(groups.params())          # in case Fourier added any new ones
    Vgroups = copy.deepcopy(groups)
    Vparams = ParameterList(default_parameters)   # for evaluating V
    Vparams.add(Vgroups.params())
    width = params0['width']
    nelements = params0['nelements']
    dim = params0['dim']
    degree = params0['degree']
    periodic = commandlineArguments.periodic
    nligands = groups.nligands()
    rhomax = params0['rhomax']
    cushion = params0['cushion']
    t0 = params0['t0']
    param_funcs = pfuncs(commandlineArguments, t0=t0, params0=params0)
    if commandlineArguments.penalties:
        rhopen = grhopen = Upen = gUpen = commandlineArguments.penalties
    else:
        rhopen = Upen = gUpen = 1.0
        grhopen = 10.0
    rhopen = params0['rhopen'] or rhopen; params0['rhopen'] = rhopen
    grhopen = params0['grhopen'] or grhopen; params0['grhopen'] = grhopen
    Upen = params0['Upen'] or Upen; params0['Upen'] = Upen
    gUpen = params0['gUpen'] or gUpen; params0['gUpen'] = gUpen
    logMAIN('list(groups.ligands())', list(groups.ligands()))
    maxscale = params0['maxscale']
    if (commandlineArguments.showparams):
        for n,p,d,h in params0.params():
            print(
                '{n}={val} -- {h}'.format(n=n, val=p(), h=h)
            )
        return
    logMAIN('params0', params0)
    def Vfunc(Us, params={}):
        Vparams.update(params)            # copy params into ligands
        return Vgroups.V(Us)               # compute V
    def Vtophat(rho, params={}):
        tanh = ufl.tanh((rho - params['rhomax'])/params['cushion'])
        return params['maxscale'] * params['sigma']**2 / 2 * (tanh + 1)
    def Vwitch(rho, params={}):
        tanh = ufl.tanh((rho - params['rhomax'])/params['cushion'])
        return (params['maxscale'] * params['sigma']**2 / 2 *
                (tanh + 1) * (rho / params['rhomax'])
        )
    Vcap = Vwitch if commandlineArguments.cappotential == 'witch' else Vtophat
    def V2(Us, rho, params={}):
        return Vfunc(Us, params=params) + Vcap(rho, params=params)
    mesh = box_mesh(width=width, dim=dim, nelements=nelements)
    if commandlineArguments.randgrid:
        rmesh = box_mesh(width=width, dim=dim,
                          nelements=commandlineArguments.randgrid)
    else:
        rmesh = box_mesh(width=width, dim=dim,
                          nelements=nelements//2)
    fe.parameters["form_compiler"]["representation"] = "uflacs"
    fe.parameters['linear_algebra_backend'] = 'PETSc'
    np.random.seed(commandlineArguments.seed)
    murho0 = params0['Nworms']/(width**dim)
    resuming = commandlineArguments.resume or commandlineArguments.restart
    if resuming:
        if comm.rank == 0:
            cpf = KSDGTimeSeries(resuming, 'r')
            tlast = cpf.sorted_times()[-1]
            if commandlineArguments.resume:
                t0 = tlast
            else:
                t0 = params0['t0']
            lastvec = cpf.retrieve_by_time(tlast)
            nlastvec = lastvec.size
            vectype = lastvec.dtype
        else:
            t0 = params0['t0']
            nlastvec = None
            vectype = None
        if commandlineArguments.resume:
            #
            # if resuming, get t0 from the datafile
            #
            t0 = comm.bcast(t0, root=0)
        nlastvec = comm.bcast(nlastvec, root=0)
        vectype = comm.bcast(vectype, root=0)
        if comm.rank != 0:
            lastvec = np.empty(nlastvec, vectype)
        logMAIN('t0', t0)
        logMAIN('nlastvec', nlastvec)
        logMAIN('vectype', vectype)
        U0s = [fe.Constant(0.0) for i in range(nligands)]
        ksdg = makeKSDGSolver(degree=degree, mesh=mesh, width=width,
                              nelements=nelements, t0=t0,
                              parameters=params0, param_funcs=param_funcs,
                              periodic=periodic,
                              ligands=groups,
                              solver_type=commandlineArguments.solver,
                              V=V2, U0=U0s,
                              rho0=fe.Constant(0.0))  
        remap = dofremap(ksdg)
        lastvec = np.array(lastvec[np.argsort(remap)])
        lastvec = comm.bcast(lastvec, root=0)
        nlastvec = lastvec.size
        vectype = lastvec.dtype
        logMAIN('lastvec', lastvec)
        logMAIN('type(lastvec)', type(lastvec))
        logMAIN('nlastvec', nlastvec)
        logMAIN('vectype', vectype)
        dofmap = ksdg.sol.function_space().dofmap()
        logMAIN('np.array(dofmap.dofs())', np.array(dofmap.dofs()))
        logMAIN('lastvec[np.array(dofmap.dofs())]', lastvec[np.array(dofmap.dofs())])
        ksdg.sol.vector()[:] = lastvec[np.array(dofmap.dofs())]
        ksdg.sol.vector().apply('insert')
    else:
        t0 = params0['t0']
        Cd1 = fe.FunctionSpace(mesh, 'CG', degree)
        logMAIN('Cd1', Cd1)
        rho0 = fe.Function(Cd1)
        logMAIN('rho0', rho0)
        random_function(rho0, mesh=rmesh, periodic=periodic,
                        mu=murho0, sigma=params0['srho0'])
        if params0['rho0']:
            rho0Exp = fe.Expression(
                params0['rho0'], degree=degree
            )
            rho0Expf = fe.interpolate(rho0Exp, Cd1)
            rho0.vector()[:] += rho0Expf.vector()
        U0s = [fe.Function(Cd1) for i in range(nligands)]
        for i, lig in enumerate(groups.ligands()):
            U0s[i].vector()[:] = (lig.s/lig.gamma)*rho0.vector()
        ksdg = makeKSDGSolver(degree=degree, mesh=mesh, width=width,
                              nelements=nelements, t0=t0,
                              parameters=params0, param_funcs=param_funcs,
                              periodic=periodic,
                              ligands=groups,
                              solver_type=commandlineArguments.solver,
                              V=V2, U0=U0s, rho0=rho0)
    logMAIN('ksdg', str(ksdg))
    options = PETSc.Options()
    options.setValue('ts_max_snes_failures', 100)
    ts = implicitTS(ksdg,
                    t0 = t0,
                    restart = not bool(resuming),
                    rtol = params0['rtol'],
                    atol = params0['atol'],
                    dt = params0['dt'],
                    tmax = params0['tmax'],
                    maxsteps = params0['maxsteps'])
    logMAIN('ts', str(ts))
    ts.setMonitor(ts.printMonitor)
    logMAIN('printMonitor set')
    if commandlineArguments.save:
        with open(commandlineArguments.save + '_options.txt', 'w') as opts:
            optd = vars(commandlineArguments)
            for k, v in optd.items():
                if k != 'petsc' and k != 'fenics' and k != 'params':
                    print('--%s=%s' % (k, v), file=opts)
            print(params0.str(), file=opts)
            try:
                popts = optd['petsc']
                if popts:
                    print('--petsc', file=opts)
                    for popt in popts:
                        print(popt, file=opts)
                    print('--', file=opts)
            except KeyError:
                pass
            try:
                fopts = optd['fenics']
                if fopts:
                    print('--fenics', file=opts)
                    for fopt in fopts:
                        print(fopt, file=opts)
                    print('--', file=opts)
            except KeyError:
                pass
        saveMonitor, closeMonitor = ts.makeSaveMonitor(
            prefix=commandlineArguments.save
        )
        ts.setMonitor(saveMonitor)
        logMAIN('saveMonitor set')
    if commandlineArguments.check:
        ts.setMonitor(
            ts.checkpointMonitor,
            (),
            {'prefix': commandlineArguments.check}
        )
        logMAIN('checkpointMonitor set')
    try:
        logMAIN('calling ts.solve', ts.solve)
        ts.solve()
    except KeyboardInterrupt as e:
        print('KeyboardInterrupt:', str(e))
    except Exception as e:
        print('Exception:', str(e))
        einfo = sys.exc_info()
        sys.excepthook(*einfo)
    if commandlineArguments.save:
        closeMonitor()
        logMAIN('saveMonitor closed')
    ts.cleanup()
    if MPI.COMM_WORLD.rank == 0:
        print("SNES failures = ", ts.getSNESFailures())
    # MPI.Finalize()

if __name__ == "__main__" and not in_notebook():
    # execute only if run as a script
    main()
