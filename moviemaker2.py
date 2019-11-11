#!/usr/bin/env python3

import IPython
import sys
import os
import pickle
savargv = sys.argv
sys.argv = sys.argv[:1]        # Hide argv from PETSc
from argparse import ArgumentParser
import fenics as fe
from KSDG import (KSDGException, fplot, Solution)
import numpy as np
import h5py
import matplotlib.pyplot as plt

def parse(args=savargv[1:]):
    parser = ArgumentParser(description='Create movie frames from a time series',
                            allow_abbrev=True)
    parser.add_argument('-p', '--prefix',
                        help='solution file prefix')
    parser.add_argument('--steps', action='store_true',
                        help='use atual time steps')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='start time')
    parser.add_argument('-e', '--end', type=float,
                        help='end time')
    parser.add_argument('-n', '--nframes', type=int, default=3001,
                        help='number frames')
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-c', '--nocolorbar', action='store_true',
                        help="don't plot colorbars")
    parser.add_argument('--names', type=str,
                        help='comma-separated subspace names (for labeling plots)')
    parser.add_argument('--label', type=str, default='t',
                        help='parameter with which to label plots')
    parser.add_argument('-ss', '--subspace', action='append',
                        default=[], help="subspaces to plot")
    parser.add_argument('--noperiodic', action='store_true',
                        help="ignore periodic BC expansion")
    parser.add_argument('frameprefix', help='prefix for frame images')
    clargs = parser.parse_args(args)
    return clargs

defplotopts=dict(
    colorbar=True,
    subspaces=[0, 1],
    periodic='auto',
    variable='auto',
    label='t',
)

def Vufl(soln, t):
    "Make a UFL object for plotting V"
    split = fe.split(soln.function)
    irho = split[0]
    iUs = split[1:]
    soln.ksdg.set_time(t)
    V = (
        soln.V(iUs, irho, params=soln.ksdg.iparams)/
        (soln.ksdg.iparams['sigma']**2/2)
    )
    fs = soln.function.function_space()
    cell = fs.ufl_cell()
    CE = fe.FiniteElement('CG', cell, soln.degree)
    CS = fe.FunctionSpace(fs.mesh(), CE)
    pV = fe.project(V, CS, solver_type='petsc')
    return pV

def plot_curves(t, soln, opts=defplotopts):
    min = soln.meshstats['xmin']
    max = soln.meshstats['xmax']
    nplots = len(opts['subspaces'])
    names = opts['names']
    dim = soln.dim
    soln.load(t)
    width = 4.0*nplots + 2.0*(nplots-1)
    fig = plt.figure(1, figsize=(width,5))
    currplot = 1
    fig.clf()
    params=soln.params(t)
    try:
        labelval = params[opts['label']]
    except KeyError:
        labelval = t
    label = '%s = %.4g'%(opts['label'], labelval)
    for name,subspace in zip(names, opts['subspaces']):
        ra = fig.add_subplot(1, nplots, currplot)
        if type(subspace) == int:
            ssfunc = soln.function.sub(subspace)
        elif subspace == 'V':
            ssfunc = Vufl(soln, t)
        if dim == 1:
            p = fplot(ssfunc, xmin=min, xmax=max)
            plt.title("%s\n%s"%(name, label))
        elif dim == 2:
            p = fe.plot(ssfunc, title="%s\n%s"%(name, label))
            if opts['colorbar']:
                try: 
                    plt.colorbar(p)
                except AttributeError:
                    pass
        else:
            raise KSDGException("can only plot 1 or 2 dimensions")
        dofs = np.array(
            ssfunc.function_space().dofmap().dofs()
        )
        fvec = ssfunc.vector()[:]
        ymin, ymax = np.min(fvec[dofs]), np.max(fvec[dofs])
        plt.xlabel('(%7g, %7g)'%(ymin,ymax), axes=ra)
        currplot += 1
    return(fig)

def decode_subspace(ss):
    try:
        ret = int(ss)
    except ValueError:
        ret = str(ss)
    return ret

def main():
    clargs = parse()
    soln = Solution(clargs.prefix, noperiodic=clargs.noperiodic)
    tmin, tmax = soln.tmin, soln.tmax
    start = clargs.start
    end = clargs.end if clargs.end else tmax
    n = clargs.nframes
    if clargs.steps:
        frname = 'step'
        times = [ t for t in soln.tstimes if t >= start and t <= end ]
    else:
        frname = 'frame'
        times = np.linspace(start, end, num=n)
    if soln.ligands:
        nsubspaces = 1 + soln.ligands.nligands()
    else:
        nsubspaces = 2
    if soln.periodic and soln.noperiodic:
        nsubspaces *= 2**dim
    subspaces = [ decode_subspace(ss) for ss in clargs.subspace ]
    if subspaces == []:
        subspaces = list(range(nsubspaces))
    names = list(['y'+str(i) for i in subspaces])
    if clargs.names:
        nopt = clargs.names.split(',')
        if len(nopt) < len(names):
            names[:len(nopt)] = nopt
        else:
            names = nopt
    plotopts = dict(
        colorbar=not clargs.nocolorbar,
        periodic=soln.periodic and (not clargs.noperiodic),
        subspaces=subspaces,
        variable=soln.variable,
        names=names,
        label=clargs.label,
    )
    xmin = soln.meshstats['xmin']
    xmax = soln.meshstats['xmax']
    for k,t in enumerate(times):
        if t < start:
            continue
        if t > end:
            break
        fig = plot_curves(t, soln, opts=plotopts)
        frame = clargs.frameprefix + '_' + frname + '%05d'%k + '.png'
        if clargs.verbose:
            print('plotting %s %d, t= %7g, %s'%(frname, k, t, frame))
        fig.savefig(frame)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

