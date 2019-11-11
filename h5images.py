#!/usr/bin/env python3 -u

import IPython
import sys
import os
import pickle
import json
savargv = sys.argv
sys.argv = sys.argv[:1]        # Hide argv from PETSc
from argparse import ArgumentParser
import fenics as fe
from KSDG import (KSDGException, fplot, Solution)
import numpy as np
import h5py

def parse(args=savargv[1:]):
    parser = ArgumentParser(description='Save solution time points in individual HDF5 files',
                            allow_abbrev=True)
    parser.add_argument('-p', '--prefix',
                        help='solution file prefix')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='start time')
    parser.add_argument('-e', '--end', type=float,
                        help='end time')
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('imageprefix', help='prefix for image HDF5 files')
    clargs = parser.parse_args(args)
    return clargs

def main():
    clargs = parse()
    soln = Solution(clargs.prefix)
    tmin, tmax = soln.tmin, soln.tmax
    start = clargs.start
    end = clargs.end if clargs.end else tmax
    frname = 'step'
    times = [ t for t in soln.tstimes if t >= start and t <= end ]
    for k,t in enumerate(times):
        if t < start:
            continue
        if t > end:
            break
        frame = clargs.imageprefix + '_' + frname + '%05d'%k + '.h5'
        if clargs.verbose:
            print(
                'saving %s %d, t= %7g, %s'%(frname, k, t, frame),
                flush=True
            )
        images = soln.images(t)
        h5img = h5py.File(frame, 'w')
        h5img['t'] = t
        h5img['images'] = images
        try:
            params = soln.params(t)
        except KeyError:
            params = {'t': t}
        h5img['params'] = json.dumps(params)
        h5img.close()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

