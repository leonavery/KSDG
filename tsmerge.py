#!/usr/bin/env python3

import sys
savargv = sys.argv
sys.argv = sys.argv[:1]        # Hide argv from PETSc
from argparse import ArgumentParser
from KSDG import KSDGTimeSeries
import numpy as np

def main():
    parser = ArgumentParser(description='Merge time series',
                            allow_abbrev=True)
    parser.add_argument('-o', '--outfile',
                        help='merged file name')
    parser.add_argument('infiles', nargs='+', help='files to merge')
    clargs = parser.parse_args(savargv[1:])
    sys.argv = sys.argv[0:0]
    ins = [KSDGTimeSeries(ifn, 'r') for ifn in clargs.infiles]
    out = KSDGTimeSeries(clargs.outfile, 'w')
    times = np.empty((0), dtype=float)
    files = np.empty((0), dtype=int)
    for f,ts in enumerate(ins):
        st = ts.sorted_times()
        times = np.append(times, st)
        files = np.append(files, np.full_like(st, f, dtype=int))
    order = times.argsort()
    lastt = float('nan')
    k = 0
    for point in order:
        t = times[point]
        if t == lastt: continue
        f = files[point]
        ts = ins[f]
        vec = ts.retrieve_by_time(t)
        out.store(vec, t, k=k)
        lastt = t
        k += 1
    out.close()
    for ts in ins:
        ts.close()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

