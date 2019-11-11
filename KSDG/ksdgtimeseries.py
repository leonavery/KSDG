import h5py, os
from petsc4py import PETSc
import fenics as fe
from fenics import HDF5File
import numpy as np
from mpi4py import MPI
from .ksdgdebug import log

def logSERIES(*args, **kwargs):
    log(*args, system='SERIES', **kwargs)

class KSDGTimeSeries():

    def __init__(self, tsname, mode='r+'):
        """Open an H5TimeSeries."""
        dirname = os.path.dirname(os.path.abspath(tsname))
        os.makedirs(dirname, exist_ok=True)
        self.tsf = h5py.File(tsname, mode)
        self.lastk = 0
        if 'times' in self.tsf:
            self.ts = np.array(self.tsf['times'])
            try:
                self.ks = np.array(self.tsf['ks'])
            except KeyError:
                self.ks = np.arange(len(self.ts))
            self.order = np.array(self.tsf['order'])
        else:
            self.ts = np.array([], dtype=float)
            self.ks = np.array([], dtype=int)
            self.order = np.array([], dtype=int)
        self.sorted = False

    def try_to_set(self, key, val):
        try:
            del self.tsf[key]
        except KeyError:
            pass
        try:
            self.tsf[key] = val
        except ValueError:
            pass

        
    def _sort(self):
        if self.sorted: return
        self.try_to_set('times', self.ts)
        self.order = self.ts.argsort()
        self.try_to_set('order', self.order)
        self.sts = self.ts
        self.sts.sort()
        self.try_to_set('ks', self.ks)
        self.try_to_set('lastk', self.lastk)
        self.sorted = True
        
    def close(self):
        if not hasattr(self, 'tsf'): return
        self._sort()
        self.tsf.close()
        del self.tsf

    def __del__(self):
        self.close()

    def store(self, data, t, k=None):
        logSERIES('k, t', k, t)
        if k is None:
            k = self.lastk + 1
        self.lastk = k
        self.ks = np.append(self.ks, k)
        self.ts = np.append(self.ts, t)
        key = 'data' + str(k)
        self.tsf[key] = data
        self.tsf[key].attrs['k'] = k
        self.tsf[key].attrs['t'] = t
        self.sorted = False

    def times(self):
        self._sort()
        return self.ts

    def steps(self):
        self._sort()
        return self.ks

    def sorted_times(self):
        self._sort()
        return self.sts

    def sorted_steps(self):
        self._sort()
        return self.order

    def retrieve_by_number(self, k):
        key = 'data' + str(k)
        return np.array(self.tsf[key])

    def retrieve_by_time(self, t):
        """
        Retrieve a time point.
        
        Arguments:
        t: the time to be retrieved.
        """
        self._sort()
        if (t <= self.sts[0]):
            t = self.sts[0]
            a = 0
        elif (t >= self.sts[-1]):
            t = self.sts[-1]
            a = len(self.sts) - 1
        else:
            a = self.sts.searchsorted(t)
        na = self.order[a]
        ta = self.sts[a]
        adata = self.retrieve_by_number(self.ks[na])
        if (a >= len(self.order) - 1):
            data = adata
        elif ta == t:
            data = adata
        else:
            b = a + 1
            nb = self.order[b]
            tb = self.sts[b]
            bdata = self.retrieve_by_number(self.ks[nb])
            data = ((t-ta)*bdata + (tb-t)*adata)/(tb-ta)
        return(data)
