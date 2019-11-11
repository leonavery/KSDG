#!/usr/bin/env python3
"""Class for extracting information about a PDE solution.

Expected usage:
    from KSDG import Solution

    soln = Solution(prefix)
    values = soln.project(t)
    coords = soln.coords()
    params = soln.params(t)

General ideas: First, you should need only to pass a prefix at
creation time. The prefix is like a filename, but various things are
appended to it to get the names of the actual files in which the
solution is stored. (E.g., prefix + '_ts.h5' is the name of the
TimeSeries file in which numerical results are stored.) You shouldn't
need to pass in any other information -- all necessary information
about the problem and solution are extracted from the files. Some of
this information is made available as member attributes,
e.g. soln.ksdg is the KSDGSolver, soln.timeSeries is the TimeSeries,
and soln.periodic is True iff the problem was solved with periodic
boundary conditions. Information requiring more analysis is extracted
through functions. Typically such a function will take time as an
argument.

Second, this module is intended to be used as an interface module with
Mathematica. Therefore, functions will typically return simple types
that can be converted cleanly to Mathematica types.

The basic framework of this module was taken from moviemaker.py. (It
is likely that there will be a new version of moviemaker that uses
this module.)

Note: this class is not MPI-safe. Run as a single process only.
"""

import sys
import os
import copy
import dill, pickle
import numpy as np
import h5py
import collections
import fenics as fe
import ufl
from fenics import (Mesh, Constant, FiniteElement, FunctionSpace,
                    Function, MixedElement)
from .ksdgtimeseries import KSDGTimeSeries
from .ksdgexception import KSDGException
from .ksdgmakesolver import makeKSDGSolver
from .ksdggather import (remap_from_files, fsinfo_filename,
                         integerify_transform, integerify, subspaces,
                         local_dofs)
from .ksdgperiodic import mesh_stats
from .ksdgexpand import ExpandedMesh
from .ksdgligand import ParameterList

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

def CGelement(fs):
    """Make a CG element corresponding to a DG FunctionSpace."""
    elements = []
    for ss in (fs.split()):
        CE = FiniteElement('CG', ss.ufl_cell(), ss.ufl_element().degree())
        elements.append(CE)
    return MixedElement(elements)

class Solution():
    def __init__(self,
                 prefix,
                 noperiodic=False
        ):
        """Access a KSDG solution.

        Required argument:
        prefix: This is the prefix for the names of the files in which
        the solution is stored. (Typically, it would be the value of
        the --save option to ksdgsolver<d>.py.

        optional argument:
        noperiodic=False: Treat solution as periodic, even if it wasn't.
        """
        self.prefix = os.path.expanduser(prefix)
        self.prefix = os.path.expandvars(self.prefix)
        self.noperiodic = noperiodic
        self.meshfile = self.prefix + '_omesh.xml.gz'
        if os.path.isfile(self.meshfile):
            self.mesh = self.omesh = Mesh(self.meshfile)
        else:
            self.meshfile = prefix + '_mesh.xml.gz'
            self.mesh = Mesh(self.meshfile)
        self.optionsfile = self.prefix + '_options.txt'
        with open(self.optionsfile, 'r') as optsf:
            self.options = optsf.readlines()
        self.fsinfofile = fsinfo_filename(self.prefix, 0)
        with h5py.File(self.fsinfofile, 'r') as fsf:
            self.dim, self.degree = (
                int(fsf['dim'][()]),
                int(fsf['degree'][()]),
            )
            try:
                # self.periodic = fsf['periodic'].value #deprecated
                self.periodic = fsf['periodic'][()]
            except KeyError:
                self.periodic = False
            try:
                self.ligands = pickle.loads(fsf['ligands'][()])
            except KeyError:
                self.ligands = False
            try:
                self.param_names = pickle.loads(fsf['param_names'][()])
                self.variable = True
            except KeyError:
                self.param_names = []
                self.variable = False
            try:
                self.params0 = pickle.loads(fsf['params0'][()])
            except KeyError:
                self.params0 = {}
            try:
                pfuncs = fsf['param_funcs'][()]
                self.param_funcs = dill.loads(pfuncs.tobytes())
            except (KeyError, ValueError):
                self.param_funcs = {}
        def identity(t, params={}):
            return t
        self.param_funcs['t'] = identity
        if self.params0 and self.ligands:
            capp = [s for s in self.options if '--cappotential' in s]
            if 'tophat' in capp[0]:
                self.cappotential = 'tophat'
            elif 'witch' in capp[0]:
                self.cappotential = 'witch'
            self.Vgroups = copy.deepcopy(self.ligands)
            self.Vparams = ParameterList(default_parameters)
            self.Vparams.add(self.Vgroups.params())
            def Vfunc(Us, params={}):
                self.Vparams.update(params) # copy params into ligands
                return self.Vgroups.V(Us)   # compute V
            def Vtophat(rho, params={}):
                tanh = ufl.tanh((rho - params['rhomax'])/params['cushion'])
                return params['maxscale'] * params['sigma']**2 / 2 * (tanh + 1)
            def Vwitch(rho, params={}):
                tanh = ufl.tanh((rho - params['rhomax'])/params['cushion'])
                return (params['maxscale'] * params['sigma']**2 / 2 *
                        (tanh + 1) * (rho / params['rhomax'])
                )
            Vcap = Vwitch if self.cappotential == 'witch' else Vtophat
            def V2(Us, rho, params={}):
                return Vfunc(Us, params=params) + Vcap(rho, params=params)
            self.V = V2
        self.tsfile = prefix + '_ts.h5'
        self.ts = self.timeSeries = KSDGTimeSeries(self.tsfile, 'r')
        self.tstimes = self.ts.sorted_times()
        self.tmin, self.tmax = self.tstimes[0], self.tstimes[-1]
        kwargs = dict(
            mesh=self.mesh,
            dim=self.dim,
            degree=self.degree,
            t0=self.tmin,
            ligands=self.ligands,
            parameters=self.params0,
            V=V2,
            periodic=self.periodic
        )
        if self.variable:
            kwargs['param_funcs'] = self.param_funcs
        self.ksdg = makeKSDGSolver(**kwargs)
        # self.V = self.ksdg.V
        if self.periodic and self.noperiodic:
            self.mesh = self.ksdg.mesh
            self.meshstats = mesh_stats(self.ksdg.mesh)
        else:
            try:
                self.meshstats = mesh_stats(self.ksdg.omesh)
            except AttributeError:
                self.meshstats = mesh_stats(self.ksdg.mesh)
        if self.periodic and not self.noperiodic:
            self.ex_mesh = ExpandedMesh(self.ksdg.sol)
            self.function = self.ex_mesh.expanded_function
        else:
            self.ex_mesh = None
            self.function = self.ksdg.sol
        self.fs = self.function.function_space()
        self.transform = integerify_transform(
            np.reshape(self.fs.tabulate_dof_coordinates(), (-1, self.dim))
        )

    def params(self, t):
        pd = collections.OrderedDict(
            [(name, self.param_funcs[name](t, params=self.params0))
             for name in self.param_names]
        )
        return pd

    def load(self, t):
        """Load solution for time t."""
        vec = self.ts.retrieve_by_time(t)
        if self.periodic and not self.noperiodic:
            self.ex_mesh.sub_function.vector()[:] = vec
            self.ex_mesh.sub_function.vector().apply('insert')
            self.ex_mesh.expand()
        else:
            self.ksdg.sol.vector()[:] = vec
            self.ksdg.sol.vector().apply('insert')
        
    def project(self, t=None):
        """Project solution onto CG subspace.

        soln.project(t)

        project projects the solution at time t onto a CG
        FunctionSpace. The projected function is left in
        self.CGfunction. If argument t is not supplied, the currently
        loaded solution will be used. (In this case, you must have
        called load(t) before calling project().

        project returns self.CGfunction as its value.
        """
        solver_type = 'petsc'
        if t is not None:
            self.load(t)
        if not hasattr(self, 'CS'):
            #
            # create CG FunctionSpace
            #
            self.CE = CGelement(self.fs)
            self.CS = FunctionSpace(self.fs.mesh(), self.CE)
        self.CGfunction = fe.project(self.function, self.CS,
                   solver_type=solver_type)
        return self.CGfunction

    def images(self, t=None):
        self.project(t)
        if not hasattr(self, 'ims'):
            self.CGdofs = local_dofs(self.CS)
            order = np.argsort(self.CGdofs[:, -1])
            self.CGdofs = self.CGdofs[order]
            self.CGintdofs = np.empty_like(self.CGdofs, dtype=int)
            self.CGintdofs[:, :2] = np.rint(self.CGdofs[:, :2],
                                            casting='unsafe')
            self.CGintdofs[:, -1] = np.rint(self.CGdofs[:, -1],
                                            casting='unsafe')
            self.CGintdofs[:, 2:-1] = integerify(self.CGdofs[:, 2:-1],
                                                 transform=self.transform)
            self.sss = np.unique(self.CGintdofs[:, 1])
            self.nss = len(self.sss)
            assert np.alltrue(self.sss == np.arange(self.nss))
            self.dims = np.zeros(shape=(self.dim,), dtype=int)
            for i in range(self.dim):
                ixs = np.unique(self.CGintdofs[:, 2+i])
                self.dims[i] = len(ixs)
                assert np.alltrue(ixs == np.arange(self.dims[i]))
            self.imshape = (self.nss,) + tuple(self.dims)
            assert self.CS.dim() == np.prod(self.imshape)
            checknums = 1 + np.arange(self.CS.dim())
            self.iims = np.zeros(shape=self.imshape, dtype=int)
            self.CGixs = (
                (self.CGintdofs[:, 1],) +
                tuple(self.CGintdofs[:, 2:-1].transpose())
            )
            self.iims[self.CGixs] = checknums
            assert np.alltrue(self.iims > 0)
            del(self.iims)
            self.ims = np.empty(shape=self.imshape)
        self.ims[self.CGixs] = self.CGfunction.vector()[:]
        return self.ims

    def Vufl(self, t):
        "Make a UFL object for plotting V"
        self.project(t)
        split = fe.split(self.function)
        irho = split[0]
        iUs = split[1:]
        self.ksdg.set_time(t)
        V = (
            self.V(iUs, irho, params=self.ksdg.iparams)/
            (self.ksdg.iparams['sigma']**2/2)
        )
        fs = self.function.function_space()
        cell = fs.ufl_cell()
        self.SE = fe.FiniteElement('CG', cell, self.degree)
        self.SS = fe.FunctionSpace(fs.mesh(), self.SE)
        pV = fe.project(V, self.SS, solver_type='petsc')
        return pV

    def Vimage(self, t=None):
        pV = self.Vufl(t)
        self.Vimshape = tuple(self.dims)
        self.VCGcoords = np.reshape(
            self.SS.tabulate_dof_coordinates(), (-1, self.dim)
        )
        self.VCGintcoords = integerify(self.VCGcoords,
                                       transform=self.transform)
        self.Vixs = (
            tuple(self.VCGintcoords.transpose())
        )
        self.Vimg = np.zeros(self.Vimshape, dtype=float)
        self.Vimg[self.Vixs] = pV.vector()[:]
        return self.Vimg

    
def main():
    prefix = sys.argv[1]
    soln = Solution(prefix)
    imgs = soln.images(0)
    Vimg = soln.Vimage(0)

if __name__ == "__main__":
    # execute only if run as a script
    main()
                
