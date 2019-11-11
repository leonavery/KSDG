"""Random functions that might come in handy some day.

This module contains functions I wrote but ended up mostly not
using. I put them here in the thought that they might be useful
someday.

"""

import sys
import numpy as np
# As a general rule, I import only fenics classes by name.
import fenics as fe
from fenics import (FiniteElement, MixedElement, VectorElement,
                    TestFunctions, FunctionAssigner, FunctionSpace,
                    VectorFunctionSpace, TrialFunction, TestFunction,
                    Function, Constant, Measure)
import matplotlib.pyplot as plt
import ufl
from .ksdgdebug import log
from .ksdgexception import KSDGException

#
# Handy utility function: I used this constantly during
# development. It would be better to beef it up so it's a little
# smarter about plotting, though.
#
def fplot(f, compare=None, dim=None, xmin=0, xmax=1, npoints=100,
          axes=None, legend=None, mesh=None):
    """plot a function f

    Positional parameter:
    f: the function to be plotted

    Keyword parameters:
    dim=1: dimensionality of the argument of f
    npoints=100: number points to plot
    xmin=0, xmax=1: range to plot
    """
    if not dim:
        if mesh:
            dim = mesh.geometry().dim()
        else:
            dim = 1
    if (dim > 1):
        #
        # compare ignored in this case
        #
        p = fe.plot(f, mesh=mesh, axes=axes)
        plt.colorbar(p)
        return p
    if not axes:
        axes = plt.gca()
    xs = np.linspace(xmin, xmax, npoints)
    if compare:
        p = axes.plot(xs, [f(x) for x in xs], 'b',
                      xs, [compare(x) for x in xs], 'r')
        if legend:
            plt.legend(p, legend)
    else:
        p = axes.plot(xs, [f(x) for x in xs], 'b')
    return p

def project_gradient_neumann(
        f0,
        degree=None,
        mesh=None,
        solver_type='gmres',
        preconditioner_type='default'
    ):
    """Find an approximation to f0 that has the same gradient

    The resulting function also satisfies homogeneous Neumann boundary
    conditions.

    Parameters:    
    f0: the function to approximate
    mesh=None: the mesh on which to approximate it If not provided, the
        mesh is extracted from f0.
    degree=None: degree of the polynomial approximation. extracted
        from f0 if not provided. 
    solver_type='gmres': The linear solver type to use.
    preconditioner_type='default': Preconditioner type to use
    """
    if not mesh: mesh = f0.function_space().mesh()
    element = f0.ufl_element()
    if not degree:
        degree = element.degree()
    CE = FiniteElement('CG', mesh.ufl_cell(), degree)
    CS = FunctionSpace(mesh, CE)
    DE = FiniteElement('DG', mesh.ufl_cell(), degree)
    DS = FunctionSpace(mesh, DE)
    CVE = VectorElement('CG', mesh.ufl_cell(), degree - 1)
    CV = FunctionSpace(mesh, CVE)
    RE = FiniteElement('R', mesh.ufl_cell(), 0)
    R = FunctionSpace(mesh, RE)
    CRE = MixedElement([CE, RE])
    CR = FunctionSpace(mesh, CRE)
    f = fe.project(f0, CS,
                   solver_type=solver_type,
                   preconditioner_type=preconditioner_type)
    g = fe.project(fe.grad(f), CV,
                   solver_type=solver_type,
                   preconditioner_type=preconditioner_type)
    lf = fe.project(fe.nabla_div(g), CS,
                    solver_type=solver_type,
                    preconditioner_type=preconditioner_type)
    tf, tc = TrialFunction(CR)
    wf, wc = TestFunctions(CR)
    dx = Measure('dx', domain=mesh,
                 metadata={'quadrature_degree': min(degree, 10)})
    a = (fe.dot(fe.grad(tf), fe.grad(wf)) + tc * wf + tf * wc) * dx
    L = (f * wc - lf * wf) * dx
    igc = Function(CR)
    fe.solve(a == L, igc,
             solver_parameters={'linear_solver': solver_type,
                                 'preconditioner': preconditioner_type}
    )
    ig, c = igc.sub(0), igc.sub(1)
    igd = fe.project(ig, DS,
                     solver_type=solver_type,
                     preconditioner_type=preconditioner_type)
    return igd

    
def project_gradient(
        f0,
        mesh=None,
        degree=None,
        debug=False,
        solver_type='gmres',
        preconditioner_type='default'
    ):
    """Find an approximation to f0 that has the same gradient

    Parameters:    
    f0: the function to approximate
    mesh=None: the mesh on which to approximate it. If not provided, the
        mesh is extracted from f0.
    degree=None: degree of the polynomial approximation. extracted
        from f0 if not provided. 
    solver_type='gmres': The linear solver type to use.
    preconditioner_type='default': Preconditioner type to use
    """
    if not mesh: mesh = f0.function_space().mesh()
    element = f0.ufl_element()
    if not degree:
        degree = element.degree()
    CE = FiniteElement('CG', mesh.ufl_cell(), degree)
    CS = FunctionSpace(mesh, CE)
    DE = FiniteElement('DG', mesh.ufl_cell(), degree)
    DS = FunctionSpace(mesh, DE)
    CVE = VectorElement('CG', mesh.ufl_cell(), degree - 1)
    CV = FunctionSpace(mesh, CVE)
    RE = FiniteElement('R', mesh.ufl_cell(), 0)
    R = FunctionSpace(mesh, RE)
    CRE = MixedElement([CE, RE])
    CR = FunctionSpace(mesh, CRE)
    f = fe.project(f0, CS,
                   solver_type=solver_type,
                   preconditioner_type= preconditioner_type)
    g = fe.project(fe.grad(f), CV,
                   solver_type=solver_type,
                   preconditioner_type= preconditioner_type)
    tf, tc = TrialFunction(CR)
    wf, wc = TestFunctions(CR)
    dx = Measure('dx', domain=mesh,
                 metadata={'quadrature_degree': min(degree, 10)})
    a = (fe.dot(fe.grad(tf), fe.grad(wf)) + tc * wf + tf * wc) * fe.dx
    L = (f * wc + fe.dot(g, fe.grad(wf))) * fe.dx
    igc = Function(CR)
    fe.solve(a == L, igc,
             solver_parameters={'linear_solver': solver_type,
                                 'preconditioner': preconditioner_type}
    )
    if debug:
        print('igc', igc.vector()[:])
    assigner = FunctionAssigner(CS, CR.sub(0))
#    ig = Function(CS)
#    assigner.assign(ig, igc.sub(0))
#    fe.assign(ig, igc.sub(0))
#    if debug:
#        print('ig', igc.sub(0).vector()[:])
    igd = fe.project(igc.sub(0), DS,
                     solver_type=solver_type,
                     preconditioner_type=preconditioner_type)
    if debug:
        print('igd', igd.vector()[:])
    return igd
