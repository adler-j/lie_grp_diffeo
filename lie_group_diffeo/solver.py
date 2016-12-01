"""Definitions of solvers."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import odl
from lie_group_diffeo.action import LieAction


__all__ = ('gradient_flow_solver',)


def gradient_flow_solver(x, f, g, action, niter, line_search=1,
                         Ainv=None, callback=None, method='compose'):
    """Gradient flow solver for diffeomorphic image reconstruction.

    Notes
    -----
    Given a functional :math:`f : M \\rightarrow \mathbb{R}`, a lie group
    :math:`G` an action :math:`\\Phi : G \\times M \\rightarrow M` and a
    template :math:`x_0`, this solver solves

    .. math::
        \\min_{g \in G} f(\Phi(g, x_0))
    """

    assert isinstance(f, odl.solvers.Functional)
    assert isinstance(action, LieAction)
    assert x in f.domain
    assert action.domain == f.domain
    assert action.lie_group == g.lie_group
    assert method in ('compose', 'add')

    if not callable(line_search):
        line_search = odl.solvers.ConstantLineSearch(line_search)

    algebra = action.lie_group.associated_algebra

    if Ainv is None:
        Ainv = odl.IdentityOperator(algebra)
    else:
        assert isinstance(Ainv, odl.Operator)
        assert Ainv.domain == algebra
        assert Ainv.range == algebra

    x0 = x.copy()

    for i in range(niter):
        fgrad_x = f.gradient(x)
        u = Ainv(action.momentum_map(x, fgrad_x))

        direction = -action.inf_action(u)(x)
        dir_derivative = fgrad_x.inner(direction)
        steplen = line_search(x, direction, dir_derivative)

        if 0:
            x -= steplen * direction
        else:
            g = g.compose(algebra.exp(-steplen * u))
            x.assign(action.action(g)(x0))

        if callback is not None:
            callback(x)

    return g
