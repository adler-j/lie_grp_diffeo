from SOn import SOn
from GLn import GLn
import odl
import numpy as np
import scipy as sp

if __name__ == '__main__':
    space = odl.uniform_discr(-1, 1, 1000, interp='linear')

    v0 = space.element(lambda x: np.exp(-x**2 / 0.2**2))
    v1 = space.element(lambda x: np.exp(-(x-0.02)**2 / 0.3**2))

    f1 = odl.solvers.L2NormSquared(space).translated(v1)

    W = space.tangent_bundle
    w1 = W.element([lambda x: x])
    # Something is seriously wrong here, this should not need to be negative.
    f2 = -0.1 * odl.solvers.L2NormSquared(W).translated(w1)

    lie_grp = GLn(1)
    assalg = lie_grp.associated_algebra

    Ainv = lambda x: x

    v = v0.copy()
    w = w1.copy()
    g = lie_grp.identity

    callback = odl.solvers.CallbackShow()

    callback(v0)
    callback(v1)

    eps = 0.5
    for i in range(20):
        u = Ainv(assalg.inf_action_adj(space, v, f1.gradient(v)) +
                 assalg.inf_action_adj(W, w, f2.gradient(w)))

        if 0:
            v -= eps * u.inf_action(space)(v)
            w -= eps * u.inf_action(W)(w)
        else:
            g = g.compose(assalg.exp(-eps * u))
            v = g.action(space)(v0)
            w = g.action(W)(w1)

        callback(v)
        print(f1(v) - f2(w))
