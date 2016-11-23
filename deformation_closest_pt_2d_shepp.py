from SOn import SOn
from GLn import GLn
import odl
import numpy as np
import scipy as sp

# linear interpolation has boundary problems.
space = odl.uniform_discr([-1, -1], [1, 1], [300, 300], interp='linear')


if 0:
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: x[0] * 0.3 + x[1] * 0.2,
                                      lambda x: x[0] * 0.1 + x[1] * 0.3]))
else:
    theta = 0.2
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: (np.cos(theta) - 1) * x[0] + np.sin(theta) * x[1],
                                      lambda x: -np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1]]))

v0 = odl.phantom.shepp_logan(space, modified=True)
v1 = transform(v0)

f1 = odl.solvers.L2NormSquared(space).translated(v1)

W = odl.ProductSpace(odl.rn(2), 2)
w1 = W.element([[1, 0],
                [0, 1]])
# Something is seriously wrong here, this should not need to be negative.
f2 = 0.01 * odl.solvers.L2NormSquared(W).translated(w1)

lie_grp = SOn(2)
assalg = lie_grp.associated_algebra

Ainv = lambda x: x

v = v0.copy()
w = w1.copy()
g = lie_grp.identity

callback = odl.solvers.CallbackShow(display_step=50)

v0.show('starting point')
v1.show('target point')

eps = 0.005
for i in range(1000):
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
    print(f1(v) + f2(w))
