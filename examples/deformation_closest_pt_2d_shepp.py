from lie_group_diffeo import GLn, SOn, MatrixImageAction, MatrixVectorAction, ProductSpaceAction
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
image_action = MatrixImageAction(lie_grp, space)
point_action = ProductSpaceAction(MatrixVectorAction(lie_grp, W[0]), 2)

Ainv = lambda x: x

v = v0.copy()
w = w1.copy()
g = lie_grp.identity

callback = odl.solvers.CallbackShow(display_step=50)

v0.show('starting point')
v1.show('target point')

eps = 0.005
for i in range(1000):
    u = Ainv(image_action.inf_action_adj(v, f1.gradient(v)) +
             point_action.inf_action_adj(w, f2.gradient(w)))

    if 0:
        v -= eps * image_action.inf_action(u)(v)
        w -= eps * point_action.inf_action(u)(w)
    else:
        g = g.compose(assalg.exp(-eps * u))
        v = image_action.action(g)(v0)
        w = point_action.action(g)(w1)

    callback(v)
    print(f1(v) + f2(w))
