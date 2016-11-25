import lie_group_diffeo as lgd
import odl
import numpy as np

# linear interpolation has boundary problems.
space = odl.uniform_discr([-1, -1], [1, 1], [200, 200], interp='nearest')

# Select template and target
v0 = space.element(lambda x: np.exp(-(5 * (x[0] + 0.2)**2 + x[1]**2) / 0.4**2))
v1 = space.element(lambda x: np.exp(-(1 * (x[0] + 0.2)**2 + x[1]**2) / 0.4**2))

# Select functional for data matching
f1 = odl.solvers.L2NormSquared(space).translated(v1)

# Select lie group of deformations
lie_grp = lgd.GLn(2)
#lie_grp = lgd.SOn(2)
deform_action = lgd.MatrixImageAction(lie_grp, space)
#lie_grp = lgd.AffineGroup(2)
#deform_action = lgd.MatrixImageAffineAction(lie_grp, space)

regularizer = 'determinant'
if regularizer == 'image':
    W = space.tangent_bundle
    w1 = W.element([lambda x: x[0],
                    lambda x: x[1]])

    # Create regularizing functional
    f2 = 0.01 * odl.solvers.L2NormSquared(W).translated(w1)

    # Create action
    regularizer_action = lgd.ProductSpaceAction(deform_action, W.size)
elif regularizer == 'point':
    W = odl.ProductSpace(odl.rn(2), 2)
    w1 = W.element([[1, 0],
                    [0, 1]])

    # Create regularizing functional
    f2 = 0.01 * odl.solvers.L2NormSquared(W).translated(w1)

    # Create action
    point_action = lgd.MatrixVectorAction(lie_grp, W[0])
    regularizer_action = lgd.ProductSpaceAction(deform_action, W.size)
elif regularizer == 'determinant':
    W = odl.rn(1)
    w1 = W.element([1])

    # Create regularizing functional
    f2 = 0.01 * odl.solvers.L1Norm(W).translated(w1)

    # Create action
    regularizer_action = lgd.MatrixDeterminantAction(lie_grp, W)
else:
    assert False


assalg = lie_grp.associated_algebra

Ainv = lambda x: x

v = v0.copy()
w = w1.copy()
g = lie_grp.identity

callback = odl.solvers.CallbackShow(display_step=10)

v0.show('starting point')
v1.show('target point')

eps = 0.005
for i in range(1000):
    u = Ainv(deform_action.inf_action_adj(v, f1.gradient(v)) +
             regularizer_action.inf_action_adj(w, f2.gradient(w)))

    if 0:
        v -= eps * deform_action.inf_action(u)(v)
        w -= eps * regularizer_action.inf_action(u)(w)
    else:
        g = g.compose(assalg.exp(-eps * u))
        v = deform_action.action(g)(v0)
        w = regularizer_action.action(g)(w1)

    callback(v)
    print(f1(v) + f2(w))
