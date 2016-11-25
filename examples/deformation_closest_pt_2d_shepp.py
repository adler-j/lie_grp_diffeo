import lie_group_diffeo as lgd
import odl
import numpy as np

# linear interpolation has boundary problems.
space = odl.uniform_discr([-1, -1], [1, 1], [300, 300], interp='linear')

transform_type = 'affine'
if transform_type == 'affine':
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: x[0] * 0.3 + x[1] * 0.2,
                                      lambda x: x[0] * 0.1 + x[1] * 0.3]))
elif transform_type == 'rotate':
    theta = 0.2
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: (np.cos(theta) - 1) * x[0] + np.sin(theta) * x[1],
                                      lambda x: -np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1]]))
else:
    assert False

v0 = odl.phantom.shepp_logan(space, modified=True)
v1 = transform(v0)

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
    f2 = 0.2 * odl.solvers.L2NormSquared(W).translated(w1)

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
