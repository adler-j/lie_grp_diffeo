import lie_group_diffeo as lgd
import odl
import numpy as np

typ = 'Affine'

space = odl.uniform_discr(-1, 1, 1000, interp='nearest')

v0 = space.element(lambda x: np.exp(-x**2 / 0.2**2))
v1 = space.element(lambda x: np.exp(-(x-0.1)**2 / 0.3**2))

f1 = odl.solvers.L2NormSquared(space).translated(v1)

# Define the lie group to use.
lie_grp_type = 'affine'
if lie_grp_type == 'gln':
    lie_grp = lgd.GLn(1)
    deform_action = lgd.MatrixImageAction(lie_grp, space)
elif lie_grp_type == 'son':
    lie_grp = lgd.SOn(1)
    deform_action = lgd.MatrixImageAction(lie_grp, space)
elif lie_grp_type == 'sln':
    lie_grp = lgd.SLn(1)
    deform_action = lgd.MatrixImageAction(lie_grp, space)
elif lie_grp_type == 'affine':
    lie_grp = lgd.AffineGroup(1)
    deform_action = lgd.MatrixImageAffineAction(lie_grp, space)
elif lie_grp_type == 'rigid':
    lie_grp = lgd.EuclideanGroup(1)
    deform_action = lgd.MatrixImageAffineAction(lie_grp, space)
else:
    assert False

# Define what regularizer to use
regularizer = 'determinant'
if regularizer == 'image':
    W = space.tangent_bundle
    w1 = W.element([lambda x: x[0]])

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
    f2 = 0.02 * odl.solvers.L2NormSquared(W).translated(w1)

    # Create action
    regularizer_action = lgd.MatrixDeterminantAction(lie_grp, W)
else:
    assert False

assalg = lie_grp.associated_algebra

Ainv = lambda x: x

v = v0.copy()
w = w1.copy()
g = lie_grp.identity

callback = odl.solvers.CallbackShow(lie_grp_type)

callback(v0)
callback(v1)

eps = 0.2
for i in range(30):
    u = Ainv(deform_action.inf_action_adj(v, f1.gradient(v)) +
             regularizer_action.inf_action_adj(w, f2.gradient(w)))

    if 0:
        v -= eps * deform_action.inf_action(u)(v)
        w -= eps * regularizer_action.inf_action(u)(w)
    else:
        g = g.compose(assalg.exp(-eps * u))
        v = deform_action.action(g)(v0)
        w = regularizer_action.action(g)(w1)

    #callback(v)
    print(f1(v) + f2(w))

callback(v)