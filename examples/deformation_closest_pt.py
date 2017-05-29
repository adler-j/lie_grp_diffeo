import lie_group_diffeo as lgd
import odl
import numpy as np

space = odl.uniform_discr(-1, 1, 1000, interp='nearest')

# Define template
template = space.element(lambda x: np.exp(-x**2 / 0.2**2))
target = space.element(lambda x: np.exp(-(x-0.1)**2 / 0.3**2))

# Define data matching functional
data_matching = odl.solvers.L2NormSquared(space).translated(target)

# Define the lie group to use.
lie_grp_type = 'rigid'
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
regularizer = 'image'
if regularizer == 'image':
    # Create set of all points in space
    W = space.tangent_bundle
    w = W.element(space.points().T)

    # Create regularizing functional
    regularizer = 0.01 * odl.solvers.L2NormSquared(W).translated(w)

    # Create action
    regularizer_action = lgd.ProductSpaceAction(deform_action, W.size)
elif regularizer == 'point':
    W = odl.ProductSpace(odl.rn(space.ndim), 2)
    w = W.element([[0], [1]])

    # Create regularizing functional
    regularizer = 0.01 * odl.solvers.L2NormSquared(W).translated(w)

    # Create action
    if lie_grp_type == 'affine' or lie_grp_type == 'rigid':
        point_action = lgd.MatrixVectorAffineAction(lie_grp, W[0])
    else:
        point_action = lgd.MatrixVectorAction(lie_grp, W[0])
    regularizer_action = lgd.ProductSpaceAction(point_action, W.size)
elif regularizer == 'determinant':
    W = odl.rn(1)
    w = W.element([1])

    # Create regularizing functional
    regularizer = 0.2 * odl.solvers.L2NormSquared(W).translated(w)

    # Create action
    regularizer_action = lgd.MatrixDeterminantAction(lie_grp, W)
else:
    assert False

# Initial guess
g = lie_grp.identity

# Combine action and functional into single object.
action = lgd.ProductSpaceAction(deform_action, regularizer_action)
x = action.domain.element([template, w]).copy()
f = odl.solvers.SeparableSum(data_matching, regularizer)

# Show some results, reuse the plot
fig = template.show()
target.show(fig=fig)

# Create callback that displays the current iterate and prints the function
# value
callback = odl.solvers.CallbackShow(lie_grp_type, step=10,
                                    indices=0, fig=[fig])
callback &= odl.solvers.CallbackPrint(f)

# Solve via gradient flow
lgd.gradient_flow_solver(x, f, g, action,
                         niter=50, line_search=0.2, callback=callback)
