import lie_group_diffeo as lgd
import odl
import numpy as np

space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='linear')
coord_space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='linear').tangent_bundle

# Select deformation type of the target
transform_type = 'affine'
if transform_type == 'affine':
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: x[0] * 0.1 + x[1] * 0.1,
                                      lambda x: x[0] * 0.03 + x[1] * 0.1]))
elif transform_type == 'rotate':
    theta = 0.2
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: (np.cos(theta) - 1) * x[0] + np.sin(theta) * x[1],
                                      lambda x: -np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1]]))
else:
    assert False

# Create template and target
template = odl.phantom.shepp_logan(space, modified=True)
target = transform(template)

# Define data matching functional
data_matching = odl.solvers.L1Norm(space).translated(target)

lie_grp = lgd.Diff(space, coord_space=coord_space)
deform_action = lgd.GeometricDeformationAction(lie_grp, space)

w = space.one()

# Create regularizing functional
regularizer = 5 * odl.solvers.KullbackLeibler(space, prior=w)

# Create action
regularizer_action = lgd.JacobianDeterminantScalingAction(lie_grp, space)

# Initial guess
g = lie_grp.identity

# Combine action and functional into single object.
action = lgd.ProductSpaceAction(deform_action, regularizer_action)
x = action.domain.element([template, w]).copy()
f = odl.solvers.SeparableSum(data_matching, regularizer)

# Show some results, reuse the plot
template.show('template')
target.show('target')

# Create callback that displays the current iterate and prints the function
# value
callback = odl.solvers.CallbackShow('diffemorphic matching', display_step=20)
callback &= odl.solvers.CallbackPrint(f)

# Solve via gradient flow
result = lgd.gradient_flow_solver(x, f, g, action,
                                  niter=2000, line_search=0.00001,
                                  callback=callback)

result.data.show('Resulting diffeo')
(result.data - lie_grp.identity.data).show('translations')
