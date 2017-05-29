import lie_group_diffeo as lgd
import odl
import numpy as np

action_type = 'geometric'
transform_type = 'affine'

space = odl.uniform_discr([-1, -1], [1, 1], [101, 101], interp='linear')
coord_space = odl.uniform_discr([-1, -1], [1, 1], [101, 101], interp='linear').tangent_bundle

# Select deformation type of the target
if transform_type == 'affine':
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: x[0] * 0.1 + x[1] * 0.1,
                                      lambda x: x[0] * 0.03 + x[1] * 0.1]) * 0.5)
elif transform_type == 'rotate':
    theta = 0.2
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: (np.cos(theta) - 1) * x[0] + np.sin(theta) * x[1],
                                      lambda x: -np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1]]))
else:
    assert False

# Create template and target
template = odl.phantom.shepp_logan(space, modified=True)
#template = odl.phantom.derenzo_sources(space)
target = transform(template)

# template, target = target, template

# Define data matching functional
data_matching = odl.solvers.L2Norm(space).translated(target)

lie_grp = lgd.Diff(space, coord_space=coord_space)
geometric_deform_action = lgd.GeometricDeformationAction(lie_grp, space)
scale_action = lgd.JacobianDeterminantScalingAction(lie_grp, space)
if action_type == 'mass_preserving':
    deform_action = lgd.ComposedAction(geometric_deform_action, scale_action)
elif action_type == 'geometric':
    deform_action = geometric_deform_action
else:
    assert False

w = space.one()

grid = space.element(lambda x: np.cos(x[0] * np.pi * 5)**20 + np.cos(x[1] * np.pi * 5)**20)

# Create regularizing functional
regularizer = 1 * odl.solvers.KullbackLeibler(space, prior=w)
#regularizer = 2 * odl.solvers.L2NormSquared(space).translated(w)

# Create action
regularizer_action = lgd.JacobianDeterminantScalingAction(lie_grp, space)

# Initial guess
g = lie_grp.identity

# Combine action and functional into single object.
action = lgd.ProductSpaceAction(deform_action, regularizer_action, geometric_deform_action)
x = action.domain.element([template, w, grid]).copy()
f = odl.solvers.SeparableSum(data_matching, regularizer, odl.solvers.ZeroFunctional(space))

# Show some results, reuse the plot
template.show('template')
target.show('target')

# Create callback that displays the current iterate and prints the function
# value
callback = odl.solvers.CallbackShow('diffemorphic matching', step=20)
callback &= odl.solvers.CallbackPrint(f)

# Smoothing
filter_width = 1.0  # standard deviation of the Gaussian filter
ft = odl.trafos.FourierTransform(space)
c = filter_width ** 2 / 4.0 ** 2
gaussian = ft.range.element(lambda x: np.exp(-np.sqrt((x[0] ** 2 + x[1] ** 2) * c)))
convolution = ft.inverse * gaussian * ft
class AinvClass(odl.Operator):
    def _call(self, x):
        return [convolution(di) for di in x.data]
Ainv = AinvClass(domain=lie_grp.associated_algebra, range=lie_grp.associated_algebra, linear=True)

# Step length method
def steplen(itern):
    # print(5e-2 / np.log(2 + itern))
    return 1e-2 / np.log(10 + itern)
line_search = odl.solvers.PredefinedLineSearch(steplen)
# line_search = 3e-4

# Solve via gradient flow
result = lgd.gradient_flow_solver(x, f, g, action, Ainv=Ainv,
                                  niter=2000, line_search=line_search,
                                  callback=callback)

result.data.show('Resulting diffeo')
(result.data - lie_grp.identity.data).show('translations')
(result.data_inv - lie_grp.identity.data).show('translations inverse')
