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
        space.tangent_bundle.element([lambda x: -0.1,
                                      lambda x: 0.1 + x[1] * 0.1]))
elif transform_type == 'rotate':
    theta = 0.2
    transform = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: (np.cos(theta) - 1) * x[0] + np.sin(theta) * x[1],
                                      lambda x: -np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1]]))
else:
    assert False

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(np.pi / 3, 2 * np.pi * 2.0 / 3, 10)
detector_partition = odl.uniform_partition(-2, 2, 300)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create template and target
template = odl.phantom.shepp_logan(space, modified=True)
target = transform(template)

target, template = template, target

data = ray_trafo(target)

# template, target = target, template

# Define data matching functional
data_matching = odl.solvers.L2Norm(ray_trafo.range).translated(data) * ray_trafo

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
weighting = coord_space[0].element(
    lambda x: np.exp(-sum((xi/0.70)**10 for xi in x)))

grid = space.element(lambda x: np.cos(x[0] * np.pi * 5)**20 + np.cos(x[1] * np.pi * 5)**20)

# Create regularizing functional
# regularizer = 3 * odl.solvers.KullbackLeibler(space, prior=w)
regularizer = 0.5 * (odl.solvers.L2NormSquared(space) * weighting).translated(w)

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
callback = odl.solvers.CallbackShow('diffemorphic matching', display_step=20)
callback &= odl.solvers.CallbackPrint(f)

# Smoothing
filter_width = 0.5  # standard deviation of the Gaussian filter
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
    return 3e-1 / (10 + itern)
line_search = odl.solvers.PredefinedLineSearch(steplen)

# Solve via gradient flow
result = lgd.gradient_flow_solver(x, f, g, action, Ainv=Ainv,
                                  niter=2000, line_search=line_search,
                                  callback=callback)

result.data.show('Resulting diffeo')
(result.data - lie_grp.identity.data).show('translations')
(result.data_inv - lie_grp.identity.data).show('translations inverse')
