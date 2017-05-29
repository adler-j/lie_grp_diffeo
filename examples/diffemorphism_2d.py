import lie_group_diffeo as lgd
import odl
import numpy as np

space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='nearest')
coord_space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='linear').tangent_bundle

# Define template
template = space.element(lambda x: np.exp(-(3 * x[0]**2 + x[1]**2) / 0.4**2))
target = space.element(lambda x: np.exp(-(1 * (x[0] + 0.1)**2 + x[1]**2) / 0.4**2))

# Define data matching functional
data_matching = odl.solvers.L1Norm(space).translated(target)

lie_grp = lgd.Diff(space, coord_space=coord_space)
deform_action = lgd.GeometricDeformationAction(lie_grp, space)

# Initial guess
g = lie_grp.identity

# Combine action and functional into single object.
action = deform_action
x = template.copy()
f = data_matching

# Show some results, reuse the plot
template.show('template')
target.show('target')

# Create callback that displays the current iterate and prints the function
# value
callback = odl.solvers.CallbackShow('diffemorphic matching', step=50)
callback &= odl.solvers.CallbackPrint(f)

# Solve via gradient flow
result = lgd.gradient_flow_solver(x, f, g, action,
                                  niter=2000, line_search=0.00005,
                                  callback=callback)

result.data.show('Resulting diffeo')
(result.data - lie_grp.identity.data).show('translations')
