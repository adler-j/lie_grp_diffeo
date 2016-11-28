import lie_group_diffeo as lgd
import odl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

r3 = odl.rn(3)
W = odl.ProductSpace(r3, 3)
v0 = -r3.element([1, 0, 0])

v1 = r3.element([2, 3, 4])
v1 *= 1.5 / v1.norm()
f1 = odl.solvers.L2NormSquared(r3).translated(v1)

w1 = W.element([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
f2 = 0.2 * odl.solvers.L2NormSquared(W).translated(w1)

# SELECT GLn or SOn here
# lie_grp = GLn(3)
# lie_grp = lgd.SOn(3)
# point_action = lgd.MatrixVectorAction(lie_grp, r3)

lie_grp = lgd.AffineGroup(3)
point_action = lgd.MatrixVectorAffineAction(lie_grp, r3)

assalg = lie_grp.associated_algebra
power_action = lgd.ProductSpaceAction(point_action, 3)

# Combine action and functional into single object.
action = lgd.ProductSpaceAction(point_action, power_action)
x = action.domain.element([v0, w1])
f = odl.solvers.SeparableSum(f1, f2)

# Initial guess
g = lie_grp.identity

# Create some plotting info
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_aspect('equal')
ax.scatter(v0[0], v0[1], v0[2], c='r')
ax.scatter(v1[0], v1[1], v1[2], c='r')
ax.scatter(0, 0, 0, c='k')
ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]])


def callback(x):
    # Show trajectory of target point
    ax.scatter(x[0][0], x[0][1], x[0][2], c='b')

    # Also show trajectory of reference points
    ax.scatter(x[1][0][0], x[1][0][1], x[1][0][2], c='c')
    ax.scatter(x[1][1][0], x[1][1][1], x[1][1][2], c='c')
    ax.scatter(x[1][2][0], x[1][2][1], x[1][2][2], c='c')

lgd.gradient_flow_solver(x, f, g, action,
                         niter=100, line_search=0.1, callback=callback)
