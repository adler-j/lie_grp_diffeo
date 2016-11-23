from lie_group_diffeo import GLn, SOn, MatrixVectorAction, ProductSpaceAction
import odl
import numpy as np

if __name__ == '__main__':
    # linear interpolation has boundary problems.
    space = odl.uniform_discr([-1, -1], [1, 1], [200, 200], interp='nearest')

    v0 = space.element(lambda x: np.exp(-sum(xi**2 for xi in x) / 0.4**2))
    v1 = space.element(lambda x: np.exp(-(x[0]**2 + 0.5 * x[1]**2) / 0.4**2))

    f1 = odl.solvers.L2NormSquared(space).translated(v1)

    W = space.tangent_bundle
    w1 = W.element([lambda x: x[0],
                    lambda x: x[1]])
    # Something is seriously wrong here, this should not need to be negative.
    f2 = -0.1 * odl.solvers.L2NormSquared(W).translated(w1)

    lie_grp = GLn(2)
    assalg = lie_grp.associated_algebra
    point_action = MatrixVectorAction(lie_grp, space)
    power_action = ProductSpaceAction(point_action, W.size)

    Ainv = lambda x: x

    v = v0.copy()
    w = w1.copy()
    g = lie_grp.identity

    callback = odl.solvers.CallbackShow(display_step=10)

    v0.show('starting point')
    v1.show('target point')

    eps = 0.01
    for i in range(1000):
        u = Ainv(point_action.inf_action_adj(v, f1.gradient(v)) +
                 power_action.inf_action_adj(w, f2.gradient(w)))

        if 0:
            v -= eps * point_action.inf_action(u)(v)
            w -= eps * power_action.inf_action(u)(w)
        else:
            g = g.compose(assalg.exp(-eps * u))
            v = point_action.action(g)(v0)
            w = power_action.action(g)(w1)

        callback(v)
        print(f1(v) - f2(w))
