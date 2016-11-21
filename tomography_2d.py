from SOn import SOn
from GLn import GLn
import odl
import numpy as np
import scipy as sp

if __name__ == '__main__':
    # linear interpolation has boundary problems.
    space = odl.uniform_discr([-1, -1], [1, 1], [200, 200], interp='nearest')

    scaling = odl.deform.LinDeformFixedDisp(
        space.tangent_bundle.element([lambda x: x[0] * 0.3 + x[1] * 0.2,
                                      lambda x: x[0] * 0.1 + x[1] * 0.3]))

    v0 = odl.phantom.shepp_logan(space, modified=True)

    W = space.tangent_bundle
    w1 = W.element([lambda x: x[0],
                    lambda x: x[1]])

    # Make a parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
    # Detector: uniformly sampled, n = 558, min = -30, max = 30
    detector_partition = odl.uniform_partition(-30, 30, 558)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    # Ray transform (= forward projection). We use ASTRA CUDA backend.
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')


    v1 = scaling(v0)

    f1 = odl.solvers.L2NormSquared(ray_trafo.range).translated(ray_trafo(v1)) * ray_trafo


    # Something is seriously wrong here, this should not need to be negative.
    f2 = -0.1 * odl.solvers.L2NormSquared(W).translated(w1)

    lie_grp = GLn(2)
    assalg = lie_grp.associated_algebra

    Ainv = lambda x: x

    v = v0.copy()
    w = w1.copy()
    g = lie_grp.identity

    callback = odl.solvers.CallbackShow(display_step=10)

    v0.show('starting point')
    v1.show('target point')

    eps = 0.01
    for i in range(1000):
        u = Ainv(assalg.inf_action_adj(ray_trafo.domain, v, f1.gradient(v)) +
                 assalg.inf_action_adj(W, w, f2.gradient(w)))

        if 0:
            v -= eps * u.inf_action(space)(v)
            w -= eps * u.inf_action(W)(w)
        else:
            g = g.compose(assalg.exp(-eps * u))
            v = g.action(space)(v0)
            w = g.action(W)(w1)

        callback(v)
        print(f1(v) - f2(w))
