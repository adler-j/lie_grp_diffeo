from SOn import SOn
from GLn import GLn
import odl
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == '__main__':
    r3 = odl.rn(3)
    W = odl.ProductSpace(r3, 3)
    v0 = -r3.element([1, 0, 0])

    v1 = r3.element([2, 3, 4])
    v1 *= 1.5 / v1.norm()
    f1 = odl.solvers.L2NormSquared(r3).translated(v1)

    w1 = W.element([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    f2 = 0.1 * odl.solvers.L2NormSquared(W).translated(w1)

    lie_grp = GLn(3)
    assalg = lie_grp.associated_algebra

    Ainv = lambda x: x

    v = v0.copy()
    w = w1.copy()

    g = lie_grp.identity

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

    eps = 0.01
    for i in range(10):
        u = Ainv(assalg.inf_action_adj(r3, v, f1.gradient(v)) +
                 assalg.inf_action_adj(W, w, f2.gradient(w)))

        if 0:
            v -= eps * u.inf_action(r3)(v)
            w -= eps * u.inf_action(W)(w)
        else:
            print('a', (eps * u), assalg.exp(-eps * u))
            g = g.compose(assalg.exp(-eps * u))
            v = g.action(r3)(v0)
            w = g.action(W)(w1)

        ax.scatter(v[0], v[1], v[2], c='b')

        print(v, w)
        print(f1(v) + f2(w))
