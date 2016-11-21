from SOn import SOn
from GLn import GLn
import odl
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == '__main__':
    space = odl.uniform_discr(-1, 1, 1000)

    el = space.element(lambda x: np.exp(-x**2 / 0.2**2))

    lie_grp = GLn(1)
    assalg = lie_grp.associated_algebra
    lie_el = assalg.exp(assalg.element(np.array([[0.1]])))

    el.show('el')
    lie_el.action(space)(el).show('deformed')
