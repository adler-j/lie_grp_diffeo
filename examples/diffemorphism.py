import lie_group_diffeo as lgd
import odl
import numpy as np


space = odl.uniform_discr(-1, 1, 30, interp='nearest')
tgtspc = space.tangent_bundle

pts = space.points().T
def_pts = tgtspc.element(pts**2)
def_pts2 = tgtspc.element(pts**3)

def_pts3 = tgtspc.element([def_pts[0].interpolation(def_pts2)])
def_pts3.show()
tgtspc.element(pts**6).show()