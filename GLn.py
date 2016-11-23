import odl
import numpy as np
import scipy as sp
from lie_group import LieGroup, LieGroupElement, LieAlgebra, LieAlgebraElement


__all__ = ('GLn',)


class GLn(LieGroup):
    def __init__(self, n):
        self.n = n

    def element(self, arg):
        return GLnElement(self, arg)

    @property
    def associated_algebra(self):
        return GLnAlgebra(self)

    @property
    def identity(self):
        return self.element(np.eye(self.n))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.n)


class GLnElement(LieGroupElement):
    def __init__(self, group, arr):
        self.group = group
        self.arr = np.asarray(arr, dtype=float)

    def action(self, domain):
        if isinstance(domain, odl.ProductSpace):
            subops = [self.action(domi) for domi in domain]
            return odl.DiagonalOperator(*subops)
        elif isinstance(domain, odl.DiscreteLp):
            pts = domain.points()
            deformed_pts = self.arr.dot(pts.T) - pts.T
            deformed_pts = domain.tangent_bundle.element(deformed_pts)
            return odl.deform.LinDeformFixedDisp(deformed_pts)
        elif (isinstance(domain, odl.space.base_ntuples.FnBase) and
              domain.size == self.group.n):
            return odl.MatVecOperator(self.arr, domain, domain)
        else:
            assert False

    def compose(self, other):
        return self.group.element(self.arr.dot(other.arr))

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.group, self.arr)


class GLnAlgebra(LieAlgebra):
    def __init__(self, gln):
        self.gln = gln

    def identity(self):
        return self.element(np.eye(self.gln.n))

    def zero(self):
        return self.element(np.zeros([self.gln.n, self.gln.n]))

    def element(self, arg):
        return GLnAlgebraElement(self, arg)

    def exp(self, el):
        return self.gln.element(sp.linalg.expm(el.arr))

    def inf_action_adj(self, domain, v, m):
        if isinstance(domain, odl.ProductSpace):
            return sum((self.inf_action_adj(domi, vi, mi)
                        for domi, vi, mi in zip(domain, v, m)),
                       self.zero())
        elif isinstance(domain, odl.DiscreteLp):
            pts = domain.tangent_bundle.element(domain.points().T)
            grad = odl.Gradient(domain, method='central')
            gradv = grad(v)
            result = np.zeros([self.gln.n, self.gln.n])
            for i in range(self.gln.n):
                for j in range(self.gln.n):
                    result[i, j] = m.inner(gradv[i] * pts[j])
            return self.element(result)
        elif (isinstance(domain, odl.space.base_ntuples.FnBase) and
              domain.size == self.gln.n):
            return self.element(np.outer(m, v))
        else:
            assert False


class GLnAlgebraElement(LieAlgebraElement):
    def __init__(self, algebra, arr):
        self.algebra = algebra
        self.arr = arr

    def inf_action(self, domain):
        if isinstance(domain, odl.ProductSpace):
            subops = [self.inf_action(domi) for domi in domain]
            return odl.DiagonalOperator(*subops)
        elif isinstance(domain, odl.DiscreteLp):
            grad = odl.Gradient(domain)
            pts = domain.points()
            deformed_pts = self.arr.dot(pts.T) - pts.T
            deformed_pts = grad.range.element(deformed_pts)
            return odl.PointwiseInner(grad.range, deformed_pts) * grad
        elif (isinstance(domain, odl.space.base_ntuples.FnBase) and
              domain.size == self.algebra.gln.n):
            return odl.MatVecOperator(self.arr, domain, domain)
        else:
            assert False

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.algebra, self.arr)
