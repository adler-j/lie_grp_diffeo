import odl
import numpy as np
import scipy as sp


__all__ = ('SOn',)


class SOn(object):
    def __init__(self, n):
        self.n = n

    def element(self, arg):
        return SOnElement(self, arg)

    @property
    def associated_algebra(self):
        return SOnAlgebra(self)

    @property
    def identity(self):
        return self.element(np.eye(self.n))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.n)


class SOnElement(object):
    def __init__(self, group, arr):
        self.group = group
        self.arr = arr

    def action(self, domain):
        if isinstance(domain, odl.ProductSpace):
            subops = [self.action(domi) for domi in domain]
            return odl.DiagonalOperator(*subops)
        elif (isinstance(domain, odl.space.base_ntuples.FnBase) and
              domain.size == self.group.n):
            mat_vec_op = odl.MatVecOperator(self.arr, domain, domain)
            return mat_vec_op
        elif isinstance(domain, odl.DiscreteLp):
            pts = domain.points()
            deformed_pts = self.arr.dot(pts.T) - pts.T
            deformed_pts = domain.tangent_bundle.element(deformed_pts)
            return odl.deform.LinDeformFixedDisp(deformed_pts)
        else:
            assert False

    def compose(self, other):
        return self.group.element(self.arr.dot(other.arr))

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.group, self.arr)


class SOnAlgebra(object):
    def __init__(self, son):
        self.son = son

    def zero(self):
        return self.element(np.zeros([self.son.n, self.son.n]))

    def element(self, arg):
        return SOnAlgebraElement(self, arg)

    def exp(self, el):
        return self.son.element(sp.linalg.expm(el.arr))

    def inf_action_adj(self, domain, v, m):
        if isinstance(domain, odl.ProductSpace):
            return sum((self.inf_action_adj(domi, vi, mi)
                        for domi, vi, mi in zip(domain, v, m)),
                       self.zero())
        elif (isinstance(domain, odl.space.base_ntuples.FnBase) and
              domain.size == self.son.n):
            return self.element((np.outer(m, v) - np.outer(v, m)) / 2)
        elif isinstance(domain, odl.DiscreteLp):
            pts = domain.tangent_bundle.element(domain.points().T)
            grad = odl.Gradient(domain)
            gradv = grad(v)
            result = np.zeros([self.son.n, self.son.n])
            for i in range(self.son.n):
                for j in range(self.son.n):
                    result[i, j] = m.inner(gradv[i] * pts[j])

            result = result - result.T
            return self.element(result)
        else:
            assert False

    def __repr__(self):
        return '{}.associated_algebra'.format(self.son)


class SOnAlgebraElement(object):
    def __init__(self, algebra, arr):
        self.algebra = algebra
        self.arr = arr
        assert np.all(arr == -arr.T)

    def inf_action(self, domain):
        if isinstance(domain, odl.ProductSpace):
            mat_vec_ops = [odl.MatVecOperator(self.arr, domi, domi)
                           for domi in domain]
            return odl.DiagonalOperator(*mat_vec_ops)
        elif (isinstance(domain, odl.space.base_ntuples.FnBase) and
              domain.size == self.algebra.son.n):
            mat_vec_op = odl.MatVecOperator(self.arr, domain, domain)
            return mat_vec_op
        elif isinstance(domain, odl.DiscreteLp):
            grad = odl.Gradient(domain)
            pts = domain.points()
            deformed_pts = self.arr.dot(pts.T) - pts.T
            deformed_pts = grad.range.element(deformed_pts)
            return odl.PointwiseInner(grad.range, deformed_pts) * grad
        else:
            assert False

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.algebra, self.arr)

    def __add__(self, other):
        return self.algebra.element(self.arr + other.arr)

    def __mul__(self, other):
        return self.algebra.element(self.arr * other)

    def __rmul__(self, other):
        return self.algebra.element(other * self.arr)
