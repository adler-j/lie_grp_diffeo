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
            mat_vec_ops = [odl.MatVecOperator(self.arr, domi, domi)
                           for domi in domain]
            return odl.DiagonalOperator(*mat_vec_ops)
        else:
            mat_vec_op = odl.MatVecOperator(self.arr, domain, domain)
            return mat_vec_op

    def compose(self, other):
        return self.group.element(self.arr.dot(other.arr))

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.group, self.arr)


class SOnAlgebra(object):
    def __init__(self, son):
        self.son = son

    def element(self, arg):
        return SOnAlgebraElement(self, arg)

    def exp(self, el):
        return self.son.element(sp.linalg.expm(el.arr))

    def inf_action_adj(self, domain, v, m):
        if isinstance(domain, odl.ProductSpace):
            return self.element(sum((np.outer(mi, vi) - np.outer(vi, mi)) / 2
                                    for mi, vi in zip(m, v)))
        else:
            return self.element((np.outer(m, v) - np.outer(v, m)) / 2)

    def __repr__(self):
        return '{}.associated_algebra'.format(self.gln)


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
        else:
            mat_vec_op = odl.MatVecOperator(self.arr, domain, domain)
            return mat_vec_op

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.algebra, self.arr)

    def __add__(self, other):
        return self.algebra.element(self.arr + other.arr)

    def __mul__(self, other):
        return self.algebra.element(self.arr * other)

    def __rmul__(self, other):
        return self.algebra.element(other * self.arr)