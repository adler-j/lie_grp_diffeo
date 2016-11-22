import odl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


__all__ = ('GLn',)


class GLn(object):
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


class GLnElement(object):
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
            mat_vec_op = odl.MatVecOperator(self.arr, domain, domain)
            return mat_vec_op
        else:
            assert False

    def compose(self, other):
        return self.group.element(self.arr.dot(other.arr))

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.group, self.arr)


class GLnAlgebra(object):
    def __init__(self, gln):
        self.gln = gln

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
            grad = odl.Gradient(domain)
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

    def __repr__(self):
        return '{}.associated_algebra'.format(self.gln)


class GLnAlgebraElement(object):
    def __init__(self, algebra, arr):
        self.algebra = algebra
        self.arr = arr

    def inf_action(self, domain):
        if isinstance(domain, odl.ProductSpace):
            subops = [self.inf_action(domi) for domi in domain]
            return odl.DiagonalOperator(*subops)
        elif isinstance(domain, odl.DiscreteLp):
            pts = domain.points()
            deformed_pts = self.arr.dot(pts.T) - pts.T
            deformed_pts = domain.tangent_bundle.element(deformed_pts)
            return odl.deform.LinDeformFixedDisp(deformed_pts)
        elif (isinstance(domain, odl.space.base_ntuples.FnBase) and
              domain.size == self.algebra.gln.n):
            mat_vec_op = odl.MatVecOperator(self.arr, domain, domain)
            return mat_vec_op
        else:
            assert False

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.algebra, self.arr)

    def __add__(self, other):
        return self.algebra.element(self.arr + other.arr)

    def __mul__(self, other):
        return self.algebra.element(other * self.arr)

    def __rmul__(self, other):
        return self.algebra.element(other * self.arr)

if __name__ == '__main__':
    # Create algebra and an element in the algebra
    gl2 = GLn(2)
    gl2alg = gl2.associated_algebra
    alg_el = gl2alg.element(np.array([[0.1, 0.2], [0.1, 0.2]]))
    el = gl2alg.exp(alg_el)

    # Apply action on r2
    r2 = odl.rn(2)
    el_action_on_r2 = el.action(r2)
    print('real', el_action_on_r2([1, 2]))

    # Apply action on c2
    c2 = odl.cn(2)
    el_action_on_c2 = el.action(c2)
    print('complex', el_action_on_c2([1, 2]))

    # Compose with other element
    alg_el_2 = gl2alg.element(np.array([[1.0, 0.0], [0.1, 2.0]]))
    el_2 = gl2alg.exp(alg_el_2)
    composed_el = el.compose(el_2)

    # Apply action on r2
    r2 = odl.rn(2)
    el_action_on_r2 = composed_el.action(r2)
    print('real', el_action_on_r2([1, 2]))

    # Apply action on c2
    c2 = odl.cn(2)
    el_action_on_c2 = composed_el.action(c2)
    print('complex', el_action_on_c2.adjoint([1, 2]))
