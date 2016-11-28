"""Definitions of concrete Matrix Lie groups."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import odl
import numpy as np
import scipy as sp
from lie_group_diffeo.lie_group import (LieGroup, LieGroupElement, LieAlgebra,
                                        LieAlgebraElement)
from lie_group_diffeo.action import LieAction


__all__ = ('GLn', 'SLn', 'SOn',
           'MatrixVectorAction', 'MatrixImageAction',
           'MatrixDeterminantAction',
           'AffineGroup', 'EuclideanGroup',
           'MatrixVectorAffineAction', 'MatrixImageAffineAction')


class MatrixGroup(LieGroup):

    """Group of square matrices with matrix mult as group operation."""

    def __init__(self, size):
        self.size = size

    def element(self, array=None):
        """Create element from ``array``."""
        if array is None:
            return self.identity
        else:
            return self.element_type(self, array)

    @property
    def identity(self):
        return self.element(np.eye(self.size))

    def __eq__(self, other):
        return ((isinstance(self, type(other)) or
                 isinstance(other, type(self))) and
                self.size == other.size)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.size)


class MatrixGroupElement(LieGroupElement):

    """A Matrix."""

    def __init__(self, lie_group, arr):
        LieGroupElement.__init__(self, lie_group)
        self.arr = np.asarray(arr, dtype=float)

    def compose(self, other):
        """Compose two elements via matrix multiplication."""
        return self.lie_group.element(self.arr.dot(other.arr))

    @property
    def inverse(self):
        """Inverse is given by matrix inversion."""
        return self.lie_group.element(np.linalg.inv(self.arr))

    def __repr__(self):
        return '{!r}.element(\n{!r})'.format(self.lie_group, self.arr)


class MatrixAlgebra(LieAlgebra):

    """Vector space of matrices.

    The linear combination is defined pointwise.

    The inner product is the Frobenious inner product.
    """

    def __init__(self, lie_group):
        LieAlgebra.__init__(self, lie_group=lie_group)

    def project(self, array):
        """Project array on the algebra."""
        raise NotImplementedError('abstract method')

    @property
    def size(self):
        return self.lie_group.size

    def _lincomb(self, a, x1, b, x2, out):
        out.arr[:] = a * x1.arr + b * x2.arr

    def _inner(self, x1, x2):
        """Frobenious inner product."""
        return np.trace(x1.arr.T.dot(x2.arr))

    def one(self):
        return self.element(np.eye(self.size))

    def zero(self):
        return self.element(np.zeros([self.size, self.size]))

    def element(self, array=None):
        if array is None:
            return self.zero()
        else:
            return self.element_type(self, self.project(array))

    def exp(self, el):
        """Exponential map via matrix exponential."""
        return self.lie_group.element(sp.linalg.expm(el.arr))


class MatrixAlgebraElement(LieAlgebraElement):

    """A Matrix."""

    def __init__(self, lie_group, arr):
        LieAlgebraElement.__init__(self, lie_group)
        self.arr = np.asarray(arr, dtype=float)

    def __repr__(self):
        return '{!r}.element(\n{!r})'.format(self.space, self.arr)


class GLn(MatrixGroup):

    """The set of all invertible matrices."""

    @property
    def associated_algebra(self):
        return GLnAlgebra(self)

    @property
    def element_type(self):
        return GLnElement


class GLnElement(MatrixGroupElement):
    pass


class GLnAlgebra(MatrixAlgebra):
    @property
    def element_type(self):
        return GLnAlgebraElement

    def project(self, array):
        """Project array on the algebra."""
        return array


class GLnAlgebraElement(MatrixAlgebraElement):
    pass


class SLn(MatrixGroup):

    """The set of all matrices with determinant 1."""

    @property
    def associated_algebra(self):
        return SLnAlgebra(self)

    @property
    def element_type(self):
        return SLnElement


class SLnElement(MatrixGroupElement):
    pass


class SLnAlgebra(MatrixAlgebra):
    @property
    def element_type(self):
        return SLnAlgebraElement

    def project(self, array):
        return array - np.eye(self.size) * np.trace(array) / self.size


class SLnAlgebraElement(MatrixAlgebraElement):
    pass


class SOn(MatrixGroup):

    """The set of all orthogonal matrices.

    A matrix ``Q`` is orthogonal if::

        Q^T Q = Q Q^T = I
    """

    @property
    def associated_algebra(self):
        return SOnAlgebra(self)

    @property
    def element_type(self):
        return SOnElement


class SOnElement(MatrixGroupElement):
    pass


class SOnAlgebra(MatrixAlgebra):
    @property
    def element_type(self):
        return SOnAlgebraElement

    def project(self, array):
        return array - array.T


class SOnAlgebraElement(MatrixAlgebraElement):
    pass


class AffineGroup(MatrixGroup):

    """Affine group represented via matrices.

    The matrices have the form::

        A v
        0 1

    where ``A`` is an invertible ``n x n`` matrix, ``v`` is a ``n x 1`` vector,
    0 is a ``1 x n`` vector with all zeros and ``1`` is simply a scalar.
    """

    def __init__(self, n):
        MatrixGroup.__init__(self, n + 1)

    @property
    def associated_algebra(self):
        return AffineGroupAlgebra(self)

    @property
    def element_type(self):
        return AffineGroupElement

    def __repr__(self):
        """Return ``repr(self).``"""
        return '{}({})'.format(self.__class__.__name__, self.size - 1)


class AffineGroupElement(MatrixGroupElement):
    pass


class AffineGroupAlgebra(MatrixAlgebra):
    @property
    def element_type(self):
        return AffineGroupAlgebraElement

    def project(self, array):
        cpy = array.copy()
        cpy[-1] = 0
        return cpy


class AffineGroupAlgebraElement(MatrixAlgebraElement):
    pass


class EuclideanGroup(MatrixGroup):
    """Euclidean group represented via matrices.

    The euclidean group is the set of rigid transformations.

    The matrices have the form::

        Q v
        0 1

    where ``Q`` is an orthogonal ``n x n`` matrix, ``v`` is a ``n x 1`` vector,
    0 is a ``1 x n`` vector with all zeros and ``1`` is simply a scalar.
    """
    def __init__(self, n):
        MatrixGroup.__init__(self, n + 1)

    @property
    def associated_algebra(self):
        return EuclideanGroupAlgebra(self)

    @property
    def element_type(self):
        return EuclideanGroupElement

    def __repr__(self):
        """Return ``repr(self).``"""
        return '{}({})'.format(self.__class__.__group__, self.size - 1)


class EuclideanGroupElement(MatrixGroupElement):
    pass


class EuclideanGroupAlgebra(MatrixAlgebra):
    @property
    def element_type(self):
        return EuclideanGroupAlgebraElement

    def project(self, array):
        cpy = array.copy()
        cpy[:-1, :-1] = cpy[:-1, :-1] - cpy[:-1, :-1].T
        cpy[-1] = 0
        return cpy


class EuclideanGroupAlgebraElement(MatrixAlgebraElement):
    pass


class MatrixVectorAction(LieAction):

    """Action by matrix vector product."""

    def __init__(self, lie_group, domain):
        LieAction.__init__(self, lie_group, domain)
        assert lie_group.size == domain.size

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        return odl.MatVecOperator(lie_grp_element.arr,
                                  self.domain, self.domain)

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        return odl.MatVecOperator(lie_alg_element.arr,
                                  self.domain, self.domain)

    def inf_action_adj(self, v, m):
        assert v in self.domain
        assert m in self.domain
        return self.lie_group.associated_algebra.element(np.outer(m, v))


class MatrixImageAction(LieAction):

    """Action on image via coordinate transformation."""

    def __init__(self, lie_group, domain, gradient=None):
        LieAction.__init__(self, lie_group, domain)
        assert lie_group.size == domain.ndim
        if gradient is None:
            # should use method=central, others introduce a bias.
            self.gradient = odl.Gradient(self.domain, method='central')
        else:
            self.gradient = gradient

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        pts = self.domain.points()
        deformed_pts = lie_grp_element.arr.dot(pts.T) - pts.T
        deformed_pts = self.domain.tangent_bundle.element(deformed_pts)
        return odl.deform.LinDeformFixedDisp(deformed_pts)

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        pts = self.domain.points()
        deformed_pts = lie_alg_element.arr.dot(pts.T) - pts.T
        deformed_pts = self.domain.tangent_bundle.element(deformed_pts)
        pointwise_inner = odl.PointwiseInner(self.gradient.range, deformed_pts)
        return pointwise_inner * self.gradient

    def inf_action_adj(self, v, m):
        assert v in self.domain
        assert m in self.domain
        size = self.lie_group.size
        pts = self.domain.tangent_bundle.element(self.domain.points().T)
        gradv = self.gradient(v)
        result = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                result[i, j] = m.inner(gradv[i] * pts[j])

        return self.lie_group.associated_algebra.element(result)


class MatrixDeterminantAction(LieAction):

    """Action by matrix vector product."""

    def __init__(self, lie_group, domain):
        LieAction.__init__(self, lie_group, domain)
        assert domain.size == 1

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        return odl.ScalingOperator(self.domain,
                                   np.linalg.det(lie_grp_element.arr))

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        return odl.ScalingOperator(self.domain,
                                   np.trace(lie_alg_element.arr))

    def inf_action_adj(self, v, m):
        assert v in self.domain
        assert m in self.domain
        eye = float(m * v) * np.eye(self.lie_group.size)
        return self.lie_group.associated_algebra.element(eye)


class MatrixVectorAffineAction(LieAction):

    """Action by affine matrix vector product."""

    def __init__(self, lie_group, domain):
        LieAction.__init__(self, lie_group, domain)
        assert lie_group.size == domain.size + 1

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        matrix_op = odl.MatVecOperator(lie_grp_element.arr[:-1, :-1],
                                       self.domain, self.domain)
        translation_vec = self.domain.element(lie_grp_element.arr[:-1, -1])
        return matrix_op + translation_vec

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        matrix_op = odl.MatVecOperator(lie_alg_element.arr[:-1, :-1],
                                       self.domain, self.domain)
        translation_vec = self.domain.element(lie_alg_element.arr[:-1, -1])
        return matrix_op + translation_vec

    def inf_action_adj(self, v, m):
        assert v in self.domain
        assert m in self.domain
        v_appended = np.ones(v.size + 1)
        v_appended[:-1] = v
        m_appended = np.ones(m.size + 1)
        m_appended[:-1] = m
        outer = np.outer(m_appended, v_appended)
        return self.lie_group.associated_algebra.element(outer)


class MatrixImageAffineAction(LieAction):

    """Action on image via affine coordinate transformation."""

    def __init__(self, lie_group, domain, gradient=None):
        LieAction.__init__(self, lie_group, domain)
        assert lie_group.size == domain.ndim + 1
        if gradient is None:
            # should use method=central, others introduce a bias.
            self.gradient = odl.Gradient(self.domain, method='central')
        else:
            self.gradient = gradient

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        pts = self.domain.points()
        deformed_pts = lie_grp_element.arr[:-1, :-1].dot(pts.T) - pts.T
        deformed_pts += lie_grp_element.arr[:-1, -1][:, None]
        deformed_pts = self.domain.tangent_bundle.element(deformed_pts)
        return odl.deform.LinDeformFixedDisp(deformed_pts)

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        pts = self.domain.points()
        deformed_pts = lie_alg_element.arr[:-1, :-1].dot(pts.T) - pts.T
        deformed_pts += lie_alg_element.arr[:-1, -1][:, None]
        deformed_pts = self.domain.tangent_bundle.element(deformed_pts)
        pointwise_inner = odl.PointwiseInner(self.gradient.range, deformed_pts)
        return pointwise_inner * self.gradient

    def inf_action_adj(self, v, m):
        assert v in self.domain
        assert m in self.domain
        size = self.lie_group.size
        pts = self.domain.tangent_bundle.element(self.domain.points().T)
        gradv = self.gradient(v)
        result = np.zeros([size, size])
        for i in range(size - 1):
            for j in range(size - 1):
                result[i, j] = m.inner(gradv[i] * pts[j])

        for i in range(size - 1):
            result[i, -1] = m.inner(gradv[i])

        return self.lie_group.associated_algebra.element(result)
