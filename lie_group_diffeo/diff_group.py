import odl
import numpy as np
from lie_group import LieGroup, LieGroupElement, LieAlgebra, LieAlgebraElement
from action import LieAction


__all__ = ('Diff', 'GeometricDeformationAction',
           'JacobianDeterminantScalingAction')


def _pspace_el_asarray(element):
    """Convert productspace element to flat array."""
    assert isinstance(element.space, odl.ProductSpace)
    assert element.space.is_power_space
    pts = np.empty([element.size,
                    element[0].size])
    for i, xi in enumerate(element):
        pts[i] = xi.asarray().ravel(order=xi.order)
    return pts


def _pspace_el_asmatrix(vec):
    """Convert ``x`` to an array."""
    domain = vec[0].space
    shape = domain[0].shape + (len(vec),) + domain.shape
    arr = np.empty(shape, dtype=domain.dtype)
    for i, xi in enumerate(vec):
        for j, xij in enumerate(xi):
            arr[..., i, j] = xij.asarray()

    return arr


class Diff(LieGroup):
    """Group of diffeomorphisms on some manifold M."""
    def __init__(self, domain, coord_space=None):
        assert isinstance(domain, odl.DiscreteLp)
        self.domain = domain
        if coord_space is None:
            coord_space = domain.tangent_bundle  # not really correct, but W/E
        assert isinstance(coord_space, odl.ProductSpace)
        assert coord_space.is_power_space
        self.coord_space = coord_space

    def element(self, inp=None, inp_inv=None):
        """Create element from ``inp``."""
        if inp is None and inp_inv is None:
            return self.identity
        elif inp is None or inp_inv is None:
            raise TypeError('Need to provide both inv and inp_inv')
        else:
            inp = self.coord_space.element(inp)
            inp_inv = self.coord_space.element(inp_inv)
            return self.element_type(self, inp, inp_inv)

    @property
    def identity(self):
        """The mapping x -> x."""
        pts = self.coord_space[0].points().T
        return self.element(pts, pts)

    @property
    def associated_algebra(self):
        return DiffAlgebra(self)

    @property
    def element_type(self):
        return DiffElement

    def __eq__(self, other):
        return ((isinstance(self, type(other)) or
                 isinstance(other, type(self))) and
                self.coord_space == other.coord_space)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.coord_space)


class DiffElement(LieGroupElement):
    def __init__(self, lie_group, data, data_inv):
        LieGroupElement.__init__(self, lie_group)
        self.data = data
        self.data_inv = data_inv

    @property
    def inverse(self):
        return self.lie_group.element(self.data_inv, self.data)

    def compose(self, other):
        pts = _pspace_el_asarray(other.data)
        def_pts = [dati.interpolation(pts) for dati in self.data]
        pts_inv = _pspace_el_asarray(self.data_inv)
        def_pts_inv = [dati.interpolation(pts_inv) for dati in other.data_inv]
        return self.lie_group.element(def_pts, def_pts_inv)

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.lie_group, self.data)


class DiffAlgebra(LieAlgebra):
    def __init__(self, lie_group):
        LieAlgebra.__init__(self, lie_group=lie_group)

        # Not technically correct but W/E for now
        self.data_space = lie_group.coord_space

        # TODO: This needs to be severely improved
        self.indicator = lie_group.coord_space[0].element(
            lambda x: np.exp(-sum((xi/0.92)**50 for xi in x)))

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination by data space."""
        out.data.lincomb(a, x1.data, b, x2.data)

    def _inner(self, x1, x2):
        """Inner product by data space."""
        return x1.data.inner(x2.data)

    def one(self):
        return self.element(self.data_space.one())

    def zero(self):
        return self.element(self.data_space.zero())

    def element(self, inp=None):
        if inp is None:
            return self.zero()
        else:
            inp = self.data_space.element(inp)
            return self.element_type(self, self.project(inp))

    def project(self, vectors):
        """Project vectors on the algebra."""
        for i in range(len(vectors)):
            vectors[i] *= self.indicator
        return vectors

    def exp(self, el):
        """Exponential map via addition."""
        pts = self.data_space[0].points().T
        increment = _pspace_el_asarray(el.data)
        return self.lie_group.element(pts + increment, pts - increment)

    def __eq__(self, other):
        return (isinstance(other, DiffAlgebra) and
                self.lie_group == other.lie_group)

    @property
    def element_type(self):
        return DiffAlgebraElement


class DiffAlgebraElement(LieAlgebraElement):
    def __init__(self, lie_algebra, data):
        LieAlgebraElement.__init__(self, lie_algebra)
        self.data = data

    def __repr__(self):
        return '{!r}.element({!r})'.format(self.space, self.data)


class GeometricDeformationAction(LieAction):

    """Action via geometric deformation of image."""

    def __init__(self, lie_group, domain, gradient=None):
        LieAction.__init__(self, lie_group, domain)
        assert lie_group.domain == domain
        if gradient is None:
            self.gradient = odl.Gradient(self.domain, method='central')
        else:
            self.gradient = gradient

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        pts = self.domain.points().T
        pts = self.lie_group.coord_space.element(pts)
        deformed_pts = lie_grp_element.data - pts
        return odl.deform.LinDeformFixedDisp(deformed_pts,
                                             templ_space=self.domain)

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        deformed_pts = lie_alg_element.data
        pointwise_inner = odl.PointwiseInner(self.gradient.range, deformed_pts)
        return pointwise_inner * self.gradient

    def momentum_map(self, f, m):
        assert f in self.domain
        assert m in self.domain
        gradf = self.gradient(f)
        return self.lie_group.associated_algebra.element(gradf * m)


class JacobianDeterminantScalingAction(LieAction):

    """Action via multiplication with inverse of jacobian determinant."""

    def __init__(self, lie_group, domain):
        LieAction.__init__(self, lie_group, domain)
        assert lie_group.domain == domain
        self.gradient = odl.Gradient(self.domain, method='forward',
                                     pad_mode='order1')
        self.partials = [odl.PartialDerivative(self.domain, axis=i,
                                               method='forward',
                                               pad_mode='order1')
                         for i in range(domain.ndim)]

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        jacobian = _pspace_el_asmatrix([self.gradient(xi)
                                        for xi in lie_grp_element.data])
        det_jac = 1 / np.linalg.det(jacobian)
        det_jac = self.domain.element(det_jac)
        return odl.MultiplyOperator(det_jac)

    def inf_action(self, lie_alg_element):
        scale = sum(
            parti(dati)
            for parti, dati in zip(self.partials, lie_alg_element.data))
        return odl.MultiplyOperator(-scale)

    def momentum_map(self, f, m):
        assert f in self.domain
        assert m in self.domain
        part_derivs = [-parti.adjoint(f * m) for parti in self.partials]
        return self.lie_group.associated_algebra.element(part_derivs)
