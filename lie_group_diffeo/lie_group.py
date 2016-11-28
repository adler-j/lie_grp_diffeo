import odl


__all__ = ('LieGroup', 'LieAlgebra')


class LieGroup(odl.Set):
    @property
    def associated_algebra(self):
        """Return the associated Lie algebra."""
        raise NotImplementedError('abstract method')

    @property
    def identity(self):
        """Return the identity element."""
        raise NotImplementedError('abstract method')

    @property
    def element_type(self):
        """Type of elements in this group."""
        raise NotImplementedError('abstract method')

    def __contains__(self, other):
        """Return ``other in self``."""
        return isinstance(other, self.element_type) and other.lie_group == self


class LieGroupElement(object):
    def __init__(self, lie_group):
        self.lie_group = lie_group

    def compose(self, other):
        """Compose this element with other."""
        raise NotImplementedError('abstract method')


class LieAlgebra(odl.LinearSpace):
    def __init__(self, lie_group):
        odl.LinearSpace.__init__(self, odl.RealNumbers())
        self.lie_group = lie_group

    def exp(self, el):
        """Compose this element with other."""
        raise NotImplementedError('abstract method')

    @property
    def element_type(self):
        """Type of elements in this algebra."""
        raise NotImplementedError('abstract method')

    def __repr__(self):
        return '{!r}.associated_algebra'.format(self.lie_group)


class LieAlgebraElement(odl.LinearSpaceElement):
    pass
