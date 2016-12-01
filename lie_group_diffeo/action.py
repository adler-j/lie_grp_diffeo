"""Definitions of abstract actions with Lie groups."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import odl
from numbers import Integral


__all__ = ('LieAction', 'ProductSpaceAction', 'IdentityAction',
           'ComposedAction', 'InverseAction')


class LieAction(object):

    """Action of a lie group on some set."""

    def __init__(self, lie_group, domain):
        self.lie_group = lie_group
        self.domain = domain

    def action(self, lie_grp_element):
        """Return the action, an odl.Operator associated with lie_grp_element.
        """
        raise NotImplementedError('abstract method')

    def inf_action(self, lie_grp_element):
        """Return the infinitesimal action, an odl.Operator associated with
        ``lie_grp_element``.
        """
        raise NotImplementedError('abstract method')

    def momentum_map(self, v, m):
        """The momentum map corresponding to the infinitesimal action.
        Returns a `lie_group.associated_algebra` object associated with
        base point ``v`` and momentum ``m``.
        """
        raise NotImplementedError('abstract method')


class ProductSpaceAction(LieAction):

    """Action on a product space as defined by several "sub-actions"."""

    def __init__(self, *actions):
        # Allow ProductSpaceAction(action, 3) style syntax.
        if (len(actions) == 2 and
                isinstance(actions[0], LieAction) and
                isinstance(actions[1], Integral)):
            actions = [actions[0]] * actions[1]
        assert all(ac.lie_group == actions[0].lie_group for ac in actions)

        lie_group = actions[0].lie_group
        domain = odl.ProductSpace(*[ac.domain for ac in actions])
        self.actions = actions

        LieAction.__init__(self, lie_group, domain)

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        subops = [ac.action(lie_grp_element) for ac in self.actions]
        return odl.DiagonalOperator(*subops)

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        subops = [ac.inf_action(lie_alg_element) for ac in self.actions]
        return odl.DiagonalOperator(*subops)

    def momentum_map(self, v, m):
        assert v in self.domain
        assert m in self.domain
        return sum((ac.momentum_map(vi, mi)
                    for ac, vi, mi in zip(self.actions, v, m)),
                   self.lie_group.associated_algebra.zero())


def IdentityAction(LieAction):

    """The action that maps any element to itself."""

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group
        return odl.IdentityOperator(self.domain)

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra
        return odl.ZeroOperator(self.domain)

    def momentum_map(self, v, m):
        assert v in self.domain
        assert m in self.domain
        return self.lie_group.associated_algebra.zero()


class ComposedAction(LieAction):

    """The action of two composed actions.

    phi(g, x) = phi_2(g, phi_1(g, x))

    Here, we need to assume phi_2 is linear in the second argument.
    """

    def __init__(self, inner, outer):
        assert isinstance(inner, LieAction)
        assert isinstance(outer, LieAction)

        assert inner.lie_group == outer.lie_group
        assert inner.domain == outer.domain

        self.inner = inner
        self.outer = outer

        LieAction.__init__(self, inner.lie_group, inner.domain)

    def action(self, lie_grp_element):
        assert lie_grp_element in self.lie_group

        action_inner = self.inner.action(lie_grp_element)
        action_outer = self.outer.action(lie_grp_element)
        return action_outer * action_inner

    def inf_action(self, lie_alg_element):
        assert lie_alg_element in self.lie_group.associated_algebra

        inf_action_inner = self.inner.inf_action(lie_alg_element)
        inf_action_outer = self.outer.inf_action(lie_alg_element)
        return inf_action_inner + inf_action_outer

    def momentum_map(self, v, m):
        assert v in self.domain
        assert m in self.domain

        momentum_map_inner = self.inner.momentum_map(v, m)
        momentum_map_outer = self.outer.momentum_map(v, m)
        return momentum_map_inner + momentum_map_outer


class InverseAction(LieAction):

    """The apply action, but with the inverse element."""

    def __init__(self, base_action):
        assert isinstance(base_action, LieAction)
        LieAction.__init__(self, base_action.lie_group, base_action.domain)

    def action(self, lie_grp_element):
        return self.base_action.action(lie_grp_element.inverse)

    def inf_action(self, lie_alg_element):
        return -self.base_action.inf_action(lie_alg_element.inverse)

    def momentum_map(self, v, m):
        return -self.base_action.momentum_map(v, m)
