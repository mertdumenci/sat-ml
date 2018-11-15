

import enum
import collections
from satml.helpers import fixpoint


Type = enum.Enum('Type', 'VAR AND OR NOT CONST')

# My fake approximation of a sum type in Python.
# This code is in Python because I want to use its ML libraries.
_Expression = collections.namedtuple('_Expression', 'typ l_val r_val')


_TYPE_PRINT = {
    Type.AND: '∧',
    Type.OR: '∨'
}


def pprint(an_exp: _Expression):
    """Pretty prints an `Expression`"""
    def pprint_s(exp):
        typ, l_val, r_val = exp.typ, exp.l_val, exp.r_val

        if typ == Type.CONST:
            return 'T' if l_val else 'F'
        if typ == Type.VAR:
            return l_val
        if typ == Type.NOT:
            return '-{}'.format(pprint_s(l_val))

        return '({} {} {})'.format(
            pprint_s(l_val), _TYPE_PRINT[typ], pprint_s(r_val)
        )

    return pprint_s(an_exp)


def expr(tpl) -> _Expression:
    """Convenience initializer for `Expression`"""
    if isinstance(tpl, _Expression):
        return tpl
    if isinstance(tpl, str):
        if tpl[0] == '-':
            return _Expression(Type.NOT, expr(tpl[1:]), None)
        else:
            return _Expression(Type.VAR, tpl, None)
    if isinstance(tpl, bool):
        return _Expression(Type.CONST, tpl, None)
    if tpl is None:
        return None

    assert len(tpl) == 3, "Expecting (type, l_val, r_val)"
    typ, l_val, r_val = tpl
    assert isinstance(typ, Type), "Expecting first arg to be a Type instance"

    if typ == Type.CONST or typ == Type.VAR:
        return _Expression(typ, l_val, r_val)

    return _Expression(typ, expr(l_val), expr(r_val))


def _distribute(an_exp: _Expression) -> _Expression:
    """Distributes disjunctions over conjunctions."""
    if not isinstance(an_exp, _Expression):
        return an_exp
    if an_exp.typ == Type.AND or an_exp.typ == Type.NOT:
        return expr((an_exp.typ, _distribute(an_exp.l_val), _distribute(an_exp.r_val)))
    if an_exp.typ != Type.OR:
        return an_exp

    # At this point, we have an invariant s.t.
    #   :type l_val: Expression
    #   :type r_val: Expression
    l_expr = an_exp.l_val
    r_expr = an_exp.r_val

    # (X AND Y) OR Z -> (X OR Z) AND (Y OR Z)
    if l_expr.typ == Type.AND:
        return expr((
                Type.AND,
                (Type.OR, _distribute(l_expr.l_val), _distribute(r_expr)),
                (Type.OR, _distribute(l_expr.r_val), _distribute(r_expr))
            ))
    # X OR (Y AND Z) -> (X AND Y) OR (X AND Z)
    if r_expr.typ == Type.AND:
        return expr((
                Type.AND,
                (Type.OR, _distribute(l_expr), _distribute(r_expr.l_val)),
                (Type.OR, _distribute(l_expr), _distribute(r_expr.r_val))
            ))

    return expr((Type.OR, _distribute(l_expr), _distribute(r_expr)))


def _push_negations(an_exp: _Expression) -> _Expression:
    """Pushes negations inward to the literals."""
    if not isinstance(an_exp, _Expression):
        return an_exp

    # Clean up double negatives.
    if an_exp.typ == Type.NOT and an_exp.l_val.typ == Type.NOT:
        return an_exp.l_val.l_val
    # Push negation in the AND.
    if an_exp.typ == Type.NOT and an_exp.l_val.typ == Type.AND:
        return expr((
            Type.OR,
            _push_negations(expr((Type.NOT, an_exp.l_val.l_val, None))),
            _push_negations(expr((Type.NOT, an_exp.l_val.r_val, None)))
        ))
    # Push negation in the OR.
    if an_exp.typ == Type.NOT and an_exp.l_val.typ == Type.OR:
        return expr((
            Type.AND,
            _push_negations(expr((Type.NOT, an_exp.l_val.l_val, None))),
            _push_negations(expr((Type.NOT, an_exp.l_val.r_val, None)))
        ))
    # Flip constants.
    if an_exp.typ == Type.NOT and an_exp.l_val.typ == Type.CONST:
        return expr((
            Type.CONST,
            not an_exp.l_val.l_val,
            None
        ))

    # Otherwise, push the operation in.
    return expr((an_exp.typ, _push_negations(an_exp.l_val), _push_negations(an_exp.r_val)))


def cnf(an_exp: _Expression) -> _Expression:
    """Conjunctive normal form"""
    def compare(x, y):
        return x == y

    def _cnf(x):
        return _distribute(_push_negations(x))

    return fixpoint(an_exp, _cnf, compare)


def free(an_exp: _Expression):
    """Finds free variables."""
    if not isinstance(an_exp, _Expression):
        return set()

    typ = an_exp.typ
    if typ == Type.VAR:
        # Python treats the string as an iterator and splits it to its char when put into the ctor.
        s = set()
        s.add(an_exp.l_val)

        return s

    return free(an_exp.l_val).union(free(an_exp.r_val))
