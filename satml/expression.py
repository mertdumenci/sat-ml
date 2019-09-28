
"""
The Boolean expression library.
Author: Mert Dumenci
"""

# TODO(mert): Implement certain codepaths as iterative vs. recursive...

import enum
import collections
import functools
from typing import Set, Any, Tuple

from satml import types

"""A CNF expression type."""
Type = enum.Enum('Type', 'VAR AND OR NOT CONST')
"""A CNF expression."""
Expression = collections.namedtuple('Expression', 'typ l_val r_val')


def trivially_sat(exp: Expression) -> bool:
    return exp.typ == Type.CONST and exp.l_val == True


def trivially_unsat(exp: Expression) -> bool:
    return exp.typ == Type.CONST and exp.l_val == False


def rename(exp: Expression, var_map=None) -> Expression:
    """Renames all variables in first order of occurrence in the expression AST."""
    if isinstance(exp, str):
        if exp in var_map:
            return str(var_map[exp])
        else:
            max_name = max(var_map.values()) if len(var_map) > 0 else 0
            var_map[exp] = max_name + 1
        
            return str(max_name + 1)
    elif isinstance(exp, Expression):
        typ, l_val, r_val = exp
        var_map = var_map if var_map is not None else {}
        return Expression(typ, rename(l_val, var_map), rename(r_val, var_map))
    else:
        return exp


def fixpoint(arg, func, comp):
    """Applies `func` to `arg` repeatedly until `comp` returns YES."""
    prev = arg
    cur = func(arg)

    while not comp(cur, prev):
        prev = cur
        cur = func(cur)

    return cur


def expr(tpl) -> Expression:
    """Convenience initializer for `Expression`"""
    if isinstance(tpl, Expression):
        return tpl
    # TODO(mert): I don't like that these are strings. Should be integers for performance
    if isinstance(tpl, str):
        if tpl[0] == '-':
            return Expression(Type.NOT, expr(tpl[1:]), None)
        else:
            return Expression(Type.VAR, tpl, None)
    if isinstance(tpl, bool):
        return Expression(Type.CONST, tpl, None)
    if tpl is None:
        return None

    assert len(tpl) == 3, "Expecting (type, l_val, r_val)"
    typ, l_val, r_val = tpl
    assert isinstance(typ, Type), "Expecting first arg to be a Type instance"

    if typ == Type.CONST or typ == Type.VAR:
        return Expression(typ, l_val, r_val)

    return Expression(typ, expr(l_val), expr(r_val))


def _distribute(an_exp: Expression) -> Expression:
    """Distributes disjunctions over conjunctions."""
    if not isinstance(an_exp, Expression):
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


def _push_negations(an_exp: Expression) -> Expression:
    """Pushes all negations inward to the literals."""
    if not isinstance(an_exp, Expression):
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


def cnf(an_exp: Expression) -> Expression:
    """Takes an `Expression` in CNF."""
    def compare(x, y):
        return x == y

    def _cnf(x):
        return _distribute(_push_negations(x))

    return fixpoint(an_exp, _cnf, compare)


def free(an_exp: Expression) -> Set[Any]:
    """Finds the free variables in an `Expression`."""
    if not isinstance(an_exp, Expression):
        return set()

    typ = an_exp.typ
    if typ == Type.VAR:
        # Python treats the string as an iterator and splits it to its char when put into the ctor.
        s = set()
        s.add(an_exp.l_val)

        return s

    return free(an_exp.l_val).union(free(an_exp.r_val))


_TYPE_PRINT = {
    Type.AND: '∧',
    Type.OR: '∨'
}


def pprint(an_exp: Expression) -> str:
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


def _clause(string: str) -> Expression:
    """Parses a 0-delimited Dimacs clause."""
    vars = string.split(' ')[:-1]
    return functools.reduce(
        lambda c, v: expr((Type.OR, v, c)),
        vars[1:],
        expr((vars[0]))
    )


def from_dimacs(string: types.Dimacs) -> Expression:
    """Parses a Dimacs file."""
    lines = string.split('\n')
    clauses = [_clause(line) for line in lines if line and line[0] != 'c' and line[0] != 'p']

    return functools.reduce(
        lambda ca, cb: expr((Type.AND, ca, cb)),
        clauses[1:],
        clauses[0]
    )


def to_dimacs(exp: Expression) -> types.Dimacs:
    """Makes a Dimacs CNF file."""
    exp = simplify(cnf(exp))

    lines, num_ands = _to_dimacs(exp)
    num_conjuncts = num_ands + 1

    # Assemble lines (by adding line delimiters)
    dimacs = '\n'.join([line + ' 0' for line in lines])
    
    # Write out the header.
    num_variables = len(free(exp))
    header = "p cnf {} {}\n".format(num_variables, num_conjuncts)

    return header + dimacs


def _to_dimacs(exp: Expression) -> Tuple[types.Dimacs, int]:
    """Internal interface for dimacs conversion."""
    # Unpack the expression.
    assert isinstance(exp, Expression)
    typ, l_val, r_val = exp
    assert typ != Type.CONST

    if typ == Type.VAR:
        return str(l_val), 0
    if typ == Type.NOT:
        assert l_val.typ == Type.VAR
        return '-' + str(l_val.l_val), 0
    if typ == Type.AND:
        l1, n1 = _to_dimacs(l_val)
        l2, n2 = _to_dimacs(r_val)

        if not isinstance(l1, list):
            l1 = [l1]
        if not isinstance(l2, list):
            l2 = [l2]

        return l1 + l2, n1 + n2 + 1
    if typ == Type.OR:
        return ' '.join([_to_dimacs(l_val)[0], _to_dimacs(r_val)[0]]), 0

    assert False, "Should never reach here"    
    return None


def assign(an_exp, var, val):
    """Fixes a given variable."""
    if not isinstance(an_exp, Expression):
        return an_exp

    typ = an_exp.typ
    if typ == Type.VAR:
        cur_var = an_exp.l_val

        if cur_var == var:
            return expr((Type.CONST, val, None))
        else:
            return an_exp
    elif typ == Type.CONST:
        return an_exp

    return expr((
        typ, assign(an_exp.l_val, var, val), assign(an_exp.r_val, var, val)
    ))


_FALSE = expr(False)
_TRUE = expr(True)


def simplify(an_exp):
    """Gets rid of constants. (Or reduces to a constant.)"""
    typ = an_exp.typ

    if typ == Type.OR:
        s_l_val = simplify(an_exp.l_val)
        s_r_val = simplify(an_exp.r_val)

        if s_l_val == _TRUE or s_r_val == _TRUE:
            return _TRUE
        elif s_l_val != _FALSE and s_r_val != _FALSE:
            return expr((typ, s_l_val, s_r_val))
        elif s_l_val != _FALSE and s_r_val == _FALSE:
            return s_l_val
        elif s_l_val == _FALSE and s_r_val != _FALSE:
            return s_r_val
        else:
            return _FALSE
    elif typ == Type.AND:
        s_l_val = simplify(an_exp.l_val)
        s_r_val = simplify(an_exp.r_val)

        if s_l_val == _FALSE or s_r_val == _FALSE:
            return _FALSE
        elif s_l_val != _TRUE and s_r_val != _TRUE:
            return expr((typ, s_l_val, s_r_val))
        elif s_l_val != _TRUE and s_r_val == _TRUE:
            return s_l_val
        elif s_l_val == _TRUE and s_r_val != _TRUE:
            return s_r_val
        else:
            return _TRUE
    elif typ == Type.CONST or typ == Type.VAR:
        return an_exp
    elif typ == Type.NOT:
        if an_exp.l_val.typ == Type.CONST:
            return expr((Type.CONST, not an_exp.l_val.l_val, None))

        return expr((Type.NOT, simplify(an_exp.l_val), None))
