

import random
from satml.expression import free, expr, Type, _Expression, pprint


def branch_on_variable(an_exp, var, val):
    """Fixes a given variable."""
    if not isinstance(an_exp, _Expression):
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
        typ, branch_on_variable(an_exp.l_val, var, val), branch_on_variable(an_exp.r_val, var, val)
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


def satisfiable(an_exp):
    """SAT solving (modulo model for unsat, TODO)"""
    free_v = list(free(an_exp))
    # Solved.
    if not free_v:
        if an_exp.typ == Type.CONST:
            return bool(an_exp.l_val)
        else:
            raise ValueError('Reduced to {}?'.format(pprint(an_exp)))

    split = random.choice(free_v)
    true = simplify(branch_on_variable(an_exp, split, True))
    false = simplify(branch_on_variable(an_exp, split, False))

    return satisfiable(true) or satisfiable(false)