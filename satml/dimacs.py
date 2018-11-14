"""
Reading DIMACS strings.
"""

import functools
from satml.expression import expr, Type


def _clause(string):
    """Expects `string` to be in the DIMACS clause format."""
    vars = string.split(' ')[:-1]
    return functools.reduce(
        lambda c, v: expr((Type.OR, v, c)),
        vars[1:],
        expr((vars[0]))
    )


def from_dimacs(string):
    """Builds a CNF from a DIMACS file."""
    lines = string.split('\n')
    clauses = [_clause(line) for line in lines if line and line[0] != 'c' and line[0] != 'p']

    return functools.reduce(
        lambda ca, cb: expr((Type.AND, ca, cb)),
        clauses[1:],
        clauses[0]
    )

