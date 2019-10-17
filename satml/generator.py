
"""
Generates example Boolean formulas.
Author: Mert Dumenci
"""

import subprocess
from typing import Union, Tuple
from satml import solver, types, expression


def generate_pigeonhole(
    pigeons: int,
    holes: int,
    solver: solver.Solver) -> Tuple[types.Dimacs, expression.Expression]:
    gen = subprocess.run(
        ["cnfgen", "php", str(pigeons), str(holes)],
        stdout=subprocess.PIPE
    )

    assert gen.returncode == 0, gen.stderr.decode('utf-8')

    dimacs = gen.stdout.decode('utf-8').strip()
    return dimacs, expression.from_dimacs(dimacs)


def generate_clique_color(
    vertices: int,
    clique_size: int,
    colors: int,
    solver: solver.Solver) -> Tuple[types.Dimacs, expression.Expression]:
    gen = subprocess.run(
        ["cnfgen", "cliquecoloring", str(vertices), str(clique_size), str(colors)],
        stdout=subprocess.PIPE
    )

    assert gen.returncode == 0, gen.stderr.decode('utf-8')
    
    dimacs = gen.stdout.decode('utf-8').strip()
    return dimacs, expression.from_dimacs(dimacs)


def generate_random(
    clause_width: int,
    num_vars: int,
    num_clauses: int,
    solver=None,
    force_satisfiable=False) -> Tuple[types.Dimacs, expression.Expression, types.Satisfiability]:
    """
    Generate a random K-CNF formula.
    If `force_satisfiable = True`, then `solver` must be supplied.
    """
    assert not force_satisfiable or (force_satisfiable and solver)

    if not force_satisfiable:
        return _generate_random(clause_width, num_vars, num_clauses)

    while True:
        candidate_dimacs, candidate = _generate_random(clause_width, num_vars, num_clauses)
        s, num_dec = solver.solve(candidate_dimacs)

        if s:
            return candidate_dimacs, candidate, (s, num_dec)


def _generate_random(
    clause_width: int,
    num_vars: int,
    num_clauses: int) -> Tuple[types.Dimacs, expression.Expression]:
    gen = subprocess.run(
        ["cnfgen", "randkcnf", str(clause_width), str(num_vars), str(num_clauses)],
        stdout=subprocess.PIPE
    )

    assert gen.returncode == 0, gen.stderr.decode('utf-8')
    dimacs = gen.stdout.decode('utf-8').strip()

    return dimacs, expression.from_dimacs(dimacs)
