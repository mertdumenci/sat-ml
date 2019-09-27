
"""
Generates random K-CNF SAT problems.
Author: Mert Dumenci
"""

import subprocess
from typing import Union, Tuple
from satml import solver, types, expression


def generate_pigeonhole(
    pigeons: int,
    holes: int,
    solver: solver.Solver) -> Tuple[types.Dimacs, expression.Expression, types.Solution]:
    gen = subprocess.run(
        ["cnfgen", "php", str(pigeons), str(holes)],
        capture_output=True
    )

    assert gen.returncode == 0, gen.stderr.decode('utf-8')

    dimacs = gen.stdout.decode('utf-8').strip()
    s, history = solver.solve(dimacs)

    return dimacs, expression.from_dimacs(dimacs), (s, history)


def generate_clique_color(
    vertices: int,
    clique_size: int,
    colors: int,
    solver: solver.Solver) -> Tuple[types.Dimacs, expression.Expression, types.Solution]:
    gen = subprocess.run(
        ["cnfgen", "cliquecoloring", str(vertices), str(clique_size), str(colors)],
        capture_output=True
    )

    assert gen.returncode == 0, gen.stderr.decode('utf-8')

    dimacs = gen.stdout.decode('utf-8').strip()
    s, history = solver.solve(dimacs)

    return dimacs, expression.from_dimacs(dimacs), (s, history)



def generate_random(
    clause_width: int,
    num_vars: int,
    num_clauses: int,
    force_satisfiable=False,
    solver=solver.Solver) -> Union[Tuple[types.Dimacs, expression.Expression], Tuple[types.Dimacs, expression.Expression, types.Solution]]:
    """
    Generate a random K-CNF formula.
    If `force_satisfiable = True`, then `solver` must be supplied.
    """
    assert not force_satisfiable or (force_satisfiable and solver)

    if not force_satisfiable:
        return _generate_random(clause_width, num_vars, num_clauses)

    while True:
        candidate_dimacs, candidate = _generate_random(clause_width, num_vars, num_clauses)
        s, history = solver.solve(candidate_dimacs)

        if s:
            print("DIMACS:\n{}".format(candidate_dimacs))
            return candidate_dimacs, candidate, (s, history)


def _generate_random(
    clause_width: int,
    num_vars: int,
    num_clauses: int) -> Tuple[types.Dimacs, expression.Expression]:
    gen = subprocess.run(
        ["cnfgen", "randkcnf", str(clause_width), str(num_vars), str(num_clauses)], 
        capture_output=True
    )

    assert gen.returncode == 0, gen.stderr.decode('utf-8')
    dimacs = gen.stdout.decode('utf-8').strip()

    # `cnfgen` is not the best in naming variables and that confuses the solver. Let's rename them.
    f = expression.from_dimacs(dimacs)
    f = expression.rename(f)
    return expression.to_dimacs(f), f
