
"""
Generates random K-CNF SAT problems.
Author: Mert Dumenci
"""

import subprocess
from typing import Union, Tuple
from satml import solver, types


def generate_random(
    clause_width: int,
    num_vars: int,
    num_clauses: int,
    force_satisfiable=False,
    solver=solver.Solver) -> Union[types.Dimacs, Tuple[types.Dimacs, types.Solution]]:
    """
    Generate a random K-CNF formula.
    If `force_satisfiable = True`, then `solver` must be supplied.
    """
    assert not force_satisfiable or (force_satisfiable and solver)

    if not force_satisfiable:
        return _generate_random(clause_width, num_vars, num_clauses)

    while True:
        candidate = _generate_random(clause_width, num_vars, num_clauses)
        s, history = solver.solve(candidate)

        if s:
            return candidate, (s, history)


def _generate_random(
    clause_width: int,
    num_vars: int,
    num_clauses: int) -> types.Dimacs:
    gen = subprocess.run(
        ["cnfgen", "randkcnf", str(clause_width), str(num_vars), str(num_clauses)], 
        capture_output=True
    )

    assert gen.returncode == 0, gen.stderr.decode('utf-8')
    return gen.stdout.decode('utf-8').strip()
