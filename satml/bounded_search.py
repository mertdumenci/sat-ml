"""
Explores the SAT search space, tries to improve on the solver heuristic.
Author: Mert Dumenci
"""

from typing import List
from satml import expression, types, solver


def h_star(f: expression.Expression, solver: solver.Solver, depth: int, verbose=False) -> List[types.Decision]:
    """
    Given a formula, run an exhaustive search for a given number of levels
    improving the solver heuristic if possible.

    The number of decisions made by the solver heuristic `h` is an _upper bound_
    on the number of decisions made by the heuristic this method defines.

    We can only improve on it.

    :pre: `f` must be satisfiable.
    """
    current_formula = f
    current_depth = 0

    sat, initial_solver_guess = solver.solve(expression.to_dimacs(f))
    assert sat, "`f` must be satisfiable."

    decisions = []
    num_frontier_decisions = -1
    
    if verbose:
        print("🎬 Starting with initial solver guess of {} decisions".format(initial_solver_guess))

    while current_formula and current_depth < depth:
        if verbose:
            print("✅ Level {} with decision trail {}".format(current_depth, decisions))
    
        free_variables = expression.free(current_formula)
        num_frontier_decisions, best_decision, best_decision_formula = -1, None, None

        # Try all assignments to all free variables, referencing the solver for the
        # quality of the decision.
        for variable in free_variables:
            for assignment in [True, False]:
                fp = expression.simplify(expression.assign(current_formula, variable, assignment))

                if expression.trivially_sat(fp):
                    sat, num_decisions = True, 0
                elif expression.trivially_unsat(fp):
                    sat, num_decisions = False, 0
                else:
                    sat, num_decisions = solver.solve(expression.to_dimacs(fp))

                if not sat:
                    continue

                if num_frontier_decisions == -1 or num_decisions < num_frontier_decisions:
                    best_decision = (variable, assignment)
                    num_frontier_decisions = num_decisions
                    best_decision_formula = fp

        if verbose:
            print("💈 Best decision has frontier {}".format(num_frontier_decisions))
        decisions.append(best_decision)
        current_formula = best_decision_formula
        current_depth += 1

    total_num_decisions = depth + num_frontier_decisions
    if verbose:
        print("Initial solver decisions: {}, found number of decisions: {}".format(initial_solver_guess, total_num_decisions))

    if total_num_decisions < initial_solver_guess:
        return decisions, (initial_solver_guess, total_num_decisions)
    
    return None
