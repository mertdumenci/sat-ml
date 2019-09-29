"""
Explores the SAT search space, tries to improve on the solver heuristic.
Author: Mert Dumenci
"""

import random
from typing import List, Tuple
from satml import expression, types, solver


def h_star(f: expression.Expression, solver: solver.Solver, depth: int, verbose=False) -> List[Tuple[expression.Expression, types.Decision]]:
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
    num_vars = max((int(v) for v in expression.free(f)))
    num_frontier_decisions = -1
    
    if verbose:
        print("🎬 Starting with initial solver guess of {} decisions".format(initial_solver_guess))

    while current_formula and current_depth < depth:
        if verbose:
            print("✅ Level {}".format(current_depth))
    
        free_variables = list(expression.free(current_formula))
        # We don't want any internal Python set-iteration-order bias to seep in our variable selection metrics.
        # Also, why is this in-place Python????
        random.shuffle(free_variables)
        num_frontier_decisions, best_decision, best_decision_formula = -1, None, None

        if verbose:
            print("Initial: {}".format(expression.pprint(current_formula)))

        # Try all assignments to all free variables, referencing the solver for the
        # quality of the decision.
        for variable in free_variables:
            for assignment in [True, False]:
                fp = expression.simplify(expression.assign(current_formula, variable, assignment))

                if expression.trivially_sat(fp):
                    print("Trivially sat")
                    sat, num_decisions = True, 0
                elif expression.trivially_unsat(fp):
                    print("Trivially unsat")
                    sat, num_decisions = False, 0
                else:
                    sat, num_decisions = solver.solve(expression.to_dimacs(fp, num_vars=num_vars))

                # Need to normalize the number of decisions because we make the solver "decide" for level more variables
                # by injecting the total # of variables in the dimacs at every level. (so that deciding the last variable doesn't restrict the
                # search space and artificially deflate the decisions in the solver.)
                num_decisions = max(0, num_decisions - current_depth)

                if verbose and sat:
                    print("{} {} = {}, Current Best Frontier: {}, New Frontier: {}".format("💚" if num_decisions < num_frontier_decisions else "🔴", variable, assignment, num_frontier_decisions, num_decisions))
                    print(expression.pprint(fp))
                    print(expression.to_dimacs(fp))

                if sat and (num_frontier_decisions == -1 or num_decisions < num_frontier_decisions):
                    best_decision = current_formula, (variable, assignment)
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

    if num_frontier_decisions > 0 and total_num_decisions < initial_solver_guess:
        return decisions, initial_solver_guess, total_num_decisions
    
    return None
