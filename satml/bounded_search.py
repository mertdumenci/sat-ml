
"""
Explores the SAT search space, tries to improve on the solver heuristic.
Author: Mert Dumenci
"""

import random
from typing import List, Tuple
from satml import fast_cnf, types, solver


def h_star(dimacs: types.Dimacs, solver: solver.Solver, depth: int) -> List[Tuple[fast_cnf.FastCNF, types.Decision]]:
    """
    Given a formula, run an exhaustive search for a given number of levels
    improving the solver heuristic if possible.

    The number of decisions made by the solver heuristic `h` is an _upper bound_
    on the number of decisions made by the heuristic this method defines.

    We can only improve on it.

    :pre: `f` must be satisfiable.
    """
    current_depth = 0

    sat, initial_solver_guess = solver.solve(dimacs)
    assert sat, "`dimacs` must be satisfiable."

    current_formula = fast_cnf.from_dimacs(dimacs)
    # Normalize variable names.
    fast_cnf.rename_(current_formula)

    decisions = []
    num_vars = max(fast_cnf.free(current_formula))
    num_frontier_decisions = -1

    while current_formula and current_depth < depth:
        free_variables = list(fast_cnf.free(current_formula))
        # We don't want any internal Python set-iteration-order bias to seep in our variable selection metrics.
        random.shuffle(free_variables)
        num_frontier_decisions, best_decision, best_decision_formula = -1, None, None

        # Try all assignments to all free variables, referencing the solver for the
        # quality of the decision.
        for variable in free_variables:
            for assignment in [True, False]:
                fp = fast_cnf.assign(current_formula, variable, assignment)
            
                # if expression.trivially_sat(fp):
                #     sat, num_decisions = True, 0
                # elif expression.trivially_unsat(fp):
                #     sat, num_decisions = False, 0
                # else:
                sat, num_decisions = solver.solve(fast_cnf.to_dimacs(fp, num_vars=num_vars))

                # Need to normalize the number of decisions, because every time we assign to the maximally named variable,
                # the solver makes one less decision (because in its eyes the number of variables go down.)
                # 
                # Assigning 2 or 3 in (1 or 2 or 3) shouldn't change the number of decisions made, they should both output 3 (one trivial.)
                # We do this in the above line, and just subtract the number of decisions we know we've made from the solver's output here.
                num_decisions = max(0, num_decisions - current_depth)

                if sat and (num_frontier_decisions == -1 or num_decisions < num_frontier_decisions):
                    best_decision = current_formula, (variable, assignment)
                    num_frontier_decisions = num_decisions
                    best_decision_formula = fp

        decisions.append(best_decision)
        current_formula = best_decision_formula
        current_depth += 1

    total_num_decisions = depth + num_frontier_decisions

    if num_frontier_decisions > 0 and total_num_decisions < initial_solver_guess:
        return decisions, initial_solver_guess, total_num_decisions
    
    return None
