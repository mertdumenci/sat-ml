
"""
A vanilla Python SAT solver. Rudimentary--doesn't do any clause learning, branching strategy is random.
Author: Mert Dumenci
"""

import random

from satml import types, solver
from satml.expression import free, expr, Type, _Expression, pprint


class Pysolver(solver.Solver):
    def solve(self, dimacs: types.Dimacs) -> types.Solution:
        """SAT solving."""
        free_v = list(free(an_exp))
        # Solved.
        if not free_v:
            if an_exp.typ == Type.CONST:
                return bool(an_exp.l_val)
            else:
                raise ValueError('Reduced to {}?'.format(pprint(an_exp)))

        split = random.choice(free_v)
        true = simplify(assign_variable(an_exp, split, True))
        false = simplify(assign_variable(an_exp, split, False))

        # TODO(mert): Implement history tracking.
        return (self.solve(true)[0] or self.solve(false)[0], None)


def _satisfiable_branch_all(an_exp, level=0):
    """
    Instead of picking a branch at each level, branch
    on all variables, report back what was locally best.

    (The "best" metric is how many times have we branched after
     this decision, using the same greedy method.)

    `sat :: an_exp -> sat, var, num_branches, history`
    """
    # TODO(mert): Fix this function, perhaps instead of (2n)! operations we can guide ourselves
    # using `cryptominisat`.
    assert False, "This function isn't implemented correctly."
    
    free_v = list(free(an_exp))
    if not free_v:
        if an_exp.typ == Type.CONST:
            return bool(an_exp.l_val), None, True, 0, set()
        else:
            raise ValueError('Reduced to {}?'.format(pprint(an_exp)))

    sat, best_var, assignment, num_branches, history = False, None, True, float('inf'), None
    for var in free_v:
        true = simplify(assign_variable(an_exp, var, True))
        false = simplify(assign_variable(an_exp, var, False))

        # Branch with T.
        sat_t, _, num_t, _, history = _satisfiable_branch_all(true, level=level + 1)
        num_t_p = num_t + 1  # Count this branch.

        if sat_t and num_t_p < num_branches:
            best_var = var
            assignment = True
            num_branches = num_t_p

        # Branch with F.
        sat_f, _, num_f, _, hist_f = _satisfiable_branch_all(false, level=level + 1)
        num_f_p = num_f + 1 # Count this branch.

        if sat_f and num_f_p < num_branches:
            best_var = var
            assignment = False
            num_branches = num_f_p

        sat = sat_t or sat_f

    cur_hist_entry = (an_exp, sat, best_var, assignment, num_branches)
    return sat, best_var, assignment, num_branches, history.union({cur_hist_entry})
