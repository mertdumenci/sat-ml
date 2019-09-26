
"""
Generates data for the training process.
"""

import torch
from torch.utils import data
from typing import List, Tuple

from satml import generator, types, expression
from satml.solvers import cryptominisat
import satml.solver


def expand_history(
    formula: expression.Expression,
    history: types.DecisionHistory) -> List[Tuple[int, bool, expression.Expression]]:
    print("[info] Expanding history for {}: {}".format(expression.pprint(formula), history))

    expanded_history = []
    cur_formula = formula

    for var, decision in history:
        # My expression library only accepts string variables.
        var = str(var)

        # The point in the decision history belongs to the previous formula
        expanded_history.append((var, decision, cur_formula))
        cur_formula = expression.simplify(expression.assign(cur_formula, var, decision))

    return expanded_history


class HeuristicSamples(data.Dataset):
    """
    Characterizes a dataset composed of `formula, decision` tuples necessary to
    learn a SAT branching heuristic.
    """
    def __init__(
        self, 
        solver: satml.solver.Solver,
        num_formulas: int,
        clause_width: int,
        max_num_vars: int,
        max_num_clauses: int):
        self.solver = solver
        self.num_formulas = num_formulas
        self.clause_width = clause_width
        self.max_num_vars = max_num_vars
        self.max_num_clauses = max_num_clauses

        # TODO(mert): Maybe use a faster representation of a CNF
        # rather than an AST?

        # :type: List[Tuple[int, bool, expression.Expression]]
        self.next_k = []

    def __len__(self):
        return self.num_formulas

    def __getitem__(self, i) -> Tuple[expression.Expression, Tuple[int, bool]]:
        if len(self.next_k) == 0:
            # Generate random satisfiable formula
            dimacs, (_, history) = generator.generate_random(
                self.clause_width,
                self.max_num_vars,
                self.max_num_clauses,
                force_satisfiable=True,
                solver=self.solver
            )

            # Parse out Dimacs, expand history
            f = expression.from_dimacs(dimacs)
            self.next_k = expand_history(f, history)

        var, decision, f = self.next_k.pop(0)
        return f, (var, decision)


if __name__ == '__main__':
    solver = cryptominisat.Cryptominisat()
    samples = HeuristicSamples(solver, 5, 3, 10, 5)

    for i in range(5):
        f, (var, decision) = samples[i]

        print("{} = {} for formula: {}".format(var, decision, expression.pprint(f)))