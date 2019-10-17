
from functools import reduce
from satml import fast_cnf

import torch
import torch.utils.data as data


def adjacency(formulas):
    """
    Takes any number of formulas and returns an adjacency matrix representing the formulas
    (disconnected if > 1) formulas.
    """
    if not isinstance(formulas, list):
        formulas = [formulas]

    # Find how many variables we'll have, so that we can index beyond
    # that set for negative literals. This is a running sum.
    # 1, 4, 4, 6, 20, 30
    num_variables = reduce(lambda l, f: l + [(0 if not l else l[-1]) + fast_cnf.max_var(f)], formulas, [])
    num_clauses = 0

    # The sparse tensor we're building up
    indices = []

    for n, formula in enumerate(formulas):
        # How many variables have we seen before?
        prev_vars = 0 if n == 0 else num_variables[n - 1]

        for clause in formula:
            for j, literal in enumerate(clause):
                clause_index = num_clauses
                # Literals are 1 indexed
                literal_index = prev_vars + abs(literal) - 1
                # Negative literals go after all positive literals
                if literal < 0:
                    literal_index += num_variables[-1]
                
                indices.append([literal_index, clause_index])

            num_clauses += 1

    indices = torch.LongTensor(indices)
    values = torch.zeros(len(indices)) + 1
    # 2m + n where m is the number of literals, and n is the number of clauses
    size = torch.Size([2 * num_variables[-1], num_clauses])
    adj_matrix = torch.sparse.FloatTensor(indices.t(), values, size)

    return adj_matrix, [fast_cnf.max_var(f) for f in formulas]


def collator(batch):
    """Expects `batch` to be a list of (X, y) pairs."""
    X, y = zip(*batch)
    X = list(X)
    
    adj, num_vars = adjacency(X)
    return (adj.to_dense(), torch.LongTensor(num_vars)), torch.LongTensor(y)


class AdjacencyDataset(data.Dataset):
    def __init__(self, decisions):
        self.decisions = decisions

        # Extract the maximum number of variables in the dataset, we use this
        # for labeling.
        self.max_var = max((fast_cnf.max_var(f) for f, _ in decisions))
        
    def __len__(self):
        return len(self.decisions)
    
    def __getitem__(self, index):
        formula, (var, branch) = self.decisions[index]

        label = 0
        if branch is True:
            label = var - 1
        else:
            label = self.max_var + label

        return formula, label
