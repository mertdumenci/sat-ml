
"""
A faster version of `expression` with a very, very limited set of operations.
"""

from typing import List, Set


FastCNF = List[List[int]]


def free(cnf: FastCNF) -> Set[int]:
    """Free variables in a fast CNF."""
    var = set()
    var.update(*((abs(v) for v in clause) for clause in cnf))

    return var


def from_dimacs(dimacs: str) -> FastCNF:
    """Parses a DIMACS file into a fast CNF expression."""
    clauses = []
    lines = dimacs.split('\n')

    # Avoiding function calls
    for line in lines:
        if len(line) == 0 or line[0] == 'c' or line[0] == 'p':
            continue
        
        variables = [int(v) for v in line[:-1].strip().split(' ')]
        clauses.append(variables)

    return clauses


def to_dimacs(cnf: FastCNF, num_vars=None) -> str:
    """Writes a DIMACS file from a fast CNF expression."""
    max_var = max(free(cnf))
    len_clause = len(cnf)

    dimacs = f"p cnf {num_vars if num_vars else max_var} {len_clause}\n"
    for clause in cnf:
        dimacs += f"{' '.join((str(v) for v in clause))} 0\n"
    
    return dimacs


def rename_(cnf: FastCNF, var_map=None):
    """Normalizes variable names in a CNF expression. Runs *in-place*."""
    if not var_map:
        var_map = {}

    max_var = 0

    # Linear in the number of literals
    for i, clause in enumerate(cnf):
        for j, v in enumerate(clause):
            abs_v = abs(v)

            if abs_v in var_map:
                abs_v = var_map[abs_v]
            else:
                max_var += 1
                var_map[abs_v] = max_var
                abs_v = max_var

            clause[j] = abs_v if v > 0 else -abs_v


def assign(cnf: FastCNF, var: int, val: bool) -> FastCNF:
    """var = val in a fast CNF, simplifying the resulting formula on a clause-level."""
    new_cnf = []

    for clause in cnf:
        i, skipping_clause = 0, False
        new_clause = []

        while i < len(clause) and not skipping_clause:
            candidate_var = clause[i]
            
            if abs(candidate_var) != abs(var):
                new_clause.append(candidate_var)
    
            # If polarities match, the whole clause is true.
            elif (candidate_var < 0 and val == False) or \
                (candidate_var > 0 and val == True):
                skipping_clause = True
            
            # If polarities are opposing, then this variable is out of
            # the clause.

            i += 1

        if not skipping_clause:
            new_cnf.append(new_clause)
        
    return new_cnf


if __name__ == '__main__':
    DIMACS = "p cnf 3 2\n1 2 0\n3 -2 0\n"

    a = [[3, 2, 1], [-1, 3, 2], [-1, 1, 2]]
    print(a)
    rename_(a)
    print(a)
    print(assign(a, 1, True))
    print(assign(a, 1, False))
    print(assign(a, 3, True))

    print(from_dimacs(DIMACS))
    print(to_dimacs(from_dimacs(DIMACS)))
    assert DIMACS == to_dimacs(from_dimacs(DIMACS))
