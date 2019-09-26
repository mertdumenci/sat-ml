
from satml.expression import pprint, Type, cnf, expr
from satml.solver import satisfiable, simplify
from satml.dimacs import from_dimacs

import itertools
import os
import pickle
import subprocess
import multiprocessing


x_y_z = expr((Type.OR, 'x', (Type.AND, 'y', 'z')))
x_y_z_c = cnf(x_y_z)

complex_1 = expr((Type.OR, 'a', (Type.NOT, (Type.AND, 'b', (Type.OR, 'a', 'c')), None)))
complex_1_c = cnf(complex_1)

# print(pprint(x_y_z))
# print(pprint(x_y_z_c))

# print(pprint(complex_1))
# print(pprint(complex_1_c))


# print(satisfiable(expr((Type.OR, 'x', '-x'))))
# print(satisfiable(expr((Type.AND, 'x', '-x'))))


SOLVABLE_DIMACS_TEST = """
c Pigeonhole principle formula for 1 pigeons and 2 holes
c Generated with `cnfgen`
c (C) 2012-2018 Massimo Lauria <lauria.massimo@gmail.com>
c https://massimolauria.github.io/cnfgen
c
c COMMAND LINE: cnfgen php 1 2
c
p cnf 2 1
1 2 0
"""

SOLVABLE_DIMACS_TEST_2 = """
c Pigeonhole principle formula for 3 pigeons and 4 holes
c Generated with `cnfgen`
c (C) 2012-2018 Massimo Lauria <lauria.massimo@gmail.com>
c https://massimolauria.github.io/cnfgen
c
c COMMAND LINE: cnfgen php 3 4
c
p cnf 12 15
1 2 3 4 0
5 6 7 8 0
9 10 11 12 0
-1 -5 0
-1 -9 0
-5 -9 0
-2 -6 0
-2 -10 0
-6 -10 0
-3 -7 0
-3 -11 0
-7 -11 0
-4 -8 0
-4 -12 0
-8 -12 0
"""

UNSOLVABLE_DIMACS_TEST_2 = """
c Pigeonhole principle formula for 6 pigeons and 4 holes
c Generated with `cnfgen`
c (C) 2012-2018 Massimo Lauria <lauria.massimo@gmail.com>
c https://massimolauria.github.io/cnfgen
c
c COMMAND LINE: cnfgen php 6 4
c
p cnf 24 66
1 2 3 4 0
5 6 7 8 0
9 10 11 12 0
13 14 15 16 0
17 18 19 20 0
21 22 23 24 0
-1 -5 0
-1 -9 0
-1 -13 0
-1 -17 0
-1 -21 0
-5 -9 0
-5 -13 0
-5 -17 0
-5 -21 0
-9 -13 0
-9 -17 0
-9 -21 0
-13 -17 0
-13 -21 0
-17 -21 0
-2 -6 0
-2 -10 0
-2 -14 0
-2 -18 0
-2 -22 0
-6 -10 0
-6 -14 0
-6 -18 0
-6 -22 0
-10 -14 0
-10 -18 0
-10 -22 0
-14 -18 0
-14 -22 0
-18 -22 0
-3 -7 0
-3 -11 0
-3 -15 0
-3 -19 0
-3 -23 0
-7 -11 0
-7 -15 0
-7 -19 0
-7 -23 0
-11 -15 0
-11 -19 0
-11 -23 0
-15 -19 0
-15 -23 0
-19 -23 0
-4 -8 0
-4 -12 0
-4 -16 0
-4 -20 0
-4 -24 0
-8 -12 0
-8 -16 0
-8 -20 0
-8 -24 0
-12 -16 0
-12 -20 0
-12 -24 0
-16 -20 0
-16 -24 0
-20 -24 0
"""


def pigeonhole(pigeons, holes):
    program = 'cnfgen php {} {}'.format(pigeons, holes)
    return from_dimacs(os.popen(program).read())


def run_extended(an_exp):
    print("On expression: {}".format(pprint(an_exp)))

    def _print_info(history_entry):
        e, s, v, a, n = history_entry
        print("Formula: {}\n{} where best variable was {} = {} with {} branches.".format(
            pprint(e),
            "Satisfiable" if s else "Unsatisfiable",
            v,
            a,
            n
        ))

    sat, best_var, assignment, num_b, hist = satisfiable(an_exp, branch_all=True)
    _print_info((an_exp, sat, best_var, assignment, num_b))

    # Filter history to only contain satisfiable formulas. We don't care about anything else.
    hist = [(e, s, v, a, n) for e, s, v, a, n in hist if s == True]

    return hist


def random_dimacs_cnf(width, n_var, n_clauses):
    program = 'cnfgen randkcnf {} {} {}'.format(width, n_var, n_clauses)
    return os.popen(program).read()


def fast_check_sat(dimacs):
    sat = subprocess.Popen(
        ["cryptominisat5", "--verb", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    out, err = sat.communicate(input=dimacs.encode())
    return sat.returncode == 10


def random_satisfiable_cnf(width, n_var, n_clauses):
    cnf = None

    while cnf is None:
        rnd = random_dimacs_cnf(width, n_var, n_clauses)
        if fast_check_sat(rnd):
            cnf = from_dimacs(rnd)

    return cnf


NUM_INITIAL_FORMULAS = 3000
NUM_VARIABLES = 15
WIDTH = 3
NUM_CLAUSES = 5

# TODO Mert (September 20, 2019)
# We have a problem!
# Formula: ((3 ∨ (-2 ∨ 1)) ∧ (-3 ∨ (-2 ∨ -1)))
# Satisfiable where best variable was 3 with 4 branches.

# TEST_CASE = expr((Type.AND, 
#         (Type.OR,
#          (Type.VAR, '3', None),
#          (Type.OR, 
#           (Type.NOT, (Type.VAR, '2', None), None),
#           (Type.VAR, '1', None))
#         ),
#         (Type.OR,
#          (Type.NOT, (Type.VAR, '3', None), None),
#          (Type.OR, 
#           (Type.NOT, (Type.VAR, '2', None), None),
#           (Type.NOT, (Type.VAR, '1', None), None))
#         )
# ))

# run_extended(TEST_CASE)


def spawn_one_formula(i):
    print("Running iteration", i)
    formula = random_satisfiable_cnf(WIDTH, NUM_VARIABLES, NUM_CLAUSES)
    hist = run_extended(formula)

    print("Generated {} formulas".format(len(hist)))

    return hist


pool = multiprocessing.Pool(processes=8)

all_history = set().union(*pool.imap_unordered(spawn_one_formula, range(NUM_INITIAL_FORMULAS)))
print("Generated {} examples.".format(len(all_history)))

output_file = 'data/history_{}_vars_{}_clauses_{}_width_{}_initials.pickle'.format(NUM_VARIABLES, NUM_CLAUSES, WIDTH, NUM_INITIAL_FORMULAS)
with open(output_file, 'wb') as f:
    pickle.dump(all_history, f)

# print(spawn_one_formula(1))
