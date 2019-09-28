
"""
Generates data for the training process.
"""

import sys
import multiprocessing
import argparse
from typing import List, Tuple

import tqdm
from satml import generator, types, expression, bounded_search
from satml.solvers import cryptominisat
import satml.solver


def generator_instance(packed_args):
    num_vars, num_clauses = packed_args

    solver = cryptominisat.Cryptominisat()
    _, f, _ = generator.generate_random(3, num_vars=num_vars, num_clauses=num_clauses, solver=solver, force_satisfiable=True)
    return bounded_search.h_star(f, solver, 5)


parser = argparse.ArgumentParser('Generates sat-ml data')
parser.add_argument('--num_formulas', type=int, help='Number of formulas to generate.')
parser.add_argument('--num_vars', type=int, help='Number of variables in random formulas.')
parser.add_argument('--num_clauses', type=int, help='Number of clauses in random formulas.')
parser.add_argument('--output', type=str, help='Where to write the pickled file.')

args = parser.parse_args()

with multiprocessing.Pool(8) as p:
    # This is weird... Surely there's a better way of doing this.
    pool_map = p.imap(generator_instance, ((args.num_vars, args.num_clauses) for _ in range(args.num_formulas)))
    h_stars = list(tqdm.tqdm(pool_map, total=args.num_formulas))
