
"""
Generates data for the training process.
"""

import sys
import multiprocessing
import argparse
import pickle
from typing import List, Tuple

import tqdm
from satml import generator, types, expression, bounded_search
from satml.solvers import cryptominisat
import satml.solver


def generator_instance(packed_args):
    num_vars, num_clauses, expansion_depth = packed_args

    solver = cryptominisat.Cryptominisat()
    _, f, _ = generator.generate_random(3, num_vars=num_vars, num_clauses=num_clauses, solver=solver, force_satisfiable=True)
    return bounded_search.h_star(f, solver, expansion_depth)


parser = argparse.ArgumentParser('Generates sat-ml data')
parser.add_argument('--num_formulas', type=int, help='Number of formulas to generate.')
parser.add_argument('--num_vars', type=int, help='Number of variables in random formulas.')
parser.add_argument('--num_clauses', type=int, help='Number of clauses in random formulas.')
parser.add_argument('--expansion_depth', type=int, help='H_star expansion depth', default=5)
parser.add_argument('--output', type=str, help='Where to write the pickled file.')

args = parser.parse_args()


with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    # This is weird... Surely there's a better way of doing this.
    pool_map = p.imap(generator_instance, ((args.num_vars, args.num_clauses, args.expansion_depth) for _ in range(args.num_formulas)))
    
    results = list(tqdm.tqdm(pool_map, total=args.num_formulas))
    # Filter out unsuccessful searches
    results = [result for result in results if result is not None]
    dataset = []

    # Unpack decisions, generate statistics
    avg_improvement_factor = 0.0
    for decisions, solver_guess, found_decisions in results:
        avg_improvement_factor += (solver_guess / found_decisions)
        dataset += decisions

    avg_improvement_factor /= len(results)
    print("✅ {} formulas with {:.2f}x average reduction in decision trail length".format(len(dataset), avg_improvement_factor))

    # Write to disk
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)
