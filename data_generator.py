
"""
Generates data for the training process.
"""

import sys
import os
import glob
import multiprocessing
import argparse
import pickle
from typing import List, Tuple

import tqdm
from satml import generator, types, fast_cnf, bounded_search
from satml.solvers import cryptominisat
import satml.solver


def generator_instance(packed_args):
    dimacs_path, expansion_depth = packed_args

    solver = cryptominisat.Cryptominisat()
    with open(dimacs_path, 'r') as dimacs:
        f = dimacs.read()

    return bounded_search.h_star(f, solver, expansion_depth)


parser = argparse.ArgumentParser('Generates sat-ml data')
parser.add_argument('--dimacs', type=str, help='Folder of formulas in `dimacs` format.')
parser.add_argument('--expansion_depth', type=int, help='H_star expansion depth', default=5)
parser.add_argument('--output', type=str, help='Where to write the pickled file.')
parser.add_argument('--single_threaded', type=bool, help='Should we run in single-threaded mode or not', default=False)

args = parser.parse_args()


cpu_count = 1 if args.single_threaded else multiprocessing.cpu_count()
with multiprocessing.Pool(cpu_count) as p:
    pattern = os.path.join(args.dimacs, "*.dimacs")
    names = list(glob.glob(pattern))
    
    # Run the search
    pool_map = p.imap(generator_instance, ((path, args.expansion_depth) for path in names))
    results = list(tqdm.tqdm(pool_map, total=len(names)))

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
