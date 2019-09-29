
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
from satml import generator, types, expression, bounded_search
from satml.solvers import cryptominisat
import satml.solver


def generator_instance(packed_args):
    num_vars, num_clauses, expansion_depth, dimacs_path = packed_args

    solver = cryptominisat.Cryptominisat()
    if dimacs_path:
        with open(dimacs_path, 'r') as dimacs:
            dimacs = dimacs.read()
            f = expression.from_dimacs(dimacs)
    else:
        _, f, _ = generator.generate_random(3, num_vars=num_vars, num_clauses=num_clauses, solver=solver, force_satisfiable=True)

    return bounded_search.h_star(f, solver, expansion_depth)


parser = argparse.ArgumentParser('Generates sat-ml data')
parser.add_argument('--sr_dimacs', type=str, help='Should we use a `neurosat` generated random formula set instead of generating it ourselves', required=False)
parser.add_argument('--num_formulas', type=int, help='Number of formulas to generate.', required=False)
parser.add_argument('--num_vars', type=int, help='Number of variables in random formulas.', required=False)
parser.add_argument('--num_clauses', type=int, help='Number of clauses in random formulas.', required=False)
parser.add_argument('--expansion_depth', type=int, help='H_star expansion depth', default=5)
parser.add_argument('--output', type=str, help='Where to write the pickled file.')
parser.add_argument('--single_threaded', type=bool, help='Should we run in single-threaded mode or not', default=False)

args = parser.parse_args()


cpu_count = 1 if args.single_threaded else multiprocessing.cpu_count()
with multiprocessing.Pool(cpu_count) as p:
    sr_dimacs = None
    if args.sr_dimacs:
        pattern = os.path.join(args.sr_dimacs, "*.dimacs")
        sr_dimacs = list(glob.glob(pattern))

    num_formulas = len(sr_dimacs) if not args.num_formulas else args.num_formulas

    if sr_dimacs:
        pool_map = p.imap(generator_instance, ((0, 0, args.expansion_depth, dimacs_path) for dimacs_path in sr_dimacs[:num_formulas]))
    else:
        pool_map = p.imap(generator_instance, ((args.num_vars, args.num_clauses, args.expansion_depth, sr_dimacs[i] if sr_dimacs else None) for i in range(args.num_formulas)))

    results = list(tqdm.tqdm(pool_map, total=num_formulas))
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
