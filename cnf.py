
from satml.expression import pprint, Type, cnf, expr
from satml.solver import satisfiable, simplify
from satml.dimacs import from_dimacs


x_y_z = expr((Type.OR, 'x', (Type.AND, 'y', 'z')))
x_y_z_c = cnf(x_y_z)

complex_1 = expr((Type.OR, 'a', (Type.NOT, (Type.AND, 'b', (Type.OR, 'a', 'c')), None)))
complex_1_c = cnf(complex_1)

print(pprint(x_y_z))
print(pprint(x_y_z_c))

print(pprint(complex_1))
print(pprint(complex_1_c))


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

print(satisfiable(from_dimacs(SOLVABLE_DIMACS_TEST)))
print(satisfiable(from_dimacs(SOLVABLE_DIMACS_TEST_2)))
print(satisfiable(from_dimacs(UNSOLVABLE_DIMACS_TEST_2)))
