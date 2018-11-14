
from satml.expression import pprint, Type, cnf, expr
from satml.solver import satisfiable
from satml.dimacs import from_dimacs


x_y_z = expr((Type.OR, 'x', (Type.AND, 'y', 'z')))
x_y_z_c = cnf(x_y_z)

complex_1 = expr((Type.OR, 'a', (Type.NOT, (Type.AND, 'b', (Type.OR, 'a', 'c')), None)))
complex_1_c = cnf(complex_1)

pprint(x_y_z)
pprint(x_y_z_c)

pprint(complex_1)
pprint(complex_1_c)


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

UNSOLVABLE_DIMACS_TEST = """
c Pigeonhole principle formula for 2 pigeons and 1 holes
c Generated with `cnfgen`
c (C) 2012-2018 Massimo Lauria <lauria.massimo@gmail.com>
c https://massimolauria.github.io/cnfgen
c
c COMMAND LINE: cnfgen php 2 1
c
p cnf 2 3
1 0
2 0
-1 -2 0
"""

print(satisfiable(from_dimacs(SOLVABLE_DIMACS_TEST)))
print(satisfiable(from_dimacs(UNSOLVABLE_DIMACS_TEST)))
