
"""
Types used in `satml`.
Author: Mert Dumenci
"""

from typing import List, Tuple

"""
DIMACS representations of SAT problems.
"""
Dimacs = str

"""
The decision history of the solver.
`[(literal, polarity [true = Positive, false = Negative])]`
"""
DecisionHistory = List[Tuple[int, bool]]

"""
A SAT solution.
"""
Solution = Tuple[bool, DecisionHistory] # TODO(mert): Perhaps add interpretation here?
