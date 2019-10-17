
"""
Types used in `satml`.
Author: Mert Dumenci
"""

from typing import List, Tuple, Any

"""
DIMACS representations of SAT problems.
"""
Dimacs = str

"""
Whether or not a formula is satisfiable.
`(satisfiable, num_decisions)`
"""
Satisfiability = Tuple[bool, int]

"""
Represents a branching decision.
"""
Decision = Tuple[Tuple[Any], str, bool]
