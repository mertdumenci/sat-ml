
"""
Abstract class for solvers.
Author: Mert Dumenci
"""

import abc
from satml import types


class Solver(abc.ABC):
    @abc.abstractmethod
    def solve(self, dimacs: types.Dimacs) -> types.Solution:
        pass
