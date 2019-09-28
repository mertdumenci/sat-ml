
"""
Solver interface around `cryptominisat`.
Author: Mert Dumenci
"""

import subprocess
import tempfile
import re

from satml import types, solver


class Cryptominisat(solver.Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions_regex = re.compile("c\ decisions\ *:\ ([0-9]+).*")

    def solve(self, dimacs: types.Dimacs) -> types.Satisfiability:
        instance = subprocess.Popen(
            ["cryptominisat5", "--verb", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        out, _ = instance.communicate(input=dimacs.encode())
        out = out.decode('utf-8')

        # Fetch satisfiability information
        satisfiable = instance.returncode == 10
        # Fetch how many decisions were made (we're going to try to optimize for this!)
        matches = self.decisions_regex.search(out)
        assert matches is not None, "No decisions output?"
        num_decisions = int(matches.group(1))

        return satisfiable, num_decisions
