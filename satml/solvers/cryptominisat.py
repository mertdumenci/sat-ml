
"""
An interface to `cryptominisat`.
Author: Mert Dumenci
"""

import subprocess
import tempfile

from satml import types, solver


class Cryptominisat(solver.Solver):
    def solve(self, dimacs: types.Dimacs) -> types.Solution:
        # We make a temporary file as `cryptominisat` writes decisions only to a file
        with tempfile.NamedTemporaryFile() as dec_hist_file:
            instance = subprocess.Popen(
                ["cryptominisat5", "--verb", "0", "--dumpdecformodel", dec_hist_file.name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            _ = instance.communicate(input=dimacs.encode())
            
            satisfiable = instance.returncode == 10
            dec_hist = self._parse_decision_history(dec_hist_file)

            return satisfiable, dec_hist

    def _parse_decision_history(
        self, 
        dec_hist_file: tempfile.NamedTemporaryFile) -> types.DecisionHistory:
        """
        Parses the decision history output of `cryptominisat`.
        """
        history = []

        for line in dec_hist_file:
            line = line.decode('utf-8')
            encoded_lit, _, = [int(f) for f in line.split(' ')]

            history.append((
                abs(encoded_lit),
                encoded_lit > 0
            ))
        
        return history
