"""
n: new file. Patterns for the Dyck (balanced brackets) task, used to mark positions in the plots.

Every line is a sequence of "(" and ")" whose nesting depth never goes below 0 or above the maximum
depth of the generator (5, see data/synthetic_dyck.py). Most next characters are random, except at
two kinds of positions, where the next character is FORCED and a model that tracks the depth can be sure:
    - depth 0:          only "(" is allowed next
    - depth = maximum:  only ")" is allowed next
"""
import os
from typing import Dict

import numpy as np

__all__ = ["patterns"]

DYCK_MAX_DEPTH = int(os.environ.get("DYCK_MAX_DEPTH", 5))


def patterns(text: str) -> Dict[str, np.ndarray]:
    N = len(text)
    depth = np.zeros(N, dtype=int)  # nesting depth AFTER reading each character
    d = 0
    for t, ch in enumerate(text):
        if ch == "(":
            d += 1
        elif ch == ")":
            d = max(0, d - 1)
        elif ch == "\n":
            d = 0  # every line is a new sequence
        depth[t] = d

    is_bracket = np.array([ch in "()" for ch in text], dtype=bool)
    return {
        "Open brackets": np.array([ch == "(" for ch in text], dtype=bool),
        "Close brackets": np.array([ch == ")" for ch in text], dtype=bool),
        "Forced open (depth 0)": is_bracket & (depth == 0),
        "Forced close (max depth)": is_bracket & (depth == DYCK_MAX_DEPTH),
        "Deep (depth 3+)": is_bracket & (depth >= 3),  # c: no ">" in the name, it becomes a file name (not allowed on Windows)
    }


