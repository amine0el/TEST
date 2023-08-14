"""Pareto utilities."""
from copy import deepcopy
from typing import List

import numpy as np


def get_non_dominated(candidates: set) -> set:
    """This function returns the non-dominated subset of elements.

    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    The code provided in all the stackoverflow answers is wrong. Important changes have been made in this function.

    Args:
        candidates: The input set of candidate vectors.

    Returns:
        The non-dominated subset of this input set.
    """
    candidates = np.array(list(candidates))  # Turn the input set into a numpy array.
    candidates = candidates[candidates.sum(1).argsort()[::-1]]  # Sort candidates by decreasing sum of coordinates.
    for i in range(candidates.shape[0]):  # Process each point in turn.
        n = candidates.shape[0]  # Check current size of the candidates.
        if i >= n:  # If we've eliminated everything up until this size we stop.
            break
        non_dominated = np.ones(candidates.shape[0], dtype=bool)  # Initialize a boolean mask for undominated points.
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        non_dominated[i + 1 :] = np.any(candidates[i + 1 :] > candidates[i], axis=1)
        candidates = candidates[non_dominated]  # Grab only the non-dominated vectors using the generated bitmask.

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))  # Add the non dominated vectors to a set again.

    return non_dominated
