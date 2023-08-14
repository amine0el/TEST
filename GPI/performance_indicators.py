"""Performance indicators for multi-objective RL algorithms.

We mostly rely on pymoo for the computation of axiomatic indicators (HV and IGD), but some are customly made.
"""
from copy import deepcopy
from typing import Callable, List

import numpy as np
import numpy.typing as npt




def expected_utility(front: List[np.ndarray], weights_set: List[np.ndarray], utility: Callable = np.dot) -> float:
    """Expected Utility Metric.

    Expected utility of the policies on the PF for various weights.
    Similar to R-Metrics in MOO. But only needs one PF approximation.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the eum on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: eum metric
    """
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def maximum_utility_loss(
    front: List[np.ndarray], reference_set: List[np.ndarray], weights_set: np.ndarray, utility: Callable = np.dot
) -> float:
    """Maximum Utility Loss Metric.

    Maximum utility loss of the policies on the PF for various weights.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the mul on
        reference_set: reference set (e.g. true Pareto front) to compute the mul on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: mul metric
    """
    max_scalarized_values_ref = [np.max([utility(weight, point) for point in reference_set]) for weight in weights_set]
    max_scalarized_values = [np.max([utility(weight, point) for point in front]) for weight in weights_set]
    utility_losses = [max_scalarized_values_ref[i] - max_scalarized_values[i] for i in range(len(max_scalarized_values))]
    return np.max(utility_losses)
