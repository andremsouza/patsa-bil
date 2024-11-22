"""This module contains utility functions for PCA analysis."""

from typing import Callable, Optional

import numpy as np
from sklearn.decomposition import PCA


def broken_stick(X: np.ndarray, normalize: bool = False) -> int:
    """
    Calculate the number of principal components to retain using the broken stick method.

    Args:
        X (np.ndarray): The input data matrix of shape (n_samples, n_features).
        normalize (bool, optional): Whether to normalize the data. Defaults to False.

    Returns:
        int: The number of principal components to retain.

    """
    if normalize:
        # Z-normalize the data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Initialize and fit PCA
    pca = PCA()
    pca.fit(X)

    # Calculate the explained variance
    explained_variance = pca.explained_variance_ratio_
    expected_values: list[float] = []
    d: int = len(explained_variance)
    c: int = 1

    # Calculate the expected values
    for i in range(1, d + 1):
        j: int = d - i + 1

        total_sum: float = 0.0
        for k in range(1, (d - j + 1) + 1):
            total_sum += 1 / (d + 1 - k)
        total_sum *= c / d

        expected_values.append(total_sum)

    # Compare the expected values with the explained variance

    def compare_greater(x, y):
        return x > y

    def compare_less(x, y):
        return x < y

    func_compare: Optional[Callable] = None
    if expected_values[0] > explained_variance[0]:
        func_compare = compare_greater
    else:
        func_compare = compare_less

    ans: int = 1
    while (
        func_compare(expected_values[ans - 1], explained_variance[ans - 1]) and ans != d
    ):
        ans += 1

    return max(2, ans)
