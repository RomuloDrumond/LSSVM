"""Label encoding utilities for LSSVM algorithms.

This module provides functions to convert between different label encoding formats.
"""

import numpy as np


def dummie2multilabel(X: np.ndarray) -> np.ndarray:
    """Convert dummy variables to multilabel format.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input array with dummy variables (one-hot encoded).

    Returns
    -------
    ndarray of shape (n_samples, 1)
        Multilabel encoded array where each row contains the index
        of the active class.
    """
    N = len(X)
    X_multi = np.zeros((N, 1), dtype="int")
    for i in range(N):
        X_multi[i] = np.where(X[i] == 1)[0]
    return X_multi
