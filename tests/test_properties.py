"""Property-based tests for the model."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.random import Generator

from tests.typing import AnyLSSVCClass


@pytest.mark.property
def test_reproducibility(
    blobs_data: tuple[np.ndarray, np.ndarray],
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
    kernel_type: str,
) -> None:
    """Train model twice with the same data, assert identical predictions."""
    X, y = blobs_data

    # Train first model
    model1 = model_class(kernel=kernel_type)
    model1.fit(X, y)
    predictions1 = to_numpy(model1.predict(X))

    # Train second model
    model2 = model_class(kernel=kernel_type)
    model2.fit(X, y)
    predictions2 = to_numpy(model2.predict(X))

    assert np.array_equal(predictions1, predictions2)


@pytest.mark.property
def test_permutation_invariance(
    blobs_data: tuple[np.ndarray, np.ndarray],
    rng: Generator,
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
    kernel_type: str,
) -> None:
    """Train on original and shuffled data, assert same predictions for test point."""
    X, y = blobs_data

    # Train on original data
    model1 = model_class(kernel=kernel_type)
    model1.fit(X, y)

    # Create a shuffled version of the data
    p = rng.permutation(len(X))
    X_shuffled, y_shuffled = X[p], y[p]

    # Train on shuffled data
    model2 = model_class(kernel=kernel_type)
    model2.fit(X_shuffled, y_shuffled)

    # Test point
    test_point = np.array([[0, 0]])

    pred1 = to_numpy(model1.predict(test_point))
    pred2 = to_numpy(model2.predict(test_point))

    assert np.array_equal(pred1, pred2)
