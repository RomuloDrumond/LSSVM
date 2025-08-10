"""Tests for edge cases and robustness."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.conftest import assert_predictions_valid
from tests.typing import AnyLSSVCClass


@pytest.mark.parametrize("gamma", [1e-6, 1e6])
def test_extreme_gamma_values(
    blobs_data: tuple[np.ndarray, np.ndarray],
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
    gamma: float,
) -> None:
    """Test behavior with very high and very low `gamma` values."""
    X, y = blobs_data
    model = model_class(gamma=gamma)
    model.fit(X, y)

    predictions = to_numpy(model.predict(X))
    assert_predictions_valid(predictions)


@pytest.mark.parametrize(
    "test_case",
    [
        ("duplicate_samples", "Test model handles duplicate samples without crashing"),
        (
            "zero_variance_features",
            "Test features with zero variance don't result in `NaN` or `Inf`",
        ),
    ],
)
def test_robustness_scenarios(
    blobs_data: tuple[np.ndarray, np.ndarray],
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
    kernel_type: str,
    test_case: tuple[str, str],
) -> None:
    """Test model robustness across various edge case scenarios."""
    case_name, _ = test_case
    X, y = blobs_data

    if case_name == "duplicate_samples":
        # Create duplicate samples
        X_test = np.vstack([X, X[:10]])
        y_test = np.concatenate([y, y[:10]])
    elif case_name == "zero_variance_features":
        # Add a feature with zero variance
        X_test = np.hstack([X, np.ones((X.shape[0], 1))])
        y_test = y

    model = model_class(kernel=kernel_type)
    model.fit(X_test, y_test)

    predictions = to_numpy(model.predict(X_test))
    assert_predictions_valid(predictions)

    # Additional checks for specific cases
    if case_name == "duplicate_samples":
        assert predictions.shape[0] == X_test.shape[0]


def test_high_dimensional_data(
    rng: Any,
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
    kernel_type: str,
) -> None:
    """Test model solves correctly when d > n (dimensions > samples)."""
    n_samples = 50
    n_features = 100

    X = rng.random((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    y[y == 0] = -1

    model = model_class(kernel=kernel_type)
    model.fit(X, y)

    predictions = to_numpy(model.predict(X))

    assert predictions.shape[0] == n_samples
    assert_predictions_valid(predictions)
