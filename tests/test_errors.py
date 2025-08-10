"""Tests for error handling and edge cases."""

from typing import Any

import numpy as np
import pytest

from tests.typing import AnyLSSVCClass


@pytest.mark.parametrize(
    "params,expected_gamma,expected_kernel",
    [
        ({"gamma": 10.0, "kernel": "rbf"}, 10.0, "rbf"),
        ({}, 1.0, "rbf"),  # default values
    ],
)
def test_hyperparameter_initialization(
    model_class: AnyLSSVCClass,
    params: dict[str, Any],
    expected_gamma: float,
    expected_kernel: str,
) -> None:
    """Test that hyperparameters are set correctly during initialization."""
    model = model_class(**params)
    assert model.gamma == expected_gamma
    assert model.kernel_ == expected_kernel


@pytest.mark.parametrize(
    "invalid_params,expected_exception",
    [
        ({"kernel": "invalid_kernel"}, KeyError),
        ({"gamma": -1.0}, ValueError),
        ({"gamma": 0}, ValueError),
        ({"kernel": "rbf", "sigma": -0.5}, ValueError),
        ({"kernel": "poly", "d": 0}, ValueError),
    ],
)
def test_invalid_hyperparameters(
    model_class: AnyLSSVCClass,
    invalid_params: dict[str, Any],
    expected_exception: type[Exception],
) -> None:
    """Test that invalid hyperparameters raise appropriate exceptions."""
    with pytest.raises(expected_exception):
        model_class(**invalid_params)


def test_input_shape_validation(model_class: AnyLSSVCClass) -> None:
    """Test ValueError when X and y have mismatched lengths during fit."""
    model = model_class()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1])
    with pytest.raises(ValueError):
        model.fit(X, y)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_invalid_data_values(
    blobs_data: tuple[np.ndarray, np.ndarray],
    model_class: AnyLSSVCClass,
    invalid_value: float,
) -> None:
    """Test ValueError when X contains NaN or infinity values."""
    X, y = blobs_data
    model = model_class()

    X_invalid = X.copy()
    X_invalid[0, 0] = invalid_value
    with pytest.raises(ValueError):
        model.fit(X_invalid, y)
