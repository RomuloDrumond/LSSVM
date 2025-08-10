"""Tests for kernel functions."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from utils.kernel import _linear_np, _poly_np, _rbf_np, get_kernel


def test_linear_kernel() -> None:
    """Test the linear kernel function."""
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[5, 6], [7, 8]])
    expected = np.dot(x1, x2.T)
    result = _linear_np(x1, x2)
    assert np.allclose(result, expected)

    x3 = np.array([1, 2])
    x4 = np.array([3, 4])
    expected = np.dot(x3, x4.T)
    result = _linear_np(x3, x4)
    assert np.allclose(result, expected)


def test_rbf_kernel() -> None:
    """Test the RBF kernel function."""
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[1, 2], [3, 5]])

    # Manual calculation
    diff_00 = x1[0] - x2[0]
    expected_val_00 = np.exp(-np.dot(diff_00, diff_00) / (0.1**2))

    diff_11 = x1[1] - x2[1]
    expected_val_11 = np.exp(-np.dot(diff_11, diff_11) / (0.1**2))

    result = _rbf_np(x1, x2, sigma=0.1)
    assert np.allclose(result[0, 0], expected_val_00)
    assert np.allclose(result[1, 1], expected_val_11)


def test_polynomial_kernel() -> None:
    """Test the polynomial kernel function."""
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[5, 6], [7, 8]])

    # Manual calculation
    expected = (np.dot(x1, x2.T) + 1) ** 3

    result = _poly_np(x1, x2, d=3)
    assert np.allclose(result, expected)


@pytest.mark.property
@given(
    X=arrays(
        np.float64,
        (10, 2),
        elements=floats(
            min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
    )
)
@pytest.mark.parametrize("kernel_name", ["linear", "poly", "rbf"])
def test_kernel_symmetry(X: np.ndarray, kernel_name: str) -> None:
    """Property test for kernel symmetry K(X, X) == K(X, X).T."""
    kernel_func = get_kernel(kernel_name)
    K = kernel_func(X, X)
    assert np.allclose(K, K.T)
