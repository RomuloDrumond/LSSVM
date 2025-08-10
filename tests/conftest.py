"""Fixtures for the test suite."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.random import Generator
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score

from lssvm.LSSVC import LSSVC

from .typing import AnyLSSVC, AnyLSSVCClass

try:
    import torch

    from lssvm.LSSVC_GPU import LSSVC_GPU

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


@pytest.fixture
def to_numpy() -> Callable[[Any], np.ndarray]:
    """Return a function to convert data to a NumPy array."""

    def converter(data: Any) -> np.ndarray:
        if TORCH_AVAILABLE and hasattr(data, "cpu"):
            return cast(np.ndarray, data.cpu().numpy())
        return np.asarray(data)

    return converter


@pytest.fixture
def rng() -> Generator:
    """Return a reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture(params=["linear", "poly", "rbf"])
def kernel_type(request: pytest.FixtureRequest) -> str:
    """Return kernel types for parametrized testing."""
    return cast(str, request.param)


@pytest.fixture
def blobs_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a linearly separable dataset."""
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.2)
    # Ensure labels are -1 and 1
    y[y == 0] = -1
    return X, y


@pytest.fixture
def moons_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a non-linearly separable dataset."""
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    # Ensure labels are -1 and 1
    y[y == 0] = -1
    return X, y


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple multiclass dataset (3 classes)."""
    X, y = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
    return X, y


# Common model configuration fixtures
@pytest.fixture
def high_gamma_model(model_class: AnyLSSVCClass) -> AnyLSSVC:
    """Return a model with high gamma for perfect accuracy tests."""
    return model_class(gamma=100)


@pytest.fixture
def standard_rbf_model(model_class: AnyLSSVCClass) -> AnyLSSVC:
    """Return a model with standard RBF configuration."""
    return model_class(gamma=10.0, kernel="rbf", sigma=0.5)


# Dynamically create model class parameters
# TODO: type checking got crazy, need to address this in the future
_model_classes_params = [pytest.param(LSSVC, id="LSSVC_cpu")]
if TORCH_AVAILABLE and CUDA_AVAILABLE:
    _model_classes_params.append(
        pytest.param(
            LSSVC_GPU,
            id="LSSVC_gpu",
            marks=[pytest.mark.slow],
        )
    )


@pytest.fixture(params=_model_classes_params)
def model_class(request: pytest.FixtureRequest) -> AnyLSSVCClass:
    """Return a list of all model classes."""
    return cast(AnyLSSVCClass, request.param)


# Helper functions for common assertions
def assert_predictions_valid(predictions: Any) -> None:
    """Assert that predictions are finite and don't contain NaN."""
    predictions = np.asarray(predictions)
    assert np.isfinite(predictions).all(), "Predictions contain non-finite values"
    assert not np.isnan(predictions).any(), "Predictions contain NaN values"


def assert_perfect_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Assert 100% accuracy between true and predicted labels."""
    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy:.2%}"


def assert_high_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.95
) -> None:
    """Assert accuracy above specified threshold."""
    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy >= threshold, (
        f"Expected accuracy >= {threshold:.2%}, got {accuracy:.2%}"
    )
