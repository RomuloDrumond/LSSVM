"""Tests for the multiclass classification strategy."""

from collections.abc import Callable
from typing import Any

import numpy as np

from tests.conftest import assert_high_accuracy
from tests.typing import AnyLSSVC, AnyLSSVCClass


def test_multiclass_internals(
    multiclass_data: tuple[np.ndarray, np.ndarray], model_class: AnyLSSVCClass
) -> None:
    """Test that multiclass models create the correct internal structure."""
    X, y = multiclass_data
    n_classes = len(np.unique(y))

    model = model_class()
    model.fit(X, y)

    # One-vs-Rest should train n_classes binary classifiers
    assert model.b is not None and model.b.shape[0] == n_classes
    assert model.alpha is not None and model.alpha.shape[0] == n_classes


def test_multiclass_prediction(
    multiclass_data: tuple[np.ndarray, np.ndarray],
    standard_rbf_model: AnyLSSVC,
    to_numpy: Callable[[Any], np.ndarray],
) -> None:
    """Test prediction accuracy on a multiclass dataset."""
    X, y = multiclass_data
    model = standard_rbf_model
    model.fit(X, y)

    predictions = to_numpy(model.predict(X))
    assert_high_accuracy(y, predictions, threshold=0.95)
