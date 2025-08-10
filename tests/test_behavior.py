"""Tests for the model's classification behavior."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.conftest import assert_perfect_accuracy
from tests.typing import AnyLSSVCClass


@pytest.mark.parametrize(
    "test_case,data_fixture",
    [
        ("perfect_accuracy", "blobs_data"),
        ("alternative_labels", "blobs_data"),
    ],
)
def test_classification_scenarios(
    request: pytest.FixtureRequest,
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
    kernel_type: str,
    test_case: str,
    data_fixture: str,
) -> None:
    """Test classification accuracy across different scenarios and label formats."""
    X, y = request.getfixturevalue(data_fixture)

    if test_case == "alternative_labels":
        # Change labels from {-1, 1} to {0, 1}
        y = y.copy()
        y[y == -1] = 0

    # Use high gamma to reduce regularization for perfect accuracy
    model = model_class(gamma=100, kernel=kernel_type)
    model.fit(X, y)

    predictions = to_numpy(model.predict(X))
    assert_perfect_accuracy(y, predictions)
