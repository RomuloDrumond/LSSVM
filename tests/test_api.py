"""Tests for the model's public API."""

import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tests.typing import AnyLSSVCClass


def test_predict_output_shape(
    blobs_data: tuple[np.ndarray, np.ndarray],
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
) -> None:
    """Assert that .predict() output shape matches number of input samples."""
    X, y = blobs_data
    model = model_class()
    model.fit(X, y)

    # Test with single sample
    X_single = X[:1]
    predictions_single = to_numpy(model.predict(X_single))
    assert predictions_single.shape[0] == 1

    # Test with multiple samples
    X_multi = X[:10]
    predictions_multi = to_numpy(model.predict(X_multi))
    assert predictions_multi.shape[0] == 10

    # Test with all samples
    predictions_all = to_numpy(model.predict(X))
    assert predictions_all.shape[0] == X.shape[0]


def test_fit_returns_self(
    blobs_data: tuple[np.ndarray, np.ndarray], model_class: AnyLSSVCClass
) -> None:
    """Assert that .fit(X, y) returns self."""
    X, y = blobs_data
    model = model_class()
    result = model.fit(X, y)
    assert result is model


def test_input_immutability(
    blobs_data: tuple[np.ndarray, np.ndarray], model_class: AnyLSSVCClass
) -> None:
    """Assert that .fit(X, y) does not mutate input arrays X and y."""
    X, y = blobs_data
    X_original = X.copy()
    y_original = y.copy()

    model = model_class()
    model.fit(X, y)

    assert np.array_equal(X, X_original)
    assert np.array_equal(y, y_original)


def test_pickle_compatibility(
    blobs_data: tuple[np.ndarray, np.ndarray],
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
) -> None:
    """Assert model can be pickled/unpickled with identical predictions."""
    X, y = blobs_data
    model = model_class()
    model.fit(X, y)

    predictions_before = to_numpy(model.predict(X))

    # Pickle and unpickle the model
    pickled_model = pickle.dumps(model)
    unpickled_model = pickle.loads(pickled_model)

    predictions_after = to_numpy(unpickled_model.predict(X))

    assert np.array_equal(predictions_before, predictions_after)


def test_pandas_compatibility(
    blobs_data: tuple[np.ndarray, np.ndarray],
    model_class: AnyLSSVCClass,
    to_numpy: Callable[[Any], np.ndarray],
) -> None:
    """Assert model works with pandas DataFrame and Series."""
    X, y = blobs_data
    X_df = pd.DataFrame(X, columns=["feature1", "feature2"])
    y_s = pd.Series(y, name="target")

    model = model_class()
    model.fit(X_df, y_s)

    predictions = to_numpy(model.predict(X_df))

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == X_df.shape[0]
