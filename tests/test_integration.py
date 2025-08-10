"""Integration tests for the model."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tests.conftest import assert_high_accuracy
from tests.typing import AnyLSSVC, AnyLSSVCClass


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "test_case,threshold",
    [
        ("train_test_split", 0.90),
        ("cross_validation", 0.94),
    ],
)
def test_iris_performance(
    standard_rbf_model: AnyLSSVC,
    to_numpy: Callable[[Any], np.ndarray],
    test_case: str,
    threshold: float,
) -> None:
    """Test model performance on Iris dataset with different evaluation strategies."""
    # Load and preprocess data
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if test_case == "train_test_split":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = standard_rbf_model
        model.fit(X_train, y_train)
        predictions = to_numpy(model.predict(X_test))

        assert_high_accuracy(y_test, predictions, threshold=threshold)

    elif test_case == "cross_validation":
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = standard_rbf_model.__class__(gamma=10.0, kernel="rbf", sigma=0.5)
            model.fit(X_train, y_train)
            predictions = to_numpy(model.predict(X_test))

            from sklearn.metrics import accuracy_score

            accuracies.append(accuracy_score(y_test, predictions))

        mean_accuracy = np.mean(accuracies)
        assert mean_accuracy >= threshold


@pytest.mark.slow
@pytest.mark.integration
def test_moons_sklearn_comparison(
    model_class: AnyLSSVCClass, to_numpy: Callable[[Any], np.ndarray]
) -> None:
    """Assert accuracy >= 98% comparable to `sklearn.svm.SVC` with RBF kernel."""
    # Generate non-linearly separable data
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # LSSVC model
    lssvc = model_class(gamma=50.0, kernel="rbf", sigma=0.5)
    lssvc.fit(X, y)
    lssvc_preds = to_numpy(lssvc.predict(X))

    # Scikit-learn SVC model for comparison
    svc = SVC(C=1.0, kernel="rbf", gamma="auto")
    svc.fit(X, y)
    svc_preds = svc.predict(X)

    from sklearn.metrics import accuracy_score

    lssvc_accuracy = accuracy_score(y, lssvc_preds)
    svc_accuracy = accuracy_score(y, svc_preds)

    assert lssvc_accuracy >= 0.98
    assert lssvc_accuracy >= svc_accuracy - 0.02  # Allow for small differences
