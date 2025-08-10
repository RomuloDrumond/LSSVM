"""Least Squares Support Vector Machine for classification tasks.

This module implements the LSSVC algorithm using numpy for CPU computation.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy import dot

from utils.conversion import numpy_json_encoder
from utils.import_export import dump_model, load_model
from utils.kernel import get_kernel


class LSSVC:
    """A class that implements the Least Squares Support Vector Machine for classification tasks.

    It uses Numpy pseudo-inverse function to solve the dual optimization
    problem with ordinary least squares. In multiclass classification
    problems the approach used is one-vs-all, so, a model is fit for each
    class while considering the others a single set of the same class.

    Parameters
    ----------
    gamma : float, default=1.0
        Constant that control the regularization of the model, it may vary
        in the set (0, +infinity). The closer gamma is to zero, the more
        regularized the model will be.
    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        The kernel used for the model, if set to 'linear' the model
        will not take advantage of the kernel trick, and the LSSVC maybe only
        useful for linearly separable problems.
    kernel_params : dict, default=depends on 'kernel' choice
        If kernel = 'linear', these parameters are ignored. If kernel = 'poly',
        'd' is accepted to set the degree of the polynomial, with default = 3.
        If kernel = 'rbf', 'sigma' is accepted to set the radius of the
        gaussian function, with default = 1.

    Attributes
    ----------
    alpha : Optional[np.ndarray]
        ndarray of shape (1, n_support_vectors) if in binary
        classification and (n_classes, n_support_vectors) for
        multiclass problems. Each column is the optimum value of the dual
        variable for each model (using the one-vs-all approach we have
        n_classes == n_classifiers), it can be seen as the weight given
        to the support vectors (sv_x, sv_y). As usually there is no alpha == 0,
        we have n_support_vectors == n_train_samples.
    b : Optional[np.ndarray]
        ndarray of shape (1,) if in binary classification and (n_classes,)
        for multiclass problems. The optimum value of the bias of the model.
    sv_x : Optional[np.ndarray]
        ndarray of shape (n_support_vectors, n_features). The set of the
        supporting vectors attributes, it has the shape of the training data.
    sv_y : Optional[np.ndarray]
        ndarray of shape (n_support_vectors, n). The set of the supporting
        vectors labels. If the label is represented by an array of n elements,
        the sv_y attribute will have n columns.
    y_labels : Optional[np.ndarray]
        ndarray of shape (n_classes, n). The set of unique labels. If the
        label is represented by an array of n elements, the y_label attribute
        will have n columns.
    K : function, default=rbf()
        Kernel function.
    """

    alpha: np.ndarray | None = None
    b: np.ndarray | np.float64 | None = None
    sv_x: np.ndarray | None = None
    sv_y: np.ndarray | None = None
    y_labels: np.ndarray | None = None
    K: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(
        self, gamma: float = 1, kernel: str = "rbf", **kernel_params: Any
    ) -> None:
        # Hyperparameters
        self.gamma = gamma
        self.kernel_ = kernel
        self.kernel_params = kernel_params

        if self.gamma <= 0:
            raise ValueError("Gamma must be > 0")

        if self.kernel_ == "rbf" and self.kernel_params.get("sigma", 1) <= 0:
            raise ValueError("Sigma must be > 0 for RBF kernel")

        if self.kernel_ == "poly" and self.kernel_params.get("d", 3) <= 0:
            raise ValueError("Degree 'd' must be > 0 for polynomial kernel")

        # Model parameters are initialized as class attributes
        self.K = get_kernel(kernel, **kernel_params)

    def _optimize_parameters(
        self, X: np.ndarray, y_values: np.ndarray
    ) -> tuple[np.float64, np.ndarray]:
        """Optimize the dual variables through the use of the kernel matrix pseudo-inverse.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y_values : ndarray of shape (n_samples, 1)
            Training labels converted to -1/+1 format.

        Returns
        -------
        tuple[np.float64, np.ndarray]
            A tuple containing (b, alpha) where b is the bias term and alpha
            are the dual variables.
        """
        sigma = np.multiply(y_values * y_values.T, self.K(X, X))

        A = np.block(
            [
                [0, y_values.T],
                [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))],
            ]
        )
        B = np.array([0] + [1] * len(y_values))

        A_cross = np.linalg.pinv(A)

        solution = dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:].flatten()

        return (b, alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSSVC":
        """Fit the model given the set of X attribute vectors and y labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Training labels.
        """
        # Convert pandas to numpy
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # Input validation
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)

        self.sv_x = X
        self.sv_y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)

        if len(self.y_labels) == 2:  # binary classification
            # converting to -1/+1
            y_values = np.where((y_reshaped == self.y_labels[0]).all(axis=1), -1, +1)[
                :, np.newaxis
            ]  # making it a column vector

            self.b, self.alpha = self._optimize_parameters(X, y_values)

        else:  # multiclass classification, one-vs-all approach
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))

            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all
                # other classes
                y_values = np.where(
                    (y_reshaped == self.y_labels[i]).all(axis=1), +1, -1
                )[:, np.newaxis]

                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels of data X given a trained model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        ndarray
            Predicted labels.

        Raises
        ------
        Exception
            If the model has not been fitted yet.
        """
        if (
            self.alpha is None
            or self.sv_x is None
            or self.sv_y is None
            or self.y_labels is None
            or self.b is None
        ):
            raise Exception(
                "The model doesn't see to be fitted, try running .fit() method first"
            )

        # Convert pandas to numpy
        if hasattr(X, "values"):
            X = X.values

        X_reshaped = X.reshape(1, -1) if X.ndim == 1 else X

        KxX = self.K(self.sv_x, X_reshaped)

        if len(self.y_labels) == 2:  # binary classification
            y_values = np.where((self.sv_y == self.y_labels[0]).all(axis=1), -1, 1)

            assert isinstance(self.b, np.float64 | float), (
                "b should be scalar for binary classification"
            )
            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)

            y_pred_labels = np.where(y == -1, self.y_labels[0], self.y_labels[1])

        else:  # multiclass classification, one-vs-all approach
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                # converting to +1 for the desired class and -1 for all
                # other classes
                y_values = np.where(
                    (self.sv_y == self.y_labels[i]).all(axis=1), +1, -1
                )[:, np.newaxis]
                assert isinstance(self.b, np.ndarray), (
                    "b should be array for multiclass"
                )
                y[i] = (
                    dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
                )

            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])

        return y_pred_labels

    def dump(self, filepath: str = "model", only_hyperparams: bool = False) -> None:
        """Save the model in a JSON format.

        Parameters
        ----------
        filepath : str, default='model'
            File path to save the model's json.
        only_hyperparams : bool, default=False
            To either save only the model's hyperparameters or not, it
            only affects trained/fitted models.
        """
        model_json: dict[str, Any] = {
            "type": "LSSVC",
            "hyperparameters": {
                "gamma": self.gamma,
                "kernel": self.kernel_,
                "kernel_params": self.kernel_params,
            },
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json["parameters"] = {
                "alpha": self.alpha,
                "b": self.b,
                "sv_x": self.sv_x,
                "sv_y": self.sv_y,
                "y_labels": self.y_labels,
            }

        dump_model(
            model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath
        )

    @classmethod
    def load(cls, filepath: str, only_hyperparams: bool = False) -> "LSSVC":
        """Load a model from a .json file.

        Parameters
        ----------
        filepath : str
            The model's .json file path.
        only_hyperparams : bool, default=False
            To either load only the model's hyperparameters or not, it
            only has effects when the dump of the model as done with the
            model's parameters.

        Returns
        -------
        LSSVC
            The loaded LSSVC model.

        Raises
        ------
        Exception
            If the model type doesn't match 'LSSVC'.
        """
        model_json = load_model(filepath=filepath)

        if model_json["type"] != "LSSVC":
            raise Exception(f"Model type '{model_json['type']}' doesn't match 'LSSVC'")

        lssvc = LSSVC(
            gamma=model_json["hyperparameters"]["gamma"],
            kernel=model_json["hyperparameters"]["kernel"],
            **model_json["hyperparameters"]["kernel_params"],
        )

        if (model_json.get("parameters") is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json["parameters"]["alpha"])
            lssvc.b = np.array(model_json["parameters"]["b"])
            lssvc.sv_x = np.array(model_json["parameters"]["sv_x"])
            lssvc.sv_y = np.array(model_json["parameters"]["sv_y"])
            lssvc.y_labels = np.array(model_json["parameters"]["y_labels"])

        return lssvc
