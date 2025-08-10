"""GPU-accelerated Least Squares Support Vector Machine for classification.

This module implements the LSSVC algorithm using PyTorch for GPU computation.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from utils.conversion import torch_json_encoder
from utils.import_export import dump_model, load_model
from utils.kernel import torch_get_kernel


class LSSVC_GPU:
    """A class GPU variation that implements the Least Squares Support Vector Machine for classification tasks.

    It uses PyTorch pseudo-inverse function to solve the dual optimization
    problem with ordinary least squares. In multiclass classification problems
    the approach used is one-vs-all, so, a model is fit for each class while
    considering the others a set of the same class.

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
    alpha : torch.Tensor | None
        Tensor of size [1, n_support_vectors] if in binary
        classification and [n_classes, n_support_vectors] for
        multiclass problems. Each column is the optimum value of the dual
        variable for each model (using the one-vs-all approach we have
        n_classes == n_classifiers), it can be seen as the weight given
        to the support vectors (sv_x, sv_y). As usually there is no alpha == 0,
        we have n_support_vectors == n_train_samples.
    b : torch.Tensor | None
        Tensor of size [1] if in binary classification and [n_classes]
        for multiclass problems. The optimum value of the bias of the model.
    sv_x : torch.Tensor | None
        Tensor of size [n_support_vectors, n_features]. The set of the
        supporting vectors attributes, it has the size of the training data.
    sv_y : torch.Tensor | None
        Tensor of size [n_support_vectors, n]. The set of the supporting
        vectors labels. If the label is represented by an array of n elements,
        the sv_y attribute will have n columns.
    y_labels : torch.Tensor | None
        Tensor of size [n_classes, n]. The set of unique labels. If the
        label is represented by an array of n elements, the y_label attribute
        will have n columns.
    K : Callable, default=rbf()
        Kernel function.
    """

    alpha: torch.Tensor | None = None
    b: torch.Tensor | None = None
    sv_x: torch.Tensor | None = None
    sv_y: torch.Tensor | None = None
    y_labels: torch.Tensor | None = None
    K: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self, gamma: float = 1, kernel: str = "rbf", **kernel_params: Any
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.K = torch_get_kernel(kernel, **kernel_params)

    def _optimize_parameters(
        self, X: torch.Tensor, y_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimize the dual variables through the use of the kernel matrix pseudo-inverse.

        Parameters
        ----------
        X : torch.Tensor
            Training data tensor.
        y_values : torch.Tensor
            Training labels converted to -1/+1 format.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing (b, alpha) where b is the bias term and alpha
            are the dual variables.
        """
        sigma = torch.mm(y_values, torch.t(y_values)) * self.K(X, X)

        A = torch.cat(
            (
                # block matrix
                torch.cat(
                    (
                        torch.tensor(0, dtype=X.dtype, device=self.device).view(1, 1),
                        torch.t(y_values),
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        y_values,
                        sigma
                        + self.gamma**-1
                        * torch.eye(len(y_values), dtype=X.dtype, device=self.device),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        B = torch.tensor(
            [0] + [1] * len(y_values), dtype=X.dtype, device=self.device
        ).view(-1, 1)

        A_cross = torch.pinverse(A)

        solution = torch.mm(A_cross, B)
        b = solution[0]
        alpha = solution[1:].view(-1)

        return (b, alpha)

    def fit(self, X_np: np.ndarray, y_np: np.ndarray, verboses: int = 0) -> "LSSVC_GPU":
        """Fit the model given the set of X attribute vectors and y labels.

        Parameters
        ----------
        X_np : ndarray of shape (n_samples, n_features)
            Training data.
        y_np : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Training labels. If the label is represented by an array of n elements,
            the y parameter must have n columns.
        verboses : int, default=0
            Verbosity level.
        """
        # Convert pandas to numpy
        if hasattr(X_np, "values"):
            X_np = X_np.values
        if hasattr(y_np, "values"):
            y_np = y_np.values

        y_reshaped = y_np.reshape(-1, 1) if y_np.ndim == 1 else y_np

        # converting to tensors and passing to GPU
        X = torch.from_numpy(X_np).to(self.device)
        y = torch.from_numpy(y_reshaped).to(self.device)

        self.sv_x = X
        self.sv_y = y
        self.y_labels = torch.unique(y, dim=0)

        if len(self.y_labels) == 2:  # binary classification
            # converting to -1/+1
            y_values = torch.where(
                (y == self.y_labels[0]).all(dim=1),
                torch.tensor(-1, dtype=X.dtype, device=self.device),
                torch.tensor(+1, dtype=X.dtype, device=self.device),
            ).view(-1, 1)  # making it a column vector

            self.b, self.alpha = self._optimize_parameters(X, y_values)

        else:  # multiclass classification, one-vs-all approach
            n_classes = len(self.y_labels)
            self.b = torch.empty((n_classes,), dtype=X.dtype, device=self.device)
            self.alpha = torch.empty(
                (n_classes, len(y)), dtype=X.dtype, device=self.device
            )
            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all
                # other classes
                y_values = torch.where(
                    (y == self.y_labels[i]).all(dim=1),
                    torch.tensor(+1, dtype=X.dtype, device=self.device),
                    torch.tensor(-1, dtype=X.dtype, device=self.device),
                ).view(-1, 1)  # making it a column vector

                b_i, alpha_i = self._optimize_parameters(X, y_values)
                if self.b is not None and self.alpha is not None:
                    self.b[i] = b_i
                    self.alpha[i] = alpha_i

        return self

    def predict(self, X_np: np.ndarray) -> np.ndarray | torch.Tensor:
        """Predict the labels of data X given a trained model.

        Parameters
        ----------
        X_np : ndarray of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
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
        if hasattr(X_np, "values"):
            X_np = X_np.values

        X_reshaped = X_np.reshape(1, -1) if X_np.ndim == 1 else X_np

        X = torch.from_numpy(X_reshaped).to(self.device)
        KxX = self.K(self.sv_x, X)

        if len(self.y_labels) == 2:  # binary classification
            y_values = torch.where(
                (self.sv_y == self.y_labels[0]).all(dim=1),
                torch.tensor(-1, dtype=X.dtype, device=self.device),
                torch.tensor(+1, dtype=X.dtype, device=self.device),
            )

            y = torch.sign(torch.mm((self.alpha * y_values).view(1, -1), KxX) + self.b)

            y_pred_labels = torch.where(
                y.flatten() == -1, self.y_labels[0], self.y_labels[1]
            )

        else:  # multiclass classification, ONE-VS-ALL APPROACH
            y = torch.empty(
                (len(self.y_labels), len(X)), dtype=X.dtype, device=self.device
            )
            for i in range(len(self.y_labels)):
                y_values = torch.where(
                    (self.sv_y == self.y_labels[i]).all(dim=1),
                    torch.tensor(+1, dtype=X.dtype, device=self.device),
                    torch.tensor(-1, dtype=X.dtype, device=self.device),
                )

                y[i] = torch.mm((self.alpha[i] * y_values).view(1, -1), KxX) + self.b[i]

            predictions = torch.argmax(y, dim=0)
            y_pred_labels = torch.stack([self.y_labels[i] for i in predictions])

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
            model_dict=model_json, file_encoder=torch_json_encoder, filepath=filepath
        )

    @classmethod
    def load(cls, filepath: str, only_hyperparams: bool = False) -> "LSSVC_GPU":
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
        LSSVC_GPU
            The loaded LSSVC_GPU model.

        Raises
        ------
        Exception
            If the model type doesn't match 'LSSVC'.
        """
        model_json = load_model(filepath=filepath)

        if model_json["type"] != "LSSVC":
            raise Exception(f"Model type '{model_json['type']}' doesn't match 'LSSVC'")

        lssvc = LSSVC_GPU(
            gamma=model_json["hyperparameters"]["gamma"],
            kernel=model_json["hyperparameters"]["kernel"],
            **model_json["hyperparameters"]["kernel_params"],
        )

        if (model_json.get("parameters") is not None) and (not only_hyperparams):
            params = model_json["parameters"]
            device = lssvc.device

            lssvc.alpha = torch.Tensor(params["alpha"]).double().to(device)
            lssvc.b = torch.Tensor(params["b"]).double().to(device)
            lssvc.sv_x = torch.Tensor(params["sv_x"]).double().to(device)
            lssvc.sv_y = torch.Tensor(params["sv_y"]).double().to(device)
            lssvc.y_labels = torch.Tensor(params["y_labels"]).double().to(device)

        return lssvc
