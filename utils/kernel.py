"""Kernel functions for LSSVM algorithms.

This module provides kernel functions for both numpy and PyTorch implementations
of the LSSVM algorithms.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import numpy as np
import torch
from scipy.spatial.distance import cdist


def _linear_np(x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    """Numpy linear kernel function."""
    return cast(np.ndarray, np.dot(x_i, x_j.T))


def _poly_np(x_i: np.ndarray, x_j: np.ndarray, d: int = 3) -> np.ndarray:
    """Numpy polynomial kernel function."""
    return cast(np.ndarray, (np.dot(x_i, x_j.T) + 1) ** d)


def _rbf_np(x_i: np.ndarray, x_j: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Numpy radial basis function (RBF) kernel."""
    return cast(np.ndarray, np.exp(-(cdist(x_i, x_j) ** 2) / sigma**2))


def get_kernel(
    name: str, **params: Any
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return the kernel function given the kernel name and parameters.

    Parameters
    ----------
    name : str
        The name of the kernel function. Must be one of 'linear', 'poly', or 'rbf'.
    **params : dict
        Kernel-specific parameters.

    Returns
    -------
    Callable
        The kernel function.

    Raises
    ------
    KeyError
        If the kernel name is not supported.
    """
    kernels: dict[str, Callable[..., np.ndarray]] = {
        "linear": _linear_np,
        "poly": _poly_np,
        "rbf": _rbf_np,
    }
    if name not in kernels:
        raise KeyError(
            f"Kernel '{name}' is not defined, try one in the list: "
            f"{list(kernels.keys())}."
        )

    kernel_func = kernels[name]

    if name == "poly":
        return partial(kernel_func, d=params.get("d", 3))
    if name == "rbf":
        return partial(kernel_func, sigma=params.get("sigma", 1.0))

    return kernel_func


def _linear_torch(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    """PyTorch linear kernel function."""
    return torch.mm(x_i, torch.t(x_j))


def _poly_torch(x_i: torch.Tensor, x_j: torch.Tensor, d: int = 3) -> torch.Tensor:
    """PyTorch polynomial kernel function."""
    return (torch.mm(x_i, torch.t(x_j)) + 1) ** d


def _rbf_torch(
    x_i: torch.Tensor, x_j: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    """PyTorch radial basis function (RBF) kernel."""
    return torch.exp(-(torch.cdist(x_i, x_j) ** 2) / sigma**2)


def torch_get_kernel(
    name: str, **params: Any
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the PyTorch kernel function given the kernel name and parameters.

    Parameters
    ----------
    name : str
        The name of the kernel function. Must be one of 'linear', 'poly', or 'rbf'.
    **params : dict
        Kernel-specific parameters.

    Returns
    -------
    Callable
        The PyTorch kernel function.

    Raises
    ------
    KeyError
        If the kernel name is not supported.
    """
    kernels: dict[str, Callable[..., torch.Tensor]] = {
        "linear": _linear_torch,
        "poly": _poly_torch,
        "rbf": _rbf_torch,
    }

    if name not in kernels:
        raise KeyError(
            f"Kernel '{name}' is not defined, try one in the list: "
            f"{list(kernels.keys())}."
        )

    kernel_func = kernels[name]

    if name == "poly":
        return partial(kernel_func, d=params.get("d", 3))
    if name == "rbf":
        return partial(kernel_func, sigma=params.get("sigma", 1.0))

    return kernel_func
