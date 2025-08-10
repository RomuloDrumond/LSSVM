"""Data conversion utilities for LSSVM algorithms.

This module provides functions to convert various data types to JSON-serializable
formats for model serialization and deserialization.
"""

from typing import Any

import numpy as np
import torch


def torch_json_encoder(obj: Any) -> Any:
    """Encode PyTorch tensors to JSON-serializable format.

    Parameters
    ----------
    obj : Any
        The object to encode. If it's a PyTorch tensor, it will be converted
        to a numpy array and then to a list.

    Returns
    -------
    Any
        The JSON-serializable representation of the object.
    """
    if type(obj).__module__ == torch.__name__ and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def numpy_json_encoder(obj: Any) -> Any:
    """Encode numpy arrays to JSON-serializable format.

    Parameters
    ----------
    obj : Any
        The object to encode. If it's a numpy array, it will be converted
        to a list.

    Returns
    -------
    Any
        The JSON-serializable representation of the object.
    """
    if type(obj).__module__ == np.__name__ and isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
