"""Model import and export utilities for LSSVM algorithms.

This module provides functions to save and load LSSVM models in JSON format.
"""

import codecs
import json
from collections.abc import Callable
from typing import Any


def dump_model(
    model_dict: dict[str, Any],
    file_encoder: Callable[[Any], Any],
    filepath: str = "model",
) -> None:
    """Save a model dictionary to a JSON file.

    Parameters
    ----------
    model_dict : dict
        The model dictionary to save.
    file_encoder : Callable
        A function to encode non-JSON serializable objects.
    filepath : str, default="model"
        The filepath to save the model. The .json extension will be added if not present.
    """
    with open(f"{filepath.replace('.json', '')}.json", "w") as fp:
        json.dump(model_dict, fp, default=file_encoder)


def load_model(filepath: str = "model") -> dict[str, Any]:
    """Load a model dictionary from a JSON file.

    Parameters
    ----------
    filepath : str, default="model"
        The filepath to load the model from. The .json extension will be added if not present.

    Returns
    -------
    dict
        The loaded model dictionary.
    """
    helper_filepath = filepath if filepath.endswith(".json") else f"{filepath}.json"
    with codecs.open(helper_filepath, "r", encoding="utf-8") as fp:
        file_text = fp.read()
    model_json: dict[str, Any] = json.loads(file_text)
    return model_json
