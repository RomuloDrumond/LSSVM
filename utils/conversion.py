import torch
import numpy as np

def torch_json_encoder(obj):
    if type(obj).__module__ == torch.__name__:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f"""Unable to  "jsonify" object of type :', {type(obj)}""")


def numpy_json_encoder(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f"""Unable to  "jsonify" object of type :', {type(obj)}""")

