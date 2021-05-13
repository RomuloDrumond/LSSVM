from numpy import dot, exp
from scipy.spatial.distance import cdist
import torch


def get_kernel(name, **params):
    """The method that returns the kernel function, given the 'kernel'
    parameter.
    """

    def linear(x_i, x_j):
        return dot(x_i, x_j.T)

    def poly(x_i, x_j, d=params.get('d', 3)):
        return (dot(x_i, x_j.T) + 1) ** d

    def rbf(x_i, x_j, sigma=params.get('sigma', 1)):
        return exp(-cdist(x_i, x_j) ** 2 / sigma ** 2)

    kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}

    if kernels.get(name) is None:
        raise KeyError(
            f"Kernel '{name}' is not defined, try one in the list: "
            f"{list(kernels.keys())}."
        )
    else:
        return kernels[name]


def torch_get_kernel(name, **params):
    """The method that returns the kernel function, given the 'kernel'
        parameter.
        """

    def linear(x_i, x_j):
        return torch.mm(x_i, torch.t(x_j))

    def poly(x_i, x_j, d=params.get('d', 3)):
        return (torch.mm(x_i, torch.t(x_j)) + 1) ** d

    def rbf(x_i, x_j, sigma=params.get('sigma', 1)):
        return torch.exp(-torch.cdist(x_i, x_j) ** 2 / sigma ** 2)

    kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}

    if kernels.get(name) is None:
        raise KeyError(
            f"Kernel '{name}' is not defined, try one in the list: "
            f"{list(kernels.keys())}."
        )
    else:
        return kernels[name]
