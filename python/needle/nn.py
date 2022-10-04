"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, device=device, dtype=dtype, requires_grad=True
        ))  # [Cin, Cout]
        if bias:
            self.bias = Parameter(init.kaiming_uniform(
                out_features, 1, device=device, dtype=dtype, requires_grad=True
            ).reshape([1, out_features]))  # [1, Cout]
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # X: [N, Cin]
        y = X @ self.weight  # [N, Cout]
        if self.bias is not None:
            y = y + self.bias.broadcast_to(y.shape)
        return y



class Flatten(Module):
    def forward(self, X):
        dim = 1
        for i in range(1, len(X.shape)):
            dim *= X.shape[i]
        return X.reshape([X.shape[0], dim])


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x
        

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        n_batch = logits.shape[0]
        n_cls = logits.shape[1]
        return (ops.logsumexp(logits, axes=(1,)) - (logits * init.one_hot(n_cls, y)).sum(axes=(1,))).sum() / n_batch


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)


    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_size = x.shape[0]

            mean = x.sum(axes=(0,)) / batch_size
            self.running_mean.cached_data = ((1 - self.momentum) * self.running_mean + self.momentum * mean).numpy()

            mean = mean.reshape([1, self.dim]).broadcast_to(x.shape)
            var = ((x - mean) ** 2).sum(axes=(0,)) / batch_size
            self.running_var.cached_data = ((1 - self.momentum) * self.running_var + self.momentum * var).numpy()

            var = var.reshape([1, self.dim]).broadcast_to(x.shape)
        else:
            mean = self.running_mean.reshape([1, self.dim]).broadcast_to(x.shape)
            var = self.running_var.reshape([1, self.dim]).broadcast_to(x.shape)
        
        weight = self.weight.reshape([1, self.dim]).broadcast_to(x.shape)
        bias = self.bias.reshape([1, self.dim]).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))


    def forward(self, x: Tensor) -> Tensor:
        mean = x.sum(axes=(1,)) / self.dim
        mean = mean.reshape([x.shape[0], 1]).broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum(axes=(1,)) / self.dim
        var = var.reshape([x.shape[0], 1]).broadcast_to(x.shape)
        weight = self.weight.reshape([1, self.dim]).broadcast_to(x.shape)
        bias = self.bias.reshape([1, self.dim]).broadcast_to(x.shape)
        return (x - mean) / ((var + self.eps) ** 0.5) * weight + bias


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            keep = init.randb(*x.shape, p=1 - self.p)
            x = x * keep / (1 - self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)



