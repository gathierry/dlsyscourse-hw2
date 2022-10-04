"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * self.scalar * (a ** (self.scalar - 1)),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, - out_grad * a / (b ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if self.axes is None:
            self.axes = (-1, -2)

    def compute(self, a):
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad.transpose(self.axes[::-1]),)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return (out_grad.reshape(a.shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        i = len(out_grad.shape) - 1
        j = len(a.shape) - 1
        axes = []
        while i >= 0:
            if j >= 0:
                if a.shape[j] != out_grad.shape[i]:
                    axes.append(i)
            else:
                axes.append(i)
            i -= 1
            j -= 1
        axes = tuple(axes)
        return (out_grad.sum(axes).reshape(a.shape),)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        """
        a: [d0, d1, ..., dk]
        out_grad: [d0, d2, ..., dk]
        """
        grad = out_grad
        if self.axes is not None:
            shape = list(a.shape)
            for axis in self.axes:
                shape[axis] = 1
            grad = grad.reshape(shape)
        return (grad.broadcast_to(a.shape),)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        """
        a: [*, m, n] or [m, n]
        b: [*, n, k] or [n, k]
        out_grad: [*, m, k]
        """
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad
        axes = tuple(i for i in range(len(grad_a.shape) - len(a.shape)))
        if len(axes):
            grad_a = grad_a.sum(axes)
        axes = tuple(i for i in range(len(grad_b.shape) - len(b.shape)))
        if len(axes):
            grad_b = grad_b.sum(axes)
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return (-out_grad,)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (Tensor(array_api.ones(a.shape), requires_grad=False) / a * out_grad,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * exp(a),)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (
            Tensor(
                array_api.where(a.realize_cached_data() > 0, 1, 0)
            ) * out_grad,
        )


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def _exp_sum(self, Z):
        max_z = array_api.max(Z, self.axes)
        if self.axes is None:
            reshape_max_z = max_z
        else:
            shape = list(Z.shape)
            for i in self.axes:
                shape[i] = 1
            reshape_max_z = max_z.reshape(shape)
        exp_z = array_api.exp(
            Z - array_api.broadcast_to(reshape_max_z, Z.shape)
        )
        sum_z = array_api.sum(exp_z, self.axes)
        return sum_z, exp_z, max_z
        

    def compute(self, Z):
        sum_z, exp_z, max_z = self._exp_sum(Z)
        log_z = array_api.log(sum_z)
        return log_z + max_z

    def gradient(self, out_grad, node):
        Z = node.inputs[0].numpy()
        sum_z, exp_z, _ = self._exp_sum(Z)
        if self.axes is None:
            sum_z = array_api.broadcast_to(sum_z, Z.shape)
            out_grad = out_grad.broadcast_to(Z.shape)
        else:
            shape = list(Z.shape)
            for i in self.axes:
                shape[i] = 1
            sum_z = array_api.broadcast_to(sum_z.reshape(shape), Z.shape)
            out_grad = out_grad.reshape(shape).broadcast_to(Z.shape)
        grad = Tensor(1 / sum_z * exp_z)
        return (grad * out_grad,)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
