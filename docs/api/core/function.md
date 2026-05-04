# Function & Context

`Function` is the unit at which the autograd engine knows how to compute a gradient. Every differentiable operation in simplegrad — `_Add`, `_Mul`, `_Matmul`, `_Relu`, `_CELoss`, ... — subclasses `Function`.

If you only **use** simplegrad you typically don't touch this class directly. If you **extend** it with a new differentiable op, this is the API you implement against.

## Skeleton

```python
import numpy as np
from simplegrad.core import Tensor, Function, Context


class _Square(Function):
    oper = "square"  # short label shown on the graph node

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.x_values = x.values         # save anything we need for backward
        return x.values ** 2            # return a numpy array

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return 2 * ctx.x_values * grad_output


def square(x: Tensor) -> Tensor:
    return _Square.apply(x)
```

The base class handles graph wiring, gradient accumulation, broadcast reduction, and lazy/eager dispatch. You only write the math.

## Key rules

- `forward` returns a backend ndarray and saves intermediates on `ctx`. It must **not** write to `.grad`.
- `backward` returns one gradient array per `Tensor` input (or a tuple). It must **not** accumulate — the base class does that.
- Override `output_shape(*inputs)` for ops where the output shape differs from the first input's shape (matmul, sum, conv...).
- Use `xp = ctx.backend` instead of `np` so the op works on both CPU (numpy) and GPU (cupy).
- Set `differentiable = False` for ops with no gradient (e.g. `argmax`).

## Multi-input

For ops with more than one input, `backward` returns a tuple of gradients in the **same order** as the input tensors:

```python
class _Mul(Function):
    oper = "*"

    @staticmethod
    def output_shape(x, y):
        return np.broadcast_shapes(x.shape, y.shape)

    @staticmethod
    def forward(ctx, x, y):
        ctx.x_values = x.values
        ctx.y_values = y.values
        return x.values * y.values

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.y_values, grad_output * ctx.x_values
```

Return `None` for inputs that have no gradient.

---

::: simplegrad.core.autograd.Function

::: simplegrad.core.autograd.Context
