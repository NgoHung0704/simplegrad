# Factory Functions

Convenience constructors for creating common tensor initializations. They all return regular `Tensor`s — the difference from `Tensor(values=...)` is that you specify a shape and the values are filled in for you.

## At a glance

```python
import simplegrad as sg

a = sg.zeros((2, 3))                                # all zeros
b = sg.ones((2, 3))                                 # all ones
c = sg.full((2, 3), fill_value=7.5)                 # constant fill
d = sg.normal((2, 3), mu=0, sigma=1)                # N(0, 1)
e = sg.uniform((2, 3), low=-0.1, high=0.1)          # U(-0.1, 0.1)
```

Every factory accepts the same set of common keyword arguments:

| Argument    | Default          | Meaning                                  |
|-------------|------------------|------------------------------------------|
| `shape`     | required         | Output shape tuple.                       |
| `dtype`     | `"float32"`      | String dtype key.                         |
| `comp_grad` | global flag      | Track gradients on the new tensor.        |
| `label`     | `None`           | Optional name shown in the graph.         |
| `device`    | global default   | `"cpu"` or `"cuda:N"`.                    |

## Initialization recipes

```python
import math

# Xavier / Glorot uniform: fan-in based bound
fan_in, fan_out = 64, 32
bound = math.sqrt(6 / (fan_in + fan_out))
W = sg.uniform((fan_in, fan_out), low=-bound, high=bound, label="W")

# Kaiming / He normal for ReLU
sigma = math.sqrt(2 / fan_in)
W = sg.normal((fan_in, fan_out), mu=0, sigma=sigma, label="W")

# Bias initialized to zero
b = sg.zeros((fan_out,), label="b")
```

---

::: simplegrad.core.factory.zeros

::: simplegrad.core.factory.ones

::: simplegrad.core.factory.normal

::: simplegrad.core.factory.uniform

::: simplegrad.core.factory.full
