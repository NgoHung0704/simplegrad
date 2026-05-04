# Module

`Module` is the base class for every neural network layer. Subclass it, set parameters as attributes, and implement `forward()` — that's the entire contract.

## Defining a layer

```python
import simplegrad as sg
from simplegrad.core import Tensor, Module


class AffineScale(Module):
    """y = scale * x + bias  (broadcast over the last dim)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.scale = sg.ones((num_features,), label="scale")
        self.bias  = sg.zeros((num_features,), label="bias")

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


layer = AffineScale(num_features=8)
y = layer(sg.ones((2, 8)))     # __call__ is wired to forward()
```

`scale` and `bias` are leaf Tensors stored as attributes. `Module._get_parameters` walks `self.__dict__` and finds them automatically — no manual registration.

## Composition rules

Any of the following are picked up by `parameters()` and `submodules()`:

| Attribute kind                                  | Treated as                |
|-------------------------------------------------|---------------------------|
| `Tensor`                                        | parameter                 |
| `Module`                                        | sub-module (recursed into)|
| `list` / `tuple` of `Tensor`s and/or `Module`s  | indexed children          |

This is what makes `Sequential`, `Conv2d`, `Embedding`, and your own custom modules just work with the optimizers.

## Train vs. eval mode

```python
model.set_eval_mode()    # disables Dropout, etc.
preds = model(x_test)
model.set_train_mode()
```

Custom layers that need similar behavior should check `self.eval_mode` in `forward`.

## Moving to GPU

```python
model.to_device("cuda:0")
```

Walks every parameter and replaces its underlying array with a CuPy array on the target device. The model itself is mutated in place.

## Inspecting

```python
model = sg.nn.Sequential(
    sg.nn.Linear(4, 16), sg.nn.ReLU(), sg.nn.Linear(16, 3),
)
print(model)
model.summary()
```

`summary()` prints a table of every parameter, its shape, and the parameter count.

---

::: simplegrad.core.module.Module
