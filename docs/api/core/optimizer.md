# Optimizer

`Optimizer` is the base class for all parameter-update rules. simplegrad ships [`SGD`](../optimizers/sgd.md) and [`Adam`](../optimizers/adam.md). Subclass `Optimizer` to add your own.

## Three lines per step

1. **`zero_grad()`** — reset `.grad` on every parameter (gradients accumulate by default).
2. **`loss.backward()`** — fills in `.grad` everywhere.
3. **`step()`** — read `.grad`, write a new value back into `param.values`.

Only `step()` must be implemented by subclasses.

## Single-group construction

For most use cases, pass a `Module` directly and the optimizer puts every parameter in one default group:

```python
optimizer = sg.opt.SGD(model, lr=0.01, momentum=0.9)
```

## Parameter groups

Different hyperparameters for different parts of the model:

```python
optimizer = sg.opt.Adam(
    lr=1e-3,                                                  # group default
    param_groups=[
        {"params": model.encoder, "label": "enc"},            # uses 1e-3
        {"params": model.decoder, "label": "dec", "lr": 1e-4},
    ],
)
```

`params` accepts either a `Module` (its `parameters()` are used) or a `dict[str, Tensor]`.

Update at runtime — for example, from a learning-rate scheduler:

```python
optimizer.set_param("lr", 5e-4)                 # update all groups
optimizer.set_param("lr", 1e-5, group="enc")    # one group only
```

## Subclassing

```python
class GradientClipSGD(Optimizer):
    def __init__(self, model, lr, max_norm: float = 1.0):
        super().__init__(lr, model, max_norm=max_norm)

    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            max_norm = group["max_norm"]
            for name, p in group["params"].items():
                grad = p.grad
                norm = (grad * grad).sum() ** 0.5
                if norm > max_norm:
                    grad = grad * (max_norm / norm)
                p.values -= lr * grad
```

---

::: simplegrad.core.optimizer.Optimizer
