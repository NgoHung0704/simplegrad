# Tensor & Autograd

The `Tensor` class is the central type in simplegrad. It wraps a NumPy (or CuPy) array and records every operation applied to it, so that gradients can be computed automatically by calling `.backward()`.

```python
import simplegrad as sg

x = sg.Tensor([[1.0, 2.0], [3.0, 4.0]], label="x")
w = sg.Tensor([[0.5], [-0.5]], label="w")

loss = sg.mean(x @ w)
loss.backward()

print(x.grad)   # gradient of loss w.r.t. x
print(w.grad)   # gradient of loss w.r.t. w
```

Operations on a `Tensor` build a dynamic graph; `.backward()` walks the graph in reverse and accumulates the result into `.grad` of every leaf with `comp_grad=True`.

## Tensor attributes

| Attribute   | Meaning |
|-------------|---------|
| `values`    | Backing NumPy / CuPy array (or `None` if lazy and not realized). |
| `shape`     | Tuple shape, available even before realization. |
| `dtype`     | String dtype like `"float32"`. |
| `device`    | `"cpu"` or `"cuda:N"`. |
| `comp_grad` | Whether gradients are tracked. |
| `grad`      | Accumulated gradient after `.backward()`. `None` until then. |
| `is_leaf`   | True if produced by user code, False if produced by an op. |

::: simplegrad.core.autograd.Tensor

---

## Execution mode

simplegrad supports two execution modes — **eager** (default) and **lazy**. Use the context managers below to switch.

### `no_grad`

Disable gradient tracking inside a block. Useful for evaluation:

```python
model.set_eval_mode()
with sg.no_grad():
    preds = model(x_test)
model.set_train_mode()
```

::: simplegrad.core.autograd.no_grad

### `lazy`

Build a graph without executing it; call `.realize()` (or `.backward()`) to run all forwards in one pass:

```python
with sg.lazy():
    out = model(x)          # graph nodes only — out.values is None

out.realize()               # numpy now runs in topological order
```

::: simplegrad.core.autograd.lazy

### `mode`

Persistently switch the global execution mode. Prefer the `lazy()` context manager for scoped control:

```python
sg.mode("lazy")     # all subsequent ops are lazy
sg.mode("eager")    # back to default
```

::: simplegrad.core.autograd.mode

::: simplegrad.core.autograd.is_lazy
