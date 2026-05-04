# Compound Ops

A "compound op" is a function that produces its output by composing other simplegrad operations rather than calling NumPy directly. Examples shipped with the framework: `softmax`, `mse_loss`.

Compound ops add nothing mathematically — they are regular Python functions — but they get **two visualization benefits**:

1. The graph renderer wraps the entire compound op in a labelled black-border rectangle.
2. Each call gets its own rectangle, so two calls to `softmax` produce two separate boxes.

## Decorate a function

```python
from simplegrad.core import compound_op
from simplegrad.functions import exp, sum

@compound_op
def softmax(x, dim=None):
    """Softmax along the given dimension."""
    exps = exp(x)
    return exps / sum(exps, dim)
```

When you visualize a graph that uses `softmax`, the `exp`, `sum`, and division nodes appear inside a `softmax` rectangle.

## Group ops manually

If you want to group ops at a specific spot without wrapping a whole function, use `graph_group`:

```python
from simplegrad.core import graph_group

with graph_group("attention_head"):
    q = w_q(x); k = w_k(x); v = w_v(x)
    scores = sg.softmax((q @ k.T) / sqrt_dk, dim=-1)
    out = scores @ v
```

Every Tensor created inside the `with` block — including ones returned by inner ops — receives a shared group identifier and is rendered together.

---

::: simplegrad.core.compound_ops.compound_op

::: simplegrad.core.compound_ops.graph_group
