# Transform Functions

Shape transformations that don't move data — the same buffer with a different view. Both ops are differentiable: gradients flow through unchanged.

---

## flatten

Flatten a contiguous range of dims into a single dim.

**Shape:**

| Input | Output |
|---|---|
| `(d0, ..., d_start, ..., d_end, ..., dN)` | `(d0, ..., d_start * ... * d_end, ..., dN)` |

```python
>>> x = sg.Tensor([[[1.0, 2.0], [3.0, 4.0]],
...                [[5.0, 6.0], [7.0, 8.0]]])
>>> x.shape
(2, 2, 2)
>>> sg.flatten(x, start_dim=1).shape    # collapse last 2 dims
(2, 4)
```

::: simplegrad.functions.tranform.flatten

---

## reshape

Same data, new shape. Total element count must match.

```python
>>> x = sg.Tensor([[1.0, 2.0, 3.0, 4.0]])
>>> sg.reshape(x, (2, 2)).values
array([[1., 2.],
       [3., 4.]], dtype=float32)
```

::: simplegrad.functions.tranform.reshape
