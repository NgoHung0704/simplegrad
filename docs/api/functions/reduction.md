# Reductions

Reductions collapse one or more dimensions of a tensor.

| Function           | Forward                                | Differentiable |
|--------------------|----------------------------------------|----------------|
| `sum(x, dim)`      | $\sum_i x_i$ along `dim`               | :material-check: |
| `mean(x, dim)`     | $\dfrac{1}{N}\sum_i x_i$ along `dim`   | :material-check: |
| `trace(x)`         | $\sum_i x_{ii}$ (diagonal sum)         | :material-check: |
| `argmax(x, dim)`   | index of $\max_i x_i$                  | :material-close: |
| `argmin(x, dim)`   | index of $\min_i x_i$                  | :material-close: |

`sum` and `mean` always return tensors with the reduced dimensions kept as size 1 (`keepdims=True`), so broadcasting against the original tensor stays straightforward.

---

## sum

$$
\mathrm{sum}(x)_{\dots} = \sum_i x_{\dots,\,i,\,\dots}
$$

```python
>>> x = sg.Tensor([[1.0, 2.0, 3.0],
...                [4.0, 5.0, 6.0]])
>>> sg.sum(x).values            # all elements
array([[21.]], dtype=float32)
>>> sg.sum(x, dim=0).values     # column sums
array([[5., 7., 9.]], dtype=float32)
```

::: simplegrad.functions.reduction.sum

---

## mean

$$
\mathrm{mean}(x) = \frac{1}{N}\sum_i x_i
$$

```python
>>> sg.mean(x, dim=1).values
array([[2.],
       [5.]], dtype=float32)
```

::: simplegrad.functions.reduction.mean

---

## trace

$$
\mathrm{tr}(X) = \sum_i X_{ii}
$$

```python
>>> sg.trace(sg.Tensor([[1.0, 2.0], [3.0, 4.0]])).values
array([[5.]], dtype=float32)
```

::: simplegrad.functions.reduction.trace

---

## argmax / argmin

```python
>>> preds = sg.Tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
>>> sg.argmax(preds, dim=-1).values
array([1, 0], dtype=int32)
```

`argmax` and `argmin` are **not differentiable** — the engine sets `comp_grad=False` on their output and refuses to backprop through them.

::: simplegrad.functions.reduction.argmax

::: simplegrad.functions.reduction.argmin
