# Loss Functions

Loss functions reduce a tensor of per-sample errors to a single scalar that can be back-propagated.

| Loss       | Formula                                                  | Use case |
|------------|----------------------------------------------------------|----------|
| `ce_loss`  | $-\sum_i y_i \log\,\mathrm{softmax}(z)_i$                | Multi-class classification. |
| `mse_loss` | $\dfrac{1}{N}\sum_i (p_i - y_i)^2$                        | Regression. |

Both losses accept a `reduction` argument — `"mean"` (default), `"sum"`, or `None` (return per-element losses).

---

## ce_loss

$$
\mathcal{L}_{\text{CE}}(z, y) = -\sum_i y_i \log\!\left(\frac{e^{z_i}}{\sum_j e^{z_j}}\right)
$$

`ce_loss` includes a numerically stable log-softmax via the log-sum-exp trick — feed it raw logits, **not** probabilities. The target `y` is a probability distribution (one-hot for hard labels, soft labels for distillation).

```python
>>> logits = sg.Tensor([[2.0, 0.5, -1.0],
...                     [0.3, 1.5,  0.7]], label="logits")
>>> target = sg.Tensor([[1.0, 0.0, 0.0],          # class 0
...                     [0.0, 1.0, 0.0]],         # class 1
...                    label="target", comp_grad=False)
>>> loss = sg.ce_loss(logits, target)
>>> loss.values
0.394...
>>> loss.backward()
```

::: simplegrad.functions.losses.ce_loss

---

## mse_loss

$$
\mathcal{L}_{\text{MSE}}(p, y) = \frac{1}{N}\sum_i (p_i - y_i)^2
$$

```python
>>> preds = sg.Tensor([[1.0], [2.0], [3.0]], label="preds")
>>> truth = sg.Tensor([[1.5], [1.8], [2.7]], comp_grad=False)
>>> sg.mse_loss(preds, truth).values
0.127...
```

::: simplegrad.functions.losses.mse_loss
