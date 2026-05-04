# Dropout

Randomly zeros elements of the input during training as a regularizer. In evaluation mode (`model.set_eval_mode()`) dropout is automatically disabled.

$$
\mathrm{Dropout}(x)_i =
\begin{cases}
0 & \text{with probability } p \\
x_i & \text{with probability } 1 - p
\end{cases}
$$

## Example

```python
import simplegrad as sg

model = sg.nn.Sequential(
    sg.nn.Linear(64, 128),
    sg.nn.ReLU(),
    sg.nn.Dropout(p=0.3),
    sg.nn.Linear(128, 10),
)

# Training: dropout is active
model.set_train_mode()
out_train = model(x)

# Evaluation: dropout becomes a no-op
model.set_eval_mode()
out_eval = model(x)
```

A fresh mask is sampled on every forward pass, so two calls with the same input produce different outputs in training mode.

!!! note
    This implementation zeros elements but does **not** rescale the survivors by $1/(1-p)$. If you need standard inverse-scaling dropout, write a small custom layer.

---

::: simplegrad.nn.dropout.Dropout
