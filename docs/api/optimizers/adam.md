# Adam

Adaptive Moment Estimation with bias correction. The default optimizer for most modern deep learning workflows.

## Update rule

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^{\,t}} \\
\theta_{t+1} &= \theta_t - \mathrm{lr}\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

`m_t` is an exponential moving average of the gradient, `v_t` is an EMA of the squared gradient. The denominator $\sqrt{\hat v} + \epsilon$ rescales each parameter independently.

## Defaults

| Hyperparameter | Default | Typical range    |
|----------------|---------|------------------|
| `lr`           | `1e-3`  | `1e-4` … `3e-3`  |
| `beta_1`       | `0.9`   | `0.85` … `0.95`  |
| `beta_2`       | `0.999` | `0.99` … `0.9999`|
| `eps`          | `1e-8`  | `1e-8` … `1e-6`  |

## Example

```python
import simplegrad as sg

optimizer = sg.opt.Adam(model, lr=1e-3)

for step in range(num_steps):
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

## Parameter groups

Smaller `lr` on a pretrained backbone, full `lr` on the head:

```python
optimizer = sg.opt.Adam(
    lr=1e-3,                                  # default for the head
    param_groups=[
        {"params": model.backbone, "label": "bk", "lr": 1e-5},
        {"params": model.head,     "label": "hd"},
    ],
)
```

---

::: simplegrad.optimizers.adam.Adam
