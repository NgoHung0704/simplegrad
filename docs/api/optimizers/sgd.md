# SGD

Stochastic Gradient Descent with optional momentum and dampening.

## Update rule

Without momentum:

$$
\theta_{t+1} = \theta_t - \mathrm{lr} \cdot g_t
$$

With momentum $\beta$ and dampening $\tau$:

$$
\begin{aligned}
v_t &= \beta\, v_{t-1} - \mathrm{lr}\,(1 - \tau)\, g_t \\
\theta_{t+1} &= \theta_t + v_t
\end{aligned}
$$

where $g_t = \nabla_\theta L$ is the gradient at step $t$.

## Example

```python
import simplegrad as sg

model = sg.nn.Sequential(sg.nn.Linear(4, 4), sg.nn.ReLU(), sg.nn.Linear(4, 2))
optimizer = sg.opt.SGD(model, lr=0.01, momentum=0.9)

for step in range(1000):
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

## Parameter groups

Different `lr` per group:

```python
optimizer = sg.opt.SGD(
    lr=1e-2,                                # default
    momentum=0.9,
    param_groups=[
        {"params": model.encoder, "label": "enc"},
        {"params": model.decoder, "label": "dec", "lr": 1e-3, "momentum": 0.5},
    ],
)
optimizer.set_param("lr", 5e-3, group="enc")  # change at runtime
```

---

::: simplegrad.optimizers.sgd.SGD
