# Schedulers

A learning-rate scheduler wraps an `Optimizer` and updates its `lr` over time. Call `scheduler.step()` after every optimizer step (or after every epoch — your choice, just be consistent).

| Scheduler                                                                              | Schedule                                                              |
|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`LinearLR`](#simplegrad.schedulers.func_based.LinearLR)                               | linear interpolation between two values                               |
| [`ExponentialLR`](#simplegrad.schedulers.func_based.ExponentialLR)                     | $\mathrm{lr}_t = \mathrm{lr}_0 \cdot \gamma^{\,t}$                    |
| [`CosineAnnealingLR`](#simplegrad.schedulers.func_based.CosineAnnealingLR)             | cosine warm restarts                                                  |
| [`ReduceLROnPlateauLR`](#simplegrad.schedulers.metric_based.ReduceLROnPlateauLR)        | step-down when a metric stops improving                               |

## Training loop shape

```python
optimizer = sg.opt.Adam(model, lr=1e-3)
scheduler = sg.sch.LinearLR(optimizer, start_lr=1e-3, end_lr=1e-5, total_steps=10_000)

for step in range(10_000):
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

For `ReduceLROnPlateauLR`, pass the monitored metric:

```python
plateau = sg.sch.ReduceLROnPlateauLR(optimizer, factor=0.1, patience=5)

for epoch in range(num_epochs):
    train_one_epoch()
    val_loss = evaluate()
    plateau.step(val_loss)
```

---

## LinearLR

$$
\mathrm{lr}_t = \mathrm{lr}_{\text{start}} + r \cdot t,
\quad r = \frac{\mathrm{lr}_{\text{end}} - \mathrm{lr}_{\text{start}}}{T}
$$

```python
sg.sch.LinearLR(optimizer, start_lr=1e-3, end_lr=1e-5, total_steps=10_000)
```

::: simplegrad.schedulers.func_based.LinearLR

---

## ExponentialLR

$$
\mathrm{lr}_t = \mathrm{lr}_{\text{start}} \cdot \gamma^{\,t}
$$

```python
sg.sch.ExponentialLR(optimizer, start_lr=1e-3, gamma=0.99)
```

::: simplegrad.schedulers.func_based.ExponentialLR

---

## CosineAnnealingLR

$$
\mathrm{lr}_t = \mathrm{lr}_{\min} + \tfrac{1}{2}\,(\mathrm{lr}_{\max} - \mathrm{lr}_{\min})\,
\bigl(1 + \cos(\pi\, t_{\text{cur}} / T_i)\bigr)
$$

After each period $T_i$ expires, $t_{\text{cur}} \leftarrow 0$ and $T_i \leftarrow T_i \cdot T_{\text{mult}}$.

```python
sg.sch.CosineAnnealingLR(optimizer, T_0=1000, T_mult=2, lr_min=1e-5)
```

::: simplegrad.schedulers.func_based.CosineAnnealingLR

---

## ReduceLROnPlateauLR

When the monitored metric has not improved for `patience` steps, multiply `lr` by `factor`.

```python
sg.sch.ReduceLROnPlateauLR(
    optimizer,
    factor=0.1,           # multiply lr by 0.1 when triggered
    patience=5,           # 5 bad steps before triggering
    cooldown=2,           # ignore 2 steps after a reduction
    threshold=1e-4,
    threshold_mode="rel",
    maximize_metric=False,    # metric is a loss
    verbose=True,
)
```

::: simplegrad.schedulers.metric_based.ReduceLROnPlateauLR
