# Scheduler

`Scheduler` is the base class for learning-rate schedules. A scheduler wraps an `Optimizer` and changes its `lr` over time, typically to anneal it down so that early steps explore broadly and late steps fine-tune.

The schedules shipped with simplegrad are documented [here](../schedulers/linear.md).

## Training-loop shape

```python
scheduler = sg.sch.LinearLR(optimizer, start_lr=1e-3, end_lr=1e-5, total_steps=10_000)

for step in range(10_000):
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()        # advance the schedule
```

## Writing your own scheduler

Subclass `Scheduler`, implement `step()`, and call `optimizer.set_param("lr", new_lr)` to push a new learning rate. Increment `self.steps`.

```python
class WarmupLR(Scheduler):
    def __init__(self, optimizer, warmup_steps, base_lr):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr

    def step(self):
        progress = min(1.0, (self.steps + 1) / self.warmup_steps)
        self.optimizer.set_param("lr", self.base_lr * progress)
        self.steps += 1
```

---

::: simplegrad.core.scheduler.Scheduler
