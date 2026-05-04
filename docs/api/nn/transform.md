# Flatten

Collapse a contiguous range of dimensions into a single dimension. By default `Flatten()` keeps the batch dim (`start_dim=1`) and merges everything after it — exactly what you want between a conv stack and a linear head.

## Example

```python
import simplegrad as sg

model = sg.nn.Sequential(
    sg.nn.Conv2d(1, 16, kernel_size=3),
    sg.nn.ReLU(),
    sg.nn.MaxPool2d(2, 2),
    sg.nn.Flatten(),                          # (B, 16, 13, 13) -> (B, 2704)
    sg.nn.Linear(16 * 13 * 13, 10),
)
```

Flatten any contiguous range:

```python
flat = sg.nn.Flatten(start_dim=2, end_dim=-1)   # keeps (B, C) prefix
```

---

::: simplegrad.nn.transform.Flatten
