# MaxPool2d

A `Module` wrapper around [`functions.max_pool2d`](../functions/pooling.md). Slides a window over the spatial dims and keeps the maximum of each window.

**Shape:**

| Tensor | Shape                                              |
|--------|----------------------------------------------------|
| Input  | $(N, C, H, W)$ or $(C, H, W)$                       |
| Output | $(N, C, H_{\text{out}}, W_{\text{out}})$            |

$$
H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - kH}{s_H} \right\rfloor + 1, \quad
W_{\text{out}} = \left\lfloor \frac{W + 2 p_W - kW}{s_W} \right\rfloor + 1
$$

## Example

```python
import simplegrad as sg

pool = sg.nn.MaxPool2d(kernel_size=2, stride=2)
x = sg.normal((1, 16, 28, 28))
y = pool(x)
print(y.shape)   # (1, 16, 14, 14)
```

The classic place to put it is between conv blocks:

```python
sg.nn.Sequential(
    sg.nn.Conv2d(1, 16, kernel_size=3, pad_width=1),
    sg.nn.ReLU(),
    sg.nn.MaxPool2d(2, 2),       # halves H and W
)
```

---

::: simplegrad.nn.pooling.MaxPool2d
