# Conv2d

Applies a 2-D convolution. Wraps the differentiable [`functions.conv2d`](../functions/conv.md) op and owns its own weight and bias tensors.

**Shape:**

| Tensor | Shape                                              |
|--------|----------------------------------------------------|
| Input  | $(N, C_{\text{in}}, H, W)$ or $(C_{\text{in}}, H, W)$ |
| Weight | $(C_{\text{out}}, C_{\text{in}}, kH, kW)$           |
| Bias   | $(C_{\text{out}},)$ *(if `use_bias=True`)*          |
| Output | $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$ |

Output spatial dims:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - kH}{s_H} \right\rfloor + 1, \quad
W_{\text{out}} = \left\lfloor \frac{W + 2 p_W - kW}{s_W} \right\rfloor + 1
$$

## Example

```python
import simplegrad as sg

conv = sg.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, pad_width=1)
x = sg.normal((4, 3, 32, 32))     # batch of 4 RGB 32x32 images
y = conv(x)
print(y.shape)                    # (4, 16, 32, 32) — same spatial size due to pad
```

Inside a CNN:

```python
model = sg.nn.Sequential(
    sg.nn.Conv2d(1, 16, kernel_size=3, pad_width=1),
    sg.nn.ReLU(),
    sg.nn.MaxPool2d(2, 2),
    sg.nn.Conv2d(16, 32, kernel_size=3, pad_width=1),
    sg.nn.ReLU(),
    sg.nn.MaxPool2d(2, 2),
    sg.nn.Flatten(),
    sg.nn.Linear(32 * 7 * 7, 10),
)
```

## Initialization

Weights are sampled from $\mathcal{U}(-b, +b)$ with $b = \dfrac{1}{\sqrt{C_{\text{in}} \cdot kH \cdot kW}}$ — Kaiming-uniform scaled to the receptive field.

---

::: simplegrad.nn.conv.Conv2d
