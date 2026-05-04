# Convolution

`conv2d` applies a 2-D convolution. Internally it uses the **im2col + matmul** trick: receptive fields are gathered into a 2-D matrix and the convolution is computed as a single batched matmul.

For most users the layer wrapper [`nn.Conv2d`](../nn/conv.md) is more convenient — it owns the weight and bias tensors. `functions.conv2d` is the underlying differentiable op.

## Output shape

For input $(N, C_{\text{in}}, H, W)$, weight $(C_{\text{out}}, C_{\text{in}}, kH, kW)$, stride $(s_H, s_W)$, padding $(p_H, p_W)$:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - kH}{s_H} \right\rfloor + 1
\qquad
W_{\text{out}} = \left\lfloor \frac{W + 2 p_W - kW}{s_W} \right\rfloor + 1
$$

| Tensor       | Shape                                        |
|--------------|----------------------------------------------|
| Input        | $(N, C_{\text{in}}, H, W)$                   |
| Weight       | $(C_{\text{out}}, C_{\text{in}}, kH, kW)$    |
| Bias         | $(C_{\text{out}},)$ *(optional)*             |
| Output       | $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$ |

## Example

```python
import simplegrad as sg

x = sg.normal((1, 3, 5, 5))     # 1 RGB image, 5x5
W = sg.normal((4, 3, 3, 3))     # 4 filters, 3x3, over 3 input channels

y = sg.conv2d(x, W, stride=1, pad_width=1)
print(y.shape)                  # (1, 4, 5, 5) -- "same" padding
```

`pad_width` accepts an `int` (same padding all sides) or a 4-tuple `(top, bottom, left, right)`. `pad_mode` is forwarded to `numpy.pad` — only `"constant"` is supported in the backward pass.

---

::: simplegrad.functions.conv.pad

::: simplegrad.functions.conv.conv2d
