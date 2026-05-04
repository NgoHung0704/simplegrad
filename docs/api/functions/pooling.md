# Pooling

Pooling slides a window over the spatial dimensions and collapses each window to a single number — the maximum, in this case. It downsamples feature maps and gives the network local translation invariance.

The layer wrapper [`nn.MaxPool2d`](../nn/pooling.md) is the convenient way to use this in a `Sequential`.

## Output shape

For input $(N, C, H, W)$, kernel $(kH, kW)$, stride $(s_H, s_W)$, padding $(p_H, p_W)$:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2 p_H - kH}{s_H} \right\rfloor + 1
\qquad
W_{\text{out}} = \left\lfloor \frac{W + 2 p_W - kW}{s_W} \right\rfloor + 1
$$

If `stride` is `None`, it defaults to `kernel_size` (non-overlapping windows — the standard choice).

## Example

```python
import simplegrad as sg

x = sg.Tensor([[[[ 1,  2,  3,  4],
                 [ 5,  6,  7,  8],
                 [ 9, 10, 11, 12],
                 [13, 14, 15, 16]]]], comp_grad=False)

y = sg.max_pool2d(x, kernel_size=2, stride=2)
print(y.values)
# [[[[ 6.  8.]
#    [14. 16.]]]]
```

---

::: simplegrad.functions.pooling.max_pool2d
