# Linear

Applies a linear transformation:

$$
y = x\,W + b
$$

**Shape:**

| Tensor | Shape                                  |
|--------|----------------------------------------|
| Input  | $(\dots, \text{in\_features})$         |
| Weight | $(\text{in\_features}, \text{out\_features})$ |
| Bias   | $(\text{out\_features},)$ *(if `use_bias=True`)* |
| Output | $(\dots, \text{out\_features})$        |

`Linear` is the building block of every multi-layer perceptron. Stack it with an activation to make a hidden layer.

## Example

```python
import simplegrad as sg

layer = sg.nn.Linear(in_features=4, out_features=8)

x = sg.Tensor([[1.0, 2.0, 3.0, 4.0]])
y = layer(x)
print(y.shape)   # (1, 8)
```

Inside an MLP:

```python
mlp = sg.nn.Sequential(
    sg.nn.Linear(784, 128),
    sg.nn.ReLU(),
    sg.nn.Linear(128, 10),
)
```

## Initialization

Weights are drawn from $\mathcal{U}\!\left(-\dfrac{1}{\sqrt{\text{in\_features}}},\, +\dfrac{1}{\sqrt{\text{in\_features}}}\right)$ — Kaiming-uniform. Pass `weight=` and `bias=` to override.

---

::: simplegrad.nn.linear.Linear
