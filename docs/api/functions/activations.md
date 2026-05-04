# Activations

Differentiable nonlinearities. Each is also available as a `Module` wrapper in [`nn.activations`](../nn/activations.md), for use inside a `Sequential`.

## Quick reference

| Function                              | Formula                                   | Output range        |
|---------------------------------------|-------------------------------------------|---------------------|
| [`relu`](#simplegrad.functions.activations.relu)       | $\mathrm{ReLU}(x) = \max(0,\, x)$         | $[0, \infty)$       |
| [`tanh`](#simplegrad.functions.activations.tanh)       | $\tanh(x)$                                | $(-1, 1)$           |
| [`sigmoid`](#simplegrad.functions.activations.sigmoid) | $\sigma(x) = \dfrac{1}{1 + e^{-x}}$       | $(0, 1)$            |
| [`elu`](#simplegrad.functions.activations.elu)         | $x \text{ if } x>0;\ \alpha(e^x{-}1)$ otherwise | $(-\alpha, \infty)$ |
| [`gelu`](#simplegrad.functions.activations.gelu)       | $0.5\,x\,(1 + \mathrm{erf}(x/\sqrt 2))$   | $\mathbb{R}$        |
| [`softmax`](#simplegrad.functions.activations.softmax) | $\dfrac{e^{x_i}}{\sum_j e^{x_j}}$         | sums to 1 along `dim` |

For multi-class classification, prefer [`nn.CELoss`](../nn/losses.md) over an explicit `softmax` followed by cross-entropy — `CELoss` does both in one numerically stable step.

---

## relu

$$
\mathrm{ReLU}(x) = \max(0,\, x)
$$

```python
>>> sg.relu(sg.Tensor([-1.0, 0.0, 2.0])).values
array([0., 0., 2.], dtype=float32)
```

::: simplegrad.functions.activations.relu

---

## tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

```python
>>> sg.tanh(sg.Tensor([0.0, 1.0, -1.0])).values
array([ 0.   ,  0.762, -0.762], dtype=float32)
```

::: simplegrad.functions.activations.tanh

---

## sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

```python
>>> sg.sigmoid(sg.Tensor([0.0, 1.0, -1.0])).values
array([0.5  , 0.731, 0.269], dtype=float32)
```

::: simplegrad.functions.activations.sigmoid

---

## elu

$$
\mathrm{ELU}(x) =
\begin{cases}
x & \text{if } x > 0 \\
\alpha\,(e^x - 1) & \text{otherwise}
\end{cases}
$$

```python
>>> sg.elu(sg.Tensor([-1.0, 0.0, 1.0])).values
array([-0.632,  0.   ,  1.   ], dtype=float32)
```

::: simplegrad.functions.activations.elu

---

## gelu

$$
\mathrm{GELU}(x) = 0.5\,x\,\bigl(1 + \mathrm{erf}(x / \sqrt{2})\bigr)
$$

A smooth alternative to ReLU, used by BERT, GPT, and most modern transformers. The fast `mode="tanh"` approximation is accurate to within 0.02% for all `x`:

$$
\mathrm{GELU}_{\text{tanh}}(x) \approx 0.5\,x\,\bigl(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3))\bigr)
$$

```python
>>> sg.gelu(sg.Tensor([-1.0, 0.0, 1.0])).values
array([-0.159,  0.   ,  0.841], dtype=float32)
```

::: simplegrad.functions.activations.gelu

---

## softmax

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

```python
>>> sg.softmax(sg.Tensor([[1.0, 2.0, 3.0]]), dim=1).values
array([[0.090, 0.245, 0.665]], dtype=float32)
```

::: simplegrad.functions.activations.softmax
