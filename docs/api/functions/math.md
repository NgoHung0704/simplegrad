# Math Functions

Element-wise differentiable math: `log`, `exp`, `sin`, `cos`, `tan`. Each call builds a graph node that knows how to compute its gradient.

| Function   | Forward    | Gradient        |
|------------|------------|-----------------|
| `log(x)`   | $\ln x$    | $1/x$           |
| `exp(x)`   | $e^x$      | $e^x$           |
| `sin(x)`   | $\sin x$   | $\cos x$        |
| `cos(x)`   | $\cos x$   | $-\sin x$       |
| `tan(x)`   | $\tan x$   | $1/\cos^2 x$    |

---

## log

$$
\log(x) = \ln(x), \quad \frac{d\log}{dx} = \frac{1}{x}
$$

```python
>>> x = sg.Tensor([1.0, 2.0, 4.0])
>>> sg.log(x).values
array([0.   , 0.693, 1.386], dtype=float32)
```

!!! warning "Domain"
    `log` raises `ValueError` if `x` contains a non-positive value (eager mode only).

::: simplegrad.functions.math.log

---

## exp

$$
\exp(x) = e^x, \quad \frac{d\exp}{dx} = e^x
$$

```python
>>> sg.exp(sg.Tensor([0.0, 1.0, 2.0])).values
array([1.   , 2.718, 7.389], dtype=float32)
```

::: simplegrad.functions.math.exp

---

## sin

$$
\sin(x), \quad \frac{d\sin}{dx} = \cos(x)
$$

```python
>>> sg.sin(sg.Tensor([0.0, 1.5708, 3.1416])).values
array([ 0.   ,  1.   , -0.   ], dtype=float32)
```

::: simplegrad.functions.math.sin

---

## cos

$$
\cos(x), \quad \frac{d\cos}{dx} = -\sin(x)
$$

```python
>>> sg.cos(sg.Tensor([0.0, 1.5708, 3.1416])).values
array([ 1.   ,  0.   , -1.   ], dtype=float32)
```

::: simplegrad.functions.math.cos

---

## tan

$$
\tan(x), \quad \frac{d\tan}{dx} = \frac{1}{\cos^2(x)}
$$

```python
>>> sg.tan(sg.Tensor([0.0, 0.7854])).values   # 0, π/4
array([0.   , 1.   ], dtype=float32)
```

::: simplegrad.functions.math.tan
