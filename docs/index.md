---
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="images/header.png" alt="simplegrad" width="720" />
</p>

<p align="center"><strong>A lightweight, NumPy-backed deep learning framework with autograd.</strong><br/>
Read the source. Follow the math. Train real models.</p>

<p align="center">
  <a href="introduction/" class="md-button md-button--primary">Get started</a>
  <a href="api/core/autograd/" class="md-button">API reference</a>
</p>

---

## Why simplegrad?

simplegrad is a small deep learning framework built from scratch on top of NumPy. Every part of the stack — tensors, autograd, layers, optimizers, schedulers, experiment tracking — is plain Python you can read in an afternoon. It demystifies what frameworks like PyTorch and TensorFlow actually do, while still being capable enough to train real classifiers.

- **Reverse-mode autograd** built around a `Tensor` class.
- **Familiar nn API**: `Module`, `Sequential`, `Linear`, `Conv2d`, `Dropout`, `Embedding`, ReLU/Tanh/Sigmoid/ELU/GELU, MSE & cross-entropy losses.
- **SGD & Adam** optimizers with parameter groups; Linear / Exponential / Cosine / Plateau schedulers.
- **CPU and CUDA** through a unified `numpy` / `cupy` backend.
- **Eager and lazy** execution.
- **Experiment tracking** to SQLite + inline graph and metric visualization.

## A 30-second taste

```python
import simplegrad as sg

x = sg.Tensor([[1.0, 2.0], [3.0, 4.0]], label="x")
w = sg.Tensor([[0.5], [-0.5]], label="w")

y = sg.mean(x @ w)
y.backward()

print(x.grad)   # d(mean(x @ w)) / dx
print(w.grad)   # d(mean(x @ w)) / dw
```

That's the entire workflow: build a graph by running ops, call `.backward()` to fill `.grad`, repeat. Layers, optimizers, and schedulers are all built on top of this idea.

See the [Introduction](introduction.md) for the full walkthrough, then jump into the [API reference](api/core/autograd.md).
