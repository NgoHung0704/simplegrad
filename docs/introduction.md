# Introduction

simplegrad is an **educational deep learning framework** built on top of NumPy. Every part of the stack — from the autograd engine to the Adam optimizer — is written in plain Python so that you can read the source, follow the math, and understand how modern deep learning works from the ground up.

If you have ever wondered *"what does PyTorch actually do when I call `.backward()`?"*, simplegrad is the answer in slow motion.

## Installation

```bash
pip install simplegrad
```

For local development:

```bash
git clone https://github.com/simplegrad/simplegrad
cd simplegrad
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

| Extra      | What it adds |
|------------|--------------|
| `dev`      | `pytest`, `black`, `mypy`, `torch` for cross-checks. |
| `docs`     | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`. |
| `cuda12`   | `cupy-cuda12x` so tensors can live on `cuda:N`. |
| `cuda13`   | `cupy-cuda13x` for CUDA 13. |

## Mental model

1. **Forward.** When you write `y = x @ w`, simplegrad runs the matmul **and** records a graph node remembering its inputs and how to compute the gradient.
2. **Backward.** When you call `loss.backward()`, the engine walks that graph in reverse, applies the chain rule at each node, and accumulates a gradient on every leaf tensor (`.grad`).
3. **Update.** An `Optimizer` reads the gradients and writes new values into the parameters in place.

That loop — forward, backward, update — is the entire training algorithm.

## Architecture

The package is organized in **strict layers**. A layer may only import from layers below it:

```
core/         Tensor, autograd engine, base classes (Module, Optimizer, Scheduler)
  ↑
functions/    Differentiable math, activations, losses, pooling, conv
  ↑
nn/           High-level neural network layers (Linear, Conv2d, Dropout, ...)
  ↑
optimizers/   SGD, Adam
schedulers/   LinearLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateauLR
```

Supporting modules (`track/`, `visual/`, `simpleboard/`) sit alongside this hierarchy and import from it but are not imported by it.

| Package | What's in it |
|---|---|
| **`core/`** | `Tensor`, the `Function` base class, and `Module`/`Optimizer`/`Scheduler` base classes. |
| **`functions/`** | Differentiable ops: math, activations, reductions, losses, conv, pooling, transforms. |
| **`nn/`** | `Module` wrappers around the functional ops: `Linear`, `Conv2d`, `MaxPool2d`, `Dropout`, `Embedding`, `Flatten`, `Sequential`, plus activation and loss layers. |
| **`optimizers/`** | `SGD` (momentum, dampening) and `Adam` (bias-corrected moments). Both support parameter groups. |
| **`schedulers/`** | `LinearLR`, `ExponentialLR`, `CosineAnnealingLR`, `ReduceLROnPlateauLR`. |
| **`track/`** | `Tracker` records scalar metrics and computation graphs to SQLite. |
| **`visual/`** | `graph()` renders a Tensor's computation graph; `plot()` and `scatter()` draw training curves. |

## First steps

```python
import simplegrad as sg

x = sg.Tensor([1.0, 2.0, 3.0], label="x")     # leaf tensor
y = sg.mean(x ** 2)                            # builds the graph
y.backward()                                   # fills x.grad

print(x.grad)   # array([0.667, 1.333, 2.0])
```

Operators (`+`, `*`, `@`, `**`) and functional ops (`sg.relu`, `sg.softmax`, `sg.sum`, `sg.mean`, ...) all build graph nodes with known gradients. Calling `.backward()` on a scalar tensor walks the graph in reverse and accumulates `.grad` on every leaf.

## Full training example

```python
import simplegrad as sg

# Build a small classifier
model = sg.nn.Sequential(
    sg.nn.Linear(4, 16),
    sg.nn.ReLU(),
    sg.nn.Linear(16, 3),
)
loss_fn = sg.nn.CELoss()
optimizer = sg.opt.Adam(model, lr=1e-3)

# Toy data — 8 samples of a 4-dim feature, all in class 0
x_train = sg.Tensor([[0.1, 0.2, 0.3, 0.4]] * 8, label="x")
y_train = sg.Tensor([[1, 0, 0]] * 8, label="y")

# Training loop
for step in range(200):
    optimizer.zero_grad()             # clear .grad on every parameter
    logits = model(x_train)           # forward
    loss = loss_fn(logits, y_train)   # scalar tensor
    loss.backward()                   # populate .grad everywhere
    optimizer.step()                  # update weights in-place

    if step % 50 == 0:
        print(f"step {step}  loss {loss.values:.4f}")
```

Every concept here — `Tensor`, `Module`, `Sequential`, `CELoss`, `Adam` — is documented in detail in the API reference.

## Lazy mode

By default simplegrad runs **eagerly**. Lazy mode lets you build a graph and execute it in one shot:

```python
with sg.lazy():
    a = sg.Tensor([1.0, 2.0])
    b = sg.Tensor([3.0, 4.0])
    c = a + b          # not computed yet — c.values is None
    d = sg.mean(c)     # also deferred

d.realize()            # executes the full graph
print(d.values)        # 3.5
```

`backward()` calls `realize()` for you, so you almost never need to invoke it manually.

## Experiment tracking

```python
from simplegrad.track import Tracker

tracker = Tracker("./experiments")
tracker.set_experiment("mnist")
run_id = tracker.start_run(name="baseline", config={"lr": 1e-3})

for step in range(100):
    # ... training step ...
    tracker.record("loss", loss.values.item(), step=step)

tracker.end_run()
```

Every metric becomes a row in a SQLite database under `./experiments/`. Plot inline with `simplegrad.visual.plot()` or browse them in the SimpleBoard dashboard.

## Computation graph visualization

```python
from simplegrad.visual import graph

x = sg.Tensor([1.0, 2.0], label="x")
y = sg.Tensor([3.0, 4.0], label="y")
z = sg.mean(x * y + x)

graph(z)   # renders an SVG inline in the notebook
```

Salmon = leaf, blue = intermediate, gold = operation. Functions decorated with `@compound_op` (and `Module.forward` methods) are wrapped in a labelled rectangle.

## Where to go next

- Build a model: [`Module`](api/core/module.md), [`Sequential`](api/nn/sequential.md), [`Linear`](api/nn/linear.md).
- Differentiate it: [`Tensor`](api/core/autograd.md), [`Function`](api/core/function.md).
- Train it: [`SGD`](api/optimizers/sgd.md), [`Adam`](api/optimizers/adam.md), [Schedulers](api/schedulers/linear.md).
- Track and visualize: [Tracking](api/track.md), [Visualization](api/visual.md).
