# Visualization

The `simplegrad.visual` package gives you two kinds of inline visualizations: **computation graphs** (Graphviz / SVG) and **training curves** (matplotlib).

---

## Computation graph

```python
import simplegrad as sg
from simplegrad.visual import graph

x = sg.Tensor([1.0, 2.0], label="x")
y = sg.Tensor([3.0, 4.0], label="y")
z = sg.mean(sg.softmax(x * y + x))

graph(z)              # returns a graphviz.Digraph; renders inline in Jupyter
graph(z, path="dag")  # also writes dag.svg to disk
```

Color legend:

- :material-circle:{ style="color: lightsalmon" } **Salmon** — leaf tensors (inputs, parameters)
- :material-circle:{ style="color: lightsteelblue" } **Steel blue** — intermediate (non-leaf) tensors
- :material-circle:{ style="color: lightgoldenrodyellow" } **Gold** — operation nodes

Functions decorated with `@compound_op` (and most `Module.forward` methods) are wrapped in a labelled rectangle.

!!! note "Graphviz binary required"
    `graph()` shells out to the system Graphviz binary. Install it with `brew install graphviz`, `apt install graphviz`, or [download for Windows](https://graphviz.org/download/).

::: simplegrad.visual.inline_comp_graph.graph

---

## Training plots

Pair with [`Tracker`](track.md):

```python
from simplegrad.visual import plot, scatter

results = tracker.get_results(run_id)         # {metric: [RecordInfo(step, value), ...]}
plot(results)                                  # line chart per metric
scatter(results, selected=["loss"], color="#1f77b4")
```

Pass `path=...` to save the figure rather than just displaying it.

::: simplegrad.visual.inline_training_graphs.plot

::: simplegrad.visual.inline_training_graphs.scatter
