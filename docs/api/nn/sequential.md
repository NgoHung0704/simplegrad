# Sequential

A trivial container that applies a fixed list of `Module`s in order. Use it whenever your model is a straight pipeline with no branching.

## Example

```python
import simplegrad as sg

model = sg.nn.Sequential(
    sg.nn.Linear(784, 128),
    sg.nn.ReLU(),
    sg.nn.Dropout(p=0.2),
    sg.nn.Linear(128, 10),
)

logits = model(sg.normal((32, 784)))
print(logits.shape)   # (32, 10)
```

`Sequential.parameters()` returns every parameter of every wrapped module — pass the whole thing to an optimizer:

```python
optimizer = sg.opt.Adam(model, lr=1e-3)
```

If your model has skip connections, multiple inputs, or any branching, write a custom [`Module`](../core/module.md) subclass instead.

---

::: simplegrad.nn.sequential.Sequential
