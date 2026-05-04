# Activation Layers

`Module` wrappers around the [functional activations](../functions/activations.md). Use these when you want to drop an activation into a `Sequential`:

```python
import simplegrad as sg

mlp = sg.nn.Sequential(
    sg.nn.Linear(8, 32),
    sg.nn.ReLU(),                  # the layer here
    sg.nn.Linear(32, 4),
    sg.nn.Softmax(dim=-1),
)
```

If you call an activation outside a `Sequential`, the function form is shorter:

```python
y = sg.relu(layer(x))
```

Both forms produce the same graph node — the layers exist purely for ergonomics.

| Layer | Equivalent function |
|-------|---------------------|
| [`ReLU`](#simplegrad.nn.activation_layers.ReLU)       | [`relu`](../functions/activations.md#simplegrad.functions.activations.relu) |
| [`ELU`](#simplegrad.nn.activation_layers.ELU)         | [`elu`](../functions/activations.md#simplegrad.functions.activations.elu) |
| [`Tanh`](#simplegrad.nn.activation_layers.Tanh)       | [`tanh`](../functions/activations.md#simplegrad.functions.activations.tanh) |
| [`Sigmoid`](#simplegrad.nn.activation_layers.Sigmoid) | [`sigmoid`](../functions/activations.md#simplegrad.functions.activations.sigmoid) |
| [`GELU`](#simplegrad.nn.activation_layers.GELU)       | [`gelu`](../functions/activations.md#simplegrad.functions.activations.gelu) |
| [`Softmax`](#simplegrad.nn.activation_layers.Softmax) | [`softmax`](../functions/activations.md#simplegrad.functions.activations.softmax) |

---

::: simplegrad.nn.activation_layers.ReLU

::: simplegrad.nn.activation_layers.ELU

::: simplegrad.nn.activation_layers.Tanh

::: simplegrad.nn.activation_layers.Sigmoid

::: simplegrad.nn.activation_layers.GELU

::: simplegrad.nn.activation_layers.Softmax
