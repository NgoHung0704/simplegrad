# Loss Layers

`Module` wrappers around the [functional losses](../functions/losses.md). They expose the same API but let you keep loss configuration alongside the model.

| Layer                                                     | Underlying op                                                              | Notes |
|-----------------------------------------------------------|----------------------------------------------------------------------------|-------|
| [`CELoss`](#simplegrad.nn.loss_layers.CELoss)             | [`ce_loss`](../functions/losses.md#simplegrad.functions.losses.ce_loss)     | Built-in stable softmax — feed logits, not probabilities. |
| [`MSELoss`](#simplegrad.nn.loss_layers.MSELoss)            | [`mse_loss`](../functions/losses.md#simplegrad.functions.losses.mse_loss)   | Standard regression loss. |

## Example

```python
import simplegrad as sg

loss_fn = sg.nn.CELoss(dim=-1, reduction="mean")

logits = model(x_train)
loss = loss_fn(logits, y_train)
loss.backward()
```

---

::: simplegrad.nn.loss_layers.CELoss

::: simplegrad.nn.loss_layers.MSELoss
