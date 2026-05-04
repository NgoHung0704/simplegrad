# Embedding

A learnable lookup table that maps integer indices (token IDs) to dense vectors. The standard first layer of any language model.

$$
\mathrm{Embedding}(i) = W_{i,:}
$$

**Shape:**

| Tensor | Shape                                          |
|--------|------------------------------------------------|
| Input  | $(\dots)$ — integer indices, any shape         |
| Weight | $(\text{num\_embeddings}, \text{embedding\_dim})$ |
| Output | $(\dots, \text{embedding\_dim})$               |

## Example

```python
import simplegrad as sg

embedding = sg.nn.Embedding(num_embeddings=1000, embedding_dim=32)

token_ids = sg.Tensor([[5, 17, 42], [99, 0, 100]],
                      dtype="int32", comp_grad=False)
vectors = embedding(token_ids)
print(vectors.shape)   # (2, 3, 32) -- batch x sequence x embed_dim
```

When you call `.backward()`, only the rows referenced by `token_ids` receive gradient updates. Duplicate indices accumulate correctly (same token twice in a sequence contributes twice to that row's gradient).

The default weight is sampled from $\mathcal{N}(0, 1)$. Pass a pre-built `weight` tensor of shape $(\text{num\_embeddings}, \text{embedding\_dim})$ to use a custom initialization (e.g., loaded pretrained vectors).

---

::: simplegrad.nn.embedding.Embedding
