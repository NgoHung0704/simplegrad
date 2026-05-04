# Dtypes

simplegrad uses string keys to identify NumPy dtypes:

| Key          | NumPy class      |
|--------------|------------------|
| `"int8"`     | `np.int8`        |
| `"int16"`    | `np.int16`       |
| `"int32"`    | `np.int32`       |
| `"int64"`    | `np.int64`       |
| `"float16"`  | `np.float16`     |
| `"float32"`  | `np.float32` *(default)* |
| `"float64"`  | `np.float64`     |

## Set the global default

```python
import simplegrad as sg

sg.default_dtype("float64")
x = sg.Tensor([1.0, 2.0])      # x.dtype == 'float64'
sg.default_dtype("float32")
```

This affects every tensor created **without** an explicit `dtype=...` argument.

## Per-tensor dtype

```python
x = sg.Tensor([1.0, 2.0], dtype="float16")
y = x.convert_to("float32", inplace=False)   # new tensor
```

---

::: simplegrad.core.dtypes.default_dtype

::: simplegrad.core.dtypes.get_default_dtype

::: simplegrad.core.dtypes.get_default_dtype_class

::: simplegrad.core.dtypes.get_dtype_class

::: simplegrad.core.dtypes.as_array

::: simplegrad.core.dtypes.convert_to_dtype
