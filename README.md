# layeredconfig

ConfigParser with layered precedence, provenance, comment-preserving writes, and safe expressions.

## Install

- Base (no numpy):

```bash
pip install .
```

- With numpy extras for safe numpy expressions:

```bash
pip install .[numpy]
```

## Quick start

```python
from layeredconfig import LayeredConfig

config = LayeredConfig()
config.add_from_file("defaults.ini")
config.add_user_config("user.ini")

# Simple values
value = config.get("core", "option")

# Expression (literal)
points = config.getexpression("plot", "ticks", backend="literal")

# Expression (safe with numpy)
arr = config.getexpression("plot", "bins", backend="safe", allow_numpy=True)

# Provenance
info = config.explain("plot", "bins")
print(info)
```

## Security notes

- Expressions default to `backend="literal"` for safety.
- The `safe` backend only allows a small AST subset and a very small set of symbols.
- NumPy is opt-in via `allow_numpy=True` and the `numpy` extra.

## Extensibility

You can add your own safe symbols:

```python
import math
config.register_symbol("sqrt", math.sqrt)
```

## License

BSD-3-Clause
