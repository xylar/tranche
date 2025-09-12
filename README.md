# tranche

ConfigParser with layered precedence, provenance, comment-preserving writes, and safe expressions.

## Documentation

Full docs: https://xylar.github.io/tranche/

## Install

- From PyPI (no NumPy):

```bash
pip install tranche
```

- With NumPy extras (to enable safe NumPy expressions):

```bash
pip install tranche[numpy]
```

- From source (local checkout):

```bash
pip install .
```

## Quick start

```python
from tranche import Tranche

config = Tranche()
config.add_from_file("defaults.cfg")
config.add_user_config("user.cfg")

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

## Links

- Docs: https://xylar.github.io/tranche/
- Source: https://github.com/xylar/tranche
- Issues: https://github.com/xylar/tranche/issues

## License

BSD-3-Clause
