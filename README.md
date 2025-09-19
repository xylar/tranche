![tranche logo](https://raw.githubusercontent.com/xylar/tranche/main/docs/logo/tranche_logo_small.png)

# tranche

ConfigParser with layered precedence, provenance, comment-preserving writes, and safe expressions.

The name **tranche** comes from the French word for *slice* — a nod to how the
library lets you cut cleanly through multiple configuration layers to get a
single, effective view of your settings.

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

# Expression (literal backend chosen automatically)
points = config.getexpression("plot", "ticks")

# NumPy expression (auto-selects safe backend because allow_numpy=True)
arr = config.getexpression("plot", "bins", allow_numpy=True)

# Or via convenience helper
arr2 = config.getnumpy("plot", "bins")

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
