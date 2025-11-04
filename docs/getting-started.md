# Getting Started

This library builds layered configuration files with provenance and safe expression evaluation.

## Installation

Optional NumPy support can be installed via the extra:

```
pip install tranche[numpy]
```

## Quick example

```python
from tranche import Tranche

cfg = Tranche()
# Add files in order of increasing precedence
cfg.add_from_file('defaults.cfg')
cfg.add_user_config('user.cfg')

value = cfg.get('section', 'option')
print(value)
```

### Layering semantics (at a glance)

- Add base files with `add_from_file(...)` in order; later base files can override earlier base files.
- Add a user file with `add_user_config(...)`; user values always take precedence over base files.
- You can combine two Tranche objects:

```python
higher = Tranche(); higher.add_from_file('override.cfg')
lower = Tranche(); lower.add_from_file('base.cfg')
lower.append(higher)  # entries from 'higher' win
```

## Safe expressions

To parse list/tuple/dict values, or evaluate expressions:

```python
# Literal-only (numbers, strings, containers)
vals = cfg.getexpression('calc', 'values')

# NumPy-enabled expression auto-selects safe backend
grid = cfg.getexpression('grid', 'levels', allow_numpy=True)

# Or use the helper for NumPy
grid2 = cfg.getnumpy('grid', 'levels')
```

Register custom callables for the safe backend (available when backend resolves to 'safe'):

```python
import math
cfg.register_symbol('sqrt', math.sqrt)
```

When using `allow_numpy=True` with the safe backend, a limited `np` namespace is available.

### Section-level helpers

You can access a section as an object and use tranche's helpers directly on it. This keeps
code concise when working within a single section:

```python
sec = cfg['calc']

# Same as cfg.getlist('calc', 'values', dtype=int)
values = sec.getlist('values', dtype=int)

# Same as cfg.getexpression('calc', 'levels', allow_numpy=True)
levels = sec.getnumpy('levels')

# Provenance for a single option
info = sec.explain('values')  # {"value": ..., "source": ..., "layer": ...}
```

## Write a combined config (with provenance)

Write the merged configuration back to disk while preserving original comments and including `# source:` for each option:

```python
with open('combined.cfg', 'w') as f:
	cfg.write(f, include_sources=True, include_comments=True)
```