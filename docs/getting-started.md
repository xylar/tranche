# Getting Started

This library builds layered configuration files with provenance and safe expression evaluation.

## Installation

Optional NumPy support can be installed via the extra:

```
pip install layeredconfig[numpy]
```

## Quick example

```python
from layeredconfig import LayeredConfig

cfg = LayeredConfig()
# Add files in order of increasing precedence
cfg.add_from_file('defaults.cfg')
cfg.add_user_config('user.cfg')

value = cfg.get('section', 'option')
print(value)
```

### Layering semantics (at a glance)

- Add base files with `add_from_file(...)` in order; later base files can override earlier base files.
- Add a user file with `add_user_config(...)`; user values always take precedence over base files.
- You can combine two LayeredConfig objects:

```python
higher = LayeredConfig(); higher.add_from_file('override.cfg')
lower = LayeredConfig(); lower.add_from_file('base.cfg')
lower.append(higher)  # entries from 'higher' win
```

## Safe expressions

To parse list/tuple/dict values, or evaluate expressions using a safe AST evaluator:

```python
cfg.getexpression('calc', 'values', backend='safe')
```

Register custom callables for the safe backend:

```python
import math
cfg.register_symbol('sqrt', math.sqrt)
```

When using `allow_numpy=True` with the safe backend, a limited `np` namespace is available.

## Write a combined config (with provenance)

Write the merged configuration back to disk while preserving original comments and including `# source:` for each option:

```python
with open('combined.cfg', 'w') as f:
	cfg.write(f, include_sources=True, include_comments=True)
```