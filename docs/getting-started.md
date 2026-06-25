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

## ConfigParser-style access

`Tranche` mirrors much of the `configparser.ConfigParser` interface, so code
written against `ConfigParser` mostly works unchanged while still getting
tranche's layering, provenance and comment handling:

```python
cfg.sections()                 # ['section', ...]
cfg.options('section')         # ['option', ...]
cfg.items('section')           # [('option', 'value'), ...]
'section' in cfg               # membership test
for name in cfg: ...           # iterate section names
len(cfg)                       # number of sections (counts DEFAULT)
```

You can also load configuration from memory rather than a file. Each loader
takes a `source` label used for provenance and a `user` flag to mark user
overrides:

```python
cfg.read_string('[section]\noption = value\n', source='<inline>')
cfg.read_dict({'section': {'option': 'value'}})
with open('extra.cfg') as f:
    cfg.read_file(f)
```

Mutation is supported too. `remove_option` and `remove_section` act across
*all* layers, so the option truly disappears from the combined view:

```python
cfg.add_section('new')
cfg.remove_option('section', 'option')
cfg.remove_section('old')

cfg['section'] = {'option': 'value'}   # replace a whole section
del cfg['section']
```

Sections behave like mutable mappings, with changes routed back through the
parent `Tranche` so they persist across recombination:

```python
sec = cfg['section']
sec['option'] = 'value'                # set a single option
sec.update({'a': 1, 'b': 2})
value = sec.setdefault('option', 'fallback')
sec.pop('a')
del sec['b']
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