---
title: Why not just use configparser/ConfigUpdater/ConfigObj?
---

# Why layeredconfig?

layeredconfig builds on Python's configparser but adds a few capabilities that are hard to get elsewhere in one place: multi-file layering with precedence, provenance, safe expression evaluation (optional NumPy), interpolation, and comment-preserving writes.

## Comparison

Legend: ✓ supported · ✗ not supported · — not applicable

| Feature | configparser | ConfigUpdater | ConfigObj | layeredconfig |
| --- | :---: | :---: | :---: | :---: |
| Multi-file layering with explicit user precedence | ✗ | ✗ | ✗ | ✓ |
| Provenance (where did this value come from?) | ✗ | ✗ | ✗ | ✓ |
| Comment round-trip when writing | ✗ | ✓ | ✓ | ✓ |
| Extended interpolation | ✓ | — | ✗ | ✓ |
| Expressions as values | ✗ | ✗ | ✗ | ✓ |
| Optional NumPy in expressions | — | — | ✗ | ✓ |
| Write out with provenance | ✗ | ✗ | ✗ | ✓ |

Notes:
- ConfigUpdater is excellent for editing INI files while preserving formatting, but it is not a layered reader/combiner.
- ConfigObj offers comment round‑tripping and a different API, but not multi-layer precedence or provenance.

## Typical layering flow

Example: merge defaults, site, and user config files with user taking precedence.

```python
from layeredconfig import LayeredConfig

cfg = LayeredConfig()
cfg.add_from_file('defaults.cfg')   # shipped with package
cfg.add_from_file('/etc/myapp.cfg') # site‑wide
cfg.add_user_config('~/.config/myapp.cfg')  # highest precedence

# Access values (with ExtendedInterpolation enabled by default)
val = cfg.get('general', 'output_dir')
```

You can also combine two LayeredConfig objects explicitly:

```python
higher = LayeredConfig(); higher.add_from_file('override.cfg')
lower = LayeredConfig(); lower.add_from_file('base.cfg')
lower.append(higher)  # values from 'higher' win
```

## Interpolation and expressions

Extended interpolation works out of the box:

```ini
[paths]
root = /data/run
output = ${paths:root}/out
```

Expressions are read via `getexpression`:

```ini
[calc]
# List literal
levels = [0, 10, 20]
# With numpy (opt‑in, safe backend)
lev_np = np.linspace(0, 1, 5)
```

```python
# Literal backend (safest):
levels = cfg.getexpression('calc', 'levels', backend='literal')

# Safe backend with a constrained whitelist; enable numpy explicitly:
levels_np = cfg.getexpression('calc', 'lev_np', backend='safe', allow_numpy=True)
```

You can expose additional safe helpers:

```python
import statistics
cfg.register_symbol('mean', statistics.mean)
# Now INI can use: avg = mean([1,2,3])
```

## Writing out with comments and provenance

Preserve original comments and include `# source:` lines showing where each value came from:

```python
with open('combined.cfg', 'w') as f:
    cfg.write(f, include_sources=True, include_comments=True)
```

This yields, for example:

```ini
[general]
# original comment about threads
# source: /etc/myapp.cfg
threads = 8
```

You can also get the source programmatically:

```python
info = cfg.explain('general', 'threads')
# {'value': '8', 'source': '/etc/myapp.cfg', 'layer': 'base'}
```

## Security notes

- Expression evaluation is opt-in and safe by default. The default `backend='literal'` uses `ast.literal_eval`.
- The `backend='safe'` evaluator allows a small, whitelisted set of operations. NumPy is disabled unless you pass `allow_numpy=True`, and only a minimal subset is exposed (`np.arange`, `np.linspace`, `np.array`).
- You can register additional helpers with `register_symbol`; names with dunders or dots are rejected.
