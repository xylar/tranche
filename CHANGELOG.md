# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

## [0.6.0] - 2026-08-29
### Changed
- **Breaking:** `getlist()` now raises `NoOptionError` or `NoSectionError` for a
  missing option when no `fallback` is given, matching `get()`, `getint()`,
  `getfloat()` and `getboolean()`.  It previously returned `None`, which turned
  a mistyped option name into a `TypeError: 'NoneType' is not iterable` far
  from its cause.  Pass `fallback=None` to keep the old result for one call.
- `get()`, `getint()`, `getfloat()`, `getboolean()` and `getlist()` are now
  typed with `configparser`-style overloads, so they return a plain value when
  no `fallback` is given and `value | fallback` when one is.  They were
  annotated as always returning `X | None`, which forced callers to guard
  against a `None` that only an explicit `fallback` could produce.
- **Breaking:** `set()` raises `TypeError` for a value that is neither a `str`
  nor `None`, naming the section, the option and the file and line it was set
  from.  Such a value was previously stored and only rejected later, when the
  config was combined, by an error that identified none of those things.
  (#23)
- `raw`, `vars` and `fallback` are now declared and documented on every getter
  rather than hidden behind `**kwargs`.  They have been forwarded to
  `configparser` since 0.4.0, but nothing said so.  (#22)

### Fixed
- Removed the `python_version` pin from the mypy configuration.  mypy applies
  it when parsing installed type stubs as well as first-party code, so a pin
  below the version those stubs are written for stopped mypy before it checked
  anything.

## [0.5.0] - 2026-06-25
### Added
- Broader `configparser.ConfigParser` compatibility so tranche can be used as a
  more complete drop-in wrapper:
  - Query methods: `sections()`, `options()`, `items()`, `defaults()`,
    `keys()`, `values()`.
  - Container protocol: `__contains__`, `__iter__`, `__len__`.
  - In-memory loaders: `read_string()`, `read_file()`, `read_dict()` (with
    provenance source labels and env-var interpolation for text loaders).
  - Mutation: `add_section()`, `remove_option()`, `remove_section()`, plus
    mapping assignment/deletion via `config[section] = {...}` and
    `del config[section]`.  `remove_option`/`remove_section` act across all
    layers so options truly disappear from the combined view.
- `Section` is now a more complete `configparser.SectionProxy` wrapper:
  `keys()`, `values()`, `items()`, `setdefault()`, `update()`, `pop()`,
  `popitem()`, `clear()`, item assignment/deletion, and a `name` property.
  Mutations are routed through the parent `Tranche` so they persist across
  recombination.

### Changed
- `Section` now re-derives a live `SectionProxy` on each access instead of
  caching one, so a held `Section` reflects later mutations rather than a stale
  snapshot.

## [0.4.0] - 2025-12-04
### Added
- Keyword arguments are forwarded from `get()`, `getint()`, `getfloat()`,
  `getboolean()` and `getlist()` to the underlying
  `configparser.ConfigParser` methods, so `fallback` and the other
  `configparser` keyword arguments can be passed through.

## [0.3.0] - 2025-11-04
### Added
- A `Section` wrapper giving access to one config section's options through
  tranche's getters, with tests and documentation.

## [0.2.3.post1] - 2025-09-23
### Notes
- Release metadata only; no code changes.

## [0.2.2] - 2025-09-22
### Changed
- Internal maintenance release: bumped version metadata and prepared packaging.

### Fixed
- Minor documentation clarifications and release process notes (no code changes).

### Notes
- Patch release with no public API changes; safe to upgrade.

## [0.2.1] - 2025-09-20
### Fixed
- Corrected minor doc typos and clarified NumPy expression usage examples.

### Changed
- Adjusted packaging metadata / workflow tweaks (no runtime code changes).

### Notes
- Patch release; no API or behavior changes.

## [0.2.0] - 2025-09-19
### Added
- `Tranche.getnumpy()` convenience helper for NumPy-enabled expressions.

### Changed
- `getexpression()` now accepts `backend=None` (default). It auto-selects
  `safe` when `allow_numpy=True` and `literal` otherwise.
- Removed implicit literal->safe fallback/print; behavior is now deterministic
  based on arguments.

### Notes
- This is an intentional minor version bump due to a public API signature
  change (`backend` can now be `None`). Existing code that passed an explicit
  backend string continues to work unchanged.

## [0.1.1] - 2025-09-12
### Changed
- Publishing workflow for PyPI updated: publish on GitHub Release instead of on tag.
- Internal CI/CD tweaks; no functional code changes in the package.

## [0.1.0] - 2025-09-12
### Added
- Initial release of tranche.
- Layered configuration with clear precedence: base files and user overrides.
- Provenance via `explain(section, option)` reporting value, layer, and source file.
- Comment-preserving `write()` with optional source annotations.
- Safe expression evaluation:
  - Literal backend for simple Python literals.
  - Safe AST backend with a small whitelist and optional NumPy namespace.
  - Optional dtype casting for list/tuple/dict values.
- Extended interpolation and environment variable support (`${env:VAR}`).
- Validation hook to run custom checks on the combined configuration.
- Typed package (PEP 561) with `py.typed`.
- Documentation (Sphinx + MyST), tests, and CI workflows.
