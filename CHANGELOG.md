# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

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
