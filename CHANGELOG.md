# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

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
