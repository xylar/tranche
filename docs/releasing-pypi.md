---
title: Releasing on PyPI
---

# Releasing tranche on PyPI (via CI)

Releases are published by GitHub Actions. You generally don’t need to build or upload locally.

## Prerequisites (one-time)

- Choose a unique PyPI distribution name in `pyproject.toml` (`[project].name`).
- Configure publishing auth (choose one):
  - API tokens: add repo secrets `PYPI_API_TOKEN` and (optional) `TEST_PYPI_API_TOKEN`.
  - Trusted Publishing (OIDC): enable a Trusted Publisher on PyPI. The workflow already grants `id-token: write`. If PyPI asks for an “Environment name,” use `pypi` (the workflow sets it by default, and `testpypi` for manual TestPyPI runs).
- Optional: create GitHub Environments named `pypi` and `testpypi` to gate runs.

## Normal release flow

1) Bump the version in `tranche/version.py` and commit.

2) Tag the release (no leading `v`) and push the tag:

```bash
git tag -a 0.1.1 -m "tranche 0.1.1"
git push origin 0.1.1
```

This triggers `.github/workflows/publish.yml` to:
- build sdist and wheel,
- check metadata,
- publish to PyPI.

Publishing also runs if you publish a GitHub Release for that tag.

## Optional: publish to TestPyPI

Use the manual workflow and choose `testpypi`:

1) GitHub → Actions → “Build and publish to PyPI” → Run workflow → repository: `testpypi`.
2) (Optional) Verify install from TestPyPI in a clean env.

## Troubleshooting

- 403 “not allowed to upload to project”: the name is taken on PyPI. Pick a new `[project].name` or pursue a PEP 541 transfer.
- 409 “File already exists”: bump `__version__` and re-tag.
- Build/metadata failures: open the failed workflow, check the “Build” or “twine check” step logs.
- Python compatibility: controlled by `requires-python` in `pyproject.toml` (`>=3.10`).

## Optional local verification

If you want to sanity check locally:

```bash
python -m build
python -m twine check dist/*
```

This project is pure Python and produces a universal wheel (`py3-none-any`).
