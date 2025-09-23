"""Package version information.

This module holds the single source of truth for the package version.
Build backends read tranche.version.__version__ via pyproject.
"""

__all__ = ["__version__"]

# Update this value for releases; pre-releases can use semantic versions
# like "0.2.0rc1". The packaging config reads this attribute.
__version__ = "0.2.3.post1"
