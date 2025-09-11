"""
Sphinx configuration for layeredconfig documentation.

Uses MyST Markdown, autodoc, and autosummary to generate API docs.
"""
import importlib.util as _importlib_util
from datetime import datetime
import layeredconfig as _pkg


# -- Project information -----------------------------------------------------

project = "layeredconfig"
author = "Xylar Asay-Davis"
copyright = f"{datetime.now().year}, {author}"
release = str(_pkg.__version__)


# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Render type hints in the description instead of function signature
autodoc_typehints = "description"

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]


# -- Options for HTML output -------------------------------------------------

html_theme = "furo" if _importlib_util.find_spec("furo") else "alabaster"

html_static_path = ["_static"]


# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# -- Source suffixes ---------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}
