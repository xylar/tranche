import ast
import inspect
import math
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from configparser import (
    ConfigParser,
    ExtendedInterpolation,
    NoSectionError,
    RawConfigParser,
)
from importlib.resources import files as imp_res_files
from io import StringIO
from re import Match
from types import ModuleType
from typing import Any, TextIO, TypeVar, cast, overload

from .section import Section

CombinedParser = ConfigParser | RawConfigParser
T = TypeVar("T")
F = TypeVar("F")

# Sentinel distinguishing "argument not provided" from an explicit ``None``.
_UNSET: Any = object()


class Tranche:
    """
    A "meta" config parser that keeps a dictionary of config parsers and their
    sources to combine when needed.  The custom config parser allows provenance
    of the source of different config options and allows the "user" config
    options to always take precedence over other config options (even if they
    are added later).

    Example
    -------
    >>> config = Tranche()
    >>> config.add_from_file('default.cfg')
    >>> config.add_user_config('user.cfg')
    >>> value = config.get('section', 'option')

    Attributes
    ----------
    combined : {None, configparser.ConfigParser}
        The combined config options

    combined_comments : {None, dict}
        The combined comments associated with sections and options

    sources : {None, dict}
        The source of each section or option
    """

    class _SafeNamespace:
        """Read-only namespace exposing only whitelisted attributes."""

        def __init__(self, allowed: dict[str, Any]):
            self._allowed = dict(allowed)

        def __getattr__(self, name: str) -> Any:
            if "__" in name:
                raise ValueError("Use of dunder attributes is not allowed")
            if name not in self._allowed:
                raise NameError(f"Attribute '{name}' is not allowed")
            return self._allowed[name]

    @staticmethod
    def _default_symbols(allow_numpy: bool) -> dict[str, Any]:
        """
        Default symbol table for the safe expression evaluator.

        If allow_numpy is True, expose a constrained 'np'/'numpy' namespace
        with only a few safe callables.
        """
        symbols: dict[str, Any] = {
            "range": range,
            "int": int,
            "float": float,
            "pi": math.pi,
            # Expose 'math' with only 'pi' to start;
            # extend later via register_symbol
            "math": Tranche._SafeNamespace({"pi": math.pi}),
        }
        if allow_numpy:
            try:
                import numpy as _np  # local import to keep numpy optional
            except ImportError as e:
                raise ImportError(
                    "NumPy is required for allow_numpy=True. Install "
                    "'tranche[numpy]' to enable this feature."
                ) from e
            np_ns = Tranche._SafeNamespace(
                {
                    "arange": _np.arange,
                    "linspace": _np.linspace,
                    "array": _np.array,
                }
            )
            symbols.update(
                {
                    "np": np_ns,
                    "numpy": np_ns,
                }
            )
        return symbols

    @staticmethod
    def _safe_eval(
        expression: str,
        allow_numpy: bool = False,
        extra_symbols: dict[str, Any] | None = None,
    ) -> Any:
        """
        Evaluate a restricted Python expression safely using an AST whitelist.

        Allowed nodes: Expression, Constant, List, Tuple, Dict, Name,
        Attribute, Call, UnaryOp, BinOp. Names containing '__' are rejected.
        Attribute access is only allowed on SafeNamespace instances.
        """

        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Name,
            ast.Attribute,
            ast.Call,
            ast.UnaryOp,
            ast.BinOp,
        )

        tree = ast.parse(expression, mode="eval")

        symbols = Tranche._default_symbols(allow_numpy)
        # Merge any user-registered symbols (simple names only)
        if extra_symbols:
            for key, val in extra_symbols.items():
                if "__" in key or "." in key:
                    raise ValueError(
                        "Illegal symbol name; '__' and '.' are not allowed"
                    )
                symbols[key] = val

        def eval_node(node: ast.AST) -> Any:
            if not isinstance(node, allowed_nodes):
                raise ValueError(
                    f"Unsupported expression element: {type(node).__name__}"
                )

            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.List):
                return [eval_node(elt) for elt in node.elts]
            if isinstance(node, ast.Tuple):
                return tuple(eval_node(elt) for elt in node.elts)
            if isinstance(node, ast.Dict):
                # Disallow dict unpacking like {**x}
                if any(k is None for k in node.keys):
                    raise ValueError("Dict unpacking is not allowed")
                keys_list: list[Any] = []
                for k in node.keys:
                    # mypy: after the check above, k is not None
                    assert k is not None
                    keys_list.append(eval_node(k))
                vals = [eval_node(v) for v in node.values]
                return {k: v for k, v in zip(keys_list, vals, strict=True)}
            if isinstance(node, ast.Name):
                if "__" in node.id:
                    raise ValueError("Use of dunder names is not allowed")
                if node.id not in symbols:
                    raise NameError(f"Unknown name: {node.id}")
                return symbols[node.id]
            if isinstance(node, ast.Attribute):
                value = eval_node(node.value)
                if not isinstance(value, Tranche._SafeNamespace):
                    raise ValueError("Attribute access only allowed on safe namespaces")
                return getattr(value, node.attr)
            if isinstance(node, ast.Call):
                func = eval_node(node.func)
                # Only allow calling plain callables coming from our
                # symbols/namespaces
                if not callable(func):
                    raise ValueError("Attempted to call a non-callable")
                args = [eval_node(a) for a in node.args]
                kwargs = {
                    kw.arg: eval_node(kw.value)
                    for kw in node.keywords
                    if kw.arg is not None
                }
                # Reject **kwargs or *args (ast.keyword.arg is None for
                # **kwargs)
                if any(kw.arg is None for kw in node.keywords):
                    raise ValueError("Star-args or **kwargs are not allowed")
                return func(*args, **kwargs)
            if isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.Not):
                    return not operand
                if isinstance(node.op, ast.Invert):
                    return ~operand
                raise ValueError("Unsupported unary operator")
            if isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op = node.op
                if isinstance(op, ast.Add):
                    return left + right
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right
                if isinstance(op, ast.FloorDiv):
                    return left // right
                if isinstance(op, ast.Mod):
                    return left % right
                if isinstance(op, ast.Pow):
                    return left**right
                raise ValueError("Unsupported binary operator")

            # Should be unreachable due to the isinstance gate
            raise ValueError(f"Unsupported node: {type(node).__name__}")

        return eval_node(tree)

    def __init__(self) -> None:
        """
        Make a new (empty) config parser
        """
        # per-file configs and comments
        self._configs: dict[str, RawConfigParser] = {}
        self._user_config: dict[str, RawConfigParser] = {}
        self._comments: dict[str, dict[str | tuple[str, str], str]] = {}

        # extra safe-eval symbols registered by the user
        self._extra_symbols: dict[str, Any] = {}

        # combined state
        self.combined: CombinedParser | None = None
        self.combined_comments: dict[str | tuple[str, str], str] | None = None
        self.sources: dict[tuple[str, str], str] | None = None

    def add_user_config(self, filename: str) -> None:
        """
        Add a the contents of a user config file to the parser.  These options
        take precedence over all other options.

        Parameters
        ----------
        filename : str
            The relative or absolute path to the config file
        """
        self._add(filename, user=True)

    def add_from_file(self, filename: str) -> None:
        """
        Add the contents of a config file to the parser.

        Parameters
        ----------
        filename : str
            The relative or absolute path to the config file
        """
        self._add(filename, user=False)

    def add_from_package(
        self,
        package: str | ModuleType,
        config_filename: str,
        exception: bool = True,
    ) -> None:
        """
        Add the contents of a config file to the parser.

        Parameters
        ----------
        package : str or Package
            The package where ``config_filename`` is found

        config_filename : str
            The name of the config file to add

        exception : bool, optional
            Whether to raise an exception if the config file isn't found
        """
        try:
            path = imp_res_files(package) / config_filename
            self._add(str(path), user=False)
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            if exception:
                raise

    def read_string(
        self,
        string: str,
        source: str = "<string>",
        user: bool = False,
    ) -> None:
        """
        Add config options read from a string.

        This mirrors :meth:`configparser.ConfigParser.read_string` while
        preserving comments, provenance and layered precedence.  Environment
        variables of the form ``${env:VAR}`` are interpolated, just as for
        :meth:`add_from_file`.

        Parameters
        ----------
        string : str
            The config contents to parse.

        source : str, optional
            A label used as the provenance "source" for these options.  A later
            call with the same ``source`` replaces the earlier layer.

        user : bool, optional
            Whether these options should be treated as user overrides that take
            precedence over all base options.
        """
        self._ingest(string.splitlines(keepends=True), source, user=user)

    def read_file(
        self,
        f: Any,
        source: str | None = None,
        user: bool = False,
    ) -> None:
        """
        Add config options read from an open file or iterable of lines.

        This mirrors :meth:`configparser.ConfigParser.read_file` while
        preserving comments, provenance and layered precedence.

        Parameters
        ----------
        f : iterable of str
            An open file object or any iterable yielding config lines.

        source : str, optional
            A label used as the provenance "source" for these options.  If not
            given, ``f.name`` is used when available, otherwise ``"<stream>"``.
            A later call with the same ``source`` replaces the earlier layer.

        user : bool, optional
            Whether these options should be treated as user overrides that take
            precedence over all base options.
        """
        if source is None:
            source = getattr(f, "name", "<stream>")
        self._ingest(list(f), cast(str, source), user=user)

    def read_dict(
        self,
        dictionary: Mapping[str, Mapping[str, Any]],
        source: str = "<dict>",
        user: bool = False,
    ) -> None:
        """
        Add config options read from a nested dictionary.

        This mirrors :meth:`configparser.ConfigParser.read_dict`.  Keys and
        values are converted to strings.  Options added this way have empty
        comments.

        Parameters
        ----------
        dictionary : Mapping[str, Mapping[str, Any]]
            A mapping of the form ``{section: {option: value, ...}, ...}``.

        source : str, optional
            A label used as the provenance "source" for these options.  If
            several calls share a source, later options overwrite earlier ones.

        user : bool, optional
            Whether these options should be treated as user overrides that take
            precedence over all base options.
        """
        config_dict = self._user_config if user else self._configs
        config = config_dict.setdefault(source, RawConfigParser())
        comments = self._comments.setdefault(source, {})
        for section, options in dictionary.items():
            section = str(section)
            if not config.has_section(section):
                config.add_section(section)
            comments.setdefault(section, "")
            for option, value in options.items():
                option = str(option).lower()
                config.set(section, option, str(value))
                comments[(section, option)] = ""
        self._invalidate()

    @overload
    def get(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
    ) -> str: ...

    @overload
    def get(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
        fallback: F,
    ) -> str | F: ...

    def get(self, section: str, option: str, **kwargs: Any) -> Any:
        """
        Get an option value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        raw : bool, optional
            Whether to return the raw value, leaving any references to other
            config options uninterpolated

        vars : dict, optional
            Options consulted before the config's own sections

        fallback : optional
            A value to return if the option is not present.  Without it, a
            missing option raises, as it does for
            :meth:`configparser.ConfigParser.get`.

        Returns
        -------
        value : str
            The value of the config option, or ``fallback`` if it was
            given and the option is not present
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        # ConfigParser.get is not precisely typed, so help mypy a bit
        value = combined.get(section, option, **kwargs)
        return cast(str, value)

    @overload
    def getint(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
    ) -> int: ...

    @overload
    def getint(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
        fallback: F,
    ) -> int | F: ...

    def getint(self, section: str, option: str, **kwargs: Any) -> Any:
        """
        Get an option integer value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        raw : bool, optional
            Whether to return the raw value, leaving any references to other
            config options uninterpolated

        vars : dict, optional
            Options consulted before the config's own sections

        fallback : optional
            A value to return if the option is not present.  Without it, a
            missing option raises, as it does for
            :meth:`configparser.ConfigParser.getint`.

        Returns
        -------
        value : int
            The value of the config option, or ``fallback`` if it was
            given and the option is not present
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        value = combined.getint(section, option, **kwargs)
        return cast(int, value)

    @overload
    def getfloat(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
    ) -> float: ...

    @overload
    def getfloat(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
        fallback: F,
    ) -> float | F: ...

    def getfloat(self, section: str, option: str, **kwargs: Any) -> Any:
        """
        Get an option float value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        raw : bool, optional
            Whether to return the raw value, leaving any references to other
            config options uninterpolated

        vars : dict, optional
            Options consulted before the config's own sections

        fallback : optional
            A value to return if the option is not present.  Without it, a
            missing option raises, as it does for
            :meth:`configparser.ConfigParser.getfloat`.

        Returns
        -------
        value : float
            The value of the config option, or ``fallback`` if it was
            given and the option is not present
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        value = combined.getfloat(section, option, **kwargs)
        return cast(float, value)

    @overload
    def getboolean(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
    ) -> bool: ...

    @overload
    def getboolean(
        self,
        section: str,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
        fallback: F,
    ) -> bool | F: ...

    def getboolean(self, section: str, option: str, **kwargs: Any) -> Any:
        """
        Get an option boolean value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        raw : bool, optional
            Whether to return the raw value, leaving any references to other
            config options uninterpolated

        vars : dict, optional
            Options consulted before the config's own sections

        fallback : optional
            A value to return if the option is not present.  Without it, a
            missing option raises, as it does for
            :meth:`configparser.ConfigParser.getboolean`.

        Returns
        -------
        value : bool
            The value of the config option, or ``fallback`` if it was
            given and the option is not present
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        value = combined.getboolean(section, option, **kwargs)
        return cast(bool, value)

    def explain(self, section: str, option: str) -> dict:
        """
        Explain the provenance of an option.

        Returns a dictionary with the effective value, the source file path,
        and which layer provided it ("user" or "base").

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        info : dict
            {"value": ..., "source": <path>, "layer": "user"|"base"}
        """
        if self.combined is None or self.sources is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        sources = cast(dict[tuple[str, str], str], self.sources)

        # This will raise if the section/option does not exist, mirroring
        # ConfigParser behavior
        value = combined.get(section, option)

        key = (section, option)
        if key not in sources:
            raise KeyError(f"No provenance found for {section}.{option}")
        source = sources[key]
        layer = "user" if source in self._user_config else "base"
        return {"value": value, "source": source, "layer": layer}

    @overload
    def getlist(
        self,
        section: str,
        option: str,
        dtype: Callable[[str], T] = ...,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
    ) -> list[T]: ...

    @overload
    def getlist(
        self,
        section: str,
        option: str,
        dtype: Callable[[str], T] = ...,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
        fallback: F,
    ) -> list[T] | F: ...

    def getlist(
        self,
        section: str,
        option: str,
        dtype: Callable[[str], T] = str,  # type: ignore[assignment]
        *,
        fallback: Any = _UNSET,
        **kwargs: Any,
    ) -> Any:
        """
        Get an option value as a list for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        dtype : {Type[str], Type[int], Type[float]}
            The type of the elements in the list.

        fallback : list, optional
            A value to return if the option is not found.  Without it, a
            missing option raises, as it does for :meth:`get` and the other
            typed getters.

        raw : bool, optional
            Whether to return the raw value, leaving any references to other
            config options uninterpolated

        vars : dict, optional
            Options consulted before the config's own sections

        fallback : optional
            A value to return if the option is not present.  Without it, a
            missing option raises, as it does for
            :meth:`configparser.ConfigParser.get`.

        Returns
        -------
        value : list
            The value of the config option parsed into a list, or ``fallback``
            if it was given and the option is not present.

        Raises
        ------
        configparser.NoSectionError
            If the section does not exist and no ``fallback`` was given.

        configparser.NoOptionError
            If the option does not exist and no ``fallback`` was given.
        """
        if fallback is _UNSET:
            # let ConfigParser raise, so that a missing option is reported the
            # same way here as it is by get(), getint() and the rest
            raw = self.get(section, option, **kwargs)
        else:
            raw = self.get(section, option, fallback=None, **kwargs)
            if raw is None:
                # a fallback is returned as given, without being parsed
                return fallback
        parts = raw.replace(',', ' ').split()
        items: list[T] = [dtype(value) for value in parts]
        return items

    def getexpression(
        self,
        section: str,
        option: str,
        dtype: type | None = None,
        backend: str | None = None,
        allow_numpy: bool = False,
        fallback: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Get an option as an expression (typically a list, though tuples and
        dicts are also available).  The expression is required to have valid
        python syntax, so that string entries are required to be in single or
        double quotes.

        Parameters
        ----------
        section : str
            The section in the config file

        option : str
            The option in the config file

        dtype : {Type[bool], Type[int], Type[float], Type[list], Type[tuple], Type[str]}, optional
            If supplied, each element in a list or tuple, or
            each value in a dictionary are cast to this type.  This is likely
            most useful for ensuring that all elements in a list of numbers are
            of type float, rather than int, when the distinction is important.

        backend : {"literal", "safe"} or None
            - None (default): choose "safe" if allow_numpy=True else "literal".
            - "literal": use ast.literal_eval (safest, literals only)
            - "safe": use a whitelisted AST evaluator optionally allowing numpy

        allow_numpy : bool, optional
            If True and backend="safe", enable limited numpy functions
            (np.arange, np.linspace, np.array) under names "np"/"numpy".

        fallback : Any, optional
            A fallback value to return if the option is not found.

        raw : bool, optional
            Whether to return the raw value, leaving any references to other
            config options uninterpolated

        vars : dict, optional
            Options consulted before the config's own sections

        fallback : optional
            A value to return if the option is not present.  Without it, a
            missing option raises, as it does for
            :meth:`configparser.ConfigParser.get`.
        """  # noqa: E501
        expression_string = self.get(section, option, fallback=None, **kwargs)

        if expression_string is None:
            return fallback

        # Adaptive backend selection
        if backend is None:
            backend = "safe" if allow_numpy else "literal"

        if backend == "literal":
            result = ast.literal_eval(expression_string)
        elif backend == "safe":
            result = Tranche._safe_eval(
                expression_string,
                allow_numpy=allow_numpy,
                extra_symbols=self._extra_symbols,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if dtype is not None:
            if isinstance(result, list):
                result = [dtype(element) for element in result]
            elif isinstance(result, tuple):
                result = tuple(dtype(element) for element in result)
            elif isinstance(result, dict):
                for key in result:
                    result[key] = dtype(result[key])

        return result

    # Convenience helper for numpy-enabled expressions
    def getnumpy(
        self,
        section: str,
        option: str,
        dtype: type | None = None,
        backend: str | None = None,
        fallback: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Shortcut for expressions requiring NumPy.

        Equivalent to ``getexpression(..., allow_numpy=True)`` with adaptive
        backend selection (safe if backend is None).

        Parameters
        ----------
        section : str
            Section name.
        option : str
            Option name.
        dtype : type, optional
            Cast list/tuple/dict elements to this type.
        backend : {"literal", "safe"} or None, optional
            Override backend.  None => choose "safe".
        fallback : Any, optional
            A fallback value to return if the option is not found.
        raw : bool, optional
            Whether to return the raw value, leaving any references to other
            config options uninterpolated

        vars : dict, optional
            Options consulted before the config's own sections

        fallback : optional
            A value to return if the option is not present.  Without it, a
            missing option raises, as it does for
            :meth:`configparser.ConfigParser.get`.
        """
        return self.getexpression(
            section,
            option,
            dtype=dtype,
            backend=backend,
            allow_numpy=True,
            fallback=fallback,
            **kwargs,
        )

    def register_symbol(self, name: str, obj: Any) -> None:
        """
        Make 'name' available to the safe expression backend.

        Restrictions:

        - name must not contain '__' or '.'

        - cannot override reserved core names: 'np', 'numpy', 'math',
          'range', 'int', 'float', 'pi'

        Examples
        --------
        Register a stdlib function (math.sqrt) and call it from the config:

        >>> import math
        >>> config = Tranche()
        >>> config.register_symbol('sqrt', math.sqrt)
        >>> # INI has:
        >>> #  [calc]
        >>> #  val = sqrt(9)
        >>> config.getexpression('calc', 'val', backend='safe')
        3.0

        Register a reducer (statistics.mean):

        >>> import statistics
        >>> config.register_symbol('mean', statistics.mean)
        >>> # INI has:
        >>> #  [data]
        >>> #  avg = mean([1, 2, 3])
        >>> config.getexpression('data', 'avg', backend='safe')
        2

        Register a type (datetime.timedelta) supporting keyword args:

        >>> from datetime import timedelta
        >>> config.register_symbol('timedelta', timedelta)
        >>> # INI has:
        >>> #  [sched]
        >>> #  interval = timedelta(days=2)
        >>> config.getexpression('sched', 'interval', backend='safe')
        datetime.timedelta(days=2)

        Parameters
        ----------
        name : str
            The name to register

        obj : any
            The object to register under the given name
        """
        if "__" in name or "." in name:
            raise ValueError("Symbol names cannot contain '__' or '.'")
        reserved = {"np", "numpy", "math", "range", "int", "float", "pi"}
        if name in reserved:
            raise ValueError(f"Cannot override reserved symbol: {name}")
        self._extra_symbols[name] = obj

    def has_section(self, section: str) -> bool:
        """
        Whether the given section is part of the config

        Parameters
        ----------
        section : str
            The name of the config section

        Returns
        -------
        found : bool
            Whether the option was found in the section
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.has_section(section)

    def has_option(self, section: str, option: str) -> bool:
        """
        Whether the given section has the given option

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        found : bool
            Whether the option was found in the section
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.has_option(section, option)

    def sections(self) -> list[str]:
        """
        Get the list of section names in the combined config.

        The special ``[DEFAULT]`` section is not included, mirroring
        :meth:`configparser.ConfigParser.sections`.

        Returns
        -------
        sections : list of str
            The names of the config sections
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.sections()

    def options(self, section: str) -> list[str]:
        """
        Get the list of option names available in the given section.

        Parameters
        ----------
        section : str
            The name of the config section

        Returns
        -------
        options : list of str
            The names of the options in the section
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.options(section)

    def defaults(self) -> Mapping[str, str]:
        """
        Get the options in the special ``[DEFAULT]`` section.

        Returns
        -------
        defaults : Mapping[str, str]
            The fallback options applied to every section
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.defaults()

    def items(
        self,
        section: str = _UNSET,
        **kwargs: Any,
    ) -> Any:
        """
        Get section or option items from the combined config.

        Mirrors :meth:`configparser.ConfigParser.items`.

        Parameters
        ----------
        section : str, optional
            If given, return a list of ``(option, value)`` pairs for that
            section.  If omitted, return a list of ``(name, section)`` pairs for
            every section, where each ``section`` is a
            :class:`tranche.section.Section` wrapper.

        **kwargs : Any
            Additional keyword arguments (e.g. ``raw``, ``vars``) forwarded to
            :meth:`configparser.ConfigParser.items` when ``section`` is given.

        Returns
        -------
        items : list
            Either ``(name, Section)`` pairs or ``(option, value)`` pairs.
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        if section is _UNSET:
            return [(name, self[name]) for name in combined.sections()]
        return combined.items(section, **kwargs)

    def keys(self) -> list[str]:
        """
        Get the section names in the combined config (including ``DEFAULT``).

        Returns
        -------
        keys : list of str
            The section names, mirroring mapping access via ``config[section]``
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return list(combined.keys())

    def values(self) -> list[Section]:
        """
        Get a :class:`tranche.section.Section` wrapper for each section.

        Returns
        -------
        values : list of tranche.section.Section
            The sections of the combined config
        """
        return [self[name] for name in self.keys()]

    def __contains__(self, section: object) -> bool:
        """Whether ``section`` is a section in the combined config."""
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return section in combined

    def __iter__(self) -> Iterator[str]:
        """Iterate over the section names (including ``DEFAULT``)."""
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return iter(combined)

    def __len__(self) -> int:
        """The number of sections (including ``DEFAULT``) in the config."""
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return len(combined)

    def set(
        self,
        section: str,
        option: str,
        value: str | None = None,
        comment: str | None = None,
        user: bool = False,
    ) -> None:
        """
        Set the value of the given option in the given section.  The file from
         which this function was called is also retained for provenance.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        value : str, optional
            The value to set the option to

        comment : str, optional
            A comment to include with the config option when it is written
            to a file

        user : bool, optional
            Whether this config option was supplied by the user (e.g. through
            a command-line flag) and should take priority over other sources

        Raises
        ------
        TypeError
            If ``value`` is neither a ``str`` nor ``None``.
        """
        option = option.lower()
        calling_frame = inspect.stack(context=2)[1]
        filename = os.path.abspath(calling_frame.filename)

        if value is not None and not isinstance(value, str):
            # ConfigParser stores the value as given and only objects when the
            # config is combined or written, by which point nothing says which
            # option was at fault or where it came from.  Both are known here.
            raise TypeError(
                f'The value for [{section}] {option} must be a str, not '
                f'{type(value).__name__}.  It was set at '
                f'{filename}:{calling_frame.lineno}.  Convert it with '
                f'str() at the call.'
            )

        if user:
            config_dict = self._user_config
        else:
            config_dict = self._configs
        if filename not in config_dict:
            config_dict[filename] = RawConfigParser()
        config = config_dict[filename]
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, option, value)
        self._invalidate()
        if filename not in self._comments:
            self._comments[filename] = dict()
        self._comments[filename].setdefault(section, '')
        if comment is None:
            comment = ''
        else:
            comment = ''.join([f'# {line}\n' for line in comment.split('\n')])
        self._comments[filename][(section, option)] = comment

    def add_section(self, section: str, user: bool = False) -> None:
        """
        Add an empty section.  The file from which this function was called is
        retained for provenance, mirroring :meth:`set`.

        Parameters
        ----------
        section : str
            The name of the config section to add

        user : bool, optional
            Whether this section was supplied by the user and should take
            priority over other sources

        Raises
        ------
        configparser.DuplicateSectionError
            If the section already exists in the calling file's layer
        """
        calling_frame = inspect.stack(context=2)[1]
        filename = os.path.abspath(calling_frame.filename)
        config_dict = self._user_config if user else self._configs
        if filename not in config_dict:
            config_dict[filename] = RawConfigParser()
        config_dict[filename].add_section(section)
        self._comments.setdefault(filename, {}).setdefault(section, '')
        self._invalidate()

    def remove_option(self, section: str, option: str) -> bool:
        """
        Remove an option from a section across all layers.

        The option is removed from every contributing file (both base and user
        layers) so that it no longer appears in the combined config.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        existed : bool
            Whether the option existed in any layer before removal

        Raises
        ------
        configparser.NoSectionError
            If the section is not present in any layer
        """
        option = option.lower()
        section_found = False
        existed = False
        for config_dict in (self._configs, self._user_config):
            for source, config in config_dict.items():
                if not config.has_section(section):
                    continue
                section_found = True
                if config.remove_option(section, option):
                    existed = True
                    self._comments.get(source, {}).pop((section, option), None)
        if not section_found:
            raise NoSectionError(section)
        if existed:
            self._invalidate()
        return existed

    def remove_section(self, section: str) -> bool:
        """
        Remove a section and all of its options across all layers.

        The section is removed from every contributing file (both base and user
        layers) so that it no longer appears in the combined config.

        Parameters
        ----------
        section : str
            The name of the config section

        Returns
        -------
        existed : bool
            Whether the section existed in any layer before removal
        """
        existed = False
        for config_dict in (self._configs, self._user_config):
            for source, config in config_dict.items():
                if config.remove_section(section):
                    existed = True
                comments = self._comments.get(source)
                if comments is None:
                    continue
                comments.pop(section, None)
                for key in [
                    key
                    for key in comments
                    if isinstance(key, tuple) and key[0] == section
                ]:
                    comments.pop(key, None)
        if existed:
            self._invalidate()
        return existed

    def __setitem__(self, section: str, options: Mapping[str, Any]) -> None:
        """
        Set the options of a section, replacing any existing section.

        Mirrors ``ConfigParser.__setitem__``: the existing section (in every
        layer) is removed first, then the provided options are added to a
        runtime layer.

        Parameters
        ----------
        section : str
            The name of the config section

        options : Mapping[str, Any]
            The options to set in the section
        """
        self.remove_section(section)
        for option, value in dict(options).items():
            self._set_runtime(section, option, value)

    def __delitem__(self, section: str) -> None:
        """
        Remove a section across all layers.

        Parameters
        ----------
        section : str
            The name of the config section

        Raises
        ------
        KeyError
            If the section does not exist
        """
        if section not in self:
            raise KeyError(section)
        self.remove_section(section)

    def write(
        self,
        fp: TextIO,
        include_sources: bool = True,
        include_comments: bool = True,
        raw: bool = True,
    ) -> None:
        """
        Write the config options to the given file pointer.

        Parameters
        ----------
        fp : typing.TextIO
            The file pointer to write to.

        include_sources : bool, optional
            Whether to include a comment above each option indicating the
            source file where it was defined

        include_comments : bool, optional
            Whether to include the original comments associated with each
            section or option

        raw : bool, optional
            Whether to write "raw" config options, rather than using extended
            interpolation
        """
        self.combine(raw=raw)
        combined = cast(CombinedParser, self.combined)
        combined_comments = cast(
            dict[str | tuple[str, str], str], self.combined_comments
        )
        sources = cast(dict[tuple[str, str], str], self.sources)
        for section in combined.sections():
            section_items = combined.items(section=section)
            if include_comments and section in combined_comments:
                fp.write(combined_comments[section])
            fp.write(f'[{section}]\n\n')
            for option, value in section_items:
                if include_comments:
                    fp.write(combined_comments[(section, option)])
                if include_sources:
                    source = sources[(section, option)]
                    fp.write(f'# source: {source}\n')
                value = str(value).replace('\n', '\n\t')
                if not raw:
                    value = value.replace('$', '$$')
                fp.write(f'{option} = {value}\n\n')
            fp.write('\n')

        if raw:
            # since we combined in "raw" mode, force recombining on future
            # access commands
            self.combined = None

    def list_files(self) -> list[str]:
        """
        Get a list of files contributing to the combined config options

        Returns
        -------
        filenames : list of str
            A list of file paths

        """
        filenames = list(self._configs.keys()) + list(self._user_config.keys())
        return filenames

    def copy(self) -> "Tranche":
        """
        Get a deep copy of the config parser

        Returns
        -------
        config_copy : tranche.Tranche
            The deep copy
        """
        config_copy = Tranche()
        for filename, config in self._configs.items():
            config_copy._configs[filename] = Tranche._deepcopy(config)

        for filename, config in self._user_config.items():
            config_copy._user_config[filename] = Tranche._deepcopy(config)

        config_copy._comments = dict(self._comments)
        return config_copy

    def append(self, other: "Tranche") -> None:
        """
        Append a deep copy of another config parser to this one.  Config
        options from ``other`` will take precedence over those from this config
        parser.

        Parameters
        ----------
        other : tranche.Tranche
                The other, higher priority config parser
        """
        other = other.copy()
        self._configs.update(other._configs)
        self._user_config.update(other._user_config)
        self._comments.update(other._comments)
        self._invalidate()

    def prepend(self, other: "Tranche") -> None:
        """
        Prepend a deep copy of another config parser to this one.  Config
        options from this config parser will take precedence over those from
        ``other``.

        Parameters
        ----------
        other : tranche.Tranche
                The other, higher priority config parser
        """
        other = other.copy()

        configs = dict(other._configs)
        configs.update(self._configs)
        self._configs = configs

        user_config = dict(other._user_config)
        user_config.update(self._user_config)
        self._user_config = user_config

        comments = dict(other._comments)
        comments.update(self._comments)
        self._comments = comments
        self._invalidate()

    def __getitem__(self, section: str) -> Section:
        """
        Get get the config options for a given section.

        Parameters
        ----------
        section : str
            The name of the section to retrieve.

        Returns
        -------
        section : tranche.section.Section
            The config options for the given section with tranche helpers.
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        proxy = combined[section]
        return Section(self, proxy, section)

    def combine(self, raw: bool = False) -> None:
        """
        Combine the config files into one.  This is normally handled
        automatically.

        Parameters
        ----------
        raw : bool, optional
            Whether to combine "raw" config options, rather than using extended
            interpolation
        """
        if raw:
            self.combined = RawConfigParser()
        else:
            self.combined = ConfigParser(interpolation=ExtendedInterpolation())
        self.sources = dict()
        self.combined_comments = dict()
        for configs in [self._configs, self._user_config]:
            for source, config in configs.items():
                for section in config.sections():
                    if section in self._comments[source]:
                        self.combined_comments[section] = self._comments[source][
                            section
                        ]
                    if not self.combined.has_section(section):
                        self.combined.add_section(section)
                    for option, value in config.items(section):
                        self.sources[(section, option)] = source
                        self.combined.set(section, option, value)
                        self.combined_comments[(section, option)] = self._comments[
                            source
                        ][(section, option)]

    def validate(self, validator: Callable[[dict[str, dict[str, str]]], None]) -> None:
        """
        Call a user-provided validator function on the config as a nested dict.

        The validator is called with a dictionary of the form:

            {section: {option: value, ...}, ...}

        where all values are strings as returned by `get()`.

        This allows integration with validation libraries such as Pydantic or
        voluptuous, or custom validation logic. If the validator raises an
        exception, it will propagate to the caller.

        Parameters
        ----------
        validator : Callable[[dict[str, dict[str, str]]], None]
            A function that takes a nested dictionary of config values and
            raises on error.

        Examples
        --------
        >>> def my_validator(cfg):
        ...     assert 'main' in cfg
        ...     assert 'foo' in cfg['main']
        >>> config.validate(my_validator)
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        config_dict: dict[str, dict[str, str]] = {}
        for section in combined.sections():
            config_dict[section] = {
                option: self.get(section, option)
                for option in combined.options(section)
            }
        validator(config_dict)

    def _add(self, filename: str, user: bool) -> None:
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Config file does not exist: {filename}')
        with open(filename, encoding="utf-8") as f:
            lines = f.readlines()
        self._ingest(lines, filename, user=user)

    def _ingest(self, lines: list[str], source: str, user: bool) -> None:
        """
        Parse ``lines`` of config text and store them under ``source``.

        Shared by :meth:`_add`, :meth:`read_string` and :meth:`read_file`.
        Environment variables (``${env:VAR}``) are interpolated and comments are
        preserved.
        """
        interpolated = [self._interpolate_env_vars(line) for line in lines]
        config = RawConfigParser()
        config.read_file(StringIO(''.join(interpolated)), source=source)
        # Parse comments from the original (un-interpolated) text
        comments = self._parse_comments(
            StringIO(''.join(lines)), source, comments_before=True
        )
        if user:
            self._user_config[source] = config
        else:
            self._configs[source] = config
        self._comments[source] = comments
        self._invalidate()

    def _set_runtime(self, section: str, option: str, value: Any) -> None:
        """
        Set a single option in the shared in-memory ``<dict>`` base layer.

        Used by mapping-style assignment (:meth:`__setitem__` and
        ``Section.__setitem__``) where there is no originating file to attribute.
        """
        source = '<dict>'
        config = self._configs.setdefault(source, RawConfigParser())
        comments = self._comments.setdefault(source, {})
        if not config.has_section(section):
            config.add_section(section)
            comments.setdefault(section, '')
        option = option.lower()
        config.set(section, option, str(value))
        comments[(section, option)] = ''
        self._invalidate()

    def _invalidate(self) -> None:
        """Reset cached combined state so it is rebuilt on next access."""
        self.combined = None
        self.combined_comments = None
        self.sources = None

    def _interpolate_env_vars(self, value: str) -> str:
        """
        Replace ${env:VAR} with the value of the environment variable VAR.
        """

        def replacer(match: 'Match[str]') -> str:
            var = match.group(1)
            return os.environ.get(var, '')

        pattern = re.compile(r'\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}')
        return pattern.sub(replacer, value)

    @staticmethod
    def _parse_comments(
        fp: TextIO, filename: str, comments_before: bool = True
    ) -> dict[str | tuple[str, str], str]:
        """
        Parse the comments in a config file into a dictionary.

        Parameters
        ----------
        fp : file-like
            Open file pointer to the config file.
        filename : str
            Name of the config file (for error messages).
        comments_before : bool, optional
            If True, associate comments before a section/option with that item.

        Returns
        -------
        comments : dict
            Mapping of (section, option) or section to comment strings.
        """
        comments: dict[str | tuple[str, str], str] = {}
        current_comment = ''
        section_name: str | None = None
        option_name: str | None = None
        indent_level = 0
        for line_number, line in enumerate(fp, start=1):
            value = line.strip()
            is_comment = value.startswith('#')
            if is_comment:
                current_comment = current_comment + line
            if len(value) == 0 or is_comment:
                # end of value
                indent_level = sys.maxsize
                continue

            cur_indent_level = len(line) - len(line.lstrip())
            is_continuation = cur_indent_level > indent_level
            # a section header or option header?
            if section_name is None or option_name is None or not is_continuation:
                indent_level = cur_indent_level
                # is it a section header?
                is_section = value.startswith('[') and value.endswith(']')
                if is_section:
                    if not comments_before:
                        if option_name is None:
                            if section_name is not None:
                                comments[section_name] = current_comment
                        else:
                            if section_name is not None:
                                comments[(section_name, option_name)] = current_comment
                    section_name = value[1:-1].strip()
                    option_name = None

                    if comments_before and section_name is not None:
                        comments[section_name] = current_comment
                    current_comment = ''
                # an option line?
                else:
                    delimiter_index = value.find('=')
                    if delimiter_index == -1:
                        raise ValueError(
                            f'Expected to find "=" on line {line_number} of {filename}'
                        )

                    if not comments_before:
                        if option_name is None:
                            if section_name is not None:
                                comments[section_name] = current_comment
                        else:
                            if section_name is not None and option_name is not None:
                                comments[(section_name, option_name)] = current_comment

                    option_name = value[:delimiter_index].strip().lower()
                    if comments_before and section_name is not None:
                        comments[(section_name, option_name)] = current_comment
                    current_comment = ''

        if not comments_before:
            if section_name is not None:
                comments[section_name] = current_comment
            if section_name is not None and option_name is not None:
                comments[(section_name, option_name)] = current_comment
        return comments

    @staticmethod
    def _deepcopy(config: RawConfigParser) -> RawConfigParser:
        config_copy = RawConfigParser()
        for section in config.sections():
            if not config_copy.has_section(section):
                config_copy.add_section(section)
            for option, value in config.items(section):
                config_copy.set(section, option, value)
        return config_copy
