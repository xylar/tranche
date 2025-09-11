import ast
import math
import inspect
import os
import sys
from configparser import (
    ConfigParser,
    ExtendedInterpolation,
    RawConfigParser,
    SectionProxy,
)
from importlib.resources import files as imp_res_files
from io import StringIO
from types import ModuleType
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    TextIO,
    Type,
    TypeVar,
    Callable,
)

import numpy as np

CombinedParser = Union[ConfigParser, RawConfigParser]
T = TypeVar("T")


class LayeredConfig:
    """
    A "meta" config parser that keeps a dictionary of config parsers and their
    sources to combine when needed.  The custom config parser allows provenance
    of the source of different config options and allows the "user" config
    options to always take precedence over other config options (even if they
    are added later).

    Example
    -------
    >>> config = LayeredConfig()
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

        def __init__(self, allowed: Dict[str, Any]):
            self._allowed = dict(allowed)

        def __getattr__(self, name: str) -> Any:
            if "__" in name:
                raise ValueError("Use of dunder attributes is not allowed")
            if name not in self._allowed:
                raise NameError(f"Attribute '{name}' is not allowed")
            return self._allowed[name]

    @staticmethod
    def _default_symbols(allow_numpy: bool) -> Dict[str, Any]:
        """
        Default symbol table for the safe expression evaluator.

        If allow_numpy is True, expose a constrained 'np'/'numpy' namespace
        with only a few safe callables.
        """
        symbols: Dict[str, Any] = {
            "range": range,
            "int": int,
            "float": float,
            "pi": math.pi,
            # Expose 'math' with only 'pi' to start;
            # extend later via register_symbol
            "math": LayeredConfig._SafeNamespace({"pi": math.pi}),
        }
        if allow_numpy:
            np_ns = LayeredConfig._SafeNamespace(
                {
                    "arange": np.arange,
                    "linspace": np.linspace,
                    "array": np.array,
                }
            )
            symbols.update({
                "np": np_ns,
                "numpy": np_ns,
            })
        return symbols

    @staticmethod
    def _safe_eval(expression: str, *, allow_numpy: bool = False) -> Any:
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

        symbols = LayeredConfig._default_symbols(allow_numpy)

        def eval_node(node: ast.AST) -> Any:
            if not isinstance(node, allowed_nodes):
                raise ValueError(
                    f"Unsupported expression element: "
                    f"{type(node).__name__}"
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
                keys = [
                    eval_node(cast(ast.AST, k))  # type: ignore[arg-type]
                    for k in node.keys
                ]
                vals = [eval_node(v) for v in node.values]
                return {k: v for k, v in zip(keys, vals)}
            if isinstance(node, ast.Name):
                if "__" in node.id:
                    raise ValueError("Use of dunder names is not allowed")
                if node.id not in symbols:
                    raise NameError(f"Unknown name: {node.id}")
                return symbols[node.id]
            if isinstance(node, ast.Attribute):
                value = eval_node(node.value)
                if not isinstance(value, LayeredConfig._SafeNamespace):
                    raise ValueError(
                        "Attribute access only allowed on safe "
                        "namespaces"
                    )
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
                    return left ** right
                raise ValueError("Unsupported binary operator")

            # Should be unreachable due to the isinstance gate
            raise ValueError(f"Unsupported node: {type(node).__name__}")

        return eval_node(tree)

    def __init__(self) -> None:
        """
        Make a new (empty) config parser
        """
        # per-file configs and comments
        self._configs: Dict[str, RawConfigParser] = {}
        self._user_config: Dict[str, RawConfigParser] = {}
        self._comments: Dict[
            str, Dict[Union[str, Tuple[str, str]], str]
        ] = {}

        # combined state
        self.combined: Optional[Union[ConfigParser, RawConfigParser]] = None
        self.combined_comments: Optional[
            Dict[Union[str, Tuple[str, str]], str]
        ] = None
        self.sources: Optional[Dict[Tuple[str, str], str]] = None

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
        package: Union[str, ModuleType],
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

    def get(self, section: str, option: str) -> str:
        """
        Get an option value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : str
            The value of the config option
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.get(section, option)

    def getint(self, section: str, option: str) -> int:
        """
        Get an option integer value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : int
            The value of the config option
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.getint(section, option)

    def getfloat(self, section: str, option: str) -> float:
        """
        Get an option float value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : float
            The value of the config option
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.getfloat(section, option)

    def getboolean(self, section: str, option: str) -> bool:
        """
        Get an option boolean value for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        Returns
        -------
        value : bool
            The value of the config option
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined.getboolean(section, option)

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
        sources = cast(Dict[Tuple[str, str], str], self.sources)

        # This will raise if the section/option does not exist, mirroring
        # ConfigParser behavior
        value = combined.get(section, option)

        key = (section, option)
        if key not in sources:
            raise KeyError(f"No provenance found for {section}.{option}")
        source = sources[key]
        layer = "user" if source in self._user_config else "base"
        return {"value": value, "source": source, "layer": layer}

    def getlist(
        self, section: str, option: str, dtype: Callable[[str], T] = str
    ) -> List[T]:
        """
        Get an option value as a list for a given section.

        Parameters
        ----------
        section : str
            The name of the config section

        option : str
            The name of the config option

        dtype : {Type[str], Type[int], Type[float]}
            The type of the elements in the list

        Returns
        -------
        value : list
            The value of the config option parsed into a list
        """
        values = self.get(section, option)
        values = [dtype(value) for value in values.replace(',', ' ').split()]
        return values

    def getexpression(
        self,
        section: str,
        option: str,
        dtype: Optional[Type] = None,
        backend: str = "literal",
        allow_numpy: Optional[bool] = None,
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

        backend : {"literal", "safe"}
            - "literal": use ast.literal_eval (default, safest)
            - "safe": use a whitelisted AST evaluator optionally allowing numpy

        allow_numpy : bool, optional
            If True and backend="safe", enable limited numpy functions
            (np.arange, np.linspace, np.array) under names "np"/"numpy".
        """  # noqa: E501
        expression_string = self.get(section, option)

        if allow_numpy is None:
            allow_numpy = False

        if backend == "literal":
            result = ast.literal_eval(expression_string)
        elif backend == "safe":
            result = LayeredConfig._safe_eval(
                expression_string, allow_numpy=allow_numpy
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

    def set(
        self,
        section: str,
        option: str,
        value: Optional[str] = None,
        comment: Optional[str] = None,
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
        """
        option = option.lower()
        calling_frame = inspect.stack(context=2)[1]
        filename = os.path.abspath(calling_frame.filename)

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
        self.combined = None
        self.combined_comments = None
        self.sources = None
        if filename not in self._comments:
            self._comments[filename] = dict()
        if comment is None:
            comment = ''
        else:
            comment = ''.join([f'# {line}\n' for line in comment.split('\n')])
        self._comments[filename][(section, option)] = comment

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
            Dict[Union[str, Tuple[str, str]], str], self.combined_comments
        )
        sources = cast(Dict[Tuple[str, str], str], self.sources)
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

    def list_files(self) -> List[str]:
        """
        Get a list of files contributing to the combined config options

        Returns
        -------
        filenames : list of str
            A list of file paths

        """
        filenames = list(self._configs.keys()) + list(self._user_config.keys())
        return filenames

    def copy(self) -> "LayeredConfig":
        """
        Get a deep copy of the config parser

        Returns
        -------
        config_copy : layeredconfig.LayeredConfig
            The deep copy
        """
        config_copy = LayeredConfig()
        for filename, config in self._configs.items():
            config_copy._configs[filename] = LayeredConfig._deepcopy(config)

        for filename, config in self._user_config.items():
            config_copy._user_config[filename] = LayeredConfig._deepcopy(
                config
            )

        config_copy._comments = dict(self._comments)
        return config_copy

    def append(self, other: "LayeredConfig") -> None:
        """
        Append a deep copy of another config parser to this one.  Config
        options from ``other`` will take precedence over those from this config
        parser.

        Parameters
        ----------
    other : layeredconfig.LayeredConfig
            The other, higher priority config parser
        """
        other = other.copy()
        self._configs.update(other._configs)
        self._user_config.update(other._user_config)
        self._comments.update(other._comments)
        self.combined = None
        self.combined_comments = None
        self.sources = None

    def prepend(self, other: "LayeredConfig") -> None:
        """
        Prepend a deep copy of another config parser to this one.  Config
        options from this config parser will take precedence over those from
        ``other``.

        Parameters
        ----------
    other : layeredconfig.LayeredConfig
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
        self.combined = None
        self.combined_comments = None
        self.sources = None

    def __getitem__(self, section: str) -> SectionProxy:
        """
        Get get the config options for a given section.

        Parameters
        ----------
        section : str
            The name of the section to retrieve.

        Returns
        -------
        section_proxy : configparser.SectionProxy
            The config options for the given section.
        """
        if self.combined is None:
            self.combine()
        combined = cast(CombinedParser, self.combined)
        return combined[section]

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
                        self.combined_comments[section] = self._comments[
                            source
                        ][section]
                    if not self.combined.has_section(section):
                        self.combined.add_section(section)
                    for option, value in config.items(section):
                        self.sources[(section, option)] = source
                        self.combined.set(section, option, value)
                        self.combined_comments[(section, option)] = (
                            self._comments[source][(section, option)]
                        )

    def _add(self, filename: str, user: bool) -> None:
        filename = os.path.abspath(filename)
        config = RawConfigParser()
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Config file does not exist: {filename}')
        config.read(filenames=filename)
        with open(filename) as fp:
            comments = self._parse_comments(fp, filename, comments_before=True)

        if user:
            self._user_config[filename] = config
        else:
            self._configs[filename] = config
        self._comments[filename] = comments
        self.combined = None
        self.combined_comments = None
        self.sources = None

    @staticmethod
    def _parse_comments(
        fp: TextIO, filename: str, comments_before: bool = True
    ) -> Dict[Union[str, Tuple[str, str]], str]:
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
        comments = dict()
        current_comment = ''
        section_name = None
        option_name = None
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
            if (
                section_name is None
                or option_name is None
                or not is_continuation
            ):
                indent_level = cur_indent_level
                # is it a section header?
                is_section = value.startswith('[') and value.endswith(']')
                if is_section:
                    if not comments_before:
                        if option_name is None:
                            comments[section_name] = current_comment
                        else:
                            comments[(section_name, option_name)] = (
                                current_comment
                            )
                    section_name = value[1:-1].strip()
                    option_name = None

                    if comments_before:
                        comments[section_name] = current_comment
                    current_comment = ''
                # an option line?
                else:
                    delimiter_index = value.find('=')
                    if delimiter_index == -1:
                        raise ValueError(
                            f'Expected to find "=" on line '
                            f'{line_number} of {filename}'
                        )

                    if not comments_before:
                        if option_name is None:
                            comments[section_name] = current_comment
                        else:
                            comments[(section_name, option_name)] = (
                                current_comment
                            )

                    option_name = value[:delimiter_index].strip().lower()

                    if comments_before:
                        comments[(section_name, option_name)] = current_comment
                    current_comment = ''

        return comments

    @staticmethod
    def _deepcopy(
        config: Union[ConfigParser, RawConfigParser]
    ) -> ConfigParser:
        """
        Make a deep copy of the ConfigParser object.

        Parameters
        ----------
        config : configparser.ConfigParser
            The config parser to copy.

        Returns
        -------
        new_config : configparser.ConfigParser
            A deep copy of the config parser.
        """
        config_string = StringIO()
        config.write(config_string)
        # We must reset the buffer to make it ready for reading.
        config_string.seek(0)
        new_config = ConfigParser()
        new_config.read_file(config_string)
        return new_config
