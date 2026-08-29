from __future__ import annotations

from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from configparser import SectionProxy
from typing import TYPE_CHECKING, Any, TypeVar, overload

# Import Tranche only for type checking to avoid circular imports at runtime
if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .tranche import Tranche

T = TypeVar("T")
F = TypeVar("F")

# Sentinel distinguishing "argument not provided" from an explicit ``None``.
_UNSET: Any = object()


class Section:
    """
    Wrapper around ``configparser.SectionProxy`` exposing tranche helpers.

    Provides section-scoped convenience methods while delegating all
    other behavior to the underlying ``SectionProxy``.
    """

    def __init__(
        self,
        tranche: Tranche,
        proxy: SectionProxy,
        name: str,
    ) -> None:
        """
        Initialize a Section wrapper.

        Parameters
        ----------
        tranche : Tranche
            Parent configuration object providing helper methods.
        proxy : configparser.SectionProxy
            Underlying section proxy (unused except as a hint; a live proxy is
            re-derived on each access so the wrapper never goes stale after the
            combined config is rebuilt).
        name : str
            Name of the section represented by this wrapper.
        """
        self._tranche = tranche
        self._name = name

    @property
    def _proxy(self) -> SectionProxy:
        """The live ``SectionProxy`` from the parent's combined config.

        Re-derived on each access so the wrapper reflects mutations to the
        underlying :class:`Tranche` rather than a stale snapshot.
        """
        if self._tranche.combined is None:
            self._tranche.combine()
        combined = self._tranche.combined
        assert combined is not None  # for type checkers; combine() set it
        return combined[self._name]

    # ---- Convenience getters backed by Tranche methods ----

    # As in :meth:`tranche.Tranche.getlist`, the no-``dtype`` case gets its
    # own overloads rather than an elided default, which would leave ``T``
    # unsolved, and comes first so that it is the one matched.

    @overload
    def getlist(
        self,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
    ) -> list[str]: ...

    @overload
    def getlist(
        self,
        option: str,
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
        fallback: F,
    ) -> list[str] | F: ...

    @overload
    def getlist(
        self,
        option: str,
        dtype: Callable[[str], T],
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
    ) -> list[T]: ...

    @overload
    def getlist(
        self,
        option: str,
        dtype: Callable[[str], T],
        *,
        raw: bool = ...,
        vars: Mapping[str, str] | None = ...,
        fallback: F,
    ) -> list[T] | F: ...

    def getlist(
        self,
        option: str,
        dtype: Callable[[str], T] = str,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> Any:
        """
        Get an option value parsed as a list.

        Parameters
        ----------
        option : str
            Option name within this section.

        dtype : Callable[[str], T], optional
            Converter applied to each item. Defaults to ``str``.

        **kwargs : Any
            ``raw``, ``vars`` and ``fallback``, forwarded to
            :meth:`tranche.Tranche.getlist`.

        Returns
        -------
        list of T
            Parsed list with elements converted by ``dtype``, or ``fallback``
            if it was given and the option is not present.
        """
        return self._tranche.getlist(self._name, option, dtype=dtype, **kwargs)

    def getexpression(
        self,
        option: str,
        dtype: type | None = None,
        backend: str | None = None,
        allow_numpy: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Evaluate an option as a Python expression safely.

        Parameters
        ----------
        option : str
            Option name within this section.
        backend : {"literal", "safe"} or None, optional
            Evaluation backend. ``None`` chooses ``"safe"`` when
            ``allow_numpy`` is True, otherwise ``"literal"``.
        allow_numpy : bool, optional
            If True and using the "safe" backend, expose limited numpy
            functions under ``np``/``numpy``.
        **kwargs : Any
            ``raw``, ``vars`` and ``fallback``, forwarded to
            :meth:`tranche.Tranche.getexpression`.

        Returns
        -------
        Any
            Result of the evaluated expression, optionally cast.
        """
        return self._tranche.getexpression(
            self._name,
            option,
            dtype=dtype,
            backend=backend,
            allow_numpy=allow_numpy,
            **kwargs,
        )

    def getnumpy(
        self,
        option: str,
        dtype: type | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Evaluate an expression with NumPy enabled.

        Shortcut equivalent to ``getexpression(..., allow_numpy=True)``.

        Parameters
        ----------
        option : str
            Option name within this section.
        dtype : type, optional
            If provided, cast list/tuple elements or dict values.
        backend : {"literal", "safe"} or None, optional
            Backend override. ``None`` chooses ``"safe"``.
        **kwargs : Any
            ``raw``, ``vars`` and ``fallback``, forwarded to
            :meth:`tranche.Tranche.getnumpy`.

        Returns
        -------
        Any
            Result of the evaluated expression, optionally cast.
        """
        return self._tranche.getnumpy(
            self._name,
            option,
            dtype=dtype,
            backend=backend,
            **kwargs,
        )

    def explain(self, option: str) -> dict:
        """
        Explain provenance for an option in this section.

        Returns a dictionary with the effective value, the source file
        path, and which layer provided it ("user" or "base").

        Parameters
        ----------
        option : str
            Option name within this section.

        Returns
        -------
        dict
            Dictionary with keys ``{"value", "source", "layer"}``.
        """
        return self._tranche.explain(self._name, option)

    def has_option(self, option: str) -> bool:
        """
        Check whether this section contains an option.

        Parameters
        ----------
        option : str
            Option name to check.

        Returns
        -------
        bool
            True if the option exists, else False.
        """
        return self._tranche.has_option(self._name, option)

    @property
    def name(self) -> str:
        """The name of this section."""
        return self._name

    # ---- Mapping-like behavior and delegation ----

    def __getitem__(self, option: str) -> str:  # keep parity with SectionProxy
        return self._proxy[option]

    def __setitem__(self, option: str, value: Any) -> None:
        """
        Set a single option in this section via the parent :class:`Tranche`.

        The value is stored in tranche's runtime layer so that it survives
        recombination, unlike assigning directly to the underlying
        ``SectionProxy``.  Other layers and their provenance are left intact.
        """
        self._tranche._set_runtime(self._name, option, value)

    def __delitem__(self, option: str) -> None:
        """Remove an option from this section across all layers."""
        if not self._tranche.remove_option(self._name, option):
            raise KeyError(option)

    def __contains__(self, option: object) -> bool:
        return option in self._proxy

    def __iter__(self) -> Iterator[str]:
        return iter(self._proxy)

    def __len__(self) -> int:
        return len(self._proxy)

    def keys(self) -> KeysView[str]:
        """The option names in this section."""
        return self._proxy.keys()

    def values(self) -> ValuesView[str]:
        """The option values in this section."""
        return self._proxy.values()

    def items(self) -> ItemsView[str, str]:
        """The ``(option, value)`` pairs in this section."""
        return self._proxy.items()

    # ---- Mapping mutation routed through the parent Tranche ----
    #
    # ``SectionProxy`` is a ``MutableMapping``; its mixin mutators (``pop``,
    # ``popitem``, ``clear``, ``setdefault``, ``update``) would otherwise edit
    # the transient combined parser and be lost on the next ``combine()``.  We
    # override them to mutate tranche's layers so changes persist.

    def setdefault(self, option: str, value: Any = "") -> str:
        """
        Return ``option``'s value, setting it to ``value`` first if absent.

        Parameters
        ----------
        option : str
            Option name within this section.
        value : Any, optional
            Value to set (and return) if the option does not yet exist.

        Returns
        -------
        str
            The existing or newly set value.
        """
        if self._tranche.has_option(self._name, option):
            return self._proxy[option]
        self._tranche._set_runtime(self._name, option, value)
        return str(value)

    def update(
        self,
        other: Any = (),
        /,
        **kwargs: Any,
    ) -> None:
        """
        Update options from a mapping or iterable of pairs and/or keywords.

        Mirrors ``dict.update``; each value is stored in tranche's runtime
        layer.
        """
        if hasattr(other, "keys"):
            for option in other.keys():
                self._tranche._set_runtime(self._name, option, other[option])
        else:
            for option, value in other:
                self._tranche._set_runtime(self._name, option, value)
        for option, value in kwargs.items():
            self._tranche._set_runtime(self._name, option, value)

    def pop(self, option: str, default: Any = _UNSET) -> Any:
        """
        Remove ``option`` and return its value across all layers.

        Parameters
        ----------
        option : str
            Option name within this section.
        default : Any, optional
            Value to return if the option is absent.  If omitted, a
            ``KeyError`` is raised for a missing option.

        Returns
        -------
        Any
            The removed value, or ``default`` if the option was absent.
        """
        if not self._tranche.has_option(self._name, option):
            if default is _UNSET:
                raise KeyError(option)
            return default
        value = self._proxy[option]
        self._tranche.remove_option(self._name, option)
        return value

    def popitem(self) -> tuple[str, str]:
        """
        Remove and return an arbitrary ``(option, value)`` pair.

        Raises
        ------
        KeyError
            If the section has no options.
        """
        options = self._tranche.options(self._name)
        if not options:
            raise KeyError(f"section {self._name!r} is empty")
        option = options[-1]
        value = self._proxy[option]
        self._tranche.remove_option(self._name, option)
        return option, value

    def clear(self) -> None:
        """Remove all options from this section across all layers."""
        for option in self._tranche.options(self._name):
            self._tranche.remove_option(self._name, option)

    def __repr__(self) -> str:  # helpful debugging
        return f"Section(name={self._name!r}, proxy={self._proxy!r})"

    def __getattr__(self, name: str) -> Any:
        # Delegate attributes/methods not defined here to the underlying proxy
        return getattr(self._proxy, name)
