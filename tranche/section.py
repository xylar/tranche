from __future__ import annotations

from collections.abc import Callable, Iterator
from configparser import SectionProxy
from typing import TYPE_CHECKING, Any, TypeVar, cast

# Import Tranche only for type checking to avoid circular imports at runtime
if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .tranche import Tranche

T = TypeVar("T")


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
            Underlying section proxy to delegate standard behavior to.
        name : str
            Name of the section represented by this wrapper.
        """
        self._tranche = tranche
        self._proxy = proxy
        self._name = name

    # ---- Convenience getters backed by Tranche methods ----

    def getlist(
        self,
        option: str,
        dtype: Callable[[str], T] | None = None,
    ) -> list[T]:
        """
        Get an option value parsed as a list.

        Parameters
        ----------
        option : str
            Option name within this section.
        dtype : Callable[[str], T], optional
            Converter applied to each item. Defaults to ``str``.

        Returns
        -------
        list of T
            Parsed list with elements converted by ``dtype``.
        """
        if dtype is None:
            dtype = cast(Callable[[str], T], str)
        return self._tranche.getlist(self._name, option, dtype=dtype)

    def getexpression(
        self,
        option: str,
        dtype: type | None = None,
        backend: str | None = None,
        allow_numpy: bool = False,
    ) -> Any:
        """
        Evaluate an option as a Python expression safely.

        Parameters
        ----------
        option : str
            Option name within this section.
        dtype : type, optional
            If provided, cast list/tuple elements or dict values.
        backend : {"literal", "safe"} or None, optional
            Evaluation backend. ``None`` chooses ``"safe"`` when
            ``allow_numpy`` is True, otherwise ``"literal"``.
        allow_numpy : bool, optional
            If True and using the "safe" backend, expose limited numpy
            functions under ``np``/``numpy``.

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
        )

    def getnumpy(
        self,
        option: str,
        dtype: type | None = None,
        backend: str | None = None,
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

        Returns
        -------
        Any
            Result of the evaluated expression, optionally cast.
        """
        return self._tranche.getnumpy(self._name, option, dtype=dtype, backend=backend)

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

    # ---- Mapping-like behavior and delegation ----

    def __getitem__(self, option: str) -> str:  # keep parity with SectionProxy
        return self._proxy[option]

    def __contains__(self, option: object) -> bool:
        return option in self._proxy

    def __iter__(self) -> Iterator[str]:
        return iter(self._proxy)

    def __len__(self) -> int:
        return len(self._proxy)

    def __repr__(self) -> str:  # helpful debugging
        return f"Section(name={self._name!r}, proxy={self._proxy!r})"

    def __getattr__(self, name: str) -> Any:
        # Delegate attributes/methods not defined here to the underlying proxy
        return getattr(self._proxy, name)
