import textwrap
from pathlib import Path

import pytest

from tranche.section import Section
from tranche.tranche import Tranche


def write_tmp_cfg(tmp_path: Path, name: str, contents: str) -> str:
    path = tmp_path / name
    path.write_text(textwrap.dedent(contents).lstrip())
    return str(path)


def test_getitem_returns_section_wrapper(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "basic.cfg",
        """
        [main]
        a = 1
        b = 2
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    sec = cfg["main"]
    assert isinstance(sec, Section)
    # Delegated mapping behavior
    assert sec["a"] == "1"
    assert "b" in sec
    assert len(sec) == 2
    assert sorted(list(sec)) == ["a", "b"]
    # Delegated SectionProxy API still works
    assert sec.getint("a") == 1


def test_section_getlist(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "list.cfg",
        """
        [vals]
        items = 1, 2, 3
        words = foo bar baz
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)
    sec = cfg["vals"]

    assert sec.getlist("items", dtype=int) == [1, 2, 3]
    assert sec.getlist("words") == ["foo", "bar", "baz"]


def test_section_getexpression_literal_and_cast(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "expr.cfg",
        """
        [expr]
        x = [1, 2, 3]
        y = {"a": 1, "b": 2}
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)
    sec = cfg["expr"]

    x = sec.getexpression("x")
    assert isinstance(x, list) and x == [1, 2, 3]

    xf = sec.getexpression("x", dtype=float)
    assert xf == [1.0, 2.0, 3.0]

    y = sec.getexpression("y", dtype=str)
    assert y == {"a": "1", "b": "2"}


def test_section_getnumpy_and_backend_default(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    cfg_path = write_tmp_cfg(
        tmp_path,
        "np.cfg",
        """
        [expr]
        arr = np.arange(4)
        grid = numpy.linspace(0, 1, 3)
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)
    sec = cfg["expr"]

    arr = sec.getnumpy("arr")
    assert np.array_equal(arr, np.arange(4))

    # backend None with allow_numpy=True inside helper should choose safe
    grid = sec.getnumpy("grid")
    assert np.allclose(grid, np.linspace(0, 1, 3))


def test_section_explain_and_has_option(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "prov.cfg",
        """
        [prov]
        a = 10
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)
    sec = cfg["prov"]

    assert sec.has_option("a") is True
    info = sec.explain("a")
    assert info["value"] == "10"
    assert info["layer"] == "base"
    assert info["source"].endswith("prov.cfg")
