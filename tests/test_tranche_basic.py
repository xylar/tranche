import io
import textwrap
from pathlib import Path

import pytest

from tranche.tranche import Tranche


def write_tmp_cfg(tmp_path: Path, name: str, contents: str) -> str:
    path = tmp_path / name
    path.write_text(textwrap.dedent(contents).lstrip())
    return str(path)


def test_add_and_get_from_file(tmp_path: Path) -> None:
    base = write_tmp_cfg(
        tmp_path,
        "base.cfg",
        """
        [general]
        answer = 41
        name = base
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(base)

    assert cfg.get("general", "answer") == "41"
    assert cfg.getint("general", "answer") == 41
    assert cfg.get("general", "name") == "base"


def test_user_config_precedence(tmp_path: Path) -> None:
    base = write_tmp_cfg(
        tmp_path,
        "base.cfg",
        """
        [general]
        answer = 41
        name = base
        """,
    )
    user = write_tmp_cfg(
        tmp_path,
        "user.cfg",
        """
        [general]
        answer = 42
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(base)
    cfg.add_user_config(user)

    # user value overrides base
    assert cfg.getint("general", "answer") == 42
    # name remains from base (no override)
    assert cfg.get("general", "name") == "base"


def test_getlist_and_getexpression(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "cfg.cfg",
        """
        [nums]
        ints = 1, 2, 3
        expr = [1, 2, 3]
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    assert cfg.getlist("nums", "ints", int) == [1, 2, 3]

    # literal backend
    assert cfg.getexpression("nums", "expr", backend="literal") == [1, 2, 3]


def test_explain_reports_source(tmp_path: Path) -> None:
    base = write_tmp_cfg(
        tmp_path,
        "base.cfg",
        """
        [a]
        x = 1
        """,
    )
    user = write_tmp_cfg(
        tmp_path,
        "user.cfg",
        """
        [a]
        x = 2
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(base)
    cfg.add_user_config(user)

    info = cfg.explain("a", "x")
    assert info["value"] == "2"
    assert info["layer"] == "user"
    assert info["source"].endswith("user.cfg")


def test_set_and_write_roundtrip(tmp_path: Path) -> None:
    cfg = Tranche()
    cfg.set("sec", "opt", "val", comment="hello")

    out = io.StringIO()
    cfg.write(out)
    text = out.getvalue()

    assert "[sec]" in text
    assert "opt = val" in text
    assert "# hello" in text


def test_safe_eval_with_numpy_disabled(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "cfg.cfg",
        """
        [expr]
        val = [1, 2, 3]
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    # safe backend without numpy should work for pure literals
    assert cfg.getexpression("expr", "val", backend="safe") == [1, 2, 3]


@pytest.mark.parametrize("backend", ["literal", "safe"])
@pytest.mark.parametrize("dtype", [int, float, str])
def test_getexpression_dtype_casting(tmp_path: Path, backend: str, dtype: type) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "cfg.cfg",
        """
        [vals]
        arr = [1, 2, 3]
        tup = (1, 2, 3)
        dct = {"a": 1, "b": 2}
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    res = cfg.getexpression("vals", "arr", backend=backend, dtype=dtype)
    assert isinstance(res, list)
    for v in res:
        assert isinstance(v, dtype)

    res = cfg.getexpression("vals", "tup", backend=backend, dtype=dtype)
    assert isinstance(res, tuple)
    for v in res:
        assert isinstance(v, dtype)

    res = cfg.getexpression("vals", "dct", backend=backend, dtype=dtype)
    assert isinstance(res, dict)
    for v in res.values():
        assert isinstance(v, dtype)


def test_extended_interpolation_section_option(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "inter.cfg",
        """
        [base]
        foo = hello
        bar = ${base:foo}
        baz = ${base:bar} world
        """,
    )
    cfg = Tranche()
    cfg.add_from_file(cfg_path)
    assert cfg.get("base", "bar") == "hello"
    assert cfg.get("base", "baz") == "hello world"


def test_get_kwargs_fallback(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "fallback.cfg",
        """
        [base]
        present = 1
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    # Direct Tranche getters with fallback
    assert cfg.get("base", "missing", fallback="x") == "x"
    assert cfg.getint("base", "missing", fallback=5) == 5
    assert cfg.getfloat("base", "missing", fallback=1.5) == 1.5
    assert cfg.getboolean("base", "missing", fallback=True) is True

    # Section helpers should also support kwargs through to Tranche
    base = cfg["base"]
    assert base.get("missing", fallback="y") == "y"


def test_getlist_and_getexpression_nonstring_fallback(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "fallback_expr.cfg",
        """
        [vals]
        list = 1, 2, 3
        expr = [1, 2, 3]
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    # getlist with pre-parsed iterable fallback should return that
    # value directly
    assert cfg.getlist("vals", "missing", fallback=["4", "5"]) == ["4", "5"]

    # getexpression with string fallback should return the fallback unchanged
    assert (
        cfg.getexpression(
            "vals",
            "missing",
            backend="literal",
            fallback="[8, 9]",
        )
        == "[8, 9]"
    )

    # getexpression with already-evaluated fallback should return it
    fallback_value = (10, 11)
    assert (
        cfg.getexpression(
            "vals",
            "missing",
            backend="literal",
            fallback=fallback_value,
        )
        is fallback_value
    )


def test_getlist_and_getexpression_ignore_fallback_when_present(
    tmp_path: Path,
) -> None:
    """Fallback is ignored when the option exists in the config."""

    cfg_path = write_tmp_cfg(
        tmp_path,
        "present_expr.cfg",
        """
        [vals]
        list = 1, 2, 3
        expr = [1, 2, 3]
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    # getlist should return the parsed config value, not the fallback
    assert cfg.getlist("vals", "list", int, fallback=[4, 5]) == [1, 2, 3]

    # getexpression should evaluate the config value, ignoring fallback
    assert cfg.getexpression(
        "vals",
        "expr",
        backend="literal",
        fallback=[9, 9, 9],
    ) == [1, 2, 3]


def test_safe_eval_rejects_dunder_and_nonwhitelisted_attr(
    tmp_path: Path,
) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "expr.cfg",
        """
        [expr]
        val1 = (1).__class__
        val2 = (1).__add__(2)
        val3 = (1).real
        """,
    )
    cfg = Tranche()
    cfg.add_from_file(cfg_path)
    # Dunder name usage should be rejected
    with pytest.raises(ValueError):
        cfg.getexpression("expr", "val1", backend="safe")
    with pytest.raises(ValueError):
        cfg.getexpression("expr", "val2", backend="safe")
    # Attribute chain on non-whitelisted object should be rejected
    with pytest.raises(ValueError):
        cfg.getexpression("expr", "val3", backend="safe")
