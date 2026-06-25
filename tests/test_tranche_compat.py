"""Tests for ConfigParser-compatibility additions to Tranche and Section."""

import io
import textwrap
from configparser import DuplicateSectionError, NoSectionError
from pathlib import Path

import pytest

from tranche.section import Section
from tranche.tranche import Tranche


def make_config() -> Tranche:
    cfg = Tranche()
    cfg.read_string(
        textwrap.dedent(
            """
            [alpha]
            x = 1
            y = 2

            [beta]
            z = 3
            """
        ).lstrip(),
        source="base.cfg",
    )
    return cfg


# ---- Read loaders ---------------------------------------------------------


def test_read_string_with_env_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANCHE_TEST_VAR", "hello")
    cfg = Tranche()
    cfg.read_string("[s]\nv = ${env:TRANCHE_TEST_VAR}\n")
    assert cfg.get("s", "v") == "hello"


def test_read_dict_and_precedence() -> None:
    cfg = Tranche()
    cfg.read_dict({"main": {"a": 1, "b": 2}})
    cfg.read_dict({"main": {"a": 99}}, source="<override>", user=True)
    assert cfg.get("main", "a") == "99"
    assert cfg.getint("main", "b") == 2


def test_read_file_uses_stream_name(tmp_path: Path) -> None:
    cfg = Tranche()
    stream = io.StringIO("[s]\nv = 5\n")
    cfg.read_file(stream, source="mem.cfg")
    assert cfg.get("s", "v") == "5"
    assert cfg.explain("s", "v")["source"] == "mem.cfg"


# ---- Query methods --------------------------------------------------------


def test_sections_options_items() -> None:
    cfg = make_config()
    assert cfg.sections() == ["alpha", "beta"]
    assert cfg.options("alpha") == ["x", "y"]
    assert cfg.items("alpha") == [("x", "1"), ("y", "2")]

    all_items = cfg.items()
    names = [name for name, _ in all_items]
    assert names == ["alpha", "beta"]
    assert all(isinstance(sec, Section) for _, sec in all_items)


def test_keys_values_defaults() -> None:
    cfg = make_config()
    assert cfg.keys() == ["DEFAULT", "alpha", "beta"]
    assert [s.name for s in cfg.values()] == ["DEFAULT", "alpha", "beta"]
    assert dict(cfg.defaults()) == {}


# ---- Container protocol ---------------------------------------------------


def test_contains_iter_len() -> None:
    cfg = make_config()
    assert "alpha" in cfg
    assert "missing" not in cfg
    assert list(iter(cfg)) == ["DEFAULT", "alpha", "beta"]
    # DEFAULT is counted by ConfigParser, so two sections -> len 3
    assert len(cfg) == 3


# ---- Mutation: add/remove across layers -----------------------------------


def test_add_section_and_duplicate() -> None:
    cfg = make_config()
    cfg.add_section("gamma")
    assert "gamma" in cfg.sections()
    with pytest.raises(DuplicateSectionError):
        cfg.add_section("gamma")


def test_remove_option_across_layers() -> None:
    cfg = make_config()
    # user layer also defines alpha.x
    cfg.read_dict({"alpha": {"x": 100}}, source="<user>", user=True)
    assert cfg.get("alpha", "x") == "100"
    assert cfg.remove_option("alpha", "x") is True
    # removed from every layer, so it is gone entirely
    assert not cfg.has_option("alpha", "x")
    assert cfg.remove_option("alpha", "x") is False


def test_remove_option_missing_section_raises() -> None:
    cfg = make_config()
    with pytest.raises(NoSectionError):
        cfg.remove_option("nope", "x")


def test_remove_section_returns_bool_and_clears_comments() -> None:
    cfg = make_config()
    assert cfg.remove_section("beta") is True
    assert "beta" not in cfg.sections()
    assert cfg.remove_section("beta") is False
    # surviving section is unaffected
    assert cfg.get("alpha", "x") == "1"


# ---- Mapping mutation -----------------------------------------------------


def test_setitem_replaces_section() -> None:
    cfg = make_config()
    cfg["alpha"] = {"only": "kept"}
    assert cfg.options("alpha") == ["only"]
    assert cfg.get("alpha", "only") == "kept"


def test_delitem_section() -> None:
    cfg = make_config()
    del cfg["beta"]
    assert "beta" not in cfg
    with pytest.raises(KeyError):
        del cfg["beta"]


def test_section_setitem_survives_recombination() -> None:
    cfg = make_config()
    section = cfg["alpha"]
    section["new"] = "val"
    # mutate the parent so the combined config is rebuilt
    cfg.remove_section("beta")
    # the cached Section must still see the new value (live proxy)
    assert section["new"] == "val"
    assert cfg.get("alpha", "new") == "val"


def test_section_mapping_mutators() -> None:
    cfg = make_config()
    section = cfg["alpha"]

    assert section.setdefault("x") == "1"
    assert section.setdefault("brand", "new") == "new"
    assert cfg.get("alpha", "brand") == "new"

    section.update({"p": 1}, q=2)
    assert cfg.get("alpha", "p") == "1"
    assert cfg.get("alpha", "q") == "2"

    assert section.pop("p") == "1"
    assert section.pop("absent", "fallback") == "fallback"
    with pytest.raises(KeyError):
        section.pop("absent")

    option, _ = section.popitem()
    assert option not in cfg.options("alpha")

    section.clear()
    assert cfg.options("alpha") == []


def test_section_keys_values_items() -> None:
    cfg = make_config()
    section = cfg["alpha"]
    assert list(section.keys()) == ["x", "y"]
    assert list(section.values()) == ["1", "2"]
    assert list(section.items()) == [("x", "1"), ("y", "2")]
    assert section.name == "alpha"
