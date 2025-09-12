import os
import tempfile

import pytest

from tranche import Tranche


def test_env_var_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_VAR", "env_value")
    cfg: Tranche = Tranche()
    with tempfile.NamedTemporaryFile("w+t", delete=False) as f:
        f.write("""
[main]
foo = ${env:MY_TEST_VAR}
bar = static
""")
        fname: str = f.name
    try:
        cfg.add_from_file(fname)
        assert cfg.get("main", "foo") == "env_value"
        assert cfg.get("main", "bar") == "static"
    finally:
        os.remove(fname)


def test_validation_hook_success() -> None:
    cfg: Tranche = Tranche()
    with tempfile.NamedTemporaryFile("w+t", delete=False) as f:
        f.write("""
[main]
foo = 123
bar = abc
""")
        fname: str = f.name
    try:
        cfg.add_from_file(fname)

        def validator(d: dict[str, dict[str, str]]) -> None:
            assert "main" in d
            assert d["main"]["foo"] == "123"
            assert d["main"]["bar"] == "abc"

        cfg.validate(validator)
    finally:
        os.remove(fname)


def test_validation_hook_failure() -> None:
    cfg: Tranche = Tranche()
    with tempfile.NamedTemporaryFile("w+t", delete=False) as f:
        f.write("""
[main]
foo = 123
bar = abc
""")
        fname: str = f.name
    try:
        cfg.add_from_file(fname)

        def validator(d: dict[str, dict[str, str]]) -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            cfg.validate(validator)
    finally:
        os.remove(fname)
