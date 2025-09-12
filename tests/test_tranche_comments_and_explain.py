import io
import os
import textwrap
from configparser import NoOptionError, NoSectionError
from pathlib import Path

import pytest

from tranche.tranche import Tranche


def write_tmp_cfg(tmp_path: Path, name: str, contents: str) -> str:
    path = tmp_path / name
    path.write_text(textwrap.dedent(contents).lstrip())
    return str(path)


def test_write_preserves_comments_and_sources(tmp_path: Path) -> None:
    base = write_tmp_cfg(
        tmp_path,
        "base.cfg",
        """
        # section comment line 1
        # section comment line 2
        [sec]

        # opt1 comment line 1
        # opt1 comment line 2
        opt1 = 1

        # opt2 comment
        opt2 = 2
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(base)

    out = io.StringIO()
    cfg.write(out, include_sources=True, include_comments=True, raw=True)
    text = out.getvalue()

    # section comments appear before the section header
    sec_idx = text.index("[sec]")
    assert text.index("# section comment line 1") < sec_idx
    assert text.index("# section comment line 2") < sec_idx

    # option comments appear before their respective source lines
    opt1_cmt_idx = text.index("# opt1 comment line 1")
    opt1_src_idx = text.index("# source:")
    opt1_line_idx = text.index("opt1 = 1")
    assert opt1_cmt_idx < opt1_src_idx < opt1_line_idx

    # second option also has its comment preserved
    assert "# opt2 comment" in text

    # source shows the actual file path
    assert os.path.basename(base) in text


def test_write_toggle_include_flags(tmp_path: Path) -> None:
    path = write_tmp_cfg(
        tmp_path,
        "cfg.cfg",
        """
        [sec]
        # comment
        a = 10
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(path)

    # no sources
    out = io.StringIO()
    cfg.write(out, include_sources=False, include_comments=True)
    text = out.getvalue()
    assert "# source:" not in text
    assert "# comment" in text

    # no comments
    out = io.StringIO()
    cfg.write(out, include_sources=True, include_comments=False)
    text = out.getvalue()
    assert "# source:" in text
    assert "# comment" not in text


def test_explain_base_only_and_multiple_bases(tmp_path: Path) -> None:
    base1 = write_tmp_cfg(
        tmp_path,
        "base1.cfg",
        """
        [s]
        x = 1
        y = 10
        """,
    )
    base2 = write_tmp_cfg(
        tmp_path,
        "base2.cfg",
        """
        [s]
        x = 2
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(base1)
    cfg.add_from_file(base2)  # later base should take precedence for x

    # y exists only in base1
    info_y = cfg.explain("s", "y")
    assert info_y["value"] == "10"
    assert info_y["layer"] == "base"
    assert info_y["source"].endswith("base1.cfg")

    # x overridden by base2
    info_x = cfg.explain("s", "x")
    assert info_x["value"] == "2"
    assert info_x["layer"] == "base"
    assert info_x["source"].endswith("base2.cfg")


def test_explain_missing_section_and_option(tmp_path: Path) -> None:
    path = write_tmp_cfg(
        tmp_path,
        "cfg.cfg",
        """
        [a]
        x = 1
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(path)

    # missing option in existing section -> NoOptionError
    with pytest.raises(NoOptionError):
        cfg.explain("a", "y")

    # missing section -> NoSectionError
    with pytest.raises(NoSectionError):
        cfg.explain("b", "x")
