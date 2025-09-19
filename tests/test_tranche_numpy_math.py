import math
import textwrap
from pathlib import Path

import pytest

from tranche.tranche import Tranche


def write_tmp_cfg(tmp_path: Path, name: str, contents: str) -> str:
    path = tmp_path / name
    path.write_text(textwrap.dedent(contents).lstrip())
    return str(path)


def test_safe_eval_math_pi_and_name(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "math.cfg",
        """
        [expr]
        a = math.pi
        b = pi
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    # math namespace exposes only pi by default
    assert cfg.getexpression("expr", "a", backend="safe") == math.pi
    # 'pi' is also available as a top-level symbol
    assert cfg.getexpression("expr", "b", backend="safe") == math.pi


def test_safe_eval_math_no_sqrt_by_default(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "math.cfg",
        """
        [expr]
        bad = math.sqrt(4)
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    with pytest.raises(NameError):
        cfg.getexpression("expr", "bad", backend="safe")


def test_safe_eval_register_math_function(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "sqrt.cfg",
        """
        [expr]
        val = sqrt(9)
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)
    cfg.register_symbol("sqrt", math.sqrt)

    assert cfg.getexpression("expr", "val", backend="safe") == 3.0


def test_numpy_disabled_raises_for_np_usage(tmp_path: Path) -> None:
    cfg_path = write_tmp_cfg(
        tmp_path,
        "np.cfg",
        """
        [expr]
        arr = np.arange(3)
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    # With numpy not allowed, attempting to use np should fail
    with pytest.raises(NameError):
        cfg.getexpression("expr", "arr", backend="safe", allow_numpy=False)


def test_safe_eval_with_numpy_enabled(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    cfg_path = write_tmp_cfg(
        tmp_path,
        "np.cfg",
        """
        [expr]
        a = np.arange(3)
        b = numpy.linspace(0, 1, 5)
        c = np.array([1, 2, 3])
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    a = cfg.getexpression("expr", "a", backend="safe", allow_numpy=True)
    b = cfg.getexpression("expr", "b", backend="safe", allow_numpy=True)
    c = cfg.getexpression("expr", "c", backend="safe", allow_numpy=True)

    assert a.__class__.__name__ == "ndarray" and a.shape == (3,)
    assert np.array_equal(a, np.arange(3))
    assert b.__class__.__name__ == "ndarray" and b.shape == (5,)
    assert np.allclose(b, np.linspace(0, 1, 5))
    assert c.__class__.__name__ == "ndarray"
    assert np.array_equal(c, np.array([1, 2, 3]))


def test_numpy_auto_backend(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    cfg_path = write_tmp_cfg(
        tmp_path,
        "auto_np.cfg",
        """
        [expr]
        grid = numpy.linspace(0, -4, 5)
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    # backend left as default (None internally), allow_numpy triggers safe
    # backend
    grid = cfg.getexpression("expr", "grid", allow_numpy=True)
    assert np.allclose(grid, np.linspace(0, -4, 5))


def test_getnumpy_helper(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    cfg_path = write_tmp_cfg(
        tmp_path,
        "helper_np.cfg",
        """
        [expr]
        arr = np.arange(4)
        """,
    )

    cfg = Tranche()
    cfg.add_from_file(cfg_path)

    arr = cfg.getnumpy("expr", "arr")
    assert np.array_equal(arr, np.arange(4))
