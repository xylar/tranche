"""
Check what a type checker infers from tranche's getters.

The typed getters are declared with ``@overload``, and an overload can be
wrong in a way nothing else notices: the runtime is unaffected, every
behavioural test still passes, and the damage lands in the callers, who are
told a value has a type it does not.  0.6.0 shipped exactly that -- the
overloads for ``getlist`` spelled ``dtype``'s default as ``...``, so mypy had
nothing to bind the type variable from and inferred ``list[Never]`` where
it should have inferred ``list[str]``.

These tests therefore assert on inference rather than on values.  They run
mypy over a snippet and read back what it reveals, which is the only way to
see an overload from the outside.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

mypy = pytest.importorskip('mypy', reason='mypy is not installed')

#: Each case is an expression and the type mypy should reveal for it.  The
#: expressions are evaluated with ``cfg`` bound to a Tranche and ``sec`` to
#: one of its sections.
CASES = [
    # the plain getters return a value, not an optional one
    ("cfg.get('s', 'o')", 'str'),
    ("cfg.getint('s', 'o')", 'int'),
    ("cfg.getfloat('s', 'o')", 'float'),
    ("cfg.getboolean('s', 'o')", 'bool'),
    # ...and widen to include the fallback when one is given
    ("cfg.get('s', 'o', fallback=None)", 'str | None'),
    ("cfg.getint('s', 'o', fallback=0)", 'int'),
    # getlist without a dtype is a list of str; this is what 0.6.0 broke,
    # and both of the cases below reveal list[Never] until it is fixed
    ("cfg.getlist('s', 'o')", 'list[str]'),
    ("cfg.getlist('s', 'o', dtype=int)", 'list[int]'),
    ("cfg.getlist('s', 'o', dtype=float)", 'list[float]'),
    # a fallback widens the list type rather than replacing it, and the
    # element type still comes from dtype (or defaults to str)
    ("cfg.getlist('s', 'o', fallback=None)", 'list[str] | None'),
    ("cfg.getlist('s', 'o', dtype=float, fallback=None)", 'list[float] | None'),
    # raw and vars are accepted without disturbing the element type
    ("cfg.getlist('s', 'o', raw=True)", 'list[str]'),
    # the same, through a Section
    ("sec.getlist('o')", 'list[str]'),
    ("sec.getlist('o', dtype=int)", 'list[int]'),
    ("sec.getlist('o', fallback=None)", 'list[str] | None'),
    ("sec.getlist('o', dtype=float, fallback=None)", 'list[float] | None'),
    ("sec.getlist('o', raw=True, vars={'a': 'b'})", 'list[str]'),
]

PREAMBLE = 'from tranche import Tranche\n\ncfg = Tranche()\nsec = cfg["s"]\n'


#: the repository root, so that mypy resolves ``tranche`` from the source
#: being tested rather than from whatever happens to be installed
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize(revealed: str) -> str:
    """
    Put a revealed type into the short spelling the cases are written in.

    mypy 2 reveals ``list[str] | None`` where mypy 1 reveals
    ``Union[builtins.list[builtins.str], None]``.  The two say the same
    thing, so the expectations are written once, in the shorter form, and
    the older spelling is folded onto it here.
    """
    text = revealed.replace('builtins.', '').replace('typing.', '')
    # the only unions in the cases are optionals, so one rewrite is enough
    return re.sub(r'Union\[(.+), None\]', r'\1 | None', text)


def _run_mypy(tmp_path: Path, body: str) -> str:
    """Run mypy over a snippet and return its output."""
    path = tmp_path / 'snippet.py'
    path.write_text(PREAMBLE + body)
    env = dict(os.environ, MYPYPATH=_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'mypy',
            '--no-incremental',
            '--no-error-summary',
            str(path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout


@pytest.fixture(scope='module')
def revealed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """The type mypy reveals for each case, keyed by the expression."""
    tmp_path = tmp_path_factory.mktemp('inference')
    body = ''.join(f'reveal_type({expression})\n' for expression, _ in CASES)
    output = _run_mypy(tmp_path, body)
    types = [
        _normalize(line.split('Revealed type is')[1].strip().strip('"'))
        for line in output.split('\n')
        if 'Revealed type is' in line
    ]
    assert len(types) == len(CASES), (
        f'expected {len(CASES)} revealed types but got {len(types)}; mypy '
        f'said:\n{output}'
    )
    return dict(zip([case[0] for case in CASES], types, strict=True))


@pytest.mark.parametrize(('expression', 'expected'), CASES)
def test_the_getters_infer_the_type_they_return(
    revealed: dict[str, str], expression: str, expected: str
) -> None:
    """
    A getter whose overloads are wrong still runs correctly, so nothing else
    in this suite would notice.  Callers would.
    """
    assert revealed[expression] == expected, (
        f'{expression} is inferred as {revealed[expression]}, not '
        f'{expected}.  The overloads for that getter do not describe what '
        f'it returns, so every caller is told the wrong type while the '
        f'runtime stays correct.'
    )


def test_a_getter_result_needs_no_annotation(tmp_path: Path) -> None:
    """
    Binding a getter's result must not require the caller to annotate it.
    An unsolved type variable shows up this way rather than as a wrong type,
    which is how the 0.6.0 regression reached a release.
    """
    body = (
        'fields = cfg.getlist("s", "o")\n'
        'numbers = cfg.getlist("s", "o", dtype=float)\n'
        'name = cfg.get("s", "o")\n'
        'count = cfg.getint("s", "o")\n'
        'items = sec.getlist("o")\n'
    )
    output = _run_mypy(tmp_path, body)
    assert 'Need type annotation' not in output, (
        f'a caller has to annotate a getter result, which means a type '
        f'variable was left unsolved by the overloads:\n{output}'
    )
    assert 'error:' not in output, output
