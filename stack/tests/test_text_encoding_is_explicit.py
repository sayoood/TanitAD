"""Text I/O declares its encoding, so the suite is runnable on the dev box.

⛔ WHY THIS EXISTS. Four suite failures on 2026-08-21 looked like four unrelated
logic bugs. They were one environment bug with two faces, and both faces are
invisible on Linux:

  READ  a text-mode ``subprocess`` call with no ``encoding=`` decodes with the
        LOCALE codec. On Windows that is cp1252, and every tool in this repo
        prints star / stop / warning glyphs by house style — so a tool's OWN
        tests die with ``UnicodeDecodeError: 'charmap' codec can't decode byte
        0x8f``, which reads as a crash in the tool rather than in the harness.
        23 files carried it.

  WRITE a bare text-mode ``open`` for writing encodes with the locale codec.
        ``Paper/figures/make_lf0_bev_panels.py`` emits U+2265 into its SVG, so
        on Windows THE FIGURE GENERATOR COULD NOT WRITE ITS OUTPUT AT ALL —
        a production defect, not a test defect.

⚠️ THE FIX IS DELIBERATELY SCOPED, and this file encodes that scope. 110 bare
text-mode opens exist repo-wide; most are pod-side scripts that only ever run on
Linux, and a few read logs that are NOT clean UTF-8, where strict decoding would
INTRODUCE failures (``thor_bench_report.py`` already passes ``errors="replace"``
for exactly that reason). So opens are policed only where they bite — the tests
and the figure generators — while subprocess is policed everywhere, because
there the thing being decoded is always our own tool's stdout.

⛔ SCANNED WITH ``ast``, NOT REGEX. The first cut used regexes over source text
and flagged its own docstring and its own probe string — a guard that fails on
its own explanation is worse than no guard.
"""
from __future__ import annotations

import ast
import pathlib

STACK = pathlib.Path(__file__).resolve().parents[1]
REPO = STACK.parent
SELF = pathlib.Path(__file__).resolve()

_SUBPROC_FUNCS = {"run", "check_output", "Popen", "check_call"}
_TEXT_FLAGS = ("text", "universal_newlines")


def _py_files(*roots: pathlib.Path):
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in str(p) or p.resolve() == SELF:
                continue
            yield p


def _tree(p: pathlib.Path):
    """⚠️ utf-8-SIG, not utf-8. Six files in this repo carry a BOM; Python
    strips it on import but `ast.parse` of a BOM-prefixed string raises
    `SyntaxError: invalid non-printable character U+FEFF`. Reading with plain
    utf-8 made this guard SILENTLY SKIP those six — a hole exactly where a
    guard is supposed to be watching."""
    try:
        return ast.parse(p.read_text(encoding="utf-8-sig"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_subprocess_call(call: ast.Call) -> bool:
    f = call.func
    return (isinstance(f, ast.Attribute) and f.attr in _SUBPROC_FUNCS
            and isinstance(f.value, ast.Name) and f.value.id == "subprocess")


def _is_bare_open(call: ast.Call) -> bool:
    """A module-level ``open(...)`` — not ``Path.open``, not a method."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "open"):
        return False
    if _kw(call, "encoding") is not None:
        return False
    mode = _kw(call, "mode")
    if mode is None and len(call.args) >= 2:
        mode = call.args[1]
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str) \
            and "b" in mode.value:
        return False                      # binary: no encoding applies
    return True


def test_subprocess_text_mode_always_names_its_encoding():
    """Repo-wide: what is being decoded is our own tool's stdout, and that is
    UTF-8 on every platform this programme runs on."""
    bad = []
    for p in _py_files(STACK, REPO / "Paper", REPO / "tools"):
        t = _tree(p)
        if t is None:
            continue
        for node in ast.walk(t):
            if not (isinstance(node, ast.Call) and _is_subprocess_call(node)):
                continue
            if _kw(node, "encoding") is not None:
                continue
            if any(_is_true(_kw(node, f)) for f in _TEXT_FLAGS):
                bad.append(f"{p.relative_to(REPO)}:{node.lineno}")
    assert not bad, (
        "text-mode subprocess without encoding= decodes with the LOCALE codec; "
        "on Windows that is cp1252 and any non-Latin-1 glyph in the child's "
        "output raises UnicodeDecodeError, which reads as a bug in the tool. "
        'Add encoding="utf-8":\n  ' + "\n  ".join(bad))


def test_tests_and_figures_open_text_files_with_an_explicit_encoding():
    """Scoped to the places that MUST work on the dev box."""
    bad = []
    for p in _py_files(STACK / "tests", REPO / "Paper" / "figures"):
        t = _tree(p)
        if t is None:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.Call) and _is_bare_open(node):
                bad.append(f"{p.relative_to(REPO)}:{node.lineno}")
    assert not bad, (
        "bare text-mode open() in a test or figure generator. On Windows the "
        "locale codec cannot encode U+2265, so make_lf0_bev_panels.py could "
        'not write its SVG at all. Add encoding="utf-8":\n  ' + "\n  ".join(bad))


def test_the_guard_can_actually_fail():
    """⛔ A guard that cannot fail teaches nothing. Both detectors are exercised
    on synthetic source, so a green run means the repo is clean rather than that
    the detectors quietly stopped matching."""
    src = ("import subprocess\n"
           "subprocess.run(['x'], capture_output=True, text=True)\n"
           "subprocess.run(['x'], capture_output=True, text=True, encoding='utf-8')\n"
           "open('f.txt', 'w')\n"
           "open('f.txt', 'w', encoding='utf-8')\n"
           "open('f.bin', 'wb')\n")
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]
    subs = [n for n in calls if _is_subprocess_call(n)]
    flagged = [n for n in subs if _kw(n, "encoding") is None
               and any(_is_true(_kw(n, f)) for f in _TEXT_FLAGS)]
    assert len(subs) == 2 and len(flagged) == 1, "subprocess detector drifted"
    opens = [n for n in calls if isinstance(n.func, ast.Name)
             and n.func.id == "open"]
    assert len(opens) == 3, "the probe lost an open() call"
    assert [_is_bare_open(n) for n in opens] == [True, False, False], (
        "the open() detector must flag the bare text write, and must NOT flag "
        "the encoded one or the binary one")
