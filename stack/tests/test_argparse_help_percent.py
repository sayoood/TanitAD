"""A BARE `%` IN AN argparse `help=` STRING BREAKS `--help` — statically pinned.

⛔ THE FAILURE THIS EXISTS FOR, MEASURED 2026-09-05. `--goal-kappa-turn` was
added to `taniteval/tools/refav1_arm.py` with help text quoting measurements:
`"... on 90.6 % of GT-turn windows ..."`. `argparse` runs `%`-formatting over
every help string in `HelpFormatter._expand_help`, so `"% o"` is read as the
`%o` conversion and `--help` died with

    TypeError: %o format: an integer is required, not dict

⚠️ **The run path was fine — only `--help` was dead**, which is exactly why it
could have shipped. A tool whose `--help` crashes is a tool nobody can discover
the flags of, and this programme's flags carry the measurements that justify
them.

⭐ AND THE THING THAT CAUGHT IT WAS A SAME-BREATH CONTROL, not the check itself:
grepping the help output for the new flag returned 0, which reads exactly like
"my grep is wrong" — until the control grep for the SIBLING flag
(`--lat-logit-bias`, which had always been there) ALSO returned 0. Two zeros,
one of which had to be non-zero, is what turned "my grep is wrong" into "the
whole help output is missing". *CLAUDE.md's rule that a count of 0 from a
channel that could have failed is not evidence of absence.*

This test is STATIC — it parses the file with `ast` and never imports it, so it
costs milliseconds and cannot be defeated by a missing dependency.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Tools whose --help must render. Named explicitly rather than globbed: a glob
# would silently cover nothing if the layout moved, and this file would then
# pass forever while testing zero files (see `test_z_the_control` below).
TOOLS = [
    "taniteval/tools/refav1_arm.py",
    "stack/scripts/refc_v3_train.py",
]

# `%%` is the escaped form; `%(default)s`-style named specs are also legal.
_NAMED = re.compile(r"%\((?:[^)]*)\)[sdrfgxo]")


def _bare_percents(text: str) -> list[str]:
    t = _NAMED.sub("", text).replace("%%", "")
    return [text] if "%" in t else []


def _help_strings(path: pathlib.Path):
    """Every `help=` keyword value in an `add_argument(...)` call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                val = ast.literal_eval(kw.value)
            except Exception:
                continue                      # a non-literal help is not ours
            if isinstance(val, str):
                out.append((getattr(node, "lineno", -1), val))
    return out


@pytest.mark.parametrize("rel", TOOLS)
def test_a_no_bare_percent_in_argparse_help(rel):
    p = REPO / rel
    if not p.exists():
        pytest.skip(f"{rel} not present in this checkout")
    helps = _help_strings(p)
    # ⛔ CONTROL: a file we assert about must actually HAVE help strings. Zero
    # would make the assertion below vacuously true — the failure mode this
    # whole file is about.
    assert helps, f"{rel}: parsed 0 help strings — the check would be vacuous"
    bad = [(ln, h) for ln, h in helps if _bare_percents(h)]
    assert not bad, (
        f"{rel}: {len(bad)} argparse help string(s) carry a BARE '%', which "
        f"argparse reads as a format spec and which kills --help. Write '%%'. "
        f"First at line {bad[0][0]}: {bad[0][1][:160]!r}")


def test_z_the_control_detects_a_bare_percent():
    """The detector must FIRE on a known-bad string and STAY SILENT on the
    escaped and named forms — otherwise `test_a` passing proves nothing."""
    assert _bare_percents("90.6 % of windows")           # the real bug
    assert _bare_percents("100% done")
    assert not _bare_percents("90.6 %% of windows")      # escaped
    assert not _bare_percents("default: %(default)s")    # named spec
    assert not _bare_percents("no percent here at all")
