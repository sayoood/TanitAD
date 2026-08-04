"""Repo-wide: no fidelity harness may decode the reference video without an offset.

**Why a class-level guard and not per-file fixes.** `R-2026-08-03-k` was one defect with
four call sites: `render_quality.py`, `rs_sweep.py`, `rs_seam_control.py` and
`rs_frame_offset.py` all decode `camera_front_wide_120fov.mp4` through `load_refs`, and
every absolute number any of them produced was scored against a reference **6 frames too
early**.  Fixing them one at a time is the signature this repo already knows
(`test_no_frame_shadowing_repo_wide.py`: two instances of one defect, fixed separately,
the second live for a window).  So the rule is pinned here:

    every `load_refs(...)` call outside the offset-measuring instrument itself must pass
    `ref_offset=`, and no module may hard-code the literal 6 as the offset.

The second half matters as much as the first.  The offset is **+6 on `00040136` and +5 on
`7c72937c`** (both MEASURED by the renderer's own neighbour scan, `argmax_histogram`
`{6: 12}` and `{5: 12}`).  A hard-coded 6 is a fix that is wrong on the very next scene —
and "one scene has already refuted one generalisation" is how this file came to exist.
"""
from __future__ import annotations

import ast
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1] / "experiments"

# `rs_frame_offset.py` MEASURES the offset by scanning reference indices, so it must
# decode without one. It is the only admissible exemption, and it is named, not a pattern.
_EXEMPT = {"rs_frame_offset.py"}


def _py_files() -> list[Path]:
    return [p for p in _EXP.rglob("*.py") if "__pycache__" not in p.parts]


def _load_refs_calls(tree: ast.AST) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else None)
            if name == "load_refs":
                out.append(node)
    return out


def test_every_load_refs_call_passes_a_reference_offset():
    offenders = []
    for p in _py_files():
        if p.name in _EXEMPT:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for call in _load_refs_calls(tree):
            has_kw = any(k.arg == "ref_offset" for k in call.keywords)
            has_pos = len(call.args) >= 4
            if not (has_kw or has_pos):
                offenders.append(f"{p.relative_to(_EXP)}:{call.lineno}")
    assert not offenders, (
        "load_refs() called without ref_offset= (R-2026-08-03-k): " + ", ".join(offenders))


def test_the_offset_is_never_hard_coded_as_a_literal_six():
    """+6 is `00040136`'s answer, not the rule. `7c72937c` is +5."""
    offenders = []
    for p in _py_files():
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for call in _load_refs_calls(tree):
            vals = [k.value for k in call.keywords if k.arg == "ref_offset"]
            vals += list(call.args[3:4])
            for v in vals:
                if isinstance(v, ast.Constant) and v.value in (6, 5):
                    offenders.append(f"{p.relative_to(_EXP)}:{call.lineno}")
    assert not offenders, (
        "reference offset hard-coded to a per-scene value: " + ", ".join(offenders))


def test_the_exempt_instrument_still_exists_and_still_scans():
    """If `rs_frame_offset.py` is renamed or gutted, the exemption above silently starts
    excusing nothing — or worse, a real offender inherits its name."""
    p = _EXP / "alpasim-gsplat" / "rs_frame_offset.py"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    assert "load_refs(" in src
    assert "adjudicate(" in src, "the scanner must adjudicate, not take a bare argmax"


def test_render_quality_exposes_the_gate_and_the_offset_flag():
    """The gate is only a gate if the CLI actually offers it and defaults it ON."""
    src = (_EXP / "alpasim-gsplat" / "render_quality.py").read_text(encoding="utf-8")
    assert "--ref-offset" in src
    assert "--no-align-check" in src        # opt-OUT, therefore on by default
    assert "assert_reference_aligned" in src
    assert "MIN_WRONG_GAP" in src
