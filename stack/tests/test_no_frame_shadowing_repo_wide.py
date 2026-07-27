"""No module may rebind a `CanonicalFrame` name in a `for` loop — repo-wide.

**Why this exists, and why per-file guards were not enough.** Commit ``fdc5b4f``
introduced ``fr = as_frame(...)`` next to a ``for fr in c.decode(stream)`` loop
in ``physicalai.py``; the loop rebound the geometry to a PyAV video frame and
``_decode_mp4`` raised ``AttributeError`` on **every** path, deployed included.
``4cb37f4`` fixed that file — **and missed a second live instance in
``scripts/v2_compressed.py``**, where the rebound name was read through a
*closure*, so it was harder to see and broke every v2 build path. v2 is the only
storage-viable route to a wide corpus, so for a window **v5 had no build step at
all**.

Two instances of one defect, fixed one at a time, is the signature of a missing
class-level guard. The per-file AST checks that shipped with each fix pin the
two known sites; **this one pins the class**, so the third instance fails in CI
instead of being found by an agent three hours into a build.

It matters now specifically because the wide-FOV rollout threads a
``CanonicalFrame`` through every decoder, and ``cosmos_drive.py``, ``l2d.py``
and ``scripts/validate_geometry.py`` all already contain ``for fr in
c.decode(...)`` loops. None binds ``as_frame`` **today** — so they are safe
today and this test passes — but each is one edit away from reintroducing the
exact defect.

The rule: if a module binds a name from ``as_frame(...)``, no ``for`` statement
in that module may use the same name as its target.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[1]
_SKIP_DIRS = {"tests", ".git", "__pycache__", "experiments"}


def _py_files() -> list[Path]:
    return [p for p in _STACK.rglob("*.py")
            if not _SKIP_DIRS & set(p.relative_to(_STACK).parts)]


def _frame_names(tree: ast.AST) -> dict[str, int]:
    """Names bound from `as_frame(...)` -> the line that binds them."""
    out: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "as_frame"):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                out[tgt.id] = node.lineno
    return out


def _loop_targets(tree: ast.AST) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        tgt = node.target
        names = ([t.id for t in tgt.elts if isinstance(t, ast.Name)]
                 if isinstance(tgt, ast.Tuple)
                 else [tgt.id] if isinstance(tgt, ast.Name) else [])
        for n in names:
            out.setdefault(n, node.lineno)
    return out


def test_no_module_rebinds_a_canonicalframe_name_in_a_loop():
    offenders: list[str] = []
    scanned = 0
    for path in _py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # not ours to police
            continue
        scanned += 1
        frames = _frame_names(tree)
        if not frames:
            continue
        loops = _loop_targets(tree)
        for name, bind_line in frames.items():
            if name in loops:
                offenders.append(
                    f"{path.relative_to(_STACK)}: `{name}` bound from "
                    f"as_frame() at line {bind_line} is rebound by a for-loop "
                    f"at line {loops[name]}")

    assert scanned > 20, (
        f"only {scanned} files scanned — the walk is broken and this test "
        f"would pass vacuously")
    assert not offenders, (
        "a CanonicalFrame name is rebound by a decode loop. This is the "
        "fdc5b4f defect; it has now occurred twice (physicalai.py, then "
        "v2_compressed.py through a closure) and silently broke the deployed "
        "path both times:\n  " + "\n  ".join(offenders))


def test_the_guard_can_actually_fail():
    """The failing direction. Without this the test above could be vacuous."""
    bad = ast.parse("fr = as_frame(None, 256, 266.0)\n"
                    "for fr in decoder:\n    pass\n")
    assert set(_frame_names(bad)) & set(_loop_targets(bad)) == {"fr"}, (
        "the detector must flag the exact shape of the original defect")

    good = ast.parse("fr = as_frame(None, 256, 266.0)\n"
                     "for vframe in decoder:\n    pass\n")
    assert not (set(_frame_names(good)) & set(_loop_targets(good))), (
        "the repaired shape must not be flagged, or the guard is unusable")


@pytest.mark.parametrize("mod", ["tanitad/data/cosmos_drive.py",
                                 "tanitad/data/l2d.py",
                                 "scripts/validate_geometry.py"])
def test_known_decoders_stay_clean_as_wide_fov_lands(mod):
    """These carry `for fr in c.decode(...)` but do not bind as_frame yet.

    Named individually so that whoever threads a CanonicalFrame through them
    for the wide-FOV rollout gets a red test naming the file, rather than an
    AttributeError inside a build worker hours later.
    """
    tree = ast.parse((_STACK / mod).read_text(encoding="utf-8"))
    clash = set(_frame_names(tree)) & set(_loop_targets(tree))
    assert not clash, f"{mod} now rebinds {sorted(clash)} — see this module's docstring"
