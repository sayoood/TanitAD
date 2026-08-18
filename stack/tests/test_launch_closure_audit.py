"""Self-tests for ``scripts/launch_closure_audit.py`` — the C99 class fix.

C99 (2026-08-18): a ship set derived from *"what I edited"* instead of from the
import closure left a 2.6×-stale dependency on Thor, and **three green md5s**
reported success. This instrument replaces the hand-list. These tests pin the
four properties that make it trustworthy, each of which has a real failure
behind it:

* the walk follows **function-level** imports (that is where C99's file lived);
* CRLF-vs-LF is classified as ``CRLF_ONLY``, **never** as ``DRIFT`` (a naive
  comparison produced 7 false drift rows out of 10 on 2026-08-16, and 47 of 50
  in tonight's Thor audit);
* an import inside ``try/except ImportError`` is marked **guarded**, so a probe
  that bypasses the guard cannot fabricate a blocker;
* Git-Bash MSYS argument mangling is undone (untreated, it made the audit report
  **120/120 MISSING_REMOTE** — a clean, plausible, entirely false answer).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "launch_closure_audit", _SCRIPTS / "launch_closure_audit.py")
LCA = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register BEFORE exec: @dataclass resolves annotations via
# sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules["launch_closure_audit"] = LCA
_spec.loader.exec_module(LCA)


# --------------------------------------------------------------------------
# 1. the walk reaches function-level imports — C99's actual failure
# --------------------------------------------------------------------------


def _write(root: Path, rel: str, src: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8")
    return p


def test_closure_follows_function_level_imports(tmp_path, monkeypatch):
    """A hand-list would miss ``deep``; the closure must not.

    ``entry`` imports ``mid`` at module level; ``mid`` imports ``deep`` ONLY
    inside a function body. That is exactly the shape of
    ``v6.py`` -> ``tanitad.data.anchor_goal``, the file this audit found absent
    on Thor.
    """
    monkeypatch.setattr(LCA, "_REPO", tmp_path)
    _write(tmp_path, "s/entry.py", "import mid\n")
    _write(tmp_path, "s/mid.py", "def f():\n    import deep\n    return deep\n")
    _write(tmp_path, "s/deep.py", "X = 1\n")

    c = LCA.compute_closure([tmp_path / "s/entry.py"], [tmp_path / "s"])
    assert set(c.files) == {"s/entry.py", "s/mid.py", "s/deep.py"}
    # and the reach axis must separate them
    assert "s/deep.py" in c.deferred_only
    assert "s/mid.py" in c.eager and "s/entry.py" in c.eager
    assert "s/deep.py" not in c.eager


def test_importlib_string_and_relative_imports_are_followed(tmp_path, monkeypatch):
    monkeypatch.setattr(LCA, "_REPO", tmp_path)
    _write(tmp_path, "s/entry.py",
           "import importlib\nimportlib.import_module('pkg.a')\n")
    _write(tmp_path, "s/pkg/__init__.py", "")
    _write(tmp_path, "s/pkg/a.py", "from . import b\n")
    _write(tmp_path, "s/pkg/b.py", "Y = 2\n")

    c = LCA.compute_closure([tmp_path / "s/entry.py"], [tmp_path / "s"])
    assert {"s/pkg/__init__.py", "s/pkg/a.py", "s/pkg/b.py"} <= set(c.files)


def test_external_modules_are_recorded_not_walked(tmp_path, monkeypatch):
    monkeypatch.setattr(LCA, "_REPO", tmp_path)
    _write(tmp_path, "s/entry.py", "import torch\nimport json\n")
    c = LCA.compute_closure([tmp_path / "s/entry.py"], [tmp_path / "s"])
    assert c.files == ["s/entry.py"]
    assert "torch" in c.external and "json" in c.external


# --------------------------------------------------------------------------
# 2. ⚠️ CRLF is never reported as drift
# --------------------------------------------------------------------------


def test_crlf_only_is_not_drift():
    """The repo is CRLF, Thor is LF. This distinction IS the audit's validity."""
    crlf = b"import os\r\ndef f():\r\n    return 1\r\n"
    lf = crlf.replace(b"\r\n", b"\n")
    raw_a, lf_a, n_a = LCA._digests(crlf)
    raw_b, lf_b, n_b = LCA._digests(lf)

    assert raw_a != raw_b, "the raw digests must differ, else there is nothing to trap"
    assert lf_a == lf_b, "LF-normalised digests must agree"
    assert n_a - n_b == 3, "one \\r per line — the byte delta equals the line count"

    v = LCA.verdict({"present": True, "md5_raw": raw_a, "md5_lf": lf_a},
                    {"present": True, "md5_raw": raw_b, "md5_lf": lf_b})
    assert v == "CRLF_ONLY"


def test_real_drift_survives_lf_normalisation():
    a, b = b"A = 1\r\nB = 2\r\n", b"A = 1\nB = 3\n"
    ra, la, _ = LCA._digests(a)
    rb, lb, _ = LCA._digests(b)
    assert LCA.verdict({"present": True, "md5_raw": ra, "md5_lf": la},
                       {"present": True, "md5_raw": rb, "md5_lf": lb}) == "DRIFT"


def test_identical_bytes_are_same_and_absence_is_named():
    d = LCA._digests(b"x = 1\n")
    same = {"present": True, "md5_raw": d[0], "md5_lf": d[1]}
    assert LCA.verdict(same, dict(same)) == "SAME"
    assert LCA.verdict(same, {"present": False}) == "MISSING_REMOTE"
    assert LCA.verdict({"present": False}, same) == "MISSING_LOCAL"


# --------------------------------------------------------------------------
# 3. guarded imports must not be reported as blockers
# --------------------------------------------------------------------------


def test_try_except_importerror_marks_the_import_guarded(tmp_path, monkeypatch):
    """``probe_latent_state.py:117`` wraps ``lead_state_gate`` exactly like this.

    Thor cannot import it (no pandas) and the launch does not care. A probe that
    called that a blocker would be a fabricated finding.
    """
    monkeypatch.setattr(LCA, "_REPO", tmp_path)
    _write(tmp_path, "s/entry.py",
           "try:\n    from opt import K\nexcept ImportError:\n    K = 1\n"
           "import hard\n")
    _write(tmp_path, "s/opt.py", "K = 2\n")
    _write(tmp_path, "s/hard.py", "Z = 3\n")

    c = LCA.compute_closure([tmp_path / "s/entry.py"], [tmp_path / "s"])
    assert "s/opt.py" in c.guarded_only
    assert "s/hard.py" not in c.guarded_only


@pytest.mark.parametrize("clause,guarded", [
    ("except ImportError:", True),
    ("except ModuleNotFoundError:", True),
    ("except (ImportError, ValueError):", True),
    ("except Exception:", True),
    ("except:", True),
    ("except ValueError:", False),
    ("except OSError:", False),
])
def test_handler_catches_import(clause, guarded):
    src = f"try:\n    import x\n{clause}\n    pass\n"
    handler = ast.parse(src).body[0].handlers[0]
    assert LCA._handler_catches_import(handler) is guarded


def test_else_branch_of_a_guarded_try_is_not_protected(tmp_path, monkeypatch):
    """Only the ``try:`` BODY is guarded — ``else:`` runs unprotected."""
    monkeypatch.setattr(LCA, "_REPO", tmp_path)
    _write(tmp_path, "s/entry.py",
           "try:\n    import a\nexcept ImportError:\n    a = None\n"
           "else:\n    import b\n")
    _write(tmp_path, "s/a.py", "")
    _write(tmp_path, "s/b.py", "")
    c = LCA.compute_closure([tmp_path / "s/entry.py"], [tmp_path / "s"])
    assert "s/a.py" in c.guarded_only
    assert "s/b.py" not in c.guarded_only


# --------------------------------------------------------------------------
# 4. ⛔ MSYS argument mangling — it produced a 100 %-wrong audit
# --------------------------------------------------------------------------


def test_msys_path_mangling_is_undone():
    assert (LCA._demangle_posix("C:/Program Files/Git/home/nvidia/TanitAD")
            == "/home/nvidia/TanitAD")
    assert (LCA._demangle_posix("C:/Program Files/Git/workspace/TanitAD")
            == "/workspace/TanitAD")
    # already correct, or absent — left exactly alone
    assert LCA._demangle_posix("/home/nvidia/TanitAD") == "/home/nvidia/TanitAD"
    assert LCA._demangle_posix(None) is None
    assert LCA._demangle_posix("relative/path") == "relative/path"


# --------------------------------------------------------------------------
# 5. the framing the probe-hygiene rules require
# --------------------------------------------------------------------------


def test_unframe_takes_only_the_opaque_marker():
    """Remote answers arrive inside ``ZZ…ZZ``; chatter around them is ignored.

    Never grep the raw stream for the words the command itself contains — a
    monitor doing that has invented a false failure three times.
    """
    stream = ("Warning: Permanently added host\n"
              "some banner text mentioning DRIFT and Traceback\n"
              "ZZpayload-here-ZZ\n")
    assert LCA._unframe(stream) == "payload-here-"
    with pytest.raises(RuntimeError):
        LCA._unframe("no marker at all\n")


def test_default_entries_and_roots_exist_in_the_repo():
    """The shipped defaults must name real files, or the audit silently narrows."""
    repo = Path(__file__).resolve().parents[2]
    for rel in LCA.DEFAULT_ENTRIES:
        assert (repo / rel).is_file(), f"missing default entry point: {rel}"
    for rel in LCA.DEFAULT_ROOTS:
        assert (repo / rel).is_dir(), f"missing default sys.path root: {rel}"


def test_the_v6_ladder_closure_is_far_larger_than_a_hand_list():
    """The whole point: a hand-listed ship set is a guess.

    MEASURED 2026-08-18: 120 files from the 7 v6 entry points, where C99 shipped
    3. This asserts the order of magnitude, not the exact count, so ordinary repo
    growth does not fail the suite — but a collapse to a hand-list-sized set
    (i.e. the walk silently stopped following imports) does.
    """
    repo = Path(__file__).resolve().parents[2]
    c = LCA.compute_closure([repo / e for e in LCA.DEFAULT_ENTRIES],
                            [repo / r for r in LCA.DEFAULT_ROOTS])
    assert len(c.files) > 60, (
        f"closure collapsed to {len(c.files)} files — the walk is not following "
        "imports; a launch audit built on this would repeat C99")
    assert "stack/scripts/refc_dump_latents.py" in c.files, (
        "the exact file C99 missed must be in the computed closure")
    assert "stack/tanitad/data/anchor_goal.py" in c.files
    assert c.deferred_only, "the deferred-only risk set must be populated"
