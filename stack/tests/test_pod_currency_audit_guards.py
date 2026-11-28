"""The pre-launch currency gate must REFUSE rather than report a false all-clear.

Two failures it could not previously distinguish from "the box is current":

  1. MSYS/Git-Bash rewrites `--pod-root /workspace/TanitAD/stack` into a Windows
     path before a native python.exe sees it. MEASURED 2026-09-07: this makes the
     programme's own pre-launch gate unrunnable from the dev box.
  2. `find` over a root that does not exist on the box exits quietly. Every marker
     is still emitted and the payload validates, so `pod_scan` SUCCEEDS with ZERO
     entries -- and the audit then compares the repo against nothing.

⛔ Both produce a PASS-shaped result from a check that looked at nothing, which is
worse than a crash. Same family as the `df` / cgroup / `step_s` traps: a probe
reporting the wrong scope, read as an answer.

Every expectation below is a LITERAL, and `test_the_guard_can_go_RED` reinstalls
the historical behaviour so a guard that cannot fail is caught.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "pod_currency_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("_pca", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MOD = _load()

# (root, must_be_refused)
CASES = [
    ("/workspace/TanitAD/stack", False),
    ("/home/nvidia/TanitAD/stack", False),
    ("/workspace/experiments", False),
    ("C:/Program Files/Git/workspace/TanitAD/stack", True),
    ("C:/workspace/TanitAD/stack", True),
    ("D:/Git/workspace", True),
    ("workspace/TanitAD/stack", True),
    ("", True),
]


@pytest.mark.parametrize("root,must_refuse", CASES)
def test_pod_root_validation(root: str, must_refuse: bool) -> None:
    msg = MOD.check_pod_root(root)
    if must_refuse:
        assert msg != "", f"{root!r} must be refused; the gate would scan nothing"
    else:
        assert msg == "", f"{root!r} is a legitimate remote root; got {msg!r}"


def test_the_msys_message_names_the_workaround() -> None:
    msg = MOD.check_pod_root("C:/Program Files/Git/workspace/TanitAD/stack")
    assert "MSYS_NO_PATHCONV=1" in msg, (
        "an operator who hits this must be told how to fix it, not just that it broke")


def test_empty_scan_is_refused_not_reported_clean(monkeypatch, tmp_path) -> None:
    """A scan that found nothing must exit 2, not print a clean table."""
    monkeypatch.setattr(MOD, "pod_scan", lambda *a, **k: {})
    rc = MOD.main(["--host", "nosuchhost", "--pod-root", "/workspace/TanitAD/stack",
                   "--repo", str(tmp_path)])
    assert rc == 2, "0 files scanned is UNTRUSTWORTHY, never a PASS"


def test_the_guard_actually_PREVENTS_THE_SCAN(monkeypatch, tmp_path) -> None:
    """The real question is BEHAVIOURAL: does a mangled root ever reach the box?

    ⛔ Asserting that a neutered validator returns "" would be a tautology -- an
    expected value of whatever the code does. So assert on what CHANGES: with the
    guard in place `pod_scan` is NEVER CALLED for a mangled root; with the guard
    removed it IS called, against the Windows path, which is the historical bug.
    Both arms exit 2, so the exit code cannot discriminate and is not used.
    """
    MANGLED = "C:/Program Files/Git/workspace/TanitAD/stack"

    calls: list = []
    monkeypatch.setattr(MOD, "pod_scan", lambda host, root, *a, **k: calls.append(root) or {})

    rc_guarded = MOD.main(["--host", "nosuchhost", "--pod-root", MANGLED,
                           "--repo", str(tmp_path)])
    assert calls == [], "GUARDED: a mangled root must never reach the box"
    assert rc_guarded == 2

    # deliberate regression: the historical tool had no validator at all
    monkeypatch.setattr(MOD, "check_pod_root", lambda root: "")
    rc_regressed = MOD.main(["--host", "nosuchhost", "--pod-root", MANGLED,
                             "--repo", str(tmp_path)])
    assert calls == [MANGLED], (
        "REGRESSED: without the validator the scan runs against the Windows path -- "
        "if this list is still empty the guarded arm above proves nothing")
    assert rc_regressed == 2      # caught downstream by the empty-scan guard, not by root
