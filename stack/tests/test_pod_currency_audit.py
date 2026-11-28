#!/usr/bin/env python3
"""Pins for `stack/scripts/pod_currency_audit.py` — the repo→box currency checker.

⛔ WHY THESE EXIST. The programme has burned two runs on pod code that silently
predated the repo (refcv3's 40,284 steps on a synthetic anchor fallback; the
aborted refcv4's 8-wide tactical vocabulary indexed by a 3-wide contract, which
did not crash only because the width assertion `c37fa68` was never on the pod).
The checker exists so that staleness is mechanical rather than noticed.

The tests pin the two properties that decide whether the checker can be trusted,
and BOTH of them were real defects in its first revision, found on 2026-09-04
while auditing `tanitad-refcv3`:

1. **A truncated or corrupt pull must FAIL, not under-report.** The transport is
   a PTY that drops ~5 of every 14 lines on bulk pulls (MEASURED 2026-08-12), and
   a short table would silently read as "fewer files, all fine".

2. **A history probe that could not RUN must never read as a verdict.** The first
   revision called git with `check=False`, so a G:-mount flap returned an empty
   log and every one of the 14 mismatches was reported `POD-DIVERGED` —
   a fabricated classification, produced with total confidence, for files that may
   simply have been `POD-BEHIND`. That is CLAUDE.md's rule in tool form: *an empty
   result is a claim about the SEARCH, not about the CONTENT.* The probe now
   raises and the row reads `HISTORY-UNKNOWN`, which FAILS the audit by default.

`REPO-ONLY` is deliberately NOT fatal by default, but it is not cosmetic either:
on `tanitad-refcv3` the REPO-ONLY set included `tanitad/data/v72_eval_clip_digests.json`,
the data a safety oracle in `parity.py` loads — which is why the `.py`-only default
is itself a trap and `--all-files` is the honest sweep.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "pod_currency_audit.py"


def _load():
    if not _SCRIPT.exists():                                 # pragma: no cover
        pytest.skip(f"{_SCRIPT} not present in this checkout")
    spec = importlib.util.spec_from_file_location("pod_currency_audit_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pod_currency_audit_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load()


# --------------------------------------------------------------------------- #
# 1. the framed pull                                                            #
# --------------------------------------------------------------------------- #
def _frame(rows, *, lines=None, b64=None, md5=None, length=None):
    """Build a pod_scan stdout payload, optionally with a deliberate defect."""
    tsv = "".join("%s\t%s\t%d\t%d\t%s\n" % (r[0], r[1], r[2], r[3], r[4]) for r in rows)
    real = base64.b64encode(gzip.compress(tsv.encode())).decode()
    body = real if b64 is None else b64
    return "\n".join([
        "ZZLINES%dZZ" % (len(rows) if lines is None else lines),
        "ZZB64MD5%sZZ" % (hashlib.md5(real.encode()).hexdigest() if md5 is None else md5),
        "ZZB64LEN%dZZ" % (len(real) if length is None else length),
        "ZZPAYLOADSTART", body, "ZZPAYLOADEND", "ZZDONE0ZZ", ""])


class _FakeProc:
    def __init__(self, out):
        self.returncode, self.stdout, self.stderr = 0, out.encode(), b""


def _patch_ssh(monkeypatch, out):
    monkeypatch.setattr(MOD.subprocess, "run", lambda *a, **k: _FakeProc(out))


ROWS = [("aa" * 16, "bb" * 16, 10, 1, "scripts/a.py"),
        ("cc" * 16, "dd" * 16, 20, 2, "tanitad/b.py")]


def test_good_payload_parses(monkeypatch):
    _patch_ssh(monkeypatch, _frame(ROWS))
    got = MOD.pod_scan("host", "/root", [], 60)
    assert set(got) == {"scripts/a.py", "tanitad/b.py"}
    assert got["scripts/a.py"]["md5"] == "aa" * 16
    assert got["tanitad/b.py"]["size"] == 20


def test_truncated_payload_raises(monkeypatch):
    """A PTY that eats lines must produce an ERROR, never a shorter drift map."""
    good = base64.b64encode(gzip.compress(b"x")).decode()
    _patch_ssh(monkeypatch, _frame(ROWS, b64=good[:-4]))
    with pytest.raises(RuntimeError, match="TRUNCATED"):
        MOD.pod_scan("host", "/root", [], 60)


def test_corrupt_payload_raises(monkeypatch):
    _patch_ssh(monkeypatch, _frame(ROWS, md5="0" * 32))
    with pytest.raises(RuntimeError, match="CORRUPT"):
        MOD.pod_scan("host", "/root", [], 60)


def test_row_count_mismatch_raises(monkeypatch):
    """The pod's own line count disagreeing with the decoded table is fatal."""
    _patch_ssh(monkeypatch, _frame(ROWS, lines=99))
    with pytest.raises(RuntimeError):
        MOD.pod_scan("host", "/root", [], 60)


def test_missing_marker_raises(monkeypatch):
    _patch_ssh(monkeypatch, "ZZPAYLOADSTART\nZZPAYLOADEND\n")
    with pytest.raises(RuntimeError):
        MOD.pod_scan("host", "/root", [], 60)


# --------------------------------------------------------------------------- #
# 2. classification — the failed probe must not become a verdict                #
# --------------------------------------------------------------------------- #
def _stub_classify(monkeypatch, *, pod_md5_lf, ref_md5_lf, worktree_lf=None,
                   history=None, history_raises=False):
    monkeypatch.setattr(MOD, "ref_blobs", lambda *a, **k: {"stack/x.py": "sha-ref"})
    monkeypatch.setattr(MOD, "blob_md5s", lambda *a, **k: {"sha-ref": ("raw", ref_md5_lf)})
    monkeypatch.setattr(MOD, "worktree_md5",
                        lambda *a, **k: None if worktree_lf is None else ("w", worktree_lf))
    monkeypatch.setattr(MOD, "run_git", lambda *a, **k: "deadbeef")

    def _hist(*a, **k):
        if history_raises:
            raise RuntimeError("fatal: not a git repository")
        return history
    monkeypatch.setattr(MOD, "history_match", _hist)
    pod = {"x.py": {"md5": "podraw", "md5_lf": pod_md5_lf, "size": 1, "mtime": 1}}
    return MOD.classify("/repo", "HEAD", "stack", pod, [".py"], [])


def _state(res):
    return res["rows"][0]["state"]


def test_history_probe_failure_is_not_a_verdict(monkeypatch):
    """⛔ THE REGRESSION THIS FILE EXISTS FOR.

    A git call that could not complete must NOT be reported as "matches no
    revision".  Before the fix this returned POD-DIVERGED — a confident, wrong
    classification manufactured out of a mount outage.
    """
    res = _stub_classify(monkeypatch, pod_md5_lf="p", ref_md5_lf="r", history_raises=True)
    assert _state(res) == "HISTORY-UNKNOWN"
    assert "error" in res["rows"][0]["history"]


def test_history_unknown_fails_the_audit_by_default():
    """A probe that could not run must never be scored as clean."""
    default = "pod-behind,pod-diverged,history-unknown"
    assert "history-unknown" in default
    bad = {s.strip().upper() for s in default.split(",")}
    assert "HISTORY-UNKNOWN" in bad


def test_pod_behind_named_when_history_finds_an_ancestor(monkeypatch):
    res = _stub_classify(monkeypatch, pod_md5_lf="p", ref_md5_lf="r",
                         history={"searched": 3,
                                  "hit": {"commit": "c" * 40, "direction": "BEHIND",
                                          "commits_since": 7, "subject": "old"}})
    assert _state(res) == "POD-BEHIND"


def test_diverged_only_when_the_search_actually_ran(monkeypatch):
    res = _stub_classify(monkeypatch, pod_md5_lf="p", ref_md5_lf="r",
                         history={"searched": 12, "hit": None})
    assert _state(res) == "POD-DIVERGED"
    assert res["rows"][0]["history"]["searched"] == 12


def test_line_endings_alone_are_not_drift(monkeypatch):
    """Pod files arrive CRLF from Windows ship steps; that is not staleness."""
    res = _stub_classify(monkeypatch, pod_md5_lf="same", ref_md5_lf="same")
    assert _state(res) == "IDENTICAL"
    assert res["rows"][0].get("eol_only") is True


def test_pod_matches_uncommitted_worktree(monkeypatch):
    """pod == dirty worktree != HEAD: the box runs a change that was never committed."""
    res = _stub_classify(monkeypatch, pod_md5_lf="w", ref_md5_lf="r", worktree_lf="w")
    assert _state(res) == "POD-MATCHES-DIRTY"


def test_repo_only_and_pod_only_are_reported(monkeypatch):
    monkeypatch.setattr(MOD, "ref_blobs", lambda *a, **k: {"stack/only_repo.py": "s1"})
    monkeypatch.setattr(MOD, "blob_md5s", lambda *a, **k: {"s1": ("a", "a")})
    monkeypatch.setattr(MOD, "worktree_md5", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "run_git", lambda *a, **k: "deadbeef")
    pod = {"only_pod.py": {"md5": "z", "md5_lf": "z", "size": 1, "mtime": 1}}
    res = MOD.classify("/repo", "HEAD", "stack", pod, [".py"], [])
    states = {r["path"]: r["state"] for r in res["rows"]}
    assert states == {"only_repo.py": "REPO-ONLY", "only_pod.py": "POD-ONLY"}


def test_ext_filter_hides_non_py_and_all_files_does_not(monkeypatch):
    """The `.py` default is a trap: a JSON sidecar can be a safety oracle's DATA.

    MEASURED 2026-09-04 on tanitad-refcv3: `tanitad/data/v72_eval_clip_digests.json`
    was absent from the pod while `parity.py` was stale, so the v7.2 eval-exclusion
    oracle was doubly disabled — and a `.py`-only sweep could not see the half of it
    that was data.
    """
    monkeypatch.setattr(MOD, "ref_blobs",
                        lambda *a, **k: {"stack/d/side.json": "s1", "stack/d/m.py": "s2"})
    monkeypatch.setattr(MOD, "blob_md5s", lambda *a, **k: {"s1": ("a", "a"), "s2": ("b", "b")})
    monkeypatch.setattr(MOD, "worktree_md5", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "run_git", lambda *a, **k: "deadbeef")
    only_py = MOD.classify("/repo", "HEAD", "stack", {}, [".py"], [])
    assert [r["path"] for r in only_py["rows"]] == ["d/m.py"]
    every = MOD.classify("/repo", "HEAD", "stack", {}, None, [])
    assert sorted(r["path"] for r in every["rows"]) == ["d/m.py", "d/side.json"]


def test_scan_script_tokens_are_disjoint_from_what_we_grep():
    """A monitor whose filter contains the pattern it searches for matches its own echo.

    MEASURED three times in this programme (most recently 2026-08-12: a healthy run
    reported `Traceback CUDA out of memory` because the PTY echoed the command line).
    The pod scanner must therefore never EMIT a marker by writing that marker's literal
    text into a searchable position — the values are computed pod-side and wrapped in
    opaque ZZ…ZZ tokens, and the payload carries its own md5 so an echo cannot forge it.
    """
    assert "ZZB64MD5${h}ZZ" in MOD.POD_SCAN_SH        # the value is interpolated, not literal
    assert "ZZLINES${lines}ZZ" in MOD.POD_SCAN_SH
    # and the scanner must never run git on the pod: a fetch there HANGS, and a failed
    # fetch followed by a checkout RESETS the tree to an ancient commit.
    assert "git " not in MOD.POD_SCAN_SH


def test_history_walks_the_ref_before_all_refs(monkeypatch):
    """`git log --all -- <path>` walks EVERY ref and stalled a whole audit on this mount.

    MEASURED 2026-09-04: the audit process sat at 0.8 s CPU for 15 minutes, all of it inside
    one such call.  "Is the box BEHIND?" is by definition a question about ANCESTORS of the
    ref, so the ref's own history answers it -- and `--all` is kept only as a fallback so a
    pod that is AHEAD or on a side branch is still not misreported as DIVERGED.
    """
    seen = []

    def fake_run_git(args, repo, **kw):
        if args[0] == "log" and "--format=%H" in args:
            seen.append(args[1])
            return "" if args[1] == "--all" else "a" * 40
        return ""

    monkeypatch.setattr(MOD, "run_git", fake_run_git)
    monkeypatch.setattr(MOD, "run_git_stdin",
                        lambda *a, **k: (b"b" * 40 + b" blob 10\n"))
    monkeypatch.setattr(MOD, "blob_md5s", lambda *a, **k: {"b" * 40: ("raw", "nomatch")})
    res = MOD.history_match("/repo", "stack/x.py", "want", "HEAD")
    assert seen[0] == "HEAD", "the ref must be walked FIRST, not --all"
    assert "--all" in seen, "--all must still be the fallback"
    assert res["hit"] is None and res["searched"] >= 1
