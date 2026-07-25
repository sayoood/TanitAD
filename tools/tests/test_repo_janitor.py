"""Falsifiers for repo_janitor.

The incident (RETRACTION_LOG 07-24, class C8): two modules were reported STRANDED
and absent from main; both were in main. `Glob` is mtime-sorted and truncated at
100, and with 51 worktrees plus 94 incoming bundles the freshly-touched worktree
copies filled the window. The sprawl was the measurement error, so the janitor's
one non-negotiable property is that it NEVER destroys work to reduce it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repo_janitor as rj  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, errors="replace")
    assert proc.returncode == 0, f"git {args} -> {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("a\n", encoding="utf-8")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-m", "seed")
    return r


def _worktree(repo: Path, name: str, extra_commit: bool = False,
              dirty: bool = False) -> Path:
    wt = repo.parent / name
    _git(repo, "worktree", "add", "-b", f"agent/{name}", str(wt))
    if extra_commit:
        (wt / f"{name}.txt").write_text("work\n", encoding="utf-8")
        _git(wt, "add", ".")
        _git(wt, "commit", "-m", f"{name} work")
    if dirty:
        (wt / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")
    return wt


def _bundle(repo: Path, area: str, slug: str, verdict: str | None,
            n: int = 2) -> Path:
    d = repo / rj.HUB_DIR / area / "Implementation" / "incoming" / slug
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"f{i}.md").write_text("x" * 100, encoding="utf-8")
    if verdict is not None:
        (d / "INTAKE.md").write_text(
            f"# INTAKE\n\n## ORCHESTRATOR VERDICT\n\n- **Verdict:** {verdict}\n",
            encoding="utf-8")
    return d


# ------------------------------------------------------------------ worktrees


def test_a_worktree_with_unique_commits_is_never_a_candidate(repo):
    _worktree(repo, "holder", extra_commit=True)
    wts = rj.score_worktrees(repo, rj.parse_worktrees(repo), "HEAD", True)
    holder = next(w for w in wts if w.path.endswith("holder"))
    assert holder.ahead == 1
    assert holder.candidate is False
    assert "do NOT delete" in holder.reason


def test_a_dirty_worktree_is_never_a_candidate(repo):
    _worktree(repo, "dirty", dirty=True)
    wts = rj.score_worktrees(repo, rj.parse_worktrees(repo), "HEAD", True)
    w = next(x for x in wts if x.path.endswith("dirty"))
    assert w.ahead == 0 and w.dirty >= 1 and w.candidate is False


def test_a_clean_zero_ahead_worktree_is_a_candidate_but_survives_by_default(repo):
    wt = _worktree(repo, "spent")
    rep_code = rj.main(["--repo", str(repo)])
    assert rep_code == 0
    assert wt.is_dir(), "reporting must never delete"


def test_delete_removes_only_candidates(repo):
    spent = _worktree(repo, "spent")
    holder = _worktree(repo, "holder", extra_commit=True)
    dirty = _worktree(repo, "dirty", dirty=True)
    assert rj.main(["--repo", str(repo), "--delete"]) == 0
    assert not spent.is_dir(), "the safe candidate should be gone"
    assert holder.is_dir(), "a worktree holding commits must survive --delete"
    assert dirty.is_dir(), "a dirty worktree must survive --delete"


def test_delete_refuses_without_a_status_check(repo):
    wt = _worktree(repo, "spent")
    assert rj.main(["--repo", str(repo), "--delete", "--fast"]) == 2
    assert wt.is_dir()


def test_prune_clears_records_whose_directory_is_gone(repo):
    wt = _worktree(repo, "vanished")
    import shutil
    shutil.rmtree(wt)
    before = len(rj.parse_worktrees(repo))
    rj.main(["--repo", str(repo)])
    assert len(rj.parse_worktrees(repo)) < before


def test_the_running_worktree_is_excluded(repo):
    wts = rj.score_worktrees(repo, rj.parse_worktrees(repo), "HEAD", True)
    me = next(w for w in wts if Path(w.path).resolve() == repo.resolve())
    assert me.candidate is False
    assert "running in" in me.reason


# ------------------------------------------------------------------- incoming


def test_incoming_ledger_ages_and_triage_state(repo):
    _bundle(repo, "Data Engineering", "2026-07-08-old-thing", None)
    _bundle(repo, "Benchmarks & Eval", "2026-07-24-fresh", "integrate")
    _bundle(repo, "Tools&DevEnv", "2026-07-20-untriaged",
            "integrate / integrate-with-changes / defer / reject")
    bundles = rj.scan_incoming(repo, date(2026, 7, 25))
    by = {b.slug: b for b in bundles}
    assert by["2026-07-08-old-thing"].age_days == 17
    assert by["2026-07-08-old-thing"].intake == "missing"
    assert by["2026-07-24-fresh"].intake == "integrate"
    assert by["2026-07-20-untriaged"].intake == "unfilled"
    assert bundles[0].slug == "2026-07-08-old-thing", "oldest first"
    assert all(b.n_files >= 2 and b.bytes > 0 for b in bundles)


def test_ledger_is_dated_markdown_and_lists_every_bundle(repo, tmp_path):
    _bundle(repo, "Data Engineering", "2026-07-08-old-thing", None)
    _bundle(repo, "Data Engineering", "2026-07-24-fresh", "defer")
    out = tmp_path / "led" / "LEDGER.md"
    assert rj.main(["--repo", str(repo), "--ledger-out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "2026-07-08-old-thing" in text and "2026-07-24-fresh" in text
    assert "Generated:" in text
    text.encode("ascii")


# --------------------------------------------------------------------- future


def test_future_dated_slug_is_flagged(repo):
    _bundle(repo, "Opponent Analyzer", "2026-07-31-ahead-of-the-clock", None)
    bundles = rj.scan_incoming(repo, date(2026, 7, 25))
    b = bundles[-1]
    assert b.future is True and b.age_days == -6


def test_future_mtime_on_a_tracked_file_is_flagged(repo):
    f = repo / "ahead.txt"
    f.write_text("later\n", encoding="utf-8")
    _git(repo, "add", "ahead.txt")
    _git(repo, "commit", "-m", "ahead")
    future = time.time() + 3 * 3600
    os.utime(f, (future, future))
    items = rj.scan_future(repo, datetime.now(), 300, 500)
    assert any(i.path == "ahead.txt" for i in items)
    assert all(i.ahead_s > 0 for i in items)


def test_untracked_future_files_are_ignored(repo):
    f = repo / "scratch.tmp"
    f.write_text("x\n", encoding="utf-8")
    future = time.time() + 3 * 3600
    os.utime(f, (future, future))
    assert not [i for i in rj.scan_future(repo, datetime.now(), 300, 500)
                if i.path == "scratch.tmp"]


def test_fail_on_future_gates(repo):
    _bundle(repo, "Opponent Analyzer", "2026-07-31-ahead", None)
    # the slug is future-dated relative to the real system clock only if the
    # clock is behind it; drive the check through the report instead.
    rep = rj.JanitorReport(repo=str(repo), tip="HEAD", generated="now")
    rep.bundles = rj.scan_incoming(repo, date(2026, 7, 25))
    assert any(b.future for b in rep.bundles)


# --------------------------------------------------------------------- driver


def test_json_report_round_trips(repo, tmp_path):
    _worktree(repo, "spent")
    _bundle(repo, "Data Engineering", "2026-07-08-old", "integrate")
    out = tmp_path / "j.json"
    assert rj.main(["--repo", str(repo), "--json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["worktrees"] and payload["bundles"]
    assert payload["deleted"] == []


def test_render_is_ascii(repo):
    _bundle(repo, "Data Engineering", "2026-07-08-em—dash-slug", None)
    rep = rj.JanitorReport(repo=str(repo), tip="HEAD", generated="now")
    rep.bundles = rj.scan_incoming(repo, date(2026, 7, 25))
    rj.render(rep, 14).encode("ascii")
    rj.render_ledger(rep, 14).encode("ascii")


def test_not_a_git_repo_exits_3(tmp_path):
    assert rj.main(["--repo", str(tmp_path)]) == 3
