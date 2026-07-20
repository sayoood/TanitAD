"""Falsifier tests for session_guard — each builds a tiny throwaway git repo and
drives the real guard end-to-end (it shells out to git), fast (<1 s each). These
ARE the D-026 backlog falsifiers made executable:

  (b) uncommitted hub deliverable        -> BLOCK (exit 1)
      clean hub tree                      -> PASS  (exit 0)
  (a) an agent/* branch ahead of the tip -> WARNed (exit 0) / BLOCK under --strict
      current branch ahead                -> info only, never blocks
  (c) INTAKE with unfilled verdict + old  -> escalated; filled OR fresh -> not
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import session_guard as sg  # noqa: E402

TEMPLATE_VERDICT = "- **Verdict:** integrate / integrate-with-changes / defer / reject"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, f"git {' '.join(args)}: {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@t.t")
    _git(r, "config", "user.name", "t")
    (r / "README.md").write_text("seed\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "seed")
    return r


def _intake(repo: Path, slug: str, verdict_line: str) -> Path:
    d = repo / "TanitAD Research Hub" / "Tools&DevEnv" / "Implementation" / "incoming" / slug
    d.mkdir(parents=True)
    body = (
        "# INTAKE — test\n\n## What\nx\n\n"
        "## ORCHESTRATOR VERDICT (filled by the MVP stream)\n\n"
        f"{verdict_line}\n- **Date / by:** x\n"
    )
    (d / "INTAKE.md").write_text(body, encoding="utf-8")
    return d / "INTAKE.md"


# --------------------------------------------------------- (b) uncommitted hub

def test_clean_hub_passes(repo):
    assert sg.main(["--repo", str(repo)]) == 0


def test_uncommitted_hub_file_blocks(repo):
    note = repo / "TanitAD Research Hub" / "Tools&DevEnv" / "Research" / "2026-07-18-x.md"
    note.parent.mkdir(parents=True)
    note.write_text("finding\n", encoding="utf-8")  # untracked deliverable
    assert sg.main(["--repo", str(repo)]) == 1


def test_modified_project_state_blocks(repo):
    (repo / "PROJECT_STATE.md").write_text("state\n", encoding="utf-8")
    _git(repo, "add", "PROJECT_STATE.md")
    _git(repo, "commit", "-qm", "add state")
    (repo / "PROJECT_STATE.md").write_text("state changed\n", encoding="utf-8")  # dirty
    assert sg.main(["--repo", str(repo)]) == 1


def test_uncommitted_non_hub_file_does_not_block(repo):
    (repo / "scratch.txt").write_text("noise\n", encoding="utf-8")  # outside hub
    rep = sg.build_report(repo, "HEAD", sg.date.today(), 3)
    assert rep.uncommitted_hub == []
    assert sg.main(["--repo", str(repo)]) == 0


# ------------------------------------------------------- (a) unmerged branches

def test_stray_agent_branch_warns_not_blocks(repo):
    # branch with a commit the tip (main) does not have; we are NOT on it
    _git(repo, "checkout", "-q", "-b", "agent/x-20260101")
    (repo / "f.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "stray")
    _git(repo, "checkout", "-q", "main")
    rep = sg.build_report(repo, "HEAD", sg.date.today(), 3)
    strays = [b for b in rep.unmerged if b.name == "agent/x-20260101"]
    assert strays and strays[0].ahead == 1 and not strays[0].is_current
    assert sg.main(["--repo", str(repo)]) == 0            # warn only
    assert sg.main(["--repo", str(repo), "--strict"]) == 1  # strict blocks


def test_current_branch_ahead_is_info_only(repo):
    _git(repo, "checkout", "-q", "-b", "agent/tools-devenv-20260718")
    (repo / "g.txt").write_text("g\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "my work")
    # tip = main (behind), we are on the agent branch that is +1
    rep = sg.build_report(repo, "main", sg.date.today(), 3)
    cur = [b for b in rep.unmerged if b.is_current]
    assert cur and cur[0].ahead == 1
    # current-branch-ahead must never block, even in --strict
    assert sg.main(["--repo", str(repo), "--base", "main", "--strict"]) == 0


def test_merged_branch_not_flagged(repo):
    _git(repo, "checkout", "-q", "-b", "agent/done-20260101")
    (repo / "h.txt").write_text("h\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "done")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge", "agent/done-20260101")
    rep = sg.build_report(repo, "HEAD", sg.date.today(), 3)
    assert not any(b.name == "agent/done-20260101" for b in rep.unmerged)


# --------------------------------------------------------- (c) stale INTAKEs

def test_old_unfilled_intake_is_stale(repo):
    _intake(repo, "2026-07-01-old", TEMPLATE_VERDICT)
    rep = sg.build_report(repo, "HEAD", sg.date(2026, 7, 18), 3)
    assert [s.slug for s in rep.stale_intakes] == ["2026-07-01-old"]


def test_fresh_unfilled_intake_not_stale(repo):
    _intake(repo, "2026-07-17-fresh", TEMPLATE_VERDICT)
    rep = sg.build_report(repo, "HEAD", sg.date(2026, 7, 18), 3)  # 1 day old
    assert rep.stale_intakes == []


def test_filled_verdict_not_stale(repo):
    _intake(repo, "2026-07-01-triaged", "- **Verdict:** integrate")
    rep = sg.build_report(repo, "HEAD", sg.date(2026, 7, 18), 3)
    assert rep.stale_intakes == []


def test_stale_intake_warns_blocks_under_strict(repo):
    _intake(repo, "2026-07-01-old", TEMPLATE_VERDICT)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add intake")  # commit so (b) stays clean
    assert sg.main(["--repo", str(repo), "--now", "2026-07-18"]) == 0
    assert sg.main(["--repo", str(repo), "--now", "2026-07-18", "--strict"]) == 1


def test_missing_verdict_line_is_unfilled(repo):
    d = repo / "TanitAD Research Hub" / "X" / "Implementation" / "incoming" / "2026-06-01-noverdict"
    d.mkdir(parents=True)
    (d / "INTAKE.md").write_text("# INTAKE\n\nno verdict section here\n", encoding="utf-8")
    rep = sg.build_report(repo, "HEAD", sg.date(2026, 7, 18), 3)
    assert [s.slug for s in rep.stale_intakes] == ["2026-06-01-noverdict"]


# ------------------------------------------------------------------- plumbing

def test_verdict_unfilled_parser():
    assert sg._verdict_unfilled("## ORCHESTRATOR VERDICT\n- **Verdict:** \n")[0] is True
    assert sg._verdict_unfilled("## ORCHESTRATOR VERDICT\n" + TEMPLATE_VERDICT)[0] is True
    # non-committal placeholders under a real heading are still "no decision"
    assert sg._verdict_unfilled("## ORCHESTRATOR VERDICT\n- **Verdict:** _pending_\n")[0] is True
    assert sg._verdict_unfilled("## ORCHESTRATOR VERDICT\n- **Verdict:** TBD\n")[0] is True
    # no ORCHESTRATOR VERDICT heading at all -> unfilled (author-section verdict ignored)
    assert sg._verdict_unfilled("# INTAKE\n- **Verdict:** integrate\n")[0] is True
    ok = sg._verdict_unfilled("## ORCHESTRATOR VERDICT\n- **Verdict:** defer\n")
    assert ok[0] is False and ok[1] == "defer"


def test_json_mode_emits_and_returns_code(repo, capsys):
    note = repo / "PROJECT_STATE.md"
    note.write_text("x\n", encoding="utf-8")  # untracked hub deliverable -> block
    code = sg.main(["--repo", str(repo), "--json"])
    out = capsys.readouterr().out
    assert '"uncommitted_hub"' in out
    assert code == 1


def test_not_a_git_repo_returns_3(tmp_path):
    assert sg.main(["--repo", str(tmp_path)]) == 3


# --------------------------------------------- (b2) uncommitted source (v2, 2026-07-20)
#
# Added after the guard's own live run missed the biggest strand on the fleet: the
# shared Drive working tree held 40 uncommitted stack/ paths -- 12 untracked test
# modules (135 tests) and 9 untracked tanitad/lake/* modules -- while the hub check
# reported the tree "clean". The hub check only ever looked at hub prefixes.

def test_untracked_source_is_reported_separately_from_modified(repo):
    (repo / "stack" / "tests").mkdir(parents=True)
    (repo / "stack" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed stack")
    (repo / "stack" / "keep.py").write_text("x = 2\n", encoding="utf-8")       # modified
    (repo / "stack" / "tests" / "test_new.py").write_text("", encoding="utf-8")  # untracked

    src = sg.check_uncommitted_source(repo)
    assert src.untracked == ["stack/tests/test_new.py"]
    assert src.modified == ["stack/keep.py"]
    assert src.total == 2


def test_uncommitted_source_warns_but_does_not_block(repo):
    """A mid-work tree is legitimately dirty -- WARN is the honest default; only
    --strict turns it into a session-end block."""
    (repo / "stack").mkdir()
    (repo / "stack" / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert sg.main(["--repo", str(repo)]) == 0
    assert sg.main(["--repo", str(repo), "--strict"]) == 1


def test_clean_source_tree_reports_nothing(repo):
    assert sg.check_uncommitted_source(repo).total == 0


def test_source_check_ignores_paths_outside_the_source_prefixes(repo):
    (repo / "scratch.txt").write_text("noise\n", encoding="utf-8")
    assert sg.check_uncommitted_source(repo).total == 0


def test_source_rows_survive_quoted_paths_with_spaces(repo):
    """git quotes paths containing spaces; the destination must still be matched
    against the prefixes (the repo is full of 'Name With Spaces' dirs)."""
    (repo / "stack" / "a b").mkdir(parents=True)
    (repo / "stack" / "a b" / "c.py").write_text("x\n", encoding="utf-8")
    assert sg.check_uncommitted_source(repo).untracked == ["stack/a b/c.py"]


def test_render_lists_the_source_paths_for_the_operator(repo):
    (repo / "stack").mkdir()
    (repo / "stack" / "new.py").write_text("x = 1\n", encoding="utf-8")
    rep = sg.build_report(repo, "HEAD", sg.date.today(), 3)
    text, code = sg.render(rep, strict=False)
    assert code == 0
    assert "?? stack/new.py" in text and "UNTRACKED" in text
