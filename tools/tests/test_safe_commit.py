"""Falsifiers for safe_commit — every one drives a throwaway git repo end-to-end,
because the failures being guarded against are git's own behaviour and cannot be
unit-tested against a mock.

Each test IS one of the incidents in CLAUDE.md section "Git hygiene":

  * a sibling's staged work is swept into the wrong commit  (2x, 60265d3/3d41bd0)
  * `Keys.txt` reaches a commit                             (git-ignored, add -f)
  * a token pastes into a staged file
  * a stale .git/index.lock reads as "another git process"  (07-25 C8)
  * the tool itself emits `git commit -- <pathspec>`        (the segfaulting form)
  * a commit lands on main                                  (invariant)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import safe_commit  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, errors="replace", encoding="utf-8")
    assert proc.returncode == 0, f"git {args} -> {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-b", "work")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    (r / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(r, "add", "seed.txt")
    _git(r, "commit", "-m", "seed")
    return r


def _stage(repo: Path, rel: str, body: str = "x\n", force: bool = False) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    _git(repo, "add", *(["-f"] if force else []), rel)


def _run(repo: Path, *extra: str) -> int:
    return safe_commit.main(["--repo", str(repo), *extra])


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


# --------------------------------------------------------- the whole-index sweep


def test_declared_paths_only_commits(repo):
    _stage(repo, "tools/mine.py")
    before = _head(repo)
    assert _run(repo, "-p", "tools/", "-m", "mine") == 0
    assert _head(repo) != before
    assert _git(repo, "show", "--name-only", "--format=", "HEAD") == "tools/mine.py"


def test_siblings_staged_work_aborts_the_commit(repo):
    """60265d3 and 3d41bd0, made impossible: my file plus somebody else's
    half-finished module in the same index."""
    _stage(repo, "tools/mine.py")
    _stage(repo, "stack/tanitad/sibling_wip.py", "def half_written(:\n")
    before = _head(repo)
    assert _run(repo, "-p", "tools/", "-m", "mine") == 1
    assert _head(repo) == before, "nothing may be committed when the guard trips"


def test_accept_index_names_the_foreign_work_in_the_message(repo):
    """CLAUDE.md's escape hatch: commit the whole index, but SAY SO -- splitting
    it out would need the pathspec form, which segfaults."""
    _stage(repo, "tools/mine.py")
    _stage(repo, "stack/sibling.py")
    assert _run(repo, "-p", "tools/", "--accept-index", "-m", "mine") == 0
    body = _git(repo, "log", "-1", "--format=%B")
    assert "stack/sibling.py" in body
    assert "sibling" in body.lower()


def test_no_declaration_and_no_accept_index_refuses(repo):
    _stage(repo, "tools/mine.py")
    before = _head(repo)
    assert _run(repo, "-m", "no declaration") == 1
    assert _head(repo) == before


def test_empty_index_refuses(repo):
    assert _run(repo, "-p", "tools/", "-m", "nothing") == 1


def test_directory_prefix_and_glob_declarations_match(repo):
    _stage(repo, "TanitAD Research Hub/Data Engineering/x.md")
    _stage(repo, "tools/a.py")
    assert _run(repo, "-p", "TanitAD Research Hub/", "-p", "tools/*.py",
                "-m", "both") == 0


# ------------------------------------------------------------------ secret guard


def test_gitignored_keys_txt_is_refused(repo):
    """The real shape: Keys.txt is git-ignored, so it can only reach the index
    via `git add -f` -- and that leaves git's own ignore verdict as evidence."""
    (repo / ".gitignore").write_text("Keys.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _stage(repo, "Keys.txt", "hf_" + "A" * 34 + "\n", force=True)
    before = _head(repo)
    assert _run(repo, "-p", ".", "--accept-index", "-m", "oops") == 1
    assert _head(repo) == before


def test_keys_txt_refused_even_when_not_ignored(repo):
    """Two probes, not one (CLAUDE.md operating standard 2): the filename shape
    catches it in a repo whose .gitignore has drifted."""
    _stage(repo, "Keys.txt", "notes\n")
    assert _run(repo, "-p", "Keys.txt", "-m", "keys") == 1


def test_token_in_staged_content_is_refused(repo):
    _stage(repo, "tools/config.py", f'TOKEN = "hf_{"b" * 34}"\n')
    before = _head(repo)
    assert _run(repo, "-p", "tools/", "-m", "config") == 1
    assert _head(repo) == before


def test_token_is_never_echoed(repo, capsys):
    secret = "hf_" + "c" * 34
    _stage(repo, "tools/config.py", f'TOKEN = "{secret}"\n')
    _run(repo, "-p", "tools/", "-m", "config")
    out = capsys.readouterr()
    assert secret not in (out.out + out.err), "the tool must never print a token"
    assert "redacted" in out.out


def test_ordinary_hf_filenames_are_not_blocked(repo):
    """The loose `hf_[A-Za-z0-9]+` pattern from the brief matches this repo's own
    committed files (hf_export.py, hf_repo_state_2026-07-25.json). Blocking on it
    would make the tool unusable, so those are ADVISORY only."""
    _stage(repo, "tools/hf_export.py", "def hf_relay():\n    return 'hf_repo_state'\n")
    assert _run(repo, "-p", "tools/", "-m", "hf helper") == 0


def test_allow_secret_path_never_overrides_a_token(repo):
    # ⚠️ The PEM header is ASSEMBLED, never written as one literal. This file is
    # itself scanned by tools/secret_scan.py, and a whole-literal fixture makes
    # the repo-wide gate fire on its own test suite -- the same self-match trap
    # as a monitor whose filter contains the pattern it searches for. MEASURED
    # 2026-08-18: as a single literal this line was 1 of the 7 blocking findings
    # in the whole-tracked-repo scan, and it was the only artifact of its class.
    pem = "-----BEGIN RSA PRIVATE" + " KEY-----\n"
    _stage(repo, "tools/x.pem", pem)
    assert _run(repo, "-p", "tools/", "--allow-secret-path", "-m", "pem") == 1


# ------------------------------------------------------------------- the lock


def test_stale_index_lock_is_cleared_and_the_commit_proceeds(repo):
    """A crash leaves .git/index.lock; the next attempt then reports 'Another git
    process seems to be running', which reads like contention and is debris."""
    _stage(repo, "tools/mine.py")
    lock = repo / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    assert _run(repo, "-p", "tools/", "--force-lock", "-m", "after a crash") == 0
    assert not lock.exists()


def test_clear_stale_lock_reports_no_lock(repo):
    assert safe_commit.clear_stale_lock(repo, 0.0, True, log=lambda *_: None) is False


# ------------------------------------------------------- the form of the command


def test_the_tool_never_emits_a_pathspec_or_amend(repo, monkeypatch):
    """The load-bearing invariant. `git commit -- <pathspec>` segfaults ~50 % of
    the time on this repo (exit 139 / 0xC0000005) and three root-cause theories
    for it were falsified in one session, so the ONLY defence is never emitting
    the form."""
    seen: list[list[str]] = []
    real = subprocess.run

    def spy(cmd, *a, **kw):
        if isinstance(cmd, list) and cmd[:2] == ["git", "commit"]:
            seen.append(list(cmd))
        return real(cmd, *a, **kw)

    monkeypatch.setattr(safe_commit.subprocess, "run", spy)
    _stage(repo, "tools/mine.py")
    assert _run(repo, "-p", "tools/", "-m", "m") == 0
    assert seen, "no commit was attempted"
    for cmd in seen:
        assert "--" not in cmd, f"pathspec form emitted: {cmd}"
        assert "--amend" not in cmd, f"--amend emitted: {cmd}"
        assert cmd[2] == "-F" and len(cmd) == 4, f"unexpected commit form: {cmd}"


def test_a_crashed_git_that_still_committed_is_not_retried(repo, monkeypatch):
    """HEAD movement is the only trustworthy success signal: the crash can land
    the commit and still return 139. Retrying there would double-commit."""
    calls = {"n": 0}
    real = subprocess.run

    def spy(cmd, *a, **kw):
        if isinstance(cmd, list) and cmd[:2] == ["git", "commit"]:
            calls["n"] += 1
            real(cmd, *a, **kw)                     # the commit really happens
            return subprocess.CompletedProcess(cmd, 139, "", "Segmentation fault")
        return real(cmd, *a, **kw)

    monkeypatch.setattr(safe_commit.subprocess, "run", spy)
    _stage(repo, "tools/mine.py")
    before = _head(repo)
    assert _run(repo, "-p", "tools/", "-m", "m") == 0
    assert calls["n"] == 1, "must not retry once HEAD has moved"
    assert _head(repo) != before


def test_a_true_crash_is_retried(repo, monkeypatch):
    calls = {"n": 0}
    real = subprocess.run

    def spy(cmd, *a, **kw):
        if isinstance(cmd, list) and cmd[:2] == ["git", "commit"]:
            calls["n"] += 1
            if calls["n"] == 1:                     # crash without committing
                return subprocess.CompletedProcess(cmd, 139, "", "Segmentation fault")
        return real(cmd, *a, **kw)

    monkeypatch.setattr(safe_commit.subprocess, "run", spy)
    _stage(repo, "tools/mine.py")
    assert _run(repo, "-p", "tools/", "-m", "m", "--retries", "3") == 0
    assert calls["n"] == 2


# ------------------------------------------------------------------ misc guards


def test_commit_on_main_is_refused(tmp_path):
    r = tmp_path / "m"
    r.mkdir()
    _git(r, "init", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("a\n", encoding="utf-8")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-m", "seed")
    (r / "b.txt").write_text("b\n", encoding="utf-8")
    _git(r, "add", "b.txt")
    assert safe_commit.main(["--repo", str(r), "-p", "b.txt", "-m", "x"]) == 1
    assert safe_commit.main(["--repo", str(r), "-p", "b.txt", "-m", "x",
                             "--allow-main"]) == 0


def test_dry_run_commits_nothing(repo):
    _stage(repo, "tools/mine.py")
    before = _head(repo)
    assert _run(repo, "-p", "tools/", "-m", "m", "--dry-run") == 0
    assert _head(repo) == before


def test_print_index_is_read_only(repo):
    _stage(repo, "tools/mine.py")
    before = _head(repo)
    assert _run(repo, "--print-index") == 0
    assert _head(repo) == before


def test_message_file_is_used_verbatim(repo, tmp_path):
    msg = tmp_path / "msg.txt"
    msg.write_text("subject line\n\nbody with $dollar and `backtick`\n",
                   encoding="utf-8")
    _stage(repo, "tools/mine.py")
    assert _run(repo, "-p", "tools/", "-F", str(msg)) == 0
    assert "backtick" in _git(repo, "log", "-1", "--format=%B")


def test_paths_with_spaces_survive_the_index_listing(repo):
    _stage(repo, "TanitAD Research Hub/Data Engineering/note with space.md")
    plan = safe_commit.build_plan(repo, ["TanitAD Research Hub/"], False, True,
                                  False, False)
    assert plan.index == ["TanitAD Research Hub/Data Engineering/note with space.md"]
    assert not plan.foreign


@pytest.mark.parametrize("path,declared,ok", [
    ("tools/a.py", ["tools/"], True),
    ("tools/a.py", ["tools"], True),
    ("tools/a.py", ["tools/*.py"], True),
    ("tools/sub/a.py", ["tools/*.py"], False),
    ("stack/a.py", ["tools/"], False),
    ("toolsX/a.py", ["tools/"], False),
])
def test_declaration_matching(path, declared, ok):
    assert safe_commit.matches_declared(path, declared) is ok


def test_not_a_git_repo_exits_3(tmp_path):
    assert safe_commit.main(["--repo", str(tmp_path), "--print-index"]) == 3
