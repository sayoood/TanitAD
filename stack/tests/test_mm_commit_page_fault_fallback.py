"""mm_commit.py must survive the two failures that actually happened.

Both are reproduced at the REAL subprocess boundary by a git shim, not mocked:

  1. `git add` exiting 0xC0000006 (STATUS_IN_PAGE_ERROR) with empty stderr.
     MEASURED 2026-09-07: it did this on all 8 retries for minutes, on 13 paths,
     WHILE `git hash-object` on the same paths in the same seconds succeeded --
     so the mount was not down, one code path was. The tool died there, which
     blocks committing entirely.
  2. `git diff --cached --name-only` returning EMPTY for a commit that really
     changed files. CLAUDE.md records the mount doing this for 146 files. The
     tool read that as "nothing to commit" and EXITED 0 -- a false success that
     tells an agent its work landed when HEAD never moved.

Each fix ships with its DELIBERATE-REGRESSION arm: the historical code is
reconstructed by patching the fix back out, and must FAIL. A guard that cannot
go RED proves nothing (CLAUDE.md: "a check that shares the defect it checks for
is green forever"), and every expectation below is a LITERAL, never an
expression over the code under test.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "mm_commit.py"
SHIM = Path(__file__).resolve().parent / "_data" / "mm_commit_gitshim.py"


def _git(repo: Path, *a: str) -> str:
    r = subprocess.run(["git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}", *a],
                       capture_output=True, text=True)
    return r.stdout.strip()


@pytest.fixture()
def repo(tmp_path_factory) -> Path:
    r = Path(tempfile.mkdtemp(prefix="mmtest-"))
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t"), ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(r), "config", k, v], check=True)
    (r / "kept.txt").write_text("untouched\n", encoding="utf-8")
    (r / "target.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(r), "add", "kept.txt", "target.txt"], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-qm", "base"], check=True)
    (r / "target.txt").write_text("AFTER-THE-EDIT\n", encoding="utf-8")
    return r


def _run(tool: Path, repo: Path, *, fail_add: bool = False, empty_diff: bool = False,
         timeout: int = 40):
    msg = repo / "_msg.txt"
    msg.write_text("test commit\n", encoding="utf-8")
    env = dict(os.environ)
    env["MM_COMMIT_REPO"] = str(repo)
    env["MM_COMMIT_GIT"] = f'"{sys.executable}" "{SHIM}"'
    env["PYTHONIOENCODING"] = "utf-8"
    # the ladder's DURATION is not under test; its OUTCOME is
    env["MM_COMMIT_SLEEP_SCALE"] = "0.02"
    if fail_add:
        env["MM_SHIM_FAIL_ADD"] = "1"
    if empty_diff:
        env["MM_SHIM_EMPTY_DIFF"] = "1"
    try:
        r = subprocess.run([sys.executable, str(tool), str(msg), "target.txt"],
                           capture_output=True, text=True, env=env,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""
        return None, out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)


def _regressed(kind: str) -> Path:
    """The HISTORICAL tool: the fix patched back out. The deliberate-regression arm."""
    src = TOOL.read_text(encoding="utf-8")
    if kind == "add":
        # before the fallback existed, staging was a bare `git add`
        assert "_stage(p, env)" in src
        src = src.replace("        _stage(p, env)",
                          '        git("add", "--", p, env=env)')
    elif kind == "diff":
        # before the positive assertion, an empty delta meant "nothing to commit"
        marker = "        really = []"
        assert marker in src
        head, tail = src.split(marker, 1)
        rest = tail.split('        raise SystemExit(0)', 1)[1]
        src = (head
               + '        print("  nothing to commit (already identical to HEAD)")\n'
               + "        _rm(scratch_dir)\n"
               + "        raise SystemExit(0)" + rest)
    else:                                                    # pragma: no cover
        raise AssertionError(kind)
    p = Path(tempfile.mkdtemp(prefix="mmreg-")) / "mm_commit_REGRESSED.py"
    p.write_text(src, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# 1. the page-faulting `git add`
# --------------------------------------------------------------------------

def test_shim_is_actually_in_the_loop(repo: Path) -> None:
    """Control: without it, every assertion below would be about the real git."""
    env = dict(os.environ, MM_SHIM_FAIL_ADD="1")
    r = subprocess.run([sys.executable, str(SHIM), "add", "--", "target.txt"],
                       capture_output=True, text=True, env=env, cwd=repo)
    assert r.returncode == 3221225478, (
        "the shim must reproduce 0xC0000006 exactly; got %r" % r.returncode)


def test_add_page_fault_still_commits_via_cacheinfo(repo: Path) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    rc, out = _run(TOOL, repo, fail_add=True)
    after = _git(repo, "rev-parse", "HEAD")
    assert rc == 0, f"the commit must land despite add failing.\n{out}"
    assert after != before, f"HEAD must move.\n{out}"
    assert "cacheinfo" in out, f"it must say it fell back.\n{out}"
    assert _git(repo, "show", "HEAD:target.txt") == "AFTER-THE-EDIT", out
    # and the commit stays SCOPED -- the unnamed file is untouched
    assert _git(repo, "show", "HEAD:kept.txt") == "untouched", out


def test_add_page_fault_KILLS_the_historical_tool(repo: Path) -> None:
    """RED arm. If this ever passes, the test above is measuring nothing."""
    before = _git(repo, "rev-parse", "HEAD")
    rc, out = _run(_regressed("add"), repo, fail_add=True, timeout=60)
    assert rc != 0, f"the pre-fix tool must DIE on the page fault.\n{out}"
    assert _git(repo, "rev-parse", "HEAD") == before, "and must not have committed"


# --------------------------------------------------------------------------
# 2. the falsely-empty `git diff --cached`
# --------------------------------------------------------------------------

def test_empty_diff_is_not_believed(repo: Path) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    rc, out = _run(TOOL, repo, empty_diff=True, timeout=25)
    assert rc != 0, f"an empty diff must NEVER be reported as success.\n{out}"
    assert "the mount lied" in out, f"it must name what happened.\n{out}"
    assert _git(repo, "rev-parse", "HEAD") == before


def test_empty_diff_FALSELY_SUCCEEDS_on_the_historical_tool(repo: Path) -> None:
    """RED arm: this is the exact false success that shipped."""
    before = _git(repo, "rev-parse", "HEAD")
    rc, out = _run(_regressed("diff"), repo, empty_diff=True, timeout=60)
    assert rc == 0, f"the pre-fix tool exited 0 here; that was the defect.\n{out}"
    assert "nothing to commit" in out, out
    assert _git(repo, "rev-parse", "HEAD") == before, (
        "and it said so while HEAD never moved -- the whole point")
