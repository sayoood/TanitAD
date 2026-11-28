"""G-SUITE-PINNED's throwaway repo on a filesystem that records NO file ownership (exFAT).

MEASURED 2026-10-12 on the dev box: `--suite-work E:/lgs` (exFAT) made every git call in the
throwaway repo die "detected dubious ownership", its HEAD read None, and G-SUITE-PINNED ERRORed
after a 13-minute, 34 GB archive -- before a single pinned test ran. The fix gives the throwaway
repo, and the pinned tests' environment, ONE command-scope `safe.directory` entry for that repo
only (`launch_gate.safe_dir_env`); no git config file is written.

exFAT is not needed to reproduce it: git's own test knob `GIT_TEST_ASSUME_DIFFERENT_OWNER=1` makes
git treat any repository as foreign-owned. The CONTROL below asserts the knob really refuses on this
host's git, so the fix arm cannot pass against a git that was never refusing; the REGRESSION arm
removes the fix and must read HEAD None again. Expectations are literals or the repo's own sha.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import launch_gate as LG  # noqa: E402

KNOB = "GIT_TEST_ASSUME_DIFFERENT_OWNER"


def _clean() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(*args: str, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "src"
    src.mkdir()
    env = _clean()
    assert _git("init", "-q", cwd=src, env=env).returncode == 0
    (src / "stack").mkdir()
    (src / "stack" / "a.py").write_bytes(b"A = 1\n")
    assert _git("add", "stack/a.py", cwd=src, env=env).returncode == 0
    assert _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c",
                cwd=src, env=env).returncode == 0
    sha = _git("rev-parse", "HEAD", cwd=src, env=env).stdout.strip()
    assert len(sha) == 40
    return src, sha


def _archive(tmp_path: Path, name: str) -> Path:
    dest = tmp_path / name
    (dest / "stack").mkdir(parents=True)
    (dest / "stack" / "a.py").write_bytes(b"A = 1\n")
    return dest


def test_safe_dir_env_APPENDS_one_entry_and_leaves_the_callers_entries_alone(tmp_path):
    base = {"PATH": "p", "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "core.autocrlf", "GIT_CONFIG_VALUE_0": "false",
            "GIT_CONFIG_KEY_1": "user.name", "GIT_CONFIG_VALUE_1": "t"}
    before = dict(base)
    got = LG.safe_dir_env(base, tmp_path)
    assert base == before                                  # the caller's dict is not mutated
    assert got["GIT_CONFIG_COUNT"] == "3"
    assert got["GIT_CONFIG_KEY_0"] == "core.autocrlf" and got["GIT_CONFIG_VALUE_0"] == "false"
    assert got["GIT_CONFIG_KEY_1"] == "user.name" and got["GIT_CONFIG_VALUE_1"] == "t"
    assert got["GIT_CONFIG_KEY_2"] == "safe.directory"
    assert got["GIT_CONFIG_VALUE_2"] == tmp_path.resolve().as_posix()
    fresh = LG.safe_dir_env({"PATH": "p"}, tmp_path)
    assert sorted(fresh) == ["GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "PATH"]
    assert fresh["GIT_CONFIG_COUNT"] == "1" and fresh["GIT_CONFIG_KEY_0"] == "safe.directory"


def test_the_throwaway_repo_reaches_HEAD_where_git_sees_a_FOREIGN_owner(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.fail("git is not on PATH -- G-SUITE-PINNED cannot run here either")
    src, sha = _source_repo(tmp_path)

    # CONTROL: on this host's git the knob must really refuse (else the fix arm proves nothing)
    ctl = _archive(tmp_path, "control")
    assert _git("init", "-q", cwd=ctl, env=_clean()).returncode == 0
    knob = dict(_clean(), **{KNOB: "1"})
    r = _git("rev-parse", "--git-dir", cwd=ctl, env=knob)
    assert r.returncode == 128 and "dubious ownership" in r.stderr, (r.returncode, r.stderr[-300:])

    monkeypatch.setenv(KNOB, "1")

    # REGRESSION ARM: without the safe.directory entry the 2026-10-12 failure comes back
    monkeypatch.setattr(LG, "safe_dir_env", lambda env, dest: dict(env))
    broken = LG.throwaway_repo(_archive(tmp_path, "broken"), str(src / ".git"), sha)
    assert broken["head"] is None
    # errors keep stderr's LAST 300 chars: "dubious ownership" is cut, git's advice line survives
    assert any("safe.directory" in e for e in broken["errors"]), broken["errors"]
    monkeypatch.undo()
    monkeypatch.setenv(KNOB, "1")

    # FIX ARM: the same foreign-owned repo reaches HEAD == the commit, and is clean against it
    dest = _archive(tmp_path, "archive")
    got = LG.throwaway_repo(dest, str(src / ".git"), sha)
    assert got["head"] == sha, got
    assert got["errors"] == []

    # ... and the PINNED TESTS' environment can run git inside it too (secret_scan's hook test)
    env = LG.pinned_env(dest, {"omp": 4, "cpu_only": True}, dict(_clean(), **{KNOB: "1"}))
    assert _git("rev-parse", "HEAD", cwd=dest / "stack", env=env).stdout.strip() == sha
    st = _git("status", "--porcelain", cwd=dest, env=env)
    assert st.returncode == 0 and st.stdout == "", (st.returncode, st.stdout, st.stderr[-300:])
    # the entry names THIS repo only -- a sibling foreign-owned repo is still refused
    other = _archive(tmp_path, "other")
    assert _git("init", "-q", cwd=other, env=_clean()).returncode == 0
    assert _git("rev-parse", "--git-dir", cwd=other, env=env).returncode == 128
