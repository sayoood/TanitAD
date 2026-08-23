"""secret_scan: renames of tracked paths are not new exposure; paper titles are
not credential files — and the guard STILL catches a real `git add -f`.

MEASURED 2026-08-23: the Research Hub -> Research Lab migration staged 5,342
renames. The scanner reported 42 long-tracked, force-added media files as
"git-ignored-but-staged" (a rename is not a new `git add -f`) and blocked on
`2305.18290_Direct-Preference-Optimization-...-is-Secret.pdf` (a paper title
matching `*secret*`). Both narrowed; the negative control below proves the
narrowing did not blind the guard.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import secret_scan as S  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("*.mp4\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "commit", "-qm", "ignore mp4")
    return tmp_path


def test_rename_of_a_tracked_ignored_file_is_not_flagged(repo: Path):
    """A force-added file, committed long ago, then RENAMED: no new exposure."""
    (repo / "old.mp4").write_bytes(b"\x00" * 64)
    _git(repo, "add", "-f", "old.mp4")
    _git(repo, "commit", "-qm", "force-added media (the decision was made here)")
    _git(repo, "mv", "old.mp4", "new.mp4")
    rep = S.scan_staged(repo)
    ignored = [f for f in rep.findings if f.pattern == "git-ignored-but-staged"]
    assert ignored == [], (
        "a RENAME of an already-tracked force-added file must not be reported "
        f"as a fresh `git add -f`: {[f.path for f in ignored]}")


def test_NEGATIVE_CONTROL_a_fresh_force_add_is_STILL_caught(repo: Path):
    """⛔ The guard must still bite on the case it exists for."""
    (repo / "leak.mp4").write_bytes(b"\x00" * 64)
    _git(repo, "add", "-f", "leak.mp4")
    rep = S.scan_staged(repo)
    ignored = [f for f in rep.findings if f.pattern == "git-ignored-but-staged"]
    assert any(f.path.endswith("leak.mp4") for f in ignored), (
        "the narrowing BLINDED the guard: a fresh `git add -f` of an ignored "
        "file was not reported")


def test_a_paper_title_containing_secret_is_not_a_credential_filename():
    hits = S.scan_path_shape(
        "TanitAD Research Lab/Library/papers/"
        "2305.18290_Direct-Preference-Optimization-Your-Language-Model-is-Secret.pdf")
    assert not [h for h in hits if h.pattern == "credential-filename"], (
        "a banked paper whose TITLE contains 'secret' is a document, not a "
        "credential file; its content is still scanned")


def test_NEGATIVE_CONTROL_a_real_secret_filename_is_STILL_caught():
    hits = S.scan_path_shape("config/my_secret_token.json")
    assert [h for h in hits if h.pattern == "credential-filename"], (
        "the .pdf exemption must not leak into other suffixes")
