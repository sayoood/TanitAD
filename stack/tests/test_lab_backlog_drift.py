"""The backlog-drift guard: does it FIRE when findings outrun the backlog?

⛔ WHY THE DELIBERATE-REGRESSION CASE IS THE IMPORTANT ONE. Per the validation
standard: if the gate does not FAIL the regression arm, a PASS on the fixed arm means
nothing. A drift checker that always reports zero is indistinguishable from a healthy
repo, and that is exactly the failure mode it was written to catch -- the rule it
enforces had already been silently broken for two days.

⚠️ Builds a real temp git repo rather than mocking `git`. A mock would test the mock's
idea of `git log`, and the thing most likely to be wrong here is the argument form
(--git-dir/--work-tree, the \\x1f separator, the -- pathspec), which a mock cannot check.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lab_backlog_drift as m  # noqa: E402


def _run(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


def _commit(repo: Path, rel: str, text: str, msg: str):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    _run(repo, "add", "--", rel)
    _run(repo, "commit", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@t")
    _run(r, "config", "user.name", "t")
    _commit(r, m.BACKLOG, "# backlog\n", "backlog: initial")
    return r


def test_reports_zero_when_the_backlog_is_current(repo, capsys):
    """The healthy case must be clean AND exit 0, or nothing can gate on it."""
    assert m.main.__module__  # the module imported
    b = m.last_touch(repo, m.BACKLOG)
    assert b, "the backlog must have a commit"
    assert m.findings_since(repo, b) == []


def test_FIRES_when_a_finding_outruns_the_backlog(repo):
    """⭐ THE REGRESSION CASE. This is the situation that actually occurred:
    findings registered while the backlog sat untouched."""
    b = m.last_touch(repo, m.BACKLOG)
    _commit(repo, m.FINDING_FILES[0], "# claims\nMM-E99 something\n",
            "MM-E99: a finding nobody proposed")
    drift = m.findings_since(repo, b)
    assert len(drift) == 1, "a finding after the backlog commit must be reported"
    assert "MM-E99" in drift[0][2]


def test_a_backlog_update_CLEARS_the_drift(repo):
    """⚠️ The guard must be satisfiable — one that can never go green gets muted."""
    b = m.last_touch(repo, m.BACKLOG)
    _commit(repo, m.FINDING_FILES[0], "# claims\nMM-E99\n", "MM-E99: finding")
    assert m.findings_since(repo, b)
    _commit(repo, m.BACKLOG, "# backlog\n\n## PROPOSED\nMM-E99 row\n",
            "backlog: propose MM-E99")
    assert m.findings_since(repo, m.last_touch(repo, m.BACKLOG)) == []


def test_finding_files_is_EXPLICIT_not_a_wildcard():
    """⛔ A wildcard would sweep in formatting commits and train readers to ignore
    the output — a guard that cries wolf is worse than none."""
    assert isinstance(m.FINDING_FILES, tuple) and m.FINDING_FILES
    assert all(f.endswith(".md") for f in m.FINDING_FILES)
    assert not any("*" in f for f in m.FINDING_FILES)


def test_printed_output_is_ASCII_only():
    """⚠️ The dev box is cp1252. A guard that dies on its own house glyphs fails
    closed for the wrong reason (MEASURED 2026-08-21). Glyphs belong in comments."""
    src = (Path(m.__file__)).read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("print(") or s.startswith('print(f"'):
            assert s.isascii(), f"non-ASCII in a printed line: {s[:70]}"
