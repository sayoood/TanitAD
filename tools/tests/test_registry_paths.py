"""Tests for tools/registry_paths.py.

The failure mode being guarded is symmetric and both halves have already
happened here:

  * calling a path MISSING when the file plainly exists (the first cut reported
    85 MISSING, of which ~70 were bare filenames and one — `RESULTS_camcond.md`
    — was a real file the matcher could not see because the ellipsis elided a
    *prefix of a path component*); and
  * calling a pod path MISSING because it cannot be reached from the repo,
    which is an absence claim from a single failed location.

Both manufacture defects out of a weak matcher, which is exactly the class the
registry is supposed to protect against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import registry_paths as rp  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    """A miniature repo with the shapes the registry actually cites."""
    hub = tmp_path / "TanitAD Research Hub"
    (hub / "Arch" / "Implementation" / "incoming" /
     "2026-07-22-own-dynamics-encoder").mkdir(parents=True)
    (hub / "Arch" / "Implementation" / "incoming" /
     "2026-07-22-own-dynamics-encoder" / "RESULTS_camcond.md").write_text("x")
    (tmp_path / "taniteval" / "results").mkdir(parents=True)
    (tmp_path / "taniteval" / "results" / "driving_a.json").write_text("{}")
    (tmp_path / "stack" / "scripts").mkdir(parents=True)
    (tmp_path / "stack" / "scripts" / "ci.py").write_text("")
    # a pod-rescue COPY of a canonical artifact
    resc = tmp_path / "stack" / "experiments" / "pod-rescue-20260802" / "pod3" / \
        "root" / "taniteval" / "results"
    resc.mkdir(parents=True)
    (resc / "driving_a.json").write_text("{}")
    # an agent worktree copy — must never be indexed
    wt = tmp_path / ".claude" / "worktrees" / "agent-x" / "stack" / "scripts"
    wt.mkdir(parents=True)
    (wt / "ci.py").write_text("")
    rp._INDEX_CACHE.clear()
    yield tmp_path
    rp._INDEX_CACHE.clear()


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tok,kind", [
    ("taniteval/results/x.json", "repo"),
    ("ckpt.pt", "name"),
    ("/workspace/experiments/a.pt", "abspath"),
    ("tanitad-pod2:/workspace/a.pt", "remote"),
    ("a/{b,c}.json", "glob"),
    ("results/<key>.json", "glob"),
    ("eff_*.json", "glob"),
    ("…/incoming/x/y.json", "ellipsis"),
    (".../incoming/x/y.json", "ellipsis"),
])
def test_classify(tok, kind):
    assert rp.classify(tok) == kind


# ---------------------------------------------------------------------------
# The two fabricated-defect classes
# ---------------------------------------------------------------------------
def test_ellipsis_matches_a_dated_directory_prefix(tree):
    """`…/own-dynamics-encoder/X.md` must find `2026-07-22-own-dynamics-encoder/X.md`."""
    r = rp.resolve("…/own-dynamics-encoder/RESULTS_camcond.md", tree)
    assert r["status"] == rp.EXISTS
    assert r["resolved"].endswith(
        "2026-07-22-own-dynamics-encoder/RESULTS_camcond.md")


def test_abspath_is_not_checked_never_missing(tree):
    r = rp.resolve("/workspace/experiments/nope.pt", tree)
    assert r["status"] == rp.NOT_CHECKED
    assert r["stranded"] is True


def test_abspath_reports_a_repo_counterpart_when_banked(tree):
    r = rp.resolve("/workspace/TanitAD/stack/scripts/ci.py", tree)
    assert r["status"] == rp.NOT_CHECKED
    assert r["repo_counterpart"] == "stack/scripts/ci.py"
    assert r["stranded"] is False


def test_remote_host_path_is_not_checked(tree):
    r = rp.resolve("tanitad-pod2:/workspace/experiments/x/ckpt.pt", tree)
    assert r["status"] == rp.NOT_CHECKED
    assert r["host"] == "tanitad-pod2"


def test_bare_name_is_name_only_not_missing(tree):
    r = rp.resolve("ckpt.pt", tree)
    assert r["status"] == rp.NAME_ONLY


def test_brace_expansion_is_not_a_path(tree):
    r = rp.resolve("taniteval/results/{windows,fan}_refc-{base,xl}-30k.pt", tree)
    assert r["status"] == rp.NOT_A_PATH


def test_placeholder_template_is_not_a_path(tree):
    assert rp.resolve("results/<key>.json", tree)["status"] == rp.NOT_A_PATH


# ---------------------------------------------------------------------------
# Index hygiene
# ---------------------------------------------------------------------------
def test_worktrees_are_never_indexed(tree):
    """`.claude/worktrees/*` are transient copies of this repo. Indexing them
    turned every real hit into an 8-way AMBIGUOUS."""
    hits = rp.file_index(tree).get("ci.py", [])
    assert len(hits) == 1
    # Assert on path COMPONENTS, not a substring of the whole path: pytest's
    # own tmp dir for this test is named `test_worktrees_are_never_index0`,
    # so a substring check passes/fails for the wrong reason.
    assert "worktrees" not in hits[0].relative_to(tree).parts
    assert hits[0].relative_to(tree).parts == ("stack", "scripts", "ci.py")


def test_pod_rescue_copy_does_not_make_a_citation_ambiguous(tree):
    """A rescue dump is a COPY, not a second artifact."""
    r = rp.resolve("taniteval/results/driving_a.json", tree)
    assert r["status"] == rp.EXISTS
    assert "pod-rescue" not in r["resolved"]


def test_existing_repo_path_resolves(tree):
    assert rp.resolve("stack/scripts/ci.py", tree)["status"] == rp.EXISTS


def test_genuinely_missing_repo_path_is_missing(tree):
    assert rp.resolve("stack/scripts/ghost.py", tree)["status"] == rp.MISSING


# ---------------------------------------------------------------------------
# Extraction + exit code
# ---------------------------------------------------------------------------
def test_extract_only_takes_backticked_artifact_like_tokens():
    text = ("see `taniteval/results/a.json` and `stack/x.py` but not "
            "taniteval/results/b.json nor `just prose` nor `https://x/y.json`")
    got = rp.extract_citations(text)
    assert "taniteval/results/a.json" in got
    assert "stack/x.py" in got
    assert "taniteval/results/b.json" not in got
    assert not any(g.startswith("http") for g in got)


def test_sweep_counts_and_exit_code(tree):
    reg = tree / "reg.md"
    reg.write_text("`stack/scripts/ci.py` and `stack/scripts/ghost.py`",
                   encoding="utf-8")
    res = rp.sweep(reg, tree)
    assert res["counts"][rp.EXISTS] == 1
    assert res["counts"][rp.MISSING] == 1
    assert rp.exit_code(res) == 1


def test_exit_code_zero_when_clean(tree):
    reg = tree / "reg.md"
    reg.write_text("`stack/scripts/ci.py`", encoding="utf-8")
    assert rp.exit_code(rp.sweep(reg, tree)) == 0


def test_name_only_does_not_trip_the_exit_code(tree):
    reg = tree / "reg.md"
    reg.write_text("`ckpt.pt` `README.md`", encoding="utf-8")
    assert rp.exit_code(rp.sweep(reg, tree)) == 0
