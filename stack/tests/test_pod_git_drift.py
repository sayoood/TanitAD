"""Tests for the box-vs-git drift detector.

The verdict logic is what protects us from the 2026-07-20 failure mode (code
living on exactly one pod disk), so it is tested directly rather than through
ssh. HOST_ONLY is the finding that matters: it means the file exists nowhere
else on earth.

⚠️ Four of these pin the 2026-08-18 repair, and each has a measured defect
behind it: the index used to walk ``.claude/worktrees/`` (**8,079 indexed files
against the repo's 2,132**); it matched by basename anywhere in the repo; it did
no CRLF normalisation (**28.1 % of ``stack/`` sources contain CRLF**, and ~94 %
of the rows it would print as drift were that artifact); and an unreachable host
returned ``[]``, which downstream read as *"clean"*.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pod_git_drift import (  # noqa: E402
    CRLF_ONLY, DRIFTED, HOST_ONLY, IN_GIT, NAME_ONLY, POD_ONLY, _match_by_path,
    classify, repo_index, sha256_bytes, sha256_lf,
)


def _index(files: dict[str, bytes]) -> dict:
    """Build an index directly, mirroring repo_index's output shape."""
    by_hash, by_lf, by_name, by_path = {}, {}, {}, {}
    for path, data in files.items():
        raw, lf = sha256_bytes(data), sha256_lf(data)
        by_hash.setdefault(raw, []).append(path)
        by_lf.setdefault(lf, []).append(path)
        by_name.setdefault(path.rsplit("/", 1)[-1], set()).add(raw)
        by_path[path] = {"raw": raw, "lf": lf}
    return {"by_hash": by_hash, "by_lf": by_lf, "by_name": by_name,
            "by_path": by_path}


def test_identical_content_is_in_git():
    idx = _index({"stack/scripts/a.py": b"print(1)\n"})
    got = classify([(sha256_bytes(b"print(1)\n"), "/root/a.py")], idx)
    assert got[0]["verdict"] == IN_GIT


def test_same_path_different_content_is_drifted():
    """The box is running something we cannot rebuild from HEAD."""
    idx = _index({"stack/scripts/refb_train.py": b"v1\n"})
    got = classify(
        [(sha256_bytes(b"v2-modified\n"),
          "/root/TanitAD/stack/scripts/refb_train.py")], idx)
    assert got[0]["verdict"] == DRIFTED
    assert "stack/scripts/refb_train.py" in got[0]["note"]


def test_unknown_file_is_host_only():
    """This is the REF-B v2 / TanitEval case — one disk, no copy anywhere."""
    idx = _index({"stack/scripts/a.py": b"x\n"})
    got = classify([(sha256_bytes(b"secret sauce\n"), "/root/refb_v4.py")], idx)
    assert got[0]["verdict"] == HOST_ONLY == POD_ONLY
    assert "no file of this name" in got[0]["note"]


# --------------------------------------------------------------------------
# ⚠️ the 2026-08-18 repair — four defects, each of which read as an answer
# --------------------------------------------------------------------------


def test_crlf_difference_is_its_own_verdict_and_is_never_drift():
    """⛔ The largest single effect of the repair.

    The repo tree is CRLF, every box is LF. MEASURED: 28.1 % of ``stack/`` and
    19.3 % of ``taniteval/`` sources contain CRLF, and on a well-synced box
    ~94 % of the rows this printed as drift were that artifact (C105, 47 of 50).
    """
    crlf = b"import os\r\ndef f():\r\n    return 1\r\n"
    lf = crlf.replace(b"\r\n", b"\n")
    idx = _index({"stack/scripts/roll.py": crlf})
    got = classify([(sha256_bytes(lf),
                     "/home/nvidia/TanitAD/stack/scripts/roll.py")], idx)
    assert got[0]["verdict"] == CRLF_ONLY
    assert "NOT drift" in got[0]["note"]
    # and the byte delta really is one \r per line — the arithmetic C105 used
    assert len(crlf) - len(lf) == 3


def test_a_genuine_edit_survives_lf_normalisation_and_is_still_drift():
    """The normaliser must not become a way to hide real drift."""
    idx = _index({"stack/scripts/roll.py": b"A = 1\r\nB = 2\r\n"})
    got = classify([(sha256_bytes(b"A = 1\nB = 3\n"),
                     "/home/nvidia/TanitAD/stack/scripts/roll.py")], idx)
    assert got[0]["verdict"] == DRIFTED


def test_a_same_basename_in_an_unrelated_tree_is_not_called_drift():
    """⛔ The basename lottery.

    ``/root/experiments/scratch/utils.py`` has nothing to do with
    ``stack/tanitad/utils.py``. Calling that DRIFTED manufactures a finding —
    and ``__init__.py`` alone has dozens of repo copies.
    """
    idx = _index({"stack/tanitad/utils.py": b"real\n"})
    got = classify([(sha256_bytes(b"unrelated\n"),
                     "/root/experiments/scratch/utils.py")], idx)
    assert got[0]["verdict"] == NAME_ONLY
    assert "not drift" in got[0]["note"]


def test_longest_path_suffix_wins_over_a_shallow_coincidence():
    """A full-path agreement must outrank a two-segment collision."""
    by_path = {"stack/scripts/v6_chain.py": {}, "tools/v6_chain.py": {}}
    assert (_match_by_path("/home/nvidia/TanitAD/stack/scripts/v6_chain.py",
                           by_path) == "stack/scripts/v6_chain.py")
    # nothing plausible-but-wrong when there is no match at all
    assert _match_by_path("/root/other/zzz.py", by_path) is None


def test_an_empty_host_list_is_an_error_not_a_clean_result():
    """⛔ ``--hosts`` with no values scanned nothing and reported ``TOTAL: 0``.

    Same shape as the four dead default pods: a reassuring answer to a question
    that was never asked. Absence of a finding is not a finding of absence.
    """
    from pod_git_drift import main
    assert main(["--hosts"]) == 2


def test_repo_index_excludes_worktree_copies(tmp_path):
    """⛔ 14 stale full repo copies inflated the index 3.8× (8,079 vs 2,132).

    Every IN_GIT / DRIFTED judgement was being made against that pool.
    """
    (tmp_path / "stack").mkdir()
    (tmp_path / "stack/live.py").write_bytes(b"current\n")
    wt = tmp_path / ".claude" / "worktrees" / "agent-x" / "stack"
    wt.mkdir(parents=True)
    (wt / "live.py").write_bytes(b"ancient\n")
    (wt / "ghost.py").write_bytes(b"only in the worktree\n")

    idx = repo_index(tmp_path)

    assert set(idx["by_path"]) == {"stack/live.py"}
    assert "ghost.py" not in idx["by_name"], (
        "a worktree-only file must not mask a genuine HOST_ONLY finding")
    assert sha256_bytes(b"ancient\n") not in idx["by_hash"]


def test_relocated_file_still_counts_as_in_git():
    """Content match wins over path — a rescued file lands at a new repo path."""
    idx = _index({"stack/experiments/refb-v2/refb_v4.py": b"arch\n"})
    got = classify([(sha256_bytes(b"arch\n"), "/root/refb_v4.py")], idx)
    assert got[0]["verdict"] == IN_GIT


def test_mixed_batch_classifies_each_independently():
    idx = _index({"s/a.py": b"A\n", "s/b.sh": b"B\n"})
    got = classify(
        [
            (sha256_bytes(b"A\n"), "/root/s/a.py"),
            (sha256_bytes(b"B-changed\n"), "/root/s/b.sh"),
            (sha256_bytes(b"C\n"), "/root/s/c.py"),
        ],
        idx,
    )
    assert [g["verdict"] for g in got] == [IN_GIT, DRIFTED, HOST_ONLY]


def test_repo_index_picks_up_sources_and_skips_noise(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "real.py").write_bytes(b"code\n")
    (tmp_path / "pkg" / "tool.sh").write_bytes(b"#!/bin/sh\n")
    (tmp_path / "pkg" / "notes.md").write_bytes(b"# doc\n")
    cache = tmp_path / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "real.py").write_bytes(b"compiled\n")

    idx = repo_index(tmp_path)

    assert "real.py" in idx["by_name"]
    assert "tool.sh" in idx["by_name"]
    assert "notes.md" not in idx["by_name"]          # not a source suffix
    # __pycache__ copy must not mask a genuine pod-only finding
    assert idx["by_name"]["real.py"] == {sha256_bytes(b"code\n")}
