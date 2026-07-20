"""Tests for the pod-vs-git drift detector.

The verdict logic is what protects us from the 2026-07-20 failure mode (code
living on exactly one pod disk), so it is tested directly rather than through
ssh. POD_ONLY is the finding that matters: it means the file exists nowhere
else on earth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pod_git_drift import (  # noqa: E402
    DRIFTED, IN_GIT, POD_ONLY, classify, repo_index, sha256_bytes,
)


def _index(files: dict[str, bytes]) -> dict:
    """Build an index directly, mirroring repo_index's output shape."""
    by_hash, by_name = {}, {}
    for path, data in files.items():
        digest = sha256_bytes(data)
        by_hash.setdefault(digest, []).append(path)
        by_name.setdefault(path.rsplit("/", 1)[-1], set()).add(digest)
    return {"by_hash": by_hash, "by_name": by_name}


def test_identical_content_is_in_git():
    idx = _index({"stack/scripts/a.py": b"print(1)\n"})
    got = classify([(sha256_bytes(b"print(1)\n"), "/root/a.py")], idx)
    assert got[0]["verdict"] == IN_GIT


def test_same_name_different_content_is_drifted():
    """The pod is running something we cannot rebuild from HEAD."""
    idx = _index({"stack/scripts/refb_train.py": b"v1\n"})
    got = classify([(sha256_bytes(b"v2-modified\n"), "/root/refb_train.py")], idx)
    assert got[0]["verdict"] == DRIFTED


def test_unknown_file_is_pod_only():
    """This is the REF-B v2 / TanitEval case — one disk, no copy anywhere."""
    idx = _index({"stack/scripts/a.py": b"x\n"})
    got = classify([(sha256_bytes(b"secret sauce\n"), "/root/refb_v4.py")], idx)
    assert got[0]["verdict"] == POD_ONLY
    assert "no file of this name" in got[0]["note"]


def test_relocated_file_still_counts_as_in_git():
    """Content match wins over path — a rescued file lands at a new repo path."""
    idx = _index({"stack/experiments/refb-v2/refb_v4.py": b"arch\n"})
    got = classify([(sha256_bytes(b"arch\n"), "/root/refb_v4.py")], idx)
    assert got[0]["verdict"] == IN_GIT


def test_mixed_batch_classifies_each_independently():
    idx = _index({"s/a.py": b"A\n", "s/b.sh": b"B\n"})
    got = classify(
        [
            (sha256_bytes(b"A\n"), "/root/a.py"),
            (sha256_bytes(b"B-changed\n"), "/root/b.sh"),
            (sha256_bytes(b"C\n"), "/root/c.py"),
        ],
        idx,
    )
    assert [g["verdict"] for g in got] == [IN_GIT, DRIFTED, POD_ONLY]


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
