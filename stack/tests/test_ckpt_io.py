"""Milestone archiving must be atomic.

The trainers guard archiving with ``if not arch.exists()``. So if an
interrupted copy ever leaves a TRUNCATED file at the final name, that guard
treats the milestone as permanently done and the gate protocol silently loads
a corrupt checkpoint later. These tests pin the property that makes that
impossible.
"""

import shutil
from pathlib import Path

import pytest

from tanitad.train.ckpt_io import atomic_archive


def test_archive_copies_content_and_leaves_no_temp(tmp_path: Path):
    src = tmp_path / "ckpt.pt"
    src.write_bytes(b"weights" * 4096)
    dst = tmp_path / "ckpt_step5000.pt"

    atomic_archive(src, dst)

    assert dst.exists()
    assert dst.read_bytes() == src.read_bytes()
    assert not dst.with_name(dst.name + ".tmp").exists()


def test_interrupted_copy_never_creates_the_final_name(tmp_path, monkeypatch):
    """The actual regression: quota-exceeded mid-copy (how it bit us on pod2)."""
    src = tmp_path / "ckpt.pt"
    src.write_bytes(b"x" * 4096)
    dst = tmp_path / "ckpt_step5000.pt"

    def partial_then_fail(s, d, *a, **k):
        Path(d).write_bytes(b"x" * 10)          # partial write ...
        raise OSError("Disk quota exceeded")     # ... then die

    monkeypatch.setattr(shutil, "copy2", partial_then_fail)

    with pytest.raises(OSError):
        atomic_archive(src, dst)

    # Both must hold for the milestone to self-heal on the next save:
    assert not dst.exists(), "partial archive appeared under the FINAL name"
    assert not dst.with_name(dst.name + ".tmp").exists(), "multi-GB temp leaked"


def test_archive_is_reattempted_after_failure(tmp_path, monkeypatch):
    """After a failed attempt the trainer's `not arch.exists()` guard is still
    True, so the next save archives successfully."""
    src = tmp_path / "ckpt.pt"
    src.write_bytes(b"payload" * 1024)
    dst = tmp_path / "ckpt_step5000.pt"

    real_copy2 = shutil.copy2
    calls = {"n": 0}

    def fail_once(s, d, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            Path(d).write_bytes(b"partial")
            raise OSError("Disk quota exceeded")
        return real_copy2(s, d, *a, **k)

    monkeypatch.setattr(shutil, "copy2", fail_once)

    with pytest.raises(OSError):
        atomic_archive(src, dst)
    assert not dst.exists()          # guard still lets us retry

    atomic_archive(src, dst)         # the next save
    assert dst.read_bytes() == src.read_bytes()
