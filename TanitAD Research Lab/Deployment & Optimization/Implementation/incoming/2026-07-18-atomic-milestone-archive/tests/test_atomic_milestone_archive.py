"""Witnesses for the non-atomic milestone-archive bug + the atomic fix.

Compliance review #3 (Production & Optimization, 2026-07-18), F-5/6/7 ops-fragility.

The current mainline archives gate milestones with ``shutil.copy2(ckpt, arch)``
guarded by ``not arch.exists()`` (train_flagship4b.py:337, refb_train.py:358,
refa_train_plus.py:540). These tests reproduce the silent-corrupt-milestone
failure that a kill mid-copy causes, and prove the ``ckpt_io.atomic_archive``
fix self-heals.

Run:  pytest test_atomic_milestone_archive.py -q   (torch on CPU; no GPU needed)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ckpt_io import archive_milestones, atomic_archive  # noqa: E402

MILESTONES = (5000, 15000, 20000, 30000)


def _write_valid_ckpt(path: Path, step: int) -> None:
    """A small but real torch checkpoint (loads back cleanly)."""
    torch.save({"model": {"w": torch.arange(16).float()}, "step": step}, path)


# --- the CURRENT mainline behaviour (buggy), inlined verbatim -------------- #
def _current_copy2_archive(ckpt_path: Path, step: int) -> None:
    """Exact logic at train_flagship4b.py:333-337 / refb_train.py:353-358."""
    for m in MILESTONES:
        if step >= m:
            arch = ckpt_path.with_name(f"ckpt_step{m}.pt")
            if not arch.exists():
                shutil.copy2(ckpt_path, arch)


# =========================================================================== #
#  1. WITNESS — the current copy2 path leaves a corrupt milestone that the
#     re-archive guard then adopts forever (the live bug).
# =========================================================================== #
def test_current_copy2_corrupt_milestone_survives_guard(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    _write_valid_ckpt(ckpt, step=5000)
    arch = tmp_path / "ckpt_step5000.pt"

    # Simulate a kill DURING copy2: a truncated milestone file is left behind.
    good = ckpt.read_bytes()
    arch.write_bytes(good[: len(good) // 2])          # half-written -> corrupt
    with pytest.raises(Exception):                    # torch cannot load it
        torch.load(arch, weights_only=True)

    # Next save runs the current guard. It sees arch.exists() -> SKIPS re-archive.
    _write_valid_ckpt(ckpt, step=6000)
    _current_copy2_archive(ckpt, step=6000)

    # BUG: the corrupt milestone still stands and is still unloadable.
    assert arch.exists()
    with pytest.raises(Exception):
        torch.load(arch, weights_only=True)


# =========================================================================== #
#  2. FIX — atomic_archive never leaves a partial file at the FINAL path when
#     the copy is interrupted (only the .partial sidecar).
# =========================================================================== #
def test_atomic_archive_no_partial_at_final_on_crash(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt.pt"
    _write_valid_ckpt(ckpt, step=5000)
    arch = tmp_path / "ckpt_step5000.pt"

    # Make the copy raise mid-way (simulated kill / disk-full / Errno122).
    def boom(src, dst, *a, **k):
        Path(dst).write_bytes(b"\x00\x01\x02")        # a few bytes then die
        raise OSError(122, "Disk quota exceeded")     # the pod2 signature
    monkeypatch.setattr("ckpt_io.shutil.copy2", boom)

    with pytest.raises(OSError):
        atomic_archive(ckpt, arch)

    # The FINAL milestone path never appeared -> guard will re-archive.
    assert not arch.exists()
    assert (tmp_path / "ckpt_step5000.pt.partial").exists()


# =========================================================================== #
#  3. FIX self-heals — after a crash leaves a .partial, the next archive call
#     produces a valid, loadable milestone (no manual cleanup needed).
# =========================================================================== #
def test_atomic_archive_reheals_after_crash(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt.pt"
    _write_valid_ckpt(ckpt, step=5000)
    arch = tmp_path / "ckpt_step5000.pt"

    calls = {"n": 0}
    real_copy2 = shutil.copy2

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:                           # first attempt dies
            Path(dst).write_bytes(b"\x00\x01")
            raise OSError(122, "Disk quota exceeded")
        return real_copy2(src, dst, *a, **k)          # retry succeeds
    monkeypatch.setattr("ckpt_io.shutil.copy2", flaky)

    with pytest.raises(OSError):                       # crash on step 5000 save
        archive_milestones(ckpt, step=5000, milestones=MILESTONES)
    assert not arch.exists()                           # nothing corrupt adopted

    made = archive_milestones(ckpt, step=6000, milestones=MILESTONES)  # next save
    assert arch.name in made
    loaded = torch.load(arch, weights_only=True)       # loads cleanly now
    assert loaded["step"] == 5000
    assert torch.equal(loaded["model"]["w"], torch.arange(16).float())


# =========================================================================== #
#  4. FIX round-trip — normal path archives an identical, loadable milestone,
#     and is idempotent (a second call at a higher step does not re-copy it).
# =========================================================================== #
def test_archive_milestones_roundtrip_and_idempotent(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    _write_valid_ckpt(ckpt, step=5000)
    arch = tmp_path / "ckpt_step5000.pt"

    made1 = archive_milestones(ckpt, step=5000, milestones=MILESTONES)
    assert made1 == ["ckpt_step5000.pt"]
    assert not (tmp_path / "ckpt_step5000.pt.partial").exists()   # no leftover
    loaded = torch.load(arch, weights_only=True)
    assert loaded["step"] == 5000

    mtime = arch.stat().st_mtime_ns
    _write_valid_ckpt(ckpt, step=7000)                            # ckpt advances
    made2 = archive_milestones(ckpt, step=7000, milestones=MILESTONES)
    assert "ckpt_step5000.pt" not in made2                        # not re-copied
    assert arch.stat().st_mtime_ns == mtime                       # untouched
