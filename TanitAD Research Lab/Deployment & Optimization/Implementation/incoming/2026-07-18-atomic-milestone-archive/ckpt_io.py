"""Atomic checkpoint-archive helper (Production & Optimization review #3, 2026-07-18).

Proposed target: ``stack/tanitad/train/ckpt_io.py`` (new shared module).

Motivation — a LIVE ops-fragility bug on the current mainline. All three pod
trainers archive gate milestones with a **non-atomic** ``shutil.copy2`` straight
to the final path:

  - ``scripts/train_flagship4b.py:337``   (flagship, pod2)
  - ``scripts/refb_train.py:358``         (REF-B, pod1)
  - ``experiments/reset-speed4b/refa_train_plus.py:540`` (REF-A, pod3)

each guarded by ``if step >= m and not arch.exists(): shutil.copy2(ckpt, arch)``.

Failure mode (the documented pod2 self-kill / OOM-kill / Errno122-quota-full
history): a kill *during* ``copy2`` leaves ``ckpt_step{m}.pt`` **truncated but
existing**. On the next save the guard sees ``arch.exists() == True`` -> it
**never re-archives**, so the corrupt milestone silently stands in. The gate
protocol later ``torch.load``s that milestone for D1/D2/D3 -> crash or garbage
metrics. It is the same silent-corrupt class the atomic *resume*-ckpt write
(``tmp.replace(ckpt)``) already guards against — the archive path was missed.

Fix — mirror the resume-write pattern for the archive: copy to a sidecar
``.partial`` then ``os.replace`` (atomic rename on one filesystem). A kill
mid-copy leaves only ``.partial``; the final path never exists half-written, so
the ``not arch.exists()`` re-archive guard stays correct and self-heals next save.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def atomic_archive(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """Copy ``src`` to ``dst`` atomically.

    Copies to ``dst + '.partial'`` first, then ``os.replace`` into place. If the
    process dies during the copy, only the ``.partial`` sidecar exists and
    ``dst`` never appears half-written — so an ``if not dst.exists()`` archive
    guard can never adopt a truncated file. A stale ``.partial`` from a prior
    crash is overwritten by the next copy.
    """
    src, dst = Path(src), Path(dst)
    partial = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, partial)      # crash here -> only `.partial`, dst absent
    os.replace(partial, dst)        # atomic rename on the same filesystem


def archive_milestones(ckpt_path: str | os.PathLike, step: int,
                       milestones=(5000, 15000, 20000, 30000)) -> list[str]:
    """Archive ``ckpt_path`` to ``ckpt_step{m}.pt`` for each reached milestone.

    Drop-in for the inline loop in the trainers. Returns the names newly
    archived this call. Atomic + idempotent: a truncated ``.partial`` left by a
    prior kill does not block re-archiving (the final file is what the guard
    checks, and it only appears via the atomic rename).
    """
    ckpt_path = Path(ckpt_path)
    made: list[str] = []
    for m in milestones:
        if step < m:
            continue
        arch = ckpt_path.with_name(f"ckpt_step{m}.pt")
        if not arch.exists():
            atomic_archive(ckpt_path, arch)
            made.append(arch.name)
    return made
