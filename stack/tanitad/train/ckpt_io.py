"""Checkpoint I/O helpers — atomic milestone archiving.

WHY THIS EXISTS. Milestone archiving used a bare ``shutil.copy2(src, dst)``,
which writes straight to the FINAL name. If the process dies mid-copy (OOM
kill, a full quota, an operator kill) the archive is left TRUNCATED at its
real name. The archiving guard is ``if not arch.exists()``, so from then on
the corrupt file looks archived and is never rewritten — and the gate
protocol silently loads a corrupt milestone hours or days later.

The fix is the same trick the main checkpoint save already uses: copy to a
temp sibling, then rename. ``Path.replace`` is atomic within a filesystem, so
the final name only ever appears complete. An interrupted copy leaves a stray
``*.pt.tmp`` that the ``exists()`` guard ignores, so the next save re-archives
correctly and the milestone self-heals.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def atomic_archive(src: str | Path, dst: str | Path) -> Path:
    """Copy ``src`` to ``dst`` so ``dst`` is never observable half-written.

    Returns the destination path. On failure the partial temp file is removed
    before re-raising — these checkpoints are multi-GB and the training volumes
    are quota-constrained, so a leaked partial is not harmless.

    A hard kill (SIGKILL) cannot run cleanup, but that case is safe by design:
    only the ``.tmp`` sibling exists, the real name does not, and the guard
    re-archives on the next save.
    """
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return dst
