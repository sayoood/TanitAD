#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""restart item 4 -- `V2CompressedCache.__getstate__` drops `newest_frame_only`.

THE DEFECT (step-cost package sec. 7, re-read from source 2026-09-28). `__init__` sets
`self.newest_frame_only`; `__getstate__` returns cache_dir / lru_size / files / frame / allow_lossy
and NOT newest_frame_only; `__setstate__` restores those five. `decode_stacked_range` reads
`self.newest_frame_only` on every decode. So a cache that crosses a PICKLE boundary -- a DataLoader
worker started by SPAWN (Windows, macOS, Python >= 3.14 on Linux, or any explicit
`multiprocessing_context="spawn"/"forkserver"`) -- has no such attribute and the first decode dies
with `AttributeError`, whatever the flag's value was (the attribute is missing, not just False).

WHY THE LIVE RUN IS UNAFFECTED (MEASURED 2026-09-28, read-only on Thor):
* the trainer's train DataLoader (`refc_v3_train.py`, `dl = torch.utils.data.DataLoader(...
  num_workers=args.workers ...)`) passes NO `multiprocessing_context` and nothing in the trainer
  calls `set_start_method` (grep: 0 hits); Thor runs CPython 3.12.3 on Linux, whose default start
  method is fork;
* the six live `pt_data_worker` processes carry the TRAINER'S OWN argv in /proc/<pid>/cmdline --
  a forked child inherits its parent's argv, a spawned one would read `-c "from multiprocessing
  .spawn import spawn_main ..."`. Fork shares the dataset by memory; nothing is pickled.

⛔ CLOSURE-TOUCHING: `stack/tanitad/data/v2_dataset.py` (blob ee61ea938f2a) is in BOTH binding
closures. PACKAGE-ONLY; apply at a restart that already re-runs the bindings.

Usage: python apply_getstate_fix.py <tree-root> [--check] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REL = "stack/tanitad/data/v2_dataset.py"
BASE = "ee61ea938f2a7ec249513e28d0029fe8ab2d6d5d"
RESULT = "f20aca4c55b7ccc34dc37a6ccb7f612471bc44a5"   # verified 2026-09-28 (15 passed, raw/getstate)

EDITS = [
    ('                "frame": None if self.frame is None else self.frame.to_dict(),\n'
     '                "allow_lossy": self.allow_lossy}\n',
     '                "frame": None if self.frame is None else self.frame.to_dict(),\n'
     '                "allow_lossy": self.allow_lossy,\n'
     '                # ⛔ 2026-09-28: every __init__ attribute but the LRU crosses the pickle\n'
     '                # boundary -- a SPAWN worker without it died on its first decode\n'
     '                "newest_frame_only": self.newest_frame_only}\n'),
    ('        self.allow_lossy = bool(s.get("allow_lossy", False))\n'
     '        self._lru = None\n',
     '        self.allow_lossy = bool(s.get("allow_lossy", False))\n'
     '        # an OLD pickle (no key) predates the switch: False is its behaviour\n'
     '        self.newest_frame_only = bool(s.get("newest_frame_only", False))\n'
     '        self._lru = None\n'),
]


def git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    p = Path(a.tree) / REL
    raw = p.read_bytes()
    got = git_blob(raw)
    if RESULT and got == RESULT:
        print(f"[getstate] {REL}: ALREADY PATCHED ({got})")
        return 0
    if got != BASE:
        raise SystemExit(f"[getstate] {REL}: base blob {got} != the launch blob {BASE} -- REFUSED")
    txt = raw.decode("utf-8")
    for old, new in EDITS:
        if txt.count(old) != 1:
            raise SystemExit(f"[getstate] anchor count {txt.count(old)} != 1: {old[:80]!r}")
        txt = txt.replace(old, new)
    new_b = txt.encode("utf-8")
    nb = git_blob(new_b)
    if RESULT and nb != RESULT:
        raise SystemExit(f"[getstate] result blob {nb} != the verified {RESULT} -- REFUSED")
    print(f"[getstate] {REL}: {BASE[:12]} -> {nb}")
    if not a.check:
        dst = (Path(a.out_dir) / REL) if a.out_dir else p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(new_b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
