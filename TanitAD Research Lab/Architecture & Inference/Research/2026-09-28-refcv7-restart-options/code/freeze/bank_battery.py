#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bank a numerics battery into the package: per run `run_spec.json`, `digests.json.gz`, the
trainer's own `metrics.jsonl.gz`, and the log TAIL with the trainer's `[v3]` banner lines dropped
(the full logs stay in the work dir); plus every `cmp_*.json`, the conversion record + log.
Timing-rig checkpoints (`run/ckpt.pt`, 0.6 GB each) are NOT banked -- they are worthless as models.

Usage: bank_battery.py <work dir> <package raw dir>
"""
from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path


def main() -> int:
    w, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("batch.log", "convert.log", "U6_convert_record.json", "conv_argv.json"):
        if (w / f).is_file():
            shutil.copyfile(w / f, dst / f)
    for f in sorted(w.glob("cmp_*.json")):
        shutil.copyfile(f, dst / f.name)
    for d in sorted(p for p in w.iterdir() if p.is_dir() and (p / "digests.json").is_file()):
        o = dst / d.name
        o.mkdir(exist_ok=True)
        shutil.copyfile(d / "run_spec.json", o / "run_spec.json")
        with open(d / "digests.json", "rb") as fi, gzip.open(o / "digests.json.gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        m = d / "run" / "metrics.jsonl"
        if m.is_file():
            with open(m, "rb") as fi, gzip.open(o / "metrics.jsonl.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
        log = w / f"{d.name}.log"
        if log.is_file():
            lines = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if not ln.startswith(("[v3] ", "[parity]", "[v2] "))]
            (o / "log_tail.txt").write_text("\n".join(lines[-60:]) + "\n", encoding="utf-8")
        print("banked", d.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
