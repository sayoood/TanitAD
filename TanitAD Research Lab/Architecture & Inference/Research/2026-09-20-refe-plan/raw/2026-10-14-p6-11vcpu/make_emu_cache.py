#!/usr/bin/env python3
"""Dev-box emulation fixture: M-LAB's cached navtest parts (NAVSIM MetricCache pickles) laid out as a D9 cache (<root>/<log>/<token>.pkl.xz), which
online_labeller.load_parts reads as-is. Writes <root>/tokens.txt = "log:token,..." (all 240 of M-LAB's cached_r1 plan).
    python make_emu_cache.py <root>
"""
import json
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
mlab = Path(__file__).resolve().parents[1] / "2026-10-11-m-lab"
plan = json.load(open(mlab / "plan" / "cached_r1.json", encoding="utf-8"))
toks, size = [], 0
for t, p in plan["paths"].items():
    p = p.replace(chr(92), "/")
    lg = p.split("/")[-4]
    (root / lg).mkdir(parents=True, exist_ok=True)
    out = root / lg / (t + ".pkl.xz")
    if not out.exists():
        shutil.copyfile(p, out)
    size += out.stat().st_size
    toks.append(f"{lg}:{t}")
(root / "tokens.txt").write_text(",".join(toks), encoding="utf-8")
print(len(toks), "parts,", round(size / 1e6, 1), "MB")
