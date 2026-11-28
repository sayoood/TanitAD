"""Staging gate: no clip id, 8-hex prefix or 12-hex tail in the files about to be staged.

A scan that reads 0 proves nothing on its own, so a CONTROL file known to carry legacy
prefixes must read > 0 with the same scanner, or the gate fails.

    python leak_scan.py --ids <ids.txt> [--ids ...] [--universe clip_universe.json]
                        --control <file> --out <report.txt> <file or dir> ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

H8 = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")
H12 = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")


def scan(text: str, ids: set, heads: set, tails: set) -> dict:
    return {"full_ids": sum(1 for i in ids if i in text),
            "prefixes": sum(1 for t in H8.findall(text) if t in heads),
            "tails": sum(1 for t in H12.findall(text) if t in tails)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", action="append", default=[])
    ap.add_argument("--universe")
    ap.add_argument("--control", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    ids = set()
    for f in a.ids:
        ids |= set(Path(f).read_text(encoding="utf-8").split())
    if a.universe:
        ids |= set(json.loads(Path(a.universe).read_text(encoding="utf-8"))["ids"])
    heads, tails = {i[:8] for i in ids}, {i[-12:] for i in ids}
    files = []
    for p in map(Path, a.paths):
        files += sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    lines, total = [], 0
    for f in files:
        c = scan(f.read_bytes().decode("utf-8", errors="replace"), ids, heads, tails)
        n = sum(c.values())
        total += n
        lines.append(f"{n:>3}  {c}  {f.name}")
    ctl = scan(Path(a.control).read_bytes().decode("utf-8", errors="replace"), ids, heads, tails)
    ok = total == 0 and ctl["prefixes"] > 0
    rep = [f"# leak scan over {len(ids)} known clip ids (+ their 8-hex prefixes and 12-hex tails)",
           f"# control {Path(a.control).name}: {ctl}  (must read > 0 prefixes)",
           *lines, f"# files {len(files)} | hits {total} | GATE {'PASS' if ok else 'FAIL'}"]
    Path(a.out).write_text("\n".join(rep) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(rep[:2] + rep[-1:]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
