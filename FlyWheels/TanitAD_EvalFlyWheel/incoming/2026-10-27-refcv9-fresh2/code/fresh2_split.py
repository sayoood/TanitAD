"""fresh2_split.py -- PREREG_FRESH2 s2 (fixed before any label existed): usable FRESH2 clips in candidate order (sha256 ascending);
CONFIRM = the first ceil(2/3 x n_usable); GROW = the rest.  Mechanical; reads nothing but the usable id list.
  writes  F:/l1c/fresh2/split.json                    {"confirm": [clip_id...], "grow": [...]}  (raw ids; F: only)
          <pkg>/select/fresh2_split_sha12.json        sha12 only
NEW file (FRESH2 agent, 2026-10-27).  No clip id is printed.
"""
import hashlib
import json
import math
from pathlib import Path

IDS = Path("F:/l1c/fresh2/fresh2_ids.json")
OUT = Path("F:/l1c/fresh2/split.json")
PKG = Path(__file__).resolve().parent.parent


def main():
    usable = json.load(open(IDS, encoding="utf-8"))
    shas = [hashlib.sha256(r["clip_id"].encode()).hexdigest() for r in usable]
    assert shas == sorted(shas) and len(set(shas)) == len(shas), "usable list is not in candidate (sha256 ascending) order"
    n = len(usable)
    nc = math.ceil(2 * n / 3)
    conf, grow = usable[:nc], usable[nc:]
    if OUT.exists():
        old = json.load(open(OUT, encoding="utf-8"))
        assert old == {"confirm": [r["clip_id"] for r in conf], "grow": [r["clip_id"] for r in grow]}, "split.json exists and differs: the split is fixed once"
    json.dump({"confirm": [r["clip_id"] for r in conf], "grow": [r["clip_id"] for r in grow]}, open(OUT, "w", encoding="utf-8"))
    pub = {"prereg": "PREREG_FRESH2.md s2", "rule": "usable clips in candidate order (sha256(clip_id) ascending); CONFIRM = first ceil(2/3 x n_usable); GROW = rest",
           "n_usable": n, "n_confirm": len(conf), "n_grow": len(grow),
           "confirm_sha12": [s[:12] for s in shas[:nc]], "grow_sha12": [s[:12] for s in shas[nc:]],
           "boundary": {"last_confirm_sha12": shas[nc - 1][:12], "first_grow_sha12": shas[nc][:12] if grow else None}}
    (PKG / "select").mkdir(exist_ok=True)
    (PKG / "select" / "fresh2_split_sha12.json").write_text(json.dumps(pub, indent=1), encoding="utf-8")
    print(f"ZZF2SPLITZZ usable {n} confirm {len(conf)} grow {len(grow)}", flush=True)


if __name__ == "__main__":
    main()
