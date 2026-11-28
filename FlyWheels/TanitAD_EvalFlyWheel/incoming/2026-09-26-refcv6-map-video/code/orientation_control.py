"""Orientation control for render_refcv6_map_video: the ego's OWN future path must lie on SAM3-drivable road.

A control that must read a known value, derived independently of the renderer's drawing code:

* the GT future slots come from ``per_frame.jsonl`` (``refb_labels.waypoint_targets``: +x forward,
  +y LEFT, the trainer's label function);
* the SAM3 drivable fraction comes straight from the GT file (``cart_frac`` uint8/255, read here with
  numpy by its sha12 file name, identity checked against ``meta.source.clip_sha12``), at the window's
  own ``map_raw_frame``;
* the cell of a point is computed from the FILE CONTRACT (``semantic_map_gt.py:16-17``): row
  ``floor(x / 0.5)``, col ``floor((y + 16) / 0.5)`` with col 0 on the RIGHT.

The same points are then scored under three mappings: the contract, a LATERAL MIRROR (col -> 63 - col)
and a TIME SHIFT of +-2 s on the label frame. Only points with x >= 4 m (ahead of the hood) on cells
SAM3 saw are counted. A correct mapping puts the ego's own path on drivable road far more often
than the mirror does on windows whose path actually moves laterally. That is the check this makes.

Writes ``raw/orientation_control.json`` (sha12 only).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
RAW = PKG / "raw"
GT_DIR = Path("D:/refcv6_eval_kit/data/sam3_gt_eval_thor137")
DRV, NOT_SEEN, FS = 1, 8, 255


def main():
    rows = [json.loads(ln) for ln in open(RAW / "per_frame.jsonl", encoding="utf-8") if ln.strip()]
    out = {"what": __doc__.split("\n")[0], "rule": "point on drivable <=> cart_frac[drivable] >= 128/255 "
           "(the IoU rule's >= 0.5); seen <=> 2*(255 - not_seen) >= 255", "clips": {}}
    for s12 in sorted({r["clip_sha12"] for r in rows}):
        f = GT_DIR / f"{s12}.sam3mapgt.npz"
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            cart = z["cart_frac"]
        stored = (meta.get("source") or {}).get("clip_sha12")
        if stored != s12:
            raise SystemExit(f"{f.name} stores sha12 {stored!r} -- not this clip's file")
        T = cart.shape[0]
        rs = [r for r in rows if r["clip_sha12"] == s12]
        acc = {k: [0, 0] for k in ("contract", "mirror", "shift-20", "shift+20")}
        lat_acc = {k: [0, 0] for k in ("contract", "mirror")}
        per_win = []
        for r in rs:
            fr0 = int(r["map_raw_frame"])
            pts = [p for p, ok in zip(r["gt_slots_m"], r["gt_slot_valid"]) if ok and p[0] >= 4.0]
            lateral = max((abs(p[1]) for p in pts), default=0.0) >= 1.0
            win = {}
            for name, fr, mirror in (("contract", fr0, False), ("mirror", fr0, True),
                                     ("shift-20", fr0 - 20, False), ("shift+20", fr0 + 20, False)):
                if not 0 <= fr < T:
                    continue
                hit = n = 0
                for x, y in pts:
                    i = int(math.floor(x / 0.5))
                    j = int(math.floor((y + 16.0) / 0.5))
                    if not (0 <= i < 120 and 0 <= j < 64):
                        continue
                    if mirror:
                        j = 63 - j
                    ns = int(cart[fr, NOT_SEEN, i, j])
                    if 2 * (FS - ns) < FS:                       # not seen by SAM3 -> not counted
                        continue
                    n += 1
                    hit += int(cart[fr, DRV, i, j]) >= 128
                acc[name][0] += hit
                acc[name][1] += n
                if lateral and name in lat_acc:
                    lat_acc[name][0] += hit
                    lat_acc[name][1] += n
                win[name] = (hit, n)
            per_win.append({"win_rank": r["win_rank"], "lateral_path": lateral, **{
                k: v for k, v in win.items()}})
        out["clips"][s12] = {
            "nav": rs[0]["nav"], "n_windows": len(rs),
            "n_windows_lateral_ge_1m": sum(1 for w in per_win if w["lateral_path"]),
            "path_points_on_drivable": {k: {"hit": h, "n": n, "frac": (h / n if n else None)}
                                        for k, (h, n) in acc.items()},
            "lateral_windows_only": {k: {"hit": h, "n": n, "frac": (h / n if n else None)}
                                     for k, (h, n) in lat_acc.items()},
        }
    p = RAW / "orientation_control.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for s12, c in out["clips"].items():
        pd = c["path_points_on_drivable"]
        lw = c["lateral_windows_only"]
        print(f"{s12} {c['nav']:6s} contract {pd['contract']['frac']:.4f} (n={pd['contract']['n']})  "
              f"mirror {pd['mirror']['frac']:.4f}  shift-2s {pd['shift-20']['frac']}  shift+2s "
              f"{pd['shift+20']['frac']}  | lateral windows {c['n_windows_lateral_ge_1m']}: contract "
              f"{lw['contract']['frac']} mirror {lw['mirror']['frac']}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
