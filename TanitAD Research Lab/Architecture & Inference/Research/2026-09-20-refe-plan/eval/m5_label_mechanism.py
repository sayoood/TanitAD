#!/usr/bin/env python3
"""Measure 5 -- EXPLORATORY, NOT GATING: what the v4 LABELS say about a 0.75x copy versus ITS OWN source proposal.

Each v4 set carries 8 copies of the dump-time scorer's top-8 (`slow.src`) plus STOP, in 9 replaced non-source slots, so
every copy's source is still in the set with its own labels. Per (copy, source) pair this reads the TRAINER's targets
(ScorerBank.components + the NAVSIM drivable-area override, m5_finetune_eval.trainer_targets) -- NC, DAC, EP, TTC, C,
DDC -- and the source's last-pose heading defect (|yaw[19] - tangent of the last segment|, wrapped; Amendment 7's
D-REFE-LASTYAW-1), to show WHICH component tells the scorer that slowing is better, and whether that component tracks
the heading defect a copy never reaches (a 0.75x copy ends at the source's t = 3.0 s).

    python eval/m5_label_mechanism.py [--sets <dir>] [--out <json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
HEADS = ("NC", "DAC", "EP", "TTC", "C", "DDC")


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def main() -> int:
    import m5_finetune_eval as FE
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=os.path.join(FE.M5, "sets"))
    ap.add_argument("--out", default=os.path.join(FE.OUT, "label_mechanism.json"))
    a = ap.parse_args()
    d_all, defect, n_sets = [], [], 0
    by_key, n_lines = {}, 0
    for p in sorted(glob.glob(os.path.join(a.sets, "onpolicy_r0_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "onpolicy_set":
                n_lines += 1
                by_key[f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r['rank'])}"] = r   # one per key, as data()
    for r in by_key.values():
        if True:
            sl = r.get("slow") or {}
            if not sl.get("carries"):
                continue
            n_sets += 1
            comp = np.array([FE.trainer_targets(t) for t in r["targets"]], np.float64)          # [64, 6]
            traj = np.asarray(r["traj"], np.float64)
            yaw = np.asarray(r["yaw"], np.float64)
            for slot, src, f in zip(sl["slots"], sl["src"], sl["factor"]):
                if src < 0 or f != 0.75:
                    continue
                d_all.append(comp[slot] - comp[src])
                seg = traj[src, -1] - traj[src, -2]
                tan = math.atan2(seg[1], seg[0]) if np.hypot(*seg) >= 0.05 else yaw[src, -2]
                defect.append(abs(wrap(yaw[src, -1] - tan)))
    D = np.asarray(d_all)
    H = np.asarray(defect)
    big = H > 0.5
    res = {"_label": "EXPLORATORY, NOT GATING: v4 label differences, copy minus ITS source (trainer targets)",
           "sets_dir": a.sets, "set_lines": n_lines, "distinct_keys": len(by_key), "n_sets_carrying": n_sets, "n_pairs": int(len(D)),
           "source_last_heading_defect_rad": {"median": float(np.median(H)) if len(H) else None,
                                              "frac_gt_0.5": float(big.mean()) if len(H) else None},
           "mean_diff": {h: float(D[:, i].mean()) for i, h in enumerate(HEADS)} if len(D) else {},
           "frac_copy_higher": {h: float((D[:, i] > 1e-9).mean()) for i, h in enumerate(HEADS)} if len(D) else {},
           "frac_copy_lower": {h: float((D[:, i] < -1e-9).mean()) for i, h in enumerate(HEADS)} if len(D) else {},
           "comfort_diff_by_source_defect": {
               "defect_gt_0.5": float(D[big, 4].mean()) if big.any() else None,
               "defect_le_0.5": float(D[~big, 4].mean()) if (~big).any() else None,
               "n_gt": int(big.sum()), "n_le": int((~big).sum())} if len(D) else {}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
