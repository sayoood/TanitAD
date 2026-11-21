"""detres_event_finder.py -- CPU-only: WHICH greedy-match events carry the 2 m AP step of `eval_box3d_det_ap2_automobile_0_20`.

On the banked packs (g0.packs.npz, arms s0 / fp32_s0) the AP-vs-threshold curve of the automobile 0-20 m cell is flat 1.95 -> 2.0 (+0.0044) and jumps 2.0 -> 2.05
(+0.0197) with the TP count unchanged: a few detections sit just beyond 2.0 m of their GT. This script finds them: per window it re-runs the repo's own
`greedy_rows` at 2.0 and 2.0 + dt (dt = 0.05 and 0.10), lists the slots whose kind (TP / FP / DontCare) flips, and measures each window's own AP effect
(that ONE window judged at 2.0 + dt, every other window at 2.0).  Evidence class: MEASURED on banked packs (descriptive; no model, no GPU).

    python detres_event_finder.py --packs E:/refcv8_eval_kit/battery/step5000/g0.packs.npz --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import detres_pack_analysis as PA  # noqa: E402  (same banked-pack loader / term(); sets torch threads to 2)
from tanitad.eval import detection_metrics as dm  # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES  # noqa: E402

KIND = {1: "TP", 0: "FP", -1: "DontCare"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dts", default="0.05,0.10")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ci = list(AGENT_CLASSES).index(PA.CLS)
    out = {"tool": "detres_event_finder.py", "evidence": "MEASURED on banked packs (descriptive)", "arms": {}}
    for arm in ("s0", "fp32_s0"):
        packs = PA.load_arm(a.packs, arm)
        if a.smoke:
            packs = packs[:4]
        base = PA.term(packs, 2.0)
        res = {"base_ap5": base["ap5"], "n_gt": base["n_gt"], "windows": len(packs), "dt": {}}
        for dt in [float(x) for x in a.dts.split(",")]:
            events = []
            for w, pk in enumerate(packs):
                r0 = {r[3]: r for r in dm.greedy_rows(pk, 2.0, ci)}
                r1 = {r[3]: r for r in dm.greedy_rows(pk, 2.0 + dt, ci)}
                flips = []
                for s in sorted(set(r0) | set(r1)):
                    k0, k1 = (r0[s][1] if s in r0 else None), (r1[s][1] if s in r1 else None)
                    if k0 != k1:
                        row = r1.get(s) or r0.get(s)
                        g = row[4]
                        d = None
                        if g is not None and g >= 0:
                            d = float(np.linalg.norm(pk["xy"][s] - pk["gt_xy"][g]))
                        flips.append({"slot": int(s), "kind_2.0": KIND.get(k0), f"kind_{2.0 + dt:.2f}": KIND.get(k1),
                                      "conf": float(row[0]), "band": row[2], "dist_to_matched_gt_m": d,
                                      "slot_cx_m": float(pk["xy"][s, 0]), "gt_cx_m": (float(pk["gt_xy"][g, 0]) if g is not None and g >= 0 else None)})
                if not flips:
                    continue
                alt = list(packs)
                # judge ONLY this window at 2.0 + dt: the pooled AP is rebuilt from rows, so swap this window's rows
                rows = []
                for j, pj in enumerate(packs):
                    rows += dm.greedy_rows(pj, 2.0 + dt if j == w else 2.0, ci)
                rb = [r for r in rows if r[2] == PA.BAND]
                ap_w = dm._ap(rb, base["n_gt"])
                events.append({"window": w, "delta_ap_this_window_only": round(float(ap_w - base["ap"]), 5), "flips": flips})
            events.sort(key=lambda e: -abs(e["delta_ap_this_window_only"]))
            res["dt"][f"{dt:.2f}"] = {"n_windows_with_flips": len(events), "sum_single_window_deltas": round(sum(e["delta_ap_this_window_only"] for e in events), 5),
                                      "top": events[:6]}
        out["arms"][arm] = res
        print("DETRES_EVENTS_ARM", arm, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print("DETRES_EVENTS_DONE")


if __name__ == "__main__":
    main()
