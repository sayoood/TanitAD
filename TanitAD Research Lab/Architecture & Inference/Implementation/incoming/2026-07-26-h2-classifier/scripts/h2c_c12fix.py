"""H2 classifier — the CORRECTED C12 conjunct diagnostic (dev box, CPU).

**Why this file exists — a defect in my own instrument, found by reading the result.**
`T_seen` (`a_req_seen < tau*`) fires on **96.7 %** of frames. The pre-registered training recipe is
BCE with `pos_weight` — a RARE-POSITIVE recipe — so on `T_seen` it up-weights the MAJORITY class,
and the head it produces cannot say anything about the rare, informative side. Reading that head's
complement would have been an over-read, so it is not read.

The same question posed correctly is **`NOT_T_seen`** — *"an agent the encoder CAN see requires
braking >= tau*"* — as a rare-positive target. It is **~3.3 %** of frames, i.e. **~5x better
powered than the composite**, and it is the cleanest available test of whether the FROZEN v1
representation exposes "there is something ahead I must brake for" at all. Same head, same CV, same
split, same estimator; diagnostic only — **not an arm in the primary comparison.**

usage:  python h2c_c12fix.py --run <run_c12fix dir> --out <artifacts dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h2c_stats import average_precision, boot_stat, roc_auc  # noqa: E402
from h2c_eval import _paired_ap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    HO = np.load(os.path.join(args.run, "scores_heldout.npz"))
    eid = HO["clip"]
    y = 1.0 - HO["EX"][:, 1].astype(float)          # NOT_T_seen at frame level
    res = {"target": "NOT_T_seen = (a_req_seen >= tau*) — an agent INSIDE the encoder crop "
                     "requires braking >= 0.5 m/s^2",
           "why": "T_seen is a 96.7 %-positive target and the pre-registered BCE+pos_weight recipe "
                  "up-weights its MAJORITY class; NOT_T_seen poses the same question as a "
                  "rare-positive target and is ~5x better powered than the composite",
           "n_frames": int(y.size), "n_positives": int(y.sum()), "base_rate": float(y.mean()),
           "n_positive_clips": int(len(np.unique(eid[y > 0]))),
           "n_clips": int(len(np.unique(eid))), "arms": {}}
    chance = np.zeros_like(y)
    for k in HO.files:
        if not (k.startswith("s__") and k.endswith("__NOT_T_seen")):
            continue
        arm = k[3:].replace("__NOT_T_seen", "")
        s = HO[k][:, 0]
        res["arms"][arm] = {
            "AP": boot_stat(lambda y, s, **_: average_precision(y, s), eid,
                            n_boot=args.boot, y=y, s=s),
            "AP_over_base": float(average_precision(y, s) / max(y.mean(), 1e-12)),
            "AUROC": float(roc_auc(y, s)),
            "paired_AP_vs_chance": _paired_ap(y, s, chance, eid, args.boot),
        }
    json.dump(res, open(os.path.join(args.out, "c12_fix.json"), "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
