#!/usr/bin/env python3
"""P4b - the VALIDATION panel, banked so the val-vs-test optimism is quotable from an artifact.

Val AP (rule B) for `const`, `prior` (per-cell TRAIN marginal) and every trained arm's saved
best-checkpoint val predictions, next to the same arms' TEST AP from the scored panel.
Writes `raw/p4_val_reference.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p4_bev_head as H  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="main,main_s1,shuffled,pixel,s16")
    ap.add_argument("--panel", default=str(HERE.parent / "raw" / "p4_panel_levers.json"))
    args = ap.parse_args()
    from sklearn.metrics import average_precision_score as aps
    D = H.load_panel_data()
    runs = H.WORK / "runs"
    arms = [a for a in args.arms.split(",") if a]
    r0 = np.load(runs / arms[0] / "rows.npz")
    tr, va = r0["train"], r0["val"]
    occ, mB = D["occ"], D["mB"]
    y = occ[va][mB[va]]
    prior = ((occ[tr] & mB[tr]).sum(0) / np.maximum(mB[tr].sum(0), 1)).astype(np.float32)
    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    out = {"schema": "tanitad.bevhead_val_reference/1", "evidence_class": "MEASURED (ours)",
           "val_clips": int(len(np.unique(D["clip_of_row"][va]))), "val_rows": int(len(va)),
           "val_scored_cells": int(mB[va].sum()), "val_prevalence": float(y.mean()), "arms": {}}
    out["arms"]["const"] = {"val_ap_B": float(y.mean()), "test_ap_B": panel["arms"]["const"]["ap_all_B"]}
    s = np.broadcast_to(prior, occ[va].shape)[mB[va]]
    out["arms"]["prior"] = {"val_ap_B": float(aps(y, s)), "test_ap_B": panel["arms"]["prior"]["ap_all_B"]}
    for a in arms:
        r = np.load(runs / a / "rows.npz")
        assert np.array_equal(r["val"], va)
        pv = np.load(runs / a / "val_probs.npy").astype(np.float32)
        out["arms"][a] = {"val_ap_B": float(aps(y, pv[mB[va]])),
                          "test_ap_B": panel["arms"].get(a, {}).get("ap_all_B")}
    pv, pt = out["arms"]["prior"]["val_ap_B"], out["arms"]["prior"]["test_ap_B"]
    for a in arms:
        v = out["arms"][a]
        if v["test_ap_B"] is not None:
            v["val_margin_over_prior"] = v["val_ap_B"] - pv
            v["test_margin_over_prior"] = v["test_ap_B"] - pt
            v["val_over_test_margin_ratio"] = (v["val_margin_over_prior"] / v["test_margin_over_prior"]
                                               if abs(v["test_margin_over_prior"]) > 1e-9 else None)
    (HERE.parent / "raw" / "p4_val_reference.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
