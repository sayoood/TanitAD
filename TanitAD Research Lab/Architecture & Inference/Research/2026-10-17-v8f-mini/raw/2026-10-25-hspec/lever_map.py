#!/usr/bin/env python3
"""lever_map.py -- DESCRIPTIVE diagnostics run AFTER the PREREG_V8F_HSPEC outcome was fixed (READ.json). NOT a bar, NOT an arm, changes nothing in the outcome table.

Rule Zero: a refutation must leave the next lever behind. Two zero-training maps from the SAME banked arrays and the SAME machinery:
  D1  T0 DIAGNOSTIC (eval-oracle): v fitted ON eval-141 itself (reads GT) with the stage-1 objective and with the stage-2 objective at the TRAIN lambda -> the CEILING of
      'shrink per cosine coefficient'. If even the oracle does not separate from P10 on accel / curvature, the lever is exhausted, not under-fitted.
  D2  lambda map: stage-2 fits on TRAIN windows only with lambda = m x (the PREREG's rule value), m in {0.25, 1, 4, 16, 64, 256}, read on eval-141. Picks nothing: m = 1 is the registered
      arm; any other m was looked at AFTER the HSPEC numbers and could only be adopted by a NEW PREREG with a train-only selection rule.
All rows: HS-variant minus P10 and minus H, paired episode-cluster bootstrap, B = 2000, 141 episodes.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v8f_hspec as S  # noqa: E402

H = S.H
KEYS = ("ade2", "ade6", "fw_lon_accel_mae", "fw_lat_curvature_mae_1pm", "fw_lat_yaw_rate_mae_degps", "fw_lat_heading_mae_deg")


def rows(plan, W, nb):
    pw = H.arm_pw(plan, W["gt"], W["BE"])
    sub = {k: pw[k] for k in KEYS}
    out = {"vs_P10": H.rows_for(sub, {k: W["PW"]["P10"][k] for k in KEYS}, W["eid"], W["BE"], nb),
           "vs_H": H.rows_for(sub, {k: W["PW"]["H"][k] for k in KEYS}, W["eid"], W["BE"], nb),
           "mean": {k: float(np.nanmean(pw[k])) for k in KEYS}}
    return out


def main(nb: int = 2000) -> int:
    t0 = time.time()
    D2 = S.d2_matrix()
    res = {"label": ("DESCRIPTIVE lever map, computed AFTER READ.json fixed the PREREG outcome. D1 reads GT on eval (T0 ceiling); D2 is a lambda sweep looked at after the HSPEC numbers. "
                     "Neither is a bar nor an arm; adopting any of it needs its own PREREG."), "per_seed": {}}
    for seed in S.SEEDS:
        W = S.load_world(seed)
        T = S.train_arrays(seed)
        lam_rule = S.lam_rule(T["c"], T["w"], T["d"], T["B30"], D2)
        r = {"lambda_rule": lam_rule, "D2_lambda_map": {}, "D1_eval_oracle": {}}
        for m in (0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0):
            f = S.fit_v(T["c"], T["w"], T["d"], T["B30"], m * lam_rule, D2)
            plan = S.hs_plan(W["P"], W["c"], W["w"], f["v"], W["B30"])
            r["D2_lambda_map"][f"x{m:g}"] = {"v": f["v"], "v_mean": float(f["v"].mean()), **rows(plan, W, nb)}
            print(f"[lever {seed}] D2 x{m:g}  v mean {f['v'].mean():.3f}  " + "  ".join(
                f"{k.replace('fw_', '')[:14]} {r['D2_lambda_map'][f'x{m:g}']['vs_P10'][k]['mean']:+.4f}" for k in ("ade6", "fw_lon_accel_mae", "fw_lat_curvature_mae_1pm")), flush=True)
        d_ev = W["gt"] - W["P"]
        for nm, lam in (("stage1_objective", 0.0), ("stage2_objective_train_lambda", lam_rule)):
            f = S.fit_v(W["c"], W["w"], d_ev, W["B30"], lam, D2)
            plan = S.hs_plan(W["P"], W["c"], W["w"], f["v"], W["B30"])
            r["D1_eval_oracle"][nm] = {"v": f["v"], "v_mean": float(f["v"].mean()), **rows(plan, W, nb)}
            print(f"[lever {seed}] D1 oracle {nm}  v mean {f['v'].mean():.3f}  " + "  ".join(
                f"{k.replace('fw_', '')[:14]} {r['D1_eval_oracle'][nm]['vs_P10'][k]['mean']:+.4f}[{r['D1_eval_oracle'][nm]['vs_P10'][k]['lo']:+.4f},"
                f"{r['D1_eval_oracle'][nm]['vs_P10'][k]['hi']:+.4f}]" for k in ("fw_lon_accel_mae", "fw_lat_curvature_mae_1pm")), flush=True)
        res["per_seed"][seed] = r
    res["seconds"] = time.time() - t0
    H.jdump(res, f"{HERE}/lever_map.json")
    print(f"[lever] DONE {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
