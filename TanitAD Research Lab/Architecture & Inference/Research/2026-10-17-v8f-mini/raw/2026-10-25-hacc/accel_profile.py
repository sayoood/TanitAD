#!/usr/bin/env python3
"""accel_profile.py -- DESCRIPTIVE (no bar, no arm; for the NEXT pre-registration). Where along the 6 s horizon does H's accel error against
the GT exceed P10's? Per 0.2 s accel cell k = 1..29: mean |a_plan - a_gt| over windows, for H (HBLEND), P10 and B (the model's own plan).

Definition (the family's, as written in test_v8f_oacc.py's oracle): speed_k = |p_k - p_(k-1)| / 0.2 with p_0 = the origin (the ego at NOW),
accel_k = (speed_(k+1) - speed_k) / 0.2.
CONTROL (before any profile is read): the profile's per-window mean must reproduce the banked per-window `fw_lon_accel_mae` of H and P10
(W["PW"]) to 2e-6. If it does not, the definition here is not the family's and nothing below is printed.
AMENDED before any profile number existed: the first run computed in float64 and its control read 5.7e-5 .. 9.3e-5 (no profile was printed);
the same definition in FLOAT32 reproduces the banked values to 9.5e-7 (H) / 4.8e-7 (P10), so the family computes in float32. This script now
does too, and the bar is 2e-6 (float32 epsilon on values ~1), not 1e-6.
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v8f_hspec as S  # noqa: E402

DT = 0.2


def accel(p):
    p = p.astype(np.float32)
    q = np.concatenate([np.zeros_like(p[:, :1]), p], 1)
    v = np.linalg.norm(np.diff(q, axis=1), axis=-1) / np.float32(DT)
    return np.diff(v, axis=1) / np.float32(DT)                                   # [N, 29]


def main() -> int:
    out = {"label": "DESCRIPTIVE per-step accel-error profile (after HSPEC's outcome; no bar)", "per_seed": {}}
    for seed in S.SEEDS:
        W = S.load_world(seed)
        ag = accel(W["gt"])
        prof, ctl = {}, {}
        for nm, arr in (("H", W["H"]), ("P10", W["P_file"]), ("B", W["tau"])):
            e = np.abs(accel(arr) - ag).astype(np.float64)
            prof[nm] = e
            if nm in W["PW"]:
                ref = np.asarray(W["PW"][nm]["fw_lon_accel_mae"], np.float64)
                mine = np.abs(accel(arr) - ag).mean(1).astype(np.float64)
                ok = np.isfinite(ref)
                ctl[nm] = float(np.abs(mine[ok] - ref[ok]).max())
        out["per_seed"][seed] = {"control_maxabs_vs_banked": ctl}
        if max(ctl.values()) > 2e-6:
            print(f"[{seed}] CONTROL FAILED: {ctl} -- the definition is not the family's; no profile read", flush=True)
            continue
        rows = []
        for k in range(29):
            h, p, b = (float(prof[n][:, k].mean()) for n in ("H", "P10", "B"))
            rows.append({"cell": k + 1, "t_s": round((k + 1) * DT, 1), "H": h, "P10": p, "B": b, "H_minus_P10": h - p})
        gap = np.array([r["H_minus_P10"] for r in rows])
        out["per_seed"][seed].update({"rows": rows, "gap_total": float(gap.sum() / 29), "gap_first3_share": float(gap[:3].sum() / gap.sum()),
                                      "gap_first1_share": float(gap[:1].sum() / gap.sum())})
        print(f"[{seed}] control {ctl}  mean gap H-P10 {gap.mean():+.4f}  share in cell 1: {gap[0] / gap.sum():.2f}  cells 1-3: {gap[:3].sum() / gap.sum():.2f}", flush=True)
        for r in rows:
            print(f"  k={r['cell']:2d} t={r['t_s']:.1f}s  H {r['H']:.3f}  P10 {r['P10']:.3f}  B {r['B']:.3f}  H-P10 {r['H_minus_P10']:+.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "accel_profile.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
