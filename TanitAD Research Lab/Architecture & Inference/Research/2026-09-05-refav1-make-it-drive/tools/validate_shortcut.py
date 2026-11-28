#!/usr/bin/env python
"""Is the REAL planner's output exactly ``canonical_controls(decoded token)``?

⭐ WHY THIS MATTERS MORE THAN IT LOOKS. `assert_plan_gate.py` measured, through
the real `plan()` on a TINY random-init model, that the returned controls are
exactly the decoded token's canonical profile. If that also holds on the
**trained 21,109-step checkpoint** over **real banked windows**, then the
planner is a deterministic LOOKUP on the goal head's argmax, and the four
metric families + ADE for ANY decision rule can be computed on CPU in seconds
without ever running iCEM again.

That is a large claim, so it is CHECKED, not assumed — against
``cl_controls``, the actual controls `refav1_arm.run_dump` banked from
`res.controls`.

⛔ CONTROLS:
  * a POSITIVE control that must MISMATCH: comparing `cl_controls` against the
    canonical profile of a DIFFERENT (shifted) token must disagree on most
    windows. Without it, an all-zeros bug on both sides would read as a match.
  * the match rate is reported SEPARATELY for windows whose decoded token is
    zero-curvature (LANE_KEEP / ABORT_LC) and for turn tokens: a shortcut that
    only works where everything is zero is not a shortcut.
  * per-window max-abs-diff percentiles, so "match" is a number, not a bool.

Run: python validate_shortcut.py --dump <dir> --out shortcut.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="banked refav1 dump dir")
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)

    import torch
    from tanitad.refs.refa_v1 import canonical_controls, RefAV1Config
    from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v

    cfg = RefAV1Config()
    LAT = list(tactical_lat_actions("v7.0"))
    LON = list(tactical_lon_actions_v("v7.0"))
    OP_STEPS, OP_DT = cfg.op_steps, cfg.op_dt

    CL, GLAT, GLON, V0, WS, CIX = [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.dump, "ep*.npz"))):
        dec_f = os.path.join(a.dump, "decisions", os.path.basename(f))
        if not os.path.exists(dec_f):
            continue
        with np.load(f) as z, np.load(dec_f) as d:
            CL.append(d["cl_controls"]); GLAT.append(d["goal_lat_cl"])
            GLON.append(d["goal_lon_cl"]); V0.append(z["v0"])
            WS.append(z["ws"])
            CIX.append(np.repeat(np.asarray(z["clip_index"]).ravel()[0],
                                 len(z["ws"])))
    if not CL:
        print("NO WINDOWS READ — inconclusive, not a pass")
        return 2
    CL = np.concatenate(CL); GLAT = np.concatenate(GLAT)
    GLON = np.concatenate(GLON); V0 = np.concatenate(V0)
    WS = np.concatenate(WS); CIX = np.concatenate(CIX)
    n, K = CL.shape[0], CL.shape[1]
    print(f"[read] {n} windows, controls [{K}, {CL.shape[2]}] from {a.dump}")

    def canon(lat_i, lon_i, v0):
        return canonical_controls(LAT[lat_i], LON[lon_i], float(v0),
                                  OP_STEPS, OP_DT)[:K].numpy()

    # ---- the comparison -------------------------------------------------- #
    # ⭐ PER CHANNEL. Channel 0 is ACCEL, channel 1 is CURVATURE, and they are
    # planned against different parts of the cost: a `decel_1.5` baseline can
    # win the LONGITUDINAL channel (its magnitude 1.5 is exactly the diff_max
    # first observed) while the LATERAL channel is still the token's canonical
    # profile. Pooling the two hides precisely the distinction the gate is
    # about, so the pooled number is reported but never used for the verdict.
    CAN = np.stack([canon(GLAT[i], GLON[i], V0[i]) for i in range(n)])
    diff = np.abs(CL - CAN).max(axis=(1, 2))
    diff_acc = np.abs(CL[:, :, 0] - CAN[:, :, 0]).max(axis=1)
    diff_kap = np.abs(CL[:, :, 1] - CAN[:, :, 1]).max(axis=1)
    # POSITIVE CONTROL that must MISMATCH: shift the lat token by 1 (mod 8).
    # If this also "matches", the comparison is not discriminating anything.
    diff_ctl = np.array([np.abs(CL[i] - canon((GLAT[i] + 1) % len(LAT),
                                              GLON[i], V0[i])).max()
                         for i in range(n)])

    ZERO_TOK = {LAT.index("LANE_KEEP"), LAT.index("ABORT_LC")}
    is_zero = np.array([int(g) in ZERO_TOK for g in GLAT])
    match = diff <= a.tol

    match_kap = diff_kap <= a.tol
    match_acc = diff_acc <= a.tol

    def blk(m):
        if m.sum() == 0:
            return {"n": 0, "match_rate": None}
        return {"n": int(m.sum()),
                "match_rate_POOLED": float(match[m].mean()),
                "match_rate_CURVATURE": float(match_kap[m].mean()),
                "match_rate_ACCEL": float(match_acc[m].mean()),
                "kappa_diff_p95": float(np.percentile(diff_kap[m], 95)),
                "kappa_diff_max": float(diff_kap[m].max()),
                "accel_diff_p95": float(np.percentile(diff_acc[m], 95)),
                "accel_diff_max": float(diff_acc[m].max())}

    R = {
        "dump": a.dump, "n_windows": n, "K": int(K), "tol": a.tol,
        "lat_vocab": LAT,
        "overall": blk(np.ones(n, bool)),
        "zero_curvature_tokens": blk(is_zero),
        "curvature_carrying_tokens": blk(~is_zero),
        "decoded_lat_hist": {LAT[i]: int((GLAT == i).sum())
                             for i in range(len(LAT)) if (GLAT == i).sum()},
        "CONTROL_shifted_token": {
            "match_rate": float((diff_ctl <= a.tol).mean()),
            "diff_p50": float(np.percentile(diff_ctl, 50)),
            "note": "must be LOW; a high value means the test discriminates "
                    "nothing (e.g. both sides all-zero)"},
    }
    # ⭐ THE VERDICT IS ABOUT THE LATERAL CHANNEL. That is the channel the
    # curvature gate governs and the channel a turn lives in. The longitudinal
    # channel is reported beside it and is expected to DIFFER — `decel_1.5` is
    # an injected baseline that can win the accel column on modelled cost.
    ok_kap = R["overall"]["match_rate_CURVATURE"] == 1.0
    ok_turn = (R["curvature_carrying_tokens"]["match_rate_CURVATURE"] == 1.0
               if R["curvature_carrying_tokens"]["n"] else False)
    ok_ctl = R["CONTROL_shifted_token"]["match_rate"] < 0.5
    R["PASS"] = bool(ok_kap and ok_turn and ok_ctl)
    R["verdict"] = {
        "LATERAL_is_a_lookup_on_the_decoded_token": bool(ok_kap),
        "holds_on_turn_windows_too": bool(ok_turn),
        "control_discriminates": bool(ok_ctl),
        "LONGITUDINAL_also_a_lookup": bool(
            R["overall"]["match_rate_ACCEL"] == 1.0)}

    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "lat_vocab"}, indent=1))
    print("\nOVERALL:", "PASS" if R["PASS"] else "FAIL")
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
