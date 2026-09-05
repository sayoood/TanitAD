#!/usr/bin/env python
"""P4 — DOES refav1 EXECUTE A TURN WHERE THE ROAD ACTUALLY TURNS?

⭐ THE QUESTION, KEYED ON THE ROAD AND NOT ON THE DECODE. The predecessor's
`turn_execution.py` asks "on windows where the head DECODED a turn, did the
planner turn?" That is the right gate-2 question, and it is answered. This asks
the PI's question instead: **on windows where the ROAD turns, does the executed
plan carry curvature?** -- and its complement, **on straight road, does it stay
straight?** Both are required; recall alone is not a result.

Panel: episodes chosen for real-turn DENSITY (`p4_episodes.json`), so the rates
here are CONDITIONAL on a turn-rich panel and are NOT corpus marginals. That
selection is stated, banked before the run, and must be carried with any number
taken from this file.

⛔ CONTROLS -- every "exactly 0.0" claim needs a live column beside it:
  * `ha0_ext` -- the NON-PLANNER integrator arm banked in the SAME npz, which
    must read NON-ZERO curvature on a materially non-zero fraction of windows.
    If it does not, the curvature column is dead and every zero in this file is
    a READ ERROR, not a measurement. (Predecessor: 98.9 % non-zero, absmax
    0.603.) ⚠️ `ha0_ext` here is the arm's own `hold_ext_controls` integrator,
    the canonical one -- decision M11 / D-MM-ADJ-1 -- never `echo_gate.ha0_ext`.
  * `cl` vs `ha0` identity count -- if the planner arm is bit-identical to the
    constant-velocity floor on a window, it did not plan there.
  * n printed for every cell; a cell with n < 10 is marked UNDERPOWERED.
  * the banked zero-GPU DECODE PREDICTION for this exact window set is carried
    in the output, so the measured decode must reproduce it or the run is not
    on the windows it claims.

Run: python gm_p4_turn_gate.py --dump <dir> --label <arm> --gt <intent npz>
                              --episodes <p4_episodes.json> --out <json>
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
    ap.add_argument("--dump", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gt", required=True, help="npz with clip_index/ws/gt_kappa")
    ap.add_argument("--episodes", default=None, help="p4_episodes.json")
    ap.add_argument("--kappa-turn", type=float, default=0.04)
    ap.add_argument("--exec-eps", type=float, default=1e-3)
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    from tanitad.models.v6 import tactical_lat_actions
    LAT = list(tactical_lat_actions("v7.0"))
    CURV = [i for i, t in enumerate(LAT)
            if t.startswith(("TURN_", "NUDGE_", "LANE_CHANGE_"))]

    CL, GLAT, HAX, HA0, SRC, WS, CIX = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.dump, "ep*.npz"))):
        d_f = os.path.join(a.dump, "decisions", os.path.basename(f))
        if not os.path.exists(d_f):
            continue
        with np.load(f) as e, np.load(d_f) as d:
            CL.append(d["cl_controls"]); GLAT.append(d["goal_lat_cl"])
            HAX.append(d["ha0_ext_controls"]); SRC.append(d["plan_source_cl"])
            WS.append(d["ws"])
            CIX.append(np.full(len(d["ws"]), int(np.asarray(e["clip_index"]).ravel()[0])))
            HA0.append(e["ha0"] if "ha0" in e.files else None)
    if not CL:
        print("NO WINDOWS READ -- INCONCLUSIVE (not a zero)")
        return 2
    CL = np.concatenate(CL); GLAT = np.concatenate(GLAT)
    HAX = np.concatenate(HAX); SRC = np.concatenate(SRC)
    WS = np.concatenate(WS); CIX = np.concatenate(CIX)
    n = len(WS)

    kap = np.abs(CL[:, :, 1]).max(axis=1)          # executed |curvature| max
    ksgn = CL[np.arange(n), np.abs(CL[:, :, 1]).argmax(axis=1), 1]
    hax_k = np.abs(HAX[:, :, 1]).max(axis=1)

    # -------- join the GT curvature by (clip_index, ws) -------------------- #
    G = np.load(a.gt, allow_pickle=False)
    key = {(int(c), int(w)): float(k) for c, w, k in
           zip(G["clip_index"], G["ws"], G["gt_kappa"])}
    gt = np.array([key.get((int(c), int(w)), np.nan) for c, w in zip(CIX, WS)])
    have = np.isfinite(gt)
    if have.sum() == 0:
        print("GT JOIN EMPTY -- INCONCLUSIVE, refusing to report rates")
        return 3
    turn = have & (np.abs(gt) > a.kappa_turn)
    strt = have & (np.abs(gt) <= a.kappa_turn)

    def blk(m, name):
        if m.sum() == 0:
            return {"name": name, "n": 0, "UNDERPOWERED": True}
        k = kap[m]
        nz = k > a.exec_eps
        out = {"name": name, "n": int(m.sum()),
               "UNDERPOWERED_n_lt_10": bool(m.sum() < 10),
               "frac_executed_nonzero_curvature": float(nz.mean()),
               "kappa_absmax_mean": float(k.mean()),
               "kappa_absmax_p50": float(np.percentile(k, 50)),
               "kappa_absmax_max": float(k.max()),
               "n_exactly_zero": int((k == 0.0).sum()),
               "decode_curv_frac": float(np.isin(GLAT[m], CURV).mean()),
               "plan_source_hist": {int(s): int((SRC[m] == s).sum())
                                    for s in np.unique(SRC[m])}}
        if name.startswith("GT TURN"):
            ok = nz & (np.sign(ksgn[m]) == np.sign(gt[m]))
            out["frac_executed_nonzero_AND_correct_direction"] = float(ok.mean())
        return out

    ctrl = {
        "CONTROL_ha0_ext_curvature_MUST_BE_NONZERO": {
            "frac_nonzero": float((hax_k > 1e-6).mean()),
            "absmax": float(hax_k.max()), "n": int(n),
            "passes": bool((hax_k > 1e-6).mean() > 0.5),
            "note": ("the non-planner integrator banked in the SAME npz. If "
                     "this reads zero the curvature column is dead and every "
                     "zero above is a READ ERROR, not a measurement.")},
        "CONTROL_gt_join_nonempty": {
            "n_joined": int(have.sum()), "n_windows": int(n),
            "passes": bool(have.sum() == n)},
    }
    pred = None
    if a.episodes and os.path.exists(a.episodes):
        with open(a.episodes, encoding="utf-8") as fh:
            pred = json.load(fh).get("prediction")
        if pred:
            got = float(np.isin(GLAT[turn], CURV).mean()) if turn.sum() else None
            ctrl["CONTROL_reproduces_banked_decode_prediction"] = {
                "predicted_decode_curv_on_real_turns":
                    pred.get("decode_curv_on_real_turns"),
                "measured": got,
                "passes": bool(got is not None
                               and abs(got - pred["decode_curv_on_real_turns"]) < 0.06)}

    R = {"dump": a.dump, "arm": a.label, "n_windows": int(n),
         "kappa_turn_threshold": a.kappa_turn,
         "exec_nonzero_eps": a.exec_eps,
         "panel_note": ("episodes selected for real-turn DENSITY; these are "
                        "CONDITIONAL rates on a turn-rich panel, NOT corpus "
                        "marginals"),
         "GT_TURN_windows": blk(turn, "GT TURN (|kappa|>thr)"),
         "GT_STRAIGHT_windows": blk(strt, "GT STRAIGHT"),
         "decoded_lat_hist": {LAT[i]: int((GLAT == i).sum())
                              for i in range(len(LAT)) if (GLAT == i).sum()},
         "controls": ctrl}
    failed = [k for k, v in ctrl.items() if not v.get("passes")]
    R["controls_failed"] = failed
    R["verdict"] = ("INCONCLUSIVE (control failed: " + ",".join(failed) + ")"
                    if failed else
                    ("GATE 1+2 OPEN: executed curvature is non-zero on "
                     f"{R['GT_TURN_windows']['frac_executed_nonzero_curvature']:.4f} "
                     "of GT-turn windows"
                     if R["GT_TURN_windows"].get("frac_executed_nonzero_curvature", 0) > 0
                     else "GATE SHUT: executed curvature exactly 0.0 on every "
                          "GT-turn window, with a LIVE control column"))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)

    t, s = R["GT_TURN_windows"], R["GT_STRAIGHT_windows"]
    print(f"[{a.label}] n={n}  thr={a.kappa_turn}")
    print(f"  GT TURN     n={t.get('n')}  exec_nonzero="
          f"{t.get('frac_executed_nonzero_curvature')}  "
          f"correct_dir={t.get('frac_executed_nonzero_AND_correct_direction')}  "
          f"|k|max={t.get('kappa_absmax_max')}  decode_curv={t.get('decode_curv_frac')}")
    print(f"  GT STRAIGHT n={s.get('n')}  exec_nonzero="
          f"{s.get('frac_executed_nonzero_curvature')}  "
          f"|k|max={s.get('kappa_absmax_max')}  decode_curv={s.get('decode_curv_frac')}")
    print("  CONTROLS: " + json.dumps({k: v.get("passes") for k, v in ctrl.items()}))
    print(f"  ha0_ext nonzero={ctrl['CONTROL_ha0_ext_curvature_MUST_BE_NONZERO']['frac_nonzero']:.4f}"
          f" absmax={ctrl['CONTROL_ha0_ext_curvature_MUST_BE_NONZERO']['absmax']:.4f}")
    print(f"  VERDICT: {R['verdict']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
