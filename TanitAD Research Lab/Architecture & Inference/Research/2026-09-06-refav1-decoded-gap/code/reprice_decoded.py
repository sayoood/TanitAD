#!/usr/bin/env python3
"""D-REFAV1-DK-DECODED BAR-2 -- does the cost still RE-RANK when its gap comes
from VISION instead of from the label?  ZERO-GPU: it re-prices the ALREADY-BANKED
141-episode 21109 panel, once with the ORACLE gap and once with the CROSS-FITTED
DECODED gap, on the SAME windows and the SAME two baselines.

⭐ THE COMPARISON IS CLEAN BY CONSTRUCTION. On the planner's own
constant-acceleration baselines both surviving regularisers are IDENTICALLY zero
(`mean(jerk^2)` = 0 for any constant `a`; `kappa` = 0 for both) and `w_vend` was
never armed -- so the distance-keeping term is the ONLY discriminator and every
flip is attributable to it alone.

⛔ THE ORACLE PASS IS A REPRODUCTION CONTROL, NOT DECORATION. It must recover
`D-REFAV1-DK-COST`'s published counts (21 violating of 90 LEAD windows, 21/21
flips). If it does not, the decoded numbers beside it are unreadable and the run
is reported as such rather than quoted.

⛔ BOTH ERROR DIRECTIONS ARE SCORED, because only one of them is flattering:
  * SENSITIVITY -- of the oracle's violating windows, how many does the decoded
    gap also fire on AND flip?
  * SPECIFICITY -- of the oracle's ADEQUATE windows, how many does it correctly
    leave alone?  ⚠️ A head that simply UNDER-PREDICTS every gap fires everywhere
    and reproduces the flips trivially; specificity is what catches that, and it
    is why the bar was pre-registered with both halves.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

DT = 0.2
K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--oof", default=None, help="oof_gap_all_lead.npz; "
                    "omit for the ORACLE-only reproduction pass")
    ap.add_argument("--emit-window-list", default=None,
                    help="write the oracle VIOLATING + ADEQUATE windows as an "
                         "explicit (episode name, t) list for --window-list")
    ap.add_argument("--oof-present", default=None, help="oof_present.npz")
    ap.add_argument("--stack", required=True)
    ap.add_argument("--taniteval", required=True)
    ap.add_argument("--present-thr", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    sys.path.insert(0, a.taniteval)
    sys.path.insert(0, os.path.join(a.taniteval, "tools"))
    from tanitad.refs import refav1_lon_cost as dk

    files = sorted(os.path.join(a.dump, x) for x in os.listdir(a.dump)
                   if x.startswith("ep") and x.endswith(".npz"))
    manifest = json.load(open(os.path.join(a.dump, "manifest.json"),
                              encoding="utf-8"))
    eps = manifest.get("episodes") or []
    by_fi = {int(e.get("file_index", i)): e for i, e in enumerate(eps)}

    blk = np.load(a.block, allow_pickle=True)
    idx = {}
    b_cid = blk["clip_id"].astype(str)
    b_fr = blk["frame"].astype(np.int64)
    for i in range(b_cid.size):
        idx[(b_cid[i], int(b_fr[i]))] = i
    b_gap = blk["gap0_m"].astype(np.float64)
    b_state = blk["state"].astype(str)
    b_speed = blk["speeds"].astype(np.float64)

    # ---- the panel's windows, with their (clip, RAW frame) keys -------------
    clip_w, frame_w, v0_w, eid_w = [], [], [], []
    for fi, f in enumerate(files):
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(int).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float64).reshape(-1)
        clip = (by_fi.get(fi) or {}).get("clip_id") or ""
        for t, v in zip(ws, v0):
            clip_w.append(str(clip))
            frame_w.append(2 * int(t))
            v0_w.append(float(v))
            eid_w.append(fi)
    clip_w = np.array(clip_w)
    frame_w = np.array(frame_w, dtype=np.int64)
    v0_w = np.array(v0_w, dtype=np.float64)
    eid_w = np.array(eid_w, dtype=np.int64)
    W = v0_w.size

    rows = np.array([idx.get((c, int(fr)), -1) for c, fr in zip(clip_w, frame_w)],
                    dtype=np.int64)
    have = rows >= 0
    state_w = np.array(["NO_LABEL"] * W, dtype=object)
    gap_o = np.full(W, np.nan)
    state_w[have] = b_state[rows[have]]
    gap_o[have] = b_gap[rows[have]]
    # ⛔ LABEL-FREE ALIGNMENT PROOF on the joined windows (the same one the arm
    # runs at plan time). A mis-join would price the plan against another clip.
    dv = np.abs(v0_w[have] - b_speed[rows[have]])
    speed_max = float(np.nanmax(dv)) if dv.size else float("nan")

    # ---- the DECODED gap, out-of-fold, on the same key ----------------------
    if a.oof:
        z = np.load(a.oof, allow_pickle=True)
        dmap = {(str(c), int(f)): float(p) for c, f, p in
                zip(z["clip_id"], z["frame"], z["pred_field"])}
        pmap = {(str(c), int(f)): float(p) for c, f, p in
                zip(z["clip_id"], z["frame"], z["pred_pix"])}
        gap_d = np.array([dmap.get((c, int(fr)), np.nan)
                          for c, fr in zip(clip_w, frame_w)])
        gap_px = np.array([pmap.get((c, int(fr)), np.nan)
                           for c, fr in zip(clip_w, frame_w)])
    else:
        gap_d = np.full(W, np.nan)
        gap_px = np.full(W, np.nan)
    pres_d = np.full(W, np.nan)
    if a.oof_present and os.path.exists(a.oof_present):
        zp = np.load(a.oof_present, allow_pickle=True)
        ppm = {(str(c), int(f)): float(p) for c, f, p in
               zip(zp["clip_id"], zp["frame"], zp["pred_field"])}
        pres_d = np.array([ppm.get((c, int(fr)), np.nan)
                           for c, fr in zip(clip_w, frame_w)])

    is_lead = np.isfinite(gap_o) & (state_w == "LEAD")
    s_star = dk.DK_D0_M + dk.DK_TAU_TARGET_S * v0_w
    viol_o = is_lead & (gap_o < s_star)
    adeq_o = is_lead & ~viol_o

    # ---- the two baselines iCEM actually compares --------------------------
    spec = dk.DistanceKeepingSpec(w_dk=1.0, gap_source=dk.GAP_SOURCE_ORACLE)

    a_zero = np.zeros(K)
    a_dec = np.full(K, -1.5)

    def pass_over(gap_arr, mask, name):
        """-> per-window (fires, flips) under `gap_arr`."""
        fires = np.zeros(W, dtype=bool)
        flips = np.zeros(W, dtype=bool)
        cz = np.full(W, np.nan)
        cd = np.full(W, np.nan)
        for i in np.flatnonzero(mask):
            g = gap_arr[i]
            if not np.isfinite(g):
                continue
            g = max(0.0, float(g))          # the arm clamps a negative decode
            fires[i] = g < s_star[i]
            aa = torch.tensor(a_zero, dtype=torch.float64)[None]
            c0 = torch.stack([aa, torch.zeros_like(aa)], dim=-1)
            cz[i] = float(dk.distance_keeping_cost(
                c0, v0=float(v0_w[i]), gap0_m=g, dt=DT, spec=spec))
            ad = torch.tensor(a_dec, dtype=torch.float64)[None]
            c1 = torch.stack([ad, torch.zeros_like(ad)], dim=-1)
            cd[i] = float(dk.distance_keeping_cost(
                c1, v0=float(v0_w[i]), gap0_m=g, dt=DT, spec=spec))
            flips[i] = cd[i] < cz[i]
        return fires, flips, cz, cd

    f_o, fl_o, cz_o, cd_o = pass_over(gap_o, is_lead, "oracle")
    f_d, fl_d, cz_d, cd_d = pass_over(gap_d, is_lead, "decoded")
    f_p, fl_p, _, _ = pass_over(gap_px, is_lead, "pixel")

    n_lead = int(is_lead.sum())
    n_viol_o = int(viol_o.sum())
    n_flip_o = int((viol_o & fl_o).sum())

    # ---- BAR-2 ---------------------------------------------------------------
    sens_n = int((viol_o & f_d & fl_d).sum())
    sens = sens_n / max(n_viol_o, 1)
    spec_n = int((adeq_o & ~f_d).sum())
    speci = spec_n / max(int(adeq_o.sum()), 1)
    sens_px = int((viol_o & f_p & fl_p).sum()) / max(n_viol_o, 1)
    spec_px = int((adeq_o & ~f_p).sum()) / max(int(adeq_o.sum()), 1)

    ok_repro = (n_lead == 90 and n_viol_o == 21 and n_flip_o == 21)
    bar2 = bool(sens >= 0.80 and speci >= 0.70)

    def stat(m, arr):
        v = arr[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {"n": int(v.size), "mean": round(float(v.mean()), 4),
                "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    res = {
        "task": "D-REFAV1-DK-DECODED BAR-2 -- re-ranking reproduction, decoded vs oracle gap",
        "evidence_class": "MEASURED (ours) -- zero-GPU re-pricing of the banked "
                          "refav1-21109 141-episode panel",
        "tier": "NOT a driving number. Re-prices a BANKED panel's candidate set; "
                "nothing is rolled and no model is loaded.",
        "panel": {"n_windows": W, "n_episodes": int(np.unique(eid_w).size),
                  "n_lead_windows": n_lead,
                  "states": {s: int((state_w == s).sum())
                             for s in sorted(set(map(str, state_w)))},
                  "speed_check_max_mps": speed_max},
        "oracle_reproduction_control": {
            "n_lead_windows": n_lead, "expected": 90,
            "n_in_violation": n_viol_o, "expected_violating": 21,
            "n_flips": n_flip_o, "expected_flips": 21,
            "reproduces_published": bool(ok_repro),
            "note": "if this is not 90/21/21 the decoded numbers beside it are "
                    "not readable against the published ceiling"},
        "decoded": {
            "n_with_decode": int(np.isfinite(gap_d[is_lead]).sum()),
            "gap_oracle": stat(is_lead, gap_o),
            "gap_decoded": stat(is_lead, gap_d),
            "gap_err_m": {
                "mae": round(float(np.nanmean(np.abs(gap_d[is_lead]
                                                     - gap_o[is_lead]))), 4),
                "bias": round(float(np.nanmean(gap_d[is_lead]
                                               - gap_o[is_lead])), 4)},
            "n_fires": int((is_lead & f_d).sum()),
            "n_flips": int((is_lead & f_d & fl_d).sum()),
        },
        "BAR2": {
            "sensitivity": round(sens, 4), "sensitivity_n": sens_n,
            "sensitivity_of": n_viol_o, "bar": 0.80,
            "specificity": round(speci, 4), "specificity_n": spec_n,
            "specificity_of": int(adeq_o.sum()), "bar_spec": 0.70,
            "PASS": bar2,
            "pixel_floor_sensitivity": round(sens_px, 4),
            "pixel_floor_specificity": round(spec_px, 4),
        },
        "gap_source_ladder_note":
            "the ORACLE pass uses gap AND gate from the label; the DECODED pass "
            "uses the cross-fitted vision head's gap on the SAME oracle-selected "
            "LEAD windows, which is the `decoded_gap` rung. The `decoded` rung "
            "(gate also decoded) is measured by the planner A/B, not here.",
    }
    if np.isfinite(pres_d).any():
        gate_d = pres_d >= a.present_thr
        res["decoded_gate"] = {
            "thr": a.present_thr,
            "tp": int((gate_d & is_lead).sum()),
            "fp": int((gate_d & ~is_lead & np.isfinite(pres_d)).sum()),
            "fn": int((~gate_d & is_lead).sum()),
            "tn": int((~gate_d & ~is_lead & np.isfinite(pres_d)).sum()),
            "recall_on_lead": round(float(np.mean(gate_d[is_lead])), 4),
            "recall_on_violating": round(float(np.mean(gate_d[viol_o])), 4)
            if n_viol_o else None,
        }
    if a.emit_window_list:
        # ⛔ The subset is ORACLE-SELECTED and IDENTICAL FOR EVERY ARM, so it is a
        # measurement choice and not an input to any arm. Episode NAMES (never
        # indices) because an index is meaningful only against one loader build.
        eplist = [os.path.splitext(os.path.basename(f))[0] for f in files]
        # dump file name != loader episode name; take it from the manifest
        wl = []
        for i in np.flatnonzero(viol_o | adeq_o):
            nm = (by_fi.get(int(eid_w[i])) or {}).get("name")
            if not nm:
                raise SystemExit("manifest episode %d has no `name`" % eid_w[i])
            wl.append([nm, int(frame_w[i] // 2)])
        doc = {"rule": ("D-REFAV1-DK-DECODED BAR-3: the ORACLE's LEAD windows of "
                        "the banked 21109 panel -- %d VIOLATING + %d ADEQUATE. "
                        "Identical for every arm; the gap SOURCE is the only "
                        "variable." % (int(viol_o.sum()), int(adeq_o.sum()))),
               "windows": wl,
               "n_violating": int(viol_o.sum()), "n_adequate": int(adeq_o.sum()),
               "violating": [[(by_fi.get(int(eid_w[i])) or {}).get("name"),
                              int(frame_w[i] // 2)]
                             for i in np.flatnonzero(viol_o)]}
        json.dump(doc, open(a.emit_window_list, "w"), indent=1)
        print("window list -> %s (%d windows, %d violating)"
              % (a.emit_window_list, len(wl), int(viol_o.sum())))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in
                      ("oracle_reproduction_control", "decoded", "BAR2")}, indent=1))
    print("\nBAR-2 %s   (sens %.3f >= 0.80 ? %s ; spec %.3f >= 0.70 ? %s)"
          % ("PASS" if bar2 else "FAIL", sens, sens >= 0.80, speci, speci >= 0.70))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
