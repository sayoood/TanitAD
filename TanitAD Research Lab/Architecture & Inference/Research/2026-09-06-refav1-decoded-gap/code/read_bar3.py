#!/usr/bin/env python3
"""D-REFAV1-DK-DECODED BAR-3 reader -- the gap-source ladder's STRUCTURAL rows.

⛔ WHAT IS READABLE AT THIS n AND WHAT IS NOT. 29 windows over 17 episodes carries
no family claim, and refav1's planner SAMPLES: its inference-seed floor is
~0.30 m ADE and nothing here was run at >= 3 seeds. ⇒ the ONLY quantities quoted
are EXACT-EQUALITY COUNTS under a fixed seed with one flag moved -- identities
about this set of runs, not estimates. Metric deltas are printed for completeness
and are explicitly NOT claimed.

⭐ THE SAME-BREATH CONTROLS: the LATERAL family must be BIT-IDENTICAL across arms
(the term touches only the longitudinal channel) and `ha0`, the trivial
constant-velocity arm, must be unchanged (it does not plan).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True)
    ap.add_argument("--oof", default=None)
    ap.add_argument("--recal", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tags = ["off", "oracle", "decgap", "decoded"]
    res = {"task": "D-REFAV1-DK-DECODED BAR-3 -- the gap-source ladder",
           "evidence_class": "MEASURED (ours), dev-box RTX 4060",
           "tier": "T1 (self-action open loop)",
           "n_note": ("29 windows / 17 episodes -- NO family claim. The planner "
                      "SAMPLES (seed floor ~0.30 m ADE) and no arm was run at >=3 "
                      "inference seeds, so only EXACT-EQUALITY COUNTS are quoted."),
           "arms": {}}
    for t in tags:
        p = os.path.join(a.ab, "out_%s.json" % t)
        if not os.path.exists(p):
            res["arms"][t] = {"status": "NOT RUN"}
            continue
        d = json.load(open(p, encoding="utf-8"))
        r = d.get("refav1", {})
        tp = get(r, "trivial_profile", "arms", "cl", default={})
        row = {
            "status": "OK",
            "n_windows": d.get("n_windows"), "n_episodes": d.get("n_episodes"),
            "cl_const_speed_frac": tp.get("const_speed_frac"),
            "cl_trivial_frac": tp.get("trivial_frac"),
            "cl_identical_to_ha0_n": get(tp, "identical_to", "ha0", "n"),
            "cl_identical_to_ha0_frac": get(tp, "identical_to", "ha0", "frac"),
            "ha0_trivial_frac": get(r, "trivial_profile", "arms", "ha0",
                                    "trivial_frac"),
        }
        cov = get(r, "distance_keeping", default={})
        row["distance_keeping"] = {k: cov.get(k) for k in
                                   ("status", "n", "mean_headway_min_m",
                                    "mean_time_gap_min_s", "mean_min_ttc_s")
                                   if k in cov}
        # ⭐ the per-arm four-family rows: `arms.cl.four_families.*`. The LATERAL
        # block is the SAME-BREATH CONTROL -- a longitudinal-only term must leave
        # it bit-identical -- and `ha0`'s rows must be unchanged because it does
        # not plan.
        row["families"] = {}
        for fk in ("lateral", "longitudinal"):
            f = get(d, "arms", "cl", "four_families", fk)
            if isinstance(f, dict):
                row["families"][fk] = {k: v for k, v in f.items()
                                       if isinstance(v, (int, float))}
        f = get(d, "arms", "ha0", "four_families", "lateral")
        if isinstance(f, dict):
            row["ha0_lateral"] = {k: v for k, v in f.items()
                                  if isinstance(v, (int, float))}
        for ak in ("cl", "ha0"):
            adev = get(d, "arms", ak, "ade_m")
            if adev is None:
                adev = get(d, "arms", ak, "four_families", "ade")
            row["%s_ade" % ak] = adev
        # the dk plan-time coverage record lives at
        # refav1.manifest.goal_rule.distance_keeping_cost
        for key in ("distance_keeping_cost",):
            v = get(r, "manifest", "goal_rule", key)
            if isinstance(v, dict):
                row.setdefault("dk_record", {})[key] = {
                    k: (vv if not isinstance(vv, list) else
                        {"n": len(vv),
                         "mean": (float(np.nanmean([x for x in vv
                                                    if x is not None]))
                                  if any(x is not None for x in vv) else None)})
                    for k, vv in v.items() if k != "spec"}
                if "spec" in v:
                    sp = v["spec"]
                    row.setdefault("dk_record", {})["spec"] = {
                        k: sp.get(k) for k in
                        ("w_dk", "tau_target_s", "d0_m", "gap_source",
                         "vision_only", "is_ceiling", "lead_speed_closure")}
                    if sp.get("gap_head"):
                        row["dk_record"]["gap_head"] = {
                            k: sp["gap_head"].get(k) for k in
                            ("head_sha256", "head_version", "ckpt_step",
                             "fit_corpus", "fit_scheme", "n_fit_rows",
                             "n_fit_clips")}
        res["arms"][t] = row

    ctrl = res["arms"].get("off", {})
    base_id = ctrl.get("cl_identical_to_ha0_n")
    res["BAR3"] = {
        "control_cl_identical_to_ha0_n": base_id,
        "control_const_speed_frac": ctrl.get("cl_const_speed_frac"),
        "note": ("the SPEC's phrasing 'moves off 1.0000' was the predecessor's "
                 "value on a 4-window dev-box slice. On THIS 29-window panel the "
                 "unarmed control is not 1.0000, so the criterion is read in its "
                 "intended form: does the ARMED arm's exact-equality count move "
                 "BELOW the unarmed control's, with the lateral family unchanged?"),
        "rows": {}}
    for t in ("oracle", "decgap", "decoded"):
        r = res["arms"].get(t, {})
        if r.get("status") != "OK":
            res["BAR3"]["rows"][t] = {"status": r.get("status", "MISSING")}
            continue
        n = r.get("cl_identical_to_ha0_n")
        lat_same = None
        if r.get("families") and ctrl.get("families"):
            lat_same = (json.dumps(r["families"].get("lateral"), sort_keys=True)
                        == json.dumps(ctrl["families"].get("lateral"),
                                      sort_keys=True))
        ha0_same = None
        if r.get("ha0_lateral") and ctrl.get("ha0_lateral"):
            ha0_same = (json.dumps(r["ha0_lateral"], sort_keys=True)
                        == json.dumps(ctrl["ha0_lateral"], sort_keys=True))
        res["BAR3"]["rows"][t] = {
            "ha0_lateral_bit_identical_to_control": ha0_same,
            "cl_identical_to_ha0_n": n,
            "moved_off_control": (None if (n is None or base_id is None)
                                  else bool(n < base_id)),
            "delta_windows": (None if (n is None or base_id is None)
                              else int(base_id - n)),
            "const_speed_frac": r.get("cl_const_speed_frac"),
            "lateral_bit_identical_to_control": lat_same,
            "ha0_trivial_frac": r.get("ha0_trivial_frac")}

    # ---------------------------------------------------------------------- #
    # ⭐ END-TO-END CROSS-CHECK: the gap the PLANNER decoded in-loop, against the
    # gap the BANK decoded offline for the same windows. They come through two
    # different code paths -- the arm re-encodes `feats` inside the window loop,
    # the bank read the fp8 cache in a separate process -- so agreement is real
    # evidence that the head applied in the planner IS the head that was scored.
    # ⛔ Without it, "the head is in the loop" is an argument, not a measurement.
    # ---------------------------------------------------------------------- #
    if a.oof and a.recal and os.path.exists(a.oof):
        z = np.load(a.oof, allow_pickle=True)
        rc = json.load(open(a.recal, encoding="utf-8"))["recalibration"]["per_fold"]
        pred = {}
        for c, fr, pv, fo in zip(z["clip_id"].astype(str), z["frame"],
                                 z["pred_field"], z["fold"]):
            r = rc[str(int(fo))]
            pred[(str(c), int(fr))] = r["alpha"] + r["beta"] * float(pv)
        for t in ("decgap", "decoded"):
            rec = get(res["arms"].get(t, {}), "dk_record",
                      "distance_keeping_cost", default={})
            dg = rec.get("decoded_gap")
            if not isinstance(dg, dict):
                continue
            # the dump stores the in-loop values in window order; compare the
            # SORTED vectors, which needs no order reconstruction
            raw = json.load(open(os.path.join(a.ab, "out_%s.json" % t),
                                 encoding="utf-8"))
            vals = get(raw, "refav1", "manifest", "goal_rule",
                       "distance_keeping_cost", "decoded_gap", default=[])
            og = get(raw, "refav1", "manifest", "goal_rule",
                     "distance_keeping_cost", "oracle_gap", default=[])
            # restrict to the windows this arm actually saw, via the ORACLE gaps
            # it recorded alongside (they key the same rows)
            ogs = sorted(x for x in og if x is not None)
            got = sorted(float(x) for x in vals)
            # match by oracle gap: build the expected set from the same rows
            oo = {}
            for c, fr, y in zip(z["clip_id"].astype(str), z["frame"], z["y"]):
                oo.setdefault(round(float(y), 4), []).append((str(c), int(fr)))
            sel = []
            for x in ogs:
                hit = oo.get(round(float(x), 4))
                if hit:
                    sel.append(hit[0])
            exp = sorted(pred[k] for k in sel if k in pred)
            n = min(len(exp), len(got))
            if n:
                d = float(np.max(np.abs(np.array(exp[:n]) - np.array(got[:n]))))
                res["arms"][t]["inloop_vs_bank"] = {
                    "n_matched": n, "n_inloop": len(got), "n_expected": len(exp),
                    "max_abs_delta_m": d,
                    "agrees": bool(d < 1e-3),
                    "note": ("the arm re-encodes `feats` in the window loop; the "
                             "bank read the fp8 cache in another process. "
                             "Agreement is evidence the planner's head IS the "
                             "scored head.")}
                print("in-loop vs bank [%s]: n=%d max|delta| = %.3e m -> %s"
                      % (t, n, d, "AGREES" if d < 1e-3 else "DISAGREES"))

    print("%-9s %-8s %-10s %-24s %-12s" % ("arm", "n_win", "const_sp",
                                           "cl identical to ha0", "lateral=="))
    for t in tags:
        r = res["arms"].get(t, {})
        if r.get("status") != "OK":
            print("%-9s %s" % (t, r.get("status", "MISSING")))
            continue
        b = res["BAR3"]["rows"].get(t, {})
        print("%-9s %-8s %-10s %2s/%-2s = %-14s %-12s"
              % (t, r.get("n_windows"), r.get("cl_const_speed_frac"),
                 r.get("cl_identical_to_ha0_n"), r.get("n_windows"),
                 r.get("cl_identical_to_ha0_frac"),
                 b.get("lateral_bit_identical_to_control", "-")))
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
