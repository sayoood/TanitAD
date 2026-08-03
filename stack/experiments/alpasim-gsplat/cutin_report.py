#!/usr/bin/env python3
"""STREAM B report: does a CLOSE lead vehicle change either arm's driving?

The prior with-objects vs empty-road panel was NULL, but it only ever tested DISTANT
traffic. `scene_geometry.py` then established at TWO independent probes that this scene
contains no close-following geometry at all (min in-lane headway 46.2 m), so the close
conditions are CONSTRUCTED by `synth_actor.py` from the scene's own gaussians.

WHAT MAKES THE RESULT QUOTABLE — three controls, all reported whether they pass or fail:

  1. NOISE FLOOR (`behind`). The same actor gaussians, posed behind a forward-facing
     camera: rendering is provably unchanged (0 pixels differ, `synth_verify.json`), so
     `behind - empty` is the harness's own reproducibility. ⛔ A lead-vs-empty delta that
     does not EXCEED this is not interpretable, no matter what its CI says.
  2. DOSE-RESPONSE. lead25 / lead15 / lead8 are three points on one axis. A genuine
     reaction to a lead vehicle should grow monotonically as the gap closes; a rendering
     artefact has no reason to.
  3. SELF-CONSISTENCY. Every family-level point estimate is recomputed here, directly
     from the per-window components, and compared with the estimator's own output. A
     mismatch is reported as a FAILURE and blocks the number — this is the control that
     once caught a 5347x curvature inflation in a reducer.

Every contrast is the PAIRED episode-cluster bootstrap on identical (start, k) windows,
and all FOUR FAMILIES are reported alongside ADE, never ADE alone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import cl_metrics as M

LEAD_CONDS = ("behind", "lead25", "lead15", "lead8", "cutin")
ARMS = ("flagship-v1", "refc-base")

# the metrics carried through every contrast, grouped by the binding families
FAMILY_OF = {
    "ade_0_2s": "ADE", "dist_to_gt_traj_m": "ADE",
    "abs_target_speed_err_ms": "LONGITUDINAL", "along_track_ade_m": "LONGITUDINAL",
    "abs_executed_speed_err_ms": "LONGITUDINAL",
    "synth_lead_headway_m": "LONGITUDINAL", "synth_lead_time_gap_s": "LONGITUDINAL",
    "synth_lead_frac_tg_below_1s": "LONGITUDINAL",
    "synth_lead_collision_rate": "LONGITUDINAL",
    "heading_err_rad": "LATERAL", "curvature_err_1pm": "LATERAL",
    "yawrate_err_rads": "LATERAL", "cross_track_abs_m": "LATERAL",
    "lateral_ade_m": "LATERAL",
    "manoeuvre_plan_eq_logged": "TACTICAL",
    "route_corridor_departure_rate": "STRATEGIC",
}


def rollouts_path(root: Path, arm: str, cond: str) -> Path:
    return root / f"{arm}_{cond}" / f"rollouts_{arm}_{cond}.json"


def paired(rowsA, eidA, rowsB, eidB):
    """Pair on identical (start, k) windows. Returns (pair_dict, n_common)."""
    key = lambda rows, eids: {(int(e), int(r["k"])): r for r, e in zip(rows, eids)}
    KA, KB = key(rowsA, eidA), key(rowsB, eidB)
    common = sorted(set(KA) & set(KB))
    pe = [c[0] for c in common]
    fns = {
        "ade_0_2s": lambda r: r["ade"],
        "dist_to_gt_traj_m": lambda r: r["dist_to_gt"],
        "abs_target_speed_err_ms": lambda r: abs(r["speed_err"]),
        "abs_executed_speed_err_ms": lambda r: abs(r["speed_track_err"]),
        "along_track_ade_m": lambda r: r["lon_ade"],
        "synth_lead_headway_m": lambda r: r.get("synth_headway"),
        "synth_lead_time_gap_s": lambda r: r.get("synth_time_gap"),
        "synth_lead_frac_tg_below_1s": lambda r: r.get("synth_tg_below_1s"),
        "synth_lead_collision_rate": lambda r: r.get("synth_collision"),
        "heading_err_rad": lambda r: abs(r["heading_err"]),
        "curvature_err_1pm": lambda r: r["curv_err"],
        "yawrate_err_rads": lambda r: r["yawrate_err"],
        "cross_track_abs_m": lambda r: abs(r["cross_track"]),
        "lateral_ade_m": lambda r: r["lat_ade"],
        "manoeuvre_plan_eq_logged": lambda r: float(r["man_plan"] == r["man_gt"]),
        "route_corridor_departure_rate": lambda r: float(r["corridor_departure"]),
    }
    out, checks = {}, {}
    for lab, fn in fns.items():
        a = [np.nan if fn(KA[c]) is None else fn(KA[c]) for c in common]
        b = [np.nan if fn(KB[c]) is None else fn(KB[c]) for c in common]
        r = M._paired(a, b, pe)
        r["family"] = FAMILY_OF.get(lab, "?")
        out[lab] = r
        # ---- SELF-CONSISTENCY: recompute the point estimate from the components ----
        av, bv = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(av) & np.isfinite(bv)
        if ok.any() and "delta" in r:
            direct = float(av[ok].mean() - bv[ok].mean())
            err = abs(direct - r["delta"])
            checks[lab] = {"direct_mean_delta": round(direct, 6),
                           "estimator_delta": r["delta"],
                           "abs_diff": round(err, 8),
                           "PASS": bool(err <= max(5e-4, 0.02 * abs(direct)))}
    return out, len(common), checks


def load(root, arm, cond, lead_ref):
    p = rollouts_path(root, arm, cond)
    if not p.exists():
        return None
    d, rows, eids, summ = M.collect(p, None, lead_ref=lead_ref)
    return {"payload": d, "rows": rows, "eids": eids, "summ": summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default=None, help="synth_verify.json (the falsifier)")
    ap.add_argument("--geometry", default=None, help="scene_geometry.json (the 2 probes)")
    a = ap.parse_args()
    root = Path(a.root)

    res = {
        "what": "close-following / cut-in discriminating panel",
        "provenance": {
            "geometry_FOUND_in_scene": False,
            "geometry_note": ("NOT FOUND. Two independent probes agree: "
                              "sequence_tracks.json min in-lane headway 46.246 m and "
                              "clipgt/obstacle.parquet 46.224 m, both with 0 rows under "
                              "25 m / 2 s and 0 cut-in events. The close conditions below "
                              "are CONSTRUCTED (synth_actor.py) from the scene's own "
                              "gaussians on the ego's own logged path."),
        },
        "estimator_note": (
            "paired episode-cluster bootstrap (taniteval.ci) on identical (start, k) "
            "windows. Clusters are the 9 ROLLOUT STARTS — disjoint segments of ONE clip, "
            "not 40 independent val episodes."),
        "within_sim_note": (
            "WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on these NuRec "
            "reconstructions vs 0.4728 on real footage (3.21x OOD)."),
        "arms": {}, "arm_contrast": {}, "self_consistency": {},
    }
    for f in ("verify", "geometry"):
        p = getattr(a, f)
        if p and Path(p).exists():
            res["provenance"][f] = json.loads(Path(p).read_text()) if f == "verify" else \
                json.loads(Path(p).read_text())["summary"]
            if f == "verify":
                res["provenance"]["verify"].pop("per_cond", None)

    for arm in ARMS:
        base = load(root, arm, "empty", None)
        if base is None:
            res["arms"][arm] = {"error": "no empty control"}
            continue
        entry = {"empty": {"n_windows": len(base["rows"]),
                           "n_clusters": int(len(set(base["eids"].tolist()))),
                           "families": M.families(base["rows"], base["eids"], base["summ"])}}
        for cond in LEAD_CONDS:
            cur = load(root, arm, cond, cond)
            if cur is None:
                entry[cond] = {"error": "missing run"}
                continue
            ctl = load(root, arm, "empty", cond)     # same lead_ref -> counterfactual
            pr, n, chk = paired(cur["rows"], cur["eids"], ctl["rows"], ctl["eids"])
            entry[cond] = {
                "n_windows": len(cur["rows"]),
                "n_paired_windows": n,
                "n_clusters": int(len(set(cur["eids"].tolist()))),
                "synth_attach": cur["payload"].get("synth_attach"),
                "lead_path_extrapolated_calls":
                    cur["payload"].get("lead_path_extrapolated_calls"),
                "families": M.families(cur["rows"], cur["eids"], cur["summ"]),
                "paired_vs_empty": pr,
                "rollout_min_headway_m": [s.get("synth_min_headway_m") for s in cur["summ"]],
            }
            res["self_consistency"][f"{arm}/{cond}"] = chk
        res["arms"][arm] = entry

    # ---- arm-vs-arm inside each condition -------------------------------------- #
    for cond in ("empty",) + LEAD_CONDS:
        A = load(root, ARMS[0], cond, None if cond == "empty" else cond)
        B = load(root, ARMS[1], cond, None if cond == "empty" else cond)
        if A is None or B is None:
            continue
        pr, n, chk = paired(A["rows"], A["eids"], B["rows"], B["eids"])
        res["arm_contrast"][cond] = {"A": ARMS[0], "B": ARMS[1],
                                     "n_paired_windows": n, "paired_A_minus_B": pr}
        res["self_consistency"][f"ARMS/{cond}"] = chk

    # ---- the noise floor, and what survives it ---------------------------------- #
    floor = {}
    for arm in ARMS:
        e = res["arms"].get(arm, {})
        b = e.get("behind", {}).get("paired_vs_empty")
        if not b:
            continue
        floor[arm] = {k: abs(v.get("delta", 0.0)) for k, v in b.items()}
    res["noise_floor_abs_delta_behind_minus_empty"] = floor
    surv = {}
    for arm in ARMS:
        if arm not in floor:
            continue
        for cond in ("lead25", "lead15", "lead8", "cutin"):
            pv = res["arms"][arm].get(cond, {}).get("paired_vs_empty")
            if not pv:
                continue
            for k, v in pv.items():
                if not v.get("separated"):
                    continue
                if abs(v["delta"]) > floor[arm].get(k, 0.0):
                    surv.setdefault(arm, {}).setdefault(cond, []).append(
                        {"metric": k, "family": v.get("family"), "delta": v["delta"],
                         "lo": v["lo"], "hi": v["hi"],
                         "noise_floor": round(floor[arm].get(k, 0.0), 6)})
    res["separated_AND_above_noise_floor"] = surv

    # ---- dose-response: is the effect monotone in closeness? -------------------- #
    dose = {}
    for arm in ARMS:
        rows = {}
        for k in FAMILY_OF:
            seq = []
            for cond in ("lead25", "lead15", "lead8"):
                pv = res["arms"].get(arm, {}).get(cond, {}).get("paired_vs_empty", {})
                if k in pv and "delta" in pv[k]:
                    seq.append(pv[k]["delta"])
            if len(seq) == 3:
                mono = bool((seq[0] <= seq[1] <= seq[2]) or (seq[0] >= seq[1] >= seq[2]))
                rows[k] = {"family": FAMILY_OF[k], "lead25": seq[0], "lead15": seq[1],
                           "lead8": seq[2], "monotone": mono}
        dose[arm] = rows
    res["dose_response"] = dose

    # ---- the programme's TOP DEFECT, tested where it should be easiest to trip ---- #
    # REF-C's 5-way manoeuvre head has NEVER emitted a longitudinal class (accelerate 0,
    # brake_stop 0 over 859 open-loop windows). A lead vehicle a few metres ahead with a
    # closing gap is the strongest possible brake stimulus. If the head still emits no
    # longitudinal class here, the defect is architectural, not a data-coverage problem.
    tac = {}
    for arm in ARMS:
        for cond in ("empty",) + LEAD_CONDS:
            f = res["arms"].get(arm, {}).get(cond, {}).get("families")
            if not f:
                continue
            t = f["TACTICAL"]
            row = {"head_class_share": t.get("head_class_share"),
                   "plan_class_share": t.get("plan_class_share"),
                   "logged_class_share": t.get("logged_class_share")}
            hs = t.get("head_class_share") or {}
            row["head_emits_longitudinal"] = bool(
                (hs.get("accelerate", 0) or 0) > 0 or (hs.get("brake_stop", 0) or 0) > 0)
            ps = t.get("plan_class_share") or {}
            row["plan_emits_longitudinal"] = bool(
                (ps.get("accelerate", 0) or 0) > 0 or (ps.get("brake_stop", 0) or 0) > 0)
            tac[f"{arm}/{cond}"] = row
    res["tactical_longitudinal_emission"] = tac

    n_fail = sum(1 for d in res["self_consistency"].values()
                 for c in d.values() if not c["PASS"])
    res["self_consistency_failures"] = n_fail
    res["self_consistency_verdict"] = "PASS" if n_fail == 0 else "FAIL"

    Path(a.out).write_text(json.dumps(res, indent=1))
    # ------------------------------- readable ---------------------------------- #
    print("SELF-CONSISTENCY:", res["self_consistency_verdict"],
          f"({n_fail} failures over "
          f"{sum(len(d) for d in res['self_consistency'].values())} checks)")
    for arm in ARMS:
        e = res["arms"].get(arm, {})
        if "error" in e:
            print(arm, e); continue
        print(f"\n===== {arm} =====")
        for cond in LEAD_CONDS:
            c = e.get(cond, {})
            if "error" in c:
                print(f"  {cond}: {c['error']}"); continue
            L = c["families"]["LONGITUDINAL"]
            hw = L.get("synth_lead_headway_m", {})
            tg = L.get("synth_lead_time_gap_s", {})
            col = L.get("synth_lead_collision_rate", {})
            inl = L.get("synth_lead_inlane_rate", {})
            # THE STIMULUS ACTUALLY DELIVERED — printed before any effect, so the
            # reader knows what the arm was shown rather than trusting the label.
            print(f"  -- {cond}: n={c['n_paired_windows']} paired | STIMULUS: "
                  f"headway {hw.get('mean')} m [{hw.get('lo')},{hw.get('hi')}], "
                  f"time-gap {tg.get('mean')} s, in-lane rate {inl.get('mean')}, "
                  f"collision rate {col.get('mean')}")
            print(f"       min headway per rollout: {c['rollout_min_headway_m']}")
            for k, v in c["paired_vs_empty"].items():
                if "delta" not in v:
                    continue
                star = "*" if v.get("separated") else " "
                fl = floor.get(arm, {}).get(k, 0.0)
                above = "ABOVE-FLOOR" if abs(v["delta"]) > fl else "below-floor"
                print(f"     {star} {v.get('family','?'):12s} {k:32s} "
                      f"{v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
                      f"floor={fl:.4f} {above}")
    print("\nTACTICAL longitudinal-class emission under a close lead "
          "(the programme's top defect, maximally provoked):")
    for k, v in res["tactical_longitudinal_emission"].items():
        print(f"  {k:26s} head_emits_lon={v['head_emits_longitudinal']!s:5s} "
              f"plan_emits_lon={v['plan_emits_longitudinal']!s:5s} "
              f"head={v['head_class_share']} plan={v['plan_class_share']}")
    print("\nSURVIVORS (separated AND above the behind-vs-empty noise floor):")
    print(json.dumps(surv, indent=1) if surv else "  none")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
