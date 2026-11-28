#!/usr/bin/env python3
"""C-ENV-1 step 3 -- run the constraint panel on the B1 EVAL split.

Scores `tanitad.eval.constraints` on real trajectories and the real agent join,
so the metric's `n`, its CENSORING rate and its dynamic range are measured
rather than asserted.

ARMS -- one variable each, and two of them are DELIBERATE REGRESSIONS:

  gt          the ground-truth future trajectory. The reference. If GT itself
              violated its own envelope everywhere, the thresholds would be
              wrong and nothing below would mean anything.
  gt_x2speed  ⛔ REGRESSION. Identical PATH, speed doubled. The ceilings are
              therefore bit-identical to `gt` and only the scored quantity
              moves -- one variable. MUST fail the OVER side.
  gt_creep    ⛔ REGRESSION. Identical PATH, speed pinned to 1.0 m/s. MUST fail
              the UNDER side. This is the arm that proves the metric is
              two-sided: a creeping arm satisfies every ceiling and would score
              perfectly under a one-sided criterion.
  gt_x2speed and gt_creep share `gt`'s geometry exactly, so any difference is
  attributable to the speed channel alone.

⛔ THE VACUITY GATE travels with every number: `manoeuvre_rate` is the fraction
of scored windows whose path actually turns (|dyaw| over the band above a
threshold). MEASURED 2026-09-06 on a sibling stream, a friction-circle zero was
bought by a `turn_left` recall of exactly 0.0000 -- an arm that declines the
manoeuvre satisfies every constraint trivially.
⚠️ The turn threshold here is a REPORTING statistic for that gate. It is NOT the
banned `|dyaw| > 0.15` selection gate: nothing is selected, filtered, or scored
by it, and every window is scored either way.

⛔ NO MODEL IS RUN. These are trajectory arms over banked GT, so the numbers
below characterise the INSTRUMENT, not any TanitAD model.
"""
from __future__ import annotations

import json
import lzma
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

from tanitad.eval.constraints import (  # noqa: E402
    CLEARANCE_DATUM, LEAD_LAT_M, clearance_speed_ceiling, combined_ceiling,
    curvature_from_xy, kinematic_speed_ceiling, lead_gap,
    speed_envelope_report, trajectory_clearance,
)

DT_S = 0.1007            # the join's own recovered grid, not 0.1
K_LO, K_HI = 20, 60      # the v7 TACTICAL band, 2.0-6.0 s
STRIDE = 10
TURN_DYAW_RAD = 0.20     # reporting-only, for the vacuity gate
MAX_RANGE_M = 60.0


def ego_frame(p_ref, pts):
    """Express world points in the ego frame at pose `p_ref` (+x fwd, +y left)."""
    c, s = math.cos(-p_ref[2]), math.sin(-p_ref[2])
    d = pts - p_ref[:2]
    return np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])


def load_join(xz_path):
    per_clip = defaultdict(dict)
    cls_seen = Counter()
    with lzma.open(xz_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ags = [(a["cx"], a["cy"], a["yaw"], a["l"], a["w"], a["cls"])
                   for a in r["agents"]]
            per_clip[r["clip_id"]][int(r["frame"])] = ags
            for a in r["agents"]:
                cls_seen[a["cls"]] += 1
    return per_clip, cls_seen


def under_side_sweep(ceil, v_gt, has_lead):
    """⛔ IS THE UNDER-DRIVING SIDE SCOREABLE AGAINST THESE CEILINGS AT ALL?

    Sweeps the two thresholds that define "the situation allows max speed" and
    "this counts as under-driving", and reports the GT under-rate at each. GT is
    competent human driving: a criterion that flags GT as under-driving on most
    windows is measuring the CEILING's uninformativeness, not the driver.

    Also splits by whether a LEAD was present. The kinematic ceiling on a
    straight road is ~50 m/s, so every legal speed sits far below it -- if the
    under-rate collapses once a real lead-gap ceiling binds, that localises the
    degeneracy to the no-lead case and says exactly which label is missing.
    """
    out = {"_reads": ("GT is competent human driving. A high GT under-rate is "
                      "evidence the CEILING is uninformative, not that the "
                      "driver is slow."),
           "n_steps": int(ceil.size), "grid": []}
    for allows in (8.0, 12.0, 16.0, 20.0):
        for margin in (3.0, 5.0, 10.0):
            m = ceil >= allows
            if not m.any():
                out["grid"].append({"allows_ms": allows, "margin_ms": margin,
                                    "status": "UNPOWERED", "n_allows": 0})
                continue
            und = m & ((ceil - v_gt) > margin)
            ml, mn = m & has_lead, m & ~has_lead
            out["grid"].append({
                "allows_ms": allows, "margin_ms": margin,
                "n_allows": int(m.sum()),
                "gt_under_rate": round(float(und.sum() / m.sum()), 4),
                "gt_under_rate_LEAD": (round(float((und & ml).sum() / ml.sum()), 4)
                                       if ml.any() else None),
                "n_lead": int(ml.sum()),
                "gt_under_rate_NO_LEAD": (round(float((und & mn).sum() / mn.sum()), 4)
                                          if mn.any() else None),
                "n_no_lead": int(mn.sum()),
            })
    return out


def main(eps_dir: Path, join_xz: Path, out_json: Path):
    t_start = time.time()
    per_clip, cls_seen = load_join(join_xz)
    files = sorted(eps_dir.glob("*.v2ep.pt"))

    arms = ("gt", "gt_x2speed", "gt_creep")
    acc = {a: defaultdict(float) for a in arms}
    n_win = 0
    n_clip_used = 0
    n_no_join = 0
    clearances, kin_ceils, clr_ceils = [], [], []
    censor = Counter()
    turn_flags = []
    # banked for the UNDER-SIDE SENSITIVITY SWEEP: the pair (ceiling, gt speed)
    # for every scored step, plus which family supplied the binding ceiling.
    sweep_ceil, sweep_v, sweep_has_lead = [], [], []

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        cid = d["clip_id"]
        if cid not in per_clip:
            n_no_join += 1
            continue
        n_clip_used += 1
        P = d["poses"].numpy().astype(np.float64)
        frames = per_clip[cid]
        T = P.shape[0]
        for t0 in range(0, T - K_HI - 1, STRIDE):
            idx = np.arange(t0 + K_LO, t0 + K_HI + 1)
            if idx[-1] >= T:
                continue
            traj_w = P[idx, :2]
            traj0 = ego_frame(P[t0], traj_w)        # planned path in ego@t0
            v_gt = P[idx, 3]
            dyaw = float(abs(np.unwrap(P[idx, 2])[-1] - P[idx, 2][0]))
            turn_flags.append(dyaw > TURN_DYAW_RAD)

            kappa, kgood = curvature_from_xy(traj0)
            v_kin, kin_ok = kinematic_speed_ceiling(kappa)
            kin_ok = kin_ok & kgood

            # ⛔ TWO DIFFERENT QUANTITIES, kept apart deliberately (see
            # constraints.lead_gap): the CEILING uses the in-corridor lead gap;
            # the PROXIMITY family uses the min distance to any agent. Feeding
            # the latter into the ceiling made GT violate its own envelope on
            # 71.2 % of steps.
            gaps = np.full(len(idx), np.nan)
            per_step_clear = []
            for j, k in enumerate(idx):
                ags = frames.get(int(k))
                if not ags:
                    continue
                boxes = [(a[0], a[1], a[2], a[3], a[4]) for a in ags]
                g = lead_gap(boxes, lat_m=LEAD_LAT_M, max_range_m=MAX_RANGE_M)
                if g is not None:
                    gaps[j] = g
                r = trajectory_clearance(np.zeros((1, 2)), boxes,
                                         max_range_m=MAX_RANGE_M)
                if not r.censored:
                    per_step_clear.append(r.min_clearance_m)
            v_clr, clr_ok = clearance_speed_ceiling(gaps)
            ceil, ceil_ok = combined_ceiling(v_kin, kin_ok, v_clr, clr_ok)

            if not ceil_ok.any():
                censor["no_ceiling_at_all"] += 1
            censor["steps_total"] += len(idx)
            censor["steps_no_kin"] += int((~kin_ok).sum())
            censor["steps_no_clr"] += int((~clr_ok).sum())
            censor["steps_no_ceiling"] += int((~ceil_ok).sum())

            if per_step_clear:
                clearances.extend(per_step_clear)
            kin_ceils.extend(v_kin[kin_ok].tolist())
            clr_ceils.extend(v_clr[clr_ok].tolist())
            sweep_ceil.extend(ceil[ceil_ok].tolist())
            sweep_v.extend(v_gt[ceil_ok].tolist())
            sweep_has_lead.extend(clr_ok[ceil_ok].tolist())

            speeds = {"gt": v_gt, "gt_x2speed": v_gt * 2.0,
                      "gt_creep": np.full(len(idx), 1.0)}
            for a in arms:
                rep = speed_envelope_report(speeds[a], v_ceiling=ceil,
                                            ceiling_valid=ceil_ok)
                if rep["status"] != "OK":
                    acc[a]["windows_censored"] += 1
                    continue
                acc[a]["windows_scored"] += 1
                acc[a]["n_scored"] += rep["n_scored"]
                acc[a]["n_over"] += rep["n_over"]
                acc[a]["n_allows"] += rep["n_situation_allows"]
                acc[a]["n_under"] += rep["n_under"]
                acc[a]["sum_overshoot"] += rep["mean_overshoot_ms"] * rep["n_over"]
                acc[a]["sum_shortfall"] += rep["mean_shortfall_ms"] * rep["n_under"]
            n_win += 1

    def summarise(a):
        s = acc[a]
        ns, no, nal, nu = (s["n_scored"], s["n_over"], s["n_allows"], s["n_under"])
        return {
            "windows_scored": int(s["windows_scored"]),
            "windows_censored": int(s["windows_censored"]),
            "n_steps_scored": int(ns),
            "frac_over_ceiling": round(no / ns, 6) if ns else None,
            "n_over": int(no),
            "mean_overshoot_ms": round(s["sum_overshoot"] / no, 4) if no else 0.0,
            "n_situation_allows": int(nal),
            "frac_under_when_allowed": round(nu / nal, 6) if nal else None,
            "n_under": int(nu),
            "mean_shortfall_ms": round(s["sum_shortfall"] / nu, 4) if nu else 0.0,
        }

    def dist(xs, name):
        if not xs:
            return {"status": "EMPTY", "n": 0}
        v = np.asarray(xs)
        return {"n": len(v), "mean": round(float(v.mean()), 4),
                "p05": round(float(np.percentile(v, 5)), 4),
                "p50": round(float(np.percentile(v, 50)), 4),
                "p95": round(float(np.percentile(v, 95)), 4),
                "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    out = {
        "_what": ("constraint panel on the B1 EVAL split: two-sided speed "
                  "envelope + clearance, with two deliberate-regression arms"),
        "_evidence_class": "MEASURED (ours; artifact = this json)",
        "_no_model_was_run": True,
        "_arms_are_trajectory_arms": ("gt_x2speed and gt_creep hold gt's PATH "
                                      "fixed, so the ceilings are identical and "
                                      "only the speed channel moves"),
        "clearance_datum": CLEARANCE_DATUM,
        "params": {"dt_s": DT_S, "band_steps": [K_LO, K_HI],
                   "band_s": [round(K_LO * DT_S, 2), round(K_HI * DT_S, 2)],
                   "stride": STRIDE, "max_range_m": MAX_RANGE_M,
                   "turn_dyaw_rad_REPORTING_ONLY": TURN_DYAW_RAD},
        "corpus": {"eps_dir": str(eps_dir), "n_episodes_on_disk": len(files),
                   "n_clips_with_join": n_clip_used,
                   "n_clips_without_join": n_no_join,
                   "n_windows": n_win},
        "class_histogram_seen": dict(cls_seen.most_common()),
        "n_classes_seen": len(cls_seen),
        "vacuity_gate": {
            "manoeuvre_rate": round(float(np.mean(turn_flags)), 6) if turn_flags else None,
            "n_turning_windows": int(np.sum(turn_flags)),
            "n_windows": len(turn_flags),
            "_reads": ("report this beside every constraint number: a zero "
                       "bought by declining the manoeuvre is not a safety "
                       "result"),
        },
        "censoring": {k: int(v) for k, v in censor.items()},
        "censoring_fractions": {
            "steps_without_kinematic_ceiling": round(
                censor["steps_no_kin"] / max(censor["steps_total"], 1), 4),
            "steps_without_clearance_ceiling": round(
                censor["steps_no_clr"] / max(censor["steps_total"], 1), 4),
            "steps_without_ANY_ceiling": round(
                censor["steps_no_ceiling"] / max(censor["steps_total"], 1), 4),
        },
        "distributions": {
            "clearance_m": dist(clearances, "clearance"),
            "kinematic_ceiling_ms": dist(kin_ceils, "kin"),
            "clearance_ceiling_ms": dist(clr_ceils, "clr"),
        },
        "arms": {a: summarise(a) for a in arms},
        "under_side_sensitivity": under_side_sweep(
            np.asarray(sweep_ceil), np.asarray(sweep_v),
            np.asarray(sweep_has_lead, dtype=bool)),
        "_runtime_s": None,
    }
    out["_runtime_s"] = round(time.time() - t_start, 1)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"clips {n_clip_used}/{len(files)} (no join: {n_no_join})  "
          f"windows {n_win}  classes {len(cls_seen)}")
    print(f"manoeuvre_rate {out['vacuity_gate']['manoeuvre_rate']}  "
          f"({out['vacuity_gate']['n_turning_windows']}/"
          f"{out['vacuity_gate']['n_windows']})")
    cf = out["censoring_fractions"]
    print(f"censored steps: no_kin {cf['steps_without_kinematic_ceiling']}  "
          f"no_clr {cf['steps_without_clearance_ceiling']}  "
          f"no_ANY {cf['steps_without_ANY_ceiling']}")
    d = out["distributions"]["clearance_m"]
    print(f"clearance m: n={d['n']} p05={d['p05']} p50={d['p50']} p95={d['p95']}")
    for a in arms:
        s = out["arms"][a]
        print(f"  {a:12s} over={s['frac_over_ceiling']} (n={s['n_over']}) "
              f"under_when_allowed={s['frac_under_when_allowed']} "
              f"(n={s['n_under']}/{s['n_situation_allows']})")
    print(f"wrote {out_json} in {out['_runtime_s']}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
