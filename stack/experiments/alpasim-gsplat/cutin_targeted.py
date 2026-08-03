#!/usr/bin/env python3
"""The CUT-IN test, re-powered — and its exposure clock corrected.

═══ WHY THIS FILE EXISTS ═══
`cutin_window_subset.py` isolated the cut-in windows and then said so itself: the subset
is UNDERPOWERED. Its ticks landed in 3 of 9 rollout-start clusters, so the paired
bootstrap had THREE resampling units and its lone separation (lateral_ade +0.1227) was
chance-level across 26 tests. Its author named the fix — put the rollout starts ON the
cut-in ticks. This file is that experiment, plus two defects found while building it.

── DEFECT 1: THE SELECTOR READ THE WRONG CLOCK, AND THE WRONG CLOCK IS THE POLICY'S ──
The predecessor selected a window as "cut-in" when `i_gt` — the nearest LOGGED ego index
to where the policy ACTUALLY DROVE — fell in the annotated cut-in tick set. But the actor
is drawn by TIME: `per_step_metrics` places it at `frac = (t_us - ts0)/(ts1 - ts0)`, and
the renderer looks the track up by timestamp. On this scene the arms do not track the
log, so the two clocks come apart — in OPPOSITE DIRECTIONS for the two arms.

  MEASURED on the banked scene2 rollouts (450 windows each):
      arm/condition            median i_gt - wall_tick   cut-in rows   clusters
      flagship-v1 / objects            +2.0                107 (i_gt)     3
      flagship-v1 / empty              +1.0                108 (i_gt)     3
      refc-base   / objects            -4.0                 99 (i_gt)     3
      refc-base   / empty              -3.0                101 (i_gt)     3
      ANY of the four, by WALL TICK      0                 129            5

  So the old selector (a) threw away 21-38 genuinely exposed rows and 2 whole clusters,
  and (b) — the serious half — chose a DIFFERENT WINDOW SET FOR EACH ARM AND EACH
  CONDITION, because the selector was a function of the treatment. A paired contrast whose
  pairing depends on the thing being tested is not a paired contrast.

The wall tick is EXOGENOUS: it is `argmin |gt.ts_us - step.t_us|`, a property of the
harness, identical for all four files by construction. It is the clock the renderer uses,
so it is the clock the exposure indicator must use.

── DEFECT 2: "44 CUT-IN EVENTS" IS 3 CROSSINGS, AND ONLY ONE IS RENDERABLE ──
`find_cutins` emits one record per approach tick, so a single lane entry becomes ~10-20
"events". Merged per track, scene 7c72937c has THREE distinct crossings:
      track  34 (id 21)  ticks  83- 92   NOT renderable
      track  65 (id 70)  ticks  79- 94   NOT renderable
      track 117 (id 30)  ticks 118-143   RENDERABLE
"21 renderable cut-in events" is ONE crossing. Every power statement has to say so.

This matters for the brief's own recipe. It asks for starts on "k=83-92, 118-143" — but
83-92 is track 34, whose cuboids are ABSENT from the dynamic_rigids layer. Starts placed
there buy annotation-labelled windows in which THAT vehicle is not drawn, so they dilute
the objects-vs-empty contrast rather than concentrate it. Only 118-143 is a treatment.

── DEFECT 3: A PER-STEP PAIRED CONTRAST IN CLOSED LOOP IS NOT AN EVENT RESPONSE ──
⚠️ I built the non-renderable epoch as a "placebo" and it SEPARATED on 12 of 28 metrics.
That is not evidence of a broken instrument — it is evidence that MY CONTROL WAS WRONG,
and the error is worth stating because it invalidates the obvious reading of the whole
panel. `objects` draws ALL 92 renderable agents at EVERY tick, not only during cut-ins.
So there is no tick at which the two conditions render the same pixels, and no epoch of
this design is a placebo.

Worse, the conditions diverge: once the ego has been displaced at tick 90 it is somewhere
else at tick 140, and every metric differs thereafter for reasons that have nothing to do
with the cut-in. A per-(start,k) contrast inside the exposure window therefore measures
ACCUMULATED divergence, not a response to the event.

The instrument that survives this is EVENT-ALIGNED DIFFERENCE-IN-DIFFERENCES on the
between-condition trajectory gap `d_AB(k) = ||ego_objects(k) - ego_empty(k)||`:

    did(start) = [d_AB(k_e + H) - d_AB(k_e)]/H   -   [d_AB(k_e) - d_AB(k_e - H)]/H

with k_e the step at which the entry tick lands in that rollout. Each start contributes
ONE number, the pre-window is that rollout's own control, and accumulated divergence
cancels to first order because it enters both terms. THAT is what the targeted starts are
for: they place k_e at a known, spread-out step in every rollout.

The only true instrument null available here is DETERMINISM — the same arm and condition
run twice must give exactly zero on every metric. It is run separately and reported first.

═══ WHAT IS REPORTED ═══
Negative controls FIRST, then the positive control, then the four families + ADE.
Every rate carries its precision and both denominators. Every family that cannot be
computed says so with its reason and its n, rather than vanishing from the table (which
is how the real-lead LONGITUDINAL rows disappeared from the predecessor's printout).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EGO_LEN = 4.7
LEAD_HALF_W = 2.0
LEAD_MAX_X = 80.0
POST_TICKS = 20          # 2.0 s reaction window after the entry completes
# MEASURED, from the panel's own rollout header (`f_eff`) and the scene camera block:
# the renderer draws 1920x1080 at 945.61 px/rad, both arms consume a 256x256 f-theta
# canonical crop at 266.17 px/rad. Apparent sizes below are in the MODEL's raster.
F_EFF_MODEL = 266.1706644208189
F_PX_PER_RAD_RENDER = 945.61
VEH_W_M, VEH_H_M = 1.8, 1.5      # nominal automobile box, for an apparent-size estimate
# k = steps since the rollout's first decision. At k=0 BOTH conditions sit at the same
# logged pose, so the early bands are the CAUSALLY CLEAN window: the only thing that
# differs is what was rendered. By the late bands the trajectories have separated and the
# contrast is measuring where they ended up, not what they decided.
K_BANDS = ((0, 5), (5, 13), (13, 26), (26, 38), (38, 50))
MAN_NAMES = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")
ROUTE_NAMES = ("left", "straight", "right", "unknown")


# --------------------------------------------------------------------------------- #
# the scene's own cut-in annotation, merged into DISTINCT crossings                    #
# --------------------------------------------------------------------------------- #
def crossings(geom_json):
    """Merge overlapping per-tick cut-in records into distinct lane entries."""
    g = json.loads(Path(geom_json).read_text())
    s = g.get("summary", g)
    by = {}
    for c in s.get("cutins_ALL", []):
        e = by.setdefault(int(c["track"]), {"track": int(c["track"]),
                                            "track_id": c.get("track_id"),
                                            "label": c.get("label"),
                                            "renderable": bool(c.get("renderable")),
                                            "k_start": int(c["k_start"]),
                                            "k_end": int(c["k_end"]),
                                            "headway_end_m": c.get("headway_end_m"),
                                            "time_gap_end_s": c.get("time_gap_end_s"),
                                            "area_frac_end": c.get("area_frac_end"),
                                            "n_raw_records": 0})
        e["k_start"] = min(e["k_start"], int(c["k_start"]))
        e["k_end"] = max(e["k_end"], int(c["k_end"]))
        e["n_raw_records"] += 1
    return sorted(by.values(), key=lambda e: e["k_start"]), s


def tickset(cross, renderable):
    t = set()
    for c in cross:
        if bool(c["renderable"]) != renderable:
            continue
        for k in range(c["k_start"], c["k_end"] + POST_TICKS + 1):
            t.add(k)
    return t


# --------------------------------------------------------------------------------- #
# the EXOGENOUS clock                                                                 #
# --------------------------------------------------------------------------------- #
def wall_tick_map(path):
    """(start_frame, k) -> wall tick, and (start_frame, k) -> ego xy.

    The wall tick is exogenous by construction: `t_us` is set by the harness clock (t0
    from the logged frame, +0.1 s per step), never by the policy. Asserted monotone below.
    """
    d = json.loads(Path(path).read_text())
    ts = np.asarray([g["ts_us"] for g in d["gt"]], float)
    wt, ego = {}, {}
    for r in d["rollouts"]:
        s = int(r["start_frame"])
        for st in r["steps"]:
            wt[(s, int(st["k"]))] = int(np.argmin(np.abs(ts - float(st["t_us"]))))
            ego[(s, int(st["k"]))] = (float(st["ego"][0]), float(st["ego"][1]))
    return wt, ego, d


# --------------------------------------------------------------------------------- #
# EVENT-ALIGNED DIFFERENCE-IN-DIFFERENCES on the between-condition trajectory gap      #
# --------------------------------------------------------------------------------- #
def divergence_did(egoA, egoB, wt, entry_tick, H=10):
    """Does the objects/empty trajectory gap grow FASTER once the cut-in is visible?

    `d_AB(k) = ||ego_A(k) - ego_B(k)||` is zero at k=0 by construction (both rollouts
    start from the same logged pose) and grows as the two conditions drive apart. A
    per-step contrast inside the exposure window cannot separate "reacted to the cut-in"
    from "was already somewhere else"; this can, because each rollout's own PRE-entry
    growth is the control for its POST-entry growth, and the accumulated offset enters
    both terms and cancels to first order.

    One number per rollout start => the bootstrap resamples starts, which is the only
    level at which these data are even arguably exchangeable.
    """
    per = {}
    for (s, k), t in wt.items():
        per.setdefault(s, {})[k] = t
    rows, curves = [], {}
    for s, kk in sorted(per.items()):
        inv = {t: k for k, t in kk.items()}
        if entry_tick not in inv:
            continue
        ke = inv[entry_tick]
        kmax = max(kk)
        if ke - H < 0 or ke + H > kmax:
            rows.append({"start": s, "k_entry": ke, "usable": False,
                         "reason": f"needs k_entry in [{H}, {kmax - H}]"})
            continue
        d = lambda k: float(np.hypot(egoA[(s, k)][0] - egoB[(s, k)][0],
                                     egoA[(s, k)][1] - egoB[(s, k)][1]))
        pre = (d(ke) - d(ke - H)) / H
        post = (d(ke + H) - d(ke)) / H
        rows.append({"start": s, "k_entry": ke, "usable": True,
                     "d_at_entry_m": round(d(ke), 4),
                     "pre_growth_m_per_step": round(pre, 5),
                     "post_growth_m_per_step": round(post, 5),
                     "did_m_per_step": round(post - pre, 5)})
        curves[s] = {str(j): round(d(ke + j), 4) for j in range(-H, H + 1)}
    use = [r for r in rows if r.get("usable")]
    out = {"H_steps": H, "entry_tick": entry_tick, "per_start": rows,
           "n_usable_starts": len(use),
           "gap_curve_aligned_on_entry_m": curves}
    if len(use) >= 3:
        from taniteval.ci import episode_cluster_bootstrap, paired_episode_cluster_bootstrap
        v = [r["did_m_per_step"] for r in use]
        eid = [r["start"] for r in use]
        r = episode_cluster_bootstrap(np.asarray(v, float), eid)
        r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
        out["DID_mean_m_per_step"] = r
        p = paired_episode_cluster_bootstrap(
            np.asarray([x["post_growth_m_per_step"] for x in use], float),
            np.asarray([x["pre_growth_m_per_step"] for x in use], float), eid)
        p["separated"] = bool(p["lo"] > 0 or p["hi"] < 0)
        out["paired_post_minus_pre"] = p
    else:
        out["DID_mean_m_per_step"] = {
            "n": len(use),
            "reason": "fewer than 3 usable starts — this design cannot carry a CI"}
    return out


def assert_geometry(wt, label, expect_step1=True):
    """Geometry is ASSERTED, never assumed: the wall clock must advance one tick per
    step inside every rollout, or the exposure indicator is not a clock at all."""
    bad = []
    per = {}
    for (s, k), t in wt.items():
        per.setdefault(s, {})[k] = t
    for s, kk in per.items():
        ks = sorted(kk)
        d = np.diff([kk[k] for k in ks])
        if expect_step1 and len(d) and not np.all(d == 1):
            bad.append({"start": s, "bad_steps": int((d != 1).sum()),
                        "unique_diffs": sorted(set(int(x) for x in d))})
    return {"label": label, "n_rollouts": len(per), "monotone_step1": not bad,
            "violations": bad[:5]}


# --------------------------------------------------------------------------------- #
def cutin_actor_geometry(row, track_idx):
    """Ego-frame geometry of THE cut-in vehicle itself, not of "the lead".

    The predecessor's LONGITUDINAL family scored `real_lead_*`, which is whichever
    renderable agent happens to be nearest in-lane. During a cut-in the vehicle of
    interest is by definition NOT yet in-lane, so the metric that is supposed to measure
    the response to the cut-in was, for most of the entry, measuring a different car —
    or nothing at all. These fields follow track `track_idx` specifically.
    """
    a = None
    for x in row.get("actors", ()):
        if int(x["idx"]) == track_idx:
            a = x
            break
    if a is None:
        return {"cutin_present": 0.0, "cutin_x_m": np.nan, "cutin_abs_y_m": np.nan,
                "cutin_headway_m": np.nan, "cutin_time_gap_s": np.nan,
                "cutin_ttc_s": np.nan, "cutin_inlane": np.nan,
                "cutin_ahead_infov": 0.0}
    x, y = float(a["rig"][0]), float(a["rig"][1])
    ahead = x > 0 and x < LEAD_MAX_X
    hw = (x - EGO_LEN) if ahead else np.nan
    v_ego = float(row["v_ego"])
    tg = (hw / max(v_ego, 0.1)) if np.isfinite(hw) else np.nan
    ttc = np.nan
    vl = a.get("v")
    if np.isfinite(hw) and hw > 0 and vl is not None:
        closing = v_ego - float(vl)
        if closing > 0.1:
            ttc = hw / closing
    # ⛔ NEVER SCORE AN ARM OFF ITS RASTER. A null is only a statement about the policy
    # if the stimulus was actually present in the pixels the policy SEES. The scene is
    # rendered at 1920x1080 with f = 945.61 px/rad, but both arms consume a 256x256
    # f-theta canonical crop whose measured edge-referenced focal length is
    # f_eff = 266.17 px/rad — a 3.55x linear reduction, 12.6x in area. So the apparent
    # size is computed in the MODEL'S OWN RASTER, not the renderer's.
    d = float(np.hypot(x, y))
    apx = ((VEH_W_M / d) * F_EFF_MODEL) if d > 0.5 else np.nan
    apy = ((VEH_H_M / d) * F_EFF_MODEL) if d > 0.5 else np.nan
    return {"cutin_present": 1.0, "cutin_x_m": x, "cutin_abs_y_m": abs(y),
            "cutin_headway_m": hw, "cutin_time_gap_s": tg, "cutin_ttc_s": ttc,
            "cutin_inlane": float(abs(y) <= LEAD_HALF_W),
            "cutin_ahead_infov": float(ahead),
            "cutin_w_px_model": apx, "cutin_h_px_model": apy,
            "cutin_area_px_model": (apx * apy if np.isfinite(apx) else np.nan)}


# --------------------------------------------------------------------------------- #
# metric table — four families + ADE, exactly as the binding rule requires             #
# --------------------------------------------------------------------------------- #
def metric_table(cutin_track):
    G = cutin_actor_geometry
    return (
        ("ADE", "ade_0_2s", lambda r: r["ade"]),
        ("ADE", "de_2s", lambda r: r["de"][-1]),
        ("ADE", "dist_to_gt_traj_m", lambda r: r["dist_to_gt"]),
        # ---- LONGITUDINAL: speed AND distance-keeping ----
        ("LONGITUDINAL", "abs_target_speed_err_ms", lambda r: abs(r["speed_err"])),
        ("LONGITUDINAL", "target_speed_err_ms", lambda r: r["speed_err"]),
        ("LONGITUDINAL", "executed_speed_err_ms", lambda r: r["speed_track_err"]),
        ("LONGITUDINAL", "along_track_ade_m", lambda r: r["lon_ade"]),
        ("LONGITUDINAL", "v_target_ms", lambda r: r["v_target"]),
        ("LONGITUDINAL", "real_lead_headway_m", lambda r: _f(r["headway"])),
        ("LONGITUDINAL", "real_lead_time_gap_s", lambda r: _f(r["time_gap"])),
        ("LONGITUDINAL", "real_lead_ttc_s_when_closing", lambda r: _f(r.get("ttc"))),
        ("LONGITUDINAL", "real_lead_frac_tg_below_1s",
         lambda r: (float(r["time_gap"] < 1.0) if r["time_gap"] is not None else np.nan)),
        # distance-keeping TO THE CUT-IN VEHICLE ITSELF
        ("LONGITUDINAL", "cutin_headway_m",
         lambda r: G(r, cutin_track)["cutin_headway_m"]),
        ("LONGITUDINAL", "cutin_time_gap_s",
         lambda r: G(r, cutin_track)["cutin_time_gap_s"]),
        ("LONGITUDINAL", "cutin_ttc_s_when_closing",
         lambda r: G(r, cutin_track)["cutin_ttc_s"]),
        ("LONGITUDINAL", "cutin_apparent_w_px_model_raster",
         lambda r: G(r, cutin_track)["cutin_w_px_model"]),
        # ---- LATERAL ----
        ("LATERAL", "heading_err_rad", lambda r: abs(r["heading_err"])),
        ("LATERAL", "curvature_err_1pm", lambda r: r["curv_err"]),
        ("LATERAL", "yawrate_err_rads", lambda r: r["yawrate_err"]),
        ("LATERAL", "cross_track_abs_m", lambda r: abs(r["cross_track"])),
        ("LATERAL", "lateral_ade_m", lambda r: r["lat_ade"]),
        # ---- TACTICAL ----
        ("TACTICAL", "manoeuvre_plan_eq_logged",
         lambda r: float(r["man_plan"] == r["man_gt"])),
        ("TACTICAL", "manoeuvre_head_eq_logged",
         lambda r: (float(r["man_head"] == r["man_gt"])
                    if r["man_head"] is not None else np.nan)),
        ("TACTICAL", "manoeuvre_head_eq_plan",
         lambda r: (float(r["man_head"] == r["man_plan"])
                    if r["man_head"] is not None else np.nan)),
        ("TACTICAL", "plan_is_brake_stop", lambda r: float(r["man_plan"] == 4)),
        ("TACTICAL", "plan_is_accelerate", lambda r: float(r["man_plan"] == 3)),
        # ---- STRATEGIC ----
        ("STRATEGIC", "route_corridor_departure_rate",
         lambda r: float(r["corridor_departure"])),
        ("STRATEGIC", "route_head_eq_logged",
         lambda r: (float(r["route_head"] == r["route_gt"])
                    if (r["route_head"] is not None and r["route_valid"]) else np.nan)),
        ("STRATEGIC", "route_gt_valid_rate", lambda r: float(bool(r["route_valid"]))),
    )


def _f(v):
    return np.nan if v is None else float(v)


# --------------------------------------------------------------------------------- #
def contrast(KA, KB, keys, table, M):
    """Paired episode-cluster bootstrap on identical (start, k) windows."""
    if not keys:
        return {"n": 0, "reason": "no paired window in this stratum"}
    pe = [c[0] for c in keys]
    out = {}
    for fam, lab, fn in table:
        va = [_f(fn(KA[c])) for c in keys]
        vb = [_f(fn(KB[c])) for c in keys]
        ok = int(np.sum(np.isfinite(va) & np.isfinite(vb)))
        if ok == 0:
            out[lab] = {"family": fam, "n": 0, "n_used": 0,
                        "reason": "metric undefined on every window in this stratum "
                                  "(e.g. no renderable agent in the in-lane search band)"}
            continue
        r = M._paired(va, vb, pe)
        r["family"] = fam
        r["mean_A"] = round(float(np.nanmean(va)), 5)
        r["mean_B"] = round(float(np.nanmean(vb)), 5)
        r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
        out[lab] = r
    n_t = sum(1 for v in out.values() if v.get("n_used"))
    n_s = sum(1 for v in out.values() if v.get("separated"))
    out["_summary"] = {"n_metrics_testable": n_t, "n_separated": n_s,
                       "expected_false_positives_at_alpha_0_05": round(0.05 * n_t, 2),
                       "n_windows": len(keys),
                       "n_clusters": len({c[0] for c in keys})}
    return out


def exposure_label(verdict_path):
    """Make the curve-invariant verdict TRAVEL WITH the numbers.

    Without this the table is a well-powered contrast on a window set that a reader will
    call "the cut-in windows" because that is what the starts were chosen for — which is
    exactly how a refuted premise gets re-quoted from a results file.
    """
    if not verdict_path or not Path(verdict_path).exists():
        return ("NOT SUPPLIED — pass --cutin-verdict CUTIN_IS_REAL.json. Without it this "
                "table does not carry the evidence about what its window contains.")
    v = json.loads(Path(verdict_path).read_text())
    n_ok = v.get("n_surviving_the_curve_invariant_test")
    return {
        "n_distinct_crossings": v.get("n_distinct_crossings_after_merge"),
        "n_surviving_curve_invariant_test": n_ok,
        "per_crossing_verdict": [{"track": x["track"],
                                  "renderable": x.get("renderable"),
                                  "IS_A_REAL_CUT_IN": x.get("IS_A_REAL_CUT_IN"),
                                  "verdict": x.get("verdict")}
                                 for x in v.get("per_crossing", [])],
        "THEREFORE": (
            "⛔ the EXPOSURE stratum below is NOT a cut-in stratum. NO annotated crossing "
            "in this scene survives the curve-invariant test. What the stratum actually "
            "contains is the highest-agent-exposure window of the clip: the ego turning "
            "~95 deg through a junction with the vehicle it has followed since tick 0 "
            "at 19-34 m ahead, and all 92 renderable agents drawn. Every contrast below "
            "is a valid objects-vs-empty test ON THAT WINDOW and must not be quoted as a "
            "cut-in result." if n_ok == 0 else
            f"{n_ok} annotated crossing(s) survive the curve-invariant test."),
    }


def nav_echo_guard(K, keys):
    """Is the route head just the nav command the harness FED the policy?

    Carried forward from `cl_metrics.families`: on the scene-2 panel flagship-v1's route
    head was a deterministic bijection of `nav` (nav=1 -> head=0 on 369/369, nav=0 ->
    head=1 on 81/81), and it scored route_head_eq_logged 1.0000 [1.0000, 1.0000]. That is
    the echo of its own conditioning input, not strategic skill. The guard must travel
    with the STRATEGIC family wherever the family is reported, or the same number gets
    re-quoted in a new table.
    """
    rows = [K[c] for c in keys]
    if not rows or not all(r.get("nav") is not None for r in rows):
        return {"verdict": "not evaluated (nav missing on at least one window)"}
    m = {}
    for r in rows:
        m.setdefault(int(r["nav"]), set()).add(r["route_head"])
    n_nav = len(m)
    circular = all(len(v) == 1 for v in m.values()) if n_nav >= 2 else None
    if circular:
        v = ("CIRCULAR — route_head_eq_logged reproduces the nav command the policy was "
             "GIVEN; do_not_quote")
    elif circular is False:
        v = "not an echo — the head is not a function of nav on these windows"
    else:
        v = (f"UNIDENTIFIABLE — nav takes only {n_nav} distinct value(s) on these "
             f"{len(rows)} windows; an echo cannot be told from a constant head. NOT a "
             "clearance.")
    return {"head_is_deterministic_function_of_nav": circular, "n_distinct_nav": n_nav,
            "nav_to_head_map": {str(k): sorted(x for x in s if x is not None)
                                for k, s in m.items()},
            "n": len(rows), "verdict": v,
            "do_not_quote_route_head_eq_logged": bool(circular)}


def class_pr(K, keys, M):
    """Precision AND recall for every decision head, on this stratum."""
    out = {}
    mh = [(K[c]["man_gt"], K[c]["man_head"]) for c in keys
          if K[c]["man_head"] is not None]
    if mh:
        out["manoeuvre_head_vs_logged"] = M.per_class_pr(
            [a for a, _ in mh], [b for _, b in mh], MAN_NAMES)
    mp = [(K[c]["man_gt"], K[c]["man_plan"]) for c in keys]
    if mp:
        out["manoeuvre_plan_vs_logged"] = M.per_class_pr(
            [a for a, _ in mp], [b for _, b in mp], MAN_NAMES)
    rh = [(K[c]["route_gt"], K[c]["route_head"]) for c in keys
          if K[c]["route_head"] is not None and K[c]["route_valid"]]
    out["route_head_vs_logged"] = (M.per_class_pr([a for a, _ in rh],
                                                  [b for _, b in rh], ROUTE_NAMES)
                                   if rh else
                                   {"_n": 0, "reason": "no window has BOTH a route head "
                                                       "and a valid logged route"})
    return out


# --------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="condition A rollouts json (treated)")
    ap.add_argument("--b", required=True, help="condition B rollouts json (control)")
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--renderable-from", required=True)
    ap.add_argument("--cutin-track", type=int, default=None,
                    help="track index of the RENDERABLE crossing; default = infer")
    ap.add_argument("--track-hash", default=None,
                    help="md5 of the renderer used, recorded for provenance")
    ap.add_argument("--label", default="A_minus_B")
    ap.add_argument("--cutin-verdict", default=None,
                    help="CUTIN_IS_REAL.json from cutin_is_real.py. Embedded so the "
                         "curve-invariant verdict TRAVELS WITH the numbers and nobody "
                         "re-reads this table as a cut-in result.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cl_metrics as M
    from gsplat_renderer import ActorTracks

    cross, summ = crossings(a.geometry)
    rend = [c for c in cross if c["renderable"]]
    nonr = [c for c in cross if not c["renderable"]]
    if a.cutin_track is None:
        if len(rend) != 1:
            raise SystemExit(f"expected exactly 1 renderable crossing, got {len(rend)} "
                             "— pass --cutin-track explicitly")
        a.cutin_track = rend[0]["track"]

    T_EXP = tickset(cross, True)          # renderable  -> a real treatment
    T_SHAM = tickset(cross, False)        # annotated but NOT rendered -> placebo
    T_SHAM = {t for t in T_SHAM if t not in T_EXP}

    tr = ActorTracks(a.tracks)
    am = json.loads(Path(a.renderable_from).read_text())
    keep = {int(x["best_track"]) for x in am["per_track"] if x["accepted"]}
    if a.cutin_track not in keep:
        raise SystemExit(f"track {a.cutin_track} is called renderable by the geometry "
                         f"but is NOT in the accepted actor map — refusing to score")

    wtA, egoA, dAraw = wall_tick_map(a.a)
    wtB, egoB, dBraw = wall_tick_map(a.b)
    geomA = assert_geometry(wtA, "A")
    geomB = assert_geometry(wtB, "B")
    if not (geomA["monotone_step1"] and geomB["monotone_step1"]):
        raise SystemExit("wall clock is not step-1 monotone — "
                         + json.dumps([geomA, geomB]))

    dA, rA, eA, sA = M.collect(a.a, tr, lead_ref=None, keep_tracks=keep)
    dB, rB, eB, sB = M.collect(a.b, tr, lead_ref=None, keep_tracks=keep)
    KA = {(int(e), int(r["k"])): r for r, e in zip(rA, eA)}
    KB = {(int(e), int(r["k"])): r for r, e in zip(rB, eB)}
    common = sorted(set(KA) & set(KB))

    wall = {c: wtA[c] for c in common}
    assert all(wtA[c] == wtB[c] for c in common), \
        "the wall clock differs between conditions — it is not exogenous"

    exp = [c for c in common if wall[c] in T_EXP]
    sham = [c for c in common if wall[c] in T_SHAM]
    rest = [c for c in common if wall[c] not in T_EXP and wall[c] not in T_SHAM]
    # what the OLD, policy-dependent selector would have picked, for the record
    old_A = [c for c in common if int(KA[c]["i_gt"]) in T_EXP]
    old_B = [c for c in common if int(KB[c]["i_gt"]) in T_EXP]

    table = metric_table(a.cutin_track)

    # ---- MANIPULATION CHECK: was the treatment actually delivered? ----------------
    def manip(K, keys, tag):
        if not keys:
            return {"n": 0, "reason": "empty stratum"}
        g = [cutin_actor_geometry(K[c], a.cutin_track) for c in keys]
        pres = np.array([x["cutin_present"] for x in g])
        ahead = np.array([x["cutin_ahead_infov"] for x in g])
        inl = np.array([x["cutin_inlane"] for x in g], float)
        hw = np.array([x["cutin_headway_m"] for x in g], float)
        aw = np.array([x["cutin_w_px_model"] for x in g], float)
        ah = np.array([x["cutin_h_px_model"] for x in g], float)
        q = lambda v, f: (round(float(f(v[np.isfinite(v)])), 3)
                          if np.isfinite(v).any() else None)
        return {"tag": tag, "n": len(keys),
                "frac_cutin_track_alive": round(float(pres.mean()), 4),
                "frac_cutin_ahead_and_within_80m": round(float(ahead.mean()), 4),
                "frac_cutin_in_ego_lane": (round(float(np.nanmean(inl)), 4)
                                           if np.isfinite(inl).any() else None),
                "median_headway_to_cutin_m": q(hw, np.median),
                "min_headway_to_cutin_m": q(hw, np.min),
                "STIMULUS_in_model_raster_256px": {
                    "f_eff_px_per_rad": F_EFF_MODEL,
                    "median_w_px": q(aw, np.median), "median_h_px": q(ah, np.median),
                    "max_w_px": q(aw, np.max), "max_h_px": q(ah, np.max),
                    "min_w_px": q(aw, np.min),
                    "note": "nominal 1.8 x 1.5 m box at the measured range, projected at "
                            "the arms' own f_eff. This is the question a null has to "
                            "survive: was the cut-in vehicle big enough to see AT ALL in "
                            "the 256x256 the policy actually consumes?"}}

    # ---- TRACKING stratum: is the arm anywhere near the log? ----------------------
    # ⚠️ cross-track is POST-treatment, so this stratum is conditional on an outcome and
    # cannot carry a causal reading. It is reported to answer one question only: "could a
    # null be masked by the arm having driven away from the scene?" The rule is SYMMETRIC
    # (both conditions tracking) so the pairing survives.
    def tracking_keys(keys, thr):
        return [c for c in keys
                if abs(KA[c]["cross_track"]) <= thr and abs(KB[c]["cross_track"]) <= thr]

    res = {
        "what": f"CUT-IN TARGETED PANEL — {a.label}",
        "evidence_class": "MEASURED (ours)",
        "A": {"file": str(a.a), "arm": dA["arm"], "condition": dA["condition"],
              "ckpt": dA.get("ckpt")},
        "B": {"file": str(a.b), "arm": dB["arm"], "condition": dB["condition"],
              "ckpt": dB.get("ckpt")},
        "renderer_md5_on_pod": a.track_hash,
        "⛔_WHAT_THE_EXPOSURE_WINDOW_ACTUALLY_CONTAINS": exposure_label(a.cutin_verdict),
        "scene_cutin_structure": {
            "headline_in_geometry_file": {
                "n_cutin_events_ALL": summ.get("n_cutin_events_ALL"),
                "n_cutin_events_RENDERABLE": summ.get("n_cutin_events_RENDERABLE")},
            "DISTINCT_crossings_after_merge": cross,
            "note": "find_cutins emits one record per approach tick, so the headline "
                    "counts approach ticks, not lane entries. THREE crossings exist; "
                    "ONE is renderable.",
        },
        "exposure_clock": {
            "used": "wall tick = argmin |gt.ts_us - step.t_us| (EXOGENOUS)",
            "geometry_assert_A": geomA, "geometry_assert_B": geomB,
            "n_exposure_ticks_renderable": len(T_EXP),
            "exposure_tick_range": [min(T_EXP), max(T_EXP)] if T_EXP else None,
            "n_sham_ticks_nonrenderable": len(T_SHAM),
            "sham_tick_range": [min(T_SHAM), max(T_SHAM)] if T_SHAM else None,
            "OLD_selector_i_gt_would_pick": {
                "n_rows_using_A_i_gt": len(old_A), "n_clusters_A": len({c[0] for c in old_A}),
                "n_rows_using_B_i_gt": len(old_B), "n_clusters_B": len({c[0] for c in old_B}),
                "disagreement_rows": len(set(old_A) ^ set(old_B)),
                "note": "the old selector is a function of where the POLICY drove, so it "
                        "picks a different window set per arm and per condition.",
            },
        },
        "denominator": {
            "n_all_paired_windows": len(common),
            "n_exposure_windows": len(exp),
            "n_sham_windows": len(sham),
            "n_other_windows": len(rest),
            "n_clusters_all": len({c[0] for c in common}),
            "n_clusters_exposure": len({c[0] for c in exp}),
            "n_clusters_sham": len({c[0] for c in sham}),
            "starts": sorted({c[0] for c in common}),
            "n_rows_A_before_pairing": len(rA), "n_rows_B_before_pairing": len(rB),
            "dropped_by_truncation_or_pairing_A": len(rA) - len(common),
            "dropped_by_truncation_or_pairing_B": len(rB) - len(common),
        },
        "estimator": "paired episode-cluster bootstrap over rollout starts "
                     "(taniteval.ci.paired_episode_cluster_bootstrap, 2000 resamples). "
                     "⚠️ the clusters are disjoint-START segments of ONE 20 s clip whose "
                     "windows OVERLAP in the exposure region; they are not independent "
                     "episodes and the CI is optimistic to that extent.",
        # ---------------- CONTROLS FIRST ----------------
        "CONTROL_1_determinism": {
            "what": "run this script with --a X --b X_REPEAT (same arm, same condition, "
                    "same starts, second run). That is the ONLY true instrument null in "
                    "this design: every metric must come out exactly 0.",
            "is_this_run_the_determinism_control": bool(dA["arm"] == dB["arm"]
                                                        and dA["condition"] == dB["condition"]),
        },
        "CONTROL_2_matched_non_cutin_epoch": {
            "definition": "ticks of the two NON-RENDERABLE crossings (tracks "
                          + ",".join(str(c["track"]) for c in nonr) + ").",
            "⚠️_THIS_IS_NOT_A_PLACEBO": "I first built this as one and it separated on "
                                        "12/28 metrics. The reason is not a broken "
                                        "instrument: `objects` draws ALL 92 renderable "
                                        "agents at EVERY tick, so no epoch of this design "
                                        "renders the same pixels in both conditions. Read "
                                        "it as 'the same contrast on a matched epoch "
                                        "WITHOUT the renderable crossing' — it answers "
                                        "whether any effect is cut-in-SPECIFIC, not "
                                        "whether the instrument is sound.",
            "manipulation_check_A": manip(KA, sham, "A/non-cutin epoch"),
            "result": contrast(KA, KB, sham, table, M),
        },
        "CONTROL_3_event_aligned_DiD": divergence_did(
            egoA, egoB, {c: wall[c] for c in common},
            entry_tick=(rend[0]["k_start"] if rend else None)) if rend else
            {"reason": "no renderable crossing"},
        "POSITIVE_CONTROL_manipulation": {
            "A_exposure": manip(KA, exp, "A/exposure"),
            "B_exposure": manip(KB, exp, "B/exposure"),
            "renderable_crossing_projected_area_frac_at_entry_end":
                rend[0].get("area_frac_end") if rend else None,
            "note": "`area_frac_end` is the crossing vehicle's projected share of the "
                    "1920x1080 frame at the end of its entry — the size of the stimulus.",
        },
        # ---------------- the experiment ----------------
        "EXPOSURE_windows": contrast(KA, KB, exp, table, M),
        "ALL_windows": contrast(KA, KB, common, table, M),
        "OTHER_windows_no_cutin": contrast(KA, KB, rest, table, M),
        "EXPOSURE_windows_TRACKING_ONLY_2m": {
            "caveat": "cross-track is POST-treatment; this stratum answers 'is the null "
                      "an artefact of the arm having left the scene', not a causal "
                      "question. Rule is symmetric (BOTH conditions within threshold).",
            "n_windows": len(tracking_keys(exp, 2.0)),
            "n_clusters": len({c[0] for c in tracking_keys(exp, 2.0)}),
            "result": contrast(KA, KB, tracking_keys(exp, 2.0), table, M),
        },
        "EXPOSURE_windows_TRACKING_ONLY_5m": {
            "n_windows": len(tracking_keys(exp, 5.0)),
            "n_clusters": len({c[0] for c in tracking_keys(exp, 5.0)}),
            "result": contrast(KA, KB, tracking_keys(exp, 5.0), table, M),
        },
        # ⚠️ THE EPOCH SPLIT IS CONFOUNDED WITH ELAPSED CLOSED-LOOP TIME.
        # The exposure epoch (ticks 118-163) sits at LARGER k than the matched epoch
        # (79-117) in most rollouts, because the starts were placed to bracket tick 118.
        # So "the sign flips between epochs" and "the sign flips as divergence
        # accumulates" predict the same table. These strata separate them: k-bands cut
        # across BOTH epochs, and the targeted starts deliberately put the same tick at
        # 12 different k, which is the only reason the two are separable at all here.
        "BY_K_BAND_all_windows": {
            f"k_{lo}_{hi}": contrast(KA, KB,
                                     [c for c in common if lo <= c[1] < hi], table, M)
            for lo, hi in K_BANDS},
        "BY_K_BAND_exposure_only": {
            f"k_{lo}_{hi}": contrast(KA, KB,
                                     [c for c in exp if lo <= c[1] < hi], table, M)
            for lo, hi in K_BANDS},
        "PRECISION_WITH_RECALL": {
            "A_exposure": class_pr(KA, exp, M),
            "B_exposure": class_pr(KB, exp, M),
        },
        "STRATEGIC_nav_echo_guard": {
            "A_exposure": nav_echo_guard(KA, exp),
            "B_exposure": nav_echo_guard(KB, exp),
        },
        "progress_vs_log": {
            "A": [{"start": s["start"], "driven_m": round(s["driven_m"], 2),
                   "gt_dist_m": round(s["gt_dist_m"], 2),
                   "progress_rel": (round(s["progress_rel"], 3)
                                    if s.get("progress_rel") else None)} for s in sA],
            "B": [{"start": s["start"], "driven_m": round(s["driven_m"], 2),
                   "gt_dist_m": round(s["gt_dist_m"], 2),
                   "progress_rel": (round(s["progress_rel"], 3)
                                    if s.get("progress_rel") else None)} for s in sB],
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))

    # ------------------------------- console -------------------------------------- #
    def show(title, block):
        print(f"\n--- {title} ---")
        if not isinstance(block, dict) or block.get("n") == 0:
            print("   ", block.get("reason", block))
            return
        for lab, v in block.items():
            if lab == "_summary" or not isinstance(v, dict):
                continue
            if not v.get("n_used"):
                print("    %-13s %-32s  n=0  %s" % (v.get("family", "?"), lab,
                                                    v.get("reason", "")))
                continue
            print("    %-13s %-32s d=%+.4f [%+.4f,%+.4f] n=%-4d %s"
                  % (v["family"], lab, v["delta"], v["lo"], v["hi"], v["n_used"],
                     "*SEP*" if v["separated"] else ""))
        s = block.get("_summary", {})
        print("    -> %s/%s separated, %s expected by chance, %s windows in %s clusters"
              % (s.get("n_separated"), s.get("n_metrics_testable"),
                 s.get("expected_false_positives_at_alpha_0_05"),
                 s.get("n_windows"), s.get("n_clusters")))

    print(json.dumps(res["denominator"], indent=1))
    print("\nMANIPULATION CHECK  A:", json.dumps(res["POSITIVE_CONTROL_manipulation"]["A_exposure"]))
    print("MANIPULATION CHECK  B:", json.dumps(res["POSITIVE_CONTROL_manipulation"]["B_exposure"]))
    did = res["CONTROL_3_event_aligned_DiD"]
    if isinstance(did, dict) and "DID_mean_m_per_step" in did:
        print("\n--- EVENT-ALIGNED DiD on the objects-vs-empty trajectory gap ---")
        for r in did["per_start"]:
            if r.get("usable"):
                print("    start %3d  k_entry %2d  gap@entry %7.3f m  pre %+.5f  "
                      "post %+.5f  DiD %+.5f m/step"
                      % (r["start"], r["k_entry"], r["d_at_entry_m"],
                         r["pre_growth_m_per_step"], r["post_growth_m_per_step"],
                         r["did_m_per_step"]))
            else:
                print("    start %3d  k_entry %2d  UNUSABLE (%s)"
                      % (r["start"], r["k_entry"], r.get("reason")))
        d0 = did["DID_mean_m_per_step"]
        if "delta" in d0 or "mean" in d0:
            c = d0.get("delta", d0.get("mean"))
            print("    DiD mean %+.5f m/step [%+.5f,%+.5f]  n_starts=%d  %s"
                  % (c, d0["lo"], d0["hi"], did["n_usable_starts"],
                     "*SEP*" if d0.get("separated") else "NOT separated"))
    show("CONTROL 2 — matched epoch WITHOUT the renderable crossing (NOT a placebo)",
         res["CONTROL_2_matched_non_cutin_epoch"]["result"])
    show("EXPOSURE windows (the renderable crossing)", res["EXPOSURE_windows"])
    show("EXPOSURE windows, tracking <= 2 m", res["EXPOSURE_windows_TRACKING_ONLY_2m"]["result"])
    show("ALL windows", res["ALL_windows"])
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
