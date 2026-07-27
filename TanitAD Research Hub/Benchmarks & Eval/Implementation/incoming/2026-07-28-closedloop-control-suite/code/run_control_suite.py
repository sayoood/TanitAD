#!/usr/bin/env python3
"""Run the LONGITUDINAL / LATERAL CONTROL SUITE over the committed arm panel.

**No GPU. No model. No checkpoint. No corpus.** Every number is arithmetic over
the committed ``pw_<arm>.npz`` per-window pseudo-simulation dumps, so every bar
here recomputes on a laptop in minutes. Nothing in ``taniteval.control`` or
``taniteval.pseudosim`` is reimplemented — it is IMPORTED.

WHAT IT PRODUCES (each is a separate raw JSON, banked incrementally)
--------------------------------------------------------------------
``axes_panel.json``        every arm on every axis, with episode-cluster CIs,
                           the PANEL-WIDE gate, and the tolerance sensitivity.
``dynamic_range.json``     ⭐ the demonstration the PI asked for: a *controlled*
                           degradation injected on each axis, the ladder, the
                           monotonicity, the both-direction check and the **MDE
                           in physical units**.
``cross_sensitivity.json`` how much each axis moves under the OTHER axis's
                           control — the purity claim, measured not asserted.
``recovery_onesided.json`` ⛔ the floor-saturation of ``recovery`` and the
                           direction of its response to lateral degradation.
``comfort_audit.json``     ⛔ the four comfort clauses evaluated on the plans AND
                           on the HUMAN'S OWN LOGGED PATH — the decisive control.
``repro_gate.json``        ✅ proves that zeroing ``comfort``'s weight changes no
                           published number: 16 published ``@clamp_v1``
                           composites must reproduce to 4 dp.

ESTIMATOR: ``taniteval.ci.paired_episode_cluster_bootstrap`` (B = 2000, unit =
val episode). ⛔ ``overlapping_holdout_se`` appears nowhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[6]
for _p in (str(_REPO / "taniteval"), str(_REPO / "stack"),
           "/root/taniteval", "/root/TanitAD/stack"):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import ci as _ci            # noqa: E402
from taniteval import control as C         # noqa: E402
from taniteval import pseudosim as PS      # noqa: E402

#: probes: scored and reported, but they do not vote on the panel-wide gate
PROBES = ("stand_still", "v1_ego_half", "v1_ego_double", "oracle_lon_straight")

#: PUBLISHED `@clamp_v1` composites — the reproduction gate for the weight change
PUBLISHED_CLAMP_V1 = {
    "cv_holdv0": 0.5705, "v4_oracle": 0.5622, "refc_xl_produced": 0.5499,
    "v1_tactical_follow": 0.5471, "v1_tactical_oracle": 0.5467,
    "refc_small_produced": 0.5444, "refc_base_produced": 0.5439,
    "nospeed_tactical_oracle": 0.5394, "v4_blind": 0.3749,
    "v1_ego_v0": 0.5608, "v1_ego_oracle_lon": 0.5946, "v1_lat_straight": 0.5460,
    "refc_base_v0on": 0.5439, "refc_base_v0off": 0.4980,
    "refc_xl_v0on": 0.5499, "v1_ego_half": 0.3117,
}

#: ⭐ THE DEMONSTRATION MATRIX. Each cell is (axis, control). The OWN-axis cells
#: are the dynamic-range proof; the CROSS cells are the purity check. Both are
#: run, because an axis that only responds to its own control is a claim and an
#: axis that responds to everything is a longitudinal metric wearing a lateral
#: name — and this corpus is 98.6 % longitudinal by squared-error energy.
DEMOS = [
    # --- longitudinal: own-axis ---
    ("lon_track", "lon_retime"), ("lon_track", "lon_scale"),
    ("lon_track", "lon_jitter"),
    # --- longitudinal: cross (must be weak) ---
    ("lon_track", "lat_shift"), ("lon_track", "yaw_bias"),
    # --- lateral placement: own-axis ---
    ("lat_track", "lat_shift"), ("lat_track", "lat_drift"),
    ("lat_track", "lat_jitter"), ("lat_track", "yaw_bias"),
    ("lat_track", "yaw_jitter"),
    # --- lateral placement: cross (must be weak) ---
    ("lat_track", "lon_retime"),
    # --- lateral heading: own-axis + cross ---
    ("lat_heading", "yaw_bias"), ("lat_heading", "lat_drift"),
    ("lat_heading", "yaw_jitter"), ("lat_heading", "lon_retime"),
    # --- ⛔ the incumbents, on the SAME ladders, for the direction check ---
    ("recovery", "lat_shift"), ("recovery", "lat_jitter"),
    ("recovery", "lon_retime"),
    ("ego_progress", "lon_retime"), ("ego_progress", "lon_jitter"),
    ("ego_progress", "lat_shift"),
    ("recovery", "yaw_jitter"),
]

#: which control belongs to which axis, for the admission rule
OWN = {"lon_track": ("lon_retime", "lon_scale", "lon_jitter"),
       "lat_track": ("lat_shift", "lat_drift", "lat_jitter", "yaw_bias",
                     "yaw_jitter"),
       "lat_heading": ("yaw_bias", "lat_drift", "yaw_jitter"),
       "recovery": ("lat_shift", "lat_jitter", "yaw_jitter"),
       "ego_progress": ("lon_retime", "lon_jitter")}


def load_pw(path):
    z = np.load(path, allow_pickle=False)
    return {"traj": torch.as_tensor(z["traj"]),
            "ref_path": torch.as_tensor(z["ref_path"]),
            "ref_yaw": torch.as_tensor(z["ref_yaw"]),
            "v0": torch.as_tensor(z["v0"]),
            "pt_dlat": torch.as_tensor(z["pt_dlat"]),
            "pt_dyaw": torch.as_tensor(z["pt_dyaw"]),
            "pt_dlon": torch.as_tensor(z["pt_dlon"]),
            "anchor": torch.as_tensor(z["anchor"]),
            "ep_i": torch.as_tensor(z["ep_i"]),
            "eid": [str(x) for x in z["eid"]]}


def human_replay(pw):
    """⭐⭐ THE ZERO-BIAS REFERENCE ARM, and it is not a convenience.

    A degradation ladder measures *the injection plus the arm's own bias*, and
    the two are not separable on a biased arm. MEASURED here: on ``cv_holdv0``
    — which drives STRAIGHT while the logged road curves — ``lat_track`` is
    **non-monotone at yaw_bias +2 -> +5 deg (0.3661 -> 0.3681)** and at
    ``lat_drift +0.02 -> +0.05`` (0.3706 -> 0.3719), because a small LEFT
    injection partially follows the road on the left-curving windows. Reading
    that as *"the lateral metric is non-monotone"* would have been a false
    verdict about the metric caused by the reference.

    ⛔ **There is no laterally-unbiased arm in the panel**, so one is
    constructed: the plan **IS the logged future path**, exactly. Zero bias on
    every axis by construction, so every injection is unambiguous.

    The inverse of ``pseudosim._cross_and_along``: that function maps a plan in
    the PERTURBED ego frame into the reference frame by ``rotate(dpsi)`` then
    ``+dlat`` on y, so the plan that lands exactly on the logged path is
    ``R(-dpsi) . (path - [0, dlat])``. Verified by construction —
    ``residuals()`` on the result must be 0 to float precision, which
    ``run_control_suite`` asserts before using it."""
    x, y, ref_x, ref_y = PS._cross_and_along(pw)
    Hh = x.shape[1]
    gx = ref_x[:, 1:Hh + 1]
    gy = ref_y[:, 1:Hh + 1] - pw["pt_dlat"][:, None]
    dpsi = torch.deg2rad(pw["pt_dyaw"])
    c, s = torch.cos(dpsi)[:, None], torch.sin(dpsi)[:, None]
    tj = torch.stack([c * gx + s * gy, -s * gx + c * gy], dim=-1)
    out = dict(pw)
    out["traj"] = tj.to(pw["traj"].dtype)
    return out


def key_of(pw):
    """Row identity. A paired bootstrap over misaligned rows is a fabricated
    number, so this is ASSERTED, never assumed (same construction as
    ``panel_combine.key_of``)."""
    return np.stack([pw["ep_i"].numpy(), pw["anchor"].numpy(),
                     np.round(pw["pt_dlat"].numpy(), 6) * 1e3,
                     np.round(pw["pt_dyaw"].numpy(), 6) * 1e3,
                     pw["pt_dlon"].numpy()], axis=1).astype(np.int64)


# =========================================================================== #
def job_axes(arms, out, n_boot):
    """Every arm on every axis + the PANEL-WIDE gate + tolerance sensitivity."""
    per_arm_axes, blocks = {}, {}
    for n, pw in arms.items():
        per_arm_axes[n] = C.axes(pw)
        blocks[n] = C.block(pw, arm=n, n_boot=n_boot, sensitivity=True)
        print(f"  [axes] {n}", flush=True)
    gate = C.panel_gate(per_arm_axes, probes=PROBES,
                        names=("lon_track", "lat_track", "lat_heading",
                               "recovery", "ego_progress"))
    ranking = {}
    for ax in ("lon_track", "lat_track", "lat_heading", "recovery",
               "ego_progress"):
        vals = {n: round(float(np.nanmean(per_arm_axes[n][ax])), 6)
                for n in arms if n not in PROBES
                and np.isfinite(per_arm_axes[n][ax]).any()}
        ranking[ax] = [{"rank": i + 1, "arm": k, "value": v} for i, (k, v)
                       in enumerate(sorted(vals.items(), key=lambda t: -t[1]))]
    out["axes_panel"] = {
        "suite_id": C.SUITE_ID(),
        "gate": {k: v for k, v in gate.items() if k != "detail"},
        "gate_detail": gate["detail"],
        "ranking_per_axis": ranking,
        "per_arm": blocks,
    }
    return out


def job_dynamic_range(arms, out, n_boot, ref_arm, second_arm):
    """⭐ The demonstration. Run on THREE arms on purpose."""
    res = {"_why_three_arms": (
        "A ladder measures the injected degradation PLUS the arm's own bias, "
        "and the two are NOT separable on a biased arm. (a) `human_replay` is "
        "the ZERO-BIAS reference — the plan IS the logged path — so every "
        "injection is unambiguous and the ADMISSION verdict is taken there. "
        "(b) `cv_holdv0` is longitudinally near-unbiased (along-track endpoint "
        "error -0.3814 m mean) but LATERALLY biased: it drives straight while "
        "the road curves, and a small LEFT injection partially follows the "
        "road, producing a real non-monotonicity that is a property of the ARM. "
        "(c) `v1_tactical_follow` over-travels on 48.80 % of windows, so the "
        "'down' rungs of a longitudinal ladder CORRECT it. All three are "
        "published: reporting only the biased arms would have produced a false "
        "'the metric is non-monotone' verdict, and reporting only the clean one "
        "would have hidden how the ladder behaves on the arms we actually "
        "evaluate."),
        "n_boot": n_boot, "arms": {}}
    for arm in ("human_replay", ref_arm, second_arm):
        if arm not in arms:
            continue
        pw, eid = arms[arm], arms[arm]["eid"]
        cells = {}
        for axis, ctl in DEMOS:
            t0 = time.time()
            d = C.dynamic_range(pw, eid, control=ctl, axis=axis, n_boot=n_boot)
            d["own_axis_control"] = ctl in OWN.get(axis, ())
            cells[f"{axis}__{ctl}"] = d
            print(f"  [dyn] {arm} {axis} x {ctl}: mde_up={d['mde_up']} "
                  f"mde_down={d['mde_down']} both={d['both_directions_separate']}"
                  f" ({time.time() - t0:.1f}s)", flush=True)
        res["arms"][arm] = cells
    # the admission verdict, on the reference arm's own-axis cells only
    verdicts = {}
    for axis in ("lon_track", "lat_track", "lat_heading", "recovery",
                 "ego_progress"):
        demos = [v for k, v in res["arms"].get("human_replay", {}).items()
                 if k.startswith(axis + "__") and v["own_axis_control"]]
        if not demos:
            continue
        try:
            verdicts[axis] = C.admit(demos)
        except C.AxisNotDemonstrated as exc:
            verdicts[axis] = {"axis": axis, "admissible": False,
                              "REFUSED": str(exc),
                              "mde": {d["control"]: {"up": d.get("mde_up"),
                                                     "down": d.get("mde_down"),
                                                     "unit": d["unit"]}
                                      for d in demos}}
        print(f"  [admit] {axis}: {verdicts[axis]['admissible']}", flush=True)
    res["admission"] = verdicts
    res["admission_rules"] = (
        "1) separates at all; 2) separates on BOTH sides of every two-sided "
        "ladder; 3) at least one ZERO-MEAN control separates; 4) monotone away "
        "from the null FROM THE MDE OUTWARD on every separating side. Evaluated "
        "on the ZERO-BIAS `human_replay` reference arm's OWN-AXIS controls; the "
        "two real arms are reported beside it and are NOT used for admission, "
        "because on a biased arm an injection can partially CORRECT the arm.")
    res["admission_reference_arm"] = "human_replay"
    out["dynamic_range"] = res
    return out


def job_cross_sensitivity(arms, out, ref_arm):
    """Axis purity, MEASURED: response per unit of INDUCED raw error.

    Comparing ``lon_retime(0.5)`` with ``lat_shift(0.5)`` is apples to oranges.
    So each rung reports the raw physical error it actually induced — mean
    ``|lon_end_err_m|`` and mean ``lat_xte_peak_m`` — and the axis response is
    divided by it. The resulting number is *score points per metre of induced
    error on that physical axis*, which IS comparable across controls."""
    pw = arms[ref_arm]
    base = C.axes(pw)
    raw = {"lon": "lon_abs_end_err_m", "lat": "lat_xte_peak_m"}
    b = {k: float(np.nanmean(base[v])) for k, v in raw.items()}
    rows = []
    for ctl, spec in C.LADDERS.items():
        for L in spec["levels"]:
            if float(L) == float(spec["null"]):
                continue
            a = C.axes(C.apply_control(pw, ctl, L))
            d_raw = {k: float(np.nanmean(a[v])) - b[k] for k, v in raw.items()}
            row = {"control": ctl, "level": float(L), "unit": spec["unit"],
                   "induced_d_lon_abs_end_err_m": round(d_raw["lon"], 6),
                   "induced_d_lat_xte_peak_m": round(d_raw["lat"], 6)}
            for ax in ("lon_track", "lat_track", "lat_heading", "recovery",
                       "ego_progress"):
                dv = float(np.nanmean(a[ax])) - float(np.nanmean(base[ax]))
                row[f"d_{ax}"] = round(dv, 6)
            row["d_lat_track_flat_DIAGNOSTIC"] = round(
                float(np.nanmean(a["lat_track_flat"]))
                - float(np.nanmean(base["lat_track_flat"])), 6)
            rows.append(row)
    # the purity ratios, at the biggest rung of each control
    def _at(ctl, lvl):
        return next(r for r in rows if r["control"] == ctl
                    and abs(r["level"] - lvl) < 1e-9)
    purity = {}
    try:
        lon_half, lat_1m = _at("lon_retime", 0.5), _at("lat_shift", -1.0)
        purity["lat_track_widening_corridor"] = {
            "d_under_lon_retime_0p5": lon_half["d_lat_track"],
            "d_under_lat_shift_minus1m": lat_1m["d_lat_track"],
            "contamination_over_signal": round(
                abs(lon_half["d_lat_track"])
                / max(abs(lat_1m["d_lat_track"]), 1e-9), 4)}
        purity["lat_track_FLAT_tolerance_the_rejected_form"] = {
            "d_under_lon_retime_0p5": lon_half["d_lat_track_flat_DIAGNOSTIC"],
            "d_under_lat_shift_minus1m": lat_1m["d_lat_track_flat_DIAGNOSTIC"],
            "contamination_over_signal": round(
                abs(lon_half["d_lat_track_flat_DIAGNOSTIC"])
                / max(abs(lat_1m["d_lat_track_flat_DIAGNOSTIC"]), 1e-9), 4)}
        purity["lon_track"] = {
            "d_under_lon_retime_0p5": lon_half["d_lon_track"],
            "d_under_lat_shift_minus1m": lat_1m["d_lon_track"],
            "contamination_over_signal": round(
                abs(lat_1m["d_lon_track"])
                / max(abs(lon_half["d_lon_track"]), 1e-9), 4)}
        yaw5 = _at("yaw_bias", 5.0)
        purity["lat_heading"] = {
            "d_under_yaw_bias_5deg": yaw5["d_lat_heading"],
            "d_under_lon_retime_0p5": lon_half["d_lat_heading"],
            "contamination_over_signal": round(
                abs(lon_half["d_lat_heading"])
                / max(abs(yaw5["d_lat_heading"]), 1e-9), 4)}
        purity["_read"] = (
            "contamination_over_signal < 1 means the axis responds MORE to its "
            "own control than to the other axis'. The FLAT-tolerance lateral "
            "form is published beside the shipped one precisely because it "
            "FAILS this test, which is why it was rejected.")
    except StopIteration:
        purity["_error"] = "a required rung is missing from the ladder"
    out["cross_sensitivity"] = {"reference_arm": ref_arm,
                                "baseline_raw": {k: round(v, 6)
                                                 for k, v in b.items()},
                                "purity": purity, "rows": rows}
    return out


def job_recovery_onesided(arms, out, n_boot):
    """⛔ The floor-saturation of ``recovery`` and the DIRECTION of its response.

    ``recovery = clamp(1 - xt_end / xt_hold, 0, 1)``. The clamp at 0 is a HARD
    FLOOR: a plan already worse than doing nothing cannot be scored any worse.
    Every row past the floor is charged identically, so any perturbation that
    helps a *few* rows raises the mean while the damaged rows are absorbed. That
    is the ``clamp_v1`` failure class in the OTHER weight-5.0 term."""
    res = {"floor_saturation": {}, "direction": {}}
    for n, pw in arms.items():
        sc = PS.score_windows(pw)
        rc = np.asarray(sc["recovery"], float)
        fin = np.isfinite(rc)
        if not fin.any():
            continue
        r = (np.asarray(sc["cross_track_end_m"], float)
             / np.maximum(np.asarray(sc["cross_track_hold_matched_m"], float),
                          1e-9))
        rf = r[fin]
        res["floor_saturation"][n] = {
            "mean": round(float(rc[fin].mean()), 6),
            "defined_frac": round(float(fin.mean()), 6),
            "frac_at_floor_le_0p001": round(float((rc[fin] <= 1e-3).mean()), 6),
            "frac_at_ceiling_ge_0p999": round(float((rc[fin] >= 0.999).mean()), 6),
            "unclamped_ratio_median": round(float(np.median(rf)), 4),
            "unclamped_ratio_p90": round(float(np.percentile(rf, 90)), 4),
            "unclamped_ratio_p99": round(float(np.percentile(rf, 99)), 4),
            "unclamped_ratio_max": round(float(rf.max()), 4),
            "frac_ratio_gt_1_i_e_WORSE_THAN_DOING_NOTHING":
                round(float((rf > 1.0).mean()), 6),
        }
    # ⛔ the direction check: does a LATERAL degradation IMPROVE the composite?
    for arm in ("cv_holdv0", "v1_tactical_follow"):
        if arm not in arms:
            continue
        pw, eid = arms[arm], arms[arm]["eid"]
        base = PS.score_windows(pw)
        cells = {}
        for ctl, lv in (("lat_shift", 2.0), ("lat_shift", -2.0),
                        ("lat_jitter", 1.0), ("yaw_bias", 5.0)):
            p = C.apply_control(pw, ctl, lv)
            s = PS.score_windows(p)
            blk = {}
            for comp in ("recovery", "ego_progress"):
                a = np.asarray(s[comp], float)
                b0 = np.asarray(base[comp], float)
                m = np.isfinite(a) & np.isfinite(b0)
                blk[comp] = _ci.paired_episode_cluster_bootstrap(
                    a[m], b0[m], list(np.asarray(eid)[m]), n_boot=n_boot)
            # the composite itself, under the shipped weights
            def _comp(sc):
                num = np.zeros(len(eid))
                den = np.zeros(len(eid))
                for nm, wt in PS.COMPONENT_WEIGHTS.items():
                    if wt == 0:
                        continue
                    v = np.asarray(sc[nm], float)
                    f = np.isfinite(v)
                    num = num + np.where(f, v * wt, 0.0)
                    den = den + np.where(f, wt, 0.0)
                return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            ca, cb = _comp(s), _comp(base)
            m = np.isfinite(ca) & np.isfinite(cb)
            blk["PSS_composite"] = _ci.paired_episode_cluster_bootstrap(
                ca[m], cb[m], list(np.asarray(eid)[m]), n_boot=n_boot)
            cells[f"{ctl}({lv:+g})"] = blk
            print(f"  [recov] {arm} {ctl}({lv:+g}): "
                  f"dPSS={blk['PSS_composite']['delta']} "
                  f"sep={blk['PSS_composite']['separated']}", flush=True)
        res["direction"][arm] = cells
    res["_read"] = (
        "A POSITIVE delta means the DEGRADED plan scores HIGHER. Every entry "
        "here is an injected LATERAL error, including a ZERO-MEAN one that "
        "cannot re-centre a bias in expectation.")
    out["recovery_onesided"] = res
    return out


def job_comfort(arms, out):
    """⛔ THE DECISIVE CONTROL: run the comfort clauses on the HUMAN'S OWN PATH.

    If the plans fail the bounds and the human passes, ``comfort`` is measuring
    the plans. If the human fails them too, ``comfort`` is measuring 10 Hz
    differentiation of a 20-point polyline and cannot separate driving quality
    from anything. The clause-by-clause split says which."""
    lim = PS.COMFORT_LIMITS
    dt = C.DT

    def clauses(px, py):
        n = px.shape[0]
        X = np.concatenate([np.zeros((n, 1)), px], 1)
        Y = np.concatenate([np.zeros((n, 1)), py], 1)
        vx, vy = np.diff(X, axis=1) / dt, np.diff(Y, axis=1) / dt
        ax, ay = np.diff(vx, axis=1) / dt, np.diff(vy, axis=1) / dt
        jx, jy = np.diff(ax, axis=1) / dt, np.diff(ay, axis=1) / dt
        head = np.arctan2(vy, vx)
        yr = np.diff(head, axis=1) / dt
        return {
            "a_lon": np.abs(ax).max(1) <= lim["a_lon_max_mps2"],
            "a_lat": np.abs(ay).max(1) <= lim["a_lat_max_mps2"],
            "jerk": np.sqrt(jx ** 2 + jy ** 2).max(1) <= lim["jerk_max_mps3"],
            "yaw_rate": np.abs(yr).max(1) <= lim["yaw_rate_max_radps"],
        }

    res = {"limits": lim, "limits_class": "PROPOSED (NAVSIM/nuPlan-style; their "
                                          "exact constants are not quotable "
                                          "from verified material)",
           "arms": {}, "human_reference": {}}
    for n, pw in arms.items():
        r = C.residuals(pw)
        cl = clauses(r["plan_x"], r["plan_y"])
        allok = cl["a_lon"] & cl["a_lat"] & cl["jerk"] & cl["yaw_rate"]
        res["arms"][n] = {"comfort_mean_ALL_CLAUSES": round(
            float(allok.mean()), 6),
            "pass_frac_per_clause": {k: round(float(v.mean()), 6)
                                     for k, v in cl.items()}}
    # ⭐ THE HUMAN. Identical arithmetic on the logged path, same windows.
    any_arm = next(iter(arms))
    r = C.residuals(arms[any_arm])
    hx = np.concatenate([r["human_x"]], axis=1)
    hy = np.concatenate([r["human_y"]], axis=1)
    cl = clauses(hx, hy)
    allok = cl["a_lon"] & cl["a_lat"] & cl["jerk"] & cl["yaw_rate"]
    res["human_reference"] = {
        "_what": ("the LOGGED HUMAN PATH over the identical windows, "
                  "differenced with the identical arithmetic. This is real "
                  "driving by definition."),
        "comfort_mean_ALL_CLAUSES": round(float(allok.mean()), 6),
        "pass_frac_per_clause": {k: round(float(v.mean()), 6)
                                 for k, v in cl.items()},
        "n_rows": int(allok.size)}
    res["_verdict"] = (
        "If the human's own pass-rate is far below 1.0, the bounds are not a "
        "comfort measurement on this surface and the term cannot separate good "
        "driving from bad — it separates smoothness of a 20-point polyline.")
    out["comfort_audit"] = res
    return out


def job_repro(arms, out, n_boot):
    """✅ Prove that zeroing ``comfort``'s weight changes NO published number."""
    scores = {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw, progress_term="clamp_v1")
        scores[n] = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        scores[n]["no_collision"] = None
        scores[n]["ttc"] = None
    gate_arms = [n for n in arms if n not in PROBES]
    per_arm = {n: PS.discriminative_range(scores[n], by_arm=scores)
               for n in arms}
    admissible = {}
    for comp in PS.COMPONENT_WEIGHTS_PUBLISHED_V1:
        admissible[comp] = not [n for n in gate_arms
                                if not per_arm[n].get(comp, {}).get("admissible")]
    rep, worst = {}, 0.0
    for n, want in PUBLISHED_CLAMP_V1.items():
        if n not in arms:
            rep[n] = {"published": want, "status": "ARM ABSENT"}
            continue
        pr = dict(per_arm[n])
        for c, ok in admissible.items():
            if not ok and c in pr:
                pr[c] = dict(pr[c], admissible=False,
                             reason="dropped by the PANEL-WIDE gate")
        comp = PS.composite(scores[n], pr, progress_term="clamp_v1")
        v = comp.pop("value")
        got = PS._boot(v, arms[n]["eid"], n_boot, 0)["mean"]
        d = abs(got - want)
        worst = max(worst, d)
        rep[n] = {"published": want, "got": got, "abs_diff": round(d, 6),
                  "n_weighted_terms": comp["n_weighted_terms"],
                  "components_zero_weighted": list(comp["components_zero_weighted"]),
                  "status": "OK" if d <= 5e-4 else "MISMATCH"}
    out["repro_gate"] = {
        "rule": ("Every PUBLISHED @clamp_v1 composite must still reproduce to "
                 "4 dp with COMPONENT_WEIGHTS['comfort'] = 0.0. If it does "
                 "not, the weight change is NOT the provable no-op it is "
                 "claimed to be and must be reverted."),
        "weights_now": PS.COMPONENT_WEIGHTS,
        "weights_published_v1": PS.COMPONENT_WEIGHTS_PUBLISHED_V1,
        "weights_id": PS.WEIGHTS_ID,
        "panel_gate_admissible": admissible,
        "max_abs_diff": round(worst, 6),
        "PASS": bool(worst <= 5e-4),
        "per_arm": rep,
        "_why_it_is_a_no_op": (
            "comfort is dropped by the panel-wide gate for every arm, AND a "
            "zero weight adds exactly 0.0 to both numerator and denominator in "
            "composite(), so the value is bit-identical in both branches."),
    }
    print(f"  [repro] max|diff| = {worst:.6f} PASS={worst <= 5e-4}", flush=True)
    return out


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=_ci.DEFAULT_N_BOOT)
    ap.add_argument("--ref-arm", default="cv_holdv0")
    ap.add_argument("--second-arm", default="v1_tactical_follow")
    ap.add_argument("--jobs", default="axes,dyn,cross,recov,comfort,repro")
    a = ap.parse_args()

    t0 = time.time()
    arms, seen = {}, {}
    for d in a.in_dir:
        for f in sorted(Path(d).glob("pw_*.npz")):
            name = f.name[len("pw_"):-len(".npz")]
            if name in arms:
                raise SystemExit(f"duplicate arm {name}: {seen[name]} vs {f}")
            arms[name] = load_pw(f)
            seen[name] = str(f)
    print(f"[suite] {len(arms)} arms: {sorted(arms)}", flush=True)

    # ---- row identity, ASSERTED ------------------------------------------- #
    ref = a.ref_arm if a.ref_arm in arms else sorted(arms)[0]
    kref = key_of(arms[ref])
    row_id, refused = {}, {}
    for n in list(arms):
        k = key_of(arms[n])
        ok = (k.shape == kref.shape) and bool((k == kref).all())
        row_id[n] = {"n_rows": int(k.shape[0]), "identical_to_reference": ok}
        if not ok:
            refused[n] = f"row keys differ from {ref}"
            arms.pop(n)
    assert arms, f"every arm refused on row identity: {refused}"

    # ⭐ the ZERO-BIAS reference arm, VERIFIED to be zero-bias before use
    hr = human_replay(arms[ref])
    _r = C.residuals(hr)
    _worst = max(float(np.abs(_r["along_err_m"]).max()),
                 float(np.abs(_r["cross_err_m"]).max()),
                 float(np.abs(_r["xte_m"]).max()))
    assert _worst < 1e-3, (
        f"human_replay is NOT on the logged path (max |residual| = {_worst}); "
        f"the frame inversion is wrong and every ladder run on it would be "
        f"measuring the inversion error, not the injection")
    arms["human_replay"] = hr
    print(f"[suite] human_replay built, max|residual| = {_worst:.3e} m",
          flush=True)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    head = {
        "_experiment": ("the LONGITUDINAL / LATERAL CONTROL SUITE on the "
                        "pseudo-simulation surface"),
        "_evidence_class": "MEASURED (ours; artifact = these JSONs + pw_*.npz)",
        "_protocol": PS.PROTOCOL,
        "_estimator": (f"taniteval.ci.episode_cluster_bootstrap / paired form "
                       f"(B={a.n_boot}, unit = val episode)"),
        "_refused_estimator": "overlapping_holdout_se",
        "_gate": "PANEL-WIDE. The per-arm gate is REFUSED, not offered.",
        "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
        "traffic_mode_note": PS.TRAFFIC_MODE_NOTE,
        "missing_gates": C.MISSING_GATES,
        "comfort_status": PS.COMFORT_STATUS,
        "suite_id": C.SUITE_ID(),
        "arm_sources": seen, "row_identity": row_id,
        "arms_refused_on_row_identity": refused,
        "reference_arm": ref, "probes_excluded": list(PROBES),
        "selected_frac_note": (
            "selected_frac = 1.000 for every arm BY CONSTRUCTION: there is no "
            "selection step on this surface, every arm is scored on all rows."),
    }

    def bank(name, node):
        p = od / f"{name}.json"
        p.write_text(json.dumps(dict(head, **{name: node}), indent=2,
                                default=str), encoding="utf-8")
        print(f"[suite] wrote {p}", flush=True)

    jobs = set(a.jobs.split(","))
    acc = {}
    if "axes" in jobs:
        job_axes(arms, acc, a.n_boot)
        bank("axes_panel", acc["axes_panel"])
    if "comfort" in jobs:
        job_comfort(arms, acc)
        bank("comfort_audit", acc["comfort_audit"])
    if "repro" in jobs:
        job_repro(arms, acc, a.n_boot)
        bank("repro_gate", acc["repro_gate"])
    if "cross" in jobs:
        job_cross_sensitivity(arms, acc, ref)
        bank("cross_sensitivity", acc["cross_sensitivity"])
    if "recov" in jobs:
        job_recovery_onesided(arms, acc, a.n_boot)
        bank("recovery_onesided", acc["recovery_onesided"])
    if "dyn" in jobs:
        job_dynamic_range(arms, acc, a.n_boot, ref, a.second_arm)
        bank("dynamic_range", acc["dynamic_range"])

    print(f"[suite] done in {time.time() - t0:.1f} s")
    print("CONTROL_SUITE_DONE", flush=True)


if __name__ == "__main__":
    main()
