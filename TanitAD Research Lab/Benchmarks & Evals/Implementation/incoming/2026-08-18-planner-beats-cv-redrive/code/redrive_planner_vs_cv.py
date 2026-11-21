#!/usr/bin/env python3
"""planner_beats_cv (C91) — what the BANKED per-window data can and cannot settle.

⛔ THE DEFINITIONAL FINDING THAT DRIVES THIS WHOLE SCRIPT
`planner_beats_cv` is computed at `taniteval/taniteval/planner_p2.py:621`

    "planner_beats_cv": bool(boot["plan"]["mean"] < boot["cv"]["mean"]),

inside `analyze_openloop` (:555). `boot` there comes from `collect_openloop`'s
`plan_wp` / `cv_wp` — the OPEN-LOOP arms, n = 881 windows / 40 episodes,
stride 8. It is **NOT** the closed-loop comparison. The banked dump
`raw_windows/p2win_flagship-30k.pt` holds the CLOSED-LOOP collection only
(221 windows / 20 episodes, stride 16 — keys closed_bike/open_grnd/cv/gt).
⇒ the zero-GPU path proposed in the brief does not reach this verdict.

WHAT THIS SCRIPT DOES, ALL ON CPU, NO MODEL LOADED:
  1. Verdict inventory — every boolean in the artifact, from the artifact.
  2. Reproduction gate — recompute the BANNED estimator from banked windows and
     check it returns the published numbers (else nothing below is comparable).
  3. Open-loop arms, decision-grade — every arm that IS banked (cv / operative /
     head), plus the exact, stated flip requirement for the one that is not.
  4. NEW: the CLOSED-LOOP planner-vs-CV comparison, PAIRED — never computed
     before (published G4 compared planner vs HEAD, never vs CV).
  5. Four metric families on the banked closed-loop windows, per family, with an
     explicit reason + n wherever a family genuinely cannot be computed.

Estimator: `taniteval.ci.episode_cluster_bootstrap` /
`paired_episode_cluster_bootstrap`. Point estimates are **full_set** means.
⛔ `overlapping_holdout_se` appears ONLY in the reproduction gate (step 2) and is
never read by a verdict.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

# code/ -> <task> -> incoming -> Implementation -> "Benchmarks & Eval"
#       -> "TanitAD Research Hub" -> REPO   (six levels up)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    *([".."] * 6)))
assert os.path.isdir(os.path.join(REPO, "taniteval")), f"REPO wrong: {REPO}"
sys.path.insert(0, os.path.join(REPO, "taniteval"))
sys.path.insert(0, os.path.join(REPO, "stack"))

from taniteval import ci as _ci                     # noqa: E402
from taniteval import pathspeed as ps               # noqa: E402
# ⛔ REUSE, never reimplement: `_ade2` and `_jack_scalar` are the SAME functions
# that produced the published artifact, so the reproduction gate below cannot
# drift from the thing it is meant to reproduce.
from taniteval import planner_p2 as p2m             # noqa: E402

INC = os.path.join(REPO, "TanitAD Research Hub", "Benchmarks & Eval",
                   "Implementation", "incoming")
RERUN = os.path.join(INC, "2026-07-26-closedloop-artifact-rerun")
OUT = os.path.join(INC, "2026-08-18-planner-beats-cv-redrive", "raw")

P2WIN = os.path.join(RERUN, "raw_windows", "p2win_flagship-30k.pt")
CLWIN = os.path.join(RERUN, "raw_windows", "clwin_flagship-30k.pt")
OLWIN = os.path.join(REPO, "taniteval", "results", "windows_flagship-30k.pt")
PUB = os.path.join(RERUN, "_pod_pulled", "planner_p2_flagship-30k.json")

DT_WP = 0.5          # banked waypoints are 0.5 s apart (WP_STEPS 5,10,15,20 @10Hz)
HORIZONS = (0.5, 1.0, 1.5, 2.0)


# ---------------------------------------------------------------- helpers ---
def ade2(pred, gt):
    """[N,4,2] vs [N,4,2] -> [N] ADE over the 4 waypoints.
    THE harness function itself (planner_p2._ade2), not a copy of it."""
    return p2m._ade2(pred, gt)


def eids_s(e):
    return [str(x) for x in e]


def interval(vals, eid, seed=0):
    return _ci.episode_cluster_bootstrap(np.asarray(vals, dtype=float),
                                         eids_s(eid), n_boot=_ci.DEFAULT_N_BOOT,
                                         seed=seed)


def paired(a, b, eid, seed=0):
    return dict(_ci.paired_episode_cluster_bootstrap(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), eids_s(eid),
        n_boot=_ci.DEFAULT_N_BOOT, seed=seed))


def banned(vals, eid, n_splits=8, val_frac=0.2):
    """The BANNED estimator, REPRODUCTION ONLY — `overlapping_holdout_se`.

    Calls `planner_p2._jack_scalar`, i.e. the exact function that produced the
    published numbers, over the exact split structure the harness builds
    (`gates.split_by_episode`, seeds 0..7, val_frac 0.2). Never read by a
    verdict anywhere in this script."""
    from tanitad.eval.gates import split_by_episode
    e = eids_s(eid)
    splits = [split_by_episode(e, val_frac, s) for s in range(n_splits)]
    return p2m._jack_scalar(np.asarray(vals, dtype=float), e, splits)


def wp_speed(path):
    """Speed (m/s) on each 0.5 s waypoint leg. [N,4,2] -> [N,4].

    ⚠️ `pathspeed.step_speed` divides by DT=0.1 and is therefore WRONG on these
    0.5 s-spaced waypoints (it would report 5x). Done explicitly here instead —
    the geometric primitives below ARE reused because they are dt-independent."""
    o = torch.zeros_like(path[:, :1])
    full = torch.cat([o, path], dim=1)
    return (full[:, 1:] - full[:, :-1]).norm(dim=-1) / DT_WP


def wp_heading_deg(path):
    """Heading of each waypoint leg (deg, ego-forward = 0). dt-independent —
    reuses pathspeed.segment_tangents verbatim."""
    t = ps.segment_tangents(path)
    return torch.atan2(t[..., 1], t[..., 0]) * (180.0 / math.pi)


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def four_families(pred, gt, eid, name):
    """The four binding metric families on [N,4,2] waypoints.

    ⛔ Per CLAUDE.md an ADE-only answer is INCOMPLETE. Families that genuinely
    cannot be computed from this dump are returned with an explicit reason + n
    rather than silently dropped."""
    n = int(pred.shape[0])
    along, cross = ps.frenet_residual(pred, gt)      # dt-independent geometry
    v_p, v_g = wp_speed(pred), wp_speed(gt)
    hd_p, hd_g = wp_heading_deg(pred), wp_heading_deg(gt)
    dhead = wrap180(hd_p - hd_g)
    # curvature kappa ~ dtheta/ds on each leg; yaw rate = dtheta/dt
    ds_p = (v_p * DT_WP).clamp_min(1e-6)
    ds_g = (v_g * DT_WP).clamp_min(1e-6)
    dth_p = wrap180(torch.diff(hd_p, dim=1)) * math.pi / 180.0
    dth_g = wrap180(torch.diff(hd_g, dim=1)) * math.pi / 180.0
    kap_p = dth_p / ds_p[:, 1:]
    kap_g = dth_g / ds_g[:, 1:]
    yaw_p = dth_p / DT_WP
    yaw_g = dth_g / DT_WP

    # ⛔ CURVATURE IS UNDEFINED FOR A STOPPED EGO and must be masked, not
    # clamped. kappa = dtheta/ds divides by arc length; on the 11 windows where
    # the ego is stationary (v0 < 0.5 m/s, GT legs down to 0.0000 m) this yields
    # |kappa| up to 23004 1/m and DOMINATES the mean: unmasked GT |kappa| mean
    # is 34.82 1/m against a MEDIAN of 0.0008 1/m. Reporting that mean would be
    # reporting a division artifact as a lateral metric. Windows are kept only
    # if EVERY GT leg exceeds MIN_LEG_M; the excluded n is reported per family.
    # Heading is also ill-defined there (segment_tangents:68-70 carries the last
    # valid tangent forward over degenerate segments), so the same mask gates
    # the heading/yaw figures reported as `_moving_only`.
    MIN_LEG_M = 0.5
    o = torch.zeros_like(gt[:, :1])
    gt_leg = (torch.cat([o, gt], dim=1)[:, 1:]
              - torch.cat([o, gt], dim=1)[:, :-1]).norm(dim=-1)
    mv = gt_leg.min(dim=1).values > MIN_LEG_M          # [N] bool
    n_mv = int(mv.sum())

    def rmse(x):
        return round(float(x.pow(2).mean().sqrt()), 4)

    def mean(x):
        return round(float(x.mean()), 4)

    per_h = {}
    for i, t in enumerate(HORIZONS):
        per_h[f"{t:g}s"] = {
            "long_rmse_m": rmse(along[:, i]),
            "long_bias_m": mean(along[:, i]),
            "lat_rmse_m": rmse(cross[:, i]),
            "lat_bias_m": mean(cross[:, i]),
            "long_frac_of_sqerr": round(
                float(along[:, i].pow(2).mean()
                      / (along[:, i].pow(2).mean()
                         + cross[:, i].pow(2).mean()).clamp_min(1e-12)), 4),
            "speed_err_mps": mean((v_p - v_g)[:, i].abs()),
            "speed_bias_mps": mean((v_p - v_g)[:, i]),
            "heading_err_deg": mean(dhead[:, i].abs()),
        }
    return {
        "arm": name, "n_windows": n, "n_episodes": len(set(eids_s(eid))),
        "LONGITUDINAL": {
            "_computable": True,
            "per_horizon_long_rmse_m": {f"{t:g}s": per_h[f"{t:g}s"]["long_rmse_m"]
                                        for t in HORIZONS},
            "speed_err_2s_mps": per_h["2s"]["speed_err_mps"],
            "speed_bias_2s_mps": per_h["2s"]["speed_bias_mps"],
            "speed_profile_rmse_mps": rmse(v_p - v_g),
            "long_frac_of_2s_sqerr": per_h["2s"]["long_frac_of_sqerr"],
            "_distance_keeping": {
                "_computable": False, "n": n,
                "_reason": "headway / time-gap / TTC needs a LEAD-AGENT track. "
                           "This dump banks ego waypoints only, and the P2 cost "
                           "itself skips the gap term for the same reason "
                           "(planner_p2.py:36-38 'no lead-agent labels in our "
                           "front-cam+pose data'). obstacle.offline (the 3D "
                           "agent-track feature, 97.44% coverage) is a POD-SIDE "
                           "join and was not part of this eval."},
        },
        "LATERAL": {
            "_computable": True,
            "crosstrack_rmse_2s_m": per_h["2s"]["lat_rmse_m"],
            "per_horizon_lat_rmse_m": {f"{t:g}s": per_h[f"{t:g}s"]["lat_rmse_m"]
                                       for t in HORIZONS},
            "heading_err_2s_deg": per_h["2s"]["heading_err_deg"],
            "heading_err_mean_deg": mean(dhead.abs()),
            # --- moving-only: curvature/heading/yaw need a non-degenerate path
            "_moving_only": {
                "n_used": n_mv, "n_excluded": n - n_mv,
                "criterion": f"every GT leg > {MIN_LEG_M} m",
                "_reason": "curvature divides heading change by ARC LENGTH and "
                           "is undefined for a stopped ego; heading itself is "
                           "carried forward over degenerate segments. Unmasked, "
                           "the GT |kappa| MEAN is 34.82 1/m against a MEDIAN "
                           "of 0.0008 1/m — a division artifact, not a metric.",
                "curvature_err_mean_invm": mean((kap_p - kap_g)[mv].abs()),
                "curvature_rmse_invm": rmse((kap_p - kap_g)[mv]),
                "gt_curvature_mean_invm": mean(kap_g[mv].abs()),
                "heading_err_mean_deg": mean(dhead[mv].abs()),
                "heading_err_2s_deg": mean(dhead[mv][:, -1].abs()),
                "yawrate_err_mean_degps": round(
                    float(((yaw_p - yaw_g)[mv].abs().mean())
                          * 180.0 / math.pi), 4),
                "yawrate_rmse_degps": round(
                    float(((yaw_p - yaw_g)[mv].pow(2).mean().sqrt())
                          * 180.0 / math.pi), 4),
            },
            # yaw rate does NOT divide by arc length, so it is also reported on
            # all windows; it is still heading-based, hence the moving-only twin.
            "yawrate_err_mean_degps_ALL": round(
                float(((yaw_p - yaw_g).abs().mean()) * 180.0 / math.pi), 4),
            "_note": "curvature/yaw are computed on 0.5 s legs — the banked "
                     "resolution. They are COARSER than the K=20 (0.1 s) path "
                     "the live harness uses, and are comparable ACROSS ARMS "
                     "here because every arm is measured identically.",
        },
        "TACTICAL": {
            "_computable": False, "n": n,
            "_reason": "manoeuvre-decision quality needs the selected vs "
                       "executed manoeuvre class. (a) This dump banks "
                       "waypoints only — no manoeuvre logits. (b) More "
                       "fundamentally the P2 CEM planner EMITS NO MANOEUVRE "
                       "CLASS: it searches a continuous action sequence "
                       "(planner_p2.py:280 cem_plan), so there is no discrete "
                       "decision to score. The class exists only for the "
                       "5-way head baseline, which is the arm P2 replaces.",
            "_work_item": "scoring a tactical family for a continuous planner "
                          "needs a manoeuvre LABELLER applied to both the "
                          "planned and the GT path. Not implemented.",
        },
        "STRATEGIC": {
            "_computable": False, "n": n,
            "_reason": "strategic decision + route/goal quality is not defined "
                       "for this arm: the P2 cost carries NO route or goal term "
                       "at all (planner_p2.py:44-51, 'LONGITUDINAL + comfort + "
                       "progress only ... no lateral / route / goal term; the "
                       "strategic goal module is P3'). There is no strategic "
                       "output to score, so this is a genuine N/A, not a "
                       "missing measurement.",
        },
        "_per_horizon": per_h,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"_what": "planner_beats_cv (C91): what banked data settles, and what "
                    "it provably cannot",
           "_compute": "CPU only. No model loaded, no GPU, no pod contacted "
                       "(Thor is training).",
           "_estimator": {
               "decision": "episode_cluster_bootstrap / "
                           "paired_episode_cluster_bootstrap (taniteval/ci.py)",
               "point_estimate": "full_set mean (NEVER the heldout split-mean)",
               "n_boot": _ci.DEFAULT_N_BOOT, "seed": 0,
               "deprecated_used_only_for": "the step-2 reproduction gate"}}

    # ---------------------------------------------------------------- 1 ----
    pub = json.load(open(PUB))
    bools = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, bool):
            bools.append({"path": path, "value": o})
    walk(pub)
    distinct = sorted({b["path"].split(".")[-1].split("[")[0] for b in bools})
    res["1_verdict_inventory"] = {
        "_rule": "C91: ENUMERATE every verdict in the artifact itself.",
        "n_boolean_instances": len(bools),
        "n_distinct_verdict_names": len(distinct),
        "distinct_names": distinct,
        "instances": bools,
    }

    # ---------------------------------------------------------------- 2 ----
    p2 = torch.load(P2WIN, map_location="cpu", weights_only=False)
    cl = torch.load(CLWIN, map_location="cpu", weights_only=False)
    ol = torch.load(OLWIN, map_location="cpu", weights_only=False)

    cl_ade = {k: ade2(p2[k], p2["gt"]).numpy()
              for k in ("closed_bike", "open_grnd", "cv")}
    repro = {}
    pub_cl = pub["closed_loop"]
    for k, pubkey in (("closed_bike", "closed_bike_ade2s"),
                      ("open_grnd", "open_grnd_ade2s"),
                      ("cv", "cv_ade2s")):
        b = banned(cl_ade[k], p2["eid"])
        exact = bool(abs(b["mean"] - pub_cl[pubkey]["mean"]) < 5e-5)
        drift = abs(b["mean"] - pub_cl[pubkey]["mean"]) / pub_cl[pubkey]["mean"]
        # closed_bike is the ONLY model-in-the-loop arm here and the ONLY one
        # that goes through the CEM. The CEM was UNSEEDED until fa4b3d1, so a
        # ~0.02 % residual is EXPECTED there and is not a reproduction failure;
        # cv and open_grnd are deterministic and must match bit-exactly.
        cem_arm = (k == "closed_bike")
        repro[k] = {
            "published_mean": pub_cl[pubkey]["mean"],
            "published_ci95": pub_cl[pubkey]["ci95"],
            "recomputed_banned_mean": round(float(b["mean"]), 4),
            "recomputed_banned_ci95": round(float(b["ci95"]), 4),
            "exact_to_4dp": exact,
            "rel_drift_pct": round(100.0 * drift, 4),
            "goes_through_unseeded_cem": cem_arm,
            "gate_pass": bool(exact or (cem_arm and drift < 1e-3)),
            "_note": ("expected: unseeded-CEM residual, 0.019 % in "
                      "planner_p2_G4.CORRECTED.json; seeded at fa4b3d1"
                      if cem_arm else
                      "deterministic arm — must be bit-exact"),
        }
    res["2_reproduction_gate"] = {
        "_why": "recomputing the BANNED estimator from banked windows must "
                "return the published numbers, else the banked windows are not "
                "the objects the 2026-07-19 gate ran on and nothing below is "
                "comparable. cv is MODEL-FREE, so its exact reproduction is the "
                "load-bearing one.",
        "closed_loop_221": repro,
        "cv_is_model_free_and_reproduces_bit_exactly":
            repro["cv"]["exact_to_4dp"],
        "all_arms_pass_gate": all(v["gate_pass"] for v in repro.values()),
    }

    # ---------------------------------------------------------------- 3 ----
    # open-loop arms that ARE banked. `pred` in windows_*.pt is the operative
    # rollout under TRUE actions; `plan_direct` in clwin is the tactical head.
    assert ol["eid"] == cl["eid"], "open-loop banks must share the window set"
    assert torch.allclose(ol["gt"], cl["gt"], atol=1e-5), "GT must be identical"
    ol_eid = ol["eid"]
    ol_arms = {
        "constant_velocity": ade2(ol["cv"], ol["gt"]).numpy(),
        "operative_rollout_trueA": ade2(ol["pred"], ol["gt"]).numpy(),
        "tactical_head": ade2(cl["plan_direct"], cl["gt"]).numpy(),
    }
    ol_boot, ol_banned = {}, {}
    for k, v in ol_arms.items():
        ol_boot[k] = interval(v, ol_eid)
        ol_banned[k] = {kk: round(float(vv), 4)
                        for kk, vv in banned(v, ol_eid).items()
                        if kk in ("mean", "ci95")}
    pub_ol = pub["open_loop"]["ade2s"]
    cv_corr = ol_boot["constant_velocity"]["mean"]
    plan_banned = pub_ol["planner"]["mean"]
    res["3_openloop_the_actual_verdict"] = {
        "_definition": "planner_beats_cv := plan_full_set_mean < cv_full_set_mean"
                       " (planner_p2.py:621, inside analyze_openloop:555)",
        "_scope": "OPEN LOOP, n=881 windows / 40 episodes, stride 8. NOT the "
                  "closed-loop block.",
        "banked_arms_decision_grade": ol_boot,
        "banked_arms_banned_reproduction": ol_banned,
        "published_banned": {k: pub_ol[k] for k in pub_ol},
        "planner_arm": {
            "_status": "NOT BANKED PER-WINDOW — the one missing input",
            "banned_mean": plan_banned,
            "banned_ci95": pub_ol["planner"]["ci95"],
            "_probes_for_absence": [
                "taniteval/results/*.pt — pred/cv/gt only, no CEM plan arm",
                "…/2026-07-26-closedloop-artifact-rerun/raw_windows/*.pt — "
                "closed-loop only (p2win) + head-driven loop (clwin)",
                "exhaustive walk of EVERY .pt in the repo (code/scan_pt.py) — "
                "no open-loop CEM planner arm at n=881 anywhere",
            ],
        },
        "flip_requirement": {
            "cv_floor_corrected_full_set": cv_corr,
            "cv_floor_banned": pub_ol["constant_velocity"]["mean"],
            "cv_correction_pct": round(
                100.0 * (pub_ol["constant_velocity"]["mean"] - cv_corr)
                / abs(cv_corr), 3),
            "planner_banned": plan_banned,
            "planner_corrected_must_be_below": cv_corr,
            "required_downward_correction_pct": round(
                100.0 * (plan_banned - cv_corr) / cv_corr, 3),
            "measured_local_envelope_pct": {
                "_windows": "the same 881-window set, same 8-split structure",
                "tactical_head": round(
                    100.0 * (ol_banned["tactical_head"]["mean"]
                             - ol_boot["tactical_head"]["mean"])
                    / abs(ol_boot["tactical_head"]["mean"]), 3),
                "operative_rollout_trueA": round(
                    100.0 * (ol_banned["operative_rollout_trueA"]["mean"]
                             - ol_boot["operative_rollout_trueA"]["mean"])
                    / abs(ol_boot["operative_rollout_trueA"]["mean"]), 3),
                "constant_velocity": round(
                    100.0 * (ol_banned["constant_velocity"]["mean"] - cv_corr)
                    / abs(cv_corr), 3)},
            "programme_wide_envelope_pct": [-6.67, 11.69],
            "_verdict": "UNDECIDED without the planner's per-window data — the "
                        "required correction sits INSIDE the programme-wide "
                        "envelope, so no bound from the published heldout mean "
                        "can settle it (JACK_IN_GATES.md §5: the banned "
                        "estimator gives 7 of 40 episodes weight exactly 0, so "
                        "the unweighted windows are unconstrained).",
        },
    }

    # ---------------------------------------------------------------- 4 ----
    # NEW: the CLOSED-LOOP planner-vs-CV comparison, paired. Never computed.
    cl_boot = {k: interval(v, p2["eid"]) for k, v in cl_ade.items()}
    pl_vs_cv = paired(cl_ade["closed_bike"], cl_ade["cv"], p2["eid"])
    op_vs_cv = paired(cl_ade["open_grnd"], cl_ade["cv"], p2["eid"])
    res["4_closedloop_planner_vs_cv_NEW"] = {
        "_what": "the CLOSED-LOOP analogue of planner_beats_cv, PAIRED on the "
                 "same 221 windows. The published G4 compared planner vs HEAD "
                 "and never vs CV, so this comparison has never been made.",
        "_tier": "T1 (action-closed loop: the model is conditioned on its OWN "
                 "actions). EVAL_DOCTRINE.md — this is the PRIMARY tier for a "
                 "capability claim, unlike the T0-flavoured open-loop block.",
        "arms_decision_grade": cl_boot,
        "paired_planner_minus_cv": pl_vs_cv,
        "paired_operative_minus_cv": op_vs_cv,
        "closedloop_planner_beats_cv": bool(
            cl_boot["closed_bike"]["mean"] < cl_boot["cv"]["mean"]),
        "closedloop_planner_beats_cv_ci_separated": bool(
            pl_vs_cv["separated"] and pl_vs_cv["delta"] < 0),
    }

    # ------------------------------------------------- 4b: other verdicts ---
    # C91's rule: report EVERY verdict, not only the one asked about.
    grid = pub["weight_sensitivity"]["grid"]
    pl_lo = min(g["planner_ade2s"] for g in grid)
    pl_hi = max(g["planner_ade2s"] for g in grid)
    hd = grid[0]["head_ade2s"]
    res["4b_every_other_verdict"] = {
        "_rule": "C91 — a verdict nobody listed cannot be re-decided, and its "
                 "silence reads exactly like agreement.",
        "G1_pass / G1...separated": {
            "status": "re-decided 2026-08-16 (JACK_IN_GATES.md), PARTIAL but "
                      "decisive; NOT re-opened here",
            "flip_needs_pct": -73.6,
            "measured_envelope_pct": [-6.909, 11.69],
            "moves_here": False,
            "_note": "independently reproduced in this run: tactical_head "
                     f"{ol_boot['tactical_head']['mean']} and operative "
                     f"{ol_boot['operative_rollout_trueA']['mean']} match "
                     "JACK_IN_GATES to 4 dp.",
        },
        "G4_pass": {
            "status": "fully re-decided 2026-08-16 on banked windows; "
                      "reproduced here",
            "planner_corrected": cl_boot["closed_bike"]["mean"],
            "planner_hi": cl_boot["closed_bike"]["hi"],
            "threshold_corrected": p2m.G4_HEAD_BASELINE_ADE2S,
            "threshold_legacy": p2m.G4_HEAD_BASELINE_ADE2S_LEGACY,
            "pass": bool(cl_boot["closed_bike"]["mean"]
                         < p2m.G4_HEAD_BASELINE_ADE2S),
            "pass_ci_separated": bool(cl_boot["closed_bike"]["hi"]
                                      < p2m.G4_HEAD_BASELINE_ADE2S),
            "moves_here": False,
        },
        "beats_head (x9) + beats_head_all": {
            "_finding": "NOT ESTIMATOR-DEPENDENT AT ANY PLAUSIBLE MAGNITUDE. "
                        "The grid compares planner ADE@2s against a constant "
                        "head ADE@2s on an 8-episode subset; whatever estimator "
                        "produced them, the margin is a RATIO of ~4.7x.",
            "planner_ade2s_range": [pl_lo, pl_hi],
            "head_ade2s": hd,
            "worst_case_ratio": round(hd / pl_hi, 3),
            "flip_needs_planner_to_rise_pct": round(100.0 * (hd - pl_hi) / pl_hi, 1),
            "flip_needs_head_to_fall_pct": round(100.0 * (hd - pl_lo) / hd, 1),
            "measured_envelope_pct": [-6.67, 11.69],
            "moves_here": False,
            "_caveat": "the exact `_sweep` that produced these rows is NOT in "
                       "this repo's git history (`git log -S beats_head -- "
                       "taniteval/taniteval/planner_p2.py` returns nothing — "
                       "the P2 harness was stranded on a pod before it landed), "
                       "so the estimator behind them is UNVERIFIED. It does not "
                       "matter: no measured estimator error is within ~30x of "
                       "closing a 4.7x ratio.",
        },
        "planner_beats_cv": {
            "status": "⛔ UNDECIDED — the subject of this task; see §3",
            "moves_here": False,
            "_why": "the open-loop CEM planner arm is not banked per-window",
        },
    }

    # ---------------------------------------------------------------- 5 ----
    res["5b_four_metric_families_openloop"] = {
        "_binding": "same four-family rule, at the tier where planner_beats_cv "
                    "actually lives (OPEN LOOP, n=881 / 40 episodes).",
        "_tier": "open-loop / true-action-conditioned — T0-flavoured for the "
                 "operative arm. ⛔ NOT a driving-performance claim "
                 "(EVAL_DOCTRINE.md).",
        "_planner_arm": "ABSENT — not banked per-window. This is exactly the "
                        "profile the re-drive would fill in, and it is why the "
                        "four families cannot be closed for the planner at this "
                        "tier without the ~400 s GPU job.",
        "constant_velocity": four_families(ol["cv"], ol["gt"], ol_eid,
                                           "constant_velocity"),
        "operative_rollout_trueA": four_families(ol["pred"], ol["gt"], ol_eid,
                                                 "operative_rollout_trueA"),
        "tactical_head": four_families(cl["plan_direct"], cl["gt"], ol_eid,
                                       "tactical_head"),
    }

    res["5_four_metric_families_closedloop"] = {
        "_binding": "CLAUDE.md — every eval reports LONGITUDINAL / LATERAL / "
                    "TACTICAL / STRATEGIC, per family, never pooled. ADE alone "
                    "is an INCOMPLETE result.",
        "_tier": "T1 closed loop, n=221 / 20 episodes",
        "planner_closed_bike": four_families(p2["closed_bike"], p2["gt"],
                                             p2["eid"], "planner_closed_bike"),
        "constant_velocity": four_families(p2["cv"], p2["gt"], p2["eid"],
                                           "constant_velocity"),
        "operative_rollout_trueA": four_families(p2["open_grnd"], p2["gt"],
                                                 p2["eid"], "open_grnd"),
    }

    out = os.path.join(OUT, "planner_beats_cv_banked_analysis.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"WROTE {out}")

    # ------------------------------------------------------------- print ---
    print("\n=== 1. VERDICT INVENTORY ===")
    print(f"  boolean instances: {len(bools)}   distinct names: {len(distinct)}")
    for d in distinct:
        print(f"    - {d}")
    print("\n=== 2. REPRODUCTION GATE (banned estimator vs published) ===")
    for k, v in repro.items():
        print(f"  {k:14s} pub {v['published_mean']:.4f}+-{v['published_ci95']:.4f}"
              f"  repro {v['recomputed_banned_mean']:.4f}"
              f"+-{v['recomputed_banned_ci95']:.4f}"
              f"  exact={str(v['exact_to_4dp']):5s}"
              f" drift={v['rel_drift_pct']:.4f}%  GATE={v['gate_pass']}")
    print("\n=== 3. OPEN LOOP (the actual planner_beats_cv scope) ===")
    for k, v in ol_boot.items():
        print(f"  {k:26s} banned {ol_banned[k]['mean']:.4f}"
              f"  -> corrected {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
    fr = res["3_openloop_the_actual_verdict"]["flip_requirement"]
    print(f"  planner (banned)           {plan_banned:.4f}  -> NOT BANKED")
    print(f"  CV floor corrected         {cv_corr:.4f}")
    print(f"  planner must drop by       {fr['required_downward_correction_pct']:.3f}%"
          f"  (local envelope "
          f"{min(fr['measured_local_envelope_pct'][k] for k in ('tactical_head','operative_rollout_trueA','constant_velocity')):.3f}"
          f"..{max(fr['measured_local_envelope_pct'][k] for k in ('tactical_head','operative_rollout_trueA','constant_velocity')):.3f}%)")
    print("\n=== 4. CLOSED LOOP planner vs CV — NEW, PAIRED (T1) ===")
    for k, v in cl_boot.items():
        print(f"  {k:14s} {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
    print(f"  paired planner-cv delta {pl_vs_cv['delta']:+.4f} "
          f"[{pl_vs_cv['lo']:+.4f}, {pl_vs_cv['hi']:+.4f}] "
          f"separated={pl_vs_cv['separated']} "
          f"p(d>0)={pl_vs_cv['p_delta_gt0']}")
    print(f"  closedloop_planner_beats_cv = "
          f"{res['4_closedloop_planner_vs_cv_NEW']['closedloop_planner_beats_cv']}")
    print("\n=== 5. FOUR FAMILIES (T1, n=221) ===")
    for arm in ("planner_closed_bike", "constant_velocity",
                "operative_rollout_trueA"):
        ff = res["5_four_metric_families_closedloop"][arm]
        print(f"  {arm}")
        print(f"     LONG  long_rmse@2s {ff['LONGITUDINAL']['per_horizon_long_rmse_m']['2s']:.4f} m"
              f"  speed_err@2s {ff['LONGITUDINAL']['speed_err_2s_mps']:.4f} m/s"
              f"  speed_bias {ff['LONGITUDINAL']['speed_bias_2s_mps']:+.4f} m/s"
              f"  long_frac {ff['LONGITUDINAL']['long_frac_of_2s_sqerr']:.4f}")
        mo = ff["LATERAL"]["_moving_only"]
        print(f"     LAT   crosstrack@2s {ff['LATERAL']['crosstrack_rmse_2s_m']:.4f} m"
              f"  heading@2s {ff['LATERAL']['heading_err_2s_deg']:.3f} deg"
              f"  [moving n={mo['n_used']}] curv {mo['curvature_err_mean_invm']:.5f} 1/m"
              f"  yawrate {mo['yawrate_err_mean_degps']:.3f} deg/s")
        print(f"     TAC   N/A - {ff['TACTICAL']['_reason'][:66]}...")
        print(f"     STRAT N/A - {ff['STRATEGIC']['_reason'][:66]}...")

    print("\n=== 5b. FOUR FAMILIES, OPEN LOOP (n=881) — planner arm ABSENT ===")
    for arm in ("constant_velocity", "operative_rollout_trueA", "tactical_head"):
        ff = res["5b_four_metric_families_openloop"][arm]
        mo = ff["LATERAL"]["_moving_only"]
        print(f"  {arm}")
        print(f"     LONG  long_rmse@2s {ff['LONGITUDINAL']['per_horizon_long_rmse_m']['2s']:.4f} m"
              f"  speed_err@2s {ff['LONGITUDINAL']['speed_err_2s_mps']:.4f} m/s"
              f"  long_frac {ff['LONGITUDINAL']['long_frac_of_2s_sqerr']:.4f}")
        print(f"     LAT   crosstrack@2s {ff['LATERAL']['crosstrack_rmse_2s_m']:.4f} m"
              f"  [moving n={mo['n_used']}] curv {mo['curvature_err_mean_invm']:.5f} 1/m"
              f"  yawrate {mo['yawrate_err_mean_degps']:.3f} deg/s")

    print("\n=== 4b. EVERY OTHER VERDICT ===")
    bh = res["4b_every_other_verdict"]["beats_head (x9) + beats_head_all"]
    print(f"  beats_head: planner {bh['planner_ade2s_range']} vs head "
          f"{bh['head_ade2s']} -> ratio {bh['worst_case_ratio']}x; "
          f"flip needs +{bh['flip_needs_planner_to_rise_pct']}% -> NOT estimator-reachable")
    g4 = res["4b_every_other_verdict"]["G4_pass"]
    print(f"  G4_pass: {g4['planner_corrected']} (hi {g4['planner_hi']}) vs "
          f"{g4['threshold_corrected']} -> pass={g4['pass']} "
          f"ci_separated={g4['pass_ci_separated']}")
    return res


if __name__ == "__main__":
    main()
