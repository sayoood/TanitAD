"""Evaluator side of the v8F rung-beta gate (``Project Steering/PREREG_V8F_BETA.md``): per-window family components, plan-level route
adherence, the paired rows, and the pre-registered decision table. Pure numpy / torch, no model.

* :func:`family_windows` -- the per-window scalars behind the binding LONGITUDINAL / LATERAL / TACTICAL families, built on
  ``four_families._seq_geometry`` / ``maneuver_kinematics`` (no second geometry), so a PAIRED episode-cluster bootstrap can be formed.
  ``four_families`` publishes each arm's family block with its own CI but not the per-window values a paired delta needs. The
  pooled statistics are pinned to the family blocks' printed numbers in ``stack/tests/test_v8f_beta.py``.
* :func:`route_adherence` -- the STRATEGIC family read on the PLAN: distance from the plan's point at 30 / 50 / 80 m of arc to the
  GT route checkpoint (v9 RC-A). beta does not move S's own checkpoints; it moves the plan.
* :func:`decide_beta` -- the PREREG table. Both outcomes were committed before any M2 checkpoint existed.
"""
from __future__ import annotations

import math

import numpy as np

from taniteval import ci as _ci

CKPT_ARCS_M = (30.0, 50.0, 80.0)
#: PREREG_V8F_BETA.md §3 -- committed before any M2 checkpoint exists. Changing a number here is a goalpost move.
GATE = {"floor_mult": 2.0,             # |delta ADE6| must exceed this multiple of the seed floor F = |ADE6(M2_s0) - ADE6(M2_s1)|
        "inert_replaced_frac": 0.05,   # beta replaces M2's plan on fewer than this fraction of windows (at BOTH seeds) => the gate is inert
        "signal_recovery": 0.5,        # oracle-goal ranking must recover this fraction of the random-to-ceiling range
        "n_seeds": 2}

#: component -> (family, lower_is_better). The non-regression read (P3) covers every one of them.
COMPONENTS = {
    "lon_speed_mae": ("LONGITUDINAL", True), "lon_accel_mae": ("LONGITUDINAL", True), "lon_along_mae": ("LONGITUDINAL", True),
    "lat_cross_mae": ("LATERAL", True), "lat_heading_mae_deg": ("LATERAL", True),
    "lat_yaw_rate_mae_degps": ("LATERAL", True), "lat_curvature_mae_1pm": ("LATERAL", True),
    "tac_lat_acc": ("TACTICAL", False), "tac_lon_acc": ("TACTICAL", False),
    "strat_route_err_30m": ("STRATEGIC (plan)", True), "strat_route_err_50m": ("STRATEGIC (plan)", True),
    "strat_route_err_80m": ("STRATEGIC (plan)", True),
    "consistency_viol_any": ("CONSISTENCY", True),
}

NEXT_ACTION = {
    "PASS": "beta clears its bar on v8F-mini: rung gamma may be pre-registered (still needs the AR-1..AR-4 readiness gate before any A40 request).",
    "FAIL-INERT": "the gate is inert (beta almost never replaces M2's plan): per plan §6 score in z1 only and re-run; a learned scorer over "
                  "WM-imagined features (paper §3.4) is the next lever. Nothing is trained by this gate.",
    "FAIL-RANKING-DEAD": "z2 is not a usable ranking space even with a PERFECT goal (the oracle-goal arm does not recover the fan's range). "
                         "This is the world model's defect, not the goal's: apply the plan §5/§6 remedies to M2 (IDM at level 2 only, gamma x2, "
                         "the time-repulsion floor) and re-measure H-HJ-COST-1 before any goal head is built.",
    "FAIL-GOAL": "z2 ranks candidates (the oracle goal works) but S's route is not a good enough goal: build the plan's trained g2 head "
                 "(vision + nav + turn distance only, target = the level-2 latent +6 s ahead) under its own PREREG and launch gate.",
    "FAIL-CONTROL": "beta's gain is not attributable to the goal (the shuffled-goal control does not lose): treat it as a vocabulary / "
                    "quantisation effect, not a world-model result; do not promote.",
    "FAIL-NR": "beta improves ADE6 but a family is separated-worse than M2: report the trade-off per family; do not promote on ADE.",
    "UNDECIDABLE": "fewer than 2 training seeds, or a required arm / block is missing: the gate cannot be read.",
}


# ------------------------------------------------------------------------------------------------ per-window components
def family_windows(pred, gt, dt: float = 0.2) -> dict:
    """Per-window family scalars for ``pred`` / ``gt`` [N, H, 2] on a ``dt`` grid. Keys: ``lon_speed_mae`` / ``lon_accel_mae`` /
    ``lon_along_mae`` / ``lat_cross_mae`` (window means of the family block's absolute errors; NaN-free), ``lat_heading_mae_deg``
    (window mean over the steps both paths move; NaN when none), ``lat_yaw_rate_mae_degps`` / ``lat_curvature_mae_1pm`` (the same over
    the step PAIRS both paths move through, as the block masks them), ``lat_heading_sum_rad`` / ``lat_heading_n`` /
    ``lat_yaw_rate_sum_rps`` / ``lat_curvature_sum_1pm`` / ``lat_pair_n`` (the pooled ratios of sums the block prints), ``tac_lat_acc`` / ``tac_lon_acc`` (1.0 where the executed lateral / longitudinal decision equals the human's)."""
    import torch
    from taniteval import four_families as ff
    P, G = torch.as_tensor(np.asarray(pred), dtype=torch.float32), torch.as_tensor(np.asarray(gt), dtype=torch.float32)
    Pg, Gg = ff._seq_geometry(P, dt), ff._seq_geometry(G, dt)
    both = Pg["valid"] & Gg["valid"]
    dh = ((Pg["heading"] - Gg["heading"] + math.pi) % (2 * math.pi) - math.pi).abs()
    n = both.sum(1)
    hsum = (dh * both).sum(1)
    pair = Pg["pair_valid"] & Gg["pair_valid"]                        # the family block masks yaw rate / curvature by step PAIRS
    npair = pair.sum(1)
    ysum = ((Pg["yaw_rate"] - Gg["yaw_rate"]).abs() * pair).sum(1)
    csum = ((Pg["curvature"] - Gg["curvature"]).abs() * pair).sum(1)
    nan = torch.full_like(hsum, float("nan"))
    out = {"lon_speed_mae": (Pg["speed"] - Gg["speed"]).abs().mean(1), "lon_accel_mae": (Pg["accel"] - Gg["accel"]).abs().mean(1),
           "lon_along_mae": (Pg["along"] - Gg["along"]).abs().mean(1), "lat_cross_mae": (Pg["cross"] - Gg["cross"]).abs().mean(1),
           "lat_heading_sum_rad": hsum, "lat_heading_n": n.float(),
           "lat_heading_mae_deg": torch.where(n > 0, hsum / n.clamp_min(1), nan) * (180.0 / math.pi),
           "lat_yaw_rate_sum_rps": ysum, "lat_curvature_sum_1pm": csum, "lat_pair_n": npair.float(),
           "lat_yaw_rate_mae_degps": torch.where(npair > 0, ysum / npair.clamp_min(1), nan) * (180.0 / math.pi),
           "lat_curvature_mae_1pm": torch.where(npair > 0, csum / npair.clamp_min(1), nan)}
    from tanitad.refs.refc_tactical import factor_from_kinematics
    kp, kg = ff.maneuver_kinematics(P, dt), ff.maneuver_kinematics(G, dt)
    lat_p, lon_p = factor_from_kinematics(*kp[:4])
    lat_g, lon_g = factor_from_kinematics(*kg[:4])
    out["tac_lat_acc"], out["tac_lon_acc"] = (lat_p == lat_g).float(), (lon_p == lon_g).float()
    return {k: v.double().numpy() for k, v in out.items()}


def route_adherence(plan_xy, rc, rc_ok) -> np.ndarray:
    """[N, 3]: ||plan point at 30 / 50 / 80 m of ARC - GT route checkpoint||, NaN where the plan is shorter than the arc or the checkpoint
    has no label. ``plan_xy`` [N, 30, 2] ego frame from the origin; ``rc`` [N, 3, >=2]; ``rc_ok`` [N, 3] bool."""
    plan = np.asarray(plan_xy, np.float64)
    n = plan.shape[0]
    pts = np.concatenate([np.zeros((n, 1, 2)), plan], 1)
    s = np.concatenate([np.zeros((n, 1)), np.cumsum(np.linalg.norm(np.diff(pts, axis=1), axis=-1), 1)], 1)       # [N, 31]
    ar = np.arange(n)
    out = np.full((n, 3), np.nan)
    for m, arc in enumerate(CKPT_ARCS_M):
        reach = s[:, -1] >= arc
        k = np.argmax(s >= arc, axis=1).clip(1, s.shape[1] - 1)
        s0, s1 = s[ar, k - 1], s[ar, k]
        f = np.clip((arc - s0) / np.maximum(s1 - s0, 1e-9), 0.0, 1.0)
        pt = pts[ar, k - 1] + f[:, None] * (pts[ar, k] - pts[ar, k - 1])
        err = np.linalg.norm(pt - np.asarray(rc, np.float64)[:, m, :2], axis=-1)
        out[:, m] = np.where(reach & np.asarray(rc_ok)[:, m].astype(bool), err, np.nan)
    return out


# ------------------------------------------------------------------------------------------------ paired rows
def paired_row(a, b, eid, n_boot: int = 2000, lower_is_better: bool = True, seed: int = 0) -> dict:
    """``mean(a) - mean(b)`` with the PAIRED episode-cluster bootstrap (same resampled episodes for both arms) over the windows where
    both are finite. ``worse`` / ``better`` already account for the metric's direction."""
    a, b, eid = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or len(set(eid[ok].tolist())) < 2:
        return {"status": "UNAVAILABLE", "reason": "fewer than 2 finite paired windows / episodes", "n_windows": int(ok.sum())}
    r = _ci.paired_episode_cluster_bootstrap(a[ok], b[ok], eid[ok], n_boot=n_boot, seed=seed)
    sep, d = bool(r["separated"]), float(r["delta"])
    return {"mean": d, "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": sep, "lower_is_better": bool(lower_is_better),
            "better": bool(sep and (d < 0 if lower_is_better else d > 0)), "worse": bool(sep and (d > 0 if lower_is_better else d < 0)),
            "n_windows": int(r["n_windows"]), "n_episodes": int(r["n_episodes"]), "estimator": r["estimator"]}


# ------------------------------------------------------------------------------------------------ the decision
def signal_recovery(rand: float, oracle: float, ceiling: float) -> float:
    """(random - oracle-goal beta) / (random - GT-best fan member), fan-only ranking: 1.0 = the oracle goal finds the fan's best member,
    0.0 = no better than a random pick, < 0 = worse than a random pick. NaN when the fan has no range."""
    den = rand - ceiling
    return float("nan") if not den > 1e-9 else float((rand - oracle) / den)


def decide_beta(seeds: list, m2_ade6: list) -> dict:
    """The pre-registered verdict from the per-seed records ``seeds`` (each: ``ade6_beta_minus_m2``, ``ade6_shuf_minus_beta``,
    ``families`` {component: paired_row (beta - M2)}, ``replaced_frac``, ``signal`` {``recovery``, ``oracle_minus_random`` paired_row})
    and the two seeds' M2 ADE6 means (the floor F = |m2_s0 - m2_s1|). Order: INERT, RANKING-DEAD, GOAL, CONTROL, NR, PASS."""
    if len(seeds) < GATE["n_seeds"] or len(m2_ade6) < GATE["n_seeds"]:
        return {"verdict": "UNDECIDABLE", "next": NEXT_ACTION["UNDECIDABLE"], "n_seeds": len(seeds)}
    F = float(abs(m2_ade6[0] - m2_ade6[1]))
    thr = GATE["floor_mult"] * F
    p1 = [bool(r["ade6_beta_minus_m2"].get("better")) and abs(r["ade6_beta_minus_m2"]["mean"]) > thr for r in seeds]
    p2 = [bool(r["ade6_shuf_minus_beta"].get("worse")) for r in seeds]       # shuffled - beta > 0, separated: the control LOSES
    worse = sorted({n for r in seeds for n, row in r["families"].items() if row.get("worse")})
    inert = all(r["replaced_frac"] < GATE["inert_replaced_frac"] for r in seeds)
    sig = [bool(r["signal"]["oracle_minus_random"].get("better")) and (r["signal"]["recovery"] >= GATE["signal_recovery"])
           for r in seeds]
    p3 = not worse
    if inert:
        v = "FAIL-INERT"
    elif not all(sig):
        v = "FAIL-RANKING-DEAD"
    elif not all(p1):
        v = "FAIL-GOAL"
    elif not all(p2):
        v = "FAIL-CONTROL"
    elif not p3:
        v = "FAIL-NR"
    else:
        v = "PASS"
    return {"verdict": v, "next": NEXT_ACTION[v], "seed_floor_F_m": F, "threshold_2F_m": thr,
            "P1_ade6_beats_m2_separated_beyond_2F": p1, "P2_shuffled_goal_control_loses": p2,
            "P3_no_family_separated_worse": p3, "families_separated_worse": worse,
            "signal_ok_per_seed": sig, "inert": inert,
            "replaced_frac": [r["replaced_frac"] for r in seeds], "gate": GATE}
