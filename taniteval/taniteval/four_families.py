"""TanitEval — the FOUR METRIC FAMILIES that every eval must report alongside ADE.

⛔ **BINDING (Sayed, 2026-08-02).** *"don't consider only ADE at the different horizons… I want you
to ADD (not replace)… Any future eval must include these metrics and this is binding."*

ADE is a scalar average over waypoint distances. An arm can **win ADE while setting the wrong
speed**, while choosing the **wrong manoeuvre**, or with **no working strategic level at all** —
none of which ADE can see. 88.7 % of our measured oracle gap is longitudinal, and our largest known
architectural defect is a 5-way softmax that MIXES lateral and longitudinal decisions. Neither is
visible in an ADE column.

The four families, ADDED to ADE, never replacing it:

    LONGITUDINAL   right speed, and keeping distance
    LATERAL        heading, curvature, yaw-rate, cross-track
    TACTICAL       manoeuvre decision + tactical goal setting
    STRATEGIC      strategic decision + route/goal setting

## What this module computes, and from what

Everything in LONGITUDINAL and LATERAL is derived from the **waypoint sequences already collected**
(``win["pred"]`` / ``win["gt"]``, ego-frame metres at 10 Hz), so it costs no extra forward pass and
applies to every arm that produces a trajectory.

⚠️ **TACTICAL and STRATEGIC need the model's DECODED DECISIONS**, which a pure world-model fidelity
pass does not produce — ``run_one`` prints exactly that: *"the scored pass did NOT traverse the
hierarchy (missing: ['strategic','tactical','operative_intent'])"*. When those are absent this
module reports the family as **UNAVAILABLE with the reason and n**, which the binding rule requires,
rather than silently dropping it. ⛔ **A family reported as unavailable is a work item, not a pass.**

## Frame and sign conventions, stated because they are easy to get wrong

Waypoints are **ego-frame** at the window's last observed pose: ``x`` forward (along-track),
``y`` left (cross-track), in **metres**, one row per 0.1 s step. Therefore:

* ``along`` error is signed ``x_pred - x_gt``: **positive = the model is AHEAD of the human**
  (over-predicted speed). We already know v1 over-predicts by +0.66 m/s at high speed, and the sign
  is the whole point — an unsigned longitudinal error hides whether we brake late or early.
* ``heading`` is ``atan2(dy, dx)`` of successive waypoints, so it is the **path tangent**, not the
  vehicle yaw. On a 2 s horizon at low speed the two diverge; that caveat travels with the number.
* ``curvature`` is ``dheading / ds``. ⚠️ It is **undefined when the vehicle is stopped** (``ds→0``)
  and explodes near it, so steps below ``MIN_DS_M`` are excluded and **counted** — the count is
  reported, because a curvature number computed over a silently-shrinking subset is not comparable
  across arms.
"""
from __future__ import annotations

import math

import torch

DT_S = 0.1               # 10 Hz waypoint cadence
MIN_DS_M = 0.05          # below this a step carries no reliable heading/curvature
_EPS = 1e-8


def _seq_geometry(wp: torch.Tensor):
    """wp [n,H,2] ego-frame metres -> per-step speed, heading, yaw-rate, curvature.

    Returns dict of tensors; heading/yaw/curvature are masked where the step displacement is
    below ``MIN_DS_M`` (a stopped or crawling vehicle has no meaningful path tangent).
    """
    # prepend the origin so step 0 is measured from the ego's own position
    zero = torch.zeros_like(wp[:, :1])
    p = torch.cat([zero, wp], dim=1)                    # [n,H+1,2]
    d = p[:, 1:] - p[:, :-1]                            # [n,H,2] per-step displacement
    ds = torch.linalg.norm(d, dim=-1)                   # [n,H] arc length per step
    speed = ds / DT_S                                   # m/s

    valid = ds > MIN_DS_M
    heading = torch.atan2(d[..., 1], d[..., 0])         # path tangent, radians

    # unwrap along the horizon so a +pi/-pi crossing does not create a fake spike
    dh = heading[:, 1:] - heading[:, :-1]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi       # wrap to (-pi, pi]
    yaw_rate = dh / DT_S                                # rad/s
    # curvature = dheading/ds, using the mean arc length of the two steps involved
    ds_mid = 0.5 * (ds[:, 1:] + ds[:, :-1])
    curvature = dh / (ds_mid + _EPS)                    # 1/m
    pair_valid = valid[:, 1:] & valid[:, :-1]

    accel = (speed[:, 1:] - speed[:, :-1]) / DT_S       # m/s^2
    return {"speed": speed, "heading": heading, "valid": valid,
            "yaw_rate": yaw_rate, "curvature": curvature,
            "pair_valid": pair_valid, "accel": accel, "along": p[..., 0][:, 1:],
            "cross": p[..., 1][:, 1:]}


def _masked(x: torch.Tensor, m: torch.Tensor) -> tuple[float, int]:
    """-> (mean over the masked entries, n kept). Returns (nan, 0) when nothing is valid."""
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float(x[m].mean()), n


def longitudinal(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Is the arm setting the RIGHT SPEED, and does it keep distance?

    ⚠️ ``distance_keeping`` requires a LEAD-AGENT track. PhysicalAI-AV ships
    ``obstacle.offline`` (3D agent tracks on 97.44 % of the corpus) but our episode ingest does not
    read it, so headway/TTC is **UNAVAILABLE** here and is reported as such with the reason.
    """
    P, G = _seq_geometry(pred), _seq_geometry(gt)
    sp_err = P["speed"] - G["speed"]
    al_err = P["along"] - G["along"]
    ac_err = P["accel"] - G["accel"]
    return {
        # --- speed setting ---
        "speed_mae_mps": round(float(sp_err.abs().mean()), 4),
        "speed_bias_mps": round(float(sp_err.mean()), 4),          # + = too fast
        "speed_rmse_mps": round(float((sp_err ** 2).mean().sqrt()), 4),
        # --- along-track position, signed so late/early braking is visible ---
        "along_mae_m": round(float(al_err.abs().mean()), 4),
        "along_bias_m": round(float(al_err.mean()), 4),            # + = ahead of the human
        "along_final_bias_m": round(float(al_err[:, -1].mean()), 4),
        # --- acceleration profile ---
        "accel_mae_mps2": round(float(ac_err.abs().mean()), 4),
        # --- distance keeping ---
        "distance_keeping": {
            "status": "UNAVAILABLE",
            "reason": ("no lead-agent track in the episode cache — PhysicalAI-AV ships "
                       "obstacle.offline (3D agent tracks, 97.44 % of the corpus) but our "
                       "ingest does not read it. Implementing it is a WORK ITEM, not a pass."),
            "n": 0,
        },
        "n_windows": int(pred.shape[0]),
    }


def lateral(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Heading, curvature, yaw-rate and cross-track — not cross-track alone.

    A path can be smooth and wrong: matching cross-track at the waypoints while turning with the
    wrong curvature. That is invisible to ADE and to cross-track, and it is what these catch.
    """
    P, G = _seq_geometry(pred), _seq_geometry(gt)
    both = P["valid"] & G["valid"]
    both_pair = P["pair_valid"] & G["pair_valid"]

    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    head_mae, n_head = _masked(dh.abs(), both)
    yaw_mae, _ = _masked((P["yaw_rate"] - G["yaw_rate"]).abs(), both_pair)
    curv_mae, n_curv = _masked((P["curvature"] - G["curvature"]).abs(), both_pair)
    curv_bias, _ = _masked(P["curvature"] - G["curvature"], both_pair)

    ct_err = P["cross"] - G["cross"]
    return {
        "heading_mae_deg": round(math.degrees(head_mae), 4) if n_head else None,
        "yaw_rate_mae_degps": round(math.degrees(yaw_mae), 4) if n_head else None,
        "curvature_mae_1pm": round(curv_mae, 6) if n_curv else None,
        "curvature_bias_1pm": round(curv_bias, 6) if n_curv else None,
        "cross_mae_m": round(float(ct_err.abs().mean()), 4),
        "cross_bias_m": round(float(ct_err.mean()), 4),            # + = drifts LEFT of the human
        "cross_final_mae_m": round(float(ct_err[:, -1].abs().mean()), 4),
        # transparency: how much of the horizon was usable
        "n_steps_heading": n_head,
        "n_steps_curvature": n_curv,
        "excluded_below_min_ds": int((~both).sum()),
        "min_ds_m": MIN_DS_M,
        "n_windows": int(pred.shape[0]),
    }


def _decision_family(win: dict, level: str, pred_key: str, gt_key: str,
                     classes=None) -> dict:
    """Shared shape for TACTICAL and STRATEGIC: accuracy + per-class confusion, or an honest
    UNAVAILABLE carrying the reason and n. ⛔ Never silently omitted."""
    p, g = win.get(pred_key), win.get(gt_key)
    if p is None or g is None:
        missing = [k for k, v in ((pred_key, p), (gt_key, g)) if v is None]
        return {
            "status": "UNAVAILABLE",
            "reason": (f"{level} decisions not present in the scored pass (missing "
                       f"{missing}). A world-model FIDELITY pass does not traverse the "
                       f"hierarchy — run_one prints this explicitly. Producing this family "
                       f"needs a hierarchy-traversing eval, which is a WORK ITEM."),
            "n": 0,
        }
    p = torch.as_tensor(p).flatten()
    g = torch.as_tensor(g).flatten()
    n = int(min(p.numel(), g.numel()))
    p, g = p[:n], g[:n]
    correct = (p == g)
    out = {"status": "OK", "n": n,
           "accuracy": round(float(correct.float().mean()), 4)}
    labels = sorted(set(g.tolist()) | set(p.tolist()))
    per = {}
    for c in labels:
        sel = g == c
        n_c = int(sel.sum())
        name = classes[c] if classes and c < len(classes) else str(c)
        per[name] = {
            "n_true": n_c,
            "recall": round(float(correct[sel].float().mean()), 4) if n_c else None,
            "n_pred": int((p == c).sum()),
        }
    out["per_class"] = per
    # ⭐ the class the programme cares about most: never-predicted classes are a silent failure
    out["never_predicted"] = [k for k, v in per.items()
                              if v["n_true"] > 0 and v["n_pred"] == 0]
    return out


def tactical(win: dict, hier: dict | None = None) -> dict:
    """Manoeuvre decision + tactical goal setting.

    ⭐ Why ``never_predicted`` is surfaced: our measured longitudinal failure is that the 5-way
    manoeuvre softmax MIXES lateral and longitudinal classes, and the arm emitted **0 of 881**
    'accelerate' decisions. An accuracy scalar hides a class the model never chooses; that list
    does not.

    ``hier`` is a ``taniteval.hierarchy.run`` result. The hierarchy pass DOES traverse the brains,
    so when it is supplied the family is populated from it instead of reporting UNAVAILABLE.
    """
    if hier and not hier.get("skipped"):
        cons = hier.get("consistency", {}) or {}
        mvt = cons.get("maneuver_vs_trajectory", {}) or {}
        seams = (hier.get("thesis_read", {}) or {}).get(
            "A_conditioning_helps_conditioned_layer", {}) or {}
        h18 = hier.get("h18_grounded_vs_ungrounded", {}) or {}
        out = {
            "status": "OK",
            "source": "hierarchy.run",
            # does the DECLARED manoeuvre match the trajectory actually driven?
            "maneuver_vs_trajectory_kappa": mvt.get("kappa"),
            "maneuver_vs_trajectory_agreement": mvt.get("agreement"),
            # is the tactical layer's conditioning load-bearing at all?
            "seams_beneficial_of_3": seams.get("n_of_3_seams_beneficial"),
            "seam_verdict": seams.get("verdict"),
            # H18: grounded operative rollout vs the ungrounded tactical head
            "grounded_op_rollout_ade_2s": h18.get("grounded_op_rollout_ade_2s"),
            "ungrounded_tactical_head_ade_2s": h18.get(
                "ungrounded_tactical_head_ade_2s"),
            "n_windows": hier.get("n_windows"),
        }
        # ⛔ κ near 0 means the declared manoeuvre and the driven path are unrelated — a decision
        # error a scalar ADE cannot see. Surface it as a verdict, not a bare number.
        k = mvt.get("kappa")
        if isinstance(k, (int, float)):
            out["maneuver_consistency_verdict"] = (
                "DECORATIVE — declared manoeuvre is ~unrelated to the driven path" if k < 0.1
                else "WEAK" if k < 0.4 else "SUBSTANTIAL")
        return out
    try:
        from tanitad.refs.refb import MANEUVER_CLASSES as _MC
        classes = list(_MC)
    except Exception:
        classes = None
    return _decision_family(win, "tactical", "maneuver_pred", "maneuver_gt", classes)


def strategic(win: dict, hier: dict | None = None) -> dict:
    """Strategic decision + route/goal setting.

    ⚠️ GATE_PROTOCOL §0.7: ``nonav_route_beats_majority`` is VOID BY CONSTRUCTION. If a strategic
    number looks impossible, adjudicate **INSTRUMENT-FAIL**, never MODEL-FAIL — a healthy arm has
    already nearly died on that label bug. The void flag is carried in the output so no downstream
    reader can quote that comparison as a model verdict.
    """
    if hier and not hier.get("skipped"):
        # ⚠️ VERIFIED key name, not guessed: hierarchy.py:857 stores the route block as
        # "seam_nav_to_strategic". A wrong key here would return None for every strategic
        # metric and read as "the model has no route skill" — a silent instrument failure.
        r = hier.get("seam_nav_to_strategic") or {}
        out = {
            "status": "OK",
            "source": "hierarchy.run",
            # route/goal setting under three conditioning regimes
            "route_acc_nav": r.get("route_acc_nav"),
            "route_acc_follow": r.get("route_acc_follow"),
            "route_acc_zeronav": r.get("route_acc_zeronav"),
            # the two baselines any route number must be read against
            "majority_straight_rate": r.get("majority_straight_rate"),
            "chance_1_of_3": r.get("chance_1_of_3"),
            "follow_pred_distribution": r.get("follow_pred_distribution"),
            # paired contrasts (episode-cluster bootstrap inside hierarchy.py)
            "delta_nav_vs_follow": r.get("delta_nav_vs_follow"),
            "delta_nav_vs_zeronav": r.get("delta_nav_vs_zeronav"),
            "n_valid": r.get("n_valid"),
            "⛔_void_by_construction": (
                "GATE_PROTOCOL §0.7 — `nonav_route_beats_majority` is VOID BY CONSTRUCTION. "
                "If route accuracy looks impossible, adjudicate INSTRUMENT-FAIL, never "
                "MODEL-FAIL. `route_skill_vs_majority` is NOT admissible as a model verdict."),
        }
        # ⛔ THE BASELINE COMPARISON MUST USE route_acc_FOLLOW, NOT route_acc_nav.
        # `route_acc_nav` feeds the model the NAV COMMAND — the answer is an input, so a value
        # near 1.0 measures COPYING, not route reasoning. (MEASURED 2026-08-02: v1 scores
        # route_acc_nav = 1.0000 while its vision-only route_acc_follow = 0.9474, which is
        # EXACTLY the majority-straight rate — i.e. it predicts "straight" always.)
        # This mirrors hierarchy.py:658's own `vision_route_beats_majority`, margin included.
        acc, maj = r.get("route_acc_follow"), r.get("majority_straight_rate")
        if isinstance(acc, (int, float)) and isinstance(maj, (int, float)):
            out["beats_majority_baseline"] = bool(acc > maj + 0.03)
            out["_baseline_used"] = ("route_acc_follow (vision-only) vs majority_straight_rate, "
                                     "margin 0.03 — route_acc_nav is PRIVILEGED and is reported "
                                     "for reference only, never as the skill test")
            out["_reading"] = (
                "vision-only route accuracy does NOT beat always-predict-straight — the "
                "strategic layer is not demonstrably doing route work"
                if acc <= maj + 0.03 else
                "vision-only route accuracy exceeds the majority baseline")
        return out
    try:
        from tanitad.refs.refb import ROUTE_CLASSES as _RC
        classes = list(_RC)
    except Exception:
        classes = None
    return _decision_family(win, "strategic", "route_pred", "route_gt", classes)


def all_families(win: dict, hier: dict | None = None) -> dict:
    """The full binding block for one arm. Attach to every eval result, beside ADE.

    ``win`` is a ``rollout.collect``/``refb_eval``/``refc_eval`` window dict; ``pred``/``gt`` are
    required. ``hier`` is an optional ``taniteval.hierarchy.run`` result — supply it and the
    TACTICAL and STRATEGIC families are populated instead of reporting UNAVAILABLE.

    ⭐ Pass ``hier``. A fidelity pass alone cannot see a decision error, and the binding rule
    treats an absent family as a work item rather than a pass.
    """
    pred = torch.as_tensor(win["pred"]).float()
    gt = torch.as_tensor(win["gt"]).float()
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"expected pred [n,H,2] ego-frame metres, got {tuple(pred.shape)}")
    if pred.shape != gt.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != gt {tuple(gt.shape)}")
    fam = {
        "longitudinal": longitudinal(pred, gt),
        "lateral": lateral(pred, gt),
        "tactical": tactical(win, hier),
        "strategic": strategic(win, hier),
    }
    unavailable = [k for k, v in fam.items()
                   if isinstance(v, dict) and v.get("status") == "UNAVAILABLE"]
    fam["_binding_rule"] = (
        "Sayed 2026-08-02: every eval reports LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC "
        "in ADDITION to ADE. Per-family, never pooled. A family reported UNAVAILABLE is a WORK "
        "ITEM, not a pass.")
    fam["_families_unavailable"] = unavailable
    fam["_complete"] = not unavailable and \
        fam["longitudinal"]["distance_keeping"]["status"] == "OK"
    return fam
