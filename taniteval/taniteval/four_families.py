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

DT_S = 0.1               # 10 Hz waypoint cadence — the DENSE grid's spacing
MIN_DS_MPS = 0.5         # below this SPEED a step carries no reliable heading/curvature
MIN_DS_M = MIN_DS_MPS * DT_S    # 0.05 m at 10 Hz — kept as a name for back-compat
_EPS = 1e-8

#: ⛔ THE DEFECT THIS CONSTANT EXISTS TO PREVENT (MEASURED 2026-08-03, Thor, real weights).
#: ``all_families`` reads ``win["pred"]`` / ``win["gt"]``, and for BOTH ``rollout.collect`` and
#: ``refc_eval.collect`` those are the **SPARSE 4-waypoint view at WP_STEPS = (5, 10, 15, 20)** —
#: i.e. **0.5 s spacing**, not 0.1 s. Every derivative here was nevertheless divided by the
#: hard-coded ``DT_S = 0.1``, so every published LONGITUDINAL/LATERAL rate was inflated:
#:
#:     speed   x5      accel   x25      yaw_rate  x5      curvature/heading/positions  correct
#:
#: NEGATIVE CONTROL that proves it, not an argument: on 859 real held-out windows the ego's own
#: recorded speed (``poses[:, 3]``) is **12.4565 m/s** while ``_seq_geometry(gt)["speed"]`` returned
#: **62.9789 m/s** — ratio **5.0559**, and dividing by 5 lands at 12.5958 m/s (**1.1 %** of truth,
#: the residual being chord-vs-instantaneous speed on curves).
#: ⇒ dt is now DERIVED from the window's own ``wp_steps``/``dt_s`` contract and CARRIED IN THE
#: OUTPUT, so a rate can never again be quoted without the grid it was computed on.
_DT_CONTRACT = ("dt is derived from win['wp_steps'] x win.get('dt_s', 0.1); a sparse 4-waypoint "
                "view at WP_STEPS=(5,10,15,20) is a 0.5 s grid, NOT the 0.1 s the module "
                "constant names")


def _seq_geometry(wp: torch.Tensor, dt: float = DT_S):
    """wp [n,H,2] ego-frame metres on a **dt-second grid** -> speed, heading, yaw-rate, curvature.

    ⛔ ``dt`` is the spacing BETWEEN the supplied waypoints, not the model's tick. Passing the
    sparse 4-waypoint view (0.5 s apart) with ``dt=0.1`` inflates every rate — see
    :data:`_DT_CONTRACT`. Callers should use :func:`infer_dt` rather than assume.

    Returns dict of tensors; heading/yaw/curvature are masked where the step displacement is below
    ``MIN_DS_MPS * dt`` (a stopped or crawling vehicle has no meaningful path tangent). The
    threshold SCALES with dt — a fixed 0.05 m gate on a 0.5 s grid excludes essentially nothing and
    silently lets crawling windows into the curvature statistic.
    """
    min_ds = MIN_DS_MPS * dt
    # prepend the origin so step 0 is measured from the ego's own position
    zero = torch.zeros_like(wp[:, :1])
    p = torch.cat([zero, wp], dim=1)                    # [n,H+1,2]
    d = p[:, 1:] - p[:, :-1]                            # [n,H,2] per-step displacement
    ds = torch.linalg.norm(d, dim=-1)                   # [n,H] arc length per step
    speed = ds / dt                                     # m/s

    valid = ds > min_ds
    heading = torch.atan2(d[..., 1], d[..., 0])         # path tangent, radians

    # unwrap along the horizon so a +pi/-pi crossing does not create a fake spike
    dh = heading[:, 1:] - heading[:, :-1]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi       # wrap to (-pi, pi]
    yaw_rate = dh / dt                                  # rad/s
    # curvature = dheading/ds, using the mean arc length of the two steps involved.
    # ⓘ dt-INVARIANT by construction (both dh and ds are geometric), which is why curvature and
    # heading were the only two rate-like metrics the old hard-coded dt did NOT corrupt.
    ds_mid = 0.5 * (ds[:, 1:] + ds[:, :-1])
    curvature = dh / (ds_mid + _EPS)                    # 1/m
    pair_valid = valid[:, 1:] & valid[:, :-1]

    accel = (speed[:, 1:] - speed[:, :-1]) / dt         # m/s^2
    return {"speed": speed, "heading": heading, "valid": valid,
            "yaw_rate": yaw_rate, "curvature": curvature,
            "pair_valid": pair_valid, "accel": accel, "along": p[..., 0][:, 1:],
            "cross": p[..., 1][:, 1:], "dt_s": dt, "min_ds_m": min_ds}


def infer_dt(win: dict) -> tuple[float, str]:
    """-> (dt seconds between the waypoints in ``win['pred']``, provenance string).

    Reads the window dict's OWN sampling contract. ``rollout.collect`` publishes
    ``wp_steps=[5,10,15,20]`` and ``dt_s=0.1``; ``refc_eval``/``refb_eval`` publish ``wp_steps``
    too. The spacing must be uniform AND the first step must equal the spacing, because
    :func:`_seq_geometry` prepends the origin as step 0 — a non-uniform or offset grid would make
    the first displacement mean something different from the rest.

    ⛔ Falls back to :data:`DT_S` ONLY when no contract is present, and says so in the provenance
    so the caller can stamp it. It never guesses silently.
    """
    steps = win.get("wp_steps")
    tick = float(win.get("dt_s", DT_S) or DT_S)
    if not steps:
        return DT_S, (f"NO wp_steps IN WINDOW — assumed dt={DT_S}s. {_DT_CONTRACT}")
    steps = [int(s) for s in steps]
    if len(steps) == 1:
        return steps[0] * tick, f"single wp_step {steps[0]} x dt_s {tick}"
    gaps = {steps[i + 1] - steps[i] for i in range(len(steps) - 1)}
    if len(gaps) != 1:
        return DT_S, (f"NON-UNIFORM wp_steps {steps} — cannot derive a single dt; assumed "
                      f"{DT_S}s. {_DT_CONTRACT}")
    gap = gaps.pop()
    if steps[0] != gap:
        return gap * tick, (f"wp_steps {steps} are uniformly spaced by {gap} ticks but start at "
                            f"{steps[0]} — the prepended origin makes step 0 span "
                            f"{steps[0] * tick}s while the rest span {gap * tick}s; using the "
                            f"spacing. Prefer the dense path for this window shape.")
    return gap * tick, f"derived from wp_steps {steps} x dt_s {tick} -> {gap * tick}s grid"


def _masked(x: torch.Tensor, m: torch.Tensor) -> tuple[float, int]:
    """-> (mean over the masked entries, n kept). Returns (nan, 0) when nothing is valid."""
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float(x[m].mean()), n


def longitudinal(pred: torch.Tensor, gt: torch.Tensor, dt: float = DT_S,
                 lead: dict | None = None) -> dict:
    """Is the arm setting the RIGHT SPEED, and does it keep distance?

    ⛔ ``dt`` is the spacing between the supplied waypoints. Speed scales as 1/dt and acceleration
    as 1/dt², so a wrong dt inflates them by 5x and 25x on the sparse 4-waypoint view — see
    :data:`_DT_CONTRACT`. Positional metrics (``along_*``) are dt-invariant.

    ``lead`` (2026-08-03) supplies the LEAD-AGENT track and turns ``distance_keeping`` on. It is a
    dict with ``leads`` [n,H,2] (lead centres in the SAME window-origin ego frame as ``pred``, NaN
    where absent), ``lead_lens`` [n] and ``speeds`` [n] (ego speed at t0, the time-gap denominator).
    `taniteval.lead_metrics` computes headway / time-gap / min-TTC from it; build it with the
    Architecture & Inference package `2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py`,
    which reads `obstacle.offline` and composes the rig->world->t0 frame chain.

    ⚠️ Without ``lead`` the half of the family Sayed made binding is **UNAVAILABLE** — reported as
    such with its reason, never as a pass. The instrument was admitted by the pre-registered
    GT-vs-CV control **D-LEAD-1** (2026-08-03): Δ min-TTC **+1.7474 s** [1.5813, 1.9218],
    Δ headway **+0.9769 m** [0.883, 1.0758], Δ time-gap **+0.1641 s** [0.1499, 0.1786], paired
    episode-cluster bootstrap over 14,027 windows / 1,431 clip clusters, all separated.
    """
    P, G = _seq_geometry(pred, dt), _seq_geometry(gt, dt)
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
        # --- EGO PROGRESS: along-track distance covered / the human's (2026-08-03, Stream E) ---
        # ⭐ PUBLISHED (arXiv 2605.00066): Ego Progress ALONE is the strongest single predictor of
        # closed-loop Driving Score (rho = 0.83) while traditional L2 gives rho = -0.36, p = 0.43,
        # NOT significant — on n = 8 with no CI, so the DIRECTION only. dt-invariant, so it is
        # immune to the sparse-grid defect that inflated every published speed_* by 5x.
        "ego_progress": _ego_progress(pred, gt),
        # --- the grid these rates were computed on. NEVER quote a rate without it. ---
        "dt_s": dt,
        "rate_scaling_note": ("speed ~ 1/dt, accel ~ 1/dt^2. Numbers computed with a wrong dt are "
                              "off by those powers; along_* are dt-invariant."),
        # --- distance keeping ---
        "distance_keeping": _distance_keeping(pred, dt, lead),
        "n_windows": int(pred.shape[0]),
    }


def _ego_progress(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """``progress.progress`` over the arm's own waypoints — the LONGITUDINAL scalar the
    closed-loop literature says carries the signal.

    Kept in its own numpy-pure module (:mod:`taniteval.progress`) rather than inlined, because the
    CONVENTION is the substance: projecting on the human's own direction makes GT score exactly
    1.0, while pseudosim's published t0-axis reading charges the human for its own curvature. That
    module documents both and reports the size of the difference.
    """
    from taniteval.progress import progress as _progress
    return _progress(pred.detach().cpu().numpy(), gt.detach().cpu().numpy())


def _distance_keeping(pred: torch.Tensor, dt: float, lead: dict | None) -> dict:
    """``lead_metrics.distance_keeping`` over the arm's own predicted path, or the reason why not.

    Kept out of :func:`longitudinal` so the UNAVAILABLE branch reads as one thing: a WORK ITEM with
    a reason and an ``n``, exactly as the binding rule's clause 5 requires.
    """
    if lead is None:
        return {
            "status": "UNAVAILABLE",
            "reason": ("no lead-agent track supplied — pass `lead=` (see this function's caller "
                       "docstring). PhysicalAI-AV ships obstacle.offline (3D agent tracks, "
                       "97.44 % of the corpus); the reader is "
                       "`Architecture & Inference/Implementation/incoming/"
                       "2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py`. "
                       "Not supplying it is a WORK ITEM, not a pass."),
            "n": 0,
        }
    from taniteval.lead_metrics import distance_keeping
    paths = pred.detach().cpu().numpy()
    out = distance_keeping(paths, lead["leads"], lead["lead_lens"], lead["speeds"], dt)
    # per-window arrays are for a paired bootstrap, not for a report — keep them out of the JSON
    # summary but reachable, so nobody re-derives them from a rounded mean.
    out["_per_window"] = {k: out.pop(k) for k in
                          ("headway_min_m", "time_gap_min_s", "min_ttc_s",
                           "n_steps_in_corridor")}
    # ⭐ SPEED-STRATIFIED read (2026-08-03). Emitted whenever the lead block carries the window
    # ids the episode-cluster bootstrap needs. A pooled distance-keeping number averages over
    # regimes that do not resemble each other on this corpus — MEASURED, the tactical lossy rate
    # runs 38.2 % at 1-3 m/s down to 1.8 % at 10-15 m/s — so the pooled value hides the regime
    # that matters. `lead["state"]` (lead_source.LEAD / NO_LEAD / NO_LABEL) is passed through so
    # each stratum reports WHY its denominator is what it is; without it NO_LABEL windows would
    # be indistinguishable from free flow.
    if lead.get("eid") is not None:
        from taniteval.lead_metrics import distance_keeping_by_speed
        out["by_speed"] = distance_keeping_by_speed(
            out, lead["speeds"], lead["eid"], states=lead.get("state"),
            n_boot=int(lead.get("n_boot", 2000)), seed=int(lead.get("seed", 0)))
    else:
        out["by_speed"] = {
            "status": "UNAVAILABLE",
            "reason": ("no per-window episode/clip id in the lead block, so the episode-cluster "
                       "bootstrap cannot be formed. Pass lead['eid']. ⛔ A pooled number is NOT "
                       "a substitute — this corpus's behaviour is strongly speed-dependent."),
            "n": 0,
        }
    out["admitted_by"] = ("D-LEAD-1 discrimination control, 2026-08-03 — GT vs hold-v0 CV, "
                          "PASS on all three metrics, paired episode-cluster bootstrap, "
                          "14,027 windows / 1,431 clusters")
    return out


def lateral(pred: torch.Tensor, gt: torch.Tensor, dt: float = DT_S) -> dict:
    """Heading, curvature, yaw-rate and cross-track — not cross-track alone.

    A path can be smooth and wrong: matching cross-track at the waypoints while turning with the
    wrong curvature. That is invisible to ADE and to cross-track, and it is what these catch.

    ⛔ ``dt`` scales ``yaw_rate`` (1/dt). ``heading``, ``curvature`` and ``cross_*`` are
    dt-invariant — which is why the 2026-08-03 dt defect corrupted exactly one metric in this
    family and left the other three correct. See :data:`_DT_CONTRACT`.
    """
    P, G = _seq_geometry(pred, dt), _seq_geometry(gt, dt)
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
        # ⛔ the gate SCALES with dt now. A fixed 0.05 m on a 0.5 s grid excluded ~nothing and let
        # crawling windows into the curvature statistic.
        "min_ds_m": MIN_DS_MPS * dt,
        "min_ds_mps": MIN_DS_MPS,
        "dt_s": dt,
        "dt_invariant": ["heading_mae_deg", "curvature_*", "cross_*"],
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


def strategic(win: dict, hier: dict | None = None, optionset: dict | None = None) -> dict:
    """Strategic decision + route/goal setting.

    ⭐ **THE OPTION-SET PATH (2026-08-03) — preferred, and it takes precedence.**
    Pass ``optionset`` (or put it on ``win["optionset"]``) and the family is scored against
    **map-derived option sets** by :mod:`taniteval.strategic_optionset`::

        optionset = {"labels": <{scene: strategic_gt report}>,        # load_label_reports(...)
                     "predictions": <{event_id: {"class":…, "road":…}}>,
                     "arm": "refc-xl-30k"}                            # optional

    ⛔ **Why this exists and why it OVERRIDES the legacy block below.** Both legacy paths score a
    route class against a label derived from the ego's own future — which cannot separate *"took
    the left branch"* from *"drifted left on a curving road"*, and **cannot see whether there was
    a branch at all**. That is how the closed-loop harness published
    ``route_head_eq_logged = 1.0000`` on a clip where MEASURED from ``map.xodr`` every junction
    admitted exactly ONE continuation: a constant-predictor tie reported as a perfect score.
    The option-set path refuses to score a single-option junction, states its denominator, and
    compares against the **best constant predictor** by a paired episode-cluster bootstrap.

    ⚠️ GATE_PROTOCOL §0.7: ``nonav_route_beats_majority`` is VOID BY CONSTRUCTION. If a strategic
    number looks impossible, adjudicate **INSTRUMENT-FAIL**, never MODEL-FAIL — a healthy arm has
    already nearly died on that label bug. The void flag is carried in the output so no downstream
    reader can quote that comparison as a model verdict.
    """
    opt = optionset if optionset is not None else win.get("optionset")
    if opt:
        from .strategic_optionset import strategic_family
        out = strategic_family(opt["labels"], opt.get("predictions") or {},
                               arm=opt.get("arm", "arm"),
                               n_boot=opt.get("n_boot", 2000),
                               seed=opt.get("seed", 0),
                               # ⛔ the INPUT-ECHO guard. Without a sweep the family returns
                               # STRATEGIC_SKILL_ADMISSIBLE=None (UNTESTED), which is the
                               # honest state — MEASURED 2026-08-03, flagship-v1's route head
                               # moves with `nav` at 100 % of 6 660 swept poses, and an echo
                               # beats every constant so BEST_CONSTANT cannot catch it.
                               conditioning_sweeps=opt.get("conditioning_sweeps"))
        out["source"] = "strategic_optionset.strategic_family (map.xodr option sets)"
        out["_supersedes"] = (
            "the ego-yaw route label (route_from_future_v21 / seam_nav_to_strategic). Those "
            "cannot see whether a branch EXISTED, which is how route_head_eq_logged reached "
            "1.0000 on a single-option clip.")
        return out
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
    out = _decision_family(win, "strategic", "route_pred", "route_gt", classes)
    if out.get("status") == "UNAVAILABLE":
        out["how_to_populate"] = (
            "supply `optionset` (map-derived option sets from "
            "stack/experiments/nurec-gsplat/strategic_gt.py, consumed by "
            "taniteval.strategic_optionset). A route label read off the ego's own future yaw is "
            "NOT a substitute: it cannot tell whether the map admitted a choice.")
    return out


def all_families(win: dict, hier: dict | None = None, prefer_dense: bool = True,
                 optionset: dict | None = None) -> dict:
    """The full binding block for one arm. Attach to every eval result, beside ADE.

    ``win`` is a ``rollout.collect``/``refb_eval``/``refc_eval`` window dict; ``pred``/``gt`` are
    required. ``hier`` is an optional ``taniteval.hierarchy.run`` result — supply it and the
    TACTICAL and STRATEGIC families are populated instead of reporting UNAVAILABLE.

    ⭐ Pass ``hier``. A fidelity pass alone cannot see a decision error, and the binding rule
    treats an absent family as a work item rather than a pass.

    ⭐ Pass ``optionset`` (2026-08-03) — ``{"labels":…, "predictions":…}`` from
    :mod:`taniteval.strategic_optionset`. It is the ONLY strategic path that can tell a real
    choice from a single-option junction, and it takes precedence over ``hier``. See
    :func:`strategic`.

    ⭐ Pass ``win["lead"]`` too (2026-08-03). Without it the LONGITUDINAL family reports its
    distance-keeping half UNAVAILABLE and ``_complete`` stays False — which is the honest state,
    not a pass. See :func:`longitudinal` for the dict's shape and the D-LEAD-1 admission.

    ⛔ ``prefer_dense`` (default True, changed 2026-08-03). When the window carries the true 10 Hz
    ``pred_dense``/``gt_dense`` path, the rate families are computed on it. That is the grid the
    derivatives were designed for: 20 samples instead of 4, and a genuine 0.1 s tick. When only the
    sparse 4-waypoint view exists, the grid is DERIVED from ``wp_steps`` (see :func:`infer_dt`)
    rather than assumed to be 0.1 s — the defect that inflated every published speed by 5x and
    every acceleration by 25x. Set ``prefer_dense=False`` to reproduce a historical sparse-grid
    number; the grid actually used is always reported in ``_grid``.
    """
    dense = prefer_dense and win.get("pred_dense") is not None \
        and win.get("gt_dense") is not None
    if dense:
        pred = torch.as_tensor(win["pred_dense"]).float()
        gt = torch.as_tensor(win["gt_dense"]).float()
        dt = float(win.get("dt_s", DT_S) or DT_S)
        prov = (f"DENSE path ({tuple(pred.shape)}) at dt_s {dt} — the grid the derivatives are "
                f"defined on")
    else:
        pred = torch.as_tensor(win["pred"]).float()
        gt = torch.as_tensor(win["gt"]).float()
        dt, prov = infer_dt(win)
        prov = f"SPARSE waypoint view; {prov}"
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"expected pred [n,H,2] ego-frame metres, got {tuple(pred.shape)}")
    if pred.shape != gt.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != gt {tuple(gt.shape)}")
    fam = {
        "longitudinal": longitudinal(pred, gt, dt, win.get("lead")),
        "lateral": lateral(pred, gt, dt),
        "tactical": tactical(win, hier),
        "strategic": strategic(win, hier, optionset),
    }
    fam["_grid"] = {
        "used": "dense" if dense else "sparse",
        "dt_s": dt, "horizon_steps": int(pred.shape[1]),
        "provenance": prov,
        "⛔_history": (
            "BEFORE 2026-08-03 this module hard-coded dt=0.1 s while reading the SPARSE "
            "4-waypoint view (0.5 s spacing), so EVERY published speed_* was x5, EVERY accel_* "
            "x25 and EVERY yaw_rate_* x5. Positions, heading and curvature were unaffected. "
            "Cross-arm comparisons stay valid (common factor); ABSOLUTE quotations and any "
            "comparison to a physical bar do not. MEASURED negative control: GT ego speed "
            "12.4565 m/s vs _seq_geometry 62.9789 m/s = 5.0559x, on 859 real held-out windows."),
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
