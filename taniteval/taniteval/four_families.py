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

#: Tolerance bands for TARGET-SPEED ACCURACY. ⚠️ PROPOSED reporting tolerances,
#: not a gate and not measured from anything — they exist so the LONGITUDINAL
#: family answers "how often is the speed right", which is what the binding rule
#: names, instead of only "how wrong is it on average".
TARGET_SPEED_BANDS_MPS = (0.5, 1.0, 2.0)

DT_S = 0.1               # 10 Hz waypoint cadence — the DENSE grid's spacing
MIN_DS_MPS = 0.5         # below this SPEED a step carries no reliable heading/curvature
MIN_DS_M = MIN_DS_MPS * DT_S    # 0.05 m at 10 Hz — kept as a name for back-compat

# --------------------------------------------------------------------------- #
# ⛔ THE COHERENCE-VERDICT LADDER — ONE DEFINITION, BECAUSE IT IS PUBLISHED     #
# --------------------------------------------------------------------------- #
#: Cohen's-κ bands for the manoeuvre-coherence verdict WORD. This ladder is the
#: PUBLISHED one: it is the source of every coherence adjective the programme
#: has quoted.
#:
#:   ``MODEL_REGISTRY.md:1111``  — "κ **0.6033** (SUBSTANTIAL)" (the v1arch row)
#:   ``Paper/TANITAD_PAPER.md``  — "TACTICAL — κ 0.6033 (substantial …)"
#:   ``V5_FLAGSHIP_DEEP_REVIEW.md:74`` — "0.253 WEAK | 0.0072 DECORATIVE"
#:   ``stack/scripts/v5_guard.py:216-217`` — the v5 GPU-spend guard, same bands
#:
#: ⛔ IT MUST NOT BE RESTATED. It was restated once — ``hierarchy._gate_sensitivity``
#: tested ``verdict_stable`` against a bare ``κ >= 0.2`` that appears in NO
#: published ladder — and the two drifted into contradicting each other on real
#: artifacts: ``hier_v1-lf19.json`` emitted κ 0.253 as *"SUPPORTED … cohere"*
#: while ``V5_FLAGSHIP_DEEP_REVIEW.md:74`` published the SAME κ 0.253 as
#: **WEAK**. A field named for a verdict must test the verdict that ships, or it
#: is an instrument that cannot fail when the published claim is wrong.
#: (Found 2026-08-15, `…/2026-08-15-dir-yaw-gate-reread/`; fixed 2026-08-16,
#: `…/2026-08-16-verdict-stable-kappa/`.)
#:
#: Read as "the first band whose upper bound the κ falls under".
KAPPA_VERDICT_LADDER = ((0.1, "DECORATIVE"), (0.4, "WEAK"),
                        (float("inf"), "SUBSTANTIAL"))

#: The long published gloss, kept byte-identical to what banked artifacts carry.
_KAPPA_BAND_GLOSS = {
    "DECORATIVE": "DECORATIVE — declared manoeuvre is ~unrelated to the driven path",
}


def kappa_band(k):
    """Short band NAME for a Cohen's κ — ``DECORATIVE`` / ``WEAK`` / ``SUBSTANTIAL``.

    ``None`` for a κ that is not computable (undefined kappa, NaN), so a caller
    cannot mistake "not measurable" for "no agreement" — the same distinction
    ``hierarchy._kappa`` makes by returning ``None`` when ``pe == 1``.

    This is the comparison key for verdict STABILITY: two κ values agree iff
    they land in the same band. Comparing raw κ against a single cut answers a
    different question than the ladder does, which is exactly the defect this
    module-level constant exists to prevent.
    """
    if k is None:
        return None
    try:
        k = float(k)
    except (TypeError, ValueError):
        return None
    if math.isnan(k):
        return None
    for hi, name in KAPPA_VERDICT_LADDER:
        if k < hi:
            return name
    return KAPPA_VERDICT_LADDER[-1][1]


def kappa_verdict(k):
    """The PUBLISHED verdict STRING for a κ (band name, plus its gloss if any).

    ``kappa_band`` is what you compare; this is what you print. Byte-identical
    to the string this module has published since the four-family panel landed,
    so banked artifacts stay comparable."""
    band = kappa_band(k)
    return None if band is None else _KAPPA_BAND_GLOSS.get(band, band)
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
        # ⭐ TARGET-SPEED ACCURACY in the literal sense the binding rule names:
        # the FRACTION of steps whose speed is inside a tolerance band. MAE and
        # RMSE are dominated by the tail, so an arm can carry a respectable MAE
        # while rarely being at the right speed at all — a distinction an error
        # magnitude cannot express, and the one the word "accuracy" asks for.
        "target_speed_acc": {
            f"within_{b}_mps": round(float((sp_err.abs() <= b).float().mean()), 4)
            for b in TARGET_SPEED_BANDS_MPS},
        "target_speed_acc_note": (
            f"fraction of the {int(pred.shape[0] * pred.shape[1])} horizon STEPS "
            f"(not windows) whose |speed error| is within the band. Bands are "
            f"PROPOSED reporting tolerances, not a gate."),
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
    dt_dk = dt
    # ⛔ THE LEAD TRACK IS OFTEN ON A COARSER TIME GRID THAN THE PATH. The banked
    # val40 block samples the lead at ts_rel_s = (0.5, 1.0, 1.5, 2.0) while a
    # dense arm path runs at 0.1 s, so K differs and `distance_keeping` rightly
    # refuses the shape. The fix is a TIME join, never a truncation: the block
    # declares which steps of the path it is defined on, those steps are
    # selected, and the spacing BETWEEN THEM becomes the dt the closing rate (and
    # therefore TTC) is computed with. Truncating to the first K steps instead
    # would silently score a 2 s lead track against 0.4 s of path.
    steps = lead.get("path_steps")
    if steps is not None:
        idx = [int(s) for s in steps]
        if max(idx) >= paths.shape[1] or min(idx) < 0:
            return {"status": "UNAVAILABLE",
                    "reason": (f"lead block declares path_steps {idx} but the "
                               f"path has only {paths.shape[1]} steps — refusing "
                               f"a join that would index off the grid"),
                    "n": 0}
        paths = paths[:, idx]
        dt_dk = float(lead.get("dt_s") or (dt * (idx[1] - idx[0]) if len(idx) > 1
                                           else dt))
    out = distance_keeping(paths, lead["leads"], lead["lead_lens"],
                           lead["speeds"], dt_dk)
    if steps is not None:
        out["path_steps"] = [int(s) for s in steps]
        out["_time_join"] = (
            f"the lead track is on a COARSER grid than the path; scored on the "
            f"path's steps {list(steps)} at dt {dt_dk}s (the spacing BETWEEN the "
            f"lead's own samples), NOT on a truncated prefix")
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


# ============================================================================ #
# TACTICAL from a TRAJECTORY-ONLY dump (2026-08-11)                            #
#                                                                              #
# ⛔ THE GAP THIS CLOSES. Before today TACTICAL had exactly two inputs: a       #
# ``hierarchy.run`` result (needs the MODEL and a trained tactical brain) or a  #
# pre-decoded ``win["maneuver_pred"]/["maneuver_gt"]``. A **T1 action-closed    #
# dump has neither** — it carries trajectories only — so every T1 number the    #
# programme has produced reported TACTICAL ``UNAVAILABLE``. Under the binding   #
# rule that is a WORK ITEM, and this is the work.                              #
#                                                                              #
# ⭐ WHY A TRAJECTORY IS A LEGITIMATE INPUT **AT T1, AND ONLY AT T1.** In the   #
# action-closed loop the predictor consumes the planner's OWN actions, so the   #
# rolled path IS the model's tactical decision made manifest — there is no      #
# recorded future steering it. Comparing the arm's EXECUTED manoeuvre against   #
# the human's EXECUTED manoeuvre is therefore a real decision comparison.       #
# ⚠️ At **T0** the path is teacher-forced by the RECORDED actions, so a         #
# trajectory-derived manoeuvre is substantially an echo of the label's own      #
# source — the §1.12 action-echo defect in another costume. The block stamps    #
# the tier and carries that warning; the caller must not read a T0 agreement    #
# number as tactical skill.                                                    #
#                                                                              #
# ⛔ WHAT IT IS **NOT**: it is NOT "selected vs executed". That contrast needs   #
# a DECLARED decision from a tactical head (``hierarchy.run``'s               #
# ``maneuver_vs_trajectory`` kappa) and stays unavailable on a dump. Stated in  #
# the output rather than blurred, because a decision the model never declared   #
# cannot be scored against the path it drove.                                   #
#                                                                              #
# ⛔ NO NEW THRESHOLDS. The classifier is the programme's own canonical         #
# ``tanitad.refs.refc_tactical.factor_from_kinematics`` — the FACTORED labeller #
# (lat 3-way x lon 3-way, collapsed to the legacy 5-way by its own             #
# ``COLLAPSE_TABLE``). Inventing a manoeuvre gate here would have been a second #
# implementation that drifts from the trainer's.                                #
#                                                                              #
# ⭐ AND THE FACTORED READ IS THE POINT, not a convenience: CLAUDE.md names     #
# "the 5-way softmax that MIXES lat+lon" as the programme's single largest      #
# known architectural defect. Reporting LAT and LON agreement SEPARATELY beside #
# the collapsed 5-way is the only way the mixing is visible in a metric.        #
# ============================================================================ #

#: Cohen's kappa is reported beside every accuracy because accuracy alone is
#: unreadable on this corpus's class balance — ``lane_keep`` dominates, so a
#: constant predictor scores high. kappa is the agreement ABOVE that chance.
_KAPPA_NOTE = ("accuracy on this corpus is dominated by the majority class; "
               "kappa is the agreement above chance and is the readable number")


def _kappa_k(a, b, k: int):
    """Cohen's kappa over ``k`` classes. ``None`` when it is undefined.

    ⛔ Generalised rather than reusing ``hierarchy._kappa``, which is hard-coded
    to ``{0,1,2}`` and would silently ignore classes 3/4 of the 5-way label —
    an agreement number computed over a truncated class set, which is exactly
    the class of silent instrument failure this module documents elsewhere.

    ``None`` (never a fake 1.0) when ``1 - pe`` vanishes, i.e. both raters are
    constant on the same class: there is no chance-corrected agreement to
    report there, and returning 1.0 would read as perfect tactical skill on a
    window set where nothing happened.
    """
    import numpy as _np
    a = _np.asarray(a, dtype=int).reshape(-1)
    b = _np.asarray(b, dtype=int).reshape(-1)
    if a.size == 0 or a.size != b.size:
        return None
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in range(k)))
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else None


def _class_report(gt, pred, names) -> dict:
    """Per-class recall/precision/support + the confusion matrix + the classes
    the arm NEVER predicts.

    ⭐ ``never_predicted`` is surfaced first-class because it is a MEASURED
    failure mode here, not a hypothetical: the deployed arm emitted **0 of 881**
    'accelerate' decisions. An accuracy scalar cannot show a class the model has
    silently deleted from its vocabulary.
    """
    import numpy as _np
    g = _np.asarray(gt, dtype=int).reshape(-1)
    p = _np.asarray(pred, dtype=int).reshape(-1)
    k = len(names)
    cm = _np.zeros((k, k), dtype=int)
    for gi, pi in zip(g, p):
        if 0 <= gi < k and 0 <= pi < k:
            cm[gi, pi] += 1
    per = {}
    for c, nm in enumerate(names):
        n_true, n_pred = int(cm[c].sum()), int(cm[:, c].sum())
        per[nm] = {
            "n_true": n_true,
            "n_pred": n_pred,
            "recall": round(float(cm[c, c] / n_true), 4) if n_true else None,
            "precision": round(float(cm[c, c] / n_pred), 4) if n_pred else None,
        }
    return {
        "per_class": per,
        "confusion_gt_rows_pred_cols": cm.tolist(),
        "class_order": list(names),
        "never_predicted": [nm for nm, v in per.items()
                            if v["n_true"] > 0 and v["n_pred"] == 0],
    }


def maneuver_kinematics(wp: torch.Tensor, dt: float):
    """path [n,K,2] ego-frame metres -> ``(dyaw, dv, v0, v1)`` + a provenance dict.

    These are the FOUR inputs ``refc_tactical.factor_from_kinematics`` takes.
    In the window-origin ego frame ``yaw(t0) == 0`` by construction, so
    ``dyaw = yaw(t0+H) - yaw(t0)`` is just the path tangent at the horizon.

    ⚠️ **TWO APPROXIMATIONS, both stated because they are the honest cost of
    reading a decision off a path**:

    1. ``dyaw`` is the **path tangent**, not the vehicle yaw (the module docstring
       already carries this caveat for the LATERAL family). They diverge at low
       speed, which is why the stationary handling below is not cosmetic.
    2. ``v0``/``v1`` are **chord speeds** over one step, not the recorded
       ``poses[:, 3]``. On a curve a chord under-reads the instantaneous speed by
       ~1 % at this corpus's speeds (MEASURED negative control in
       :data:`_DT_CONTRACT`: 12.5958 vs 12.4565 m/s).

    ⛔ **STATIONARY HANDLING — the branch that would otherwise bias the result
    against exactly the class we most need to see.** A window that brakes to a
    stop ends with ``ds -> 0``, where the tangent is undefined and explodes. But
    ``brake_stop`` is precisely that window, so dropping it would delete the
    class from the denominator. Instead the heading is **held at the last step
    that moved** — which is what a stopped vehicle physically does — and the
    fallback is COUNTED. A window that never moved at all gets ``dyaw = 0``
    (lane_keep laterally, which is correct: it did not turn) and its
    longitudinal class still resolves from ``dv``.
    """
    g = _seq_geometry(wp, dt)
    speed, valid = g["speed"], g["valid"]              # [n,K]
    heading = g["heading"]                              # [n,K]
    n, K = speed.shape

    # last VALID step index per window; -1 when the window never moved
    ar = torch.arange(K, device=speed.device).expand(n, K)
    last_valid = torch.where(valid, ar, torch.full_like(ar, -1)).max(dim=1).values
    never_moved = last_valid < 0
    idx = last_valid.clamp_min(0)
    dyaw = heading.gather(1, idx[:, None]).squeeze(1)
    dyaw = torch.where(never_moved, torch.zeros_like(dyaw), dyaw)
    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

    v0, v1 = speed[:, 0], speed[:, -1]
    prov = {
        "n_windows": int(n),
        "n_heading_held_from_last_moving_step": int((~valid[:, -1] & ~never_moved).sum()),
        "n_never_moved": int(never_moved.sum()),
        "min_ds_m": MIN_DS_MPS * dt,
        "dt_s": dt,
        "caveats": ["dyaw is the PATH TANGENT at the horizon, not the vehicle yaw",
                    "v0/v1 are one-step CHORD speeds, not the recorded poses[:,3]",
                    "heading is HELD at the last moving step when the window ends "
                    "stationary — dropping those windows would delete brake_stop "
                    "from the denominator"],
    }
    return dyaw, v1 - v0, v0, v1, prov


def _agreement_block(gt_cls, pred_cls, names, eid, n_boot, seed, tier=None) -> dict:
    """accuracy + kappa + per-class report, each with an episode-cluster CI.

    ⛔ The interval is the **episode-cluster bootstrap** (``taniteval.ci``) and
    the point estimate is the **full-set** value — never ``overlapping_holdout_se``,
    which biases the POINT ESTIMATE and not only the interval.

    kappa is bootstrapped through the SAME estimator by encoding each window's
    ``(gt, pred)`` pair as one integer ``gt * k + pred`` and decoding inside a
    callable reducer (``ci.resolve_reducer`` accepts callables precisely so
    kappa/F1/AUC need not invent their own interval).
    """
    import numpy as _np

    from . import ci as _ci
    g = _np.asarray(gt_cls, dtype=int).reshape(-1)
    p = _np.asarray(pred_cls, dtype=int).reshape(-1)
    k = len(names)
    out = {"status": "OK", "n": int(g.size), "n_windows": int(g.size),
           "classes": list(names), "_kappa_note": _KAPPA_NOTE}
    if tier is not None:
        out["tier"] = tier
    out.update(_class_report(g, p, names))
    out["accuracy"] = round(float((g == p).mean()), 4) if g.size else None
    kap = _kappa_k(g, p, k)
    out["kappa"] = round(kap, 4) if kap is not None else None
    out["kappa_undefined_reason"] = (
        None if kap is not None else
        "1 - p_expected vanishes: both label streams are constant on one class, "
        "so there is no chance-corrected agreement to report. NOT reported as 1.0")
    if eid is None or len(eid) != g.size or g.size == 0:
        out["ci"] = {"status": "UNAVAILABLE",
                     "reason": ("no per-window episode id aligned to these "
                                "windows, so the episode-cluster bootstrap "
                                "cannot be formed. ⛔ A bare point estimate is "
                                "not decision-grade."),
                     "n": int(g.size)}
        return out
    code = (g * k + p).astype(_np.float64)

    def _acc(v):
        v = _np.rint(v).astype(int)
        return float((v // k == v % k).mean())

    def _kap(v):
        v = _np.rint(v).astype(int)
        r = _kappa_k(v // k, v % k, k)
        return float("nan") if r is None else float(r)

    out["ci"] = {
        "accuracy": _ci.episode_cluster_bootstrap(code, eid, reduce=_acc,
                                                  n_boot=n_boot, seed=seed),
        "kappa": _ci.episode_cluster_bootstrap(code, eid, reduce=_kap,
                                               n_boot=n_boot, seed=seed),
    }
    return out


def tactical_goal(pred: torch.Tensor, gt: torch.Tensor, eid=None,
                  n_boot: int = 2000, seed: int = 0, tier=None) -> dict:
    """TACTICAL GOAL-SETTING — the goal point the arm commits to at the horizon.

    ⛔ **WHAT THIS IS AND IS NOT.** The binding rule names *"tactical goal-setting
    (… goal/anchor selection)"*. **Anchor SELECTION quality is UNAVAILABLE on a
    trajectory dump** — selecting implies a fan of candidates, and an arm that
    commits to one path has no fan to score. That half is reported UNAVAILABLE
    with its n and the instrument that would close it (``taniteval.selgap`` over
    a ``<arm>_fan_err``/``<arm>_sel_idx`` surface), never silently dropped.

    What IS computable is the goal point the arm actually set, decomposed so it
    is not merely FDE under another name:

    * ``goal_bearing_mae_deg`` — the DIRECTION of the goal, which is the
      tactical choice; a correct bearing with a short reach is a speed error,
      not a goal error, and the two are different work items.
    * ``goal_range_ratio`` — reach / human reach. **1.0 is correct**; < 1 is
      under-committing (the timid failure), > 1 over-committing.
    * ``goal_long_bias_m`` / ``goal_lat_bias_m`` — SIGNED, so late-vs-early and
      left-vs-right are visible rather than absorbed into a magnitude.

    ⚠️ ``goal_point_error_m`` IS the final-displacement error. It is reported for
    continuity with ADE/FDE and labelled as such — it is not offered as a new
    capability.
    """
    import numpy as _np

    from . import ci as _ci
    pe = pred[:, -1].detach().cpu().numpy().astype(_np.float64)
    ge = gt[:, -1].detach().cpu().numpy().astype(_np.float64)
    err = _np.linalg.norm(pe - ge, axis=-1)
    rp = _np.linalg.norm(pe, axis=-1)
    rg = _np.linalg.norm(ge, axis=-1)
    bp = _np.arctan2(pe[:, 1], pe[:, 0])
    bg = _np.arctan2(ge[:, 1], ge[:, 0])
    db = _np.degrees((bp - bg + _np.pi) % (2 * _np.pi) - _np.pi)
    # ⛔ a bearing to a goal 0.02 m away is noise, not a decision. Windows whose
    # HUMAN reach is below the gate are excluded from the bearing statistic and
    # COUNTED — the same discipline the curvature gate follows above.
    move = rg > max(MIN_DS_M, 0.5)
    ratio = _np.where(rg > 1e-6, rp / _np.maximum(rg, 1e-6), _np.nan)
    out = {
        "status": "OK",
        "n": int(err.size), "n_windows": int(err.size),
        "goal_point_error_m": round(float(err.mean()), 4),
        "_goal_point_error_is": "the final-displacement error (FDE) at the "
                                "tactical horizon — reported for continuity, "
                                "NOT offered as a new metric",
        "goal_bearing_mae_deg": (round(float(_np.abs(db[move]).mean()), 4)
                                 if move.any() else None),
        "goal_bearing_bias_deg": (round(float(db[move].mean()), 4)
                                  if move.any() else None),
        "n_bearing": int(move.sum()),
        "n_excluded_goal_below_0.5m": int((~move).sum()),
        "goal_range_ratio": (round(float(_np.nanmean(ratio[move])), 4)
                             if move.any() else None),
        "goal_long_bias_m": round(float((pe[:, 0] - ge[:, 0]).mean()), 4),
        "goal_lat_bias_m": round(float((pe[:, 1] - ge[:, 1]).mean()), 4),
        "sign_conventions": {"goal_long_bias_m": "+ = commits FURTHER ahead than the human",
                             "goal_lat_bias_m": "+ = commits LEFT of the human",
                             "goal_range_ratio": "1.0 = correct reach; <1 under-commits"},
        "anchor_selection": {
            "status": "UNAVAILABLE",
            "reason": ("this dump's arm commits to ONE path per window, so there "
                       "is no candidate fan and no selection to score. Closing it "
                       "needs a fan+selector surface (<arm>_fan_err / "
                       "<arm>_sel_idx), which taniteval.selgap then scores. A "
                       "WORK ITEM, not a pass."),
            "n": int(err.size)},
    }
    if tier is not None:
        out["tier"] = tier
    if eid is not None and len(eid) == err.size and err.size:
        comps = {"goal_point_error_m": (err, "mean"),
                 "goal_long_bias_m": (pe[:, 0] - ge[:, 0], "mean"),
                 "goal_lat_bias_m": (pe[:, 1] - ge[:, 1], "mean")}
        out["ci"] = _ci.bootstrap_metrics(comps, eid, n_boot=n_boot, seed=seed)
        if move.any():
            out["ci"]["goal_bearing_mae_deg"] = _ci.episode_cluster_bootstrap(
                _np.abs(db[move]), [e for e, m in zip(eid, move) if m],
                n_boot=n_boot, seed=seed)
    else:
        out["ci"] = {"status": "UNAVAILABLE",
                     "reason": "no aligned per-window eid — no episode-cluster "
                               "bootstrap can be formed",
                     "n": int(err.size)}
    return out


def tactical_from_trajectory(pred: torch.Tensor, gt: torch.Tensor, dt: float,
                             eid=None, n_boot: int = 2000, seed: int = 0,
                             tier=None) -> dict:
    """The TACTICAL family from trajectories alone — see this section's header.

    Returns the FACTORED agreement (lateral 3-way, longitudinal 3-way) beside
    the collapsed legacy 5-way, plus :func:`tactical_goal`. Every block carries
    its n, its estimator and — when supplied — its tier.
    """
    try:
        from tanitad.refs.refc_tactical import (COLLAPSE_TABLE, LAT_CLASSES,
                                                LON_CLASSES, MAN5_NAMES,
                                                factor_from_kinematics)
    except Exception as e:                                  # pragma: no cover
        return {"status": "UNAVAILABLE",
                "reason": (f"the canonical factored labeller "
                           f"`tanitad.refs.refc_tactical` is not importable "
                           f"({type(e).__name__}: {e}). ⛔ It is NOT re-implemented "
                           f"here on purpose — a second manoeuvre gate would drift "
                           f"from the trainer's. Put the repo's `stack/` on "
                           f"PYTHONPATH. A WORK ITEM, not a pass."),
                "n": int(pred.shape[0])}
    dyaw_p, dv_p, v0_p, v1_p, prov_p = maneuver_kinematics(pred, dt)
    dyaw_g, dv_g, v0_g, v1_g, prov_g = maneuver_kinematics(gt, dt)
    lat_p, lon_p = factor_from_kinematics(dyaw_p, dv_p, v0_p, v1_p)
    lat_g, lon_g = factor_from_kinematics(dyaw_g, dv_g, v0_g, v1_g)
    tbl = torch.as_tensor(COLLAPSE_TABLE, dtype=torch.long)
    man_p = tbl[lat_p, lon_p]
    man_g = tbl[lat_g, lon_g]

    out = {
        "status": "OK",
        "source": "trajectory-derived (four_families.tactical_from_trajectory)",
        "n": int(pred.shape[0]),
        "n_windows": int(pred.shape[0]),
        "labeller": ("tanitad.refs.refc_tactical.factor_from_kinematics — the "
                     "programme's OWN canonical gate (v1 branch, kappa=None); "
                     "no thresholds are defined in this module"),
        "lateral_decision": _agreement_block(lat_g, lat_p, LAT_CLASSES, eid,
                                             n_boot, seed, tier),
        "longitudinal_decision": _agreement_block(lon_g, lon_p, LON_CLASSES, eid,
                                                  n_boot, seed, tier),
        "maneuver_5way_collapsed": _agreement_block(man_g, man_p, MAN5_NAMES, eid,
                                                    n_boot, seed, tier),
        "goal_setting": tactical_goal(pred, gt, eid, n_boot, seed, tier),
        "kinematics_provenance": {"pred": prov_p, "gt": prov_g},
        "_factored_before_collapsed": (
            "⭐ Read LAT and LON first. CLAUDE.md names the 5-way softmax that "
            "MIXES lateral and longitudinal decisions as the programme's single "
            "largest known architectural defect; the collapsed row cannot show "
            "it, and a turn absorbs the longitudinal decision entirely "
            "(refc_tactical.COLLAPSE_TABLE)."),
        "_is_not": (
            "⛔ NOT 'selected vs executed'. Both label streams are EXECUTED "
            "manoeuvres — the arm's own and the human's. Scoring a DECLARED "
            "decision against the driven path needs a tactical head "
            "(hierarchy.run's maneuver_vs_trajectory kappa) and stays "
            "unavailable on a trajectory dump."),
        "_estimator": ("full_set point estimate; episode-cluster bootstrap "
                       "intervals (taniteval.ci). ⛔ overlapping_holdout_se is "
                       "NOT used — it biases the POINT ESTIMATE."),
    }
    if tier is not None:
        out["tier"] = tier
        if str(tier).upper() == "T0":
            out["⛔_tier_warning"] = (
                "TIER T0 — this path is TEACHER-FORCED by the RECORDED actions, "
                "so a trajectory-derived manoeuvre is substantially an ECHO of "
                "the label's own source (§1.12: open-loop lateral skill was an "
                "action echo, 97.9 % open-loop vs ~5 % closed-loop). Read this "
                "block as a WM diagnostic ONLY. The tactical capability claim "
                "requires T1.")
    return out


#: ⛔ THE STRATEGIC REASON, stated once so every arm reports it identically.
#: SETTLED at five independent probes (CLAUDE.md) — do not re-litigate it.
STRATEGIC_UNAVAILABLE_REASON = (
    "PhysicalAI-AV carries NO map, NO lane graph, NO junction/roundabout label, "
    "NO traffic-light feature and NO route/goal signal — the dataset card says "
    "verbatim 'we do not include open maps data', and obstacle.offline's enum "
    "over 87,481 cuboids is 10 classes, ALL DYNAMIC AGENTS. `egomotion` carries "
    "no lat/lon/GNSS either (clip-local metres), so OSM map-matching on our "
    "traces is impossible. ⇒ there is no admissible strategic LABEL on this "
    "split, and therefore no strategic decision or route/goal quality to score. "
    "⛔ The two label sources that DO exist are both inadmissible, for stated "
    "reasons and not for want of effort: (a) a route class read off the ego's "
    "OWN FUTURE yaw cannot tell whether the map admitted a choice at all — that "
    "is how the closed-loop harness once published route_head_eq_logged = 1.0000 "
    "on a clip where every junction had exactly ONE continuation, and "
    "GATE_PROTOCOL §0.7 declares nonav_route_beats_majority VOID BY "
    "CONSTRUCTION; (b) a supplied route is optimistic by construction here "
    "because our only route supplier is the ego's own future path.")

STRATEGIC_INSTRUMENT_THAT_WOULD_CLOSE_IT = (
    "the VLM strategic-labelling pipeline PH0 -> PH1 -> PH2 "
    "(`…/incoming/2026-08-07-hierarchical-wm-redesign/VLM_STRATEGIC_LABELING.md`, "
    "pre-registered in PREREG_PH0_VLM.md; PH2 is the g_str supervision stream, "
    "V6_TRAINING_MEASURES.md S2). It derives strategic goals from hindsight "
    "geometry + signage evidence with an honest abstain, which is the only "
    "route to a strategic label that does not read the ego's own future. "
    "V6_TRAINER_DESIGN.md:519 already states the consequence: until it lands the "
    "STRATEGIC family is n/a WITH ITS REASON AND n.")


def strategic_unavailable(n_windows: int, tier=None) -> dict:
    """The STRATEGIC family's honest n/a — reason + n + the closing instrument.

    ⛔ Clause 5 of the binding rule: *"Where a family genuinely cannot be
    computed, say so PER FAMILY with the reason and the n, rather than silently
    dropping it."* ``n`` here is the number of windows the family WOULD have had
    — it is not zero, and reporting 0 would understate what is missing.
    """
    out = {
        "status": "UNAVAILABLE",
        "n": int(n_windows),
        "n_windows_it_would_have_had": int(n_windows),
        "reason": STRATEGIC_UNAVAILABLE_REASON,
        "instrument_that_would_close_it": STRATEGIC_INSTRUMENT_THAT_WOULD_CLOSE_IT,
        "_is_a_work_item": ("⛔ A family reported UNAVAILABLE is a WORK ITEM, not "
                            "a pass. This one is blocked on a CORPUS fact, not on "
                            "eval engineering — no rescore of these windows can "
                            "produce it."),
        "_settled": ("CLAUDE.md operating standard rule 2 — settled at five "
                     "independent probes. Do not re-ask; the strategic topology "
                     "must come from AlpaSim or an external corpus."),
    }
    if tier is not None:
        out["tier"] = tier
    return out


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


def tactical(win: dict, hier: dict | None = None, traj: dict | None = None) -> dict:
    """Manoeuvre decision + tactical goal setting.

    ``traj`` (2026-08-11) is ``{"pred":…, "gt":…, "dt":…, "eid":…, "tier":…}`` and
    turns on :func:`tactical_from_trajectory` — the path that populates this
    family from a **trajectory-only dump**, which is what a T1 action-closed run
    produces. It is used only when ``hier`` is absent, because a real tactical
    head's DECLARED decision is strictly more informative than a manoeuvre read
    back off the driven path.

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
            # ⛔ via KAPPA_VERDICT_LADDER — never a restated 0.1/0.4 here. The
            # band NAME travels alongside so a consumer (and `hierarchy`'s gate
            # sweep) can compare verdicts without re-parsing the gloss.
            out["maneuver_consistency_verdict"] = kappa_verdict(k)
            out["maneuver_consistency_band"] = kappa_band(k)
        return out
    if traj is not None:
        return tactical_from_trajectory(
            traj["pred"], traj["gt"], traj["dt"], traj.get("eid"),
            n_boot=int(traj.get("n_boot", 2000)), seed=int(traj.get("seed", 0)),
            tier=traj.get("tier"))
    try:
        from tanitad.refs.refb import MANEUVER_CLASSES as _MC
        classes = list(_MC)
    except Exception:
        classes = None
    out = _decision_family(win, "tactical", "maneuver_pred", "maneuver_gt", classes)
    if out.get("status") == "UNAVAILABLE":
        out["how_to_populate"] = (
            "on a TRAJECTORY-ONLY dump (e.g. a T1 action-closed run) pass "
            "`traj={'pred':…,'gt':…,'dt':…,'eid':…,'tier':'T1'}` — or call "
            "all_families(win, tactical_from_trajectory=True) — and the family is "
            "computed from the EXECUTED manoeuvres by the programme's own "
            "canonical labeller. See four_families.tactical_from_trajectory.")
    return out


def strategic(win: dict, hier: dict | None = None, optionset: dict | None = None,
              no_label: dict | None = None) -> dict:
    """Strategic decision + route/goal setting.

    ``no_label`` (2026-08-11) is ``{"n": n_windows, "tier": …}`` and is the
    EXPLICIT declaration that the corpus being scored carries no admissible
    strategic label — it returns :func:`strategic_unavailable`, i.e. the honest
    per-family n/a **with its reason and its n**, which is what clause 5 of the
    binding rule requires. It is opt-in and never inferred, because the fact is
    a property of the CORPUS (true of PhysicalAI-AV, false of a map-carrying
    corpus like the nurec-gsplat clips) and guessing it would let a real
    strategic gap hide behind a corpus excuse.

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
    if no_label is not None:
        return strategic_unavailable(int(no_label.get("n", 0)),
                                     tier=no_label.get("tier"))
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
                 optionset: dict | None = None,
                 tactical_from_traj: bool = False,
                 strategic_no_label: bool = False,
                 tier: str | None = None, n_boot: int = 2000,
                 seed: int = 0) -> dict:
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
    traj = None
    if tactical_from_traj and hier is None:
        traj = {"pred": pred, "gt": gt, "dt": dt, "eid": win.get("eid"),
                "tier": tier, "n_boot": n_boot, "seed": seed}
    fam = {
        "longitudinal": longitudinal(pred, gt, dt, win.get("lead")),
        "lateral": lateral(pred, gt, dt),
        "tactical": tactical(win, hier, traj),
        "strategic": strategic(win, hier, optionset,
                               no_label=({"n": int(pred.shape[0]), "tier": tier}
                                         if strategic_no_label else None)),
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
    # ⛔ TWO DIFFERENT QUESTIONS, and conflating them is how an incomplete block
    # gets presented as a compliant one. `_complete` asks whether all four
    # families carry NUMBERS. `_rule_satisfied` asks whether the block obeys the
    # binding rule — which clause 5 lets a family satisfy by reporting n/a WITH
    # ITS REASON AND ITS n. A STRATEGIC n/a on PhysicalAI-AV is blocked on a
    # CORPUS fact no rescore can fix, so a block can be rule-satisfied and
    # permanently incomplete at the same time, and both facts must be visible.
    fam["_rule_satisfied"] = all(
        isinstance(fam[k], dict)
        and (fam[k].get("status", "OK") != "UNAVAILABLE"
             or (fam[k].get("reason") and fam[k].get("n") is not None))
        for k in ("longitudinal", "lateral", "tactical", "strategic"))
    fam["_rule_satisfied_note"] = (
        "clause 5: a family that genuinely cannot be computed satisfies the rule "
        "by stating its REASON and its n. `_complete` is the stricter question — "
        "whether all four carry numbers — and stays False while STRATEGIC is n/a.")
    if tier is not None:
        fam["_tier"] = tier
        for k in ("longitudinal", "lateral", "tactical", "strategic"):
            if isinstance(fam[k], dict):
                fam[k].setdefault("tier", tier)
    return fam
