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


# ============================================================================ #
# ⭐ INTERVALS ON EVERY REPORTED COMPONENT (2026-08-23)                         #
#                                                                              #
# ⛔ THE HOLE THIS CLOSES, MEASURED. Before today NEITHER `longitudinal()` nor  #
# `lateral()` took an `eid` or produced a single interval: every number in both #
# families was a BARE POINT ESTIMATE, and the intervals that existed were       #
# computed by the CALLERS over a hand-picked subset —                          #
#                                                                              #
#   tools/eval_four_families.py  7 components                                  #
#   tools/ff_rescore.py         10 components                                  #
#   tools/t1_eval.py             6 components                                  #
#                                                                              #
# — so `target_speed_acc`, `speed_bias`, `speed_rmse`, `accel_mae`,            #
# `along_bias`, `along_final_bias`, `ego_progress`, `yaw_rate_mae`,           #
# `curvature_bias`, `cross_bias`, `cross_final_mae` and every pooled           #
# `distance_keeping` scalar shipped with NO interval at all. CLAUDE.md's        #
# estimator rule ("never quote an interval without its estimator") has a silent #
# twin — **never quote a POINT ESTIMATE without an interval** — and a           #
# hand-picked subset is exactly how the twin gets broken without anyone lying.  #
#                                                                              #
# ⭐ THE DESIGN RULE THAT MAKES THIS SAFE: the interval's point estimate is the #
# BLOCK'S OWN NUMBER, recomputed by the same arithmetic, never a second         #
# implementation. `ff_rescore.score_arm` already states why ("a second          #
# implementation lets the interval and the point estimate drift apart           #
# silently"). Concretely:                                                      #
#                                                                              #
#   * an UNMASKED component (speed, along, cross, accel, target-speed bands) is #
#     a flat mean over [n, H] entries; every window has the same H, so the mean #
#     of the per-window means IS that number, exactly.                          #
#   * a MASKED component (heading / yaw-rate / curvature) is a mean over a      #
#     ragged subset, and the mean of per-window means is a DIFFERENT statistic  #
#     (macro vs micro). Those therefore bootstrap a **ratio of sums** —         #
#     `sum(per-window error sums) / sum(per-window counts)` — which reproduces  #
#     the block's masked mean to float64, so the interval is centred on the     #
#     number that is printed and not on a near-miss of it.                      #
#                                                                              #
# ⚠️ NOTHING HERE MOVES AN EXISTING VALUE. Every emitted metric keeps its exact #
# arithmetic; this layer only ADDS a `ci` sub-block beside it. `tests/          #
# test_eval_pipeline.py` pins that equality in both directions.                 #
# ============================================================================ #

#: ⛔ THE DECLARED COVERAGE CONTRACT. Every scalar the LONGITUDINAL/LATERAL
#: blocks report is named here with the dotted path it lives at, because
#: coverage that is only implicit cannot be AUDITED: a metric added without an
#: interval, or an interval quietly dropped, both read as "the block looks fine".
#: :func:`ci_coverage` diffs the emitted block against these two tuples, and
#: `_NON_METRIC_KEYS` below closes the other direction — a numeric leaf that is
#: neither declared nor known provenance is reported as UNDECLARED.
LONGITUDINAL_COMPONENTS = (
    "speed_mae_mps", "speed_bias_mps", "speed_rmse_mps",
    # ⛔ DERIVED from the bands tuple, never re-typed: the emitter builds these
    # keys with the same f-string, so a band added to TARGET_SPEED_BANDS_MPS
    # cannot appear in the block without also appearing in the coverage contract.
    *(f"target_speed_acc.within_{b}_mps" for b in TARGET_SPEED_BANDS_MPS),
    "along_mae_m", "along_bias_m", "along_final_bias_m",
    "accel_mae_mps2",
    "ego_progress.progress_ratio_mean", "ego_progress.progress_ratio_median",
    "ego_progress.progress_error_mean", "ego_progress.under_progress_rate",
    "ego_progress.gt_progress_mean_m", "ego_progress.t0_axis_gt_self_ratio",
    "distance_keeping.mean_headway_min_m",
    "distance_keeping.mean_time_gap_min_s",
    "distance_keeping.mean_min_ttc_s",
)

LATERAL_COMPONENTS = (
    "heading_mae_deg", "yaw_rate_mae_degps",
    "curvature_mae_1pm", "curvature_bias_1pm",
    "cross_mae_m", "cross_bias_m", "cross_final_mae_m",
)

#: Numeric leaves that are PROVENANCE / DENOMINATORS, not measurements — they
#: must NOT carry an interval, and listing them explicitly is what lets
#: :func:`ci_coverage` flag anything else that appears without one.
_NON_METRIC_KEYS = frozenset({
    "dt_s", "n", "n_windows", "n_boot", "seed", "min_ds_m", "min_ds_mps",
    "n_steps_heading", "n_steps_curvature", "n_steps_yaw_rate",
    "excluded_below_min_ds", "n_excluded_low_gt_progress", "min_progress_m",
    "n_time_gap", "n_closing", "ttc_cap_s", "n_episodes", "n_total",
    "n_windows_dropped_no_valid_step", "n_components", "n_with_interval",
    "n_unavailable", "n_undeclared", "path_steps", "alpha",
})

#: The unit the bootstrap resamples. Stated in every emitted CI block, because
#: an interval whose cluster unit is the WINDOW rather than the EPISODE is
#: anti-conservative by roughly the within-clip correlation and looks identical.
CI_RESAMPLING_UNIT = ("EPISODE (win['eid']) — consecutive windows of one clip "
                      "are not independent, so the episode is the cluster unit")


def _ci_unavailable(reason: str, n: int, **extra) -> dict:
    """The ONLY admissible shape for a component that cannot carry an interval.

    ⛔ Never a bare point estimate and never silence: the binding rule's clause 5
    shape — status + reason + n — one level down from the family."""
    out = {"status": "UNAVAILABLE", "reason": reason, "n": int(n)}
    out.update(extra)
    return out


def _ratio_reducer(sums, counts, scale: float = 1.0, label: str = "masked_mean"):
    """A ``ratio of sums`` reducer for :func:`taniteval.ci.bootstrap_metrics`.

    ⛔ WHY THE VALUES ARE INDICES. ``ci`` reducers receive ONE 1-D array per
    component, and a ratio of sums needs TWO aligned arrays (the per-window error
    sums and the per-window valid-step counts). The supported escape hatch is the
    CALLABLE reducer — ``ci.resolve_reducer`` exists precisely so kappa/F1/AUC can
    bootstrap without inventing their own estimator — so the component's "values"
    are the row indices and the reducer closes over both arrays. The resampling,
    the cluster unit and the percentile arithmetic are ``ci``'s, untouched.

    ⚠️ The caller MUST restrict the rows to ``counts > 0`` before calling. Then
    every resampled episode contributes at least one valid step and the
    denominator can never be zero; the dropped rows contribute 0 to BOTH sums, so
    the restriction does not change the statistic.
    """
    import numpy as np
    s = np.asarray(sums, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)

    def _r(idx):
        i = np.asarray(idx, dtype=np.int64)
        tot = float(c[i].sum())
        if tot <= 0:
            return float("nan")
        return float(scale * (float(s[i].sum()) / tot))

    _r.__name__ = f"{label}=sum(err)/sum(valid_steps)"
    return _r


def _run_ci_groups(groups: dict, n_boot: int, seed: int) -> dict:
    """Run one :func:`ci.bootstrap_metrics` per (eid-subset) GROUP.

    ``groups`` maps a group key -> ``{"eid": [...], "components": {name: (values,
    reducer, dp)}, "meta": {name: {...extra fields...}}}``. Grouping is what keeps
    the intervals *mutually consistent* wherever they can be: everything scored on
    the full window set moves together inside a single resampling, exactly as
    ``ci.bootstrap_metrics`` documents. Components with a different denominator
    (progress excludes stopped windows; distance-keeping excludes free flow) get
    their own draw, because they genuinely have a different n.
    """
    from taniteval import ci as _ci
    out = {}
    for spec in groups.values():
        comps, eid = spec["components"], spec["eid"]
        if not comps:
            continue
        res = _ci.bootstrap_metrics(comps, eid, n_boot=n_boot, seed=seed)
        for name, block in res.items():
            block["resampling_unit"] = CI_RESAMPLING_UNIT
            block.update(spec.get("meta", {}).get(name, {}))
            out[name] = block
    return out


def _ci_block(components: dict, unavailable: dict, n_windows: int,
              eid_note: str | None = None) -> dict:
    """Assemble the per-family ``ci`` sub-block, with its own completeness read."""
    n_ok = len(components)
    n_na = len(unavailable)
    blk = {
        "estimator": "episode_cluster_bootstrap",
        "point_estimate": "full_set (the block's own printed value, recomputed "
                          "by the same arithmetic — never a second implementation)",
        "resampling_unit": CI_RESAMPLING_UNIT,
        "n_windows": int(n_windows),
        "components": components,
        "unavailable": unavailable,
        "n_components": n_ok + n_na,
        "n_with_interval": n_ok,
        "n_unavailable": n_na,
        "complete": n_na == 0,
        "⛔_rule": ("every component this family REPORTS carries an "
                   "episode-cluster-bootstrap interval, or an explicit "
                   "{status,reason,n}. A bare point estimate is neither."),
    }
    if eid_note:
        blk["_eid_note"] = eid_note
    return blk


def _resolve_eid(eid, n: int) -> tuple[list | None, str | None]:
    """-> (eid list of length n, or (None, why-not)). Never guesses a cluster id."""
    if eid is None:
        return None, ("no per-window episode id (win['eid']) — the episode-cluster "
                      "bootstrap has no cluster unit, so NO interval can be formed. "
                      "⛔ A window-level resampling is NOT a substitute: consecutive "
                      "windows of one clip are strongly dependent and it would be "
                      "anti-conservative. Supply win['eid'].")
    e = list(eid)
    if len(e) != n:
        return None, (f"win['eid'] has {len(e)} entries for {n} windows — refusing "
                      f"to form clusters from a misaligned id array")
    if n == 0:
        return None, "0 windows — nothing to resample"
    return e, None


def longitudinal(pred: torch.Tensor, gt: torch.Tensor, dt: float = DT_S,
                 lead: dict | None = None, win: dict | None = None,
                 n_boot: int = 2000, seed: int = 0, eid=None) -> dict:
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

    ⛔ ``win`` (2026-08-16) supplies ``v0``/``eid`` to the **ANTI-ECHO CONTROLS**, which are
    attached to EVERY longitudinal block rather than requested. **Sayed, verbatim, 2026-08-16:**
    *"We can use v0 as input since it is measured and is not the future, but we should assure that
    the model/planner later is not cheating by just outputting v0 as longitudinal plan."* Holding
    the current speed is a strong baseline on most windows, so a planner can emit "keep doing v0"
    and score well — **skill attributed to a copy**, the same shape as the nav-echo (1.0000), the
    T1 action echo (97.9 % open-loop / 0.0 % hold-action) and the P1 speed echo (R² 0.995 → −0.72).
    ⇒ ``anti_echo`` carries the hold-v0 baseline (paired episode-cluster bootstrap), the copy
    detector's scalar, and the v0-shuffle control. See :mod:`taniteval.v0_antiecho`.
    """
    P, G = _seq_geometry(pred, dt), _seq_geometry(gt, dt)
    sp_err = P["speed"] - G["speed"]
    al_err = P["along"] - G["along"]
    ac_err = P["accel"] - G["accel"]
    dk = _distance_keeping(pred, dt, lead)
    out = {
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
        "distance_keeping": dk,
        # ⛔ THE ANTI-ECHO CONTROLS (PI 2026-08-16) — ALWAYS ON, never on request.
        # Every speed number above can be earned by copying v0; these are the
        # three controls that say whether it was. `longitudinal_claim_admissible`
        # inside is the bit a report may not lose.
        "anti_echo": _anti_echo(pred, gt, dt, win, lead, n_boot, seed),
        "n_windows": int(pred.shape[0]),
    }
    # ⭐ EVERY component above now carries an episode-cluster interval, or says
    # in the clause-5 shape why it cannot. See the block comment above
    # LONGITUDINAL_COMPONENTS for the hole this closes.
    if eid is None and isinstance(win, dict):
        eid = win.get("eid")
    out["ci"] = _longitudinal_ci(out, sp_err, al_err, ac_err, pred, gt, lead,
                                 eid, n_boot, seed)
    return out


def _longitudinal_ci(out: dict, sp_err, al_err, ac_err, pred, gt, lead,
                     eid, n_boot: int, seed: int) -> dict:
    """Episode-cluster intervals for EVERY scalar :func:`longitudinal` reports."""
    import numpy as np

    n = int(pred.shape[0])
    ids, why = _resolve_eid(eid, n)
    if ids is None:
        return _ci_block({}, {c: _ci_unavailable(why, n)
                              for c in LONGITUDINAL_COMPONENTS}, n)

    groups: dict = {}

    def _add(group, name, values, reduce="mean", dp=4, eids=None, **meta):
        g = groups.setdefault(group, {"eid": eids if eids is not None else ids,
                                      "components": {}, "meta": {}})
        g["components"][name] = (values, reduce, dp)
        if meta:
            g["meta"][name] = meta

    # -- group ALL: one resampling, so these intervals move together ---------- #
    # every window has the same H, so the mean of the per-window means IS the
    # flat mean the block prints (pinned by test_eval_pipeline).
    _add("ALL", "speed_mae_mps", sp_err.abs().mean(1).numpy())
    _add("ALL", "speed_bias_mps", sp_err.mean(1).numpy())
    _add("ALL", "speed_rmse_mps", (sp_err ** 2).mean(1).numpy(), reduce="rms")
    for b in TARGET_SPEED_BANDS_MPS:
        _add("ALL", f"target_speed_acc.within_{b}_mps",
             (sp_err.abs() <= b).float().mean(1).numpy())
    _add("ALL", "along_mae_m", al_err.abs().mean(1).numpy())
    _add("ALL", "along_bias_m", al_err.mean(1).numpy())
    _add("ALL", "along_final_bias_m", al_err[:, -1].numpy())
    if ac_err.shape[1] > 0:
        _add("ALL", "accel_mae_mps2", ac_err.abs().mean(1).numpy())

    unavailable: dict = {}
    if ac_err.shape[1] == 0:
        unavailable["accel_mae_mps2"] = _ci_unavailable(
            "the horizon has fewer than 2 steps, so no acceleration exists to "
            "difference — the block's value is NaN for the same reason", n)

    # -- EGO PROGRESS: its own denominator (stopped windows are EXCLUDED) ----- #
    prog = out.get("ego_progress", {})
    if prog.get("status") == "OK":
        from taniteval.progress import progress_per_window
        w = progress_per_window(pred.detach().cpu().numpy(),
                                gt.detach().cpu().numpy(), "human_dir")
        v = w["valid"]
        pe = [e for e, k in zip(ids, v) if k]
        drop = int((~v).sum())
        _add("PROGRESS", "ego_progress.progress_ratio_mean", w["ratio"][v],
             eids=pe, n_excluded_low_gt_progress=drop)
        _add("PROGRESS", "ego_progress.progress_ratio_median", w["ratio"][v],
             reduce="median", eids=pe, n_excluded_low_gt_progress=drop)
        _add("PROGRESS", "ego_progress.progress_error_mean", w["error"][v],
             eids=pe, n_excluded_low_gt_progress=drop)
        _add("PROGRESS", "ego_progress.under_progress_rate",
             (w["ratio"][v] < 1.0).astype(float), eids=pe,
             n_excluded_low_gt_progress=drop)
        _add("PROGRESS", "ego_progress.gt_progress_mean_m",
             w["gt_progress_m"][v], eids=pe, n_excluded_low_gt_progress=drop)
        gs = progress_per_window(gt.detach().cpu().numpy(),
                                 gt.detach().cpu().numpy(), "t0_axis")
        gv = gs["valid"]
        _add("PROGRESS_T0AXIS", "ego_progress.t0_axis_gt_self_ratio",
             gs["ratio"][gv], eids=[e for e, k in zip(ids, gv) if k],
             _what=("a property of the GROUND TRUTH's own geometry (the "
                    "curvature confound in pseudosim's published reading), not "
                    "of the arm — its interval is corpus dispersion"))
    else:
        why_p = prog.get("reason", "ego_progress is not OK")
        for c in LONGITUDINAL_COMPONENTS:
            if c.startswith("ego_progress."):
                unavailable[c] = _ci_unavailable(why_p, int(prog.get("n", 0)))

    # -- DISTANCE KEEPING: pooled scalars, on the windows that HAVE a lead ---- #
    dk = out.get("distance_keeping", {})
    dk_pw = dk.get("_per_window") if isinstance(dk, dict) else None
    dk_eid = (lead or {}).get("eid") if isinstance(lead, dict) else None
    dk_names = ("distance_keeping.mean_headway_min_m",
                "distance_keeping.mean_time_gap_min_s",
                "distance_keeping.mean_min_ttc_s")
    if dk.get("status") != "OK" or dk_pw is None:
        why_d = dk.get("reason") or (
            "distance_keeping did not produce per-window arrays, so no interval "
            "can be formed over them")
        for c in dk_names:
            unavailable[c] = _ci_unavailable(why_d, int(dk.get("n", 0) or 0))
    elif dk_eid is None or len(list(dk_eid)) != n:
        for c in dk_names:
            unavailable[c] = _ci_unavailable(
                "the lead block carries no per-window episode id (lead['eid']) "
                "aligned to the scored windows, so the episode-cluster bootstrap "
                "has no cluster unit. ⛔ A pooled point estimate is not a "
                "substitute.", int(dk.get("n", 0) or 0))
    else:
        de = [str(x) for x in dk_eid]
        hw = np.asarray(dk_pw["headway_min_m"], dtype=np.float64)
        tg = np.asarray(dk_pw["time_gap_min_s"], dtype=np.float64)
        tt = np.asarray(dk_pw["min_ttc_s"], dtype=np.float64)
        have = np.isfinite(hw)
        have_tg = np.isfinite(tg)
        he = [e for e, k in zip(de, have) if k]
        _add("DK_HAVE", "distance_keeping.mean_headway_min_m", hw[have],
             eids=he, n_windows_with_lead=int(have.sum()))
        _add("DK_HAVE", "distance_keeping.mean_min_ttc_s", tt[have], eids=he,
             n_windows_with_lead=int(have.sum()),
             censoring_note=dk.get("censoring_note"))
        if have_tg.any():
            _add("DK_TG", "distance_keeping.mean_time_gap_min_s", tg[have_tg],
                 eids=[e for e, k in zip(de, have_tg) if k],
                 n_time_gap=int(have_tg.sum()),
                 _note=("the time gap is NaN below lead_metrics.MIN_SPEED_MPS "
                        "and is never clamped, so its denominator is smaller "
                        "than headway's"))
        else:
            unavailable["distance_keeping.mean_time_gap_min_s"] = _ci_unavailable(
                "no window has a finite time gap (every ego speed is below "
                "lead_metrics.MIN_SPEED_MPS) — the block reports NaN for the "
                "same reason", 0)

    comps = _run_ci_groups(groups, n_boot, seed)
    return _ci_block(comps, unavailable, n)


def _anti_echo(pred: torch.Tensor, gt: torch.Tensor, dt: float,
               win: dict | None, lead: dict | None, n_boot: int, seed: int) -> dict:
    """:mod:`taniteval.v0_antiecho` over this arm's own plan, or the reason why not.

    Kept out of :func:`longitudinal` for the same reason ``_distance_keeping`` is: the
    UNAVAILABLE branch must read as ONE thing — a WORK ITEM with a reason and an ``n`` —
    rather than being mistaken for a pass.

    ⭐ ``lead`` is folded into the lookup because a lead block carries ``speeds`` (the
    time-gap denominator), which IS the ego speed at t0. A caller who supplied a lead
    block has therefore already supplied ``v0`` without knowing it, and the control
    should not report UNAVAILABLE beside a dict that contains exactly what it needs.
    """
    from taniteval import v0_antiecho as _ae
    ctx = dict(win or {})
    if lead is not None and "lead" not in ctx:
        ctx["lead"] = lead
    return _ae.anti_echo(pred, gt, dt, ctx, n_boot=n_boot, seed=seed)


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


def lateral(pred: torch.Tensor, gt: torch.Tensor, dt: float = DT_S, eid=None,
            n_boot: int = 2000, seed: int = 0) -> dict:
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
    out = {
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
        # ⚠️ ADDITIVE (2026-08-23). `yaw_rate_mae_degps` is emitted under the
        # HEADING count (`if n_head`) while it is masked by `both_pair`, so it
        # has never carried its own denominator — and when a window set has
        # valid single steps but no valid PAIRS it emits NaN under a non-zero
        # n_head. The correct denominator was already computed (it is the same
        # mask as curvature); it is now NAMED rather than discarded, which is
        # also what its interval has to be formed over. The emitted VALUE is
        # deliberately unchanged — moving it would silently rewrite banked
        # numbers, and this is the honest first half of that fix.
        "n_steps_yaw_rate": n_curv,
        "excluded_below_min_ds": int((~both).sum()),
        # ⛔ the gate SCALES with dt now. A fixed 0.05 m on a 0.5 s grid excluded ~nothing and let
        # crawling windows into the curvature statistic.
        "min_ds_m": MIN_DS_MPS * dt,
        "min_ds_mps": MIN_DS_MPS,
        "dt_s": dt,
        "dt_invariant": ["heading_mae_deg", "curvature_*", "cross_*"],
        "n_windows": int(pred.shape[0]),
    }
    out["ci"] = _lateral_ci(out, P, G, ct_err, both, both_pair, dh, eid,
                            n_boot, seed)
    return out


def _lateral_ci(out: dict, P: dict, G: dict, ct_err, both, both_pair, dh,
                eid, n_boot: int, seed: int) -> dict:
    """Episode-cluster intervals for EVERY scalar :func:`lateral` reports.

    ⛔ heading / yaw-rate / curvature are MASKED means over a ragged subset of
    steps, so their interval is a **ratio of sums** (see :func:`_ratio_reducer`)
    and NOT a mean of per-window means — the latter is a different statistic
    (macro vs micro) and would centre the interval on a number the block does
    not print. cross-track is unmasked and reduces exactly.
    """
    import numpy as np

    n = int(ct_err.shape[0])
    ids, why = _resolve_eid(eid, n)
    if ids is None:
        return _ci_block({}, {c: _ci_unavailable(why, n)
                              for c in LATERAL_COMPONENTS}, n)

    groups: dict = {}

    def _add(group, name, values, reduce="mean", dp=4, eids=None, **meta):
        g = groups.setdefault(group, {"eid": eids if eids is not None else ids,
                                      "components": {}, "meta": {}})
        g["components"][name] = (values, reduce, dp)
        if meta:
            g["meta"][name] = meta

    unavailable: dict = {}

    def _masked_component(group, name, err, mask, scale, dp, label, why_empty):
        """Per-window (sum, count) -> a ratio-of-sums bootstrap on rows with count>0."""
        cnt = mask.sum(1).numpy().astype(np.float64)
        s = (err * mask).sum(1).numpy().astype(np.float64)
        keep = cnt > 0
        if not keep.any():
            unavailable[name] = _ci_unavailable(why_empty, 0)
            return
        sk, ck = s[keep], cnt[keep]
        _add(group, name, np.arange(sk.size, dtype=np.float64),
             reduce=_ratio_reducer(sk, ck, scale=scale, label=label), dp=dp,
             eids=[e for e, k in zip(ids, keep) if k],
             n_steps=int(ck.sum()),
             n_windows_dropped_no_valid_step=int((~keep).sum()),
             _reduction=("MICRO mean = sum(per-window error sums) / "
                         "sum(per-window valid-step counts) — identical to the "
                         "masked mean the block prints, NOT a mean of "
                         "per-window means"))

    _masked_component(
        "MASK_HEADING", "heading_mae_deg", dh.abs(), both, 180.0 / math.pi, 4,
        "heading_mae_deg",
        "no step in any window clears MIN_DS — a stopped path has no tangent, so "
        "the block reports None for the same reason")
    _masked_component(
        "MASK_PAIR", "yaw_rate_mae_degps",
        (P["yaw_rate"] - G["yaw_rate"]).abs(), both_pair, 180.0 / math.pi, 4,
        "yaw_rate_mae_degps",
        "no window has two consecutive valid steps, so no yaw rate is defined")
    _masked_component(
        "MASK_PAIR", "curvature_mae_1pm",
        (P["curvature"] - G["curvature"]).abs(), both_pair, 1.0, 6,
        "curvature_mae_1pm",
        "no window has two consecutive valid steps, so no curvature is defined")
    _masked_component(
        "MASK_PAIR", "curvature_bias_1pm",
        P["curvature"] - G["curvature"], both_pair, 1.0, 6,
        "curvature_bias_1pm",
        "no window has two consecutive valid steps, so no curvature is defined")

    # cross-track is unmasked: every window contributes every step.
    _add("ALL", "cross_mae_m", ct_err.abs().mean(1).numpy())
    _add("ALL", "cross_bias_m", ct_err.mean(1).numpy())
    _add("ALL", "cross_final_mae_m", ct_err[:, -1].abs().numpy())

    comps = _run_ci_groups(groups, n_boot, seed)
    return _ci_block(comps, unavailable, n)


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


def _dig_loose(obj, path: str) -> tuple[bool, object]:
    """Resolve a dotted path where a KEY may itself contain a dot.

    ⛔ WHY THE OBVIOUS `path.split('.')` IS WRONG HERE. The target-speed bands
    are emitted as ``within_0.5_mps`` — the band value is *in the key* — so a
    naive split produces ``['target_speed_acc', 'within_0', '5_mps']`` and the
    component resolves to NOTHING. MEASURED while building this layer: all three
    ``target_speed_acc`` bands were reported as "not present in the block" while
    they were present, correct, and already carrying intervals. A resolver that
    silently misses is exactly the ``absence found at one location`` trap, so it
    matches the LONGEST key prefix at each level instead of assuming keys are
    dot-free. Returns ``(found, value)``.
    """
    if not isinstance(obj, dict):
        return False, None
    if path in obj:
        return True, obj[path]
    for k in obj:
        ks = str(k)
        if path.startswith(ks + "."):
            found, val = _dig_loose(obj[k], path[len(ks) + 1:])
            if found:
                return True, val
    return False, None


def _numeric_leaves(obj, prefix: str = "") -> dict:
    """Every numeric leaf under ``obj``, by dotted path. Booleans are NOT numbers
    here — a flag is a verdict, not a measurement."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                out[p] = v
            elif isinstance(v, dict):
                out.update(_numeric_leaves(v, p))
    return out


def ci_coverage(family_block: dict, family: str) -> dict:
    """⭐ AUDIT one emitted LONGITUDINAL/LATERAL block for interval coverage.

    This is the instrument that makes a MISSING interval fail rather than read as
    a clean record. It answers three questions, and the third is the one a
    hand-maintained list cannot:

    ``missing``      a DECLARED component that the block reports but that carries
                     neither an interval nor a ``{status,reason,n}``. ⛔ VIOLATION.
    ``unavailable``  declared, reported, and honestly declined. A WORK ITEM.
    ``undeclared``   a numeric leaf in the block that is neither a declared
                     component nor known provenance (:data:`_NON_METRIC_KEYS`) —
                     i.e. **a metric someone added without an interval**. That is
                     how the hole this whole layer closes was dug in the first
                     place, so it is detected structurally rather than by
                     remembering to update a list.

    ⚠️ A component the block does NOT report (value ``None``, or a key that never
    appears because its input was absent) is not counted as missing — there is
    nothing to bound. It shows up in ``not_reported``.
    """
    declared = {"longitudinal": LONGITUDINAL_COMPONENTS,
                "lateral": LATERAL_COMPONENTS}.get(family)
    if declared is None:
        raise ValueError(f"ci_coverage covers longitudinal/lateral, not {family!r}")
    ci = family_block.get("ci")
    if not isinstance(ci, dict):
        return {"family": family, "status": "ABSENT",
                "reason": ("the block carries NO `ci` sub-block at all — every "
                           "number in it is a bare point estimate"),
                "missing": list(declared), "unavailable": [], "undeclared": [],
                "not_reported": [], "complete": False}
    comps = ci.get("components", {}) or {}
    nas = ci.get("unavailable", {}) or {}

    missing, unavail, not_reported = [], [], []
    for c in declared:
        if c in comps and "lo" in comps[c] and "hi" in comps[c]:
            continue
        if c in nas and nas[c].get("reason") and nas[c].get("n") is not None:
            unavail.append(c)
        elif not _dig_loose(family_block, c)[0]:
            not_reported.append(c)
        else:
            missing.append(c)

    # the other direction: a numeric leaf nobody declared
    skip_roots = ("ci.", "anti_echo.", "distance_keeping.by_speed.",
                  "distance_keeping._per_window.", "ego_progress.n",
                  "distance_keeping.n")
    undeclared = []
    for path in _numeric_leaves(family_block):
        if path.startswith(skip_roots):
            continue
        leaf = path.rsplit(".", 1)[-1]
        if leaf in _NON_METRIC_KEYS or path in declared:
            continue
        undeclared.append(path)

    return {
        "family": family,
        "status": "OK",
        "n_declared": len(declared),
        "n_with_interval": len(declared) - len(missing) - len(unavail) - len(not_reported),
        "missing": sorted(missing),
        "unavailable": sorted(unavail),
        "undeclared": sorted(undeclared),
        "not_reported": sorted(not_reported),
        "complete": not missing and not undeclared,
        "⛔_verdict": ("VIOLATION — a reported component carries no interval and "
                       "no reason" if (missing or undeclared) else
                       "every reported component carries an interval or a "
                       "reasoned refusal"),
    }


def all_families(win: dict, hier: dict | None = None, prefer_dense: bool = True,
                 optionset: dict | None = None,
                 tactical_from_traj: bool = False,
                 strategic_no_label: bool = False,
                 tier: str | None = None, n_boot: int = 2000,
                 seed: int = 0, protocol: dict | None = None) -> dict:
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
        "longitudinal": longitudinal(pred, gt, dt, win.get("lead"), win=win,
                                     n_boot=n_boot, seed=seed,
                                     eid=win.get("eid")),
        "lateral": lateral(pred, gt, dt, eid=win.get("eid"),
                           n_boot=n_boot, seed=seed),
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
    # ⭐ INTERVAL COVERAGE, at the top of the block (2026-08-23). `_complete`
    # asks whether the four families carry NUMBERS; this asks whether those
    # numbers carry their UNCERTAINTY. Both were needed: until today LON and LAT
    # were `_complete: true` while every scalar in them was a bare point
    # estimate, and nothing in the record said so.
    fam["_ci_coverage"] = {
        "longitudinal": ci_coverage(fam["longitudinal"], "longitudinal"),
        "lateral": ci_coverage(fam["lateral"], "lateral"),
    }
    fam["_intervals_complete"] = bool(
        fam["_ci_coverage"]["longitudinal"].get("complete")
        and fam["_ci_coverage"]["lateral"].get("complete"))
    fam["_intervals_rule"] = (
        "every component the LONGITUDINAL and LATERAL blocks REPORT carries an "
        "episode-cluster-bootstrap interval (taniteval.ci, cluster unit = the "
        "episode), or an explicit {status,reason,n}. ⛔ overlapping_holdout_se is "
        "never used: it biases the POINT ESTIMATE, not only the interval. Any "
        "TWO-ARM delta must use ci.paired_episode_cluster_bootstrap — never a "
        "combination of two single-arm intervals in quadrature.")
    fam["_complete"] = not unavailable and \
        fam["longitudinal"]["distance_keeping"]["status"] == "OK" and \
        fam["longitudinal"]["anti_echo"]["status"] == "OK"
    # ⛔ PROMOTED TO THE TOP OF THE BLOCK (2026-08-16), because a bit buried three
    # levels down is a bit a report loses. The PI's condition is that NO
    # longitudinal claim is admissible until the arm beats hold-v0, separated —
    # so the block states, at its own top level, whether that has been discharged.
    _ae = fam["longitudinal"]["anti_echo"]
    fam["_longitudinal_claim_admissible"] = bool(
        _ae.get("longitudinal_claim_admissible", False))
    fam["_anti_echo_summary"] = _ae.get("summary", _ae.get("reason"))
    fam["_anti_echo_rule"] = (
        "Sayed 2026-08-16: v0 is ADMISSIBLE as a planner input, PROVIDED the "
        "planner is shown not to be 'just outputting v0 as longitudinal plan'. "
        "⛔ A LONGITUDINAL number emitted while _longitudinal_claim_admissible "
        "is False is a fidelity diagnostic, NOT a longitudinal capability "
        "result, and must not be presented as one.")
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
    else:
        # ⛔ AN UNSTAMPED BLOCK MUST SAY SO. Omitting the tier used to omit the
        # T0 echo warning with it (the warning lives under `if tier is not None`),
        # so the one caller that passed no tier produced a block that looked
        # CLEANER than a correctly-stamped T0 one — the guard disappeared exactly
        # when it was needed. MEASURED 2026-08-23: `eval_four_families.py` never
        # passed `tier=` and 93 of 135 in-scope artifacts carry no tier at all.
        # Same class as C133: silence read as compliance.
        fam["_tier"] = None
        fam["⛔_tier_missing"] = (
            "NO TIER SUPPLIED to all_families(). This block is NOT QUOTABLE: a "
            "number without its tier cannot be read as driving performance or as "
            "a WM diagnostic, and the T0 action-echo warning is suppressed when "
            "the tier is absent. Pass tier='T0'|'T1'|'T2' at the call site.")

    # ---- PROTOCOL DECLARATIONS ------------------------------------------- #
    # ⛔ THREE BINDING FACTS THAT ONLY THE CALLER KNOWS, AND THAT AN ARTIFACT
    # MUST NOT LEAVE SILENT: what the model consumed at inference (vision-only
    # is binding, PI 2026-08-03), whether the goal path is information-disjoint
    # from the situation classifier (PI 2026-08-03), and WHICH corpus was
    # scored (parity is sacred).
    #
    # This block never GUESSES them. An emitter that invented "vision_only:
    # true" would be manufacturing compliance, which is worse than the gap. So
    # an absent declaration is written out as UNDECLARED, with what to pass —
    # the same contract as the tier above: the gap is visible, never silent.
    # MEASURED 2026-08-23: the release gate scored RG-07/RG-08/RG-12 as FAIL on
    # our best T1 artifacts purely because nothing recorded these, not because
    # anything was violated.
    _UNDECLARED = "UNDECLARED — pass `protocol=` to all_families(); NOT assumed compliant"
    proto = dict(protocol or {})
    fam["_protocol"] = {
        "inference_inputs": proto.get("inference_inputs", _UNDECLARED),
        "vision_only": proto.get("vision_only", _UNDECLARED),
        "goal_source": proto.get("goal_source", _UNDECLARED),
        "goal_situation_disjoint": proto.get("goal_situation_disjoint", _UNDECLARED),
        "corpus": proto.get("corpus", _UNDECLARED),
        "parity_key": proto.get("parity_key", _UNDECLARED),
        "_binding": (
            "vision-only at inference (labels MAY use ego); the goal input must "
            "not carry the situation classifier's output in any form; the corpus "
            "identity makes cross-arm deltas interpretable. An UNDECLARED value "
            "is a WORK ITEM — it is never read as compliance."),
    }
    fam["_protocol_undeclared"] = sorted(
        k for k, v in fam["_protocol"].items()
        if not k.startswith("_") and v == _UNDECLARED)
    return fam
