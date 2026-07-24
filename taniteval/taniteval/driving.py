"""TanitEval v2 — DRIVING-CAPABILITY panel (tier 0), a DEFAULT axis of every eval.

WHAT THIS IS
------------
The executable form of ``TanitAD Research Hub/Benchmarks & Eval/
TANITEVAL_V2_METRIC_SUITE.md`` tier-0 set: every driving-capability metric that
is computable **today, on CPU, from an already-persisted
``results/windows_<arm>.pt``** — no GPU, no pod, no re-run, no new logging.

    L1 CRUISE-QUALITY · L2 TRANSIENT-RESPONSE · L3 along-track error + bias ·
    L4 progress error · T1 cross-track error + bias · T2 speed-decoupled path
    geometry · T3 heading error stratified by curvature · T4 curvature SIGN
    agreement · S1 kinematic scenario strata · ADE / FDE / miss

each with an **episode-cluster bootstrap** and a **paired** test against BOTH
trivial floors (**CV** and **hold-v0**).

WHY IT EXISTS — one scalar was hiding three different competencies
-----------------------------------------------------------------
MEASURED on flagship v1's own 881 windows (`v2_tier0_probe.py`, reproduced by
this module to 4 decimals): ADE beats CV by +0.4106 m [+0.2050, +0.6240]
separated — but the **cross-track** half beats it by +0.7720 [+0.4166, +1.1914]
(separated) while the **along-track** half is +0.2543 [−0.0278, +0.5304] (NOT
separated) and **speed MAE is −0.0032 [−0.1285, +0.1182]** — the deployed arm
does not track speed better than constant velocity at all. On the 639 steady
windows it is **2.0× worse than hold-v0** (0.4231 vs 0.2109 m/s, paired Δ
−0.2122 separated *against* us) while winning brake/accel decisively. On
straights its heading MAE is **7.980°** vs CV's **1.399°**. A single ADE column
hides all three; this module makes them columns.

RELATIONSHIP TO THE REST OF THE PACKAGE
---------------------------------------
* ``v2_tier0_probe.py`` — the standalone reference implementation this module
  is a promotion of. The geometry here is **ported verbatim** (same tangent
  convention, same arc-length resampling, same Menger curvature) precisely so
  the probe's sanity pin against MODEL_REGISTRY §1.2 still holds. If the two
  ever disagree, one of them changed and the pin fails loud.
* ``efficiency.py`` — the integration pattern copied here: runs automatically
  inside ``runner.run_one``, writes a self-describing versioned block into the
  same ``results/<key>.json``, ships its own dashboard panel, is test-pinned.
* ``ci.py`` — the ONLY admissible estimator. The legacy ``heldout ± ci95`` is
  ``overlapping_holdout_se``, measured **1.28–2.06× too narrow**; this module
  refuses to emit it (:func:`assert_no_deprecated_estimator`, run on every
  block before it is returned).

THE SURFACE, AND WHAT IT COSTS
------------------------------
``rollout.save_windows`` persists ``pred/gt/cv [N,4,2] · eid · speed ·
head_deg`` — 4 waypoints 0.5 s apart, because ``rollout.collect`` computes the
dense ``wp_full [b,20,2]`` and discards 16 of 20 steps at ``rollout.py:94``.
Everything needing 10 Hz derivatives (jerk, adopted comfort bounds, a real
curvature *profile*, decel-onset lead time) is therefore **tier 1** and blocked
on that one line, not on new science (suite §7 E2).

REFUSALS HONOURED (suite §6) — deliberately NOT implemented here
----------------------------------------------------------------
headway / distance-keeping / TTC (no lead-agent state exists — ``lead_state``
is a ``None`` stub) · any VTARGET-referenced 2 s target-speed metric (refuted
with numbers: 1.65 vs 0.475 MAE against holding v0) · intersection / roundabout
/ merge *capability* at a 2 s horizon (the events are 5–20 s; S1 emits
**kinematic signatures** which must never be renamed "intersection") ·
lane-centre deviation (no lane geometry exists; ``LANE_HALF_M`` is an assumed
constant) · naive curvature MAE at this resolution (measured 24× the signal).

Run:
    python -m taniteval.driving --model flagship-30k
    python -m taniteval.driving --model all          # backfill every dump
    python -m taniteval.driving --leaderboard        # markdown table
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import ci as _ci  # noqa: E402

RES = Path("/root/taniteval/results")

# --------------------------------------------------------------------------- #
# Block identity — every artifact says exactly which contract produced it       #
# --------------------------------------------------------------------------- #
BLOCK = "taniteval.driving/tier0"
VERSION = "2.0.0"
SPEC = ("TanitAD Research Hub/Benchmarks & Eval/TANITEVAL_V2_METRIC_SUITE.md"
        " (v1, 2026-07-21)")

# --------------------------------------------------------------------------- #
# Constants. Every threshold is marked MEASURED / PUBLISHED / PROPOSED, per the  #
# suite's label legend. An unmarked threshold is a bug in this file.             #
# --------------------------------------------------------------------------- #
DT_WP = 0.5          # MEASURED — the 4 persisted waypoints are 0.5 s apart
HORIZON_S = 2.0      # MEASURED — WP_STEPS (5,10,15,20) @10 Hz
EPS = 1e-6
N_BOOT = _ci.DEFAULT_N_BOOT              # 2000
# MEASURED 2026-07-21 on this box: the WHOLE block (16 headline intervals, 22
# paired floor tests, 3 longitudinal regimes, 4 speed strata, 3 curvature
# buckets, 3 kinematic signatures) costs **2.4 s of CPU at the full
# decision-grade B=2000** on 881 windows. There is therefore no "cheap inline
# variant" to trade off: the block that runs inside every eval is the same
# block the leaderboard quotes.
N_BOOT_INLINE = N_BOOT

BRAKE_A = 0.5        # PROPOSED (suite §3.1 L1/L2) — |mean realised accel|, m/s^2
K_SIGNAL = 1e-3      # PROPOSED (suite §3.2 T4) — 1/m; below this GT curvature
#                      is "straight" and the predicted sign is free
STOP_V_MS = 0.3      # MEASURED — refb_labels.STOP_V_MS
MOVING_V_MS = 1.0    # MEASURED — refb_labels.MOVING_V_MS
SUSTAINED_TURN_DEG = 15.0   # PROPOSED (suite §3.5 S1)
YAW_SIGNIF_DEG = 1.0        # PROPOSED — |per-knot heading increment| below this
#                             is knot jitter, not a turn
HEADING_EXCEED_DEG = 5.0    # PROPOSED — R5 exceedance companion to the median
MIN_N_STRATUM = 30          # MEASURED convention — pathspeed.run's `min_n`

# ⚠️ CURVATURE BUCKETS. The suite (§3.2 T3) and every MEASURED number in it use
# straight <5° / gentle 5–15° / sharp ≥15°, which is what `v2_tier0_probe.py`
# implements and what this module reproduces (n = 634 / 103 / 144 on
# flagship-30k). `driving_diagnostic.curvature_bucket` — used by `bench.py`'s
# own `by_curvature` panel — puts the gentle/sharp boundary at
# CURV_GENTLE_DEG = **20.0**, not 15. The two panels therefore bucket
# differently. Thresholds are recorded in every emitted block so a number can
# never be read against the wrong split; the divergence is escalated (E9).
CURV_STRAIGHT_DEG = 5.0     # MEASURED — driving_diagnostic.CURV_STRAIGHT_DEG
CURV_SHARP_DEG = 15.0       # PROPOSED here; driving_diagnostic says 20.0

# --------------------------------------------------------------------------- #
# Estimator policy — enforced, not documented                                   #
# --------------------------------------------------------------------------- #
DECISION_ESTIMATORS = frozenset({
    "episode_cluster_bootstrap", "paired_episode_cluster_bootstrap"})
DEPRECATED_ESTIMATOR = "overlapping_holdout_se"
ESTIMATOR_NOTE = (
    "episode-cluster bootstrap over the val EPISODES (paired form for every "
    "arm-vs-floor comparison). The legacy `heldout ± ci95` is "
    "`overlapping_holdout_se`, measured 1.28-2.06x too narrow across 10 arms "
    "(Project Steering/CI_RECOMPUTE_2026-07-20.json); it is refused by this "
    "block, not merely discouraged.")

# The MODEL_REGISTRY §1.2 pin. If these drift the artifact or the geometry
# convention changed — fail loud rather than publish. Identical to
# `v2_tier0_probe.SANITY` by construction (same numbers, same tolerance).
SANITY_ARM = "flagship-30k"
SANITY = {"ade_0_2s": 0.4271, "cv_ade_0_2s": 0.8377,
          "long_rmse_2s_m": 1.042, "lat_rmse_2s_m": 0.360,
          "top10pct_speed_bias_mps": 0.659, "top10pct_long_rmse_2s_m": 1.379}
SANITY_TOL = 0.005

# Metrics where SMALLER is better. Used only for presentation (the paired tests
# are always oriented floor − model, i.e. positive = the model wins).
LOWER_IS_BETTER = {
    "ade_0_2s", "fde_2s", "miss_2m", "long_abs_2s_m", "speed_mae_mps",
    "progress_abs_err_m", "lat_abs_2s_m", "heading_mae_2s_deg",
    "heading_med_2s_deg", "heading_exceed_5deg", "pathgeom_crosstrack_m"}
# Signed diagnostics — a bias has no floor comparison, only a direction.
SIGNED_DIAGNOSTICS = ("long_signed_2s_m", "lat_signed_2s_m", "speed_bias_mps",
                      "progress_signed_err_m")


# ========================================================================== #
# geometry — PORTED VERBATIM from v2_tier0_probe.py (the pathspeed.py           #
# conventions at the 4-waypoint surface). Do not "clean up": the registry pin   #
# is a pin on these exact conventions.                                          #
# ========================================================================== #
def prepend_origin(p):
    """[N,4,2] -> [N,5,2] with the observed pose (0,0) at t=0."""
    return torch.cat([torch.zeros(p.shape[0], 1, 2, dtype=p.dtype), p], dim=1)


def tangents(p):
    """Unit direction of travel over each 0.5 s segment. [N,4,2] -> [N,4,2]."""
    full = prepend_origin(p)
    d = full[:, 1:] - full[:, :-1]
    n = d.norm(dim=-1, keepdim=True)
    t = torch.where(n > EPS, d / n.clamp_min(EPS), torch.zeros_like(d))
    fwd = torch.tensor([1.0, 0.0])
    for i in range(t.shape[1]):                     # carry last valid forward
        bad = t[:, i].norm(dim=-1) <= EPS
        if bad.any():
            t[bad, i] = t[bad, i - 1] if i > 0 else fwd
    return t


def frenet(pred, gt):
    """Signed along/cross residual of pred vs GT on the GT tangent frame.

    along + = pred is AHEAD of GT along the path; cross + = pred is LEFT.
    Orthonormal basis => along^2 + cross^2 == ||pred-gt||^2 exactly."""
    tg = tangents(gt)
    nv = torch.stack([-tg[..., 1], tg[..., 0]], dim=-1)
    r = pred - gt
    return (r * tg).sum(-1), (r * nv).sum(-1)


def step_speed(p, dt=DT_WP):
    """Mean speed over each 0.5 s segment (m/s).

    NOT an instantaneous speed — at this surface it is a 0.5 s box-average,
    which is why jerk is tier-1 and why the CQ/TR numbers below are
    CONSERVATIVE (smoothing hides jitter; the dense path will make them
    stricter, not looser)."""
    full = prepend_origin(p)
    return (full[:, 1:] - full[:, :-1]).norm(dim=-1) / dt


def heading_deg(p):
    t = tangents(p)
    return torch.atan2(t[..., 1], t[..., 0]) * 180.0 / math.pi


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def arclength(p):
    full = prepend_origin(p)
    return (full[:, 1:] - full[:, :-1]).norm(dim=-1).sum(1)


def menger_curvature(p):
    """Signed Menger curvature (1/m) at each interior knot of the polyline.

    CAVEAT, MEASURED: at 0.5 s knot spacing this is dominated by knot jitter at
    low speed (spacing ~1.93 m, |k_gt| 0.0495 1/m, model curvature MAE 1.2015
    1/m = 24x the signal). Use the SIGN, not the magnitude, at tier-0."""
    full = prepend_origin(p)
    a, b, c = full[:, :-2], full[:, 1:-1], full[:, 2:]
    cross = ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
             - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    denom = ((b - a).norm(dim=-1) * (c - b).norm(dim=-1)
             * (c - a).norm(dim=-1)).clamp_min(EPS)
    return 2.0 * cross / denom


def path_geometry_crosstrack(pred, gt, m=8):
    """T2 — SPEED-DECOUPLED lateral error: resample both paths at the SAME arc
    lengths d_j = j/m * min(L_pred, L_gt), take the RMS perpendicular deviation.

    Two paths tracing the same GEOMETRY at different speeds score ~0, which is
    how a lateral error is attributed to *geometry* rather than to speed leaking
    into the cross-track term. This is ``pathspeed.path_geometry_crosstrack`` at
    the 4-knot surface."""
    fp, fg = prepend_origin(pred), prepend_origin(gt)

    def cum(f):
        return torch.cat([torch.zeros(f.shape[0], 1),
                          (f[:, 1:] - f[:, :-1]).norm(dim=-1).cumsum(1)], 1)
    cp, cg = cum(fp), cum(fg)
    L = torch.minimum(cp[:, -1], cg[:, -1])
    j = torch.arange(1, m + 1, dtype=torch.float32)
    q = L[:, None] * (j[None, :] / m)

    def interp(poly, c, q):
        qc = torch.minimum(q, c[:, -1:])
        idx = torch.searchsorted(c.contiguous(), qc.contiguous(),
                                 right=True).clamp(1, poly.shape[1] - 1)
        c0, c1 = torch.gather(c, 1, idx - 1), torch.gather(c, 1, idx)
        w = ((qc - c0) / (c1 - c0).clamp_min(EPS)).clamp(0, 1).unsqueeze(-1)
        g0 = torch.gather(poly, 1, (idx - 1).unsqueeze(-1).expand(-1, -1, 2))
        g1 = torch.gather(poly, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        return g0 * (1 - w) + g1 * w
    Gp, Gg = interp(fp, cp, q), interp(fg, cg, q)
    tg = torch.empty_like(Gg)
    tg[:, 1:] = Gg[:, 1:] - Gg[:, :-1]
    tg[:, 0] = Gg[:, 0]
    tn = tg / tg.norm(dim=-1, keepdim=True).clamp_min(EPS)
    nv = torch.stack([-tn[..., 1], tn[..., 0]], -1)
    cross = ((Gp - Gg) * nv).sum(-1)
    pg = cross.pow(2).mean(1).sqrt()
    return torch.where(L > EPS, pg, torch.zeros_like(pg))


# ========================================================================== #
# trivial floors constructible from the persisted meta alone                   #
# ========================================================================== #
def hold_v0(v0, n=4):
    """HOLD-V0 — go straight at the observed entry speed.

    The floor every LONGITUDINAL metric must beat (suite R1), and the one
    VTARGET provably loses to at 2 s (MAE 1.65 vs 0.475, MODEL_REGISTRY §4.1).
    On a steady window it is very nearly the right answer — which is exactly
    the point of L1."""
    t = torch.arange(1, n + 1, dtype=torch.float32) * DT_WP
    return torch.stack([v0[:, None] * t[None, :], torch.zeros(len(v0), n)], -1)


FLOORS = ("cv", "holdv0")
FLOOR_DESC = {
    "cv": "constant velocity (persisted by rollout.collect)",
    "holdv0": "hold-v0: go straight at the observed entry speed",
}


# ========================================================================== #
# per-window components — ONE place, keys are the suite's metric names          #
# ========================================================================== #
def per_window(pred, gt):
    de = torch.linalg.norm(pred - gt, dim=-1)              # [N,4]
    al, cr = frenet(pred, gt)
    vp, vg = step_speed(pred), step_speed(gt)
    kp, kg = menger_curvature(pred), menger_curvature(gt)
    straight = kg.abs() < K_SIGNAL
    head_err = wrap_deg(heading_deg(pred)[:, -1] - heading_deg(gt)[:, -1]).abs()
    return {
        # --- trajectory ---------------------------------------------------- #
        "ade_0_2s": de.mean(1),
        "fde_2s": de[:, -1],
        "miss_2m": (de[:, -1] > 2.0).float(),
        # --- LONGITUDINAL (L1-L4) ------------------------------------------ #
        "long_abs_2s_m": al[:, -1].abs(),
        "long_signed_2s_m": al[:, -1],
        "long_sq_2s": al[:, -1].pow(2),
        "speed_mae_mps": (vp - vg).abs().mean(1),
        "speed_bias_mps": (vp - vg).mean(1),
        "progress_abs_err_m": (arclength(pred) - arclength(gt)).abs(),
        "progress_signed_err_m": arclength(pred) - arclength(gt),
        # --- LATERAL (T1-T4) ------------------------------------------------ #
        "lat_abs_2s_m": cr[:, -1].abs(),
        "lat_signed_2s_m": cr[:, -1],
        "lat_sq_2s": cr[:, -1].pow(2),
        "heading_mae_2s_deg": head_err,
        # R5: heading is heavy-tailed (CI [2.34, 12.02] around a mean of 6.61
        # on flagship v1) -> the median + an exceedance RATE are the honest
        # reducers; the mean is kept only for continuity with published rows.
        "heading_med_2s_deg": head_err,
        "heading_exceed_5deg": (head_err > HEADING_EXCEED_DEG).float(),
        "pathgeom_crosstrack_m": path_geometry_crosstrack(pred, gt),
        "curv_sign_agree": ((kp.sign() == kg.sign()) | straight).float().mean(1),
    }


# How each component becomes a scalar (ci.REDUCERS name or a callable).
REDUCE = {"heading_med_2s_deg": "median"}

# Emitted as headline rows (long_sq/lat_sq are internal, surfaced via `rmse`).
HEADLINE = ("ade_0_2s", "fde_2s", "miss_2m",
            "long_abs_2s_m", "long_signed_2s_m", "speed_mae_mps",
            "speed_bias_mps", "progress_abs_err_m", "progress_signed_err_m",
            "lat_abs_2s_m", "lat_signed_2s_m", "heading_mae_2s_deg",
            "heading_med_2s_deg", "heading_exceed_5deg",
            "pathgeom_crosstrack_m", "curv_sign_agree")
# Paired against BOTH floors (suite deliverable 1). Signed biases and
# curv_sign_agree are excluded: a bias has no "floor" and sign agreement is
# reported against the floor's own rate, not as a paired win.
PAIRED = ("ade_0_2s", "fde_2s", "miss_2m", "long_abs_2s_m", "speed_mae_mps",
          "progress_abs_err_m", "lat_abs_2s_m", "heading_mae_2s_deg",
          "heading_med_2s_deg", "heading_exceed_5deg",
          "pathgeom_crosstrack_m")


# ========================================================================== #
# strata (S1) — kinematic SIGNATURES. Never rename these "intersection" or      #
# "roundabout": the events are 5-20 s and the horizon is 2 s (suite §6.3).      #
# ========================================================================== #
def regimes(gt, v0):
    """Longitudinal regime from the REALISED future speed — the only
    target-speed reference that survived measurement (VTARGET at 2 s is
    refuted, MODEL_REGISTRY §4.1)."""
    a = ((step_speed(gt)[:, -1] - v0) / HORIZON_S).numpy()
    return np.where(a <= -BRAKE_A, "brake",
                    np.where(a >= BRAKE_A, "accel", "steady"))


def curv_buckets(head_deg):
    h = np.abs(np.asarray(head_deg, dtype=float))
    return np.where(h < CURV_STRAIGHT_DEG, "straight",
                    np.where(h < CURV_SHARP_DEG, "gentle", "sharp"))


def speed_strata(v0):
    q = torch.quantile(v0, torch.tensor([1 / 3, 2 / 3, 0.90]))
    return ({"low": v0 < q[0], "med": (v0 >= q[0]) & (v0 < q[1]),
             "high": v0 >= q[1], "top10pct": v0 >= q[2]},
            [round(float(x), 3) for x in q])


def kinematic_strata(gt, v0, head_deg):
    """S1 — every stratum derivable from `poses` alone, at tier 0.

    `lane_change_signature` and the episode-level turn/roundabout signature are
    tier-1 (they need the LATMANEUVER mint / episode-level aggregation) and are
    deliberately absent."""
    vs = step_speed(gt)                                   # [N,4]
    v_end, v_min = vs[:, -1], vs.min(dim=1).values
    h = heading_deg(gt)
    inc = wrap_deg(torch.cat([h[:, :1], h[:, 1:] - h[:, :-1]], dim=1))
    pos = (inc > YAW_SIGNIF_DEG).sum(1)
    neg = (inc < -YAW_SIGNIF_DEG).sum(1)
    same_sign = (pos == 0) | (neg == 0)
    net = torch.as_tensor(np.abs(np.asarray(head_deg, dtype=float)),
                          dtype=torch.float32)
    return {
        "launch_from_stop": ((v0 < STOP_V_MS) & (v_end > MOVING_V_MS)).numpy(),
        "stop_approach": (v_min < STOP_V_MS).numpy(),
        "sustained_turn": (same_sign & (net >= SUSTAINED_TURN_DEG)).numpy(),
    }


# ========================================================================== #
# bootstrap driver — ONE set of episode resamples reused by every metric        #
# ========================================================================== #
class _Draws:
    """Cached episode-cluster resamples.

    ``ci.episode_cluster_bootstrap`` regenerates its draws per call, which is
    correct but O(metrics x n_boot) index concatenations. This suite asks ~30
    metrics x 2 floors the SAME question over the SAME episodes, so the draws
    are built once and shared — which also makes every interval in one block
    mutually consistent (ade@2s and along-track move together exactly as they
    do in reality), the same property ``ci.bootstrap_metrics`` provides.
    """

    def __init__(self, eid, n_boot=N_BOOT, seed=0):
        self.uniq, idx = _ci.episode_index(eid)
        self.n_boot, self.seed = int(n_boot), int(seed)
        self.sel = [s for s in _ci._draws(self.uniq, idx, self.n_boot, seed)]

    @property
    def n_episodes(self):
        return int(len(self.uniq))


def _reducer(name_or_call):
    if callable(name_or_call):
        return name_or_call
    return _ci.REDUCERS[name_or_call]


def _interval(vals, draws, reduce="mean"):
    """Percentile CI on a per-window statistic, resampling EPISODES.

    Carries its own provenance so the number can never be quoted without the
    construction that produced it (suite R0)."""
    v = np.asarray(vals, dtype=np.float64)
    red = _reducer(reduce)
    b = np.fromiter((red(v[s]) for s in draws.sel), dtype=np.float64,
                    count=draws.n_boot)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": round(float(red(v)), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "ci95": round(float((hi - lo) / 2.0), 4),
            "se": round(float(b.std(ddof=1)), 4),
            "reducer": reduce if isinstance(reduce, str) else "callable",
            "n_windows": int(v.size), "n_episodes": draws.n_episodes,
            "n_boot": draws.n_boot,
            "estimator": "episode_cluster_bootstrap"}


def _paired(a, b, draws, reduce="mean"):
    """CI on reduce(a) - reduce(b) with the SAME resampled episodes each draw.

    Orientation everywhere in this module is **floor - model**, so a POSITIVE
    delta means the model wins. ``separated`` is the decision predicate: never
    combine two single-arm intervals in quadrature, the estimates are not
    independent."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    red = _reducer(reduce)
    d = np.fromiter((red(a[s]) - red(b[s]) for s in draws.sel),
                    dtype=np.float64, count=draws.n_boot)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(red(a) - red(b)), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "ci95": round(float((hi - lo) / 2.0), 4),
            "p_delta_gt0": round(float((d > 0).mean()), 4),
            "separated": bool(lo > 0 or hi < 0),
            "favours": ("model" if (lo > 0) else
                        "floor" if (hi < 0) else "tie"),
            "reducer": reduce if isinstance(reduce, str) else "callable",
            "n_windows": int(a.size), "n_episodes": draws.n_episodes,
            "n_boot": draws.n_boot,
            "estimator": "paired_episode_cluster_bootstrap"}


def _sub_draws(eid, mask, n_boot, seed):
    idx = np.flatnonzero(np.asarray(mask))
    sub_eid = [str(eid[i]) for i in idx]
    return idx, _Draws(sub_eid, n_boot=n_boot, seed=seed)


# ========================================================================== #
# the estimator policy, ENFORCED                                               #
# ========================================================================== #
def assert_no_deprecated_estimator(obj, _path="block"):
    """Walk an emitted block; raise if any interval names the deprecated
    estimator or omits its estimator entirely.

    This is the mechanical form of the program rule *"never quote an interval
    without its estimator"*. `overlapping_holdout_se` is 1.28-2.06x too narrow;
    a driving-capability verdict computed with it would be a false separation,
    which is worse than no number at all. Called on every block before it is
    returned or written, so the refusal cannot be forgotten."""
    if isinstance(obj, dict):
        est = obj.get("estimator")
        if isinstance(est, str) and est not in DECISION_ESTIMATORS:
            raise ValueError(
                f"{_path}: refusing to emit a driving metric computed with "
                f"estimator {est!r}. Decision-grade estimators are "
                f"{sorted(DECISION_ESTIMATORS)}; {DEPRECATED_ESTIMATOR!r} is "
                f"measured 1.28-2.06x too narrow and is not admissible.")
        # An interval-shaped dict with no NAMED estimator is equally
        # inadmissible — that is how an unlabelled interval got published for
        # months. `ci`/`delta` fields inside a `verdict` row carry the estimator
        # forward explicitly, so this catches omissions, not restatements.
        if not isinstance(est, str) and {"lo", "hi"} <= set(obj):
            raise ValueError(f"{_path}: interval without a named estimator "
                             f"(keys {sorted(obj)})")
        for k, v in obj.items():
            assert_no_deprecated_estimator(v, f"{_path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_deprecated_estimator(v, f"{_path}[{i}]")
    return True


# ========================================================================== #
# THE BLOCK                                                                    #
# ========================================================================== #
def tier0(win, n_boot=N_BOOT, seed=0, arm=None) -> dict:
    """The full tier-0 driving-capability block for one arm's window dump.

    ``win`` is the in-memory dict from ``rollout.collect`` / the loaded
    ``windows_<key>.pt``: ``pred/gt/cv [N,4,2] · eid · speed · head_deg``.
    CPU-only, no GPU touched, seconds of wall clock — so it can run inside
    every eval without competing for the GPU or delaying anything."""
    pred, gt, cv = win["pred"].float(), win["gt"].float(), win["cv"].float()
    eid, v0 = list(win["eid"]), win["speed"].float()
    hv = hold_v0(v0, n=pred.shape[1])
    pw = {"model": per_window(pred, gt), "cv": per_window(cv, gt),
          "holdv0": per_window(hv, gt)}
    draws = _Draws(eid, n_boot=n_boot, seed=seed)

    out = {
        "block": BLOCK, "version": VERSION, "spec": SPEC,
        "arm": arm, "n_windows": int(pred.shape[0]),
        "n_episodes": draws.n_episodes,
        "wp_steps": win.get("wp_steps"),
        "surface": ("4 waypoints 0.5 s apart (tier-0). The dense 20-step path "
                    "is computed by rollout.collect and discarded at "
                    "rollout.py:94 -> jerk, comfort bounds, curvature PROFILE "
                    "and decel-onset lead time are tier-1 (suite E2)."),
        "claim_strength": "open-loop / weak (arXiv:2605.00066)",
        "estimator": {
            "interval": "episode_cluster_bootstrap",
            "delta": "paired_episode_cluster_bootstrap",
            "n_boot": int(n_boot), "seed": int(seed),
            "resampling_unit": "val episode",
            "orientation": "every paired delta is floor - model; "
                           "positive = the model wins",
            "deprecated_and_refused": DEPRECATED_ESTIMATOR,
            "estimator_note": ESTIMATOR_NOTE},
        "thresholds": {
            "brake_accel_mps2": BRAKE_A, "brake_accel_mark": "PROPOSED",
            "curvature_signal_1_per_m": K_SIGNAL,
            "curv_straight_deg": CURV_STRAIGHT_DEG,
            "curv_sharp_deg": CURV_SHARP_DEG,
            "curv_bucket_note": (
                "straight <5 / gentle 5-15 / sharp >=15 deg, as in the suite "
                "and v2_tier0_probe.py. NOTE: driving_diagnostic."
                "curvature_bucket (used by bench.by_curvature) puts the "
                "gentle/sharp boundary at 20 deg - the two panels bucket "
                "differently on purpose-of-record, not by accident (E9)."),
            "stop_v_ms": STOP_V_MS, "moving_v_ms": MOVING_V_MS,
            "sustained_turn_deg": SUSTAINED_TURN_DEG,
            "heading_exceed_deg": HEADING_EXCEED_DEG,
            "min_n_stratum": MIN_N_STRATUM},
        "floors": {f: FLOOR_DESC[f] for f in FLOORS},
        "refused": {
            "headway_ttc_distance_keeping":
                "no lead-agent state exists (lead_state is a None stub)",
            "vtarget_referenced_speed_at_2s":
                "refuted: 1.65 vs 0.475 MAE against holding v0 at 2 s",
            "intersection_roundabout_merge_capability":
                "events are 5-20 s, horizon is 2 s; S1 emits KINEMATIC "
                "signatures only and they must never be renamed",
            "lane_centre_deviation":
                "no lane geometry exists; LANE_HALF_M is an assumed constant",
            "curvature_mae_at_this_resolution":
                "measured 24x the signal; SIGN agreement only at tier 0"},
    }

    # ---- headline, every metric with its interval ------------------------- #
    out["headline"] = {k: _interval(pw["model"][k], draws,
                                    REDUCE.get(k, "mean"))
                       for k in HEADLINE}

    # ---- gate-readable cluster-bootstrap block ---------------------------- #
    # The headline intervals ARE the episode-cluster bootstrap; re-expose them
    # in the ``{"model": {metric: {...}}}`` shape ``run_gate._read_eval_metric``
    # reads, so a v4 gate finds a decision-grade interval for the primary and
    # every interval-bearing secondary and NEVER falls back to the deprecated
    # ``overlapping_holdout_se`` (the ⭐ gate bug, 2026-07-22). Additive: nothing
    # else in the block changes, and every node names episode_cluster_bootstrap
    # so ``assert_no_deprecated_estimator`` stays satisfied.
    out["primary_ci"] = "episode_cluster_bootstrap"
    out["cluster_bootstrap"] = {"model": dict(out["headline"])}
    out["floor_values"] = {
        f: {k: {"value": round(float(_reducer(REDUCE.get(k, "mean"))(
            np.asarray(pw[f][k], dtype=float))), 4),
            "reducer": REDUCE.get(k, "mean")} for k in HEADLINE}
        for f in FLOORS}

    # RMSE forms (the registry quotes RMSE, not MAE, for long/lat)
    lsq = float(pw["model"]["long_sq_2s"].mean())
    csq = float(pw["model"]["lat_sq_2s"].mean())
    out["rmse"] = {"long_rmse_2s_m": round(math.sqrt(lsq), 4),
                   "lat_rmse_2s_m": round(math.sqrt(csq), 4),
                   "long_frac_of_2s_sqerr": round(lsq / (lsq + csq + EPS), 4)}

    # ---- the decisive test: WHERE does the win live? ---------------------- #
    out["vs_floor_paired"] = {
        f: {k: _paired(pw[f][k], pw["model"][k], draws, REDUCE.get(k, "mean"))
            for k in PAIRED} for f in FLOORS}

    # ---- L1 CRUISE-QUALITY / L2 TRANSIENT-RESPONSE ------------------------ #
    reg = regimes(gt, v0)
    out["longitudinal_regime"] = {}
    for r in ("brake", "steady", "accel"):
        m = reg == r
        if not m.any():
            continue
        idx, sd = _sub_draws(eid, m, n_boot, seed)
        row = {"n": int(m.sum()),
               "metric": "L1 CRUISE-QUALITY" if r == "steady"
               else "L2 TRANSIENT-RESPONSE",
               "low_confidence": bool(int(m.sum()) < MIN_N_STRATUM)}
        for f in ("model",) + FLOORS:
            row[f"{f}_ade"] = round(float(pw[f]["ade_0_2s"][idx].mean()), 4)
            row[f"{f}_speed_mae"] = round(
                float(pw[f]["speed_mae_mps"][idx].mean()), 4)
        row["vs_holdv0_speed_mae_paired"] = _paired(
            np.asarray(pw["holdv0"]["speed_mae_mps"])[idx],
            np.asarray(pw["model"]["speed_mae_mps"])[idx], sd)
        row["vs_cv_speed_mae_paired"] = _paired(
            np.asarray(pw["cv"]["speed_mae_mps"])[idx],
            np.asarray(pw["model"]["speed_mae_mps"])[idx], sd)
        row["vs_holdv0_long_abs_paired"] = _paired(
            np.asarray(pw["holdv0"]["long_abs_2s_m"])[idx],
            np.asarray(pw["model"]["long_abs_2s_m"])[idx], sd)
        out["longitudinal_regime"][r] = row

    # ---- speed strata ------------------------------------------------------ #
    masks, thr = speed_strata(v0)
    out["speed_strata_thresholds_mps"] = thr
    out["by_speed"] = {}
    for lab, m in masks.items():
        idx = m.nonzero(as_tuple=True)[0]
        sa = float(pw["model"]["long_sq_2s"][idx].mean())
        sc = float(pw["model"]["lat_sq_2s"][idx].mean())
        out["by_speed"][lab] = {
            "n": int(len(idx)),
            "low_confidence": bool(len(idx) < MIN_N_STRATUM),
            "model_ade": round(float(pw["model"]["ade_0_2s"][idx].mean()), 4),
            "cv_ade": round(float(pw["cv"]["ade_0_2s"][idx].mean()), 4),
            "holdv0_ade": round(float(pw["holdv0"]["ade_0_2s"][idx].mean()), 4),
            "long_rmse_2s_m": round(math.sqrt(sa), 4),
            "lat_rmse_2s_m": round(math.sqrt(sc), 4),
            "long_frac_of_2s_sqerr": round(sa / (sa + sc + EPS), 4),
            "speed_bias_mps": round(
                float(pw["model"]["speed_bias_mps"][idx].mean()), 4)}

    # ---- T3/T4 by curvature ------------------------------------------------ #
    cb = curv_buckets(win["head_deg"])
    out["by_curvature"] = {}
    for lab in ("straight", "gentle", "sharp"):
        m = cb == lab
        if not m.any():
            continue
        idx, sd = _sub_draws(eid, m, n_boot, seed)
        out["by_curvature"][lab] = {
            "n": int(m.sum()),
            "low_confidence": bool(int(m.sum()) < MIN_N_STRATUM),
            "model_heading_mae_deg": round(
                float(pw["model"]["heading_mae_2s_deg"][idx].mean()), 3),
            "cv_heading_mae_deg": round(
                float(pw["cv"]["heading_mae_2s_deg"][idx].mean()), 3),
            "holdv0_heading_mae_deg": round(
                float(pw["holdv0"]["heading_mae_2s_deg"][idx].mean()), 3),
            "model_heading_med_deg": round(float(np.median(
                np.asarray(pw["model"]["heading_mae_2s_deg"])[idx])), 3),
            "cv_heading_med_deg": round(float(np.median(
                np.asarray(pw["cv"]["heading_mae_2s_deg"])[idx])), 3),
            "model_curv_sign_agree": round(
                float(pw["model"]["curv_sign_agree"][idx].mean()), 4),
            "cv_curv_sign_agree": round(
                float(pw["cv"]["curv_sign_agree"][idx].mean()), 4),
            "holdv0_curv_sign_agree": round(
                float(pw["holdv0"]["curv_sign_agree"][idx].mean()), 4),
            "vs_cv_heading_paired": _paired(
                np.asarray(pw["cv"]["heading_mae_2s_deg"])[idx],
                np.asarray(pw["model"]["heading_mae_2s_deg"])[idx], sd),
            "vs_cv_ade_paired": _paired(
                np.asarray(pw["cv"]["ade_0_2s"])[idx],
                np.asarray(pw["model"]["ade_0_2s"])[idx], sd)}

    # ---- S1 kinematic signatures ------------------------------------------ #
    out["kinematic_strata"] = {
        "_naming_contract": ("KINEMATIC SIGNATURES derived from poses alone. "
                             "These are NOT scenario labels: no map, no agents "
                             "and no scenario ground truth exist (suite §2.3). "
                             "Never rename these 'intersection' or 'roundabout'.")}
    for lab, m in kinematic_strata(gt, v0, win["head_deg"]).items():
        n = int(m.sum())
        row = {"n": n, "frac": round(n / max(1, len(eid)), 4),
               "low_confidence": bool(n < MIN_N_STRATUM)}
        if n:
            idx, sd = _sub_draws(eid, m, n_boot, seed)
            for f in ("model",) + FLOORS:
                row[f"{f}_ade"] = round(float(pw[f]["ade_0_2s"][idx].mean()), 4)
                row[f"{f}_speed_mae"] = round(
                    float(pw[f]["speed_mae_mps"][idx].mean()), 4)
            if n >= 2:
                row["vs_cv_ade_paired"] = _paired(
                    np.asarray(pw["cv"]["ade_0_2s"])[idx],
                    np.asarray(pw["model"]["ade_0_2s"])[idx], sd)
        out["kinematic_strata"][lab] = row

    # ---- the READ: what a single ADE column would have hidden -------------- #
    out["verdict"] = _verdict(out)

    # ---- sanity pin vs MODEL_REGISTRY (flagship-30k only) ------------------ #
    if arm == SANITY_ARM:
        got = {"ade_0_2s": out["headline"]["ade_0_2s"]["mean"],
               "cv_ade_0_2s": out["floor_values"]["cv"]["ade_0_2s"]["value"],
               "long_rmse_2s_m": out["rmse"]["long_rmse_2s_m"],
               "lat_rmse_2s_m": out["rmse"]["lat_rmse_2s_m"],
               "top10pct_speed_bias_mps":
                   out["by_speed"]["top10pct"]["speed_bias_mps"],
               "top10pct_long_rmse_2s_m":
                   out["by_speed"]["top10pct"]["long_rmse_2s_m"]}
        out["sanity_vs_registry"] = {
            k: {"expected": SANITY[k], "got": got[k],
                "ok": bool(abs(got[k] - SANITY[k]) <= SANITY_TOL)}
            for k in SANITY}
        out["sanity_all_ok"] = all(v["ok"] for v in
                                   out["sanity_vs_registry"].values())

    assert_no_deprecated_estimator(out)
    return out


def _verdict(out) -> dict:
    """The one-screen read. Every field is a `separated` predicate, never a
    bare number — an unseparated win is a tie (suite R2)."""
    vs = out["vs_floor_paired"]

    def _rd(floor, key):
        d = vs[floor][key]
        return {"delta": d["delta"], "ci": [d["lo"], d["hi"]],
                "separated": d["separated"], "favours": d["favours"],
                "estimator": d["estimator"]}
    reg = out.get("longitudinal_regime", {})
    steady = reg.get("steady", {}).get("vs_holdv0_speed_mae_paired")
    straight = out.get("by_curvature", {}).get("straight", {})
    v = {
        "ade_vs_cv": _rd("cv", "ade_0_2s"),
        "along_track_vs_cv": _rd("cv", "long_abs_2s_m"),
        "cross_track_vs_cv": _rd("cv", "lat_abs_2s_m"),
        "speed_mae_vs_cv": _rd("cv", "speed_mae_mps"),
        "speed_mae_vs_holdv0": _rd("holdv0", "speed_mae_mps"),
        "path_geometry_vs_cv": _rd("cv", "pathgeom_crosstrack_m"),
    }
    if steady:
        v["cruise_speed_vs_holdv0"] = {
            "n": reg["steady"]["n"], "delta": steady["delta"],
            "ci": [steady["lo"], steady["hi"]],
            "separated": steady["separated"], "favours": steady["favours"],
            "estimator": steady["estimator"]}
    if straight:
        v["straight_heading_model_vs_cv_deg"] = {
            "n": straight["n"], "model": straight["model_heading_mae_deg"],
            "cv": straight["cv_heading_mae_deg"],
            "separated": straight["vs_cv_heading_paired"]["separated"],
            "favours": straight["vs_cv_heading_paired"]["favours"],
            "estimator": straight["vs_cv_heading_paired"]["estimator"]}
    # The headline sentence: where does an ADE win actually live?
    lat_sep = v["cross_track_vs_cv"]["separated"] and \
        v["cross_track_vs_cv"]["favours"] == "model"
    lon_sep = v["along_track_vs_cv"]["separated"] and \
        v["along_track_vs_cv"]["favours"] == "model"
    spd_sep = v["speed_mae_vs_cv"]["separated"] and \
        v["speed_mae_vs_cv"]["favours"] == "model"
    ade_sep = v["ade_vs_cv"]["separated"] and \
        v["ade_vs_cv"]["favours"] == "model"
    v["where_the_win_lives"] = (
        "lateral only" if (lat_sep and not lon_sep) else
        "longitudinal only" if (lon_sep and not lat_sep) else
        "both axes" if (lat_sep and lon_sep) else
        "neither axis separated")
    v["tracks_speed_better_than_cv"] = bool(spd_sep)
    v["beats_cv_ade_separated"] = bool(ade_sep)
    return v


# ========================================================================== #
# entry points — the efficiency.py wiring pattern, exactly                     #
# ========================================================================== #
def quick(win, n_boot=N_BOOT_INLINE, arm=None) -> dict:
    """The DEFAULT axis: runs inside every ``runner run``.

    CPU-only, ~2.4 s of wall clock at the full B=2000, torch on CPU tensors
    only — so it never competes for the GPU, never delays an eval, and cannot
    move an accuracy number. The block it produces is decision-grade, not a
    preview: the backfill path computes the identical thing."""
    out = tier0(win, n_boot=n_boot, arm=arm)
    out["inline"] = True
    return out


def from_windows(path, n_boot=N_BOOT, arm=None) -> dict:
    win = torch.load(str(path), map_location="cpu", weights_only=False)
    out = tier0(win, n_boot=n_boot, arm=arm)
    out["artifact"] = str(path)
    return out


def arms_with_windows(res_dir=RES):
    """Every arm with a persisted window dump — the tier-0 population.

    NOT `registry.MODELS`: 9 of the 24 dumps (flagship-v16-ab-ft, flagship-v2-6k,
    the refc-v12 family, the overfit probes) have no registry entry, and the
    whole point of tier-0 is that it runs on whatever has been persisted."""
    return sorted(p.stem[len("windows_"):]
                  for p in Path(res_dir).glob("windows_*.pt"))


def run_and_save(key, res_dir=RES, n_boot=N_BOOT, seed=0) -> dict:
    """Tier-0 block for one arm -> results/driving_<key>.json, and merged into
    results/<key>.json when that arm has a benchmark artifact."""
    res_dir = Path(res_dir)
    wpath = res_dir / f"windows_{key}.pt"
    if not wpath.exists():
        raise SystemExit(
            f"{wpath} not found. Tier-0 needs a persisted window dump; "
            f"available: {arms_with_windows(res_dir)}")
    out = from_windows(wpath, n_boot=n_boot, arm=key)
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"driving_{key}.json").write_text(
        json.dumps(out, indent=2, default=str))
    acc = res_dir / f"{key}.json"
    if acc.exists():                      # keep the block WITH the accuracy row
        try:
            d = json.loads(acc.read_text())
            d["driving"] = out
            acc.write_text(json.dumps(d, indent=2, default=str))
        except Exception as ex:            # never let the panel break a result
            print(f"[driving] {key}: could not merge into {acc.name}: "
                  f"{type(ex).__name__}: {str(ex)[:120]}", flush=True)
    v = out["verdict"]
    sp = out["headline"]
    print(f"[driving] {key} n={out['n_windows']}/{out['n_episodes']}eps "
          f"ade={sp['ade_0_2s']['mean']:.4f} "
          f"[{sp['ade_0_2s']['lo']:.4f}, {sp['ade_0_2s']['hi']:.4f}] "
          f"| along {sp['long_abs_2s_m']['mean']:.4f} / cross "
          f"{sp['lat_abs_2s_m']['mean']:.4f} "
          f"| speed MAE {sp['speed_mae_mps']['mean']:.4f} vs hold-v0 "
          f"{out['floor_values']['holdv0']['speed_mae_mps']['value']:.4f} "
          f"| win lives: {v['where_the_win_lives']} "
          f"| tracks speed > CV: {v['tracks_speed_better_than_cv']}",
          flush=True)
    if out.get("sanity_all_ok") is False:
        raise SystemExit(f"[driving] {key}: SANITY PIN FAILED vs "
                         f"MODEL_REGISTRY — {out['sanity_vs_registry']}. Do "
                         f"not publish until the discrepancy is explained.")
    return out


def run_all(res_dir=RES, n_boot=N_BOOT, seed=0) -> dict:
    """Backfill every arm that has a `windows_*.pt` (the `eff-all` analogue)."""
    got = {}
    for key in arms_with_windows(res_dir):
        try:
            got[key] = run_and_save(key, res_dir=res_dir, n_boot=n_boot,
                                    seed=seed)
        except Exception as e:
            print(f"[driving-all] {key} FAILED: {type(e).__name__}: "
                  f"{str(e)[:160]}", flush=True)
    return got


# ========================================================================== #
# report panel (dashboard 04c) + leaderboard rows                              #
# ========================================================================== #
def _load_blocks(res_dir=RES):
    """Canonical `driving_<key>.json` artifacts only."""
    out = {}
    for f in sorted(Path(res_dir).glob("driving_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        key = d.get("arm") or f.stem[len("driving_"):]
        if f.name != f"driving_{key}.json" or d.get("block") != BLOCK:
            continue
        out[key] = d
    return out


def sep_tag(d, style="text"):
    """Three-way outcome of a paired floor test — NEVER two-way.

    "separated" alone is ambiguous: a separated interval that favours the FLOOR
    means the trivial baseline beat the model, which is the opposite reading.
    Six of our arms are CI-separated *against* themselves on speed MAE; a
    sep/tie rendering would have printed them as wins."""
    if not d:
        return "—"
    if not d.get("separated"):
        return {"text": "tie", "md": "**tie**"}[style]
    if d.get("favours") == "model":
        return {"text": "win", "md": "win"}[style]
    return {"text": "LOST", "md": "**LOST**"}[style]


def _eff_p50(res_dir, key):
    """This arm's p50 planning-tick from the efficiency panel, if it has one."""
    f = Path(res_dir) / f"eff_{key}.json"
    if not f.exists():
        return None, None
    try:
        d = json.loads(f.read_text())
        if "QUARANTINED" in d:
            return None, None
        for p in ("fp32", "tf32", "amp16"):
            if p in d:
                return d[p]["plan_step"]["p50_ms"], p
    except Exception:
        pass
    return None, None


def panel_rows(res_dir=RES) -> str:
    """HTML rows for the dashboard's driving-capability panel (04c)."""
    blocks = _load_blocks(res_dir)
    rows = []
    for key, d in sorted(blocks.items(),
                         key=lambda kv: kv[1]["headline"]["ade_0_2s"]["mean"]):
        h, fl, v = d["headline"], d["floor_values"], d["verdict"]
        ms, prec = _eff_p50(res_dir, key)
        st = d.get("by_curvature", {}).get("straight", {})
        cruise = v.get("cruise_speed_vs_holdv0", {})

        def pill(ok, txt):
            return (f"<span class='pill {'good' if ok else 'crit'}'>{txt}"
                    f"</span>")
        win = v["where_the_win_lives"]
        wincls = "good" if win == "both axes" else \
            "warn" if win in ("lateral only", "longitudinal only") else "crit"
        cr_ok = cruise.get("favours") != "floor"
        rows.append(
            f"<tr><td><div class='mname'>{key}</div>"
            f"<div class='meta'>{d['n_windows']} windows / "
            f"{d['n_episodes']} eps · B={d['estimator']['n_boot']}"
            f"{' · inline' if d.get('inline') else ''}</div></td>"
            f"<td class='r'><span class='big'>{h['ade_0_2s']['mean']:.4f}</span>"
            f"<div class='meta mono'>[{h['ade_0_2s']['lo']:.3f}, "
            f"{h['ade_0_2s']['hi']:.3f}] ep-cluster boot</div></td>"
            f"<td class='r mono'>{h['long_abs_2s_m']['mean']:.3f} / "
            f"{h['lat_abs_2s_m']['mean']:.3f}"
            f"<div class='meta'>{sep_tag(v['along_track_vs_cv'])}"
            f" / {sep_tag(v['cross_track_vs_cv'])}"
            f" vs CV</div></td>"
            f"<td class='r mono'>{h['speed_mae_mps']['mean']:.3f}"
            f"<div class='meta'>hold-v0 "
            f"{fl['holdv0']['speed_mae_mps']['value']:.3f} · "
            f"vs CV {sep_tag(v['speed_mae_vs_cv'])}</div></td>"
            f"<td class='r'>{pill(cr_ok, ('%.3f' % cruise['delta']) if cruise else '—')}"
            f"<div class='meta'>n={cruise.get('n', '—')} steady</div></td>"
            f"<td class='r mono'>{st.get('model_heading_mae_deg', '—')}°"
            f"<div class='meta'>CV {st.get('cv_heading_mae_deg', '—')}° · "
            f"n={st.get('n', '—')}</div></td>"
            f"<td class='r mono'>{h['curv_sign_agree']['mean']:.3f}</td>"
            f"<td class='r mono'>{'—' if ms is None else f'{ms:.1f} ms'}"
            f"<div class='meta'>{prec or ''}</div></td>"
            f"<td class='r'><span class='pill {wincls}'>{win}</span>"
            f"<div class='meta'>speed&gt;CV: "
            f"{'yes' if v['tracks_speed_better_than_cv'] else 'NO'}</div></td>"
            f"</tr>")
    return "\n".join(rows)


LEADERBOARD_COLS = ("arm", "ADE@2s m [ep-cluster boot CI95]",
                    "along / cross @2s m (vs CV)",
                    "speed MAE m/s: model vs hold-v0 (vs CV)",
                    "cruise Δ m/s vs hold-v0", "heading on straights °",
                    "κ-sign", "tick p50", "where the win lives")


def leaderboard_md(res_dir=RES, arms=None) -> str:
    """The markdown table for `Benchmarks & Eval/LEADERBOARD.md`.

    Regenerable: `python -m taniteval.driving --leaderboard`. Every cell is
    read from a `driving_<key>.json` / `eff_<key>.json` artifact — no number in
    it is transcribed by hand."""
    blocks = _load_blocks(res_dir)
    if arms:
        blocks = {k: v for k, v in blocks.items() if k in arms}
    lines = ["| " + " | ".join(LEADERBOARD_COLS) + " |",
             "|" + "|".join(["---"] * len(LEADERBOARD_COLS)) + "|"]
    order = arms or sorted(blocks,
                           key=lambda k: blocks[k]["headline"]["ade_0_2s"]["mean"])
    for key in order:
        d = blocks.get(key)
        if not d:
            continue
        h, fl, v = d["headline"], d["floor_values"], d["verdict"]
        ms, prec = _eff_p50(res_dir, key)
        st = d.get("by_curvature", {}).get("straight", {})
        cr = v.get("cruise_speed_vs_holdv0", {})
        crs = "—" if not cr else f"{cr['delta']:+.3f} {sep_tag(cr, 'md')}"
        lines.append(
            f"| {key} "
            f"| **{h['ade_0_2s']['mean']:.4f}** [{h['ade_0_2s']['lo']:.4f}, "
            f"{h['ade_0_2s']['hi']:.4f}] "
            f"| {h['long_abs_2s_m']['mean']:.3f} "
            f"{sep_tag(v['along_track_vs_cv'], 'md')} / "
            f"{h['lat_abs_2s_m']['mean']:.3f} "
            f"{sep_tag(v['cross_track_vs_cv'], 'md')} "
            f"| {h['speed_mae_mps']['mean']:.3f} vs "
            f"{fl['holdv0']['speed_mae_mps']['value']:.3f} "
            f"{sep_tag(v['speed_mae_vs_cv'], 'md')} "
            f"| {crs} "
            f"| {st.get('model_heading_mae_deg', '—')} vs CV "
            f"{st.get('cv_heading_mae_deg', '—')} "
            f"| {h['curv_sign_agree']['mean']:.3f} "
            f"| {'—' if ms is None else f'{ms:.1f} ms ({prec})'} "
            f"| {v['where_the_win_lives']} |")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser("taniteval.driving")
    ap.add_argument("--model", default=None,
                    help="registry key / window-dump key, or 'all'")
    ap.add_argument("--results", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leaderboard", action="store_true",
                    help="print the markdown leaderboard table and exit")
    ap.add_argument("--arms", default=None,
                    help="comma list restricting the leaderboard rows/order")
    a = ap.parse_args(argv)
    res = Path(a.results)
    if a.leaderboard:
        arms = [s.strip() for s in a.arms.split(",")] if a.arms else None
        print(leaderboard_md(res, arms))
        return
    if not a.model:
        ap.error("--model <key>|all, or --leaderboard")
    if a.model == "all":
        run_all(res, n_boot=a.n_boot, seed=a.seed)
    else:
        run_and_save(a.model, res_dir=res, n_boot=a.n_boot, seed=a.seed)


if __name__ == "__main__":
    main()
