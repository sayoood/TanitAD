"""taniteval.v0_antiecho — the THREE anti-echo controls the PI made binding 2026-08-16.

THE RULING THAT CREATED THIS MODULE
-----------------------------------
**Sayed, verbatim, 2026-08-16** (``Project Steering/V6F_PLANNER_DESIGN.md`` §1.4):

    "We can use v0 as input since it is measured and is not the future, but we
    should assure that the model/planner later is not cheating by just
    outputting v0 as longitudinal plan."

``v0`` (ego speed at t0) is therefore **ADMISSIBLE as a planner input** — it is
measured present state, available to any real vehicle at inference, and it is not
the answer. That half of the ruling *removes an argument*. This module implements
the other half, which **adds an obligation**.

⛔ WHY THE OBLIGATION IS THE LOAD-BEARING HALF
----------------------------------------------
Admitting ``v0`` opens a new way to fake competence: a planner can emit **"keep
doing v0"** as its longitudinal plan and score well, because holding the current
speed is a strong baseline on most windows. That is **skill attributed to a
copy** — and this programme has been fooled by that exact shape THREE times,
each of which looked like capability until a control was run:

============================  =========================================  ==============
failure                       what it scored                             what it was
============================  =========================================  ==============
**nav-echo** (flagship v1)    ``route_acc_nav`` = **1.0000**             an exact bijection of
                                                                         the nav it was fed
**T1 action echo**            S-curve reproduction **97.9 %** open-loop  **0.0 %** hold-action,
                                                                         ~5 % closed-loop
**P1 speed echo**             R² **0.995**                               **−0.72** under the
                                                                         v0 shuffle
============================  =========================================  ==============

⇒ **No longitudinal claim is admissible until the arm is shown to BEAT hold-v0,
separated, on the LONGITUDINAL family.** Not beating it is *not a small miss*: it
means the longitudinal head has learned nothing beyond its own input.

⚠️ **AND ADMITTING ``v0`` DID NOT CLOSE A DEFICIT.** MEASURED: even vision **+
v0** sat at σ/ADE **3.527** on the REF-C surface — still worse than a
**0-parameter constant-yaw-rate rule at 1.1888**. The ruling removed an
*argument*, not a *gap*.

THE THREE CONTROLS, AND WHAT EACH CAN AND CANNOT SEE
-----------------------------------------------------
1. :func:`holdv0_baseline` — **the baseline**. Scores the identical longitudinal
   metrics for the arm and for a plan that sustains ``v0`` with zero commanded
   acceleration, on the SAME windows, and reports the difference under a
   **paired episode-cluster bootstrap** (``taniteval.ci``). This is the one the
   ruling makes binding, and it is the only one that yields an *admissibility*
   verdict.
2. :func:`copy_detector` — **the scalar**. A direct, per-window correlation of
   the emitted longitudinal plan against the constant-``v0`` trajectory, reported
   jointly with the commanded acceleration, so a near-1 correlation with a
   near-zero command cannot be buried in a table. Runs from a banked dump, needs
   no model, costs milliseconds.
3. :func:`shuffle_control` — **the falsifier**. Permute ``v0`` across windows and
   re-run: a planner genuinely *using* speed degrades sharply, one merely
   *echoing* it tracks the shuffled value. ⛔ This one **needs the model**, so
   from a banked dump it reports UNAVAILABLE *with its reason and its n* and
   names the command that produces the input — never a silent pass.

⛔ WHERE THE SHUFFLE MACHINERY ALREADY LIVES — REUSED, NOT REINVENTED
---------------------------------------------------------------------
The model-side permutation that produced the P1 row's **R² 0.995 → −0.72** is
``stack/scripts/probe_latent_state.py``:

* ``collect_grid(..., speed_echo_control=True)`` — **line 578**, permutes
  ``b["pose_last"][:, 3]`` (the ``v0`` that ``lift_actions3`` appends as the
  predictor's 3rd action channel) across the batch: *same frames, same recorded
  actions, WRONG speed scalar*;
* CLI ``--speed-echo-control`` — **line 775**;
* already wired into the standing probe by ``stack/ops/pbattery_watcher.py:80``.

:func:`shuffle_v0` here is the **scorer-side canonical form of that same
permutation** — seeded, so a control is reproducible — and exists so a caller
re-running a planner uses one construction rather than a second one that drifts.
It does **not** replace the pod-side path; :data:`SHUFFLE_PRODUCER` points at it.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
* It does **not** re-derive ``v0`` from ``gt`` or ``pred``. The first step of the
  GT path is a *future* quantity; imputing ``v0`` from it would make the control
  unfalsifiable — the exact defect class the ruling exists to prevent. No ``v0``
  ⇒ UNAVAILABLE with a reason.
* It does **not** re-implement the geometry. Speeds, accelerations and along-track
  displacements come from :func:`taniteval.four_families._seq_geometry`, the same
  function the LONGITUDINAL family itself uses, so the control and the number it
  guards can never drift apart.
* It does **not** pool the controls into a score. Per the binding four-families
  rule they are reported per-family, each with its estimator, its CI and its n.

Only numpy + torch + :mod:`taniteval.ci`. CPU, milliseconds, no pod.
"""
from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "VERSION",
    "BLOCK",
    "SHUFFLE_PRODUCER",
    "hold_v0_path",
    "shuffle_v0",
    "resolve_v0",
    "copy_detector",
    "holdv0_baseline",
    "shuffle_control",
    "anti_echo",
]

BLOCK = "taniteval.v0_antiecho"
VERSION = "1.0.0"

#: The PI ruling this module is the executable form of. Stamped into every block
#: so a banked artifact carries the rule it was produced under.
RULING = ("Sayed 2026-08-16: v0 is ADMISSIBLE as a planner input (measured "
          "present state, not the future) PROVIDED the planner is shown not to "
          "be 'just outputting v0 as longitudinal plan'. See "
          "Project Steering/V6F_PLANNER_DESIGN.md §1.4.")

#: ⛔ The model-side shuffle is NOT implemented here. It already exists; this is
#: where it lives, so nobody builds a second one.
SHUFFLE_PRODUCER = (
    "stack/scripts/probe_latent_state.py — collect_grid(..., "
    "speed_echo_control=True) at line 578 permutes b['pose_last'][:, 3] across "
    "the batch (same frames, same recorded actions, WRONG speed scalar); CLI "
    "flag --speed-echo-control at line 775; already wired by "
    "stack/ops/pbattery_watcher.py:80. This is the machinery that produced the "
    "P1 speed-echo row (R2 0.995 -> -0.72). Re-run the planner under it and pass "
    "the resulting waypoints as win['pred_v0shuffled'] (+ win['v0_shuffled']).")

# --------------------------------------------------------------------------- #
# Thresholds. Every one is PROPOSED and every one is applied to the ARM AND TO  #
# THE GT ALIKE, so the headline is a DIFFERENCE and no threshold decides it     #
# alone. An unmarked threshold in this file is a bug.                          #
# --------------------------------------------------------------------------- #
#: PROPOSED — |Pearson r| between the emitted along-track profile and hold-v0's
#: above which the two are "the same plan" for flagging purposes. Deliberately
#: severe: a monotone ramp correlates highly with almost any other monotone ramp,
#: which is precisely why ``r`` MUST NOT be reported alone (see :func:`copy_detector`).
ECHO_R_MIN = 0.999
#: PROPOSED — mean |commanded acceleration| (m/s^2) below which a plan is
#: "commanding nothing". 0.1 m/s^2 over a 2 s horizon is ~0.2 m/s of speed change.
ECHO_ACCEL_MPS2 = 0.10
#: PROPOSED — RMS speed departure from v0 (m/s) below which a window's plan is
#: "flat". Same scale as ECHO_ACCEL_MPS2 x horizon.
ECHO_DEV_MPS = 0.10
#: PROPOSED — RMS speed departure from v0 (m/s) above which the HUMAN demonstrably
#: did something other than hold. Used only to define the discriminating subset
#: ``echo_frac_where_human_acted``.
HUMAN_ACTED_DEV_MPS = 0.50
#: PROPOSED — echo_index at or above which the verdict is ECHO outright: on
#: essentially every window the plan is hold-v0.
ECHO_INDEX_ECHO = 0.99
#: PROPOSED — excess of the arm's echo_index over the GT's above which the
#: verdict is SUSPECT: the arm is markedly flatter than the human it is copying.
ECHO_EXCESS_SUSPECT = 0.20
#: ⛔ PROPOSED — RMS speed departure (m/s) below which ``dev_r`` is DEGENERATE
#: rather than small. This constant exists because the can-it-fire proof caught
#: a real instrument defect, and it is worth stating rather than burying:
#:
#: MEASURED 2026-08-16 on 240 synthetic windows — the round trip
#: ``hold_v0_path(v0) -> _seq_geometry -> speed`` is **NOT bit-exact in
#: float32**. It leaves a residue of **1.9073e-06 m/s max, 5.6974e-07 m/s RMS**
#: (and 7.6294e-06 m/s^2 on the accel), because the geometry recovers speed from
#: ``norm(diff(positions))/dt`` rather than reading it back.
#:
#: ⇒ An exact ``== 0`` degeneracy test therefore **CORRELATED FLOAT NOISE** and
#: reported ``dev_r = -0.0133`` for a **PURE ECHO** — a number, where the honest
#: answer is "undefined". A reader would have taken that as a genuine-but-bad
#: planner (weak negative correlation with the human) instead of the copy it is.
#: The gate is physical, not numeric: 1e-3 m/s is ~1750x the measured residue and
#: ~100x below ECHO_DEV_MPS, so it can only ever catch a departure that is nil.
DEV_R_MIN_RMS_MPS = 1e-3

_EPS = 1e-8

#: The longitudinal per-window metrics the hold-v0 baseline is run on. ALL are
#: "lower is better", so a NEGATIVE paired delta (arm - hold) is an arm win.
#: `speed_mae_mps` is the PRIMARY: it is the literal "target-speed accuracy"
#: metric the binding four-families rule names, and it is the one an echo
#: cannot win by construction.
PRIMARY_METRIC = "speed_mae_mps"
BASELINE_METRICS = ("speed_mae_mps", "along_abs_m", "target_speed_miss_1mps")
BASELINE_METRIC_DESC = {
    "speed_mae_mps": "mean |speed error| over the horizon, m/s (PRIMARY — the "
                     "binding rule's target-speed accuracy)",
    "along_abs_m": "|along-track position error| at the final horizon step, m — "
                   "the POSITIONAL consequence of the speed plan; dt-invariant",
    "target_speed_miss_1mps": "fraction of horizon steps whose |speed error| "
                              "exceeds 1.0 m/s (1 - target_speed_acc.within_1_mps)",
}


# --------------------------------------------------------------------------- #
# The baseline plan itself                                                     #
# --------------------------------------------------------------------------- #
def hold_v0_path(v0, n: int, dt: float) -> torch.Tensor:
    """HOLD-v0 — go straight at the observed entry speed, zero commanded accel.

    ``v0`` [N] m/s, ``n`` horizon steps, ``dt`` seconds between steps ->
    ``[N, n, 2]`` ego-frame metres, axis 0 = along-track (forward), axis 1 =
    lateral (left), exactly the convention ``pred``/``gt`` use.

    ⛔ **This is the generalised form of** :func:`taniteval.driving.hold_v0`,
    which is hard-coded to the tier-0 surface (``n=4``, ``DT_WP=0.5``). It is
    pinned **bit-identical** to that function at ``(4, 0.5)`` by
    ``tests/test_v0_antiecho.py::test_hold_v0_path_is_bit_identical_to_the_driving_floor``
    — the programme already has one hold-v0 and must not grow a second that
    drifts from it.

    ⚠️ It is a **STRAIGHT** line. That is deliberate and it is a limitation worth
    stating: hold-v0 is the LONGITUDINAL floor and nothing else. It cannot turn,
    so it must never be used to judge a lateral claim (the defect that made every
    pre-2026-08-02 turn verdict a comparison against two predictors structurally
    unable to turn — ``driving.FLOORS``' CTRV note).
    """
    v = torch.as_tensor(v0, dtype=torch.float32).reshape(-1)
    if n < 1:
        raise ValueError(f"hold_v0_path needs n >= 1, got {n}")
    if not (dt > 0):
        raise ValueError(f"hold_v0_path needs dt > 0, got {dt}")
    t = torch.arange(1, n + 1, dtype=torch.float32) * float(dt)
    return torch.stack([v[:, None] * t[None, :],
                        torch.zeros(len(v), n)], -1)


def shuffle_v0(v0, seed: int = 0) -> tuple[torch.Tensor, np.ndarray]:
    """Permute ``v0`` ACROSS WINDOWS -> ``(v0_shuffled [N], permutation [N])``.

    The scorer-side canonical form of the permutation
    ``probe_latent_state.collect_grid`` applies pod-side (see
    :data:`SHUFFLE_PRODUCER`). Same construction — a uniform random permutation
    of the ``v0`` scalar with the frames and the recorded actions untouched — but
    **seeded**, because a control whose draw cannot be reproduced cannot be
    re-checked.

    ⚠️ ``probe_latent_state`` permutes **within each batch** (``torch.randperm``
    on the batch dim), which is a coarser shuffle than this whole-array one: it
    can leave a window paired with a ``v0`` from a neighbouring window of the
    same episode. Both are valid controls; the whole-array form is strictly
    stronger and is what this module scores against. Stated rather than blurred,
    because the two do not produce identical numbers.

    ⛔ A permutation can map a window to ITSELF. On a real 881-window grid the
    expected number of fixed points is exactly 1 regardless of N, which is
    negligible — but it is reported (``n_fixed_points``) rather than silently
    resampled, so a degenerate tiny-N control is visible instead of hidden.
    """
    v = torch.as_tensor(v0, dtype=torch.float32).reshape(-1)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(v.shape[0])
    return v[torch.as_tensor(perm, dtype=torch.long)], perm


# --------------------------------------------------------------------------- #
# v0 resolution — from the window dict, NEVER from the future                  #
# --------------------------------------------------------------------------- #
#: Where ``v0`` may legitimately come from, in priority order. Each entry is
#: ``(key path, provenance string)``.
V0_SOURCES = (
    ("v0", "win['v0'] — supplied explicitly by the caller"),
    ("speed", "win['speed'] — rollout.collect's ego speed at t0 "
              "(ep.poses[last, 3]), the same scalar the --speed-input arms feed "
              "the predictor"),
    ("lead.speeds", "win['lead']['speeds'] — the lead block's time-gap "
                    "denominator, which IS the ego speed at t0"),
)


def resolve_v0(win: dict, n: int):
    """``(v0 [n] float32, provenance str)`` or ``(None, reason str)``.

    ⛔ **NEVER derived from ``gt`` or ``pred``.** The GT path's first step is a
    FUTURE displacement; imputing ``v0`` from it would make every control in this
    module compare the arm against a quantity the arm's own error moves, i.e. it
    would be unfalsifiable by construction. That is the defect class the ruling
    exists to prevent, so the honest branch is a refusal with a reason.
    """
    for key, prov in V0_SOURCES:
        if key == "lead.speeds":
            lead = win.get("lead")
            v = lead.get("speeds") if isinstance(lead, dict) else None
        else:
            v = win.get(key)
        if v is None:
            continue
        t = torch.as_tensor(np.asarray(v, dtype=np.float32)).reshape(-1).float()
        if t.numel() != n:
            continue
        if not bool(torch.isfinite(t).all()):
            continue
        return t, prov
    return None, (
        "no ego speed at t0 in the window dict. Tried, in order: "
        + "; ".join(k for k, _ in V0_SOURCES)
        + f" (each must be a finite [n={n}] array). ⛔ v0 is NOT imputed from "
        "gt/pred — the first GT step is a FUTURE displacement and deriving v0 "
        "from it would make the anti-echo controls unfalsifiable. Supply "
        "win['v0'] (rollout.collect already publishes win['speed']). Not "
        "supplying it is a WORK ITEM, not a pass.")


# --------------------------------------------------------------------------- #
# geometry — borrowed from four_families, never re-implemented                 #
# --------------------------------------------------------------------------- #
def _geom(wp: torch.Tensor, dt: float):
    """``four_families._seq_geometry``, imported at call time.

    Lazy so :mod:`taniteval.four_families` can import THIS module from inside
    :func:`~taniteval.four_families.longitudinal` without a cycle — the same
    idiom ``_ego_progress``/``_distance_keeping`` already use there.
    """
    from taniteval.four_families import _seq_geometry
    return _seq_geometry(wp, dt)


def _rowwise_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row Pearson r for ``[N, H]`` pairs. NaN where a row is constant.

    A constant row has zero variance and NO correlation — returning 0.0 or 1.0
    there would invent a verdict. NaN propagates into ``nanmean`` and into the
    reported ``n``, so the degenerate rows are counted rather than absorbed.
    """
    ac = a - a.mean(dim=1, keepdim=True)
    bc = b - b.mean(dim=1, keepdim=True)
    num = (ac * bc).sum(dim=1)
    den = ac.pow(2).sum(dim=1).sqrt() * bc.pow(2).sum(dim=1).sqrt()
    out = torch.where(den > _EPS, num / den.clamp_min(_EPS),
                      torch.full_like(num, float("nan")))
    return out


def _nan(x) -> float | None:
    f = float(x)
    return None if math.isnan(f) else round(f, 4)


# --------------------------------------------------------------------------- #
# CONTROL 2 — the direct copy-detector                                         #
# --------------------------------------------------------------------------- #
def copy_detector(pred: torch.Tensor, gt: torch.Tensor, v0: torch.Tensor,
                  dt: float) -> dict:
    """Is the emitted longitudinal plan a COPY of ``v0``? -> one headline scalar.

    ⭐ **The headline is** ``echo_index``: the fraction of windows whose emitted
    along-track profile is simultaneously

    * correlated with the constant-``v0`` trajectory at ``r >= ECHO_R_MIN``, and
    * commanding ``|accel| <= ECHO_ACCEL_MPS2``

    — i.e. the two conditions the ruling names, **jointly**, never separately. A
    pure hold-v0 planner scores exactly **1.0000**.

    ⛔ **Why ``r`` ALONE IS NOT THE DETECTOR, and reporting it alone would be an
    instrument defect.** Both the arm's along-track profile and hold-v0's are
    monotone ramps from the origin, and any two monotone ramps correlate near 1.
    A planner that brakes hard still scores ``r`` ~0.99. The commanded-accel
    conjunct is what makes the statistic discriminating; ``r_vs_holdv0`` is
    published beside it as a *component*, explicitly labelled as necessary and
    not sufficient.

    ⭐ **Why ``echo_index_gt`` IS PUBLISHED, and why the headline is the EXCESS.**
    On this corpus most windows genuinely are near-constant-speed cruising, so
    the *human* also scores a high ``echo_index``. Quoting the arm's number
    without the GT's would charge the arm for the corpus. ``echo_index_excess =
    echo_index - echo_index_gt`` is the honest read, and it is
    threshold-ROBUST: the same PROPOSED thresholds are applied to both paths, so
    a shift in either moves both terms together.

    ⭐ ``dev_r`` is the scalar a pure echo **cannot** fake. It correlates the
    arm's DEPARTURE from hold-v0 (``speed_arm - v0``) against the human's
    departure (``speed_gt - v0``), pooled over every (window, step). A genuine
    longitudinal head that brakes when the human brakes scores > 0; a pure echo
    has an identically-zero departure, so the correlation is UNDEFINED — reported
    as ``None`` with ``dev_r_degenerate: true``, which IS the echo signature, not
    a missing number.
    """
    P, G = _geom(pred, dt), _geom(gt, dt)
    n, H = int(pred.shape[0]), int(pred.shape[1])
    v = torch.as_tensor(v0, dtype=torch.float32).reshape(-1)
    hold = hold_v0_path(v, H, dt)
    A_hold = hold[..., 0]                       # [N,H] along-track of hold-v0
    v_row = v[:, None].expand(-1, H)            # [N,H] the constant-v0 profile

    def _one(Q, name):
        r = _rowwise_pearson(Q["along"], A_hold)
        a_cmd = Q["accel"].abs().mean(dim=1) if H > 1 else torch.zeros(n)
        dev = Q["speed"] - v_row
        dev_rms = dev.pow(2).mean(dim=1).sqrt()
        flat = (a_cmd <= ECHO_ACCEL_MPS2)
        corr = torch.nan_to_num(r, nan=0.0) >= ECHO_R_MIN
        idx = float((corr & flat).float().mean())
        return {
            "echo_index": round(idx, 4),
            "r_vs_holdv0_mean": _nan(torch.nanmean(r)),
            "cmd_accel_mae_mps2": round(float(a_cmd.mean()), 4),
            "speed_dev_rms_mps": round(float(dev_rms.mean()), 4),
            "n_rows_r_undefined": int(torch.isnan(r).sum()),
            "_dev": dev, "_dev_rms": dev_rms, "_flat": flat, "_r": r,
            "_name": name,
        }

    arm, ref = _one(P, "arm"), _one(G, "gt")
    excess = arm["echo_index"] - ref["echo_index"]

    # --- the departure correlation: the number a pure echo cannot fake ------ #
    da, dg = arm["_dev"].reshape(-1), ref["_dev"].reshape(-1)
    dev_r = float(_rowwise_pearson(da[None, :], dg[None, :])[0])
    # ⛔ The degeneracy gate is PHYSICAL, not numeric. `hold_v0_path ->
    # _seq_geometry` is not bit-exact in float32 (MEASURED residue 1.9e-06 m/s),
    # so a `== 0` test would correlate FLOAT NOISE and hand a PURE ECHO a real
    # number — measured, it produced dev_r = -0.0133. See DEV_R_MIN_RMS_MPS.
    rms_arm = float(da.pow(2).mean().sqrt())
    rms_gt = float(dg.pow(2).mean().sqrt())
    dev_degenerate = bool(math.isnan(dev_r)
                          or rms_arm <= DEV_R_MIN_RMS_MPS
                          or rms_gt <= DEV_R_MIN_RMS_MPS)
    dev_ratio = (float(arm["_dev_rms"].mean()) /
                 max(float(ref["_dev_rms"].mean()), _EPS))

    # --- the discriminating subset: flat arm where the human demonstrably acted #
    human_acted = ref["_dev_rms"] >= HUMAN_ACTED_DEV_MPS
    n_acted = int(human_acted.sum())
    arm_flat = arm["_dev_rms"] <= ECHO_DEV_MPS
    frac_acted = (float((arm_flat & human_acted).float().sum()) / n_acted
                  if n_acted else float("nan"))

    if arm["echo_index"] >= ECHO_INDEX_ECHO:
        verdict, why = "ECHO", (
            f"echo_index {arm['echo_index']:.4f} >= {ECHO_INDEX_ECHO}: on "
            f"essentially every window the emitted longitudinal plan is "
            f"hold-v0 (r >= {ECHO_R_MIN} AND |a_cmd| <= {ECHO_ACCEL_MPS2} "
            f"m/s^2). ⛔ This is the copy the PI's 2026-08-16 condition names. "
            f"No longitudinal claim may be made from this arm.")
    elif excess >= ECHO_EXCESS_SUSPECT:
        verdict, why = "SUSPECT", (
            f"echo_index_excess {excess:+.4f} >= {ECHO_EXCESS_SUSPECT}: the arm "
            f"holds v0 on {excess * 100:.1f} pp more windows than the human "
            f"does. Not a proof of copying, but every longitudinal number from "
            f"this arm must be read beside the hold-v0 baseline below.")
    else:
        verdict, why = "CLEAN", (
            f"echo_index_excess {excess:+.4f} < {ECHO_EXCESS_SUSPECT}: the arm "
            f"is not measurably flatter than the human on these windows. ⚠️ "
            f"CLEAN here is NOT an admissibility verdict — that is "
            f"holdv0_baseline's job. This detector can only refute a copy, "
            f"never establish skill.")

    return {
        "status": "OK",
        "verdict": verdict,
        "verdict_reason": why,
        # ⭐ THE HEADLINE, and the two terms it is a difference of.
        "echo_index": arm["echo_index"],
        "echo_index_gt": ref["echo_index"],
        "echo_index_excess": round(excess, 4),
        "echo_index_def": (
            f"fraction of windows with r(along, hold-v0) >= {ECHO_R_MIN} AND "
            f"mean |commanded accel| <= {ECHO_ACCEL_MPS2} m/s^2, JOINTLY. A pure "
            f"hold-v0 planner scores exactly 1.0. Reported beside the GT's own "
            f"value because most of this corpus genuinely is cruising; the "
            f"EXCESS is the read."),
        # --- components. r is necessary, NOT sufficient — labelled as such. --- #
        "r_vs_holdv0_mean": arm["r_vs_holdv0_mean"],
        "r_vs_holdv0_mean_gt": ref["r_vs_holdv0_mean"],
        "r_alone_is_not_the_detector": (
            "⛔ both profiles are monotone ramps from the origin, so ANY "
            "forward-moving plan correlates ~1 with hold-v0 — a hard-braking "
            "planner still scores r ~0.99. r is a COMPONENT of echo_index, "
            "never a verdict on its own."),
        "cmd_accel_mae_mps2": arm["cmd_accel_mae_mps2"],
        "cmd_accel_mae_mps2_gt": ref["cmd_accel_mae_mps2"],
        "speed_dev_rms_mps": arm["speed_dev_rms_mps"],
        "speed_dev_rms_mps_gt": ref["speed_dev_rms_mps"],
        "dev_ratio": round(dev_ratio, 4),
        "dev_ratio_def": ("RMS(speed_arm - v0) / RMS(speed_gt - v0). 0.0 = the "
                          "arm commands NOTHING while the human did; 1.0 = it "
                          "commands the same magnitude (says nothing about the "
                          "sign — that is dev_r)."),
        # ⭐ the one a pure echo cannot fake
        "dev_r": None if dev_degenerate else round(dev_r, 4),
        "dev_r_degenerate": bool(dev_degenerate),
        "dev_r_rms_arm_mps": round(rms_arm, 6),
        "dev_r_rms_gt_mps": round(rms_gt, 6),
        "dev_r_def": ("Pearson r between (speed_arm - v0) and (speed_gt - v0) "
                      "pooled over every (window, step) — the correlation of "
                      "the DEPARTURE from the echo. A pure hold-v0 planner has "
                      "a nil departure, so this is UNDEFINED "
                      "(dev_r_degenerate=true), which IS the signature. ⛔ The "
                      f"degeneracy gate is PHYSICAL (RMS <= "
                      f"{DEV_R_MIN_RMS_MPS} m/s), not `== 0`: the geometry "
                      f"recovers speed from norm(diff(positions))/dt and leaves "
                      f"a ~1.9e-06 m/s float32 residue, which an exact test "
                      f"correlated into a spurious dev_r = -0.0133 for a PURE "
                      f"ECHO."),
        "echo_frac_where_human_acted": _nan(torch.tensor(frac_acted)),
        "echo_frac_where_human_acted_def": (
            f"of the {n_acted} windows where the human's own speed departed "
            f"from v0 by >= {HUMAN_ACTED_DEV_MPS} m/s RMS, the fraction on "
            f"which the arm nevertheless stayed within {ECHO_DEV_MPS} m/s of "
            f"v0. The discriminating subset: holding was demonstrably the wrong "
            f"answer and the arm held anyway."),
        "n_windows_human_acted": n_acted,
        "n_windows": n,
        "horizon_steps": H,
        "dt_s": dt,
        "thresholds": {
            "ECHO_R_MIN": ECHO_R_MIN, "ECHO_ACCEL_MPS2": ECHO_ACCEL_MPS2,
            "ECHO_DEV_MPS": ECHO_DEV_MPS,
            "HUMAN_ACTED_DEV_MPS": HUMAN_ACTED_DEV_MPS,
            "ECHO_INDEX_ECHO": ECHO_INDEX_ECHO,
            "ECHO_EXCESS_SUSPECT": ECHO_EXCESS_SUSPECT,
            "_class": "ALL PROPOSED, and ALL applied to the arm AND the GT "
                      "alike so the headline (the excess) is threshold-robust.",
        },
    }


# --------------------------------------------------------------------------- #
# CONTROL 1 — the hold-v0 baseline (the binding one)                           #
# --------------------------------------------------------------------------- #
def _long_components(pred: torch.Tensor, gt: torch.Tensor, dt: float) -> dict:
    """Per-window LONGITUDINAL error components, all lower-is-better."""
    P, G = _geom(pred, dt), _geom(gt, dt)
    sp = P["speed"] - G["speed"]
    al = P["along"] - G["along"]
    return {
        "speed_mae_mps": sp.abs().mean(dim=1).numpy().astype(np.float64),
        "along_abs_m": al[:, -1].abs().numpy().astype(np.float64),
        "target_speed_miss_1mps":
            (sp.abs() > 1.0).float().mean(dim=1).numpy().astype(np.float64),
    }


def holdv0_baseline(pred: torch.Tensor, gt: torch.Tensor, v0: torch.Tensor,
                    dt: float, eid=None, n_boot: int = 2000,
                    seed: int = 0) -> dict:
    """⭐ THE BINDING CONTROL. The arm's longitudinal metrics BESIDE hold-v0's.

    Every metric in :data:`BASELINE_METRICS` is computed twice on the SAME
    windows — once for the arm, once for a plan that sustains ``v0`` with zero
    commanded acceleration — and the difference is put through
    :func:`taniteval.ci.paired_episode_cluster_bootstrap`, the programme's only
    decision-grade estimator.

    ⛔ **The verdict is an ADMISSIBILITY verdict, not a score.** Per the ruling:
    an arm that does not beat hold-v0, *separated*, has not made a small miss —
    **its longitudinal head has learned nothing beyond its own input**, and no
    longitudinal claim may be made from it. ``admissible`` is that bit.

    ⛔ Without ``eid`` there is no episode-cluster bootstrap and therefore no
    decision-grade interval. The point deltas are still reported — they are real
    — but ``estimator`` says UNAVAILABLE with its reason and ``admissible`` is
    False, because an unseparated delta cannot discharge a separation
    requirement. A quadrature combination of two single-arm intervals is NOT a
    substitute and is not offered.
    """
    from taniteval import ci as _ci
    H = int(pred.shape[1])
    v = torch.as_tensor(v0, dtype=torch.float32).reshape(-1)
    hold = hold_v0_path(v, H, dt)
    arm_c = _long_components(pred, gt, dt)
    hold_c = _long_components(hold, gt, dt)

    have_eid = eid is not None and len(list(eid)) == int(pred.shape[0])
    metrics: dict[str, dict] = {}
    for m in BASELINE_METRICS:
        a, b = arm_c[m], hold_c[m]
        row = {
            "arm": round(float(np.nanmean(a)), 4),
            "holdv0": round(float(np.nanmean(b)), 4),
            "what": BASELINE_METRIC_DESC[m],
            "lower_is_better": True,
        }
        if have_eid:
            pb = _ci.paired_episode_cluster_bootstrap(
                a, b, list(eid), n_boot=n_boot, seed=seed)
            row.update(pb)
            row["delta_sign_note"] = ("delta = arm - holdv0; NEGATIVE and "
                                      "separated is an arm WIN")
            if pb["separated"] and pb["delta"] < 0:
                row["verdict"] = "BEATS_HOLDV0"
            elif pb["separated"]:
                row["verdict"] = "LOSES_TO_HOLDV0"
            else:
                row["verdict"] = "NOT_SEPARATED"
        else:
            row["delta"] = round(float(np.nanmean(a) - np.nanmean(b)), 4)
            row["separated"] = False
            row["estimator"] = "UNAVAILABLE"
            row["verdict"] = "NO_INTERVAL"
            row["reason"] = (
                "no per-window episode id, so the episode-cluster bootstrap "
                "cannot be formed. The point delta is real; the SEPARATION the "
                "ruling requires is not established. Pass win['eid']. ⛔ A "
                "combination of two single-arm intervals in quadrature is NOT a "
                "substitute — the two arms are scored on the same windows and "
                "are not independent.")
        metrics[m] = row

    primary = metrics[PRIMARY_METRIC]
    admissible = primary.get("verdict") == "BEATS_HOLDV0"
    verdict = primary.get("verdict", "NO_INTERVAL")
    if verdict == "BEATS_HOLDV0":
        why = (f"the arm beats hold-v0 on {PRIMARY_METRIC} by "
               f"{-primary['delta']:.4f} m/s [{-primary['hi']:.4f}, "
               f"{-primary['lo']:.4f}], separated, paired episode-cluster "
               f"bootstrap over {primary['n_episodes']} episodes. The "
               f"longitudinal head carries something its v0 input does not.")
    elif verdict == "LOSES_TO_HOLDV0":
        why = (f"⛔ the arm is WORSE than hold-v0 on {PRIMARY_METRIC} by "
               f"{primary['delta']:.4f} m/s [{primary['lo']:.4f}, "
               f"{primary['hi']:.4f}], separated AGAINST it. A 0-parameter "
               f"copy of the arm's own input outperforms the arm. No "
               f"longitudinal claim is admissible.")
    elif verdict == "NOT_SEPARATED":
        why = (f"⛔ the arm is NOT separated from hold-v0 on {PRIMARY_METRIC} "
               f"(delta {primary['delta']:+.4f} m/s [{primary['lo']:.4f}, "
               f"{primary['hi']:.4f}]). ⛔ THIS IS NOT A SMALL MISS: it means "
               f"the longitudinal head has learned nothing beyond its own v0 "
               f"input. No longitudinal claim is admissible.")
    else:
        why = ("no episode ids were supplied, so no separation was established. "
               "The ruling's condition is UNDISCHARGED, which is not the same "
               "as failed — see the per-metric `reason`.")

    return {
        "status": "OK",
        "verdict": verdict,
        "verdict_reason": why,
        "admissible": bool(admissible),
        "admissible_def": (
            "⛔ Sayed 2026-08-16: no longitudinal claim is admissible until the "
            "planner BEATS hold-v0, SEPARATED, on the LONGITUDINAL family. This "
            "bit is that condition, evaluated on " + PRIMARY_METRIC + "."),
        "primary_metric": PRIMARY_METRIC,
        "metrics": metrics,
        "estimator": ("paired_episode_cluster_bootstrap (taniteval.ci) — ⛔ NOT "
                      "overlapping_holdout_se, which is anti-conservative AND "
                      "biases the point estimate"),
        "n_boot": int(n_boot), "seed": int(seed),
        "n_windows": int(pred.shape[0]),
        "n_episodes": (int(len(set(map(str, eid)))) if have_eid else 0),
        "dt_s": dt, "horizon_steps": H,
        "baseline": ("hold-v0: along = v0 * t, lateral = 0, zero commanded "
                     "acceleration. 0 parameters, 0 vision, 0 training."),
        "⚠️_scope": ("hold-v0 is a STRAIGHT line and is the LONGITUDINAL floor "
                     "ONLY. Beating it says nothing about lateral or tactical "
                     "quality, and it must never be used to judge a turn — for "
                     "that the floor family is driving.FLOORS (cv/holdv0/ctrv)."),
    }


# --------------------------------------------------------------------------- #
# CONTROL 3 — the v0 shuffle                                                   #
# --------------------------------------------------------------------------- #
def shuffle_control(pred: torch.Tensor, gt: torch.Tensor, v0: torch.Tensor,
                    dt: float, pred_shuffled=None, v0_shuffled=None,
                    eid=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Permute ``v0`` across windows and re-run: does the plan FOLLOW the lie?

    ⛔ **This control needs the MODEL.** A banked dump carries one set of
    predictions; the shuffled-``v0`` predictions are a second forward pass. When
    they are absent the block reports UNAVAILABLE **with its reason and its n**
    and names the producer (:data:`SHUFFLE_PRODUCER`) — clause 5 of the binding
    four-families rule, never a silent pass.

    Given ``pred_shuffled`` (the arm re-run with ``v0`` permuted) it reports two
    things, and the SECOND is the discriminating one:

    * ``degradation`` — how much the longitudinal error grows under the lie. A
      planner genuinely using speed degrades sharply. ⚠️ Large degradation alone
      does **not** prove skill: an echo degrades too, because it is now copying
      the wrong number.
    * ⭐ ``tracks_shuffled_v0`` — whether the shuffled-run plan follows the
      **shuffled** value rather than the true one:
      ``RMS(speed_shuf - v0_shuffled) / RMS(speed_shuf - v0_true)``. A pure echo
      drives this toward **0** (it tracks the lie exactly); a planner that reads
      the scene and merely conditions on speed keeps it near or above 1. This is
      the P1-row shape (R² 0.995 -> -0.72) expressed as a ratio.

    ⚠️ **The top band is named NOT_AN_ECHO, deliberately not "USES_SPEED".** An
    arm that ignores ``v0`` entirely produces an identical plan under the shuffle
    and therefore lands in it — MEASURED on the synthetic pair: degradation
    exactly **0.0 [0.0, 0.0]**, ratio **3.725**. This control can only ever
    REFUTE a copy; whether the arm has longitudinal skill is
    :func:`holdv0_baseline`'s question.
    """
    n, H = int(pred.shape[0]), int(pred.shape[1])
    if pred_shuffled is None:
        return {
            "status": "UNAVAILABLE",
            "reason": ("no shuffled-v0 re-run supplied. This control needs a "
                       "SECOND forward pass of the model with v0 permuted "
                       "across windows — it cannot be computed from a banked "
                       "dump. " + SHUFFLE_PRODUCER + " Not running it is a WORK "
                       "ITEM, not a pass."),
            "n": 0,
            "n_windows_available": n,
            "producer": SHUFFLE_PRODUCER,
            "how_to_supply": ("win['pred_v0shuffled'] [n,H,2] (+ optionally "
                              "win['v0_shuffled'] [n]; when absent the "
                              "permutation is regenerated with "
                              "shuffle_v0(v0, seed) and the seed is stamped)"),
        }
    ps = torch.as_tensor(pred_shuffled, dtype=torch.float32)
    if ps.shape != pred.shape:
        return {"status": "UNAVAILABLE",
                "reason": (f"pred_v0shuffled has shape {tuple(ps.shape)} but "
                           f"pred is {tuple(pred.shape)} — refusing to score a "
                           f"control on different windows"),
                "n": 0}
    v = torch.as_tensor(v0, dtype=torch.float32).reshape(-1)
    if v0_shuffled is None:
        vs, perm = shuffle_v0(v, seed)
        perm_prov = (f"REGENERATED here with shuffle_v0(v0, seed={seed}). ⚠️ "
                     f"This is only the true permutation if the re-run used the "
                     f"same seeded helper; if the model was run under "
                     f"probe_latent_state --speed-echo-control (a per-BATCH "
                     f"randperm) the permutation differs and win['v0_shuffled'] "
                     f"MUST be supplied instead.")
    else:
        vs = torch.as_tensor(v0_shuffled, dtype=torch.float32).reshape(-1)
        perm = None
        perm_prov = "supplied by the caller alongside the re-run"
    if vs.numel() != v.numel():
        return {"status": "UNAVAILABLE",
                "reason": (f"v0_shuffled has {vs.numel()} entries, v0 has "
                           f"{v.numel()} — refusing"),
                "n": 0}

    from taniteval import ci as _ci
    true_c = _long_components(pred, gt, dt)[PRIMARY_METRIC]
    shuf_c = _long_components(ps, gt, dt)[PRIMARY_METRIC]
    have_eid = eid is not None and len(list(eid)) == n
    if have_eid:
        deg = _ci.paired_episode_cluster_bootstrap(shuf_c, true_c, list(eid),
                                                   n_boot=n_boot, seed=seed)
        deg["delta_sign_note"] = ("delta = shuffled - true on " + PRIMARY_METRIC
                                  + "; POSITIVE means the lie hurt")
    else:
        deg = {"delta": round(float(np.nanmean(shuf_c) - np.nanmean(true_c)), 4),
               "separated": False, "estimator": "UNAVAILABLE",
               "reason": "no per-window eid — point delta only"}

    # ⭐ the discriminating statistic: does the shuffled run FOLLOW the lie?
    S = _geom(ps, dt)
    d_lie = (S["speed"] - vs[:, None]).pow(2).mean().sqrt()
    d_true = (S["speed"] - v[:, None]).pow(2).mean().sqrt()
    ratio = float(d_lie) / max(float(d_true), _EPS)

    if ratio <= 0.25:
        verdict, why = "ECHO", (
            f"under the shuffle the plan tracks the FALSE v0 "
            f"{1 / max(ratio, _EPS):.1f}x more closely than the true one "
            f"(ratio {ratio:.4f}). The longitudinal plan is a copy of its own "
            f"speed input. ⛔ This is the P1 speed-echo shape (R2 0.995 -> "
            f"-0.72) and no longitudinal claim survives it.")
    elif ratio >= 0.90:
        verdict, why = "NOT_AN_ECHO", (
            f"under the shuffle the plan does NOT follow the false v0 (ratio "
            f"{ratio:.4f} >= 0.90) — its speed profile is anchored somewhere "
            f"other than its own v0 input. ⚠️ **NAMED FOR WHAT IT MEASURES.** "
            f"This verdict REFUTES a copy and establishes nothing else: an arm "
            f"that ignores v0 ENTIRELY scores it too (measured: an arm whose "
            f"output is unchanged by the shuffle has degradation exactly 0.0 "
            f"and ratio 3.725). Whether the arm has longitudinal SKILL is "
            f"holdv0_baseline's verdict, not this one.")
    else:
        verdict, why = "PARTIAL", (
            f"ratio {ratio:.4f} sits between the ECHO (<=0.25) and NOT_AN_ECHO "
            f"(>=0.90) bands: the plan partially follows the false v0. Report "
            f"the number, do not round it into either verdict.")

    return {
        "status": "OK",
        "verdict": verdict,
        "verdict_reason": why,
        "tracks_shuffled_v0": round(ratio, 4),
        "tracks_shuffled_v0_def": (
            "RMS(speed_shuffled_run - v0_shuffled) / RMS(speed_shuffled_run - "
            "v0_true). -> 0 means the plan followed the LIE exactly (a copy); "
            ">= 1 means it ignored it."),
        "rms_to_shuffled_v0_mps": round(float(d_lie), 4),
        "rms_to_true_v0_mps": round(float(d_true), 4),
        "degradation": deg,
        "degradation_note": (
            "⚠️ degradation alone does NOT prove the planner uses speed: an "
            "ECHO also degrades, because it is copying the wrong number. "
            "tracks_shuffled_v0 is what separates the two."),
        "permutation_provenance": perm_prov,
        "n_fixed_points": (int((np.asarray(perm) ==
                                np.arange(len(perm))).sum())
                           if perm is not None else None),
        "producer": SHUFFLE_PRODUCER,
        "n_windows": n, "horizon_steps": H, "dt_s": dt,
        "bands": {"ECHO": "<= 0.25", "PARTIAL": "0.25 - 0.90",
                  "NOT_AN_ECHO": ">= 0.90",
                  "_class": "PROPOSED — bands for reading the ratio, not gates",
                  "_naming": ("the top band is NOT_AN_ECHO, not 'USES_SPEED': "
                              "an arm that IGNORES v0 entirely lands there too, "
                              "so the label may only claim the refutation it "
                              "actually supports.")},
    }


# --------------------------------------------------------------------------- #
# The wired block — what four_families.longitudinal attaches automatically      #
# --------------------------------------------------------------------------- #
def anti_echo(pred: torch.Tensor, gt: torch.Tensor, dt: float,
              win: dict | None = None, n_boot: int = 2000,
              seed: int = 0) -> dict:
    """All three controls, or the reason none of them could run.

    Called automatically by :func:`taniteval.four_families.longitudinal` — the
    controls are **not** opt-in, because the ruling makes them a condition on
    every longitudinal number rather than a diagnostic to request.

    ``win`` supplies ``v0`` (see :func:`resolve_v0`), ``eid`` for the estimator,
    and optionally ``pred_v0shuffled``/``v0_shuffled`` for control 3.
    """
    win = win or {}
    n = int(pred.shape[0])
    v0, prov = resolve_v0(win, n)
    if v0 is None:
        return {
            "status": "UNAVAILABLE",
            "reason": prov,
            "n": 0,
            "n_windows_available": n,
            "ruling": RULING,
            "block": BLOCK, "version": VERSION,
            "⛔_consequence": (
                "the PI's 2026-08-16 anti-echo condition is UNDISCHARGED on "
                "this block. Every LONGITUDINAL number beside it must be read "
                "as UNVERIFIED against a hold-v0 copy — it may not be presented "
                "as a longitudinal capability result."),
        }
    eid = win.get("eid")
    base = holdv0_baseline(pred, gt, v0, dt, eid, n_boot=n_boot, seed=seed)
    det = copy_detector(pred, gt, v0, dt)
    shuf = shuffle_control(pred, gt, v0, dt,
                           pred_shuffled=win.get("pred_v0shuffled"),
                           v0_shuffled=win.get("v0_shuffled"),
                           eid=eid, n_boot=n_boot, seed=seed)
    flagged = (det["verdict"] == "ECHO" or shuf.get("verdict") == "ECHO"
               or not base["admissible"])
    return {
        "status": "OK",
        "block": BLOCK, "version": VERSION,
        "ruling": RULING,
        "v0_provenance": prov,
        # ⭐ ONE bit a report cannot lose: may a longitudinal claim be made?
        "longitudinal_claim_admissible": bool(base["admissible"]
                                              and det["verdict"] != "ECHO"
                                              and shuf.get("verdict") != "ECHO"),
        "flagged": bool(flagged),
        "summary": (
            f"holdv0={base['verdict']} · copy_detector={det['verdict']} "
            f"(echo_index {det['echo_index']:.4f} vs GT "
            f"{det['echo_index_gt']:.4f}) · shuffle={shuf.get('verdict', shuf['status'])}"),
        "holdv0_baseline": base,
        "copy_detector": det,
        "shuffle_control": shuf,
        "_why_all_three": (
            "holdv0_baseline is the ADMISSIBILITY test (does the arm beat the "
            "copy, separated). copy_detector is the cheap always-on SCALAR "
            "(is the emitted plan literally the copy). shuffle_control is the "
            "FALSIFIER (does the plan follow a v0 we know to be wrong). The "
            "first can pass while the arm still copies on the subset that "
            "matters, the second cannot establish skill, and the third needs "
            "the model — so none of the three substitutes for another."),
        "⚠️_context": (
            "admitting v0 removed an ARGUMENT, not a DEFICIT: MEASURED, vision "
            "+ v0 sat at sigma/ADE 3.527 on the REF-C surface, still worse than "
            "a 0-parameter constant-yaw-rate rule at 1.1888."),
    }
