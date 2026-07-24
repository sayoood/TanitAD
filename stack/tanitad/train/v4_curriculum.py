"""Flagship v4 joint-training curriculum primitives (P4 of V4_FLAGSHIP_DESIGN §15).

The pure, GPU-free pieces of the joint WM + planner training loop — kept in one
importable module so the correctness-critical schedule is unit-tested rather than
buried in the trainer script. Four things live here:

* :func:`lambda_plan_at` — the λ_plan GRADIENT scale (O-20), scheduled across the
  three phases. λ_plan=0 in Phase A (LP: heads train, trunk unmoved by the
  planner), a linear ramp 0→1 in Phase B (the only phase where the WM/planner
  trade is negotiated), 1 in Phase C (Sayed's "at the same time", full).
* :func:`produced_goal_frac` — scheduled sampling on the goal AND on S3, sharing
  the SAME phase boundaries as λ_plan. This is the O-17 fix: the ramps used to end
  at step 20,000 on a clock unrelated to λ_plan's, which put the G1 gate (step
  10,000, with a produced-goal primary and a KILL secondary) inside the ramp — a
  near-certain restart for a schedule reason, from a family capped at 2. With the
  boundaries aligned, G1 sees the arm in exactly the configuration it is scored on.
* :func:`plan_smoothness_loss` — jerk + curvature-rate on the DENSE 20-step
  emitted plan (§7). v1's ``--jerk-weight`` acted on a 4-point, non-scored head and
  contributed ≤1e-4 of a ~4.0 loss ([PM] #7); this is "the other mechanism" [PM]
  asks for — it acts on the path v4 actually ships.
* :func:`factorised_ce` — the LAT×LON×DIST cross-entropies (§4.3/§6.2), each with
  its ``unknown``/``d_unknown`` sentinel masked out of the CE exactly like
  ``ROUTE_UNKNOWN`` (§6.5: a logit no label can train is a dead parameter).
* :class:`CanaryController` — the WM-integrity canary AS A CAP-AND-HOLD CONTROLLER
  (§5.5): it reduces λ_plan but HOLDS it at a floor and never kills a run (a mid-run
  kill would violate GATE_PROTOCOL §1; only the pre-registered gate step may kill).
  ⭐ v4.2 fix: the earlier naive halve-to-**zero** starved v4.1's planner; this holds
  a floor so the planner always keeps a real gradient into the trunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# Masked target sentinel — the labeler writes this for every window whose LAT/LON/
# DIST/route mode is ``unknown``; F.cross_entropy skips it (§6.5 discipline).
IGNORE_INDEX = -100


# ============================================================================
# The three-phase curriculum (λ_plan + scheduled sampling)
# ============================================================================

@dataclass(frozen=True)
class CurriculumPhases:
    """Phase boundaries, in steps. Phase A = [0, phase_a); Phase B = [phase_a,
    phase_b) is the ramp; Phase C = [phase_b, end). λ_plan and BOTH scheduled-
    sampling ramps (goal + S3) share these boundaries (O-17)."""
    phase_a: int = 2000
    phase_b: int = 8000

    def __post_init__(self):
        if not 0 < self.phase_a < self.phase_b:
            raise ValueError(f"need 0 < phase_a < phase_b, got "
                             f"{self.phase_a}, {self.phase_b}")


def _ramp(step: int, lo: int, hi: int) -> float:
    """0 below ``lo``, linear over [lo, hi), 1 at/above ``hi``."""
    if step < lo:
        return 0.0
    if step >= hi:
        return 1.0
    return (step - lo) / (hi - lo)


def lambda_plan_at(step: int, phases: CurriculumPhases, mode: str = "sched") -> float:
    """The planner→trunk gradient scale at ``step``.

    ``mode``: ``"0"`` reproduces the frozen-trunk v1.5 regime byte-identically
    (Phase A held forever — the ``--lambda-plan 0`` attributability control);
    ``"1"`` is full joint training from step 0; ``"sched"`` is the three-phase
    curriculum. A canary controller may only reduce the scheduled value, never
    raise it (:class:`CanaryController`)."""
    if mode == "0":
        return 0.0
    if mode == "1":
        return 1.0
    if mode != "sched":
        raise ValueError(f"lambda-plan mode must be 0|1|sched, got {mode!r}")
    return _ramp(step, phases.phase_a, phases.phase_b)


def produced_goal_frac(step: int, phases: CurriculumPhases) -> float:
    """Fraction of the batch trained on the PRODUCED goal / own-selected plan
    (vs teacher-forced GT) at ``step`` — scheduled sampling for both the goal (S1)
    and S3. Shares λ_plan's boundaries so that at the G1 gate step the arm is
    fully in its evaluated (produced-goal) configuration (O-17)."""
    return _ramp(step, phases.phase_a, phases.phase_b)


def gate_is_after_ramp(gate_step: int, phases: CurriculumPhases) -> bool:
    """O-17 invariant: the pre-registered gate must fall in Phase C, so the arm is
    scored on the configuration it has actually trained under. False here means the
    gate sits inside a ramp — the exact defect O-17 fixed."""
    return gate_step >= phases.phase_b


# ============================================================================
# Plan smoothness on the DENSE emitted plan (§7)
# ============================================================================

def plan_smoothness_loss(plan: Tensor, w_jerk: float = 0.02,
                         w_curv: float = 0.01) -> tuple[Tensor, dict]:
    """Jerk + curvature-rate penalty on the dense plan ``[B, S, 2]`` (S≥4).

    Jerk is the third position difference (∝ jerk at fixed dt); curvature-rate is
    the second difference of heading (change in turn rate). Both are computable
    ONLY because the emitted plan is dense — a 4-point path admits exactly one
    third difference (§3.3), which is why v4 ships the 20-step plan and puts the
    smoothness term on IT rather than on a sparse, unscored head.
    """
    if plan.shape[-2] < 4:
        raise ValueError(f"smoothness needs a dense plan (S>=4), got S="
                         f"{plan.shape[-2]} — this is the §7 precondition")
    vel = plan[..., 1:, :] - plan[..., :-1, :]            # [B, S-1, 2]
    acc = vel[..., 1:, :] - vel[..., :-1, :]              # [B, S-2, 2]
    jerk = acc[..., 1:, :] - acc[..., :-1, :]             # [B, S-3, 2]
    jerk_pen = (jerk ** 2).sum(dim=-1).mean()

    # heading and its rate-of-change (curvature-rate proxy), guarded at low speed
    theta = torch.atan2(vel[..., 1], vel[..., 0])         # [B, S-1]
    dtheta = torch.atan2(torch.sin(theta[..., 1:] - theta[..., :-1]),
                         torch.cos(theta[..., 1:] - theta[..., :-1]))  # wrap
    curv_rate = dtheta[..., 1:] - dtheta[..., :-1]        # [B, S-3]
    curv_pen = (curv_rate ** 2).mean()

    loss = w_jerk * jerk_pen + w_curv * curv_pen
    return loss, {"jerk": float(jerk_pen.detach()),
                  "curv_rate": float(curv_pen.detach())}


# ============================================================================
# Factorised LAT × LON × DIST cross-entropies (§4.3, §6.2)
# ============================================================================

def _masked_ce(logits: Tensor, target: Tensor) -> Tensor:
    """CE with IGNORE_INDEX rows dropped; 0 (no grad) if every row is masked."""
    if bool((target != IGNORE_INDEX).any()):
        return F.cross_entropy(logits.float(), target, ignore_index=IGNORE_INDEX)
    return logits.sum() * 0.0            # keep it in the graph, contribute nothing


def factorised_ce(out: dict, lat_tgt: Tensor, lon_tgt: Tensor, dist_tgt: Tensor,
                  w_lat: float = 0.05, w_lon: float = 0.05,
                  w_dist: float = 0.05) -> tuple[Tensor, dict]:
    """The three tactical-mode CEs. Each ``unknown`` target is IGNORE_INDEX and is
    masked out — the LONMODE term is the one that does not exist in REF-C today
    (its 5-way softmax cannot express a live longitudinal mode; ``accelerate`` is
    predicted on 0/881 windows). ``w_* = 0`` reproduces the baseline selection
    path (the grafts stay at their zero init), which is the ``isolate select``
    one-lever diff (§16)."""
    l_lat = _masked_ce(out["lat_logits"], lat_tgt)
    l_lon = _masked_ce(out["lon_logits"], lon_tgt)
    l_dist = _masked_ce(out["dist_logits"], dist_tgt)
    loss = w_lat * l_lat + w_lon * l_lon + w_dist * l_dist
    return loss, {"lat_ce": float(l_lat.detach()), "lon_ce": float(l_lon.detach()),
                  "dist_ce": float(l_dist.detach())}


# ============================================================================
# Strategic goal-scalar regression (§4.3, §7A.4) — masked, per-channel scaled
# ============================================================================

# The four strategic goal scalars, in the fixed column order the labeler and the
# head both use (``v4_labels.STRAT_SCALAR_NAMES``): time-to-maneuver (s),
# curvature@3 s (1/m), curvature@5 s (1/m), target-speed@5 s (m/s). Their natural
# units differ by ~200x (seconds vs 1/m), so each column is divided by its scale
# before the smooth-L1 — otherwise the seconds term alone dictates the gradient.
# PROPOSED constants (a §14.4 knob, not a finding): 10 s, one tight-junction
# curvature 1/20 m, again, and SPEED_SCALE.
STRAT_SCALAR_SCALES: tuple[float, ...] = (10.0, 0.05, 0.05, 10.0)


def strategic_scalar_loss(pred: Tensor, target: Tensor, mask: Tensor,
                          scales: tuple[float, ...] = STRAT_SCALAR_SCALES,
                          weight: float = 0.05, beta: float = 1.0
                          ) -> tuple[Tensor, dict]:
    """Masked smooth-L1 over the strategic goal scalars (§4.3 / §7A.4).

    ``pred`` / ``target`` are ``[B, C]`` in PHYSICAL units; ``mask`` ``[B, C]``
    bool marks the windows where each scalar has a valid label (many are
    out-of-horizon — ttm needs a maneuver in range, the 5 s terms need 5 s of
    future). Each column is normalised by ``scales[c]`` before the loss so the
    heterogeneous units are comparable, and a fully-masked column contributes NO
    gradient — the same IGNORE discipline the factorised CE uses, so a scalar with
    2 % coverage is masked, never regressed against a fabricated 0.

    Returns ``(loss, log)`` with per-channel MEASURED coverage and mean error so
    the acceptance metric (a head training on real, non-ignored targets) is
    directly readable.
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"strategic_scalar_loss shape mismatch: pred "
                         f"{tuple(pred.shape)} target {tuple(target.shape)} "
                         f"mask {tuple(mask.shape)}")
    c = pred.shape[-1]
    sc = torch.as_tensor(scales[:c], dtype=pred.dtype, device=pred.device)
    m = mask.to(pred.dtype)
    resid = (pred.float() - target.float()) / sc                     # [B, C]
    per = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=beta,
                           reduction="none") * m                     # [B, C]
    denom = m.sum().clamp_min(1.0)
    # keep the term in the graph even when everything is masked (contributes 0)
    loss = weight * (per.sum() / denom if bool(m.any()) else pred.sum() * 0.0)
    cov = mask.float().mean(dim=0)                                   # [C]
    log = {"strat_scalar_loss": float(loss.detach()),
           "strat_scalar_cov": [round(float(x), 4) for x in cov],
           "strat_scalar_n_valid": int(mask.sum())}
    return loss, log


# ============================================================================
# The WM-integrity canary — a CONTROLLER, never a kill (§5.5)
# ============================================================================

@dataclass
class CanaryController:
    """Turns the plan-free WM canary into an in-loop CAP-AND-HOLD controller on λ_plan.

    It NEVER kills a run — a canary-triggered mid-run kill would violate
    GATE_PROTOCOL §1 (a run is killed only at the pre-registered gate step). It may
    only pull λ_plan DOWN (§14.4 O-14: *"the canary may only move it down"*), but it
    **HOLDS at a floor** and never decays the planner gradient to ~0.

    ⭐ **This is the v4.2 fix (2026-07-23).** v4.1 ran the earlier NAIVE
    HALVE-TO-ZERO controller: a soft breach multiplied ``_mult`` by ``ctrl_factor``
    with NO lower bound, and three hard breaches drove it to **exactly 0**. Because
    on v4.1's cut ``lr_trunk`` *any* meaningful planner gradient briefly breached the
    canary (it read 0.60/0.63 at neighbouring evals while sitting healthy at 0.46 at
    step 10 k), the down-only ratchet fired on nearly every eval and drove
    ``lam_mult`` monotonically to **1.5e-5 by step 10 k / 3.8e-6 by 11 k** — the
    planner→trunk coupling was effectively OFF from ~step 2000. The WM stayed healthy
    (canary 0.46) precisely *because the planner had been sacrificed*: held-out
    ``ade_0_2s`` 0.8522, oracle_in_fan 0.4838, both FAIL (MEASURED,
    ``…/2026-07-23-v4-eval-harness/flagship-v4.1-10k.json``).

    The fix is the design's own R1 intent (V4_FLAGSHIP_DESIGN §11 / the 2026-07-23
    synthesis §4/§7): **cap-and-hold at a floor**. ``_mult`` is bounded below by
    ``mult_floor`` at ALL times, so once λ_plan has ramped up the planner ALWAYS
    keeps at least ``mult_floor`` of its gradient into the trunk. The controller
    still only moves DOWN (O-14 preserved: ``_mult`` is monotone non-increasing, now
    bounded below by the floor instead of 0), still never kills, and the gate at step
    10,000 still adjudicates ``wm_canary_ade_2s`` separately — the gate, not the
    controller, is the only thing that may stop a genuinely collapsing WM. WM
    protection here comes from ``lr_trunk`` (1e-4 in v4.2) + the gate; the controller
    only guarantees the planner is not starved to fix a noisy canary.
    """
    baseline: float                          # step-0 canary on the warm trunk (v1: 0.452)
    ctrl_thresh: float = 0.05                # soft: canary_vs_base above this -> ×ctrl_factor (clamped at floor)
    alarm_thresh: float = 0.30               # hard: above this on alarm_evals in a row -> floor (NOT 0)
    ctrl_factor: float = 0.5
    alarm_evals: int = 3
    mult_floor: float = 0.25                 # ⭐ CAP-AND-HOLD floor: _mult (hence the planner→trunk
                                             # gradient) is NEVER reduced below this. The v4.2 fix for
                                             # the halve-to-zero starvation. A §14.4 knob (--lam-mult-floor).
    _hard_streak: int = 0
    _mult: float = 1.0                       # accumulated multiplier on the scheduled λ, in [mult_floor, 1]

    def __post_init__(self):
        if not 0.0 < self.mult_floor <= 1.0:
            raise ValueError(f"mult_floor must be in (0, 1], got {self.mult_floor} "
                             "— a floor of 0 IS the v4.1 halve-to-zero bug this fixes")
        # a resumed / hand-set _mult must respect the floor (never re-open below it)
        self._mult = max(self.mult_floor, min(1.0, self._mult))

    def update(self, canary_value: float) -> tuple[float, str]:
        """Feed the latest canary; return ``(lambda_multiplier, action)``. The
        multiplier only ratchets DOWN (monotone non-increasing) and is HELD AT the
        floor — matching "the canary may only move λ_plan down" (§14.4 O-14) while
        never starving the planner to ~0 (the v4.1 bug)."""
        delta = canary_value - self.baseline
        if delta > self.alarm_thresh:
            # HARD breach: on ``alarm_evals`` in a row, drop to — and HOLD at — the
            # floor. NEVER to zero (that was the v4.1 starvation). The run continues;
            # only the pre-registered gate may stop a genuinely collapsing WM.
            self._hard_streak += 1
            if self._hard_streak >= self.alarm_evals:
                self._mult = self.mult_floor
                action = "alarm_lambda_to_floor"
            else:
                action = f"hard_breach_{self._hard_streak}/{self.alarm_evals}"
        elif delta > self.ctrl_thresh:
            # SOFT breach: step down by ctrl_factor, CLAMPED at the floor (cap-and-hold).
            self._hard_streak = 0
            self._mult = max(self.mult_floor, self._mult * self.ctrl_factor)
            action = ("controller_held_at_floor"
                      if self._mult <= self.mult_floor + 1e-12
                      else "controller_halved_lambda")
        else:
            # within the acceptable band: HOLD. A healthy (even noisy-but-healthy)
            # canary must NOT decay the planner — the exact failure v4.1 hit.
            self._hard_streak = 0
            action = "ok"
        return self._mult, action

    def effective_lambda(self, scheduled: float) -> float:
        """The scheduled λ_plan after the controller's (down-only, floored) multiplier."""
        return scheduled * self._mult
