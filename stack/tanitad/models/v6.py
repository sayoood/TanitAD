"""v6 — the STAGED hierarchical world-model composition module.

WHAT THIS MODULE IS. The v6 *composition* layer: per-layer predictors
(operative / tactical / strategic), the shared goal-token vocabulary, the
60-step unicycle emission that covers BOTH bands of the binding 6 s horizon,
and — the part that is machinery rather than decoration — the **gradient
isolation matrix enforced in code** and checkable by :meth:`V6Stack.assert_isolation`.

Everything reusable is IMPORTED, never re-implemented:
  * :class:`tanitad.models.encoder.ViTEncoder` + :class:`~tanitad.models.readout.SpatialGridReadout`
    (the geometry firewall — a wide input still yields the same ``state_dim``);
  * :class:`tanitad.models.predictor.OperativePredictor` — the operative
    predictor, FiLM-conditioned; its ``intent`` port IS the ``g_tac``
    conditioning seam (``P_O(z_op, (a,κ) | g_tac)``);
  * :class:`tanitad.models.tactical.FTac` — the residual-MLP latent dynamics
    family, used at BOTH the tactical and the strategic clock;
  * :class:`tanitad.models.sigreg.SigReg` (+ ``position_relaxed``) — O6;
  * ``scripts/train_v58f_unicycle_head.UnicycleEmission`` / ``unicycle_rollout``
    — the W4-proven ``a = a_max·tanh``, ``κ = κ_max·tanh`` emission, lazily
    imported through the ``tanitad/models/v58f.py:_ensure_scripts`` precedent
    (``stack/scripts`` is not an importable package at ``tanitad`` module scope);
  * :func:`tanitad.eval.spectral.effective_rank` / ``participation_ratio`` — O6's
    standing spectrum monitor.

THE SPECS THIS IMPLEMENTS (they are the requirements, quoted where binding):
  * ``V6_TRAINING_MEASURES.md`` — O1–O6 / T1–T5 / S1–S3 / C1–C2 / X1–X5 and the
    staged protocol **S-W → S-T → S-S → (optional) S-J** (X5);
  * ``HIERARCHY_VOCABULARY.md`` v0.2 — §3/§4 the goal & action token vocabulary,
    §4b the **binding 6 s horizon spec**, §5 the wiring;
  * ``JEPA_PHYSICS_SURVEY.md`` — LF0–LF4, and the staged-training evidence
    (Drive-JEPA / V-JEPA 2-AC / DINO-WM: *nobody at the frontier co-trains the
    planning gradient into the encoder from step 0*).

⛔ THE THREE BINDING RULES THIS FILE ENFORCES MECHANICALLY

1. **X3 gradient-isolation matrix.** Planner/goal heads NEVER backprop into any
   encoder; higher layers reach lower layers' latents ONLY through stop-grad or
   an EMA-slow copy. Both are config flags (:class:`V6Config`), both are
   MEASURED by :meth:`V6Stack.assert_isolation` — a real autograd probe over the
   module's own DECLARED planner-side surface, not a comment.
2. **One vocabulary, two views (§5).** The goal-token embedding table is the
   SAME ``nn.Module`` object in the emitting head above and the consuming
   conditioner below (``id(...)`` identity, unit-pinned). A second table would
   be two vocabularies wearing one name.
3. **§4b seam-free by construction.** There is ONE 60-step (a, κ)@10 Hz control
   sequence integrated through ONE unicycle rollout 0→6 s. 0–2 s is the
   operative band, 2–6 s the tactical band — *bands are SLICES of one rollout*,
   never two stitched trajectories, so the 2 s seam cannot be discontinuous and
   X2's seam metrics VERIFY rather than repair.

⚠️ GOAL ADMISSIBILITY (Sayed, 2026-08-03, binding). A goal input is admissible;
the **output of the situation classifier is not**, in any form. Stated for this
module, as the rule requires: the ONLY inputs to every goal head here are the
layer's own latent (vision-derived) and the goal embedding handed DOWN from the
layer above. There is no situation-classifier port, no ego-state port, and no
``**kwargs`` back door (signatures are pinned by ``tests/test_v6_staged.py``).
The shared-trunk disclosure: all layers may read one encoder (the E-ENC arm
``shared_encoder=True``), and information flows trunk→heads only — no head's
OUTPUT re-enters another head's input path.

⚠️ VISION-ONLY AT INFERENCE (Sayed, 2026-08-03, binding). The layer latents come
from the vision encoder. ``v0`` (initial speed) enters ONLY the unicycle
INTEGRATOR — it is the integration constant of a kinematic rollout, not a
perception input, exactly as in the W4-gated ``UnicycleEmission``. A deployment
that cannot read its own speedometer cannot integrate any trajectory at all.

TIER STAMP. This module is MECHANISM, not measurement. No number produced by
calling anything here is quotable as a result; capability claims come from the
T1 instrument (``taniteval/tools/t1_eval.py``) with its estimator and its four
metric families (EVAL_DOCTRINE.md; Sayed 2026-08-02).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig
from tanitad.eval.spectral import effective_rank, participation_ratio
from tanitad.models.encoder import ViT5Encoder, ViTEncoder
from tanitad.models.metric_dynamics import StepDisplacementReadout
from tanitad.models.predictor import OperativePredictor
from tanitad.models.readout import SpatialGridReadout
from tanitad.models.sigreg import SigReg
from tanitad.models.tactical import FTac

__all__ = [
    # vocabulary (HIERARCHY_VOCABULARY §3/§4)
    "STRATEGIC_GOAL_TOKENS", "STRATEGIC_ACTION_TOKENS", "TACTICAL_GOAL_TOKENS",
    "TACTICAL_LAT_ACTIONS", "TACTICAL_LON_ACTIONS", "CONSTRAINT_SLOTS",
    "GOAL_ARG_SLOTS", "GOAL_ARG_NAMES",
    # ⭐ g_tac factored LAT x LON + the typed categorical arg channel
    "TACTICAL_GOAL_TOKENS_LAT", "TACTICAL_GOAL_TOKENS_LON",
    "GOAL_AXIS_ABSTAIN", "goal_token_axis",
    "GOAL_CAT_ARG_NAMES", "GOAL_CAT_ARG_TOKENS", "STOP_REASONS",
    "LIGHT_STATES",
    # config + stack
    "V6Config", "V6Stack", "GoalVocabulary", "GoalHead", "GoalConditioner",
    "GoalDistanceScorer", "MLPCandidateScorer", "AnchorGoalHead",
    "IsolationViolation", "PARAM_BUDGET",
    # horizon (§4b)
    "PLAN_STEPS", "DT", "HORIZON_S", "OP_BAND_S", "TAC_BAND_S",
    # staging (X5)
    "STAGES", "STAGE_GROUPS", "MODULE_GROUPS", "stage_trainable_groups",
    "apply_stage_freeze",
    # measure primitives (O2/O3/O4/O6)
    "time_to_reach", "time_to_reach_weights", "half_weight_distance_m",
    "readout_grid_ranges", "sample_cell_block_mask", "near_field_band_mask",
    "kinematic_saliency", "saliency_weights", "InteractionSampler",
    "spectrum_report", "SpectrumAccumulator", "o6_rank_verdict",
    "O6_ADMISSIBLE_CEILING", "O6_RANK_FLOOR",
]

# ============================================================================
# §4b — THE BINDING HORIZON SPEC (PI 2026-08-11, HIERARCHY_VOCABULARY.md)
# ============================================================================
#: ONE control sequence, 60 steps @ 10 Hz = 6.0 s, integrated by ONE unicycle
#: rollout. Emission k scales 20 → 60; roll selection rolls to 6 s.
PLAN_STEPS = 60
DT = 0.1                       # 10 Hz tick — the dense-horizon contract
HORIZON_S = PLAN_STEPS * DT    # 6.0 s
OP_BAND_S = (0.0, 2.0)         # operative band — fine control authority
TAC_BAND_S = (2.0, 6.0)        # tactical band — same controls, g_tac-shaped
#: sub-300M is a programme INVARIANT (CLAUDE.md), not a preference.
PARAM_BUDGET = 300_000_000


# ============================================================================
# HIERARCHY_VOCABULARY v0.2 §3/§4 — the tokenizable goal/action vocabulary
# ============================================================================
#: §3 strategic goals `g_str` (8–30 s+). ``FOLLOW_MAIN_ROAD`` is THE DEFAULT
#: whenever no navigation route is set up (PI 2026-08-11); ``NONE_ABSTAIN``
#: survives only for genuinely ambiguous geometry.
STRATEGIC_GOAL_TOKENS: tuple[str, ...] = (
    "KEEP_CORRIDOR", "LANE_TARGET", "EXIT_RIGHT", "EXIT_LEFT",
    "TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH", "ROUTE_TO", "STOP_AT",
    "FOLLOW_MAIN_ROAD", "NONE_ABSTAIN",
)
#: §3 strategic actions `a_str` — emitted by S, condition S's OWN predictor.
STRATEGIC_ACTION_TOKENS: tuple[str, ...] = (
    "PREPARE_LANE_CHANGE", "HOLD_CORRIDOR", "REDUCE_TO", "PREPARE_EXIT",
    "PREPARE_STOP", "RESUME_CRUISE",
)
#: §4 tactical goals `g_tac` (2–6 s — the earlier 2–8 s note is SUPERSEDED by
#: §4b). ``SPEED_BAND`` is a TACTICAL responsibility (PI decision 2026-08-11):
#: target speed from sign/OCR + corridor speed priors, with the strategic
#: ``REDUCE_TO`` acting only as an upper envelope.
TACTICAL_GOAL_TOKENS: tuple[str, ...] = (
    "ANCHOR_GOAL", "CORRIDOR_OFFSET", "GAP_TARGET", "SPEED_BAND", "YIELD_AT",
    "STOP_POINT", "WAIT_FOR_ONCOMING", "EVADE_IN_CORRIDOR",
    "TRAFFIC_LIGHT_REACT",
)
#: §4 tactical actions `a_tac`, FACTORED LAT × LON — the 5-way-mixed softmax
#: (the programme's single largest known defect) is retired BY DESIGN.
TACTICAL_LAT_ACTIONS: tuple[str, ...] = (
    "LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L",
    "NUDGE_R",
)
TACTICAL_LON_ACTIONS: tuple[str, ...] = (
    "FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
)
#: §2 "every goal token carries OPTIONAL temporal and spatial constraint slots
#: … uniformly typed". Unset = unconstrained (the mask says which are set).
CONSTRAINT_SLOTS: tuple[str, ...] = (
    "within_m", "by_time_s", "at_arc_m", "hold_for_s",
)
#: Four token-specific typed slots + the four uniform constraint slots. Args
#: are PHYSICAL UNITS (m, s, m/s) — "no free text at inference" (§2).
GOAL_ARG_NAMES: tuple[str, ...] = (
    "arg0", "arg1", "arg2", "arg3", *CONSTRAINT_SLOTS,
)
GOAL_ARG_SLOTS = len(GOAL_ARG_NAMES)          # 8

# ----------------------------------------------------------------------------
# ⭐ g_tac FACTORED LAT × LON — the SAME factoring `a_tac` already carries
# ----------------------------------------------------------------------------
# ⛔ WHY. MEASURED 2026-08-16 (`…/incoming/2026-08-16-anchor-goal-supervision/`,
# E-AG1/E-AG2, 881 windows / 40 episodes, LOEO, episode-cluster bootstrap):
#   * the 2 s goal point's corpus variance is **98.8 % LONGITUDINAL** — the
#     zero-information null splits sigma_long 19.0578 vs sigma_lat 2.0723
#     (9.2x in sigma, 84x in variance);
#   * a K-way `anchor_id` classifier is **near-adequate laterally** (1.3310
#     against a 0.6802 floor, 1.96x) and **hopeless longitudinally** (13.3502
#     against a 0.8954 floor, 14.9x);
#   * the quantisation itself is ISOTROPIC (0.5674 long / 0.5599 lat) — the
#     shipped FPS vocabulary spends half its resolution on the axis carrying
#     1.2 % of the variance.
# ⇒ ONE K-way `anchor_id` forces ONE categorical decision to carry BOTH axes.
# That is the 5-way manoeuvre softmax defect — "the programme's single largest
# known defect" — surviving one level up into the goal vocabulary, AFTER
# `a_tac` was explicitly factored LAT x LON to retire it. These two tuples are
# that same retirement applied to `g_tac`.
#
# ⚠️ THE PARTITION IS BY WHICH AXIS THE TOKEN CONSTRAINS, and it is TOTAL:
# LAT + LON reproduce TACTICAL_GOAL_TOKENS exactly (pinned by
# ``tests/test_v6_factored_goal.py::test_the_partition_is_total_and_disjoint``).
# Each side carries its own ABSTAIN, because a factored head that cannot say
# "this axis is unconstrained" must invent a constraint on every window (§2:
# "Unset = unconstrained").
TACTICAL_GOAL_TOKENS_LAT: tuple[str, ...] = (
    "ANCHOR_GOAL", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "LAT_UNCONSTRAINED",
)
TACTICAL_GOAL_TOKENS_LON: tuple[str, ...] = (
    "SPEED_BAND", "GAP_TARGET", "YIELD_AT", "STOP_POINT", "WAIT_FOR_ONCOMING",
    "TRAFFIC_LIGHT_REACT", "LON_UNCONSTRAINED",
)
#: the two abstains are NOT part of the §4 nine — they are the factoring's own.
GOAL_AXIS_ABSTAIN: tuple[str, str] = ("LAT_UNCONSTRAINED", "LON_UNCONSTRAINED")

# ----------------------------------------------------------------------------
# ⭐ THE CATEGORICAL ARG CHANNEL — the arg-TYPE gap, MEASURED 2026-08-16 §2.3
# ----------------------------------------------------------------------------
# ⛔ THE GAP. `GOAL_ARG_NAMES` is eight slots of PHYSICAL UNITS (m, s, m/s);
# both ends are continuous — ``GoalHead.arg_head`` emits 8 floats and
# ``GoalVocabulary.arg_proj`` consumes 8 floats. But **SEVEN of the nine
# `g_tac` tokens carry at least one CATEGORICAL arg** (`anchor_id`,
# `agent_slot_id`, `gap_slot`, `reason`, `oncoming_slot`, `obstacle_slot`,
# `light_slot_id`, `state`). An index is not a physical quantity: with an
# FPS-ordered vocabulary anchor 5 is not "between" anchors 4 and 6 in any
# geometry, so regressing it is a TYPE ERROR. ⇒ before this channel existed,
# **`goal_head_tac` could express exactly 2 of the 9 tokens** even with perfect
# labels — a CODE gap, not a data gap.
#
# ⚠️ THE FIVE SLOT-VALUED ARGS ARE ONE KIND, NOT FIVE. `agent_slot_id`,
# `gap_slot`, `oncoming_slot`, `obstacle_slot` and `light_slot_id` are all "an
# id into the window's agent-slot vocabulary", and no token needs two of them
# at once (checked token by token against HIERARCHY_VOCABULARY.md:84-92), so
# ONE `agent_slot` channel serves all five.
#: The typed categorical slots. ``anchor_id`` indexes the FULL 2-D anchor
#: table; ``lat_bin`` indexes the FACTORED lateral sub-vocabulary — they are
#: DIFFERENT vocabularies and a token uses one or the other, never both.
#: Conflating them would be the very type error this channel exists to remove.
GOAL_CAT_ARG_NAMES: tuple[str, ...] = (
    "anchor_id", "lat_bin", "agent_slot", "reason", "state",
)
#: `STOP_POINT(position_arc_m, reason)` — HIERARCHY_VOCABULARY.md:89.
STOP_REASONS: tuple[str, ...] = ("sign", "light", "queue", "hazard")
#: `TRAFFIC_LIGHT_REACT(light_slot_id, state, stopline_arc_m)` — :92. The B2
#: VLM schema emits exactly these four.
LIGHT_STATES: tuple[str, ...] = ("red", "amber", "green", "none")
#: §2.2's per-token table, turned from prose into a tested artefact: which
#: categorical slot(s) each `g_tac` token needs. A token absent here needs
#: none (`CORRIDOR_OFFSET` and `SPEED_BAND` — the only two that were
#: expressible before this channel). ``ANCHOR_GOAL`` is listed with BOTH ids
#: because the joint and the factored formulations are separate arms.
GOAL_CAT_ARG_TOKENS: dict[str, tuple[str, ...]] = {
    "ANCHOR_GOAL": ("anchor_id", "lat_bin"),
    "GAP_TARGET": ("agent_slot",),
    "YIELD_AT": ("agent_slot",),
    "STOP_POINT": ("reason",),
    "WAIT_FOR_ONCOMING": ("agent_slot",),
    "EVADE_IN_CORRIDOR": ("agent_slot",),
    "TRAFFIC_LIGHT_REACT": ("agent_slot", "state"),
}


def goal_token_axis(token: str) -> str:
    """``"lat"`` / ``"lon"`` for a `g_tac` token — the ONE place the partition
    is read, so the programme has one convention and not two."""
    if token in TACTICAL_GOAL_TOKENS_LAT:
        return "lat"
    if token in TACTICAL_GOAL_TOKENS_LON:
        return "lon"
    raise KeyError(f"{token!r} is in neither factored g_tac axis")


def _ensure_scripts() -> None:
    """Make ``stack/scripts`` importable — the ``models/v58f.py:_ensure_scripts``
    precedent (``tanitad/models/v6.py`` -> ``parents[2]`` == the stack root).
    Used ONLY for the W4-gated unicycle emission, which lives in a script."""
    sp = str(Path(__file__).resolve().parents[2] / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)


# ============================================================================
# O2 — TIME-TO-REACH weighting (HIERARCHY_VOCABULARY §2, PI correction)
# ============================================================================
# ⚠️ "Near-field is TIME-scaled, not metre-scaled (PI correction): a fixed 40 m
#    band cannot cover a 6 s horizon (180 m at 30 m/s). All near-field weighting
#    (V6 measure O2) and constraint defaults use TIME-TO-REACH (arc_length /
#    v_ego, capped at the 6 s horizon) — speed-adaptive by construction."
# The V6_TRAINING_MEASURES O2 row still says "0–40 m band"; §2 of the vocabulary
# is the LATER, correcting statement and is what is implemented here.

def time_to_reach(dist_m: Tensor, v_ego: Tensor, *, v_floor: float = 1.0,
                  horizon_s: float = HORIZON_S) -> Tensor:
    """Time-to-reach ``arc / v_ego`` in seconds, floored in speed and CAPPED at
    ``horizon_s``.

    ``dist_m`` [..., C] arc-length (metres) per readout cell, ``v_ego`` [B] or
    broadcastable ego speed (m/s). ``v_floor`` keeps a stopped ego from sending
    every cell to +inf (at v=0 nothing is reachable inside the horizon, which
    the cap already expresses). Returns the same broadcast shape as
    ``dist_m * v_ego``.
    """
    if v_floor <= 0.0:
        raise ValueError(f"v_floor must be > 0, got {v_floor}")
    if horizon_s <= 0.0:
        raise ValueError(f"horizon_s must be > 0, got {horizon_s}")
    v = v_ego.to(dist_m.dtype).clamp_min(v_floor)
    while v.ndim < dist_m.ndim:
        v = v.unsqueeze(-1)
    return (dist_m.abs() / v).clamp(0.0, horizon_s)


def time_to_reach_weights(dist_m: Tensor, v_ego: Tensor, *, tau_s: float = 2.0,
                          v_floor: float = 1.0, horizon_s: float = HORIZON_S,
                          normalize: bool = True) -> Tensor:
    """O2's per-cell loss weights: ``exp(-t_reach / tau_s)``.

    SPEED-ADAPTIVE BY CONSTRUCTION, and that is the whole point: two cells with
    the SAME time-to-reach get the SAME weight whatever their metre distance,
    so a faster ego automatically widens the metre band it is asked to model
    while the TIME band stays fixed (:func:`half_weight_distance_m` makes the
    widening explicit and is unit-pinned).

    ``normalize`` rescales so the weights MEAN 1 over the cell axis, which keeps
    the O2 term's gradient magnitude comparable to an unweighted latent loss —
    a weighting must re-allocate the loss, not silently rescale it.
    """
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be > 0, got {tau_s}")
    t = time_to_reach(dist_m, v_ego, v_floor=v_floor, horizon_s=horizon_s)
    w = torch.exp(-t / tau_s)
    if normalize:
        w = w / w.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return w


def half_weight_distance_m(v_ego: Tensor | float, *, tau_s: float = 2.0,
                           v_floor: float = 1.0) -> Tensor:
    """The metre distance at which :func:`time_to_reach_weights` halves:
    ``v · tau · ln2``. Linear in speed — the measured statement of "higher
    speed ⇒ wider metre band, same time band"."""
    v = torch.as_tensor(v_ego, dtype=torch.float32).clamp_min(v_floor)
    return v * float(tau_s) * math.log(2.0)


def readout_grid_ranges(grid_h: int, grid_w: int, *, near_m: float = 3.0,
                        far_m: float = 80.0) -> Tensor:
    """Per-cell nominal ego-frame range [grid_h, grid_w] for the readout grid.

    ⚠️ EVIDENCE CLASS: **ESTIMATED — a declared monotone image-row prior, NOT
    calibrated depth.** PhysicalAI-AV ships no map and no depth channel (the
    five-probe settled result in CLAUDE.md), so there is no measured cell→metre
    table to import. What IS defensible is the monotonicity: in a forward
    camera the lower image rows are nearer. Rows are laid out image-order (row
    0 = TOP = far, row ``grid_h-1`` = BOTTOM = near) and spaced GEOMETRICALLY,
    because image row maps roughly to inverse depth. Columns share a row's
    range (no lateral depth cue without calibration — saying so is the point).

    A calibrated table can be dropped in later without touching O2: everything
    downstream consumes metres, and :func:`time_to_reach_weights` converts.
    """
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"bad grid {grid_h}x{grid_w}")
    if not 0.0 < near_m < far_m:
        raise ValueError(f"need 0 < near_m < far_m, got {near_m}, {far_m}")
    if grid_h == 1:
        rows = torch.full((1,), math.sqrt(near_m * far_m))
    else:
        # row 0 (top) -> far_m ; row grid_h-1 (bottom) -> near_m
        frac = torch.linspace(1.0, 0.0, grid_h)
        rows = near_m * (far_m / near_m) ** frac
    return rows[:, None].expand(grid_h, grid_w).contiguous()


# ============================================================================
# O3 — masked SPATIAL-latent prediction over the readout grid (I-JEPA adapted)
# ============================================================================

def sample_cell_block_mask(grid_h: int, grid_w: int, *, n_blocks: int = 2,
                           block_h: int = 2, block_w: int = 2,
                           batch: int = 1,
                           generator: torch.Generator | None = None,
                           device=None) -> Tensor:
    """CONTIGUOUS spatial block masks over the readout grid — the
    occluded-vehicle surrogate of O3 (I-JEPA masking adapted to BEV-ish cell
    tokens). Returns bool ``[batch, grid_h*grid_w]``, ``True`` == MASKED.

    Contiguity is the load-bearing property: scattered per-cell dropout is
    trivially inpainted from neighbours and teaches nothing about permanence,
    which is exactly RC3 in ``JEPA_PHYSICS_SURVEY.md`` §2. Blocks may overlap
    (a rejection loop would make the mask rate data-dependent and the loss
    non-stationary); the realised rate is therefore ≤ the nominal one and is
    reported by the caller, never assumed.
    """
    if block_h < 1 or block_w < 1 or n_blocks < 0:
        raise ValueError(f"bad block spec {n_blocks}x({block_h},{block_w})")
    if block_h > grid_h or block_w > grid_w:
        raise ValueError(f"block ({block_h},{block_w}) does not fit in grid "
                         f"({grid_h},{grid_w})")
    m = torch.zeros(batch, grid_h, grid_w, dtype=torch.bool, device=device)
    if n_blocks == 0:
        return m.reshape(batch, grid_h * grid_w)
    hi_r, hi_c = grid_h - block_h + 1, grid_w - block_w + 1
    r0 = torch.randint(hi_r, (batch, n_blocks), generator=generator)
    c0 = torch.randint(hi_c, (batch, n_blocks), generator=generator)
    for b in range(batch):
        for j in range(n_blocks):
            r, c = int(r0[b, j]), int(c0[b, j])
            m[b, r:r + block_h, c:c + block_w] = True
    return m.reshape(batch, grid_h * grid_w)


def near_field_band_mask(grid_h: int, grid_w: int, *, rows: int = 1,
                         batch: int = 1, device=None) -> Tensor:
    """O3's second masking mode — mask whole NEAR-FIELD bands (the bottom
    ``rows`` readout rows, where physics bites). Bool ``[batch, H*W]``,
    ``True`` == masked. Deterministic on purpose: it is a fixed stressor, and
    a stressor that moves cannot be compared across steps."""
    if not 0 <= rows <= grid_h:
        raise ValueError(f"rows must be in [0, {grid_h}], got {rows}")
    m = torch.zeros(batch, grid_h, grid_w, dtype=torch.bool, device=device)
    if rows:
        m[:, grid_h - rows:, :] = True
    return m.reshape(batch, grid_h * grid_w)


# ============================================================================
# O4 — INTERACTION-WEIGHTED SAMPLING, from ACTIONS ONLY (label-free = LF1)
# ============================================================================

def kinematic_saliency(actions: Tensor, *, dt: float = DT,
                       w_jerk: float = 1.0, w_decel: float = 1.0,
                       w_reversal: float = 1.0,
                       jerk_scale: float = 5.0, decel_scale: float = 2.0
                       ) -> Tensor:
    """Per-window ego-kinematic saliency from the RECORDED ACTIONS ALONE.

    ``actions`` ``[N, T, >=2]`` in the corpus 2-channel layout (steer, accel);
    extra channels (the speed column ``lift_actions3`` appends) are ignored.
    Returns ``[N]``, non-negative.

    Three terms, each the doc's own words (O4 / LF1: *"|jerk|, |decel|,
    steering reversals — from actions ONLY, label-free"*):
      * ``|jerk|``     mean ``|Δaccel| / dt``, scaled by ``jerk_scale`` m/s³;
      * ``|decel|``    mean ``relu(-accel)``, scaled by ``decel_scale`` m/s²;
      * ``reversals``  fraction of steering-rate SIGN CHANGES (already in [0,1]).

    ⛔ NO PERCEPTION LABEL ENTERS. That is what makes O4 admissible under the
    JEPA thesis and what makes it LF1 rather than an aux-label lever: the
    obstacle join and the VLM fields are frozen-probe/eval-strata material,
    never a training-time selector.
    """
    if actions.ndim != 3 or actions.shape[-1] < 2:
        raise ValueError(f"actions must be [N, T, >=2], got "
                         f"{tuple(actions.shape)}")
    if actions.shape[1] < 3:
        raise ValueError(f"saliency needs T >= 3 (a jerk and a reversal need "
                         f"two differences), got T={actions.shape[1]}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    a = actions.float()
    steer, accel = a[..., 0], a[..., 1]
    jerk = torch.diff(accel, dim=-1).abs() / dt              # [N, T-1]
    s_jerk = (jerk.mean(dim=-1) / jerk_scale)
    s_decel = (accel.clamp_max(0.0).abs().mean(dim=-1) / decel_scale)
    d_steer = torch.diff(steer, dim=-1)                      # [N, T-1]
    rev = (torch.sign(d_steer[..., 1:]) * torch.sign(d_steer[..., :-1]) < 0)
    s_rev = rev.float().mean(dim=-1)                         # [N]
    return (w_jerk * s_jerk + w_decel * s_decel
            + w_reversal * s_rev).clamp_min(0.0)


def saliency_weights(scores: Tensor, *, alpha: float = 1.0,
                     floor: float = 0.25, normalize: bool = True) -> Tensor:
    """Saliency -> sampling weights: ``(floor + s)**alpha``, optionally
    normalised to sum 1.

    ``floor > 0`` is NOT cosmetic — it keeps every free-flow window reachable.
    A sampler that can never draw free flow re-selects the corpus, and
    re-selecting the corpus is the one thing parity forbids (CLAUDE.md: the
    canonical corpus is ``physicalai-train-e438721ae894``, skip-hash
    ``f09e44db``). O4 REWEIGHTS the draw; it never removes a window, so every
    arm still sees the same 2376 episodes and cross-arm comparability holds.
    ``alpha=0`` reproduces uniform sampling exactly — the attributability
    control.
    """
    if floor < 0:
        raise ValueError(f"floor must be >= 0, got {floor}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    s = scores.float().clamp_min(0.0)
    w = (floor + s) ** alpha
    if normalize:
        w = w / w.sum().clamp_min(1e-12)
    return w


class InteractionSampler:
    """O4's episode-grouped, saliency-weighted batch sampler.

    Wraps the MEASURED I/O shape of ``train_v58f_unicycle_head.make_sampler``
    (few episodes × many windows per batch — random windows over a MooseFS LRU
    is ~30 cold payload loads per batch; grouping cuts that ~8×) and draws the
    windows INSIDE the chosen episodes by :func:`saliency_weights` instead of
    uniformly. Episodes are drawn uniformly so no episode is starved.

    ``weights`` is a per-window tensor aligned with ``index`` (the dataset's
    ``(episode, t)`` list). ``alpha=0`` ⇒ byte-identical behaviour to uniform
    within-episode sampling, which is how the O4 ablation arm is run.
    """

    def __init__(self, index, weights: Tensor, *, eps_per_batch: int = 4,
                 generator: torch.Generator | None = None):
        if len(index) != len(weights):
            raise ValueError(f"index ({len(index)}) and weights "
                             f"({len(weights)}) must align 1:1")
        self.ep2idx: dict[int, list[int]] = {}
        for i, (e, _t) in enumerate(index):
            self.ep2idx.setdefault(int(e), []).append(i)
        self.ep_ids = list(self.ep2idx)
        self.weights = weights.float().clamp_min(0.0)
        self.eps_per_batch = int(eps_per_batch)
        self.gen = generator

    def __call__(self, bs: int) -> list[int]:
        n_ep = min(self.eps_per_batch, len(self.ep_ids))
        pick = torch.randint(len(self.ep_ids), (n_ep,), generator=self.gen)
        out: list[int] = []
        gi = 0
        while len(out) < bs:
            pool = self.ep2idx[self.ep_ids[int(pick[gi % n_ep])]]
            w = self.weights[torch.tensor(pool)]
            if float(w.sum()) <= 0:
                w = torch.ones_like(w)
            j = int(torch.multinomial(w, 1, generator=self.gen))
            out.append(pool[j])
            gi += 1
        return out


# ============================================================================
# O6 — the standing SPECTRUM monitor (participation ratio, effective rank)
# ============================================================================

#: The reading is ADMISSIBLE for a rank verdict only above this ceiling.
#: ⛔ WHY A CEILING EXISTS AT ALL: a centred covariance built from ``n`` rows has
#: rank ≤ n-1, so ``effective_rank`` cannot exceed n-1 however healthy ``d``
#: dimensions are. The live v6F S-W run measures n=48 rows of d=2048 — the
#: reading is bounded by **47**, and "15 of 2048" is a category error: it is
#: 15 of 47. MEASURED (`SIGREG_GATE_POWER.md`, 2026-08-16): at n=48 an
#: isotropic d=2048 population (true effective rank 2048) reads **46.86**, and
#: a population whose true rank has collapsed 7.3× to 281 still reads 22.6 —
#: the estimator saturates and stops carrying the information the gate needs.
#: 1024 is the smallest power of two that clears the pre-registered absolute
#: floor (:data:`O6_RANK_FLOOR`) by 16×.
O6_ADMISSIBLE_CEILING = 1024

#: Pre-registered absolute floor on a POOLED ``effective_rank`` (see
#: ``SIGREG_GATE_POWER.md`` §5 for the both-outcomes registration). MEASURED in
#: that document at pool 32 (ceiling 1535): a healthy α=2 power-law population
#: reads **122.4**, the same population collapsed to 16 retained directions
#: reads **19.4**. 64 sits between them with ~2× margin either way, and is 8×
#: the ``top_k`` the energy share is taken over.
O6_RANK_FLOOR = 64.0


def spectrum_report(z: Tensor, *, top_k: int = 8, ci_reps: int = 0,
                    block: int = 1, generator=None) -> dict:
    """Collapse diagnostics for a latent batch ``[n, d]`` (leading dims are
    flattened): participation ratio, entropy effective rank, top-k energy
    share, the raw n/d, **and the rank ceiling the reading is bounded by**.

    Both statistics are IMPORTED from ``tanitad.eval.spectral`` — the same
    functions the orthogonality/spectral instruments use, so the O6 training
    monitor and the offline analysis cannot drift apart. Computed on the
    CENTRED covariance eigenvalues (= squared singular values of the centred
    batch): collapse is a statement about VARIANCE directions, and an
    uncentred spectrum would be dominated by the mean vector.

    ⛔ ``rank_ceiling`` AND ``effective_rank_frac`` ARE NOT DECORATION. Without
    them the number is routinely read against ``d``, which is wrong by
    construction (see :data:`O6_ADMISSIBLE_CEILING`). They are emitted
    unconditionally so no record can be quoted without its own ceiling.

    ``ci_reps > 0`` adds an interval on the reading, with the CLUSTER as the
    resampling unit (``block`` consecutive rows) — the correct unit when the
    rows are ``W`` consecutive frames of the same window rather than
    independent draws.

    ⛔ THE INTERVAL IS A **LEAVE-ONE-CLUSTER-OUT JACKKNIFE**, NOT A BOOTSTRAP,
    AND THAT IS A MEASURED CHOICE. Bootstrap-with-replacement DUPLICATES
    blocks, and duplicated rows are exactly rank-deficient — for a RANK
    functional that is a systematic downward bias, not noise. MEASURED coverage
    of the finite-n estimand (``SIGREG_GATE_POWER.md`` §4, 60 datasets):

    ======================  =========  =========
    interval                 48 rows    384 rows
    ======================  =========  =========
    percentile bootstrap        0.250      0.000
    pivotal bootstrap           0.300      0.000
    **cluster jackknife**   **0.850**  **0.867**
    ======================  =========  =========

    ⚠️ This is a deliberate, evidence-backed carve-out from the programme rule
    that decision-grade intervals are the episode-cluster BOOTSTRAP. That rule
    is right for mean-like eval metrics; it fails here because the estimand is
    a rank. The bootstrap bounds are still emitted, LABELLED as diagnostics
    with their measured coverage, so the carve-out is visible in every record.
    ⚠️ 0.85-0.87 against a nominal 0.95 is mildly ANTI-CONSERVATIVE: treat the
    interval as a working uncertainty, not an exact guarantee.

    Defaults (``ci_reps=0, block=1``) leave the computation bit-for-bit what
    it was, and draw NOTHING from the global RNG — the live v6F S-W run is
    training from this file.
    """
    z2 = z.detach().float().reshape(-1, z.shape[-1])
    n, d = z2.shape
    if n < 2:
        raise ValueError(f"spectrum needs n >= 2 rows, got {n}")
    zc = z2 - z2.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(zc.double())
    eig = sv ** 2
    k = min(int(top_k), int(eig.numel()))
    share = float(eig[:k].sum() / eig.sum().clamp_min(1e-30))
    ceiling = min(n - 1, d)
    er = effective_rank(sv)
    out = {"n": int(n), "d": int(d), "top_k": k,
           "participation_ratio": participation_ratio(eig),
           "effective_rank": er,
           "top_k_share": share,
           # ---- the ceiling, stamped in the record itself ------------------
           "rank_ceiling": int(ceiling),
           "effective_rank_frac": float(er / max(ceiling, 1)),
           "rank_admissible": bool(ceiling >= O6_ADMISSIBLE_CEILING),
           "ceiling_note": f"a centred covariance from n={n} rows has rank "
                           f"<= {ceiling}; effective_rank is bounded by that, "
                           f"NOT by d={d}"}
    if int(ci_reps) > 0:
        out["effective_rank_ci95"] = _cluster_interval_er(
            zc, int(block), int(ci_reps), er, generator)
    return out


def _er_from_gram(g: Tensor) -> float:
    """``effective_rank`` of a row set given its (uncentred) Gram matrix.

    Double-centring the Gram is equivalent to centring the rows, and the
    singular values of the centred rows are the square roots of the centred
    Gram's eigenvalues — so ``d`` is paid once and every resample costs
    ``O(n^3)`` instead of ``O(n^2 d)``.
    """
    m = g.mean(0, keepdim=True)
    gc = g - m - m.T + g.mean()
    return effective_rank(torch.linalg.eigvalsh(gc).clamp_min(0).flip(0).sqrt())


def _cluster_interval_er(zc: Tensor, block: int, reps: int, theta: float,
                         generator=None) -> dict:
    """95 % interval for ``effective_rank`` with the CLUSTER as the unit.

    Primary: **leave-one-cluster-out jackknife** — no duplicated blocks, so no
    manufactured rank deficiency, and the only one of the three candidates that
    covers (see :func:`spectrum_report`). The bootstrap bounds are computed
    alongside and returned as LABELLED DIAGNOSTICS, never as the interval.
    """
    zd = zc.double()
    g_full = zd @ zd.T
    block = max(1, int(block))
    nb = zd.shape[0] // block
    if nb < 4:
        return {"status": "n/a",
                "reason": f"only {nb} blocks of {block} rows — a 4-block "
                          f"interval is noise"}
    # ---- the interval: leave-one-cluster-out jackknife ---------------------
    jk = []
    for b in range(nb):
        keep = torch.cat([torch.arange(b * block),
                          torch.arange((b + 1) * block, nb * block)])
        jk.append(_er_from_gram(g_full[keep][:, keep]))
    j = torch.tensor(jk, dtype=torch.float64)
    se = float(((nb - 1) / nb * ((j - j.mean()) ** 2).sum()).sqrt())
    out = {"lo": theta - 1.96 * se, "hi": theta + 1.96 * se,
           "kind": "leave-one-cluster-out jackknife", "se": se,
           "block_rows": block, "n_blocks": int(nb),
           "measured_coverage": "0.85 (48 rows) / 0.867 (384 rows) vs nominal "
                                "0.95 — SIGREG_GATE_POWER.md §4"}
    # ---- diagnostics only: the bootstrap that does NOT cover ---------------
    if int(reps) > 0:
        vals = []
        off = torch.arange(block)
        for _ in range(int(reps)):
            idx = torch.randint(0, nb, (nb,), generator=generator)
            rows = (idx[:, None] * block + off).reshape(-1)
            vals.append(_er_from_gram(g_full[rows][:, rows]))
        v = torch.tensor(vals, dtype=torch.float64)
        out["bootstrap_DIAGNOSTIC_do_not_quote"] = {
            "percentile_lo": float(torch.quantile(v, 0.025)),
            "percentile_hi": float(torch.quantile(v, 0.975)),
            "reps": int(reps),
            "why_not_used": "resampling blocks WITH replacement duplicates "
                            "them; duplicated rows are exactly rank-deficient, "
                            "so this is biased DOWN for a rank functional. "
                            "MEASURED coverage 0.25 / 0.00."}
    return out


class SpectrumAccumulator:
    """A bounded ring of raw latent rows, so the spectrum can be estimated from
    MANY consecutive steps instead of one batch.

    ⛔ THE PROBLEM IT SOLVES. One spectrum call sees ``B*W`` rows — on the live
    run 48, which are 8 windows × 6 CONSECUTIVE frames drawn from only 4
    episodes (``--eps-per-batch 4``). That is not 48 independent samples of the
    representation; it is ~4 scenes. The rank ceiling is 47 and the estimator's
    variance is set by the cluster count, not the row count. Pooling ``capacity``
    consecutive steps raises BOTH: rows ``capacity*48`` and distinct episodes
    ``~capacity*4``.

    ⚠️ WHY CONSECUTIVE STEPS AND NOT CONSECUTIVE SPECTRUM CALLS. Pooling 32
    calls at ``--spectrum-every 200`` would span 6 400 steps, over which the
    representation genuinely moves — the pooled spectrum would then measure the
    UNION over training and read high for the wrong reason. 32 consecutive
    steps span 32 steps (~14 min of Thor wall-clock at 26.35 s/step), where
    drift is negligible.

    Rows are stored on CPU in ``float32``: ``capacity=32`` costs
    32 × 48 × 2048 × 4 B ≈ **12.6 MB**, and the per-step cost is one detach and
    one D2H copy of 393 KB against a 26 s step.
    """

    def __init__(self, capacity: int = 32, block: int = 1):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self.block = max(1, int(block))
        self._buf: list[Tensor] = []

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def n_rows(self) -> int:
        return sum(int(t.shape[0]) for t in self._buf)

    def push(self, z: Tensor) -> None:
        """Bank one step's ``[..., d]`` latent (flattened to rows)."""
        r = z.detach().float().reshape(-1, z.shape[-1]).cpu()
        self._buf.append(r)
        if len(self._buf) > self.capacity:
            del self._buf[0]

    def clear(self) -> None:
        self._buf.clear()

    def report(self, *, top_k: int = 8, ci_reps: int = 0,
               generator=None) -> dict:
        """The pooled reading, stamped with how it was pooled."""
        if not self._buf:
            raise ValueError("SpectrumAccumulator is empty — nothing pushed")
        z = torch.cat(self._buf, dim=0)
        rep = spectrum_report(z, top_k=top_k, ci_reps=ci_reps,
                              block=self.block, generator=generator)
        rep["pooled_steps"] = len(self._buf)
        rep["pool_capacity"] = self.capacity
        rep["pool_block_rows"] = self.block
        return rep


def o6_rank_verdict(cur: dict, ref: dict | None = None, *,
                    retention: float = 0.8,
                    floor: float = O6_RANK_FLOOR,
                    ceiling_min: int = O6_ADMISSIBLE_CEILING) -> dict:
    """The RE-DERIVED O6 gate criterion — see ``SIGREG_GATE_POWER.md`` §5.

    The criterion it replaces (*"≥ 0.8× effective rank across phases"*) named no
    estimator, no ``n`` and no interval, and MEASURED at the live run's n=48 it
    fires on nothing at between **9 %** (model-based null) and **38 %** (the
    run's own banked spread) — a guard that goes off when nothing happened. Its
    power against a 1.43× true collapse was **0.11** at that false-positive
    rate, i.e. very nearly uninformative.

    The replacement has three clauses and can return INCONCLUSIVE, which the old
    one could not:

    1. **ADMISSIBILITY.** A reading whose ``rank_ceiling`` is below
       :data:`O6_ADMISSIBLE_CEILING` is INCONCLUSIVE — never PASS, never FAIL.
       A single 48-row batch is exactly this case, and saying so is the honest
       reading of the whole banked series.
    2. **RETENTION (relative), CI-based.** FAIL when the retention interval lies
       WHOLLY BELOW ``retention``; PASS when it lies wholly at or above it;
       INCONCLUSIVE when it straddles. Firing therefore requires the interval to
       exclude the threshold, so noise alone cannot trip it.
    3. **FLOOR (absolute).** FAIL when the pooled ``effective_rank`` is below
       ``floor`` regardless of retention — retention alone cannot see a
       representation that was ALREADY collapsed when the reference was taken.

    ``ref=None`` evaluates clauses 1 and 3 only — the first admissible reading
    of a phase has nothing to retain against — and returns INCONCLUSIVE saying
    so, never PASS.
    """
    ceiling = int(cur.get("rank_ceiling", min(int(cur.get("n", 2)) - 1,
                                              int(cur.get("d", 1)))))
    er = float(cur["effective_rank"])
    out: dict = {"criterion": "O6_rank_retention v2 (SIGREG_GATE_POWER.md)",
                 "effective_rank": er, "rank_ceiling": ceiling,
                 "effective_rank_frac": er / max(ceiling, 1),
                 "retention_threshold": float(retention),
                 "absolute_floor": float(floor)}
    out["ceiling_min"] = int(ceiling_min)
    if ceiling < ceiling_min:
        out |= {"pass": None, "status": "INCONCLUSIVE",
                "reason": f"rank_ceiling {ceiling} < {ceiling_min}: "
                          f"a centred covariance from n={cur.get('n')} rows "
                          f"cannot resolve rank. Pool more steps "
                          f"(SpectrumAccumulator) before asking this question."}
        return out
    if er < floor:
        out |= {"pass": False, "status": "FAIL",
                "reason": f"effective_rank {er:.3f} < the pre-registered "
                          f"absolute floor {floor} at an ADMISSIBLE ceiling "
                          f"{ceiling} — clause 3."}
        return out
    if ref is None:
        out |= {"pass": None, "status": "INCONCLUSIVE",
                "reason": "no reference reading — clause 2 (retention) needs a "
                          "phase-start reading at the SAME pooling. Clauses 1 "
                          "and 3 passed."}
        return out
    er0 = float(ref["effective_rank"])
    if er0 <= 0:
        out |= {"pass": None, "status": "INCONCLUSIVE",
                "reason": "reference effective_rank is non-positive"}
        return out
    out["reference_effective_rank"] = er0
    out["retention"] = er / er0
    ci_c, ci_r = cur.get("effective_rank_ci95"), ref.get("effective_rank_ci95")
    if not (isinstance(ci_c, dict) and "lo" in ci_c
            and isinstance(ci_r, dict) and "lo" in ci_r):
        out |= {"pass": None, "status": "INCONCLUSIVE",
                "reason": "both readings must carry effective_rank_ci95 "
                          "(spectrum_report(..., ci_reps=N)). A point ratio "
                          "with no interval is the defect this replaces."}
        return out
    # Conservative interval on the RATIO from the two marginal intervals:
    # widest possible, so it can only ever be too wide. Both bounds are
    # clamped at 0 first — a normal-approximation jackknife bound can go
    # negative when the SE is large, and a negative effective rank is not a
    # thing. Every failure mode of these clamps widens the interval, which
    # makes the verdict INCONCLUSIVE rather than firing.
    lo = max(ci_c["lo"], 0.0) / max(ci_r["hi"], 1e-12)
    hi = max(ci_c["hi"], 0.0) / max(ci_r["lo"], 1e-12)
    out["retention_ci95"] = {"lo": lo, "hi": hi,
                             "kind": "ratio of cluster-JACKKNIFE bounds "
                                     "(conservative: lo/hi and hi/lo)"}
    if hi < retention:
        out |= {"pass": False, "status": "FAIL",
                "reason": f"retention interval [{lo:.3f}, {hi:.3f}] lies wholly "
                          f"below {retention} — clause 2."}
    elif lo >= retention:
        out |= {"pass": True, "status": "PASS",
                "reason": f"retention interval [{lo:.3f}, {hi:.3f}] lies wholly "
                          f"at/above {retention}, and effective_rank {er:.3f} "
                          f">= floor {floor} at ceiling {ceiling}."}
    else:
        out |= {"pass": None, "status": "INCONCLUSIVE",
                "reason": f"retention interval [{lo:.3f}, {hi:.3f}] straddles "
                          f"{retention} — the reading cannot decide. Pool more "
                          f"steps or take more reference readings."}
    return out


# ============================================================================
# THE GOAL-TOKEN VOCABULARY — one table, two views (§5)
# ============================================================================

class GoalVocabulary(nn.Module):
    """A tokenizable goal vocabulary: ``(TYPE, [args])`` (§2).

    Holds the ONE embedding table plus the arg projection. The emitting head
    ABOVE and the consuming conditioner BELOW are both handed THIS OBJECT, so
    ``head.vocab is cond.vocab`` and the two views share parameters by
    identity, not by a copy that happens to start equal.

    ``encode(ids|probs, args) -> [B, d_embed]``: the token embedding plus a
    projection of the typed continuous slots, masked by ``arg_mask`` so an
    UNSET constraint slot contributes exactly zero (§2: "Unset =
    unconstrained"; the same IGNORE discipline as ``v4_curriculum``'s masked
    CE — a slot no label can fill must not be regressed against a fabricated 0).
    A SOFT input (``probs`` [B, n_tokens]) keeps the seam differentiable for
    the staged trainer; a HARD input (``ids`` [B] long) is the inference path.
    """

    def __init__(self, tokens: tuple[str, ...], d_embed: int = 128,
                 n_args: int = GOAL_ARG_SLOTS):
        super().__init__()
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"duplicate goal tokens in {tokens}")
        self.tokens = tuple(tokens)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.d_embed, self.n_args = int(d_embed), int(n_args)
        self.table = nn.Embedding(len(self.tokens), self.d_embed)
        nn.init.trunc_normal_(self.table.weight, std=0.02)
        self.arg_proj = nn.Linear(self.n_args, self.d_embed)
        self.norm = nn.LayerNorm(self.d_embed)
        # ⛔ THE CATEGORICAL CHANNEL IS NOT BUILT HERE, ON PURPOSE. Building it
        # in __init__ would draw RNG in the MIDDLE of V6Stack's construction
        # order and shift every module initialised after this vocabulary — so
        # merely turning the flag on would perturb pre-existing weights and the
        # live S-W resume's byte-identity argument would be gone. It is
        # attached at the END of V6Stack.__init__ instead (attach_cat_channel).
        self.cat_names: tuple[str, ...] | None = None
        self.cat_cards: tuple[int, ...] | None = None
        self.cat_emb: nn.Linear | None = None

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    # -- the categorical arg channel (§2.3 of ANCHOR_GOAL_SUPERVISION) --------
    @property
    def n_cat(self) -> int:
        """Total width of the concatenated categorical block (0 when off)."""
        return 0 if self.cat_cards is None else sum(self.cat_cards)

    def attach_cat_channel(self, cards, names=GOAL_CAT_ARG_NAMES,
                           usage: dict[str, tuple[str, ...]] | None = None
                           ) -> None:
        """⭐ Give this vocabulary a TYPED CATEGORICAL arg channel.

        ``cards`` is one cardinality per name in ``names``. The consuming side
        is ONE ``nn.Linear(sum(cards), d_embed, bias=False)`` — i.e. the
        concatenated per-slot embedding tables expressed as a matmul, exactly
        the trick :meth:`embed_tokens` already uses so a HARD one-hot and a SOFT
        posterior travel the SAME code path (a second, "soft-only" path is how
        two conventions get invented).

        ``usage`` maps token -> the slots that token actually uses; it becomes
        the ``cat_usage`` buffer, and it is what lets an UNSET slot contribute
        exactly zero (§2: "Unset = unconstrained"; the IGNORE discipline — a
        slot no label can fill must not be regressed against a fabricated 0).
        Non-persistent: it is derived from module constants, so shipping it in
        the checkpoint would be shipping a copy of the source.
        """
        if self.cat_cards is not None:
            raise RuntimeError("this vocabulary already has a categorical "
                               "channel — attaching twice would leave two "
                               "tables wearing one name (§5)")
        cards, names = tuple(int(c) for c in cards), tuple(names)
        if len(cards) != len(names):
            raise ValueError(f"{len(cards)} cardinalities for {len(names)} "
                             f"categorical slots {names}")
        if any(c < 1 for c in cards):
            raise ValueError(f"every categorical cardinality must be >= 1, "
                             f"got {cards}")
        self.cat_names, self.cat_cards = names, cards
        self.cat_emb = nn.Linear(sum(cards), self.d_embed, bias=False)
        nn.init.trunc_normal_(self.cat_emb.weight, std=0.02)
        use = torch.zeros(self.n_tokens, len(names))
        for tok, slots in (usage or GOAL_CAT_ARG_TOKENS).items():
            if tok not in self.index:
                continue                      # a token of a DIFFERENT axis
            for s in slots:
                if s not in names:
                    raise KeyError(f"{tok} needs categorical slot {s!r}, which "
                                   f"this channel does not have: {names}")
                use[self.index[tok], names.index(s)] = 1.0
        self.register_buffer("cat_usage", use, persistent=False)

    def cat_slice(self, name: str) -> slice:
        """The slice of the concatenated categorical block owned by ``name``."""
        if self.cat_cards is None:
            raise RuntimeError("this vocabulary has no categorical channel")
        i = self.cat_names.index(name)
        lo = sum(self.cat_cards[:i])
        return slice(lo, lo + self.cat_cards[i])

    def cat_mask_from_tokens(self, ids_or_probs: Tensor) -> Tensor:
        """[B, n_cat_slots] — the SOFT probability that the emitted token uses
        each categorical slot.

        Under a hard id this is the token's own usage row; under a soft
        posterior it is ``probs @ cat_usage``, which is the differentiable
        generalisation and NOT a second convention: at a one-hot posterior the
        two coincide exactly (pinned by a test).
        """
        if self.cat_cards is None:
            raise RuntimeError("this vocabulary has no categorical channel")
        u = self.cat_usage.to(self.table.weight.dtype)
        if ids_or_probs.dtype in (torch.long, torch.int32, torch.int64):
            return u[ids_or_probs.long()]
        # ⚠️ clamped because MULTI-LABEL gates are not a simplex: two emitted
        # tokens that both use `agent_slot` would otherwise give that slot a
        # mask of 2 and silently AMPLIFY its embedding. A mask says whether a
        # slot is set; it is not a weight.
        return (ids_or_probs.to(u.dtype) @ u).clamp(0.0, 1.0)

    def expand_cat_mask(self, cat_mask: Tensor) -> Tensor:
        """[B, n_cat_slots] -> [B, sum(cards)], each slot's mask broadcast over
        its own block."""
        if self.cat_cards is None:
            raise RuntimeError("this vocabulary has no categorical channel")
        if cat_mask.shape[-1] != len(self.cat_cards):
            raise ValueError(f"cat_mask must be [B, {len(self.cat_cards)}], "
                             f"got {tuple(cat_mask.shape)}")
        reps = torch.as_tensor(self.cat_cards, device=cat_mask.device)
        return torch.repeat_interleave(cat_mask, reps, dim=-1)

    def id_of(self, token: str) -> int:
        if token not in self.index:
            raise KeyError(f"{token!r} not in this vocabulary {self.tokens}")
        return self.index[token]

    def embed_tokens(self, ids_or_probs: Tensor) -> Tensor:
        """[B] long ids -> [B, d] ; or [B, n_tokens] float probs -> [B, d]
        (the differentiable soft view: ``probs @ table.weight``)."""
        if ids_or_probs.dtype in (torch.long, torch.int32, torch.int64):
            return self.table(ids_or_probs.long())
        if ids_or_probs.shape[-1] != self.n_tokens:
            raise ValueError(f"soft goal must be [B, {self.n_tokens}], got "
                             f"{tuple(ids_or_probs.shape)}")
        return ids_or_probs.to(self.table.weight.dtype) @ self.table.weight

    def encode(self, ids_or_probs: Tensor, args: Tensor | None = None,
               arg_mask: Tensor | None = None, cat: Tensor | None = None,
               cat_mask: Tensor | None = None) -> Tensor:
        e = self.embed_tokens(ids_or_probs)
        if args is not None:
            if args.shape[-1] != self.n_args:
                raise ValueError(f"args must be [B, {self.n_args}], got "
                                 f"{tuple(args.shape)}")
            a = args.to(e.dtype)
            if arg_mask is not None:
                a = a * arg_mask.to(e.dtype)
            e = e + self.arg_proj(a)
        if cat is not None:
            if self.cat_emb is None:
                raise ValueError(
                    "this vocabulary has NO categorical arg channel but was "
                    "handed one — an undeclared conditioning path is exactly "
                    "what the disjointness audit (X1) forbids. Build the stack "
                    "with goal_cat_args=True.")
            if cat.shape[-1] != self.n_cat:
                raise ValueError(f"cat must be [B, {self.n_cat}] (the "
                                 f"concatenated slots {self.cat_names} with "
                                 f"cards {self.cat_cards}), got "
                                 f"{tuple(cat.shape)}")
            c = cat.to(e.dtype)
            if cat_mask is not None:
                c = c * self.expand_cat_mask(cat_mask).to(e.dtype)
            e = e + self.cat_emb(c)
        return self.norm(e)


class GoalHead(nn.Module):
    """The goal-EMITTING head of a layer: ``z -> (logits over the vocabulary,
    typed continuous args)``.

    ⛔ ADMISSIBILITY (Sayed 2026-08-03). Inputs are ``z`` (the layer's own
    vision-derived latent) and OPTIONALLY ``cond`` (the goal embedding handed
    DOWN from the layer above — goals flow down, §5). No situation-classifier
    output in any form; no ego state; no ``**kwargs``. The check to run for any
    goal signal is *"could this have been computed from the situation
    classifier's output?"* — here, no, because no path from any classifier
    exists in this module.

    ⚠️ NOT zero-initialised on purpose. A zero output layer would make the head
    emit a constant AND make its gradient w.r.t. its own input exactly zero,
    which would silently defeat :meth:`V6Stack.assert_isolation` — an isolation
    check that passes because a path carries no gradient TODAY is not an
    isolation check. (The emission head IS zero-init, for the CV warm start;
    that is why the isolation probe backprops the DECLARED planner-side surface
    including the pre-emission feature, not just the emitted controls.)
    """

    def __init__(self, vocab: GoalVocabulary, d_in: int, *,
                 d_cond: int = 0, hidden: int = 256):
        super().__init__()
        self.vocab = vocab                          # SHARED object (§5)
        self.d_in, self.d_cond = int(d_in), int(d_cond)
        self.trunk = nn.Sequential(
            nn.Linear(self.d_in + self.d_cond, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.type_head = nn.Linear(hidden, vocab.n_tokens)
        self.arg_head = nn.Linear(hidden, vocab.n_args)
        self.hidden = int(hidden)
        # ⛔ Both extensions are attached LATE and only when asked, for the same
        # reason the vocabulary's channel is (see GoalVocabulary.__init__):
        # drawing RNG here would move every module built after this head.
        # ``multilabel`` holds NO parameters — it is a second READING of the
        # same ``type_head`` logits — so it can be switched at any time.
        self.cat_head: nn.Linear | None = None
        self.multilabel = False

    def attach_cat_head(self) -> None:
        """⭐ The EMITTING half of the categorical arg channel: one linear over
        the concatenated per-slot blocks, read back as per-slot softmaxes.

        Refuses when the vocabulary has no channel — a head that emitted
        categorical logits into a conditioner that cannot read them would be a
        capability on paper only, which is the exact shape of the gap this
        closes (§2.3: 2 of 9 tokens expressible).
        """
        if self.vocab.cat_cards is None:
            raise RuntimeError("attach the vocabulary's categorical channel "
                               "first — an emitter with no consumer is not a "
                               "channel")
        if self.cat_head is not None:
            raise RuntimeError("categorical head already attached")
        self.cat_head = nn.Linear(self.hidden, self.vocab.n_cat)

    def enable_multilabel(self, on: bool = True) -> None:
        """⭐ MULTI-LABEL EMISSION. A single 9-way softmax can emit exactly ONE
        `g_tac` token per step, and §6.4's measurement says the tactical goal
        must be **`ANCHOR_GOAL` (lateral) AND `SPEED_BAND` (longitudinal)** at
        the same time — a PAIR, which a simplex cannot represent.

        ``gates = logits.sigmoid()`` are INDEPENDENT per-token, so a set is
        expressible; ``probs`` stays in the output unchanged so the
        single-choice reading and every existing consumer are untouched. Holds
        no parameters: it is the same ``type_head``, read a second way.

        ⚠️ The FACTORED heads are the STRUCTURED form of the same fix and are
        strictly better attributable — one token per AXIS, so the pair is
        emitted by construction and each half is separately scoreable. This
        flag is the unstructured fallback for the un-factored head.
        """
        self.multilabel = bool(on)

    def forward(self, z: Tensor, cond: Tensor | None = None) -> dict:
        if z.ndim != 2 or z.shape[-1] != self.d_in:
            raise ValueError(f"z must be [B, {self.d_in}], got "
                             f"{tuple(z.shape)}")
        if self.d_cond:
            if cond is None or cond.shape[-1] != self.d_cond:
                raise ValueError(f"this head needs cond [B, {self.d_cond}]")
            h = self.trunk(torch.cat([z, cond.to(z.dtype)], dim=-1))
        else:
            if cond is not None:
                raise ValueError("head built with d_cond=0 was handed a cond "
                                 "— an undeclared conditioning path is exactly "
                                 "what the disjointness audit (X1) forbids")
            h = self.trunk(z)
        logits = self.type_head(h)
        out = {"logits": logits, "args": self.arg_head(h),
               "probs": logits.softmax(dim=-1)}
        if self.multilabel:
            out["gates"] = logits.sigmoid()
        if self.cat_head is not None:
            cl = self.cat_head(h)
            out["cat_logits"] = cl
            # per-slot softmax: each typed slot is its OWN categorical variable,
            # so ONE softmax over the concatenation would make picking an
            # `anchor_id` compete with picking a `reason` — a different (and
            # wrong) model.
            out["cat_probs"] = torch.cat(
                [cl[..., self.vocab.cat_slice(n)].softmax(dim=-1)
                 for n in self.vocab.cat_names], dim=-1)
        return out


class GoalConditioner(nn.Module):
    """The goal-CONSUMING side of the seam, one layer BELOW the emitter.

    Holds the SAME :class:`GoalVocabulary` object as the head above (§5: "Token
    embeddings are shared between the goal-emitting head (above) and the
    goal-consuming conditioner (below) — one vocabulary, two views") and turns
    a goal into the conditioning vector the lower predictor consumes.
    """

    def __init__(self, vocab: GoalVocabulary, d_out: int | None = None):
        super().__init__()
        self.vocab = vocab                          # SHARED object (§5)
        d_out = int(d_out or vocab.d_embed)
        self.proj = (nn.Identity() if d_out == vocab.d_embed
                     else nn.Linear(vocab.d_embed, d_out))
        self.d_out = d_out

    def forward(self, ids_or_probs: Tensor, args: Tensor | None = None,
                arg_mask: Tensor | None = None, cat: Tensor | None = None,
                cat_mask: Tensor | None = None) -> Tensor:
        return self.proj(self.vocab.encode(ids_or_probs, args, arg_mask,
                                           cat, cat_mask))


class GoalDistanceScorer(nn.Module):
    """SELECTION over the fan, by distance to a PREDICTED goal point (+267).

    ``score_i = − ‖endpoint_i − ĝ‖ / tau + b_i`` with ``ĝ = W · e_g_tac + c``.

    ⭐ WHY THIS SHAPE AND NOT A LEARNED COST OVER THE FAN. MEASURED 2026-08-15
    on the banked in-repo REF-C-XL fan (881 windows / 40 episodes / 256
    candidates, `sel_winners_curse_law.py`, paired episode-cluster bootstrap):

      * a world-model ROLL-CONSISTENCY score — the W7 quantity, present in that
        dump as ``cons_score`` — selects at **6.4501 m** against a fan oracle of
        **0.1639 m**, is **+5.9787 [+5.3217, +6.7625] WORSE** than the shipped
        supervised selector, and its normalised error-rank **RISES** with the
        candidate count (0.241 at N=4 → 0.286 at N=256) while its lower-tail hit
        rate **COLLAPSES** (0.57 → 0.28). That is the winner's curse, replicated
        independently of W7 on a different model and a different grid.
      * selection by distance to a goal point behaves the OPPOSITE way: its
        normalised error-rank FALLS with N (0.006 → 0.001 at sigma 0) and its
        lower-tail hit rate is **1.00**, because a candidate-INDEPENDENT
        reference has no degenerate minimiser — inaction cannot minimise it.
      * the requirement curve is measured, not assumed: at **sigma 0.5 m** the
        goal rule is **−0.1591 [−0.2300, −0.0894] BETTER** than the trained
        selector (separated); at **sigma 1.0 m** it is **+0.0943 [+0.0241,
        +0.1650] WORSE** (separated). ⇒ **the goal head must reach ≈0.8 m
        1-sigma endpoint accuracy to be worth having**, and that is a gate, not
        a hope.

    ⛔ THE HONESTY HAZARD, STATED RATHER THAN BURIED. A goal head trained to
    regress the future endpoint IS a trajectory predictor; selecting on it would
    move the planning into the goal head and call the result a hierarchy. Two
    things bound that, both structural:
      1. **The goal is 2 numbers**, decoded from the goal embedding only — an
         information bottleneck against a 60x2 plan. A "goal" wide enough to
         carry the path is not a goal, and this module cannot emit one.
      2. **The goal-echo control** (replace ĝ by its corpus marginal) must be
         reported beside every number. MEASURED on the same fan: the echo
         selects at **7.8237 m** against the live goal's 0.7862 m, so the null
         is far away and a win cannot be an artefact of "any goal point works".

    ⛔ ADMISSIBILITY (PI 2026-08-03). The only input is ``e_g_tac``, which comes
    from ``goal_head_tac(z_tac_p, cond=e_g_str)`` — vision-derived latents and
    the strategic goal. No situation-classifier output in any form, no ego state
    at inference. The check to run is *"could this have been computed from the
    situation classifier's output?"* — no such path exists into this module.
    """

    def __init__(self, d_goal_embed: int, n_candidates: int, *,
                 tau_m: float = 1.0):
        super().__init__()
        # ĝ: the predicted goal POINT (x, y) in the ego frame at the horizon.
        self.goal_point = nn.Linear(int(d_goal_embed), 2)
        nn.init.zeros_(self.goal_point.weight)
        nn.init.zeros_(self.goal_point.bias)
        # per-candidate prior. Zero-init so the head starts as a pure
        # goal-distance rule and any learned per-candidate bias is visible as a
        # DEPARTURE from that, not as the thing being measured.
        self.cand_bias = nn.Parameter(torch.zeros(int(n_candidates)))
        self.log_tau = nn.Parameter(torch.tensor(float(math.log(tau_m))))

    def forward(self, waypoints: Tensor, g_embed: Tensor,
                goal_point: Tensor | None = None) -> dict:
        """``waypoints`` [B, N, T, 2] · ``g_embed`` [B, d_goal_embed] ->
        ``{"score" [B, N], "goal_point" [B, 2], "goal_dist" [B, N]}``.

        Higher score == better, so ``score.argmax(-1)`` is the incumbent rule
        and any noise-robust aggregator is a strictly later decision.

        ``goal_point`` [B, 2] OVERRIDES the internal free decode — this is the
        seam :class:`AnchorGoalHead` plugs into, so a STRUCTURED goal (snapped
        laterally, continuous longitudinally) can be scored by the identical
        rule. Default ``None`` keeps the incumbent behaviour bit-for-bit.

        ⭐ When it IS overridden the free decode is still computed and returned
        as ``goal_point_free``. Two reasons, both load-bearing: (1) a parameter
        that no declared output reaches is invisible to the X3 probe — the
        ``intent_proj`` defect, where a path present in the diagram was absent
        from the optimisation; (2) it hands E-AG2's paired comparison — the FREE
        decode against the STRUCTURED one — on the same window in one forward,
        at zero extra parameters. It is zero-init, so an unfitted free decode
        reads as exactly that rather than as noise.
        """
        if waypoints.ndim != 4 or waypoints.shape[-1] != 2:
            raise ValueError(f"waypoints must be [B, N, T, 2], got "
                             f"{tuple(waypoints.shape)}")
        free = self.goal_point(g_embed.to(waypoints.dtype))       # [B, 2]
        extra = {}
        if goal_point is not None:
            if goal_point.shape != (waypoints.shape[0], 2):
                raise ValueError(f"goal_point must be [B, 2], got "
                                 f"{tuple(goal_point.shape)}")
            g, extra = goal_point.to(waypoints.dtype), {"goal_point_free": free}
        else:
            g = free
        d = (waypoints[:, :, -1] - g[:, None]).norm(dim=-1)       # [B, N]
        tau = self.log_tau.exp().clamp_min(1e-3).to(d.dtype)
        return {"score": -d / tau + self.cand_bias.to(d.dtype)[None],
                "goal_point": g, "goal_dist": d} | extra


class MLPCandidateScorer(nn.Module):
    """⭐ THE CAPACITY CONTROL for :class:`GoalDistanceScorer`. Same inputs,
    far more parameters, and **no distance prior whatsoever**.

    ⛔ WHY THIS MODULE HAS TO EXIST BEFORE A ``"goal"`` ARM IS JUDGED.
    ``GoalDistanceScorer`` wins with **+267** parameters and a hard-wired
    ``−‖endpoint − ĝ‖`` rule. If that arm beats the incumbent, exactly two
    stories fit the observation:

      1. **MECHANISM** — a candidate-INDEPENDENT reference has no degenerate
         minimiser, which is why its normalised error-rank FALLS with N while
         the roll-consistency score's RISES (0.241 → 0.286 at N=256);
      2. **CAPACITY** — the selector head was simply underpowered, and any
         extra parameters on the same inputs would have done as well.

    A ``"goal"``-only experiment cannot separate them, and reading (2) as (1)
    is the **C6 confound** verbatim — a decoder compared on its marginal —
    which this programme has already been burned by once. §5.3 of
    ``V6F_PLANNER_DESIGN.md`` pre-registers the refutation: *if ``"mlp"``
    matches or beats ``"goal"``, SEL-1's story is wrong.*

    ⚠️ **INFORMATION-MATCHED ON PURPOSE, NOT INFORMATION-ENRICHED.** The inputs
    are exactly what the goal rule reads — the candidate ENDPOINT
    (``waypoints[:, :, -1]``) and ``e_g_tac`` — and nothing else. Handing the
    control the full 60x2 path would make it a *different experiment* (an
    information control), and its result would no longer speak to capacity.

    ⚠️ **THE CONTROL IS DELIBERATELY GENEROUS.** It gets ~127x the goal rule's
    parameters. That is the conservative direction for the conclusion we most
    want to avoid over-claiming: if a control this large still loses, "capacity"
    is a weak explanation; if it wins, SEL-1 is refuted and we want to know.

    ⛔ ADMISSIBILITY (PI 2026-08-03) is inherited unchanged: the only inputs are
    the emitted trajectory and ``e_g_tac``, which comes from
    ``goal_head_tac(z_tac_p, cond=e_g_str)``. No situation-classifier output in
    any form, no ego state at inference.

    ⚠️ It emits **no** ``goal_point`` / ``goal_dist``. It has no goal point, and
    a zero-filled field would be a fabricated number that later reads as a
    measurement. ``mechanism="mlp"`` is emitted instead so a dump is
    self-identifying.
    """

    def __init__(self, d_goal_embed: int, n_candidates: int, *,
                 hidden: int = 256):
        super().__init__()
        self.d_in = 2 + int(d_goal_embed)
        self.fc1 = nn.Linear(self.d_in, int(hidden))
        self.fc2 = nn.Linear(int(hidden), 1)
        # Zero-init the OUTPUT layer, mirroring GoalDistanceScorer's zero-init
        # discipline: the control starts as a flat score over the fan, so any
        # ranking it acquires is visible as something it LEARNED rather than
        # something its initialisation handed it.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.cand_bias = nn.Parameter(torch.zeros(int(n_candidates)))

    def forward(self, waypoints: Tensor, g_embed: Tensor) -> dict:
        """``waypoints`` [B, N, T, 2] · ``g_embed`` [B, d_goal_embed] ->
        ``{"score" [B, N], "mechanism": "mlp"}``. Higher score == better, the
        same convention as :class:`GoalDistanceScorer`, so the two are drop-in
        comparable and ``sel_*`` logging is unchanged."""
        if waypoints.ndim != 4 or waypoints.shape[-1] != 2:
            raise ValueError(f"waypoints must be [B, N, T, 2], got "
                             f"{tuple(waypoints.shape)}")
        end = waypoints[:, :, -1]                                  # [B, N, 2]
        b, n = end.shape[0], end.shape[1]
        g = g_embed.to(end.dtype)[:, None].expand(b, n, -1)
        h = torch.nn.functional.gelu(self.fc1(torch.cat([end, g], dim=-1)))
        return {"score": self.fc2(h).squeeze(-1) + self.cand_bias.to(h.dtype),
                "mechanism": "mlp"}


class AnchorGoalHead(nn.Module):
    """⭐ ``ANCHOR_GOAL``, emitted as REGRESS-THEN-SNAP — with the one-hot K-way
    classifier kept reachable as the CONTROL.

    ⛔ WHY THIS SHAPE, MEASURED — the failure is the ESTIMATOR, not the
    ESTIMAND (`…/2026-08-16-anchor-goal-supervision/` §6.2, 881 windows /
    40 episodes, LOEO, paired episode-cluster bootstrap):

      * ``snap`` — the SAME ridge prediction, rounded to the nearest anchor —
        is **NOT separated** from the free ridge: Δ **−0.0002 [−0.1031,
        +0.0703]** at K=256, and **+0.0383 [−0.2125, +0.2338]** even in the
        much stronger ``v0`` regime. ⇒ **quantising onto the shipped vocabulary
        is FREE.**
      * a K-way one-hot classifier on the same features costs **+4.7502
        [+3.0514, +6.3981] WORSE**, separated at every K from 8 to 256 and
        under BOTH vocabulary constructions (FPS and k-means), and it
        replicates on REF-C-base (**+5.4570 [+3.8345, +7.1073]**).
      * and that failure mode was already measured on an INDEPENDENT surface:
        a one-hot target is metric-BLIND — it scores "picked the adjacent
        anchor" and "picked one 40 m away" identically — and **E-OBJ-1**
        measured swapping one-hot CE for ``softade`` recovering **−0.0974 m
        (base) / −0.1670 m (XL), separated**, with the recovery LONGITUDINAL.

    ⇒ ``"snap_lat"`` / ``"snap_xy"`` are the DEFAULT and ``"onehot"`` is the
    pre-registered CONTROL. A comparison with no control is unattributable
    (the C6 confound), which is why the refuted arm stays buildable.

    ⭐ ``"snap_lat"`` IS THE FACTORED FORM, and it is the mode the measurement
    actually prescribes. The regression is 2-D; only the **LATERAL** coordinate
    is quantised, onto a lateral sub-vocabulary; the **LONGITUDINAL** coordinate
    stays a continuous progress arg. §6.4: the goal's variance is 98.8 %
    longitudinal while the FPS quantisation is isotropic (0.5674 / 0.5599), so
    a joint K-way index spends half its resolution on the axis carrying 1.2 %
    of the variance — and the classifier's own residual is 1.96x the floor
    laterally against **14.9x** longitudinally.

    ⚠️ THE SNAP IS STRAIGHT-THROUGH, and that is what makes "regress-then-snap"
    a trainable object rather than a post-hoc rounding: the emitted point is
    quantised while the gradient reaches the CONTINUOUS regression, so the loss
    can stay metric-aware (an endpoint distance) instead of becoming the
    metric-blind CE the measurement refuses.

    ⛔ THE 6 s BLOCKER, ENFORCED RATHER THAN DOCUMENTED. Every anchor vocabulary
    the programme owns stops at **step 20 = 2.0 s** (MEASURED 2026-08-16 over
    all five banked tables: `refc_anchors_full_REBUILD.pt` and
    `refc_anchors_small64.pt` horizons ``[5,10,15,20]``; `anchors_dev256.pt`
    and `flagship_v4_anchors_dense.pt` ``[1..20]``), while ``PLAN_STEPS`` is 60
    and the v6f selector scores the **6 s** endpoint. Scoring a 6 s ground
    truth against a 2 s anchor would produce a NUMBER rather than an error, so
    :meth:`load_anchor_table` **REFUSES** any table whose requested step is not
    the plan horizon — the same refusal
    ``e_ag1_anchor_floor.shipped_vocab_arm`` makes, and the same one
    ``tanitad.data.anchor_goal.anchor_endpoints`` makes on the label side.
    ⇒ **this head cannot be run at 6 s today, and it says so instead of
    inventing a number.**

    ⛔ ADMISSIBILITY (PI 2026-08-03) is inherited unchanged from
    :class:`GoalDistanceScorer`: the only input is ``e_g_tac``, which comes from
    ``goal_head_tac(z_tac_p, cond=e_g_str)`` — vision-derived latents and the
    strategic goal. No situation-classifier output in any form, no ego state at
    inference. The anchor table is a FROZEN buffer, identical for every window,
    so it carries zero per-window information by construction — the quantity
    the goal-echo control (goal <- the vocabulary's centroid, MEASURED
    13.5553 m against live arms at 0.79–9.49) exists to bound.
    ⚠️ ``v0`` is NOT read here and nothing here depends on ``v0`` being
    admissible — that is an OPEN PI DECISION (`V6F_PLANNER_DESIGN.md` §1.4 vs
    `e_wc2_sigma_star.py:188`), worth a MEASURED 2.85x on this very quantity.
    """

    MODES: tuple[str, ...] = ("snap_lat", "snap_xy", "onehot")

    def __init__(self, d_goal_embed: int, n_anchors: int, *,
                 mode: str = "snap_lat", plan_horizon_s: float = HORIZON_S,
                 n_lat_bins: int = 16):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f"anchor_goal mode must be one of {self.MODES}, got {mode!r}. "
                f"'snap_lat' is the FACTORED default the measurement "
                f"prescribes; 'onehot' is the metric-blind CONTROL that "
                f"E-AG2 measured +4.7502 [+3.0514, +6.3981] WORSE.")
        if int(n_anchors) < 2:
            raise ValueError(f"n_anchors must be >= 2, got {n_anchors}")
        if int(n_lat_bins) < 2:
            raise ValueError(f"n_lat_bins must be >= 2, got {n_lat_bins}")
        self.mode = mode
        self.n_anchors, self.n_lat_bins = int(n_anchors), int(n_lat_bins)
        self.plan_horizon_s = float(plan_horizon_s)
        # the free regression — the metric-aware half. Zero-init on the same
        # discipline as GoalDistanceScorer.goal_point: the head starts at the
        # origin so any goal it acquires is LEARNED, not handed to it by init.
        # ⚠️ Built ONLY in the snap modes, and the classifier ONLY in "onehot":
        # the two arms are DIFFERENT ESTIMATORS and giving each the other's
        # parameters would leave dead weight at random init in both — the
        # `intent_proj` defect. They are matched where it matters (identical
        # input, ``e_g_tac``, and nothing else), which is what makes the
        # comparison a comparison.
        self.goal_point = (nn.Linear(int(d_goal_embed), 2)
                           if mode != "onehot" else None)
        if self.goal_point is not None:
            nn.init.zeros_(self.goal_point.weight)
            nn.init.zeros_(self.goal_point.bias)
        self.cls = (nn.Linear(int(d_goal_embed), self.n_anchors)
                    if mode == "onehot" else None)
        # FROZEN GEOMETRY. Persistent: `anchor_id` is meaningless without the
        # exact table, and a rebuilt table silently re-labels the whole corpus
        # (§1 field 3) — so the table SHIPS WITH THE CHECKPOINT.
        self.register_buffer("anchors", torch.zeros(self.n_anchors, 2))
        self.register_buffer("lat_bins", torch.zeros(self.n_lat_bins))
        self.register_buffer("table_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("table_horizon_s", torch.zeros(()))

    # ---- the table ---------------------------------------------------------
    @torch.no_grad()
    def load_anchor_table(self, anchors: Tensor, horizons=None,
                          dt: float = DT, *, step: int | None = None) -> dict:
        """Install the frozen anchor endpoints and derive the lateral bins.

        ``anchors`` is either ``[K, S, 2]`` **with** its ``horizons`` (the
        shipped `build_refc_anchors.py` format) or a pre-reduced ``[K, 2]``
        endpoint table, in which case ``horizons`` must be ``None`` and the
        caller is asserting the endpoints are already at the plan horizon.

        ⛔ REFUSES a horizon mismatch. Returns the provenance dict a report can
        quote.
        """
        a = torch.as_tensor(anchors).float()
        if a.ndim == 3:
            if horizons is None:
                raise ValueError("a [K, S, 2] table needs its `horizons` — "
                                 "`anchors[:, -1]` silently means '2.0 s' for "
                                 "both a 4-point and a 20-point vocabulary")
            want = int(round(self.plan_horizon_s / float(dt))) \
                if step is None else int(step)
            from tanitad.data.anchor_goal import anchor_endpoints  # lazy: the
            # tanitad.data package init pulls the corpus adapters, and this
            # module is imported by every trainer. Same precedent as
            # `_ensure_scripts`. Imported rather than re-implemented so the
            # horizon refusal has ONE definition in the programme.
            ends = anchor_endpoints(a, horizons, want, dt=float(dt))
            got_h = want * float(dt)
        elif a.ndim == 2 and a.shape[-1] == 2:
            if horizons is not None:
                raise ValueError("a [K, 2] endpoint table takes no `horizons`")
            ends, got_h = a, self.plan_horizon_s
        else:
            raise ValueError(f"anchors must be [K, S, 2] or [K, 2], got "
                             f"{tuple(a.shape)}")
        if abs(got_h - self.plan_horizon_s) > 1e-6:
            raise ValueError(
                f"REFUSING an anchor table at {got_h} s against a "
                f"{self.plan_horizon_s} s plan horizon. Every anchor "
                f"vocabulary the programme owns stops at 2.0 s (MEASURED over "
                f"all five banked tables) while the v6f selector scores the 6 s "
                f"endpoint — scoring one against the other would produce a "
                f"number rather than an error.")
        if ends.shape[0] != self.n_anchors:
            raise ValueError(f"this head was built for K={self.n_anchors}, the "
                             f"table has {ends.shape[0]}")
        self.anchors.copy_(ends)
        # ⚠️ EVIDENCE CLASS: ESTIMATED — a DECLARED construction, not a measured
        # optimum. The lateral bins are EQUAL-MASS quantiles of the anchor
        # table's own lateral marginal, i.e. resolution allocated by the
        # vocabulary's lateral density. §7.1 pre-registers E-AG4, which is the
        # experiment that would settle whether an anisotropic construction beats
        # the isotropic FPS one; until it runs, this is a construction and is
        # labelled as one.
        q = torch.linspace(0.0, 1.0, self.n_lat_bins + 1)[:-1] \
            + 0.5 / self.n_lat_bins
        self.lat_bins.copy_(torch.quantile(ends[:, 1], q.to(ends.dtype)))
        self.table_ready.fill_(True)
        self.table_horizon_s.fill_(float(got_h))
        return {"n_anchors": int(ends.shape[0]),
                "horizon_s": float(got_h), "n_lat_bins": self.n_lat_bins,
                "lat_bin_centres": [float(x) for x in self.lat_bins]}

    # ---- emission ----------------------------------------------------------
    @staticmethod
    def _straight_through(raw: Tensor, snapped: Tensor) -> Tensor:
        """``snapped`` forward, ``raw``'s gradient backward — the whole point of
        regress-then-snap: the estimand is quantised, the ESTIMATOR stays a
        metric-aware regression (the half E-AG2 exonerated)."""
        return raw + (snapped - raw).detach()

    def forward(self, g_embed: Tensor) -> dict:
        """``g_embed`` [B, d_goal_embed] -> the emitted ``ANCHOR_GOAL``.

        Always: ``goal_point`` [B, 2] (the EMITTED, possibly quantised goal) and
        ``goal_point_raw`` [B, 2] (the free regression, so the quantisation cost
        is readable rather than inferred).
        ``snap_lat`` adds ``lat_bin`` [B]; ``snap_xy`` and ``onehot`` add
        ``anchor_id`` [B]; ``onehot`` adds ``cls_logits`` [B, K].
        """
        if g_embed.ndim != 2:
            raise ValueError(f"g_embed must be [B, d], got "
                             f"{tuple(g_embed.shape)}")
        if not bool(self.table_ready):
            raise RuntimeError(
                "no anchor table is loaded — call load_anchor_table(). A "
                "zero table would snap every goal to the origin and still "
                "return a number, which is the failure mode this refusal "
                "exists to make impossible.")
        if self.mode == "onehot":
            # ⛔ THE CONTROL. Its target is a one-hot `anchor_id` and its
            # emitted point is a HARD table lookup — metric-blind by
            # construction, which is the property being controlled for. No
            # straight-through path is offered: adding one would quietly turn
            # the control into a third arm.
            logits = self.cls(g_embed.to(self.anchors.dtype))
            k = logits.argmax(dim=-1)
            return {"mode": self.mode, "cls_logits": logits, "anchor_id": k,
                    "goal_point": self.anchors[k]}
        raw = self.goal_point(g_embed.to(self.anchors.dtype))        # [B, 2]
        out = {"goal_point_raw": raw, "mode": self.mode}
        if self.mode == "snap_xy":
            d = (raw[:, None, :] - self.anchors[None]).norm(dim=-1)  # [B, K]
            k = d.argmin(dim=-1)
            out |= {"anchor_id": k, "quant_err_m": d.gather(1, k[:, None])[:, 0],
                    "goal_point": self._straight_through(raw, self.anchors[k])}
            return out
        # ⭐ snap_lat — the FACTORED default: the LATERAL axis is the only one
        # quantised; the LONGITUDINAL axis stays a continuous progress arg.
        j = (raw[:, 1:2] - self.lat_bins[None]).abs().argmin(dim=-1)  # [B]
        y_q = self.lat_bins[j]
        out |= {"lat_bin": j,
                "quant_err_m": (raw[:, 1] - y_q).abs(),
                "goal_point": torch.stack(
                    [raw[:, 0], self._straight_through(raw[:, 1], y_q)],
                    dim=-1)}
        return out


# ============================================================================
# X3 — the gradient-isolation matrix, as a config + a MEASURED check
# ============================================================================

class IsolationViolation(RuntimeError):
    """Raised by :meth:`V6Stack.assert_isolation` when a forbidden gradient
    edge of the X3 matrix is actually LIVE in the autograd graph."""


#: X3, written out. ``row -> may backprop into``. Anything absent is FORBIDDEN.
#: The matrix is the spec; :meth:`V6Stack.assert_isolation` is the measurement.
ISOLATION_MATRIX: dict[str, tuple[str, ...]] = {
    # WM-side: the operative predictor is the physical substrate and DOES train
    # the encoder — that is the S-W stage's entire purpose.
    "predictor_op": ("encoder", "readout", "predictor_op"),
    "aux": ("encoder", "readout", "aux"),
    # Higher layers reach lower latents ONLY through stop-grad / EMA-slow.
    "layer_tac": ("layer_tac",),
    "layer_str": ("layer_str",),
    # Planner/goal heads NEVER into any encoder.
    "planner": ("planner",),
}


@dataclass
class V6Config:
    """Every v6 knob that changes the model, in one serialisable place.

    Defaults are the CATALOG settings (``V6_TRAINING_MEASURES.md`` §0–§5 and
    ``HIERARCHY_VOCABULARY.md`` §4b), not preferences. Two of them are
    pre-registered ARMS rather than choices — ``shared_encoder`` (E-ENC) and
    ``uplink`` — and the trainer records which arm ran.
    """

    # ---- per-layer widths ---------------------------------------------------
    d_tac: int = 512               # §3.1 of the redesign: information decreases up
    d_str: int = 256
    # ``d_op`` is DERIVED from the readout (grid × grid_w × d_readout) — the
    # geometry firewall. It is never set independently; see :meth:`d_op`.

    # ---- per-layer clocks (Hz) ---------------------------------------------
    hz_op: float = 10.0            # operative — the 10 Hz control tick
    hz_tac: float = 2.0            # tactical  — ~2 Hz (manoeuvre scale)
    hz_str: float = 0.5            # strategic — ~0.5 Hz (route scale)

    # ---- E-ENC (the pre-registered arm, §0 Q1) -----------------------------
    #: True  = (a) ONE common encoder + per-layer adapters — the frontier
    #:         pattern (V-JEPA2 / DINO-WM / Drive-JEPA all use ONE encoder).
    #: False = (b) per-layer encoders, each seeing the input at its own
    #:         resolution/rate. Decision metric: per-layer P-battery pass rate
    #:         at MATCHED TOTAL PARAMS; a tie goes to the common encoder on the
    #:         parameter budget. Separate encoders must EARN their params.
    shared_encoder: bool = True
    #: ⭐ ViT-5 recipe encoder (PI 2026-08-13). RMSNorm + LayerScale + QK-Norm +
    #: register tokens + joint APE/2D-axial-RoPE, GeLU MLP (ViT-5 REJECTS
    #: SwiGLU: it over-gates against LayerScale and compact models suffer most).
    #: PUBLISHED: arXiv 2602.08071, 84.2 % IN-1k vs DeiT-III-B 83.8 %.
    #: ⚠️ Default False so every banked v6 number stays reproducible; turning it
    #: on CHANGES THE PARAMETER COUNT and is therefore a declared arm, not a
    #: silent upgrade.
    vit5_encoder: bool = False
    n_registers: int = 4
    adapter_hidden: int = 512

    # ---- §4b horizon spec (BINDING) ----------------------------------------
    plan_steps: int = PLAN_STEPS   # 60 steps @ 10 Hz = 6.0 s, ONE rollout
    dt: float = DT
    op_band_s: tuple[float, float] = OP_BAND_S
    tac_band_s: tuple[float, float] = TAC_BAND_S
    n_candidates: int = 8          # fan size; roll selection rolls to 6 s
    a_max: float = 4.0             # W4 census bound — feasible by construction
    kappa_max: float = 0.2
    emission_hidden: int = 256
    #: Bounding function for the emission's (a, kappa). ``"squash"`` is
    #: :func:`tanitad.models.kinematic._squash` — identity inside the range, C1
    #: rational tail, gradient alive at 100x the limit. ``"tanh"`` is the LEGACY
    #: v5.8f/W4 form, kept only for bit-exact reproduction of banked v5.8f rows:
    #: MEASURED 2026-08-15 float32, ``d/draw tanh(raw)`` is EXACTLY 0.0 from
    #: ``raw >= 10`` (the kinematic docstring's ``tanh(51)`` example understates
    #: the cliff 5x), and this run's own S-W history logged a gnorm-354,076 spike
    #: — the regime that pushes a pre-activation there and kills the head.
    #: Free to set here: ``emission.`` is in the ``planner`` group, which S-W does
    #: not train, and an activation holds no parameters, so no state_dict key or
    #: shape moves and a strict resume is unaffected.
    emission_squash: str = "squash"
    d_plan_feat: int = 256         # z_op -> emission feature projection

    # ---- SELECTION (V6F_PLANNER_DESIGN.md) ---------------------------------
    #: ⛔ DEFAULT OFF. ``"none"`` builds NO scorer and leaves the state_dict
    #: byte-identical to the pre-selector v6 (proved in
    #: ``tests/test_v6_selector.py::test_all_off_is_byte_identical_to_head``).
    #: ``"goal"`` builds :class:`GoalDistanceScorer` — **+267 parameters** — the
    #: mechanism E-WC MEASURED: a goal point of 1-sigma accuracy 0.5 m beats a
    #: trained selector by −0.1591 m [−0.2300, −0.0894] paired-separated on the
    #: banked REF-C-XL fan, while the world-model roll-consistency cost the W7
    #: line used is +5.9787 m [+5.3217, +6.7625] WORSE than that same selector.
    #: ⚠️ NEVER judge a ``"goal"`` arm without the ``"mlp"`` capacity control
    #: (V6F_PLANNER_DESIGN §5) — an arm that wins on capacity read as winning on
    #: mechanism is the C6 confound.
    #: ``"mlp"`` builds :class:`MLPCandidateScorer` — the CAPACITY CONTROL. It
    #: reads exactly the same inputs (candidate endpoint + ``e_g_tac``) with no
    #: distance prior, so a ``"goal"`` win that is really a capacity win shows
    #: up as ``"mlp"`` matching it (``V6F_PLANNER_DESIGN.md`` §5.3).
    selector: str = "none"
    #: Selection temperature (metres) for the goal-distance logit. Only read
    #: when ``selector != "none"``.
    selector_tau_m: float = 1.0
    #: Hidden width of the ``"mlp"`` capacity control. Only read when
    #: ``selector == "mlp"``; holds no parameters otherwise.
    selector_mlp_hidden: int = 256
    #: epsilon-RELAXED winner-take-all for the plan loss. ⛔ DEFAULT 0.0 = the
    #: incumbent PURE WTA, bit-identical. MEASURED consequence of pure WTA: the
    #: N−1 losing candidates receive EXACTLY ZERO gradient, so nothing bounds
    #: the fan's MEAN. The banked REF-C-XL fan shows what that regime looks like
    #: — oracle 0.1639 m against a fan MEAN of 13.9564 m — and a fan whose mean
    #: is 85x its oracle is the regime in which every selector's argmin fails.
    #: Consumed by ``scripts/train_v6_staged.py``; holds no parameters.
    plan_wta_eps: float = 0.0

    # ---- GOAL-HEAD STRUCTURE (2026-08-16, E-AG1/E-AG2 §6.4) ----------------
    # ⛔ ALL FOUR DEFAULT OFF, and the default build's state_dict is proved
    # BYTE-IDENTICAL to the pre-change one, per tensor with ``torch.equal``,
    # in ``tests/test_v6_factored_goal.py``. v6F S-W is training from a
    # checkpoint of exactly this architecture; a broken strict resume kills it.
    #: ⭐ FACTOR `g_tac` LAT x LON, the same factoring `a_tac` already carries.
    #: Builds ``goal_head_tac_lat`` / ``goal_head_tac_lon`` over the two
    #: partitioned vocabularies and composes ``e_g_tac`` from the PAIR — so
    #: `ANCHOR_GOAL` (lateral) AND `SPEED_BAND` (longitudinal) are emitted
    #: together by construction, which one 9-way softmax cannot do.
    #: ⚠️ The MIXED ``goal_head_tac`` is KEPT and still emitted: it is the
    #: CONTROL for the factored arm, and removing it would break the live
    #: resume. Only the DOWNLINK moves to the factored pair.
    goal_factored: bool = False
    #: Independent per-token gates on the UN-factored head (§2.3's second
    #: representational limit). Holds no parameters — the same ``type_head``
    #: logits read through a sigmoid instead of a softmax.
    goal_multilabel: bool = False
    #: ⭐ The TYPED CATEGORICAL arg channel. Without it 7 of the 9 `g_tac`
    #: tokens are inexpressible even given perfect labels, because their args
    #: are indices and both ends of the implemented path are PHYSICAL UNITS.
    goal_cat_args: bool = False
    #: ⭐ ``ANCHOR_GOAL`` emission. ``"none"`` builds nothing.
    #: ``"snap_lat"`` = REGRESS-THEN-SNAP, factored — quantise the LATERAL axis
    #: only, keep the LONGITUDINAL continuous (the §6.4 prescription).
    #: ``"snap_xy"`` = regress then snap to the nearest 2-D anchor (MEASURED
    #: FREE: Δ −0.0002 [−0.1031, +0.0703] vs the free ridge).
    #: ⛔ ``"onehot"`` = the K-way one-hot CLASSIFIER — the metric-blind CONTROL
    #: E-AG2 measured **+4.7502 [+3.0514, +6.3981] WORSE**, kept reachable
    #: because a comparison with no control is unattributable (C6).
    #: ⚠️ It REFUSES to run until an anchor table at the PLAN HORIZON is
    #: loaded, and no such table exists: every banked vocabulary stops at 2.0 s.
    anchor_goal: str = "none"
    #: K of the anchor vocabulary (the shipped `refc_anchors_full_REBUILD.pt`).
    n_anchors: int = 256
    #: resolution of the FACTORED lateral sub-vocabulary (``"snap_lat"``).
    n_lat_bins: int = 16
    #: size of the per-window agent-slot vocabulary the four agent-referencing
    #: tokens index (`GAP_TARGET`, `YIELD_AT`, `WAIT_FOR_ONCOMING`,
    #: `EVADE_IN_CORRIDOR`). ⚠️ `TRAFFIC_LIGHT_REACT`'s slot can NEVER come from
    #: `obstacle.offline` — 10 dynamic classes, zero infrastructure.
    n_agent_slots: int = 8

    # ---- THE g_str -> P_T CONDITIONING PORT (F-1, 2026-08-16) --------------
    #: ⛔ DEFAULT OFF, byte-identity preserved (proved per tensor in
    #: ``tests/test_v6_gstr_port.py`` against a CONTENT-anchored pre-change
    #: revision — the live v6F S-W resume is 87,893,449 params / 405 keys and a
    #: broken strict resume kills it). ON builds :attr:`V6Stack.cond_tac_dyn`,
    #: a ZERO-INIT ``Linear(d_goal_embed -> 2*d_goal_embed)`` whose output is
    #: ADDED to the tactical action pair ``e_a_tac`` before ``predictor_tac``
    #: consumes it — i.e. the spec'd ``P_T(z_tac, a_tac | g_str)``
    #: (HIERARCHY_VOCABULARY §5; the binding diagram; this class's own
    #: docstring), which the code did NOT implement until the 2026-08-16
    #: conformance audit found it (DIAGRAM_CONFORMANCE.md F-1, the top 🟥):
    #: ``FTac``'s one conditioning input was fully consumed by the LAT×LON
    #: action embeddings, and ``e_g_str`` reached only ``goal_head_tac``. An
    #: S-T launched without this port would never train the strategic→tactical
    #: DYNAMICS downlink in its own stage — the ``intent_proj`` defect one
    #: level up, and precisely the fake-hierarchy failure the PI's remarks
    #: guard against.
    #: ⚠️ The port is the SAME construction as the accepted g_tac→P_O seam one
    #: level down (``cond = act_emb(actions) + intent_proj(intent)``): additive
    #: into the predictor's existing conditioning pathway, no ``FTac`` shape
    #: change (a shape change bypasses ``STAGE_MAY_INTRODUCE``'s adjudication,
    #: which is checked by KEYS — ``load_state_dict(strict=False)`` still
    #: RAISES on shapes, measured). Zero-init makes S-T's t1 loss continuous
    #: at introduction; ``e_g_str`` enters DETACHED under the planner cut, so
    #: the tactical WM loss cannot train ``layer_str`` backwards through it.
    tac_goal_cond: bool = False

    # ---- vocabulary --------------------------------------------------------
    d_goal_embed: int = 128

    # ---- X3 gradient isolation ---------------------------------------------
    #: planner/goal heads NEVER backprop into any encoder. Setting this False
    #: is the deliberately MIS-WIRED arm — :meth:`V6Stack.assert_isolation`
    #: then FAILS, which is precisely what makes the check a check.
    isolate_planner_from_encoder: bool = True
    #: higher→lower latent paths are cut. Setting this False is the co-trained
    #: control arm (H-COTRAIN's confirmed-branch lever, applied preventively).
    isolate_uplink: bool = True
    #: what the HIGHER layer's prediction TARGET is built from once the uplink
    #: is cut: ``"stopgrad"`` = the online adapter's own detached output;
    #: ``"ema"`` = an EMA-slow copy of that adapter (the V-JEPA teacher
    #: pattern). Either way no gradient flows up — this selects the target, not
    #: the isolation.
    uplink: str = "stopgrad"
    ema_decay: float = 0.996

    # ---- sub-configs (reused wholesale) ------------------------------------
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        in_channels=9, image_size=256, image_width=640, patch_size=16,
        d_model=384, depth=8, n_heads=6))
    readout: ReadoutConfig = field(default_factory=lambda: ReadoutConfig(
        grid=4, d_readout=128))
    predictor: PredictorConfig = field(default_factory=lambda: PredictorConfig(
        d_model=768, depth=6, n_heads=12, window=6, horizons=(1, 2, 4),
        action_dim=3, residual=True))
    f_hidden_tac: int = 512
    f_hidden_str: int = 512
    f_blocks: int = 3
    aux_hidden: int = 256          # O3 masked-cell predictor width

    # ---- O6 ----------------------------------------------------------------
    sigreg_slices: int = 512
    sigreg_beta: float = 1.0
    sigreg_free_dims: int = 0

    # ---- the invariant ------------------------------------------------------
    param_budget: int = PARAM_BUDGET

    def __post_init__(self):
        if self.plan_steps < 2:
            raise ValueError(f"plan_steps must be >= 2, got {self.plan_steps}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        for name, band in (("op_band_s", self.op_band_s),
                           ("tac_band_s", self.tac_band_s)):
            if len(band) != 2 or not band[0] < band[1]:
                raise ValueError(f"{name} must be (lo, hi) with lo < hi, got "
                                 f"{band}")
        if abs(self.op_band_s[1] - self.tac_band_s[0]) > 1e-9:
            raise ValueError(
                f"the operative band must END exactly where the tactical band "
                f"BEGINS — §4b's seam-free-by-construction requirement. Got "
                f"{self.op_band_s} then {self.tac_band_s}: a gap or an overlap "
                f"here IS the stitched-trajectory defect the spec forbids.")
        if abs(self.tac_band_s[1] - self.horizon_s) > 1e-9:
            raise ValueError(
                f"the tactical band must end at the plan horizon "
                f"{self.horizon_s} s (plan_steps {self.plan_steps} x dt "
                f"{self.dt}), got {self.tac_band_s[1]}")
        if self.selector not in ("none", "goal", "mlp"):
            raise ValueError(
                f"selector must be none|goal|mlp, got {self.selector!r}. "
                f"'mlp' is the pre-registered CAPACITY CONTROL for 'goal' "
                f"(V6F_PLANNER_DESIGN.md §5.3): same inputs, no distance "
                f"prior, ~127x the parameters. A 'goal' win judged without it "
                f"is unattributable between mechanism and capacity.")
        if self.selector_tau_m <= 0.0:
            raise ValueError(f"selector_tau_m must be > 0, got "
                             f"{self.selector_tau_m}")
        if self.selector_mlp_hidden <= 0:
            raise ValueError(f"selector_mlp_hidden must be > 0, got "
                             f"{self.selector_mlp_hidden}")
        if not 0.0 <= self.plan_wta_eps <= 1.0:
            raise ValueError(f"plan_wta_eps must be in [0, 1], got "
                             f"{self.plan_wta_eps}")
        if self.anchor_goal not in ("none", *AnchorGoalHead.MODES):
            raise ValueError(
                f"anchor_goal must be none|{'|'.join(AnchorGoalHead.MODES)}, "
                f"got {self.anchor_goal!r}. 'snap_lat' is the FACTORED default "
                f"the measurement prescribes (quantisation is FREE: Δ −0.0002 "
                f"[−0.1031, +0.0703]); 'onehot' is the metric-blind CONTROL "
                f"E-AG2 measured +4.7502 [+3.0514, +6.3981] WORSE.")
        if self.anchor_goal != "none" and not self.goal_cat_args:
            raise ValueError(
                "anchor_goal needs goal_cat_args=True. `anchor_id` / `lat_bin` "
                "are CATEGORICAL and the physical-units arg slots cannot hold "
                "them (§2.3's arg-TYPE gap) — an emitted id that reaches "
                "nothing downstream is a head wearing an emission's name, and "
                "writing it into a metres slot is the type error this channel "
                "exists to remove.")
        for nm, v in (("n_anchors", self.n_anchors),
                      ("n_lat_bins", self.n_lat_bins),
                      ("n_agent_slots", self.n_agent_slots)):
            if int(v) < 2:
                raise ValueError(f"{nm} must be >= 2, got {v}")
        if self.uplink not in ("stopgrad", "ema"):
            raise ValueError(f"uplink must be stopgrad|ema, got {self.uplink!r}")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got "
                             f"{self.ema_decay}")
        for nm, hz in (("hz_op", self.hz_op), ("hz_tac", self.hz_tac),
                       ("hz_str", self.hz_str)):
            if hz <= 0:
                raise ValueError(f"{nm} must be > 0, got {hz}")
        if not self.hz_op >= self.hz_tac >= self.hz_str:
            raise ValueError(
                f"clocks must be non-increasing up the hierarchy "
                f"(op {self.hz_op} >= tac {self.hz_tac} >= str {self.hz_str}) "
                f"— a higher layer ticking faster than a lower one is not a "
                f"hierarchy")
        if abs(1.0 / self.hz_op - self.dt) > 1e-9:
            raise ValueError(f"hz_op {self.hz_op} contradicts dt {self.dt} "
                             f"(the operative clock IS the control tick)")

    # ---- derived geometry ---------------------------------------------------
    @property
    def horizon_s(self) -> float:
        """The ONE plan horizon, seconds (6.0 by §4b)."""
        return self.plan_steps * self.dt

    @property
    def grid_shape(self) -> tuple[int, int]:
        gw = self.readout.grid if self.readout.grid_w is None \
            else int(self.readout.grid_w)
        return (int(self.readout.grid), gw)

    @property
    def n_cells(self) -> int:
        gh, gw = self.grid_shape
        return gh * gw

    @property
    def d_op(self) -> int:
        """Operative compact-state width — DERIVED, never set: the readout is
        the geometry firewall (``grid × grid_w × d_readout``)."""
        return self.n_cells * int(self.readout.d_readout)

    @property
    def stride_tac(self) -> int:
        """Operative ticks per tactical tick (10 Hz / 2 Hz = 5)."""
        return max(1, int(round(self.hz_op / self.hz_tac)))

    @property
    def stride_str(self) -> int:
        """Operative ticks per strategic tick (10 Hz / 0.5 Hz = 20)."""
        return max(1, int(round(self.hz_op / self.hz_str)))

    def band_slice(self, band: str) -> slice:
        """The step SLICE of the single 60-step rollout for ``"op"``/``"tac"``.

        Bands are slices of ONE trajectory (§4b) — this method is the only
        place the band boundaries are turned into indices, so there is exactly
        one convention in the programme.
        """
        lo, hi = {"op": self.op_band_s, "tac": self.tac_band_s}[band]
        return slice(int(round(lo / self.dt)),
                     min(int(round(hi / self.dt)), self.plan_steps))

    def split_bands(self, x: Tensor, dim: int = -2
                    ) -> tuple[Tensor, Tensor]:
        """Split a per-step tensor (…, plan_steps, …) into (operative,
        tactical) band views along ``dim``. Views, not copies — a band is a
        window onto the same rollout, and materialising it as a copy is how a
        "seam" gets invented."""
        if x.shape[dim] != self.plan_steps:
            raise ValueError(f"axis {dim} is {x.shape[dim]}, expected "
                             f"plan_steps={self.plan_steps}")
        idx_op, idx_tac = self.band_slice("op"), self.band_slice("tac")
        n = x.ndim
        d = dim if dim >= 0 else n + dim
        pre = (slice(None),) * d
        return x[pre + (idx_op,)], x[pre + (idx_tac,)]

    def to_dict(self) -> dict:
        import dataclasses as _dc
        out = _dc.asdict(self)
        out["_derived"] = {
            "d_op": self.d_op, "horizon_s": self.horizon_s,
            "grid_shape": list(self.grid_shape), "n_cells": self.n_cells,
            "stride_tac": self.stride_tac, "stride_str": self.stride_str,
            "op_band_steps": [self.band_slice("op").start,
                              self.band_slice("op").stop],
            "tac_band_steps": [self.band_slice("tac").start,
                               self.band_slice("tac").stop]}
        return out


# ============================================================================
# X5 — the staging protocol, as data
# ============================================================================
#: The four stages. A failed stage NEVER propagates upward (X5).
STAGES: tuple[str, ...] = ("S-W", "S-T", "S-S", "S-J")

#: Every named parameter group of :class:`V6Stack`. ``apply_stage_freeze``
#: partitions the whole module over exactly these, and RAISES if a parameter
#: escapes the partition — an unassigned parameter is a module that no stage
#: freezes and no stage trains, which is how a "frozen trunk" quietly moves.
MODULE_GROUPS: tuple[str, ...] = (
    "encoder", "readout", "predictor_op", "layer_tac", "layer_str",
    "planner", "aux",
)

#: What each stage TRAINS. Everything else is frozen.
#:   S-W  world stage: WM only, λ_plan ≡ 0, planner ABSENT.
#:   S-T  tactical layer + the operative planner it conditions, on the FROZEN
#:        S-W trunk (the Drive-JEPA "planner is a post-trained consumer" shape).
#:   S-S  strategic layer on the FROZEN S-T stack.
#:   S-J  optional brief joint polish — isolation still ON.
STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "S-W": ("encoder", "readout", "predictor_op", "aux"),
    "S-T": ("layer_tac", "planner"),
    "S-S": ("layer_str",),
    "S-J": MODULE_GROUPS,
}


def stage_trainable_groups(stage: str) -> tuple[str, ...]:
    if stage not in STAGE_GROUPS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return STAGE_GROUPS[stage]


def apply_stage_freeze(stack: "V6Stack", stage: str) -> dict:
    """Set ``requires_grad`` for ``stage`` and RETURN the audit.

    The returned dict names, per group, the trainable/frozen parameter counts
    and — explicitly — the SHARED goal tables, because a shared table moves
    whenever EITHER of its two views is in the trainable set. That is the
    intended §5 contract ("one vocabulary, two views"), and it is stated in the
    audit rather than left to be discovered: a freeze report that silently
    calls a layer "frozen" while a table inside it trains is the same class of
    lie as a probe that reports the wrong scope.
    """
    groups = set(stage_trainable_groups(stage))
    seen: set[str] = set()
    report: dict[str, dict] = {g: {"trainable": 0, "frozen": 0}
                               for g in MODULE_GROUPS}
    for name, p in stack.named_parameters():
        g = stack.group_of(name)
        seen.add(name)
        train = g in groups
        p.requires_grad_(train)
        report[g]["trainable" if train else "frozen"] += int(p.numel())
    missing = [n for n, _ in stack.named_parameters() if n not in seen]
    if missing:                                   # defensive: cannot happen
        raise RuntimeError(f"parameters escaped the group partition: {missing}")
    shared = {
        "vocab_tac": {"table": "vocab_tac.table.weight",
                      "emitted_by": "goal_head_tac (layer_tac)",
                      "consumed_by": "cond_op (planner)",
                      "group": stack.group_of("vocab_tac.table.weight")},
        "vocab_str": {"table": "vocab_str.table.weight",
                      "emitted_by": "goal_head_str (layer_str)",
                      "consumed_by": "cond_tac (layer_tac)",
                      "group": stack.group_of("vocab_str.table.weight")},
    }
    return {"stage": stage, "trainable_groups": sorted(groups),
            "per_group": report,
            "n_trainable": sum(v["trainable"] for v in report.values()),
            "n_frozen": sum(v["frozen"] for v in report.values()),
            "shared_goal_tables": shared,
            "_note": "a shared goal table moves whenever EITHER of its two "
                     "views is trainable (HIERARCHY_VOCABULARY §5)"}


# ============================================================================
# THE STACK
# ============================================================================

class _EmaCopy(nn.Module):
    """EMA-slow copy of a module (the V-JEPA teacher pattern). Parameters are
    buffers-in-spirit: ``requires_grad=False`` and excluded from every
    optimiser, so no gradient can ever flow through this branch — which is what
    makes ``uplink="ema"`` an X3-compliant target source rather than a second
    trainable path in disguise."""

    def __init__(self, src: nn.Module, decay: float = 0.996):
        super().__init__()
        import copy
        self.module = copy.deepcopy(src)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, src: nn.Module) -> None:
        d = self.decay
        for pt, ps in zip(self.module.parameters(), src.parameters()):
            pt.mul_(d).add_(ps.detach(), alpha=1.0 - d)
        for bt, bs in zip(self.module.buffers(), src.buffers()):
            bt.copy_(bs)

    def forward(self, *a, **kw):
        with torch.no_grad():
            return self.module(*a, **kw)


class MaskedCellPredictor(nn.Module):
    """O3 — predict MASKED readout-grid cells from the visible context.

    Cell tokens ``[B, C, d_r]`` + a bool mask ``[B, C]``: masked cells are
    replaced by a learned MASK token (plus their positional embedding, so the
    model knows WHICH cell it is being asked about — an I-JEPA target block is
    identified by position, and dropping that turns the task into "predict the
    average cell"), then one small transformer encoder mixes context into the
    holes. Loss is computed by the trainer on the masked cells only.
    """

    def __init__(self, n_cells: int, d_cell: int, hidden: int = 256,
                 depth: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_cells, self.d_cell = int(n_cells), int(d_cell)
        self.inp = nn.Linear(self.d_cell, hidden)
        self.pos = nn.Parameter(torch.zeros(1, self.n_cells, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, n_heads, dim_feedforward=4 * hidden, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        # enable_nested_tensor=False: norm_first layers cannot use the nested
        # fast path, and leaving it True only emits a warning on every build.
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth,
                                            enable_nested_tensor=False)
        self.out = nn.Linear(hidden, self.d_cell)

    def forward(self, cells: Tensor, mask: Tensor) -> Tensor:
        """``cells`` [B, C, d_r], ``mask`` [B, C] bool (True == masked) ->
        predicted cells [B, C, d_r] (only the masked entries are scored)."""
        if cells.shape[-2:] != (self.n_cells, self.d_cell):
            raise ValueError(f"cells must be [B, {self.n_cells}, "
                             f"{self.d_cell}], got {tuple(cells.shape)}")
        if mask.shape != cells.shape[:2]:
            raise ValueError(f"mask must be [B, {self.n_cells}], got "
                             f"{tuple(mask.shape)}")
        h = self.inp(cells)
        h = torch.where(mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        return self.out(self.blocks(h + self.pos.to(h.dtype)))


class V6Stack(nn.Module):
    """The v6 composition: three layers, one vocabulary per seam, one 6 s
    rollout, and the X3 matrix enforced in code.

    WIRING (HIERARCHY_VOCABULARY §5, verbatim):
      ``z_str_{t+K} = P_S(z_str_t, a_str_t)``
      ``z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)``
      ``z_op_{t+j}  = P_O(z_op_t, (a, κ)_t | g_tac)``
    Goals flow DOWN only; latents flow UP through stop-grad / EMA.
    ⚠️ The ``| g_str`` conditioning of P_T is the :attr:`cond_tac_dyn` port
    (F-1, DIAGRAM_CONFORMANCE.md 2026-08-16 — spec'd here all along, not built
    until then), gated by ``cfg.tac_goal_cond``: DEFAULT OFF so the live S-W
    resume stays byte-identical, and turned ON at S-T, whose
    ``STAGE_MAY_INTRODUCE`` allowance admits the fresh zero-init keys.

    THE PLANNER-SIDE SURFACE IS DECLARED, NOT INFERRED. ``forward`` returns
    ``planner_side`` — the list of tensors that must carry NO gradient into any
    encoder. :meth:`assert_isolation` backprops exactly that list. Declaring it
    is what makes the check total: a new head added without appending to the
    declaration is caught by ``tests/test_v6_staged.py::test_planner_surface_is_total``.
    """

    def __init__(self, cfg: V6Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or V6Config()
        gh, gw = cfg.grid_shape

        # ---- encoders (E-ENC arm) ------------------------------------------
        _Enc = ((lambda c: ViT5Encoder(c, n_registers=cfg.n_registers))
                if cfg.vit5_encoder else ViTEncoder)
        self.encoder = _Enc(cfg.encoder)
        self.readout = SpatialGridReadout(
            self.encoder.n_tokens, cfg.encoder.d_model, grid=cfg.readout.grid,
            d_readout=cfg.readout.d_readout,
            token_grid=self.encoder.grid_shape, grid_w=cfg.readout.grid_w)
        if self.readout.out_dim != cfg.d_op:
            raise RuntimeError(f"readout out_dim {self.readout.out_dim} != "
                               f"cfg.d_op {cfg.d_op} — the geometry firewall "
                               f"is the single source of the state width")
        # arm (b): each layer gets its OWN visual substrate at its own clock.
        self.encoder_tac = self.readout_tac = None
        self.encoder_str = self.readout_str = None
        if not cfg.shared_encoder:
            self.encoder_tac = _Enc(cfg.encoder)
            self.readout_tac = SpatialGridReadout(
                self.encoder_tac.n_tokens, cfg.encoder.d_model,
                grid=cfg.readout.grid, d_readout=cfg.readout.d_readout,
                token_grid=self.encoder_tac.grid_shape,
                grid_w=cfg.readout.grid_w)
            self.encoder_str = _Enc(cfg.encoder)
            self.readout_str = SpatialGridReadout(
                self.encoder_str.n_tokens, cfg.encoder.d_model,
                grid=cfg.readout.grid, d_readout=cfg.readout.d_readout,
                token_grid=self.encoder_str.grid_shape,
                grid_w=cfg.readout.grid_w)

        # ---- vocabularies: ONE object per seam, held by BOTH views ---------
        self.vocab_str = GoalVocabulary(STRATEGIC_GOAL_TOKENS, cfg.d_goal_embed)
        self.vocab_tac = GoalVocabulary(TACTICAL_GOAL_TOKENS, cfg.d_goal_embed)
        #: the layers' OWN action vocabularies (each conditions its own
        #: predictor). Tactical is FACTORED LAT × LON by design (§4).
        self.vocab_a_str = GoalVocabulary(STRATEGIC_ACTION_TOKENS,
                                          cfg.d_goal_embed)
        self.vocab_a_lat = GoalVocabulary(TACTICAL_LAT_ACTIONS, cfg.d_goal_embed)
        self.vocab_a_lon = GoalVocabulary(TACTICAL_LON_ACTIONS, cfg.d_goal_embed)

        # ---- layer O: predictor + the g_tac conditioner ---------------------
        self.predictor_op = OperativePredictor(cfg.predictor, cfg.d_op,
                                               intent_dim=cfg.d_goal_embed)
        #: metric grounding of the operative layer — the head that turns a
        #: latent TRANSITION into a per-step Δpose. O1's response-form L_ctrl
        #: (``train_stage_a.stage_a_losses``) decodes through exactly this, so
        #: it is a WM-side module (group ``predictor_op``, trained in S-W) and
        #: NOT part of the planner: a mis-grouping here would let the planner
        #: sculpt the metric decode, which is the C6 confound in miniature.
        self.step_readout_op = StepDisplacementReadout(cfg.d_op)
        self.cond_op = GoalConditioner(self.vocab_tac, cfg.d_goal_embed)

        # ---- layer T: adapter + predictor + goal head + g_str conditioner ---
        d_uplink_tac = cfg.d_op + (0 if cfg.shared_encoder else cfg.d_op)
        self.adapter_tac = nn.Sequential(
            nn.Linear(d_uplink_tac, cfg.adapter_hidden), nn.GELU(),
            nn.Linear(cfg.adapter_hidden, cfg.d_tac), nn.LayerNorm(cfg.d_tac))
        self.predictor_tac = FTac(cfg.d_tac, d_goal=2 * cfg.d_goal_embed,
                                  hidden=cfg.f_hidden_tac,
                                  n_blocks=cfg.f_blocks)
        self.cond_tac = GoalConditioner(self.vocab_str, cfg.d_goal_embed)
        self.goal_head_tac = GoalHead(self.vocab_tac, cfg.d_tac,
                                      d_cond=cfg.d_goal_embed)
        self.act_head_lat = GoalHead(self.vocab_a_lat, cfg.d_tac)
        self.act_head_lon = GoalHead(self.vocab_a_lon, cfg.d_tac)

        # ---- layer S: adapter + predictor + goal head + action head --------
        d_uplink_str = cfg.d_tac + (0 if cfg.shared_encoder else cfg.d_op)
        self.adapter_str = nn.Sequential(
            nn.Linear(d_uplink_str, cfg.adapter_hidden), nn.GELU(),
            nn.Linear(cfg.adapter_hidden, cfg.d_str), nn.LayerNorm(cfg.d_str))
        self.predictor_str = FTac(cfg.d_str, d_goal=cfg.d_goal_embed,
                                  hidden=cfg.f_hidden_str,
                                  n_blocks=cfg.f_blocks)
        self.goal_head_str = GoalHead(self.vocab_str, cfg.d_str)
        self.act_head_str = GoalHead(self.vocab_a_str, cfg.d_str)

        # ---- EMA-slow uplink targets (uplink="ema") -------------------------
        self.ema_adapter_tac = self.ema_adapter_str = None
        if cfg.uplink == "ema":
            self.ema_adapter_tac = _EmaCopy(self.adapter_tac, cfg.ema_decay)
            self.ema_adapter_str = _EmaCopy(self.adapter_str, cfg.ema_decay)

        # ---- emission: ONE 60-step (a, κ) sequence, W4-proven --------------
        self.plan_proj = nn.Linear(cfg.d_op, cfg.d_plan_feat)
        self.cand_queries = nn.Embedding(cfg.n_candidates, cfg.d_plan_feat)
        nn.init.trunc_normal_(self.cand_queries.weight, std=0.02)
        _ensure_scripts()
        from train_v58f_unicycle_head import UnicycleEmission  # noqa: E402
        self.emission = UnicycleEmission(
            feat_dim=cfg.d_plan_feat + cfg.d_goal_embed, k=cfg.plan_steps,
            hidden=cfg.emission_hidden, a_max=cfg.a_max,
            kappa_max=cfg.kappa_max, dt=cfg.dt,
            squash=cfg.emission_squash)

        # ---- O3 aux + O6 ---------------------------------------------------
        self.masked_cells = MaskedCellPredictor(
            cfg.n_cells, int(cfg.readout.d_readout), hidden=cfg.aux_hidden)
        self.sigreg = SigReg(cfg.sigreg_slices, cfg.sigreg_beta)
        #: per-cell nominal ranges for O2 (ESTIMATED prior — see the function).
        self.register_buffer("cell_ranges_m",
                             readout_grid_ranges(gh, gw).reshape(-1),
                             persistent=False)

        # ---- SELECTION (default OFF) — built LAST, and only when asked ------
        # Constructed at the very END so the ``"none"`` path consumes NO RNG and
        # creates NO state_dict key, and the ``"goal"`` path disturbs no other
        # module's initialisation. That is what makes the byte-identity proof in
        # tests/test_v6_selector.py TOTAL rather than approximate.
        self.cand_score = None
        if cfg.selector == "goal":
            self.cand_score = GoalDistanceScorer(
                cfg.d_goal_embed, cfg.n_candidates, tau_m=cfg.selector_tau_m)
        elif cfg.selector == "mlp":
            self.cand_score = MLPCandidateScorer(
                cfg.d_goal_embed, cfg.n_candidates,
                hidden=cfg.selector_mlp_hidden)

        # ---- GOAL-HEAD STRUCTURE (2026-08-16) — built LAST, only when asked --
        # ⛔ EVERYTHING BELOW OBEYS THE SAME RULE AS ``cand_score`` AND FOR THE
        # SAME REASON: it is constructed at the very END of __init__ and ONLY
        # when its flag is set, so the DEFAULT path draws NO random numbers,
        # creates NO state_dict key, and every earlier module's initialisation
        # is bit-for-bit what it was before these flags existed. That is what
        # makes the byte-identity proof in tests/test_v6_factored_goal.py TOTAL
        # rather than approximate — and v6F S-W is resuming from a checkpoint of
        # exactly this architecture RIGHT NOW.
        self.goal_head_tac_lat = self.goal_head_tac_lon = None
        self.vocab_tac_lat = self.vocab_tac_lon = None
        self.cond_op_lat = self.cond_op_lon = None
        self.anchor_head = None
        if cfg.goal_factored:
            # ⭐ the SAME idiom `a_tac` already uses (``act_head_lat`` /
            # ``act_head_lon``): two partitioned vocabularies, two heads, one
            # per axis — not a second idiom for the same job.
            self.vocab_tac_lat = GoalVocabulary(TACTICAL_GOAL_TOKENS_LAT,
                                                cfg.d_goal_embed)
            self.vocab_tac_lon = GoalVocabulary(TACTICAL_GOAL_TOKENS_LON,
                                                cfg.d_goal_embed)
            self.goal_head_tac_lat = GoalHead(self.vocab_tac_lat, cfg.d_tac,
                                              d_cond=cfg.d_goal_embed)
            self.goal_head_tac_lon = GoalHead(self.vocab_tac_lon, cfg.d_tac,
                                              d_cond=cfg.d_goal_embed)
            # ONE vocabulary, two views (§5): the conditioners below hold the
            # SAME objects the heads above hold. Both are Identity projections
            # (d_out == d_embed), so the factoring's parameter cost is the two
            # heads and the two tables and NOTHING hidden in the seam.
            self.cond_op_lat = GoalConditioner(self.vocab_tac_lat,
                                               cfg.d_goal_embed)
            self.cond_op_lon = GoalConditioner(self.vocab_tac_lon,
                                               cfg.d_goal_embed)
        if cfg.goal_cat_args:
            cards = (cfg.n_anchors, cfg.n_lat_bins, cfg.n_agent_slots,
                     len(STOP_REASONS), len(LIGHT_STATES))
            for v in (self.vocab_tac, self.vocab_tac_lat, self.vocab_tac_lon):
                if v is not None:
                    v.attach_cat_channel(cards)
            for h in (self.goal_head_tac, self.goal_head_tac_lat,
                      self.goal_head_tac_lon):
                if h is not None:
                    h.attach_cat_head()
        if cfg.goal_multilabel:
            # holds no parameters — the same logits, read a second way
            self.goal_head_tac.enable_multilabel(True)
        if cfg.anchor_goal != "none":
            self.anchor_head = AnchorGoalHead(
                cfg.d_goal_embed, cfg.n_anchors, mode=cfg.anchor_goal,
                plan_horizon_s=cfg.horizon_s, n_lat_bins=cfg.n_lat_bins)

        # ---- THE g_str -> P_T PORT (F-1) — built LAST, only when asked ------
        # ⛔ Same rule as ``cand_score`` and the goal-structure levers, same
        # reason: constructed at the very END of __init__ and ONLY under its
        # flag, so the default path draws NO RNG, creates NO state_dict key,
        # and every earlier module's initialisation is bit-for-bit what it was.
        # ZERO-INIT is load-bearing twice over: (1) at introduction the port is
        # an exact no-op, so S-T's t1 loss is CONTINUOUS when the flag first
        # turns on over an S-W checkpoint (`STAGE_MAY_INTRODUCE["S-T"]` admits
        # the fresh keys); (2) it is the same discipline every scorer here
        # follows (GoalDistanceScorer.goal_point, MLPCandidateScorer.fc2, the
        # emission head) — anything the port later does is LEARNED, never
        # handed to it by init. The aliveness-at-zero concern the GoalHead
        # docstring warns about does not bite here: the port's INPUT is
        # detached BY DESIGN (its upstream gradient is meant to be zero), and
        # its own parameters still receive gradient at zero weights, which
        # ``tests/test_v6_gstr_port.py`` measures rather than assumes.
        self.cond_tac_dyn = None
        if cfg.tac_goal_cond:
            self.cond_tac_dyn = nn.Linear(cfg.d_goal_embed,
                                          2 * cfg.d_goal_embed)
            nn.init.zeros_(self.cond_tac_dyn.weight)
            nn.init.zeros_(self.cond_tac_dyn.bias)

    # -- grouping ------------------------------------------------------------
    #: prefix -> group. Longest matching prefix wins, so ``predictor_tac``
    #: cannot be swallowed by ``predictor_op``'s entry.
    _GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
        ("encoder.", "encoder"), ("readout.", "readout"),
        ("encoder_tac.", "encoder"), ("readout_tac.", "readout"),
        ("encoder_str.", "encoder"), ("readout_str.", "readout"),
        # ⛔ THE g_tac SEAM TRAINS WITH THE PLANNER, NOT WITH THE TRUNK.
        # MEASURED 2026-08-13 (PI question "how will the operative predictor
        # learn to react to tactical goals if trained alone?"): with intent_proj
        # under `predictor_op`, S-W trains it while intent=None (zero gradient —
        # dead weight at random init) and S-T FREEZES it exactly when g_tac
        # first flows. The goal-injection port would stay random until S-J and
        # the hierarchy's downlink would silently not learn in its own stage.
        # The DYNAMICS stay trunk-frozen in S-T; only the goal-injection port
        # moves with the goal-conditioning side. Listed BEFORE the general
        # prefix: group_of picks the LONGEST match, so order is documentation,
        # correctness comes from specificity.
        ("predictor_op.intent_proj.", "planner"),
        ("predictor_op.intent_gate", "planner"),
        ("predictor_op.", "predictor_op"),
        ("step_readout_op.", "predictor_op"),
        ("adapter_tac.", "layer_tac"), ("predictor_tac.", "layer_tac"),
        ("cond_tac.", "layer_tac"), ("goal_head_tac.", "layer_tac"),
        ("act_head_lat.", "layer_tac"), ("act_head_lon.", "layer_tac"),
        ("vocab_tac.", "layer_tac"), ("vocab_a_lat.", "layer_tac"),
        ("vocab_a_lon.", "layer_tac"), ("ema_adapter_tac.", "layer_tac"),
        # the FACTORED g_tac pair sits in the SAME group as the mixed head it
        # replaces — the factoring is a shape change, not a stage change, and a
        # head that trained in a different stage would not be its control.
        ("goal_head_tac_lat.", "layer_tac"), ("goal_head_tac_lon.", "layer_tac"),
        ("vocab_tac_lat.", "layer_tac"), ("vocab_tac_lon.", "layer_tac"),
        # ⭐ THE g_str->P_T PORT (F-1) IS `layer_tac`, NOT `planner` — and the
        # asymmetry with `intent_proj` (grouped planner) is deliberate, not
        # drift. intent_proj lives inside predictor_op, whose group trains in
        # S-W while intent=None (the measured dead-weight defect) — regrouping
        # it to planner made it train exactly when g_tac first flows. THIS port
        # conditions predictor_tac, which is ALREADY layer_tac and ALREADY
        # trains in S-T, the stage whose t1 loss flows through the conditioned
        # prediction — so grouping it with the dynamics it conditions gives the
        # same train-when-live property with no regrouping. It must NOT be
        # `planner`: its output feeds `zh_tac` (WM-side, declared uplink_side),
        # not the plan, and a planner-group parameter unreachable from the
        # declared planner_side surface would fail
        # test_planner_surface_is_total by construction.
        ("cond_tac_dyn.", "layer_tac"),
        ("adapter_str.", "layer_str"), ("predictor_str.", "layer_str"),
        ("goal_head_str.", "layer_str"), ("act_head_str.", "layer_str"),
        ("vocab_str.", "layer_str"), ("vocab_a_str.", "layer_str"),
        ("ema_adapter_str.", "layer_str"),
        ("cond_op.", "planner"), ("plan_proj.", "planner"),
        ("cond_op_lat.", "planner"), ("cond_op_lon.", "planner"),
        ("cand_queries.", "planner"), ("emission.", "planner"),
        # ⭐ ANCHOR_GOAL is a PLANNER module for the same reason the selector is:
        # it reads e_g_tac and decodes the goal POINT the selection consumes, so
        # it must be retrained on the trunk it consumes (registry §1.14 — the
        # frozen selector read 0.7933 -> 4.4159 when the trunk moved under it).
        ("anchor_head.", "planner"),
        # SELECTION is a PLANNER module: it is trained in S-T on the trunk it
        # consumes, never in S-W. "You cannot repair a trunk and keep its
        # planner" (registry §1.14) applies to the selector first of all — the
        # frozen selector read 0.7933 -> 4.4159 when the trunk moved under it.
        ("cand_score.", "planner"),
        ("masked_cells.", "aux"), ("sigreg.", "aux"),
    )

    def group_of(self, param_name: str) -> str:
        """Which :data:`MODULE_GROUPS` entry owns ``param_name``. Raises on an
        unmapped parameter — a new submodule must declare its group, because a
        parameter no stage owns is a parameter no stage freezes."""
        best, best_len = None, -1
        for pre, grp in self._GROUP_PREFIXES:
            if param_name.startswith(pre) and len(pre) > best_len:
                best, best_len = grp, len(pre)
        if best is None:
            raise KeyError(
                f"parameter {param_name!r} belongs to no group. Add it to "
                f"V6Stack._GROUP_PREFIXES — an ungrouped parameter is one no "
                f"stage freezes and no stage trains.")
        return best

    def group_parameters(self, *groups: str):
        want = set(groups)
        for n, p in self.named_parameters():
            if self.group_of(n) in want:
                yield n, p

    # -- the encoder-protected set (X3's "any encoder") ----------------------
    def encoder_parameters(self):
        """Every parameter the planner may NEVER reach: all encoders AND the
        readouts. The readout is part of the representation path — a gradient
        that stops at the readout has still sculpted the state the probes read,
        so protecting only ``encoder.*`` would be protecting the wrong scope."""
        yield from self.group_parameters("encoder", "readout")

    # ---------------------------------------------------------------------- #
    # forward pieces                                                          #
    # ---------------------------------------------------------------------- #
    def encode_window(self, frames: Tensor) -> Tensor:
        """``frames`` [B, W, C, H, W'] -> operative states [B, W, d_op]."""
        b, w = frames.shape[:2]
        flat = frames.reshape(b * w, *frames.shape[2:])
        return self.readout(self.encoder(flat)).reshape(b, w, -1)

    def cells(self, z_op: Tensor) -> Tensor:
        """Compact state [B, d_op] -> readout CELL tokens [B, C, d_readout].

        The DINO-WM lesson made structural: *"pooling is where geometry goes to
        die"*. RC1 in ``JEPA_PHYSICS_SURVEY.md`` §2 is precisely the hypothesis
        that lead geometry lives in these cells and dies in aggregation — so
        the cells are exposed, and O3 predicts them directly."""
        return z_op.reshape(*z_op.shape[:-1], self.cfg.n_cells,
                            int(self.cfg.readout.d_readout))

    def _cut(self, z: Tensor, isolate: bool) -> Tensor:
        return z.detach() if isolate else z

    def uplink_tac(self, z_op: Tensor, own: Tensor | None = None
                   ) -> tuple[Tensor, Tensor]:
        """z_op -> (z_tac ONLINE, z_tac TARGET).

        The stop-grad is UNCONDITIONAL under ``isolate_uplink`` — that is X3.
        ``uplink`` then selects what the TARGET is built from: the online
        adapter's own detached output (``stopgrad``) or an EMA-slow copy of it
        (``ema``, the V-JEPA teacher). Neither carries gradient upward.
        """
        cfg = self.cfg
        src = self._cut(z_op, cfg.isolate_uplink)
        if cfg.shared_encoder:
            if own is not None:
                raise ValueError("shared_encoder=True takes no per-layer "
                                 "features — that is the E-ENC arm (b) input")
            x = src
        else:
            if own is None:
                raise ValueError("shared_encoder=False needs the tactical "
                                 "layer's OWN encoded features")
            x = torch.cat([src, self._cut(own, cfg.isolate_uplink)], dim=-1)
        online = self.adapter_tac(x)
        if self.ema_adapter_tac is not None:
            target = self.ema_adapter_tac(x)
        else:
            target = online.detach()
        return online, target.detach()

    def uplink_str(self, z_tac: Tensor, own: Tensor | None = None
                   ) -> tuple[Tensor, Tensor]:
        """z_tac -> (z_str ONLINE, z_str TARGET). Same contract as
        :meth:`uplink_tac`, one level up."""
        cfg = self.cfg
        src = self._cut(z_tac, cfg.isolate_uplink)
        if cfg.shared_encoder:
            if own is not None:
                raise ValueError("shared_encoder=True takes no per-layer "
                                 "features")
            x = src
        else:
            if own is None:
                raise ValueError("shared_encoder=False needs the strategic "
                                 "layer's OWN encoded features")
            x = torch.cat([src, self._cut(own, cfg.isolate_uplink)], dim=-1)
        online = self.adapter_str(x)
        if self.ema_adapter_str is not None:
            target = self.ema_adapter_str(x)
        else:
            target = online.detach()
        return online, target.detach()

    @torch.no_grad()
    def layer_targets(self, z_op_future: Tensor, own_tac: Tensor | None = None,
                      own_str: Tensor | None = None) -> dict:
        """The higher layers' PREDICTION TARGETS at a FUTURE tick.

        ``z_op_future`` is the operative latent one tactical (resp. strategic)
        tick ahead, encoded from the true future frame. Run under ``no_grad``,
        so this branch can never become a second gradient path upward — the
        target of a latent-prediction loss must be a target, not a co-trained
        partner (BYOL/V-JEPA discipline, and X3's rule for our seams).

        ⚠️ Call it with the frame at ``cfg.stride_tac`` steps ahead for the
        tactical target and ``cfg.stride_str`` ahead for the strategic one —
        each layer's own clock. Predicting a target one OPERATIVE tick ahead
        and calling it a tactical prediction would make the tactical loss an
        identity map wearing a hierarchy's name.
        """
        z_tac, z_tac_t = self.uplink_tac(z_op_future, own_tac)
        _z_str, z_str_t = self.uplink_str(z_tac, own_str)
        return {"z_tac": z_tac_t.detach(), "z_str": z_str_t.detach()}

    @torch.no_grad()
    def ema_update(self) -> None:
        """Advance the EMA-slow uplink copies. No-op under ``stopgrad``."""
        if self.ema_adapter_tac is not None:
            self.ema_adapter_tac.update(self.adapter_tac)
        if self.ema_adapter_str is not None:
            self.ema_adapter_str.update(self.adapter_str)

    @staticmethod
    def _encode_goal(cond: GoalConditioner, head: dict) -> Tensor:
        """Turn a :class:`GoalHead` output into the embedding the layer below
        consumes — the ONE place the token view and the arg channels are
        selected, so a new channel cannot reach one seam and miss another.

        * MULTI-LABEL: ``gates`` (independent per-token) when the head emits
          them, otherwise ``probs``. Both are ``[B, n_tokens]`` floats and
          travel the SAME ``probs @ table.weight`` path, so a SET of goals is a
          sum of token embeddings and needs no second convention.
        * CATEGORICAL args: passed with the mask implied by WHICH token was
          emitted, so a slot the emitted token does not use contributes exactly
          zero (§2: "Unset = unconstrained").

        ⚠️ With neither channel on this is exactly ``cond(probs, args)`` — the
        pre-2026-08-16 call, unchanged, which is what keeps the default build's
        forward bit-for-bit what it was.
        """
        tok = head.get("gates", head["probs"])
        if "cat_probs" not in head:
            return cond(tok, head["args"])
        return cond(tok, head["args"], cat=head["cat_probs"],
                    cat_mask=cond.vocab.cat_mask_from_tokens(tok))

    def emit(self, z_op: Tensor, g_tac_embed: Tensor, v0: Tensor) -> dict:
        """THE 6 s EMISSION (§4b). ``z_op`` [B, d_op], ``g_tac_embed``
        [B, d_goal_embed], ``v0`` [B] ->

          ``controls``  [B, N, 60, 2]  the (a, κ) sequence, feasible by
                        construction (``a = a_max·tanh``, ``κ = κ_max·tanh``);
          ``waypoints`` [B, N, 60, 2]  ONE unicycle rollout 0 → 6 s;
          ``feat``      [B, N, F]      the pre-emission feature — DECLARED
                        planner-side, and the tensor the isolation probe
                        backprops (the emission's final layer is zero-init for
                        the CV warm start, so probing only the controls would
                        report "no gradient" for a live mis-wire).

        ``N == cfg.n_candidates``; ``squeeze_candidates`` collapses N=1.
        """
        cfg = self.cfg
        if z_op.ndim != 2 or z_op.shape[-1] != cfg.d_op:
            raise ValueError(f"z_op must be [B, {cfg.d_op}], got "
                             f"{tuple(z_op.shape)}")
        if v0.ndim != 1 or v0.shape[0] != z_op.shape[0]:
            raise ValueError(f"v0 must be [B={z_op.shape[0]}], got "
                             f"{tuple(v0.shape)}")
        b = z_op.shape[0]
        base = self.plan_proj(z_op)[:, None, :]                    # [B,1,F0]
        feat = base + self.cand_queries.weight[None]               # [B,N,F0]
        g = g_tac_embed[:, None, :].expand(b, cfg.n_candidates, -1)
        feat = torch.cat([feat, g.to(feat.dtype)], dim=-1)         # [B,N,F]
        a_ctl, kappa, wp = self.emission(feat, v0)
        out = {"a": a_ctl, "kappa": kappa,
               "controls": torch.stack([a_ctl, kappa], dim=-1),
               "waypoints": wp, "feat": feat}
        if self.anchor_head is not None:
            # ⭐ the STRUCTURED goal point. Emitted BEFORE selection because the
            # selection consumes it — and computed from ``g_tac_embed`` alone,
            # the identical input the free decode reads, so the two are matched.
            out |= {f"anchor_{k}": v
                    for k, v in self.anchor_head(g_tac_embed).items()}
        if self.cand_score is not None:
            # ⚠️ the scorer reads the EMITTED trajectory, i.e. the RANKED
            # OBJECT itself. E-S1-0's dose-response is the sharpest statement in
            # the programme on this point — the supervised t=0 confidence
            # selects at 0.4728 while THE SAME WEIGHTS' unsupervised refined
            # readout selects at 1.3100, a 2.8x penalty purely for scoring
            # off-distribution. Reproduced 2026-08-15 on the banked XL fan:
            # shipped 0.4714 vs refined 1.3901 (2.95x).
            # ⚠️ the ANCHOR goal point overrides the selector's own free decode
            # ONLY for the distance rule — the "mlp" CAPACITY CONTROL is left
            # information-MATCHED on ``e_g_tac`` and NOT handed a ready-made
            # goal point, because handing it one would make it an information
            # control and its result would stop speaking to capacity (§5.3).
            kw = {}
            if isinstance(self.cand_score, GoalDistanceScorer) \
                    and "anchor_goal_point" in out:
                kw["goal_point"] = out["anchor_goal_point"]
            out |= {f"sel_{k}": v
                    for k, v in self.cand_score(wp, g_tac_embed, **kw).items()}
        return out

    def forward(self, frames: Tensor, actions: Tensor, v0: Tensor, *,
                own_frames_tac: Tensor | None = None,
                own_frames_str: Tensor | None = None) -> dict:
        """One full hierarchy pass.

        ``frames``  [B, W, C, H, W'] · ``actions`` [B, W, A] (the 3-channel
        lifted format the predictor trains with) · ``v0`` [B] m/s.
        ``own_frames_*`` [B, C, H, W'] are the per-layer encoder inputs of the
        E-ENC arm (b); they are REFUSED under ``shared_encoder=True`` so an arm
        cannot be run half-configured.

        Returns every layer's latents, goals, actions and the 6 s plan, plus
        ``planner_side`` — the DECLARED surface :meth:`assert_isolation` probes.
        """
        cfg = self.cfg
        z_op_win = self.encode_window(frames)                      # [B,W,d_op]
        z_op = z_op_win[:, -1]

        own_tac = own_str = None
        if not cfg.shared_encoder:
            if own_frames_tac is None or own_frames_str is None:
                raise ValueError("shared_encoder=False needs own_frames_tac "
                                 "and own_frames_str (E-ENC arm b)")
            own_tac = self.readout_tac(self.encoder_tac(own_frames_tac))
            own_str = self.readout_str(self.encoder_str(own_frames_str))
        elif own_frames_tac is not None or own_frames_str is not None:
            raise ValueError("shared_encoder=True takes no per-layer frames")

        # ---- up: O -> T -> S (WM-side; cut by isolate_uplink) ---------------
        z_tac, z_tac_tgt = self.uplink_tac(z_op, own_tac)
        z_str, z_str_tgt = self.uplink_str(z_tac, own_str)

        # ---- planner-side VIEWS (cut by isolate_planner_from_encoder) ------
        # ⚠️ The two cuts are INDEPENDENT levers on purpose. The uplink cut is a
        # WM-side rule (higher layers reach lower latents only through
        # stop-grad/EMA); the planner cut is the "no goal head in any encoder"
        # rule. If the goal heads read the RAW layer latents, then disabling the
        # uplink alone would silently also open a planner→encoder path, and the
        # two control arms would stop being one lever each — the `--v2`
        # conflation failure in miniature.
        cut = cfg.isolate_planner_from_encoder
        z_plan = self._cut(z_op, cut)
        z_tac_p = self._cut(z_tac, cut)
        z_str_p = self._cut(z_str, cut)

        # ---- down: goals S -> T -> O ---------------------------------------
        g_str = self.goal_head_str(z_str_p)
        a_str = self.act_head_str(z_str_p)
        e_g_str = self.cond_tac(g_str["probs"], g_str["args"])
        e_a_str = self.vocab_a_str.encode(a_str["probs"], a_str["args"])

        g_tac = self.goal_head_tac(z_tac_p, cond=e_g_str)
        a_lat = self.act_head_lat(z_tac_p)
        a_lon = self.act_head_lon(z_tac_p)
        # ⭐ THE FACTORED g_tac PAIR. Both halves read exactly what the mixed
        # head reads (``z_tac_p`` and ``e_g_str``) — the factoring changes the
        # DECISION's shape, not its information, so a difference cannot be an
        # information effect. The mixed head is still emitted: it is this arm's
        # CONTROL, and a comparison with no control is unattributable (C6).
        g_tac_lat = g_tac_lon = None
        if self.goal_head_tac_lat is not None:
            g_tac_lat = self.goal_head_tac_lat(z_tac_p, cond=e_g_str)
            g_tac_lon = self.goal_head_tac_lon(z_tac_p, cond=e_g_str)
            # the MEAN, not the sum: a factored embedding twice the mixed one's
            # scale would change the conditioning magnitude the operative FiLM
            # sees, and a scale change between an arm and its control is a
            # confound wearing an architecture's name.
            e_g_tac = 0.5 * (self._encode_goal(self.cond_op_lat, g_tac_lat)
                             + self._encode_goal(self.cond_op_lon, g_tac_lon))
        else:
            e_g_tac = self._encode_goal(self.cond_op, g_tac)
        e_a_tac = torch.cat(
            [self.vocab_a_lat.encode(a_lat["probs"], a_lat["args"]),
             self.vocab_a_lon.encode(a_lon["probs"], a_lon["args"])], dim=-1)

        # ---- each layer's predictor rolls under its OWN action --------------
        zh_str = self.predictor_str(z_str, e_a_str)
        # ⭐ THE g_str -> P_T PORT (F-1): `z_tac_{t+k} = P_T(z_tac, a_tac |
        # g_str)` — the spec'd but previously unbuilt fifth downward port. The
        # strategic goal is projected (zero-init) and ADDED to the action-pair
        # conditioning, exactly the idiom the g_tac->P_O seam uses one level
        # down (`cond = act_emb(actions) + intent_proj(intent)`). Default OFF
        # keeps `g_cond_tac` the SAME tensor object as `e_a_tac` — bit-for-bit
        # the pre-F-1 forward.
        # ⚠️ e_g_str enters DETACHED under the planner cut, the same downward-
        # port rule as the intent port below: the goal STEERS the tactical
        # dynamics, but the tactical WM loss (t1) must not train the strategic
        # goal path (`goal_head_str`/`vocab_str`/`cond_tac`, all above this
        # seam) backwards through it — gradients reach the port's OWN
        # parameters only. Goals flow down BY DESIGN; gradient does not flow
        # back up except into the port itself.
        g_cond_tac = e_a_tac
        if self.cond_tac_dyn is not None:
            g_cond_tac = e_a_tac + self.cond_tac_dyn(self._cut(e_g_str, cut))
        zh_tac = self.predictor_tac(z_tac, g_cond_tac)
        # ⚠️ the g_tac conditioning enters the OPERATIVE predictor detached
        # under isolation: the goal STEERS the operative dynamics, but the
        # operative WM loss must not train the goal head backwards through it
        # (that would be the planner gradient re-entering the trunk by the
        # conditioning port instead of the encoder — the same defect, one door
        # along).
        zh_op = self.predictor_op(z_op_win, actions,
                                  intent=self._cut(e_g_tac, cut))

        # ---- THE g_tac->OPERATIVE SEAM, exercised on DETACHED trunk inputs --
        # ⛔ Added 2026-08-13 after a PI question exposed that NO S-T loss
        # flowed through the goal-conditioned operative prediction: t1_latent
        # trains layer_tac, lambda_plan trains the planner, and zh_op fed
        # neither — so intent_proj (the hierarchy's downlink) would have
        # stayed at random init until S-J. The same failure shape as v5's
        # nav-echo: a conditioning path present in the diagram, absent from
        # the optimisation.
        # Trunk inputs are DETACHED so gradient through this tensor reaches
        # ONLY the goal-injection side (intent_proj + the goal embeddings) —
        # which is what makes it admissible in `planner_side` under X3: it
        # cannot carry gradient into the encoder BY CONSTRUCTION.
        zh_op_seam = self.predictor_op(z_op_win.detach(), actions.detach(),
                                       intent=self._cut(e_g_tac, cut))

        # ---- the ONE 6 s plan ----------------------------------------------
        plan = self.emit(z_plan, e_g_tac, v0)
        # The selector is planner-side and MUST be probed by X3 like every other
        # planner output — a head added without appending to the declaration is
        # exactly what test_planner_surface_is_total exists to catch.
        sel_side = [plan[k] for k in
                    ("sel_score", "sel_goal_point", "sel_goal_point_free",
                     # ⭐ the ANCHOR_GOAL head is planner-group, so every one of
                     # its parameters must be REACHABLE from this declaration or
                     # test_planner_surface_is_total fails — which is exactly
                     # the guard that catches a head added without declaring it.
                     # ``anchor_cls_logits`` is what makes the one-hot CONTROL
                     # reachable: its emitted point is a HARD table lookup and
                     # carries no gradient at all, by design.
                     "anchor_goal_point", "anchor_goal_point_raw",
                     "anchor_cls_logits")
                    if k in plan]
        # the FACTORED pair is planner-side for the same reason the mixed head
        # is: it is a goal head, and X3 forbids any goal head reaching an
        # encoder.
        fact_side = [t for h in (g_tac_lat, g_tac_lon) if h is not None
                     for t in (h["logits"], h["args"])]
        cat_side = [h["cat_logits"] for h in (g_str, a_str, g_tac, g_tac_lat,
                                              g_tac_lon, a_lat, a_lon)
                    if h is not None and "cat_logits" in h]

        return {
            "z_op_win": z_op_win, "z_op": z_op, "z_plan": z_plan,
            "z_tac": z_tac, "z_tac_target": z_tac_tgt,
            "z_str": z_str, "z_str_target": z_str_tgt,
            "g_str": g_str, "a_str": a_str, "g_tac": g_tac,
            "g_tac_lat": g_tac_lat, "g_tac_lon": g_tac_lon,
            "a_lat": a_lat, "a_lon": a_lon,
            "e_g_str": e_g_str, "e_g_tac": e_g_tac,
            "zhat_op": zh_op, "zhat_op_seam": zh_op_seam,
            "zhat_tac": zh_tac, "zhat_str": zh_str,
            "plan": plan,
            # ⛔ THE DECLARED PLANNER-SIDE SURFACE (X3). Every tensor here must
            # carry ZERO gradient into any encoder/readout parameter.
            "planner_side": [
                g_str["logits"], g_str["args"], a_str["logits"],
                a_str["args"], g_tac["logits"], g_tac["args"],
                a_lat["logits"], a_lon["logits"],
                plan["feat"], plan["a"], plan["kappa"], plan["waypoints"],
                *sel_side, *fact_side, *cat_side,
                # the seam: detached-trunk, goal-conditioned — reaches
                # intent_proj and nothing in the encoder (X3-safe by the
                # .detach() above, and the probe now VERIFIES that)
                *zh_op_seam.values(),
            ],
            # higher-layer latents: must carry zero gradient into the layer(s)
            # BELOW them (stop-grad / EMA uplink).
            "uplink_side": {"tac": [z_tac, zh_tac], "str": [z_str, zh_str]},
        }

    # ---------------------------------------------------------------------- #
    # X3 — the MEASURED isolation check                                       #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _probe_scalar(tensors) -> Tensor:
        """A deterministic, NON-SYMMETRIC linear reduction of a tensor list.

        Linear (not squared) so a zero-valued tensor still yields gradient 1 —
        the emission head is zero-init, and ``x.pow(2).sum()`` at ``x = 0`` has
        zero gradient, which would make a LIVE mis-wire look isolated.
        Non-uniform weights so two paths cannot cancel by symmetry.
        """
        total = None
        for i, t in enumerate(tensors):
            tf = t.float().reshape(-1)
            w = torch.linspace(1.0, 2.0, tf.numel(), device=tf.device,
                               dtype=tf.dtype) * (1.0 + 0.37 * i)
            s = (tf * w).sum()
            total = s if total is None else total + s
        if total is None:
            raise ValueError("nothing to probe")
        return total

    @staticmethod
    def _live_edges(scalar: Tensor, named_params, *, atol: float = 0.0
                    ) -> list[str]:
        pairs = [(n, p) for n, p in named_params if p.requires_grad]
        if not pairs:
            return []
        grads = torch.autograd.grad(scalar, [p for _, p in pairs],
                                    allow_unused=True, retain_graph=True,
                                    materialize_grads=False)
        live = []
        for (n, _p), g in zip(pairs, grads):
            if g is not None and float(g.abs().max()) > atol:
                live.append(n)
        return live

    def synthetic_batch(self, batch: int = 2, device=None, seed: int = 0
                        ) -> dict:
        """A random CPU/GPU batch with the exact forward contract — used by
        :meth:`assert_isolation` and by the trainer's ``--dry-run`` so the pod
        launch is verifiable before any corpus exists."""
        cfg = self.cfg
        g = torch.Generator().manual_seed(seed)
        c = cfg.encoder.in_channels
        h, w = cfg.encoder.image_hw()
        frames = torch.randn(batch, cfg.predictor.window, c, h, w,
                             generator=g)
        actions = torch.randn(batch, cfg.predictor.window,
                              cfg.predictor.action_dim, generator=g) * 0.1
        v0 = torch.rand(batch, generator=g) * 20.0 + 1.0
        out = {"frames": frames, "actions": actions, "v0": v0}
        if not cfg.shared_encoder:
            out["own_frames_tac"] = torch.randn(batch, c, h, w, generator=g)
            out["own_frames_str"] = torch.randn(batch, c, h, w, generator=g)
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    def assert_isolation(self, batch: dict | None = None, *,
                         batch_size: int = 2, device=None,
                         strict: bool = True) -> dict:
        """MEASURE the X3 gradient-isolation matrix on a real autograd graph.

        Three edges are probed, each by backpropagating the module's OWN
        declared surface and reading which parameters actually received
        gradient (``torch.autograd.grad(..., allow_unused=True)`` — it never
        touches ``.grad``, so calling this mid-training is safe):

          1. **planner → encoder** — no goal head, no action head and no
             emission may reach any encoder or readout parameter;
          2. **tactical → below** — the tactical layer may not reach the
             operative predictor or any encoder (the uplink is stop-grad/EMA);
          3. **strategic → below** — likewise for the strategic layer.

        Returns the report; raises :class:`IsolationViolation` under ``strict``
        naming the live parameters. ⚠️ This is a MEASUREMENT, not a proof over
        all inputs: it certifies the graph built by THIS forward. That is
        exactly the class of evidence a training run needs (the graph it will
        actually backprop), and it is strictly more than a comment claiming the
        detach is there.
        """
        was_training = self.training
        # ⚠️ X3 is an ARCHITECTURE property, not a stage property. A frozen
        # parameter records no autograd edge, so a check run mid-S-T would find
        # the encoder "isolated" simply because it is frozen — a vacuous pass,
        # and vacuous passes are how an isolation guarantee rots. Every
        # parameter is therefore made differentiable FOR THE PROBE ONLY (before
        # the forward, since requires_grad decides what the graph records) and
        # the exact prior state is restored in the ``finally``.
        saved = [(p, bool(p.requires_grad)) for p in self.parameters()]
        self.eval()
        try:
            for p, _ in saved:
                p.requires_grad_(True)
            b = batch or self.synthetic_batch(batch_size, device)
            out = self.forward(**b)
            enc = list(self.encoder_parameters())
            below_tac = list(self.group_parameters(
                "encoder", "readout", "predictor_op"))
            below_str = list(self.group_parameters(
                "encoder", "readout", "predictor_op", "layer_tac"))
            checks = {
                "planner_to_encoder": self._live_edges(
                    self._probe_scalar(out["planner_side"]), enc),
                "tactical_to_below": self._live_edges(
                    self._probe_scalar(out["uplink_side"]["tac"]), below_tac),
                "strategic_to_below": self._live_edges(
                    self._probe_scalar(out["uplink_side"]["str"]), below_str),
            }
        finally:
            for p, rg in saved:
                p.requires_grad_(rg)
            self.train(was_training)
        report = {
            "matrix": {k: list(v) for k, v in ISOLATION_MATRIX.items()},
            "config": {
                "isolate_planner_from_encoder":
                    self.cfg.isolate_planner_from_encoder,
                "isolate_uplink": self.cfg.isolate_uplink,
                "uplink": self.cfg.uplink,
                "shared_encoder": self.cfg.shared_encoder},
            "violations": {k: v[:12] for k, v in checks.items() if v},
            "n_violations": {k: len(v) for k, v in checks.items()},
            "n_probed": {"planner_to_encoder": len(enc),
                         "tactical_to_below": len(below_tac),
                         "strategic_to_below": len(below_str)},
            "pass": not any(checks.values()),
            "_evidence_class": "MEASURED (ours; autograd probe on this graph)",
        }
        if strict and not report["pass"]:
            first = {k: v[:5] for k, v in checks.items() if v}
            raise IsolationViolation(
                f"X3 gradient-isolation VIOLATED — live forbidden edges "
                f"{first}. Config: {report['config']}. A planner gradient in "
                f"an encoder is the co-training failure the staged protocol "
                f"exists to prevent (JEPA_PHYSICS_SURVEY §4).")
        return report

    # ---------------------------------------------------------------------- #
    # the sub-300M invariant                                                  #
    # ---------------------------------------------------------------------- #
    def param_report(self, budget: int | None = None) -> dict:
        """Per-group parameter counts + the sub-300M verdict.

        Counts EVERY parameter (not only ``requires_grad``): a stage freezes
        modules, it does not delete them, so the deployed model carries them
        all. Shared objects (the goal tables, held by two views each) are
        counted ONCE — ``named_parameters`` de-duplicates by identity, which is
        also the check that the sharing is real: if a table were accidentally
        copied, this total would jump.
        """
        budget = int(budget or self.cfg.param_budget)
        per = {g: 0 for g in MODULE_GROUPS}
        for n, p in self.named_parameters():
            per[self.group_of(n)] += int(p.numel())
        total = sum(per.values())
        return {
            "total": total, "budget": budget,
            "within_budget": total <= budget,
            "headroom": budget - total,
            "per_group": per,
            "trainable": sum(int(p.numel()) for p in self.parameters()
                             if p.requires_grad),
            "arm": ("shared-encoder+adapters" if self.cfg.shared_encoder
                    else "per-layer-encoders"),
            "d_op": self.cfg.d_op, "d_tac": self.cfg.d_tac,
            "d_str": self.cfg.d_str,
            "_note": "E-ENC decides at MATCHED TOTAL PARAMS — compare arms on "
                     "this number, not on per-layer widths",
            "_evidence_class": "MEASURED (ours; count at instantiation)",
        }

    def assert_param_budget(self, budget: int | None = None) -> dict:
        rep = self.param_report(budget)
        if not rep["within_budget"]:
            raise ValueError(
                f"v6 stack is {rep['total']:,} params, over the sub-300M "
                f"INVARIANT ({rep['budget']:,}). Per group: {rep['per_group']}")
        return rep


def matched_param_config(base: V6Config, target_total: int, *,
                         probe: tuple[int, ...] | None = None,
                         lo: int = 192, hi: int = 1600) -> tuple[V6Config, dict]:
    """E-ENC helper: the ``predictor.d_model`` whose TOTAL param count comes
    closest to ``target_total`` without exceeding the budget.

    ⚠️ The E-ENC decision metric is *"per-layer P-battery pass rate at MATCHED
    total params"*. Matching by eye is how an arm wins on CAPACITY and gets
    read as winning on ARCHITECTURE — the same confound class as the C6
    "decoder compared on its marginal". So the matching is done here, MEASURED
    by instantiation, and the chosen width plus the residual gap are returned
    for the run row rather than asserted.

    The default probe grid is the multiples of ``lcm(n_heads, 64)`` in
    ``[lo, hi]`` — attention needs ``d_model % n_heads == 0``, and a grid that
    ignores that silently prunes to a handful of widths and reports a bad match
    as the best one (MEASURED here at n_heads=12: a multiples-of-64 grid leaves
    only 384/768/1152).

    Returns ``(config, report)``; ``report["gap_frac"]`` is the residual
    mismatch as a fraction of the target — quote it, because "matched" with a
    30 % gap is not matched.
    """
    nh = int(base.predictor.n_heads)
    step = nh * 64 // math.gcd(nh, 64)
    grid = tuple(probe) if probe else tuple(
        range(max(step, (lo // step) * step or step), hi + 1, step))
    rows = []
    best, best_gap = None, None
    for d in grid:
        if d % nh:
            continue
        cfg = replace(base, predictor=replace(base.predictor, d_model=d))
        total = V6Stack(cfg).param_report()["total"]
        rows.append({"d_model": d, "total": total,
                     "over_budget": total > cfg.param_budget})
        if total > cfg.param_budget:
            continue
        gap = abs(total - target_total)
        if best_gap is None or gap < best_gap:
            best, best_gap = cfg, gap
    if best is None:
        raise ValueError(f"no probed predictor width fits the budget for "
                         f"target {target_total:,} (grid {grid})")
    return best, {
        "target_total": int(target_total),
        "chosen_d_model": best.predictor.d_model,
        "chosen_total": int(target_total + 0) if best_gap is None else
        int(V6Stack(best).param_report()["total"]),
        "gap": int(best_gap), "gap_frac": best_gap / max(target_total, 1),
        "grid": list(grid), "probed": rows,
        "_evidence_class": "MEASURED (ours; counts at instantiation)"}
