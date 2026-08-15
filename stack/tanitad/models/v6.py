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
    # config + stack
    "V6Config", "V6Stack", "GoalVocabulary", "GoalHead", "GoalConditioner",
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
    "spectrum_report",
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

def spectrum_report(z: Tensor, *, top_k: int = 8) -> dict:
    """Collapse diagnostics for a latent batch ``[n, d]`` (leading dims are
    flattened): participation ratio, entropy effective rank, top-k energy
    share, and the raw n/d so a report can say what it was measured on.

    Both statistics are IMPORTED from ``tanitad.eval.spectral`` — the same
    functions the orthogonality/spectral instruments use, so the O6 training
    monitor and the offline analysis cannot drift apart. Computed on the
    CENTRED covariance eigenvalues (= squared singular values of the centred
    batch): collapse is a statement about VARIANCE directions, and an
    uncentred spectrum would be dominated by the mean vector.

    O6's gate is "effective rank retention ≥ 0.8× across any curriculum phase",
    so what matters is that this is measured EVERY N steps and logged — a
    ratio needs a series, not a single reading.
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
    return {"n": int(n), "d": int(d), "top_k": k,
            "participation_ratio": participation_ratio(eig),
            "effective_rank": effective_rank(sv),
            "top_k_share": share}


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

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

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
               arg_mask: Tensor | None = None) -> Tensor:
        e = self.embed_tokens(ids_or_probs)
        if args is not None:
            if args.shape[-1] != self.n_args:
                raise ValueError(f"args must be [B, {self.n_args}], got "
                                 f"{tuple(args.shape)}")
            a = args.to(e.dtype)
            if arg_mask is not None:
                a = a * arg_mask.to(e.dtype)
            e = e + self.arg_proj(a)
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
        return {"logits": logits, "args": self.arg_head(h),
                "probs": logits.softmax(dim=-1)}


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
                arg_mask: Tensor | None = None) -> Tensor:
        return self.proj(self.vocab.encode(ids_or_probs, args, arg_mask))


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
    d_plan_feat: int = 256         # z_op -> emission feature projection

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
            kappa_max=cfg.kappa_max, dt=cfg.dt)

        # ---- O3 aux + O6 ---------------------------------------------------
        self.masked_cells = MaskedCellPredictor(
            cfg.n_cells, int(cfg.readout.d_readout), hidden=cfg.aux_hidden)
        self.sigreg = SigReg(cfg.sigreg_slices, cfg.sigreg_beta)
        #: per-cell nominal ranges for O2 (ESTIMATED prior — see the function).
        self.register_buffer("cell_ranges_m",
                             readout_grid_ranges(gh, gw).reshape(-1),
                             persistent=False)

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
        ("adapter_str.", "layer_str"), ("predictor_str.", "layer_str"),
        ("goal_head_str.", "layer_str"), ("act_head_str.", "layer_str"),
        ("vocab_str.", "layer_str"), ("vocab_a_str.", "layer_str"),
        ("ema_adapter_str.", "layer_str"),
        ("cond_op.", "planner"), ("plan_proj.", "planner"),
        ("cand_queries.", "planner"), ("emission.", "planner"),
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
        return {"a": a_ctl, "kappa": kappa,
                "controls": torch.stack([a_ctl, kappa], dim=-1),
                "waypoints": wp, "feat": feat}

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
        e_g_tac = self.cond_op(g_tac["probs"], g_tac["args"])
        e_a_tac = torch.cat(
            [self.vocab_a_lat.encode(a_lat["probs"], a_lat["args"]),
             self.vocab_a_lon.encode(a_lon["probs"], a_lon["args"])], dim=-1)

        # ---- each layer's predictor rolls under its OWN action --------------
        zh_str = self.predictor_str(z_str, e_a_str)
        zh_tac = self.predictor_tac(z_tac, e_a_tac)
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

        return {
            "z_op_win": z_op_win, "z_op": z_op, "z_plan": z_plan,
            "z_tac": z_tac, "z_tac_target": z_tac_tgt,
            "z_str": z_str, "z_str_target": z_str_tgt,
            "g_str": g_str, "a_str": a_str, "g_tac": g_tac,
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
