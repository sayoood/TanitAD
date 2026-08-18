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
from tanitad.models.agent_slots import (N_QUERIES_DEFAULT, AgentSlotDecoder,
                                        SlotDecodeRanges)
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
    # ⭐ the three gated diagram cells (2026-08-16): proposals / MPC / fallback
    "DiffusionProposalGenerator", "MpcRefiner", "FallbackTrigger",
    "P7_GATE_RHO",
    # ⭐ F-18 — the PERCEPTION agent-slot decoder (re-exported from its own
    # module so `from tanitad.models.v6 import ...` reaches the whole v6
    # surface; the implementation lives in tanitad/models/agent_slots.py)
    "AgentSlotDecoder",
    # ⭐ F-7 / catalog T2 — manoeuvre contrastives (label-free)
    "ManoeuvreContrastiveHead", "lane_mirror_window", "time_reverse_window",
    "photometric_jitter_window", "T2_AUGMENTATIONS",
    "T2_MANOEUVRE_PRESERVING", "T2_MANOEUVRE_REVERSING",
    "IsolationViolation", "PARAM_BUDGET",
    # horizon (§4b)
    "PLAN_STEPS", "DT", "HORIZON_S", "OP_BAND_S", "TAC_BAND_S",
    # staging (X5)
    "STAGES", "STAGE_GROUPS", "MODULE_GROUPS", "LADDER_UNTRAINED_GROUPS",
    "stage_trainable_groups", "apply_stage_freeze",
    # ⛔ the frozen-external guard (E-XENC-1's live trap)
    "FROZEN_EXTERNAL_FLAG", "FrozenExternalViolation",
    "declare_frozen_external", "frozen_external_prefixes",
    "reassert_frozen_external", "assert_frozen_external",
    # measure primitives (O2/O3/O4/O6)
    "time_to_reach", "time_to_reach_weights", "half_weight_distance_m",
    "readout_grid_ranges", "sample_cell_block_mask", "near_field_band_mask",
    "kinematic_saliency", "saliency_weights", "InteractionSampler",
    # ⭐ F-9 / catalog T3 — the interaction CURRICULUM (zero parameters)
    "multi_agent_kinematic_entropy", "T3Curriculum", "t3_rank_control",
    "T3_MASS_SCALE", "T3_CONTROL_MIN_N",
    # ⭐ F-10 / catalog S3 — the DOMAIN-STRATIFIED MIX (zero parameters)
    "DomainMix", "StratifiedEpisodeSampler", "domain_mix_control",
    "DOMAIN_MIX_CONTROL_MIN_N", "DOMAIN_MIX_MAX_AMPLIFICATION",
    "DOMAIN_MIX_MIN_STRATUM_EPISODES",
    "spectrum_report", "SpectrumAccumulator", "o6_rank_verdict",
    "O6_ADMISSIBLE_CEILING", "O6_RANK_FLOOR",
    # X4 — the O6 spectrum pattern applied PER LAYER (op / tac / str)
    "X4_LAYER_POLICY", "layer_spectrum_policy", "x4_rank_verdict",
    "LayerSpectrumMonitor", "sigreg_trend_verdict",
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
#:
#: ⛔ **F-14 BLOCKER — BOTH NAMED DERIVATION INPUTS ARE UNAVAILABLE ON THIS
#: CORPUS, AND ONE OF THEM IS FORBIDDEN RATHER THAN MERELY MISSING.** The
#: vocabulary above is correct and stays; what follows is what a derivation may
#: NOT be built from, MEASURED 2026-08-18. ⚠️ Read this before wiring any
#: SPEED_BAND supervision — the blocker is the thing that rots when nobody
#: revisits it, so it is recorded where the implementer looks.
#:
#: 1. **sign/OCR — FORBIDDEN.** The SAM3/VLM sign channel is released ONLY as a
#:    presence flag (per-clip at 0.5, per-detection at >=0.70); sign **KIND and
#:    TEXT stay forbidden** (`RETRACTION_LOG.md` C87; `DATA_STRATEGY_REFRESH.md`
#:    :134), and the G1 sign-text gate is **CLOSED at 0/31**
#:    (`MODEL_REGISTRY.md`:3599). A speed-limit prior needs exactly `kind ==
#:    "speed"` AND `text`, i.e. precisely the two forbidden fields.
#:    ⭐ AND THE FAILURE MODE IS THIS CELL'S OWN: the two highest-scoring FALSE
#:    positives are a **dashboard `30` roundel (0.927)** and a hoarding (0.778),
#:    both ABOVE true signs — so a confidence threshold removes the harmless
#:    errors and KEEPS the harmful ones. A dashboard roundel is the EGO
#:    SPEEDOMETER: a sign-derived target speed here would be an **ego echo
#:    arriving through the vision channel**, which a vision-only admissibility
#:    audit does not watch.
#: 2. **corridor speed priors — NO CORRIDOR EXISTS.** PhysicalAI-AV carries no
#:    map or lane graph (dataset card, verbatim: *"we do not include open maps
#:    data"*); `taniteval/corridor.py` says the same in code — its strata are
#:    *"a KINEMATIC signature, never a topology"*.
#:
#: ⚠️ **AND THE ADMISSIBLE-LOOKING SUBSTITUTE IS NOT THIS.**
#: ``tanitad.lake.vtarget.vtarget_guarded`` is a leak-guarded target-speed
#: label, banked for all 2376 parity-train episodes and MEASURED admissible AS
#: A LABEL — but it is **hindsight EGO geometry** (the 85th percentile of the
#: ego's own future free-flow speed), i.e. *"what speed did this driver settle
#: at"*, not *"what speed is permitted here"*. Substituting it silently turns a
#: regulatory prior into a behavioural one, and the two differ exactly where
#: this token matters. It is also measured that on ego inputs NOTHING beats
#: repeating v0's band (0.4066 free vs 0.2465 for the trained classifier), so
#: any SPEED_BAND head must clear that bar from VISION or it is a dead
#: parameter. That measurement (`…/2026-08-04-target-speed/code/
#: vt_band_from_vision.py`) has NEVER BEEN RUN. See
#: `…/incoming/2026-08-18-f10-f14-cells/F10_F14_CELLS.md` §2.
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

# ---------------------------------------------------------------------------- #
# ⭐ v6.1 — TURN_L / TURN_R, APPENDED. Prepared 2026-08-18, NOT the default.
# ---------------------------------------------------------------------------- #
#: ⛔ WHY THIS IS A SECOND TUPLE AND NOT AN EDIT TO THE FIRST.
#: `TACTICAL_LAT_ACTIONS` SIZES LIVE TENSORS: `GoalVocabulary(...).table.weight`
#: is `(n_tokens, d_goal_embed)` and `GoalHead.type_head` is
#: `(n_tokens, hidden)`. The v6F S-W 30k run on Thor holds `vocab_a_lat.
#: table.weight (6, 128)` and `act_head_lat.type_head.weight (6, 256)` in its
#: checkpoint, under a TENSOR-STRICT resume contract. Editing the tuple in place
#: would not disturb the RUNNING process — but it would brick its AUTO-RESUME
#: the moment `supervise_run.sh` restarted it, and that failure would arrive
#: hours later, silently, as a shape mismatch on a 4.6-day run. ⇒ v6.1 is a
#: SEPARATE tuple behind a version switch that DEFAULTS TO v6.0.
#:
#: ⭐ WHY APPEND AND NEVER INSERT. Indices 0-5 keep their meaning exactly, so
#: every existing label, dump and artifact stays valid without re-derivation,
#: and a 6-wide head can be widened to 8 by padding rather than retraining.
#:
#: ⚠️ WHAT THESE TWO TOKENS ARE, AND HOW THEY DIFFER FROM THE OTHER SIX.
#: The other six are LANE-RELATIVE (keep / change / abort / nudge) over a 2-6 s
#: horizon. `TURN_L`/`TURN_R` are JUNCTION TRAVERSAL, and they are the only
#: lateral tokens whose meaning is CONDITIONED ON THE STRATEGIC ROUTE being set:
#: the route says which arm of the junction, this says the ego is traversing it
#: now. That coupling is declared here because it exists in the world — the
#: alternative to naming it was `LANE_KEEP` inside a junction, which is false
#: (there are no lanes to keep), or a permanent abstention on the 186 clips
#: (3.94 % of 4,729) where Alpamayo says "Turn Left"/"Turn Right".
#:
#: ⛔ REPRESENTABLE, NOT SCOREABLE. n = 85 (TURN_L) and 101 (TURN_R) in the
#: Alpamayo corpus; on a 40-episode val split that is ~2 per class. They may be
#: emitted and supervised; any PER-CLASS metric on them must be REFUSED for
#: under-power, exactly as `cost_fidelity` refuses below n = 200. Pinned by
#: `stack/tests/test_tactical_vocab_v61.py`.
TACTICAL_LAT_ACTIONS_V61: tuple[str, ...] = TACTICAL_LAT_ACTIONS + (
    "TURN_L", "TURN_R",
)

#: Per-class metrics on these are refused below this n (representable ≠ scoreable).
TACTICAL_LAT_UNDERPOWERED: frozenset[str] = frozenset({"TURN_L", "TURN_R"})
TACTICAL_LAT_MIN_N_FOR_METRIC: int = 200

TACTICAL_VOCAB_VERSIONS: dict[str, tuple[str, ...]] = {
    "v6.0": TACTICAL_LAT_ACTIONS,
    "v6.1": TACTICAL_LAT_ACTIONS_V61,
}


def tactical_lat_actions(version: str = "v6.0") -> tuple[str, ...]:
    """The lateral action vocabulary for ``version``. Default is v6.0 — the
    live run's shape — so importing this module changes nothing."""
    try:
        return TACTICAL_VOCAB_VERSIONS[version]
    except KeyError:
        raise ValueError(
            f"unknown tactical vocabulary version {version!r}; "
            f"known: {sorted(TACTICAL_VOCAB_VERSIONS)}") from None
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
# F-9 / catalog T3 — the INTERACTION CURRICULUM primitives
#
# Spec, two independent locations (established BEFORE a line was written):
#   * `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:66` —
#     *"T3 | interaction curriculum: windows ranked by MULTI-AGENT kinematic
#     entropy measured from the O-layer's own predicted occupancy
#     (self-supervised, after O2/O3 make it non-degenerate) | curriculum from
#     free-flow -> dense interaction | P7 calibration rho >=0.3 held on
#     interaction-rich strata, not just pooled"*
#   * `…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:59` — *"O4 is
#     the ego-kinematic version only; T3's multi-agent extension needs the P8
#     occupancy readout in the loop. Fix F-9 (gated on P8 maturity)"*, and
#     `:214` — *"F-9 | P3 | T3 interaction curriculum (multi-agent entropy from
#     the P8 occupancy readout) — gated on P8 maturity."*
#
# ⛔ WHAT T3 IS *NOT*: it is not O4. O4 scores EGO kinematics from the action
# channels; T3 scores OTHER AGENTS' motion from the O-layer's own predicted
# occupancy, and it is a CURRICULUM (a schedule) where O4 is a static weight.
# Both halves of that sentence are load-bearing and both are built here.
#
# ⭐ ZERO PARAMETERS. Nothing below is an `nn.Module`; the default build is
# untouched at 87,893,449 params / 405 keys (MEASURED, see the test suite).
# ============================================================================

#: The occupancy MASS (summed absolute per-cell change over the rollout) at
#: which the mass gate reaches 1 - 1/e. Units are "raster cells of occupancy
#: probability changed", so this is a scene-scale constant, not a tuning knob.
T3_MASS_SCALE: float = 4.0

#: ⛔ Below this many windows :func:`t3_rank_control` REFUSES a verdict.
#: Same discipline as ``T2_CONTROL_MIN_N``: a separation ratio quoted without
#: its ``n`` is noise wearing a number's clothes (MEASURED for T2 — at n=4 the
#: null ratio spanned 0.397-3.361).
T3_CONTROL_MIN_N: int = 32


def multi_agent_kinematic_entropy(occ: Tensor, *,
                                  mass_scale: float = T3_MASS_SCALE,
                                  eps: float = 1e-12) -> Tensor:
    """Catalog T3's ranking signal: **multi-agent kinematic entropy** of an
    occupancy rollout.

    ``occ`` — ``[B, K, H, W]`` occupancy **probabilities** (not logits) over
    ``K >= 2`` ticks, i.e. ``sigmoid(decode(z_hat_{t+k}))`` from the P8 readout
    (``scripts/train_p8_occupancy.py``). Returns ``[B]`` in ``[0, 1]``.

    The definition, and why each piece is there:

    1. **kinematic** — the signal is the per-cell occupancy CHANGE across
       consecutive ticks, never a static snapshot. A car parked at the kerb is
       not an interaction; a car crossing the ego's path is.
    2. **multi-agent** — Shannon entropy over cells of that change mass,
       normalised by ``log(H*W)``. One agent moving concentrates the mass
       (low entropy); several agents moving in several places spread it (high).
    3. ⛔ **the MASS GATE, which is the whole reason this is not a one-liner.**

    ⛔ **THE DEGENERATE READING, MEASURED AND REJECTED.** The obvious
    implementation — Shannon entropy of the normalised occupancy raster — is
    **MAXIMAL ON AN EMPTY ROAD**. Normalising a near-zero field divides noise
    by noise and yields a near-UNIFORM distribution, whose entropy is the
    maximum the functional can return. A curriculum ranked by it would drive
    training *towards* empty scenes while its name said the opposite. Pinned by
    ``test_bare_spatial_entropy_is_maximal_on_an_empty_raster`` — the failure is
    demonstrated on the bare functional, and then shown to be absent here.

    The gate ``1 - exp(-M / mass_scale)`` is **exactly 0 at zero mass**, so an
    empty scene scores exactly 0 whatever its entropy is. This is the same
    discipline as F-8's flat-plan control: know your functional's degenerate
    input before you rank a corpus with it.

    ⚠️ **This function does NOT decide admissibility.** The occupancy raster it
    consumes comes from a decoder trained on the obstacle join — a LABEL path.
    Using it as a training-time SELECTOR is admissible only as a declared data
    mix (the F-10 precedent: *"admissible for the data MIX (it is not a model
    input) but must be declared"*, DIAGRAM_CONFORMANCE.md:57). The declaration
    is enforced trainer-side by the provenance stamp, not here.
    """
    if occ.dim() != 4:
        raise ValueError(
            f"⛔ occupancy must be [B, K, H, W], got {tuple(occ.shape)}. A "
            f"[B, H, W] snapshot has no kinematics to measure — T3's signal is "
            f"the CHANGE across ticks, and a single tick cannot express one.")
    b, k, h, w = occ.shape
    if k < 2:
        raise ValueError(
            f"⛔ multi-agent KINEMATIC entropy needs K >= 2 ticks, got K={k}. "
            f"With one tick the change field is empty and the score would be "
            f"0 for every window — a ranking signal that ranks nothing.")
    lo, hi = float(occ.min()), float(occ.max())
    if lo < -1e-4 or hi > 1.0 + 1e-4:
        raise ValueError(
            f"⛔ occupancy must be PROBABILITIES in [0, 1], got [{lo:.4g}, "
            f"{hi:.4g}] — this looks like raw logits. P8 emits logits; apply "
            f"`sigmoid` at the call site. Entropy of a logit field is not a "
            f"quantity anyone specified.")
    if mass_scale <= 0:
        raise ValueError(f"mass_scale must be > 0, got {mass_scale}")
    o = occ.float()
    # 1. kinematics: absolute per-cell change, accumulated over the rollout
    m = (o[:, 1:] - o[:, :-1]).abs().sum(dim=1).reshape(b, h * w)   # [B, HW]
    mass = m.sum(dim=-1)                                            # [B]
    # 2. entropy of WHERE the motion is, in [0, 1]
    p = m / mass.clamp_min(eps).unsqueeze(-1)
    ent = -(p * (p + eps).log()).sum(dim=-1) / math.log(h * w)
    # 3. the mass gate — exactly 0 when nothing moved, so an empty raster
    #    cannot inherit the maximal entropy of normalised noise
    gate = 1.0 - torch.exp(-mass / mass_scale)
    return (gate * ent).clamp(0.0, 1.0)


@dataclass(frozen=True)
class T3Curriculum:
    """Catalog T3's *"curriculum from free-flow -> dense interaction"*.

    Maps training ``progress`` in ``[0, 1]`` to the exponent of the same
    ``(floor + s) ** alpha`` weighting O4 uses, ramped linearly from
    ``alpha_start`` to ``alpha_end`` over the first ``warmup_frac`` of training
    and held at ``alpha_end`` thereafter.

    ⭐ **WHY A SIGNED EXPONENT, AND WHY THIS IS NOT** :func:`saliency_weights`.
    Read literally, *"free-flow -> dense"* is the ordinary easy->hard
    curriculum: **start biased towards free flow**, end biased towards dense
    interaction. Biasing *towards* the low-score end requires ``alpha < 0``, and
    :func:`saliency_weights` refuses that. **The O4 guard was NOT weakened to
    make room for T3** — this is T3's own weighting, so O4's contract is
    exactly what it was. (F-7's lesson: do not edit a shared module to fit a new
    cell into it.)

    ``alpha_start = 0`` is *uniform*, not free-flow-biased; the default
    ``-1.0 -> +1.0`` is the honest reading of the catalog row. A run wanting the
    "no curriculum" control sets ``alpha_start == alpha_end`` and gets a static
    weighting — which is O4's shape with T3's score, and is a legitimate arm.

    ⛔ ``floor > 0`` is REQUIRED here, not merely advisable: with ``alpha < 0``
    a zero floor makes the weight of a zero-score window **infinite**. It is
    also what keeps every window reachable at every alpha, so the curriculum
    reweights the draw and never re-selects the corpus (the parity invariant).
    """

    alpha_start: float = -1.0
    alpha_end: float = 1.0
    warmup_frac: float = 0.5
    floor: float = 0.25

    def __post_init__(self) -> None:
        if not (0.0 < self.warmup_frac <= 1.0):
            raise ValueError(
                f"⛔ warmup_frac must be in (0, 1], got {self.warmup_frac}. At "
                f"0 the ramp is a step change at step 0, which is not a "
                f"curriculum; above 1 it never reaches alpha_end at all.")
        if self.floor <= 0:
            raise ValueError(
                f"⛔ floor must be > 0, got {self.floor} — with alpha < 0 a "
                f"zero floor makes a zero-score window infinitely likely, and "
                f"a floor of 0 also stops every window being reachable, which "
                f"is the one thing parity forbids.")
        if self.alpha_end < self.alpha_start:
            raise ValueError(
                f"⛔ alpha_end ({self.alpha_end}) < alpha_start "
                f"({self.alpha_start}) is the catalog row REVERSED: it "
                f"curricula from dense interaction towards free flow. If that "
                f"arm is wanted it must be declared as such, not reached by "
                f"swapping two numbers.")

    def alpha_at(self, progress: float) -> float:
        """The exponent in force at ``progress`` (0 = step 0, 1 = last step)."""
        if not (0.0 <= progress <= 1.0):
            raise ValueError(f"progress must be in [0, 1], got {progress}")
        f = min(1.0, progress / self.warmup_frac)
        return float(self.alpha_start + f * (self.alpha_end - self.alpha_start))

    def weights_at(self, scores: Tensor, progress: float) -> Tensor:
        """Per-window sampling weights at ``progress``, normalised to sum 1."""
        s = scores.float().clamp_min(0.0)
        w = (self.floor + s) ** self.alpha_at(progress)
        return w / w.sum().clamp_min(1e-12)


def t3_rank_control(scores: Tensor, dense: Tensor, *,
                    min_n: int = T3_CONTROL_MIN_N) -> dict:
    """⛔ **The trivial-proxy control for T3's ranking signal.**

    ``scores`` ``[n]`` T3 entropies, ``dense`` ``[n]`` bool — the independent
    "this window really is interaction-rich" judgement (e.g. the obstacle-join
    agent count above a threshold, held out of the scorer). Returns the mean
    score on each side, their ratio, both SEMs, and a ``verdict``.

    ⭐ **WHY IT EXISTS.** C92: a headline died because a readout was echoing ego
    speed. "The curriculum weighted some windows more" is **not** evidence that
    T3 found interaction — the ratio is. And the specific failure this catches
    is the one :func:`multi_agent_kinematic_entropy` documents: a functional
    that is maximal on an empty road would come back with ``ratio < 1`` here,
    i.e. **inverted**, while every other log line looked healthy.

    ⛔ **REFUSES below ``min_n`` per side.** MEASURED for the sibling T2 control
    at random init, where the true ratio is 1 by construction: at n=4 the null
    ratio spanned **0.397-3.361**. A verdict from a handful of windows is noise.
    """
    if scores.dim() != 1 or dense.dim() != 1:
        raise ValueError(f"scores {tuple(scores.shape)} and dense "
                         f"{tuple(dense.shape)} must both be 1-D")
    if scores.shape != dense.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and dense "
                         f"{tuple(dense.shape)} must align 1:1")
    d = dense.bool()
    s = scores.float()
    n_d, n_f = int(d.sum()), int((~d).sum())
    out = {"n_dense": n_d, "n_free": n_f, "min_n": int(min_n)}
    if n_d < min_n or n_f < min_n:
        out["verdict"] = "REFUSED_TOO_FEW"
        out["_note"] = (f"⛔ need >= {min_n} windows per side, have "
                        f"dense={n_d} free={n_f}. A ratio quoted without its n "
                        f"is not a verdict.")
        return out
    sd, sf = s[d], s[~d]
    m_d, m_f = float(sd.mean()), float(sf.mean())
    out |= {
        "mean_dense": m_d, "mean_free": m_f,
        "sem_dense": float(sd.std(unbiased=True) / math.sqrt(n_d)),
        "sem_free": float(sf.std(unbiased=True) / math.sqrt(n_f)),
        "ratio": m_d / m_f if m_f > 0 else float("inf"),
    }
    if m_f <= 0 and m_d <= 0:
        out["verdict"] = "DEGENERATE_ALL_ZERO"
        out["_note"] = ("⛔ every window scored 0 — the signal ranks nothing. "
                        "Check the occupancy rollout is not empty.")
    elif out["ratio"] < 1.0:
        out["verdict"] = "INVERTED"
        out["_note"] = ("⛔ free-flow windows score HIGHER than dense ones. "
                        "This is the empty-road-is-maximal-entropy failure; a "
                        "curriculum on this signal trains the opposite of T3.")
    else:
        out["verdict"] = "OK"
    return out


# ============================================================================
# F-10 / catalog S3 — the DOMAIN-STRATIFIED TRAINING MIX
#
# Spec, two independent locations (established BEFORE a line was written):
#   * `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:81` —
#     *"S3 | domain-stratified training mix (geographic/domain diversity beats
#     volume — arXiv 2607.04500) | the S1 scaling-ladder data-mix arm folds in
#     here | cross-domain P-battery deltas reported per stratum"*
#   * `…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:69` — *"domain-
#     diverse mix (catalog S3) | NOT BUILT | no domain-stratified sampling in
#     `train()` (episode draw is uniform / O4-weighted only). Needs the
#     VLM/scena strata as a SAMPLER input — which is admissible for the data
#     MIX (it is not a model input) but must be declared. Fix F-10"*, and
#     `:215` — *"F-10 | P3 | S3 domain-stratified mix — VLM strata as SAMPLER
#     input (admissible: data mix, not a model input; declare it)."*
#
# ⛔ THE ARCHITECTURAL FINDING THAT DETERMINED THIS IMPLEMENTATION — MEASURED,
# NOT REASONED. F-10 acts on the **EPISODE** axis and it MUST, because
# :class:`InteractionSampler` draws episodes UNIFORMLY and consults ``weights``
# only *within* the drawn episode (``__call__``: ``pick = torch.randint(...)``
# then ``w = self.weights[pool]``). A domain label is an EPISODE property, so
# expressed as a per-window weight it is CONSTANT inside every episode — and a
# constant vector through ``torch.multinomial`` is exactly uniform.
#
# MEASURED (`test_v6_domain_mix.py::
# test_a_per_window_domain_weight_is_EXACTLY_a_noop_on_the_episode_mix`): a
# domain-balanced per-window weight and an all-ones weight produce the
# **bit-identical draw sequence** over 4,000 draws, and the achieved domain
# share stays at the corpus proportion (0.675/0.325 against a 0.5/0.5 target).
# ⇒ **A per-window domain weight is a TERM OVER AN INVARIANT — a no-op wearing
# the name of the lever.** Same class as C115 (a loss over `z_tac`'s
# non-existent temporal extent). This is why F-10 introduces its own episode
# draw instead of reusing O4's weight vector.
#
# ⛔ AND THE OBVIOUS "DIVERSITY" OBJECTIVE IS MAXIMAL AT TWO OPPOSITE
# DEGENERATE INPUTS (the C119 shape). Perfect stratum balance is achieved, at
# every temperature, BOTH when there is exactly ONE stratum and when EVERY
# EPISODE IS ITS OWN STRATUM — and both are exactly the uniform draw. A balance
# metric therefore cannot detect either failure; only
# :meth:`DomainMix.report`'s amplification and ``n_eff_episodes`` can. Both are
# refused by :meth:`DomainMix.episode_weights`, not merely documented.
#
# ⛔ ZERO PARAMETERS. A sampling mix is a schedule over the data, not a module:
# nothing here is an ``nn.Module``, nothing enters any ``state_dict``, and the
# default build stays at 87,893,449 params / 405 keys (MEASURED by BUILDING
# through ``build_stack_from_args`` — see the test suite).
# ============================================================================

#: ⛔ Below this many episodes per side :func:`domain_mix_control` REFUSES a
#: verdict and returns no ratio at all. Same discipline (and the same measured
#: rationale) as ``T2_CONTROL_MIN_N`` / ``T3_CONTROL_MIN_N``: at n=4 a null
#: ratio spanned 0.397-3.361, so a verdict there is noise wearing a number's
#: clothes.
DOMAIN_MIX_CONTROL_MIN_N: int = 32

#: ⛔ The default ceiling on how much more often the MOST up-weighted episode
#: may be drawn than under a uniform draw. It is the guard the catalog row's
#: own headline demands: *"diversity beats volume"* is a TRADE, and balancing a
#: stratum of 5 episodes against one of 2,000 spends the corpus on those 5.
#:
#: **MEASURED on the parity corpus size (N = 2376), at tau = 1 (full balance)**
#: — the calibration is a measurement, not a guess::
#:
#:     shape                  amp@tau=1   n_eff   n_eff_frac   smallest stratum
#:     6-stratum realistic       11.00    650.6      0.274            36
#:     3-stratum coarse           4.50   1030.1      0.434           176
#:     10-stratum fine           14.85    705.6      0.297            16
#:     long-tail 12              24.75    339.0      0.143             8
#:
#: 20x ADMITS full balance on every shape whose smallest stratum holds >= 16
#: episodes, and REFUSES the long-tail shape whose smallest holds 8 — which is
#: exactly the case the cap exists for. Raising it is a declared decision.
DOMAIN_MIX_MAX_AMPLIFICATION: float = 20.0

#: ⛔ A stratum with fewer episodes than this cannot carry an equal share of a
#: 2,376-episode corpus without becoming a memorisation target: at tau = 1 with
#: S strata it receives ``1/S`` of EVERY batch however few episodes it holds.
#: Refused rather than silently amplified. ⚠️ This guard and
#: :data:`DOMAIN_MIX_MAX_AMPLIFICATION` are NOT redundant — MEASURED above, the
#: long-tail-12 shape PASSES this one (its smallest stratum is exactly 8) and
#: is caught only by the amplification ceiling.
DOMAIN_MIX_MIN_STRATUM_EPISODES: int = 8


@dataclass(frozen=True)
class DomainMix:
    """Catalog S3's *"domain-stratified training mix"*, as a TEMPERATURE.

    ``tau`` interpolates the episode draw between the two readings the catalog
    row leaves open:

    * ``tau = 0`` — **PROPORTIONAL**. Stratum mass tracks stratum size, so
      every episode is equally likely. This is the matched CONTROL arm: it runs
      the same code path and consumes the same RNG as a live mix, and its draw
      is distributionally identical to today's uniform episode draw.
      ⚠️ It is NOT *stream*-identical to the incumbent — the incumbent
      (``InteractionSampler``, i.e. ``--domain-strata`` absent) draws episodes
      with ``randint`` and this draws with ``multinomial``. Two different
      controls, both legitimate, and a run must say which it used.
    * ``tau = 1`` — **BALANCED**. Every stratum receives an equal share of the
      draw regardless of how many episodes it holds. This is the literal
      reading of *"domain diversity beats volume"*.

    Formally: stratum ``k`` of size ``n_k`` receives mass ``q_k`` proportional
    to ``n_k ** (1 - tau)``; episodes inside a stratum are equiprobable, so an
    episode's weight is proportional to ``n_k ** (-tau)``.

    ⛔ **WHAT THIS IS NOT: O4 or T3.** Those weight WINDOWS by a saliency score
    and act *inside* an episode. This weights EPISODES by a stratum label and
    acts on the episode draw. The two axes are orthogonal by construction and
    compose without conflation — which is exactly why F-10 is NOT refused
    alongside ``--o4-alpha`` the way ``--t3-scores`` is (T3 and O4 are two
    levers on ONE axis; F-10 is a lever on a DIFFERENT axis).

    ⛔ **PARITY.** This REWEIGHTS the episode draw and removes no episode: with
    ``tau <= 1`` and every stratum non-empty, every episode keeps strictly
    positive probability, so all 2,376 episodes of
    ``physicalai-train-e438721ae894`` remain reachable and cross-arm
    comparability holds. An episode with NO stratum label is REFUSED rather
    than dropped, because dropping it would be a re-selection.
    """

    tau: float = 1.0
    max_amplification: float = DOMAIN_MIX_MAX_AMPLIFICATION
    min_stratum_episodes: int = DOMAIN_MIX_MIN_STRATUM_EPISODES

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(
                f"⛔ tau must be in [0, 1], got {self.tau}. Below 0 the mix "
                f"ANTI-balances (it concentrates on the largest stratum, the "
                f"catalog row reversed); above 1 a small stratum is drawn MORE "
                f"often in total than a large one, which is not a mix but an "
                f"inversion. Either arm must be declared, not reached by "
                f"passing a number out of range.")
        if self.max_amplification < 1.0:
            raise ValueError(
                f"⛔ max_amplification must be >= 1, got "
                f"{self.max_amplification}: below 1 no balancing at all is "
                f"expressible and the lever is inert by construction.")
        if self.min_stratum_episodes < 1:
            raise ValueError(
                f"⛔ min_stratum_episodes must be >= 1, got "
                f"{self.min_stratum_episodes}.")

    # -- the two degenerate stratifications, named ---------------------------
    @staticmethod
    def _refuse_degenerate(n_strata: int, n_episodes: int) -> None:
        """⛔ Both extremes are EXACTLY the uniform draw at EVERY tau.

        A "balance" reading calls both of them perfect, which is why this is a
        refusal and not a warning: the score cannot see it.
        """
        if n_strata < 2:
            raise ValueError(
                f"⛔ {n_strata} distinct stratum over {n_episodes} episodes: a "
                f"one-stratum mix is EXACTLY the uniform draw at every tau "
                f"(q_1 = 1, so every episode weight is 1/n). The lever is "
                f"inert and would be advertised in the launch line while "
                f"changing nothing — the same silent no-op --domain-strata "
                f"exists to avoid. ⚠️ A balance metric reads this as PERFECTLY "
                f"BALANCED; only the fact that it is inert distinguishes it.")
        if n_strata >= n_episodes:
            raise ValueError(
                f"⛔ {n_strata} strata over {n_episodes} episodes — every "
                f"episode is its own stratum, and every n_k = 1, so "
                f"n_k ** (-tau) = 1 and the draw is EXACTLY UNIFORM at every "
                f"tau. ⚠️ This is the OPPOSITE degenerate input to the "
                f"one-stratum case and a balance metric calls it PERFECTLY "
                f"BALANCED too. A stratification is a partition into a few "
                f"domains, not an episode id.")

    def episode_weights(self, strata) -> Tensor:
        """Per-episode draw weights for ``strata`` (one label per episode).

        ``strata`` is a sequence of hashable labels aligned 1:1 with the
        episode list. Returns a ``[n_episodes]`` tensor summing to 1.

        Refuses: an unlabelled episode (``None``/empty — a parity break), the
        two degenerate stratifications above, a stratum below
        ``min_stratum_episodes``, and an amplification above
        ``max_amplification``.
        """
        labels = list(strata)
        n = len(labels)
        if n == 0:
            raise ValueError("⛔ episode_weights got an EMPTY stratum list.")
        missing = [i for i, s in enumerate(labels)
                   if s is None or (isinstance(s, str) and not s.strip())]
        if missing:
            raise ValueError(
                f"⛔ {len(missing)} of {n} episodes carry NO stratum label "
                f"(first at index {missing[0]}). ⛔ They must NOT be dropped: "
                f"dropping an episode is a RE-SELECTION of the corpus, which "
                f"parity forbids (canonical train "
                f"`physicalai-train-e438721ae894`, skip-hash `f09e44db`). "
                f"Either label them or run without --domain-strata.")
        sizes: dict = {}
        for s in labels:
            sizes[s] = sizes.get(s, 0) + 1
        self._refuse_degenerate(len(sizes), n)
        small = {k: v for k, v in sizes.items()
                 if v < self.min_stratum_episodes}
        if small:
            raise ValueError(
                f"⛔ {len(small)} stratum/strata hold fewer than "
                f"{self.min_stratum_episodes} episodes ({small}). Balancing "
                f"them against the rest spends the corpus on a handful of "
                f"clips: at tau=1 each such stratum receives "
                f"1/{len(sizes)} = {1.0 / len(sizes):.4f} of EVERY batch. "
                f"Merge them into an OTHER stratum, or lower tau, but do not "
                f"reach a balanced mix by memorising a few episodes.")
        # q_k proportional to n_k ** (1 - tau); an episode's weight is q_k/n_k.
        qs = {k: float(v) ** (1.0 - self.tau) for k, v in sizes.items()}
        z = sum(qs.values())
        w = torch.tensor([qs[s] / (sizes[s] * z) for s in labels],
                         dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)
        amp = float(w.max()) * n
        if amp > self.max_amplification:
            raise ValueError(
                f"⛔ tau={self.tau} amplifies the most up-weighted episode "
                f"{amp:.2f}x over a uniform draw, above the "
                f"{self.max_amplification:.2f}x ceiling. Stratum sizes "
                f"{dict(sorted(sizes.items(), key=lambda kv: kv[1]))}. "
                f"⚠️ 'diversity beats volume' is a TRADE and this is the price "
                f"side of it: lower tau, merge the small strata, or raise the "
                f"ceiling AS A DECLARED DECISION — but the amplification must "
                f"be a number in the run row, not an accident.")
        return w

    def report(self, strata) -> dict:
        """The mix's diagnostic — what it costs as well as what it buys.

        ⭐ ``n_eff_episodes`` (``1 / sum(w**2)``, the inverse participation
        ratio) is the honest counterpart to the catalog row's *"diversity beats
        volume"*: it is **the volume you paid**. Under a uniform draw it equals
        the episode count exactly; a mix that balances hard collapses it, and
        that collapse is invisible to any stratum-share metric.
        """
        labels = list(strata)
        w = self.episode_weights(labels)
        sizes: dict = {}
        for s in labels:
            sizes[s] = sizes.get(s, 0) + 1
        share: dict = {}
        for s, wi in zip(labels, w.tolist()):
            share[s] = share.get(s, 0.0) + wi
        n = len(labels)
        return {
            "tau": self.tau,
            "n_episodes": n,
            "n_strata": len(sizes),
            "stratum_sizes": {str(k): int(v) for k, v in sizes.items()},
            "stratum_share_corpus": {str(k): round(v / n, 6)
                                     for k, v in sizes.items()},
            "stratum_share_drawn": {str(k): round(v, 6)
                                    for k, v in share.items()},
            "max_amplification": round(float(w.max()) * n, 4),
            "min_amplification": round(float(w.min()) * n, 4),
            "n_eff_episodes": round(
                float(1.0 / (w.pow(2).sum().clamp_min(1e-12))), 2),
            "n_eff_frac": round(
                float(1.0 / (w.pow(2).sum().clamp_min(1e-12))) / n, 4),
            "_reads": ("n_eff_episodes is the VOLUME the mix paid for its "
                       "diversity: it equals n_episodes exactly under a "
                       "uniform draw and collapses as tau balances. No "
                       "stratum-share metric can see this number."),
        }


def domain_mix_control(metric_by_episode: Tensor, strata, *,
                       min_n: int = DOMAIN_MIX_CONTROL_MIN_N) -> dict:
    """⛔ **The trivial-proxy control for a stratification's INFORMATIVENESS.**

    ``metric_by_episode`` ``[n]`` any per-episode quantity the mix is supposed
    to diversify over (episode length, mean speed, a P-battery per-episode
    score); ``strata`` the labels. Returns the between-stratum spread against
    the within-stratum spread and a ``verdict``.

    ⭐ **Why this control exists.** A stratification that CUTS ACROSS the
    quantity of interest re-weights the corpus without changing what the model
    sees — the mix moves the labels around and the distribution of the thing
    that matters is unmoved. That failure is invisible to every balance metric
    (the strata ARE balanced; they are just uninformative), and it is the same
    family as C119: a score that looks healthy on exactly the input that makes
    it meaningless.

    ⛔ Refuses below ``min_n`` episodes and returns NO ratio, so there is no
    number to quote out of context.
    """
    m = metric_by_episode.detach().float().flatten()
    labels = list(strata)
    if m.numel() != len(labels):
        raise ValueError(f"⛔ metric ({m.numel()}) and strata ({len(labels)}) "
                         f"must align 1:1")
    n = int(m.numel())
    if n < min_n:
        return {"verdict": "REFUSED_TOO_FEW",
                "n": n, "min_n": int(min_n), "ratio": None,
                "_note": (f"⛔ {n} episodes < {min_n}: no ratio is returned at "
                          f"all, so none can be quoted. A separation ratio "
                          f"without its n is noise wearing a number's "
                          f"clothes.")}
    groups: dict = {}
    for lab, val in zip(labels, m.tolist()):
        groups.setdefault(lab, []).append(val)
    if len(groups) < 2:
        return {"verdict": "DEGENERATE_ONE_STRATUM", "n": n, "ratio": None,
                "_note": ("⛔ one stratum: there is no between-group spread to "
                          "measure, and the mix is the uniform draw.")}
    means = torch.tensor([sum(v) / len(v) for v in groups.values()])
    between = float(means.std(unbiased=len(means) > 1))
    within_parts = [torch.tensor(v).std(unbiased=True).item()
                    for v in groups.values() if len(v) > 1]
    within = float(sum(within_parts) / len(within_parts)) if within_parts \
        else 0.0
    sem = {str(k): (float(torch.tensor(v).std(unbiased=True)
                          / math.sqrt(len(v))) if len(v) > 1 else None)
           for k, v in groups.items()}
    ratio = None if within <= 1e-12 else between / within
    out = {"n": n, "n_strata": len(groups),
           "stratum_n": {str(k): len(v) for k, v in groups.items()},
           "stratum_mean": {str(k): round(sum(v) / len(v), 6)
                            for k, v in groups.items()},
           "stratum_sem": sem,
           "between_std": round(between, 6), "within_std": round(within, 6),
           "ratio": None if ratio is None else round(ratio, 6),
           "min_n": int(min_n)}
    if any(len(v) < 2 for v in groups.values()):
        out["verdict"] = "UNDERPOWERED_STRATUM"
        out["_note"] = ("⚠️ at least one stratum holds a single episode: its "
                        "within-spread is undefined and its mean is one "
                        "sample.")
    elif ratio is None:
        out["verdict"] = "DEGENERATE_ZERO_WITHIN"
        out["_note"] = ("⛔ within-stratum spread is zero — the metric is "
                        "constant inside every stratum, which makes the ratio "
                        "undefined rather than infinite.")
    elif ratio < 0.1:
        out["verdict"] = "UNINFORMATIVE"
        out["_note"] = ("⛔ the strata barely separate this metric "
                        "(between/within < 0.1): the mix re-weights the "
                        "corpus without changing the distribution of the "
                        "thing it is supposed to diversify. A balance metric "
                        "cannot see this — the strata ARE balanced.")
    else:
        out["verdict"] = "OK"
    return out


class StratifiedEpisodeSampler(InteractionSampler):
    """F-10's episode draw: strata-weighted episodes, unchanged windows.

    ⛔ **WHY A SUBCLASS AND NOT AN EDIT TO** :class:`InteractionSampler`.
    O4's contract is *"episodes are drawn uniformly so no episode is
    starved"* — that is a real guarantee other arms rely on, and F-10 is
    exactly the arm that must break it. Editing the shared class to make room
    for a new cell is the mistake F-7 recorded (and T3 avoided by carrying its
    own weighting). ``InteractionSampler`` is byte-unchanged; only the
    ``pick`` line is overridden here.

    The **window** draw is inherited untouched, so an F-10 mix composes with
    O4's or T3's window saliency on a genuinely different axis.

    ⚠️ RNG: this draws episodes with ``multinomial`` where the base class uses
    ``randint``. At ``tau = 0`` the two are distributionally identical but NOT
    stream-identical, so the byte-identical control is *"no ``--domain-strata``
    at all"* and the matched-path control is ``tau = 0``. Both are real; a run
    must say which it used.
    """

    def __init__(self, index, weights: Tensor, ep_weights, *,
                 eps_per_batch: int = 4,
                 generator: torch.Generator | None = None):
        super().__init__(index, weights, eps_per_batch=eps_per_batch,
                         generator=generator)
        missing = [e for e in self.ep_ids if int(e) not in ep_weights]
        if missing:
            raise ValueError(
                f"⛔ {len(missing)} of {len(self.ep_ids)} episodes in the "
                f"dataset index have NO mix weight (first: {missing[0]}). An "
                f"episode with no weight would be unreachable, and an "
                f"unreachable episode is a corpus RE-SELECTION.")
        w = torch.tensor([float(ep_weights[int(e)]) for e in self.ep_ids],
                         dtype=torch.float32)
        if float(w.min()) <= 0.0:
            raise ValueError(
                f"⛔ {int((w <= 0).sum())} episode weight(s) are <= 0. Every "
                f"episode must stay strictly reachable — parity forbids "
                f"re-selecting the corpus.")
        self.ep_weights = w / w.sum().clamp_min(1e-12)

    def __call__(self, bs: int) -> list[int]:
        n_ep = min(self.eps_per_batch, len(self.ep_ids))
        pick = torch.multinomial(self.ep_weights, n_ep, replacement=True,
                                 generator=self.gen)
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
# X4 — the O6 spectrum pattern applied PER LAYER (op / tac / str)
# ============================================================================
# The diagram's X4 row is "per-layer SIGReg + per-layer spectrum monitors (O6
# pattern at T/S scale) | rank retention per layer" (V6_TRAINING_MEASURES.md).
# Until 2026-08-16 only z_op was monitored. The machinery below extends the
# SPECTRUM half to z_tac / z_str; the per-layer SIGREG half remains
# unimplemented in the trainer (SigReg is applied to z_op only) and the records
# SAY SO rather than inventing a loss series — see :func:`sigreg_trend_verdict`.
#
# ⛔ THE ONE THING THAT MUST NOT BE COPIED FROM z_op IS ITS CONSTANTS. The
# higher layers uplink ONLY the window's LAST frame (`V6Stack.forward`:
# ``z_op = z_op_win[:, -1]``), so one step contributes B rows (8 on the live
# geometry), not B*W (48) — the per-batch rank ceiling is **7**, not 47 — and
# their d is 4x / 8x smaller, so z_op's ceiling_min = 1024 EXCEEDS what a
# centred covariance over z_str (d=256) can EVER reach. Copying it would make
# the strategic layer INCONCLUSIVE FOREVER by construction. Each layer carries
# its own measured pair below.

#: Per-layer admissibility ceiling + absolute floor for the X4 rank verdict.
#: MEASURED (ours) — `…/incoming/2026-08-16-x4-p9/code/x4_layer_power.py`,
#: artifact `raw/x4_layer_power.json`, CPU, seeded, through the REAL
#: `spectrum_report`/`o6_rank_verdict`; generative model inherited from
#: SIGREG_GATE_POWER.md's calibrated regime (power-law alpha=2, rho_ep=0.5).
#: Selection rules were PRE-COMMITTED in that script before any number:
#:
#:   ceiling_min = largest power of two <= d/2 (cap 1024), raised while the
#:                 measured pair-FP of the 0.8x point criterion is > 0 or the
#:                 healthy/collapsed separation is < 3x at the first reachable
#:                 pool;
#:   floor       = smallest power of two >= geomean(healthy, collapsed) at the
#:                 recommended pool.
#:
#: ANCHOR CHECK: both rules REPRODUCE z_op's shipped pair exactly (1024 / 64,
#: with healthy@pool32 = 121.4 vs the banked 121.57 and pair-FP@48rows = 10.7 %
#: vs SIGREG_GATE_POWER's 11.0 % lower bound), so the new layers extend the
#: O6 derivation rather than forking it. Measured selection rows:
#:
#:   tac (d=512): ceiling_min 256 — healthy 65.98 vs collapsed 11.19 at
#:                ceiling 263 (separation 5.89x, FP 0.000); floor 32
#:                (margins: healthy/floor 2.06x, floor/collapsed 2.86x)
#:   str (d=256): ceiling_min 128 — healthy 59.49 vs collapsed 11.03 at
#:                ceiling 256 (separation 5.40x, FP 0.000); floor 32
#:                (margins: 1.86x / 2.90x)
#:
#: ⚠️ ACHIEVABILITY: at 8 rows/step, ``--spectrum-accum 32`` reaches ceiling
#: 255 — ONE ROW SHORT of tac's 256. **The recommended accum is therefore 33**
#: (33*8-1 = 263; also lifts z_op's pooled ceiling 1535 -> 1583). 32 leaves
#: tac INCONCLUSIVE on clause 1, which is correct and is not a pass.
X4_LAYER_POLICY: dict[str, dict] = {
    "op": {"d": 2048, "ceiling_min": O6_ADMISSIBLE_CEILING,
           "floor": O6_RANK_FLOOR,
           "basis": "SIGREG_GATE_POWER.md §5 (pre-registered; unchanged — "
                    "x4_layer_power.py reproduces both numbers)"},
    "tac": {"d": 512, "ceiling_min": 256, "floor": 32.0,
            "basis": "MEASURED x4_layer_power.json (sep 5.89x, FP 0.000 at "
                     "ceiling 263; floor margins 2.06x/2.86x)"},
    "str": {"d": 256, "ceiling_min": 128, "floor": 32.0,
            "basis": "MEASURED x4_layer_power.json (sep 5.40x, FP 0.000 at "
                     "ceiling 256; floor margins 1.86x/2.90x)"},
}

#: The accum that makes ALL THREE layers' clause-1 admissibility reachable
#: (see the achievability note above): min over layers of the smallest N with
#: N*rows_per_step - 1 >= ceiling_min, at the live 8-rows/step geometry.
X4_RECOMMENDED_ACCUM = 33


def layer_spectrum_policy(layer: str, d: int) -> dict:
    """The (ceiling_min, floor) pair for ``layer`` at runtime dimension ``d``.

    Returns the MEASURED constants when ``d`` matches the dimension they were
    sized at. For any other ``d`` (a re-configured stack) it returns the RULE
    ceiling (largest power of two <= d/2, cap 1024) and **no floor** — a floor
    nobody measured must not fail a gate, so ``floor=None`` disables clause 3
    and the verdict stamps that the number is missing rather than inventing
    one (the C76 rule: a threshold ships with the FP rate it achieves, or it
    does not ship).
    """
    pol = X4_LAYER_POLICY.get(layer)
    if pol is not None and int(d) == pol["d"]:
        return dict(pol)
    return {"d": int(d),
            "ceiling_min": min(2 ** int(math.floor(math.log2(max(d / 2, 2)))),
                               O6_ADMISSIBLE_CEILING),
            "floor": None,
            "basis": f"RULE ONLY — d={d} does not match the measured "
                     f"dimension for layer {layer!r} "
                     f"({pol['d'] if pol else 'none on record'}); run "
                     f"x4_layer_power.py at this d before gating on a floor"}


def x4_rank_verdict(layer: str, d: int, cur: dict,
                    ref: dict | None = None, *,
                    retention: float = 0.8) -> dict:
    """Per-layer rank verdict — :func:`o6_rank_verdict` under the LAYER'S OWN
    measured policy, stamped with the layer and the policy's basis.

    Same three clauses, same INCONCLUSIVE capability. When the policy carries
    no measured floor (non-shipped ``d``), clause 3 is DISABLED and the record
    says so — an unmeasured floor that can fail a run is worse than no floor.
    """
    pol = layer_spectrum_policy(layer, d)
    floor_missing = pol["floor"] is None
    v = o6_rank_verdict(cur, ref, retention=retention,
                        floor=(0.0 if floor_missing else float(pol["floor"])),
                        ceiling_min=int(pol["ceiling_min"]))
    v["criterion"] = "X4_rank_retention v1 (per-layer; o6_rank_verdict under " \
                     "the layer's measured policy — X4_P9.md)"
    v["layer"] = str(layer)
    v["d"] = int(d)
    v["policy_basis"] = pol["basis"]
    if floor_missing:
        v["absolute_floor"] = None
        v["floor_note"] = ("clause 3 DISABLED: no measured floor for this "
                           "(layer, d); the verdict can still FAIL on "
                           "retention (clause 2) but never on the floor")
    return v


def sigreg_trend_verdict(baseline: "list[float]", current: "list[float]", *,
                         z_fire: float = 8.0, min_baseline: int = 64,
                         min_current: int = 16) -> dict:
    """The BASELINED ``o6`` LOSS-TREND guard — the collapse alarm that is
    already in every log at zero cost.

    ⭐ WHY THE LOSS AND NOT THE RANK, at small n. MEASURED
    (O6_ABLATION_AND_MASK_PROBE.md §4.1, `raw/sigreg_response.json`): on the
    SAME 48-row batch a 2x collapse (2048 -> 1024 retained dims) moves
    ``effective_rank`` **0.3 %** (46.86 -> 46.73, buried in its own noise) and
    moves the ``o6_sigreg`` loss **3.05x — 42.4 sigma** (0.4023 +- 0.0195 ->
    1.2283). The monitor is nearly blind exactly where the loss term is
    exquisitely sensitive, so between pooled spectrum readings the loss trend
    is the higher-power guard.

    Direction: collapse RAISES ``o6`` (the latent moves away from the
    isotropic-Gaussian target), so the guard fires on a sustained RISE only —
    a falling ``o6`` is the regulariser being optimised, never an alarm.

    Robust statistics (median / MAD), so single-step spikes cannot fire it;
    ``z_fire=8`` sits far above healthy noise and far below the measured
    42.4-sigma collapse response. INCONCLUSIVE below the sample minima, and
    the scale is floored at 2 % of |median| (stamped when it binds) so a
    degenerate zero-variance baseline cannot manufacture an infinite z —
    at the floor the guard needs a >= 16 % rise to fire, against the
    measured +205 % collapse response (0.4023 -> 1.2283).

    ⚠️ SCOPE, stated to keep this honest: this is a REPORTED training-time
    monitor. Promoting it into a stage-gate criterion is a PI decision
    (O6_ABLATION_AND_MASK_PROBE.md escalation 2), and its false-positive rate
    on the REAL run's series is unmeasured until the Thor records are banked
    (SIGREG_GATE_POWER.md escalation 1). Per-layer versions for z_tac / z_str
    are NOT emitted anywhere because no per-layer SIGReg loss exists — a trend
    guard without a loss series would be an invented instrument.
    """
    nb, nc = len(baseline), len(current)
    out: dict = {"criterion": f"o6 trend guard: median(current {nc}) vs "
                              f"median(baseline {nb}) + {z_fire} x robust_sd",
                 "n_baseline": nb, "n_current": nc, "z_fire": float(z_fire)}
    if nb < min_baseline or nc < min_current:
        out |= {"pass": None, "status": "INCONCLUSIVE",
                "reason": f"needs >= {min_baseline} baseline and >= "
                          f"{min_current} current samples, got {nb}/{nc}"}
        return out
    b = torch.tensor(baseline, dtype=torch.float64)
    c = torch.tensor(current, dtype=torch.float64)
    med = float(b.median())
    mad = float((b - med).abs().median())
    sd = 1.4826 * mad
    sd_floor = max(2e-2 * abs(med), 1e-12)
    scale = max(sd, sd_floor)
    cur_med = float(c.median())
    z = (cur_med - med) / scale
    out |= {"baseline_median": med, "baseline_robust_sd": sd,
            "scale": scale, "scale_floored": bool(sd < sd_floor),
            "current_median": cur_med, "z": z}
    if z >= z_fire:
        out |= {"pass": False, "status": "FAIL",
                "reason": f"o6 median rose {z:.1f} robust-sd above the "
                          f"phase-start baseline ({med:.4f} -> {cur_med:.4f})"
                          f" — the measured signature of collapse is a rise "
                          f"of this shape (42.4 sigma at a 2x collapse)"}
    else:
        out |= {"pass": True, "status": "PASS",
                "reason": f"no collapse-shaped rise: z {z:+.1f} < {z_fire} "
                          f"(falls are the regulariser training and never "
                          f"fire)"}
    return out


class LayerSpectrumMonitor:
    """X4: per-layer spectrum records with per-layer pooling, references and
    verdicts — the O6 machinery applied to ``{"tac": z_tac, "str": z_str}``
    (and to any latent dict; the trainer keeps z_op on its incumbent path so
    the live run's records stay byte-identical).

    Owns, PER LAYER: a :class:`SpectrumAccumulator` ring (when ``accum > 1``),
    the phase-start reference (taken at the first reading whose ceiling clears
    the LAYER'S OWN ``ceiling_min``), and the :func:`x4_rank_verdict`.

    ⚠️ CLUSTER UNIT, stated. z_tac / z_str rows are one-per-window and the
    live sampler INTERLEAVES episodes across the batch ([e0,e1,e2,e3,e0,...]),
    so contiguous-row blocks cannot express the episode grouping. The pooled
    jackknife therefore uses THE STEP as the cluster (``block = rows_per_
    step``) — every row in a block genuinely shares the step's 4 episodes, so
    the clustering is conservative (over-grouped), which for a guard that
    fires only when the interval EXCLUDES the threshold is the safe direction.
    The per-batch (unpooled) reading has only one step and falls back to
    ``block=1``, stamped, which UNDERSTATES episode correlation — one more
    reason the per-batch reading is INCONCLUSIVE-only under clause 1.

    ⚠️ RNG: the jackknife draws nothing; only the labelled bootstrap
    diagnostic draws, and only from the ``generator`` handed in. Default
    ``ci_reps=0`` draws NOTHING — same contract as :func:`spectrum_report`,
    pinned by the same style of test.
    """

    def __init__(self, layers: dict[str, int], *, accum: int = 1,
                 rows_per_step: dict[str, int] | None = None,
                 ci_reps: int = 0, generator=None):
        if not layers:
            raise ValueError("LayerSpectrumMonitor needs at least one layer")
        self.layers = {str(k): int(v) for k, v in layers.items()}
        self.accum = int(accum)
        self.ci_reps = int(ci_reps)
        self.generator = generator
        self.rows_per_step = {k: int((rows_per_step or {}).get(k, 1))
                              for k in self.layers}
        self._acc: dict[str, SpectrumAccumulator] = (
            {k: SpectrumAccumulator(capacity=self.accum,
                                    block=max(1, self.rows_per_step[k]))
             for k in self.layers} if self.accum > 1 else {})
        self._ref: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def references(self) -> dict:
        return dict(self._ref)

    @staticmethod
    def _get(latents: dict, k: str):
        """Accept BOTH the layer name and the forward-output key (``tac`` or
        ``z_tac``). The failure class this closes: a caller handing the raw
        ``V6Stack.forward`` dict would otherwise get silent per-layer n/a
        records — visible, but a monitor that no-ops on a plausible input
        shape is a trap."""
        z = latents.get(k)
        return latents.get(f"z_{k}") if z is None else z

    def push(self, latents: dict) -> None:
        """Bank one step's latents into the per-layer rings (accum > 1 only)."""
        for k, acc in self._acc.items():
            z = self._get(latents, k)
            if z is not None:
                acc.push(z)

    def _annotate(self, rep: dict, pol: dict) -> dict:
        """Overlay the LAYER'S admissibility next to the O6-global one.

        ``spectrum_report`` stamps ``rank_admissible`` against the z_op/O6
        constant (1024) unconditionally — correct for z_op, and a misreading
        trap for a d=256 layer that can never reach it. The layer records
        therefore carry BOTH, labelled, so neither can be quoted as the other.
        """
        rep = dict(rep)
        rep["rank_admissible_layer"] = bool(
            rep["rank_ceiling"] >= int(pol["ceiling_min"]))
        rep["admissibility_note"] = (
            f"rank_admissible tests the O6/z_op constant "
            f"({O6_ADMISSIBLE_CEILING}); THE BINDING ADMISSIBILITY FOR THIS "
            f"LAYER is rank_admissible_layer (ceiling >= "
            f"{int(pol['ceiling_min'])}, {pol['basis'].split('(')[0].strip()})")
        return rep

    def emit(self, latents: dict, *, step: int | None = None) -> dict:
        """One per-layer record set: per-batch reading, pooled reading (when
        the ring has content), the layer verdict against the layer reference,
        and the reference bookkeeping. Layers absent from ``latents`` yield an
        explicit n/a record, never silence (rule 2)."""
        out: dict = {}
        for k, d in self.layers.items():
            pol = layer_spectrum_policy(k, d)
            z = self._get(latents, k)
            if z is None:
                out[k] = {"status": "n/a",
                          "reason": f"latent {k!r} (or 'z_{k}') not present "
                                    f"in the outputs handed to emit()"}
                continue
            rec: dict = {"spectrum": self._annotate(
                spectrum_report(z.reshape(-1, z.shape[-1]),
                                ci_reps=self.ci_reps, block=1,
                                generator=self.generator), pol)}
            if self.ci_reps:
                rec["spectrum"]["cluster_unit_note"] = (
                    "per-batch interval uses block=1 (rows are windows; the "
                    "sampler interleaves episodes so contiguous blocks cannot "
                    "group them) — episode correlation UNDERSTATED; the "
                    "pooled reading is the quotable one")
            basis = rec["spectrum"]
            acc = self._acc.get(k)
            if acc is not None and len(acc):
                pooled = self._annotate(
                    acc.report(ci_reps=self.ci_reps,
                               generator=self.generator), pol)
                pooled["cluster_unit"] = (
                    f"STEP ({max(1, self.rows_per_step[k])} rows) — "
                    f"conservative over-grouping; see class docstring")
                rec["spectrum_pooled"] = pooled
                basis = pooled
            rec["verdict"] = x4_rank_verdict(k, d, basis, self._ref.get(k))
            if (self._ref.get(k) is None and "spectrum_pooled" in rec
                    and rec["spectrum_pooled"]["rank_admissible_layer"]):
                # the phase-start reference for clause 2 — first reading that
                # clears the LAYER's ceiling. Same resume caveat as the O6
                # path: a restarted process takes a fresh reference.
                self._ref[k] = dict(rec["spectrum_pooled"], ref_step=step)
                rec["reference_taken_at_step"] = step
            out[k] = rec
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
# DIFFUSION PROPOSALS · MPC TOP-K REFINEMENT · FALLBACK TRIGGER (2026-08-16)
# — the three remaining diagram cells on the planner/selection surface, ALL
#   gated DEFAULT-OFF (DIAGRAM_CONFORMANCE.md F-15 / §3 / F-17)
# ============================================================================

#: P7's pre-registered calibration gate: Spearman rho >= 0.3 with the CI
#: excluding 0. DUPLICATED AS DATA from ``scripts/w7_roll_rerank.py`` (its
#: ``P7_GATE_RHO``) so the fallback trigger is buildable without importing an
#: instrument script's dependency chain into a model module; equality with the
#: instrument's constant is pinned by
#: ``tests/test_v6_diffusion_mpc_fallback.py`` so the two cannot drift apart.
P7_GATE_RHO = 0.3


class DiffusionProposalGenerator(nn.Module):
    """⭐ THE DIAGRAM'S OPERATIVE-BRAIN PROPOSAL CELL (F-15): diffuse the FULL
    6 s CONTROL sequence — 60 x (a, kappa) — with TEMPORALLY CORRELATED noise,
    as the candidate-fan generator. Default OFF; a declared ARM against the
    incumbent query fan, never a silent replacement.

    WHY CONTROLS AND NOT WAYPOINTS. H1 (`DIFFUSION_PLANNER_COMPARISON`)
    measured a per-waypoint offset head amplifying the same epsilon by **25x**
    in acceleration at dt 0.1, and the v5f dense fan measured **97.6 %**
    infeasible steps / **100 %** infeasible candidates. Candidates are
    CONTROLS, always: every sample here passes the same bounding function the
    emission uses and integrates through the same ``unicycle_rollout``, so the
    fan is feasible BY CONSTRUCTION (W4's property, inherited).

    WHY TEMPORALLY CORRELATED NOISE. White noise on a 60-step control sequence
    integrates to near-cancelling jitter — the fan's endpoints collapse toward
    the CV point and the diversity the selector needs never exists. An OU
    (AR(1)) draw — ``e_t = rho * e_{t-1} + sqrt(1 - rho^2) * w_t``, stationary
    unit marginal variance, lag-1 autocorrelation exactly ``rho`` — puts the
    noise energy at manoeuvre timescales. ⚠️ The autocorrelation of every draw
    is MEASURED (:meth:`measured_lag_autocorr`) and returned beside the fan
    (``noise_lag1_autocorr``), never asserted from the formula — the same rule
    as every other number in the programme.

    THE DENOISE LOOP is DDIM-style and TRUNCATED (DiffusionDrive's regime, the
    programme's own REF-C ``AnchoredDiffusionDecoder`` precedent — but over
    CONTROLS, not trajectories): ``diffusion_steps`` deterministic x0-prediction
    steps from a pure correlated-noise start. The denoiser predicts x0 as a
    RESIDUAL (``x0_hat = x_k + net(x_k, cond, t)``) with a ZERO-INIT output
    layer, so at initialisation the "denoised" fan IS the squashed correlated-
    noise prior — maximally diverse, feasible, and anything the module later
    does to it is LEARNED, never handed to it by init (the same discipline as
    ``GoalDistanceScorer.goal_point`` / ``MLPCandidateScorer.fc2`` / the
    emission head).

    HOW IT TRAINS. Through the SAME plan loss as the query fan (WTA +
    eps-relaxed WTA + ``w_select``'s softade, all reading ``plan["waypoints"]``
    generator-agnostically) — so an arm comparison is attributable to the
    GENERATOR, not to a different objective. A dedicated denoising-score-
    matching loss is a possible later lever and would be a trainer-side term;
    it is NOT built here.

    ⚠️ SEQUENCED BEHIND THE SELECTION QUESTION (F-15's own caveat): *"do not
    build before the selection question settles — proposals and selection are
    one experiment surface"*. Built now as gated-off machinery; TRAINING it is
    an S-T decision that rides the same admission the selector does.

    ⛔ ADMISSIBILITY (PI 2026-08-03 / 2026-08-16). Inputs: ``plan_proj(z_op)``
    (vision-derived, planner-cut), ``e_g_tac`` (the goal path), and ``v0``
    (measured present state — PI 2026-08-16: admissible, with the hold-v0
    anti-echo controls living eval-side). No situation-classifier output in
    any form. The noise is sampled fresh per forward and carries no
    per-window information by construction.
    """

    def __init__(self, d_cond: int, n_candidates: int, plan_steps: int, *,
                 hidden: int = 256, n_steps: int = 4, rho: float = 0.9,
                 sigma_a: float = 2.0, sigma_k: float = 0.1,
                 a_max: float = 4.0, kappa_max: float = 0.2, dt: float = DT,
                 squash: str = "squash", alpha_bar_min: float = 0.05):
        super().__init__()
        if squash not in ("tanh", "squash"):
            raise ValueError(f"squash must be tanh|squash, got {squash!r}")
        self.d_cond = int(d_cond)
        self.n_candidates = int(n_candidates)
        self.plan_steps = int(plan_steps)
        self.n_steps = int(n_steps)
        self.rho = float(rho)
        self.sigma_a, self.sigma_k = float(sigma_a), float(sigma_k)
        self.a_max, self.kappa_max = float(a_max), float(kappa_max)
        self.dt, self.squash = float(dt), squash
        self.alpha_bar_min = float(alpha_bar_min)
        cc = max(hidden // 4, 8)
        # cond + the scalar diffusion time t = k/K; v0/SPEED_SCALE is appended
        # by the caller-facing forward (the trunk's own speed-channel column).
        self.cond_proj = nn.Linear(self.d_cond + 2, cc)
        self.conv_in = nn.Conv1d(2 + cc, hidden, kernel_size=5, padding=2)
        self.conv_mid = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv_out = nn.Conv1d(hidden, 2, kernel_size=1)
        # ZERO-INIT: at init the residual x0-prediction is exactly the noisy
        # input, so the fan is the correlated-noise prior — see class docstring.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    # ---- the noise ---------------------------------------------------------
    def sample_ou_noise(self, b: int, generator: torch.Generator | None = None,
                        device=None) -> Tensor:
        """[B, N, K, 2] stationary AR(1)/OU noise along the TIME axis: unit
        marginal variance, lag-1 autocorrelation ``self.rho`` — by
        construction; the MEASURED value ships with every forward."""
        n, k = self.n_candidates, self.plan_steps
        w = torch.randn(b, n, k, 2, generator=generator, device=device)
        if self.rho == 0.0:
            return w
        e = torch.empty_like(w)
        e[..., 0, :] = w[..., 0, :]
        c = math.sqrt(1.0 - self.rho * self.rho)
        for t in range(1, k):
            e[..., t, :] = self.rho * e[..., t - 1, :] + c * w[..., t, :]
        return e

    @staticmethod
    def measured_lag_autocorr(eps: Tensor, lag: int = 1) -> float:
        """EMPIRICAL lag-``lag`` autocorrelation of a [..., K, C] noise draw,
        pooled over every leading dim and channel. This is the MEASUREMENT the
        binding constraint asks for — reported beside every fan, never
        asserted from the generating formula."""
        if eps.shape[-2] <= lag:
            raise ValueError(f"need > {lag} steps, got {eps.shape[-2]}")
        x = eps.transpose(-2, -1).reshape(-1, eps.shape[-2]).float()
        x = x - x.mean(dim=-1, keepdim=True)
        num = (x[:, :-lag] * x[:, lag:]).sum()
        den = (x * x).sum().clamp_min(1e-12)
        return float(num / den)

    def _alpha_bar(self, k: int) -> float:
        """Linear truncated schedule: ᾱ(0)=1 (clean) → ᾱ(K)=alpha_bar_min."""
        return 1.0 - (k / self.n_steps) * (1.0 - self.alpha_bar_min)

    def _net(self, x: Tensor, cond: Tensor, t: float) -> Tensor:
        """Residual x0-prediction. ``x`` [B, N, K, 2] · ``cond`` [B, C]."""
        b, n, k, _ = x.shape
        tcol = torch.full((b, 1), float(t), dtype=cond.dtype,
                          device=cond.device)
        hc = torch.nn.functional.gelu(
            self.cond_proj(torch.cat([cond, tcol], dim=-1)))       # [B, cc]
        hc = hc[:, None, :].expand(b, n, -1).reshape(b * n, -1)
        xin = x.reshape(b * n, k, 2).transpose(1, 2)               # [BN, 2, K]
        h = torch.cat([xin, hc[:, :, None].expand(-1, -1, k)], dim=1)
        h = torch.nn.functional.gelu(self.conv_in(h))
        h = torch.nn.functional.gelu(self.conv_mid(h))
        out = self.conv_out(h).transpose(1, 2).reshape(b, n, k, 2)
        return x + out

    def forward(self, plan_feat: Tensor, g_embed: Tensor, v0: Tensor, *,
                generator: torch.Generator | None = None) -> dict:
        """``plan_feat`` [B, F0] (= plan_proj(z_plan)) · ``g_embed``
        [B, d_goal_embed] · ``v0`` [B] -> the diffusion fan:
        ``{"a" [B,N,K], "kappa" [B,N,K], "waypoints" [B,N,K,2],
        "raw" [B,N,K,2], "noise_lag1_autocorr" float (MEASURED),
        "n_denoise_steps" int}``. Same output contract as
        ``UnicycleEmission.forward`` plus the measurement."""
        if plan_feat.ndim != 2 or g_embed.ndim != 2 or v0.ndim != 1:
            raise ValueError(
                f"plan_feat [B,F0], g_embed [B,G], v0 [B] expected; got "
                f"{tuple(plan_feat.shape)}, {tuple(g_embed.shape)}, "
                f"{tuple(v0.shape)}")
        _ensure_scripts()
        from train_v58f_unicycle_head import (  # noqa: E402
            SPEED_SCALE, unicycle_rollout)
        b = plan_feat.shape[0]
        vcol = (v0.to(plan_feat.dtype) / SPEED_SCALE)[:, None]
        cond = torch.cat([plan_feat, g_embed.to(plan_feat.dtype), vcol],
                         dim=-1)
        if cond.shape[-1] != self.d_cond + 1:
            raise ValueError(f"cond dim {cond.shape[-1]} != d_cond+1 = "
                             f"{self.d_cond + 1}")
        eps = self.sample_ou_noise(b, generator, device=plan_feat.device)
        lag1 = self.measured_lag_autocorr(eps.detach())
        x = eps.to(plan_feat.dtype)
        x0_hat = x
        for k in range(self.n_steps, 0, -1):
            ab_k = self._alpha_bar(k)
            x0_hat = self._net(x, cond, k / self.n_steps)
            if k > 1:
                ab_p = self._alpha_bar(k - 1)
                eps_hat = (x - math.sqrt(ab_k) * x0_hat) \
                    / math.sqrt(max(1.0 - ab_k, 1e-8))
                x = math.sqrt(ab_p) * x0_hat \
                    + math.sqrt(max(1.0 - ab_p, 0.0)) * eps_hat
        raw = torch.stack([x0_hat[..., 0] * self.sigma_a,
                           x0_hat[..., 1] * self.sigma_k], dim=-1)
        if self.squash == "tanh":                     # legacy, bit-exact v5.8f
            a_ctl = self.a_max * torch.tanh(raw[..., 0])
            kappa = self.kappa_max * torch.tanh(raw[..., 1])
        else:
            from tanitad.models.kinematic import _squash
            a_ctl = _squash(raw[..., 0], self.a_max)
            kappa = _squash(raw[..., 1], self.kappa_max)
        wp, _ = unicycle_rollout(a_ctl, kappa, v0, dt=self.dt)
        return {"a": a_ctl, "kappa": kappa, "waypoints": wp, "raw": raw,
                "noise_lag1_autocorr": lag1,
                "n_denoise_steps": self.n_steps}


class MpcRefiner(nn.Module):
    """⭐ THE SELECTION CELL'S "MPC refines the top-K" — built the way §3 of
    DIAGRAM_CONFORMANCE.md says it MUST be built, which is NOT as drawn (D-1).

    HOLDS NO PARAMETERS AND NO BUFFERS: turning it on changes no state_dict
    key, so it can be flipped over any checkpoint without touching a strict
    resume. It is a PROCEDURE, not a head.

    ⛔ THE MEASUREMENTS THAT BIND THIS DESIGN (all INHERITED here from their
    artifacts, quoted where they constrain code):

    1. **The pure roll-consistency argmin is REFUTED — winner's curse.**
       W7-FULL selected 3.3348 m over a 0.1273 m-oracle fan; on the banked
       REF-C-XL fan the roll score is **+5.9787 [+5.3217, +6.7625] WORSE**
       than the shipped selector and its error-rank RISES with N
       (0.241 → 0.286) while the goal rule's FALLS. ⇒ imagined-consistency may
       appear in the refinement cost ONLY as a REGULARIZER (``w_consist``,
       default 0), NEVER as the primary term, and NEVER in the re-score.
    2. **W7-PROG: any selection cost NEEDS a goal-conditioned
       (candidate-independent) component.** ⇒ the PRIMARY term is the distance
       to the selector's goal point ``ĝ`` — which is why this module REFUSES
       to exist without the ``"goal"`` selector (``V6Config.__post_init__``):
       the ``"mlp"`` capacity control has no goal point, and descending on its
       score would be candidate-DEPENDENT — the refuted family.
    3. **E-S1-0's dose-response** (2.8–2.95x worse purely for scoring
       off-distribution) is the standing warning for any refine-then-rescore
       loop. ⇒ after refinement the candidates are re-scored by the
       GOAL-CONDITIONED cost ONLY (``mpc_score = −goal_dist_post``), never by
       roll-consistency and never by a learned score evaluated off its
       training distribution.
    4. **Kinematic cost survives as a tie-breaker on a feasible fan**
       (§1.14: top8+kincost 0.4815 vs 0.560) ⇒ ``w_kin`` is a low-weight
       smoothness REGULARIZER with a zero-weight ablation always available.

    ⛔ INERT UNLESS A SELECTOR IS ADMISSIBLE. Structurally: the config refuses
    ``mpc_refine`` without ``selector="goal"``, and the chain-side
    ``assert_selector_admissible`` (v6_chain.py) refuses ANY selector launch
    while SEL-1 stands REFUSED (E-WC2 fired 2026-08-16: sigma/ADE 9.9915
    [7.4492, 13.5119] against the 3.0 refusal line) — so the MPC path cannot
    reach a launch command before the E-WC2-SW dump admits a selector. Warm
    start = the distilled/trained selector's scores pick the top-K; that is
    the one surviving sense of "distilled selector warm-starts".

    MECHANISM: ``mpc_steps`` iterations of plain gradient descent on a raw
    control DELTA (a CEM loop is a possible later arm; descent is the
    deterministic, testable form). The iterate is re-bounded through
    ``kinematic._squash`` every step, so EVERY iterate is feasible by
    construction (identity below 0.9x the bound — where the census says
    emitted controls live). All model inputs are DETACHED at entry and every
    output is DETACHED at exit: the refinement trains NOTHING and the training
    loss cannot backprop through the inner loop — it is an inference-time
    procedure, exactly like the fallback comparator.
    """

    def __init__(self, *, topk: int = 2, steps: int = 3, lr: float = 0.05,
                 w_goal: float = 1.0, w_kin: float = 0.1,
                 w_consist: float = 0.0, a_max: float = 4.0,
                 kappa_max: float = 0.2, dt: float = DT):
        super().__init__()
        if w_goal <= 0.0:
            raise ValueError(
                f"w_goal must be > 0, got {w_goal}: a refinement whose primary "
                f"(goal-conditioned) term is absent optimises the regularizers "
                f"alone — and an imagined-consistency-led refinement is the "
                f"REFUTED roll-cost selection rule wearing MPC's name "
                f"(W7-PROG; +5.9787 m).")
        self.topk, self.steps, self.lr = int(topk), int(steps), float(lr)
        self.w_goal, self.w_kin = float(w_goal), float(w_kin)
        self.w_consist = float(w_consist)
        self.a_max, self.kappa_max, self.dt = float(a_max), float(kappa_max), \
            float(dt)

    def _cost(self, a_ref: Tensor, k_ref: Tensor, wp: Tensor,
              goal_point: Tensor, roll_fn) -> tuple[Tensor, dict]:
        """The COMPOSED cost [B, K]: goal (primary) + kin + consist
        (regularizers). Per-term breakdown returned for the run row — an
        unattributable composite is the `--v2` conflation."""
        goal = (wp[:, :, -1] - goal_point[:, None]).norm(dim=-1)
        kin = ((a_ref[..., 1:] - a_ref[..., :-1]).abs().mean(dim=-1)
               / self.a_max
               + (k_ref[..., 1:] - k_ref[..., :-1]).abs().mean(dim=-1)
               / self.kappa_max)
        total = self.w_goal * goal + self.w_kin * kin
        parts = {"goal": goal, "kin": kin}
        if self.w_consist > 0.0 and roll_fn is not None:
            consist = roll_fn(a_ref, k_ref)
            total = total + self.w_consist * consist
            parts["consist"] = consist
        return total, parts

    def refine(self, a_ctl: Tensor, kappa: Tensor, v0: Tensor, *,
               sel_score: Tensor, goal_point: Tensor,
               roll_fn=None) -> dict:
        """``a_ctl``/``kappa`` [B, N, T] (the emitted fan) · ``v0`` [B] ·
        ``sel_score`` [B, N] (the trained selector's warm start) ·
        ``goal_point`` [B, 2] (``ĝ`` — candidate-INDEPENDENT) ->
        the refined top-K and its audit trail. Every output DETACHED."""
        _ensure_scripts()
        from train_v58f_unicycle_head import unicycle_rollout  # noqa: E402
        from tanitad.models.kinematic import _squash
        b, n, t = a_ctl.shape
        k = min(self.topk, n)
        idx = sel_score.detach().topk(k, dim=-1).indices           # [B, K]
        ar = torch.arange(b, device=a_ctl.device)[:, None]
        a0 = a_ctl.detach()[ar, idx]                               # [B, K, T]
        k0 = kappa.detach()[ar, idx]
        gp = goal_point.detach()
        v0d = v0.detach()
        with torch.enable_grad():
            delta = torch.zeros(b, k, t, 2, device=a_ctl.device,
                                dtype=a_ctl.dtype, requires_grad=True)
            pre = None
            for _ in range(self.steps):
                a_ref = _squash(a0 + delta[..., 0], self.a_max)
                k_ref = _squash(k0 + delta[..., 1], self.kappa_max)
                wp, _ = unicycle_rollout(a_ref, k_ref, v0d, dt=self.dt)
                cost, parts = self._cost(a_ref, k_ref, wp, gp, roll_fn)
                if pre is None:
                    pre = {kk: vv.detach() for kk, vv in parts.items()}
                    pre["total"] = cost.detach()
                g, = torch.autograd.grad(cost.sum(), delta)
                delta = (delta - self.lr * g).detach().requires_grad_(True)
            # final evaluation on the last iterate
            a_ref = _squash(a0 + delta[..., 0], self.a_max)
            k_ref = _squash(k0 + delta[..., 1], self.kappa_max)
            wp, _ = unicycle_rollout(a_ref, k_ref, v0d, dt=self.dt)
            cost, parts = self._cost(a_ref, k_ref, wp, gp, roll_fn)
        post = {kk: vv.detach() for kk, vv in parts.items()}
        post["total"] = cost.detach()
        # ⛔ THE RE-SCORE IS GOAL-CONDITIONED ONLY (D-1 / E-S1-0): the selected
        # refined candidate is the argmin of the CANDIDATE-INDEPENDENT goal
        # distance — never roll-consistency, never a score off-distribution.
        goal_post = post["goal"]
        sel_local = goal_post.argmin(dim=-1)                       # [B]
        arb = torch.arange(b, device=a_ctl.device)
        return {
            "controls": torch.stack([a_ref, k_ref], dim=-1).detach(),
            "waypoints": wp.detach(),
            "topk_idx": idx,
            "cost_pre": pre["total"], "cost_post": post["total"],
            "goal_dist_pre": pre["goal"], "goal_dist_post": goal_post,
            "kin_pre": pre["kin"], "kin_post": post["kin"],
            **({"consist_pre": pre["consist"],
                "consist_post": post["consist"]} if "consist" in pre else {}),
            "selected_local": sel_local,
            "selected": idx[arb, sel_local],
            "rescore": "goal_distance_only (D-1: roll-consistency argmin "
                       "REFUTED +5.9787 m; refined-readout rescoring "
                       "2.8-2.95x worse, E-S1-0)",
        }


class FallbackTrigger(nn.Module):
    """⭐ THE CONTEXT-BRAIN FALLBACK CELL (F-17): fires when imagined
    consequences disagree beyond the P7-CALIBRATED band.

    Signal = ``w_spread * fan_endpoint_spread + w_rollvar * roll_cost_var`` —
    exactly the diagram's context-row quantity (*"fan spread + roll-cost
    variance → calibrated uncertainty"*), the ONE place the roll-cost
    SURVIVES its refutation: P7 measured its calibration at **rho 0.7164
    [0.5847, 0.7696]** on the stage-A-repaired trunk (INHERITED;
    ``w7_roll_rerank.py`` owns the instrument).

    ⛔ AN UNCERTAINTY SIGNAL, NEVER A SELECTOR — kept distinct in CODE, not
    only in prose:
      1. every statistic here is PERMUTATION-INVARIANT over the candidate
         axis (std / variance over N) and NO per-candidate output exists, so
         this module CANNOT reorder or choose among candidates even in
         principle (pinned by
         ``test_fallback_is_permutation_invariant_hence_not_a_selector``);
      2. the whole computation runs under ``no_grad`` at the call site: the
         signal is MONITORED, never optimised — a trigger whose signal the
         training loss could reduce is a trigger the model learns to blind
         (the same "monitored, never optimised" rule as P2's nuisance
         non-retention).
    The roll-cost's SELECTION use stays refuted (+5.9787 m, error-rank RISES
    with N); its VARIANCE as an uncertainty input is P7's validated use.

    ⛔ THE CALIBRATED BAND IS LOADED, NEVER INVENTED. :meth:`load_calibration`
    installs the P7-fit spread→error mapping and its threshold as PERSISTENT
    buffers (they ship with the checkpoint — the anchor-table discipline) and
    REFUSES any calibration that fails P7's pre-registered gate
    (rho >= 0.3 with CI excluding 0): *a trigger calibrated on an
    uncalibrated signal is a random brake.* Until a calibration is loaded the
    comparator returns ``fired: None`` with the reason — it says so instead
    of inventing a boolean (the ``AnchorGoalHead.table_ready`` refusal
    pattern).

    THE DEFINED FALLBACK ACTION is the hold-v0 / CV baseline emission: zero
    commanded accel, zero curvature, integrated through the same
    ``unicycle_rollout`` from the true ``v0`` — feasible by construction, and
    exactly the warm start the zero-init emission head defines. It is emitted
    beside the plan on every forward (cheap, analytic) so a consumer can
    switch per-window on ``fired`` without a second pass.

    HOLDS NO TRAINABLE PARAMETERS — the calibration is an OFFLINE artifact
    from the P7 instrument, not a trained head; nothing here can be sculpted
    by any loss. (Buffers still create state_dict keys, so the flag rides
    ``STAGE_MAY_INTRODUCE`` like every other introduction.)
    """

    #: what :meth:`load_calibration` requires of the P7 artifact.
    CALIB_KEYS: tuple[str, ...] = (
        "spearman_rho", "rho_ci", "slope", "intercept", "threshold",
        "w_spread", "w_rollvar")

    def __init__(self, *, a_max: float = 4.0, kappa_max: float = 0.2,
                 dt: float = DT, plan_steps: int = PLAN_STEPS):
        super().__init__()
        self.a_max, self.kappa_max = float(a_max), float(kappa_max)
        self.dt, self.plan_steps = float(dt), int(plan_steps)
        # PERSISTENT: the band ships with the checkpoint (anchor-table rule).
        self.register_buffer("calib_ready",
                             torch.zeros((), dtype=torch.bool))
        self.register_buffer("calib_rho", torch.zeros(()))
        self.register_buffer("calib_rho_ci_lo", torch.zeros(()))
        self.register_buffer("calib_slope", torch.zeros(()))
        self.register_buffer("calib_intercept", torch.zeros(()))
        self.register_buffer("calib_threshold", torch.zeros(()))
        self.register_buffer("w_spread", torch.ones(()))
        self.register_buffer("w_rollvar", torch.ones(()))

    @torch.no_grad()
    def load_calibration(self, calib: dict) -> dict:
        """Install a P7 calibration artifact. REFUSES one that fails the
        pre-registered P7 gate. Returns the provenance dict a run row quotes."""
        missing = [k for k in self.CALIB_KEYS if k not in calib]
        if missing:
            raise ValueError(
                f"calibration is missing {missing}; required: "
                f"{self.CALIB_KEYS}. The artifact comes from the P7 "
                f"instrument (w7_roll_rerank.py's calibration block + the "
                f"spread->error band fit), never hand-written.")
        rho = float(calib["spearman_rho"])
        ci = calib["rho_ci"]
        if ci is None or len(ci) != 2:
            raise ValueError("rho_ci must be [lo, hi] — an interval-free rho "
                             "is not admissible (estimator rule)")
        ci_lo = float(ci[0])
        if rho < P7_GATE_RHO or ci_lo <= 0.0:
            raise ValueError(
                f"REFUSING calibration: rho {rho} (CI lo {ci_lo}) fails P7's "
                f"pre-registered gate rho >= {P7_GATE_RHO} with CI excluding "
                f"0. A trigger calibrated on an uncalibrated signal is a "
                f"random brake. (The repaired-trunk reference is rho 0.7164 "
                f"[0.5847, 0.7696] — P7 PASS.)")
        thr = float(calib["threshold"])
        if thr <= 0.0:
            raise ValueError(f"threshold must be > 0 (got {thr}): a "
                             f"non-positive band fires on every window, which "
                             f"is a disabled planner wearing a safety net")
        ws, wr = float(calib["w_spread"]), float(calib["w_rollvar"])
        if ws < 0.0 or wr < 0.0 or (ws == 0.0 and wr == 0.0):
            raise ValueError(f"w_spread/w_rollvar must be >= 0 and not both "
                             f"0, got {ws}/{wr}")
        self.calib_rho.fill_(rho)
        self.calib_rho_ci_lo.fill_(ci_lo)
        self.calib_slope.fill_(float(calib["slope"]))
        self.calib_intercept.fill_(float(calib["intercept"]))
        self.calib_threshold.fill_(thr)
        self.w_spread.fill_(ws)
        self.w_rollvar.fill_(wr)
        self.calib_ready.fill_(True)
        return {"rho": rho, "rho_ci": [ci_lo, float(ci[1])],
                "threshold": thr, "w_spread": ws, "w_rollvar": wr,
                "provenance": str(calib.get("provenance", "UNSTATED")),
                "gate": f"P7 rho >= {P7_GATE_RHO}, CI excluding 0 — PASSED"}

    def forward(self, waypoints: Tensor, v0: Tensor, *,
                roll_cost: Tensor | None = None) -> dict:
        """``waypoints`` [B, N, T, 2] (the LIVE fan) · ``v0`` [B] ·
        ``roll_cost`` [B, N] per-candidate roll-consistency costs (optional —
        without it the signal is spread-only and says so).

        Returns the signals, the comparator verdict (``fired`` [B] bool, or
        ``None`` + reason when uncalibrated), and the fallback emission."""
        if waypoints.ndim != 4 or waypoints.shape[-1] != 2:
            raise ValueError(f"waypoints must be [B, N, T, 2], got "
                             f"{tuple(waypoints.shape)}")
        _ensure_scripts()
        from train_v58f_unicycle_head import unicycle_rollout  # noqa: E402
        b, n = waypoints.shape[:2]
        # permutation-invariant BY CONSTRUCTION: std/var over the N axis.
        spread = waypoints[:, :, -1, :].std(dim=1, unbiased=False) \
            .norm(dim=-1)                                          # [B]
        if roll_cost is not None:
            if roll_cost.shape != (b, n):
                raise ValueError(f"roll_cost must be [B={b}, N={n}], got "
                                 f"{tuple(roll_cost.shape)}")
            rollvar = roll_cost.float().var(dim=1, unbiased=False)
        else:
            rollvar = torch.zeros_like(spread)
        signal = (self.w_spread.to(spread.dtype) * spread
                  + self.w_rollvar.to(spread.dtype) * rollvar)
        out = {"spread": spread, "roll_cost_var": rollvar, "signal": signal,
               "signal_includes_rollvar": roll_cost is not None}
        if bool(self.calib_ready):
            pred_err = self.calib_slope.to(signal.dtype) * signal \
                + self.calib_intercept.to(signal.dtype)
            out |= {"pred_err": pred_err,
                    "fired": pred_err > self.calib_threshold.to(signal.dtype),
                    "status": "CALIBRATED",
                    "calib_rho": float(self.calib_rho)}
        else:
            out |= {"pred_err": None, "fired": None,
                    "status": "UNCALIBRATED — the comparator refuses to fire "
                              "(no P7 band loaded; load_calibration() is the "
                              "only way in)"}
        # the DEFINED fallback action: hold-v0 / CV, feasible by construction.
        z = torch.zeros(b, 1, self.plan_steps, dtype=waypoints.dtype,
                        device=waypoints.device)
        fb_wp, _ = unicycle_rollout(z, z, v0, dt=self.dt)
        out |= {"controls": torch.zeros(b, self.plan_steps, 2,
                                        dtype=waypoints.dtype,
                                        device=waypoints.device),
                "waypoints": fb_wp[:, 0],
                "action": "hold_v0_straight (zero accel, zero curvature — "
                          "the CV baseline emission)"}
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
    # ⭐ INTERPRETATION HEADS (F-18's agent-slot decoder today) reach NOTHING
    # but themselves — a STRICTER row than `planner`'s in consequence, because
    # what it protects the trunk from is not a planning gradient but a
    # PERCEPTION LABEL. The diagram's header row is binding: *"no perception
    # label, map or reward in any trunk loss"*. A slot decoder whose gradient
    # reached the encoder would make `obstacle.offline` cuboids a trunk
    # supervisor, i.e. exactly the thing the whole label-free O1–O6 programme
    # is constructed to avoid — and it would also destroy the head's own
    # meaning: a readout that TRAINED its input can no longer answer "what does
    # the latent already carry" (the §1.10 latents-only discipline P8 states).
    # Measured by the `perception_to_trunk` edge in :meth:`V6Stack.assert_isolation`.
    "interp": ("interp",),
}


# ============================================================================
# F-7 / catalog T2 — MANOEUVRE CONTRASTIVES (label-free)
# ============================================================================
#: ⛔ THE SPEC, and the two places it is written down, because it is quoted
#: rather than paraphrased:
#:
#:   ``V6_TRAINING_MEASURES.md:65`` (the catalog) — *"T2 | **manoeuvre-
#:   contrastive windows** (label-free): time-reversal and lane-mirror
#:   augmentations as hard negatives for the tactical predictor | a lane change
#:   mirrored is the OPPOSITE manoeuvre — the predictor must not be invariant
#:   to it; teaches manoeuvre identity without manoeuvre labels"*
#:
#:   ``DIAGRAM_CONFORMANCE.md:56`` — *"Needs: a T2 loss (label-free
#:   augmentations of the window + a contrastive head on ``z_tac``) + a weight
#:   in ``V6LossWeights``"*
#:
#: ⚠️ **WHAT THE SPEC DOES NOT SAY, AND THEREFORE WHAT IS ASSUMED HERE.** A
#: contrastive loss needs a POSITIVE; the catalog names only the negatives.
#: Supplying one is a necessity of the named construct, but WHICH one is an
#: assumption and it is declared rather than buried:
#:
#:   * the POSITIVE is a manoeuvre-PRESERVING augmentation (default
#:     ``photometric`` — brightness/contrast, which changes appearance and
#:     leaves geometry, hence the manoeuvre, untouched). It doubles as C1's
#:     nuisance-non-retention pressure.
#:   * the HARD NEGATIVE is the manoeuvre-REVERSING augmentation the catalog
#:     names (``lane_mirror``).
#:   * the EASY NEGATIVES are the other windows in the batch (standard
#:     in-batch InfoNCE).
#:
#: ⛔ AND THE OBVIOUS "FREE" POSITIVE IS DEGENERATE — MEASURED, not reasoned.
#: ``uplink`` defaults to ``"stopgrad"`` (v6.py:2778), so
#: :meth:`V6Stack.uplink_tac` returns ``target = online.detach()``
#: (v6.py:3890): ``z_tac_target`` IS ``z_tac`` with the graph cut. Feeding it
#: to a unit-norm projector as the positive gives cosine similarity **exactly
#: 1.0** for every window regardless of what the head learned, so the positive
#: term is a constant and InfoNCE collapses to "push the negatives apart" —
#: a pure flip-detector objective. Pinned by
#: ``tests/test_v6_t2_contrastive.py::test_z_tac_target_is_a_degenerate_positive``.


def lane_mirror_window(frames: Tensor, actions: Tensor | None = None
                       ) -> tuple[Tensor, Tensor | None]:
    """T2's MANOEUVRE-REVERSING augmentation: left <-> right.

    ``frames`` [B, W, C, H, W'] -> horizontal flip (the last axis).
    ``actions`` [B, W, A] -> the LATERAL channel (index 0, steer/curvature in
    the lifted 3-channel format) is NEGATED; the longitudinal channel and the
    speed channel are untouched, because mirroring a scene does not change how
    fast the ego is going.

    This is the augmentation the catalog's own justification is about: *"a lane
    change mirrored is the OPPOSITE manoeuvre"*.
    """
    if frames.ndim != 5:
        raise ValueError(f"frames must be [B, W, C, H, W'], got "
                         f"{tuple(frames.shape)}")
    fm = torch.flip(frames, dims=(-1,))
    am = None
    if actions is not None:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, W, A], got "
                             f"{tuple(actions.shape)}")
        am = actions.clone()
        am[..., 0] = -am[..., 0]
    return fm, am


def time_reverse_window(frames: Tensor, actions: Tensor | None = None
                        ) -> tuple[Tensor, Tensor | None]:
    """T2's TIME-REVERSAL augmentation — built, and NOT in the default
    negative set. The reason is MEASURED and it is a fact about the
    architecture, not a preference:

    ⛔ ``z_tac`` HAS NO TEMPORAL EXTENT. :meth:`V6Stack.encode_window`
    (v6.py:3844-3847) flattens ``[B, W]`` into the batch axis, so the encoder
    sees every frame INDEPENDENTLY — there is no temporal mixing anywhere on
    the path to ``z_op``. ``forward`` then takes ``z_op = z_op_win[:, -1]``
    (v6.py:4197) and ``z_tac, _ = self.uplink_tac(z_op)`` (v6.py:4207).
    **``z_tac`` is a function of the LAST FRAME ALONE.** Reversing the window
    therefore does not present the tactical layer with "the manoeuvre played
    backwards" — it presents it with *the frame from W ticks earlier*, and the
    contrastive term would be teaching "the tactical latent at t must differ
    from the tactical latent at t - W".

    ⚠️ That objective is not merely off-spec, it is **the opposite of catalog
    T5** (F-8, in this same file's sibling change), which penalises the plan
    for CHANGING between nearby windows. Enabling both would put two terms of
    the same stage in direct opposition. Pinned by
    ``tests/test_v6_t2_contrastive.py::test_time_reversal_is_an_earlier_frame_not_a_reversed_manoeuvre``.

    The convention implemented is the straightforward one (reverse the sequence
    and negate both control channels, since a decelerating trajectory run
    backwards accelerates) so the arm EXISTS and can be measured the day the
    tactical path gains real temporal extent.
    """
    if frames.ndim != 5:
        raise ValueError(f"frames must be [B, W, C, H, W'], got "
                         f"{tuple(frames.shape)}")
    fr = torch.flip(frames, dims=(1,))
    ar = None
    if actions is not None:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, W, A], got "
                             f"{tuple(actions.shape)}")
        ar = torch.flip(actions, dims=(1,)).clone()
        ar[..., :2] = -ar[..., :2]
    return fr, ar


def photometric_jitter_window(frames: Tensor, actions: Tensor | None = None,
                              *, brightness: float = 0.25,
                              contrast: float = 0.25,
                              generator: torch.Generator | None = None
                              ) -> tuple[Tensor, Tensor | None]:
    """T2's MANOEUVRE-PRESERVING augmentation — the assumed POSITIVE view.

    Per-window brightness and contrast jitter. Geometry is untouched, so the
    manoeuvre is BY CONSTRUCTION identical; only appearance moves. ``actions``
    are returned unchanged (a brightness change commands no steering).

    ⚠️ DECLARED ASSUMPTION, not a spec item — see the T2 block above.
    """
    if frames.ndim != 5:
        raise ValueError(f"frames must be [B, W, C, H, W'], got "
                         f"{tuple(frames.shape)}")
    b = frames.shape[0]
    shape = (b,) + (1,) * (frames.ndim - 1)
    kw = {"device": frames.device, "dtype": frames.dtype}
    if generator is not None:
        db = torch.empty(shape, **kw).uniform_(-brightness, brightness,
                                               generator=generator)
        dc = torch.empty(shape, **kw).uniform_(1.0 - contrast, 1.0 + contrast,
                                               generator=generator)
    else:
        db = torch.empty(shape, **kw).uniform_(-brightness, brightness)
        dc = torch.empty(shape, **kw).uniform_(1.0 - contrast, 1.0 + contrast)
    mean = frames.mean(dim=(-3, -2, -1), keepdim=True)
    return (frames - mean) * dc + mean + db, actions


#: name -> the callable, so an arm is named in a launch line and resolved here
#: rather than by an ``if`` chain that can silently fall through to a no-op.
T2_AUGMENTATIONS: dict[str, object] = {
    "lane_mirror": lane_mirror_window,
    "time_reverse": time_reverse_window,
    "photometric": photometric_jitter_window,
}

#: ⛔ MANOEUVRE-PRESERVING augmentations may serve as the POSITIVE; only a
#: manoeuvre-REVERSING one is a legal HARD NEGATIVE. Stated as data so a
#: launch line that swaps them is REFUSED rather than silently training the
#: model to consider a mirrored lane change the same manoeuvre.
T2_MANOEUVRE_PRESERVING: frozenset[str] = frozenset({"photometric"})
T2_MANOEUVRE_REVERSING: frozenset[str] = frozenset({"lane_mirror",
                                                    "time_reverse"})


class ManoeuvreContrastiveHead(nn.Module):
    """F-7 / T2: the projector ``z_tac -> unit-norm embedding``.

    A standard two-layer SimCLR projector plus a learnable temperature. It is
    ``layer_tac``-grouped (``_GROUP_PREFIXES``), so it trains in S-T alongside
    the ``adapter_tac`` whose output it reads, and S-W never sees it.

    ⛔ X3 IS SATISFIED BY CONSTRUCTION, not by hope: its only input is
    ``z_tac``, and ``uplink_tac`` cuts the trunk unconditionally under
    ``isolate_uplink`` (v6.py:3875), so no T2 gradient can reach the encoder or
    the readout. The ``layer_tac`` row of :data:`ISOLATION_MATRIX` is
    ``("layer_tac",)`` and this head does not widen it.
    """

    def __init__(self, d_tac: int, *, d_proj: int = 128, hidden: int = 256,
                 tau: float = 0.1, learn_tau: bool = True):
        super().__init__()
        if d_proj < 1 or hidden < 1:
            raise ValueError("d_proj and hidden must be >= 1")
        if not (0.0 < tau):
            raise ValueError(f"tau must be > 0, got {tau}")
        self.net = nn.Sequential(
            nn.Linear(d_tac, hidden), nn.GELU(), nn.Linear(hidden, d_proj))
        lt = torch.tensor(float(math.log(tau)))
        if learn_tau:
            self.log_tau = nn.Parameter(lt)
        else:
            self.register_buffer("log_tau", lt)

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def forward(self, z_tac: Tensor) -> Tensor:
        """``z_tac`` [B, d_tac] -> L2-normalised embedding [B, d_proj]."""
        if z_tac.ndim != 2:
            raise ValueError(f"z_tac must be [B, d_tac], got "
                             f"{tuple(z_tac.shape)}")
        h = self.net(z_tac)
        return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class V6Config:
    """Every v6 knob that changes the model, in one serialisable place.

    Defaults are the CATALOG settings (``V6_TRAINING_MEASURES.md`` §0–§5 and
    ``HIERARCHY_VOCABULARY.md`` §4b), not preferences. Two of them are
    pre-registered ARMS rather than choices — ``shared_encoder`` (E-ENC) and
    ``uplink`` — and the trainer records which arm ran.
    """

    # ---- per-layer widths ---------------------------------------------------
    #: ⭐ Tactical LATERAL vocabulary version. DEFAULT v6.0 = the live
    #: v6F S-W checkpoint's shape (6 tokens). v6.1 appends TURN_L/TURN_R
    #: and is intended for the post-30k S-T launch, where
    #: STAGE_MAY_INTRODUCE legitimises the widened keys.
    tac_vocab_version: str = "v6.0"
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

    # ---- PROPOSALS · MPC · FALLBACK (2026-08-16, DIAGRAM cells) ------------
    # ⛔ ALL DEFAULT OFF, built at the very END of __init__, and the default
    # build's state_dict is proved BYTE-IDENTICAL per tensor against a
    # CONTENT-anchored pre-change revision (C75) in
    # ``tests/test_v6_diffusion_mpc_fallback.py`` — the live resumes are
    # 87,893,449/405 (default) and 336,542,025/573 (config E) and a broken
    # strict resume kills them.
    #: ⭐ the candidate-fan GENERATOR. ``"query"`` = the incumbent learned-query
    #: emission (cand_queries + UnicycleEmission). ``"diffusion"`` = the
    #: diagram's operative-brain proposal cell (F-15): diffuse the full 6 s
    #: CONTROL sequence with temporally correlated (OU) noise through
    #: :class:`DiffusionProposalGenerator`. A DECLARED ARM (new state_dict
    #: keys ⇒ introduced post-S-W via ``STAGE_MAY_INTRODUCE``), pre-registered
    #: against the query fan — with it ON the query/CV fan is STILL emitted
    #: beside it (``qfan_*``) as the paired reference on the same window.
    proposals: str = "query"
    #: truncated DDIM denoise steps (DiffusionDrive's regime; REF-C precedent).
    diffusion_steps: int = 4
    #: lag-1 autocorrelation of the OU noise along the 60-step axis. The DRAWN
    #: noise's autocorrelation is MEASURED per forward and reported
    #: (``prop_noise_lag1_autocorr``), never asserted from this number.
    diffusion_noise_rho: float = 0.9
    diffusion_hidden: int = 256
    #: raw-space (pre-squash) noise scales per channel — they set the prior
    #: fan's diversity. Declared knobs, not magic: a=2.0 against a_max 4.0,
    #: kappa=0.1 against kappa_max 0.2.
    diffusion_sigma_a: float = 2.0
    diffusion_sigma_k: float = 0.1
    #: ⭐ MPC top-K refinement (the selection cell's "MPC refines the top-K",
    #: built per the D-1 re-read: goal-conditioned PRIMARY cost + kinematic and
    #: imagined-consistency REGULARIZERS; re-score by goal distance ONLY).
    #: ⛔ REQUIRES ``selector="goal"`` — refused otherwise in __post_init__ —
    #: and therefore cannot reach a launch while SEL-1 stands REFUSED
    #: (``assert_selector_admissible`` gates every selector arm). Holds NO
    #: parameters and NO buffers: flipping it changes no state_dict key.
    mpc_refine: bool = False
    mpc_topk: int = 2
    mpc_steps: int = 3
    mpc_lr: float = 0.05
    #: P_O roll depth (10 Hz steps) for the imagined-consistency REGULARIZER.
    #: 0 = the roll never runs (and ``mpc_w_consist`` must then be 0).
    mpc_roll_k: int = 0
    mpc_w_goal: float = 1.0
    mpc_w_kin: float = 0.1
    mpc_w_consist: float = 0.0
    #: ⭐ the context-brain fallback trigger (F-17): fan spread + roll-cost
    #: variance -> P7-calibrated uncertainty; fires when the band is exceeded.
    #: The roll-cost appears here as an UNCERTAINTY SIGNAL (P7 rho 0.7164,
    #: its validated use), NEVER as a selector — the module is permutation-
    #: invariant over candidates by construction. Holds calibration BUFFERS
    #: only (no trainable parameters); ON adds keys ⇒ ``STAGE_MAY_INTRODUCE``.
    fallback_trigger: bool = False
    #: P_O roll depth for the roll-cost-variance half of the signal.
    #: 0 = spread-only (the signal says so: ``fb_signal_includes_rollvar``).
    fallback_roll_k: int = 10

    # ---- THE PERCEPTION AGENT-SLOT DECODER (F-18, 2026-08-16) --------------
    # ⛔ DEFAULT OFF, built at the very END of ``__init__``, and the default
    # build's state_dict is proved BYTE-IDENTICAL per tensor against a
    # CONTENT-anchored pre-change revision (C75) in
    # ``tests/test_v6_agent_slots.py`` — the live resumes are 87,893,449/405
    # (default) and 336,542,025/573 (config E) and a broken strict resume kills
    # them.
    #: ⭐ build :class:`~tanitad.models.agent_slots.AgentSlotDecoder` — §4.2's
    #: first interpretation-head row and the LAST unbuilt PERCEPTION cell of
    #: the diagram (DIAGRAM_CONFORMANCE.md F-18: *"NEW — design here;
    #: DETR-style slot decoder ~2–4 M params on spatial tokens"*).
    #: ⛔ VISION-ONLY AT INFERENCE (PI 2026-08-03): its ONLY input is the
    #: spatial memory computed from ``frames``. The decoder's ``forward`` takes
    #: one tensor and has no keyword through which ego, actions, goals or a
    #: situation channel could arrive — the signature IS the audit.
    #: ⚠️ It trains in NO ladder stage — its group is in
    #: :data:`LADDER_UNTRAINED_GROUPS`, so no ``STAGE_GROUPS`` entry (S-J
    #: included) may declare it and :func:`stage_trainable_groups` RAISES if one
    #: does: its targets are agent cuboids and the v6 batch has none. S-T may
    #: INTRODUCE it (``STAGE_MAY_INTRODUCE``) so a checkpoint can CARRY it —
    #: carrying is not training, and until 2026-08-16 the freeze map said
    #: otherwise whenever ``agent_slots=True``.
    agent_slots: bool = False
    #: ⭐ F-7 / catalog T2 — the MANOEUVRE-CONTRASTIVE head. DEFAULT OFF, so
    #: the default build creates NO state_dict key and the live v6F S-W resume
    #: (87,893,449 params / 405 keys, tensor-strict) is untouched. ON adds keys
    #: ⇒ ``STAGE_MAY_INTRODUCE["S-T"]`` carries ``"t2_head."``.
    #: ⛔ It trains in S-T ONLY (group ``layer_tac``); S-W never builds it,
    #: because the catalog files T2 under LAYER T.
    t2_contrastive: bool = False
    d_t2_proj: int = 128
    d_t2_hidden: int = 256
    #: InfoNCE temperature at init. Learnable (``log_tau``) — the one extra
    #: scalar parameter, and it is a parameter rather than a constant because a
    #: fixed temperature is a hyper-parameter nobody re-tunes.
    t2_tau: float = 0.1
    #: number of slot queries. ⚠️ A DECLARED PLACEHOLDER, not a fitted value —
    #: the right number is the join's measured per-frame agent-count
    #: distribution and that is UNMEASURED (no join file lives in the repo).
    #: Under-sizing it does not corrupt the loss (the farthest targets are
    #: dropped and COUNTED, ``slot_set_loss``'s ``n["dropped"]``) but it does
    #: cap what the head can express on crowded frames.
    n_slot_queries: int = N_QUERIES_DEFAULT
    slot_hidden: int = 256
    slot_depth: int = 3
    slot_heads: int = 8
    #: ⭐ THE MEMORY THE SLOTS CROSS-ATTEND INTO — a DECLARED ARM, with its
    #: control, because a null result is otherwise unattributable.
    #: ``"cells"`` (default) = the readout's spatial CELL tokens, i.e. the
    #: latent surface every other v6 probe reads (``V6Stack.cells``; O2/O3 act
    #: on exactly these). That is the surface whose CONTENT is the open
    #: question — RC1's *"lead geometry lives in these cells and dies in
    #: aggregation"*, and §4.2's "lead state absent in v5f (measured)".
    #: ``"tokens"`` = the encoder's raw patch tokens — the INFORMATION CONTROL.
    #: If ``cells`` fails and ``tokens`` succeeds, the agents ARE visible to the
    #: encoder and the READOUT is what destroys them, which is a finding about
    #: the geometry firewall and not about the world model. Without the control
    #: a failure is unattributable between "the encoder cannot see agents" and
    #: "the 4x4 grid cannot carry them" — the C6 confound, one floor down.
    slot_src: str = "cells"
    #: ⛔ X3 for the interpretation head. True (default) = the slot decoder's
    #: memory is DETACHED, so its perception-label gradient cannot reach the
    #: encoder/readout — the binding *"no perception label in any trunk loss"*
    #: header row. Setting it False is the deliberately MIS-WIRED control arm,
    #: and :meth:`V6Stack.assert_isolation`'s ``perception_to_trunk`` edge then
    #: FAILS — which is precisely what makes that edge a check rather than a
    #: comment (a guard that cannot fail is the C13 family). It exists for the
    #: probe's negative control and for NO training use.
    isolate_interp_from_encoder: bool = True

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
        # ---- PROPOSALS · MPC · FALLBACK ------------------------------------
        if self.proposals not in ("query", "diffusion"):
            raise ValueError(
                f"proposals must be query|diffusion, got {self.proposals!r}. "
                f"'diffusion' is the F-15 arm (control-sequence denoiser with "
                f"temporally correlated noise); 'query' is the incumbent "
                f"learned-query fan and the arm's control.")
        if self.diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be >= 1, got "
                             f"{self.diffusion_steps}")
        if not 0.0 <= self.diffusion_noise_rho < 1.0:
            raise ValueError(
                f"diffusion_noise_rho must be in [0, 1), got "
                f"{self.diffusion_noise_rho}: rho >= 1 is not a stationary "
                f"AR(1) process and its 'autocorrelation' would be a number "
                f"rather than a property")
        for nm, v in (("diffusion_hidden", self.diffusion_hidden),
                      ("diffusion_sigma_a", self.diffusion_sigma_a),
                      ("diffusion_sigma_k", self.diffusion_sigma_k)):
            if float(v) <= 0.0:
                raise ValueError(f"{nm} must be > 0, got {v}")
        if self.mpc_refine and self.selector != "goal":
            raise ValueError(
                f"mpc_refine requires selector='goal', got "
                f"{self.selector!r}. The refinement's PRIMARY cost is the "
                f"distance to the selector's candidate-INDEPENDENT goal point "
                f"ĝ (W7-PROG: any selection cost NEEDS a goal-conditioned "
                f"component); 'none' has no selector at all and 'mlp' — the "
                f"capacity control — emits no goal point, so descending on "
                f"its score would be candidate-DEPENDENT, i.e. the REFUTED "
                f"roll-cost family (+5.9787 m, error-rank RISING with N). "
                f"The MPC path stays INERT unless a selector is admissible "
                f"(assert_selector_admissible gates every selector launch).")
        if self.mpc_refine:
            if self.mpc_topk < 1 or self.mpc_topk > self.n_candidates:
                raise ValueError(f"mpc_topk must be in [1, n_candidates="
                                 f"{self.n_candidates}], got {self.mpc_topk}")
            if self.mpc_steps < 1:
                raise ValueError(f"mpc_steps must be >= 1, got "
                                 f"{self.mpc_steps}")
            if self.mpc_lr <= 0.0:
                raise ValueError(f"mpc_lr must be > 0, got {self.mpc_lr}")
            if self.mpc_w_goal <= 0.0:
                raise ValueError(
                    f"mpc_w_goal must be > 0, got {self.mpc_w_goal}: a "
                    f"refinement whose primary (goal-conditioned) term is "
                    f"absent optimises the regularizers alone — an "
                    f"imagined-consistency-led refinement is the REFUTED "
                    f"roll-cost selection rule wearing MPC's name.")
            if self.mpc_w_kin < 0.0 or self.mpc_w_consist < 0.0:
                raise ValueError(f"mpc_w_kin/mpc_w_consist must be >= 0, got "
                                 f"{self.mpc_w_kin}/{self.mpc_w_consist}")
            if self.mpc_w_consist > 0.0 and self.mpc_roll_k < 1:
                raise ValueError(
                    f"mpc_w_consist {self.mpc_w_consist} with mpc_roll_k "
                    f"{self.mpc_roll_k}: a consistency regularizer with no "
                    f"roll is a term that silently never computes — the same "
                    f"defect family as w_select with selector='none'.")
        if self.mpc_roll_k < 0 or self.fallback_roll_k < 0:
            raise ValueError(f"roll depths must be >= 0, got mpc_roll_k="
                             f"{self.mpc_roll_k}, fallback_roll_k="
                             f"{self.fallback_roll_k}")
        # ---- THE PERCEPTION AGENT-SLOT DECODER (F-18) ----------------------
        if self.slot_src not in ("cells", "tokens"):
            raise ValueError(
                f"slot_src must be cells|tokens, got {self.slot_src!r}. "
                f"'cells' is the readout's spatial latent — the surface whose "
                f"content is the open question (RC1); 'tokens' is the "
                f"encoder's raw patches, the INFORMATION CONTROL that "
                f"separates 'the encoder cannot see agents' from 'the readout "
                f"grid cannot carry them'.")
        if self.agent_slots:
            for nm, v in (("n_slot_queries", self.n_slot_queries),
                          ("slot_hidden", self.slot_hidden),
                          ("slot_depth", self.slot_depth),
                          ("slot_heads", self.slot_heads)):
                if int(v) < 1:
                    raise ValueError(f"{nm} must be >= 1, got {v}")
            if int(self.slot_hidden) % int(self.slot_heads):
                raise ValueError(
                    f"slot_hidden {self.slot_hidden} must divide by "
                    f"slot_heads {self.slot_heads}")
            # ⛔ THE TWO "AGENT SLOT" COUNTS ARE ONE SET OR THEY ARE A TYPE
            # ERROR. `n_agent_slots` is the CARDINALITY of the categorical
            # `agent_slot` arg that four g_tac tokens index (`GAP_TARGET`,
            # `YIELD_AT`, `WAIT_FOR_ONCOMING`, `EVADE_IN_CORRIDOR`); this
            # decoder is the thing that would POPULATE that set. If the head
            # emits 16 slots and the vocabulary can only index 8, an emitted
            # `agent_slot=12` refers to nothing — which is exactly the class of
            # type error the categorical channel was built to remove
            # ("writing it into a metres slot is the type error this channel
            # exists to remove"). Refuse at build time, before the GPU.
            if self.goal_cat_args and \
                    int(self.n_agent_slots) != int(self.n_slot_queries):
                raise ValueError(
                    f"n_agent_slots {self.n_agent_slots} != n_slot_queries "
                    f"{self.n_slot_queries} with goal_cat_args=True: the "
                    f"categorical `agent_slot` arg INDEXES the slots this "
                    f"decoder emits, so two different cardinalities make an "
                    f"emitted index that refers to nothing. Set them equal, or "
                    f"turn one of the two levers off.")
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
    # ⭐ INTERPRETATION HEADS (F-18). A group of its own, and NOT `aux`, because
    # `aux` MAY train the encoder by the X3 matrix (O3/O6 are label-free trunk
    # losses and that is their job) while an interpretation head may NOT — its
    # supervision is a PERCEPTION LABEL. Putting the slot decoder in `aux`
    # would have opened that door silently and the isolation probe would have
    # reported a pass, because the matrix would have permitted the edge.
    # ⚠️ EMPTY at the default build: `V6Config.agent_slots` is False, so this
    # group holds ZERO parameters and every per-group report gains a `0` entry
    # and nothing else.
    # ⛔ AND IT IS IN `LADDER_UNTRAINED_GROUPS` (below), so it is in NO stage's
    # trainable set — being a MODULE_GROUPS member means "apply_stage_freeze
    # partitions over it", NEVER "some stage trains it". Those two were
    # conflated by the `"S-J": MODULE_GROUPS` alias and it was a real defect;
    # the note on STAGE_GROUPS carries the measurement.
    "interp",
)

#: ⛔ THE GROUPS NO LADDER STAGE MAY TRAIN — as DATA, because this is an
#: invariant and the previous version of it was a comment.
#:
#: The v6 ladder has no loss that reaches these groups. Declaring one trainable
#: makes the freeze audit report a module as "training" while it receives
#: exactly zero gradient — the same lie ``V6LossWeights.for_stage`` zeroes its
#: planner terms to avoid (it drops ``w_anchor`` in S-S and ``w_s2_goal`` in
#: S-T for precisely this reason: a term advertised in the launch line that
#: trains nothing).
#:
#: ``interp`` — the interpretation heads (F-18's agent-slot decoder) — is the
#: sole member. Its targets are agent cuboids and the v6 training batch carries
#: frames/actions/poses/future_* only (``tanitad/data/_contract.py``; ``grep
#: obstacle tanitad/data/physicalai.py`` -> zero), so no stage CAN train it.
#: It is trained by a FROZEN-TRUNK PROBE in the P8 idiom
#: (``scripts/train_p8_occupancy.py``), which is also what the §6 status table
#: means by "interpretation heads on frozen latents"; it lives in the
#: state_dict so the checkpoint ships the interpretation head with the model,
#: and ``STAGE_MAY_INTRODUCE["S-T"]`` is what lets a later stage carry it in
#: over an S-W checkpoint that never had it. **Carrying a module is not
#: training it, and the freeze map must say so.**
#:
#: ⚠️ TO GIVE A LADDER STAGE A LOSS THAT REACHES ONE OF THESE, REMOVE THE NAME
#: HERE IN THE SAME EDIT — :func:`stage_trainable_groups` RAISES if a stage
#: declares a member, so the loss and the freeze map cannot drift apart
#: silently in either direction.
LADDER_UNTRAINED_GROUPS: frozenset[str] = frozenset({"interp"})

#: What each stage TRAINS. Everything else is frozen.
#:   S-W  world stage: WM only, λ_plan ≡ 0, planner ABSENT.
#:   S-T  tactical layer + the operative planner it conditions, on the FROZEN
#:        S-W trunk (the Drive-JEPA "planner is a post-trained consumer" shape).
#:   S-S  strategic layer on the FROZEN S-T stack.
#:   S-J  optional brief joint polish — isolation still ON.
#:
#: ⛔ S-J IS ``MODULE_GROUPS`` MINUS :data:`LADDER_UNTRAINED_GROUPS`, DERIVED
#: RATHER THAN SPELLED OUT. Both halves are load-bearing and each answers a
#: real failure:
#:
#:   * **derived** — a NEW group appended to ``MODULE_GROUPS`` is joint-polished
#:     automatically, instead of being silently absent from the one stage whose
#:     job is to train everything. A hand-written seven-name tuple would rot the
#:     other way.
#:   * **minus** — the bare alias ``"S-J": MODULE_GROUPS`` was a DEFECT,
#:     MEASURED 2026-08-16 (``…/2026-08-16-evidence-and-flake/``): with
#:     ``agent_slots=True`` it marked ``interp``'s **62 tensors / 3,207,445
#:     parameters** (production geometry) TRAINABLE in S-J while the S-J loss
#:     reached **exactly 0** of them. Latent only because the flag defaults
#:     False and the group is then empty — and the alias is precisely HOW the
#:     defect arrived: appending ``interp`` to ``MODULE_GROUPS`` (`06b8782`)
#:     changed what S-J trains without touching the line that declares S-J.
#:     ⚠️ ``STAGE_GROUPS["S-J"] is MODULE_GROUPS`` was an IDENTITY ALIAS
#:     (same ``id``, MEASURED). Tuples are immutable so nothing could mutate
#:     across it, but the coupling was total in one direction, which is worse
#:     than a mutation bug because it is invisible at the edit site.
STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "S-W": ("encoder", "readout", "predictor_op", "aux"),
    "S-T": ("layer_tac", "planner"),
    "S-S": ("layer_str",),
    "S-J": tuple(g for g in MODULE_GROUPS
                 if g not in LADDER_UNTRAINED_GROUPS),
}


def stage_trainable_groups(stage: str) -> tuple[str, ...]:
    """The groups ``stage`` trains — the ONE funnel every consumer reads.

    ⛔ RAISES if a stage declares a :data:`LADDER_UNTRAINED_GROUPS` member. The
    guard lives here rather than in :func:`apply_stage_freeze` because the
    declaration is not private plumbing: ``train_v6_staged.py`` writes it into
    the run's ``config.json`` as ``trainable_groups``, so a stale entry does not
    merely mislead the freeze audit — it SHIPS in the artifact the run is later
    quoted from. Guarding the funnel covers every consumer of the declaration,
    not only the one that flips ``requires_grad``.
    """
    if stage not in STAGE_GROUPS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    groups = STAGE_GROUPS[stage]
    overstated = tuple(g for g in groups if g in LADDER_UNTRAINED_GROUPS)
    if overstated:
        raise RuntimeError(
            f"stage {stage!r} declares {list(overstated)} trainable, but no "
            f"ladder loss reaches those groups (LADDER_UNTRAINED_GROUPS). The "
            f"freeze audit would report them as TRAINING while they receive "
            f"exactly zero gradient, and `trainable_groups` in the run's "
            f"config.json would ship that overstatement. If a stage has GAINED "
            f"a loss that reaches one, drop it from LADDER_UNTRAINED_GROUPS in "
            f"the same edit.")
    return groups


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


# ---------------------------------------------------------------------------
# ⛔ THE FROZEN-EXTERNAL GUARD (E-XENC-1's live trap)
# ---------------------------------------------------------------------------
#: Attribute name a submodule carries to declare itself a FOREIGN, FROZEN
#: backbone. Set it with :func:`declare_frozen_external`, never by hand — the
#: setter is what makes the declaration greppable and the value self-explaining.
FROZEN_EXTERNAL_FLAG = "_tanitad_frozen_external"


class FrozenExternalViolation(RuntimeError):
    """Raised by :func:`assert_frozen_external` when a submodule DECLARED a
    frozen external backbone is trainable — or when the stage's declared groups
    have gone entirely untrainable, which is the same lie in the other
    direction."""


def declare_frozen_external(module: nn.Module, why: str) -> nn.Module:
    """Mark ``module`` as a foreign backbone that must NEVER train, and freeze
    it now. Returns the module so it can wrap a constructor call."""
    setattr(module, FROZEN_EXTERNAL_FLAG, str(why))
    module.requires_grad_(False)
    return module


def frozen_external_prefixes(stack: nn.Module) -> dict[str, str]:
    """``{parameter-name prefix: reason}`` for every declared subtree."""
    out: dict[str, str] = {}
    for name, mod in stack.named_modules():
        why = getattr(mod, FROZEN_EXTERNAL_FLAG, None)
        if why:
            out[name] = str(why)
    return out


def _in_declared_subtree(param_name: str, prefixes) -> str | None:
    for pre in prefixes:
        if pre == "" or param_name == pre or param_name.startswith(pre + "."):
            return pre
    return None


def reassert_frozen_external(stack: nn.Module) -> dict:
    """Re-freeze every declared external subtree. ⛔ CALL THIS AFTER
    :func:`apply_stage_freeze`, not before — the freeze sets ``requires_grad``
    from the GROUP MAP, and a foreign backbone installed under ``encoder``
    lands in a group S-W trains."""
    pres = frozen_external_prefixes(stack)
    n = 0
    for name, p in stack.named_parameters():
        if _in_declared_subtree(name, pres) is not None:
            p.requires_grad_(False)
            n += int(p.numel())
    return {"declared_subtrees": pres, "n_params_refrozen": n}


def assert_frozen_external(stack: nn.Module, stage: str,
                           expect_n_trainable: int | None = None) -> dict:
    """⛔ THE GUARD, PINNED IN BOTH DIRECTIONS — and it must be, because this
    programme shipped a rejects-everything guard and a passes-everything guard
    **within one day** (C95/C97).

    **Direction A — it must CATCH the un-freeze.** :func:`apply_stage_freeze`
    sets ``requires_grad`` from :data:`MODULE_GROUPS`, so a frozen external
    encoder installed under the ``encoder`` group is UN-FROZEN by S-W
    (MEASURED, E-XENC-1 build: **86,580,480 foreign parameters** would have
    trained while the run called itself *"frozen external encoder"*, and
    nothing downstream would have said so). Any trainable parameter inside a
    declared subtree raises.

    **Direction B — it must NOT pass a stack that trains nothing.** A guard
    that only asserts "the foreign backbone is frozen" is satisfied by freezing
    the WHOLE model, which is the opposite failure and just as silent. So every
    group in ``stage_trainable_groups(stage)`` must still hold at least one
    trainable NATIVE parameter. If a declared subtree has swallowed a group
    whole, that raises too — with the group named.

    ``expect_n_trainable`` makes the runbook's exact-count assertion a
    parameter of the guard instead of a number retyped into a launch script.

    Returns the audit; raises :class:`FrozenExternalViolation` on either
    direction. A stack with NO declared subtree is legal and still gets
    direction B — which is why this is safe to call unconditionally.
    """
    pres = frozen_external_prefixes(stack)
    groups = set(stage_trainable_groups(stage))
    leaked: list[tuple[str, str, int]] = []
    native_trainable = {g: 0 for g in MODULE_GROUPS}
    external_by_group = {g: 0 for g in MODULE_GROUPS}
    n_trainable = 0
    for name, p in stack.named_parameters():
        g = stack.group_of(name)
        pre = _in_declared_subtree(name, pres)
        if p.requires_grad:
            n_trainable += int(p.numel())
        if pre is not None:
            external_by_group[g] += int(p.numel())
            if p.requires_grad:
                leaked.append((name, pre, int(p.numel())))
        elif p.requires_grad:
            native_trainable[g] += int(p.numel())
    if leaked:
        tot = sum(n for _, _, n in leaked)
        raise FrozenExternalViolation(
            f"stage {stage!r}: {len(leaked)} parameters ({tot:,}) inside a "
            f"DECLARED frozen-external subtree are TRAINABLE — "
            f"`apply_stage_freeze` sets requires_grad from the group map and "
            f"the subtree sits in a group this stage trains. Call "
            f"`reassert_frozen_external(stack)` AFTER `apply_stage_freeze`. "
            f"First offenders: {[n for n, _, _ in leaked[:4]]}")
    starved = sorted(g for g in groups if native_trainable[g] == 0)
    if starved:
        raise FrozenExternalViolation(
            f"stage {stage!r} declares {sorted(groups)} trainable but "
            f"{starved} hold ZERO trainable native parameters "
            f"(external in those groups: "
            f"{ {g: external_by_group[g] for g in starved} }). A guard that "
            f"only checked the backbone was frozen would PASS this — and the "
            f"run would train nothing while reporting a normal freeze.")
    if expect_n_trainable is not None and n_trainable != int(expect_n_trainable):
        raise FrozenExternalViolation(
            f"stage {stage!r}: n_trainable {n_trainable:,} != expected "
            f"{int(expect_n_trainable):,}. The arm is not the arm it claims "
            f"to be; do not start step 1.")
    return {"stage": stage, "declared_subtrees": pres,
            "n_declared_subtrees": len(pres),
            "n_trainable": n_trainable,
            "n_external_params": sum(external_by_group.values()),
            "native_trainable_per_group": native_trainable,
            "external_per_group": {g: n for g, n in external_by_group.items()
                                   if n},
            "expect_n_trainable": expect_n_trainable,
            "directions_checked": [
                "A: no parameter inside a declared frozen-external subtree is "
                "trainable",
                "B: every group in stage_trainable_groups() still has a "
                "trainable NATIVE parameter"]}


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

    ⭐ AND SINCE F-18 THERE IS A SECOND DECLARED SURFACE, ``interp_side`` — the
    PERCEPTION agent-slot decoder (``cfg.agent_slots``, default OFF). It is
    separate from ``planner_side`` because it is a stricter obligation: what
    flows through it is a PERCEPTION LABEL, which the binding diagram's header
    row forbids in ANY trunk loss, so it may reach nothing outside its own
    ``interp`` group — not merely nothing in an encoder. Its edge
    (``perception_to_trunk``) appears in :meth:`assert_isolation` only when the
    head is built, and ``isolate_interp_from_encoder=False`` is the mis-wired
    arm that makes that edge FAIL, which is what keeps it a check.
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
        self.vocab_a_lat = GoalVocabulary(
            tactical_lat_actions(getattr(cfg, "tac_vocab_version", "v6.0")),
            cfg.d_goal_embed)
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

        # ---- PROPOSALS · MPC · FALLBACK (2026-08-16) — built LAST, only when
        # asked. ⛔ Same rule as every gated lever above, same reason: the
        # default path draws NO RNG, creates NO state_dict key, and every
        # earlier module's initialisation is bit-for-bit what it was — proved
        # per tensor against a CONTENT-anchored pre-change revision (C75) in
        # tests/test_v6_diffusion_mpc_fallback.py. The live resumes
        # (87,893,449/405 default · 336,542,025/573 config E) depend on it.
        self.prop_diffusion = None
        if cfg.proposals == "diffusion":
            # conditioned on [plan_proj(z_plan) ‖ e_g_tac] (+ the v0 column it
            # appends itself) — the SAME information surface the query fan
            # reads; per-candidate diversity comes from the OU noise, not from
            # learned queries.
            self.prop_diffusion = DiffusionProposalGenerator(
                cfg.d_plan_feat + cfg.d_goal_embed, cfg.n_candidates,
                cfg.plan_steps, hidden=cfg.diffusion_hidden,
                n_steps=cfg.diffusion_steps, rho=cfg.diffusion_noise_rho,
                sigma_a=cfg.diffusion_sigma_a, sigma_k=cfg.diffusion_sigma_k,
                a_max=cfg.a_max, kappa_max=cfg.kappa_max, dt=cfg.dt,
                squash=cfg.emission_squash)
        self.mpc = None
        if cfg.mpc_refine:
            # parameter-free and buffer-free: no state_dict key moves. The
            # selector="goal" requirement is enforced by V6Config.__post_init__.
            self.mpc = MpcRefiner(
                topk=cfg.mpc_topk, steps=cfg.mpc_steps, lr=cfg.mpc_lr,
                w_goal=cfg.mpc_w_goal, w_kin=cfg.mpc_w_kin,
                w_consist=cfg.mpc_w_consist, a_max=cfg.a_max,
                kappa_max=cfg.kappa_max, dt=cfg.dt)
        self.fallback = None
        if cfg.fallback_trigger:
            # buffers only (the P7 band ships with the checkpoint); no
            # trainable parameter — the trigger is CALIBRATED, never trained.
            self.fallback = FallbackTrigger(
                a_max=cfg.a_max, kappa_max=cfg.kappa_max, dt=cfg.dt,
                plan_steps=cfg.plan_steps)

        # ---- THE PERCEPTION AGENT-SLOT DECODER (F-18) — built LAST, only
        # when asked. ⛔ Same rule as every gated lever above, same reason: the
        # default path draws NO RNG, creates NO state_dict key, and every
        # earlier module's initialisation is bit-for-bit what it was — proved
        # per tensor against a CONTENT-anchored pre-change revision in
        # tests/test_v6_agent_slots.py.
        # ⚠️ ``enforce_band=False`` HERE and only here: V6Stack is instantiated
        # at toy geometries by dozens of tests, where a §6 2–4 M band would
        # refuse every one of them. The band is NOT thereby unenforced — it is
        # the module's own default (a frozen-trunk probe script gets it for
        # free) and the PRODUCTION geometry is pinned against it by
        # test_production_geometry_is_inside_the_preregistered_band, with a
        # companion test that the band check still FIRES when asked.
        self.agent_slots = None
        if cfg.agent_slots:
            d_mem, n_mem = ((int(cfg.readout.d_readout), int(cfg.n_cells))
                            if cfg.slot_src == "cells"
                            else (int(cfg.encoder.d_model),
                                  int(self.encoder.n_tokens)))
            self.agent_slots = AgentSlotDecoder(
                d_mem, n_mem, n_queries=cfg.n_slot_queries,
                d_model=cfg.slot_hidden, depth=cfg.slot_depth,
                n_heads=cfg.slot_heads, ranges=SlotDecodeRanges(),
                enforce_band=False)

        # ---- F-7 / T2: THE MANOEUVRE-CONTRASTIVE HEAD — built LAST for the
        # same reason as every gated lever above: the default path draws NO
        # RNG and creates NO state_dict key, so the default build is
        # bit-identical to the pre-F-7 revision (pinned per tensor in
        # tests/test_v6_t2_contrastive.py).
        # ⚠️ It is deliberately NOT referenced anywhere in ``forward``. The T2
        # loss calls it directly, so ``forward``'s output dict gains no key and
        # `test_v6_gstr_port.py::
        # test_default_forward_is_bit_identical_and_emits_no_new_key` — which
        # already caught one unconditional key — cannot be tripped by this cell.
        self.t2_head = None
        if cfg.t2_contrastive:
            self.t2_head = ManoeuvreContrastiveHead(
                cfg.d_tac, d_proj=cfg.d_t2_proj, hidden=cfg.d_t2_hidden,
                tau=cfg.t2_tau)

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
        # ⭐ F-7 / T2: the contrastive projector is `layer_tac` — it reads
        # `z_tac` and nothing else, and S-T is the stage whose loss flows
        # through it. Grouping it with the adapter it consumes gives the same
        # train-when-live property the F-1 `cond_tac_dyn.` port has, with no
        # regrouping and no widening of ISOLATION_MATRIX["layer_tac"].
        ("t2_head.", "layer_tac"),
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
        # ⭐ the DIFFUSION PROPOSAL GENERATOR is a PLANNER module for the same
        # reason the emission is: it IS the fan generator, trained by the plan
        # loss in S-T on the trunk it consumes. (MpcRefiner and FallbackTrigger
        # hold NO parameters — a procedure and a calibrated comparator — so
        # they need no group: group_of partitions parameters, and there is
        # nothing of theirs for a stage to train or freeze.)
        ("prop_diffusion.", "planner"),
        ("masked_cells.", "aux"), ("sigreg.", "aux"),
        # ⭐ THE AGENT-SLOT DECODER IS `interp`, NOT `aux` — and the distinction
        # is the whole X3 argument, not tidiness. `aux` MAY backprop into the
        # encoder (ISOLATION_MATRIX: O3/O6 are label-free trunk losses and that
        # is their job); this head's supervision is a PERCEPTION LABEL, which
        # the diagram's header row forbids in any trunk loss. Filed under `aux`
        # the matrix would PERMIT the edge and `assert_isolation` would report a
        # pass over a real violation.
        ("agent_slots.", "interp"),
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
    def encode_window(self, frames: Tensor, *, return_tokens: bool = False):
        """``frames`` [B, W, C, H, W'] -> operative states [B, W, d_op].

        ``return_tokens=True`` additionally returns the encoder's PATCH TOKENS
        [B, W, n_tokens, d_model] — the ``slot_src="tokens"`` arm's memory
        (F-18). ⚠️ The default path is untouched: same call, same tensors, same
        RNG; the tokens were always computed and simply discarded by the
        readout. Keeping them is opt-in because at the full geometry they are
        B x W x 640 x 768 floats, which is not a cost to pay for every forward
        that does not want them.
        """
        b, w = frames.shape[:2]
        flat = frames.reshape(b * w, *frames.shape[2:])
        tok = self.encoder(flat)
        z = self.readout(tok).reshape(b, w, -1)
        if return_tokens:
            return z, tok.reshape(b, w, *tok.shape[1:])
        return z

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

    def roll_consistency(self, states: Tensor, actions: Tensor,
                         a_ctl: Tensor, kappa: Tensor, v0: Tensor, *,
                         intent: Tensor | None = None, k: int) -> Tensor:
        """Per-candidate IMAGINED-vs-CANDIDATE consistency cost [B, N]: roll
        ``predictor_op`` ``k`` steps under each candidate's OWN controls,
        decode the imagined per-step Δpose through ``step_readout_op``,
        accumulate SE(2), and return the mean distance to the candidate's own
        unicycle waypoints over those steps — the W7 quantity, computed on the
        v6 stack.

        ⛔ TWO USES SURVIVE ITS REFUTATION AS A SELECTOR, AND ONLY TWO:
          1. the CONTEXT BRAIN's uncertainty signal — its VARIANCE across the
             fan feeds the P7-calibrated fallback band (P7 rho 0.7164
             [0.5847, 0.7696], the validated use);
          2. a REGULARIZER inside the MPC refinement's composed cost
             (``mpc_w_consist``, default 0), behind the goal-conditioned
             primary term.
        ⛔ NEVER a selector: as a selection rule this exact quantity is
        MEASURED +5.9787 [+5.3217, +6.7625] WORSE than the trained selector,
        with error-rank RISING in N (0.241 → 0.286) — the winner's curse. No
        argmin/argmax over this tensor may pick a candidate.

        Trunk inputs (``states``, ``actions``, ``intent``) are DETACHED HERE,
        unconditionally: the roll is an INSTRUMENT — it must never train the
        trunk or the goal path (the ``zh_op_seam`` discipline, applied to a
        measurement). Gradient still flows to ``a_ctl``/``kappa`` so the MPC
        inner descent can differentiate w.r.t. the CONTROLS.

        Candidate controls enter the predictor in the RECORDED action format:
        channel 0 = steer_road_rad (``steer_of_kappa``, the corpus encoding),
        channel 1 = accel, channel 2 = ``v0/SPEED_SCALE`` held constant — the
        ``_lift3`` convention the trainer itself uses, mirrored, not improved.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        _ensure_scripts()
        from stage_a_probes import steer_of_kappa          # noqa: E402
        from train_v58f_unicycle_head import (             # noqa: E402
            SPEED_SCALE, unicycle_rollout)
        b, n, t = a_ctl.shape
        if k > t:
            raise ValueError(f"k={k} exceeds the plan horizon {t}")
        st = states.detach()
        aw = actions.detach()
        w = st.shape[1]
        # expand the shared window per candidate: [B*N, W, ·]
        st = st[:, None].expand(b, n, w, st.shape[-1]).reshape(b * n, w, -1)
        aw = aw[:, None].expand(b, n, w, aw.shape[-1]).reshape(b * n, w, -1)
        it = None
        if intent is not None:
            it = intent.detach()[:, None].expand(b, n, -1).reshape(b * n, -1)
        # candidate controls -> recorded-format future actions [B*N, k, 3]
        steer = steer_of_kappa(kappa)
        vcol = (v0.detach().to(a_ctl.dtype) / SPEED_SCALE)[:, None, None] \
            .expand(b, n, t)
        fa = torch.stack([steer, a_ctl, vcol], dim=-1) \
            .reshape(b * n, t, 3)[:, :k]
        # the same window-shift roll as metric_dynamics.rollout_transitions,
        # with the intent port added (that helper has none — stated, not
        # silently diverged: same 1-step head, same shift).
        dposes = []
        win_s, win_a = st, aw
        for j in range(k):
            z_hat = self.predictor_op(win_s, win_a, intent=it)[1]
            dposes.append(self.step_readout_op(win_s[:, -1], z_hat))
            if j < k - 1:
                win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
                win_a = torch.cat([win_a[:, 1:], fa[:, j].unsqueeze(1)],
                                  dim=1)
        from tanitad.models.metric_dynamics import accumulate_se2
        roll_wp = accumulate_se2(torch.stack(dposes, dim=1))   # [B*N, k, 2]
        uni_wp, _ = unicycle_rollout(a_ctl, kappa, v0.detach(), dt=self.cfg.dt)
        return (roll_wp.reshape(b, n, k, 2)
                - uni_wp[:, :, :k]).norm(dim=-1).mean(dim=-1)

    def emit(self, z_op: Tensor, g_tac_embed: Tensor, v0: Tensor, *,
             roll_ctx: dict | None = None,
             generator: torch.Generator | None = None) -> dict:
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

        ⭐ Under ``cfg.proposals == "diffusion"`` the LIVE fan (``a``/``kappa``/
        ``controls``/``waypoints``) comes from :class:`DiffusionProposalGenerator`
        and the query fan is STILL emitted beside it as ``qfan_*`` — the
        paired on-window reference of the pre-registered arm comparison, and
        what keeps ``emission.*``/``cand_queries`` reachable by the X3
        totality probe. ``generator`` seeds the noise draw (None = global RNG).

        ``roll_ctx`` (``{"states", "actions", "intent"}``, all DETACHED by
        the caller) enables the P_O rolls the MPC consistency regularizer and
        the fallback's roll-cost-variance signal need; without it those two
        halves are skipped and say so.
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
        if self.prop_diffusion is None:
            a_ctl, kappa, wp = self.emission(feat, v0)
            extra_fan = {}
        else:
            # the query/CV fan first — the paired reference, same window
            q_a, q_k, q_wp = self.emission(feat, v0)
            d = self.prop_diffusion(base[:, 0], g_tac_embed, v0,
                                    generator=generator)
            a_ctl, kappa, wp = d["a"], d["kappa"], d["waypoints"]
            extra_fan = {
                "prop_mechanism": "diffusion",
                # ⚠️ MEASURED on the actual draw, never asserted from cfg
                "prop_noise_lag1_autocorr": d["noise_lag1_autocorr"],
                "prop_noise_rho_target": cfg.diffusion_noise_rho,
                "prop_n_denoise_steps": d["n_denoise_steps"],
                "qfan_a": q_a, "qfan_kappa": q_k, "qfan_waypoints": q_wp}
        out = {"a": a_ctl, "kappa": kappa,
               "controls": torch.stack([a_ctl, kappa], dim=-1),
               "waypoints": wp, "feat": feat} | extra_fan
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
        if self.mpc is not None:
            # ⭐ MPC TOP-K REFINEMENT — warm-started by the trained selector's
            # scores, refined by cost descent on a composed cost (goal PRIMARY;
            # kin + imagined-consistency REGULARIZERS), re-scored by GOAL
            # DISTANCE ONLY (D-1). Config guarantees selector == "goal" here,
            # so `sel_score` and `sel_goal_point` exist. Every input is
            # detached and every output is detached: the refinement trains
            # NOTHING and nothing trains through it.
            roll_fn = None
            if cfg.mpc_w_consist > 0.0 and cfg.mpc_roll_k > 0:
                if roll_ctx is not None:
                    roll_fn = (lambda aa, kk: self.roll_consistency(
                        roll_ctx["states"], roll_ctx["actions"], aa, kk, v0,
                        intent=roll_ctx.get("intent"), k=cfg.mpc_roll_k))
                else:
                    out["mpc_consist_skipped"] = ("no roll_ctx — the "
                                                  "consistency regularizer "
                                                  "needs the window states/"
                                                  "actions")
            out |= {f"mpc_{k}": v for k, v in self.mpc.refine(
                out["a"], out["kappa"], v0, sel_score=out["sel_score"],
                goal_point=out["sel_goal_point"], roll_fn=roll_fn).items()}
        if self.fallback is not None:
            # ⭐ THE CONTEXT-BRAIN FALLBACK TRIGGER — under no_grad in FULL:
            # the uncertainty signal is MONITORED, never optimised (a signal
            # the training loss could reduce is a trigger the model learns to
            # blind). Roll-cost VARIANCE is the P7-validated use of the roll
            # quantity; its selection use stays refuted, and the module is
            # permutation-invariant over candidates so it CANNOT select.
            with torch.no_grad():
                roll_cost = None
                if cfg.fallback_roll_k > 0 and roll_ctx is not None:
                    roll_cost = self.roll_consistency(
                        roll_ctx["states"], roll_ctx["actions"],
                        out["a"], out["kappa"], v0,
                        intent=roll_ctx.get("intent"),
                        k=cfg.fallback_roll_k)
                out |= {f"fb_{k}": v for k, v in
                        self.fallback(out["waypoints"], v0,
                                      roll_cost=roll_cost).items()}
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
        # ⚠️ The token branch runs ONLY for the F-18 `slot_src="tokens"` arm.
        # With the slot decoder off (the default) this is the pre-F-18 call,
        # byte-for-byte — same tensors, same RNG, no extra allocation.
        tok_win = None
        if self.agent_slots is not None and cfg.slot_src == "tokens":
            z_op_win, tok_win = self.encode_window(frames, return_tokens=True)
        else:
            z_op_win = self.encode_window(frames)                  # [B,W,d_op]
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
        # roll context for the MPC consistency regularizer and the fallback's
        # roll-cost-variance signal — DETACHED AT CONSTRUCTION (the zh_op_seam
        # discipline): the rolls are instruments/regularizers over a frozen
        # view of the trunk, never a third gradient path into it.
        roll_ctx = None
        if (self.fallback is not None and cfg.fallback_roll_k > 0) or \
                (self.mpc is not None and cfg.mpc_w_consist > 0.0
                 and cfg.mpc_roll_k > 0):
            roll_ctx = {"states": z_op_win.detach(),
                        "actions": actions.detach(),
                        "intent": self._cut(e_g_tac, cut).detach()}
        plan = self.emit(z_plan, e_g_tac, v0, roll_ctx=roll_ctx)
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
                     "anchor_cls_logits",
                     # ⭐ the DIFFUSION arm's paired query/CV reference fan —
                     # planner-side because it flows through emission/
                     # cand_queries, and it is exactly what keeps those
                     # parameters REACHABLE by the totality probe when the
                     # live fan comes from the denoiser. (The live fan's
                     # a/kappa/waypoints below then reach prop_diffusion.)
                     # MPC outputs are DETACHED by construction and the
                     # fallback runs under no_grad — both are graph-free, so
                     # declaring them here would probe nothing; their
                     # isolation is structural, not measured-by-backprop.
                     "qfan_a", "qfan_kappa", "qfan_waypoints")
                    if k in plan]
        # the FACTORED pair is planner-side for the same reason the mixed head
        # is: it is a goal head, and X3 forbids any goal head reaching an
        # encoder.
        fact_side = [t for h in (g_tac_lat, g_tac_lon) if h is not None
                     for t in (h["logits"], h["args"])]
        cat_side = [h["cat_logits"] for h in (g_str, a_str, g_tac, g_tac_lat,
                                              g_tac_lon, a_lat, a_lon)
                    if h is not None and "cat_logits" in h]

        # ---- F-18: THE PERCEPTION AGENT-SLOT DECODER ------------------------
        # ⛔ VISION-ONLY: the memory below is a function of ``frames`` and of
        # nothing else. No v0, no actions, no goal embedding, no pose — and the
        # decoder's signature takes one positional tensor, so there is no door
        # for one to arrive through later either.
        # ⛔ X3: the memory is CUT (``_cut``) under
        # ``isolate_interp_from_encoder``, so the PERCEPTION-LABEL gradient
        # stops at the head. This is the SAME construction as ``zh_op_seam``'s
        # detached-trunk forward — a head that is admissible in a declared
        # isolation surface BY CONSTRUCTION rather than by hope — and the
        # ``perception_to_trunk`` probe in ``assert_isolation`` measures it on
        # the real graph. Setting the flag False is the mis-wired control arm
        # that makes that probe able to FAIL.
        slots = None
        if self.agent_slots is not None:
            icut = cfg.isolate_interp_from_encoder
            mem = (self._cut(tok_win[:, -1], icut) if tok_win is not None
                   else self._cut(self.cells(z_op), icut))
            slots = self.agent_slots(mem)

        out = {
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
        # ⭐ THE DECLARED INTERPRETATION SURFACE (F-18). Every tensor here must
        # carry ZERO gradient into ANY parameter outside the `interp` group — a
        # stricter target than `planner_side`'s, because what flows through
        # these tensors is a PERCEPTION LABEL.
        # ⛔ THE KEYS ARE ADDED ONLY WHEN THE HEAD EXISTS, and that is not
        # tidiness — it was CAUGHT. Returning them unconditionally (even as
        # ``None``) broke `test_v6_gstr_port.py::
        # test_default_forward_is_bit_identical_and_emits_no_new_key`, whose
        # whole job is that the DEFAULT forward's key set is part of the
        # inertness contract the live S-W resume stands on. It also keeps
        # ``assert_isolation`` honest: with no head there is no
        # ``perception_to_trunk`` key at all, rather than a probe over an
        # absent module reporting zero violations and establishing nothing.
        # ⚠️ Every emitted field is listed, not a representative one: the
        # `intent_proj` defect was a path present in the design and absent from
        # the optimisation, and an under-declared surface is the same defect
        # moved into the audit.
        if slots is not None:
            out["agent_slots"] = slots
            out["interp_side"] = [
                slots["presence_logit"], slots["cls_logits"], slots["box"],
                slots["yaw_vec"], slots["rates"], slots["occ_logit"]]
        return out

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
          4. **perception → trunk** (F-18, present ONLY when the agent-slot
             decoder is built) — the interpretation head may reach NOTHING but
             its own ``interp`` group. This edge is stricter than 1: it
             protects the trunk not from a planning gradient but from a
             PERCEPTION LABEL, which the diagram's header row forbids in any
             trunk loss. ⚠️ The key is ABSENT when the head is absent, on
             purpose — a probe over a module that does not exist reports zero
             violations and has established nothing, which is the vacuous pass
             this method's own ``requires_grad_(True)`` dance exists to avoid.

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
            # ⭐ F-18's edge — added ONLY when the head exists (see the
            # docstring). The forbidden set is EVERYTHING that is not `interp`,
            # which is the ISOLATION_MATRIX row written out: "interp": ("interp",).
            not_interp: list = []
            if out.get("interp_side") is not None:
                not_interp = [(n, p) for n, p in self.named_parameters()
                              if self.group_of(n) != "interp"]
                checks["perception_to_trunk"] = self._live_edges(
                    self._probe_scalar(out["interp_side"]), not_interp)
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
                "shared_encoder": self.cfg.shared_encoder,
                "isolate_interp_from_encoder":
                    self.cfg.isolate_interp_from_encoder,
                "agent_slots": self.agent_slots is not None,
                "slot_src": self.cfg.slot_src},
            "violations": {k: v[:12] for k, v in checks.items() if v},
            "n_violations": {k: len(v) for k, v in checks.items()},
            "n_probed": {"planner_to_encoder": len(enc),
                         "tactical_to_below": len(below_tac),
                         "strategic_to_below": len(below_str),
                         **({"perception_to_trunk": len(not_interp)}
                            if "perception_to_trunk" in checks else {})},
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
