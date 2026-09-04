"""REF-A **v1** — the redesigned frozen-encoder arm.

⭐ WHAT CHANGED FROM REF-A, AND WHY EACH CHANGE HAS A NAMED SOURCE.

REF-A (`refa.py`) was: frozen **DINOv2-B/14**, 224x224 -> **256 tokens / 51.39
deg**, a **pooling/grid adapter into a compact state**, a **supervised** head on
top, **no test-time planning**. Measured: ADE@2s **2.1675 m** (T0), plateaued.

Two independent lines of evidence then landed on the same day:

* **Ours** (`…/Research/2026-08-18-encoder-localisation/`, verdict
  ``P2-PRESERVED``): the information REF-A was accused of losing is **present
  and preserved at five measured stages** — raw features 0.5285, trained adapter
  0.4751, predictor latent 0.4762 / 0.4863 — arriving intact at the exact latent
  the eval decoded. The trained adapter did **not** collapse (per-dim std 0.8011
  vs 0.220 random). ⇒ The deficit is on the **consumption** side.
* **The literature** (`…/Research/2026-08-18-frozen-encoder-literature/`):
  frozen encoders succeed in exactly two configurations — (A) a *very large*
  frozen VLM + wide visual interface + supervised head (FROST-Drive: frozen 14B
  **8.17 RFS / 1.04 m** beats the SAME encoder fine-tuned **8.13 / 1.47**, while
  a frozen *ImageNet* ViT is the worst arm in the table at **7.39 / 2.28**), or
  (B) a *moderate* frozen encoder + **future-feature prediction** + **test-time
  planning** (DINO-WM, V-JEPA 2-AC; in driving DeepSight and LAW). REF-A had
  configuration A's consumer with configuration B's encoder class **and neither
  one's compensating strength** — the one cell nothing succeeds in.

⇒ **v1 commits to configuration B, fully**, and fixes the interface defects that
were configuration-independent.

| # | change | source |
|---|---|---|
| 1 | encoder **DINOv3** (ViT-L/16, d=1024), still frozen, still cached | DeepSight uses DINOv3-ViT-L/16 as its world-state target; PI directive |
| 2 | **640 patch tokens, 120 deg HFOV, 256x640** (was 256 tokens / 51.39 deg) | DINOv2's H/14 map is documented-insufficient for small/distant objects; our own w120 geometry decision |
| 3 | ⛔ **no bottleneck**: adapter width >= encoder width (1024) | FROST-Drive interface-width ablation: 5120-d **8.17** vs 256-d **7.68** on the SAME frozen encoder |
| 4 | primary objective = **predict future PATCH features** (L2), not a supervised head | DINO-WM: latent L2, "no auxiliary reconstruction, reward, or terminal losses", no policy head |
| 5 | **patch tokens only, never CLS/pooled** for the predictive path | DINO-WM ablation: global R3M / ResNet18 / **DINOv2 CLS** "significantly degrades" |
| 6 | behaviour from **iCEM + MPC at test time** (`refa_v1_plan.py`) | DINO-WM / V-JEPA 2-AC / GPC; repairs C101 |
| 7 | hierarchy **kept**: strategic --FiLM--> tactical --FiLM--> operative | our `fourbrain.run_hierarchy`; PI directive |
| 8 | goals enter the **planning COST**, not only a head | v3 direction (`tanitad-v3-direction`): "target-speed + mode-switching become the PLANNING COST not a head" |
| 9 | **6 s predictive horizon** at three rates, strategic on its OWN predictor over a strategy-only subspace | PI directive + three-planner hierarchy directive |

⚠️ **WHAT THIS DESIGN DOES NOT CLAIM.** Nothing here is a result. The ranking
that motivates it is a hypothesis ranking, and the recipe imports our
known-worst component (the action search) — which is why `refa_v1_plan.py`
carries a structural floor *and* a cost-fidelity gate rather than trust.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.fourbrain import StrategicPolicy, TacticalPolicy
# ⭐ THE ONE COMMAND<->GEOMETRY BRIDGE (PI ruling 2026-09-03). Imported, never
# re-derived: a second conversion is a second convention.
from tanitad.models.kinematic import _check_units, as_command
# ⛔ ONE VOCABULARY SOURCE. These tuples are IMPORTED, never re-declared — a
# second copy is a second vocabulary, and the programme has already paid for
# that once (see the defect note below).
from tanitad.models.v6 import (TACTICAL_LAT_ACTIONS,
                               TACTICAL_LON_ACTIONS, tactical_lat_actions,
                               tactical_lon_actions_v)
from tanitad.refs.refa_v1_plan import PlanConfig, icem_plan, unicycle_paths

__all__ = ["RefAV1Config", "RefAV1", "DINOV3_GEOMETRY", "SPEED_SCALE_MPS",
           "TACTICAL_LAT_ACTIONS", "TACTICAL_LON_ACTIONS"]

#: The frozen-encoder contract. Cached offline; the encoder never enters the
#: graph (REF-A stability item 2, preserved verbatim in v1).
DINOV3_GEOMETRY = {
    "model": "dinov3-vit-l-16",
    "d_enc": 1024,
    "patch": 16,
    "height": 256, "width": 640,          # 2.5 aspect — SAME as v6's 224x560,
    "grid_h": 16, "grid_w": 40,           # so this is a pure resolution scale
    "n_tokens": 640,                      # 16*40, vs REF-A's 256
    "hfov_deg": 120.0,                    # vs REF-A's 51.39
    "tokens_include_cls": False,          # ⛔ patch tokens ONLY (change #5)
}

#: Fixed normaliser for the speed channel (`RefAV1Config.speed_channel`), in
#: m/s. 30 m/s is the corpus's highway ceiling, so v/30 spans ~[0, 1] beside
#: (a, kappa), which stay in raw SI units. ⚠️ A CONTRACT with every consumer
#: that rolls the predictor (forward, plan(), any T1 harness): change it and a
#: checkpoint trained under it decodes a different speed. The flagship line
#: carries the same role as `SPEED_SCALE = 10.0` (`flagship_losses.py`,
#: registry §1.2) — a separate contract, named separately on purpose.
SPEED_SCALE_MPS = 30.0

# --------------------------------------------------------------------------- #
# ⭐ THE TACTICAL BRAIN'S DECODED ACTION AS KINEMATICS — one table, one function.
#
# `plan()` with no supplied goal imagines the goal from the tactical brain's own
# decision (design change #8: goals enter the planning COST). That rollout needs
# an (a, kappa) sequence, and the decoded action is a TOKEN. The label emitter
# (`stack/scripts/s2_geom_emit_v7.py`) DEFINES each token by a speed change over
# the tactical band [2, 6] s and by arc geometry, so each token becomes the
# SIMPLEST profile satisfying its own definition:
#   * longitudinal — a first-order approach to the token's target speed with
#     time constant GOAL_REACH_S = TACTICAL_S[0] (the band opens at 2 s), rate
#     saturated at GOAL_A_MAX = DV_BRAKE_MS per second = the `decel_1.5` floor
#     magnitude (`refa_v1_plan._baseline_controls`);
#   * lateral — TURN as a constant-curvature arc over the band width (4 s) at
#     the measured junction radius (emitter :245, R 12.2 m -> kappa 0.08);
#     LANE_CHANGE / NUDGE as S-curves sized by the unicycle small-angle relation
#     y = v^2 kappa t^2 / 2 at the MEASURED v0 (admissible: PI 2026-09-02).
# ⚠️ A CONVENTION, not a measurement — reviewable in this one place. Unknown
# tokens (a vocabulary this table predates) map to (0, 0): the goal degrades to
# "keep going", never to a crash. Nothing here reads the future.
# --------------------------------------------------------------------------- #
GOAL_REACH_S = 2.0                #: s  — s2_geom_emit_v7.TACTICAL_S[0]
GOAL_A_MAX = 1.5                  #: m/s^2 — DV_BRAKE_MS/s; = the decel_1.5 floor
GOAL_KAPPA_MAX = 0.2              #: 1/m — PlanConfig.kappa_max
GOAL_KAPPA_TURN = 0.08            #: 1/m — R 12.5 m (emitter's measured 12.2 m)
GOAL_TURN_S = 4.0                 #: s — TACTICAL_S[1] - TACTICAL_S[0]
GOAL_LANE_CHANGE = (2.0, 1.75)    #: (half-duration s, half-offset m: 3.5 m lane)
GOAL_NUDGE = (1.0, 0.5)           #: (half-duration s, half-offset m: 1 m shift)
GOAL_V_REF_MIN_MPS = 3.0          #: m/s floor for the S-curve sizing at low v0
GOAL_CREEP_MPS = 1.5              #: m/s — midpoint of the emitter's CREEP band
GOAL_CURVE_VMAX_MPS = 8.0         #: m/s — emitter TURN_MAX_VMIN_MS
#: target speed RELATIVE to v0 (m/s): the emitter's own dv thresholds.
GOAL_LON_DV_MPS = {"CRUISE": 0.0, "FOLLOW": -1.0, "ACCELERATE": +1.5,
                   "YIELD_MERGE": -1.5, "BRAKE_TO": -3.0}

#: ⭐ THE COARSE COST'S TIME GRID (`plan(cost_time_grid=...)`, 2026-09-03).
#: The planner optimises `plan_steps` OPERATIVE actions (0.2 s each); with
#: `plan_level="tactical"` the cost rolls the TACTICAL predictor, whose step is
#: `tac_dt` = 0.6 s.
#:   * ``"dense"``    — DEFAULT and byte-identical to every pre-2026-09-03
#:     caller: all `H` operative entries are fed straight to the tactical
#:     predictor, so the 2.0 s plan is imagined as 6.0 s and the candidate's
#:     tactical action `j` is operative action `j`. ⛔ This is the LEGACY path
#:     and it is wrong in time the same way ``model_action_units="kappa"`` is
#:     wrong in units: EVERY tactical action the predictor was trained on is
#:     operative action ``j*stride`` (`forward`, ``tac_a``), and the imagined
#:     GOAL is built on that same grid (`_imagine_tactical_goal`). Kept as the
#:     default only so every banked number stays reproducible.
#:   * ``"tactical"`` — tactical step `j` consumes operative action
#:     ``min(j*stride, H-1)``: exactly ``[::stride]`` while the plan lasts, then
#:     a zero-order hold of the plan's final action, which is the standard
#:     receding-horizon continuation and the only defined way to span the 6 s
#:     the cost is documented to span (`plan_horizon_s`, :247). The candidate's
#:     terminal field then sits at the goal's own 6.0 s.
COST_TIME_GRIDS = ("dense", "tactical")


def _check_cost_time_grid(grid: str) -> str:
    if grid not in COST_TIME_GRIDS:
        raise ValueError(f"cost_time_grid must be one of {COST_TIME_GRIDS}, "
                         f"got {grid!r}")
    return grid


#: ⛔ THE GOAL'S OWN TIME GRID (`plan(goal_time_grid=...)`, 2026-09-03, L0 /
#: BACKLOG R38). THE SEED IS NOT THE GOAL, and no re-gridding can make it be.
#: MEASURED (`2026-09-03-tactical-decoder/raw/seed_goal_mismatch.json`):
#: **62 of the 64 (lat, lon) token pairs** give a different tactical action
#: sequence to the GOAL and to the SEED that chases it -- under BOTH
#: `COST_TIME_GRIDS` -- with `TURN_L` over-rotating its own goal by **82.5 deg**;
#: the only agreeing pairs are `(LANE_KEEP, CRUISE)` and `(ABORT_LC, CRUISE)`,
#: i.e. the all-zero control.
#:
#: ⭐ THE CAUSE IS A TRUNCATION, NOT A REGRID. `_imagine_tactical_goal` rolls
#: the goal from operative indices ``[0, stride, ..., (tac_steps-1)*stride]`` =
#: ``[0,3,...,27]`` (6.0 s), while the seed is ``controls[:cfg.plan_steps]`` =
#: ``[0..9]`` (2.0 s, `plan_horizon_s` 2.0 / `op_dt` 0.2). **The goal's actions
#: at operative indices 12, 15, 18, 21, 24 and 27 are not in the seed at all**,
#: so no `cost_time_grid` can address them: ``"dense"`` reads ``[0..9]`` and
#: ``"tactical"`` reads ``[0,3,6,9,9,9,9,9,9,9]``.
#:   * ``"full"``  -- DEFAULT and byte-identical to every pre-2026-09-03 caller:
#:     the goal is the token's FULL `op_steps` manoeuvre subsampled onto the
#:     tactical grid. The 62/64 mismatch is preserved, because every banked
#:     number was produced under it and must stay reproducible.
#:   * ``"plan"``  -- the goal is re-rolled from the SEED'S OWN action feed: the
#:     same `_model_actions` call, the same `tac_idx` re-grid, the same
#:     predictor and the same ``z0`` a candidate gets, so the canonical seed's
#:     cost rollout and the goal rollout are THE SAME FORWARD PASS and
#:     ``1 - cos`` is 0 by construction on 64/64 tokens, in both
#:     `COST_TIME_GRIDS` and at both `plan_level`s.
#: ⚠️ ``"plan"`` BUYS IDENTITY, NOT HORIZON -- say so in any claim. The goal
#: it builds is the manoeuvre's first `plan_horizon_s` (``"dense"``) or its
#: first `plan_horizon_s` held to 6 s (``"tactical"``), NOT the 6 s manoeuvre
#: the token names. Making the seed span the token's own manoeuvre needs
#: `plan_horizon_s` 2.0 -> 6.0, which changes the optimised window, the search
#: dimensionality, the proposal head's output shape (:1093) and the length of
#: the returned plan -- a DESIGN decision, not a repair, and not this file's to
#: take unilaterally.
#: ⛔ A THIRD "REPAIR" THAT LOOKS RIGHT AND IS WRONG, recorded so it is not
#: re-proposed: packing the goal's stride-3 actions into the 10 seed slots
#: (``seed = ctrl[::stride][:plan_steps]``) does give 64/64 -- under ``"dense"``
#: ONLY -- but the plan is EXECUTED on the operative grid at `op_dt`
#: (`PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt)`) and the coarse->fine
#: re-score rolls `self.operative` over the same tensor with no regrid, so it
#: would drive a 6 s manoeuvre in 2 s. Under ``"tactical"`` it cannot work at
#: all: `tac_idx` takes 4 distinct values and cannot address 10 distinct
#: goal actions.
GOAL_TIME_GRIDS = ("full", "plan")


def _check_goal_time_grid(grid: str) -> str:
    if grid not in GOAL_TIME_GRIDS:
        raise ValueError(f"goal_time_grid must be one of {GOAL_TIME_GRIDS}, "
                         f"got {grid!r}")
    return grid


#: ⛔ THE GOAL TERM'S METRIC (`plan(cost_metric=...)`, 2026-09-03, L3 / R29).
#: The shipped goal term is ``1 - cosine_similarity(z_terminal, goal)`` in
#: FLOAT32. Near ``cos = 1`` that is a CATASTROPHIC CANCELLATION against 1.0:
#: whatever the term's own magnitude, it can only take values that are integer
#: multiples of ``spacing(1f)/2 = 5.9604645e-08``.
#:   * ``"cos"``   -- DEFAULT and byte-identical to every pre-2026-09-03 caller.
#:   * ``"chord"`` -- the Euclidean distance between the two NORMALISED fields,
#:     ``||z/||z|| - g/||g||||``. On unit vectors ``chord = sqrt(2*(1-cos))``,
#:     so it is MONOTONE-EQUIVALENT and cannot re-rank two candidates in exact
#:     arithmetic. What it changes is the ARITHMETIC: the component difference
#:     ``x_i - y_i`` of two nearly equal normalised vectors is EXACT (Sterbenz),
#:     so the result carries only the normalisation's relative error and is
#:     resolved at the DIFFERENCE's own scale instead of at 1.0's.
#: ⭐ IT IS NOT WEIGHT-NEUTRAL, AND MUST NEVER SHIP ALONE. ``chord`` is
#: ``sqrt(2x)`` of the old term, so at ``x ~ 1e-9`` it is ~4.5e-05 -- the goal
#: term is multiplied by ``1/chord``, a measured **5,792.6x** implicit
#: re-weighting against the unchanged ``0.05*kappa^2`` penalty
#: (`PREREG_TACTICAL_DECODER.md` §5 L3: *"Swapping the metric while holding
#: 0.05 fixed is therefore NOT a one-variable arm"*). Any arm that flips this
#: flag must declare `W_KAPPA` in the same breath -- see the derivation banked
#: at `.../Research/2026-09-03-cost-repair/RESULT.md` §4.
#: ⚠️ ``sqrt(2*(1-cos))`` computed FROM the float32 cosine recovers NOTHING:
#: it inherits the 5.96e-08 quantum before the square root can act. The chord
#: must be computed as a NORM OF A DIFFERENCE, which is what this does, and
#: that distinction is the deliberate-regression arm of the L3 experiment.
#:
#: ⭐⭐ ``"ccos"`` -- THE CENTRED COSINE (2026-09-04, D-REFAV1-COST-FORM).
#: ``1 - cos(z_K - z_ref, g - z_ref)`` where ``z_ref`` is the ZERO-ACTION
#: (constant-velocity) terminal field for the SAME window, rolled by the SAME
#: predictor from the SAME ``z0`` in the SAME forward pass, so no future
#: information and no extra rollout beyond the one the ``cv`` baseline already
#: pays for.
#: ⛔ IT IS A DIFFERENT KIND OF CHANGE FROM ``"chord"`` AND THE TWO MUST NEVER
#: BE CONFLATED. ``chord = sqrt(2*(1-cos))`` is MONOTONE-EQUIVALENT: in exact
#: arithmetic it cannot re-rank two candidates, and it repairs float32
#: cancellation ONLY. **Subtracting a common vector CHANGES THE RANKING** --
#: ``cos(x, y)`` and ``cos(x - r, y - r)`` are different functions of the same
#: pair, and `test_cost_ccos.py::test_k_*` exhibits an exact candidate pair
#: whose order flips in float64. That re-ranking IS the repair.
#: ⭐ WHY IT IS NEEDED (MEASURED, `.../Research/2026-09-04-refav1-cost-scale/
#: RESULT.md` §2a/§4b, step 21,109, n = 282 windows / 141 clusters): the
#: compared fields are UNCENTRED and **99.3 % common mode**, so the shipped
#: term realises ~1.2e-05 of its own [0, 2] range while the CHEAPEST jerk
#: charge in a 300-sample iCEM population is ``0.02 * 4.711 = 0.094`` -- i.e.
#: **100.00 % of the iteration-0 population is excluded before the world model
#: is consulted**. Centring takes that a-priori exclusion to **1.67 %** and the
#: left/right decision from **1.4 % to 41.3 %** of the term's own magnitude, at
#: the SHIPPED weights.
#: ⛔ NOT WEIGHT-NEUTRAL EITHER, and by a much larger factor than the chord:
#: it rescales the goal term against an unchanged `W_JERK` / `W_KAPPA` /
#: `W_VEND`. Any arm that flips this flag declares the weight triple in the
#: same breath -- see `W_KAPPA` and the measured factor banked at
#: `.../Research/2026-09-04-refav1-centred-goal/RESULT.md` §3.
#: ⚠️ AND IT CARRIES A MEASURED DEGENERACY THAT IS NOT REPAIRED HERE. The
#: do-nothing candidate IS ``z_ref``, so its centred vector is the ZERO vector
#: and ``ccos(cv) = 1.0`` EXACTLY -- the worst-but-one value in the range, by
#: definition rather than by measurement (MEASURED 40/40 windows,
#: `raw/cost_forms_devbox.json`). On the **86.5 %** of the eval grid whose
#: decoded goal is LANE_KEEP the goal itself is "hold", ``||g - z_ref||`` is
#: float32 rounding (4.18e-08 relative), and the centred direction is noise.
#: A HOLD BRANCH (gate on ``||g - z_ref||`` and fall back to a distance) is a
#: PRE-REGISTRATION ITEM belonging to the PI / Master Mind, deliberately NOT
#: taken in this file. ⇒ ``"ccos"`` is an INSTRUMENTED OPTION, not a candidate
#: default; `_check_cost_metric` accepts it and nothing selects it.
COST_METRICS = ("cos", "chord", "ccos")

#: eps of `F.cosine_similarity`'s own default, applied per-vector here so a
#: zero field cannot divide by zero. Never reached on a real rollout.
_CHORD_EPS = 1e-8


def _check_cost_metric(metric: str) -> str:
    if metric not in COST_METRICS:
        raise ValueError(f"cost_metric must be one of {COST_METRICS}, "
                         f"got {metric!r}")
    return metric


def _goal_term(zt: Tensor, g: Tensor, metric: str = "cos",
               z_ref: Tensor | None = None) -> Tensor:
    """``[n, ...]`` terminal fields + ``[n, ...]`` goal -> ``[n]`` goal cost.

    THE ONE PLACE the planner's goal distance is computed. `_cost_chunk` calls
    it and nothing else does; a probe that wants to measure the SHIPPED metric
    calls this function rather than re-implementing it.

    ``"cos"`` is bit-identical to the pre-2026-09-03 expression
    ``1.0 - F.cosine_similarity(zt.flatten(1), g.flatten(1), dim=-1)`` -- it IS
    that expression, and adding ``z_ref`` did not touch that branch:
    `test_cost_chord.py::test_a_*` and `test_cost_ccos.py::test_a_*` both pin
    it bit-for-bit. See `COST_METRICS` for why ``"chord"`` is not a cosmetic
    reformulation, why ``"ccos"`` is a DIFFERENT KIND of change (it re-ranks;
    the chord provably cannot), and why neither may ship without a weight
    statement.

    ``z_ref`` ``[1, ...]`` or ``[n, ...]`` is the CENTRING REFERENCE and is
    REQUIRED by ``"ccos"`` and ignored by the other two. `plan()` supplies the
    zero-action terminal field of the same window, same predictor, same ``z0``
    -- never a mean over anything and never anything from the future.
    """
    x = zt.flatten(1)
    y = g.flatten(1)
    if metric == "cos":
        return 1.0 - F.cosine_similarity(x, y, dim=-1)
    if metric == "ccos":
        if z_ref is None:
            raise ValueError(
                "cost_metric='ccos' needs z_ref: the ZERO-ACTION terminal "
                "field of this window. Centring on anything else (a batch "
                "mean, a running average, the goal) is a different estimator "
                "and would leak across candidates -- see COST_METRICS")
        r = z_ref.flatten(1)
        return 1.0 - F.cosine_similarity(x - r, y - r, dim=-1)
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(_CHORD_EPS)
    yn = y / y.norm(dim=-1, keepdim=True).clamp_min(_CHORD_EPS)
    return (xn - yn).norm(dim=-1)


#: ⭐ THE EXPLICIT COST WEIGHTS -- named so that L4 is addressable in ONE place.
#: These are the values every banked refav1 number was produced under and they
#: are NOT changed by the 2026-09-03 L3 work; `test_cost_chord.py` pins them.
#: ⛔ `W_KAPPA` IS 99.5 % OF ALL COST VARIATION over the candidate box
#: (`D-REFAV1-COST-SURFACE`), against a world-model contribution of 1.63e-10
#: along kappa. Under the SHIPPED ``"cos"`` metric the weight that would let a
#: correct turn win is **1.66e-08** -- ``w*kappa_max^2 = 6.6e-10``, i.e. the
#: whole penalty over the whole box is ~90x SMALLER than one representable step
#: of the term it trades against. That is a DELETION of the penalty, not a
#: weight, and it is why L4 cannot be taken alone either. The derivation, the
#: chord-side weights and the deletion test are banked at
#: `.../Research/2026-09-03-cost-repair/RESULT.md` §4.
W_JERK = 0.02                                   #: comfort, on channel 0 only
W_KAPPA = 0.05                                  #: curvature, on RAW curvature
W_VEND = 0.10                                   #: terminal speed vs target


def canonical_controls(lat: str, lon: str, v0: float, op_steps: int,
                       op_dt: float) -> Tensor:
    """(lat token, lon token, measured v0) -> ``[op_steps, 2]`` (a, kappa) on
    the operative grid. Deterministic, future-free; see the table above."""
    a = torch.zeros(op_steps)
    k = torch.zeros(op_steps)
    # --- longitudinal: approach the token's target speed, then hold ------- #
    if lon == "HOLD":
        v_t = 0.0
    elif lon == "CREEP":
        v_t = GOAL_CREEP_MPS
    elif lon == "ADAPT_SPEED_FOR_CURVE":
        v_t = min(float(v0), GOAL_CURVE_VMAX_MPS)
    else:
        v_t = max(0.0, float(v0) + GOAL_LON_DV_MPS.get(lon, 0.0))
    v = float(v0)
    for i in range(op_steps):
        a_i = max(-GOAL_A_MAX, min(GOAL_A_MAX, (v_t - v) / GOAL_REACH_S))
        a[i] = a_i
        v += a_i * op_dt
    # --- lateral: x forward, y left, so +kappa turns LEFT (kinematic.py) --- #
    sign = 1.0 if lat.endswith("_L") else -1.0
    v_ref = max(float(v0), GOAL_V_REF_MIN_MPS)

    def _s_curve(half_s: float, half_m: float) -> None:
        n = int(round(half_s / op_dt))
        kap = min(GOAL_KAPPA_MAX, 2.0 * half_m / (v_ref ** 2 * half_s ** 2))
        k[:n] = sign * kap
        k[n:2 * n] = -sign * kap

    if lat.startswith("TURN_"):
        k[:int(round(GOAL_TURN_S / op_dt))] = sign * GOAL_KAPPA_TURN
    elif lat.startswith("LANE_CHANGE_"):
        _s_curve(*GOAL_LANE_CHANGE)
    elif lat.startswith("NUDGE_"):
        _s_curve(*GOAL_NUDGE)
    return torch.stack([a, k], dim=-1)


@dataclass
class RefAV1Config:
    #: ⭐ v7 mandate: new builds default to the FROZEN FlyWheel
    #: vocabulary; recorded configs keep their version.
    tac_vocab_version: str = "v7.0"
    """Every number that defines the arm, in one place, so a launch can be
    diffed against a checkpoint."""

    # --- frozen visual interface (change #1, #2, #3) ---------------------- #
    d_enc: int = 1024
    n_tokens: int = 640
    d_state: int = 1024                   # ⛔ MUST be >= d_enc (change #3)

    # --- operative: dense, 6 s (change #9) -------------------------------- #
    op_dt: float = 0.2
    op_steps: int = 30                    # 30 * 0.2 = 6.0 s
    op_layers: int = 6
    op_heads: int = 8
    op_window: int = 4                    # observed frames fed to the predictor

    # --- tactical: coarse, 6 s, on learned query tokens ------------------- #
    tac_dt: float = 0.6
    tac_steps: int = 10                   # 10 * 0.6 = 6.0 s
    tac_queries: int = 64
    tac_layers: int = 4

    # --- strategic: its OWN predictor on a strategy-only subspace --------- #
    # ⭐ PI DECISION 2026-08-31: **3.0 s × 2** — replacing the inexpressible
    # 1.5 × 4 (7.5 operative steps, silently rounded to 8 → the rung ran at
    # 1.6 s with 3 of 4 targets; see D-REFAV1-LADDER) and the interim 1.2 × 5.
    # 3.0/0.2 = 15 exactly, 2 × 15 = 30 = op_steps, targets at 3.0 / 6.0 s —
    # and the resulting ladder 0.2 : 0.6 : 3.0 = **1 : 3 : 15** matches the
    # ratio MM-E15 read off the corpus label bands.
    str_dt: float = 3.0
    str_steps: int = 2                    # 2 * 3.0 = 6.0 s in-window, stride 15
    str_dim: int = 256
    str_layers: int = 2

    # --- ⭐ the LONG-HORIZON strategic extension (PI 2026-08-31) ----------- #
    # Verbatim: *"The strategic layer must have its long horizon predictor in
    # the abstract latent space not only 6 seconds."* ⇒ §4b's 6.0 s binds the
    # CONTROL OUTPUT and the token-field levels; the strategic SUBSPACE
    # predictor continues PAST the operative grid, in its own latent, at its
    # own rate. Extension ticks are supervised from CACHED features at
    # 6.0 + k·str_dt (the episode is 20 s, so 12 s is always in-cache), fed as
    # ``str_ext_targets`` — they cannot come from the 6 s operative grid.
    # Default 12.0 s: 2 extra ticks at 9.0 / 12.0 s, reaching INSIDE the
    # strategic label band [8, 30) at MM-E15's median manoeuvre start
    # (12.5 s) — the band the 6 s rung provably never reached.
    str_horizon_s: float = 12.0
    w_feat_str_ext: float = 0.25          # same weight class as w_feat_str

    # --- control / planning ----------------------------------------------- #
    a_dim: int = 2                        # (a, kappa) — Alpamayo-2 form; the
                                          # CONTROL width (see `a_in_dim`)
    #: ⭐ THE SPEED CHANNEL (#5). REF-A's longitudinal blindness (3.73 m) was
    #: repaired by a speed input (0.83 m — `MODEL_REGISTRY.md`, the REF-A
    #: speed reset), and the deployed flagship carries `v0` as its third
    #: action channel for the same reason (registry §1.2). refav1's predictor
    #: sees (a, kappa) only, and its tmix is depthwise, so the ego speed has
    #: no route into the state at all.
    #: PI ruling 2026-09-02, verbatim: *"It is allowed to use the velocity as
    #: initial measured state at its cycle time. It is not allowed to use the
    #: future dynamic information from the ground truth."* ⇒ when on, the
    #: predictor INPUT is (a, kappa, v_k / SPEED_SCALE_MPS) with v_k
    #: INTEGRATED from the measured anchor speed, v_k = v0 + Σ_{j<k} a_j·dt
    #: (`augment_actions`) — never indexed from a future speed. The CONTROL
    #: space the planner searches stays `a_dim` = 2; v is derived from
    #: plan()'s measured v0 before every rollout, so T0 and T1 run the same
    #: code. ⛔ DEFAULT OFF: a next-run arm, and the live run's checkpoint must
    #: load. (H-REFAV1-MOTION's refuted injection was a SCENE-motion channel;
    #: this is the EGO state — a different lever, not a re-run of that one.)
    speed_channel: bool = False
    #: `WideAdapter.tmix` groups. None ⇒ d_state (depthwise — the shipped
    #: form, each channel mixes only its own past); 1 ⇒ full cross-channel
    #: temporal mixing, the next-run arm on H-REFAV1-MOTION's defect statement.
    tmix_groups: int | None = None
    plan_horizon_s: float = 2.0           # optimised window; cost spans 6 s
    goal_times_s: tuple = (2.0, 4.0, 6.0)

    #: ⛔ WHERE THE SEARCH ROLLS, AND WHY IT IS NOT THE OPERATIVE FIELD.
    #: MEASURED on an RTX 4060 at the real geometry: a 10-step rollout of the
    #: 640x1024 operative field costs **160 ms per candidate** and scales
    #: linearly (2553 / 5129 / 10519 ms at n = 16 / 32 / 64 — the GPU is already
    #: saturated at n=16), i.e. ~6 candidates/s. DINO-WM's published
    #: configuration (300 samples x 30 iterations, ~1975 rollouts after decay)
    #: therefore costs **325 s per MPC tick** here, ~54 s on an A40-class card.
    #: That is not a planner, it is a batch job.
    #:
    #: ⭐ The hierarchy already contains the fix: the TACTICAL field is 64 query
    #: tokens instead of 640, so the same search costs ~1/10th. Searching a
    #: manoeuvre against the coarse tactical world and then VERIFYING the winner
    #: on the full operative field is coarse-to-fine, and it is what the
    #: tactical level is for. ``"operative"`` stays reachable so the deviation
    #: from DINO-WM is a measurable ablation, not an unstated compromise.
    plan_level: str = "tactical"          # "tactical" | "operative"
    verify_on_operative: bool = True      # re-score the winner on 640 tokens

    # --- hierarchy brains (kept from REF-A / flagship) -------------------- #
    tactical_cfg: TacticalPolicyConfig | None = None
    strategic_cfg: StrategicPolicyConfig | None = None

    # --- loss weights: feature prediction is PRIMARY (change #4) ---------- #
    w_feat_op: float = 1.0
    w_feat_tac: float = 0.5
    w_feat_str: float = 0.25
    # ⚠️ WAS 0.1 — set to 0.0 on 2026-08-31 so the advertisement matches the
    # code (no target existed). ⭐ 2026-09-01: THE TARGET NOW EXISTS — the
    # loader emits the demonstrated (a, kappa) sequence, so `w_aux_head > 0`
    # trains the proposal by winner-takes-all against the human demo. Still
    # AUXILIARY and default-OFF: behaviour comes from planning; this head only
    # SEEDS the search (GPC), so imitation here cannot echo into the metric.
    w_aux_head: float = 0.0               # imitation proposal: AUXILIARY only

    # --- ⭐ MULTIMODAL proposals (Drive-JEPA 2601.22032, adapted) ----------- #
    # Drive-JEPA's supervision-bottleneck point: one scene, one human
    # trajectory, inherently multimodal futures. Their answer is a proposal-
    # centric planner distilling DIVERSE (simulator + human) trajectories with
    # momentum-aware selection. The refav1-shaped version: `proposal_k` modes
    # + a score head, trained WTA (only the closest mode regresses the demo,
    # so modes specialise instead of averaging), score CE toward the winner.
    # At plan() time ALL modes seed the iCEM population (they compete inside
    # iteration 0 and can seed the mean), and the momentum role is played by
    # the planner's existing `prev_elites` memory across ticks.
    # ⛔ THE SAFETY OF THE SLOT is the point: proposals only ever SEED the
    # search — the selected behaviour still comes from imagined-consequence
    # cost, so distilled diversity cannot become an imitation echo. Simulator
    # distillation (AlpaSim rollouts / rule-scored candidates over the
    # obstacle joins) slots in later as extra demo rows, same loss.
    proposal_k: int = 1                   # 1 = the original single proposal

    # --- change #10: the COUNTERFACTUAL-ACTION term (PI 2026-08-31) -------- #
    # ⛔ WHY v1 NEEDED A TENTH CHANGE. Change #4 makes the primary loss "predict
    # the future patch features". That target is TEACHER-FORCED: it already
    # contains the action's effect, so a predictor can match it WITHOUT using
    # the action at all. UWM-JEPA (2605.25313) states it and says the finding
    # "applies beyond the unitary parameterisation"; our own campaign measured
    # the same thing three ways -- action moves the prediction 0.4-0.6% as much
    # as the scene, ego_state adds -0.0006 (t -0.48) over drift on held-out
    # data, and the FiLM gain CONVERGED rather than straining. ⇒ v1 would have
    # inherited REF-A's original symptom: scores on context, ignores actions.
    #
    # ⭐ THE TERM CARRIES ITS OWN KNOWN-VALUE CONTROL, which is why it is an
    # instrument and not just a loss: roll the SAME state under the true future
    # actions and `cf_negs` counterfactual ones, and require the true rollout to
    # be the one matching the observed future. An action-INDEPENDENT predictor
    # scores EXACTLY ln(1 + cf_negs) and cannot do better -- so `cf_excess`
    # above 0 is proof the predictor used the action, with no baseline to argue
    # about.
    #
    # ⚠️ DEFAULT OFF (w_cf = 0.0) so this changes no existing behaviour, and it
    # adds NO parameters -- the state_dict is byte-identical either way.
    w_cf: float = 0.0
    cf_negs: int = 3
    cf_tau: float = 1.0
    #: Which operative rollout step to score. Deep enough that the action has
    #: moved the world, shallow enough to stay cheap. ⚠️ At op_dt 0.2 s, step 4
    #: is 0.8 s -- the horizon our own arms trained at, chosen so a null here
    #: is comparable to the banked action-divergence numbers rather than being
    #: a different question.
    cf_at_step: int = 4

    # --- observed-MOTION injection (PI 2026-08-31: "implement and try 1") -- #
    # ⚠️ THE DEFECT, STATED CORRECTLY THE SECOND TIME. First draft claimed the
    # rollout state is "Markovian on one static frame" — MEASURED FALSE by this
    # feature's own control test: ``WideAdapter.tmix`` is a temporal conv, so
    # history DOES reach ``field[:, -1]``. But tmix is DEPTHWISE (groups =
    # d_state, kernel 3): each channel mixes only its own past — there is NO
    # cross-channel temporal path, and scene motion (an edge moving between
    # patch channels, an agent's parallax) is exactly a cross-channel signal.
    # When on, the initial state becomes  z + W_m(z_t − z_{t−1})  with a FULL
    # cross-channel W_m (down-scaled init, so training starts indistinguishable
    # from baseline), and all three levels inherit it because they read the
    # same injected state. ⛔ H-REFAV1-MOTION (2026-09-02): REFUTED for this
    # form at probe scale — the SHUFFLED-diff control gained as much as the
    # true difference (and more at k16-30), so the benefit is channel
    # augmentation, not motion. Stays default-OFF; no launch line may carry
    # it without a new prereg separating augmentation from motion.
    motion_inject: bool = False

    # --- v7.2 LABEL supervision (PI 2026-08-31: "It must be trained with this
    #     data") — auxiliary CE on the factored lat/lon decode and the route
    #     head, fed by the released v7.2 s2 label set via the loader. ⚠️
    #     AUXILIARY: future-feature prediction stays the primary loss (change
    #     #4); these make the decision heads TRAINED rather than inert.
    w_tac_label: float = 0.1
    w_str_label: float = 0.1

    # --- nav injection into the DECISION LAYERS (PI 2026-09-01) ------------ #
    # PI, after the echo concern was raised and answered: *"implement the nav
    # injection into the tactical policy and also the operative layer"* —
    # DECIDED. nav (PI-reviewed Alpamayo-CoT+ego derivation; NOT a situation-
    # classifier output, so admissible under the goal-disjointness rule) now
    # reaches: (1) the tactical policy DIRECTLY (added to its FiLM cond), and
    # (2) the operative + tactical FIELD PREDICTORS (added to the intent they
    # are conditioned on). ⛔ The strategic SUBSPACE predictor stays nav-free —
    # its prediction remains independently falsifiable. ⚠️ EVAL OBLIGATION that
    # travels with this: any nav-conditioned result carries a NAV-SHUFFLE
    # control, because a conditioned model can satisfy its conditioning
    # instead of the world (the C6 / nav-echo family).
    nav_inject: bool = True

    # --- target space for the PRIMARY (operative) term --------------------- #
    # ⛔ "adapter" (the original form) HAS A COLLAPSE MINIMUM: the target is
    # ``adapter(std(future))`` and the adapter is TRAINED, so mapping
    # everything to a constant zeroes the loss — LayerNorms raise the barrier
    # (affine γ→0 re-opens it) and the trainer's adapter_std monitor DETECTS
    # it, but nothing REMOVES the minimum. "frozen" predicts the standardised
    # DINOv3 features themselves (std has frozen buffers, fit once): the
    # target's variance is fixed at ~1 per channel, so a collapsed adapter
    # scores the target variance, not zero — DINO-WM's own arrangement.
    # ⭐⭐ DEFAULT FLIPPED TO "frozen" 2026-09-02 (PI). "adapter" was the default
    # only because it is the ORIGINAL REF-A form — a historical accident, not a
    # judgement. It is the collapse-prone one, and MEASURED on the first refav1
    # launch it collapsed: adapter_std 0.4763 -> 0.3385 over 450 steps with the
    # loss falling to 0.165 and LOOKING like the best run of the day. "frozen"
    # predicts the standardised DINOv3 features themselves (frozen std buffers)
    # — DINO-WM's own arrangement, and the only one of the two where the target
    # cannot move.
    target_space: str = "frozen"          # "adapter" | "frozen"

    # --- ANTI-COLLAPSE MECHANISMS (PI 2026-09-02) --------------------------- #
    # ⛔ WHY THESE EXIST. An audit against the banked primaries found refav1 had
    # ONE of the family's eight mechanisms (LayerNorm — whose own comment says
    # "affine gamma->0 re-opens it"). In I-JEPA (2301.08243) and BYOL
    # (2006.07733) the target comes from an EMA encoder BEHIND A STOP-GRADIENT;
    # in DINO-WM (2411.04983) from a FROZEN pretrained encoder. refav1's default
    # did neither — both sides of every feature loss flowed through the same
    # trained adapter, with gradient on both. That is not a JEPA; it is
    # self-prediction with a learnable target, the family those papers exist to
    # escape. Each knob below is SEPARATELY switchable so its effect stays
    # attributable — five changes in one arm would be the conflation error this
    # programme has already paid for.
    #: SimSiam (2011.10566): stop the TARGET chasing the prediction. The
    #: tactical and strategic targets are `_tac_field(tgt)` and
    #: `strategic.subspace(tgt)` — adapter-derived in BOTH target spaces, so
    #: `frozen` alone does not cover them.
    detach_aux_targets: bool = True
    #: ⭐ EMA TEACHER for the tactical/strategic targets (BYOL 2006.07733,
    #: I-JEPA 2301.08243). SPEC E-ARCH-TSC-1 §3 / RESULT R3: `frozen` pins the
    #: OPERATIVE target only — `tq = _tac_field(tgt)` and `st =
    #: strategic.subspace(tgt)` are derived from the TRAINED adapter in both
    #: target spaces, and `detach_aux_targets` stops the gradient but not the
    #: shrinkage: a shrinking student still shrinks them. When on, those two
    #: targets (and the str-extension target) come from a slow EMA copy of the
    #: whole target path (`_EmaTargetPath`), so the target cannot move with
    #: the student inside a step. ⛔ DEFAULT OFF: it is a next-run arm, and the
    #: live run must stay reproducible from the repo (state_dict keys and
    #: forward numerics are byte-identical when off — pinned by
    #: `tests/test_refa_v1_ema_targets.py`).
    ema_targets: bool = False
    #: Decay schedule: linear `ema_decay` -> `ema_decay_end` over the run,
    #: I-JEPA's arrangement (0.996 -> 1.0 linear; BYOL 0.996 -> 1.0 cosine).
    #: End pinned at 0.999 rather than 1.0 so the teacher never fully freezes.
    ema_decay: float = 0.996
    ema_decay_end: float = 0.999
    #: SigReg (our v6/v7 line, absent from refav1 until now): 0 = off.
    #: MEASURED discriminating: 0.477 on random vs 2.61 on collapsed.
    w_sigreg: float = 0.0
    sigreg_slices: int = 512
    #: VICReg (2105.04906) variance hinge on the adapter output, in units of
    #: the target's own scale. 0 = off.
    var_floor: float = 0.0
    #: RankMe (2210.02885) / our G-RANK gate. 0 = MONITOR ONLY (always logged);
    #: > 0 refuses when participation falls below it. Reference floor 8.56.
    min_participation: float = 0.0

    #: ⛔ TRUNCATED BPTT depth for the operative rollout. 0 = full chain (the
    #: deliberate-regression control). DEFAULT 15 = DreamerV3's imagination
    #: horizon (2301.04104) AND Looped-WM's ceil(mu_rec/2) for our 30-step
    #: rollout (2606.18208) — the two independent recipes agree on this value
    #: at our K. MEASURED without it: gnorm 3.6e3 / 5.7e7 / inf within 300 steps.
    bptt_truncate: int = 15

    def sanity(self) -> None:
        if self.d_state < self.d_enc:
            raise ValueError(
                f"d_state ({self.d_state}) < d_enc ({self.d_enc}) — change #3 "
                "forbids a bottleneck below the encoder width; FROST-Drive "
                "measured 8.17 -> 7.68 RFS on exactly this axis")
        if abs(self.op_dt * self.op_steps - 6.0) > 1e-6:
            raise ValueError("operative horizon must reach exactly 6.0 s")
        if abs(self.tac_dt * self.tac_steps - 6.0) > 1e-6:
            raise ValueError("tactical horizon must reach exactly 6.0 s")
        if abs(self.str_dt * self.str_steps - 6.0) > 1e-6:
            raise ValueError("strategic horizon must reach exactly 6.0 s")
        # ⛔⛔ THE CHECK THAT WAS MISSING, AND IT COST TWO LEVELS OF THE LADDER.
        # The abstracted levels do NOT index frames — they SUBSAMPLE the
        # operative target grid (`tgt[:, s-1::s]`). So a rate that is not an
        # integer multiple of ``op_dt`` is INEXPRESSIBLE: the code silently
        # rounds it to the nearest whole stride and trains a ladder nobody
        # chose. MEASURED on the shipped default: str_dt 1.5 / op_dt 0.2 = 7.5,
        # rounded to 8 -> the strategic rung ran at 1.6 s, and only 3 of its 4
        # steps existed inside a 30-step rollout.
        # ⚠️ The three-rates-reach-6.0-s test passed throughout, because it
        # asserted ``dt * steps == 6.0`` on the CONFIG and never once looked at
        # which future frame a prediction was regressed onto.
        for name, dt, steps in (("tac_dt", self.tac_dt, self.tac_steps),
                                ("str_dt", self.str_dt, self.str_steps)):
            ratio = dt / self.op_dt
            stride = int(round(ratio))
            if abs(ratio - stride) > 1e-6:
                raise ValueError(
                    f"{name} ({dt}) is not an integer multiple of op_dt "
                    f"({self.op_dt}): ratio {ratio}. The abstracted levels "
                    "subsample the operative grid, so this rate cannot be "
                    "represented and would be silently rounded to "
                    f"{stride * self.op_dt:.4g} s")
            if stride * steps > self.op_steps:
                raise ValueError(
                    f"{name} ladder needs operative index {stride * steps - 1} "
                    f"but the rollout has only {self.op_steps} steps: "
                    f"{steps - self.op_steps // stride} of its {steps} targets "
                    "would be silently dropped")
        # --- the long-horizon strategic extension (PI 2026-08-31) ---------- #
        if self.str_horizon_s < 6.0 - 1e-9:
            raise ValueError(
                f"str_horizon_s ({self.str_horizon_s}) below the 6.0 s "
                "in-window horizon — the extension extends, it cannot shrink")
        ext = (self.str_horizon_s - 6.0) / self.str_dt
        if abs(ext - round(ext)) > 1e-6:
            raise ValueError(
                f"str_horizon_s {self.str_horizon_s}: the extension beyond "
                f"6.0 s ({self.str_horizon_s - 6.0:.4g} s) is not an integer "
                f"number of str_dt ({self.str_dt}) ticks — the same "
                "inexpressibility class as the in-window grid rule above")
        if self.w_feat_str_ext and self.str_ext_steps == 0:
            raise ValueError(
                f"w_feat_str_ext {self.w_feat_str_ext} with str_horizon_s "
                f"{self.str_horizon_s} (zero extension ticks): the term would "
                "be advertised in the config and inert in the loss")
        if int(round(self.plan_horizon_s / self.op_dt)) > self.op_steps:
            raise ValueError("plan horizon exceeds the operative rollout")
        if self.w_cf < 0.0:
            raise ValueError(f"w_cf must be >= 0, got {self.w_cf}")
        if self.w_cf and self.cf_negs < 1:
            raise ValueError(
                f"w_cf {self.w_cf} with cf_negs {self.cf_negs}: zero negatives "
                "makes the InfoNCE a constant, so the term would be advertised "
                "in the launch line and inert in the loss")
        if self.w_cf and not (1 <= self.cf_at_step <= self.op_steps):
            raise ValueError(
                f"cf_at_step {self.cf_at_step} outside the operative rollout "
                f"[1, {self.op_steps}]")
        if self.motion_inject and self.op_window < 2:
            raise ValueError(
                f"motion_inject needs op_window >= 2 (got {self.op_window}) — "
                "a one-frame window has no z_{t-1} to difference against")
        if self.w_tac_label < 0.0 or self.w_str_label < 0.0:
            raise ValueError("label weights must be >= 0")
        if self.proposal_k < 1:
            raise ValueError(f"proposal_k must be >= 1, got {self.proposal_k}")
        if self.w_aux_head < 0.0:
            raise ValueError("w_aux_head must be >= 0")
        if self.target_space not in ("adapter", "frozen"):
            raise ValueError(f"target_space must be 'adapter' or 'frozen', "
                             f"got {self.target_space!r}")
        if not (0.0 <= self.ema_decay <= self.ema_decay_end <= 1.0):
            raise ValueError(
                f"ema_decay {self.ema_decay} / ema_decay_end "
                f"{self.ema_decay_end}: need 0 <= decay <= decay_end <= 1 — "
                "the teacher slows down over the run, it never speeds up")
        if self.tmix_groups is not None and (
                self.tmix_groups < 1 or self.d_state % self.tmix_groups):
            raise ValueError(
                f"tmix_groups {self.tmix_groups} must be >= 1 and divide "
                f"d_state {self.d_state} (None = d_state = depthwise)")

    @property
    def a_in_dim(self) -> int:
        """Width of the action tensor the PREDICTORS consume: the `a_dim`
        controls plus the derived speed channel when `speed_channel` is on.
        `a_dim` itself stays the CONTROL width — the proposal head, the
        planner's candidates and the loader's actions are all `a_dim`-wide."""
        return self.a_dim + (1 if self.speed_channel else 0)

    @property
    def plan_steps(self) -> int:
        return int(round(self.plan_horizon_s / self.op_dt))

    @property
    def str_ext_steps(self) -> int:
        """Strategic ticks BEYOND the 6.0 s in-window grid (PI 2026-08-31)."""
        return int(round((self.str_horizon_s - 6.0) / self.str_dt))


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    """Pre-norm transformer block. LayerNorm only — no BatchNorm, no dropout
    (REF-A stability item 5: I2 batch-consistency, preserved in v1)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class FeatureStandardizerV1(nn.Module):
    """Per-channel standardisation with FROZEN buffers, fit once over the train
    corpus (REF-A stability item 1, carried over unchanged). Refitting a loaded
    checkpoint raises — the stats are part of the parity contract."""

    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, feats: Tensor) -> None:
        if bool(self.fitted):
            raise RuntimeError("standardizer already fitted — refusing to refit "
                               "(the stats are part of the parity contract)")
        flat = feats.reshape(-1, feats.shape[-1]).float()
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp_min(1e-3))
        self.fitted.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class WideAdapter(nn.Module):
    """Token-preserving adapter: [B,T,N,d_enc] -> [B,T,N,d_state].

    ⛔ Deliberately NOT a readout/pooling head. REF-A's adapter mapped the token
    grid into a compact state, which is the bottleneck change #3 forbids and the
    surface DINO-WM's ablation says must stay spatial. Per-token MLP (shared
    across tokens, so parameter cost is independent of ``n_tokens``) + a learned
    spatial embedding + a depthwise temporal mix."""

    def __init__(self, cfg: RefAV1Config):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 1, cfg.n_tokens, cfg.d_state))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.d_enc),
            nn.Linear(cfg.d_enc, cfg.d_state), nn.GELU(),
            nn.Linear(cfg.d_state, cfg.d_state))
        # groups = d_state is the shipped DEPTHWISE mix (H-REFAV1-MOTION names
        # it as the defect: no cross-channel temporal path); `tmix_groups=1`
        # is the full-mixing arm. None keeps the weight shape [d, 1, 3], so
        # existing checkpoints load unchanged. `getattr`: this block is SHARED
        # with `refd.py`, whose `RefDConfig` carries no `tmix_groups` and must
        # keep building the depthwise form (tests/test_refd.py caught this).
        groups = getattr(cfg, "tmix_groups", None)
        self.tmix = nn.Conv1d(cfg.d_state, cfg.d_state, kernel_size=3,
                              padding=1,
                              groups=cfg.d_state if groups is None else groups)
        self.out = nn.LayerNorm(cfg.d_state)

    def forward(self, feats: Tensor) -> Tensor:
        b, t, n, _ = feats.shape
        x = self.proj(feats) + self.pos
        y = x.permute(0, 2, 3, 1).reshape(b * n, -1, t)      # [B*N, d, T]
        x = x + self.tmix(y).reshape(b, n, -1, t).permute(0, 3, 1, 2)
        return self.out(x)


class TokenFieldPredictor(nn.Module):
    """⭐ THE ARCHITECTURAL HEART OF v1 — DINO-WM's predictor, on our field.

    Consumes a causal window of token fields plus per-step actions and predicts
    the **future patch-feature field**. Action embedding is broadcast over tokens
    and concatenated-then-projected, which is DINO-WM's exact conditioning
    scheme. ``intent`` (from the tactical brain) is ADDED to the action
    conditioning, which is how our hierarchy already closes onto the operative
    predictor (`fourbrain.run_hierarchy`) — so change #7 costs no new mechanism.
    """

    def __init__(self, cfg: RefAV1Config, d: int, layers: int, heads: int = 8,
                 intent_dim: int | None = None):
        super().__init__()
        self.d = d
        # `a_in_dim`, not `a_dim`: the predictor consumes the controls PLUS the
        # derived speed channel when `speed_channel` is on (3-wide); the
        # control space the planner searches stays `a_dim`-wide. `getattr`:
        # this class is SHARED with `refd.py` / `refa_v1p.py`, whose configs
        # may carry only `a_dim` — for them the input width IS `a_dim`.
        a_in = getattr(cfg, "a_in_dim", cfg.a_dim)
        self.act = nn.Sequential(nn.Linear(a_in, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.intent = nn.Linear(intent_dim, d) if intent_dim else None
        self.mix = nn.Linear(2 * d, d)
        self.blocks = nn.ModuleList([_Block(d, heads) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        # The residual delta head is DOWN-SCALED. `step` returns
        # `field + self.head(x)` and `self.head` ends in a Linear fed by a
        # LayerNorm, so a default init emits O(1) per dim regardless of the
        # field's scale -- and the docstring's "near-identity at init" claim was
        # simply FALSE. MEASURED 2026-08-22 against the REAL banked DINOv3
        # field (mean|x| 0.2060, per-frame movement 0.1021):
        #     zero-action |delta| at init = 0.4497
        #                                 = 2.2x the field's own magnitude
        #                                 = 4.4x the movement it must predict
        # The identical defect in v6's `OperativePredictor` measured 580x,
        # because v6's operative latent moves only 1.9% of its magnitude per
        # tick while this field moves 49.5% -- SEVERITY SCALES WITH HOW STATIC
        # THE BASE IS, which is plausibly why REF-A trained to something and v6
        # did not.
        # NOT zero-init: zeroing an OUTPUT head sets dL/dh = W^T . dL/dout = 0
        # and stalls gradient to the whole body (18 tests caught that on v6).
        # SCOPE: initialisation only; state_dict shapes unchanged.
        from tanitad.models.predictor import RESIDUAL_HEAD_INIT_SCALE
        self.head[-1].weight.data.mul_(RESIDUAL_HEAD_INIT_SCALE)
        self.head[-1].bias.data.mul_(RESIDUAL_HEAD_INIT_SCALE)

    def step(self, field: Tensor, action: Tensor,
             intent: Tensor | None = None) -> Tensor:
        """One latent step. ``field`` [B,N,d], ``action`` [B,a_in_dim] -> [B,N,d].

        Residual by construction (``z_hat = z + delta``): the predictor learns
        the CHANGE, so a zero-action step is near-identity at init and the
        6 s rollout does not drift on the first gradient.

        WARNING -- that identity claim was FALSE until 2026-08-22. With the
        head at DEFAULT init the zero-action delta MEASURED 0.4497 against a
        real DINOv3 field of mean|x| 0.2060, i.e. 2.2x the field and 4.4x the
        movement it predicts. It is true now only because the head is
        down-scaled by ``RESIDUAL_HEAD_INIT_SCALE`` in ``__init__``. A docstring
        asserting an initialisation property is not evidence for it; the
        assertion lives in ``tests/test_residual_init_scale.py``.
        """
        a = self.act(action)
        if intent is not None and self.intent is not None:
            a = a + self.intent(intent)
        a = a[:, None, :].expand(-1, field.shape[1], -1)
        x = self.mix(torch.cat([field, a], dim=-1))
        for blk in self.blocks:
            x = blk(x)
        return field + self.head(x)

    def rollout(self, field: Tensor, actions: Tensor,
                intent: Tensor | None = None,
                last_only: bool = False) -> Tensor:
        """``actions`` [B,K,a_in_dim] -> predicted fields [B,K,N,d].

        ``actions`` is the PREDICTOR input — `RefAV1.augment_actions` output
        (controls + derived speed when `speed_channel`), never raw controls
        when that flag is on; the Linear in ``act`` refuses the wrong width.

        ⛔ ``last_only`` IS A MEMORY REQUIREMENT, NOT AN OPTION, ON THE PLANNING
        PATH. MEASURED at the real geometry: one latent field is
        640 tokens x 1024 d = 1.31 MB in fp16, so a CEM population of 300 held
        for a 10-step rollout is **300 x 10 x 1.31 MB = 3.9 GB** of stored
        intermediates for a cost that only ever reads the FINAL field. Storing
        them would have made the arm un-runnable on anything but an 80 GB card
        and the failure would have surfaced only at planning time — the C111
        class (an analysis-time failure after the compute is paid).
        """
        z = field
        # ⛔⛔ TRUNCATED BPTT (PI 2026-09-02). Until now this loop applied the
        # predictor `K` times with NO detach — a 30-deep back-prop chain through
        # shared parameters at the live config. MEASURED on the first refav1
        # launch: gnorm 3.6e3 at step 50, 5.7e7 at 150, `inf` at 300.
        #
        # ⭐ THE LAB'S ASK-1 LITERATURE PASS FOUND NO PUBLISHED RECIPE IN OUR
        # REFERENCE CLASS BACK-PROPAGATES THAT FAR, and four independent
        # mechanisms all avoid it — none of them "lower the clip":
        #   TD-MPC2 (2310.16828)      H = 3   + learned terminal value
        #   DreamerV3 (2301.04104)    H = 15  + AGC(0.3) + lambda-returns
        #   Looped-WM (2606.18208)    truncate at mu_bwd = ceil(mu_rec / 2)
        #   InfinityDrive (2412.01522) curriculum 16 -> 32 -> 64 -> 128
        # Looped-WM states our failure verbatim: "Training directly with a large
        # K is unstable because gradients must back-propagate through K x T
        # shared-parameter applications."
        #
        # ⇒ detach the carried state every `bptt_truncate` steps. The FORWARD
        # rollout is unchanged — every step still sees the true previous state,
        # so the prediction task is identical; only the gradient path is bounded.
        # 0 disables truncation (the deliberate-regression control).
        trunc = int(getattr(self, "bptt_truncate", 0) or 0)

        n_steps = actions.shape[1]

        def _carry(z_, k_):
            # Detach AFTER step k so the chain is at most `trunc` deep.
            # ⛔ NEVER on the LAST step: in `last_only` mode that tensor IS the
            # return value, and detaching it hands the caller a gradient-free
            # result — the planning path would silently backprop nothing. Caught
            # by `test_last_only_path_truncates_too`, which is the whole reason
            # that test exists: the first implementation returned a fully
            # detached field whenever K was a multiple of `trunc`, and every
            # other test still passed.
            cut = trunc and (k_ + 1) % trunc == 0 and k_ < n_steps - 1
            return z_.detach() if cut else z_

        if last_only:
            for k in range(n_steps):
                z = _carry(self.step(z, actions[:, k], intent=intent), k)
            return z
        out = []
        for k in range(actions.shape[1]):
            z = self.step(z, actions[:, k], intent=intent)
            out.append(z)          # the OUTPUT keeps its gradient path
            z = _carry(z, k)       # only the CARRIED state is cut
        return torch.stack(out, dim=1)


class StrategicSubspacePredictor(nn.Module):
    """The strategic brain's OWN predictor, on a strategy-only latent subspace.

    Three-planner directive: *"strategic gets its OWN predictor on a
    strategy-only latent subspace"*. The subspace is a learned linear read of the
    pooled field — deliberately narrow (``str_dim``), because a route hypothesis
    at 1.5 s cadence should not carry lane-level texture, and because keeping it
    separate is what makes the strategic prediction falsifiable on its own.
    """

    def __init__(self, cfg: RefAV1Config):
        super().__init__()
        self.read = nn.Sequential(nn.LayerNorm(cfg.d_state),
                                  nn.Linear(cfg.d_state, cfg.str_dim))
        self.act = nn.Linear(cfg.a_dim, cfg.str_dim)
        self.blocks = nn.ModuleList(
            [_Block(cfg.str_dim, 4) for _ in range(cfg.str_layers)])
        # The residual delta head is DOWN-SCALED, not zeroed. `rollout` does
        # `s = s + self.head(x)`, and `self.head` ends in a Linear fed by a
        # LayerNorm, so a default init emits O(1) per dim regardless of the
        # state's scale. The identical defect in v6's `OperativePredictor` left
        # it 535x WORSE than predicting NO CHANGE at step 20,000 (MEASURED
        # 2026-08-22 at the true dt=0.1s tick), and rescaling the TRAINED heads
        # could not rescue it -- error fell monotonically to alpha=0.
        # NOT zero-init: zeroing an OUTPUT head sets dL/dh = W^T . dL/dout = 0
        # and stalls gradient to the whole body. See predictor.py.
        # SCOPE: initialisation only; state_dict shapes unchanged, so existing
        # REF-A v1 checkpoints load byte-identically.
        self.head = nn.Sequential(nn.LayerNorm(cfg.str_dim),
                                  nn.Linear(cfg.str_dim, cfg.str_dim))
        from tanitad.models.predictor import RESIDUAL_HEAD_INIT_SCALE
        self.head[-1].weight.data.mul_(RESIDUAL_HEAD_INIT_SCALE)
        self.head[-1].bias.data.mul_(RESIDUAL_HEAD_INIT_SCALE)

    def subspace(self, field: Tensor) -> Tensor:
        return _read_subspace(self.read, field)       # pool tokens -> [B, str]

    def rollout(self, s: Tensor, actions: Tensor) -> Tensor:
        out = []
        for k in range(actions.shape[1]):
            x = s + self.act(actions[:, k])
            for blk in self.blocks:
                x = blk(x[:, None, :]).squeeze(1)
            s = s + self.head(x)
            out.append(s)
        return torch.stack(out, dim=1)


# --------------------------------------------------------------------------- #
# The TARGET PATH, written once — used by the student and by its EMA teacher.
# Two copies of a formula are two conventions, and a teacher that pooled
# differently from the student would be a silently different target.
# --------------------------------------------------------------------------- #
def _pool_tac(queries: Tensor, pool: nn.MultiheadAttention,
              field: Tensor) -> Tensor:
    """Token field [B,N,d] -> tactical query field [B,Q,d]."""
    q = queries.expand(field.shape[0], -1, -1)
    return pool(q, field, field, need_weights=False)[0]


def _read_subspace(read: nn.Module, field: Tensor) -> Tensor:
    """Token field [B,N,d] -> strategic subspace [B, str_dim]."""
    return read(field.mean(dim=-2))


class _EmaTargetPath(nn.Module):
    """EMA teacher for the TACTICAL and STRATEGIC targets.

    BYOL (2006.07733) and I-JEPA (2301.08243) obtain a learned-but-
    uncollapsible target from an exponential-moving-average copy of the online
    network behind a stop-gradient; the predictor asymmetry (our multi-step
    rollout against a direct target) is the other half of that recipe and
    already exists. SPEC E-ARCH-TSC-1 §3 and RESULT R3 measured why refav1
    needs it: ``--target-space frozen`` pins the OPERATIVE target only, while
    ``tq`` and ``st`` are derived from the TRAINED adapter in both spaces —
    ``detach_aux_targets`` stops their gradient, not their shrinkage.

    ⛔ WHY THE ADAPTER ALONE IS NOT THE TARGET PATH. The tactical target is
    ``tac_pool(tac_queries, adapter(std(f)))`` and the strategic one is
    ``strategic.read(mean(adapter(std(f))))`` — three trained modules and one
    trained parameter sit on the target side. An EMA adapter under a LIVE
    ``tac_pool`` still lets the target follow the student through the pool's
    output projection. So the copy covers the WHOLE path: ``adapter``,
    ``tac_queries``, ``tac_pool``, ``strategic.read``. The predictors are NOT
    copied — the rollout-vs-target asymmetry is the point.

    Every parameter here is ``requires_grad=False``: never in the optimizer,
    moved only by :meth:`RefAV1.ema_update`, which the trainer calls after
    each optimizer step. The copy lives in the state_dict (``ema.*``) so a
    resumed run restores its teacher; when ``ema_targets`` is off this module
    is never built and the state_dict is byte-identical to before it existed.
    """

    def __init__(self, model: "RefAV1"):
        super().__init__()
        self.adapter = copy.deepcopy(model.adapter)
        self.tac_queries = nn.Parameter(model.tac_queries.detach().clone(),
                                        requires_grad=False)
        self.tac_pool = copy.deepcopy(model.tac_pool)
        self.str_read = copy.deepcopy(model.strategic.read)
        for p in self.parameters():
            p.requires_grad_(False)
        # No module on this path carries a buffer today. Refuse rather than
        # silently leave one un-synced if that ever changes (a BatchNorm-style
        # running stat that never follows the student is a divergent target).
        if list(self.buffers()):
            raise RuntimeError("EMA target path acquired buffers — "
                               "ema_update() syncs parameters only")

    def tac_field(self, field: Tensor) -> Tensor:
        return _pool_tac(self.tac_queries, self.tac_pool, field)

    def subspace(self, field: Tensor) -> Tensor:
        return _read_subspace(self.str_read, field)

    def pairs(self, model: "RefAV1") -> list[tuple[Tensor, Tensor]]:
        """``(teacher_param, student_param)`` for every parameter on the path,
        matched BY NAME — a positional zip would pair silently on a refactor."""
        out = [(self.tac_queries, model.tac_queries)]
        for nm, ema_m, stu_m in (("adapter", self.adapter, model.adapter),
                                 ("tac_pool", self.tac_pool, model.tac_pool),
                                 ("strategic.read", self.str_read,
                                  model.strategic.read)):
            e = dict(ema_m.named_parameters())
            s = dict(stu_m.named_parameters())
            if e.keys() != s.keys():
                raise RuntimeError(
                    f"EMA teacher and student disagree on {nm} parameters: "
                    f"{sorted(e.keys() ^ s.keys())}")
            out.extend((e[k], s[k]) for k in e)
        return out


# --------------------------------------------------------------------------- #
# The arm
# --------------------------------------------------------------------------- #
class RefAV1(nn.Module):
    """Frozen DINOv3 field -> wide adapter -> three-rate predictive hierarchy,
    with behaviour produced by planning rather than regression.

    ``forward`` is TRAINING (feature prediction + auxiliaries). ``plan`` is
    DEPLOYMENT (iCEM/MPC over the operative predictor, cost assembled from the
    tactical and strategic goals). They share every weight; nothing in the
    planning path is trained to imitate a trajectory.
    """

    def __init__(self, cfg: RefAV1Config | None = None):
        super().__init__()
        cfg = cfg or RefAV1Config()
        cfg.sanity()
        self.cfg = cfg

        self.std = FeatureStandardizerV1(cfg.d_enc)
        self.adapter = WideAdapter(cfg)

        # ⚠️ ``d_intent`` (not ``intent_dim``) — the field name on
        # TacticalPolicyConfig. Reading the wrong attribute would have silently
        # built the predictor WITHOUT intent conditioning and quietly deleted
        # change #7 (the hierarchy) while every shape still checked out.
        intent_dim = (cfg.tactical_cfg.d_intent
                      if cfg.tactical_cfg is not None else None)
        self.operative = TokenFieldPredictor(cfg, cfg.d_state, cfg.op_layers,
                                             cfg.op_heads, intent_dim)
        self.tac_queries = nn.Parameter(
            torch.zeros(1, cfg.tac_queries, cfg.d_state))
        nn.init.trunc_normal_(self.tac_queries, std=0.02)
        self.tac_pool = nn.MultiheadAttention(cfg.d_state, 8, batch_first=True)
        self.tactical = TokenFieldPredictor(cfg, cfg.d_state, cfg.tac_layers,
                                            cfg.op_heads, intent_dim)
        self.strategic = StrategicSubspacePredictor(cfg)
        # ⛔ TRUNCATED BPTT reaches BOTH token-field predictors. The operative
        # rollout is the 30-deep one that produced gnorm inf, but `tactical`
        # is the SAME class rolling its own multi-step chain — fixing only the
        # one that happened to blow up would leave the identical defect next
        # door, which is the shape of half the bugs in this file's history.
        # The strategic predictor rolls 2 (+2 ext) steps and needs no cut.
        self.operative.bptt_truncate = int(cfg.bptt_truncate)
        self.tactical.bptt_truncate = int(cfg.bptt_truncate)

        # Hierarchy brains — the SAME classes the flagship holds, so the
        # conditioning chain is identical and comparisons stay on one axis.
        # Signature is (cfg, state_dim, window) — the brains compose on ANY
        # compact state, which is why the flagship and REF-A can share them.
        self.strategic_policy = (
            StrategicPolicy(cfg.strategic_cfg, cfg.d_state, cfg.op_window)
            if cfg.strategic_cfg is not None else None)
        self.tactical_policy = (
            TacticalPolicy(cfg.tactical_cfg, cfg.d_state, cfg.op_window,
                           d_cond=cfg.strategic_cfg.d_ctx)
            if cfg.tactical_cfg is not None else None)
        if (self.tactical_policy is not None) != (self.strategic_policy is not None):
            raise ValueError(
                "the brains are a MATCHED SET — the tactical policy is "
                "FiLM-conditioned on the strategic ctx (d_cond=d_ctx), so one "
                "without the other is a broken conditioning chain, not a "
                "smaller model")

        # ⛔ FACTORED LAT × LON TACTICAL HEADS — AND WHY THEY EXIST AT ALL.
        #
        # DEFECT FOUND 2026-08-18, after v1 was first committed: the shared
        # `TacticalPolicy` emits ONE `maneuver_logits [B, 5]` over
        # `refb.MANEUVER_CLASSES = (lane_keep, turn_left, turn_right,
        # accelerate, brake_stop)` — a softmax that MIXES the lateral and
        # longitudinal axes. `v6.py` names that mixing "the programme's single
        # largest known defect", retired BY DESIGN, and REF-C v3 already reads
        # `tac.N_LAT` / `tac.N_LON`. v1 silently inherited the retired form
        # because it reused the legacy brain with its DEFAULT config — every
        # shape checked out and nothing failed.
        #
        # MEASURED consequences of the mixed head (D-TAC1, 2026-08-03): shipped
        # 5-way decode accuracy 0.7581 / macro-recall 0.5313 with `accelerate`
        # NEVER PREDICTED, against 0.9348 / 0.8290 factored; and the 5-way label
        # destroys 9.68 % (132/1364) of the longitudinal decisions outright.
        #
        # ⇒ v1 decodes the tactical action on TWO independent heads over the
        # v6 vocabulary, imported from `v6.py` so there is exactly one source.
        # The legacy `maneuver_logits` is NOT consumed anywhere in v1.
        # ⭐ v7 mandate (PI 2026-08-27): the head vocabulary resolves through
        # the version registry. getattr fallback v6.0 = pre-field configs
        # (old checkpoints) keep their shapes.
        _vv = getattr(cfg, "tac_vocab_version", "v6.0")
        self.n_lat = len(tactical_lat_actions(_vv))
        self.n_lon = len(tactical_lon_actions_v(_vv))
        d_int = intent_dim or cfg.d_state
        self.lat_head = nn.Sequential(nn.LayerNorm(d_int),
                                      nn.Linear(d_int, self.n_lat))
        self.lon_head = nn.Sequential(nn.LayerNorm(d_int),
                                      nn.Linear(d_int, self.n_lon))

        # ⚠️ AUXILIARY imitation proposal — NOT the behaviour source. It exists
        # only to seed the planner (GPC: "generative control proposes, MPC
        # disposes"). w_aux_head is 0.1 and it never gates a metric.
        self.proposal = nn.Sequential(
            nn.LayerNorm(cfg.d_state), nn.Linear(cfg.d_state, 512), nn.GELU(),
            nn.Linear(512, cfg.proposal_k * cfg.plan_steps * cfg.a_dim))
        self.proposal_score = (nn.Sequential(nn.LayerNorm(cfg.d_state),
                                             nn.Linear(cfg.d_state,
                                                       cfg.proposal_k))
                               if cfg.proposal_k > 1 else None)

        # Nav injection (PI 2026-09-01): composed HERE, never inside the shared
        # fourbrain classes — the flagship holds the same brains, and widening
        # their signatures would change every consumer at once. Down-scaled
        # init so the decision arms start ≈ the pre-decision behaviour.
        if cfg.nav_inject and cfg.tactical_cfg is not None:
            from tanitad.models.predictor import RESIDUAL_HEAD_INIT_SCALE as _S
            n_cmd = cfg.strategic_cfg.n_commands
            d_cmd = cfg.strategic_cfg.d_cmd
            self.nav_inj_emb = nn.Embedding(n_cmd, d_cmd)
            self.nav_to_ctx = nn.Linear(d_cmd, cfg.strategic_cfg.d_ctx)
            self.nav_to_intent = nn.Linear(d_cmd, intent_dim)
            for lin in (self.nav_to_ctx, self.nav_to_intent):
                lin.weight.data.mul_(_S)
                lin.bias.data.mul_(_S)
        else:
            self.nav_inj_emb = None

        # Observed-motion injection (config-gated; params exist only when on,
        # so default state_dicts are byte-identical). Down-scaled init: the
        # injection starts near zero and the arm starts indistinguishable from
        # baseline — the experiment measures what training makes of it.
        if cfg.motion_inject:
            from tanitad.models.predictor import RESIDUAL_HEAD_INIT_SCALE
            self.motion_in = nn.Sequential(
                nn.LayerNorm(cfg.d_state), nn.Linear(cfg.d_state, cfg.d_state))
            self.motion_in[-1].weight.data.mul_(RESIDUAL_HEAD_INIT_SCALE)
            self.motion_in[-1].bias.data.mul_(RESIDUAL_HEAD_INIT_SCALE)
        else:
            self.motion_in = None

        # Frozen-target readout (config-gated, see target_space in the config).
        self.to_enc = (nn.Linear(cfg.d_state, cfg.d_enc)
                       if cfg.target_space == "frozen" else None)

        # EMA teacher for the tactical/strategic targets (config-gated; built
        # LAST so it copies the fully-initialised target path). Off ⇒ absent,
        # so the live run's checkpoint keys are untouched.
        self.ema = _EmaTargetPath(self) if cfg.ema_targets else None

    # -- EMA teacher ------------------------------------------------------- #
    def ema_decay_at(self, step: int, total_steps: int) -> float:
        """Linear ``ema_decay`` -> ``ema_decay_end`` over ``total_steps``
        (I-JEPA 2301.08243: 0.996 -> 1.0 linear over training). ``step`` is
        the trainer's 1-based step counter, so the last step reads exactly
        ``ema_decay_end``."""
        c = self.cfg
        if total_steps <= 0:
            return float(c.ema_decay)
        frac = min(max(int(step), 0), int(total_steps)) / float(total_steps)
        return float(c.ema_decay + (c.ema_decay_end - c.ema_decay) * frac)

    @torch.no_grad()
    def ema_update(self, step: int, total_steps: int) -> float:
        """``teacher <- decay * teacher + (1 - decay) * student`` on every
        target-path parameter. Called by the trainer AFTER ``opt.step()``;
        returns the decay used so the log can carry it."""
        if self.ema is None:
            raise RuntimeError(
                "ema_update() on a model built with ema_targets=False — the "
                "trainer wiring and the config disagree; refusing to no-op")
        decay = self.ema_decay_at(step, total_steps)
        for p_ema, p_stu in self.ema.pairs(self):
            p_ema.lerp_(p_stu.detach(), 1.0 - decay)
        return decay

    # -- encoding ---------------------------------------------------------- #
    def encode(self, feats: Tensor) -> Tensor:
        """Cached DINOv3 patch features [B,T,N,d_enc] -> state field."""
        if feats.shape[-1] != self.cfg.d_enc:
            raise ValueError(f"expected d_enc={self.cfg.d_enc}, "
                             f"got {feats.shape[-1]}")
        if feats.shape[-2] != self.cfg.n_tokens:
            raise ValueError(f"expected {self.cfg.n_tokens} patch tokens, got "
                             f"{feats.shape[-2]} — v1 forbids a narrowed visual "
                             "interface (change #2/#3)")
        return self.adapter(self.std(feats))

    def _tac_field(self, field: Tensor) -> Tensor:
        return _pool_tac(self.tac_queries, self.tac_pool, field)

    def _last_state(self, field: Tensor) -> Tensor:
        """The state every rollout starts from — ONE place, so training and
        planning cannot drift apart on whether motion was injected."""
        last = field[:, -1]
        if self.motion_in is not None:
            last = last + self.motion_in(field[:, -1] - field[:, -2])
        return last

    def _run_brains(self, pooled_win: Tensor, nav_cmd: Tensor | None,
                    ego: Tensor | None = None) -> dict | None:
        """The strategic→tactical chain, in ONE place for forward AND plan.

        ⭐ NAV INJECTION (PI 2026-09-01) lives here and only here: nav is added
        to the tactical policy's FiLM cond (`ctx`) and to the `intent` that
        conditions the operative and tactical FIELD predictors. Duplicating
        this block in plan() was how a conditioning change could silently
        apply at training and not at deployment — factored away.
        """
        if self.strategic_policy is None or self.tactical_policy is None:
            return None
        b = pooled_win.shape[0]
        nav = (torch.zeros(b, dtype=torch.long, device=pooled_win.device)
               if nav_cmd is None else nav_cmd)
        strat = self.strategic_policy(pooled_win, nav, ego=ego)
        ctx = strat["ctx"]
        nemb = self.nav_inj_emb(nav) if self.nav_inj_emb is not None else None
        if nemb is not None:
            ctx = ctx + self.nav_to_ctx(nemb)         # nav → tactical policy
        tac = self.tactical_policy(pooled_win, ctx, ego=ego)
        intent = tac["intent"]
        if nemb is not None:
            intent = intent + self.nav_to_intent(nemb)  # nav → field predictors
        return {"intent": intent, "ctx": ctx, "tac": tac,
                "route_logits": strat.get("route_logits")}

    # -- training ---------------------------------------------------------- #
    def _stride(self, dt: float) -> int:
        """How many operative steps one ``dt``-rate step spans.

        ⭐ THE SINGLE PLACE a rate becomes an index step, so the ladder cannot
        drift between the loss and the targets again. ``sanity()`` has already
        refused any ``dt`` that is not an integer multiple of ``op_dt``, so the
        rounding here is exact by construction rather than by luck.

        ⛔ AND THE PHASE IS THE WHOLE POINT. Targets are sliced
        ``[stride-1::stride]``, not ``[::stride]``: ``rollout()[:, 0]`` is the
        state after ONE step, so a level whose step spans ``stride`` operative
        steps must be regressed onto the observation ``stride`` steps ahead —
        not onto the one 1 step ahead. MEASURED before the fix: **10 of 10
        tactical and 4 of 4 strategic targets were wrong**, every one shifted
        early by a full stride, which trained both abstracted levels to be
        0.2 s predictors and silently shortened the ladder to 5.6 s / 5.0 s.
        """
        return max(1, int(round(dt / self.cfg.op_dt)))

    def augment_actions(self, controls: Tensor, v0: Tensor | None) -> Tensor:
        """Controls [B,K,a_dim] (a, kappa) -> predictor input [B,K,a_in_dim].

        ⭐ THE SPEED CHANNEL, under the PI ruling of 2026-09-02: *"It is
        allowed to use the velocity as initial measured state at its cycle
        time. It is not allowed to use the future dynamic information from
        the ground truth."* The third channel is therefore INTEGRATED:

            v_0 = v0                          (measured at the window anchor)
            v_k = v0 + Σ_{j<k} a_j · op_dt    (the speed OPENING step k)

        and never indexed from a future speed. Under teacher forcing the
        loader's ``a = Δv/dt`` makes this telescope to the true grid speeds —
        that is the T0 definition and is fine — but the CODE integrates, so
        the identical path is correct at T1 on the model's own actions, where
        no future speed exists to read. Normalised by ``SPEED_SCALE_MPS`` so
        the channel is O(1) beside (a, kappa). Not clamped: a candidate that
        integrates through zero is a planner-cost matter, not an input one.

        ⛔ Refuses a pre-widened tensor: the channel is DERIVED here or it
        does not exist — a caller handing over a 3-wide action has, by
        construction, read a speed from somewhere this rule cannot audit.
        """
        cfg = self.cfg
        if controls.shape[-1] != cfg.a_dim:
            raise ValueError(
                f"actions must be [.., {cfg.a_dim}] = (a, kappa) controls, got "
                f"trailing dim {controls.shape[-1]} — the speed channel is "
                "derived by augment_actions(), never supplied")
        if not cfg.speed_channel:
            return controls
        if v0 is None:
            raise ValueError(
                "speed_channel=True needs v0 [B]: the speed MEASURED at the "
                "window anchor (loader key 'v0'; plan()'s v0 argument)")
        v0 = torch.as_tensor(v0, dtype=controls.dtype,
                             device=controls.device).reshape(-1)
        if v0.shape[0] != controls.shape[0]:
            raise ValueError(f"v0 carries {v0.shape[0]} rows for a batch of "
                             f"{controls.shape[0]}")
        dv = torch.cumsum(controls[..., 0], dim=1) * cfg.op_dt    # after step k
        v = v0[:, None] + torch.cat(
            [torch.zeros_like(dv[:, :1]), dv[:, :-1]], dim=1)      # opening k
        return torch.cat([controls, (v / SPEED_SCALE_MPS)[..., None]], dim=-1)

    # -- ⭐ THE PLANNER->MODEL BOUNDARY, IN ONE PLACE ----------------------- #
    def _model_actions(self, controls: Tensor, v0: Tensor | None,
                       units: str) -> Tensor:
        """GEOMETRY controls ``[B,K,a_dim]`` -> predictor input. **The only
        planner-side crossing into the model**, so a new crossing cannot open a
        second action convention without being seen.

        ⛔ WHY THIS IS A NAMED FUNCTION AND NOT TWO ``as_command`` CALLS.
        MEASURED 2026-09-03 (`D-REFAV1-BOUNDARY-NULL`): ``as_command`` was
        called at exactly ONE site — inside `plan`'s ``_cost_chunk`` — so
        ``model_action_units="steer"`` converted the CANDIDATE while
        `_imagine_tactical_goal` still rolled the GOAL in raw curvature, and the
        two were compared across two different conventions. The converted
        canonical turn then landed one float32 ULP FURTHER from its own goal
        (goal term 5.96e-08 -> 1.19e-07; advantage over ``cv`` 0 -> -5.96e-08).
        A repair that merely adds a second ``as_command`` call leaves the same
        shape of bug available to the third crossing somebody writes next; a
        named boundary makes "did this crossing convert?" one grep.

        ⛔ PLANNER-SIDE ONLY. The TRAINING path's ``actions`` are the v2ep
        COMMAND channel ALREADY, so `forward` and `_cf_term` call
        `augment_actions` directly — routing training through here would
        double-convert every live run.

        Order is safe: ``as_command`` touches channel 1 only and
        `augment_actions` derives channel 2 from channel 0, so the two commute
        on every channel either one reads, and both commute with a slice on the
        time axis. ``units="kappa"`` returns the input OBJECT unchanged
        (`kinematic.as_command`), so the default path is byte-identical to a
        bare `augment_actions` call.
        """
        return self.augment_actions(as_command(controls, units), v0)

    def forward(self, feats: Tensor, actions: Tensor, *,
                future_feats: Tensor | None = None,
                str_ext_targets: Tensor | None = None,
                str_ext_actions: Tensor | None = None,
                lat_label: Tensor | None = None,
                lon_label: Tensor | None = None,
                route_label: Tensor | None = None,
                nav_cmd: Tensor | None = None,
                ego: Tensor | None = None,
                v0: Tensor | None = None) -> dict:
        """``feats`` [B,W,N,d_enc] observed window, ``actions`` [B,K,a_dim].

        ``v0`` [B] is the ego speed MEASURED at the window anchor (m/s, the
        loader's ``v0``). Consumed only when ``speed_channel`` is on, where
        ``augment_actions`` integrates it with ``actions`` into the
        predictors' third input channel; ``actions`` itself stays the 2-wide
        (a, kappa) control sequence everywhere.

        ``future_feats`` [B,K,N,d_enc] are the **targets** — future patch
        features in the SAME standardised space. That is the primary loss
        (change #4): the model is asked to carry the world forward, not to hit a
        trajectory label.

        ⭐ THE LONG-HORIZON STRATEGIC PAIR (PI 2026-08-31, *"not only 6
        seconds"*): ``str_ext_targets`` [B, K_ext, N, d_enc] are cached DINOv3
        features at t = 6.0 + k·str_dt (k = 1..K_ext — 9.0 s and 12.0 s at the
        default), and ``str_ext_actions`` [B, K_ext, a_dim] the action opening
        each of those windows. They CANNOT come from the 6 s operative grid —
        the loader reads them straight off the episode cache. Supervision is
        opportunistic: absent inputs skip the term (the smoke path), but one
        without the other is a contract error, refused loudly.
        """
        field = self.encode(feats)                       # [B,W,N,d]
        last = self._last_state(field)
        # ⚠️ The brains take a STATE WINDOW [B, W, D], not a single state — the
        # window length is baked into their positional embeddings, so passing
        # [B,1,D] would be a silent shape-compatible wrong input.
        pooled_win = field.mean(dim=-2)                  # [B,W,D]
        pooled = pooled_win[:, -1]

        intent = None
        out: dict = {}
        brains = self._run_brains(pooled_win, nav_cmd, ego=ego)
        if brains is not None:
            intent = brains["intent"]
            tac = brains["tac"]
            # ⚠️ `ctx` here is the AUGMENTED cond the tactical policy actually
            # consumed (nav added when nav_inject) — emitting the pre-injection
            # value would misdescribe the conditioning that happened.
            out.update({"ctx": brains["ctx"], "intent": intent,
                        "route_logits": brains["route_logits"]})
            # ⭐ THE FACTORED DECODE — v1's tactical action. Two independent
            # softmaxes, so a longitudinal decision can never be outvoted by a
            # lateral one sharing its logit space.
            out["lat_logits"] = self.lat_head(intent)
            out["lon_logits"] = self.lon_head(intent)
            # The legacy mixed head is passed through under a name that says
            # what it is, so nothing downstream can consume it by accident
            # while looking like it read a tactical action.
            out["legacy_mixed_maneuver_logits_DO_NOT_USE"] = \
                tac.get("maneuver_logits")

        # ⭐ The token-field predictors consume `acts_in` — the controls plus
        # the derived speed channel when `speed_channel` (`augment_actions`).
        # `actions` stays the raw (a, kappa) sequence for the strategic level,
        # the cf negatives and the proposal demo: those live in CONTROL space.
        acts_in = self.augment_actions(actions, v0)
        out["op_pred"] = self.operative.rollout(last, acts_in, intent=intent)
        # ⚠️ ACTIONS keep phase ``[::stride]`` while TARGETS take
        # ``[stride-1::stride]``, and the asymmetry is deliberate: a level's
        # step j spans operative steps [j*s, (j+1)*s), so it CONSUMES the
        # action that opens that window and PREDICTS the state that closes it.
        tac_a = acts_in[:, ::self._stride(self.cfg.tac_dt)]
        out["tac_pred"] = self.tactical.rollout(
            self._tac_field(last), tac_a[:, :self.cfg.tac_steps], intent=intent)
        # DECISION (2026-09-02): the strategic subspace predictor stays on the
        # 2-wide controls. Its extension ticks at 9 / 12 s (`str_ext_actions`)
        # carry ONE opening action each and no sequence to integrate over, so
        # an in-window-only speed would be a second convention on one level,
        # and a GT speed there is exactly what the ruling forbids. It also
        # keeps the strategic prediction independently falsifiable (:255).
        str_a = actions[:, ::self._stride(self.cfg.str_dt)][:, :self.cfg.str_steps]
        # ⭐ ONE continued rollout, not two: the extension ticks roll on
        # autoregressively from the in-window strategic state, which is what
        # makes this a LONG-HORIZON PREDICTOR rather than a second head.
        ext_k = self.cfg.str_ext_steps
        has_ext = str_ext_targets is not None or str_ext_actions is not None
        if has_ext:
            if str_ext_targets is None or str_ext_actions is None:
                raise ValueError(
                    "str_ext_targets and str_ext_actions are a PAIR — one "
                    "without the other would roll unsupervised ticks or "
                    "supervise ticks that were never rolled")
            if future_feats is None:
                raise ValueError(
                    "str_ext_targets without future_feats: the extension term "
                    "attaches to the training loss, which does not exist here "
                    "— the targets would be accepted and silently unused")
            if ext_k == 0:
                raise ValueError(
                    f"extension inputs supplied but str_horizon_s "
                    f"{self.cfg.str_horizon_s} yields zero extension ticks")
            for nm, t_, want in (("str_ext_targets", str_ext_targets, ext_k),
                                 ("str_ext_actions", str_ext_actions, ext_k)):
                if t_.shape[1] != want:
                    raise ValueError(f"{nm} carries {t_.shape[1]} ticks, "
                                     f"config requires {want}")
            full_a = torch.cat([str_a, str_ext_actions], dim=1)
        else:
            full_a = str_a
        str_all = self.strategic.rollout(self.strategic.subspace(last), full_a)
        out["str_pred"] = str_all[:, :self.cfg.str_steps]
        if has_ext:
            out["str_pred_ext"] = str_all[:, self.cfg.str_steps:]
            out["str_ext_target_s"] = [
                round(6.0 + (k + 1) * self.cfg.str_dt, 3) for k in range(ext_k)]
        out["proposal"] = self.proposal(pooled).reshape(
            -1, self.cfg.proposal_k, self.cfg.plan_steps, self.cfg.a_dim)
        if self.proposal_score is not None:
            out["proposal_logits"] = self.proposal_score(pooled)

        if future_feats is not None:
            tgt = self.adapter(self.std(future_feats))
            # ⛔ FOUND BY A TEST GOING NaN, 2026-08-31: a future shorter than
            # one abstracted stride slices to an EMPTY target tensor, and
            # `mse_loss` over zero elements is NaN — a poisoned total loss that
            # backpropagates NaN into every weight while looking like a batch
            # hiccup. A future too short to supervise a level is a contract
            # error, and it is refused by name, not averaged into NaN.
            for _nm, _dt in (("tactical", self.cfg.tac_dt),
                             ("strategic", self.cfg.str_dt)):
                _s = self._stride(_dt)
                if tgt.shape[1] < _s:
                    raise ValueError(
                        f"future_feats carries {tgt.shape[1]} operative steps "
                        f"but the {_nm} level's FIRST target sits at step "
                        f"{_s} — the level would train on an empty tensor "
                        "(NaN loss), not on a shorter ladder")
            k = min(tgt.shape[1], out["op_pred"].shape[1])
            if self.cfg.target_space == "frozen":
                # ⭐ Collapse-proof primary: the target is std(future) — frozen
                # buffers, no trained parameter on the target side — so its
                # per-channel variance is pinned at ~1 and a collapsed adapter
                # scores the target variance instead of zero. The prediction is
                # read back to encoder width through `to_enc`.
                tgt_op = self.std(future_feats)
                out["loss_feat_op"] = F.mse_loss(
                    self.to_enc(out["op_pred"][:, :k]), tgt_op[:, :k])
            else:
                out["loss_feat_op"] = F.mse_loss(out["op_pred"][:, :k],
                                                 tgt[:, :k])
            # ⭐ EMA TEACHER (config-gated, `ema_targets`). The tactical and
            # strategic targets come from the slow copy of the target path,
            # built under no_grad so they carry no graph. The student `tgt`
            # stays in scope because the anti-collapse terms, the cf term and
            # the participation monitor MEASURE the student, not the teacher.
            if self.ema is not None:
                with torch.no_grad():
                    tgt_aux = self.ema.adapter(self.std(future_feats))
                _tacf, _subs = self.ema.tac_field, self.ema.subspace
            else:
                tgt_aux = tgt
                _tacf, _subs = self._tac_field, self.strategic.subspace
            tq = torch.stack([_tacf(tgt_aux[:, i])
                              for i in range(tgt_aux.shape[1])], dim=1)
            step = self._stride(self.cfg.tac_dt)
            tq = tq[:, step - 1::step][:, :out["tac_pred"].shape[1]]
            # ⭐ STOP-GRADIENT ON THE TACTICAL TARGET (SimSiam 2011.10566).
            # `tq` is built from `tgt`, the TRAINED adapter's output, in BOTH
            # target spaces — so without this the target chases the prediction
            # and both can shrink to zero together. `--target-space frozen`
            # fixes only the OPERATIVE term (:262); this covers the other two.
            if self.cfg.detach_aux_targets:
                tq = tq.detach()
            kt = min(tq.shape[1], out["tac_pred"].shape[1])
            out["loss_feat_tac"] = F.mse_loss(out["tac_pred"][:, :kt], tq[:, :kt])
            sstep = self._stride(self.cfg.str_dt)
            st = _subs(tgt_aux.flatten(0, 1)).reshape(
                tgt_aux.shape[0], tgt_aux.shape[1], -1)[:, sstep - 1::sstep]
            # ⭐ THE MODEL REPORTS ITS OWN ALIGNMENT. Emitted so the realised
            # ladder is OBSERVABLE — by the test, and once per run in the log —
            # instead of being re-derived by whoever is checking. A test that
            # recomputes the slice it is auditing would pass against the very
            # bug it exists to catch; these are the indices actually used.
            out["tac_target_idx"] = list(range(step - 1, tgt.shape[1], step))[:kt]
            out["str_target_idx"] = list(
                range(sstep - 1, tgt.shape[1], sstep))[:st.shape[1]]
            if self.cfg.detach_aux_targets:
                st = st.detach()                      # same reason as `tq`
            ks = min(st.shape[1], out["str_pred"].shape[1])
            out["loss_feat_str"] = F.mse_loss(out["str_pred"][:, :ks],
                                              st[:, :ks])

            # ⭐⭐ TARGET-SCALE INSTRUMENT (2026-09-02). In `target_space =
            # "adapter"` BOTH sides of every feature loss come from the TRAINED
            # adapter, so shrinking the adapter shrinks the TARGET and the loss
            # falls toward zero with no prediction improving — the collapse
            # minimum this class documents at :263. A falling loss is then
            # indistinguishable from progress unless you can see the target.
            #
            # MEASURED 2026-09-02 on the first refav1 launch: `adapter_std`
            # 0.4763 -> 0.3385 monotonically over 450 steps while
            # `loss_feat_str` fell 0.0052 -> 0.0003 — two views of one event,
            # and the loss curve alone looked like the best run of the day.
            #
            # ⇒ emit the TARGET's own scale beside each loss. A loss is only
            # interpretable against the variance of the thing it predicts:
            # in "frozen" space that variance is pinned at ~1 (frozen std
            # buffers), so a collapsed model scores ~1.0 and CANNOT reach zero;
            # in "adapter" space it is free to fall, and this makes that
            # visible instead of inferable.
            # ⚠️ PER-CHANNEL std, NOT global — corrected 2026-09-02 after the
            # first run exposed the flaw. The global std sat pinned at 1.0000
            # while `adapter_std` fell 0.4763 -> 0.4593 on the SAME tensor:
            # under LayerNorm total variance is preserved by construction and
            # collapse shows up as variance CONCENTRATING into fewer
            # directions, not shrinking. A global std is blind to exactly the
            # failure this instrument exists to see. `adapter_std` (per-channel,
            # then averaged) is the sensitive statistic, so match it.
            def _chan_std(x):
                return float(x.detach().float().reshape(-1, x.shape[-1])
                             .std(dim=0).mean())
            out["tgt_std_op"] = _chan_std(
                tgt_op if self.cfg.target_space == "frozen" else tgt)
            out["tgt_std_tac"] = _chan_std(tq)
            out["tgt_std_str"] = _chan_std(st)
            out["loss"] = (self.cfg.w_feat_op * out["loss_feat_op"]
                           + self.cfg.w_feat_tac * out["loss_feat_tac"]
                           + self.cfg.w_feat_str * out["loss_feat_str"])

            # ---- ANTI-COLLAPSE TERMS on the adapter's own output ------------
            # ⭐ Applied to `tgt` (the adapter output) because that is the thing
            # measured collapsing: `adapter_std` 0.4763 -> 0.3385. Each is
            # separately switchable so its contribution stays attributable.
            z_flat = tgt.reshape(-1, tgt.shape[-1])
            if self.cfg.w_sigreg:
                # SigReg (v6/v7 line, absent from refav1 until 2026-09-02).
                # MEASURED discriminating: 0.477 random vs 2.61 collapsed.
                from tanitad.models.v6 import SigReg as _SigReg
                if not hasattr(self, "_sigreg"):
                    self._sigreg = _SigReg(n_slices=self.cfg.sigreg_slices)
                out["loss_sigreg"] = self._sigreg(z_flat)
                out["loss"] = out["loss"] + self.cfg.w_sigreg * out["loss_sigreg"]
            if self.cfg.var_floor:
                # VICReg's hinge (2105.04906 eq. 1): punish per-dim std BELOW
                # the floor, and only below — it must not push variance up
                # without bound, only refuse the collapse direction.
                sd = z_flat.float().std(dim=0)
                out["loss_varfloor"] = F.relu(self.cfg.var_floor - sd).mean()
                out["loss"] = (out["loss"]
                               + self.cfg.w_feat_op * out["loss_varfloor"])
            # ⭐ PARTICIPATION IS ALWAYS MONITORED, gated only when asked.
            # RankMe (2210.02885) / our G-RANK floor 8.56. A rank-1 collapse
            # reads 1.00; random 128-d reads ~13 (both MEASURED).
            with torch.no_grad():
                from tanitad.eval.spectral import (covariance_eigs,
                                                   participation_ratio)
                zc = z_flat.float()
                if zc.shape[0] > 1:
                    out["participation"] = participation_ratio(
                        covariance_eigs(zc[:4096]))
            if (self.cfg.min_participation
                    and out.get("participation", 1e9)
                    < self.cfg.min_participation):
                raise SystemExit(
                    f"⛔ participation {out['participation']:.2f} < floor "
                    f"{self.cfg.min_participation} — the adapter has collapsed. "
                    "Training on would produce a falling loss and no "
                    "representation (the 2026-09-02 refav1 failure).")


            # ---- the long-horizon strategic term (PI 2026-08-31) ----------- #
            # Same target pipeline as every other level — std -> adapter ->
            # subspace — so the extension is the SAME prediction task at a
            # longer reach, not a differently-normalised cousin.
            if has_ext:
                if self.ema is not None:
                    with torch.no_grad():
                        text = self.ema.adapter(self.std(str_ext_targets))
                else:
                    text = self.adapter(self.std(str_ext_targets))
                st_e = _subs(text.flatten(0, 1)).reshape(
                    text.shape[0], text.shape[1], -1)
                if self.cfg.detach_aux_targets:
                    st_e = st_e.detach()              # same reason as `tq`
                out["loss_feat_str_ext"] = F.mse_loss(out["str_pred_ext"], st_e)
                out["loss"] = (out["loss"] + self.cfg.w_feat_str_ext
                               * out["loss_feat_str_ext"])

            # ---- change #10: the counterfactual-action term ---------------- #
            if self.cfg.w_cf:
                out.update(self._cf_term(last, actions, tgt, intent, v0))
                out["loss"] = out["loss"] + self.cfg.w_cf * out["cf_loss"]

        # ---- v7.2 label supervision (PI 2026-08-31: "It must be trained ----
        # with this data"). ⛔ THE DEFECT THIS ENDS: lat_head / lon_head /
        # route_logits were EMITTED and appeared in NO loss term — inert
        # parameters that would have sat at init through a 30k run while the
        # eval decoded them as "the tactical action". Auxiliary by design:
        # feature prediction stays primary (change #4).
        if any(l is not None for l in (lat_label, lon_label, route_label)):
            if "lat_logits" not in out:
                raise ValueError(
                    "labels supplied but the hierarchy is off (no_hierarchy "
                    "arm) — there is no head for them to supervise")
            if future_feats is None:
                raise ValueError(
                    "labels without future_feats: the label terms attach to "
                    "the training loss, which does not exist here — they "
                    "would be accepted and silently unused")
            # ⭐ -100 IS THE NO-LABEL MARKER (cross_entropy's ignore_index) —
            # the loader emits it for out-of-band windows and unlabeled
            # episodes (97.0 % of B1 episodes carry a record, MEASURED
            # 2026-09-01: 4,572/4,713 clip ids join). The range check
            # validates only the LABELED rows, and an all-ignored family is
            # SKIPPED rather than averaged — an all-ignored CE is NaN
            # (MEASURED, pinned in tests/test_refav1_loader_labels.py), and a
            # NaN here would poison every weight while reading as a batch
            # hiccup, the exact family the short-future guard above refuses.
            tac_terms = []
            for name, lbl, key, n in (
                    ("lat", lat_label, "lat_logits", self.n_lat),
                    ("lon", lon_label, "lon_logits", self.n_lon)):
                if lbl is None:
                    continue
                valid = lbl[lbl != -100]
                if valid.numel() and (int(valid.min()) < 0
                                      or int(valid.max()) >= n):
                    raise ValueError(
                        f"{name}_label outside [0, {n}) — vocabulary "
                        f"{self.cfg.tac_vocab_version} has {n} {name} actions")
                if valid.numel() == 0:
                    continue                      # all-ignored: skip, not NaN
                out[f"loss_{name}_label"] = F.cross_entropy(out[key], lbl)
                tac_terms.append(out[f"loss_{name}_label"])
            if tac_terms:
                out["loss"] = (out["loss"] + self.cfg.w_tac_label
                               * torch.stack(tac_terms).mean())
            if route_label is not None:
                rl = out.get("route_logits")
                if rl is None:
                    raise ValueError(
                        "route_label supplied but the strategic policy emits "
                        "no route_logits")
                rvalid = route_label[route_label != -100]
                if rvalid.numel() and (int(rvalid.min()) < 0
                                       or int(rvalid.max()) >= rl.shape[-1]):
                    raise ValueError(
                        f"route_label outside [0, {rl.shape[-1]})")
                if rvalid.numel():
                    out["loss_route_label"] = F.cross_entropy(rl, route_label)
                    out["loss"] = (out["loss"] + self.cfg.w_str_label
                                   * out["loss_route_label"])

        # ---- the proposal imitation term (Drive-JEPA-adapted, 2026-09-01) --
        # The demo IS the input action sequence's first plan window — no new
        # tensor needed. WTA: only the CLOSEST mode regresses the demo, so
        # modes specialise; the score head learns to pick the winner.
        if self.cfg.w_aux_head:
            if future_feats is None:
                raise ValueError(
                    "w_aux_head without future_feats: the proposal term "
                    "attaches to the training loss, which does not exist "
                    "here — it would be advertised and silently unused")
            demo = actions[:, :self.cfg.plan_steps]
            modes = out["proposal"]                          # [B, M, P, A]
            d = (modes - demo[:, None]).pow(2).mean(dim=(-1, -2))   # [B, M]
            j = d.argmin(dim=-1)
            out["loss_proposal_wta"] = d.gather(1, j[:, None]).mean()
            prop_loss = out["loss_proposal_wta"]
            if self.proposal_score is not None:
                out["loss_proposal_pick"] = F.cross_entropy(
                    out["proposal_logits"], j)
                prop_loss = prop_loss + out["loss_proposal_pick"]
            out["loss"] = out["loss"] + self.cfg.w_aux_head * prop_loss
        return out

    def _cf_term(self, last: Tensor, actions: Tensor, tgt: Tensor,
                 intent: Tensor | None, v0: Tensor | None = None) -> dict:
        """InfoNCE over actions: which action sequence produced this future?

        ⭐ THE PROPERTY THAT MAKES IT AN INSTRUMENT, not merely a loss: an
        action-INDEPENDENT predictor emits the same rollout for every action, so
        every logit is identical, the softmax is uniform, and the loss sits at
        EXACTLY ``ln(1 + cf_negs)``. It cannot do better. ⇒ ``cf_excess > 0`` is
        proof the action was used, against a floor that is arithmetic rather
        than an empirical baseline someone can dispute.

        ⛔ NEGATIVES ARE DRAWN BY A CYCLIC ROLL, NEVER ``randperm``. A
        permutation fixes points with probability ~1/B, and a fixed point hands
        that row its OWN actions as a "counterfactual" -- pulling the loss
        toward the floor and reading as action-blindness that is not there. A
        roll by a non-zero offset is a derangement by construction. (Same defect
        class caught in the action-divergence probe and in O11.)

        ⚠️ KNOWN LIMIT, stated because it decides what a positive result means:
        the negatives come from OTHER BATCH ROWS, hence other clips. Since
        actions correlate with scene identity, "which action produced this
        future" is partly answerable as "which action belongs to this scene",
        which needs no dynamics. ⇒ a same-clip-negatives control is REQUIRED
        before a positive reading is called dynamical. It is not free -- batch
        rows are sampled i.i.d. across thousands of episodes, so same-clip
        negatives need grouped batch construction.
        """
        cfg = self.cfg
        b = actions.shape[0]
        if b < 2:
            raise ValueError(
                f"w_cf {cfg.w_cf} needs batch >= 2 for counterfactuals, got {b}")
        j = min(max(cfg.cf_at_step, 1), int(tgt.shape[1])) - 1
        n_neg = max(int(cfg.cf_negs), 1)

        def roll_to(a: Tensor) -> Tensor:
            # Negatives are OTHER rows' controls rolled from THIS row's
            # measured speed: the counterfactual is "a different action from
            # the same state", so the speed channel re-integrates per candidate.
            return self.operative.rollout(
                last, self.augment_actions(a[:, :j + 1], v0),
                intent=intent)[:, j]

        pos = roll_to(actions)
        negs = []
        for q in range(n_neg):
            off = 1 + (q % (b - 1))
            negs.append(roll_to(torch.roll(actions, shifts=off, dims=0)))

        # Distance to the OBSERVED future at the same step: the true action must
        # be the one that lands there.
        t = tgt[:, j]
        d_pos = (pos - t).flatten(1).pow(2).mean(-1)                    # [B]
        d_neg = torch.stack([(n - t).flatten(1).pow(2).mean(-1)
                             for n in negs], dim=1)                     # [B,n]
        logits = -torch.cat([d_pos[:, None], d_neg], dim=1) / cfg.cf_tau
        target = torch.zeros(b, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, target)
        floor = math.log(1.0 + n_neg)
        with torch.no_grad():
            acc = (logits.argmax(-1) == 0).float().mean()
            sep = (d_neg.mean() - d_pos.mean())
        return {"cf_loss": loss,
                "cf_no_info_floor": floor,
                # ⚠️ detach before float(): a grad-carrying tensor coerced to a
                # scalar warns, and a logged diagnostic must never look like it
                # participates in the graph.
                "cf_excess": floor - float(loss.detach()),
                "cf_pick_acc": float(acc),
                "cf_chance_acc": 1.0 / (1 + n_neg),
                "cf_sep_abs": float(sep),
                "cf_at_step": j + 1,
                "cf_negs": n_neg}

    # -- the imagined goal: the tactical brain's own 6 s field --------------- #
    def _imagine_tactical_goal(self, last: Tensor, brains: dict, v0: Tensor,
                               *, units: str = "kappa"
                               ) -> tuple[Tensor, dict]:
        """The DEFAULT planning goal (change #8), ``[B, Q, d]`` in the tactical
        query space, from vision + nav + the measured v0 and NOTHING from the
        future: the tactical policy's intent (from `_run_brains`, the one shared
        nav site) is decoded by the trained factored heads into a (lat, lon)
        token, the token becomes a control profile (`canonical_controls`), and
        the TACTICAL predictor rolls its own field to 6 s under that profile
        and its intent — 10 steps at 0.6 s, the actions subsampled off the
        operative grid exactly as `forward` builds ``tac_a``, so the speed
        channel (when on) is the same integrated v everywhere.

        ⛔ WHY THIS EXISTED ONLY IN A DOCSTRING UNTIL 2026-09-02: `plan()`
        promised this default and did ``search_goal = None``; with no goal
        term the cost was jerk + curvature only, every zero-curvature
        constant-acceleration candidate scored EXACTLY 0, and `icem_plan`'s
        floor loop kept the LAST tie — `decel_1.5`. MEASURED by the T1 adapter
        build: a constant −1.5 m/s² brake on 51/51 windows.
        ⭐ ``units`` — THE GOAL'S OWN PLANNER->MODEL CROSSING (2026-09-03,
        BACKLOG R26). ``ctrl`` is GEOMETRY (`canonical_controls`;
        ``GOAL_KAPPA_TURN`` is 1/m) and the tactical predictor consumes COMMAND,
        so the goal crosses the SAME boundary a candidate does and it now
        crosses through the SAME function (`_model_actions`). ⛔ Until this
        argument existed the crossing was HALF applied — the candidate converted
        and the goal did not — see `_model_actions` for the measurement.
        ``"kappa"`` is the legacy pass-through and is byte-identical to the
        pre-2026-09-03 expression; `plan` forwards its own
        ``model_action_units`` here, so the two sides can no longer disagree.
        """
        cfg = self.cfg
        intent = brains["intent"]
        vv = getattr(cfg, "tac_vocab_version", "v6.0")
        lat_v, lon_v = tactical_lat_actions(vv), tactical_lon_actions_v(vv)
        lat_i = self.lat_head(intent).argmax(-1).tolist()
        lon_i = self.lon_head(intent).argmax(-1).tolist()
        v0 = torch.as_tensor(v0, dtype=last.dtype, device=last.device).reshape(-1)
        if v0.shape[0] != last.shape[0]:
            raise ValueError(f"v0 carries {v0.shape[0]} rows for a batch of "
                             f"{last.shape[0]}")
        ctrl = torch.stack([
            canonical_controls(lat_v[i], lon_v[j], v, cfg.op_steps, cfg.op_dt)
            for i, j, v in zip(lat_i, lon_i, v0.tolist())]).to(last)   # [B,K,2]
        stride = self._stride(cfg.tac_dt)
        # ⭐ THE GOAL CROSSES INTO THE MODEL HERE — through the same boundary a
        # planner candidate crosses (`_model_actions`), so `units` cannot apply
        # to one side and not the other. The subsample is AFTER the crossing on
        # purpose: `augment_actions` integrates the speed channel on the
        # OPERATIVE grid, so tactical step j opens at the speed of operative
        # step j*stride — exactly as `forward` builds `tac_a`.
        acts = self._model_actions(ctrl, v0, units)[:, ::stride][:, :cfg.tac_steps]
        goal = self.tactical.rollout(self._tac_field(last), acts, intent=intent,
                                     last_only=True)                  # [B,Q,d]
        return goal, {"lat": [lat_v[i] for i in lat_i],
                      "lon": [lon_v[j] for j in lon_i], "controls": ctrl}

    @torch.no_grad()
    def imagined_goal(self, feats: Tensor, *, v0, nav_cmd: Tensor | None = None,
                      model_action_units: str = "kappa"
                      ) -> tuple[Tensor, dict]:
        """`_imagine_tactical_goal` from raw inputs — what `plan()` uses when no
        goal is supplied. ``(goal [B, Q, d], {"lat", "lon", "controls"})``.

        ``model_action_units`` is the same planner->model crossing `plan` takes
        and MUST match the value used there: a goal imagined under one spelling
        and scored against candidates converted under the other is the exact
        defect `_model_actions` documents. ``"controls"`` comes back in
        CURVATURE either way — it is a search-space object, not a model input."""
        if self.tactical_policy is None:
            raise ValueError("imagined_goal needs the hierarchy (tactical_cfg) "
                             "— without a tactical brain there is no tactical "
                             "imagination; plan() then runs goal-free")
        field = self.encode(feats)
        last = self._last_state(field)
        brains = self._run_brains(field.mean(dim=-2), nav_cmd)
        v0_t = torch.as_tensor(v0, dtype=torch.float32,
                               device=feats.device).reshape(-1)
        if v0_t.numel() == 1 and feats.shape[0] > 1:
            v0_t = v0_t.expand(feats.shape[0])
        return self._imagine_tactical_goal(last, brains, v0_t,
                                           units=model_action_units)

    # -- deployment: behaviour by PLANNING, not regression ----------------- #
    @torch.no_grad()
    def plan(self, feats: Tensor, *, v0: float, goal_field: Tensor | None = None,
             target_speed: float | None = None, nav_cmd: Tensor | None = None,
             plan_cfg: PlanConfig | None = None, prev_elites: Tensor | None = None,
             cost_chunk: int = 64, model_action_units: str = "kappa",
             cost_time_grid: str = "dense", goal_time_grid: str = "full",
             cost_metric: str = "cos"):
        """One MPC tick for ONE window (B must be 1).

        ⭐ ``model_action_units`` — THE PLANNER->MODEL CROSSING (PI ruling
        2026-09-03; contract on `kinematic.STEER_WHEELBASE_M`). The search space
        is GEOMETRY: ``PlanConfig.kappa_max``, ``_clip`` and the canonical
        controls behind an imagined goal are all curvature, and a candidate's
        metre-valued path is integrated as curvature. The PREDICTOR, however,
        was trained on the v2ep COMMAND channel — a road-wheel angle. So:

        * ``"kappa"`` (DEFAULT, byte-identical to every pre-2026-09-03 caller)
          hands the candidate to the model unconverted. ⛔ This is the LEGACY
          path and it is wrong by ``arctan(L*kappa)``: a ``GOAL_KAPPA_TURN =
          0.08`` (R = 12.5 m) candidate is imagined by the model as the
          consequence of a 0.08 rad *steering angle* — R ~ 36 m — so the cost
          surface is flat in curvature exactly where the tactical goals live.
          Kept as the default only so every banked number stays reproducible.
        * ``"steer"`` converts each candidate with ``steer = arctan(L_enc*kappa)``
          immediately before `augment_actions`, i.e. at the model boundary and
          nowhere else. The search, the clip, the cost's own curvature penalty
          and the integrated path all stay in curvature.

        ⛔ IT APPLIES TO BOTH SIDES OF THE COMPARISON since 2026-09-03
        (BACKLOG R26). It is forwarded to `_imagine_tactical_goal`, so the goal
        and the candidates cross into the model through ONE function
        (`_model_actions`) under ONE spelling. Before that fix the conversion
        was applied to the candidate only and the converted turn landed one
        float32 ULP FURTHER from its own goal — see `_model_actions`. ⚠️ The
        repair buys the planner NOTHING measurable (`D-REFAV1-BOUNDARY-NULL`:
        ``share = 0.0000 [0, 0]`` on 140/140 windows, both checkpoints); it
        removes a trap, it does not win anything.

        ⭐ ``cost_time_grid`` — THE SAME CROSSING IN TIME (BACKLOG R27).
        ``"dense"`` (DEFAULT, byte-identical to every pre-2026-09-03 caller)
        feeds all ``H`` optimised OPERATIVE actions straight to whichever
        predictor is rolling, so with ``plan_level="tactical"`` the 2.0 s plan is
        imagined as 6.0 s and candidate action ``j`` lands on tactical step
        ``j``, while the GOAL's action ``j`` is operative step ``j*stride``
        (`_imagine_tactical_goal`) — two time grids in one cosine.
        ``"tactical"`` puts the candidate on the goal's (and the TRAINING path's)
        grid: tactical step ``j`` consumes operative action
        ``min(j*stride, H-1)``. See `COST_TIME_GRIDS`. It affects only rollouts
        of `self.tactical`; the coarse->fine re-score on the operative predictor
        is untouched, and a CONSTANT candidate costs the same under both.

        ⛔ ``goal_time_grid`` -- THE SEED IS NOT THE GOAL (L0 / BACKLOG R38).
        ``"full"`` (DEFAULT, byte-identical to every pre-2026-09-03 caller)
        rolls the goal from the token's FULL `op_steps` manoeuvre, of which the
        seed -- ``controls[:cfg.plan_steps]`` -- holds only the first
        `plan_horizon_s`, so 62 of the 64 token pairs seed a control the goal
        was never rolled from and ``1 - cos = 0`` is UNREACHABLE even with a
        perfect world model. ``"plan"`` re-rolls the goal from the seed's own
        action feed, making the two the same forward pass. See
        `GOAL_TIME_GRIDS` for the measurement, for why the truncation (not the
        regrid) is the cause, and for what ``"plan"`` does NOT buy.

        ⛔ ``cost_metric`` -- THE GOAL TERM'S ARITHMETIC (L3 / R29).
        ``"cos"`` (DEFAULT, byte-identical to every pre-2026-09-03 caller) is
        ``1 - cosine_similarity``, whose float32 representable steps just below
        ``cos = 1`` are ``5.96e-08`` apart -- so the modelled kappa-response
        (MEASURED at ``1.63e-10`` over the whole candidate box,
        `D-REFAV1-COST-SURFACE`) is quantised away before any weight gets to
        act on it. ``"chord"`` computes ``||z_hat - g_hat||`` directly, which is
        monotone-equivalent (``chord = sqrt(2*(1-cos))``) and therefore cannot
        re-rank anything in exact arithmetic, but resolves the response at the
        DIFFERENCE's own scale. ⭐ **It is NOT weight-neutral**: it multiplies
        the goal term by ``1/chord`` (measured 5,792.6x) against an unchanged
        ``W_KAPPA``, so flipping it is a metric change AND an implicit
        re-weighting. See `COST_METRICS` and `W_KAPPA`; never flip it without
        declaring the weight in the same arm.

        ⚠️ All four are CALL-SITE arguments, not `RefAV1Config` fields, on
        purpose: adding a config field would change every serialised config
        dict while a training run is live. Nothing on the training path can
        see them -- MEASURED 2026-09-03 with two differently-bound probes:
        `stack/scripts/refa_v1_train.py` contains ``.plan(`` 0 times,
        ``plan_cfg`` 0 times and ``icem`` 0 times.

        The cost is where the hierarchy earns its keep (change #8): the tactical
        target speed and the strategic goal field enter as **cost terms**, not as
        head outputs to be regressed. ``goal_field`` defaults to the tactical
        brain's own imagined 6 s field (`_imagine_tactical_goal`), which is what
        makes this goal-conditioning rather than goal-following.

        ⭐ ONE GOAL SPACE — THE TACTICAL QUERY FIELD ``[Q, d]``. A supplied
        ``goal_field`` (operative ``[1, N, d]``) is pooled through
        `_tac_field`; an operative terminal field (fine-level search, and the
        coarse-to-fine re-score) is pooled the same way before the cosine. That
        pooling is the model's own `tac_pool` — the very map that defines the
        tactical predictor's training target (``tq = _tac_field(adapter(f))``),
        so like is compared with like at every level.

        Provenance travels on the result: ``res.goal_source`` is
        ``"supplied"`` | ``"tactical_imagined"`` | ``"none"`` (no hierarchy),
        ``res.goal_space``, and ``res.goal_action`` (the decoded lat/lon tokens
        + the canonical controls behind an imagined goal).
        """
        if feats.shape[0] != 1:
            raise ValueError("plan() is a single-window API (B must be 1)")
        _check_units(model_action_units)
        _check_cost_time_grid(cost_time_grid)
        _check_goal_time_grid(goal_time_grid)
        _check_cost_metric(cost_metric)
        cfg = self.cfg
        pc = plan_cfg or PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt)
        if pc.horizon != cfg.plan_steps:
            raise ValueError(f"plan horizon {pc.horizon} != cfg.plan_steps "
                             f"{cfg.plan_steps}")
        field = self.encode(feats)
        last = self._last_state(field)
        pooled_win = field.mean(dim=-2)
        pooled = pooled_win[:, -1]

        brains = self._run_brains(pooled_win, nav_cmd)
        intent = None if brains is None else brains["intent"]

        modes = self.proposal(pooled).reshape(cfg.proposal_k, cfg.plan_steps,
                                              cfg.a_dim)
        if self.proposal_score is not None:
            # the score head picks which mode takes the classic proposal slot
            # (it competes as a named baseline); the OTHER modes join the seed
            # pool below — they compete inside iCEM's iteration 0 and can
            # seed its mean, which is the Drive-JEPA proposal-set idea in the
            # planner we already have.
            order = self.proposal_score(pooled)[0].argsort(descending=True)
            modes = modes[order]
        proposal = modes[0]
        seed_pool = modes[1:] if modes.shape[0] > 1 else None
        v0_t = torch.as_tensor([v0], dtype=torch.float32, device=feats.device)

        if cfg.plan_level not in ("tactical", "operative"):
            raise ValueError(f"plan_level must be tactical|operative, "
                             f"got {cfg.plan_level!r}")
        coarse = cfg.plan_level == "tactical"
        search_pred = self.tactical if coarse else self.operative
        search_z = self._tac_field(last) if coarse else last

        # ⭐ THE COARSE COST'S TIME GRID (R27; `COST_TIME_GRIDS`). Built once,
        # outside the chunk loop, and applied ONLY to rollouts of the tactical
        # predictor: the coarse->fine re-score rolls `self.operative`, whose step
        # IS `op_dt`, so it has no defect to repair.
        # ⚠️ HOISTED ABOVE THE GOAL BLOCK 2026-09-03 (L0): with
        # `goal_time_grid="plan"` the GOAL is re-rolled from the seed's
        # own feed and needs the same re-grid, so it must exist first.
        # Pure code motion -- `tac_idx` is read only inside `_cost_chunk`
        # and by the goal repair below, both defined after this point.
        tac_idx = None
        if cost_time_grid == "tactical":
            _s = self._stride(cfg.tac_dt)
            tac_idx = torch.tensor(
                [min(j * _s, pc.horizon - 1) for j in range(cfg.tac_steps)],
                dtype=torch.long, device=feats.device)

        # ⭐ THE GOAL, IN ONE SPACE (docstring). Supplied -> pooled; absent ->
        # the tactical brain's own imagination; no hierarchy -> goal-free.
        goal_action = None
        if goal_field is not None:
            goal_t, goal_source = self._tac_field(goal_field), "supplied"
        elif brains is not None:
            # ⭐ the crossing travels to the GOAL too (R26): one spelling on
            # both sides of the cosine, or the comparison is across conventions.
            goal_t, ga = self._imagine_tactical_goal(
                last, brains, v0_t, units=model_action_units)
            goal_source = "tactical_imagined"
            goal_action = {"lat": ga["lat"][0], "lon": ga["lon"][0],
                           "controls": ga["controls"][0]}
        else:
            goal_t, goal_source = None, "none"

        # ⭐ THE DECODED ACTION ALSO SEEDS THE SEARCH (GPC: proposes, never
        # disposes). MEASURED 2026-09-02 on a random-init tiny model: without
        # this the planner returned hold_v0 on 24/24 windows against a TURN
        # goal at BOTH residual-init scales. `icem_plan`'s coloured noise is
        # zero-mean over time (`colored_noise` subtracts the mean) and its mean
        # is seeded only by the injected candidates, so a SUSTAINED curvature or
        # acceleration is unreachable unless some candidate carries it — and no
        # baseline carries curvature. The canonical controls behind the
        # imagined goal are that candidate: they join the iteration-0 seed pool
        # like a proposal mode and win only on modelled cost.
        if goal_action is not None:
            seed = goal_action["controls"][:cfg.plan_steps][None]      # [1,H,2]
            # ⛔ AND HERE IS WHERE THE SEED STOPS BEING THE GOAL (L0 / R38).
            # `[:cfg.plan_steps]` keeps operative actions [0..9] of a manoeuvre
            # the goal was rolled from at [0,3,...,27]. `goal_time_grid="plan"`
            # closes it the only way that does not change `plan_horizon_s`: by
            # re-rolling the GOAL from THIS seed's own feed -- the same
            # `_model_actions` spelling, the same `tac_idx` re-grid, the same
            # predictor and the same z0 a candidate gets in `_cost_chunk`, so
            # the canonical seed's cost rollout and the goal rollout are the
            # SAME forward pass and `1 - cos` is 0 by construction. Read
            # `GOAL_TIME_GRIDS` before quoting this: it buys IDENTITY, not
            # HORIZON. ``goal_source`` deliberately does NOT change -- the
            # provenance of this choice is the call-site argument, exactly as
            # for `cost_time_grid` and `model_action_units`.
            # ⚠️ COST: one EXTRA tactical rollout per tick under the flag --
            # `_imagine_tactical_goal` must still run, because its `ctrl` IS
            # the seed. Repairing it in place instead would need
            # `_imagine_tactical_goal` to know `plan_steps` and `tac_idx`,
            # which widens the edit past the seed site for one rollout.
            if goal_time_grid == "plan":
                g_acts = self._model_actions(seed, v0_t, model_action_units)
                if tac_idx is not None and search_pred is self.tactical:
                    g_acts = g_acts.index_select(1, tac_idx)
                g_zk = search_pred.rollout(search_z, g_acts, intent=intent,
                                           last_only=True)
                goal_t = g_zk if coarse else self._tac_field(g_zk)
            seed_pool = (seed if seed_pool is None
                         else torch.cat([seed_pool, seed], dim=0))

        # ⭐ THE CENTRING REFERENCE for `cost_metric="ccos"` (D-REFAV1-COST-FORM).
        # The ZERO-ACTION (constant-velocity) terminal field of THIS window --
        # which is exactly the `cv` / `hold_v0` baseline's rollout
        # (`refa_v1_plan._baseline_controls`: `cv` IS `torch.zeros(H, 2)`), so
        # centring uses NO privileged information and NO future information.
        # ⛔ It is keyed on (predictor, z0) and computed INSIDE the same forward
        # pass as the candidates it centres: `plan()` scores under two different
        # (pred, z0) pairs -- the coarse tactical search and the coarse->fine
        # re-score on the operative predictor -- and a reference borrowed from
        # the other one would be a different rollout. Cached because iCEM calls
        # `_cost_chunk` ~2,100 times per tick for at most TWO distinct keys.
        _zref_cache: dict = {}

        def _zero_action_ref(pred, z0) -> Tensor:
            key = (id(pred), int(z0.data_ptr()))
            hit = _zref_cache.get(key)
            if hit is not None:
                return hit
            zero = torch.zeros(1, pc.horizon, cfg.a_dim, device=feats.device,
                               dtype=v0_t.dtype)
            acts0 = self._model_actions(zero, v0_t, model_action_units)
            # the SAME time re-grid the candidates get; an all-zero control is
            # invariant to it in VALUE but not in ROLLOUT LENGTH, so it must be
            # applied or the reference sits at a different horizon.
            if tac_idx is not None and pred is self.tactical:
                acts0 = acts0.index_select(1, tac_idx)
            zk0 = pred.rollout(z0, acts0, intent=intent, last_only=True)
            ref = zk0 if pred is self.tactical else self._tac_field(zk0)
            _zref_cache[key] = ref
            return ref

        def _cost_chunk(controls: Tensor, pred=None, z0=None) -> Tensor:
            pred = pred or search_pred
            z0 = search_z if z0 is None else z0
            n = controls.shape[0]
            z = z0.expand(n, -1, -1)
            # ⭐ THE PLANNER->MODEL CROSSING. `controls` is GEOMETRY (kappa);
            # the predictor consumes COMMAND (steer). Converted at the boundary
            # and nowhere else, so the search, the clip and the integrated path
            # all stay in curvature. The GOAL crosses the SAME boundary under
            # the SAME spelling (`_imagine_tactical_goal(units=...)`), which is
            # what makes the cosine a comparison and not a units mismatch.
            # Speed channel (config-gated): each candidate's OWN accelerations
            # integrated from this tick's measured v0 — the same code path as
            # the T0 forward, so T1 cannot silently read a speed T0 never had.
            # ⚠️ channel 0 (accel) is unit-invariant, so the speed channel is
            # identical under either spelling — verified by test C3.
            acts = self._model_actions(controls, v0_t.expand(n),
                                       model_action_units)
            # ⭐ ...AND THE CROSSING IN TIME. A `tac_dt` step spans operative
            # steps [j*s, (j+1)*s) and consumes the action that OPENS it, which
            # is how `forward` builds `tac_a` and how the goal is rolled; past
            # the optimised window the plan's last action is held.
            if tac_idx is not None and pred is self.tactical:
                acts = acts.index_select(1, tac_idx)
            # last_only: the cost reads the terminal field only (see rollout).
            zk = pred.rollout(z, acts, intent=intent, last_only=True)
            c = torch.zeros(n, device=controls.device)
            if goal_t is not None:
                # an OPERATIVE terminal field is pooled into the tactical query
                # space by the model's own tac_pool (see the docstring)
                zt = zk if pred is self.tactical else self._tac_field(zk)
                g = goal_t.expand(n, -1, -1)
                # ⛔ THE ONE SITE the goal distance is computed. `"cos"` IS the
                # pre-2026-09-03 expression; `"chord"` is monotone-equivalent
                # and NOT weight-neutral; `"ccos"` DOES re-rank (that is the
                # point) and is not weight-neutral either -- see
                # `COST_METRICS` / `W_KAPPA`.
                z_ref = (_zero_action_ref(pred, z0)
                         if cost_metric == "ccos" else None)
                c = c + _goal_term(zt, g, cost_metric, z_ref)
            jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt
            c = c + W_JERK * jerk.pow(2).mean(-1)                  # comfort
            c = c + W_KAPPA * controls[..., 1].pow(2).mean(-1)     # curvature
            if target_speed is not None:
                v_end = v0 + controls[..., 0].sum(-1) * pc.dt
                c = c + W_VEND * (v_end - target_speed).pow(2)
            return c

        def cost_fn(controls: Tensor) -> Tensor:
            """Chunked so the population size is a SEARCH parameter and not a
            memory limit — DINO-WM's N=300 must remain reachable on a 24 GB
            card, which it is not if the whole population rolls at once."""
            if controls.shape[0] <= cost_chunk:
                return _cost_chunk(controls)
            return torch.cat([_cost_chunk(controls[i:i + cost_chunk])
                              for i in range(0, controls.shape[0], cost_chunk)])

        res = icem_plan(cost_fn, v0=v0, cfg=pc, proposal=proposal,
                        prev_elites=prev_elites, seed_pool=seed_pool,
                        device=feats.device)

        # ⛔ TIE-BREAK — the equivalent of "hold_v0 first, keep the first tie",
        # applied HERE because `icem_plan`'s floor loop keeps the LAST baseline
        # on `<=` (cv, hold_v0, proposal, decel_1.5 -> decel). When the cost
        # cannot tell tied baselines apart, doing nothing beats braking: prefer
        # `hold_v0`, then `cv` (the same zero controls, kept as separate names
        # by `_baseline_controls`). A tie is float-exact up to 1e-9 relative;
        # genuinely different costs are never re-ranked.
        if res.source.startswith("baseline:") and res.baseline_costs:
            tied = [k for k, c in res.baseline_costs.items()
                    if math.isclose(c, res.cost, rel_tol=1e-9, abs_tol=1e-12)]
            for pref in ("hold_v0", "cv"):
                if pref in tied:
                    if res.source != f"baseline:{pref}":
                        res.source = f"baseline:{pref}"
                        res.controls = torch.zeros_like(res.controls)
                        res.cost = float(res.baseline_costs[pref])
                    break
        # provenance for the T1 adapter's bookkeeping (plain attributes on the
        # PlanResult — declaring them as fields is a one-line change in
        # refa_v1_plan.py, escalated rather than made here)
        res.goal_source = goal_source
        res.goal_space = "tactical_query_field"
        res.goal_action = goal_action
        # ⭐ the two crossings travel ON the result, so a banked decision says
        # which conventions produced it instead of the reader inferring them
        # from a date. Both defaults are the legacy path.
        res.model_action_units = model_action_units
        res.cost_time_grid = cost_time_grid
        res.goal_time_grid = goal_time_grid
        res.cost_metric = cost_metric

        # ⭐ COARSE-TO-FINE: the search ran on the tactical field; re-score the
        # WINNER (and the baselines it beat) on the full operative field, so the
        # reported cost is the fine-grained one and a coarse-level mistake shows
        # up as a rank flip rather than disappearing. This costs a handful of
        # rollouts, not a population.
        if coarse and cfg.verify_on_operative and self.tactical is not None:
            cands = {"plan": res.controls}
            if pc.inject_baselines:
                from tanitad.refs.refa_v1_plan import _baseline_controls
                cands.update(_baseline_controls(pc, v0, feats.device, proposal))
            names = list(cands)
            stack = torch.stack([cands[k] for k in names])
            fine = _cost_chunk(stack, pred=self.operative, z0=last)
            res.fine_costs = {k: float(v) for k, v in zip(names, fine)}
            best = min(res.fine_costs, key=res.fine_costs.get)
            res.fine_best = best
            res.coarse_fine_agree = (best == "plan")
        return res

    # -- bookkeeping -------------------------------------------------------- #
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_encoder_parameters(self) -> int:
        """0 by construction — features are data tensors on disk. Kept as a
        method so a test can assert the invariant rather than a comment."""
        return 0
