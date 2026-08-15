"""v6 — the STAGED trainer (S-W → S-T → S-S → optional S-J).

WHY STAGED, AND WHY IT IS NOT A PREFERENCE (JEPA_PHYSICS_SURVEY.md §4,
PUBLISHED + our own gate history):
  * Drive-JEPA (arXiv 2601.22032): stage 1 = V-JEPA representation pretraining
    with NO planner in the loop; stage 2 = a proposal-centric planner on top.
    NAVSIM v1 93.7 PDMS / v2 87.8 EPDMS / Bench2Drive 64.52 — SOTA.
  * V-JEPA 2 → 2-AC (arXiv 2506.09985): SSL first, action-conditioning
    post-trained on 62 h of UNLABELED interaction video.
  * DINO-WM (arXiv 2411.04983): dynamics learned on FROZEN patch features.
  * ⇒ *"Nobody at the frontier co-trains the planning gradient into the encoder
    from step 0."* MEASURED on our own ladder: every STAGED component passed
    its gates (W4 emission retrofit PASS; stage-A predictor-only repair ALL
    PASS, gain 0.27 → 0.97), while the co-trained path produced the muffled
    action interface, three selector failures and the missing lead-distance
    variable.

WHAT THIS TRAINER IS. The per-stage loop for :class:`tanitad.models.v6.V6Stack`,
with the ``V6_TRAINING_MEASURES.md`` catalog wired as flagged loss terms, each
DEFAULTING TO THE CATALOG'S SETTING:

  O1  action-conditioned latent prediction with **L_ctrl in RESPONSE FORM FROM
      STEP 0** — not a post-hoc repair. IMPORTED from ``train_stage_a.py``
      (``stage_a_losses`` / ``build_cf_actions`` / ``sample_random_deltas``):
      the same counterfactual arms, the same physical envelope clamp, the same
      response form the W3 gain gate measures. ⚠️ ONE deliberate difference
      from stage A: there the encoder was FROZEN and ``states`` arrived
      detached; in S-W the encoder TRAINS, so states carry gradient. Same loss,
      different stage — said out loud because a silent detach here would make
      S-W a predictor-only run wearing a world-stage name.
  O2  near-field latent loss weighted by **TIME-TO-REACH**, not by a fixed 40 m
      band (HIERARCHY_VOCABULARY §2, PI correction: *"a fixed 40 m band cannot
      cover a 6 s horizon (180 m at 30 m/s)"*). Speed-adaptive by construction.
  O3  masked SPATIAL-latent prediction over the readout grid (I-JEPA adapted to
      cell tokens): contiguous blocks + near-field bands, predicted from
      context AND action (the rolled latent is the context in ``--o3-mode
      action``).
  O4  **interaction-weighted sampling** from ACTIONS ONLY — |jerk|, |decel|,
      steering reversals. Label-free (= LF1). Reweights the draw; never removes
      a window, so parity holds.
  O5  multi-step **rollout consistency to 6 s** — error at EVERY step, not
      endpoint-only (the P5 compounding lesson trained in).
  O6  SIGReg (LeJEPA, λ=0.1, 512 slices) + a standing **spectrum monitor**
      (participation ratio, effective rank, top-k share) every ``--spectrum-every``
      steps, so O6's "rank retention ≥ 0.8× across any curriculum phase" is a
      SERIES and not a single reading.
  T1/S1  the tactical and strategic layers' own goal-conditioned latent
      prediction, each against its layer's stop-grad/EMA target.

⛔ GATES, AND WHAT "GATED" MEANS HERE (X5: *"each stage gated by the frozen
battery BEFORE the next begins — a failed stage never propagates upward"*).
``--gate`` writes ``<out>/stage_gate.json``. Launching stage N+1 runs
:func:`assert_stage_precondition`, which reads stage N's gate and REFUSES on
``pass: false``. A gate whose required probes could not be run is
``pass: null`` = **INCONCLUSIVE, which is NOT a pass** and also refuses; it can
be overridden only by ``--allow-inconclusive-gate`` WITH ``--gate-off-reason``,
and the reason is stamped into the run config and printed as a banner (the
``train_flagship_v4`` off-reason pattern — an override with no stated reason is
how a skipped gate becomes an unremembered decision).

⛔ EVIDENCE / TIER DISCIPLINE. Every number this trainer logs is a TRAINING
number. ⚠️ *"v1.6 is best-in-program" was a trainer log, ~10 % optimistic vs
``eval_*.py``* — trainer val watches a curve; only eval output is quotable.
Capability claims come from **T1** (``taniteval/tools/t1_eval.py``) with the
**four metric families** (LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC —
Sayed 2026-08-02, binding) and the **paired episode-cluster bootstrap**
(``taniteval/ci.py``). An ADE-only table is an INCOMPLETE eval, not a result.

⚠️ POD TRAPS THIS TRAINER IS BUILT AROUND (CLAUDE.md, each has cost hours):
  * ``PYTHONPATH=/workspace/TanitAD/stack`` is REQUIRED or it dies with
    ``ModuleNotFound: tanitad``. ``--print-launch`` emits the full line.
  * ``step_s`` in the log is ACCUMULATED over ``--log-every``; this trainer logs
    ``step_s`` ALREADY DIVIDED and names the divisor, so nobody re-derives a
    "430 s/step" alarm.
  * A completed run writes its DONE-MARKER (``summary.json`` with
    ``"done": true``) in the SAME turn it finishes — a supervised run without
    one gets RESURRECTED the moment whatever broke its relaunches is fixed.
  * ``OMP_NUM_THREADS`` is set defensively (torch spawns ~113 threads/process;
    7 concurrent arms sat at 0–6 % sm for 50 minutes).

USAGE — the copy-pasteable pod lines are in ``V6_TRAINER_DESIGN.md`` §3. The
shortest useful one needs no corpus and no GPU:

  PYTHONPATH=/workspace/TanitAD/stack python3 scripts/train_v6_staged.py \\
      --stage S-W --dry-run --out /workspace/experiments/v6-dryrun
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))      # stack root

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    HORIZON_S, MODULE_GROUPS, PLAN_STEPS, STAGES, InteractionSampler,
    V6Config, V6Stack, apply_stage_freeze, kinematic_saliency,
    near_field_band_mask, sample_cell_block_mask, saliency_weights,
    spectrum_report, stage_trainable_groups, time_to_reach_weights)
from tanitad.models.sigreg import position_relaxed  # noqa: E402

# O1 — IMPORTED, never re-implemented: the response-form L_ctrl and its
# counterfactual machinery are the stage-A artifacts that PASSED the W3 gate.
from train_stage_a import (TRAIN_ARMS, sample_random_deltas,  # noqa: E402
                           stage_a_losses)
from train_v58f_unicycle_head import A_MAX, KAPPA_MAX  # noqa: E402
from stage_a_probes import DACCEL_DEFAULT, DKAPPA_DEFAULT  # noqa: E402

__all__ = [
    "V6LossWeights", "STAGE_PRECONDITION", "STAGE_GATE_SPEC",
    "STAGE_INVALIDATES", "STAGE_INVALIDATION_MECHANISM",
    "o2_near_field_loss", "o3_masked_cell_loss", "o5_rollout_consistency_loss",
    "o6_sigreg_loss", "rollout_step_weights", "build_o4_weights",
    "v6_loss_step", "stage_gate_dict", "write_stage_gate",
    "assert_stage_precondition", "GatePreconditionError",
    "build_stack_from_args", "synthetic_train_batch", "dry_run",
    "build_parser", "main",
]

# ============================================================================
# catalog defaults (V6_TRAINING_MEASURES.md §1–§3)
# ============================================================================


@dataclass
class V6LossWeights:
    """Per-measure weights. Defaults ARE the catalog's settings.

    ``lambda_plan`` is the ``--lambda-plan 0`` instrument of §0 Q2: in S-W the
    planner is ABSENT (the goal/emission heads exist but contribute exactly
    zero loss and are frozen), which is what makes S-W attributable as a pure
    world stage.
    """
    # layer O
    o1_ctrl: float = 1.0        # response-form L_ctrl, FROM STEP 0
    o1_fact: float = 1.0        # factual roll anchored to true waypoints
    o1_scene: float = 0.3       # ego/scene factorisation (P6's subspace)
    o2_nearfield: float = 1.0
    o3_masked: float = 1.0
    o5_rollout: float = 1.0
    o6_sigreg: float = 0.1      # LeJEPA's ONE validated knob — keep it fixed
    # layers T / S
    t1_latent: float = 1.0      # goal-conditioned tactical latent prediction
    s1_latent: float = 1.0      # long-horizon strategic latent prediction
    # planner
    lambda_plan: float = 0.0    # ≡ 0 in S-W by construction
    # ⛔ THE g_tac->OPERATIVE SEAM LOSS (added 2026-08-13, PI question). In S-T
    # the trunk is frozen, so the ONLY way this term can fall is for the goal
    # embedding to carry usable information about the future through
    # intent_proj — which is precisely "the operative predictor learns to be
    # actioned by the tactical goals". Zero in S-W (no goals flow) and S-S.
    seam_op: float = 1.0
    #: SELECTION (V6F_PLANNER_DESIGN.md). ⛔ DEFAULT 0.0 = OFF everywhere, so the
    #: incumbent loss is bit-identical. Needs ``cfg.selector != "none"``.
    #: The objective is E-OBJ-1's ``softade``: the EXPECTED fan error under the
    #: scorer's own softmax — METRIC-AWARE with a hard optimum. That decomposition
    #: is MEASURED, not chosen: swapping a fitted ranker's objective from the
    #: one-hot CE to ``softade`` recovered −0.0974 m (base) / −0.1670 m (XL),
    #: separated, and the recovery was LONGITUDINAL — while SOFTENING the CE
    #: target was separated WORSE (+0.0909 m) at every tau. Metric-awareness
    #: helps; target-softness hurts. ⚠️ Units: this term is in METRES while every
    #: other term is not, so its weight is a declared decision, never a default.
    w_select: float = 0.0

    def for_stage(self, stage: str) -> "V6LossWeights":
        """The weights actually in force for ``stage``.

        S-W zeroes every planner term AND every higher-layer term: a loss whose
        module is frozen still builds a graph, still costs compute, and — the
        part that bites — still appears in the log as if it were training
        something. Zeroing them here keeps the log honest about what moved.
        """
        if stage == "S-W":
            return replace(self, t1_latent=0.0, s1_latent=0.0,
                           lambda_plan=0.0, seam_op=0.0, w_select=0.0)
        if stage == "S-T":
            return replace(self, o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0,
                           o2_nearfield=0.0, o3_masked=0.0, o5_rollout=0.0,
                           o6_sigreg=0.0, s1_latent=0.0)
        if stage == "S-S":
            return replace(self, o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0,
                           o2_nearfield=0.0, o3_masked=0.0, o5_rollout=0.0,
                           o6_sigreg=0.0, t1_latent=0.0, lambda_plan=0.0,
                           seam_op=0.0, w_select=0.0)
        if stage == "S-J":
            return self
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


#: X5's ordering, as data. ``None`` = no precondition (S-W starts the ladder).
STAGE_PRECONDITION: dict[str, str | None] = {
    "S-W": None, "S-T": "S-W", "S-S": "S-T", "S-J": "S-S",
}

#: ⛔ THE ONLY MODULES A STAGE MAY INTRODUCE on top of the stage below it.
#:
#: MEASURED 2026-08-16, and it would have killed the S-T launch on its first
#: command: :func:`load_stage_init` loaded with ``strict=True``, so a stage that
#: legitimately ADDS a module was refused —
#:
#:     S-T selector="goal" -> RuntimeError: Missing key(s): cand_score.cand_bias,
#:                            cand_score.log_tau, cand_score.goal_point.weight...
#:     S-T selector="mlp"  -> RuntimeError: Missing key(s): cand_score.fc1.weight...
#:
#: The guard was right in spirit and blind in practice: it could not tell "the
#: TRUNK is missing" (fatal — the stage would train on a random encoder while
#: its log looked healthy, exactly what the docstring warns about) from "the
#: PLANNER'S NEW HEAD is missing" (expected — S-T is where the selector is
#: built). ⇒ the allowance is an EXPLICIT, per-stage ALLOWLIST rather than a
#: relaxed ``strict``, because ``strict=False`` would also wave through a
#: missing ``emission.*`` and silently random-init the whole emission head.
#:
#: ⚠️ An allowed prefix must be WHOLLY absent from the checkpoint. A PARTIALLY
#: present module is a geometry mismatch, not an introduction, and stays fatal.
STAGE_MAY_INTRODUCE: dict[str, tuple[str, ...]] = {
    "S-W": (),                  # starts the ladder; there is nothing to inherit
    "S-T": ("cand_score.",),    # the selector is built HERE, by design
    "S-S": (),                  # trains layer_str, which S-T already carried
    "S-J": (),                  # joint polish introduces nothing
}

#: ⛔ THE LADDER RUNS BACKWARDS TOO. :data:`STAGE_PRECONDITION` is the FORWARD
#: check — "the stage below passed". It cannot see the other direction: a stage
#: that trains an UPPER layer can invalidate the certificate a LOWER layer
#: already earned, because the lower layer is frozen while its INPUT moves.
#:
#: This is registry §1.14's consumer-invalidation one level up, and it lives
#: inside the ladder where it is easy to miss: S-T's gate certifies ``sel_gap``
#: and the TACTICAL family, then S-S changes the very thing they were measured
#: on and no gate re-checks them. Without this, an S-S gate could read PASS on
#: ``STRATEGIC_family`` alone and S-J would launch on an uncertified selector.
#:
#: Each entry names the stage whose certificate is invalidated and the exact
#: seam that does it, so the mechanism cannot be lost the way the "please merge"
#: requests were. ``()`` = trains nothing that any frozen consumer reads.
STAGE_INVALIDATES: dict[str, tuple[str, ...]] = {
    "S-W": (),      # starts the ladder; nothing below it exists yet
    "S-T": (),      # trains layer_tac + planner on a FROZEN S-W trunk; the
                    # trunk's inputs (pixels) are unmoved, so S-W's certificate
                    # still applies verbatim
    "S-S": ("S-T",),
    "S-J": (),      # everything trains jointly and the S-J gate's own `no_harm`
                    # probe IS the revalidation (battery FLAT across the phase)
}

#: The seam behind each :data:`STAGE_INVALIDATES` entry, quoted from source so
#: an override is a conscious act rather than a shrug at an unexplained key.
STAGE_INVALIDATION_MECHANISM: dict[str, str] = {
    "S-S": ("S-S trains `layer_str` ONLY (v6.py:995). Its output flows "
            "`goal_head_str -> e_g_str -> goal_head_tac(cond=e_g_str) -> "
            "e_g_tac` (v6.py:1520-1528), and `e_g_tac` is the SELECTOR'S ONLY "
            "INPUT (v6.py:655; score_i = -||endpoint_i - g_hat||/tau + b_i "
            "with g_hat = W.e_g_tac + c, v6.py:619). `goal_head_tac` and the "
            "selector are FROZEN in S-S — but their input distribution moves. "
            "S-T certified `sel_gap` against the S-T-era e_g_tac; that "
            "certificate does not survive S-S. Re-measure, do not assume."),
}

#: Per-stage ``λ_plan`` default, resolved when ``--lambda-plan`` is not given.
#: S-W is 0 BY CONSTRUCTION (the planner is absent — that is what makes the
#: world stage attributable); S-T is where the planner is post-trained on the
#: frozen S-W trunk (the Drive-JEPA shape); S-S trains the strategic layer only
#: and leaves the planner frozen; S-J polishes everything with isolation ON.
STAGE_LAMBDA_PLAN: dict[str, float] = {
    "S-W": 0.0, "S-T": 1.0, "S-S": 0.0, "S-J": 1.0,
}

#: The frozen-battery probes each stage's gate REQUIRES, and the entry point
#: that owns each. ``V6_TRAINING_MEASURES`` names the battery (P1–P9, I4) as
#: THE single yardstick from v5f to v6 — the point of naming the owner here is
#: that a probe reported as "n/a" must name what could not be imported, never
#: silently vanish (rule 2: absence at one location is not absence).
STAGE_GATE_SPEC: dict[str, dict] = {
    "S-W": {
        "required": ("P1", "P3", "P6"),
        "reported": ("P2", "P5", "P8", "O6_spectrum"),
        "owners": {"P1": "scripts/probe_latent_state.py",
                   "P2": "scripts/probe_latent_state.py",
                   "P3": "scripts/stage_a_probes.py",
                   "P6": "scripts/stage_a_probes.py",
                   "P5": "taniteval/tools/t1_eval.py",
                   "P8": "scripts/train_p8_occupancy.py",
                   "O6_spectrum": "tanitad.models.v6.spectrum_report"},
        "criteria": {
            "P1_retention": ">= 0.85x R2(z) at k=10 per driving target",
            "P3_sign": ">= 0.95 per channel, BOTH lat and lon",
            "P3_gain": "median gain in [0.5, 2.0], WITHOUT post-training",
            "P6_dims": "action-subspace dims (80 % var) <= 32",
            "O6_rank_retention": ">= 0.8x effective rank across phases"},
    },
    "S-T": {
        "required": ("TACTICAL_family", "sel_gap"),
        "reported": ("P7", "LATERAL_family", "X2_seam"),
        "owners": {"TACTICAL_family": "taniteval/tools/eval_four_families.py",
                   "LATERAL_family": "taniteval/tools/eval_four_families.py",
                   "sel_gap": "tanitad.models.tactical.sel_gap_tac",
                   "P7": "scripts/w7_roll_rerank.py",
                   "X2_seam": "taniteval/ci.py (PAIRED bootstrap only)"},
        "criteria": {
            "sel_gap": "<= 0.5x the fan oracle at T1 tier",
            "TACTICAL_family": "confusion improves on E4.1-derived strata",
            "P7_rho": ">= 0.3 with CI excluding 0, per stratum"},
    },
    "S-S": {
        # ⛔ THE LAST TWO ARE REVALIDATIONS, NOT NEW MEASURES. See
        # :data:`STAGE_INVALIDATES` — S-S retrains the goal that S-T's FROZEN
        # selector consumes, so S-T's certificate stops applying the moment
        # S-S starts. They are ``required`` (not ``reported``) because an S-S
        # gate that omits them must read INCONCLUSIVE, never PASS.
        "required": ("STRATEGIC_family",
                     "sel_gap_revalidated", "TACTICAL_revalidated"),
        "reported": ("S1_ade_8_30s", "X2_seam"),
        "owners": {"STRATEGIC_family": "taniteval/tools/eval_four_families.py",
                   "S1_ade_8_30s": "taniteval/tools/t1_eval.py",
                   "X2_seam": "taniteval/ci.py (PAIRED bootstrap only)",
                   "sel_gap_revalidated": "tanitad.models.tactical.sel_gap_tac "
                                          "(RE-RUN under the post-S-S g_tac)",
                   "TACTICAL_revalidated":
                       "taniteval/tools/eval_four_families.py "
                       "(RE-RUN under the post-S-S g_tac)"},
        "criteria": {
            "STRATEGIC_family": "computable at all (measured vs n/a today)",
            "S1_ade_8_30s": "beats CV/corridor baselines at T1",
            "sel_gap_revalidated":
                "still <= 0.5x the fan oracle AFTER S-S moved e_g_tac — the "
                "same bar S-T passed, re-measured on the new input "
                "distribution. PAIRED bootstrap vs the S-T reading.",
            "TACTICAL_revalidated":
                "TACTICAL family does not regress vs its S-T reading "
                "(paired episode-cluster bootstrap, same windows)"},
    },
    "S-J": {
        "required": ("X3_isolation", "no_harm"),
        "reported": ("P1", "P3", "P6", "four_families"),
        "owners": {"X3_isolation": "V6Stack.assert_isolation",
                   "no_harm": "the frozen battery, before vs after S-J"},
        "criteria": {
            "X3_isolation": "zero live forbidden edges",
            "no_harm": "battery FLAT across the joint phase (H-COTRAIN rule)"},
    },
}


class GatePreconditionError(SystemExit):
    """A stage refused to start because the stage below it did not pass."""


# ============================================================================
# the measure losses — PURE, CPU-testable (no dataset, no checkpoint)
# ============================================================================

def o2_near_field_loss(pred_cells: Tensor, true_cells: Tensor,
                       cell_ranges_m: Tensor, v_ego: Tensor, *,
                       tau_s: float = 2.0, horizon_s: float = HORIZON_S,
                       v_floor: float = 1.0) -> tuple[Tensor, dict]:
    """O2 — per-cell latent L1 weighted by TIME-TO-REACH.

    ``pred_cells`` / ``true_cells`` ``[B, C, d_r]`` readout CELL tokens (the
    DINO-WM lesson: *pooling is where geometry goes to die*, so O2 acts on
    cells and never on the pooled state); ``cell_ranges_m`` ``[C]``;
    ``v_ego`` ``[B]`` m/s.

    The weighting is TIME-scaled (HIERARCHY_VOCABULARY §2), so at 30 m/s the
    same time band covers ~180 m while at 5 m/s it covers ~30 m — one rule, no
    speed-dependent constant. Weights are mean-1 normalised over cells, so this
    RE-ALLOCATES the loss instead of rescaling it (a weighting that also
    changes the gradient magnitude is a learning-rate change in disguise).
    """
    if pred_cells.shape != true_cells.shape or pred_cells.ndim != 3:
        raise ValueError(f"cells must match and be [B, C, d_r], got "
                         f"{tuple(pred_cells.shape)} vs "
                         f"{tuple(true_cells.shape)}")
    c = pred_cells.shape[1]
    if cell_ranges_m.reshape(-1).numel() != c:
        raise ValueError(f"cell_ranges_m must have {c} entries, got "
                         f"{cell_ranges_m.reshape(-1).numel()}")
    dist = cell_ranges_m.reshape(1, c).to(pred_cells.device).float()
    w = time_to_reach_weights(dist.expand(pred_cells.shape[0], c),
                              v_ego.to(pred_cells.device), tau_s=tau_s,
                              horizon_s=horizon_s, v_floor=v_floor)
    err = (pred_cells.float() - true_cells.float()).abs().mean(dim=-1)  # [B,C]
    loss = (w * err).mean()
    return loss, {"o2_loss": float(loss.detach()),
                  "o2_w_min": float(w.min()), "o2_w_max": float(w.max()),
                  "o2_unweighted": float(err.mean().detach()),
                  "o2_tau_s": tau_s}


def o3_masked_cell_loss(masked_predictor, ctx_cells: Tensor,
                        true_cells: Tensor, mask: Tensor
                        ) -> tuple[Tensor, dict]:
    """O3 — masked SPATIAL-latent prediction over the readout grid.

    ``ctx_cells`` is what the model may look at (in ``--o3-mode action`` it is
    the ACTION-CONDITIONED rolled latent's cells, so the task is "predict the
    occluded block given the context AND the action" — O3's own wording);
    ``true_cells`` is the target; ``mask`` ``[B, C]`` bool, True == masked.

    Scored on the MASKED cells ONLY. Scoring visible cells too would let the
    model win by copying, which is exactly how a masking objective silently
    becomes an autoencoder.
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    if mask.shape != true_cells.shape[:2]:
        raise ValueError(f"mask {tuple(mask.shape)} must be [B, C] matching "
                         f"cells {tuple(true_cells.shape)}")
    n_masked = int(mask.sum())
    pred = masked_predictor(ctx_cells, mask)
    if n_masked == 0:
        z = pred.sum() * 0.0                 # in the graph, contributes nothing
        return z, {"o3_loss": 0.0, "o3_mask_rate": 0.0, "o3_n_masked": 0}
    err = (pred.float() - true_cells.float()).abs().mean(dim=-1)        # [B,C]
    loss = err[mask].mean()
    return loss, {"o3_loss": float(loss.detach()),
                  "o3_mask_rate": float(mask.float().mean()),
                  "o3_n_masked": n_masked,
                  "o3_visible_err": float(err[~mask].mean().detach())
                  if bool((~mask).any()) else None}


def rollout_step_weights(k: int, mode: str = "uniform", *,
                         device=None, dtype=torch.float32) -> Tensor:
    """Per-step weights for O5. ``uniform`` (the catalog's "error at every step,
    not endpoint-only") or ``linear-decay`` (down-weights late steps where the
    target itself is noisier). ``endpoint`` exists ONLY as the ablation control
    that reproduces the defect O5 fixes — it is never a default."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if mode == "uniform":
        w = torch.ones(k, dtype=dtype, device=device)
    elif mode == "linear-decay":
        w = torch.linspace(1.0, 0.25, k, dtype=dtype, device=device)
    elif mode == "endpoint":
        w = torch.zeros(k, dtype=dtype, device=device)
        w[-1] = 1.0
    else:
        raise ValueError(f"mode must be uniform|linear-decay|endpoint, got "
                         f"{mode!r}")
    return w / w.mean().clamp_min(1e-8)


def o5_rollout_consistency_loss(zhat_steps, z_true_steps, weights: Tensor
                                ) -> tuple[Tensor, dict]:
    """O5 — multi-step rollout consistency, error at EVERY step.

    ``zhat_steps`` / ``z_true_steps``: sequences of ``[B, S]`` latents of equal
    length (the rolled prediction and the ENCODED true future). ``weights``
    ``[k]`` from :func:`rollout_step_weights`.

    This is the P5 lesson (*compounding-error boundedness*) trained IN rather
    than measured after: an endpoint-only loss is minimisable by a trajectory
    that is wrong throughout and right at the end, and that is precisely the
    shape T1 rollouts fail with.
    """
    if len(zhat_steps) != len(z_true_steps):
        raise ValueError(f"rollout length mismatch: {len(zhat_steps)} vs "
                         f"{len(z_true_steps)}")
    k = len(zhat_steps)
    if weights.numel() != k:
        raise ValueError(f"weights must be [k={k}], got {weights.numel()}")
    per = torch.stack([(zhat_steps[j].float() - z_true_steps[j].float())
                       .abs().mean() for j in range(k)])               # [k]
    loss = (weights.to(per.device).float() * per).mean()
    return loss, {"o5_loss": float(loss.detach()),
                  "o5_k": k,
                  "o5_step1": float(per[0].detach()),
                  "o5_stepK": float(per[-1].detach()),
                  "o5_growth": float((per[-1] / per[0].clamp_min(1e-8))
                                     .detach())}


def o6_sigreg_loss(sigreg, z: Tensor, free_dims: int = 0) -> Tensor:
    """O6 — SIGReg (full_relaxed), KEPT per the PI's 2026-08-11 call.

    Delegates to :func:`tanitad.models.sigreg.position_relaxed`, which exempts
    a fixed ego-motion subspace so anti-collapse and metric-position structure
    stop cancelling. ⚠️ Never divide the Epps-Pulley statistic by n — that was
    the ALPS-4B bug that silently disabled the loss, and it is guarded inside
    ``sigreg.py``; this wrapper exists so nobody re-implements the call."""
    return position_relaxed(sigreg, z, free_dims)


def build_o4_weights(actions_per_window, *, dt: float = 0.1,
                     alpha: float = 1.0, floor: float = 0.25,
                     w_jerk: float = 1.0, w_decel: float = 1.0,
                     w_reversal: float = 1.0) -> tuple[Tensor, dict]:
    """O4 — per-window sampling weights from ACTIONS ONLY.

    ``actions_per_window`` ``[N, T, >=2]``. Returns ``(weights [N], log)``.
    ``alpha=0`` reproduces uniform sampling exactly (the attributability
    control). The log carries the saliency quantiles so a run can SAY how
    skewed its draw was rather than assert that it oversampled interaction.
    """
    s = kinematic_saliency(actions_per_window, dt=dt, w_jerk=w_jerk,
                           w_decel=w_decel, w_reversal=w_reversal)
    w = saliency_weights(s, alpha=alpha, floor=floor)
    q = torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])
    return w, {"o4_n": int(s.numel()), "o4_alpha": alpha, "o4_floor": floor,
               "o4_saliency_quantiles": [round(float(v), 5) for v in
                                         torch.quantile(s.float(), q)],
               "o4_weight_max_over_min":
                   float(w.max() / w.min().clamp_min(1e-12))}


# ============================================================================
# the per-batch loss assembly
# ============================================================================

def v6_loss_step(stack: V6Stack, batch: dict, *, stage: str,
                 weights: V6LossWeights, o1_k: int = 10, o5_k: int = 20,
                 o5_mode: str = "uniform", o3_mode: str = "action",
                 o3_blocks: int = 2, o3_block_hw: tuple[int, int] = (2, 2),
                 o3_band_rows: int = 0, o2_tau_s: float = 2.0,
                 dkappa: float = DKAPPA_DEFAULT,
                 daccel: float = DACCEL_DEFAULT,
                 rand_dk: Tensor | None = None,
                 rand_da: Tensor | None = None,
                 generator: torch.Generator | None = None,
                 rollout_grad_checkpoint: bool | None = None) -> dict:
    """One batch of the v6 staged objective.

    BATCH CONTRACT (all tensors on one device):
      ``frames``          [B, W, C, H, W']   the causal window
      ``actions2``        [B, W, 2]          recorded (steer, accel)
      ``future_actions2`` [B, H, 2]          H >= max(o1_k, o5_k) - 1
      ``v0``              [B]                m/s — the INTEGRATION constant
      ``gt_wp``           [B, o1_k, 2]       true future ego waypoints
      ``z_true_steps``    list[k] of [B, d_op] ENCODED true future latents
                          (k >= max(o1_k, o5_k)), detached by the caller
      ``own_frames_tac`` / ``own_frames_str``  E-ENC arm (b) only

    Returns ``{"loss": Tensor, **components, "log": dict}``. Terms whose weight
    is 0 for this stage are SKIPPED, not multiplied by zero — a skipped term
    costs no compute and, more importantly, cannot appear in the log looking
    like it trained something.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    w = weights.for_stage(stage)
    cfg = stack.cfg
    dev = batch["frames"].device
    log: dict = {"stage": stage}
    terms: dict[str, Tensor] = {}

    # ---- the shared forward ------------------------------------------------
    out = stack.forward(frames=batch["frames"], actions=_lift3(
        batch["actions2"], batch["v0"]), v0=batch["v0"],
        own_frames_tac=batch.get("own_frames_tac"),
        own_frames_str=batch.get("own_frames_str"))
    states = out["z_op_win"]                                   # [B, W, d_op]
    z_true = batch["z_true_steps"]

    # ---- O1: response-form L_ctrl (IMPORTED from stage A) ------------------
    if w.o1_ctrl or w.o1_fact or w.o1_scene:
        if len(z_true) < o1_k:
            raise ValueError(f"O1 needs z_true_steps >= o1_k={o1_k}, got "
                             f"{len(z_true)}")
        if batch["gt_wp"].shape[1] != o1_k:
            raise ValueError(f"gt_wp horizon {batch['gt_wp'].shape[1]} != "
                             f"o1_k={o1_k}")
        if rand_dk is None or rand_da is None:
            # the "random" counterfactual arm needs a per-window (Δκ, Δa) draw;
            # drawing it here (deterministic under ``generator``) means a
            # caller cannot accidentally run O1 with four arms instead of five.
            rand_dk, rand_da = sample_random_deltas(
                batch["v0"].shape[0],
                generator or torch.Generator().manual_seed(0),
                0.05, 3.0)
            rand_dk = rand_dk.to(dev)
            rand_da = rand_da.to(dev)
        L1 = stage_a_losses(
            stack.predictor_op, stack.step_readout_op, states,
            batch["actions2"], batch["future_actions2"], batch["v0"],
            batch["gt_wp"], z_true[o1_k - 1], o1_k, dkappa=dkappa,
            daccel=daccel, rand_dk=rand_dk, rand_da=rand_da,
            w_ctrl=w.o1_ctrl, w_fact=w.o1_fact, w_scene=w.o1_scene,
            ctrl_form="response")
        terms["o1"] = L1["loss"]
        log |= {"o1_ctrl": float(L1["l_ctrl"].detach()),
                "o1_fact": float(L1["l_fact"].detach()),
                "o1_scene": float(L1["l_scene"].detach()),
                "o1_factual_ade": float(L1["factual_ade"]),
                "o1_basis_dims": L1["basis_dims"],
                "o1_arms": list(TRAIN_ARMS)}

    # ---- the factual rollout, computed ONCE for O2 / O3 / O5 ---------------
    need_roll = bool(w.o2_nearfield or w.o3_masked or w.o5_rollout)
    if need_roll:
        from tanitad.models.metric_dynamics import rollout_transitions
        k_roll = max(o5_k, 1)
        if len(z_true) < k_roll:
            raise ValueError(f"rollout needs z_true_steps >= o5_k={o5_k}, "
                             f"got {len(z_true)}")
        aw3, fa3 = _lift3(batch["actions2"], batch["v0"]), _lift3(
            batch["future_actions2"], batch["v0"])
        # ⛔ The rollout's checkpointing is NOT the encoder's. It used to read
        # `cfg.encoder.grad_checkpoint`, which coupled two unrelated decisions
        # to one flag: the k=60 roll NEEDS checkpointing (it fixed a MEASURED
        # 37.97/44 GiB OOM), while the encoder's is a pure speed/memory trade
        # that costs ~2x the ViT forward and may be unaffordable to keep when
        # the GPU has headroom. MEASURED 2026-08-14: S-W ran at 42.8 % mean GPU
        # util with 20 GB free — i.e. paying recompute it did not need to.
        # Default preserves the old behaviour exactly when unset.
        rgc = (cfg.encoder.grad_checkpoint if rollout_grad_checkpoint is None
               else bool(rollout_grad_checkpoint))
        trans = rollout_transitions(stack.predictor_op, states, aw3, fa3,
                                    k_roll, grad_checkpoint=rgc)
        zhat_steps = [t[1] for t in trans]

        # ---- O5: error at EVERY step --------------------------------------
        if w.o5_rollout:
            sw = rollout_step_weights(k_roll, o5_mode, device=dev)
            l5, lg5 = o5_rollout_consistency_loss(zhat_steps,
                                                  z_true[:k_roll], sw)
            terms["o5"] = w.o5_rollout * l5
            log |= lg5 | {"o5_mode": o5_mode}

        # ---- O2: time-to-reach weighted near field ------------------------
        if w.o2_nearfield:
            j = min(o1_k, k_roll) - 1
            l2, lg2 = o2_near_field_loss(
                stack.cells(zhat_steps[j]), stack.cells(z_true[j]),
                stack.cell_ranges_m, batch["v0"], tau_s=o2_tau_s,
                horizon_s=cfg.horizon_s)
            terms["o2"] = w.o2_nearfield * l2
            log |= lg2 | {"o2_at_step": j + 1}

        # ---- O3: masked spatial-latent prediction -------------------------
        if w.o3_masked:
            b = states.shape[0]
            gh, gw = cfg.grid_shape
            m = sample_cell_block_mask(gh, gw, n_blocks=o3_blocks,
                                       block_h=o3_block_hw[0],
                                       block_w=o3_block_hw[1], batch=b,
                                       generator=generator)
            if o3_band_rows:
                m = m | near_field_band_mask(gh, gw, rows=o3_band_rows,
                                             batch=b)
            m = m.to(dev)
            j = min(o1_k, k_roll) - 1
            ctx = (stack.cells(zhat_steps[j]) if o3_mode == "action"
                   else stack.cells(states[:, -1]))
            l3, lg3 = o3_masked_cell_loss(stack.masked_cells, ctx,
                                          stack.cells(z_true[j]), m)
            terms["o3"] = w.o3_masked * l3
            log |= lg3 | {"o3_mode": o3_mode}

    # ---- O6: SIGReg on the operative latent --------------------------------
    if w.o6_sigreg:
        l6 = o6_sigreg_loss(stack.sigreg, states.reshape(-1, states.shape[-1]),
                            cfg.sigreg_free_dims)
        terms["o6"] = w.o6_sigreg * l6
        log["o6_sigreg"] = float(l6.detach())

    # ---- the g_tac->operative SEAM (S-T / S-J) ------------------------------
    if w.seam_op:
        if not z_true:
            raise ValueError(
                "seam_op > 0 needs at least one encoded future latent "
                "(z_true_steps) — the S-T batch must run the future encode "
                "with need_k >= 1, not skip it")
        seam = out["zhat_op_seam"]
        k1 = min(int(kk) for kk in seam)          # the 1-step head
        lseam = (seam[k1].float() - z_true[k1 - 1].float()).abs().mean()
        terms["seam"] = w.seam_op * lseam
        log["seam_op"] = float(lseam.detach())

    # ---- T1: goal-conditioned tactical latent prediction --------------------
    if w.t1_latent:
        tgt = batch.get("z_tac_next_target")
        if tgt is None:
            tgt = out["z_tac_target"]        # 1-step identity target fallback
            log["t1_target"] = "self (no z_tac_next_target in batch)"
        lt = (out["zhat_tac"].float() - tgt.float()).abs().mean()
        terms["t1"] = w.t1_latent * lt
        log["t1_latent"] = float(lt.detach())

    # ---- S1: long-horizon strategic latent prediction -----------------------
    if w.s1_latent:
        tgt = batch.get("z_str_next_target", out["z_str_target"])
        ls = (out["zhat_str"].float() - tgt.float()).abs().mean()
        terms["s1"] = w.s1_latent * ls
        log["s1_latent"] = float(ls.detach())

    # ---- planner (λ_plan) ---------------------------------------------------
    if w.lambda_plan:
        tgt = batch.get("plan_target")       # [B, plan_steps, 2] ego waypoints
        if tgt is None:
            raise ValueError("lambda_plan > 0 needs batch['plan_target'] "
                             "[B, plan_steps, 2] — a planner loss without a "
                             "target is how λ_plan silently becomes 0")
        fan = out["plan"]["waypoints"].float()                # [B, N, 60, 2]
        err = (fan - tgt.float()[:, None]).norm(dim=-1).mean(dim=-1)
        winner = err.argmin(dim=1)
        ar = torch.arange(fan.shape[0], device=fan.device)
        lp = err[ar, winner].mean()
        # ⚠️ epsilon-RELAXED WTA (default 0.0 == the incumbent PURE WTA, and the
        # term below is not even constructed then, so the graph is unchanged).
        # Under pure WTA the N−1 LOSING candidates receive EXACTLY ZERO gradient
        # and nothing bounds the fan's MEAN. MEASURED on the banked REF-C-XL fan
        # 2026-08-15: oracle 0.1639 m against a fan mean of 13.9564 m — 85x —
        # and that is the regime in which a cost's argmin is a coin flip.
        # Bounding the losers is the cheapest structural defence available, and
        # it costs zero parameters.
        if cfg.plan_wta_eps > 0.0:
            n = err.shape[1]
            if n > 1:
                loser = (err.sum(dim=1) - err[ar, winner]) / (n - 1)
                lp = lp + cfg.plan_wta_eps * loser.mean()
                log["plan_loser_mean"] = float(loser.mean().detach())
        terms["plan"] = w.lambda_plan * lp
        log["fan_mean_ade"] = float(err.mean().detach())
        log["fan_oracle_ade"] = float(err.min(dim=1).values.mean().detach())
        # §4b: report the bands SEPARATELY. A pooled 0–6 s number cannot show
        # the 2 s seam, and the seam is what X2 exists to verify.
        op_b, tac_b = cfg.split_bands(fan[ar, winner].detach(), dim=-2)
        t_op, t_tac = cfg.split_bands(tgt.float(), dim=-2)
        log |= {"plan_wta": float(lp.detach()),
                "plan_ade_0_2s": float((op_b - t_op).norm(dim=-1).mean()),
                "plan_ade_2_6s": float((tac_b - t_tac).norm(dim=-1).mean())}

        # ---- SELECTION (w_select, default 0.0 == absent) --------------------
        if w.w_select:
            if "sel_score" not in out["plan"]:
                raise ValueError(
                    "w_select > 0 with cfg.selector='none' — a selection loss "
                    "with no scorer is how a selector silently never trains. "
                    "Build the stack with --selector goal.")
            score = out["plan"]["sel_score"].float()          # [B, N]
            # E-OBJ-1 `softade`: EXPECTED fan error under the scorer's own
            # softmax. Metric-aware (a loser that misses by a centimetre is not
            # punished like one that misses by ten metres — the defect that made
            # EVERY fitted ranker in E-S1-0 separated WORSE than the incumbent),
            # and its optimum is still a sharp distribution on the low-error
            # candidate, so it optimises the LOWER TAIL that governs argmax.
            p = score.softmax(dim=-1)
            lsel = (p * err.detach()).sum(dim=-1).mean()
            terms["select"] = w.w_select * lsel
            sel_idx = score.argmax(dim=-1)
            rank = err.argsort(dim=1).argsort(dim=1)
            log |= {"sel_softade": float(lsel.detach()),
                    "sel_ade": float(err[ar, sel_idx].mean().detach()),
                    # ⭐ THE PRIMARY ENDPOINT for a ranking claim (W7-PROG's
                    # precedent). 0 = always the true best, 0.5 = a coin flip.
                    "sel_norm_err_rank": float(
                        rank[ar, sel_idx].float().mean().detach()
                        / max(err.shape[1] - 1, 1)),
                    "sel_gap": float((err[ar, sel_idx]
                                      - err.min(dim=1).values).mean().detach())}

    if not terms:
        raise RuntimeError(f"stage {stage} produced NO loss terms — every "
                           f"weight is zero, which would train nothing while "
                           f"looking like a run")
    total = torch.stack([t.float() for t in terms.values()]).sum()
    log["loss"] = float(total.detach())
    log["terms"] = sorted(terms)
    return {"loss": total, "log": log, "out": out,
            **{k: v for k, v in terms.items()}}


def _lift3(a2: Tensor, v0: Tensor) -> Tensor:
    """2-channel (steer, accel) -> the 3-channel speed-append format the
    predictor trains with. The ``canary_rollout`` / ``lift_actions3`` pattern
    (SPEED_SCALE contract), inlined for a single tensor so this module does not
    depend on the P8 trainer just to append a column."""
    from tanitad.models.flagship_v15 import SPEED_SCALE
    v = (v0.to(a2.dtype) / SPEED_SCALE)[:, None, None]
    return torch.cat([a2, v.expand(-1, a2.shape[1], -1)], dim=-1)


# ============================================================================
# X5 — the per-stage gate
# ============================================================================

def stage_gate_dict(stage: str, probes: dict, *, run: dict | None = None
                    ) -> dict:
    """Assemble ``stage_gate.json`` from whatever probes actually ran.

    ``probes`` maps probe name -> ``{"pass": bool|None, ...}``. A required probe
    that is ABSENT or reports ``pass: None`` makes the whole gate
    **INCONCLUSIVE** (``"pass": null``), never a pass. That distinction is the
    whole mechanism: a gate that quietly reads a missing probe as satisfied is
    not a gate, and X5's rule — *a failed stage never propagates upward* — is
    only enforceable if "did not run" and "ran and passed" stay different
    words.
    """
    spec = STAGE_GATE_SPEC[stage]
    req = spec["required"]
    missing = [p for p in req if p not in probes]
    inconclusive = [p for p in req
                    if p in probes and probes[p].get("pass") is None]
    failed = [p for p in req
              if p in probes and probes[p].get("pass") is False]
    if failed:
        verdict: bool | None = False
    elif missing or inconclusive:
        verdict = None
    else:
        verdict = True
    return {
        "stage": stage,
        "pass": verdict,
        "verdict": ("PASS" if verdict is True else
                    "FAIL" if verdict is False else "INCONCLUSIVE"),
        "required": list(req),
        "reported_only": list(spec["reported"]),
        "criteria": spec["criteria"],
        "owners": spec["owners"],
        "probes": probes,
        "missing_required": missing,
        "inconclusive_required": inconclusive,
        "failed_required": failed,
        "next_stage": next((s for s, p in STAGE_PRECONDITION.items()
                            if p == stage), None),
        "revalidates": {
            "stages": list(STAGE_INVALIDATES.get(stage, ())),
            "mechanism": STAGE_INVALIDATION_MECHANISM.get(stage),
            "note": "certificates this stage INVALIDATES by construction. The "
                    "re-measurements are in `required` above, so omitting them "
                    "reads INCONCLUSIVE, never PASS.",
        } if STAGE_INVALIDATES.get(stage) else None,
        "bound_outcomes": {
            "PASS": f"{stage} propagates upward; the next stage may launch",
            "FAIL": "the next stage MUST NOT launch (X5). Diagnose at THIS "
                    "layer; a failed stage never propagates upward.",
            "INCONCLUSIVE": "treated as NOT-PASS. Run the missing probes, or "
                            "override with --allow-inconclusive-gate AND "
                            "--gate-off-reason (the reason is recorded).",
        },
        "tier": "gate assembled from frozen-battery probes (T0/T1 per probe)",
        "_evidence_class": "MEASURED (ours) for probes present; probes listed "
                           "in missing_required were NOT RUN",
    }


def write_stage_gate(out_dir, gate: dict) -> Path:
    p = Path(out_dir) / "stage_gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gate, indent=1))
    return p


def assert_stage_precondition(stage: str, prev_gate_path=None, *,
                              allow_inconclusive: bool = False,
                              off_reason: str = "") -> dict:
    """REFUSE to start ``stage`` unless the stage below it PASSED (X5).

    Returns the precondition report on success; raises
    :class:`GatePreconditionError` otherwise. Three refusals, all deliberate:
      * the previous gate file is MISSING -> refuse (a stage that never ran a
        gate did not pass one);
      * ``pass: false`` -> refuse, and no flag overrides it. A FAIL is a
        finding about the layer below; propagating it upward is how a defect
        gets attributed to the wrong layer three stages later;
      * ``pass: null`` (INCONCLUSIVE) -> refuse UNLESS
        ``allow_inconclusive`` AND a non-empty ``off_reason``.
    """
    prev = STAGE_PRECONDITION.get(stage)
    if prev is None:
        return {"stage": stage, "precondition": None,
                "ok": True, "reason": "S-W starts the ladder"}
    if prev_gate_path is None:
        raise GatePreconditionError(
            f"[v6] ⛔ stage {stage} requires {prev}'s gate, but no "
            f"--prev-gate was given. X5: a stage is gated by the frozen "
            f"battery BEFORE the next begins.")
    p = Path(prev_gate_path)
    if not p.exists():
        raise GatePreconditionError(
            f"[v6] ⛔ stage {stage} requires {prev}'s gate at {p} — the file "
            f"does not exist. A stage that never ran a gate did not pass one.")
    gate = json.loads(p.read_text())
    if gate.get("stage") != prev:
        raise GatePreconditionError(
            f"[v6] ⛔ {p} is the gate for stage {gate.get('stage')!r}, but "
            f"{stage} requires {prev!r}. Pointing a stage at the wrong gate "
            f"file is not a pass.")
    verdict = gate.get("pass")
    if verdict is False:
        raise GatePreconditionError(
            f"[v6] ⛔ stage {prev} FAILED its gate "
            f"(failed_required={gate.get('failed_required')}) — {stage} MUST "
            f"NOT launch. X5: a failed stage never propagates upward. There "
            f"is no override for a FAIL; fix the layer below.")
    if verdict is None:
        if not (allow_inconclusive and off_reason.strip()):
            raise GatePreconditionError(
                f"[v6] ⛔ stage {prev}'s gate is INCONCLUSIVE "
                f"(missing={gate.get('missing_required')}, "
                f"inconclusive={gate.get('inconclusive_required')}). "
                f"INCONCLUSIVE IS NOT A PASS. Run the missing probes, or pass "
                f"--allow-inconclusive-gate --gate-off-reason '<why>'.")
        return {"stage": stage, "precondition": prev, "ok": True,
                "prev_verdict": "INCONCLUSIVE",
                "override": "allow-inconclusive-gate",
                "off_reason": off_reason, "prev_gate": str(p)}
    return {"stage": stage, "precondition": prev, "ok": True,
            "prev_verdict": "PASS", "prev_gate": str(p)}


def run_stage_gate(stack: V6Stack, stage: str, *, out_dir,
                   spectrum: dict | None = None,
                   extra_probes: dict | None = None) -> dict:
    """Run whatever frozen-battery entry points are IMPORTABLE here, assemble
    the gate, and write it.

    ⚠️ Rule 2 (*absence found at ONE location is not absence*) applied to the
    battery: a probe that cannot be imported is recorded with the ImportError
    text and the owning path from :data:`STAGE_GATE_SPEC`, so "n/a" always says
    WHAT was not reachable and WHERE it lives. It is never silently dropped,
    and it never counts as a pass.
    """
    probes: dict[str, dict] = dict(extra_probes or {})
    spec = STAGE_GATE_SPEC[stage]
    for name in tuple(spec["required"]) + tuple(spec["reported"]):
        if name in probes:
            continue
        owner = spec["owners"].get(name, "?")
        probes[name] = {"pass": None, "status": "not-run",
                        "owner": owner,
                        "reason": "no artifact supplied to --gate-probes and "
                                  "this trainer does not run the battery "
                                  "in-loop (it is a separate, frozen "
                                  "instrument by design)"}
    # X3 is the one gate this module CAN measure on its own, always.
    try:
        iso = stack.assert_isolation(batch_size=1, strict=False)
        probes["X3_isolation"] = {"pass": bool(iso["pass"]), "status": "run",
                                  "owner": "V6Stack.assert_isolation",
                                  "n_violations": iso["n_violations"],
                                  "violations": iso["violations"]}
    except Exception as exc:                                # pragma: no cover
        probes["X3_isolation"] = {"pass": None, "status": "error",
                                  "reason": f"{type(exc).__name__}: {exc}"}
    if spectrum is not None:
        probes["O6_spectrum"] = {"pass": None, "status": "reported",
                                 "owner": "tanitad.models.v6.spectrum_report",
                                 **spectrum,
                                 "reason": "rank RETENTION needs a series "
                                           "(>= 0.8x across phases); a single "
                                           "reading cannot pass or fail it"}
    gate = stage_gate_dict(stage, probes)
    gate["param_report"] = stack.param_report()
    path = write_stage_gate(out_dir, gate)
    print(f"[v6] stage gate {gate['verdict']} -> {path}", flush=True)
    return gate


# ============================================================================
# building the stack from args
# ============================================================================

def resolve_gc(a, field: str) -> bool:
    """Resolve a per-site grad-checkpoint override against the master flag.

    ``auto`` (the default) follows ``--grad-checkpoint``, so every launch
    command written before the split behaves byte-identically. ``on``/``off``
    decide that one site independently.
    """
    v = getattr(a, field, "auto")
    if v in (None, "auto"):
        return bool(getattr(a, "grad_checkpoint", False))
    return v == "on"


def build_stack_from_args(a) -> V6Stack:
    """Instantiate :class:`V6Stack` from the CLI, enforcing the sub-300M
    invariant AND the X3 matrix BEFORE any GPU time is spent. Both refusals are
    pre-launch on purpose: an over-budget or mis-wired model discovered at hour
    six of a run is a wasted GPU-day."""
    enc = EncoderConfig(in_channels=a.in_channels, image_size=a.frame_h,
                        image_width=a.frame_w, patch_size=a.patch,
                        d_model=a.enc_dim, depth=a.enc_depth,
                        n_heads=a.enc_heads,
                        grad_checkpoint=resolve_gc(a, "enc_grad_checkpoint"))
    ro = ReadoutConfig(grid=a.readout_grid, d_readout=a.readout_dim,
                       grid_w=a.readout_grid_w)
    pr = PredictorConfig(d_model=a.pred_dim, depth=a.pred_depth,
                         n_heads=a.pred_heads, window=a.window,
                         horizons=tuple(a.horizons), action_dim=3,
                         residual=True, modern=bool(a.pred_modern))
    cfg = V6Config(
        encoder=enc, readout=ro, predictor=pr,
        d_tac=a.d_tac, d_str=a.d_str, d_goal_embed=a.d_goal_embed,
        shared_encoder=not a.per_layer_encoders,
        adapter_hidden=a.adapter_hidden,
        plan_steps=a.plan_steps, dt=a.dt, n_candidates=a.n_candidates,
        a_max=a.a_max, kappa_max=a.kappa_max,
        isolate_planner_from_encoder=not a.no_isolate_planner,
        isolate_uplink=not a.no_isolate_uplink, uplink=a.uplink,
        ema_decay=a.ema_decay, sigreg_slices=a.sigreg_slices,
        sigreg_free_dims=a.sigreg_free_dims, param_budget=a.param_budget,
        f_hidden_tac=a.f_hidden_tac, f_hidden_str=a.f_hidden_str,
        f_blocks=a.f_blocks,
        selector=a.selector, selector_tau_m=a.selector_tau_m,
        selector_mlp_hidden=getattr(a, "selector_mlp_hidden", 256),
        plan_wta_eps=a.plan_wta_eps,
        vit5_encoder=bool(a.vit5_encoder), n_registers=a.n_registers)
    stack = V6Stack(cfg)
    rep = stack.assert_param_budget()
    print(f"[v6] params {rep['total']/1e6:.2f} M / budget "
          f"{rep['budget']/1e6:.0f} M · arm {rep['arm']} · per-group "
          f"{ {k: round(v/1e6, 2) for k, v in rep['per_group'].items()} }",
          flush=True)
    iso = stack.assert_isolation(batch_size=1,
                                 strict=not a.no_isolate_planner
                                 and not a.no_isolate_uplink)
    print(f"[v6] X3 isolation pass={iso['pass']} "
          f"violations={iso['n_violations']}", flush=True)
    return stack


# ============================================================================
# --dry-run: build everything, 2 synthetic CPU steps, write the config
# ============================================================================

def synthetic_train_batch(stack: V6Stack, *, batch: int = 2, k: int = 12,
                          seed: int = 0, device=None) -> dict:
    """A fully-shaped random batch — the ``--dry-run`` corpus stand-in.

    Everything the real loader supplies is here with the real shapes, so a
    dry-run exercises the SAME code path the pod will: the same forward, the
    same losses, the same optimiser step. A smoke that skips the loss assembly
    proves the model imports, not that the run will start.
    """
    cfg = stack.cfg
    g = torch.Generator().manual_seed(seed)
    c = cfg.encoder.in_channels
    h, w = cfg.encoder.image_hw()
    b = int(batch)
    out = {
        "frames": torch.randn(b, cfg.predictor.window, c, h, w, generator=g),
        "actions2": torch.randn(b, cfg.predictor.window, 2, generator=g) * 0.1,
        "future_actions2": torch.randn(b, max(k, 2), 2, generator=g) * 0.1,
        "v0": torch.rand(b, generator=g) * 20.0 + 1.0,
        "z_true_steps": [torch.randn(b, cfg.d_op, generator=g)
                         for _ in range(k)],
        "plan_target": torch.randn(b, cfg.plan_steps, 2, generator=g),
        "z_tac_next_target": torch.randn(b, cfg.d_tac, generator=g),
        "z_str_next_target": torch.randn(b, cfg.d_str, generator=g),
    }
    if not cfg.shared_encoder:
        out["own_frames_tac"] = torch.randn(b, c, h, w, generator=g)
        out["own_frames_str"] = torch.randn(b, c, h, w, generator=g)
    if device is not None:
        out = {kk: ([t.to(device) for t in v] if isinstance(v, list)
                    else v.to(device)) for kk, v in out.items()}
    return out


def dry_run(a, stack: V6Stack | None = None) -> dict:
    """Build everything, run ``--dry-steps`` (default 2) synthetic CPU steps,
    write ``config.json`` + ``dry_run.json``.

    THE POINT: verify the pod launch BEFORE the corpus is mounted. On a pod the
    checkout drifts, ``git fetch`` HANGS (no credentials), and a launch from a
    stale ``stack/`` resurrects fixed bugs — so the runbook step is *ship the
    files, then run this, then launch*, and this must exercise the real loss
    assembly, not just an import.
    """
    stack = stack or build_stack_from_args(a)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    stack = stack.to(device)
    freeze = apply_stage_freeze(stack, a.stage)
    weights = _weights_from_args(a)
    o1_k = min(a.o1_k, a.dry_k)
    o5_k = min(a.o5_k, a.dry_k)
    trainable = [p for p in stack.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)
           if trainable else None)
    gen = torch.Generator().manual_seed(a.seed)
    rows: list[dict] = []
    t0 = time.time()
    for step in range(1, int(a.dry_steps) + 1):
        b = synthetic_train_batch(stack, batch=a.dry_batch, k=a.dry_k,
                                  seed=a.seed + step, device=device)
        b["gt_wp"] = torch.randn(a.dry_batch, o1_k, 2, generator=gen)
        dk, da = sample_random_deltas(a.dry_batch, gen, a.rand_dkappa_max,
                                      a.rand_daccel_max)
        L = v6_loss_step(stack, b, stage=a.stage, weights=weights, o1_k=o1_k,
                         o5_k=o5_k, o5_mode=a.o5_mode, o3_mode=a.o3_mode,
                         o3_blocks=a.o3_blocks,
                         o3_block_hw=(a.o3_block_h, a.o3_block_w),
                         o3_band_rows=a.o3_band_rows, o2_tau_s=a.o2_tau_s,
                         dkappa=a.dkappa, daccel=a.daccel, rand_dk=dk,
                         rand_da=da, generator=gen)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, a.clip))
            opt.step()
            stack.ema_update()
        else:
            gn = 0.0
        row = L["log"] | {"step": step, "gnorm": round(gn, 4)}
        rows.append(row)
        print(f"[v6 dry {step}] {json.dumps(row)}", flush=True)
    spec = spectrum_report(torch.randn(64, min(stack.cfg.d_op, 64)))
    iso = stack.assert_isolation(batch_size=1, strict=False)
    cfg_json = _run_config(a, stack, freeze)
    (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=1))
    result = {
        "mode": "dry-run", "device": device, "steps": rows,
        "elapsed_s": round(time.time() - t0, 2),
        "freeze": freeze, "isolation": iso, "spectrum_smoke": spec,
        "param_report": stack.param_report(),
        "n_trainable_tensors": len(trainable),
        "_read": "synthetic tensors — NO corpus, NO checkpoint. This proves "
                 "the launch assembles and steps; it proves NOTHING about "
                 "driving. No number here is quotable.",
        "_evidence_class": "MEASURED (ours; synthetic smoke)",
    }
    (out_dir / "dry_run.json").write_text(json.dumps(result, indent=1))
    print(f"[v6] dry-run OK -> {out_dir}/dry_run.json + config.json",
          flush=True)
    return result


def resolve_lambda_plan(a) -> float:
    """``--lambda-plan`` if given, else the stage default. Explicit beats
    implicit, and the resolved value is written into ``config.json`` so a run
    row never has to guess which λ_plan was in force."""
    return (STAGE_LAMBDA_PLAN[a.stage] if a.lambda_plan is None
            else float(a.lambda_plan))


def _weights_from_args(a) -> V6LossWeights:
    return V6LossWeights(
        o1_ctrl=a.w_o1_ctrl, o1_fact=a.w_o1_fact, o1_scene=a.w_o1_scene,
        o2_nearfield=a.w_o2, o3_masked=a.w_o3, o5_rollout=a.w_o5,
        o6_sigreg=a.w_o6, t1_latent=a.w_t1, s1_latent=a.w_s1,
        w_select=a.w_select,
        lambda_plan=resolve_lambda_plan(a))


def _run_config(a, stack: V6Stack, freeze: dict) -> dict:
    return {
        "run": f"v6-staged-{a.stage}",
        "stage": a.stage,
        "trainable_groups": list(stage_trainable_groups(a.stage)),
        "module_groups": list(MODULE_GROUPS),
        "v6_config": stack.cfg.to_dict(),
        "loss_weights": asdict(_weights_from_args(a)),
        "loss_weights_in_force": asdict(_weights_from_args(a)
                                        .for_stage(a.stage)),
        "freeze": freeze,
        "param_report": stack.param_report(),
        "args": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in vars(a).items()},
        "horizon_spec": {
            "plan_steps": stack.cfg.plan_steps, "dt": stack.cfg.dt,
            "horizon_s": stack.cfg.horizon_s,
            "op_band_s": list(stack.cfg.op_band_s),
            "tac_band_s": list(stack.cfg.tac_band_s),
            "_binding": "HIERARCHY_VOCABULARY.md §4b — ONE 60-step (a, κ)@10 Hz "
                        "unicycle rollout 0→6 s; bands are SLICES of it"},
        "gate_spec": STAGE_GATE_SPEC[a.stage],
        "tier": "training-side config; capability claims are T1 only "
                "(EVAL_DOCTRINE.md)",
        "_evidence_class": "MEASURED (ours; this run's own configuration)",
    }


# ============================================================================
# the real training path (pod-side: GPU + the canonical v2 corpora)
# ============================================================================

def train(a) -> dict:
    """The real staged loop. POD-SIDE: needs the v2 caches and a GPU.

    Data seams are the PROVEN ones, imported not re-derived:
    ``train_v58f_unicycle_head.build_train_episodes`` (parity guard + geometry
    binding), ``eval_flagship_v4.build_v2_val_episodes``/``resolve_eval_frames``,
    ``train_flagship4b.FlagshipWindowDataset``. Parity is sacred: anything that
    re-selects episodes breaks cross-arm comparability and is refused by
    ``assert_v2_parity_cache``.
    """
    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  resolve_eval_frames)
    from train_flagship4b import FlagshipWindowDataset
    from train_flagship_v4 import _to_device
    from train_v58f_unicycle_head import build_train_episodes
    from tanitad.models.metric_dynamics import gt_ego_waypoints

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[v6] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp

    # ---- X5 precondition BEFORE anything expensive -------------------------
    pre = assert_stage_precondition(
        a.stage, a.prev_gate, allow_inconclusive=a.allow_inconclusive_gate,
        off_reason=a.gate_off_reason)
    print(f"[v6] precondition OK: {json.dumps(pre)}", flush=True)
    if pre.get("override"):
        print("=" * 72, flush=True)
        print(f"[v6] ⚠️  GATE OVERRIDE IN FORCE — previous stage verdict "
              f"{pre['prev_verdict']}; reason: {pre['off_reason']}",
              flush=True)
        print("=" * 72, flush=True)

    # ---- resume / done-marker discipline BEFORE anything expensive --------
    rg = resume_guard(out_dir, resume=a.resume, force_rerun=a.force_rerun)
    print(f"[v6] launch mode: {json.dumps(rg)}", flush=True)

    # ---- the val corpus is for probing, never training ---------------------
    overlap = (set(map(os.path.abspath, a.v2_cache))
               & set(map(os.path.abspath, a.v2_val_cache or [])))
    if overlap:
        raise SystemExit(f"[v6] ⛔ --v2-cache and --v2-val-cache share dirs "
                         f"{sorted(overlap)} — val windows in training are "
                         f"not allowed")

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    gen = torch.Generator().manual_seed(a.seed + 1)

    stack = build_stack_from_args(a)
    init_report: dict = {"init_from": None}
    if a.init_from:
        init_report = load_stage_init(stack, a.init_from, stage=a.stage)
        print(f"[v6] initialised from {json.dumps(init_report)}", flush=True)
    stack = stack.to(device)
    freeze = apply_stage_freeze(stack, a.stage)
    print(f"[v6] stage {a.stage}: trainable "
          f"{freeze['n_trainable']/1e6:.2f} M / frozen "
          f"{freeze['n_frozen']/1e6:.2f} M · groups "
          f"{freeze['trainable_groups']}", flush=True)

    cfg_eval = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg_eval,
                                                   label="train_v6_staged")
    plan = _plan(cfg_eval)
    weights = _weights_from_args(a)
    w_stage = weights.for_stage(a.stage)
    # each layer's TARGET lives one of ITS OWN ticks ahead (stride_tac = 5,
    # stride_str = 20 at the default clocks) — predicting one operative tick
    # ahead and calling it a tactical prediction is an identity map wearing a
    # hierarchy's name, so the corpus window has to actually carry those steps.
    # ⚠️ only the stages that actually run an O-layer term need the encoded
    # future latents, and encoding them is the single largest per-step cost.
    # S-T/S-S would otherwise pay max(o1_k, o5_k) = 20 extra encoder passes per
    # sample for tensors no live loss reads.
    needs_ztrue = bool(w_stage.o1_ctrl or w_stage.o1_fact or w_stage.o1_scene
                       or w_stage.o2_nearfield or w_stage.o3_masked
                       or w_stage.o5_rollout)
    need_k = max(a.o1_k, a.o5_k) if needs_ztrue else 0
    # the seam loss needs exactly ONE encoded future latent — the cheapest
    # possible future encode, and only in the stages where the seam trains
    if w_stage.seam_op and need_k < 1:
        need_k = 1
    need = max(need_k,
               stack.cfg.stride_tac if w_stage.t1_latent else 0,
               stack.cfg.stride_str if w_stage.s1_latent else 0,
               stack.cfg.plan_steps if w_stage.lambda_plan else 0,
               1)
    # ⚠️ ``plan.max_horizon`` is the FLAGSHIP-v4 loss's horizon (MEASURED 20 at
    # the current config) — it is NOT a property of the cache. ``max_horizon``
    # is a WINDOWING parameter (data/_contract.py:118-121:
    # ``t_max = frames - window - max_horizon``), so v6 sets its OWN. Inheriting
    # v4's 20 would make §4b's 6 s horizon structurally untrainable while
    # looking like a corpus limitation.
    max_h = int(a.max_horizon) if a.max_horizon else max(need,
                                                         plan.maneuver_h)
    if max_h < need:
        raise SystemExit(
            f"[v6] ⛔ --max-horizon {max_h} < the {need} future steps stage "
            f"{a.stage} needs (o1_k={a.o1_k}, o5_k={a.o5_k}, stride_tac="
            f"{stack.cfg.stride_tac}, stride_str={stack.cfg.stride_str}, "
            f"plan_steps={stack.cfg.plan_steps}). A silently shortened horizon "
            f"is not the same experiment.")
    if max_h < plan.maneuver_h:
        raise SystemExit(f"[v6] ⛔ --max-horizon {max_h} < maneuver_h "
                         f"{plan.maneuver_h} (the dataset asserts this)")
    print(f"[v6] windowing: window {stack.cfg.predictor.window} + "
          f"max_horizon {max_h} (v4's plan says {plan.max_horizon}; v6 sets "
          f"its own). ⚠️ a LONGER horizon yields FEWER windows per episode — "
          f"episode selection (parity) is untouched, the window count is not.",
          flush=True)

    train_eps, _tp = build_train_episodes(a, cache_frame=cache_frame,
                                          train_frame=model_frame)
    ds_train = FlagshipWindowDataset(
        train_eps, window=stack.cfg.predictor.window, max_horizon=max_h,
        maneuver_h=plan.maneuver_h,
        channels=stack.cfg.encoder.in_channels)
    print(f"[v6] train {len(train_eps)} eps / {len(ds_train)} windows",
          flush=True)
    if not len(ds_train):
        raise SystemExit(
            f"[v6] ⛔ 0 training windows at window "
            f"{stack.cfg.predictor.window} + max_horizon {max_h} — the "
            f"episodes are shorter than {stack.cfg.predictor.window + max_h} "
            f"frames. Lower the horizons or rebuild the cache with longer "
            f"episodes.")
    if a.v2_val_cache:
        val_eps, _vp = build_v2_val_episodes(a, cache_frame=cache_frame,
                                             train_frame=model_frame)
        ds_val = FlagshipWindowDataset(
            val_eps, window=stack.cfg.predictor.window, max_horizon=max_h,
            maneuver_h=plan.maneuver_h,
            channels=stack.cfg.encoder.in_channels)
        print(f"[v6] val {len(val_eps)} eps / {len(ds_val)} windows",
              flush=True)

    # ---- O4: interaction-weighted sampling (ACTIONS ONLY) ------------------
    if a.o4_alpha > 0:
        print(f"[v6] O4: scoring {len(ds_train)} windows by ego-kinematic "
              f"saliency (label-free, ACTIONS ONLY) ...", flush=True)
        # ⚠️ read the ACTION arrays straight off the episode providers
        # (``ep.actions[t : t+W+H]`` — the ``EpisodeWindowDataset`` slicing at
        # data/_contract.py:132-135). Going through ``ds_train[i]`` would DECODE
        # EVERY FRAME OF THE CORPUS to compute a scalar over two action
        # channels: hundreds of thousands of window payload loads on MooseFS
        # before step 1. The saliency needs no pixels by construction, and that
        # is exactly what makes O4 label-free in the first place.
        w_win = stack.cfg.predictor.window
        span = w_win + need
        acts = []
        for e_i, t in ds_train.index:
            arr = ds_train.episodes[e_i].actions[t:t + span]
            acts.append(torch.as_tensor(arr[:, :2]).float())
        # episodes near their end yield short slices — pad by edge-repeat so the
        # stack is rectangular. Edge-repeat adds ZERO jerk and ZERO reversals,
        # so a truncated window is scored on what it actually contains and is
        # never inflated by the padding.
        n_max = max(x.shape[0] for x in acts)
        acts = [x if x.shape[0] == n_max else
                torch.cat([x, x[-1:].expand(n_max - x.shape[0], 2)], dim=0)
                for x in acts]
        w4, o4log = build_o4_weights(torch.stack(acts), dt=stack.cfg.dt,
                                     alpha=a.o4_alpha, floor=a.o4_floor)
        o4log["o4_span_steps"] = int(n_max)
        print(f"[v6] O4 {json.dumps(o4log)}", flush=True)
        sample = InteractionSampler(ds_train.index, w4,
                                    eps_per_batch=a.eps_per_batch,
                                    generator=gen)
    else:
        from train_v58f_unicycle_head import make_sampler
        sample = make_sampler(ds_train, a.eps_per_batch, rng)
        o4log = {"o4_alpha": 0.0, "_note": "uniform sampling (O4 control arm)"}

    trainable = [p for p in stack.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit(f"[v6] ⛔ stage {a.stage} has NO trainable parameters "
                         f"— the freeze map and the stage disagree")
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    start_step = 0
    if rg["mode"] == "resume":
        start_step = load_resume(stack, opt, rg["from"])
        for _ in range(start_step):
            sched.step()                       # replay the LR schedule
        print(f"[v6] RESUMED at step {start_step} from {rg['from']} — "
              f"{a.steps - start_step} steps remain", flush=True)
        if start_step >= a.steps:
            raise SystemExit(
                f"[v6] ⛔ the checkpoint is already at step {start_step} >= "
                f"--steps {a.steps}. Nothing to do. If this run finished, its "
                f"summary.json should say so; write it or raise --steps.")

    cfg_json = _run_config(a, stack, freeze) | {"o4": o4log,
                                                "precondition": pre,
                                                "init": init_report,
                                                "max_horizon": max_h,
                                                "launch_mode": rg}
    (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=1))
    log_path = out_dir / "train_log.jsonl"
    fh = open(log_path, "a")
    fh.write(json.dumps({"run_start": cfg_json}) + "\n")
    fh.flush()

    history: list[dict] = []
    spectrum_last: dict | None = None
    t0 = time.time()
    steps_g = tuple(range(1, a.o1_k + 1))
    dev_type = "cuda" if device == "cuda" else "cpu"
    for step in range(start_step + 1, a.steps + 1):
        idx = sample(a.batch)
        b = _to_device(default_collate([ds_train[i] for i in idx]), device)
        aw2 = b["actions"][..., :2].float()
        fa2 = b["future_actions"][..., :2].float()
        v0 = b["pose_last"][:, 3].float()
        z_true: list = []
        if need_k:
            with torch.no_grad():
                # ONE encoder pass over all need_k future frames, not need_k
                # passes — the future-frame encode is the single largest
                # per-step cost in S-W (26 frames/sample at the defaults vs
                # v1's ~8), and a Python loop over it wastes the batch
                # dimension the GPU is built for.
                ff = b["future_frames"][:, :need_k]
                fb, fk = ff.shape[:2]
                z_flat = stack.readout(stack.encoder(
                    ff.reshape(fb * fk, *ff.shape[2:]))).reshape(fb, fk, -1)
                z_true = [z_flat[:, j].detach() for j in range(need_k)]
        dk, da = sample_random_deltas(aw2.shape[0], gen, a.rand_dkappa_max,
                                      a.rand_daccel_max)
        batch = {
            "frames": b["frames"], "actions2": aw2, "future_actions2": fa2,
            "v0": v0, "z_true_steps": z_true,
        }
        if needs_ztrue:
            batch["gt_wp"] = gt_ego_waypoints(b["pose_last"].float(),
                                              b["future_poses"].float(),
                                              steps_g)
        if not stack.cfg.shared_encoder:
            # E-ENC arm (b): each layer encodes the CURRENT frame with its own
            # encoder. ⚠️ The clock difference lives in the TARGETS (each layer
            # predicts one of ITS OWN ticks ahead), not in which frame each
            # encoder sees now — "now" is the same instant for all three
            # layers, and pretending otherwise would silently shift the
            # layers' observation times relative to each other.
            batch["own_frames_tac"] = b["frames"][:, -1]
            batch["own_frames_str"] = b["frames"][:, -1]
        # each higher layer's target sits ONE OF ITS OWN TICKS ahead
        if w_stage.t1_latent or w_stage.s1_latent:
            shared = stack.cfg.shared_encoder
            with torch.no_grad():
                for key, stride, want in (
                        ("z_tac_next_target", stack.cfg.stride_tac,
                         w_stage.t1_latent),
                        ("z_str_next_target", stack.cfg.stride_str,
                         w_stage.s1_latent)):
                    if not want:
                        continue
                    fut = b["future_frames"][:, stride - 1]
                    zf = stack.readout(stack.encoder(fut))
                    o_t = None if shared else stack.readout_tac(
                        stack.encoder_tac(fut))
                    o_s = None if shared else stack.readout_str(
                        stack.encoder_str(fut))
                    tgt = stack.layer_targets(zf, o_t, o_s)
                    batch[key] = tgt["z_tac" if key.startswith("z_tac")
                                     else "z_str"]
        if w_stage.lambda_plan:
            batch["plan_target"] = gt_ego_waypoints(
                b["pose_last"].float(), b["future_poses"].float(),
                tuple(range(1, stack.cfg.plan_steps + 1)))
        with torch.autocast(dev_type, dtype=torch.bfloat16,
                            enabled=amp_on and dev_type == "cuda"):
            L = v6_loss_step(stack, batch, stage=a.stage, weights=weights,
                             o1_k=a.o1_k, o5_k=a.o5_k, o5_mode=a.o5_mode,
                             o3_mode=a.o3_mode, o3_blocks=a.o3_blocks,
                             o3_block_hw=(a.o3_block_h, a.o3_block_w),
                             o3_band_rows=a.o3_band_rows,
                             o2_tau_s=a.o2_tau_s, dkappa=a.dkappa,
                             daccel=a.daccel, rand_dk=dk.to(device),
                             rand_da=da.to(device), generator=gen,
                             rollout_grad_checkpoint=resolve_gc(
                                 a, "rollout_grad_checkpoint"))
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.clip)
        opt.step()
        sched.step()
        stack.ema_update()

        if step % a.spectrum_every == 0:
            # O6's standing monitor — a SERIES, because retention is a RATIO
            # and a single reading can neither pass nor fail it. Measured on
            # the SAME tensor SIGReg acts on ([B*W, d_op], not just the last
            # frame): B rows would estimate a 2048-dim spectrum from 16
            # samples, which is a number, not a measurement.
            zw = L["out"]["z_op_win"].detach().float()
            spectrum_last = spectrum_report(zw.reshape(-1, zw.shape[-1]))
            fh.write(json.dumps({"step": step,
                                 "spectrum": spectrum_last}) + "\n")
        if step % a.log_every == 0:
            rec = L["log"] | {
                "step": step, "gnorm": round(float(gn), 3),
                "lr": sched.get_last_lr()[0],
                # ⚠️ ALREADY DIVIDED by --log-every. The trap this avoids:
                # trainer logs that accumulate step_s over the log interval and
                # get read as a per-step time (the false "430 s/step" alarm).
                "step_s": round((time.time() - t0)
                                / max(step - start_step, 1), 4),
                "step_s_note": f"elapsed/step over the "
                               f"{step - start_step} steps THIS process ran "
                               f"(NOT accumulated over --log-every, and NOT "
                               f"divided by the resumed step number)"}
            history.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {json.dumps(rec)}", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            _save_ckpt(out_dir / "ckpt.pt", stack=stack, opt=opt, step=step,
                       cfg_json=cfg_json)
            (out_dir / "metrics.json").write_text(json.dumps(
                {"history": history, "stage": a.stage,
                 "_read": "TRAINING numbers. Only eval output is quotable "
                          "(the v1.6 retraction); capability claims are T1.",
                 "_evidence_class": "MEASURED (ours; this run's log)"},
                indent=1))
    fh.close()

    gate = run_stage_gate(stack, a.stage, out_dir=out_dir,
                          spectrum=spectrum_last,
                          extra_probes=_load_gate_probes(a.gate_probes))
    # ⛔ DONE-MARKER, written in the SAME turn the run finishes. A supervised
    # run whose summary.json never appeared kept being RESURRECTED for two
    # days; writing this IS the correct remote off-switch.
    summary = {
        "done": True, "run": f"v6-staged-{a.stage}", "stage": a.stage,
        "steps": a.steps, "resumed_from_step": start_step,
        "out": str(out_dir),
        "gate_verdict": gate["verdict"], "gate": "stage_gate.json",
        "elapsed_s": round(time.time() - t0, 1),
        "param_report": stack.param_report(),
        "next": (f"stage {gate['next_stage']} may launch with --prev-gate "
                 f"{out_dir}/stage_gate.json"
                 if gate["pass"] is True and gate["next_stage"]
                 else "NEXT STAGE BLOCKED — see stage_gate.json"),
        "_evidence_class": "MEASURED (ours)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[v6] DONE · {json.dumps(summary)}", flush=True)
    return summary


def _load_gate_probes(path) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[v6] --gate-probes {p} does not exist")
    return json.loads(p.read_text())


def _save_ckpt(path: Path, *, stack, opt, step: int, cfg_json: dict) -> None:
    torch.save({"stack": stack.state_dict(), "opt": opt.state_dict(),
                "step": step, "config": cfg_json}, path)


def resume_guard(out_dir, *, resume: str, force_rerun: bool) -> dict:
    """Decide whether this launch RESUMES, STARTS FRESH, or is REFUSED.

    ⛔ Two refusals, and both come straight from measured incidents:

    1. **A finished run must not be relaunched.** MEASURED 2026-08-09/11: the
       v5f run completed but never wrote its done-marker; its supervisor kept
       relaunching for two days, and the moment the crash-cause was fixed a
       relaunch SUCCEEDED, resumed from a stale ``ckpt.pt``, and began
       overwriting ``config.json``/``metrics.json``/``ckpt.pt`` in the
       canonical run directory while burning GPU next to a live eval. This
       trainer writes ``summary.json {"done": true}`` in the same turn it
       finishes, and refuses to start where one already exists. **That file is
       the off-switch**; ``--force-rerun`` is the only way past it.
    2. **A fresh start must not silently overwrite a live checkpoint.**
       ``supervise_run.sh`` replays the ``TRAIN_CMD`` it captured at supervisor
       startup, so a relaunch runs the SAME command — with ``--resume off``
       that would restart at step 0 on top of an existing ``ckpt.pt``. Refused
       unless ``--force-rerun`` says so out loud.
    """
    out = Path(out_dir)
    ck, done = out / "ckpt.pt", out / "summary.json"
    if done.exists() and not force_rerun:
        try:
            marker = json.loads(done.read_text())
        except Exception:
            marker = {}
        if marker.get("done") is True:
            raise SystemExit(
                f"[v6] ⛔ {done} says this run is DONE "
                f"(stage {marker.get('stage')}, {marker.get('steps')} steps, "
                f"gate {marker.get('gate_verdict')}). Refusing to relaunch: a "
                f"finished run that gets relaunched resumes from a stale "
                f"ckpt.pt and overwrites the canonical run directory. Pass "
                f"--force-rerun ONLY if you mean to discard that run, or point "
                f"--out somewhere else.")
    if resume == "auto" and ck.exists():
        return {"mode": "resume", "from": str(ck)}
    if resume == "off" and ck.exists() and not force_rerun:
        raise SystemExit(
            f"[v6] ⛔ --resume off with an existing {ck}. A supervisor replays "
            f"the command it captured at startup, so this would restart at "
            f"step 0 ON TOP of a live checkpoint. Use --resume auto, point "
            f"--out elsewhere, or say --force-rerun.")
    return {"mode": "fresh", "from": None}


def load_resume(stack: V6Stack, opt, ckpt_path) -> int:
    """Restore stack + optimiser + step from ``ckpt.pt``. Returns the step to
    continue FROM (0 if nothing to resume)."""
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    stack.load_state_dict(ck["stack"], strict=True)
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    return int(ck.get("step", 0))


def load_stage_init(stack: V6Stack, ckpt_path, *, strict: bool = True,
                    stage: str | None = None) -> dict:
    """Initialise stage N+1 from stage N's checkpoint — the OTHER half of X5.

    A gate that says "S-W passed" is worthless if S-T then starts from random
    weights: the staged protocol's whole claim is that each layer trains ON the
    one below. This loads the full ``V6Stack`` state_dict (every stage saves the
    WHOLE stack, so the ladder is one lineage, not four unrelated models).

    ``strict=True`` by default. A key mismatch means the two stages were built
    with DIFFERENT geometry — silently allowing that is how a stage ends up
    training on a randomly-initialised trunk while its log looks healthy. The
    returned report carries the md5 of the loaded trunk so the run row can name
    exactly which S-W it stands on.
    """
    p = Path(ckpt_path)
    if not p.exists():
        raise SystemExit(f"[v6] ⛔ --init-from {p} does not exist")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("stack", ck)

    # ⚠️ ALWAYS load non-strict, then adjudicate. Under torch's ``strict=True``
    # a mismatch RAISES, so the ``missing_keys`` / ``unexpected_keys`` this
    # function reported could only ever be empty — the report was structurally
    # incapable of describing the thing it was named for.
    allowed = STAGE_MAY_INTRODUCE.get(stage, ()) if stage else ()
    missing, unexpected = stack.load_state_dict(sd, strict=False)
    introduced = [k for k in missing
                  if any(k.startswith(a) for a in allowed)]
    fatal = [k for k in missing if k not in set(introduced)]

    # ⛔ An introduction must be WHOLE. If the checkpoint already carries part
    # of an allowed module, the rest going missing is a geometry mismatch
    # wearing an allowance's clothes.
    for a in allowed:
        if any(k.startswith(a) for k in introduced) and \
                any(k.startswith(a) for k in sd):
            fatal += [k for k in introduced if k.startswith(a)]
            introduced = [k for k in introduced if not k.startswith(a)]

    if strict and (fatal or unexpected):
        raise SystemExit(
            f"[v6] ⛔ --init-from {p} is not a valid predecessor for stage "
            f"{stage!r}.\n"
            f"  missing (NOT introducible at this stage): {sorted(fatal)[:8]}\n"
            f"  unexpected (the ckpt has keys this stack does not): "
            f"{sorted(unexpected)[:8]}\n"
            f"  A stage may introduce only {allowed or '()'} — anything else "
            f"missing means the two stages were built with DIFFERENT geometry, "
            f"and loading anyway is how a stage ends up training on a "
            f"randomly-initialised trunk while its log looks healthy.")

    import hashlib
    h = hashlib.md5()
    for n, prm in sorted(stack.named_parameters()):
        if stack.group_of(n) in ("encoder", "readout", "predictor_op"):
            h.update(n.encode())
            h.update(prm.detach().cpu().numpy().tobytes())
    return {"init_from": str(p), "init_step": int(ck.get("step", -1)),
            "missing_keys": sorted(fatal), "unexpected_keys": sorted(unexpected),
            # ⭐ named separately so a run row can never confuse "this stage
            # BUILT a new head" with "this stage failed to load one".
            "introduced_keys": sorted(introduced),
            "introduced_allowance": list(allowed),
            "trunk_md5_after_load": h.hexdigest(),
            "prev_stage": (ck.get("config") or {}).get("stage"),
            "_evidence_class": "MEASURED (ours; md5 over the loaded trunk)"}


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="v6 staged trainer (S-W -> S-T -> S-S -> optional S-J)")
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--out", required=True)
    # ---- staging / gates ---------------------------------------------------
    ap.add_argument("--prev-gate", default=None,
                    help="path to the PREVIOUS stage's stage_gate.json (X5)")
    ap.add_argument("--allow-inconclusive-gate", action="store_true",
                    help="proceed on an INCONCLUSIVE (never a FAILED) previous "
                         "gate; requires --gate-off-reason")
    ap.add_argument("--gate-off-reason", default="",
                    help="why the inconclusive gate is being overridden — "
                         "recorded in config.json and printed as a banner")
    ap.add_argument("--resume", choices=("auto", "off"), default="auto",
                    help="auto = continue from <out>/ckpt.pt when present "
                         "(a supervisor replays its captured command, so a "
                         "relaunch MUST NOT restart at step 0)")
    ap.add_argument("--force-rerun", action="store_true",
                    help="⛔ discard a DONE run or overwrite a live ckpt.pt. "
                         "The done-marker is the remote off-switch; this is "
                         "the only way past it.")
    ap.add_argument("--init-from", default=None,
                    help="previous stage's ckpt.pt — S-T/S-S/S-J MUST start "
                         "from the stage below, or the ladder is four "
                         "unrelated models with a gate between them")
    ap.add_argument("--gate-probes", default=None,
                    help="JSON of externally-run battery probes to fold into "
                         "this stage's gate")
    # ---- model geometry ----------------------------------------------------
    ap.add_argument("--in-channels", type=int, default=9)
    ap.add_argument("--frame-h", type=int, default=256)
    ap.add_argument("--frame-w", type=int, default=640)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--enc-dim", type=int, default=384)
    ap.add_argument("--enc-depth", type=int, default=8)
    ap.add_argument("--enc-heads", type=int, default=6)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="master switch: checkpoint the ENCODER blocks and "
                         "(unless --rollout-grad-checkpoint overrides) the "
                         "k-step rollout")
    # ⛔ These two used to be ONE flag, which coupled unrelated decisions. The
    # k=60 rollout NEEDS checkpointing (it fixed a MEASURED 37.97/44 GiB OOM);
    # the encoder's is a pure speed/memory trade costing ~2x the ViT forward.
    # MEASURED 2026-08-14: S-W sat at 42.8 % mean GPU util with 20 GB free —
    # paying recompute it did not need. `auto` = follow --grad-checkpoint, so
    # every existing launch command behaves byte-identically.
    ap.add_argument("--enc-grad-checkpoint", choices=("auto", "on", "off"),
                    default="auto",
                    help="override checkpointing for the ENCODER only")
    ap.add_argument("--rollout-grad-checkpoint", choices=("auto", "on", "off"),
                    default="auto",
                    help="override checkpointing for the k-step ROLLOUT only; "
                         "turning this off at k=60 restores a measured OOM")
    ap.add_argument("--readout-grid", type=int, default=4)
    ap.add_argument("--readout-grid-w", type=int, default=None)
    ap.add_argument("--readout-dim", type=int, default=128)
    ap.add_argument("--pred-modern", action="store_true",
                    help="ViT-5-recipe predictor blocks (RMSNorm + QK-Norm + "
                         "LayerScale + bias-free attention). CHANGES THE "
                         "STATE-DICT KEYS: a declared arm; strict load refuses "
                         "legacy checkpoints rather than silently mis-mapping.")
    ap.add_argument("--pred-dim", type=int, default=768)
    ap.add_argument("--pred-depth", type=int, default=6)
    ap.add_argument("--pred-heads", type=int, default=12)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--d-tac", type=int, default=512)
    ap.add_argument("--d-str", type=int, default=256)
    ap.add_argument("--d-goal-embed", type=int, default=128)
    ap.add_argument("--adapter-hidden", type=int, default=512)
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--param-budget", type=int, default=300_000_000)
    # ---- SELECTION (V6F_PLANNER_DESIGN.md) — ALL DEFAULT-OFF ---------------
    ap.add_argument("--selector", choices=("none", "goal", "mlp"),
                    default="none",
                    help="'none' (default) builds NO scorer and leaves the "
                         "state_dict byte-identical. 'goal' builds the +267 "
                         "GoalDistanceScorer — the mechanism E-WC measured. "
                         "'mlp' builds the CAPACITY CONTROL: identical inputs, "
                         "no distance prior, ~127x the parameters. Judging a "
                         "'goal' arm without it cannot separate mechanism from "
                         "capacity (V6F_PLANNER_DESIGN.md §5.3).")
    ap.add_argument("--selector-tau-m", type=float, default=1.0,
                    help="selection temperature in metres (goal scorer only)")
    ap.add_argument("--selector-mlp-hidden", type=int, default=256,
                    help="hidden width of the 'mlp' capacity control "
                         "(ignored for every other --selector)")
    ap.add_argument("--plan-wta-eps", type=float, default=0.0,
                    help="epsilon-relaxed WTA: weight on the LOSING candidates' "
                         "mean error. 0.0 (default) is the incumbent pure WTA, "
                         "under which N-1 candidates get ZERO gradient and "
                         "nothing bounds the fan mean.")
    ap.add_argument("--w-select", type=float, default=0.0,
                    help="weight on the softade selection loss. 0.0 = the term "
                         "is absent. ⚠️ in METRES while the other terms are "
                         "not — a declared decision, never a default.")
    # ---- E-ENC arm (§0 Q1) -------------------------------------------------
    ap.add_argument("--f-hidden-tac", type=int, default=512,
                    help="FTac residual-MLP hidden width, tactical layer")
    ap.add_argument("--f-hidden-str", type=int, default=512,
                    help="FTac residual-MLP hidden width, strategic layer")
    ap.add_argument("--f-blocks", type=int, default=3,
                    help="FTac residual blocks per layer predictor")
    ap.add_argument("--vit5-encoder", action="store_true",
                    help="ViT-5 recipe encoder (arXiv 2602.08071): RMSNorm + "
                         "LayerScale + QK-Norm + register tokens + joint "
                         "APE/2D-axial-RoPE, GeLU MLP (ViT-5 REJECTS SwiGLU). "
                         "CHANGES THE PARAMETER COUNT -- a declared arm, never "
                         "a silent upgrade.")
    ap.add_argument("--n-registers", type=int, default=4,
                    help="register tokens; consumed internally and STRIPPED "
                         "before the readout (they are not at a place in the "
                         "image)")
    ap.add_argument("--per-layer-encoders", action="store_true",
                    help="E-ENC arm (b): per-layer encoders instead of one "
                         "common encoder + adapters. Decide at MATCHED TOTAL "
                         "PARAMS; a tie goes to the common encoder.")
    # ---- §4b horizon -------------------------------------------------------
    ap.add_argument("--plan-steps", type=int, default=PLAN_STEPS)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--a-max", type=float, default=A_MAX)
    ap.add_argument("--kappa-max", type=float, default=KAPPA_MAX)
    # ---- X3 isolation ------------------------------------------------------
    ap.add_argument("--no-isolate-planner", action="store_true",
                    help="⛔ MIS-WIRED ARM: let planner/goal heads backprop "
                         "into the encoder. Pre-registered control only.")
    ap.add_argument("--no-isolate-uplink", action="store_true",
                    help="⛔ co-trained control arm: no stop-grad on the "
                         "higher->lower latent path")
    ap.add_argument("--uplink", choices=("stopgrad", "ema"), default="stopgrad")
    ap.add_argument("--ema-decay", type=float, default=0.996)
    # ---- measures ----------------------------------------------------------
    ap.add_argument("--o1-k", type=int, default=10,
                    help="O1/L_ctrl roll horizon (the W3 probe's k)")
    ap.add_argument("--w-o1-ctrl", type=float, default=1.0)
    ap.add_argument("--w-o1-fact", type=float, default=1.0)
    ap.add_argument("--w-o1-scene", type=float, default=0.3)
    ap.add_argument("--dkappa", type=float, default=DKAPPA_DEFAULT)
    ap.add_argument("--daccel", type=float, default=DACCEL_DEFAULT)
    ap.add_argument("--rand-dkappa-max", type=float, default=0.05)
    ap.add_argument("--rand-daccel-max", type=float, default=3.0)
    ap.add_argument("--w-o2", type=float, default=1.0)
    ap.add_argument("--o2-tau-s", type=float, default=2.0,
                    help="O2 time-to-reach decay constant, SECONDS (never "
                         "metres — HIERARCHY_VOCABULARY §2)")
    ap.add_argument("--w-o3", type=float, default=1.0)
    ap.add_argument("--o3-mode", choices=("action", "static"), default="action")
    ap.add_argument("--o3-blocks", type=int, default=2)
    ap.add_argument("--o3-block-h", type=int, default=2)
    ap.add_argument("--o3-block-w", type=int, default=2)
    ap.add_argument("--o3-band-rows", type=int, default=0,
                    help="also mask the bottom N near-field readout rows")
    ap.add_argument("--o4-alpha", type=float, default=1.0,
                    help="O4 saliency exponent; 0 = uniform (control arm)")
    ap.add_argument("--o4-floor", type=float, default=0.25)
    ap.add_argument("--w-o5", type=float, default=1.0)
    ap.add_argument("--o5-k", type=int, default=20,
                    help="O5 rollout length in 10 Hz steps (60 = the §4b 6 s "
                         "horizon; needs a cache with max_horizon >= 60)")
    ap.add_argument("--o5-mode",
                    choices=("uniform", "linear-decay", "endpoint"),
                    default="uniform")
    ap.add_argument("--w-o6", type=float, default=0.1)
    ap.add_argument("--sigreg-slices", type=int, default=512)
    ap.add_argument("--sigreg-free-dims", type=int, default=0)
    ap.add_argument("--spectrum-every", type=int, default=200)
    ap.add_argument("--w-t1", type=float, default=1.0)
    ap.add_argument("--w-s1", type=float, default=1.0)
    ap.add_argument("--lambda-plan", type=float, default=None,
                    help="planner gradient scale; unset = the STAGE default "
                         f"({STAGE_LAMBDA_PLAN}). 0 in S-W BY CONSTRUCTION.")
    # ---- data --------------------------------------------------------------
    ap.add_argument("--v2-cache", nargs="+", default=[])
    ap.add_argument("--v2-val-cache", nargs="+", default=[])
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW")
    ap.add_argument("--frame-hfov", type=float, default=120.0)
    ap.add_argument("--projection", default="cylindrical")
    ap.add_argument("--require-parity", action="store_true", default=True)
    ap.add_argument("--no-require-parity", dest="require_parity",
                    action="store_false")
    ap.add_argument("--eps-per-batch", type=int, default=4)
    ap.add_argument("--max-horizon", type=int, default=None,
                    help="future steps each window carries. Unset = derived "
                         "from the stage's own needs (o1_k, o5_k, stride_tac, "
                         "stride_str, plan_steps). ⚠️ NOT inherited from the "
                         "v4 horizon plan, which is 20 and would make §4b's "
                         "6 s horizon structurally untrainable.")
    # ---- optimisation ------------------------------------------------------
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # ---- dry run -----------------------------------------------------------
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything, run --dry-steps synthetic CPU "
                         "steps, write config.json + dry_run.json, exit")
    ap.add_argument("--dry-steps", type=int, default=2)
    ap.add_argument("--dry-batch", type=int, default=2)
    ap.add_argument("--dry-k", type=int, default=12)
    ap.add_argument("--print-launch", action="store_true",
                    help="print the PYTHONPATH-correct pod launch line "
                         "and exit")
    return ap


def _launch_line(a) -> str:
    argv = " ".join(sys.argv[1:]).replace(" --print-launch", "")
    return ("PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 "
            f"python3 scripts/train_v6_staged.py {argv}")


def preflight(a) -> list[str]:
    """Refusals that must fire BEFORE a GPU-day is spent."""
    problems: list[str] = []
    if a.stage == "S-W" and resolve_lambda_plan(a):
        problems.append(
            f"--stage S-W with --lambda-plan {a.lambda_plan}: S-W is the WORLD "
            f"stage and its planner is ABSENT (λ_plan ≡ 0). A planner "
            f"gradient here destroys the stage's attributability — the exact "
            f"co-training defect the staging exists to avoid.")
    if a.stage == "S-W" and a.selector != "none":
        problems.append(
            f"--stage S-W with --selector {a.selector}: the planner group is "
            f"FROZEN in S-W, so a scorer built here would be untrainable dead "
            f"weight AND would change the state_dict — which breaks a strict "
            f"resume of the live S-W run. Selection is an S-T lever.")
    if a.w_select and a.selector == "none":
        problems.append(
            f"--w-select {a.w_select} with --selector none: a selection loss "
            f"with no scorer is how a selector silently never trains.")
    if (a.selector != "none" and not a.w_select
            and not getattr(a, "i_know_this_is_the_control_arm", False)):
        problems.append(
            f"--selector {a.selector} with --w-select 0: the scorer would be "
            f"built, consume its parameters and never receive a gradient. If "
            f"an inert-scorer control is what you want, say so explicitly by "
            f"passing --w-select 0 AND --i-know-this-is-the-control-arm.")
    if a.allow_inconclusive_gate and not a.gate_off_reason.strip():
        problems.append("--allow-inconclusive-gate needs --gate-off-reason "
                        "(an override with no stated reason is an "
                        "unremembered decision)")
    if a.no_isolate_planner or a.no_isolate_uplink:
        problems.append(
            "⚠️ ISOLATION DISABLED — this is a pre-registered CONTROL arm, not "
            "a default. If that is intended, re-run with "
            "--i-know-this-is-the-control-arm.")
    if not a.dry_run and not a.v2_cache:
        problems.append("--v2-cache is required for a real run (the canonical "
                        "corpus physicalai-train-e438721ae894; parity is "
                        "sacred)")
    if a.o5_k > a.plan_steps:
        problems.append(f"--o5-k {a.o5_k} exceeds --plan-steps {a.plan_steps}")
    if (not a.dry_run and STAGE_PRECONDITION.get(a.stage)
            and not a.init_from):
        problems.append(
            f"--stage {a.stage} without --init-from: it must start from "
            f"{STAGE_PRECONDITION[a.stage]}'s ckpt.pt. A gate saying the stage "
            f"below passed is worthless if this stage then trains on a "
            f"randomly-initialised trunk — that is not the staged protocol, "
            f"it is four unrelated models with a gate between them.")
    return problems


def main(argv=None) -> int:
    ap = build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "6")   # the 113-threads trap
    if a.print_launch:
        print(_launch_line(a))
        return 0
    problems = preflight(a)
    if a.control_arm_ack:
        problems = [p for p in problems if not p.startswith("⚠️ ISOLATION")]
    if problems:
        for p in problems:
            print(f"[v6] ⛔ {p}", flush=True)
        return 2
    if a.dry_run:
        dry_run(a)
        return 0
    train(a)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
