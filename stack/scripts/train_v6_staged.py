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
import importlib
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))      # stack root

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    GOAL_ARG_SLOTS, HORIZON_S, MODULE_GROUPS, PLAN_STEPS, STAGES,
    STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS, InteractionSampler,
    LayerSpectrumMonitor, V6Config, V6Stack, apply_stage_freeze,
    kinematic_saliency,
    near_field_band_mask, sample_cell_block_mask, saliency_weights,
    SpectrumAccumulator, o6_rank_verdict, sigreg_trend_verdict,
    spectrum_report,
    stage_trainable_groups, time_to_reach_weights)
from tanitad.models.sigreg import position_relaxed  # noqa: E402

# O1 — IMPORTED, never re-implemented: the response-form L_ctrl and its
# counterfactual machinery are the stage-A artifacts that PASSED the W3 gate.
from train_stage_a import (TRAIN_ARMS, sample_random_deltas,  # noqa: E402
                           stage_a_losses)
from train_v58f_unicycle_head import A_MAX, KAPPA_MAX  # noqa: E402
from stage_a_probes import DACCEL_DEFAULT, DKAPPA_DEFAULT  # noqa: E402

#: The canonical `s2-strategic-v1` label artifact, for `--s2-labels`' help.
#: ⛔ A LITERAL, NOT `from s2_labels import S2_CANONICAL_LABELS_REL`. MEASURED
#: 2026-08-16: the module-level import added `s2_labels` to this trainer's
#: IMPORT-TIME CLOSURE, which `tests/test_runbook_commands.py` pins because the
#: closure is exactly the set of files that must be FILE-SHIPPED to a pod
#: (pods have no git credentials). Importing a module to print a help string
#: would have made `s2_labels` mandatory for EVERY launch and given a pod
#: missing it a `ModuleNotFound` at startup — a real operational cost for a
#: cosmetic gain. `s2_labels` stays a LAZY import on the paths that use it.
#: ⚠️ The two copies cannot drift: `test_v6_s2_loss.py` asserts this string
#: equals `s2_labels.S2_CANONICAL_LABELS_REL` (C81 — audit the copies against
#: each other where a fact is written twice).
S2_CANONICAL_LABELS_REL = (
    "TanitAD Research Hub/Data Engineering/Implementation/incoming/"
    "2026-08-16-s2-v1-labels/review/labels_v2")

__all__ = [
    "V6LossWeights", "STAGE_PRECONDITION", "STAGE_GATE_SPEC",
    "STAGE_INVALIDATES", "STAGE_INVALIDATION_MECHANISM",
    "STAGE_MAY_INTRODUCE", "RESUME_CONTRACT",
    # E4 — the arm-conditional gate layer
    "GATE_APPLICABILITY", "UNMEASURED_BY_CONSTRUCTION", "SEL_GAP_TIER_NOTE",
    "probe_applies", "arm_record",
    "o2_near_field_loss", "o3_masked_cell_loss", "o5_rollout_consistency_loss",
    "o6_sigreg_loss", "rollout_step_weights", "build_o4_weights",
    "ANCHOR_OBJECTIVES", "ANCHOR_OBJ_MODES", "ANCHOR_AXIS_W_DEFAULT",
    "anchor_goal_loss",
    "S2_IGNORE_ID", "s2_goal_loss", "synthetic_s2_batch",
    "v6_loss_step", "stage_gate_dict", "write_stage_gate",
    "assert_stage_precondition", "GatePreconditionError",
    "in_spectrum_window",
    "x4_monitor_from_args", "x4_trend_record",
    "X4_TREND_BASELINE_STEPS", "X4_TREND_CURRENT_STEPS",
    "ResumeLineageError", "read_ckpt_provenance", "assert_resume_lineage",
    "resume_guard", "load_resume", "load_stage_init",
    "supersede_init_on_resume",
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
    #: THE ``ANCHOR_GOAL`` OBJECTIVE (2026-08-16). DEFAULT 0.0 = OFF everywhere,
    #: so the incumbent loss is bit-identical and the live v6F S-W resume is
    #: untouched. Needs ``cfg.anchor_goal != "none"`` AND an anchor table AT THE
    #: PLAN HORIZON -- and no such table exists (every banked vocabulary stops
    #: at 2.0 s), which is why this ships OFF and REFUSING rather than merely
    #: unused. The objective is :data:`ANCHOR_OBJECTIVES`: the DEFAULT is
    #: METRIC-AWARE and the one-hot CE is reachable only as the named,
    #: acknowledged CONTROL, because E-AG2 measured that CE +4.7502
    #: [+3.0514, +6.3981] WORSE than a ridge that was ALREADY refused.
    #: Units: METRES for ``metric``/``softanchor`` (as for ``w_select``), NATS
    #: for ``ce`` -- one more reason the two are not interchangeable and the
    #: weight is a declared decision, never a default.
    w_anchor: float = 0.0
    #: S2 — STRATEGIC GOAL SUPERVISION (2026-08-16). DEFAULT 0.0 = OFF
    #: everywhere, so the incumbent loss is bit-identical and the live resume
    #: is untouched. CE on ``g_str``/``a_str`` logits + masked L1 on their
    #: args against `s2-strategic-v1` labels (S2_STRATEGIC_GAP.md §1.2,
    #: produced by the 2026-08-16 label build), joined per clip and masked to
    #: the ``s2_valid`` band. ⛔ GOAL HEADS ONLY, NEVER A TRUNK LOSS (the
    #: binding diagram rule): the term reads the heads' emitted logits/args,
    #: whose input ``z_str_p`` is detached under the planner cut — gradient
    #: reach is MEASURED in tests/test_v6_s2_loss.py as exactly
    #: goal_head_str.* + act_head_str.* and nothing else (the vocab tables
    #: are NOT touched: the heads' logits/args come from their own
    #: trunk/type_head/arg_head, and ``vocab_str.encode`` sits only on the
    #: downstream conditioning path this loss never reads). In force only in
    #: S-S/S-J — the stages that train ``layer_str``; zeroed elsewhere so the
    #: launch line cannot advertise a term that trains nothing.
    w_s2_goal: float = 0.0

    def for_stage(self, stage: str) -> "V6LossWeights":
        """The weights actually in force for ``stage``.

        S-W zeroes every planner term AND every higher-layer term: a loss whose
        module is frozen still builds a graph, still costs compute, and — the
        part that bites — still appears in the log as if it were training
        something. Zeroing them here keeps the log honest about what moved.
        """
        if stage == "S-W":
            return replace(self, t1_latent=0.0, s1_latent=0.0,
                           lambda_plan=0.0, seam_op=0.0, w_select=0.0,
                           w_anchor=0.0, w_s2_goal=0.0)
        if stage == "S-T":
            # w_s2_goal is zeroed here for the layer_str reason w_anchor is
            # zeroed in S-S: the strategic goal heads are FROZEN in S-T
            # (STAGE_GROUPS["S-T"] has no layer_str), so an S2 term in force
            # would be advertised in the launch line and train nothing.
            return replace(self, o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0,
                           o2_nearfield=0.0, o3_masked=0.0, o5_rollout=0.0,
                           o6_sigreg=0.0, s1_latent=0.0, w_s2_goal=0.0)
        if stage == "S-S":
            # w_anchor joins w_select here for the SAME reason: the anchor head
            # is planner-group (v6.py MODULE_GROUPS: ("anchor_head.",
            # "planner")) and S-S trains ``layer_str`` ONLY, so an anchor loss
            # in force here would be a term advertised in the launch line that
            # trains nothing.
            return replace(self, o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0,
                           o2_nearfield=0.0, o3_masked=0.0, o5_rollout=0.0,
                           o6_sigreg=0.0, t1_latent=0.0, lambda_plan=0.0,
                           seam_op=0.0, w_select=0.0, w_anchor=0.0)
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
    # S-T introduces, by design: the selector (when an arm is opted into), and
    # the g_str->P_T conditioning port `cond_tac_dyn.` (F-1,
    # DIAGRAM_CONFORMANCE.md 2026-08-16 — the diagram/§5-spec'd tactical-
    # dynamics downlink the code never built; zero-init, so the introduction is
    # loss-continuous). Both are NEW MODULES rather than widened shapes, on
    # purpose: this allowance adjudicates KEYS, and a shape change bypasses it
    # entirely (`load_state_dict(strict=False)` still RAISES on shapes —
    # measured, see `trainer_argv`'s --n-candidates note).
    # ⭐ 2026-08-16, the diffusion/MPC/fallback build: S-T may also introduce
    # the diffusion proposal generator (`prop_diffusion.`, +437,954 params
    # MEASURED at production geometry — a declared fan-generator ARM) and the
    # fallback trigger's calibration buffers (`fallback.`, 8 keys, 0 params —
    # the P7 band ships with the checkpoint like the anchor table). The MPC
    # refiner needs NO entry: it holds no parameters and no buffers, so
    # flipping it changes no state_dict key at all.
    # ⭐ 2026-08-16, F-18: S-T may also introduce the PERCEPTION AGENT-SLOT
    # DECODER (`agent_slots.`, +3,207,445 params MEASURED at the §6 production
    # geometry — d_model 256 x depth 3 x 16 queries over the 16 readout cells,
    # inside the pre-registered 2-4 M band).
    # ⚠️ IT IS AN INTRODUCTION-ONLY ENTRY, AND THAT DISTINCTION MATTERS: no
    # ladder stage TRAINS it. The v6 batch carries frames/actions/poses/future_*
    # and no agent labels (see the `interp` note on STAGE_GROUPS in v6.py), so
    # this allowance exists so a run can CARRY the interpretation head forward
    # from an S-W checkpoint that never had it, while a frozen-trunk probe in
    # the P8 idiom is what optimises it. An entry here has never MEANT "this
    # stage optimises the module" — `fallback.` (0 trainable params) was
    # already the counter-example.
    # ⚠️ CARRY RULE, recorded HERE because it is NOT chain-enforced: if an S-T
    # run is ever launched WITH `--agent-slots`, its checkpoint carries
    # `agent_slots.*` and S-S/S-J must be launched with the flag too, or those
    # keys become UNEXPECTED and `load_stage_init` is fatal — exactly the
    # `--selector` / `--tac-goal-cond` failure `v6_chain.assert_geometry_carry`
    # catches from a JSON read BEFORE the corpus mounts. It does NOT catch this
    # one: that check enumerates its levers first-class and there is no
    # `Step.agent_slots`, because no chain step sets the flag (no ladder stage
    # trains the head). ⇒ plumbing it into the chain is the follow-on the moment
    # a chain step wants the head; until then `load_stage_init` still refuses
    # correctly, only later and with a less specific message.
    "S-T": ("cand_score.", "cond_tac_dyn.", "prop_diffusion.", "fallback.",
            "agent_slots."),
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

#: ⛔ WHAT A ``--resume auto`` REQUIRES OF THE CHECKPOINT IT FINDS, as data.
#:
#: MEASURED 2026-08-16 by EXECUTING the transition, not by reading it. Before
#: this, ``load_resume`` did a strict state-dict load and adopted ``ck["step"]``
#: with **no stage check at all** — every stage saves the WHOLE stack, so a
#: checkpoint written by S-T is key-for-key loadable into an S-S run.
#:
#: Cross-stage resume was stopped only INCIDENTALLY, by ``torch.optim``'s
#: param-group size check, and only because the per-stage trainable-TENSOR
#: counts happen to be distinct (MEASURED at the production geometry:
#: S-W 240 · S-T 80 · S-S 54 · S-J 374). That guard is worthless as a guard:
#:
#:   * it names nothing — the operator sees ``ValueError: loaded state dict
#:     contains a parameter group that doesn't match the size of optimizer's
#:     group``, which points at the optimiser, not at the ladder;
#:   * it is one :data:`~tanitad.models.v6.STAGE_GROUPS` edit away from two
#:     stages sharing a count, at which point it passes SILENTLY; and
#:   * ⛔ it is skipped entirely when the checkpoint carries no ``opt`` key —
#:     which is exactly the shape of ``ops/ckpt_fp16_snapshot.py``, the
#:     documented pod-handover artifact.
#:
#: A wrong-stage resume is a multi-GPU-day error that surfaces as "the model
#: got worse": the run adopts the OTHER stage's step (so the cosine schedule is
#: replayed to the wrong point), and if the counts ever collide it adopts the
#: other stage's optimiser moments — S-T's ``exp_avg`` for ``layer_tac`` landing
#: on ``layer_str`` by list position. ⇒ an EXPLICIT refusal, checked BEFORE the
#: corpus build rather than 130 lines later where ``load_resume`` sits.
RESUME_CONTRACT: dict[str, str] = {
    "same_stage": "the checkpoint's `config.stage` must EQUAL the stage being "
                  "launched. Every stage saves the whole V6Stack, so the "
                  "state_dict load cannot tell them apart — the stage label is "
                  "the only thing that can, and `_save_ckpt` has always "
                  "written it (`_run_config`: 'stage': a.stage).",
    "labelled": "an UNLABELLED checkpoint is one whose lineage cannot be "
                "verified. Every checkpoint this trainer writes carries its "
                "stage; one that does not came from somewhere else, and "
                "resuming it is an assumption wearing a resume's clothes. Use "
                "--init-from, which needs no label because it starts a NEW run "
                "at step 0 instead of inheriting a step and a schedule.",
    "has_optimiser": "a resume without optimiser moments is not a resume — it "
                     "is an --init-from that also silently inherits a step. "
                     "`ops/ckpt_fp16_snapshot.py` drops `opt` BY DESIGN (2/3 "
                     "of the bytes) and says so; it is an --init-from artifact "
                     "and must be refused here rather than half-honoured.",
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
        # X4_spectrum_layers is REPORTED, not required: the per-layer records
        # exist from step 0 but their verdicts are INCONCLUSIVE until pooled
        # (per-layer clause 1), and a required probe that is structurally
        # INCONCLUSIVE at the incumbent flags would be noise wearing a gate.
        "reported": ("P2", "P5", "P8", "O6_spectrum", "X4_spectrum_layers"),
        "owners": {"P1": "scripts/probe_latent_state.py",
                   "P2": "scripts/probe_latent_state.py",
                   "P3": "scripts/stage_a_probes.py",
                   "P6": "scripts/stage_a_probes.py",
                   "P5": "taniteval/tools/t1_eval.py",
                   "P8": "scripts/train_p8_occupancy.py",
                   "O6_spectrum": "tanitad.models.v6.spectrum_report",
                   "X4_spectrum_layers":
                       "tanitad.models.v6.LayerSpectrumMonitor"},
        "criteria": {
            "P1_retention": ">= 0.85x R2(z) at k=10 per driving target",
            "P3_sign": ">= 0.95 per channel, BOTH lat and lon",
            "P3_gain": "median gain in [0.5, 2.0], WITHOUT post-training",
            "P6_dims": "action-subspace dims (80 % var) <= 32",
            # ⛔ RE-DERIVED 2026-08-16 (SIGREG_GATE_POWER.md). The old text was
            # ">= 0.8x effective rank across phases" and named no estimator, no
            # n and no interval. MEASURED at the live run's n=48 it fires when
            # NOTHING changed between 9 % (model null) and 38 % (the run's own
            # banked spread), with power 0.11 against a 1.43x true collapse —
            # a guard that goes off when nothing happened. The replacement is
            # owned by tanitad.models.v6.o6_rank_verdict and can say
            # INCONCLUSIVE, which the old one could not.
            "O6_rank_retention":
                "o6_rank_verdict: (1) ADMISSIBLE only at rank_ceiling >= 1024 "
                "-- a single 48-row batch is INCONCLUSIVE by construction, "
                "pool with --spectrum-accum; (2) RETENTION fails only when the "
                "cluster-JACKKNIFE interval on ER_cur/ER_ref lies WHOLLY below "
                "0.8x, passes only when it lies wholly at/above, else "
                "INCONCLUSIVE; (3) FLOOR: pooled effective_rank >= 64 "
                "regardless of retention",
            # X4 (2026-08-16): the SAME three clauses per layer, under EACH
            # LAYER'S measured (ceiling_min, floor) — tac 256/32, str 128/32
            # (x4_layer_power.json; z_op's 1024/64 re-derived as the anchor).
            # ⚠️ at 8 rows/step, --spectrum-accum 32 reaches ceiling 255 — ONE
            # ROW short of tac's 256; 33 is the accum that makes all three
            # layers adjudicable. REPORTED, never required (see above).
            "X4_rank_retention":
                "x4_rank_verdict per layer {tac, str}: clause 1 at the "
                "LAYER'S ceiling_min (tac 256, str 128 -- NOT z_op's 1024, "
                "which d_str=256 can never reach), clause 2 identical, "
                "clause 3 at the layer's measured floor (32). "
                "--spectrum-accum 33 recommended (32 leaves tac one row "
                "short); INCONCLUSIVE is not a pass"},
    },
    "S-T": {
        "required": ("TACTICAL_family", "sel_gap"),
        # ⚠️ ALL FOUR families are listed, not two. The binding diagram's S-T
        # gate row is "four families at 0-2 s AND 0-6 s, T1 tier"
        # (HIERARCHY_VOCABULARY §4b eval consequence + the 2026-08-02 binding
        # rule: an eval missing a family is INCOMPLETE). LONGITUDINAL is the
        # family S-T's own thesis moves — the tactical layer OWNS target speed
        # — so omitting it here was an audited gap (DIAGRAM_CONFORMANCE.md,
        # 2026-08-16). Reported-not-required: `reported` probes are stubbed and
        # never adjudicated, so this is visibility, not a new refusal.
        "reported": ("P7", "LATERAL_family", "LONGITUDINAL_family",
                     "STRATEGIC_family", "X2_seam"),
        "owners": {"TACTICAL_family": "taniteval/tools/eval_four_families.py",
                   "LATERAL_family": "taniteval/tools/eval_four_families.py",
                   "LONGITUDINAL_family":
                       "taniteval/tools/eval_four_families.py",
                   "STRATEGIC_family": "taniteval/tools/eval_four_families.py",
                   "sel_gap": "tanitad.models.tactical.sel_gap_tac",
                   "P7": "scripts/w7_roll_rerank.py",
                   # ⭐ 2026-08-16 (F-16): the owner was a bare estimator name,
                   # which is not an owner — the instrument did not exist. It
                   # does now: taniteval/tools/seam_probe.py (+ taniteval.seam),
                   # which delegates its intervals to ci.py's PAIRED bootstrap.
                   "X2_seam": "taniteval/tools/seam_probe.py "
                              "(taniteval.seam; PAIRED bootstrap from "
                              "taniteval/ci.py only)"},
        "criteria": {
            "sel_gap": "<= 0.5x the fan oracle at T1 tier",
            # X2 is "seam metrics VERIFY, never repair" (the binding diagram).
            # A seam finding is a REPORT, never a licence for a repair term.
            "X2_seam":
                "no CONFIRMED seam row at the 2 s band edge (boundary 20) on "
                "the emitted winner: the paired episode-cluster CI on the "
                "seam-vs-within-band excess must NOT lie wholly above the "
                "materiality floor (1x the within-band step) under BOTH the "
                "global and the local null. A null counts only when it is "
                "WELL-POWERED (MDE@80% <= the floor, >= 8 episode clusters); "
                "otherwise the probe returns INCONCLUSIVE, which is not a "
                "pass. VERIFY, NEVER REPAIR",
            "TACTICAL_family": "confusion improves on E4.1-derived strata",
            "four_families_horizons":
                "every family reported at BOTH 0-2 s AND 0-6 s "
                "(HIERARCHY_VOCABULARY §4b: 'four families + oracle/selected "
                "reported at BOTH 0-2 s and 0-6 s'). A family that cannot be "
                "computed is declared per family with the reason and the n — "
                "never silently dropped (PI 2026-08-02, binding)",
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
        # ⚠️ `goal_provenance` is the BINDING diagram's second S-S gate element
        # ("gate: STRATEGIC family + goal-provenance audit"). Reported-not-
        # required for now: S2 (g_str supervision) is deliberately not wired
        # (V6_TRAINER_DESIGN §"S2 is not wired here and must not be faked"), so
        # pre-S2 the audit has nothing to audit and a required probe would be
        # vacuous. The moment S2 lands, promote it to `required` — the audited
        # gap and the promotion trigger are recorded in DIAGRAM_CONFORMANCE.md.
        "reported": ("S1_ade_8_30s", "X2_seam", "goal_provenance"),
        "owners": {"STRATEGIC_family": "taniteval/tools/eval_four_families.py",
                   "S1_ade_8_30s": "taniteval/tools/t1_eval.py",
                   "X2_seam": "taniteval/tools/seam_probe.py "
                              "(taniteval.seam; PAIRED bootstrap from "
                              "taniteval/ci.py only) — F-16, 2026-08-16",
                   "goal_provenance":
                       "config-audit over the S-S run config + S2 label "
                       "artifacts (instrument to build — see "
                       "DIAGRAM_CONFORMANCE.md, 2026-08-16)",
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

#: ⛔ E4 — THE CRITERIA THAT ONLY EXIST ON AN ARM THAT HAS A SCORER, as data.
#:
#: **The defect this closes (E4, `ST_LAUNCH_READINESS.md` §5.2).**
#: ``STAGE_GATE_SPEC["S-T"]["required"]`` contains ``sel_gap``; the default S-T
#: arm is ``--selector none`` because SEL-1 fired REFUSED; and on that arm
#: :class:`~tanitad.models.v6.V6Stack` emits **no ``sel_*`` key at all**
#: (``v6.py:3968`` — the whole block is under ``if self.cand_score is not
#: None``). So there is no ``sel_idx``, ``tactical.sel_gap_tac`` has no
#: argument, and ``taniteval.selgap`` has nothing to score. The verdict was
#: **INCONCLUSIVE by construction** — a criterion decided by the build, not by
#: the model. Same class as the three vacuous gates found that week (K3 pinned
#: at 0.5; the pre-S2 goal-provenance audit; ``_grad_census``'s zero-parameter
#: group), except this one could not *pass* rather than could not *fail*.
#:
#: ⚠️ **AND THE MECHANISM AS FIRST REPORTED WAS MISATTRIBUTED — see
#: :data:`SEL_GAP_TIER_NOTE`.** The fix is NOT "turn ``--w-select`` on".
#:
#: ⭐ **WHY THIS IS A STRENGTHENING AND NOT A LOOPHOLE.** The criterion is not
#: deleted and not demoted: it stays in ``required`` verbatim, and it stays
#: BINDING on every arm that has a scorer — which is the only arm it was ever
#: written for. What changes is that an arm which CANNOT produce the quantity
#: says so, with its reason, instead of reporting INCONCLUSIVE forever. Before
#: this, a selector arm and a no-selector arm produced the *same* verdict, so
#: the gate was equally uninformative about both; now the selector arm is the
#: only one that can be certified, and it cannot escape certification.
#:
#: ⛔ **AND "NOT APPLICABLE" IS NEVER "PASS".** A stage whose ``sel_gap`` did
#: not apply carries :data:`UNMEASURED_BY_CONSTRUCTION`'s standing record in its
#: own certificate, naming the artifact and the pre-registered threshold that
#: would make it applicable. The four-families rule, applied verbatim: *a family
#: that cannot be computed is declared per family with the reason and the n,
#: never silently dropped* (PI 2026-08-02, binding).
#:
#: Keyed ``stage -> {probe: predicate-name}``. The only predicate today is
#: ``has_scorer``, resolved from the BUILT STACK (``stack.cand_score is not
#: None``), never from the flag — a ``--selector goal`` that failed to build a
#: scorer must not be certified as though it had one.
GATE_APPLICABILITY: dict[str, dict[str, str]] = {
    "S-W": {},
    "S-T": {"sel_gap": "has_scorer"},
    # the revalidation inherits the dependency exactly: S-S re-measures the
    # SAME quantity under the post-S-S ``e_g_tac``, so it exists on exactly the
    # same arms. ⚠️ ``w_select`` is 0 at S-S (``for_stage("S-S")`` zeroes it and
    # the trainer refuses the flag) — that is irrelevant here, because the gate
    # probe is an EVAL-time T1 instrument, not the train-time log key. The
    # FROZEN scorer still runs in the forward pass and still emits ``sel_idx``.
    "S-S": {"sel_gap_revalidated": "has_scorer"},
    "S-J": {},
}

#: ⚠️ **A CORRECTION TO THE TWO REPORTS THAT FOUND E4, kept beside the fix.**
#:
#: `ST_LAUNCH_READINESS.md` §5.2 and `ST_LAUNCH_FIXES.md` §6 both name
#: ``train_v6_staged.py``'s ``if w.w_select:`` block as the reason the S-T gate
#: cannot read ``sel_gap``. That line is real, and it is **not the emitter the
#: gate consumes**. MEASURED by reading :func:`run_stage_gate`: the gate's
#: probes come from exactly four places — ``--gate-probes`` (an external JSON),
#: ``X3_isolation`` (computed in place), ``spectrum`` and ``x4_spectra``. **No
#: training-loop log key ever becomes a gate probe.**
#:
#: The two same-named quantities live at two tiers and only one is quotable:
#:
#:   * the LOG key ``sel_gap`` — a **T0 train-time monitor**, per
#:     ``tactical.sel_gap_tac``'s own docstring (*"this function is the cheap
#:     train-time monitor only"*), on the training batch, no interval;
#:   * the GATE probe ``sel_gap`` — owner ``tanitad.models.tactical.sel_gap_tac``
#:     re-run at **T1 tier** through ``taniteval.selgap`` (episode-cluster
#:     bootstrap, per-level never pooled), criterion *"<= 0.5x the fan oracle at
#:     T1 tier"*, delivered via ``--gate-probes``.
#:
#: ⇒ **Turning on ``--selector goal --w-select 1.0`` would make the LOG key
#: appear and would leave the GATE exactly as INCONCLUSIVE as before.** The
#: readable-gate requirement is two things, not one: the arm must HAVE a scorer
#: (so ``sel_idx`` exists at eval time) AND the T1 battery must be run and
#: folded in with ``--gate-probes``, like every other required probe.
#:
#: **Root-cause class: a probe read at the wrong SCOPE** — the ``df``-on-a-pod /
#: Thor-``free`` / cgroup-``usage_in_bytes`` family, here as two identically
#: named quantities at two eval tiers, the T0 one read as the T1 one. It is also
#: the EVAL_DOCTRINE rule biting inside our own source: *a number without its
#: tier stamp is incomplete*.
SEL_GAP_TIER_NOTE: str = (
    "the gate probe `sel_gap` is the T1 instrument (taniteval.selgap, "
    "episode-cluster bootstrap) supplied through --gate-probes — NOT the T0 "
    "train-time log key of the same name emitted under `if w.w_select:`. "
    "Enabling --w-select alone makes the LOG key appear and leaves the GATE "
    "unchanged; the gate needs an arm that HAS a scorer AND the T1 battery.")

#: ⛔ WHAT A NOT-APPLICABLE CRITERION LEAVES UNMEASURED, and exactly what would
#: measure it. Stamped into every gate artifact that skips one, so the gap is a
#: standing, visible work item inside the certificate rather than a silence.
UNMEASURED_BY_CONSTRUCTION: dict[str, dict[str, str]] = {
    "has_scorer": {
        "question": "TACTICAL SELECTION — does the tactical level pick a good "
                    "candidate out of the fan it proposed? (sel_gap = selected "
                    "- oracle separates 'the fan cannot propose it' from 'the "
                    "selector cannot find it'.)",
        "why_not_measured": "this arm was built with --selector none, so "
                            "V6Stack emits no sel_* key (v6.py: the block is "
                            "under `if self.cand_score is not None`). There is "
                            "no sel_idx and therefore no sel_gap to measure — "
                            "it is UNCOMPUTABLE on this arm, not merely "
                            "not-yet-run. No battery run can produce it.",
        "why_this_arm": "SEL-1 fired REFUSED 2026-08-16 (E-WC2: sigma/ADE "
                        "9.9915 [7.4492, 13.5119] against a refusal line of "
                        "3.0 pre-registered with both outcomes committed in "
                        "advance). v6_chain.assert_selector_admissible REFUSES "
                        "to launch any selector arm while that stands.",
        "what_would_make_it_applicable":
            "the E-WC2-SW measurement at the S-W -> S-T boundary "
            "(~10-25 GPU-min): dump the FROZEN S-W latents, run "
            "scripts/e_wc2_sigma_star.py, write <sw_dir>/ewc2_sw_latents.json. "
            "PRE-REGISTERED 2026-08-16 BEFORE the dump was taken: sigma(2 s) "
            "<= 0.80 m FUNDED (the arm launches with --selector goal and this "
            "criterion binds) · 0.80 < sigma <= 1.41 INCONCLUSIVE (REFUSED "
            "stands) · sigma > 1.41 REFUSED stands. The 0.80 m line is not a "
            "round number: GoalDistanceScorer's requirement curve measured the "
            "goal rule BETTER than the trained selector at sigma 0.5 m "
            "(-0.1591 [-0.2300, -0.0894]) and WORSE at sigma 1.0 m (+0.0943 "
            "[+0.0241, +0.1650]), both separated.",
        "_read": "NOT a pass, NOT a failure of the model, and NOT a criterion "
                 "that was dropped: a question this arm is structurally unable "
                 "to answer. It stays required on every arm that CAN.",
    },
}


def probe_applies(stage: str, probe: str, arm: dict | None) -> dict | None:
    """Is ``probe`` measurable on this arm? ``None`` == yes (the strict default).

    Returns a **reason dict** when the probe is NOT applicable, so a caller can
    never reduce the answer to a bare boolean and lose the explanation — the
    thing that made the three vacuous gates possible.

    ``arm`` is the run's own build record, ``{"has_scorer": bool, ...}``, taken
    from the BUILT STACK. ⛔ ``arm=None`` means *no record was supplied* and
    resolves to **APPLICABLE** — the strict reading — so no caller can weaken a
    criterion by forgetting to describe its arm.
    """
    pred = GATE_APPLICABILITY.get(stage, {}).get(probe)
    if pred is None or arm is None:
        return None
    if pred == "has_scorer":
        if arm.get("has_scorer"):
            return None
        return {"probe": probe, "predicate": pred,
                "applicable": False,
                "selector": arm.get("selector", "none"),
                **UNMEASURED_BY_CONSTRUCTION["has_scorer"]}
    raise KeyError(                                          # pragma: no cover
        f"GATE_APPLICABILITY[{stage!r}][{probe!r}] names predicate {pred!r}, "
        f"which probe_applies does not implement. A predicate that cannot be "
        f"resolved must raise, never default to applicable-or-not: both "
        f"silent answers are wrong verdicts wearing a gate.")


def arm_record(stack) -> dict:
    """The build facts a gate needs, read from the STACK, never from the args.

    ⭐ ``--selector goal`` is an intention; ``stack.cand_score is not None`` is
    what actually happened. A gate that adjudicated on the flag would certify a
    selector that failed to build — the ``intent_proj`` defect (a path present
    in the declaration and absent from the object) in a gate's costume.
    """
    scorer = getattr(stack, "cand_score", None)
    return {
        "has_scorer": scorer is not None,
        "scorer_class": type(scorer).__name__ if scorer is not None else None,
        "selector": getattr(getattr(stack, "cfg", None), "selector", "none"),
        "_source": "the BUILT stack (stack.cand_score), not the --selector flag",
    }


class GatePreconditionError(SystemExit):
    """A stage refused to start because the stage below it did not pass."""


class ResumeLineageError(SystemExit):
    """``--resume auto`` found a checkpoint that is not this run's own.

    Its own subclass so a chain script can tell "the ladder is mis-wired"
    (recoverable: point ``--out`` elsewhere, or use ``--init-from``) apart from
    the generic ``SystemExit`` every other refusal in this file raises.
    """


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


def o6_sigreg_loss(sigreg, z: Tensor, free_dims: int = 0, *,
                   generator: torch.Generator | None = None) -> Tensor:
    """O6 — SIGReg (full_relaxed), KEPT per the PI's 2026-08-11 call.

    Delegates to :func:`tanitad.models.sigreg.position_relaxed`, which exempts
    a fixed ego-motion subspace so anti-collapse and metric-position structure
    stop cancelling. ⚠️ Never divide the Epps-Pulley statistic by n — that was
    the ALPS-4B bug that silently disabled the loss, and it is guarded inside
    ``sigreg.py``; this wrapper exists so nobody re-implements the call.

    ``generator=None`` (the default, and the live v6F path) draws the slice
    directions from the GLOBAL RNG exactly as before. See
    ``v6_loss_step``'s ``sigreg_generator``."""
    return position_relaxed(sigreg, z, free_dims, generator=generator)


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
# THE ``ANCHOR_GOAL`` OBJECTIVE -- metric-aware by DEFAULT, CE as the CONTROL
# ============================================================================

#: The three objectives that can train :class:`~tanitad.models.v6.AnchorGoalHead`,
#: with the measurement that put each where it is. INHERITED from
#: `.../incoming/2026-08-16-anchor-goal-supervision/ANCHOR_GOAL_SUPERVISION.md`
#: (881 windows / 40 episodes, LOEO, paired episode-cluster bootstrap) and
#: `E-OBJ-1`; re-quoted with its stamp, never re-derived here.
#:
#: WHY A DEFAULT AND A CONTROL, RATHER THAN A CHOICE. E-AG2 held the ESTIMATOR
#: fixed and varied only the ESTIMAND: ``snap`` -- the same ridge, rounded to
#: the nearest anchor -- is NOT separated from the free ridge
#: (-0.0002 [-0.1031, +0.0703] at K=256; +0.0383 [-0.2125, +0.2338] in the much
#: stronger ``v0`` regime), while a K-way one-hot CLASSIFIER on the same
#: features costs +4.7502 [+3.0514, +6.3981] WORSE, separated at every K from 8
#: to 256, under BOTH vocabulary constructions, and replicating on REF-C-base
#: (+5.4570 [+3.8345, +7.1073]).
#: => QUANTISATION IS FREE; THE ONE-HOT TARGET IS WHAT COSTS +4.75 m.
#: A ``--anchor-objective ce`` DEFAULT would therefore ship a refuted objective;
#: it is reachable only behind ``--i-know-this-is-the-control-arm``, exactly as
#: ``--no-isolate-planner`` is.
ANCHOR_OBJECTIVES: dict[str, str] = {
    "metric": (
        "DEFAULT. Endpoint distance ||g_hat - g*|| in METRES on the EMITTED "
        "goal point. The emission is straight-through "
        "(`raw + (snapped - raw).detach()`), so the estimand is quantised "
        "while the gradient reaches the CONTINUOUS regression -- which is the "
        "half E-AG2 EXONERATED. This is 'regress-then-snap' as a TRAINED "
        "object rather than a post-hoc rounding."),
    "softanchor": (
        "The metric-aware DISTANCE-WEIGHTED target over anchors, for the K-way "
        "head: L = sum_k p_k * d_k with d_k = ||anchor_k - g*|| and p the "
        "head's own softmax. It is E-OBJ-1's `softade` one level up -- the "
        "EXPECTED anchor error under the model's own distribution, whose "
        "optimum is still all mass on the nearest anchor. "
        "Chosen over a SOFTENED CE target on evidence, not taste: E-OBJ-1 "
        "measured metric-awareness recovering -0.0974 m (base) / -0.1670 m "
        "(XL) separated, while SOFTENING the CE target was separated WORSE "
        "(+0.0909 m) at EVERY tau. Metric-awareness helps; target-softness "
        "hurts, so the softened-CE form is deliberately NOT offered."),
    "ce": (
        "CONTROL ONLY -- the pre-registered, MEASURED-REFUTED arm. One-hot "
        "cross-entropy on argmin_k ||anchor_k - g*||: metric-BLIND by "
        "construction (it scores 'picked the adjacent anchor' and 'picked one "
        "40 m away' identically), which is the property being controlled for. "
        "E-AG2: +4.7502 [+3.0514, +6.3981] WORSE than an already-refused "
        "ridge. Requires an explicit control-arm acknowledgement."),
}

#: Which :class:`~tanitad.models.v6.AnchorGoalHead` MODES each objective can
#: train, and it is a HARD coupling rather than a hint. Both directions of the
#: mismatch produce a NUMBER instead of an error, which is the failure class
#: this programme keeps paying for:
#:   * ``metric`` on ``"onehot"`` -- that head's emitted point is a HARD table
#:     lookup carrying NO gradient at all (by design, v6.py: "no straight-
#:     through path is offered"). The loss would fall to a constant and the
#:     optimiser would train NOTHING while the log showed a metre-scale term.
#:   * ``softanchor`` / ``ce`` on a snap mode -- there are no ``cls_logits``,
#:     so there is nothing to put a distribution on.
ANCHOR_OBJ_MODES: dict[str, tuple[str, ...]] = {
    "metric": ("snap_lat", "snap_xy"),
    "softanchor": ("onehot",),
    "ce": ("onehot",),
}

#: (LONGITUDINAL, LATERAL) axis weights. THE DEFAULT IS RAW METRES, AND THAT IS
#: THE EVIDENCE-WEIGHTED CHOICE RATHER THAN THE SYMMETRIC ONE.
#:
#: MEASURED (INHERITED, ANCHOR_GOAL_SUPERVISION.md 6.4, 2 s, 881 windows):
#: the goal's corpus variance is 98.8 % LONGITUDINAL (sigma_long 19.0578 vs
#: sigma_lat 2.0723 -- 9.2x in sigma, 84x in variance); every arm's RESIDUAL is
#: longitudinal too (ridge 6.6132 / 1.0667, i.e. 97.4 % of squared error), and
#: the HEADROOM is where the loss should spend: the classifier sits 1.96x above
#: the floor laterally (1.3310 vs 0.6802) against 14.9x longitudinally
#: (13.3502 vs 0.8954).
#:
#: A raw-metre loss is therefore ALREADY strongly anisotropic in effect -- it
#: allocates ~97 % of its squared-error gradient longitudinally, purely because
#: that is where the metres are. WHITENING (dividing each axis by its corpus
#: sigma) would UNDO exactly that, spending half the gradient on the axis
#: carrying 1.2 % of the variance -- which is 6.4's own diagnosis of what the
#: isotropic FPS vocabulary does wrong, repeated in the objective. Raw metres is
#: also METRIC-CONSISTENT WITH THE EVAL: ADE and the four families are scored in
#: metres, not in corpus sigmas.
#:
#: EVIDENCE CLASS of the weights themselves: DECLARED. They are exposed as two
#: floats so an arm can vary them, and the realised per-axis split is LOGGED
#: EVERY STEP (`anchor_err_lon_m` / `anchor_err_lat_m`, never pooled -- the
#: four-metric-families rule) so the allocation is MEASURED at run time instead
#: of assumed from a 2 s corpus statistic that has no 6 s counterpart.
ANCHOR_AXIS_W_DEFAULT: tuple[float, float] = (1.0, 1.0)


def anchor_goal_loss(head_out: dict, target_xy: Tensor, anchors: Tensor, *,
                     objective: str = "metric",
                     axis_w: tuple[float, float] = ANCHOR_AXIS_W_DEFAULT
                     ) -> tuple[Tensor, dict]:
    """The ``ANCHOR_GOAL`` supervision. Returns ``(loss, log)``.

    ``head_out``  the ``AnchorGoalHead.forward`` dict (UNPREFIXED keys);
    ``target_xy`` ``[B, 2]`` the TRUE ego-frame displacement at the plan
                  horizon, x forward / y left, metres -- the same convention
                  ``tanitad.data.anchor_goal`` and ``driving_diagnostic.
                  gt_ego_waypoints`` use, and the same endpoint the v6f
                  selector scores;
    ``anchors``   ``[K, 2]`` the FROZEN table (``AnchorGoalHead.anchors``).

    ADMISSIBILITY (PI 2026-08-03). ``target_xy`` is a LABEL built from FUTURE
    ego poses -- labels may use ego, inference may not. Nothing here enters the
    head: its only input is ``e_g_tac``, which is vision-derived
    (``goal_head_tac(z_tac_p, cond=e_g_str)``). The goal path stays
    information-disjoint from the situation classifier -- this function reads
    no situation, no ego state and no ``v0``, and ``d_k`` depends only on a
    frozen buffer and the label, so it is a CONSTANT w.r.t. every parameter
    (which is why, unlike ``w_select``'s ``err.detach()``, no detach is needed
    to keep the gradient where it is intended).

    THE LOG IS PER-AXIS, NEVER POOLED. LONGITUDINAL and LATERAL are separate
    families (PI 2026-08-02) and 98.8 % of this quantity's variance is
    longitudinal, so a single scalar would hide the axis that IS the problem.
    """
    if objective not in ANCHOR_OBJECTIVES:
        raise ValueError(
            f"anchor objective must be one of {tuple(ANCHOR_OBJECTIVES)}, got "
            f"{objective!r}. 'metric' is the DEFAULT the measurement "
            f"prescribes; 'ce' is the CONTROL E-AG2 measured +4.7502 "
            f"[+3.0514, +6.3981] WORSE.")
    mode = head_out.get("mode")
    if mode is not None and mode not in ANCHOR_OBJ_MODES[objective]:
        raise ValueError(
            f"objective {objective!r} needs an anchor_goal mode in "
            f"{ANCHOR_OBJ_MODES[objective]}, got {mode!r}. "
            f"{ANCHOR_OBJECTIVES[objective]}")
    if target_xy.ndim != 2 or target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy must be [B, 2], got "
                         f"{tuple(target_xy.shape)}")
    if anchors.ndim != 2 or anchors.shape[-1] != 2:
        raise ValueError(f"anchors must be [K, 2], got {tuple(anchors.shape)}")
    tgt = target_xy.float()
    w = torch.as_tensor(axis_w, dtype=torch.float32,
                        device=tgt.device).reshape(1, 2)
    if float(w.min()) < 0.0:
        raise ValueError(f"axis_w must be non-negative, got {tuple(axis_w)}")
    log: dict = {"anchor_objective": objective, "anchor_mode": mode,
                 "anchor_axis_w": [float(x) for x in axis_w]}

    # the label side, shared by all three objectives: the nearest anchor under
    # the SAME axis weighting the loss uses, so the CONTROL and the DEFAULT are
    # scored against one geometry rather than two.
    d_k = ((anchors.float()[None] - tgt[:, None]) * w[None]).norm(dim=-1)
    k_star = d_k.argmin(dim=-1)                                       # [B]

    if objective == "metric":
        g = head_out["goal_point"].float()                            # [B, 2]
        resid = (g - tgt) * w
        loss = resid.norm(dim=-1).mean()
        raw = head_out.get("goal_point_raw")
        if raw is not None:
            # the QUANTISATION COST, readable rather than inferred: E-AG2 says
            # it should be ~free, and an arm that finds otherwise has said
            # something.
            log["anchor_free_err_m"] = float(
                ((raw.float() - tgt) * w).norm(dim=-1).mean().detach())
    else:
        logits = head_out.get("cls_logits")
        if logits is None:
            raise ValueError(
                f"objective {objective!r} needs `cls_logits`, which only the "
                f"'onehot' mode emits. A distribution over anchors cannot be "
                f"put on a head that emits none.")
        logits = logits.float()
        if logits.shape[-1] != anchors.shape[0]:
            raise ValueError(f"cls_logits K={logits.shape[-1]} != anchor table "
                             f"K={anchors.shape[0]}")
        if objective == "softanchor":
            loss = (logits.softmax(dim=-1) * d_k).sum(dim=-1).mean()
        else:                                     # the refuted CE control
            loss = torch.nn.functional.cross_entropy(logits, k_star)
        p = logits.detach().softmax(dim=-1)
        top1 = logits.detach().argmax(dim=-1)
        log |= {"anchor_top1_acc": float((top1 == k_star).float().mean()),
                "anchor_chance": 1.0 / float(anchors.shape[0]),
                # what the emitted point actually costs, in METRES, whatever
                # the objective's units -- so the CE control is comparable with
                # the default on the quantity that matters.
                "anchor_expected_err_m": float((p * d_k).sum(dim=-1).mean()),
                "anchor_argmax_err_m": float(
                    d_k.gather(1, top1[:, None])[:, 0].mean())}

    # per-family, on the EMITTED point, for every objective.
    emitted = head_out.get("goal_point")
    if emitted is not None:
        e = (emitted.float() - tgt).abs().detach()
        log |= {"anchor_err_lon_m": float(e[:, 0].mean()),
                "anchor_err_lat_m": float(e[:, 1].mean()),
                # the realised allocation of squared error between the axes --
                # the number that says whether the DECLARED axis weights put the
                # gradient where the 98.8 %-longitudinal measurement says it
                # belongs, MEASURED per step instead of assumed.
                "anchor_lon_share_sq": float(
                    (e[:, 0] * w[0, 0]).pow(2).sum()
                    / ((e * w).pow(2).sum() + 1e-12))}
    log |= {"anchor_floor_m": float(d_k.gather(1, k_star[:, None])[:, 0]
                                    .mean()),
            "anchor_loss": float(loss.detach())}
    return loss, log


# ============================================================================
# S2 — strategic goal supervision (GOAL HEADS ONLY, never a trunk loss)
# ============================================================================

#: The ``ignore_index`` the S2 CE uses. A window outside the label's validity
#: band — or in a clip no label joined — contributes NOTHING: same IGNORE
#: discipline as the arg mask, one level up.
S2_IGNORE_ID = -100
_S2_ROUTE_TO_ID = STRATEGIC_GOAL_TOKENS.index("ROUTE_TO")
_S2_BATCH_KEYS = ("g_str_id", "g_str_args", "g_str_arg_mask",
                  "a_str_id", "a_str_args", "a_str_arg_mask", "s2_valid")
#: OPTIONAL per-family abstention masks (`s2_labels.S2WindowSupervision.batch`
#: emits them only for a label set that contains an abstaining record).
#: ⛔ WHY THEY EXIST. `a_str`'s vocabulary is six POSITIVE manoeuvres — there is
#: no `NONE_ABSTAIN` in `STRATEGIC_ACTION_TOKENS` (MEASURED, v6.py:157), so a
#: builder that removes a wrong action label has nowhere to put "unknown" and
#: the row falls through to `HOLD_CORRIDOR`/`REDUCE_TO` — MEASURED on the
#: v1→v2 relabel: 80 removed `PREPARE_LANE_CHANGE` became 71 + 9 of those two.
#: Deleting a wrong label MANUFACTURED a different confident label. The mask is
#: the honest alternative; an abstain TOKEN was refused because `GoalVocabulary`
#: sizes its embedding from the tuple and the live S-W run resumes tensor-level.
#: ABSENT => the family's validity is exactly `s2_valid` (the incumbent), so
#: every pre-abstain artifact keeps a bit-identical loss.
_S2_FAMILY_MASK_KEYS = ("g_str_valid", "a_str_valid")


def _s2_family(head_out: dict, ids: Tensor, args: Tensor, mask: Tensor,
               valid: Tensor, tokens: tuple, where: str
               ) -> tuple[Tensor, Tensor, dict]:
    """One family (g_str or a_str): ``(ce, arg_l1, log)`` on VALID windows.

    CE via ``ignore_index`` (invalid rows carry :data:`S2_IGNORE_ID`); the arg
    L1 is ``|pred − label| · arg_mask`` averaged over SET slots of VALID
    windows only — a slot the label leaves unconstrained sends exactly zero
    gradient (the §1.2 IGNORE discipline), and so does a window outside the
    band. The log is PER FAMILY and carries per-token counts, never a pooled
    scalar (the four-metric-families rule applied to the term's own
    telemetry)."""
    logits = head_out["logits"].float()                        # [B, V]
    pred_args = head_out["args"].float()                       # [B, 8]
    b, v = logits.shape
    if ids.shape != (b,) or ids.dtype != torch.long:
        raise ValueError(f"{where}_id must be [{b}] long, got "
                         f"{tuple(ids.shape)} {ids.dtype}")
    if args.shape != (b, GOAL_ARG_SLOTS) or mask.shape != (b, GOAL_ARG_SLOTS):
        raise ValueError(f"{where}_args/{where}_arg_mask must be "
                         f"[{b}, {GOAL_ARG_SLOTS}], got {tuple(args.shape)} / "
                         f"{tuple(mask.shape)}")
    tgt = torch.where(valid, ids, torch.full_like(ids, S2_IGNORE_ID))
    on = tgt[valid]
    if on.numel() and (int(on.min()) < 0 or int(on.max()) >= v):
        raise ValueError(f"{where}_id out of range [0, {v}) on a valid "
                         f"window — the labels and the head disagree on the "
                         f"vocabulary size")
    n_valid = int(valid.sum())
    if n_valid == 0:
        # in the graph, contributes nothing (the o3 n_masked==0 idiom) — so a
        # batch that happens to sample no in-band window neither crashes nor
        # drops the term from the log.
        z = logits.sum() * 0.0 + pred_args.sum() * 0.0
        return z, z, {f"s2_{where}_ce": None, f"s2_{where}_acc": None,
                      f"s2_{where}_n_valid": 0,
                      f"s2_{where}_arg_l1": None, f"s2_{where}_arg_slots": 0,
                      f"s2_{where}_tok_counts": {}}
    ce = torch.nn.functional.cross_entropy(logits, tgt,
                                           ignore_index=S2_IGNORE_ID)
    m = mask.float() * valid.float()[:, None]                  # [B, 8]
    n_slots = m.sum()
    arg_l1 = ((pred_args - args.float()).abs() * m).sum() \
        / n_slots.clamp_min(1.0)
    top1 = logits.detach().argmax(dim=-1)
    counts: dict[str, int] = {}
    for t in on.tolist():
        counts[tokens[t]] = counts.get(tokens[t], 0) + 1
    return ce, arg_l1, {
        f"s2_{where}_ce": float(ce.detach()),
        f"s2_{where}_n_valid": n_valid,
        f"s2_{where}_acc": float((top1[valid] == on).float().mean()),
        f"s2_{where}_arg_l1": (float(arg_l1.detach()) if float(n_slots) > 0
                               else None),
        f"s2_{where}_arg_slots": int(n_slots),
        f"s2_{where}_tok_counts": dict(sorted(counts.items()))}


def _s2_family_valid(batch: dict, key: str, valid: Tensor) -> Tensor:
    """``s2_valid & batch[key]`` when the optional per-family mask is present.

    ⛔ ABSENT IS THE DEFAULT AND IT RETURNS ``valid`` ITSELF — not a copy, not
    an all-True AND — so a batch built by any pre-abstain producer takes a code
    path identical to the incumbent one. The mask can only ever REMOVE
    supervision (it is ANDed, never ORed): a label file cannot use it to
    supervise a window the band excluded."""
    m = batch.get(key)
    if m is None:
        return valid
    if m.shape != valid.shape:
        raise ValueError(
            f"{key} must be {tuple(valid.shape)} like s2_valid, got "
            f"{tuple(m.shape)} — a per-family abstention mask that does not "
            f"align with the window axis would silently mask the wrong rows.")
    return valid & m.bool()


def s2_goal_loss(g_out: dict, a_out: dict, batch: dict) -> tuple[Tensor, dict]:
    """The S2 term (S2_STRATEGIC_GAP.md §1.2)::

        L_s2 = CE(g_str.logits, g_str_id) + CE(a_str.logits, a_str_id)
             + |g_str.args − g_str_args|·g_str_arg_mask   (mean over set slots)
             + |a_str.args − a_str_args|·a_str_arg_mask

    all masked by ``s2_valid``, and — where the label set abstains — by the
    OPTIONAL per-family masks ``g_str_valid`` / ``a_str_valid``
    (:data:`_S2_FAMILY_MASK_KEYS`). ``g_out``/``a_out`` are the forward's
    ``out["g_str"]`` / ``out["a_str"]`` dicts; the batch keys are the
    ``s2_labels.S2WindowSupervision.batch`` contract.

    ⛔ ADMISSIBILITY. The labels are hindsight ego geometry (labels may use
    ego); at inference the heads consume only ``z_str`` — vision-derived,
    ``d_cond=0``, no situation channel (`v6.py` GoalHead). The gradient of
    this loss reaches ONLY the two heads' own parameters because their input
    ``z_str_p`` enters DETACHED under the planner cut — which is why
    ``v6_loss_step`` REFUSES this term when that cut is off: without it the
    label loss would be a TRUNK loss, and "labels supervise GOAL/
    INTERPRETATION HEADS only, never any WM trunk loss" is BINDING
    (HIERARCHY_VOCABULARY §2), with no control arm.

    ⛔ ROUTE_TO is refused HERE too (defence in depth behind the loader's
    mirror of ``s2_schema.validate()``): a hand-built batch cannot smuggle
    the gated token to the head."""
    missing = [k for k in _S2_BATCH_KEYS if k not in batch]
    if missing:
        raise ValueError(
            f"w_s2_goal > 0 but the batch is missing {missing} — an S2 term "
            f"without its labels is how a supervision weight silently "
            f"becomes 0. Pass --s2-labels (the loader builds these keys).")
    valid = batch["s2_valid"].bool()
    g_valid = _s2_family_valid(batch, "g_str_valid", valid)
    a_valid = _s2_family_valid(batch, "a_str_valid", valid)
    if g_valid.any() and bool((batch["g_str_id"][g_valid]
                               == _S2_ROUTE_TO_ID).any()):
        raise ValueError(
            "a valid S2 window carries g_str_id == ROUTE_TO, which is GATED "
            "(G1 CLOSED 0/31; no categorical arg channel on vocab_str). The "
            "loader refuses it at load; refusing here too so a hand-built "
            "batch cannot reach the head with it.")
    g_ce, g_l1, g_log = _s2_family(
        g_out, batch["g_str_id"], batch["g_str_args"], batch["g_str_arg_mask"],
        g_valid, STRATEGIC_GOAL_TOKENS, "g")
    a_ce, a_l1, a_log = _s2_family(
        a_out, batch["a_str_id"], batch["a_str_args"], batch["a_str_arg_mask"],
        a_valid, STRATEGIC_ACTION_TOKENS, "a")
    loss = g_ce + a_ce + g_l1 + a_l1
    log = {"s2_n_valid": int(valid.sum()), "s2_n_windows": int(valid.numel()),
           # ⚠️ PER FAMILY. `s2_n_valid` is the WINDOW count; these are the
           # windows that were in band and whose record still DECLINED that
           # family. Both 0 on every artifact without abstention, so the
           # incumbent log reads exactly as before plus two zeros.
           "s2_g_n_abstained": int((valid & ~g_valid).sum()),
           "s2_a_n_abstained": int((valid & ~a_valid).sum()),
           **g_log, **a_log,
           "s2_loss": float(loss.detach())}
    return loss, log


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
                 sigreg_generator: torch.Generator | None = None,
                 rollout_grad_checkpoint: bool | None = None,
                 anchor_objective: str = "metric",
                 anchor_axis_w: tuple[float, float] = ANCHOR_AXIS_W_DEFAULT
                 ) -> dict:
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
      ``g_str_id``/``g_str_args``/``g_str_arg_mask`` · ``a_str_id``/
      ``a_str_args``/``a_str_arg_mask`` · ``s2_valid``
                          S2 label keys (``s2_labels.S2WindowSupervision.
                          batch``) — REQUIRED iff ``w_s2_goal`` is in force,
                          ignored otherwise
      ``g_str_valid`` / ``a_str_valid``   [B] bool — OPTIONAL per-family
                          abstention masks; absent = the incumbent behaviour
                          (both families follow ``s2_valid``)

    Returns ``{"loss": Tensor, **components, "log": dict}``. Terms whose weight
    is 0 for this stage are SKIPPED, not multiplied by zero — a skipped term
    costs no compute and, more importantly, cannot appear in the log looking
    like it trained something.

    ⛔ ``sigreg_generator`` — REPRODUCIBILITY OF ``o6``, OPT-IN (2026-08-16).
    ``SigReg`` draws its M slice directions freshly per call, and until today it
    drew them from the GLOBAL RNG, which ``generator`` does not cover. MEASURED
    (``LOSS_DETERMINISM.md``): two identical calls with the same ``generator``
    returned S-W 3.9301 vs 3.9227 — the WHOLE discrepancy in ``o6`` (0.046874 vs
    0.039470, 18.7 %), every other term bit-identical. A globally-seeded full
    training run is unaffected; every IN-PROCESS A/B was noise-dominated, so no
    ablation of any term was attributable.

    ``None`` (default) keeps the incumbent global draw BIT-FOR-BIT — v6F S-W is
    training from this code and its loss values must not move. Pass a generator
    to make ``o6`` reproducible.

    ⚠️ **Use a SEPARATE generator from ``generator``, not the same object.**
    ``generator`` also feeds ``sample_random_deltas`` (O1) and
    ``sample_cell_block_mask`` (O3), so sharing one stream re-couples the terms:
    switching O3 off changes how many draws precede O6 and ``o6`` then moves for
    a reason that has nothing to do with the ablation — which is the confound
    this parameter exists to remove.
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
                            cfg.sigreg_free_dims, generator=sigreg_generator)
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

    # ---- ANCHOR_GOAL supervision (w_anchor, default 0.0 == absent) ----------
    # NOT nested under lambda_plan: the goal head is trainable with the planner
    # loss OFF (that is the attributable arm -- a goal that moves must be
    # attributable to its OWN objective, not to a WTA fan gradient arriving
    # through the same seam), so this reads ``plan_target`` directly.
    if w.w_anchor:
        if stack.anchor_head is None:
            raise ValueError(
                "w_anchor > 0 with cfg.anchor_goal='none' -- an anchor loss "
                "with no anchor head is how a head silently never trains. "
                "Build the stack with --anchor-goal snap_lat (and its table).")
        tgt = batch.get("plan_target")
        if tgt is None:
            raise ValueError(
                "w_anchor > 0 needs batch['plan_target'] [B, plan_steps, 2] -- "
                "the ANCHOR_GOAL label is the TRUE ego-frame displacement at "
                "the plan horizon, and a goal objective with no goal is how "
                "w_anchor silently becomes 0")
        head_out = {k[len("anchor_"):]: v for k, v in out["plan"].items()
                    if k.startswith("anchor_")}
        la, lga = anchor_goal_loss(
            head_out, tgt.float()[:, -1], stack.anchor_head.anchors,
            objective=anchor_objective, axis_w=anchor_axis_w)
        terms["anchor"] = w.w_anchor * la
        log |= lga

    # ---- S2: strategic goal supervision (w_s2_goal, default 0.0 == absent) --
    # In force only where for_stage keeps it (S-S / S-J — the stages that
    # train layer_str). Reads the heads' emitted logits/args straight off the
    # shared forward: no second head pass, no RNG, no new module.
    if w.w_s2_goal:
        if not cfg.isolate_planner_from_encoder:
            raise ValueError(
                "w_s2_goal > 0 with isolate_planner_from_encoder=False: the "
                "S2 CE/L1 reads g_str/a_str, whose input z_str_p is detached "
                "ONLY by the planner cut (v6.py `_cut`) — without it the "
                "label loss reaches adapters/encoder and becomes a TRUNK "
                "loss. 'Labels supervise GOAL/INTERPRETATION HEADS only, "
                "never any WM trunk loss' is BINDING (HIERARCHY_VOCABULARY "
                "§2); unlike --no-isolate-planner's other consumers there is "
                "NO control arm for a binding rule.")
        ls2, lg_s2 = s2_goal_loss(out["g_str"], out["a_str"], batch)
        terms["s2"] = w.w_s2_goal * ls2
        log |= lg_s2

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

def stage_gate_dict(stage: str, probes: dict, *, run: dict | None = None,
                    arm: dict | None = None) -> dict:
    """Assemble ``stage_gate.json`` from whatever probes actually ran.

    ``probes`` maps probe name -> ``{"pass": bool|None, ...}``. A required probe
    that is ABSENT or reports ``pass: None`` makes the whole gate
    **INCONCLUSIVE** (``"pass": null``), never a pass. That distinction is the
    whole mechanism: a gate that quietly reads a missing probe as satisfied is
    not a gate, and X5's rule — *a failed stage never propagates upward* — is
    only enforceable if "did not run" and "ran and passed" stay different
    words.

    ⭐ **E4 — ``arm`` makes a criterion ARM-CONDITIONAL, and only downward.**
    ``arm`` is :func:`arm_record`'s build facts. A required probe that
    :func:`probe_applies` declares NOT APPLICABLE on this arm is excluded from
    the verdict and listed in ``not_applicable_required`` **with its reason and
    with what would make it applicable** — it is never silently dropped and
    never counted as a pass. ⛔ ``arm=None`` (no record supplied) adjudicates
    EVERY criterion as applicable, which is the strict reading and today's
    behaviour byte-for-byte: forgetting to describe the arm can only make the
    gate harder to pass, never easier.

    ⛔ **The one thing this must never do is manufacture a PASS.** A stage whose
    entire required set became not-applicable would "pass" while measuring
    nothing, so that case is refused outright: ``required_effective`` empty ⇒
    **INCONCLUSIVE**, with ``vacuous_gate`` naming it. A gate with nothing left
    to check is decoration, and this programme has already found three of those.

    ⛔ **AND A SUPPLIED VERDICT ALWAYS WINS OVER THE PREDICATE.** Applicability
    answers *"can this arm produce the quantity?"* — it does **not** license
    discarding a quantity somebody actually supplied. MEASURED 2026-08-17: the
    first version of this function excluded a not-applicable probe
    unconditionally, and the incumbent
    ``test_the_whole_ladder_hands_off_through_the_WRITTEN_files`` caught it — a
    planted ``sel_gap {"pass": false}`` was silently dropped and the gate read
    **PASS on a rung that had FAILED**. That is the erasure of a FAIL, the worst
    thing a gate can do, arriving through the very mechanism meant to stop
    vacuous verdicts. ⇒ a probe present with a non-``None`` ``pass`` is
    adjudicated **regardless** of the predicate, and the contradiction between
    *"this arm cannot produce it"* and *"here is a value for it"* is surfaced in
    ``applicability_conflicts`` rather than resolved silently in either
    direction.
    """
    spec = STAGE_GATE_SPEC[stage]
    req = spec["required"]
    supplied = {p for p in req
                if p in probes and probes[p].get("pass") is not None}
    skipped: dict[str, dict] = {}
    conflicts: list[dict] = []
    for p in req:
        r = probe_applies(stage, p, arm)
        if r is None:
            continue
        if p in supplied:
            conflicts.append({
                "probe": p, "supplied_pass": probes[p].get("pass"),
                "predicate": r.get("predicate"),
                "_read": "this arm was recorded as UNABLE to produce this "
                         "criterion, yet a verdict was supplied for it. The "
                         "SUPPLIED verdict is adjudicated — a predicate never "
                         "discards a measurement — but one of the two is "
                         "wrong: either the arm record or the probe's "
                         "provenance. Establish which before quoting this "
                         "gate."})
            continue
        skipped[p] = r
    eff = [p for p in req if p not in skipped]
    missing = [p for p in eff if p not in probes]
    inconclusive = [p for p in eff
                    if p in probes and probes[p].get("pass") is None]
    failed = [p for p in eff
              if p in probes and probes[p].get("pass") is False]
    vacuous = bool(req) and not eff
    if failed:
        verdict: bool | None = False
    elif missing or inconclusive or vacuous:
        verdict = None
    else:
        verdict = True
    return {
        "stage": stage,
        "pass": verdict,
        "verdict": ("PASS" if verdict is True else
                    "FAIL" if verdict is False else "INCONCLUSIVE"),
        # ⛔ the SPEC, verbatim and unedited — a criterion is never deleted from
        # the record because one arm could not produce it.
        "required": list(req),
        "required_effective": eff,
        "not_applicable_required": list(skipped.values()),
        # ⛔ never empty-and-silent: a supplied verdict for a criterion the arm
        # cannot produce is adjudicated, AND reported as the contradiction it is.
        "applicability_conflicts": conflicts,
        "arm": arm if arm is not None else {
            "_read": "no arm record was supplied, so EVERY criterion was "
                     "adjudicated as APPLICABLE (the strict default). A real "
                     "trainer-written gate always carries one — see "
                     "run_stage_gate."},
        "vacuous_gate": ({
            "refused": True,
            "_read": "every required criterion was NOT APPLICABLE on this arm, "
                     "so a PASS here would certify nothing. Forced to "
                     "INCONCLUSIVE."} if vacuous else None),
        "sel_gap_tier": SEL_GAP_TIER_NOTE,
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
            # ⛔ the fourth reading, and it is NOT one of the three verdicts:
            # a criterion this arm cannot produce. It never contributes a pass.
            "NOT_APPLICABLE": "a required criterion this ARM cannot produce "
                              "(see not_applicable_required). It is excluded "
                              "from the verdict and its question stays "
                              "UNMEASURED — never counted as satisfied. A "
                              "PASS here certifies required_effective ONLY.",
        },
        "tier": "gate assembled from frozen-battery probes (T0/T1 per probe)",
        "_evidence_class": "MEASURED (ours) for probes present; probes listed "
                           "in missing_required were NOT RUN; probes listed in "
                           "not_applicable_required are UNMEASURABLE on this "
                           "arm and each names what would change that",
    }


def write_stage_gate(out_dir, gate: dict) -> Path:
    p = Path(out_dir) / "stage_gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gate, indent=1))
    return p


def assert_stage_precondition(stage: str, prev_gate_path=None, *,
                              allow_inconclusive: bool = False,
                              off_reason: str = "",
                              dry_run: bool = False) -> dict:
    """REFUSE to start ``stage`` unless the stage below it PASSED (X5).

    Returns the precondition report on success; raises
    :class:`GatePreconditionError` otherwise. Four refusals, all deliberate:
      * the previous gate file is MISSING -> refuse (a stage that never ran a
        gate did not pass one);
      * ``pass: false`` -> refuse, and no flag overrides it. A FAIL is a
        finding about the layer below; propagating it upward is how a defect
        gets attributed to the wrong layer three stages later;
      * ``pass: null`` (INCONCLUSIVE) -> refuse UNLESS
        ``allow_inconclusive`` AND a non-empty ``off_reason``;
      * ⛔ the gate was written by a ``--dry-run`` -> refuse a REAL launch. A
        dry-run's gate is a SMOKE artifact: no battery ran, its numbers come
        from synthetic tensors, and it exists only so the chain's own advance
        logic can be executed end-to-end on CPU. ``dry_run=True`` (set by
        :func:`dry_run`) is the only thing that accepts one, so a dry ladder
        can never license a real launch — the default is the strict one.

    ⭐ **E4: a PASS is reported WITH ITS SCOPE.** When the predecessor's
    certificate skipped a required criterion as NOT APPLICABLE, this is the
    moment an operator can still act on it, so the report carries
    ``prev_not_applicable`` and the refusal-free path still PRINTS what the
    certificate did not cover. A PASS that quietly means "everything except the
    question you care about" is how a scope error becomes a claim three stages
    later — the ``heldout``-vs-``full_set`` family, in a gate.
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
    if gate.get("_dry_run") and not dry_run:
        raise GatePreconditionError(
            f"[v6] ⛔ {p} was written by a --dry-run (\"_dry_run\": true) and "
            f"this is a REAL launch. A dry-run's gate is a smoke artifact: no "
            f"frozen-battery probe ran, and every number behind it came from "
            f"synthetic tensors. Re-run stage {prev} for real, or point "
            f"--prev-gate at the real run's stage_gate.json.")
    # ⭐ E4 — the predecessor's UNMEASURED questions, surfaced at the only
    # moment an operator can still act on them. Never a refusal: the criterion
    # was genuinely unproducible on that arm, and blocking here would re-create
    # the INCONCLUSIVE-by-construction deadlock one stage higher.
    na = list(gate.get("not_applicable_required") or [])
    if na:
        print(f"[v6] ⚠️ {prev}'s certificate does NOT cover "
              f"{[x.get('probe') for x in na]} — not applicable on that arm. "
              + " | ".join(f"{x.get('probe')}: {x.get('why_not_measured','')} "
                           f"⇒ {x.get('what_would_make_it_applicable','')}"
                           for x in na), flush=True)
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
                "off_reason": off_reason, "prev_gate": str(p),
                "prev_not_applicable": na}
    return {"stage": stage, "precondition": prev, "ok": True,
            "prev_verdict": "PASS", "prev_gate": str(p),
            "prev_not_applicable": na,
            "prev_pass_scope": (
                f"{prev} PASSED on {gate.get('required_effective', gate.get('required'))}"
                + (f"; it did NOT cover "
                   f"{[x.get('probe') for x in na]} (not applicable on that "
                   f"arm — see the predecessor's stage_gate.json)"
                   if na else ""))}


def in_spectrum_window(step: int, every: int, accum: int) -> bool:
    """Is ``step`` inside the ``accum``-step block that ENDS at the next
    emission?

    Extracted from the loop so the arithmetic is testable: the block is the
    ``accum-1`` steps preceding an emission PLUS the emission step itself, i.e.
    ``step % every`` in ``{every-accum+1, …, every-1, 0}``. Off-by-one here
    silently changes what the pooled spectrum is measured over, and the record
    would still look well-formed.

    ``accum <= 1`` is the incumbent path and never pools.
    """
    if accum <= 1 or every <= 0:
        return False
    r = step % every
    return r == 0 or r > every - accum


# ============================================================================
# X4 — per-layer spectrum monitoring (tac / str) + the o6 loss-trend guard
# ============================================================================

#: The o6 TREND baseline is the first N per-step ``o6_sigreg`` values of THIS
#: process (2 spectrum intervals at the default --spectrum-every 200), and the
#: comparison window is the most recent N. Robust medians over hundreds of
#: points, so single-step spikes cannot fire the guard. Same resume caveat as
#: the spectrum reference: a restarted process re-baselines from its restart.
X4_TREND_BASELINE_STEPS = 400
X4_TREND_CURRENT_STEPS = 100

#: The layers ``--x4-spectrum-layers`` may name. ``op`` is deliberately NOT
#: acceptable: z_op's monitor is the incumbent O6 block, is not governed by
#: the X4 flag, and cannot be turned off.
X4_ALLOWED_LAYERS = ("tac", "str")


def x4_monitor_from_args(a, cfg) -> "LayerSpectrumMonitor | None":
    """Build the per-layer monitor from the CLI, or ``None`` when disabled.

    ⚠️ RNG ISOLATION: the monitor gets its OWN generator (seed+11), not the
    O6 path's ``spec_gen`` — sharing one stream would shift the z_op bootstrap
    DIAGNOSTIC sequence whenever X4 runs first, i.e. a monitoring addition
    changing another monitor's numbers under identical flags.
    """
    raw = [s.strip() for s in str(getattr(a, "x4_spectrum_layers", "")
                                  ).split(",") if s.strip()]
    if not raw or raw == ["none"]:
        return None
    bad = [s for s in raw if s not in X4_ALLOWED_LAYERS]
    if bad:
        raise SystemExit(
            f"[v6] ⛔ --x4-spectrum-layers {bad}: only {X4_ALLOWED_LAYERS} "
            f"are X4 layers ('op' is the incumbent O6 monitor and cannot be "
            f"moved under this flag; 'none' disables)")
    dims = {"tac": cfg.d_tac, "str": cfg.d_str}
    return LayerSpectrumMonitor(
        {k: dims[k] for k in raw},
        accum=a.spectrum_accum,
        # z_tac/z_str contribute one row per WINDOW, i.e. --batch rows/step
        rows_per_step={k: int(a.batch) for k in raw},
        ci_reps=a.spectrum_ci_reps,
        generator=(torch.Generator().manual_seed(a.seed + 11)
                   if a.spectrum_ci_reps else None))


def x4_trend_record(o6_baseline: "list[float]",
                    o6_current: "list[float]") -> dict:
    """The per-layer SIGReg trend block for the X4 record.

    ``op`` carries the real verdict (:func:`sigreg_trend_verdict` over the
    per-step ``o6_sigreg`` series). ``tac`` / ``str`` carry an EXPLICIT
    not-applicable: **no per-layer SIGReg loss exists** — ``v6_loss_step``
    applies ``stack.sigreg`` to z_op only, so X4's "per-layer SIGReg" half is
    unimplemented in this trainer, and a trend guard without a loss series
    would be an invented instrument (stated per the four-families rule:
    a missing metric is declared with its reason, never silently dropped).
    """
    na = {"applicable": False,
          "reason": "no per-layer SIGReg loss exists: v6_loss_step applies "
                    "SigReg to z_op only (X4's per-layer-SIGReg half is "
                    "unimplemented in the trainer). A trend guard without a "
                    "loss series would be an invented instrument."}
    if not o6_baseline and not o6_current:
        op = {"applicable": False,
              "reason": "no o6_sigreg values this stage (w_o6 == 0 or the "
                        "term is skipped) — nothing to baseline"}
    else:
        op = dict(sigreg_trend_verdict(o6_baseline, o6_current),
                  applicable=True,
                  scope="REPORTED monitor only; gate promotion is a PI "
                        "decision (O6_ABLATION_AND_MASK_PROBE.md esc. 2)")
    return {"op": op, "tac": dict(na), "str": dict(na)}


def run_stage_gate(stack: V6Stack, stage: str, *, out_dir,
                   spectrum: dict | None = None,
                   x4_spectra: dict | None = None,
                   extra_probes: dict | None = None,
                   dry_run: bool = False) -> dict:
    """Run whatever frozen-battery entry points are IMPORTABLE here, assemble
    the gate, and write it.

    ⚠️ Rule 2 (*absence found at ONE location is not absence*) applied to the
    battery: a probe that cannot be imported is recorded with the ImportError
    text and the owning path from :data:`STAGE_GATE_SPEC`, so "n/a" always says
    WHAT was not reachable and WHERE it lives. It is never silently dropped,
    and it never counts as a pass.

    ⭐ **E4: the arm record is read off the BUILT STACK here** (:func:`arm_record`)
    and handed to :func:`stage_gate_dict`, so every trainer-written certificate
    says which criteria its own build could produce. ⛔ This is also the one
    place that can tell "not run" from "UNRUNNABLE": a not-applicable criterion
    gets that status rather than the ``not-run`` boilerplate, which otherwise
    tells an operator to go and run a battery that cannot produce the number.
    """
    probes: dict[str, dict] = dict(extra_probes or {})
    spec = STAGE_GATE_SPEC[stage]
    arm = arm_record(stack)
    for name in tuple(spec["required"]) + tuple(spec["reported"]):
        if name in probes:
            continue
        owner = spec["owners"].get(name, "?")
        na = probe_applies(stage, name, arm)
        if na is not None:
            probes[name] = {"pass": None, "status": "not-applicable",
                            "owner": owner, **na}
            continue
        probes[name] = {"pass": None, "status": "not-run",
                        "owner": owner,
                        "reason": "no artifact supplied to --gate-probes and "
                                  "this trainer does not run the battery "
                                  "in-loop (it is a separate, frozen "
                                  "instrument by design)",
                        "tier_note": (SEL_GAP_TIER_NOTE
                                      if name.startswith("sel_gap") else None)}
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
        # ⛔ The verdict travels WITH the reading, so a gate artifact can never
        # again carry an effective_rank without the ceiling that bounds it.
        # ``o6_rank_verdict`` returns INCONCLUSIVE for any reading below
        # O6_ADMISSIBLE_CEILING, which every single-batch reading is.
        probes["O6_spectrum"] = {"pass": None, "status": "reported",
                                 "owner": "tanitad.models.v6.spectrum_report",
                                 **spectrum,
                                 "verdict": o6_rank_verdict(spectrum),
                                 "reason": "rank RETENTION needs a series AND "
                                           "an admissible rank ceiling; see "
                                           "SIGREG_GATE_POWER.md — at n=48 the "
                                           "reading is bounded by 47 and the "
                                           ">= 0.8x criterion fires on noise "
                                           "9-38 % of the time"}
    if x4_spectra is not None:
        # X4: the per-layer records travel with THEIR OWN verdicts (already
        # embedded per layer by LayerSpectrumMonitor.emit), each under the
        # layer's measured ceiling/floor — never z_op's. Reported, never
        # adjudicated; INCONCLUSIVE is the expected reading until pooled.
        probes["X4_spectrum_layers"] = {
            "pass": None, "status": "reported",
            "owner": "tanitad.models.v6.LayerSpectrumMonitor",
            "layers": x4_spectra,
            "reason": "per-layer rank retention (tac ceiling_min 256 / "
                      "floor 32, str 128 / 32 — x4_layer_power.json); "
                      "verdicts are INCONCLUSIVE until pooled to the "
                      "layer's own ceiling (--spectrum-accum 33 recommended: "
                      "32 leaves tac ONE ROW short at 8 rows/step)"}
    gate = stage_gate_dict(stage, probes, arm=arm)
    gate["param_report"] = stack.param_report()
    if dry_run:
        # ⛔ STAMPED, so it can never be mistaken for — or used as — a real
        # certificate. `assert_stage_precondition` refuses it for a real launch.
        gate["_dry_run"] = True
        gate["_read"] = ("SMOKE ARTIFACT. Written by --dry-run over SYNTHETIC "
                         "tensors: no corpus, no frozen-battery probe, no "
                         "eval. It exists so a chain's advance logic can be "
                         "EXECUTED end-to-end on CPU. No number here is "
                         "quotable and it licenses no real launch.")
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


def _read_anchor_table(path) -> tuple[Tensor, list | None]:
    """Read a ``build_refc_anchors.py`` artifact -> ``(anchors, horizons)``.

    The shipped format is ``{"anchors": [K, S, 2], "horizons": [...], ...}``
    (``build_refc_anchors.main``); a bare ``[K, S, 2]`` / ``[K, 2]`` tensor is
    also accepted. ``horizons`` is returned UNCHANGED, including ``None`` --
    :meth:`AnchorGoalHead.load_anchor_table` REFUSES a ``[K, S, 2]`` table with
    no horizons rather than reading ``anchors[:, -1]``, because that index
    silently means "2.0 s" for both a 4-point and a 20-point vocabulary.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[v6] anchor table {p} does not exist")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "anchors" not in obj:
            raise SystemExit(
                f"[v6] {p} has no 'anchors' key (found {sorted(obj)[:8]}) -- "
                f"this is not a build_refc_anchors.py artifact")
        return torch.as_tensor(obj["anchors"]).float(), obj.get("horizons")
    # A BARE [K, S, 2] tensor carries no horizons, and INVENTING them (1..S) is
    # exactly the "a number rather than an error" failure the refusal exists to
    # prevent -- the two real shipped shapes are [5,10,15,20] and [1..20], and
    # guessing between them mislabels the whole corpus. ``horizons=None`` is
    # passed through so ``load_anchor_table`` refuses it by name.
    return torch.as_tensor(obj).float(), None


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
        # ---- GOAL-HEAD STRUCTURE, ALL DEFAULT-OFF ---------------------------
        # These V6Config levers existed with NO CLI path to them, which is the
        # `intent_proj` defect in the launch surface: a lever present in the
        # architecture and absent from every command that can build it. Every
        # default below reproduces the incumbent build EXACTLY -- proved per
        # tensor against the pre-change revision of v6.py in
        # tests/test_v6_anchor_loss.py, not by reading these lines.
        goal_factored=bool(getattr(a, "goal_factored", False)),
        goal_multilabel=bool(getattr(a, "goal_multilabel", False)),
        goal_cat_args=bool(getattr(a, "goal_cat_args", False)),
        # F-1: the g_str->P_T port. Default False = the incumbent build,
        # byte-identical; the chain's S-T command surface turns it on.
        tac_goal_cond=bool(getattr(a, "tac_goal_cond", False)),
        anchor_goal=getattr(a, "anchor_goal", "none"),
        n_anchors=int(getattr(a, "n_anchors", 256)),
        n_lat_bins=int(getattr(a, "n_lat_bins", 16)),
        n_agent_slots=int(getattr(a, "n_agent_slots", 8)),
        # ---- PROPOSALS / MPC / FALLBACK (2026-08-16), ALL DEFAULT-OFF -------
        # Every default reproduces the incumbent build EXACTLY (per-tensor
        # C75 proof in tests/test_v6_diffusion_mpc_fallback.py). The MPC path
        # is additionally gated by V6Config itself: it refuses to build
        # without selector='goal', so it stays INERT while SEL-1 is refused
        # (assert_selector_admissible gates every selector launch).
        proposals=getattr(a, "proposals", "query"),
        diffusion_steps=int(getattr(a, "diffusion_steps", 4)),
        diffusion_noise_rho=float(getattr(a, "diffusion_noise_rho", 0.9)),
        diffusion_hidden=int(getattr(a, "diffusion_hidden", 256)),
        diffusion_sigma_a=float(getattr(a, "diffusion_sigma_a", 2.0)),
        diffusion_sigma_k=float(getattr(a, "diffusion_sigma_k", 0.1)),
        mpc_refine=bool(getattr(a, "mpc_refine", False)),
        mpc_topk=int(getattr(a, "mpc_topk", 2)),
        mpc_steps=int(getattr(a, "mpc_steps", 3)),
        mpc_lr=float(getattr(a, "mpc_lr", 0.05)),
        mpc_roll_k=int(getattr(a, "mpc_roll_k", 0)),
        mpc_w_goal=float(getattr(a, "mpc_w_goal", 1.0)),
        mpc_w_kin=float(getattr(a, "mpc_w_kin", 0.1)),
        mpc_w_consist=float(getattr(a, "mpc_w_consist", 0.0)),
        fallback_trigger=bool(getattr(a, "fallback_trigger", False)),
        fallback_roll_k=int(getattr(a, "fallback_roll_k", 10)),
        # ---- F-18 PERCEPTION AGENT SLOTS, DEFAULT-OFF ----------------------
        # Default False reproduces the incumbent build EXACTLY (per-tensor
        # C75 proof in tests/test_v6_agent_slots.py). ⛔ The head trains in NO
        # ladder stage; the flag exists so a checkpoint can CARRY it and a
        # frozen-trunk probe can read it.
        agent_slots=bool(getattr(a, "agent_slots", False)),
        n_slot_queries=int(getattr(a, "n_slot_queries", 16)),
        slot_hidden=int(getattr(a, "slot_hidden", 256)),
        slot_depth=int(getattr(a, "slot_depth", 3)),
        slot_heads=int(getattr(a, "slot_heads", 8)),
        slot_src=getattr(a, "slot_src", "cells"),
        isolate_interp_from_encoder=not bool(
            getattr(a, "no_isolate_interp", False)),
        vit5_encoder=bool(a.vit5_encoder), n_registers=a.n_registers)
    stack = V6Stack(cfg)
    # ---- the P7 fallback calibration, installed BEFORE the first forward ----
    # Same discipline as the anchor table below: the comparator refuses to
    # fire uncalibrated, and an inadmissible band (rho below P7's gate) must
    # cost milliseconds, not a GPU-day. load_calibration REFUSES rho < 0.3 or
    # a CI including 0.
    if getattr(a, "fallback_calibration", None):
        with open(a.fallback_calibration, encoding="utf-8") as fh:
            calib = json.load(fh)
        prov = stack.fallback.load_calibration(calib)
        print(f"[v6] fallback calibration {a.fallback_calibration} -> "
              f"{json.dumps(prov)}", flush=True)
    # ---- the anchor table, installed BEFORE the first forward ---------------
    # The head REFUSES to run without one (a zero table would snap every goal
    # to the origin and still return a number), so this must land before
    # ``assert_isolation`` below -- and it must land at BUILD time, not at
    # first-batch time, so a wrong-horizon vocabulary costs milliseconds
    # instead of a GPU-day. Same discipline as the --gate-probes preflight.
    if getattr(a, "anchor_table", None):
        prov = stack.anchor_head.load_anchor_table(
            *_read_anchor_table(a.anchor_table), dt=cfg.dt)
        print(f"[v6] anchor table {a.anchor_table} -> {json.dumps(prov)}",
              flush=True)
    rep = stack.assert_param_budget()
    print(f"[v6] params {rep['total']/1e6:.2f} M / budget "
          f"{rep['budget']/1e6:.0f} M · arm {rep['arm']} · per-group "
          f"{ {k: round(v/1e6, 2) for k, v in rep['per_group'].items()} }",
          flush=True)
    iso = stack.assert_isolation(batch_size=1,
                                 strict=not a.no_isolate_planner
                                 and not a.no_isolate_uplink
                                 # F-18's mis-wired arm belongs on the SAME
                                 # list: a declared control must be buildable,
                                 # or it is not a control.
                                 and not bool(getattr(a, "no_isolate_interp",
                                                      False)))
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


def synthetic_s2_batch(batch: int = 2, *, seed: int = 0,
                       valid_frac: float = 0.75, device=None) -> dict:
    """Synthetic S2 label keys — the ``--dry-run`` stand-in for the real join.

    Shapes and dtypes are EXACTLY ``s2_labels.S2WindowSupervision.batch``'s,
    including invalid rows (id ``S2_IGNORE_ID``, zero args/mask), so a dry-run
    of an S-S launch with ``--w-s2-goal`` exercises the same loss path the pod
    will — masking included. Draws only VALID token ids and never ROUTE_TO
    (the gated token would trip the loss's own refusal, correctly)."""
    g = torch.Generator().manual_seed(seed)
    b = int(batch)
    n_g, n_a = len(STRATEGIC_GOAL_TOKENS), len(STRATEGIC_ACTION_TOKENS)
    valid = torch.rand(b, generator=g) < float(valid_frac)
    if b and not bool(valid.any()):
        valid[0] = True                    # a dry step should exercise n>0
    g_id = torch.randint(n_g - 1, (b,), generator=g)
    g_id = g_id + (g_id >= _S2_ROUTE_TO_ID).long()      # skip the gated token
    g_mask = (torch.rand(b, GOAL_ARG_SLOTS, generator=g) < 0.25).float() \
        * valid[:, None]
    a_mask = (torch.rand(b, GOAL_ARG_SLOTS, generator=g) < 0.25).float() \
        * valid[:, None]
    out = {
        "g_str_id": torch.where(valid, g_id,
                                torch.full_like(g_id, S2_IGNORE_ID)),
        # unset slots carry 0.0 — the loader-enforced record convention, kept
        # here so the stand-in matches the real contract byte for byte.
        "g_str_args": torch.randn(b, GOAL_ARG_SLOTS, generator=g) * g_mask,
        "g_str_arg_mask": g_mask,
        "a_str_id": torch.where(valid,
                                torch.randint(n_a, (b,), generator=g),
                                torch.full((b,), S2_IGNORE_ID,
                                           dtype=torch.long)),
        "a_str_args": torch.randn(b, GOAL_ARG_SLOTS, generator=g) * a_mask,
        "a_str_arg_mask": a_mask,
        "s2_valid": valid,
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def dry_run(a, stack: V6Stack | None = None) -> dict:
    """Build everything, run ``--dry-steps`` (default 2) synthetic CPU steps,
    write ``config.json`` + ``dry_run.json``.

    THE POINT: verify the pod launch BEFORE the corpus is mounted. On a pod the
    checkout drifts, ``git fetch`` HANGS (no credentials), and a launch from a
    stale ``stack/`` resurrects fixed bugs — so the runbook step is *ship the
    files, then run this, then launch*, and this must exercise the real loss
    assembly, not just an import.

    ⛔ AND IT MUST EXERCISE THE LADDER SEAMS, WHICH IT DID NOT. Until
    2026-08-16 this function ignored ``--prev-gate`` and ``--init-from``
    entirely: a dry-run of S-T printed "dry-run OK" while the real launch of the
    same command died on ``Missing key(s): cand_score.*`` (the init-from launch
    blocker). A pre-launch verifier that skips the two flags the staged protocol
    is MADE of is structurally incapable of catching the class of failure it
    exists to catch — C13, in the instrument that is supposed to be the guard.
    Now, when either flag is supplied it is REALLY exercised: the X5
    precondition is adjudicated by :func:`assert_stage_precondition` and the
    predecessor is REALLY loaded by :func:`load_stage_init`. When a flag is
    absent, ``dry_run.json`` says so in its own report rather than leaving the
    reader to assume it was checked.
    """
    stack = stack or build_stack_from_args(a)
    out_dir = Path(a.out)
    # ⛔ A dry-run must never share a directory with a real run: it writes
    # config.json and a stage_gate.json, and clobbering a live run's records
    # with synthetic ones is the shared---out defect in miniature.
    if (out_dir / "ckpt.pt").exists():
        raise SystemExit(
            f"[v6] ⛔ --dry-run --out {out_dir} already contains ckpt.pt, i.e. "
            f"a REAL run lives there. A dry-run writes config.json and a "
            f"stage_gate.json and would overwrite that run's records with "
            f"synthetic ones. Point --out at a scratch directory.")
    out_dir.mkdir(parents=True, exist_ok=True)
    # ---- the ladder seams, REALLY exercised ---------------------------------
    pre = {"exercised": False,
           "_read": "--prev-gate was NOT supplied, so the X5 precondition was "
                    "NOT exercised by this dry-run."}
    if getattr(a, "prev_gate", None):
        pre = assert_stage_precondition(
            a.stage, a.prev_gate,
            allow_inconclusive=bool(getattr(a, "allow_inconclusive_gate",
                                            False)),
            off_reason=getattr(a, "gate_off_reason", "") or "",
            dry_run=True)
        pre["exercised"] = True
        print(f"[v6 dry] precondition OK · {json.dumps(pre)}", flush=True)
    device = "cpu"
    stack = stack.to(device)
    init_report = {"exercised": False,
                   "_read": "--init-from was NOT supplied, so this dry-run "
                            "stepped RANDOM weights. It proves the launch "
                            "assembles; it proves nothing about the lineage."}
    if getattr(a, "init_from", None):
        init_report = load_stage_init(stack, a.init_from, stage=a.stage)
        init_report["exercised"] = True
        print(f"[v6 dry] init-from OK · introduced="
              f"{init_report['introduced_keys']} · trunk_md5="
              f"{init_report['trunk_md5_after_load'][:12]}", flush=True)
    freeze = apply_stage_freeze(stack, a.stage)
    weights = _weights_from_args(a)
    w_stage_dry = weights.for_stage(a.stage)
    # ---- the S2 label artifact, REALLY exercised when supplied --------------
    # Same rule as --prev-gate/--init-from above: a pre-launch verifier that
    # skips a flag the launch carries is structurally incapable of catching
    # that flag's failure class. The join needs a corpus and is NOT exercised
    # here — dry_run.json says so instead of leaving it to be assumed.
    s2_report = {"exercised": False,
                 "_read": "--s2-labels was NOT supplied; the S2 loss path "
                          "runs on synthetic keys when w_s2_goal is in "
                          "force, and the loader was NOT exercised."}
    if getattr(a, "s2_labels", None):
        from s2_labels import load_s2_labels
        s2_report = load_s2_labels(a.s2_labels).report() | {
            "exercised": True,
            "join": "NOT exercised (dry-run has no corpus) — load + "
                    "validation only"}
        print(f"[v6 dry] s2-labels OK · {s2_report['n_records']} records · "
              f"g_str census {s2_report['token_census_records']['g_str']}",
              flush=True)
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
        if w_stage_dry.w_s2_goal:
            b |= synthetic_s2_batch(a.dry_batch, seed=a.seed + step,
                                    device=device)
        dk, da = sample_random_deltas(a.dry_batch, gen, a.rand_dkappa_max,
                                      a.rand_daccel_max)
        L = v6_loss_step(stack, b, stage=a.stage, weights=weights, o1_k=o1_k,
                         o5_k=o5_k, o5_mode=a.o5_mode, o3_mode=a.o3_mode,
                         o3_blocks=a.o3_blocks,
                         o3_block_hw=(a.o3_block_h, a.o3_block_w),
                         o3_band_rows=a.o3_band_rows, o2_tau_s=a.o2_tau_s,
                         dkappa=a.dkappa, daccel=a.daccel, rand_dk=dk,
                         rand_da=da, generator=gen,
                         anchor_objective=getattr(a, "anchor_objective",
                                                  "metric"),
                         anchor_axis_w=tuple(getattr(
                             a, "anchor_axis_w", ANCHOR_AXIS_W_DEFAULT)))
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
    if init_report.get("exercised"):
        cfg_json["init"] = init_report
    (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=1))
    # The gate this dry stage hands to the next one. It is assembled by the
    # REAL `run_stage_gate`, so it comes out INCONCLUSIVE exactly as a real
    # gate would with no battery artifact — the dry ladder therefore advances
    # only through the RECORDED --allow-inconclusive-gate override, never
    # through a fabricated PASS.
    gate = run_stage_gate(stack, a.stage, out_dir=out_dir, spectrum=spec,
                          extra_probes=_load_gate_probes(
                              getattr(a, "gate_probes", None)),
                          dry_run=True)
    result = {
        "mode": "dry-run", "device": device, "steps": rows,
        "elapsed_s": round(time.time() - t0, 2),
        "freeze": freeze, "isolation": iso, "spectrum_smoke": spec,
        "param_report": stack.param_report(),
        "n_trainable_tensors": len(trainable),
        "precondition": pre,
        "init": init_report,
        "s2_labels": s2_report,
        "gate_verdict": gate["verdict"],
        "_read": "synthetic tensors — NO corpus. This proves the launch "
                 "assembles and steps, and (when --prev-gate/--init-from were "
                 "supplied) that the ladder seams adjudicate and load; it "
                 "proves NOTHING about driving. No number here is quotable.",
        "_evidence_class": "MEASURED (ours; synthetic smoke)",
    }
    (out_dir / "dry_run.json").write_text(json.dumps(result, indent=1))
    print(f"[v6] dry-run OK -> {out_dir}/dry_run.json + config.json "
          f"+ stage_gate.json ({gate['verdict']}, _dry_run)", flush=True)
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
        w_select=a.w_select, w_anchor=float(getattr(a, "w_anchor", 0.0)),
        w_s2_goal=float(getattr(a, "w_s2_goal", 0.0)),
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
    # ⛔ ...and the LINEAGE of whatever it found, also before anything
    # expensive. `load_resume` sits after episode selection, dataset windowing
    # and the O4 saliency pass over every window in the corpus; a wrong-stage
    # resume discovered there has already paid for all of it.
    if rg["mode"] == "resume":
        rg["ckpt"] = assert_resume_lineage(rg["from"], stage=a.stage)
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
    # ⛔ E2, second lock — on the REAL objects, and still BEFORE the corpus.
    # `_preflight_subframe` answers this from args alone in milliseconds; this
    # compares the frame the DATA will actually be delivered at against the
    # frame the ENCODER was actually built for. Two locks because the two are
    # different objects in this trainer (see `subframe_desync`) and the failure
    # they prevent otherwise surfaces at the first forward, i.e. after the
    # corpus has mounted and the O4 saliency pass has run.
    enc_hw = tuple(int(x) for x in stack.cfg.encoder.image_hw())
    got_hw = (int(model_frame.height), int(model_frame.width))
    if got_hw != enc_hw:
        raise SystemExit(
            f"[v6] ⛔ the DATA frame {got_hw[0]}x{got_hw[1]} and the ENCODER "
            f"frame {enc_hw[0]}x{enc_hw[1]} disagree. In this trainer "
            f"--v2-subframe moves the data (it is applied to the flagship-v4 "
            f"EVAL config) while the encoder was sized from --frame-h/"
            f"--frame-w by build_stack_from_args — so the first forward would "
            f"raise `encoder input is {got_hw} but the config declares "
            f"{enc_hw}`. ⇒ drop --v2-subframe, or declare --frame-h "
            f"{got_hw[0]} --frame-w {got_hw[1]} (a DIFFERENT model, which "
            f"cannot --init-from a {enc_hw[0]}x{enc_hw[1]} checkpoint).")
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

    # ---- S2: the strategic-goal label join (S-S/S-J; default absent) --------
    # Loaded AFTER the corpus so the join is over the REAL episode ids, and
    # BEFORE the optimiser so a dead join refuses in seconds, not after the
    # O4 pass over every window. The report lands in config.json — it is the
    # raw material of the S-S gate's goal-provenance audit (label provenance
    # census + the disjointness verdict, per this run's own load).
    s2_sup = None
    s2_cfg: dict | None = None
    if a.s2_labels and w_stage.w_s2_goal:
        from s2_labels import load_s2_labels
        s2_set = load_s2_labels(a.s2_labels)
        s2_sup = s2_set.supervision(train_eps,
                                    window=stack.cfg.predictor.window,
                                    dt=a.dt, index=ds_train.index)
        s2_cfg = {"labels": s2_set.report(), "join": s2_sup.report(),
                  "w_s2_goal_in_force": w_stage.w_s2_goal}
        print(f"[v6] S2 {json.dumps(s2_cfg['join'])}", flush=True)
        if s2_sup.n_matched_episodes == 0:
            raise SystemExit(
                f"[v6] ⛔ --s2-labels {a.s2_labels} joined ZERO of "
                f"{s2_sup.n_episodes} training episodes — with --w-s2-goal "
                f"{w_stage.w_s2_goal} the term would be advertised in every "
                f"log row and never fire (the exact silent-never-fires "
                f"failure the clip index exists to prevent). Wrong corpus "
                f"for these labels, or a stale manifest without stable ids.")
        if s2_sup.n_windows_in_band == 0:
            raise SystemExit(
                f"[v6] ⛔ --s2-labels joined {s2_sup.n_matched_episodes} "
                f"episodes but ZERO windows fall inside the validity band "
                f"(t0={s2_set.t0_s}s ± {s2_set.band}) — the windowing "
                f"(window={stack.cfg.predictor.window}, max_horizon={max_h}) "
                f"never reaches the label's decision time. The term would "
                f"never fire; refusing instead.")

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
        start_step = load_resume(stack, opt, rg["from"], stage=a.stage)
        for _ in range(start_step):
            sched.step()                       # replay the LR schedule
        # ⛔ A RESUME OVERWRITES EVERY WEIGHT --init-from JUST LOADED, and both
        # flags are legal together — `supervise_run.sh` replays the command it
        # captured at startup, so the relaunch that RESUMES still carries the
        # --init-from that seeded the run. MEASURED 2026-08-16: config.json
        # recorded `init.trunk_md5_after_load = fbce009a…` while the trunk
        # actually in the model was `326034884…`, and nothing warned. That is a
        # run row naming an ancestor the run is not standing on — the exact
        # failure MODEL_REGISTRY.md exists to prevent. The init report is
        # therefore SUPERSEDED here, in place, with the truth.
        if init_report.get("init_from"):
            print(f"[v6] ⚠️  --init-from {init_report['init_from']} was "
                  f"SUPERSEDED by the resume — every weight it loaded has been "
                  f"overwritten from {rg['from']}. The lineage of this run is "
                  f"the checkpoint, not the init.", flush=True)
        init_report = supersede_init_on_resume(init_report, rg["from"])
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
    if s2_cfg is not None:
        cfg_json["s2"] = s2_cfg
    (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=1))
    log_path = out_dir / "train_log.jsonl"
    fh = open(log_path, "a")
    fh.write(json.dumps({"run_start": cfg_json}) + "\n")
    fh.flush()

    history: list[dict] = []
    spectrum_last: dict | None = None
    spectrum_ref: dict | None = None
    # ⚠️ OPT-IN, and deliberately so: --spectrum-accum defaults to 1, which
    # leaves ``spec_acc`` None and the emission path exactly what it was. The
    # live v6F S-W run resumes from an argv that carries neither flag.
    spec_acc = (SpectrumAccumulator(capacity=a.spectrum_accum,
                                    block=stack.cfg.predictor.window)
                if a.spectrum_accum > 1 else None)
    # A DEDICATED generator for the bootstrap, so switching the CI on cannot
    # consume the global stream and move the run's loss (the exact failure the
    # loss-determinism stream just fixed on the SigReg side).
    spec_gen = (torch.Generator().manual_seed(a.seed + 7)
                if a.spectrum_ci_reps else None)
    # ---- X4: per-layer (tac/str) spectrum monitor + the o6 trend series ----
    # ADDITIVE ONLY: new record keys in train_log.jsonl; no tensor, no loss,
    # no state_dict, no RNG on the default path (its generator exists only
    # when the CI is on, and is its OWN stream — see x4_monitor_from_args).
    # z_op's O6 block below is untouched and not governed by the X4 flag.
    x4_mon = x4_monitor_from_args(a, stack.cfg)
    x4_last: dict | None = None
    o6_trend_base: list[float] = []
    o6_trend_cur: deque = deque(maxlen=X4_TREND_CURRENT_STEPS)
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
        # ⛔ `or w_stage.w_anchor`: the ANCHOR_GOAL label is the SAME tensor's
        # endpoint, and the anchor objective is deliberately runnable with
        # λ_plan OFF (that is the attributable arm). Without this the target
        # would be absent exactly when the anchor loss is the only planner
        # term, and `v6_loss_step` would refuse mid-run instead of training.
        if w_stage.lambda_plan or w_stage.w_anchor:
            batch["plan_target"] = gt_ego_waypoints(
                b["pose_last"].float(), b["future_poses"].float(),
                tuple(range(1, stack.cfg.plan_steps + 1)))
        if s2_sup is not None:
            # the S2 label keys ride the SAME sampled indices as the frames —
            # the join was precomputed per episode, so this is O(batch).
            batch |= {kk: v.to(device)
                      for kk, v in s2_sup.batch(idx).items()}
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
                                 a, "rollout_grad_checkpoint"),
                             anchor_objective=getattr(a, "anchor_objective",
                                                      "metric"),
                             anchor_axis_w=tuple(getattr(
                                 a, "anchor_axis_w", ANCHOR_AXIS_W_DEFAULT)))
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.clip)
        opt.step()
        sched.step()
        stack.ema_update()

        # ---- O6's standing spectrum monitor ---------------------------------
        # ⛔ THE PER-BATCH READING CANNOT RESOLVE RANK, and the record now says
        # so itself. The tensor is [B*W, d_op] = 48 x 2048 on the live run, so
        # a centred covariance built from it has rank <= 47: "15 of 2048" is
        # 15 of 47. MEASURED (SIGREG_GATE_POWER.md): at n=48 an isotropic
        # d=2048 population reads 46.86 and a 7.3x-collapsed one still reads
        # 22.6, and the >= 0.8x criterion fires on NOTHING between 9 % and 38 %
        # of the time. --spectrum-accum pools consecutive steps to lift the
        # ceiling; it defaults to 1, which is byte-for-byte the incumbent path.
        if spec_acc is not None and in_spectrum_window(
                step, a.spectrum_every, a.spectrum_accum):
            spec_acc.push(L["out"]["z_op_win"])
        # ---- X4: the SAME pooling window, per layer -------------------------
        # o6 trend series: two bounded float lists, appended per step — the
        # loss value is already computed, so the guard costs one append.
        _o6v = L["log"].get("o6_sigreg")
        if _o6v is not None:
            if len(o6_trend_base) < X4_TREND_BASELINE_STEPS:
                o6_trend_base.append(float(_o6v))
            o6_trend_cur.append(float(_o6v))
        if x4_mon is not None and in_spectrum_window(
                step, a.spectrum_every, a.spectrum_accum):
            x4_mon.push({"tac": L["out"]["z_tac"],
                         "str": L["out"]["z_str"]})
        if step % a.spectrum_every == 0:
            zw = L["out"]["z_op_win"].detach().float()
            spectrum_last = spectrum_report(
                zw.reshape(-1, zw.shape[-1]), ci_reps=a.spectrum_ci_reps,
                block=(stack.cfg.predictor.window if a.spectrum_ci_reps else 1),
                generator=spec_gen)
            rec_s = {"step": step, "spectrum": spectrum_last}
            if spec_acc is not None and len(spec_acc):
                rec_s["spectrum_pooled"] = spec_acc.report(
                    ci_reps=a.spectrum_ci_reps, generator=spec_gen)
                rec_s["o6_verdict"] = o6_rank_verdict(
                    rec_s["spectrum_pooled"], spectrum_ref)
                if spectrum_ref is None and rec_s["spectrum_pooled"][
                        "rank_admissible"]:
                    # the phase-start reference for clause 2, taken at the
                    # first ADMISSIBLE pooled reading of the phase.
                    # ⚠️ NOT carried across a resume: a restarted process takes
                    # a fresh reference, so its retention is measured from the
                    # restart, not from the phase start. The verdict says which
                    # step the reference came from via the record it embeds —
                    # read it, do not assume the phase start.
                    spectrum_ref = dict(rec_s["spectrum_pooled"],
                                        ref_step=step)
            # ---- X4: per-layer records + the o6 trend guard -----------------
            # ADDITIVE KEY ("x4") in the same emission record. Each layer's
            # verdict runs under ITS OWN measured ceiling/floor (tac 256/32,
            # str 128/32) — z_op's 1024/64 stays on the incumbent keys above.
            if x4_mon is not None:
                x4_last = x4_mon.emit({"tac": L["out"]["z_tac"],
                                       "str": L["out"]["z_str"]}, step=step)
                rec_s["x4"] = {"layers": x4_last,
                               "sigreg_trend": x4_trend_record(
                                   o6_trend_base, list(o6_trend_cur))}
            fh.write(json.dumps(rec_s) + "\n")
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
                               f"divided by the resumed step number)",
                # ⛔ THE ONLY ADMISSIBLE MEMORY PROBE ON THE JETSON THOR.
                # MEASURED 2026-08-03: on unified memory `mem_get_info` read
                # 3.4 GB free with 60 GB allocated AND written, `free` /
                # `tegrastats` showed 106 GB "used" on an idle box, and
                # VmRSS read 0.62 GB against 24 GB — wrong in BOTH directions.
                # An in-process counter is the only one that answers the
                # question, so the trainer logs it rather than leaving an
                # operator to reach for a probe that reports the wrong scope.
                "cuda_max_mem_gb": (
                    round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                    if dev_type == "cuda" else None),
                "cuda_max_mem_note":
                    "torch.cuda.max_memory_allocated(), peak since process "
                    "start. ⛔ On Thor do NOT cross-check it against "
                    "mem_get_info/free/tegrastats/VmRSS — all four misreport "
                    "on unified memory (CLAUDE.md, MEASURED 2026-08-03)"}
            history.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {json.dumps(rec)}", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            # ⛔ X2 SEAM DUMP — DEFAULT-OFF, and the ONLY thing that banks the
            # 60-step plan. F-16's probe (taniteval/tools/seam_probe.py) is
            # built, self-tested and has produced ZERO real-arm numbers
            # because `emit()`'s output lived only inside the forward.
            # ⚠️ ZERO EXTRA GPU: `L["out"]["plan"]` is ALREADY COMPUTED for
            # this step's loss — this copies it to CPU at the checkpoint
            # boundary, never per step, and never re-runs the emission.
            # ⚠️ S-W BANKS NOTHING: the emission head is at its zero-init, so
            # the plan is all-zero and the probe would (correctly) return
            # DEGENERATE. `seam_dump_from_plan` refuses it by default, which
            # is why this is a NOTE and not a crash — the dump is for S-T and
            # later. The live v6F S-W run is exactly the refused case.
            if getattr(a, "dump_seam_plan", None):
                # ⛔ THE IMPORT MUST NOT LIVE INSIDE THE `try` WHOSE `except`
                # NAMES ITS SYMBOL. Found 2026-08-18 by the Thor closure audit.
                # If `from taniteval.seam_dump import SeamDumpError, …` raises
                # ImportError, Python then EVALUATES `except SeamDumpError` —
                # which is unbound — and the resulting UnboundLocalError
                # PROPAGATES OUT OF THE WHOLE `try` STATEMENT. The broad
                # `except Exception` below is NEVER REACHED. Measured: the
                # exception escapes as `UnboundLocalError: cannot access local
                # variable 'SeamDumpError'`.
                # ⇒ This block sits immediately before `_save_ckpt`, so the
                # failure KILLS THE TRAINER AT A CHECKPOINT BOUNDARY — and it
                # fires exactly when `taniteval` is off PYTHONPATH, which is the
                # live run's own configuration. S-W is unaffected only because
                # the chain emits `--dump-seam-plan` on S-T/S-S/S-J and not on
                # S-W; S-T would have hit it.
                # ⇒ The import is now its own guarded step, so a missing
                # optional module degrades to a printed note, which is what the
                # comment below always claimed the code did.
                try:
                    from taniteval.seam_dump import (
                        SeamDumpError, save_seam_dump, seam_dump_from_plan)
                except Exception as e:                        # noqa: BLE001
                    print(f"[v6 seam] unavailable at {step} "
                          f"({type(e).__name__}: {e}) — training continues",
                          flush=True)
                    SeamDumpError = seam_dump_from_plan = None  # noqa: N806
                if seam_dump_from_plan is not None:
                  try:
                    d = seam_dump_from_plan(
                        L["out"]["plan"],
                        eids=b["episode_id"] if "episode_id" in b
                        else range(len(v0)),
                        tier="T1", arm=f"{out_dir.name}@{step}",
                        gt=batch.get("plan_target"),
                        dt=1.0 / float(getattr(a, "fps", 10) or 10),
                        allow_degenerate=bool(
                            getattr(a, "dump_seam_plan_degenerate", False)))
                    p = save_seam_dump(
                        d, Path(a.dump_seam_plan) / f"seam_{step:06d}.pt")
                    print(f"[v6 seam] banked {p}", flush=True)
                  except SeamDumpError as e:
                    print(f"[v6 seam] NOT banked at {step}: {e}", flush=True)
                  except Exception as e:                      # noqa: BLE001
                    # ⛔ A DIAGNOSTIC MUST NEVER KILL A RUN. This is the
                    # analysis-time-refusal trap inverted: there, an optional
                    # import destroyed a finished run's output; here the whole
                    # block is optional and the training is the thing that
                    # matters. It says so loudly and continues.
                    print(f"[v6 seam] dump FAILED at {step} "
                          f"({type(e).__name__}: {e}) — continuing",
                          flush=True)
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
                          x4_spectra=x4_last,
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


def read_ckpt_provenance(ckpt_path) -> dict:
    """Read a checkpoint's LINEAGE without materialising its tensors.

    ``mmap=True`` maps the storages instead of reading them, so this costs
    milliseconds on a 3.5 GB ``ckpt.pt`` (MEASURED: 0.005 s vs 0.024 s on a
    52 MB file, and the difference is the tensor bytes, which are never
    touched). That matters because this runs on pod2, which is RAM-bound
    (~54/55 GB cgroup) — a metadata read that transiently allocates the whole
    checkpoint would be a memory-pressure event to answer a one-word question.

    Never raises on a bad file: an unreadable checkpoint is REPORTED as such
    (``readable: False``) so the caller can refuse with a diagnosis instead of
    an opaque pickle traceback.
    """
    p = Path(ckpt_path)
    out = {"path": str(p), "readable": False, "stage": None, "step": None,
           "has_opt": False, "weights_only_snapshot": False, "error": None}
    if not p.exists():
        out["error"] = "does not exist"
        return out
    try:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False,
                            mmap=True)
        except (RuntimeError, ValueError, TypeError):
            # legacy (non-zip) serialisation cannot be mmapped
            ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                       # corrupt, truncated, not a ckpt
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if not isinstance(ck, dict):
        out["error"] = f"top level is {type(ck).__name__}, not a dict"
        return out
    # ⚠️ the fp16 snapshot is a DIFFERENT shape: {"model", "_meta",
    # "_fp16_weights_only"} — its state lives under "model" and its step/config
    # under "_meta". Reading it as a ckpt.pt is how `--init-from <snapshot>`
    # came back as "not a valid predecessor: geometry mismatch", blaming the
    # architecture for a container it simply did not unwrap.
    snap = bool(ck.get("_fp16_weights_only")) or (
        "model" in ck and "stack" not in ck and "_meta" in ck)
    meta = (ck.get("_meta") or {}) if snap else ck
    cfg = meta.get("config") or {}
    out.update(readable=True, weights_only_snapshot=snap,
               has_opt=("opt" in ck),
               stage=(cfg.get("stage") if isinstance(cfg, dict) else None),
               step=(int(meta["step"]) if isinstance(meta.get("step"), int)
                     else None))
    return out


def assert_resume_lineage(ckpt_path, *, stage: str) -> dict:
    """⛔ Refuse a ``--resume auto`` onto a checkpoint that is not this run's.

    The three requirements are :data:`RESUME_CONTRACT`; each refusal quotes the
    one it violates, so the message explains the ladder rather than the
    optimiser. Runs BEFORE the corpus build — a wrong-stage resume used to die
    at ``load_resume``, which sits after episode selection, dataset windowing
    and the O4 saliency pass over every window in the corpus.
    """
    prov = read_ckpt_provenance(ckpt_path)
    if not prov["readable"]:
        raise ResumeLineageError(
            f"[v6] ⛔ --resume auto found {prov['path']} but could not read it "
            f"({prov['error']}). Refusing to resume from a checkpoint whose "
            f"lineage cannot be established. Move it aside, or point --out "
            f"somewhere else.")
    if prov["weights_only_snapshot"] or not prov["has_opt"]:
        raise ResumeLineageError(
            f"[v6] ⛔ --resume auto found {prov['path']}, which carries NO "
            f"optimiser state"
            + (" (it is an fp16 weights-only snapshot)"
               if prov["weights_only_snapshot"] else "")
            + f".\n  {RESUME_CONTRACT['has_optimiser']}\n"
            f"  ⇒ launch this as a FRESH run with --init-from {prov['path']} "
            f"(and --out somewhere without a ckpt.pt), which starts at step 0 "
            f"on purpose instead of inheriting a step this file cannot back.")
    if prov["stage"] is None:
        raise ResumeLineageError(
            f"[v6] ⛔ --resume auto found {prov['path']} with no stage label "
            f"(config.stage is absent).\n  {RESUME_CONTRACT['labelled']}")
    if prov["stage"] != stage:
        raise ResumeLineageError(
            f"[v6] ⛔ --resume auto found {prov['path']} written by stage "
            f"{prov['stage']!r} at step {prov['step']}, but this run is stage "
            f"{stage!r}.\n  {RESUME_CONTRACT['same_stage']}\n"
            f"  ⚠️ Nothing downstream would have caught this reliably: the "
            f"state_dict load SUCCEEDS across the ladder (every stage saves "
            f"the whole stack), and the only accidental barrier — the "
            f"optimiser's param-group size — holds solely because the stages "
            f"happen to train different numbers of tensors "
            f"(S-W 240 · S-T 80 · S-S 54 · S-J 374, MEASURED). The run would "
            f"have adopted step {prov['step']} and replayed the LR schedule to "
            f"the wrong point.\n"
            f"  ⇒ this is an --init-from, not a resume. Point --out at a "
            f"fresh directory and pass --init-from {prov['path']}.")
    return prov | {"stage_checked": stage,
                   "_evidence_class": "MEASURED (ours; the ckpt's own config)"}


def load_resume(stack: V6Stack, opt, ckpt_path, *, stage: str | None = None
                ) -> int:
    """Restore stack + optimiser + step from ``ckpt.pt``. Returns the step to
    continue FROM (0 if nothing to resume).

    ``stage`` is DEFENCE IN DEPTH — :func:`assert_resume_lineage` is the early
    gate and ``train`` calls it first, but this function is importable and a
    caller that names its stage gets the same refusal here. A caller that does
    NOT name one gets the old, unchecked behaviour, exactly as
    :func:`load_stage_init` treats its allowance: a check must be asked for,
    never inherited by default and never assumed to have run elsewhere.
    """
    if stage is not None:
        assert_resume_lineage(ckpt_path, stage=stage)
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    if "stack" not in ck:
        raise ResumeLineageError(
            f"[v6] ⛔ {ckpt_path} has no 'stack' key (found {sorted(ck)[:6]}). "
            f"A weights-only fp16 snapshot stores its state under 'model' and "
            f"is an --init-from artifact, not a resume point — see "
            f"ops/ckpt_fp16_snapshot.py. {RESUME_CONTRACT['has_optimiser']}")
    stack.load_state_dict(ck["stack"], strict=True)
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    return int(ck.get("step", 0))


def supersede_init_on_resume(init_report: dict, resumed_from) -> dict:
    """The run's recorded lineage after a resume overrode an ``--init-from``.

    ⛔ MEASURED 2026-08-16, and it was SILENT. ``train`` runs
    :func:`load_stage_init` first and :func:`load_resume` afterwards, so when
    both flags are present the resume overwrites every weight the init loaded —
    while ``config.json`` still carried the init's ``trunk_md5_after_load``.
    The two hashes measured ``fbce009a…`` (recorded) vs ``326034884…``
    (actually in the model): the run row named an ancestor the run was not
    standing on, which is the failure ``MODEL_REGISTRY.md`` exists to prevent.

    ⚠️ Refusing the COMBINATION would be the wrong fix. ``supervise_run.sh``
    replays the ``TRAIN_CMD`` it captured at supervisor startup, so the
    relaunch that resumes necessarily still carries the ``--init-from`` that
    seeded the run. The flag pair is normal operation; the lying record was the
    defect. ⇒ keep the init report, but demote it to what it is.
    """
    if not init_report.get("init_from"):
        return init_report
    return {
        "init_from": None,
        "superseded_by_resume": init_report | {
            "_status": "OVERWRITTEN — these weights are NOT in the model; "
                       "--resume auto ran after --init-from and replaced "
                       "every one of them. The md5 below describes the INIT, "
                       "not this run."},
        "resumed_from": str(resumed_from),
        "_evidence_class": "MEASURED (ours; this run's launch order)",
    }


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

    # ⛔ UNWRAP THE fp16 SNAPSHOT. ``ops/ckpt_fp16_snapshot.py`` writes
    # ``{"model", "_meta", "_fp16_weights_only"}`` and its docstring states the
    # snapshot "is enough for the P-battery, any eval, and --init-from". It was
    # NOT: this function looked only for ``"stack"``, fell through to the
    # wrapper dict, and refused with *"not a valid predecessor ... missing:
    # act_head_lat.arg_head.bias, ..."* — 400+ keys blamed on a GEOMETRY
    # MISMATCH when the real cause was an unopened container. MEASURED
    # 2026-08-16. That message is worse than a crash: it accuses the
    # architecture, which is where an operator would then go looking.
    # ⚠️ The snapshot is the artifact that makes a pod handover survivable
    # (the 3.53 GB ckpt.pt never once pushed to HF), so this is the path a
    # rebuilt pod actually takes.
    snap = bool(ck.get("_fp16_weights_only")) or (
        "model" in ck and "stack" not in ck and "_meta" in ck)
    meta = (ck.get("_meta") or {}) if snap else ck
    sd = ck["model"] if snap else ck.get("stack", ck)

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
    return {"init_from": str(p), "init_step": int(meta.get("step", -1)),
            "missing_keys": sorted(fatal), "unexpected_keys": sorted(unexpected),
            # ⭐ named separately so a run row can never confuse "this stage
            # BUILT a new head" with "this stage failed to load one".
            "introduced_keys": sorted(introduced),
            "introduced_allowance": list(allowed),
            "trunk_md5_after_load": h.hexdigest(),
            "prev_stage": (meta.get("config") or {}).get("stage"),
            # ⚠️ STATED, never hidden: an fp16 snapshot round-trips the weights
            # through half precision, so this trunk md5 CANNOT equal the fp32
            # source's. A run row that does not say which it stands on would
            # look like a lineage break the next time the md5s are compared.
            "init_source": "fp16_weights_only_snapshot" if snap else "ckpt",
            "init_precision": "fp16->fp32 (lossy)" if snap else "fp32",
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
    # ---- PROPOSALS / MPC / FALLBACK (2026-08-16) — ALL DEFAULT-OFF ---------
    # The three remaining diagram cells on the planner surface
    # (DIAGRAM_CONFORMANCE.md F-15 / §3-D-1 / F-17). Defaults reproduce the
    # incumbent state_dict EXACTLY (per-tensor C75 proof in
    # tests/test_v6_diffusion_mpc_fallback.py).
    ap.add_argument("--proposals", choices=("query", "diffusion"),
                    default="query",
                    help="⭐ candidate-fan GENERATOR. 'diffusion' = F-15: "
                         "diffuse the full 6 s CONTROL sequence (60 x (a,κ)) "
                         "with temporally correlated OU noise, truncated-DDIM "
                         "denoised; +437,954 params MEASURED at production "
                         "geometry; the query/CV fan is still emitted beside "
                         "it (qfan_*) as the paired on-window reference. New "
                         "keys => S-T may INTRODUCE it (STAGE_MAY_INTRODUCE); "
                         "⛔ REFUSED in S-W (planner frozen + strict-resume "
                         "break, exactly like --selector).")
    ap.add_argument("--diffusion-steps", type=int, default=4,
                    help="truncated denoise steps (DiffusionDrive's regime)")
    ap.add_argument("--diffusion-noise-rho", type=float, default=0.9,
                    help="lag-1 autocorrelation of the OU control-noise. The "
                         "DRAWN noise's autocorrelation is MEASURED per "
                         "forward and logged (prop_noise_lag1_autocorr) — "
                         "never asserted from this flag.")
    ap.add_argument("--diffusion-hidden", type=int, default=256)
    ap.add_argument("--diffusion-sigma-a", type=float, default=2.0,
                    help="raw-space noise scale, accel channel (a_max 4.0)")
    ap.add_argument("--diffusion-sigma-k", type=float, default=0.1,
                    help="raw-space noise scale, kappa channel (kappa_max 0.2)")
    ap.add_argument("--mpc-refine", action="store_true",
                    help="⭐ MPC top-K refinement (selection cell, per the D-1 "
                         "re-read): the trained selector's scores warm-start "
                         "the top-K, cost descent refines the CONTROLS on a "
                         "COMPOSED cost — goal-conditioned PRIMARY + "
                         "kinematic/imagined-consistency REGULARIZERS — and "
                         "the re-score is GOAL DISTANCE ONLY (roll-cost argmin "
                         "REFUTED +5.9787 m; refined-readout rescoring "
                         "2.8-2.95x worse, E-S1-0). 0 params, 0 keys. "
                         "⛔ REQUIRES --selector goal (V6Config refuses "
                         "otherwise), so it is INERT while SEL-1 stands "
                         "REFUSED — assert_selector_admissible gates every "
                         "selector launch.")
    ap.add_argument("--mpc-topk", type=int, default=2)
    ap.add_argument("--mpc-steps", type=int, default=3,
                    help="cost-descent iterations (CEM is a possible later arm)")
    ap.add_argument("--mpc-lr", type=float, default=0.05)
    ap.add_argument("--mpc-roll-k", type=int, default=0,
                    help="P_O roll depth for the imagined-consistency "
                         "REGULARIZER; 0 = no roll (and --mpc-w-consist must "
                         "be 0)")
    ap.add_argument("--mpc-w-goal", type=float, default=1.0,
                    help="PRIMARY term weight; must stay > 0 — a "
                         "regularizer-led refinement is the refuted roll-cost "
                         "rule wearing MPC's name")
    ap.add_argument("--mpc-w-kin", type=float, default=0.1,
                    help="kinematic smoothness REGULARIZER (the §1.14 "
                         "tie-breaker); zero-weight ablation always available")
    ap.add_argument("--mpc-w-consist", type=float, default=0.0,
                    help="imagined-consistency REGULARIZER — the ONE place "
                         "roll-consistency may enter a cost, never the "
                         "primary and never the re-score")
    ap.add_argument("--fallback-trigger", action="store_true",
                    help="⭐ context-brain fallback cell (F-17): fan spread + "
                         "roll-cost VARIANCE -> P7-calibrated uncertainty; "
                         "fires when the band is exceeded; fallback action = "
                         "hold-v0/CV emission. The roll-cost here is the "
                         "UNCERTAINTY signal (P7 rho 0.7164, its validated "
                         "use), NEVER a selector — the module is permutation-"
                         "invariant over candidates by construction. 0 "
                         "params, 8 buffer keys => S-T may INTRODUCE it; "
                         "⛔ REFUSED in S-W (strict-resume break).")
    ap.add_argument("--fallback-roll-k", type=int, default=10,
                    help="P_O roll depth for the roll-cost-variance half of "
                         "the signal; 0 = spread-only (logged as such)")
    ap.add_argument("--fallback-calibration", default=None,
                    help="P7 calibration artifact (JSON: spearman_rho, "
                         "rho_ci, slope, intercept, threshold, w_spread, "
                         "w_rollvar). load_calibration REFUSES rho < 0.3 or "
                         "a CI including 0 (P7's pre-registered gate). "
                         "Without it the comparator emits fired=None and "
                         "says why — it never invents a boolean.")
    # ---- F-18 — THE PERCEPTION AGENT-SLOT DECODER, DEFAULT-OFF -------------
    ap.add_argument("--agent-slots", action="store_true",
                    help="⭐ build the DETR-style perception agent-slot "
                         "decoder (F-18, DIAGRAM_CONFORMANCE.md §4.2 — the "
                         "LAST unbuilt PERCEPTION cell): bbox cx,cy,yaw,l,w · "
                         "state v_rel,yaw-rate,occluded · class, over the "
                         "spatial tokens. +3,207,445 params MEASURED at the "
                         "§6 2-4 M band. ⛔ VISION-ONLY at inference (its "
                         "forward takes ONE tensor). ⛔ NO ladder stage "
                         "trains it — the v6 batch has no agent labels; a "
                         "frozen-trunk probe does. S-T may INTRODUCE it; "
                         "⛔ REFUSED in S-W (strict-resume break).")
    ap.add_argument("--n-slot-queries", type=int, default=16,
                    help="slot count. ⚠️ A DECLARED PLACEHOLDER — the right "
                         "value is the join's measured per-frame agent-count "
                         "distribution, which is UNMEASURED. Over-full frames "
                         "drop their FARTHEST targets and COUNT the drop.")
    ap.add_argument("--slot-hidden", type=int, default=256)
    ap.add_argument("--slot-depth", type=int, default=3)
    ap.add_argument("--slot-heads", type=int, default=8)
    ap.add_argument("--slot-src", choices=("cells", "tokens"), default="cells",
                    help="the memory the slots cross-attend into. 'cells' = "
                         "the readout's spatial latent (the surface whose "
                         "content is the open question, RC1); 'tokens' = the "
                         "encoder's raw patches, the INFORMATION CONTROL that "
                         "separates 'the encoder cannot see agents' from 'the "
                         "readout grid cannot carry them'.")
    ap.add_argument("--no-isolate-interp", action="store_true",
                    help="⛔ THE DELIBERATELY MIS-WIRED ARM. Lets the slot "
                         "decoder's PERCEPTION-LABEL gradient reach the "
                         "encoder — which the diagram's header row forbids in "
                         "any trunk loss, and which also destroys the head's "
                         "own meaning (a readout that trained its input can no "
                         "longer say what the latent already carried). It "
                         "exists so assert_isolation's perception_to_trunk "
                         "edge can FAIL, i.e. so it is a check.")
    # ---- GOAL-HEAD STRUCTURE + ANCHOR_GOAL — ALL DEFAULT-OFF ---------------
    # These V6Config levers shipped with NO command that could build them.
    # Every default here reproduces the incumbent state_dict EXACTLY.
    ap.add_argument("--goal-factored", action="store_true",
                    help="factor g_tac LAT x LON (the same factoring a_tac "
                         "already carries). The MIXED head is KEPT and still "
                         "emitted -- it is this arm's CONTROL. +470,939 params "
                         "MEASURED at the production geometry.")
    ap.add_argument("--goal-multilabel", action="store_true",
                    help="independent per-token gates on the UN-factored head "
                         "(0 params, 0 keys). The factored pair is the "
                         "structured form of the same fix and is strictly "
                         "better attributable.")
    ap.add_argument("--goal-cat-args", action="store_true",
                    help="the TYPED categorical arg channel. Without it 7 of "
                         "the 9 g_tac tokens are inexpressible even given "
                         "perfect labels. REQUIRED by --anchor-goal.")
    ap.add_argument("--tac-goal-cond", action="store_true",
                    help="⭐ build the g_str->P_T conditioning port (F-1, "
                         "DIAGRAM_CONFORMANCE.md 2026-08-16): a ZERO-INIT "
                         "cond_tac_dyn Linear whose output is added to the "
                         "tactical action-pair conditioning, so the strategic "
                         "goal conditions the tactical DYNAMICS — "
                         "P_T(z_tac, a_tac | g_str), which the diagram, "
                         "HIERARCHY_VOCABULARY §5 and V6Stack's own docstring "
                         "spec and the code did not build. Default OFF = "
                         "byte-identical state_dict (the live S-W resume). "
                         "An S-T stage may INTRODUCE it "
                         "(STAGE_MAY_INTRODUCE); S-S/S-J must CARRY it "
                         "forward once S-T trained with it, exactly like "
                         "--selector. ⛔ REFUSED in S-W: layer_tac is frozen "
                         "there and the new keys would break the live run's "
                         "strict resume.")
    ap.add_argument("--anchor-goal",
                    choices=("none", "snap_lat", "snap_xy", "onehot"),
                    default="none",
                    help="'none' (default) builds NO anchor head and leaves "
                         "the state_dict byte-identical. 'snap_lat' is the "
                         "FACTORED regress-then-snap default the measurement "
                         "prescribes (quantise LATERAL only; 98.8 % of the "
                         "variance is LONGITUDINAL). 'snap_xy' is the arm "
                         "E-AG2 measured FREE. ⛔ 'onehot' is the metric-blind "
                         "CONTROL, MEASURED +4.7502 [+3.0514, +6.3981] WORSE.")
    ap.add_argument("--anchor-table", default=None,
                    help="a build_refc_anchors.py .pt. ⛔ REQUIRED by "
                         "--anchor-goal: the head refuses to run without one, "
                         "because a zero table would snap every goal to the "
                         "origin and still return a number. ⚠️ It must be AT "
                         "THE PLAN HORIZON and no such vocabulary exists -- "
                         "all five banked tables stop at 2.0 s against "
                         "PLAN_STEPS=60, and load_anchor_table refuses them.")
    ap.add_argument("--n-anchors", type=int, default=256,
                    help="K of the anchor vocabulary (the shipped "
                         "refc_anchors_full_REBUILD.pt is 256)")
    ap.add_argument("--n-lat-bins", type=int, default=16,
                    help="resolution of the FACTORED lateral sub-vocabulary "
                         "('snap_lat' only)")
    ap.add_argument("--n-agent-slots", type=int, default=8)
    ap.add_argument("--w-anchor", type=float, default=0.0,
                    help="weight on the ANCHOR_GOAL objective. 0.0 = the term "
                         "is absent. ⚠️ in METRES for metric/softanchor (NATS "
                         "for the ce control) while the other terms are not, "
                         "so it is a declared decision, never a default.")
    ap.add_argument("--anchor-objective",
                    choices=tuple(ANCHOR_OBJECTIVES), default="metric",
                    help="'metric' (DEFAULT) = endpoint distance on the "
                         "straight-through emitted point -- the half E-AG2 "
                         "EXONERATED (snap is NOT separated from the ridge: "
                         "-0.0002 [-0.1031, +0.0703]). 'softanchor' = the "
                         "distance-weighted target over anchors (E-OBJ-1's "
                         "softade one level up). ⛔ 'ce' is the REFUTED "
                         "one-hot control and needs "
                         "--i-know-this-is-the-control-arm.")
    ap.add_argument("--anchor-axis-w", type=float, nargs=2,
                    metavar=("LON", "LAT"),
                    default=list(ANCHOR_AXIS_W_DEFAULT),
                    help="per-axis weights. Default 1 1 = RAW METRES, which is "
                         "the EVIDENCE-weighted choice, not the symmetric one: "
                         "the residual is 97.4 % longitudinal in squared error, "
                         "so raw metres already spend the gradient there, while "
                         "whitening would move half of it onto the axis "
                         "carrying 1.2 % of the variance.")
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
    # ⛔ THE POWER FIX (SIGREG_GATE_POWER.md, 2026-08-16). One batch is 48 rows
    # over 4 episodes, so rank_ceiling = 47 and the >= 0.8x criterion fires on
    # noise 9-38 % of the time with power 0.11 against a 1.43x true collapse.
    # Pooling N CONSECUTIVE steps raises the ceiling to min(N*48-1, d_op) AND
    # the cluster count to ~4N. 32 is the recommended setting: ceiling 1535,
    # ~14 min of Thor wall-clock, ~12.6 MB of CPU ring buffer.
    # DEFAULT 1 = no accumulator, byte-for-byte the incumbent emission.
    ap.add_argument("--spectrum-accum", type=int, default=1,
                    help="pool this many CONSECUTIVE steps into the O6 "
                         "spectrum reading (1 = off, the incumbent path)")
    # >0 turns ON the interval: a leave-one-CLUSTER-out jackknife (the only
    # candidate MEASURED to cover — 0.85/0.867 vs the bootstrap's 0.25/0.00),
    # plus this many bootstrap reps kept as a labelled diagnostic. A verdict
    # REFUSES to fire without an interval, so this is required for a real gate.
    # ⚠️ COST, MEASURED on the dev box (6 threads, may differ on Thor's CPU):
    # 1536 rows x 2048 -> plain 0.291 s, with interval 28.28 s = 0.54 % of a
    # 200-step interval at 26.35 s/step. 384 rows -> 0.39 s (0.007 %).
    ap.add_argument("--spectrum-ci-reps", type=int, default=0,
                    help="0 = no interval. >0 emits the cluster JACKKNIFE "
                         "interval on effective_rank plus this many bootstrap "
                         "reps as a diagnostic; blocks are --window rows")
    # ---- X4: per-layer spectrum records (2026-08-16) -----------------------
    # ⚠️ z_tac/z_str contribute ONE row per window (the uplink reads only the
    # window's last frame), so their per-batch ceiling is B-1 = 7 on the live
    # geometry and their verdicts are INCONCLUSIVE until pooled to the
    # LAYER'S OWN ceiling (tac 256, str 128 — x4_layer_power.json; NOT z_op's
    # 1024, which d_str=256 can never reach). --spectrum-accum 33 is the
    # smallest accum that makes ALL layers adjudicable (32 leaves tac ONE ROW
    # short: 32*8-1 = 255 < 256). ADDITIVE: a new "x4" key in the emission
    # record; the z_op/O6 path is not governed by this flag.
    ap.add_argument("--x4-spectrum-layers", default="tac,str",
                    help="comma list from {tac,str} for the X4 per-layer "
                         "spectrum records, or 'none' to disable. 'op' is "
                         "refused: the O6 z_op monitor is the incumbent "
                         "block and cannot be moved under this flag.")
    ap.add_argument("--w-t1", type=float, default=1.0)
    ap.add_argument("--w-s1", type=float, default=1.0)
    # ---- S2: strategic goal supervision (S-S/S-J) — DEFAULT OFF ------------
    ap.add_argument("--w-s2-goal", type=float, default=0.0,
                    help="weight on the S2 strategic-goal supervision (CE on "
                         "g_str/a_str + masked arg L1 vs s2-strategic-v1 "
                         "labels). 0.0 = the term is absent and the loss is "
                         "bit-identical. GOAL HEADS ONLY, never a trunk loss "
                         "(binding); in force only in S-S/S-J, where "
                         "layer_str trains. Needs --s2-labels.")
    ap.add_argument("--s2-labels", default=None,
                    help="s2-strategic-v1 label artifact: the labels DIR "
                         "(clip_index.json + s2_labels_*.jsonl) or one "
                         ".jsonl with clip_index.json beside it. ⛔ THE "
                         "CANONICAL SET IS s2_labels.S2_CANONICAL_LABELS_REL "
                         f"({S2_CANONICAL_LABELS_REL}) — the ORIGINAL "
                         "…/2026-08-16-s2-v1-labels/labels/ delivery is "
                         "SUPERSEDED (the PI adjudicated its lane-change rows "
                         "~78% wrong, 06b8782) and the loader REFUSES it by "
                         "its SUPERSEDED.json marker. The join is by "
                         "tanitad.data.v2_dataset.stable_episode_id ONLY "
                         "— the legacy 16-bit id collides (69/2400 + 7/600) "
                         "and is refused. ROUTE_TO records are refused "
                         "(G1 gated), mirroring s2_schema.validate().")
    # ---- X2 seam dump: bank the 60-step plan — DEFAULT OFF ----------------
    ap.add_argument("--dump-seam-plan", default=None,
                    help="DIR to bank the emitted 60-step plan into, one "
                         "seam_<step>.pt per checkpoint save, for "
                         "taniteval/tools/seam_probe.py. Unset = nothing is "
                         "banked and the module is never imported. ZERO extra "
                         "GPU: the plan is already computed for the step's "
                         "loss. ⚠️ S-W BANKS NOTHING — the emission head is "
                         "zero-init there, so the plan is all-zero and the "
                         "probe correctly returns DEGENERATE; this is an S-T"
                         "-and-later instrument.")
    ap.add_argument("--dump-seam-plan-degenerate", action="store_true",
                    help="bank the plan even when every control is exactly "
                         "zero (the S-W zero-init case). Only for keeping the "
                         "degenerate artifact deliberately — a dump banked "
                         "this way CANNOT answer the seam question.")
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
    tgc = bool(getattr(a, "tac_goal_cond", False))
    if a.stage == "S-W" and tgc:
        problems.append(
            "--stage S-W with --tac-goal-cond: layer_tac is FROZEN in S-W, so "
            "the g_str->P_T port would be untrainable dead weight AND would "
            "add cond_tac_dyn.* keys to the state_dict — which breaks a "
            "strict resume of the LIVE S-W run. The port is an S-T lever "
            "(F-1): S-T may INTRODUCE it (STAGE_MAY_INTRODUCE['S-T']) over an "
            "S-W checkpoint that never carried it.")
    # ---- PROPOSALS / MPC / FALLBACK (2026-08-16) ---------------------------
    if a.stage == "S-W" and getattr(a, "proposals", "query") != "query":
        problems.append(
            f"--stage S-W with --proposals {a.proposals}: the planner group "
            f"is FROZEN in S-W, so the diffusion generator would be "
            f"untrainable dead weight AND would add prop_diffusion.* keys to "
            f"the state_dict — which breaks a strict resume of the live S-W "
            f"run. The fan generator is an S-T lever "
            f"(STAGE_MAY_INTRODUCE['S-T']).")
    if a.stage == "S-W" and bool(getattr(a, "fallback_trigger", False)):
        problems.append(
            "--stage S-W with --fallback-trigger: the trigger's calibration "
            "buffers add fallback.* keys to the state_dict — which breaks a "
            "strict resume of the LIVE S-W run. It is introducible at S-T "
            "(STAGE_MAY_INTRODUCE['S-T']) and holds no trainable parameter "
            "in any stage.")
    # ---- F-18 PERCEPTION AGENT SLOTS ---------------------------------------
    if a.stage == "S-W" and bool(getattr(a, "agent_slots", False)):
        problems.append(
            "--stage S-W with --agent-slots: the slot decoder adds "
            "agent_slots.* keys to the state_dict — which breaks a strict "
            "resume of the LIVE S-W run. It is introducible at S-T "
            "(STAGE_MAY_INTRODUCE['S-T']). ⚠️ And note what the introduction "
            "does NOT mean: no ladder stage TRAINS this head — the v6 batch "
            "carries no agent labels — so it is carried, and a frozen-trunk "
            "probe (the P8 idiom) is what optimises it.")
    if bool(getattr(a, "no_isolate_interp", False)) \
            and not bool(getattr(a, "agent_slots", False)):
        problems.append(
            "--no-isolate-interp without --agent-slots: a mis-wiring flag for "
            "a module that is not built is a lever that silently does nothing "
            "(the --fallback-calibration lesson). ⚠️ And when it IS built the "
            "flag is the DELIBERATELY MIS-WIRED control arm: it lets a "
            "PERCEPTION LABEL train the encoder, which the binding diagram "
            "header forbids in any trunk loss, and assert_isolation's "
            "perception_to_trunk edge then FAILS by construction — which is "
            "the point of it, not a bug.")
    if getattr(a, "slot_src", "cells") != "cells" \
            and not bool(getattr(a, "agent_slots", False)):
        problems.append(
            f"--slot-src {a.slot_src} without --agent-slots: the memory arm of "
            f"a decoder that is not built reaches nothing.")
    if bool(getattr(a, "mpc_refine", False)) and a.selector != "goal":
        problems.append(
            f"--mpc-refine with --selector {a.selector}: the refinement's "
            f"PRIMARY cost is the distance to the selector's candidate-"
            f"INDEPENDENT goal point (W7-PROG: any selection cost NEEDS a "
            f"goal-conditioned component). 'none' has no selector and 'mlp' "
            f"emits no goal point — descending on its score would be "
            f"candidate-DEPENDENT, the REFUTED roll-cost family (+5.9787 m, "
            f"error-rank RISING with N). The MPC path stays INERT unless a "
            f"selector is admissible.")
    if getattr(a, "fallback_calibration", None) \
            and not bool(getattr(a, "fallback_trigger", False)):
        problems.append(
            f"--fallback-calibration {a.fallback_calibration} without "
            f"--fallback-trigger: a calibration that reaches no comparator "
            f"is an input that silently does nothing.")
    if getattr(a, "fallback_calibration", None) \
            and not Path(a.fallback_calibration).exists():
        problems.append(
            f"--fallback-calibration {a.fallback_calibration} does not "
            f"exist. Fail in milliseconds, not after the run — the "
            f"--gate-probes lesson.")
    if a.w_select and a.selector == "none":
        problems.append(
            f"--w-select {a.w_select} with --selector none: a selection loss "
            f"with no scorer is how a selector silently never trains.")
    # ⛔ THE ACK FLAG'S DEST. ``--i-know-this-is-the-control-arm`` is registered
    # in ``main`` with ``dest="control_arm_ack"``, so the namespace NEVER has an
    # attribute named ``i_know_this_is_the_control_arm`` — the original getattr
    # here could only ever return False. MEASURED 2026-08-16: passing the flag
    # the refusal below NAMES did not clear that refusal, so the pre-registered
    # inert-scorer control arm (V6F_PLANNER_DESIGN §4.1) was unlaunchable. Both
    # spellings are accepted so a hand-built namespace still works.
    ack = bool(getattr(a, "control_arm_ack", False)
               or getattr(a, "i_know_this_is_the_control_arm", False))
    # ⛔ AND THE REFUSAL WAS STAGE-BLIND. "the scorer never receives a gradient"
    # is a defect only where the planner group TRAINS. In S-S the planner is
    # frozen BY DESIGN (STAGE_GROUPS["S-S"] == ("layer_str",)) and
    # ``V6LossWeights.for_stage("S-S")`` zeroes ``w_select`` regardless — yet
    # S-S MUST still carry ``--selector <the S-T arm>`` forward, because the
    # S-T checkpoint contains ``cand_score.*`` and a selector-less S-S stack
    # makes those UNEXPECTED keys, which ``load_stage_init`` correctly treats as
    # fatal. MEASURED 2026-08-16: every available S-S command was refused —
    # ``--selector goal`` here, ``--selector none`` at the init load, and
    # ``--w-select 1.0`` only got through by advertising a weight that is not in
    # force. Same family as the ``strict=True`` init blocker: right in spirit,
    # blind in practice.
    planner_trains = "planner" in stage_trainable_groups(a.stage)
    if a.selector != "none" and not a.w_select and planner_trains and not ack:
        problems.append(
            f"--selector {a.selector} with --w-select 0 in stage {a.stage}, "
            f"which TRAINS the planner group: the scorer would be built, "
            f"consume its parameters and never receive a gradient. If an "
            f"inert-scorer control is what you want, say so explicitly by "
            f"passing --w-select 0 AND --i-know-this-is-the-control-arm.")
    # ⛔ THE SAME INERT-MODULE FAMILY, for the g_str->P_T port: its ONLY
    # gradient source is t1 through zh_tac (v6_loss_step wires no other loss
    # through the tactical prediction), so building it in a stage that TRAINS
    # layer_tac while --w-t1 is 0 advertises a port that never trains — the
    # `intent_proj` dead-weight defect, re-created by launch line. S-S is
    # deliberately NOT refused: layer_tac is frozen there and the flag must be
    # CARRIED for geometry, exactly like --selector (a flagless S-S against an
    # S-T ckpt that trained the port dies on unexpected cond_tac_dyn.* keys).
    if tgc and "layer_tac" in stage_trainable_groups(a.stage) \
            and not a.w_t1 and not ack:
        problems.append(
            f"--tac-goal-cond with --w-t1 0 in stage {a.stage}, which TRAINS "
            f"layer_tac: t1 is the ONLY loss that flows through the "
            f"g_str-conditioned tactical prediction, so the port would be "
            f"built, consume its parameters and never receive a gradient — "
            f"the intent_proj dead-weight defect F-1 exists to close. If an "
            f"inert-port control is what you want, say so with "
            f"--i-know-this-is-the-control-arm.")
    # ---- ANCHOR_GOAL: every refusal fires in MILLISECONDS, not after a run --
    anchor_goal = getattr(a, "anchor_goal", "none")
    w_anchor = float(getattr(a, "w_anchor", 0.0))
    anchor_obj = getattr(a, "anchor_objective", "metric")
    if a.stage == "S-W" and anchor_goal != "none":
        problems.append(
            f"--stage S-W with --anchor-goal {anchor_goal}: the planner group "
            f"is FROZEN in S-W, so the anchor head would be untrainable dead "
            f"weight AND would change the state_dict -- which breaks a strict "
            f"resume of the LIVE S-W run. ANCHOR_GOAL is an S-T lever.")
    if w_anchor and anchor_goal == "none":
        problems.append(
            f"--w-anchor {w_anchor} with --anchor-goal none: an anchor "
            f"objective with no anchor head is how a head silently never "
            f"trains -- the same defect --w-select/--selector already guards.")
    if anchor_goal != "none" and not getattr(a, "anchor_table", None):
        problems.append(
            f"--anchor-goal {anchor_goal} without --anchor-table: the head "
            f"REFUSES to run without one (a zero table snaps every goal to the "
            f"origin and still returns a number). ⛔ AND NO ADMISSIBLE TABLE "
            f"EXISTS TODAY: all five banked vocabularies stop at step 20 = "
            f"2.0 s while --plan-steps is {a.plan_steps} "
            f"({a.plan_steps * a.dt:g} s), and load_anchor_table refuses the "
            f"mismatch. Build one first: build_refc_anchors.py --horizons "
            f"5,10,...,{a.plan_steps} (CPU-only, needs the TRAIN epcache).")
    if anchor_goal != "none" and not getattr(a, "goal_cat_args", False):
        problems.append(
            f"--anchor-goal {anchor_goal} without --goal-cat-args: V6Config "
            f"refuses this pairing (an emitted id that reaches nothing "
            f"downstream is a head wearing an emission's name). Pass "
            f"--goal-cat-args.")
    if w_anchor and anchor_goal not in ("none",) \
            and anchor_goal not in ANCHOR_OBJ_MODES.get(anchor_obj, ()):
        problems.append(
            f"--anchor-objective {anchor_obj} needs --anchor-goal in "
            f"{ANCHOR_OBJ_MODES[anchor_obj]}, got {anchor_goal}. "
            f"{ANCHOR_OBJECTIVES[anchor_obj]}")
    # ⛔ THE CONTROL GATE. `ce` is the objective E-AG2 MEASURED +4.7502
    # [+3.0514, +6.3981] WORSE than a ridge that was ALREADY refused, and
    # E-OBJ-1 measured the same axis independently. It stays BUILDABLE because
    # a comparison with no control is unattributable (C6) -- and it stays
    # behind the same acknowledgement `--no-isolate-planner` uses, so a refuted
    # objective can never arrive by defaulting into it.
    if w_anchor and anchor_obj == "ce" and not ack:
        problems.append(
            "--anchor-objective ce is the pre-registered, MEASURED-REFUTED "
            "CONTROL: a one-hot anchor_id target is metric-BLIND and E-AG2 "
            "measured it +4.7502 [+3.0514, +6.3981] WORSE than the free ridge "
            "(separated at every K from 8 to 256, under both vocabulary "
            "constructions, replicated on REF-C-base at +5.4570). The DEFAULT "
            "is --anchor-objective metric. If the control arm is what you "
            "want, say so with --i-know-this-is-the-control-arm.")
    if w_anchor and a.stage == "S-S":
        problems.append(
            f"--w-anchor {w_anchor} in S-S: the planner is FROZEN here and "
            f"`V6LossWeights.for_stage('S-S')` zeroes w_anchor, so the launch "
            f"line would advertise an objective that is not in force. Keep "
            f"--anchor-goal {anchor_goal} for the GEOMETRY and pass "
            f"--w-anchor 0.")
    if any(float(x) < 0.0 for x in getattr(a, "anchor_axis_w",
                                           ANCHOR_AXIS_W_DEFAULT)):
        problems.append(f"--anchor-axis-w must be non-negative, got "
                        f"{list(a.anchor_axis_w)}")
    if a.stage == "S-S" and a.w_select:
        problems.append(
            f"--w-select {a.w_select} in S-S: the planner is FROZEN here and "
            f"`V6LossWeights.for_stage('S-S')` zeroes w_select, so the launch "
            f"line would advertise a selection loss that is not in force — a "
            f"run row that lies about what moved. Pass --w-select 0 and keep "
            f"--selector {a.selector} for the GEOMETRY: S-S must carry the "
            f"S-T arm's scorer forward or --init-from fails on unexpected "
            f"cand_score.* keys.")
    # ---- S2: every incoherent combination refused in MILLISECONDS ----------
    w_s2 = float(getattr(a, "w_s2_goal", 0.0))
    s2p = getattr(a, "s2_labels", None)
    if w_s2 < 0.0:
        problems.append(f"--w-s2-goal must be non-negative, got {w_s2}")
    if w_s2 and a.stage in ("S-W", "S-T"):
        problems.append(
            f"--w-s2-goal {w_s2} in {a.stage}: the strategic goal heads "
            f"(layer_str) are FROZEN here and `V6LossWeights.for_stage"
            f"('{a.stage}')` zeroes w_s2_goal, so the launch line would "
            f"advertise a supervision that is not in force — the same "
            f"advertised-but-inert lie --w-select/--w-anchor already refuse "
            f"in S-S. S2 is an S-S/S-J lever (the stages that train "
            f"layer_str). It adds NO state_dict key, so unlike --selector "
            f"there is no geometry to carry: just drop the flag.")
    if w_s2 and not s2p and not a.dry_run:
        problems.append(
            f"--w-s2-goal {w_s2} without --s2-labels: an S2 term with no "
            f"labels is how a supervision weight silently becomes 0 — the "
            f"loss would refuse at step 1 anyway (missing batch keys), but "
            f"that is after the corpus build; this fails in milliseconds. "
            f"Point --s2-labels at the s2-strategic-v1 artifact "
            f"(labels dir with clip_index.json).")
    if s2p and not w_s2 and not ack:
        problems.append(
            f"--s2-labels {s2p} with --w-s2-goal 0: the labels would be "
            f"loaded and joined for a term that is not in force — a launch "
            f"line advertising supervision that trains nothing (the inert-"
            f"module family). If a load-only rehearsal is what you want, say "
            f"so with --i-know-this-is-the-control-arm.")
    if s2p and not Path(s2p).exists():
        problems.append(
            f"--s2-labels {s2p} does not exist. Same class as the "
            f"--gate-probes refusal: fail before the corpus build, not "
            f"after it.")
    if w_s2 and a.no_isolate_planner:
        problems.append(
            f"--w-s2-goal {w_s2} with --no-isolate-planner: the S2 CE/L1 "
            f"reads g_str/a_str, whose input z_str_p is detached ONLY by the "
            f"planner cut — without it the label loss reaches the adapters "
            f"and encoder and becomes a TRUNK loss. 'Labels supervise "
            f"GOAL/INTERPRETATION HEADS only, never any WM trunk loss' is "
            f"BINDING (HIERARCHY_VOCABULARY §2): there is NO control arm and "
            f"no acknowledgement flag for a binding rule — run the isolation "
            f"control without S2, or S2 without the control.")
    # ⛔ AN ANALYSIS-TIME REFUSAL AFTER THE COMPUTE IS PAID FOR. MEASURED
    # 2026-08-16: `--gate-probes <missing file>` is only read by
    # `_load_gate_probes` at the very END of `train()` — the whole run executes,
    # then dies with "does not exist" before writing `stage_gate.json` AND
    # before writing `summary.json`, which is the done-marker. On a 10,000-step
    # S-T that is ~3.1 GPU-days paid for, no gate produced, and a supervisor
    # left with no done-marker — the exact resurrection trap. Same class as
    # `t1_eval.py` rolling both arms and then dying on an import in `analyze()`.
    # ⇒ preflight the optional input at startup, so it fails in milliseconds.
    if a.gate_probes and not Path(a.gate_probes).exists():
        problems.append(
            f"--gate-probes {a.gate_probes} does not exist. It is not read "
            f"until AFTER the training loop, so without this refusal the run "
            f"would spend its entire budget and then die before writing "
            f"stage_gate.json or summary.json — leaving a supervisor with no "
            f"done-marker to stop on. Create the probe artifact first, or drop "
            f"the flag and supply the probes on a re-gate.")
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
    problems += _preflight_subframe(a)
    problems += _preflight_seam_dump(a)
    return problems


# ---------------------------------------------------------------------------
# ⛔ E2 — `--v2-subframe` MOVES THE DATA, NOT THE MODEL, **IN THIS TRAINER**
# ---------------------------------------------------------------------------

def subframe_desync(a) -> tuple[int, int] | None:
    """The sub-frame this run would train the DATA at, when it disagrees with
    the frame the ENCODER is built for. ``None`` = consistent.

    ⛔ WHY THIS IS NOT THE SAME FUNCTION AS ``train_flagship_v4``'s.
    ``resolve_v2_frames``'s docstring says *"The frame is applied to ``cfg``
    too, so the ENCODER is sized for what it will be fed."* That is TRUE for
    ``train_flagship_v4``, whose ``cfg`` **is** the model config, and FALSE
    here: this trainer calls ``resolve_eval_frames(a, cfg_eval)`` where
    ``cfg_eval`` is a **flagship-v4 eval config** used for the plan and the eval
    seam, while :func:`build_stack_from_args` has ALREADY sized the encoder from
    ``a.frame_h``/``a.frame_w``. The two are different objects, so a sub-frame
    moves one and not the other.

    ⚠️ MEASURED 2026-08-17 on the built production encoder
    (`…/2026-08-17-st-launch-readiness/raw/subframe_desync.json`): with
    ``--frame-h 256 --frame-w 640 --v2-subframe 176x624`` the encoder's
    ``pos`` is ``[1, 640, 768]`` and the first forward raises
    ``ValueError: encoder input is (176, 624) but the config declares
    (256, 640)``.

    ⛔ AND THE DANGEROUS PART IS THE **ORDER**. ``--init-from`` SUCCEEDS
    (checkpoint and stack are both 256x640) and ``assert_v2_geometry_matches``
    PASSES (it compares the providers against ``model_frame``, which *is*
    176x624) — so the refusal arrives at the FIRST FORWARD, after the corpus
    has mounted and the O4 saliency pass has run. A guard existed and it was in
    the wrong place. This one is args-only, so it fires in milliseconds at
    startup.
    """
    from train_flagship_v4 import parse_subframe
    hw = parse_subframe(getattr(a, "v2_subframe", None))
    if hw is None:
        return None
    if (int(hw[0]), int(hw[1])) == (int(a.frame_h), int(a.frame_w)):
        return None                      # a no-op sub-frame is consistent
    return (int(hw[0]), int(hw[1]))


def _preflight_subframe(a) -> list[str]:
    hw = subframe_desync(a)
    if hw is None:
        return []
    return [f"--v2-subframe {hw[0]}x{hw[1]} with --frame-h {a.frame_h} "
            f"--frame-w {a.frame_w}: in THIS trainer the sub-frame moves the "
            f"DATA and NOT the MODEL. build_stack_from_args sizes the encoder "
            f"from --frame-h/--frame-w; resolve_eval_frames applies the "
            f"sub-frame to a flagship-v4 EVAL config. So --init-from would "
            f"succeed, parity would pass, the corpus would mount — and the "
            f"first forward would raise `encoder input is ({hw[0]}, {hw[1]}) "
            f"but the config declares ({a.frame_h}, {a.frame_w})`, after the "
            f"compute is paid for.\n"
            f"     ⇒ drop --v2-subframe (train at the declared frame), or "
            f"declare --frame-h {hw[0]} --frame-w {hw[1]} so the encoder is "
            f"built for what it is fed. ⚠️ The second is a DIFFERENT MODEL and "
            f"cannot --init-from a {a.frame_h}x{a.frame_w} checkpoint."]


# ---------------------------------------------------------------------------
# ⛔ E5 — the seam dump's import, at STARTUP instead of 1.8 h in
# ---------------------------------------------------------------------------

def seam_dump_import_error(a) -> str:
    """``""`` when ``--dump-seam-plan``'s module imports, else the error.

    ⛔ THE ANALYSIS-TIME-IMPORT FAMILY, IN ITS MILDEST COSTUME AND ITS MOST
    PERSISTENT ONE. ``--dump-seam-plan`` is wired into the training loop at the
    ``--save-every`` boundary, and its ``except Exception`` prints *"training
    continues"* — correct for a diagnostic, and it means a `ModuleNotFoundError`
    banks NOTHING while the run looks healthy. The first attempt is at
    ``step % save_every == 0``, i.e. **~1.8 h in at --save-every 250**.

    ⚠️ MEASURED 2026-08-17 on Thor **and** the dev box: under the launch's own
    ``PYTHONPATH=<stack>``, ``import taniteval`` is a `ModuleNotFoundError` —
    ``taniteval`` is a **SIBLING of ``stack/``** (``TanitAD/taniteval/
    taniteval/``), not a member of it. F-16's probe has produced zero real-arm
    numbers three times running, and this is why.

    ⇒ The operator ASKED for the dump, so a dump that cannot happen is a
    REFUSAL, in 2 seconds, not a log line 1.8 h later. (The in-loop catch stays
    non-fatal: a diagnostic must never kill a 3-day run mid-flight. This makes
    that path near-unreachable rather than removing it.)
    """
    if not getattr(a, "dump_seam_plan", None):
        return ""
    try:
        importlib.import_module("taniteval.seam_dump")
        return ""
    except BaseException as e:                                # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _preflight_seam_dump(a) -> list[str]:
    err = seam_dump_import_error(a)
    if not err:
        return []
    return [f"--dump-seam-plan {a.dump_seam_plan} but `import "
            f"taniteval.seam_dump` FAILS: {err}\n"
            f"     `taniteval` is a SIBLING of stack/, not a member of it, so "
            f"a launch line whose PYTHONPATH is only <repo>/stack cannot see "
            f"it. Without this refusal the run would bank NOTHING while "
            f"printing '[v6 seam] dump FAILED … — training continues' at every "
            f"save boundary, the first one ~1.8 h in.\n"
            f"     ⇒ PYTHONPATH=<repo>/stack:<repo>/taniteval  (v6_chain's "
            f"launch_line and manifest_text now emit both), or drop "
            f"--dump-seam-plan and accept that X2_seam reads 'not-run'."]


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
