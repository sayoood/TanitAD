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

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.fourbrain import StrategicPolicy, TacticalPolicy
# ⛔ ONE VOCABULARY SOURCE. These tuples are IMPORTED, never re-declared — a
# second copy is a second vocabulary, and the programme has already paid for
# that once (see the defect note below).
from tanitad.models.v6 import (TACTICAL_LAT_ACTIONS,
                               TACTICAL_LON_ACTIONS, tactical_lat_actions,
                               tactical_lon_actions_v)
from tanitad.refs.refa_v1_plan import PlanConfig, icem_plan, unicycle_paths

__all__ = ["RefAV1Config", "RefAV1", "DINOV3_GEOMETRY",
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
    a_dim: int = 2                        # (a, kappa) — Alpamayo-2 form
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
        self.tmix = nn.Conv1d(cfg.d_state, cfg.d_state, kernel_size=3,
                              padding=1, groups=cfg.d_state)
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
        self.act = nn.Sequential(nn.Linear(cfg.a_dim, d), nn.GELU(),
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
        """One latent step. ``field`` [B,N,d], ``action`` [B,a_dim] -> [B,N,d].

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
        """``actions`` [B,K,a_dim] -> predicted fields [B,K,N,d].

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
        return self.read(field.mean(dim=-2))          # pool tokens -> [B, str]

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
        q = self.tac_queries.expand(field.shape[0], -1, -1)
        return self.tac_pool(q, field, field, need_weights=False)[0]

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

    def forward(self, feats: Tensor, actions: Tensor, *,
                future_feats: Tensor | None = None,
                str_ext_targets: Tensor | None = None,
                str_ext_actions: Tensor | None = None,
                lat_label: Tensor | None = None,
                lon_label: Tensor | None = None,
                route_label: Tensor | None = None,
                nav_cmd: Tensor | None = None,
                ego: Tensor | None = None) -> dict:
        """``feats`` [B,W,N,d_enc] observed window, ``actions`` [B,K,a_dim].

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

        out["op_pred"] = self.operative.rollout(last, actions, intent=intent)
        # ⚠️ ACTIONS keep phase ``[::stride]`` while TARGETS take
        # ``[stride-1::stride]``, and the asymmetry is deliberate: a level's
        # step j spans operative steps [j*s, (j+1)*s), so it CONSUMES the
        # action that opens that window and PREDICTS the state that closes it.
        tac_a = actions[:, ::self._stride(self.cfg.tac_dt)]
        out["tac_pred"] = self.tactical.rollout(
            self._tac_field(last), tac_a[:, :self.cfg.tac_steps], intent=intent)
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
            tq = torch.stack([self._tac_field(tgt[:, i])
                              for i in range(tgt.shape[1])], dim=1)
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
            st = self.strategic.subspace(tgt.flatten(0, 1)).reshape(
                tgt.shape[0], tgt.shape[1], -1)[:, sstep - 1::sstep]
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
                text = self.adapter(self.std(str_ext_targets))
                st_e = self.strategic.subspace(text.flatten(0, 1)).reshape(
                    text.shape[0], text.shape[1], -1)
                if self.cfg.detach_aux_targets:
                    st_e = st_e.detach()              # same reason as `tq`
                out["loss_feat_str_ext"] = F.mse_loss(out["str_pred_ext"], st_e)
                out["loss"] = (out["loss"] + self.cfg.w_feat_str_ext
                               * out["loss_feat_str_ext"])

            # ---- change #10: the counterfactual-action term ---------------- #
            if self.cfg.w_cf:
                out.update(self._cf_term(last, actions, tgt, intent))
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
                 intent: Tensor | None) -> dict:
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
            return self.operative.rollout(last, a[:, :j + 1], intent=intent)[:, j]

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

    # -- deployment: behaviour by PLANNING, not regression ----------------- #
    @torch.no_grad()
    def plan(self, feats: Tensor, *, v0: float, goal_field: Tensor | None = None,
             target_speed: float | None = None, nav_cmd: Tensor | None = None,
             plan_cfg: PlanConfig | None = None, prev_elites: Tensor | None = None,
             cost_chunk: int = 64):
        """One MPC tick for ONE window (B must be 1).

        The cost is where the hierarchy earns its keep (change #8): the tactical
        target speed and the strategic goal field enter as **cost terms**, not as
        head outputs to be regressed. ``goal_field`` defaults to the tactical
        brain's own imagined 6 s field, which is what makes this
        goal-conditioning rather than goal-following.
        """
        if feats.shape[0] != 1:
            raise ValueError("plan() is a single-window API (B must be 1)")
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
        search_goal = (None if goal_field is None else
                       (self._tac_field(goal_field) if coarse else goal_field))

        def _cost_chunk(controls: Tensor, pred=None, z0=None,
                        goal=None) -> Tensor:
            pred = pred or search_pred
            z0 = search_z if z0 is None else z0
            goal = search_goal if goal is None else goal
            n = controls.shape[0]
            z = z0.expand(n, -1, -1)
            # last_only: the cost reads the terminal field only (see rollout).
            zk = pred.rollout(z, controls, intent=intent, last_only=True)
            c = torch.zeros(n, device=controls.device)
            if goal is not None:
                g = goal.expand(n, -1, -1)
                c = c + (1.0 - F.cosine_similarity(
                    zk.flatten(1), g.flatten(1), dim=-1))
            jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt
            c = c + 0.02 * jerk.pow(2).mean(-1)                    # comfort
            c = c + 0.05 * controls[..., 1].pow(2).mean(-1)        # curvature
            if target_speed is not None:
                v_end = v0 + controls[..., 0].sum(-1) * pc.dt
                c = c + 0.10 * (v_end - target_speed).pow(2)
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
            fine = _cost_chunk(stack, pred=self.operative, z0=last,
                               goal=goal_field)
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
