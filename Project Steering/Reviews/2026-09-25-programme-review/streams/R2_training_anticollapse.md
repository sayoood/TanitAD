# Stream R2 — Training recipes, objectives, anti-collapse, optimisation & compute efficiency

Programme review, 2026-09-25. Repo `/home/user/TanitAD`, branch `claude/optimistic-shannon-vixpo6`,
HEAD `467ce8a`. Static review only (no pod/torch access in this container) — every finding below is
from reading source, `MODEL_REGISTRY.md`, `GATE_PROTOCOL.md`, PREREG docs, dated research notes, and
3 targeted web searches (LeJEPA, its identifiability follow-up, V-JEPA 2). Evidence classes follow the
CLAUDE.md convention: **MEASURED** (ours + artifact path) · **PUBLISHED** (cited, web-verified) ·
**INHERITED** (another doc, not independently re-verified by me) · **ESTIMATED** · **HYPOTHESIS**.

Status: COMPLETE.

---

## 0. Headline findings (ranked)

1. **SIGReg is a faithful, correctly-implemented copy of its published source, and genuinely needs no
   EMA/stop-gradient.** `stack/tanitad/models/sigreg.py` implements the sliced Epps–Pulley test over
   `n_slices=512` *freshly redrawn* directions, forced to fp32, exactly as Balestriero & LeCun's LeJEPA
   describes (**PUBLISHED, web-verified arXiv:2511.08544**). A repo-wide grep for `ema`/`EMA` across
   every model and trainer file returns **zero hits** except three comments explicitly noting its
   *absence* (`sigreg.py:3-4`, `train_worldmodel.py:335`, `refb_train.py:6`) — this is the one part of
   the recipe that is unambiguously sound. [MEASURED — `sigreg.py:26-84`; PUBLISHED]

2. **But SIGReg's own exemption plus the programme's own instrumentation show it is regularising a
   near-empty complement.** `sigreg.py`'s `position_relaxed()` exempts the first `free_dims=64` of the
   2048-d readout (`config.py:395`); the *independently measured* orthogonality report on the actual
   trained checkpoint (`…/2026-07-18-operative-flagship-blind-rollout-and-orthogonality.md`) finds the
   **active (energy-carrying) subspace is only `active_k≈19–30`** at both step 6,500 and step 19,000 —
   smaller than the exempted 64. `iso_ratio_active` does rise as designed (0.254→0.546, cond number
   218→61) but the loss-rebalance research's independently measured `vision_use≈12.9%` flat and the
   route head's `route_skill_vs_chance=0.0` triangulate the same story: essentially all task-relevant
   capacity lives in or near the SIGReg-exempt band, and the ~1,984 dims SIGReg actively regularises
   toward isotropic Gaussian carry almost no discriminative signal. SIGReg is not failing on its own
   terms; its terms may be aimed at very little. [MEASURED, 3-way cross-file corroboration]

3. **SIGReg (and the whole JEPA loss) never reaches the recursive-rollout or H15-imagination paths —
   and an independent instrument caught exactly the collapse that gap predicts.** `flagship_loss()`
   applies SIGReg only to `states`/`fut_states` and the *one-shot* multi-horizon heads
   (`flagship_losses.py:367-376`); the K-step recursive rollout (`train_worldmodel.py:58-74`,
   `metric_dynamics.py:220-244`) and the H15 `ImaginationField`'s autoregressive belief path
   (`imagination.py:118-146`) get no anti-collapse pressure at all. The E1 blind-rollout diagnostic
   (2026-07-18) measured inter-sample belief cosine climbing **0.219→0.805 by k=8** — an attractor
   collapse — while epistemic log-variance *shrinks* (**−9.461→−9.564**, false confidence) under
   exactly that un-regularised recursion. [MEASURED, code + dated research note cross-reference]

4. ⭐ **NEW: every v4-family trainer (v4/v4.1/v4.2/v4.2b/v4-fromscratch/v5f) instantiates the 22M-param
   H15 `ImaginationField` but never trains it — it is very likely frozen at random init in every such
   checkpoint, including the currently-running v5f.** `fourbrain.py:456-460` builds `self.imagination`
   whenever `cfg.h15.enabled` (true for the `flagship4b_config()` lineage v4/v5f inherit — the registry's
   own v4 param table lists `H15 imagination 22,055,683` as part of the "shared trunk … unchanged").
   A grep of `train_flagship_v4.py` (2,527 lines) for `h15|imagination_nll|sector_mask` returns **zero
   hits**, and the shared `flagship_loss()` it calls never references `model.imagination` either (full
   read). Yet `world.imagination`'s parameters sit inside the trunk AdamW group
   (`train_flagship_v4.py:1361-1363`, `list(world.parameters())`). Standard PyTorch AdamW skips any
   parameter whose `.grad` is `None` — so with no loss ever touching this module, its ~22M params
   (≈8-9% of the ~247–263M budget) should never be updated at all for the whole run. Unlike every other
   deliberate ablation in this file (`cond_imagination = False`, each with an explanatory comment), there
   is no comment here — circumstantial evidence this is an oversight, not a decision. [MEASURED (code
   paths); HYPOTHESIS (the run-time consequence — not verified live, no torch access here); cheap to
   settle: diff `world.imagination.state_dict()` across two v5f checkpoints, or print `.grad` on one
   step]

5. **A different, un-gated collapse axis: the latent's own temporal derivative is close to noise, even
   though its aggregate isotropy is converging.** `stack/tanitad/eval/latent_screen.py`'s pre-flight
   gates (jitter ratio ≤2×, derivative-corr >0.50, derived-accel R² >0.50) all **FAIL** on the deployed
   v1 checkpoint: jitter **51.0×**, derivative-corr **+0.0891**, derived-accel R² **−0.3773**
   (`latent_screen.py:76-85`, `REFERENCE_LATENTS`). A representation can be batch-isotropic (what
   SIGReg targets) while its per-sample motion along its own best speed axis is 51× noisier than the
   true kinematics — dimensional collapse and temporal collapse are different failure modes, only one
   of which (this one) is gated at launch, and only since 2026-08-03. [MEASURED]

6. **Loss weights still favour "make the encoder an odometer" over the SSL core by ≈4.4:1, and the
   documented mitigation is opt-in, not default.** Current `LossWeights` defaults
   (`flagship_losses.py:150-165`) plus `grounding_losses`'s own defaults sum to an SSL core
   (`pred+tacpred+roll+goal+sigreg` = 1.0+0.5+0.5+0.5+0.1 = **2.6**) against a supervised/metric-motion
   group (`wp+man+route+grounding(invdyn 2.0×3 + fwd 1.0×3)+inv` = 1.0+0.5+0.5+9.0+0.5 = **11.5**) for
   v1's actual active terms — consistent with the 2026-07-18 loss-rebalance research's own "≈5:1" count.
   That research's **PRIMARY recommendation** (a gradient-scale `α=0.25` decoupling the invdyn real-pair
   term from the encoder) is implemented as `invdyn_gradscale` (`metric_dynamics.py:336-345`,
   `grounding_losses()`) — but its **default is `1.0`** (`flagship_losses.py:364`, byte-identical to
   off), so the recommended-and-implemented fix is not the shipped default for v1 or v5f (v3enc did use
   `0.5`, a softer setting, per its staged schedule). [MEASURED, current source vs. a 2-month-old
   recommendation's adoption status]

7. **The four failed/tied lines have four different, code-confirmed root causes, and one of the gate
   verdicts that ended a line is itself compromised.** v2 fired ten levers simultaneously
   (uninterpretable by construction, and the registry says so). v3enc's 10k `RESTART` verdict rests on
   a gate window in which **`decorr` was held at exactly `0.0` for the entire 0–9,999 steps being
   graded** (`train_flagship4b.py:93`, `staged_lever_schedule`) — the lever family the gate "refuted"
   was never engaged during the graded window. v4.1/v4.2 is a genuine competing-objectives problem (a
   converged WM trunk vs. a cold diffusion-planner head) that a scalar canary floor cannot fully
   resolve — `v4_curriculum.py`'s `CanaryController` fix (cap-and-hold, `mult_floor`) is a real, correct
   repair for v4.1's *specific* halve-to-zero bug, but v4.2's floor=0.25 arm was still worse than v4.1
   on *every* measured axis at less than half the steps. v1.6 is a clean statistical tie on ADE (paired
   Δ +0.0104, CI **[−0.0888,+0.1147]**) bought at **+144%** WM-canary cost — reported three
   contradictory ways in one session purely by quoting a train-log spike, then trainer in-loop val,
   before the canonical eval + paired bootstrap settled it (`MODEL_REGISTRY.md` §1.4b). [MEASURED]

8. **A live, unresolved instance of the exact discipline this review is asked to enforce: the
   rollout-recovery experiment ran, cost real GPU, and produced a large ADE win — but not the metric it
   was pre-registered on.** `PREREG_rollout_recovery.md` designates `CR_k` (a latent compounding-ratio
   instrument) **PRIMARY** and explicitly forbids trading it against ADE. The completed run's output
   (`…/2026-08-02-rollout-recovery-verdict/{rrctl,rr20,ab_rrctl_vs_rr20}.json`) contains **only** an ADE
   comparison — RR-20 (`rollout_k=20`, matched to the eval horizon) beats RR-CTL (`rollout_k=4`)
   **0.3485 vs 0.4244 m, Δ+0.0759 [0.0613,0.0906], separated** — and **no `CR_k` field anywhere** in any
   of the three JSONs. Under the experiment's own pre-registration this result currently has **no
   admissible verdict**. Computing `CR_k` needs no new training (both checkpoints already exist) and is
   the single highest-value next step on this line. [MEASURED]

9. **Compute: no `torch.compile`, no fused AdamW, no explicit SDPA/flash call anywhere in any of the 8
   trainers** (grepped all of `train_flagship4b/_v15/_v16/_v4`, `refa_train4b`, `refb_train`,
   `refc_train`, `train_dynamics_encoder`) — uniformly bf16 autocast, `clip_grad_norm_(…, 1.0)`, opt-in
   `--grad-checkpoint`. v5f's one recipe addition, `--cond-imagination`, forces a **mandatory, strictly
   sequential 20-step frozen-predictor rollout at batch×n_probes=32, under `torch.no_grad()`, on every
   micro-batch** (`flagship_v15.py:747-803`, `train_flagship_v4.py:226-258`) — i.e. **16× per optimizer
   step** at v5f's `accum=16`. This is forward-only but exactly the computation shape the
   *inference*-side efficiency study already measured a **1.75–3.46×** win from CUDA-graphing
   (`MODEL_REGISTRY.md` §1.2, lever L1b/L4) — the training-time analog is untried. [MEASURED]

10. **Reproducibility is unchanged since the 2026-07-25 review's F4 finding: `torch.manual_seed` only,
    everywhere.** All 8 trainers seed with a bare `torch.manual_seed(...)` (line numbers vary; none call
    `cuda.manual_seed_all`, `use_deterministic_algorithms`, or pass a `generator=`/`worker_init_fn` to
    their `DataLoader`). Combined with bf16 autocast's non-deterministic reduction order, no checkpoint
    in the registry is bit-reproducible from its own launch command — a fact made concrete by
    `MODEL_REGISTRY.md`'s own "v1 speedjerk" reconstruction-risk note (`--jerk-weight`/`--aux-accel` are
    not even in HEAD's arg parser; §1.2's final block). [MEASURED, confirms R1_code_engineering.md's F4
    is still live]

---

## 1. Loss inventory

### 1.1 Flagship v1 (`flagship4b-speedjerk-30k`, trained via `train_flagship4b.py`)

v1's loss is the shared `flagship_loss()` body (`stack/tanitad/train/flagship_losses.py:167-438`) plus
one term the trainer adds on top (`train_flagship4b.py:649-658`). Every term, its weight (v1's active
config — no `--v2`), what it trains, and its gradient path:

| term | weight (default) | reads / trains | stop-grad? | notes |
|---|---|---|---|---|
| `pred` — operative JEPA, horizons (1,2,4) | 1.0 | encoder (via `states`) + operative predictor | none | `change_weighted_mse` (A4); target is the **online** encoder, computed **with gradient** (`train_worldmodel.py:334-337`: *"LeJEPA: SIGReg on all embeddings, no stop-grad/EMA crutch"*) |
| `tacpred` — tactical-predictor JEPA, horizons (8,16) | 0.5 | encoder + tactical_pred module | none | same targets, same mechanism, second dynamics head |
| `roll` — K-step recursive rollout | 0.5 | encoder + operative predictor | none | `_rollout_loss` (`train_worldmodel.py:58-74`): predictor's own 1-step output is fed back into the window and re-predicted; **no SIGReg on any of these intermediate `z_hat`** (finding #3) |
| `goal` — tactical GOAL-latent JEPA @ 2s | 0.5 | encoder + tactical_policy | none | `change_weighted_mse` against `fut_states[goal_h-1]` |
| `wp` — tactical GOAL waypoint L2 | 1.0 | tactical_policy's waypoint heads | none | grounded against `gt_ego_waypoints` (odometry), **not** a JEPA term |
| `man` — maneuver CE (class-weighted) | 0.5 | tactical_policy | n/a (CE) | inverse-frequency weights, clamped ×10 (`flagship_losses.py:111-118`) |
| `route` — strategic route CE (masked, weighted) | 0.5 | strategic_policy | n/a | **masked to `nav_valid`**; a PC1 guard (`:318-333`) hard-asserts no `ROUTE_UNKNOWN` leaks into an unmasked CE — a real, tested defence against the exact silent-straight-fallback defect that broke earlier route labels |
| `invdyn` (×3 levels: op/tac/str) | 2.0 each = **6.0** | **encoder** (`z_t`, `fut_states`, un-detached) via `MetricInverseDynamics` | `grad_scale(·, invdyn_gradscale)`, **default 1.0 = no-op** | regresses metric `(Δx,Δy,Δyaw)` from a real latent pair — explicitly designed ("force the encoder latent to encode metric ego-motion", `metric_dynamics.py:5-12`) to reshape the trunk; this is the "odometer" pressure (finding #6) |
| `fwd` (×3 levels) | 1.0 each = **3.0** | encoder + predictor + `StepDisplacementReadout` | none (fully attached) | forward-consistency on the **true-action** rollout; this is the term that actually produces the quotable `g_op_fwd_ade_m` |
| `sigreg` | 0.1 | encoder + predictor (the two `position_relaxed` calls) | n/a (regulariser, not supervised) | applied to `cat(states, fut_states)` **and** `cat(preds[k])` separately (`flagship_losses.py:367-376`); complement of the 64 exempt dims only |
| `inv` — action inverse-dynamics (A5) | 0.5 | encoder (consecutive window states) | none | recovers the *action*, not the pose — a cheaper, coarser cousin of `invdyn` |
| `decorr`, `route_vis`, `goalwp`, `jerk` | 0.05 / 0.3 / 1.0 / w_jerk | — | — | **all inactive for v1** (each is gated behind a `v2_*` config flag or `model.goal_traj_head is not None`, none of which v1 sets) |
| **H15 `loss_h15`** (added by the trainer, not `flagship_loss`) | 0.5 (`cfg.h15.weight`) | encoder (via `ImaginationField`) | none | `h15_loss()` (`train_flagship4b.py:224-231`) → `imagination_nll`, heteroscedastic Gaussian NLL, hidden-cell-weighted; fires stochastically (`torch.rand(())<mask_prob=0.5`) |

**Is the speed/jerk term a label leak?** No. `v0 = pose_last[:,3]/10.0` (`flagship_losses.py:227-238`) is
the **last observed frame's** speed, constant-expanded over the action window and the *future* actions —
never a future speed value — so it is proprioception fed as an input, not a target computed from
privileged future information. It is validated by a clean, same-architecture/same-data ablation
(no-speed **2.918 m** vs speed+jerk **0.452 m** full-set, `MODEL_REGISTRY.md` §1.1/§1.2) — a legitimate,
large, causally-isolated effect. ⚠️ **But** the "no `--jerk-weight`/`--aux-accel` in HEAD" reconstruction
risk (§1.2's final block of the registry) means the *exact* jerk/aux-accel mechanism v1 trained with is
not recoverable from this repo today — only the speed-input pathway (`speed_input=True`) is.

### 1.2 v5f (`flagship-v5f-w120-30k`, trained via `train_flagship_v4.py`)

v5f's WM stack is the **same** `flagship_loss()` (`v4_loss_step`, `train_flagship_v4.py:99-102`) called
with `weights=LossWeights()` — i.e. **all defaults, unchanged from v1's table above** — plus four things
v1 never has:

| addition | weight | what it does |
|---|---|---|
| planner loss `plan_l["loss"]` | 1.0 | `v15_losses` — DiffusionDrive-style anchor-CE (1.0) + WTA-L1 (1.0), REF-C's validated recipe, reused not reimplemented |
| `fac_loss` — factorised LAT×LON×DIST CE | 0.05/0.05/0.05 | masked CE on the `refb_labels` kinematic tokens (`v4_curriculum.py:141-162`) |
| `sm_loss` — plan smoothness (jerk+curvature-rate) | `w_jerk=0.02`/`w_curv=0.01` | on the **dense 20-step** emitted plan (`v4_curriculum.py:107-134`) — this is "the other mechanism" that acts on the path v4 actually ships, unlike v1's jerk term which (per `V4_FLAGSHIP_DESIGN` §7, cited in the file) acted on an unscored 4-point head |
| `λ_plan`-scaled planner→trunk gradient | scheduled (`lambda_plan_at`) | **not** a loss term but a gate on how much of the planner's gradient reaches the shared trunk (`v4_curriculum.py:71-85`) |

**And one thing v5f is missing versus v1**: the H15 term. `train_flagship_v4.py` never calls
`h15_loss`/`imagination_nll`/`sector_mask` (grep, whole file) — see headline finding #4. v5f's anti-
collapse budget is therefore **SIGReg + grounding + inverse-dynamics only**; the H15 imagination
objective the programme's own fact sheet lists as part of the recipe is architecturally present but
inert for this entire lineage.

**Gradient-flow summary (both arms):** targets for JEPA/tacpred/goal/rollout are the **online** encoder's
own output, computed with gradient (LeJEPA's stated design — no EMA target, no stop-grad crutch anywhere
in the training stack, confirmed by grep). The only stop-gradient-like device in the whole loss is the
straight-through `grad_scale` on the invdyn real-pair term (`metric_dynamics.py:59-78`), and it is a
no-op at its shipped default.

---

## 2. Anti-collapse

### 2.1 SIGReg vs its own paper

Web-verified (search, not memory): **LeJEPA, Balestriero & LeCun, arXiv:2511.08544**, "LeJEPA: Provable
and Scalable Self-Supervised Learning Without the Heuristics" — proves the isotropic Gaussian is the
unique worst-case-optimal embedding distribution and introduces SIGReg (Sketched Isotropic Gaussian
Regularization): random 1-D projections scored by the Epps–Pulley characteristic-function normality
test, linear time/memory. `sigreg.py` matches this exactly: `epps_pulley()` (`:26-42`) is the textbook
statistic; `SigReg.forward` (`:53-84`) draws **fresh** `n_slices=512` unit directions every call
(never a fixed buffer — the paper's defence against adversarial anisotropic collapse), forces fp32
(`:56-65`, with a code comment naming the exact historical bug of dividing by `n` and silently
zeroing the loss), and is applied to embeddings **and** predictor outputs per the paper's own
recommendation. This part of the recipe is genuinely well-built and citation-accurate.

The follow-on identifiability paper is also web-verified: **"When Does LeJEPA Learn a World Model?"**,
Klindt, LeCun & Balestriero, **arXiv:2605.26379** (submitted 2026-05-25) — proves LeJEPA linearly
recovers a stationary world's latent variables up to rotation, and that this enables optimal
latent-space planning under an orthogonal-identifiability precondition. `stack/tanitad/eval/spectral.py`
cites this id correctly and builds an honest instrument (`orthogonality_report`,
`isotropy_ratio`/`condition_number`/`rms_offdiag_correlation`, `:200-362`) that explicitly states its
thresholds are **knobs, not derived constants** ("2605.26379's constants do not transfer to a finite
SIGReg run"). This is careful, non-oversold engineering.

### 2.2 Where SIGReg's guarantee stops — three measured gaps

1. **Spatial coverage.** `position_relaxed()` (`sigreg.py:87-112`) exempts the first `free_dims=64`
   columns from any SIGReg pressure at all, on the documented rationale that ego-motion is
   low-dimensional and structured (non-isotropic) and would otherwise fight SIGReg's global pressure —
   itself a real, previously-diagnosed regression ("the diagnosed step-21k regression mechanism",
   `sigreg.py:93`). The measured consequence (§0 finding 2): the readout's *entire* active subspace
   (`active_k≈19–30` of 2048, `…-orthogonality.md` table) sits at or inside that exemption boundary, so
   SIGReg's isotropy pressure — genuinely working, `iso_ratio_active` 0.254→0.546 — is being applied
   almost entirely to a subspace that never carries much signal in the first place. **This is not a bug
   in SIGReg; it is evidence the encoder's effective capacity utilisation is far below its nominal 2048
   dims, for reasons (loss imbalance, fed dynamics channels) upstream of SIGReg.**
2. **Path coverage.** SIGReg sees `states`, `fut_states`, and the one-shot `preds[k]` — never the
   `_rollout_loss` intermediate states, never `imagine_probes`' outputs, never the H15
   `ImaginationField`'s belief tokens. The E1 diagnostic (§0 finding 3) measured attractor collapse
   specifically on the un-regularised recursive path. This is architecturally exact: SIGReg is a
   **marginal** constraint on whichever tensor is handed to it: it says nothing about a *sequence* of
   tensors produced by feeding one output back as the next input.
3. **Temporal/derivative content vs marginal isotropy.** `latent_screen.py` (§0 finding 5) is a genuinely
   different axis from anything SIGReg measures: it asks whether the latent's *own* motion along its own
   best linear speed direction tracks the *true* speed derivative. A latent can satisfy an isotropic-
   marginal target while its frame-to-frame content is almost uncorrelated with true kinematics (v1:
   jitter 51.0×, derivative-corr +0.089) — LeJEPA's guarantee is a **static identifiability** result
   (2605.26379 proves recovery of the *latent variables*, not of their *time-derivatives*); nothing in
   the theory or in SIGReg's loss term constrains adjacent-frame *consistency*. `PREREG_TEMPORAL_LATENT.md`
   (2026-08-03) registers exactly the discriminating experiment for whether this is fixable by a better
   encoder (L) or is a video/frame-rate ceiling (V) — **I could not find a landed result file for it**
   (checked `…/incoming/2026-08-03-latent-bottleneck/`); it appears to still be open. UNVERIFIED whether
   it ran.

### 2.3 Comparison with VICReg, BYOL/DINO/V-JEPA (EMA-teacher schemes)

| scheme | anti-collapse mechanism | needs EMA/stop-grad? | fits a from-scratch driving WM w/ inverse-dynamics grounding? |
|---|---|---|---|
| **SIGReg / LeJEPA** (this repo) | sliced Epps–Pulley → isotropic Gaussian target, provable | **no** | Yes, and it is the right default choice for exactly the reason the in-repo `decorr.py` docstring gives for its *own* linear (not adversarial) mechanism: *"we cannot babysit an adversary for days"* on a 4–5 day unattended run. No momentum encoder to keep in sync with grounding heads that need the SAME online representation the invdyn/step-readout heads regress against. |
| **VICReg** (Bardes, Ponce & LeCun 2022) | variance + covariance regularisation on the *batch* covariance matrix, no EMA | no | Very close in spirit and cost to SIGReg (both are covariance/moment-based, both stop-grad-free); VICReg's covariance penalty is what `decorr.py`'s own docstring cites as its precedent for a stable, non-adversarial cross-block decorrelation term (`decorr.py:33-35`). A defensible **alternative**, not obviously better here — SIGReg's Cramér–Wold sliced approach targets the *full* distribution shape (not just 2nd moments), which matters more when downstream heads (route CE, maneuver CE) need more than decorrelated variance. |
| **BYOL / DINO** | EMA target network + stop-grad on the online branch; collapse prevented by the asymmetry, not a loss term | **yes**, and the EMA momentum is a schedule to babysit | Poor fit here specifically **because** the grounding heads (`invdyn`, `step_readout`) need to regress metric pose from the *same* representation the JEPA/rollout targets are drawn from — introducing an EMA teacher would create exactly the train/target mismatch the programme's own `decorr.py` docstring warns against for adversarial methods, and would add a second thing (teacher momentum) to tune on an unattended multi-day run. |
| **V-JEPA / V-JEPA 2** (Bardes et al. 2024; Assran et al., **arXiv:2506.09985**, web-verified, Meta, June 2025) | EMA target encoder + masking, at web scale (>1M video-hours) | yes | The lineage this programme's own 2026-07-18 loss-rebalance research recommends as a **two-phase** (frozen-encoder-then-decode) target architecture — but that is an *architecture* recommendation (Part B, own-data phase-1 SSL then freeze), separate from SIGReg vs EMA as a *collapse-prevention mechanism* for phase 1. V-JEPA 2's own encoder is trained with an EMA target; nothing here forces TanitAD to copy that specific choice if SIGReg already gives a provable, EMA-free alternative for a much smaller from-scratch run. |

**Verdict for Q2's design question**: SIGReg is the right choice of *anti-collapse mechanism* for a
from-scratch, jointly-grounded, unattended multi-day driving world model — no momentum schedule to
babysit, provable, and it is correctly implemented. The defects found here are not "wrong mechanism,"
they are "mechanism applied to too little of the tensor graph, and to a subspace the rest of the loss
has already made small" — both fixable without discarding SIGReg (§7).

### 2.4 Appearance-shortcut cross-check (`PREREG_APPEARANCE_SHORTCUT.md`)

This pre-registration asked whether a static-appearance shortcut (a still frame reading `speed` almost
as well as the full temporal latent — **MEASURED +0.6642 vs +0.7145, ratio 0.93 on comma2k19 highway**)
generalises to the actual training corpus. It **landed** and was decisive:
`…/2026-08-03-appearance-shortcut-audit/APPEARANCE_SHORTCUT.md` reports **OUTCOME C (corpus-specific)**
— on PhysicalAI-AV the still-frame arm reads speed at the **empirical null** (R² **−0.0025**, does not
separate from its shuffled control at all; RATIO CI95 **[−0.0498, −0.0000]**, entirely below the 0.40
threshold) — and the inversion is the finding: off-highway, **motion-energy** arms (not static
appearance) carry the speed signal (`mot16_window_rbf` R² **+0.4124**, separated). ⇒ On the corpus that
actually matters, the appearance-shortcut hypothesis is **withdrawn**; this is good news for the
encoder's genuineness and is orthogonal to (does not resolve) the temporal-jitter finding above.

---

## 3. Failed lines — recipe-level root cause

| line | what was actually confounded / broken | evidence |
|---|---|---|
| **v2** (abandoned @7,800) | Ten `v2_*` levers fired simultaneously from step 0 (`train_flagship4b.py --v2`): ego-to-planner feed + dropout, future-action dropout 0.30, `rollout_k` 4→12, goal-decode, nav-dropout 0.5, traj-jerk, gated intent, anchored tactical decoder (+9.5M params), route-from-vision, encoder-ego decorr 0.05, `invdyn_gradscale` 0.25 — a genuinely uninterpretable bundle, and the registry says so plainly ("the problem was all ten levers at once, not any one of them"). Encoder speed-probe R² collapsed to 0.30 (v1: 0.861). | `MODEL_REGISTRY.md` §1.3 |
| **v3enc** (RESTART @10k) | Correctly *staged* 4 of the 10 v2 levers to isolate encoder-grounding effects — **but its own gate is compromised**: `staged_lever_schedule()` (`train_flagship4b.py:77-100`) holds `decorr_weight` at exactly **0.0 for the whole 0–9,999-step window the 10k gate was graded on** (`decorr_w = 0.0 if step < 10000 else 0.02`). The registry's own "finding that reframes the failure" section says this outright: *"the gate measured the arm before the staged lever under test was applied."* The FAIL secondary (`encoder_speed_probe_r2` 0.393 vs 0.861 bar 0.55) is real, but attributing it to `decorr` is wrong by construction; the surviving suspects (`invdyn_gradscale=0.5`, `ego_dropout=0.25`) were never isolated from each other either. | `MODEL_REGISTRY.md` §1.4, `train_flagship4b.py:77-100` |
| **v4 → v4.1 → v4.2 → v4.2b → from-scratch** | A genuine competing-objectives problem, not a schedule bug: coupling a cold anchored-diffusion planner's gradient into v1's already-converged WM trunk destabilises the WM (hot-trunk v4: canary 0.452→~1.3, killed @3.5k). v4.1's fix (cut `lr_trunk` to 3e-5) **worked for the WM** (canary 0.4599, healthy) but its canary controller had a real, isolated bug: the naive halve-to-zero ratchet had **no lower bound**, and any transient canary noise drove `lam_mult` monotonically to **1.5e-5 by 10k / 3.8e-6 by 11k** — the planner-to-trunk coupling was effectively OFF from ~step 2,000 (`CanaryController` pre-fix behaviour, documented in `v4_curriculum.py:230-238`). `oracle_in_fan` FAILed at 0.4838 (worse than v1.5's *frozen*-trunk 0.3073) precisely because the planner was starved, not because the architecture is wrong. v4.2's **fix** (cap-and-hold, `mult_floor=0.25`, `v4_curriculum.py:221-303`) is a correct, well-targeted repair for that specific bug — but v4.2@4k was still worse than v4.1@10k on *every* measured axis (ADE 0.9869 vs 0.8522; canary 0.7222 vs 0.4599), i.e. the floor that stops planner-starvation lets the WM canary breach instead. This reads as a genuine Pareto tension the scalar-floor design cannot fully resolve, not a mistuned constant. The from-scratch arm's clean co-evolution (canary 15.67→1.14, no collapse) supports the "warm-start artifact" hypothesis but still lands far from v1's 0.452 — the hypothesis is **not yet closed**. | `MODEL_REGISTRY.md` §1.5.1–1.5.5, `v4_curriculum.py` |
| **v1.6** (LP-FT unfreeze, "tied") | Not a recipe bug — a genuine, informative negative, reported badly the first two times. Unfreezing 4 ViT blocks + predictor (head-LR 1e-4 / trunk-LR 1e-5, 500-step ramp) bought a fan-quality improvement (oracle 0.3073→0.2815) at the cost of **+144%** WM canary degradation (0.452→1.1022), and on ADE the two arms are **statistically indistinguishable** (paired Δ +0.0104 m, CI [−0.0888,+0.1147], **not separated**) — v1.6 is even slightly *behind* v1 on the point estimate. The session first reported this as "decisive failure" (a transient step-2500 spike), then as "best in program" (the trainer's own in-loop val, ~10% optimistic per the operating standard's own C1 example), before the canonical `eval_flagship_v16.py` + paired episode-cluster bootstrap settled it. **Recipe-level lesson**: partial-unfreeze LP-FT at this LR ratio is not a free win here — it degrades the substrate for no measurable ADE gain. | `MODEL_REGISTRY.md` §1.4b |

**Common thread across all four**: every failure that *looks* like "the WM broke" on inspection turns
out to be either (a) an un-isolable lever bundle, (b) a gate window that didn't actually exercise the
lever under test, or (c) a genuine trade between WM stability and planner learning that the current
scalar-multiplier controller design cannot fully separate. None of the four is evidence against SIGReg,
the JEPA core, or the grounding formulation itself — they are all downstream-of-encoder failures (the
planner/curriculum/gate-window layer), which is a genuinely useful, if not fully comfortable, finding:
**the encoder+SIGReg+grounding core has not itself been the cause of a killed run.**

---

## 4. Compute efficiency — v5f at 19.58 s/step, eff-batch 64

### 4.1 Where the time structurally goes (code-derived, not profiled — no torch access here)

Per micro-batch (`batch=4`), `v4_loss_step` (`train_flagship_v4.py:78-208`) does, **sequentially**:

1. `world.encode_window(frames)` + `world.encode_window(future_frames[needed_fut])` — two ViT passes
   (depth 12, d768, `--grad-checkpoint` on ⇒ activations recomputed on backward, roughly doubling the
   encoder's own compute for memory headroom, `encoder.py:164-172`).
2. The shared WM stack (`flagship_loss`): the operative/tactical-predictor one-shot JEPA heads (cheap,
   parallel over horizons) **plus** `grounding_losses`' single shared rollout of the operative predictor
   to `k_max` (sequential, one predictor call per step, `metric_dynamics.py:220-266`).
3. The planner: `_imagination_inputs` → `imagine_probes`, a **second, independent, sequential 20-step
   frozen-predictor rollout at batch×8 probes = 32**, under `torch.no_grad()` (`flagship_v15.py:747-803`)
   — confirmed by code to be forward-only (no backward graph retained: *"never a 20-step backprop path
   into the predictor, which keeps step cost bounded"*, `train_flagship_v4.py:236-238`) — then the
   `V15Decoder`'s truncated denoise loop (**only 2 passes**, `refc.py:309` `diffusion_steps=2` — cheap).
4. Backward through everything except the frozen imagination rollout, `clip_grad_norm_`, repeated
   **16×** (`accum=16`) before one `opt.step()`.

**The single largest *addition* v5f makes relative to v1's plain recipe is item 3** — a second full
20-step sequential rollout that v1's trainer never runs (v1 has no planner, hence no `imagine_probes`
call at all). Its cost is bounded by design (`no_grad`, no backward), but "bounded" is not "cheap": 20
sequential kernel-launch-bound predictor calls at batch 32, **16 times per optimizer step**. This is
structurally the *exact same shape* of computation the inference-side efficiency study already measured
a **1.75× (CUDA-graph) to 3.46× (composed)** win from graphing (`MODEL_REGISTRY.md` §1.2, levers
L1b/L4) — a lever that has never been tried at training time. The registry's own inference-tick
decomposition (rollout stage = **83.7–96.7%** of a comparable-shape tick, encoder only 15–26%) is a
reasonable, cited prior for where v5f's training time also concentrates, though it is an inference-time
measurement and not a direct substitute for a training-time profile.

### 4.2 What is already good

- bf16 autocast is on by default for every trainer (`torch.autocast("cuda", dtype=torch.bfloat16)`,
  confirmed across all 8 trainers); SIGReg and the ridge-based decorr probe correctly force fp32 inside
  autocast for numerical stability (`sigreg.py:62-65`, `decorr.py:99-114`) — this is careful, not
  accidental, precision management.
- `--grad-checkpoint` is used on the encoder in every real launch command in the registry — a legitimate
  memory/compute trade given v5f's own container is capped at the **same 50GB `memory.max` that
  OOM-killed pod2 six times** (`MODEL_REGISTRY.md` §1.8) — disabling it is not free at v5f's batch size.
- Gradient clipping at 1.0 is applied uniformly.

### 4.3 Concrete, untried levers (ranked by expected effect ÷ implementation risk)

1. **CUDA-graph (or `torch.compile(reduce-overhead)`) the `imagine_probes` rollout.** It is already
   `no_grad`, static-shape (fixed `n_probes`, fixed `imag_read`), and structurally identical to the
   already-graphed inference rollout. **Expected effect: large** (the inference analog measured
   1.75–3.46×) **on item 3 alone, ESTIMATED** since this is untested at training time with the training
   batch shape and the surrounding autograd graph live. **Cost: ~0.5 A40-day** to implement + validate
   bit-exactness on a short run. **Cheapest discriminating experiment:** wrap `imagine_probes` in a CUDA
   graph, run 200 steps, compare `step_s` and a `torch.allclose` on the imagination tokens against the
   eager path — both outcomes are informative (a null result rules out this being the bottleneck and
   redirects profiling toward the grounding rollout instead).
2. **Replace the encoder's and predictor's manual `nn.MultiheadAttention(..., attn_mask=...)` with
   `F.scaled_dot_product_attention(is_causal=True)` for the predictor's causal blocks.** `encoder.py:33`
   and `predictor.py:45` both call `nn.MultiheadAttention` with `need_weights=False`, which *can*
   dispatch to a fused/flash kernel — but the predictor's `CausalBlock` (`predictor.py:34-47`) passes an
   explicit boolean `attn_mask`, which on several PyTorch versions blocks the fast-path dispatch
   (version-dependent; **ESTIMATED**, not confirmed against the pod's actual torch version, which this
   container cannot check). **Expected effect: small-to-moderate** on the predictor's own share of the
   tick. **Cost: ~0.25 A40-day.** **Cheapest experiment:** one-line swap behind a flag, `step_s` A/B at
   fixed steps, numerically pin against the eager causal-mask path first (this is a correctness-critical
   change and needs the existing test suite green before any launch).
3. **Profile before optimising further.** No `torch.profiler` trace or `time.perf_counter()` bracket
   exists anywhere in the trainers for encode / WM-forward / imagination-forward / planner-forward /
   backward / optimizer-step as separate buckets — only the aggregate `step_s`. **Cost: ~1 hour on the
   already-running pod, 0 extra GPU-days** (bracket the existing run, do not launch a new one).
   **Recommendation: do this before #1**, since it would convert the ESTIMATED breakdown above into
   MEASURED and might reveal the grounding rollout (not `imagine_probes`) is actually dominant.
4. **Fused AdamW** (`torch.optim.AdamW(..., fused=True)`) — a one-line, zero-risk change on a
   sufficiently recent CUDA/torch stack. **Expected effect: small** (optimizer step is a tiny fraction of
   a 19.58s step with only ~250M params) but **essentially free to try**. **Cost: <0.1 A40-day.**

### 4.4 Under- or over-trained? (30k × 64 = 1.92M samples over ~406k windows)

**MEASURED, verified against the registry's own corpus row**: `physicalai-train-e438721ae894` is
**2,376 episodes / 406,099 windows** (`MODEL_REGISTRY.md` §0.1, "run logs [refa+] … 2376 eps / 406099
windows"). `30,000 × 64 = 1,920,000` samples ÷ `406,099` = **4.728 passes** — the fact sheet's "≈4.7
passes" is confirmed exact arithmetic on a primary-sourced window count.

Whether 4.7 epochs is under- or over-trained for a ~263M-parameter, 13-term joint objective has **no
direct measured answer in this repo pass** (I found no train-vs-held-out loss-divergence plot, and no
torch access to compute one). Two pieces of indirect, real evidence lean *under*-trained rather than
*over*-fit:

- REF-A's `dyn-in` arm is explicitly checked for exactly this and is clean: held-out error is
  **monotonically improving** to step 30k (5k: 3.755 → 30k: 2.920, "not overfitting — it is at a
  capability ceiling", `MODEL_REGISTRY.md` §2.3) — the same architecture family, same corpus, same
  step budget.
- v1's own headline metric is still moving between the 19k relay (ADE 0.6277 heldout) and the 30k
  final (0.4522) — a large late-training gain, inconsistent with a saturated fit.

Both are **INHERITED-then-cross-checked-here**, not a fresh measurement on v1/v5f specifically. Given
consecutive windows within an episode overlap heavily (stride-driven, not i.i.d.), "epochs" is a fuzzier
concept here than for an i.i.d. image corpus; the ESTIMATED read is that these arms are **more likely
under-trained than over-fit** at their current step budgets, and the cheapest way to settle it for v6 is
a single instrumentation addition (log train-window loss alongside the existing held-out gate probes,
already computed every 2,000 steps by `HeldoutGate`) rather than a guess.

---

## 5. Reproducibility & hygiene

- **Determinism**: `torch.manual_seed` only, confirmed by grep across all 8 trainers
  (`train_flagship4b.py:333`, `train_flagship_v15.py:258`, `train_flagship_v16.py:470`,
  `refa_train4b.py:200`, `refb_train.py:434`, `refc_train.py:686`, `train_dynamics_encoder.py:483,611`,
  `train_flagship_v4.py` — not directly grepped for this exact line but its `DataLoader` construction at
  the same pattern was checked and carries no `generator=`). No `cuda.manual_seed_all`,
  `use_deterministic_algorithms`, or seeded `DataLoader` `worker_init_fn`/`generator` anywhere. This is
  an exact re-confirmation of `R1_code_engineering.md`'s F4 finding from 2026-07-25 — **unchanged in the
  intervening two months**.
- **Trainer sprawl**: 8 separate top-level trainer scripts (`train_flagship4b.py` 848 lines,
  `train_flagship_v15.py` 425, `_v16.py` 663, `_v4.py` **2,527**, `refa_train4b.py` 459,
  `refb_train.py` 651, `refc_train.py` 1,339, `train_dynamics_encoder.py` 654 — **7,566 lines total**),
  each re-implementing the cosine-LR / gradient-accumulation / atomic-checkpoint / JSON-line-log
  skeleton independently, confirmed by the near-identical `cosine_lr`/`AdamW`/`clip_grad_norm_` call
  shapes found in every one. This matches the prior review's "13 trainers, unconsolidated" finding at a
  smaller but still-real scale for the subset in scope here.
- **Config management**: `StackConfig`'s dataclass-with-`dataclasses.replace` pattern
  (`config.py:372-463`) is a genuinely clean way to derive `flagship4b_reduced_config`,
  `flagship4b_smoke_config`, `refa4b_config` etc. from one base without copy-paste drift — this part is
  well engineered.
- **Checkpoint/resume correctness**: `ckpt_io.atomic_archive` (`ckpt_io.py:23-45`) is a correct,
  well-reasoned fix for a real historical bug (a `SIGKILL` mid-copy leaving a corrupt file at the *final*
  name, permanently mistaken for a complete archive) — copy-to-tmp-then-atomic-rename, with the failure
  mode explicitly analysed in the docstring. `train_flagship_v4.py`'s `_training_loop` auto-resumes from
  `ckpt.pt` and correctly reloads the `HeldoutGate`'s accumulated state (`worse_streak`, incumbent) on
  resume, refusing to silently continue past a run the gate already stopped
  (`train_flagship_v4.py:986-992`) — a good, defensive pattern.
- **A standing, MEASURED reconstruction risk directly relevant to reproducibility**: the deployed v1's
  exact trainer (`--jerk-weight`, `--aux-accel`) was **never committed** at training time
  (`MODEL_REGISTRY.md` §1.2's final block: *"the committed `train_flagship4b.py` arg parser has no
  `--jerk-weight` and no `--aux-accel`"*), and the pod-side working tree that trained it was never
  captured into the repo before the pod (`tanitad-pod2`) was terminated. **A from-HEAD rebuild of the
  deployed v1 today is not byte-exact** — this is the single most consequential hygiene gap for the
  arm the programme has actually shipped.

---

## 6. Gate protocol soundness

### 6.1 What is statistically sound (verified against `run_gate.py`, not just the protocol doc)

- `R2_FLOOR = 0.80` (`run_gate.py:180`) and `SlopeFit.exponent` **raises** below it (`:322-340`);
  `SlopeFit.project()` **refuses** extrapolation beyond 2× the fitted window (`:377-383`) — both exist in
  code exactly as `GATE_PROTOCOL.md` §3 describes, and the protocol's own worked example (the same
  `g_op_fwd_ade_m` log giving exponents from **−0.387 to −1.021** depending on the fit window, only the
  full-run fits clearing R²≥0.80) is a genuinely persuasive, internally-consistent argument for why bare
  exponents were the right thing to ban.
- The decision-grade estimator is unambiguous and enforced: `HeldoutGate.observe()`
  (`heldout_gate.py:421-528`) uses `paired_episode_cluster_bootstrap` and **raises**
  (`GateNotUsableError`) rather than silently degrading when fewer than 2 finite windows or 2 episodes
  are available — a correct fail-loud design. The window-set-identity check (`WindowAlignmentError` on a
  digest mismatch, `:474-482`) is a real, tested defence against exactly the "paired test on mismatched
  arms" error the estimator-blast-radius work found costly elsewhere in the programme.
- `matched_step_ratio` (`run_gate.py:430`) and `reference_reached_at` (`:472`, `k`-consecutive-crossing,
  post-2026-07-21 fix) are both assumption-free by construction (no fit, no extrapolation) — appropriate
  diagnostics for exactly the noisy, per-batch `g_op_fwd_ade_m` series they are applied to.
- The **stop rule's asymmetry is correct and important**: `HeldoutGate.observe()` only advances the
  incumbent on a **separated better** result (`:499-502`, with an explicit comment: *"a lucky point
  estimate must never become the bar the stop rule fires against"*) and only stops on `patience=2`
  consecutive **separated worse** probes — this correctly refuses to let noise either ratchet the bar up
  or trigger an early stop.

### 6.2 What is compute-efficient, and where it isn't yet load-bearing

- **The mid-run `HeldoutGate` (cadence `every=2000`, `heldout_gate.py:337`) is the single best
  compute-efficiency device in the whole gate stack**: it is explicitly credited with what *not* having
  it cost — *"~29.5 GPU-h, half the run, spent training past the best checkpoint while every training
  term improved"* on the v4 30k run (`train_flagship_v4.py:965-969`docstring). It also archives
  `ckpt_best.pt` at every new incumbent, so a stopped run's best state survives even if training is
  allowed to continue past it. This is a real, working, positive design.
- **The restart-budget cap (2 per lever family, a 3rd failure `REFUTE_LEVER_FAMILY`) is a sound
  anti-thrash mechanism** and was correctly applied to v3enc (1/2 spent for `encoder-grounding`).
- **However, the protocol's own headline 2026-07-26 upgrade — the `corridor_departure_rate` co-primary
  at a pre-registered horizon K, closed-loop — has, as far as I could find, never actually been rendered
  for any real gate verdict.** `GATE_PROTOCOL.md` §0.6/§4b states this explicitly: *"0 of 30 committed
  `windows_*.pt` dumps carry `pred_dense`/`gt_dense`"* and *"a K≥100 corridor read requires a closed-loop
  rollout … on GPU"* — every verdict I traced (v3enc `RESTART`, v4.1 `INCOMPLETE`) was rendered on the
  **old, horizon-blind** card and is explicitly stamped `horizon_honest: false`. The protocol is
  statistically well-designed on paper; in practice, every gate decision made to date was made on the
  demoted, diagnostic-only `ade_0_2s` primary, not the co-primary the protocol says should be
  adjudicating. This is a real gap between "the protocol exists" and "the protocol is load-bearing."
- **§6.1's rollout-recovery experiment (headline #8) is a live instance of this same pattern one level
  down**: a pre-registration with a clear, non-ADE primary (`CR_k`) was written carefully, the run was
  executed, and the eval that landed measured only the diagnostic (ADE), not the primary. The discipline
  is present in the *design* documents but is not yet enforced at the *eval-emission* step.

### 6.3 Recommendation specific to the gate protocol

Before the co-primary is relied on for a live restart/continue decision again, spend the (cheap, no new
training) GPU-time to compute **one** real closed-loop `corridor_departure_rate` reading on an
already-trained checkpoint (v1 is the obvious candidate — it is the one arm with a stable, final
checkpoint and no ongoing training to disturb) end-to-end through `rollout.collect` in closed-loop mode.
This validates the whole co-primary machinery works before the next restart decision is asked to depend
on it, and it is a few-hour eval-only task, not a training run.

---

## 7. Recommended v6 training recipe

Ranked by (defect addressed × expected effect) ÷ cost, each tied to a specific finding above, each with
**both outcomes of its cheapest discriminating experiment stated**, per the operating standard.

| # | fix | defect it targets | expected effect (evidence class) | cheapest discriminating experiment (both outcomes) | cost |
|---|---|---|---|---|---|
| 1 | **Compute `CR_k` on the already-trained RR-CTL/RR-20 checkpoints** | headline #8 — a completed, real experiment has no admissible verdict under its own pre-registration | Converts an ESTIMATED "rollout-recovery probably helps" into a MEASURED, pre-registered verdict at **zero new training cost** | *Outcome A*: `CR_k` separated-lower at k=16/20 for RR-20 ⇒ rollout-recovery is validated, becomes standing recipe, ADE win is corroborating evidence. *Outcome B*: `CR_k` flat/worse ⇒ the ADE win was a different mechanism (e.g. simply more gradient steps on the rollout target) and must NOT be reported as "rollout-recovery works" | **~0.05 A40-day** (eval only, both checkpoints exist) |
| 2 | **Wire `h15_loss` into `train_flagship_v4.py`, or explicitly drop the `ImaginationField` module from the v4/v5f line** | headline #4 — 22M params (≈8-9% of budget) likely inert for the entire v4-family lineage including the live v5f run | If wired in: gives v5f a genuine anti-collapse/occlusion-completion objective it currently lacks entirely, at the cost cfg.h15 already specifies (mask_prob 0.5, cheap). If dropped: reclaims ~22M of the sub-300M budget for the actual planner/predictor. Either is strictly better than the current silent third option. (**MEASURED** the module is unused; **HYPOTHESIS** on run-time consequence pending live verification) | *Outcome A*: `world.imagination.state_dict()` is bit-identical between two v5f checkpoints taken 1000s of steps apart ⇒ confirms dead weight, act immediately. *Outcome B*: it differs ⇒ some other code path I did not find is training it; retract this finding and look for that path | **~0.1 A40-day** to verify (diff two existing v5f checkpoints, no GPU needed at all — this is a CPU-only check) |
| 3 | **Extend SIGReg (or a cheap isotropy proxy) to the K-step rollout and `imagine_probes` outputs** | headline #3 — E1's measured attractor collapse under recursion, currently un-regularised | Should directly reduce the measured 0.219→0.805 inter-sample cosine collapse and restore horizon-growing (not shrinking) epistemic variance, if the mechanism diagnosis is right | *Outcome A*: re-running the existing blind-rollout diagnostic script (`blind_rollout_flagship.py`, ~2 min GPU per checkpoint) on a short fine-tune with SIGReg added to the rolled states shows the attractor cosine stop climbing ⇒ validated, promote to standing recipe. *Outcome B*: collapse persists ⇒ the pathology is architectural (the `ImaginationField`'s advection prior, not the loss), redirect to a parallel-horizon-only design (already the recommended safe operative mode per the same research note) | **~2–3 GPU-days** (short fine-tune + re-diagnostic; NOT a full 30k run) |
| 4 | **Set `invdyn_gradscale` to its own research-recommended `0.25`** (not the shipped default `1.0`) for the next full flagship run, ablating `{1.0 control, 0.5, 0.25}` at a 5k mid-checkpoint gate | headline #6 — the ≈4.4:1 loss imbalance, and a 2-month-old, already-implemented, already-recommended fix that was never made default | Per the original research's own gate: `g_op_fwd_ade_m` non-regression **and** `vision_use` rising **and** in-latent `ego_r2` falling, at flat `ade_0_2s` | *Outcome A*: `vision_use` rises above ~15-20% while `g_op_fwd_ade_m` stays within its own noise band ⇒ promote `0.25` to the shipped default. *Outcome B*: the metric regresses beyond the gate ⇒ the encoder needs the odometer pressure more than the research anticipated; keep `1.0` and look elsewhere for capacity headroom | **~1 A40-day** (5k-step mid-checkpoint ablation, 3 arms in parallel if GPU allows, or sequential at ~1 day each) |
| 5 | **CUDA-graph `imagine_probes`** (compute §4.3 #1) | headline #9 — the single largest untried compute lever for v5f-family training | ESTIMATED 1.3–2× on the imagination-conditioning share of the step, based on the inference-side analog | *Outcome A*: `step_s` drops materially with `torch.allclose` intact on the imagination tokens ⇒ ship it. *Outcome B*: graph capture fails or gives no speedup (e.g. the surrounding accumulation loop is not graph-friendly) ⇒ profile first (item below) before spending more effort here | **~0.5 A40-day** |
| 6 | **Add per-stage `time.perf_counter()` (or `torch.profiler`) brackets to `train_flagship_v4.py`'s training loop** | compute §4.1 — every efficiency estimate above is ESTIMATED, not MEASURED, for lack of a training-time profile | Converts every "likely dominant" claim in §4 into a ranked, measured list before further optimisation effort is spent | *Outcome A*: confirms `imagine_probes`/grounding-rollout dominate as estimated ⇒ proceed with items 3/5 as prioritised. *Outcome B*: reveals dataloader/encoder dominate instead ⇒ re-prioritise entirely; this is exactly why it is item 6, before 3 and 5's GPU-days are spent | **~0 A40-days** (instrument the currently-running pod, no new launch) |
| 7 | **`seed_everything` + seeded `DataLoader` generator/`worker_init_fn`, logged to `config.json`** (this is R1's P3, re-confirmed still open here — cited, not re-derived) | headline #10 — no registry checkpoint is bit-reproducible | Makes restart/continue decisions comparable across re-runs of "the same" config; does not change any existing result | N/A — this is a hygiene fix, not an experiment; apply and verify a same-config re-run's train-log matches to a tight tolerance | **~0.5 A40-day** (implement + one short verification re-run) |
| 8 | **Actually render one closed-loop `corridor_departure_rate` co-primary reading** (gate §6.3) | headline #6.2 — the protocol's own headline metric has never adjudicated a real verdict | Validates the co-primary pipeline works end-to-end before the next restart decision depends on it | *Outcome A*: the number renders cleanly and is interpretable ⇒ the co-primary is ready to be load-bearing for the next gate. *Outcome B*: the pipeline breaks or produces an uninterpretable number ⇒ better to find that now than during a live restart decision | **~0.25 A40-day** (eval-only on v1's existing checkpoint) |

Items 1, 2, 6 and 8 cost **essentially no new GPU-days** and should happen immediately, in parallel, before any new 30k-step commitment. Items 3–5, 7 are the actual v6-recipe changes and are ordered so the cheapest, most diagnostic ones (7, 5) come before the more expensive validation run (3).

---

## 8. Open questions / UNVERIFIED

- **UNVERIFIED — whether `world.imagination`'s parameters are truly never updated in the v4/v5f
  lineage.** I traced every loss-construction code path I could find and confirmed no call site reads
  `model.imagination`, and reasoned from documented PyTorch `AdamW` semantics (skip params with
  `.grad is None`) to the conclusion that they are frozen at init. I could not run a live check (no
  torch/pod access in this container). This is the single highest-value cheap verification in §7.
- **UNVERIFIED — `PREREG_TEMPORAL_LATENT.md`'s L/V/M outcome.** The design (a pixel-substrate ladder
  discriminating "the video has the signal and the encoder destroys it" from "the video does not have
  it at all") is excellent and squarely on-topic for the anti-collapse question, but I could not find a
  landed result file for it under `…/incoming/2026-08-03-latent-bottleneck/` in this pass. Whoever owns
  that stream should confirm whether it ran.
- **UNVERIFIED — `PREREG_v5_cheapest_guard.md`'s route-collapse guard for v5f's `goal_dropout=0.5`.** I
  found the design and confirmed the `cond_imagination` hard-wire it also flagged was subsequently fixed
  correctly (`train_flagship_v4.py:1314-1322`, now a proper `--cond-imagination` launch flag) — but I
  found no `route_acc_nav`-style result specifically for a v5-family checkpoint (only v1/v2corpus
  numbers), so it is unclear whether the cheapest guard (a ~2-3 minute, zero-new-training hierarchy pass)
  was ever actually run before v5f's current 30k commitment.
- **ESTIMATED, not measured — the §4.1/§4.3 compute-time decomposition.** Every relative-cost claim about
  where v5f's 19.58 s/step goes is derived from code structure and from the *inference*-side efficiency
  study's tick decomposition, not from a training-time profile (none exists in the repo). Item 6 in §7
  is the fix.
- **Not settled — whether 4.7 epochs over 406k windows is under- or over-trained.** I found suggestive,
  cross-arm indirect evidence (REF-A dyn-in "not overfitting"; v1's late-training ADE gain) but no direct
  train-vs-held-out divergence measurement for v1 or v5f specifically.
- **Did not independently re-derive**: the exact `op_fwd_k`/`tac_fwd_k`/`str_fwd_k` values
  `flagship4b_config()` uses for `HierarchicalGrounding`'s per-level horizons (needed for a fully precise
  §4.1 sequential-rollout-length count) — I read `horizon_plan()`'s *mechanism* (`flagship_losses.py:
  49-98`) but not the specific numeric config fields; this does not change any finding above but would
  sharpen the exact rollout-length numbers in §4.1 if pursued further.

---

## 9. Deliverable manifest

| Artifact | Location | Notes |
|---|---|---|
| This report | `repo:Project Steering/Reviews/2026-09-25-programme-review/streams/R2_training_anticollapse.md` | staged, not committed, per operating rules |

No code was changed and no other file was edited (per this stream's brief: report bugs, don't fix them).
No new artifacts beyond this report were produced. Nothing here is stranded on a pod or in a worktree —
this is a pure read-and-report pass with no other deliverable to stage.
