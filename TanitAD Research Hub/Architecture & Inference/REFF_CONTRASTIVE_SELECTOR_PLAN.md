# REF-F — a contrastive state–action selector for driving, transferred from CLM-8B

**Status:** PLAN (M0), awaiting PI approval. No code, checkpoint or result exists yet. · **Date:** 2026-09-29 · **Author:** orchestrator (Opus 5.5), with research stream F-R (Sonnet) and integration survey F-S (Sonnet).
**Companion documents:** `Project Steering/PREREG_REFF_CONTRASTIVE_SELECTOR.md` (the pre-registration: hypotheses, both outcomes, estimator) · `TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-reff-prior-art-and-theory.md` (research stream) · `Project Steering/Reviews/2026-09-25-programme-review/00_PROGRAMME_REVIEW.md` (why selection is the lever).
**Evidence classes:** MEASURED · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS · UNVERIFIED, as in CLAUDE.md.

---

## 0. The idea in one page

**What CLM is.** CLM-8B, the Contrastive Language Model (Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré, Mirhoseini; Stanford + NVIDIA Research; released 2026-09-23; Apache-2.0), is a "System One" decision model. It does not generate. It maps the current state and each candidate action into one 512-d space with two small trainable heads on a frozen Qwen3-8B, and picks the candidate whose embedding best matches the state. Because action embeddings do not depend on the state, they are computed once and cached, so ranking many candidates costs one matrix product. Its headline results:
- on par with the "Jev" System-1 model on computer use, gaming and tool calling, at up to 9× lower latency;
- 81.6 % on DeepSWE and 87.6 % on Terminal-Bench 2.1 when used as a best-of-N **verifier** over sampled candidates.

All PUBLISHED via the official README; details in §1.

**Why it fits TanitAD now.** The 2026-09-25 review found the programme's decisive defect is selection, not generation:
- REF-C's fan holds a candidate at 0.1640 m ADE@2s but picks one at 0.4714 (MEASURED, `taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json`);
- the hierarchy's own decision paths lose to a flat planner 4.0–7.2× (MEASURED);
- scoring candidates with the world model would be about 22× over the Thor budget (ESTIMATED, review §4.8).

CLM is a published, fast, verifier-grade answer to exactly this problem: score many candidates against the state, cheaply.

**What REF-F is:**
- **Inputs:** camera frames plus ego state (v0, the 0.1 s speed difference `ax_fd`, speed history, yaw rate).
- **State side:** a frozen vision backbone and a small trainable state head produce the state embedding.
- **Action side:** candidate 2 s trajectories come from a vocabulary built on the train split (cached), plus any open-set candidates such as REF-C's fan. A small action encoder embeds them.
- **Selection:** a calibrated, prior-corrected cosine score picks one trajectory. A tracking controller turns it into **driving commands** (steer, acceleration) at 10 Hz, and the review's safety envelope filters them.
- **Cost:** all the trainable parts together are about 15–25 M parameters. They train on cached embeddings in GPU-hours, not GPU-days (ESTIMATED).

**The expected advantages.** Each is a pre-registered hypothesis, not a claim:
- Decisions in microseconds over thousands of candidates.
- Cheap ablations and a cheap data-efficiency curve.
- No mode averaging.
- One verifier for every candidate generator in the programme.
- Retrieval-grounded explanations.
- A calibrated action distribution for the safety layer.
- Measurable roles for the 4B hierarchy (§5).

**What could sink it** (each has a test in §4):
- the expressivity limit of dot-product scoring;
- a frozen backbone that does not carry the decision-relevant scene content;
- vocabulary coverage;
- the pointwise-mutual-information bias of InfoNCE on this corpus's imbalanced action prior (§2.5);
- copycat inertia from ego history;
- closed-loop dithering.

**Budget to the first decision-grade result** (ESTIMATED): implementation 2–3 eng-days with agents; embedding cache ≤ 0.3 A40-day; head training for all three stages ≤ 1 A40-day; val-40/val-600 evaluation ≤ 0.2 A40-day. That is about 1.5 A40-days in total, because the backbone never trains.

---

## 1. CLM-8B, precisely (what transfers and what does not)

| aspect | CLM-8B | source | transfer to REF-F |
|---|---|---|---|
| backbone | Qwen3-8B, **frozen**, last-token pooling, 4096-d, shared by state and action encoders | README, `src/clm/heads.py` (PUBLISHED) | frozen **vision** backbone for the state side (§2.2). The action side gets its own small encoder, because trajectories are low-dimensional numbers, not text |
| heads | two MLPs `hidden → width → … → 512`, GELU, optional LayerNorm/residual, L2-normalised; about 20 M parameters | `heads.py` `make_head(width, depth=2, proj=512, activation="gelu", layernorm=False, residual=False, hidden=4096)` | same form. The state head adds attention pooling over patch tokens and an ego MLP |
| score | `exp(logit_scale) · cos(state_head(s), action_head(a))`; `logit_scale` initialised to log(1/0.07), exp clamped ≤ 100 | `train/finetune.py` | same, plus a log-prior correction and a hysteresis term (§2.5) |
| loss | bidirectional InfoNCE with in-batch negatives; same-task pairs masked (`same = codes.unsqueeze(0) == codes.unsqueeze(1)`); a "Choice" mode = softmax CE over options with soft targets | `finetune.py` | both. InfoNCE for representation pre-training (masking near-duplicate trajectories); vocabulary softmax with soft targets for selection (§3.2) |
| optimisation | AdamW, `lr = 2e-3·sqrt(1024/width)·sqrt(batch/1024)`, OneCycle 10 % warm-up, batch 2048, 20 epochs, patience 5, weight decay 0, trained on **cached fp16 embeddings** | `finetune.py` | same starting point (§3.4) |
| curriculum | about 60 M Nemotron QA pairs → about 30 M synthetic hard negatives → about 1 M agent trajectories with 40 % replay; hard negatives from the start "peaks at 62.4 % then overfits" vs 69.2 % with the curriculum | README; Oaklight/krino issue #76 (PUBLISHED, secondary) | the same three stages (§3.3) |
| inference | `Engine.rank(state, candidates)`; action-embedding arena with LRU; RTX 4090 cold request 58.1 ms, cached state 1.7 → 0.6 ms, p50 28.6–28.8 ms for 3–50 new actions | README | cached vocabulary embeddings, with REF-C's fan embedded on the fly (§2.6) |
| results | zero-shot on par with Jev at up to 9×; **as a best-of-N verifier: DeepSWE 81.6 %, Terminal-Bench 2.1 87.6 %** (4.1–5.7× faster) | README (PUBLISHED; a secondary summary attributes 81.6 % to the baseline instead, and the README is taken as authoritative) | the verifier role is REF-F's mode 2 (§2.7) |
| stated limits | states truncated at 2,048 tokens; no vision yet (a multimodal version is on the roadmap); no System-2 component | README | REF-F adds System-2 re-ranking on demand (§2.8) |

**What does not transfer directly:**
1. **Frozen backbone quality.** Qwen3-8B brings broad world knowledge. TanitAD's evidence is that frozen single-frame image features fail as a *metric ego-motion* substrate: REF-A scored 2.1675 even when handed the expert's controls (MEASURED, registry §6). REF-F is designed so the backbone never has to supply metric ego-motion; ego kinematics come from the ego channel and the vocabulary is residual over a kinematic prior (§2.3–2.4). Whether a frozen backbone carries the *decision-relevant* scene content is hypothesis H-F7, tested, not assumed.
2. **Continuous actions.** CLM's options are discrete strings. Driving actions are continuous 2 s trajectories, which forces a vocabulary with a coverage/quantisation trade-off (§2.4) and a notion of "near-duplicate" actions that InfoNCE would otherwise push apart (§3.2).
3. **Sequential control.** CLM decides per request. A car decides every 100 ms, and flicker between near-equal candidates is a closed-loop failure mode (§2.5, hysteresis).
4. **The action prior.** Agent tasks present a handful of options per question. Driving presents the same vocabulary every tick, and the corpus is about 74 % straight cruising (programme docs). That imbalance changes what the contrastive score means (§2.5).

**A second, literal transfer (optional, label-time only).** The released CLM-8B could be used as-is, as an **offline teacher**:
- render the privileged state as text (ego kinematics plus `obstacle.offline` agent tracks: distances, closing speeds);
- render manoeuvres as text ("keep lane, hold speed", "brake to follow the lead at 2 s");
- let CLM-8B rank them, producing soft targets or hard negatives.

This is admissible, because privileged signals are allowed at label time, and it costs one offline pass of an 8B model over a subset. Its value is a HYPOTHESIS; it is pilot P-T in §8, not part of the core design.

---

## 2. Architecture

### 2.1 Overview

```
 FRONT-WIDE frames: 8 × (3 RGB @ 100 ms, 9-ch) 256²      EGO: v0, ax_fd, v history (8), yaw rate      [goal point: optional, predicted, never from the situation classifier]
            │                                                 │
   ┌────────▼────────┐  frozen, pluggable:                     │
   │ BACKBONE (frozen)│  F-own  = TanitAD v1 ViT encoder (87 M, TensorRT on Thor)          (default)
   │ per-stack cache  │  F-vid  = a frozen video foundation encoder (V-JEPA-2 class)
   └────────┬────────┘  F-img  = a frozen image foundation encoder (DINOv3 / RADIO class)
            │ z[8] (8 × 2048) + last-stack patch tokens pooled to 8×8×768
   ┌────────▼──────────────────────────────────────────────────▼───┐
   │ STATE HEAD  (trainable, ≈ 10–20 M)                              │
   │ temporal MLP/2-layer transformer over z[8] ⊕ learned-query      │
   │ attention pooling over patch tokens ⊕ ego MLP → MLP → 512 → L2   │
   └────────┬───────────────────────────────────────────────────────┘
            │ s ∈ S⁵¹¹
            ▼
  score_j = e^τ·⟨s, a_j⟩ + β·log p̂(a_j) + γ·consist(a_j, previous plan)     (§2.5)
            ▲ a_j ∈ S⁵¹¹  (cached for the vocabulary; on the fly for open candidates)
   ┌────────┴───────────────────────────────────────────────────────┐
   │ ACTION ENCODER (trainable, ≈ 2–5 M): residual trajectory        │
   │ (20 × lateral/longitudinal offsets vs the CTRA prior) +         │
   │ kinematic descriptors (speed profile, curvature, peak accel,    │
   │ jerk) → MLP → 512 → L2                                          │
   └────────┬───────────────────────────────────────────────────────┘
   CANDIDATES: (1) vocabulary V: N = 4,096 residual trajectories (64 lateral × 64 longitudinal, from the TRAIN split only)
               (2) open set: REF-C fan (256), kinematic primitives (hold, brake, accelerate, CTRA), the previous plan shifted by one tick
            │
   SELECT top-1 ─► (optional) refinement head: residual offset on the chosen trajectory
            │   └► System 2 when uncertain: top-8 re-ranked by the world model's reference roll + cost heads (§2.8)
   SAFETY ENVELOPE (review §10.6): candidate pruning before selection (RSS/TTC on tracked agents) + CBF-QP after selection + Simplex fallback
            │
   CONTROLLER: pure pursuit on the selected path + speed feed-forward/PID on its speed profile ─► steer, acceleration @ 10 Hz
```

### 2.2 State side

- **Backbone slot (frozen).**
  - The default is **F-own**, TanitAD's v1 ViT encoder. It is already trained on this corpus, already TensorRT-optimised on Thor, and costs no new parameters against the sub-300 M budget.
  - The foundation-encoder arms (F-vid, F-img) test whether broad pretraining carries more decision-relevant content (H-F7). They are named by class, not by product, until the research stream's latency and licence check picks specific checkpoints.
  - The backbone never trains in REF-F. Partial unfreezing is a separate, later arm (REF-F-ft), never the default. CLM's efficiency and its cached-embedding training depend on the freeze, and the review showed that planner gradients in the trunk degraded the world model (+144 % for v1.6).
- **What the state head reads:**
  - (i) the 8-step latent window `z[8]` (8 × 2048), the world model's compact state;
  - (ii) the **last stack's patch tokens pooled to 8×8×768**. The review found the 4×4×128 readout carries about 30 effective dimensions and cannot hold agents (R1 §5), so the head must see finer tokens;
  - (iii) ego kinematics.
- **Ego inputs:** v0, `ax_fd`, an 8-step speed history and yaw rate. These are the measured longitudinal levers (review §4.4). They are admissible for a planner, as v1 and REF-C already consume v0. The PI's vision-only ruling binds the *situation classifier*, and REF-F consumes no situation-classifier output in any form.
- **Copycat guard.** Ego history can let a policy copy its own recent motion (the "inertia" failure of behaviour cloning). Guards: ego-history dropout at training time (a learned null row, never zero-fill; zero-fill is the X15 lie); reporting transition windows separately from steady windows; and an image-only control arm (§4.3).

### 2.3 What an action is

An action is a **2 s future trajectory at 10 Hz**, 20 steps, expressed as a **residual over the constant-turn-rate-and-acceleration (CTRA) prior** computed from (v0, yaw rate, `ax_fd`). Decomposed in the prior path's frame:
- a **lateral** offset profile (20 values, metres);
- a **longitudinal** progress-offset profile (20 values, metres).

Three reasons:
1. **Holding the current motion becomes the zero residual.** It is always in the vocabulary and trivially representable. That targets the programme's standing defect: every arm is worse than holding speed on the 72 % steady windows (MEASURED, registry §6 reading 4).
2. **The lateral × longitudinal factorisation is explicit.** The tactical decision (lateral manoeuvre × longitudinal manoeuvre) becomes readable and scoreable, which gives the four-family *tactical* metric a native definition (selected vs executed class).
3. **Action embeddings stay independent of the state,** which is what makes them cacheable (the property CLM's speed rests on). The executed trajectory is `prior(v0, ψ̇, ax_fd) ⊕ residual`, reconstructed at inference.

The output matches TanitEval directly (20 waypoints; metrics at 0.5 / 1 / 1.5 / 2 s). Driving commands come from the controller (§2.9).

### 2.4 The vocabulary

- **Built from the parity TRAIN split only** (`physicalai-train-e438721ae894`, 2,376 episodes; the build asserts the cache key and skip-hash `f09e44db`). No val or test trajectory may enter the vocabulary, and the build writes the split key into the vocabulary file as provenance.
- **Construction.**
  1. Compute each train window's residual trajectory.
  2. Select K_lat = 64 lateral and K_lon = 64 longitudinal profiles by **farthest-point sampling** under a time-weighted L2 that weights late waypoints more.
  3. The Cartesian product gives N = 4,096 entries.

  Farthest-point sampling is the default because the programme's own trainer records why k-means fails here: "the corpus is ~74 % straight cruise" (INHERITED, `stack/scripts/train_flagship_v4.py:267`). k-means would spend most entries on straight cruising and starve the rare manoeuvres. REF-C's anchors use farthest-point sampling for the same reason.

  Ablations: k-means (density-following); a joint, unfactorised vocabulary with N = 4,096 or 8,192.
- **Frequency prior** p̂(a_j): the share of train windows assigned to each entry (soft-assigned). It is stored with the vocabulary and used by the prior correction (§2.5).
- **Coverage gate V0**, before any training (0 GPU): oracle-in-vocabulary ADE@2s on val-40 and val-600, residual vs absolute parameterisation, N ∈ {1k, 2k, 4k, 8k}. There is no pre-set pass bar. The number is reported and it sets the refinement head's job.
- **V0 preview, run 2026-09-29 on committed artifacts** (MEASURED; `Research/reff_v0_coverage.py` → `Research/reff_v0_coverage_result.json`; val-40, 881 windows / 40 episodes, full-set mean with episode-cluster bootstrap CI95, B = 2000).
  - **Set:** REF-C's 256 farthest-point anchors drawn from 200 k parity-*train* windows (provenance recorded in the file: source `…/physicalai-train-e438721ae894`). Smaller N are nested prefixes of the same order.
  - **Speed-normalised variant:** each anchor is uniformly rescaled so its implied initial speed equals the window's v0, an inference-time input. This is a crude stand-in for the residual parameterisation.

  | N | absolute oracle ADE@2s | speed-normalised oracle ADE@2s |
  |---|---|---|
  | 16 | 2.3074 [2.1111, 2.5083] | 0.4681 [0.3888, 0.5542] |
  | 64 | 1.1480 [1.0898, 1.2114] | 0.2701 [0.2254, 0.3176] |
  | 128 | 0.8032 [0.7601, 0.8416] | 0.2214 [0.1877, 0.2586] |
  | 256 | 0.5992 [0.5636, 0.6337] | **0.1787 [0.1513, 0.2074]** |

  **Reading:**
  - A fixed *absolute* vocabulary is a poor substrate: even a perfect selector over 256 absolute entries (0.60) could not match REF-C's actual pick (0.47).
  - Factoring out the current speed changes that. 64 speed-normalised entries cover better than 256 absolute ones, and 256 approach REF-C's per-scene refined fan (0.1640) with no refinement at all.

  This supports the residual-over-prior parameterisation of §2.3 and makes a vocabulary of a few thousand residual entries look ample (ESTIMATED; the real residual vocabulary needs train trajectories and is measured in M1).

  **Limits:** oracle coverage, not achievable selection; val-40 only; uniform rescaling is cruder than CTRA.

### 2.5 Scoring, and the one thing the transfer must get right

`score_j = e^τ · ⟨s, a_j⟩ + β · log p̂(a_j) + γ · consist(a_j, previous plan)`

**Why the log-prior term exists.** With negatives drawn from the data distribution, InfoNCE's optimal critic is the pointwise mutual information, `f*(s,a) = log p(a|s) − log p(a) + c(s)` (van den Oord et al. 2018; Poole et al. 2019; PUBLISHED). Arg-maxing PMI over a fixed candidate set is **not** arg-maxing the conditional likelihood. It penalises common actions.

A worked example: action prior "keep lane at speed" 0.70, "slow for lead" 0.20, "lane change" 0.10; in a given state the true conditionals are 0.60 / 0.30 / 0.10.
- PMI: log(0.60/0.70) = −0.15; log(0.30/0.20) = +0.41; log(0.10/0.10) = 0.
- So PMI picks **"slow for lead"**, while the most likely correct action is **"keep lane"**.

This corpus is about 74 % straight cruising, so an uncorrected InfoNCE selector would systematically under-choose the most common correct action, which is exactly the steady-window weakness the programme already has.

Two remedies, both pre-registered (H-F4):
- (a) add β · log p̂(a_j), with β = 1 in theory, fit on a *train* holdout;
- (b) train the selection with a **softmax over the vocabulary** ("Choice" mode, soft targets). That estimates p(a|s) directly, with the prior baked in.

The default is (b) for selection, with InfoNCE kept as stage-1 representation pre-training (§3.3).

**Checked on a toy before any GPU time (V0-toy, MEASURED on synthetic data).** `Research/reff_pmi_toy.py` trains a dual encoder with CLM's parameterisation: L2-normalised embeddings, score = (1/0.07)·cos, bidirectional in-batch InfoNCE with the same-action mask. The toy world has 60 states and 16 actions, and its action prior is dominated by "cruise" (true share 0.50). Results are means over 5 seeds (min–max in brackets), from `Research/reff_pmi_toy_result.json`:

| selection rule | hit rate (P[pick = a fresh sample from p(a\|s)]) | share of states where "cruise" is picked |
|---|---|---|
| exact MAP, argmax p(a\|s) (upper bound) | 0.597 | 0.61 |
| always the most frequent action | 0.503 | 1.00 |
| exact PMI argmax | 0.361 | **0.00** |
| trained InfoNCE (CLM default) | 0.575 [0.571, 0.578] | **0.45** |
| InfoNCE + log-prior, β = 1 | 0.586 | 0.73 (over-corrected) |
| InfoNCE + log-prior, β fitted (β* = 0.5 on all 5 seeds) | 0.594 | 0.61 |
| vocabulary softmax (CLM "Choice") | 0.592 [0.586, 0.605] | 0.60 |

**Reading:**
- The bias is real and has the predicted direction: trained InfoNCE under-selects the dominant action.
- It is smaller than exact PMI's, because a bounded cosine critic only partly represents PMI. So the correction's β must be **fitted**, not set to 1.
- The vocabulary softmax reaches MAP-level accuracy with no correction, which is why it is the default.

**Two rules follow for REF-F:**
1. Whenever contrastive scores rank candidates directly (verifier mode over REF-C's fan), use `e^τ⟨s,a⟩ + β̂·log p̂(a)`, with β̂ fitted on a train holdout.
2. An open candidate takes the frequency prior of its nearest vocabulary entry.

**Limits of this check.** It is a synthetic world, and the effect size on the real corpus is unknown. H-F4 measures it.

**Temporal consistency.** `consist(a_j, previous plan)` is the negative trajectory distance between a_j and the previous tick's plan advanced by 0.1 s. γ trades flicker against reactivity. It is fit on closed-loop runs, and its effect is reported as jerk and plan-deviation in the closed-loop panel.

**Calibration.** The temperature is re-fit on a train holdout (temperature scaling), and conformal thresholds are set on the *softmax* over candidates, giving the trust score that the safety supervisor and the System-2 trigger consume (§2.8).

### 2.6 Candidates and the cache

- **Vocabulary embeddings:** 4,096 × 512 × fp16 = 4 MiB, computed once per checkpoint, resident on device.
- **Open candidates** are embedded on the fly: REF-C's 256, primitives, the previous plan. The action encoder is about 2–5 M parameters, so about 300 candidates per tick is a small batched MLP (ESTIMATED < 1 ms on Thor).
- **Scoring cost:** 4,096 × 512 = 2.1 M multiply-adds, which is microseconds (ESTIMATED).
- **The tick is dominated by the backbone.** Caching the previous seven stacks' encodings means one new stack per tick; review §4.8 records encoder caching at 57 → 33 ms on an A40 for the whole model. The REF-F tick target is pre-registered in H-F5 and measured on Thor, not assumed.

### 2.7 Three modes, one set of weights

1. **REF-F standalone:** select from the vocabulary plus primitives. The flat System-1 reference arm.
2. **REF-F as verifier:** score REF-C's fan (best-of-256), the direct analogue of CLM's DeepSWE/Terminal-Bench verifier result. This mode answers the review's selector question with a published mechanism.
3. **REF-F as the v6 selector:** in the review's v6, REF-F replaces the hard-argmin cross-entropy selector (Δ2). Its cost heads (§3.2) are Δ3.

### 2.8 System 1 and System 2

System 1 (REF-F) runs every tick. When the calibrated margin between the top two candidates falls below θ, or the entropy exceeds h, the top-8 go to System 2: the world model's single reference roll (review Δ8) plus the cost heads re-rank them. Eight candidates fit Thor's batch saturation (SMs saturate at batch 8; CLAUDE.md). θ and h are set to cap System 2 at a pre-registered fraction of ticks (e.g. ≤ 20 %), and the fraction is reported. This is the retrieve-then-re-rank pattern of information retrieval, and it is the review's dual-process control flow (§10.11) with a measurable trigger.

### 2.9 From selection to driving commands

The selected trajectory (prior ⊕ residual, 20 waypoints) is tracked by:
- **pure pursuit**, with look-ahead max(4 m, 0.8·v·1 s), for steering: `steer = atan(L · κ)`, with L = the per-clip wheelbase when the `per_clip_v1` regime is used;
- **speed feed-forward plus PID** on the trajectory's speed profile for acceleration.

This reuses the controllers in `taniteval/taniteval/closedloop.py`. The known 2.7 m vs 2.9 m wheelbase skew (registry §0.1.1) is documented in the adapter, not silently changed. The commands (steer, acceleration) at 10 Hz are REF-F's output. The trajectory is kept for evaluation and for the safety envelope.

### 2.10 Safety envelope

A selection-based planner can only output a candidate it was offered. That makes **pruning before selection** possible: remove every candidate that violates an RSS longitudinal distance or a TTC floor against tracked agents, or exceeds kinematic limits, then select among the rest. After selection, the CBF-QP filter and the Simplex fallback of the review's envelope (§10.6) apply unchanged. REF-F's softmax trust score feeds the envelope's trust gate.

**Perception dependency.** Tracked agents at inference need a vision range and closing-rate estimate, which the 256-px crop cannot give at range (review §4.4). Until a range head exists, pruning runs only in simulation with simulator-provided agent states, labelled as an **oracle-perception** envelope test.

### 2.11 Parameter and latency budget

| part | parameters | trains? | where |
|---|---:|:--:|---|
| backbone F-own (v1 encoder) | ≈ 87 M | no | existing, TensorRT on Thor |
| state head | ≈ 10–20 M | yes | new |
| action encoder | ≈ 2–5 M | yes | new |
| refinement head (optional) | ≈ 1–2 M | yes | new |
| cost heads (optional, dot-product queries) | ≈ 1–3 M | yes | new |
| vocabulary cache | 4,096 × 512 (fp16, 4 MiB) | no | per checkpoint |
| **total** | **≈ 100–120 M** | ≈ 15–30 M trainable | well inside sub-300 M |

---

## 3. Training

### 3.1 The embedding cache (CLM's efficiency, applied)

- **One offline pass of the frozen backbone** over every parity-train and val window. The cache stores, per window, `z[8]` (8 × 2048 fp16 = 32 KiB) and the last stack's patch tokens pooled to 8×8×768 (fp16 ≈ 96 KiB).
- **Size.** About 406 k train windows × 128 KiB ≈ **52 GB**, plus the val splits. That is a pod-side artifact. Check the MooseFS quota with a real `dd` write test, never `df` (CLAUDE.md).
- **Parity.** Asserted in-process: 2,376 episodes, cache key, skip-hash. The cache file records the backbone checkpoint hash.
- **Cost:** one inference pass (ESTIMATED ≤ 0.3 A40-day). The same cache serves every head ablation, every data fraction of the slope experiment, and every backbone arm (one cache per backbone).

### 3.2 Objectives

| loss | what | why |
|---|---|---|
| `L_nce` | bidirectional InfoNCE between each state and its expert residual trajectory, in-batch negatives; **mask pairs whose expert trajectories are within ε ADE@2s** (e.g. 0.25 m, fixed before training) | CLM's representation objective. The mask generalises CLM's same-task mask: two cruising windows have near-identical futures and must not be pushed apart |
| `L_voc` | softmax cross-entropy over the vocabulary with **soft targets** q_j ∝ exp(−d(a*, v_j)² / σ²) | estimates p(a\|s) directly, so the PMI issue disappears (§2.5); uses the whole cached vocabulary as negatives. CLM's Choice mode |
| `L_cost` (stage 3) | per-candidate targets computed **offline with privileged data** (collision and TTC against `obstacle.offline` agent futures, 97.44 % clip coverage; progress; comfort, i.e. jerk and lateral acceleration), predicted by small dot-product query heads | Hydra-MDP-style multi-target distillation (arXiv 2406.06978, PUBLISHED); the review's Δ3. Drivable-area terms are impossible (no map) |
| `L_ref` (optional) | L1 on a residual offset for the selected entry | escapes vocabulary quantisation, as REF-C's anchor offsets do |

The total is `L = L_nce + λ_voc·L_voc + λ_cost·L_cost + λ_ref·L_ref`, with the weights per stage fixed in the pre-registration.

### 3.3 Curriculum: CLM's three stages, translated

| stage | CLM | REF-F | data |
|---|---|---|---|
| 1. pre-training (broad, easy) | about 60 M QA pairs, in-batch InfoNCE | `L_nce` on (state, expert trajectory) pairs | **parity arm:** the parity train split. **Scale arm (a new arm, never re-selection):** the rest of PhysicalAI-AV's HF *train* split, with an exclusion list registered first (never its val/test splits, never a clip whose NuRec scene is in the closed-loop suite or a challenge). Actions come free from egomotion |
| 2. mid-training (hard negatives) | about 30 M synthetic hard negatives | per state, kinematically plausible **near-miss perturbations** of the expert residual: lateral ±0.5–1.5 m at 2 s, speed ±10–30 %, braking early or late, a turn in the wrong direction, plus **privileged-rule negatives** (trajectories that intersect `obstacle.offline` agent futures). Introduced only after stage 1 converges, because CLM found hard negatives from the start overfit | generated on the fly from cached trajectories; CPU |
| 3. post-training (target task) | about 1 M agent trajectories, 40 % replay | `L_voc` + `L_cost` + `L_ref` on the parity corpus, with 40 % stage-1 replay. Later (Phase 2): closed-loop **DAgger-style** states visited by REF-F in NuRec/AlpaSim, labelled by the logged trajectory near the log and by a privileged rule-based teacher beyond it | parity train; NuRec/AlpaSim rollouts |

### 3.4 Hyper-parameters (starting point = CLM's defaults)

- τ initialised to log(1/0.07), with e^τ clamped ≤ 100.
- AdamW with weight decay 0; `lr = 2e-3·sqrt(1024/width)·sqrt(batch/1024)`; OneCycle with 10 % warm-up.
- Batch 2,048, which also sets the in-batch negatives; 20 epochs; early-stopping patience 5.
- Seeds via `seed_everything` plus a seeded DataLoader. The review found the programme sets only `torch.manual_seed`.
- Every choice is written into the checkpoint `cfg`, in CLM's checkpoint layout (`state_head`, `action_head`, `logit_scale`, `cfg`) plus REF-F's `vocab_hash`, `backbone_hash` and `parity_key`.

### 3.5 Compute (ESTIMATED)

| step | cost |
|---|---|
| vocabulary build and coverage gate V0 | CPU, minutes to hours |
| embedding cache per backbone | ≤ 0.3 A40-day |
| stages 1–3 on cached features | ≤ 1 A40-day for the parity arm |
| each additional ablation arm | hours |
| data-efficiency slope (5 fractions × 2 seeds) | ≤ 0.5 A40-day |
| Thor latency and closed-loop panel | Thor time; n as available |

---

## 4. Validation

### 4.1 The ladder, each rung a gate

| rung | what | GPU | passes when |
|---|---|---|---|
| **V0** | vocabulary coverage (oracle-in-vocab ADE vs N, residual vs absolute); leak guard (vocabulary provably train-only); the PMI toy check (**done 2026-09-29**, §2.5); an absolute vs speed-normalised coverage preview on REF-C's train anchors (**done 2026-09-29**, §2.4) | 0 | coverage reported; leak guard green |
| **V1** | unit tests + a CPU smoke run end-to-end on synthetic episodes (`stack/tanitad/data/toy_driving.py`); determinism test | 0 | `pytest -q` green, including the smoke |
| **V2** | head training on cached features; open loop on **val-40 and val-600**, full-set means, **paired episode-cluster bootstrap**, all **four families**, with every control arm | ≤ 1.5 A40-day | the pre-registered primaries adjudicated (H-F1–H-F4, H-F7) |
| **V3** | Thor: full-tick p50/p95; closed loop on NuRec/AlpaSim, paired vs REF-C-base, n reported (12 scenes are underpowered; grow the suite) | Thor | H-F5, H-F8 adjudicated |
| **V4** | data-efficiency slope: heads at 1 / 3 / 10 / 30 / 100 % of parity-train hours vs REF-C at the same fractions; exponents only with fit window, R² ≥ 0.80 and n (CLAUDE.md) | ≤ 0.5 A40-day + REF-C runs | H-F6 adjudicated |

### 4.2 Pre-registered hypotheses

Full text, with both outcomes, in `Project Steering/PREREG_REFF_CONTRASTIVE_SELECTOR.md`.

| id | hypothesis | primary measurement | if confirmed | if refuted |
|---|---|---|---|---|
| H-F1 | REF-F standalone is **non-inferior** to REF-C-base on ADE@2s | paired Δ upper CI bound < +0.02 m on val-600, reported on val-40 too | REF-F becomes the fast System-1 reference arm | the dot-product bottleneck binds; move to retrieve-then-re-rank (REF-F top-k → REF-C-class cross-attention scorer) |
| H-F2 | the residual vocabulary makes REF-F the first arm **not worse than holding speed** on steady windows while beating it on transients | steady-window speed MAE vs hold-v0 (paired); transient windows separately | the residual/prior parameterisation is adopted programme-wide (review Δ9) | the steady-window loss is not a parameterisation problem; look at the ego inputs and labels |
| H-F3 | as a **verifier** over REF-C-XL's 256-candidate fan, REF-F recovers ≥ 25 % of the oracle gap | paired pick-ADE vs REF-C's own pick, same windows | adopt REF-F as the v6 selector (D2b) | selection needs early interaction or cost targets; keep the E-GOAL-4 regressor line |
| H-F4 | InfoNCE-only argmax **over-selects rare manoeuvres**, and either the log-prior correction or the vocabulary softmax removes it | selected-class vs ground-truth-class frequency (KL divergence); ADE on steady windows | the correction is mandatory in any contrastive selector | drop the correction; the corpus prior is not strong enough to matter |
| H-F5 | the REF-F tick on Thor is ≤ 50 ms p95 with 4,096 cached + 300 open candidates | Thor, bf16, TensorRT backbone | REF-F leaves ≥ 50 ms of the 100 ms budget for System 2 and the envelope | profile; the backbone dominates and must be cached or shrunk |
| H-F6 | REF-F's data-efficiency curve is **flatter** than REF-C's (a smaller loss at 10 % and 3 % of hours) | four-family panel at each fraction, paired at matched fractions | the first measured evidence for the mission's "far less data" goal | no data-efficiency advantage from frozen backbones; the claim rests on other levers |
| H-F7 | a frozen **foundation** backbone (F-vid or F-img) beats the frozen v1 encoder (F-own) as REF-F's state substrate once ego kinematics come from the ego channel | H-F1 metric per backbone, paired | frozen foundation encoders are viable for decisions (unlike REF-A for odometry); settles BACKLOG B5 | the programme's own encoder is the better substrate at this scale |
| H-F8 | in closed loop REF-F is **non-inferior to REF-C-base** on pass rate and collisions, with no worse jerk than REF-C (the hysteresis works) | NuRec/AlpaSim paired, n reported | REF-F is a closed-loop reference | open-loop quality does not survive closing the loop; strengthen stage 3 (DAgger) |

### 4.3 Controls every V2 panel carries

- **kNN retrieval:** copy the expert action of the nearest training state in the same frozen feature space. If REF-F ≈ kNN, the heads learn little beyond the backbone.
- **Ego-only REF-F:** no images. This measures the vision contribution, against the programme's no-vision ego-status ceiling of 0.5735.
- **Image-only REF-F:** no ego history. This measures copycat dependence.
- **Hold-v0, CV, CTRV** floors on the same windows.
- **A random-vocabulary control** (entries sampled rather than clustered).

### 4.4 Statistics and reporting rules

These come from CLAUDE.md and are not repeated in full here:
- full-set means with the episode-cluster bootstrap (B = 2000), **paired** for any two arms, never the deprecated split-mean;
- **four families per panel.** STRATEGIC is reported as UNAVAILABLE with its reason (no route source in PhysicalAI-AV), per rule 5, never silently dropped;
- trainer logs are curve watches only; only eval output is quotable;
- every number carries its evidence class;
- REF-F enters `MODEL_REGISTRY.md` when its first checkpoint exists, with its reconstruction chain (config, cache hashes, vocabulary hash, parity key).

### 4.5 Leak and admissibility checks, run in code

- The vocabulary file carries the train split key, and the loader refuses a vocabulary whose key is not the parity train key.
- Val-window trajectories are asserted absent from the vocabulary's source set.
- No situation-classifier output is present in any REF-F input tensor (a named-input allowlist, asserted at construction).
- The scale arm refuses any clip on the registered exclusion list.

---

## 5. The advantages, stated as things to measure

1. **Decision speed.** One matrix product ranks thousands of cached candidates, so selection is effectively free and the tick is the backbone's cost (H-F5). The review found that world-model scoring of 256 candidates would be about 22× over the Thor budget.
2. **Training economy.** The heads train on cached embeddings in hours. The decision layer, the programme's measured lever, becomes cheap to iterate, and the data-efficiency curve (the mission's first priority goal) becomes affordable to measure (H-F6).
3. **No mode averaging.** Selection over a set cannot regress to the mean of two futures, which is the failure of v1's unimodal tactical head (3.3839 m, MEASURED).
4. **One verifier for everything.** REF-C's fan, kinematic primitives, the world model's proposals, or a future generator can all be scored in one space. This is CLM's verifier result applied to driving (H-F3).
5. **Explanations by precedent.** Every decision can be shown with the nearest training states and the actions they took, which is retrieval-grounded evidence for a safety case and for UN ADS / WP.29 transparency expectations.
6. **A calibrated action distribution.** The softmax over candidates gives the trust score the safety supervisor needs (review Bet B) and the System-2 trigger (§2.8).
7. **Safety by construction of the output space.** Only offered candidates can be chosen, so kinematically infeasible and rule-violating trajectories can be removed before the choice (§2.10).
8. **Continual and modular.** New actions are embedded without retraining. New data retrains only the heads. Per-rig or per-region heads are cheap, and region adaptability is a mission goal.
9. **The 4B hierarchy with measurable roles.** Tactical = the factorised lateral × longitudinal choice, scored by the four-family tactical metric. Strategic = the supervisor reading REF-F's uncertainty (review Bet B). Operative = controller plus envelope. The world model = the System-2 re-ranker. Each role has a number that can falsify it.

---

## 6. Risks, and where the plan catches each one

| risk | why it is real | caught by | mitigation |
|---|---|---|---|
| dot-product (late-interaction) expressivity | a bi-encoder cannot model every state–action interaction a cross-encoder can (retrieval literature) | H-F1, H-F3 | retrieve-then-re-rank (§2.8); a small cross-attention re-ranker as a named arm |
| frozen backbone lacks decision-relevant content | REF-A's failure (for odometry); the v1 readout's ~30 effective dimensions | H-F7, the kNN control | patch tokens in the state head; foundation-backbone arms |
| vocabulary coverage | continuous actions quantised | V0 | residual parameterisation, refinement head, open candidates |
| PMI bias | InfoNCE estimates PMI (§2.5) | H-F4 | vocabulary softmax or log-prior correction |
| false negatives | many states share near-identical futures | stage-1 ablation (mask on/off) | distance mask, soft targets |
| copycat inertia | ego history predicts ego future | image-only control; transition-window metrics | history dropout with a learned null row |
| closed-loop flicker and compounding | per-tick selection; open-loop training | H-F8, closed-loop jerk | hysteresis γ; stage-3 DAgger |
| no strategic information | the corpus has no route (review §4.2) | STRATEGIC reported UNAVAILABLE | a strategic test only on AlpaSim routes or L2D |
| the literal CLM-8B teacher adds nothing | untested | pilot P-T | drop it; the core design does not depend on it |

---

## 7. Integration into the TanitAD stack (M1)

Everything below comes from the integration survey (stream F-S, read-only, 2026-09-29). File:line references are at `main` `467ce8a`. Keys: `R` = `stack/tanitad/refs`, `S` = `stack/scripts`, `D` = `stack/tanitad/data`, `T` = `taniteval/taniteval`.

### 7.1 Facts that shape the implementation

- **Naming.** Only REF-A, REF-B and REF-C exist; there is no REF-D or REF-E. "REF-F" is free.
- **REF-C's interface**, which REF-F mirrors: `RefCModel.forward(frames[B,8,9,256,256] in [0,1], nav_cmd, v0, …)` (`R/refc.py:1834`). It returns `traj[B,4,2]` at horizons 5/10/15/20, the refined fan `anchor_traj[B,N,4,2]`, `anchor_logits[B,N]` and `sel_idx`. Anchors are farthest-point samples over parity-train ego-frame waypoint targets (`S/build_refc_anchors.py:36-77`), stored as a persistent buffer in the checkpoint.
- **Windows** come from `EpisodeWindowDataset` (`D/_contract.py:104`) via `FailLoudWindowDataset`/`load_cached_episodes` (`S/refb_train.py:122,233`):
  - window 8, stride 1, max horizon 20;
  - items carry `frames`, `future_poses[20,4]`, `pose_last`, `actions[8,2]`, and always `future_frames[20,…]` (about 47 MB per item). REF-F's cache builder must avoid that last payload.
- ⚠️ **Causality trap.** `actions[:,-1]` is the action *after* `pose_last` (`S/refb_train.py:281`). REF-F must never feed `actions` as an input. It derives its ego inputs from poses only:
  - `v0 = pose_last[3]`;
  - yaw rate = wrap(Δyaw)/0.1, as at `stack/tanitad/train/flagship_losses.py:203`;
  - `ax_fd` = (v_t − v_{t−1})/0.1 from the pose speed channel. Today `ax_fd` exists only in `S/lead_state_gate.py:190`, from the egomotion parquet, and is not in the episode cache.
- **CTRA does not exist yet.** Only CTRV (`constant_yaw_rate`, `S/driving_diagnostic.py:109`) and `T/ctrv_backfill.py:90` do. REF-F adds a CTRA prior.
- **Parity refusal is soft by default.** `D/parity.py:467 assert_parity_corpus` only prints NON-PARITY unless `require=True` (the precedent is `S/train_flagship_v4.py:708`). REF-F calls it with `require=True`.
- **Seeds.** No `seed_everything` exists: trainers call `torch.manual_seed` (`S/refc_train.py:686`) and the DataLoader shuffles with the global RNG (`:920`). REF-F adds its own seeded helper.
- **Evaluation contract.** An arm's `collect` returns `pred[N,4,2]` (waypoints 5/10/15/20), `gt`, `cv`, `eid`, `speed`, `head_deg`, `wp_steps` and `method` over windows `range(0, T−W−20, 8)` (881 windows / 40 episodes; `T/refc_eval.py:109-194`). For the four families it also needs `ctrv`, `pred_dense`/`gt_dense[N,20,2]`, `dense_steps` and `dt_s` (`T/four_families.py:507`, `T/driving.py:563`), and `maneuver_pred`/`maneuver_gt` for TACTICAL (else UNAVAILABLE, `:317`).
- **Existing fan dumps.** `taniteval/results/fan_refc-{xl,base}-30k.pt` hold `fan[W,N,4,2]`, `logits`, `sel`, `gt`, `cv`, `eid` and `v0` for val-40 (`T/refc_rerank.py:234`). REF-F's verifier mode can run on these with no new REF-C inference. Their candidates have 4 waypoints, so they are upsampled in time to 20 steps before embedding.
- **Closest precedent for a learned selector:** `RefCRescorer` (`stack/tanitad/models/refc_rescorer.py:388`, top-K 8) with its cache/train/eval chain `S/refc_v12_{cache,train,eval}.py`. Reusable pieces:
  - `geom_features` (`refc_rescorer.py:284`);
  - `reachability_mask` (`R/refc_select.py:177`);
  - farthest-point sampling (`R/refc.py:159,193,217`);
  - lateral × longitudinal class labels from poses (`R/refc_tactical.py:234`, `S/refb_labels.py:1294`);
  - ego-frame helpers (`stack/tanitad/ego_plan.py:86-216`);
  - controllers (`T/closedloop.py:172 wp_to_control`, `:196 bicycle_integrate`);
  - the feature-cache precedent `S/dino_precompute.py:96` and its shard/parity pattern in `S/refc_v12_cache.py:136`.
- **Thor export.** Only the operative predictor has a TensorRT path (`S/build_predictor_trt.py`). REF-F's heads start eager and bf16, and a TensorRT export is an M3 task.

### 7.2 Files M1 adds

| file | contents | reuses |
|---|---|---|
| `stack/tanitad/refs/reff.py` | `RefFConfig` (dataclass; presets `reff_smoke_config`, `reff_base_config`); `StateHead` (temporal mixer over `z[8]`, learned-query attention pooling over patch tokens, ego MLP with a learned null row, MLP → 512, L2); `ActionEncoder` (lateral/longitudinal residual profiles + kinematic descriptors → MLP → 512, L2); `RefFModel` with `encode_state`, `encode_actions`, `score(s, A, prior, prev)`, `select`, `refine`; losses `bidirectional_infonce(masked by ε)`, `vocab_soft_ce`, `cost_heads_loss` | CLM head form (`make_head`), in-house tensors only |
| `stack/tanitad/refs/reff_prior.py` | ego inputs from poses only (`v0`, yaw rate, `ax_fd`, 8-step speed history); **CTRA prior**; residual ↔ absolute conversion (`prior ⊕ residual`); upsampling 4-waypoint candidates to 20 steps | `ego_plan.py`, `flagship_losses.py:203` convention |
| `stack/tanitad/refs/reff_vocab.py` | farthest-point-sampled 64 × 64 residual vocabulary from the **train split only**; frequency prior; provenance (split key, source hash); `coverage(vocab, gt)`; refuses a non-train key | `R/refc.py` farthest-point sampling |
| `stack/scripts/reff_embed_cache.py` | one frozen-backbone pass per split: `z[8]` + pooled patch tokens + ego features + future residuals → shards, with `assert_parity_corpus(require=True)`, backbone hash, cache manifest; skips `future_frames` | `refc_v12_cache.py:136` shard pattern |
| `stack/scripts/reff_train.py` | the 3-stage curriculum on cached shards; seeded; `--smoke --device cpu` path; checkpoint `{state_head, action_head, logit_scale, cfg, vocab_hash, backbone_hash, parity_key}` + `config.json` in the TanitEval shape | `refc_train.py` checkpoint/config conventions |
| `taniteval/taniteval/reff_eval.py` | `collect` in three modes: standalone vocabulary, verifier over a fan dump, v6-selector hook. Emits `pred`, `pred_dense`, `gt`, `gt_dense`, `cv`, `ctrv`, `eid`, `speed`, `head_deg`, `wp_steps`, `dense_steps`, `dt_s`, and `maneuver_pred`/`maneuver_gt` from the factorised vocabulary | clone of `refc_eval.py:109-194`; `rollout.save_windows` |
| edits: `T/registry.py`, `T/loaders.py`, `T/runner.py` | registry keys `reff-*`; a loader branch mirroring REF-C's (`loaders.py:129-150`); `direct_head` + dispatch in `runner.py:59,85-99` | — |
| `stack/tests/test_reff.py` | shapes and L2 norms; InfoNCE equals a hand-computed value on a 3×3 case; the ε-mask removes exactly the near-duplicate pairs; vocabulary softmax soft targets sum to 1; the vocabulary refuses a non-train key; no `actions` tensor reaches the model (named-input allowlist); CTRA with zero acceleration equals CTRV; 4 → 20 upsampling preserves the waypoints; cache round-trip equality; determinism under a fixed seed; **CPU smoke: embed → train 50 steps → select → controller on toy episodes (the `test_refc.py` `_drive_episode` pattern)** | `stack/tests/test_refc.py:45-86,568` |
| `TanitAD Research Hub/Architecture & Inference/Research/reff_pmi_toy.py` | the V0 toy, already written (§2.5) | — |

### 7.3 Order of work in M1

Agents, per the review's routing: Sonnet implements, Opus reviews the loss and leak code.
1. `reff_prior.py` + tests (causal inputs, CTRA).
2. `reff_vocab.py` + the V0 coverage run on the committed val-40 GT (CPU).
3. `reff.py` + unit tests.
4. `reff_embed_cache.py` + `reff_train.py` with the CPU smoke.
5. `reff_eval.py` and the TanitEval wiring, including verifier mode on the committed REF-C-XL fan dump. Verifier mode is the first real number REF-F can produce without new REF-C inference.
6. `pytest -q` green in an environment that has torch. Cloud sessions currently cannot install it; a pod or the dev box can.

---

## 8. Milestones, budget and decisions

| milestone | contents | cost | gate |
|---|---|---|---|
| **M0** (this document) | research, design, pre-registration draft | — | PI approval |
| **M1** implementation | §7 files + tests + CPU smoke + V0 coverage tool | 2–3 eng-days (agents: Sonnet implements, Opus reviews) | V0, V1 |
| **M2** first result | cache + stages 1–3 + V2 panel (val-40, val-600, four families, controls) | ≈ 1.5 A40-days | H-F1–H-F4, H-F7 |
| **M3** deployability | Thor tick; closed-loop panel | Thor time | H-F5, H-F8 |
| **M4** data efficiency and adoption | slope (V4); verifier mode over REF-C; the decision on v6 (review D2b) | ≤ 1 A40-day | H-F3, H-F6 |
| **P-T** (optional pilot) | the released CLM-8B as an offline teacher over text-rendered privileged states | one offline 8B pass over a subset | its own pre-registration |

**Decisions for the PI:**
1. Approve REF-F as a reference arm on the parity corpus.
2. The default backbone (F-own proposed).
3. Whether the scale arm may use PhysicalAI-AV's HF train split beyond parity, as a new arm with a registered exclusion list.
4. Whether pilot P-T runs.
