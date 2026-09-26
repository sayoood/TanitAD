<title>DriveZero deep analysis — the RL teacher is not what wins; goal augmentation is, and that is the anti-echo mechanism our v7 gate has been missing</title>

# DriveZero (`2609.06055`, Xiaomi EV L3 Team, Sept 2026) — full teardown, successor map, comparison to TanitAD, and a reverse-engineering plan

**2026-09-20 · Research Lab (LAB-RUN-017) · Architecture & Inference · PI request · full text read locally (32 pp, 104,780 chars, banked PDF)**
⛔ **0 GPU.** All numbers below are either quoted from the paper's own tables (**PUBLISHED**, re-read from the banked PDF — not the fetch summariser) or **DERIVED** by arithmetic on the paper's stated settings (`code/dz_budget.py`, two internal consistency checks pass). Nothing here is MEASURED by us.
⭐ Plan: **`REVERSE_ENGINEERING_PLAN.md`** beside this file.

---

## 0 · Findings first — the seven that matter

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **The paper's own ablation undercuts its title.** Table 7, supervision block: **human trajectories 93.92 · DriveRL teacher trajectories 93.61 · DriveRL + goal augmentation 94.41.** The RL teacher **alone is 0.31 BELOW plain human imitation**; the **entire +0.80 gain comes from goal augmentation** — querying the goal-conditioned teacher under *alternative* intents on the same scene. ⇒ **"Beyond human demonstrations" is earned by supervision DIVERSITY, not by the teacher being better than humans.** | PUBLISHED (Table 7) |
| **F2** | ⭐⭐⭐ **That mechanism is exactly the anti-echo lever our v7 gate has been missing.** Our P-1 defect (no v7 arm beats its own hold-action control) and our nav-echo leak (369/369) both trace to the same root DriveZero names in its first paragraph: *"Each logged scene contains only one realized future, even when multiple actions would be valid."* Goal augmentation manufactures the counterfactuals a log cannot contain. ⭐ **And it is separable from the RL teacher** — which is what makes it affordable for us. | analysis on F1 |
| **F3** | ⭐⭐⭐ **I-1's untouched QUALITY half is answered externally, and the answer is SMALL.** Table 3, value-guided test-time search on a **fixed** checkpoint: N=8 **+0.11** · N=16 **+0.12** · N=32 **+0.39** · N=64 **+0.56** CLS. ⇒ an **8× candidate increase buys +0.56 points (0.60 % relative)**, and **below N=32 it is indistinguishable from noise**; one of six settings got *worse* (Test14-hard R 89.18 → 89.15). ⭐ Their N (breadth, parallel) vs L (depth, sequential) split **independently reproduces our own 2026-09-01 cost finding** (breadth 5.94× cheaper). | PUBLISHED (Table 3) |
| **F4** | ⭐⭐ **The teacher is 5.7 M parameters and the experience is almost entirely synthetic.** DERIVED from Table A1: **51.9 billion transitions**, **2,883,584 simulated driving hours ≈ 329 simulated years**, generated in **21 h wall-clock on 96 GPUs (2,016 GPU-hours)** — about **1,430 sim-hours per GPU-hour**. The 922,703 real nuPlan scenes **only seed scenes and goals**. | DERIVED (`raw/dz_budget.txt`) |
| **F5** | ⛔⭐⭐ **The privileged teacher is the most uncomfortable policy in its own table.** navhard Table 5, EC (extended comfort): **DriveRL 18.2 (S1) / 10.9 (S2)** against every other entry in the 28.4–79.1 range. ⭐ The *student* reads **51.6 / 41.3** — so WTA distillation over 20-step trajectories **launders the teacher's discomfort**. That is an unremarked property of the method and a risk for anyone copying the teacher directly. | PUBLISHED (Table 5) |
| **F6** | ⛔⭐ **Data scaling HURT the hardest closed-loop tier, and the prose steps around it.** HUGSIM Table 6, **Extreme**: RC **43.3 → 39.8** and HD-Score **28.7 → 27.6** going from DriveZero to DriveZero-Scale. The text claims SOTA *"on the Easy, Medium, and Hard tiers"* — **Extreme is omitted, not explained**, and the non-Scale model is bolded there. | PUBLISHED (Table 6) |
| **F7** | ⭐⭐ **Partial reproduction is genuinely available — and unlicensed.** `github.com/XiaomiAutoL3/DriveZero` ships **DriveRL inference code + checkpoints** (2026-09-15), Python 3.11, nuplan-devkit, CUDA 12.8, with fixed benchmark filters `driverl_val14 / test14_hard / test14_random`. **DriveZero and DriveVFM are TODO, unreleased.** ⛔ **No license statement was located in either README** ⇒ default all-rights-reserved; a PI/legal item, not a Lab decision. | PUBLISHED-RELEASE + a named absence |

---

## 1 · What DriveZero actually is

Three separately-trained parts, combined once:

```
  nuPlan logs ──seed scenes+goals──▶ [mixed-agent GPU sim] ──PPO──▶ DriveRL teacher (5.7 M, privileged)
                                                                          │ rolled out per logged frame,
                                                                          │ under AUGMENTED goals
  LAION-2B/IN-21K/SA-1B + OpenDV/nuPlan/Waymo ──agglomerative distillation──▶ DriveVFM (ViT, frozen)
                                                                          │
                            multi-view cameras ──▶ DriveVFM+LoRA ──▶ DriveZero student (338 M) ◀── WTA + score loss
```

The claim that organises everything: *"perception must understand the world, and benefits from massive and diverse visual data; action must interact with it, and requires closed-loop feedback."* Each half is pretrained in the regime that suits it, then joined. **The teacher is discarded at inference.**

### 1.1 DriveRL — the privileged teacher

| item | value |
|---|---|
| **Parameters** | **5.7 M** |
| Observation | privileged **structured** state: ego + ≤ **96 agents** × **5 frames @ 5 Hz**, ≤ **256** vector-map tokens, traffic-light states attached to the elements they govern |
| Navigation intent | **two goal points** (near + far anchor) in ego frame, look-ahead **scaling with speed**, recomputed every step, permutation-invariant — *the same representation at train and deploy* |
| Network | 3 encoders → 256-d tokens · **2** ego-to-agent cross-attn layers · **1** ego-to-map · concat + MLP → policy head |
| **Action space** | **longitudinal jerk** `[−8, 5] m/s³` and **tire steering-angle RATE** `[−0.8, 0.8] rad/s` — *rate-level*, then a kinematic bicycle model; caps 4.0 m/s², ±π/3 rad |
| Distribution | independent **Beta** per channel, shape params forced to (1, ∞) via `1 + softplus` (CaRL) ⇒ unimodal, no extreme-command mass. Train = sample, eval = analytic mode |
| Optimiser | **PPO**, γ 0.99, GAE λ 0.95, clip 0.2/0.2, entropy 0.01, 4 epochs, **Muon** optimiser, lr 3e-4 cosine |
| Reward | `r = h + (1−d)(g + (1/H)·Π_k q_k)`, H = 110 — hard-event penalty, one-time goal bonus, and **six soft scores MULTIPLIED** so they must be *jointly* satisfied |
| Critic | **decomposed one value channel per reward term**, channels summing to the total return |
| Init | **random**; no imitation pretraining |

**Mixed-agent simulation.** Background actors get independent providers behind one physical state–action interface: **log replay**, **rule-based** (batched IDM; 100 m lane search, 1.5 s headway, 0.1 s integration), **front-vehicle emergency braking** (3 % of in-corridor vehicles marked, 5 % trigger/step, −10 m/s² to a stop), and **learned self-play** (GigaFlow-inspired). Pedestrians stay on replay. **Scene-consistent actor insertion:** a late-entering actor is released only if the ego is within **5 m and 0.35 rad** of the logged pose and it would not collide — a neat fix for the "ego left the log, so the insert is nonsense" problem.

⚠️ **Self-play gave only a small benefit on nuPlan**, and the authors explain why honestly: nuPlan's own closed-loop evaluation drives background vehicles by IDM, *"and therefore does not reward more natural interaction behavior."* Self-play is reserved for **real-world deployment**.

### 1.2 Value-guided test-time action search (the part that speaks to I-1)

Not a planner — a **local policy-improvement operator**:
1. candidate 0 = the Beta **mode**; candidates 1..N−1 = **samples from the same policy** (so the search is a conservative extension of the deployed policy);
2. each candidate fixes the **first** action, then the **mode** continues from the induced state for **L = 5** steps; background actors extrapolated by **constant turn-rate and acceleration**;
3. score `S(i) = Σ γ^ℓ Π(1−d_u) r_ℓ + γ^L Π(1−d_u) V(O_L)` — **the same reward, discount and critic PPO trained on**;
4. execute the best sampled candidate **only if it beats the mode by margin δ = 0.03**, else execute the mode.

⭐ **N enlarges parallel compute; L adds sequential steps.** That is our breadth/depth distinction, stated by an opponent, and it is the design they chose for the same reason.

### 1.3 DriveVFM — agglomerative distillation into one backbone

Replaces annotated auxiliary heads (detection/lane/seg/depth) with **frozen foundation models as proxies**, following RADIO:

| teacher | supervises | rationale |
|---|---|---|
| **DINOv3** | summary **and** patch tokens | spatial structure, correspondence |
| **SigLIP2** | **summary only** | image-level / open-vocabulary semantics |
| **SAM** | **patch only** | boundary-sensitive, segmentation-like |
| **Depth Anything V2** | features (**not** its depth predictions — *"feature distillation performs better in our experiments"*) | geometry |

One shared ViT emits **one summary token per teacher** plus **shared patch tokens**; per-teacher lightweight adaptors; cosine loss on summaries, MSE on patches. Two mechanisms make it work:
- **PHI-S standardisation** of each teacher's patch features (PCA–Hadamard rotation spreading variance equally, then one scalar rescale) so no teacher's activation statistics decide its effective weight — *per-model loss weights cannot fix this, because the imbalance is also across dimensions within one model*;
- **QK-Clip** (Kimi K2) instead of QK-Norm, to stop attention-logit blow-up.

**Schedule:** 256² for **600 K** iters, then 512² for **200 K**, global batch **2,048**, AdamW ⇒ **DERIVED 1.638 billion image samples**. Corpus mixes **LAION-2B + ImageNet-21K + SA-1B** with driving images from **OpenDV + nuPlan + Waymo**. After pretraining the teachers and adaptors are **discarded**.

### 1.4 DriveZero — the camera-only student

| item | value |
|---|---|
| Cameras / resolution | **4** (`F0, B0, L0, R0`) at **960 × 512** |
| Output | **20 steps @ 5 Hz = 4 s**, **64** trajectory proposals + a predicted score each |
| Backbone | **DriveVFM ViT-L, FROZEN**; only **rank-32 Q/V LoRA** trainable |
| Tokens | 3D position embeddings, then **16 register tokens per camera** (following **DrivoR**) |
| Heads | ego-kinematics + command → one ego token added to M trajectory queries → decoder cross-attends scene tokens; a **separate scoring decoder** predicts the **six PDM components** (BCE), candidates **detached** before scoring |
| Losses | **winner-takes-all** L1 to the teacher rollout (only the closest proposal gets gradient — lets queries specialise into modes) + proposal-scoring loss |
| **Parameters** | **338.46 M full · 18.58 M trainable (5.49 %)** |
| Training | 100 K navtrain + 237 K SimScale · **16 × H20 · 25 epochs · batch 256 · 38 h (608 GPU-hours)** · FP32 |
| Inference | picks the highest predicted aggregate score; **teacher removed** |

**Goal augmentation** is the crux: the teacher is goal-conditioned, so it can be queried with *alternative* route intents at the **same** scene state, and the same augmented route is used to build the navigation command, the target trajectory **and** the proposal scores — keeping all three mutually consistent.

---

## 2 · Scale: data, hours, compute

| stage | data | compute | DERIVED |
|---|---|---|---|
| **DriveRL** | 922,703 nuPlan trainval scenes @10 Hz (41 hist + 200 fut), subsampled to 5 Hz — **seed only** | 12 nodes / **96 GPUs**, 2,048 worlds/rank = **196,608 parallel worlds**, 110-step (22 s) rollouts, 2,400 updates, **~21 h** | **51.9 B transitions · 2,883,584 sim-hours ≈ 329 sim-years · 2,016 GPU-hours · ~1,430 sim-h per GPU-h** |
| **DriveVFM** | LAION-2B + IN-21K + SA-1B + OpenDV + nuPlan + Waymo | 800 K iters @ batch 2,048 | **1.638 B image samples**; ⛔ **GPU count and hours NOT STATED — a named gap** |
| **DriveZero** | 100 K navtrain + 237 K SimScale = **337 K scenes** | **16 × H20 × 38 h** | **608 GPU-hours** |

⭐ **The shape of the bill:** the expensive half is **perception pretraining** (1.64 B samples, cost undisclosed) and the *cheap* half is the thing the title is about — the teacher costs **2,016 GPU-hours** and the student **608**. A lab with a backbone already in hand faces a far smaller bill than the headline suggests.

---

## 3 · Results, with the stamps applied

### 3.1 nuPlan closed-loop (DriveRL, privileged inputs, CLS)

| | Val14 NR / R | Test14-hard NR / R | Test14-random NR / R | mean |
|---|---|---|---|---|
| Log-Replay ("human") | 93.53 / 80.32 | 85.96 / 68.80 | 94.03 / 75.86 | — |
| CaRL (no human data) | 93.87 / 93.12 | – / 85 | – | — |
| GigaFlow (no human data) | – / 93.80 | – | – | — |
| **DriveRL** | **95.16 / 94.25** | **89.97 / 89.18** | **94.50 / 95.00** | **93.01** |
| **DriveRL-TTS** | **95.54 / 94.53** | **91.13 / 89.15** | **95.93 / 95.12** | **93.57** |

### 3.2 NAVSIM (student, camera-only)

| benchmark | stamp | DriveZero | DriveZero-Scale | prior best |
|---|---|---|---|---|
| **navtest PDMS** | NAVSIMv1, camera | **94.8** | **95.3** | DrivoR-Scale 94.6; **Human Driver 94.8** |
| **navhard EPDMS** | NAVSIMv2, **two-stage**, camera | 51.5 | **57.1** | DrivoR-Scale 54.6, SimScale 53.2, GigaPixel 50.1, ZTRS 48.1 |
| **HUGSIM HD-Score** | **true closed loop, zero-shot** | 39.4 | **46.6** | GigaPixel 38.5, DrivoR-Scale 38.1 |

⭐ **Stamp discipline applied (FS18 row 32):** `57.1` is **navhard two-stage, camera, no human trajectories** — it is the figure already in our stamp table and it stands. The `95.3` is **navtest** and belongs in a different table. Both stamps are present in the paper, unlike Drive-HWM's.

⚠️ **navtest is saturated.** Human Driver is 94.8 and PDM-Closed 89.1; the whole camera-only field from `iPad` to `DrivoR-Scale` sits inside **91.7 – 94.6**. A 0.5-point move there is not the same kind of evidence as a 6.6-point move on navhard.

### 3.3 The ablations that actually inform us (Table 7, ViT-S, navtrain only)

| study | rows | reading |
|---|---|---|
| **Backbone** | DA3-S 93.31 · DA2 ViT-S 93.72 · **DINOv2 ViT-S 93.88 · DINOv3 ViT-S 93.88** · DriveVFM ViT-S **94.41** | ⭐⭐ **DINOv2 and DINOv3 are IDENTICAL here (93.88 both)** — a matched, controlled comparison in which DINOv3 buys **nothing** over DINOv2 for driving. DriveVFM beats both by **+0.53** |
| **VFM teachers** | DINOv3+SigLIP2 93.69 → +SAM **94.10** → +DA2 **94.41** | monotone; **SAM (+0.41) > depth (+0.31)**, i.e. boundaries beat geometry here |
| **Supervision** | human 93.92 · **teacher 93.61** · teacher+goal-aug **94.41** | ⭐⭐⭐ **F1 — the teacher alone LOSES to imitation** |

**Backbone scaling (Table A13, navtrain only):** DriveVFM ViT-S **94.41** → ViT-B 94.54 → ViT-L **94.83**; **DINOv3 ViT-L 94.55**.
⚠️ The advertised margin over DINOv3 is **+0.53 at ViT-S** but only **+0.28 at ViT-L** — **the agglomerative-distillation advantage SHRINKS with scale**, which the paper does not say. And **DriveVFM ViT-S (94.41) nearly matches DINOv3 ViT-L (94.55)** at a fraction of the size, which is the more useful claim for an edge programme.

---

## 4 · Successors, rivals and lineage

| work | relation | number |
|---|---|---|
| **CaRL** `[29]`, **GigaFlow** `[14]` | ancestors — RL from scratch, no human data; DriveRL borrows CaRL's Beta parameterisation and GigaFlow's self-play | CaRL 93.87/93.12 Val14 |
| **ZTRS** `2510.24108` | rival: zero-imitation, RL on **real images + rule-based rewards, no simulated data** | **48.1** navhard |
| **GigaPixel** `2606.19641` | closest rival: pixel student, **on-policy** distillation from a privileged RL teacher | **50.1** navhard |
| **DrivoR** `2609.xxxx` `[34]` | supplies the **register-token** compression DriveZero adopts; the strongest human-supervised camera baseline | 54.6 navhard, 94.6 navtest |
| **SimScale** `[76]` | the OOD simulation corpus that supplies the `-Scale` gain | 53.2 navhard alone |
| **RADIO** `[60]`, **PHI-S** `[59]` | the agglomerative-distillation recipe DriveVFM instantiates | — |

⛔ **No successor to DriveZero was located** (published 2026-09; the search surfaced only its own components and its ancestors). **Named as an empty at one probe**, not as an absence.
⭐ **The axis the family is converging on:** *where does the counterfactual supervision come from?* ZTRS = rule-based rewards on real images; GigaPixel = on-policy distillation in the loop; DriveZero = **offline teacher rollouts under augmented goals**. DriveZero's is the **cheapest of the three at student-training time** — the student never enters a simulator and no rendering is required.

---

## 5 · Weaknesses — what a careful reader should not let pass

| # | weakness | why it matters |
|---|---|---|
| **W1** | ⛔ **No limitations section. The word "limitation" appears ZERO times in 32 pages.** | The charter's Band-D discipline applies to papers too: a document that concedes nothing has had its concessions found by the reader instead |
| **W2** | ⛔ **No confidence intervals, no seed variance, anywhere.** Every table is single runs | The decisive ablation deltas are **+0.80, +0.53, +0.41, +0.31** on a saturated 100-point scale. Our own `H-ESTIM-SEED-1` and [same-seed rerun is a fresh draw] say a single-run delta of this size is not a result |
| **W3** | ⭐ **The title is not what the ablation says** (F1) | "Beyond human demonstrations" is true of the *system*; the *teacher* alone is worse than imitation |
| **W4** | **The teacher is fed ground-truth perception** | The student inherits behaviour from a policy that never experienced a perception error. Domain randomisation is mentioned only for the **real-world** deployment, not for the benchmark teacher |
| **W5** | **EC 18.2 / 10.9 — the teacher is the most uncomfortable policy in its own table** (F5) | A multiplicative reward with a comfort term did **not** produce comfort; the student's comfort comes from WTA smoothing, which is luck, not design |
| **W6** | **Scaling hurt the Extreme tier and the prose omits it** (F6) | The one true-closed-loop, hardest-tier signal regressed |
| **W7** | ⛔ **No latency, no FLOPs, no on-vehicle compute for the 338 M student** | ⭐ This is **our** strongest comparative position: we publish **60.3 ms p50 / 63.1 ms p95 on Jetson Thor against a 100 ms budget**, measured end-to-end on real weights. DriveZero publishes nothing comparable |
| **W8** | **DriveVFM's own training cost is not stated** | 1.64 B samples is the expensive half of the system and its bill is invisible |
| **W9** | **Reproduction needs 96 GPUs + a batched GPU simulator + nuPlan** | Outside almost everyone's reach, ours included |
| **W10** | ⛔ **No license on the released code or checkpoints** | Default all-rights-reserved. **A PI/legal decision, not a Lab one** |
| **W11** | **The TTS gain is small and mostly at N=64** (F3) | A method presented as a headline capability moves the mean by 0.6 % relative |

---

## 6 · Comparison to TanitAD

| axis | DriveZero | TanitAD | reading |
|---|---|---|---|
| **Deployed params** | **338.46 M** (18.58 M trainable) | **263 M** | ⭐ we are **smaller**, and "sub-300M" survives against DriveZero — though **not** against DrivoR (~40 M) |
| **On-silicon latency** | ⛔ **not reported** | ⭐ **60.3 ms p50 / 63.1 ms p95 on Jetson Thor vs a 100 ms budget**, real weights, end-to-end | **our clearest advantage, and it is measured** |
| **Backbone** | DriveVFM ViT-L (4 VFMs agglomerated), frozen + LoRA | DINOv3-class frozen trunk, ~87.3 M encoder | their Table 7 says **DINOv3 = DINOv2**, and **+0.53 → +0.28 as scale grows** ⇒ the gain is real but modest and shrinking |
| **Supervision** | teacher rollouts under **augmented goals**; no human trajectories | **human trajectories** | ⭐⭐⭐ **the single biggest gap, and the cheapest to close** (F2) |
| **Action space** | **jerk + steering RATE**, bicycle model | `(steer, accel)` | rate-level control is a smoothness prior by construction — relevant to our comfort/jerk family |
| **Nav intent** | **two speed-scaled goal anchors**, identical at train and deploy, and **augmentable** | nav command, **measured nav-echo leak 369/369** | ⭐⭐ their augmentation is precisely an echo-breaker; our leak is the same disease |
| **Corpus** | nuPlan 922 K scenes + 337 K student scenes; **2.88 M SIM hours** | PhysicalAI-AV, 2,376 eps / 4,719 clips, strict parity | they are ~2–3 orders of magnitude larger in *experience*, and it is **synthetic** |
| **Closed loop** | HUGSIM zero-shot **46.6 HD** | AlpaSim/NuRec on Thor, **1.488 m [1.329, 1.647]**; ⛔ **no collision/offroad score** (renderer wire contract unfinished) | they have a **scored** closed loop; we have a **rendered** one without a scorer. That is our gap, and it is bounded work |
| **Benchmarks** | navtest / navhard / HUGSIM | T1 open loop + our four families | we are **not on any community leaderboard** (G3) |
| **Estimator discipline** | none (W2) | clip-cluster bootstrap, pre-registration, controls that must read known values | ⭐ **ours is stronger, and that is a real asset** — their +0.80 would not clear our bar |
| **Hierarchy** | ⛔ **flat**: one policy, one student, no tier structure | strategic / tactical / operative | the differentiation is intact; DriveZero does **not** contest H1 |

### 6.1 Where they answer our open questions

| our row | what DriveZero supplies |
|---|---|
| **I-1 quality half** (untouched after 6 passes) | ⭐⭐⭐ **answered externally: +0.56 CLS at N=64, noise below N=32** (F3). ⇒ our expected payoff should be **revised down**, and the N/L split confirms our cost finding |
| **P-1 / hold-action echo** (no v7 arm beats its control) | ⭐⭐⭐ **goal augmentation** — counterfactual supervision on the same scene (F2) |
| **nav-echo leak 369/369** | same mechanism; augmenting the route intent breaks the echo *by construction* |
| **Longitudinal blindness** | jerk-level longitudinal action + an explicit progress term; their **EP 91.5 vs human 87.5** |
| **Closed-loop gap (0.45 → 1.69)** | the teacher is trained in closed loop; the *student* still trains open-loop — so this is a **supervision** fix, not a closed-loop-training fix |
| **H1b (hierarchy)** | ⛔ **nothing** — DriveZero is flat. Our thesis is untouched by it |

### 6.2 Where we are ahead

1. **Measured edge latency on the target silicon.** They report none.
2. **Estimator and pre-registration discipline.** Their headline deltas are single-run.
3. **Hierarchy.** Untested by us, uncontested by them.
4. **A renderer on the edge device** (gsplat/NuRec at 492 FPS on Thor) — they use HUGSIM, a third-party benchmark.

---

## 7 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | DriveZero holds the navhard number in our own stamp table (57.1) and is the current top of the zero-human-demonstration family. Its central mechanism addresses **P-1**, the programme's actual problem. |
| **CONSEQUENCE** | Three, in cost order. **(a)** Goal augmentation is adoptable **without** an RL teacher, on our corpus, at tiny cost — it is the anti-echo arm v7 has never run. **(b)** I-1's expected payoff falls: the published quality gain from value-guided search is **+0.6 % relative**. **(c)** Our efficiency wedge should be restated against **338 M with no published latency**, where our 60.3 ms on Thor is the differentiator. |
| **COMBINATION** | Goal augmentation composes with our **three-planner hierarchy**: a goal-conditioned *tactical* planner can be queried under alternative tactical intents exactly as DriveRL is under alternative routes — so the mechanism is *more* natural in our architecture than in their flat one. It also composes with **LR15-2** (value head on place/progress): their critic is **decomposed per reward term**, which is the shape a progress-value head would take. |
| **CHANCES / RISKS** | **Chance:** the cheapest high-value transfer this programme has been offered in weeks, and it needs no new compute. **Risks:** (a) every number above is single-run and would not clear our own bar (W2); (b) their goal augmentation is validated only *with* an RL teacher, so its value **without** one is a HYPOTHESIS — and the falsifier is cheap; (c) nuPlan/NAVSIM is a different corpus from PhysicalAI-AV, so nothing transfers numerically; (d) ⛔ the released code is **unlicensed**. |
| **EXPERIMENT** | ⭐⭐⭐ **`E-AI-GOALAUG-1` (proposed AI20-1)** — the anti-echo arm, on the v7-tiny ladder, **no RL teacher**: augment the route/goal intent per window and generate counterfactual targets with our existing kinematic model, matched steps, with a **no-augmentation replicate** and the **hold-action control `ha0_ext`** in every arm. **Committed:** the augmented arm beats `ha0_ext` beyond the replicate floor on T1 h=6 s ADE **and** reduces the nav-echo recovery R² (0.9702) below 0.8 ⇒ goal augmentation is the anti-echo lever and earns a full arm; fails ⇒ the mechanism needs the RL teacher, and the cheap route is closed with a number. ⭐ **This is the first v7 arm with a pre-registered mechanism for the P-1 defect rather than a hypothesis about it.** |

## 8 · Stopping condition (Rule Zero)

**(3a)** the PI's question is answered across all seven requested axes. **New debts:** ⛔ the **license** status of `XiaomiAutoL3/DriveZero` (PI/legal); ⛔ **no successor located at one probe** — re-probe in a later pass, do not harden.

`Deliverables: RESULT.md · REVERSE_ENGINEERING_PLAN.md · code/dz_budget.py · raw/{dz_budget.txt, search_log.md}`
