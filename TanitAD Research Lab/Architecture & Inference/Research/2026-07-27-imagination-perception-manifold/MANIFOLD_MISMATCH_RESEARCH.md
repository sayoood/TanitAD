# THE METRIC DECODER IS ATTACHED TO IMAGINATION, NOT PERCEPTION — what the field calls it, how every other world-model line avoids it, and what it would cost us to fix

**Date:** 2026-07-27 (Europe/Berlin; pods log UTC). **Stream:** Architecture & Inference — research.
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written before any fix-evaluation literature was
read (its disclosure section states exactly what preceded it). It is **not edited**; deviations are in §10.
**Host:** dev box, **CPU / web only. No pod was touched** — pod1 (training), pod2 (owed controls), pod3
(classifier build) and the eval pod (trafficsim) were never contacted. No GPU was used. No experiment was
launched.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited — specific paper) ·
`INHERITED` (another agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` · `CONFIRMED` · `DECISION-GRADE` (CONFIRMED + pre-registered + falsifier stated
+ cost estimated).

---

# 0. VERDICT

> ## ⛔ **PRE-REGISTERED FALSIFIER F4 FIRED, ON OUR OWN TRAINING LOGS, AT ZERO COMPUTE. A decoder trained on REAL latent pairs already exists in every one of our checkpoints — `grounding.invdyn["op"]`, trained at TWICE the weight of the imagined-pair readout, with full gradient into the encoder — and after 30 000 steps it sits at 1.0129 m on TRAINING data while the imagined-pair readout on the identical batch sits at 0.0304 m. The real-pair path is not under-trained, not under-weighted and not un-attempted: it is 33× worse, on data it has already seen, and the gap is MANUFACTURED BY TRAINING (2.36× at step 0–1k → 33.3× at 28–30k).**
>
> ## ⇒ **"Re-fit the decoder on real pairs" is not an untried fix. It is what we already do, and it does not work. Any candidate whose mechanism is "train the head on the other latent source" is REFUTED on our own weights before it is proposed.**

> ## ⭐⭐ **AND THE ATTRIBUTING ABLATION ALSO ALREADY EXISTS, UNREAD, IN A COMMITTED LOG. The no-speed control (`flagship4b-phase0-30k`, `speed_input=False`) has a real-pair decode error INDISTINGUISHABLE from v1's (1.1333 vs 1.1004 m, +3.0 %) and an imagined-pair decode error 6.87× WORSE (0.3543 vs 0.0516). Removing the injected `v0` action channel destroys the imagined decode and leaves the perceived decode untouched.**
>
> ## ⇒ **The 9.4× gap is not primarily a manifold-geometry problem. It is a SHORTCUT: the metric readout decodes the speed we INJECT through the action channel, routed through the predictor. A real latent pair carries no injected speed, so there is nothing to read.** `HYPOTHESIS`, but with three independent MEASURED supports and one clean pre-existing ablation — and it is the reading that explains why the model holds `v0` to within 4 % for 18.5 s and never revises it, why feeding its own predicted speed back is catastrophic (×9.8), and why observing hurts.

**The literature answer, in three lines:**

1. **The general pathology IS named** — prior/posterior mismatch, exposure bias, the train-test gap of
   autoregressive rollout — but the field's version runs in the **OPPOSITE DIRECTION** to ours (trained on
   real, deployed on imagined). **Our direction — a head trained on imagined latents that cannot read real
   ones — is essentially unreported.** `PUBLISHED` + `CONFIRMED` by the shape of the literature (§3).
2. **Every major world-model line structurally forbids our defect**, by one of four mechanisms: an explicit
   prior→posterior pull (PlaNet latent overshooting, Dreamer KL balancing, **TD-MPC2's consistency term
   `‖z′_t − sg(h(s′_t))‖²`**), a shared discrete codebook (IRIS), decoding nothing from imagination at all
   (DINO-WM, TD-MPC2 is decoder-free), or defining the objective itself as a distance between a *predicted*
   and an *encoded* feature (the JEPA line). **We have the pull (our `_rollout_loss` is TD-MPC-shaped) but
   we apply it to the LATENT ONLY — never to the metric head.** That is the precise hole (§4).
3. **The gap as measured is NOT closable on a frozen checkpoint**, and my pre-registration commits me to
   saying so plainly. **F4 fired and F2 is corroborated** — two of four. **But the actionable conclusion is
   not "spend a GPU-week": it is that the fix belongs in the NEXT run's loss (one line, ~0 GPU cost), and
   that the two hours of frozen-checkpoint work still worth doing are DIAGNOSTIC — they decide which of
   two mechanisms to fix, and one of them (§6, X1) costs 15 minutes of CPU.** (§5, §6.)

**Two further results from the parallel surveys, each of which changes something:**

4. 🔴 **ONE paper builds it our way — and never measures the cost.** Valdi ([arXiv:2607.00917](https://arxiv.org/abs/2607.00917))
   trains *"each decoder … on an on-policy dataset collected by rolling out exactly the model whose latents
   it reconstructs"*, with the rationale asserted verbatim and **no measurement**. ⇒ **our 9.4 × is the
   number that paper is missing, and a metric SE(2) head trained exclusively on imagination pairs has NO
   published precedent in driving** — every verifiable driving model with a metric head (OccWorld, Drive-WM,
   LAW, Epona) reads the **observed** latent (§8).
5. ⭐⭐ **The horizon fix has a better form than the one we were about to adopt, and it also costs zero.**
   GraphCast, FuXi and Pangu-Weather independently converged not on *"retrain at the evaluation horizon"*
   but on **a bank of horizon-specialised readouts selected by lead time** — and **we already own the bank**
   (`op`/`tac`/`str`, all three trained, in every 4-brain checkpoint). Our own crossover is already measured
   (`str` is −0.0449 m worse at 0.5 s, better at every horizon ≥ 1 s), so the selection rule can be placed
   from data we have. **New candidate C8, §5 — zero GPU, and it strictly dominates the flat swap.** (§7)

---

# 1. PRE-REGISTRATION ADJUDICATED

Written before research; reproduced here with each falsifier's verdict. Full text: `PRE_REGISTRATION.md`.

| # | falsifier as pre-registered | verdict | on what |
|---|---|---|---|
| **F1** | *no frozen-weight repair exists in the literature* | ⚠️ **PARTLY FIRED** | Every published repair of a prior/posterior or imagined/observed decode gap is a **training-time** term. I found **no** paper that repairs it by re-fitting a small head on a frozen trunk. But I also found no paper that tried and failed — the frozen-refit route is **unattempted in the literature**, not refuted by it. `UNVERIFIED` as a negative; treated as "no precedent", not "impossible". |
| **F2** | *the gap is located in the TRUNK, not the head* | ⚠️ **CORROBORATED, not proven** | Our own §2 measurement shows a full-rate, double-weighted real-pair head plateauing at ~1.0 m on train data — consistent with the information not being there. **But the no-speed ablation (§2.3) offers a competing and better-fitting explanation** (the head is fine; the *imagined* side has a shortcut). F2 and the shortcut reading are **not yet separated**, and X1/X2 (§6) separate them. |
| **F3** | *the fix costs rollout quality* | ⛔ **DID NOT FIRE** | No source reports that tying a decoder to the observation manifold degrades imagined-rollout accuracy. PlaNet reports latent overshooting **improves** long-horizon consistency (`PUBLISHED`, Hafner et al. 2019). Nothing found on either side for a *metric* head. `UNVERIFIED` in the specific direction that matters to us. |
| **F4** | *our own invdyn heads, trained on real pairs, also fail on real pairs* | 🔴 **FIRED, DECISIVELY, AT ZERO COMPUTE** | §2.2. `g_op_mid_de_m` = **1.0129 m** at 28–30k against `g_op_fwd_ade_m` = **0.0304 m** on the **same training batch, same forward pass, same encoder states**. |

**Two of four fired or corroborated ⇒ per my own rule I must write it plainly:**

> ## ⛔ **THE GAP IS NOT FIXABLE ON A FROZEN CHECKPOINT BY RE-FITTING THE METRIC HEAD. We have already run that experiment for 30 000 steps at double weight and it plateaus 33× short. A fix that changes only the head is refuted before it is tried.**

**And the honest counterweight, which my rule also requires me to state:** *"not fixable on a frozen
checkpoint"* is **not** the same as *"needs a GPU-week"*. The repair the literature converges on is a
**loss term**, and in our code it is **~10 lines and ≈0 % step-time** because the tensors it needs are
already computed (§5, C1). The expensive thing here was never the fix — it was believing the fix had to be
discovered. **The discriminating diagnostics cost 15 CPU-minutes and ~20 GPU-minutes (§6).**

---

# 2. ⭐ THE PART OF THIS THAT IS NEW: our own logs already contained the answer

Nothing in this section required a GPU, a pod, or a new measurement. All of it is read out of
committed artifacts.

## 2.1 What the two grounding terms actually are (read out of the source, not from prose)

`stack/tanitad/train/flagship_losses.py` → `stack/tanitad/models/metric_dynamics.py::grounding_losses`
computes, **per brain level, in the same forward pass**:

* **term (a), `*_mid`** — `MetricInverseDynamics[lvl](z_t, z_{t+k})` on **REAL encoded pairs** at that
  level's horizons → relative ego-pose. Logged as **`g_{lvl}_mid_de_m`** = the metric displacement error,
  in metres, of a decoder **trained on real pairs**. For `op` the horizons are `[1, 2, 4]`
  (0.1 / 0.2 / 0.4 s). Weight `invdyn_weight = weights.invdyn = **2.0**`.
* **term (b), `*_fwd`** — `StepDisplacementReadout[lvl]` decoded on the predictor's **imagination
  rollout**, accumulated SE(2). Logged as **`g_{lvl}_fwd_ade_m`**. Weight `fwd_weight = weights.fwd =
  **1.0**`.

Both are logged from `flagship_loss` on the **training batch**
(`stack/scripts/train_flagship4b.py:513, 530-536`) — **same batch, same encoder states, same step.**
⇒ **`g_op_mid_de_m` vs `g_op_fwd_ade_m` is a perfectly paired real-pair-vs-imagined-pair contrast that
the program has been writing to disk 620 times per run and never read.** `MEASURED`, source-verified.

## 2.2 🔴 The contrast, over training

`MEASURED` · `taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl` (620 rows, ckpt step 29999) and
`…/v1-speedjerk_config.json` · v1 = `flagship4b-speedjerk-30k` · band = mean over all logged rows in the
step range.

| step band | `g_op_mid_de_m` — **REAL pair**, head trained on real pairs @ **weight 2.0** | `g_op_fwd_ade_m` — **IMAGINED** rollout @ weight 1.0 | ratio | n rows |
|---|---:|---:|---:|---:|
| 0–1k | 2.2486 | 0.9541 | **2.36×** | 20 |
| 4–6k | 1.4833 | 0.2442 | **6.07×** | 40 |
| 9–11k | 1.2821 | 0.1034 | **12.4×** | 59 |
| 19–21k | 1.1004 | 0.0516 | **21.3×** | 40 |
| **28–30k** | **1.0129** | **0.0304** | **33.3×** | 41 |

**Over training the imagined-pair decode improves 31.4× (0.9541 → 0.0304). The real-pair decode improves
2.22× (2.2486 → 1.0129) and then flattens.** The same pattern holds one level up: `str` (a single real
pair at k = 20 vs a 20-step imagined rollout) goes 16.0284 / 4.5320 (**3.54×**) → 1.8735 / 0.1606
(**11.7×**).

⇒ **The imagination/perception split is not a property the model was born with. Training creates it, and
more training widens it.**

⚠️ **Aggregation caveat, stated rather than buried.** `*_mid_de_m` is the **endpoint** displacement error
averaged over horizons {1,2,4}; `*_fwd_ade_m` is the **ADE over waypoints 1…4**. For a monotonically
growing error the endpoint statistic is the larger of the two, typically by ~2×. **Even applying a
generous 2× correction the 28–30k ratio is ~17×, and the *trend* — the ratio growing 14× over training —
is unaffected by any fixed aggregation factor.** I do not quote 33× as a clean effect size; I quote it as
a bound of the right order with the correction stated.

**Is 1.01 m "at chance"? No — and that matters.** From the committed dump
(`…/2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt`, `speed`, n = 599) the ego-speed
spread is **σ = 9.80 m/s, E|v − v̄| = 8.09 m/s** (mean 12.90, IQR 5.20–20.35). A "know nothing but the
mean speed" straight-line predictor would score **1.89–2.29 m** on the same {1,2,4} average. `ESTIMATED`
(straight-line approximation; val episode-initial subsample, not the train distribution).
⇒ **The real-pair decoder is ~2× better than knowing nothing, and ~33× worse than the imagined-pair
decoder. It extracts *some* motion — it is not blind — but it is nowhere near usable, and 30k steps at
double weight is where it stopped.**

**For scale: `invdyn["str"]` decoding a real 2 s pair scores 1.8735 m on TRAIN, against a constant-velocity
floor of 1.268 m at 2 s on val** (`…/2026-07-26-blind-imagination/artifacts/horizon_curve.json`).
⚠️ Different sets and CV is handed the true `v0`, so this is a **kind**-comparison, not a paired one — but
its kind is: **our real-pair perceptual odometry at 2 s does not beat assuming nothing changes.**

## 2.3 ⭐⭐ The attributing ablation, already in the repo, never read this way

`nospeed-phase0` is `flagship4b-phase0-30k`, the **no-speed ablation control** (`CLAUDE.md`'s own warning
about this arm's identity). Config diff against v1, computed field-by-field: **the ONLY differences are
`predictor/tactical_pred.action_dim` 2 → 3 (the `v0` channel), `aux_accel` False → True (0.53 M params),
and `jerk_weight` 0.0 → 0.02.** Encoder, predictor, seed, batch, lr, all grounding weights
(`invdyn 2.0 / fwd 1.0 / roll 0.5 / pred 1.0`), all horizons (`op [[1,2,4],4] / tac [[8,16],16] /
str [[20],20]`) and `rollout_k = 4` are **identical**. `MEASURED`, from both `*_config.json`.

Matched step band **19–21k** (both arms alive; nospeed ran to 22 950):

| arm | `speed_input` | `g_op_mid_de_m` — REAL pair | `g_op_fwd_ade_m` — IMAGINED | ratio |
|---|---|---:|---:|---:|
| v1 `flagship4b-speedjerk-30k` | **True** (`action_dim=3`) | 1.1004 | **0.0516** | **21.3×** |
| `flagship4b-phase0-30k` | **False** (`action_dim=2`) | 1.1333 | **0.3543** | **3.2×** |
| **Δ (nospeed vs v1)** | | **+3.0 %** | **+587 % (×6.87)** | |

> ### **Removing the injected speed channel leaves the REAL-pair decode statistically where it was (+3 %) and degrades the IMAGINED-pair decode by 6.87×.**

The encoder never sees `v0` — it is an **action** input to the predictor — so the real-pair path is
*expected* to be unmoved, and it is. What is not forced by construction is that the imagined path collapses
without it. ⇒ **the imagined-pair readout's accuracy is carried overwhelmingly by the speed we inject, not
by what the model perceives.** `MEASURED`, `CONFIRMED` on two arms at parity.

⚠️ **What this ablation can and cannot support.** It is a **three-variable bundle** (speed channel +
aux_accel + jerk), not a surgical single-variable ablation of the readout's input, and all three variables
are longitudinal. It therefore supports *"the model's metric competence is carried by injected longitudinal
information"* — it does **not** prove the readout literally reads `v0`. **X2 (§6) is the surgical version
and costs ~20 GPU-minutes.**

## 2.4 The reading this forces, and how it differs from the one in the source report

`BLIND_IMAGINATION.md` §3.5 states the finding as *"the decoder is attached to the imagination manifold,
not to perception"* — which is **correct as a description** and is the framing this brief was written on.
§2.2 and §2.3 sharpen the **mechanism** behind it, and the sharpening changes the fix:

| reading | what it predicts | fix it implies |
|---|---|---|
| **(M1) Manifold geometry.** `ẑ` is a contracted conditional mean; real `z` is off-manifold for the head's second slot. | Any head fed real pairs fails; a head fed real pairs *and trained on them* succeeds. | mixed-source training of the head, scheduled sampling in latent space |
| **⭐ (M2) Injected-signal shortcut.** The head reads the action-borne `v0` through the predictor; a real pair carries none. | A head trained on real pairs **also fails** (F4) ✅ **observed**; and removing `v0` **collapses the imagined side only** ✅ **observed**. | **give the ENCODER metric ego-motion, or stop calling the readout perception** |

**M2 predicts both of our surprises; M1 predicts only the first.** M1 is not eliminated — the two are not
mutually exclusive and both may contribute — but **M2 is the one consistent with every measurement we
have**, and it is the one that changes the ranking in §5.

**M2 also retro-explains three previously separate program findings** (`INHERITED`, each from its own
source, none re-verified here): blind imagination is longitudinally a **hold-`v0` predictor** holding speed
to within 4 % for 18.5 s and never revising (`BLIND_IMAGINATION.md` §3.3); **feeding its own predicted
speed back is catastrophic**, `ade_0_2s` 0.9554 → 9.3577 (§0.2 ibid.); and the historical **speed-input
fix** raised REF-A's speed-R² 0.61 → 0.965 by *injecting* speed rather than by making the encoder perceive
it. ⇒ **the program solved speed-blindness by handing the model the answer, and the metric decoder learned
to read the answer.**

---

# 3. IS IT A KNOWN PATHOLOGY, AND WHAT IS IT CALLED?

## 3.1 The names the field uses

| name | what it denotes | direction | source |
|---|---|---|---|
| **prior/posterior mismatch** | the latent dynamics (prior) and the observation-conditioned inference (posterior) drift apart, so imagination leaves the region the model was fit on | trained on posterior → used on prior | PlaNet (Hafner et al. 2019); DreamerV2 (Hafner et al. 2021) |
| **exposure bias** | trained with teacher forcing on ground-truth prefixes, deployed on its own outputs | trained on real → used on own | Scheduled Sampling (Bengio et al. 2015); Professor Forcing (Lamb et al. 2016) |
| **train-test gap of autoregressive rollout** | the modern video-generation restatement of the same thing | trained on real → used on own | **Self Forcing** (Huang et al., NeurIPS 2025) |
| **compounding / accumulating model error** | one-step error amplified along a rollout | trained on real → used on own | ubiquitous in MBRL |
| **covariate shift** (the control-theory framing) | the deployed state distribution differs from the training one | trained on expert → used on own | DAgger (Ross et al. 2011) |

## 3.2 ⛔ Our defect runs the OTHER WAY, and that direction is essentially unreported

**Every one of those names describes: *train on real, deploy on imagined*.** Ours is
**train on imagined, then try to read real** — the head is *over-adapted to its own model's output
distribution* and has never been asked to read the world.

I searched for this direction under: inverted/reverse exposure bias, decoder over-fit to model outputs,
model-generated-data over-adaptation, and "readout trained on rollouts fails on observations". **I found no
paper that names or measures it.** `UNVERIFIED` as an exhaustive negative — absence found at one search
strategy is not absence (`CLAUDE.md` rule 2) — but the *structural* reason it is unreported is
identifiable and is in §4: **almost nobody builds a head that is trained ONLY on imagined latents**, because
the standard architectures either forbid it or have no such head at all.

⇒ **The most defensible statement:** the *ingredient* pathology (prior/posterior divergence) is named,
measured and repaired in the literature; **the specific configuration that produces our 9.4× — a metric
readout supervised exclusively on rollout transitions, then fed observations — is a configuration the field
does not use, so it has not had to name its failure.** `PROVISIONAL`. **We are not looking at a
misunderstood standard defect; we are looking at an unusual construction.**

## 3.3 ⚠️ And it is not even a defect at our published operating point

Stated plainly because it changes the priority: **our headline `ade_0_2s` is produced IN-distribution.**
`rollout_decode` encodes the initial window and then never re-encodes, so the deployed decode is
imagined-pair decode — exactly what the head was trained on. **The mismatch costs us nothing on the
leaderboard number.** It costs us, precisely:

1. **re-anchoring / peeking** — a perfect error oracle loses to a fixed clock by 24.8–112.2 % (`INHERITED`,
   `BLIND_IMAGINATION.md` §4.2);
2. **the observation ceiling** — full observation is *worse* than blind imagination out to ~6 s (0.5167 vs
   0.3839), so the arm that should upper-bound us does not (`INHERITED`, ibid. §2.6e);
3. **any use of the world model as an odometry or perception module** — §2.2 says it is not one;
4. **the camera-duty-cycle efficiency claim**, which requires (1) to work.

**If the program's roadmap does not need those four, this is not worth a GPU-hour.** It does need them —
the closed-loop and sensor-gating streams both depend on (1) — so it is worth the ~0-cost loss term in the
next run and the 15 CPU-minutes of diagnosis. It is **not** worth a dedicated run.

---

# 4. ⭐ THE CRUX: who decodes from the PRIOR, who from the POSTERIOR, and what forces them to be interchangeable

This is the table the brief asked for. **"Decoder" is read broadly: any head that consumes a latent and
emits a task quantity** (pixels, reward, value, pose) — because that is what our metric readout is.

| system | head/decoder **trained on** | head/decoder **applied to** at rollout | mechanism forcing the two to be interchangeable | DEMONSTRATED vs ASSERTED |
|---|---|---|---|---|
| **PlaNet** (Hafner et al. 2019, arXiv:1811.04551) | image decoder on **POSTERIOR** states | reward model on **PRIOR** states during CEM planning | ⭐ **latent overshooting** — *"compute KL divergence between multi-step predicted latent distributions and the corresponding filtered posteriors, without generating image observations"*; gradients of the posterior are **stopped** for distances d > 1, so *"multi-step predictions are trained towards the informed posteriors, but not the other way around"* | **DEMONSTRATED** as an ablation improving performance; **ASSERTED** (not measured) that it closes a decode gap |
| **Dreamer / DreamerV2 / V3** (arXiv:2010.02193; 2301.04104) | image + reward decoders on **POSTERIOR** states — *"From the posterior model state, we reconstruct the current image x̂ₜ and predict the reward"* | actor & critic on **PRIOR** (imagined) states — *"actor and critic are trained from the same imagined trajectories"* | ⭐⭐ two mechanisms. (i) **KL balancing**: *"we minimize the KL loss faster with respect to the prior than the representations by using different learning rates, α=0.8 for the prior"*, which *"encourages learning an accurate prior over increasing posterior entropy, so that the prior better approximates the aggregate posterior."* (ii) ⭐ **imagination is SEEDED FROM POSTERIORS**: *"The trajectories start from posterior states computed during model training"* — so every imagined trajectory begins on the observation manifold | **DEMONSTRATED**: KL balancing is an ablated design choice with reported effect. The prior→posterior seeding is architectural |
| **TD-MPC / TD-MPC2** (arXiv:2310.16828) | ⭐ **reward and value heads are trained on ROLLED (PRIOR) latents** — `z_{t+1} = d(z_t, a_t)`, `z_0 = h(s_0)` — **exactly our configuration** | the same rolled latents | ⭐⭐ **the latent consistency term, Eq. 3**: `Σ λᵗ( ‖z′_t − sg(h(s′_t))‖²₂ + CE(r̂,r) + CE(q̂,q) )` — **the predicted latent is pulled directly ONTO the stop-gradient encoder output.** The prior is not merely *close in KL* to the posterior; it is regressed onto the posterior **point**. Default training horizon **H = 3**. **Decoder-free** — nothing is reconstructed | **DEMONSTRATED**: the consistency term is a named, load-bearing loss component; the decoder-free claim is in the abstract |
| **IRIS** (arXiv:2209.00588, ICLR 2023) | VQ-VAE decoder on **REAL** frames' discrete tokens | the **same** decoder on **imagined** token sequences from the transformer | ⭐ **a shared finite codebook.** An imagined token is *by construction* an element of the same codebook a real token comes from, so the decoder can never be handed an off-manifold input. Our `ẑ` is a free continuous vector and can be | **Architectural**, not measured as a gap |
| **Δ-IRIS** (arXiv:2406.19320) | as IRIS, tokenising **stochastic deltas between timesteps** | same | same codebook argument, applied to the *difference* — notable because our defect lives in exactly the pair-difference | **Architectural** |
| **DIAMOND** (Alonso et al., NeurIPS 2024 spotlight, arXiv:2405.12399) | diffusion model generating **pixels** conditioned on past frames | conditioned on its **own generated** frames | operates in **pixel space**: the conditioning is always a valid image, real or generated, so there is no separate latent manifold to fall off | **Architectural**; the paper's measured claim is that visual detail improves agent performance |
| **DINO-WM** (arXiv:2411.04983) | ⛔ **decoder-free for control** — *"model[s] visual dynamics without reconstructing the visual world"*, predicting future **DINOv2 patch features** | planning is done **entirely in feature space** | there is **no head that could be mis-attached**. The objective is a distance between a *predicted* feature and an *encoded* feature, so the two are commensurate by definition | **DEMONSTRATED** (verified by fetching the paper: no decoder) |
| **V-JEPA 2 / V-JEPA 2-AC** (arXiv:2506.09985) | action-conditioned predictor in frozen-encoder feature space; goals are **encoded real images** | planning cost compares a **predicted** feature to an **encoded** goal feature | same structural argument as DINO-WM | ⚠️ **UNVERIFIED in detail** — I confirmed the paper and that it is a latent action-conditioned world model planned with image goals, but could **not** reach the energy-function text. Do not quote the mechanism as established |

## 4.1 The pattern, and exactly where TanitAD sits in it

**Four ways to be safe, and everyone uses at least one:**

* **(A) pull the prior onto the posterior** — PlaNet (multi-step KL), Dreamer (KL balancing), **TD-MPC2
  (L2-onto-stop-grad, the strongest form)**;
* **(B) make them the same object** — IRIS/Δ-IRIS shared codebook; DIAMOND's pixel space;
* **(C) never decode from imagination** — TD-MPC2 is decoder-free; DINO-WM plans in features;
* **(D) define the objective AS a predicted-vs-encoded distance** — the JEPA line.

**TanitAD has (A) — and it is TD-MPC-shaped.** `stack/tanitad/train/train_worldmodel.py::_rollout_loss`
recursively rolls the predictor to `K = rollout_k = 4` and, at every step, minimises
`(z_hat − fut_states[idx_of[j−1]]).pow(2).mean()` — **the predicted latent regressed onto the encoded
latent**, structurally the same as TD-MPC2's Eq. 3 term (weight `roll = 0.5`), plus the JEPA prediction
loss at horizons [1,2,4] (weight `pred = 1.0`). `MEASURED`, source-verified.

> ## ⇒ **Our latents ARE tied. Our METRIC HEAD is not. The consistency machinery every other line uses to make prior and posterior interchangeable exists in our trainer and is applied to the representation only — while the head that produces every published number in the program is supervised on ONE side of it, exclusively.**

**That is the defect, stated in the field's own vocabulary, and it is a one-term omission — not an
architectural dead end.**

## 4.2 ⚠️ Why the tie we already have did not save us — and the caveat on this explanation

An L2 pull produces the **conditional mean**. `ẑ = E[z | past, action]` is systematically
**lower-variance** than a real `z`: it is the smoothed, appearance-free part. A head fed contracted,
low-noise inputs need not work on the full-variance real ones — and our (c2) error being **pure variance**
(σ 2.083 m vs bias 0.118 m at 0.5 s, `INHERITED` from `BLIND_IMAGINATION.md` §3.5) is exactly that
signature. This is the (M1) reading of §2.4 and it is a real, published mechanism in kind (the well-known
blur/regression-to-the-mean of MSE-trained predictors; DreamerV2's KL balancing exists precisely because
the degenerate way to satisfy a prior/posterior tie is to change the *posterior* rather than improve the
prior — `PUBLISHED`, arXiv:2010.02193).

⚠️ **But M1 alone does not survive our own F4 result.** If the only problem were that real pairs are
off-manifold for the head, then a head **trained on real pairs** would be fine — and ours is not (§2.2).
**M1 is at best a contributing mechanism. M2 (§2.3) is the one that fits.** I flag this because the
"contracted conditional mean" story is the *intuitive* one and it is the one a reader will reach for; our
own logs say it is not sufficient.

---

# 5. RANKED CANDIDATE FIXES

**Ranked by (expected gap closure) / (GPU-hours to a DECIDING result)**, per `PRE_REGISTRATION.md` §4 — a
fix that returns its first bit in an hour outranks a larger fix that needs a week.

⚠️ **Read the EVIDENCE-MATCH column first.** Per the brief and my pre-registration, a mechanism that
addresses "distribution shift" in the abstract is **not** evidence about a 9.4×/33× decode penalty.
`MATCHED` = published evidence about a decode-source gap. `ADJACENT` = published evidence about
compounding error or rollout drift. `NONE` = mechanism-plausible only.

| rank | fix | mechanism | published evidence | **evidence match** | implementation delta against OUR code | cost to a deciding result |
|---|---|---|---|---|---|---|
| **C0** | ⭐⭐ **Stop using one head for two sources — route by source, and STATE the readout is an action-integrator** | Not a repair; a correction of interpretation + inference-time routing. `grounding.invdyn[lvl]` already exists in every checkpoint and is the real-pair-trained head; `grounding.step[lvl]` is the imagined-pair head. Use each on its own source, and re-anchor by **resetting the SE(2) integration**, never by decoding across a real→imagined seam. | — (this is our own §2.1–2.2) | **OURS, MEASURED** | ~15 lines in `taniteval/blindimag.py`'s peek path + a registry note. **No training.** | **0 GPU.** ⚠️ Expected gain is *small* — invdyn is 1.87 m at 2 s, worse than CV — so this **caps the damage, it does not fix it**. Its real value is that it makes the peek arm interpretable. |
| **C1** | ⭐⭐ **Symmetric/mixed-source supervision of the metric head — the TD-MPC2 pattern applied one level up** | Train `grounding.step[lvl]` on a **mixture** of `(ẑ,ẑ)`, `(z,ẑ)` and `(z,z)` pairs, with the real-pair target being the true per-step Δpose. This is the standard prior/posterior tie (§4.1 family A), applied to the **head** instead of only the latent. | **TD-MPC2 Eq. 3** (`‖z′−sg(h(s′))‖²`, heads on rolled latents) `PUBLISHED`; PlaNet latent overshooting `PUBLISHED`; Dreamer KL balancing `PUBLISHED` | **ADJACENT** — all three demonstrate the *latent* tie; none measures a *head* decode gap | ⭐ **≈10 lines, ≈0 % step time.** `grounding_losses` **already has `fut_states`** (real encoded futures, computed for term (a)) and already rolls to `k_max = 20`. Add a second `decode_transitions` call on real pairs built from `fut_states`, or replace `trans[j]` with `(fut_states[j−1], fut_states[j])` with probability *p*. **No extra encode, no extra rollout.** | **Next run only** — it is a training term. ⛔ **Deciding result needs a full arm.** Ranked #1 among *training* fixes; ranked below C0/C2/X1 on time-to-first-bit. |
| **C2** | ⭐ **Set `--op-fwd-k 20` (the horizon fix), already CONFIRMED** | Decode at the horizon the head is read at. Orthogonal to the manifold defect but bundled in the same line of work. | our own; see §7 for the literature | **OURS, MEASURED + CONFIRMED on v4** | one CLI flag; `grounding_losses` already rolls to `k_max = 20`, so this adds **no rollout**, only 16 extra readout applications | **≈0 GPU** (~1–2 % step time). Already banked: `ade_0_2s` 0.3839 → 0.1950 on v1; `wm_canary_ade_2s` 1.1409 → 0.5446/0.5521 on v4 (`INHERITED`, `…/2026-07-26-tblind-ladder/`, `…/2026-07-26-tblind-rung1/`) |
| **C3** | ⭐ **Give the ENCODER metric ego-motion — an auxiliary ego-motion/odometry supervision on the encoder, not on a probe head** | If M2 is right, the latent simply does not carry metric speed and no head can. Supervise the **encoder** to produce it (aux ego-motion head with the gradient attached at full rate, or an explicit speed-regression target on `z`). | ⚠️ **the program has already done the adjacent experiment and it WORKED** — REF-A's speed-R² 0.61 → 0.965 (`INHERITED`, unverified here) — **but by INJECTING speed, not by supervising the encoder to perceive it.** ⇒ the perception version is **untested in our program** | **NONE for the perception version** — mechanism-plausible | new aux head + loss in `flagship_loss`; the targets (`future_poses`, `pose_last`) are already in the batch | **a full run.** ⛔ **Do not commit until X1 says the information is recoverable at all.** |
| **C4** | **Scheduled sampling / self-forcing in latent space** | Anneal the readout's input from real pairs to imagined pairs (or vice versa) over training. | Scheduled Sampling (Bengio et al. 2015, arXiv:1506.03099); Professor Forcing (Lamb et al. 2016, arXiv:1610.09038); **Self Forcing** (Huang et al., NeurIPS 2025, arXiv:2506.08009) — *"exposure bias, where models trained on ground-truth context must generate sequences conditioned on their own imperfect outputs"* | **ADJACENT, and pointed the WRONG WAY** — all three repair *train-real → test-imagined*. **We are already at the endpoint they anneal towards.** Applying them literally makes our defect worse | — | ⛔ **Not recommended as stated.** Its only useful form is C1 (a *mixture*, not an anneal). Listed so it is visibly considered and visibly rejected |
| **C5** | **KL / distributional tie between imagined and encoded latents (beyond the existing L2)** | Replace/augment `_rollout_loss`'s L2 with a distributional term so `ẑ` is not merely the conditional mean. | PlaNet latent overshooting; DreamerV2 KL balancing | **ADJACENT** | substantial — our predictor is deterministic; this means adding a stochastic latent. **Architecture change** | **a full run + architecture risk.** ⛔ Deliberately last among training fixes |
| **C6** | **Representation-level distillation from a metric-competent teacher** (e.g. a monocular metric-depth/VO model) | Distil a teacher that *does* recover metric scale into our encoder. | monocular metric scale is recoverable only with a scale prior — camera height / known object size / fixed calibration (`PUBLISHED`, §5.1) | **ADJACENT** | a new teacher, a new loss, and a corpus pass | **weeks.** Listed for completeness; not scheduled |
| **C7** | ⛔ **Anything that only re-fits `grounding.step` on real pairs, frozen trunk** | — | — | — | — | ⛔ **REFUTED BEFORE PROPOSED — §2.2 is that experiment, run for 30k steps at 2× weight, plateauing 33× short.** |
| **C8** | ⭐⭐ **HORIZON-BANKED READOUT SELECTION — a lead-time selection rule over the `op`/`tac`/`str` bank we already own** *(added after the §7 survey; see A4)* | Do not pick one readout. Select per horizon: `op` for the first steps, `tac`/`str` beyond the crossover. This is **exactly** the architecture three SOTA forecasting systems converged on independently | ⭐ **the strongest published evidence in this whole report.** **GraphCast** recommends it in prose (*"combining multiple models with varying numbers of AR steps, e.g., for short, medium and long lead times"*); **FuXi** ships a 3-model cascade (skillful lead time Z500 9.25 → 10.5 d); **Pangu-Weather** ships four lead-time models and beat operational NWP; **APEBench** quantifies the crossover cost at **11 %** first-step error | **MATCHED in shape** (a short-trained readout vs a long-trained one on the same rollout), though all four are weather/PDE, not driving | ⭐ **a selection rule in `taniteval/rollout.py::collect` and the two `canary_rollout`s.** The three heads **already exist and are already trained** in every 4-brain checkpoint | ⭐ **ZERO GPU, and the crossover is ALREADY MEASURED**: `str` is worse at 0.5 s by −0.0449 m [−0.0549, −0.0350] and better everywhere ≥ 1 s (`INHERITED`, `…/2026-07-26-tblind-ladder/`). ⇒ the optimal rule is **not** "always `str`", it is a crossover at ~0.5–1 s, and we have the data to place it |
| **C9** | ⚠️ **The LAW/Epona pattern — move the metric head onto the OBSERVED latent and demote imagination to auxiliary supervision** *(added after the §8 survey)* | Every verifiable driving world model with a metric head reads the **observed** latent for it; predicted latents are supervision targets only (LAW: `L_latent = Σ‖p_{t+1} − v_{t+1}‖₂`) | LAW planning L2 **0.26/0.57/1.01 m, avg 0.61**; Epona's TrajDiT reads the observation-derived latent and **names our pathology explicitly** | **MATCHED in design, NOT in measurement** — neither paper measures the gap, they merely never create it | ⛔ **a different product.** It gives up multi-step blind imagination, which is the capability this program is being asked to build. **Not a drop-in** | ⛔ **Not scheduled.** Recorded because it is the field's actual answer and the program should know it is deviating deliberately, not by oversight |

## 5.1 ⚠️ The physical constraint that bounds C3 and C6, and it is `PUBLISHED`

**Monocular metric scale is not free.** *"Monocular vision suffers from the scale ambiguity issue due to
lacking scale constraints"*; *"a single RGB image can only constrain depth up to an unknown affine
transformation"*; recovery requires a scale prior such as *"the camera height or size of known objects"*
(`PUBLISHED`, monocular VO / metric-depth literature — see the citation table).

Two consequences for us, both actionable:

1. **Our real-pair decoder's ~1.0 m plateau is not obviously a training failure — it may be near the
   information ceiling of a monocular pooled latent** unless a scale prior is learned from a fixed rig.
2. 🔴 **And our rig is NOT fixed.** *"Most monocular depth estimation models are trained for a specific
   camera calibration, and using them with another camera leads to ill-scaled predictions"* (`PUBLISHED`).
   Our own corpus fact (`INHERITED`, memory `physicalai-single-rig`, **not re-verified here**): PhysicalAI
   AV front-wide has **TWO rigs** (cy ≈ 543 / cy ≈ 755), and a geometric-centre crop is ~215 px wrong for
   rig B. ⇒ **HYPOTHESIS: our corpus contains two incompatible scale priors, which would cap any learned
   metric visual odometry.** **This is testable for free** — X1b, §6.

**If X1 + X1b confirm this, C3 and C6 are both partly futile until the rig is handled, and the honest
answer becomes: our latent cannot do metric VO, the readout is an action-integrator, and that is a design
we should own explicitly rather than repair.**

---

# 6. THE CHEAPEST EXPERIMENTS THAT WOULD CONFIRM OR REFUTE — ranked, with pre-registered falsifiers

**All five fit in a day. Four need no GPU. None touches a training pod.**

| # | experiment | cost | what it decides | ⛔ pre-registered falsifier |
|---|---|---|---|---|
| **X1** 🔴 | ⭐⭐ **The frozen-latent probe ceiling.** Cache encoded latents for N val windows with v1's frozen encoder. Fit, **on CPU**, three probes from `(z_t, z_{t+k})` → true Δpose: (i) ridge/linear, (ii) the same 2-layer MLP as `MetricInverseDynamics`, (iii) that MLP at 4× width. Report metre error at k ∈ {1,2,4,20} against the mean-speed baseline (§2.2). **This asks the only question that matters: is the metric ego-motion IN the latent at all, or did the trunk never encode it?** | **~20 GPU-min to cache latents** (eval pod or pod2 when free) **+ ~15 CPU-min to fit.** Everything after the cache is dev-box CPU | **F2 vs the head hypothesis.** If a fresh probe reaches ≲ 0.2 m at k=1, the information IS there and `invdyn`'s 1.0 m is an optimisation/multi-task-interference failure ⇒ **C1/C3 are worth a run.** If all three probes plateau near `invdyn`'s ~1.0 m, the information is NOT there ⇒ **C3 is futile without new inputs, and M2 stands** | **REFUTES the "head is the locus" hypothesis if the 4×-width MLP does not beat the trained `invdyn` head's 1.0129 m by ≥ 30 %.** Both outcomes are committed in advance and both are informative |
| **X1b** | ⭐ **Stratify X1 by rig.** Split the probe fit and score by per-clip `cy` (rig A ≈ 543 / rig B ≈ 755). Fit within-rig and cross-rig. | **+0 GPU** (same cache) | whether the two-rig scale-prior conflict (§5.1) caps metric VO | **REFUTES the two-rig-confound hypothesis if within-rig probe error is not ≥ 20 % better than pooled.** If it fires, it is a corpus fix, not a model fix — and it is cheap |
| **X2** | ⭐ **The surgical shortcut test.** On frozen v1: hold the latent window fixed, scale ONLY the `v0` action channel by ×{0.8, 0.9, 1.0, 1.1, 1.2}, roll, decode with `step["op"]`, and regress decoded speed on injected `v0`. | **~20 GPU-min, eval only** | **M1 vs M2 directly and surgically** — it removes the 3-variable-bundle caveat on §2.3 | **REFUTES M2 (the shortcut reading) if d(decoded speed)/d(v0) < 0.5.** If it is ≈ 1.0, the readout is an action-integrator and that must go in the registry |
| **X3** | **Second-slot swap.** Decode `(ẑ_t, z_true_{t+1})` and `(z_true_t, ẑ_{t+1})` separately at matched steps. §3.5 of the source report gives us the second; the first is missing. | **~10 GPU-min** (reuses `blindimag.py`, one new state-source) | **which SLOT carries the defect.** If only the second slot matters, M1's "contracted second slot" is real and C1 is well-aimed | **REFUTES the "both slots equally" null if the two arms differ by < 20 %** |
| **X4** | ⭐ **Zero-cost log sweep across every arm.** Re-run §2.2's band analysis on **all** committed `*_train_log.jsonl` (v2, v3enc, expA-nodrop, and any v4 log that carries `g_*` keys) and on the four worktree logs under `.claude/worktrees/`. | **0 GPU, ~5 CPU-min** | whether the widening real/imagined gap is universal or v1-specific — **it should be checked on v4 before v4 inherits the conclusion** | **REFUTES universality if any arm shows the ratio SHRINKING with training** |

## 6.1 ⭐ The single cheapest discriminating experiment, if only one is run

> ## **X1. It costs ~20 GPU-minutes of caching and 15 CPU-minutes of fitting, it needs no training pod, and it is the ONLY test that separates "our head is mis-attached" (fixable by C1, ~10 lines, ~0 GPU) from "our latent has no metric ego-motion" (C1 will not help, and C3/C6 are the only routes). Every other item in §5 is ranked on an assumption X1 resolves.**

**Both outcomes are committed in advance, per `CLAUDE.md` operating rule 5.** If X1 says the information is
there, we add C1 to the next run and expect the peek arm to turn positive. If X1 says it is not, we write
in the registry that **the metric readout is an action-integrator, not a perception decoder**, we stop
building re-anchoring and duty-cycle machinery on it, and the hierarchy work aims at the control loop
instead — which `…/2026-07-26-tblind-ladder/` has already MEASURED to be the dominant deployable lever
(action loop alone 0.8 → 3.2 s of `T_blind`; `INHERITED`).

---

# 7. HORIZON MISMATCH — a readout trained at k, applied at K ≫ k

*(Literature synthesis pending a parallel search; our own side is complete and is recorded here so the
section stands alone.)*

**Our measured side, for the record** (`INHERITED` from two committed streams, not re-measured here):
`op_fwd_k = 4`, `tac_fwd_k = 16`, `str_fwd_k = 20` — every published grounded number decodes with
`step["op"]` (**4-step calibrated**) read at **k = 20**, a 5× extrapolation, and at 185 in the sweep (46×).
Swapping to the 20-step-calibrated readout, same weights: `ade_0_2s` **0.3839 → 0.1950**; beats-CV horizon
**7.4 s → 18.5 s**; deployable `T_blind` **0.8 s → 2.5 s [2.5, 3.9]**; and on v4's own checkpoint
`wm_canary_ade_2s` **1.1409 → 0.5446 (`tac`) / 0.5521 (`str`)**
(`…/2026-07-26-blind-imagination/`, `…/2026-07-26-tblind-ladder/`, `…/2026-07-26-tblind-rung1/`).
⚠️ The longer-trained readout is **slightly WORSE at short horizon** (−0.0449 m [−0.0549, −0.0350] at
0.5 s, separated) — a genuine crossover, not a free win.

## 7.1 What the literature knows — and it knows a lot, in four separate fields that do not cite each other

`PUBLISHED (cited)`. ⚠️ **`INHERITED` in provenance:** this subsection is the product of a delegated
literature search run in parallel by this stream. I re-verified the two most load-bearing identifiers
myself; the rest carries the delegated agent's verification tags, which are reproduced honestly in
`CITATIONS.md` §F rather than laundered into a flat citation.

> ### ⭐ **The sign we measured is the sign the field measures, every time: train longer → slightly WORSE at short horizon, much better at long. And the fix three SOTA systems independently converged on is not "retrain at the evaluation horizon" — it is a BANK OF HORIZON-SPECIALISED READOUTS SELECTED BY LEAD TIME. We already own that bank.**

| source | what it DEMONSTRATED | how it matches us |
|---|---|---|
| ⭐ **APEBench** ([arXiv:2411.00180](https://arxiv.org/html/2411.00180v1)) | verbatim: *"when unrolling for 20 steps during training, the learned solution still performs better after 30 steps while having an **11 % increased error at the first step**"*; Fig. 2 caption: *"More unrolling improves long-term accuracy for a small sacrifice in short-term performance"* | ⭐ **the closest quantitative match anywhere.** Our 20-step readout is worse at 0.5 s by **−0.0449 m [−0.0549, −0.0350]**, separated (`INHERITED`, `…/2026-07-26-tblind-ladder/`). **Same sign, and now with a published magnitude to compare against: 11 %** |
| ⭐ **GraphCast** (Science 2023, [arXiv:2212.12794](https://arxiv.org/pdf/2212.12794), suppl. Fig. 30) | *"Models trained with fewer autoregressive steps tended to trade longer for shorter lead time accuracy"*; trained on 12 AR steps, evaluated to 10 days (~3.3× extrapolation vs our 5×). Its own recommendation: *"combining multiple models with varying numbers of AR steps, e.g., for short, medium and long lead times"* | the recommendation **is our `op`/`tac`/`str` bank** |
| ⭐ **FuXi** ([arXiv:2306.12873](https://ar5iv.labs.arxiv.org/html/2306.12873)) | *"using a single model is insufficient for achieving the best performance for both short and long lead times"* — ships a **3-model cascade**; skillful lead time Z500 **9.25 → 10.5 d**, T2M **10 → 14.5 d** | independent restatement of GraphCast's finding, **plus a deployed cascade** |
| **Pangu-Weather** (Nature 619:533, [arXiv:2211.02556](https://arxiv.org/pdf/2211.02556)) | four separate models at 1 h / 3 h / 6 h / 24 h lead time, composed greedily to minimise iteration count | horizon-specialised readouts, **deployed, beating operational NWP** |
| **Benechehab et al.** ([arXiv:2310.05672](https://ar5iv.labs.arxiv.org/html/2310.05672)) | *"models where α<1.0 suffer a worse predictive error for short horizons, but recover and beat the vanilla one-step model down-the-horizon"* — the same crossover inside MBRL dynamics models | our crossover, in our own field |
| ⚠️ **PlaNet Fig. 7 — the counter-evidence** ([arXiv:1811.04551](https://ar5iv.labs.arxiv.org/html/1811.04551)) | latent overshooting *"can substantially improve the performance of the DRNN and other models... but **slightly reduces performance of our RSSM**"* | ⛔ **multi-horizon training is NOT a free win.** It repairs *capacity-limited or misspecified* predictors and can hurt an adequate one. ⇒ **that our 20-step head wins by 2× is evidence our READOUT is the misspecified part** — which is exactly what §2 says by a different route |

## 7.2 The direct-vs-iterated (DMS/IMS) theory answer, and the genuine disagreement

The classical framing is **direct multi-step (DMS)** — fit a model whose target is *h* steps ahead — versus
**iterated multi-step (IMS)** — fit one one-step model and iterate. The trade is bias against variance. If
the one-step model is correctly specified, iterating the true one-step conditional expectation **is** the
*h*-step conditional expectation, so IMS is efficient and DMS only pays for extra parameters on
effectively less informative overlapping data. If the one-step model is **misspecified**, the
misspecification **compounds** under iteration while DMS targets the *h*-step loss directly — Chevillon's
survey concludes DMS wins in finite samples when the process is **both misspecified and non-stationary**
(Chevillon 2007, *J. Econ. Surveys* 21(4):746–785). Ing shows the choice **cannot be settled by identifying
the true model order** — method and order must be selected jointly (Ing 2003/2004).

⚠️ **The literature genuinely disagrees on the empirical winner, and the disagreement must be stated.** The
largest clean test — 170 US monthly macro series, 1959–2002 — found **iterated typically BEATS direct, and
the iterated advantage GROWS with horizon** (Marcellino, Stock & Watson 2006, *J. Econometrics*
135(1–2):499–526). The ML benchmark on 111 NN5 series found multi-output strategies beat both pure forms
(Taieb et al. 2012, [arXiv:1108.3259](https://arxiv.org/abs/1108.3259)).

> ### **Why MSW does not bind on us, stated so nobody has to re-derive it:** MSW's DMS cost is **re-estimating an entire model per horizon** from ~500 scarce observations — a pure variance penalty. **We re-estimate nothing.** Our rollout is shared and fixed; only a ~0.8 M-parameter head's loss horizon changes, on the same data. We pay essentially none of DMS's variance cost, and §2/§7 give direct evidence the short-horizon readout is misspecified for 20-step accumulation. **In the regime we occupy, "calibrate the readout at the horizon you read it at" is close to unopposed — but if we ever fit SEPARATE FULL PREDICTORS per horizon, MSW starts applying.**

## 7.3 ⛔ What the literature does NOT have, and what that makes our result

* ⛔ **Nobody has run our experiment.** No paper freezes a latent rollout, trains several readout heads at
  different *k* on that **same** rollout, and swaps them at test time with **no retraining**.
  GraphCast/FuXi/Pangu/APEBench all retrain or fine-tune the whole model. ⇒ **our readout swap is a
  confirmation of a known mechanism in a genuinely new place**, and that is the honest framing — not a
  discovery of the mechanism.
* ⛔ **There is no scaling law** for the extrapolation penalty as a function of *K/k*. Our data point
  (**5× extrapolation → 1.97× error ratio**) has **no published comparator**.
* ⛔ **No AV / trajectory-prediction paper ablates the SUPERVISION horizon of a trajectory head.** The field
  universally reports ADE@1/2/3 s and universally fails to vary the training horizon. 🔴 **That is a
  publishable gap in our own field and we are sitting on the experiment.**
* ⚠️ **A false lead, logged so it is not re-followed** (`RETRACTION_LOG` class: *leading-prompt confirmation
  on a summarising tool*): *"Closing the Train-Test Gap in World Models for Gradient-Based Planning"*
  ([arXiv:2512.09929](https://arxiv.org/abs/2512.09929)) looks like a perfect hit from its title and a
  first fetch appeared to confirm it — **that was the fetch model echoing the leading question back.** The
  verified abstract shows the gap is *next-state-prediction objective vs. test-time action estimation*,
  **not** horizon mismatch. **Do not cite it for this question.**

---

# 8. DRIVING-SPECIFIC WORLD MODELS — does anyone report an observed-vs-imagined decode gap?

`PUBLISHED (cited)`. ⚠️ Same provenance caveat as §7.1: delegated parallel search, tags preserved in
`CITATIONS.md` §F. I independently re-verified Valdi's identity and authorship.

> ## ⛔ **NO DRIVING WORLD MODEL MEASURES OUR DEFECT — because essentially none of them is built the way we are built. In every driving world model whose wiring could be verified, the decoder is trained on ENCODED (OBSERVED) latents. Decoding an IMAGINED latent is the UNTRAINED direction for them; for us it is the TRAINED one. Our 9.4× is the same axis at the opposite polarity.**

## 8.1 ⭐ The single most important finding: one paper builds it our way, states our rationale, and never measures the cost

**Valdi — "Valdi: Value Diffusion World Models", Lindenberg & Chitta** ([arXiv:2607.00917](https://arxiv.org/abs/2607.00917), 2026-07-01).
Trains each decoder on **the model's own imagined latents** — *"each decoder is trained on an on-policy
dataset collected by rolling out exactly the model whose latents it reconstructs"* — with the rationale
stated verbatim: *"This ensures each decoder is trained on the latent distribution induced by its own
model, so the decoded futures faithfully reflect that model's predictions rather than an
out-of-distribution mapping."* (App. G.2.)

**It never measures what that costs.** `PUBLISHED (ASSERTED, prose — not demonstrated)`.

⚠️ **Verification status, stated precisely.** I fetched the paper page myself and **independently confirmed
title, authors and that the experiments are in CarRacing** — *"In preliminary experiments on the CarRacing
environment…"*, i.e. **not driving-scale**. The App. G.2 quotes are **`INHERITED`** from the delegated
search's full-text read and I could not reach them.

> ### ⇒ **We have the number that paper is missing. The design choice Valdi asserts is safe, we measured at 9.4× (eval) / 33× (train). That is a genuine, defensible contribution — and it is the first thing to say if this program publishes.**

## 8.2 The per-family answer

| family | what the decoder is trained on | reports an observed-vs-imagined decode gap? | has a METRIC (metres) head, and what does it read? |
|---|---|---|---|
| **GAIA-1** ([2309.17080](https://arxiv.org/abs/2309.17080)) | tokens from the image tokenizer on **real frames**; the video decoder is trained **independently** of the world model — *"During training, our video diffusion model is conditioned on the image tokens obtained by discretizing input images with the pre-trained image tokenizer"*, *"During inference … on the predicted image tokens from the world model"* | ⛔ **no — the paper has no quantitative results table at all** | no; speed + curvature are **inputs** |
| **GAIA-2** ([2503.20523](https://arxiv.org/abs/2503.20523)) | encoded latents of **real frames** (video tokenizer = autoencoder) | ⛔ no — **no tokenizer-reconstruction / oracle row anywhere**; reconstruction quality is invisible in the results | no; ego action is conditioning only |
| **Vista** ([2405.17398](https://arxiv.org/abs/2405.17398)) | SVD VAE, pretrained; frozen-vs-finetuned **not stated in the paper** | ⛔ no; 15 s rollouts shown **qualitatively only** | no — trajectory in metres is an **input** condition |
| **GenAD** ([2403.09630](https://arxiv.org/abs/2403.09630)) | ⛔ **UNVERIFIED — every full-text route failed** (ar5iv conversion error, `/html/` 404, PDF over size limit, CVPR OA 403, Semantic Scholar 404). Title/authors/abstract only | **UNVERIFIED** | **UNVERIFIED** |
| **DriveDreamer / -2** ([2309.09777](https://arxiv.org/abs/2309.09777), [2403.06845](https://arxiv.org/abs/2403.06845)) | not explicitly stated; SD backbone frozen | ⛔ no | v1 has an action branch (yaw + velocity); v2 pixels only |
| **Drive-WM** ([2311.17918](https://arxiv.org/abs/2311.17918)) | encoded latents of **real frames** | ⛔ no isolated real-vs-generated perception gap | ⭐ **yes — planning L2 in metres (Tab. 3), and the planner reads REAL observations** |
| **MUVO** ([2311.11762](https://arxiv.org/abs/2311.11762)) | ⚠️ **not specified** — RSSM-style prior/posterior, but the paper never says which the decoders see in training. **A real ambiguity in the source, not a search failure** | ⛔ no | no |
| ⭐ **MILE** ([2210.07729](https://arxiv.org/abs/2210.07729)) | decoders take the **POSTERIOR**; prior matched by KL only | ⭐ **indirectly YES** — Fig. 4 *"Driving in imagination"*: at 0 / 30 / 60 % imagining, driving score ≈62 → ≈61 → ≈50, BEV IoU ≈60 → ≈55 → ≈40 ⚠️ (read off a plot) | action head a ∈ ℝ², not metric poses |
| **OccWorld** ([2311.16038](https://arxiv.org/abs/2311.16038)) | stage 1 on **GT occupancy**; stage 2 freezes tokenizer, decodes **predicted** tokens | ⭐ yes, via I²-World (§8.3) | ⭐ **yes — ego displacement head; planning L2 0.43/1.08/1.99 m** |
| **Copilot4D** ([2311.01017](https://arxiv.org/abs/2311.01017)) | **encoded (observed)** tokens — *"The tokenizer is trained end-to-end to reconstruct the observation"* | ⛔ no dedicated oracle row | no; ego SE(3) poses are **inputs** |
| **iVideoGPT** ([2405.15223](https://arxiv.org/abs/2405.15223)) | encoded (observed) tokens; tokenizer **not** retrained on predicted tokens | ⛔ no | ⚠️ **and it is not a driving model** — pre-trained for robotic manipulation |
| **UniSim / NeuRAD / DrivingGaussian / S-NeRF++ / MARS** | reconstruction of **observed** sensor data; no latent dynamics | ⭐ **NeuRAD and UniSim: YES** (§8.3). DrivingGaussian, S-NeRF++, MARS: ⛔ **no — all test on the RECORDED trajectory**, so off-distribution decode is never measured | no |
| ⭐ **LAW** ([2406.08481](https://arxiv.org/abs/2406.08481)) | predicts future latents supervised **against the real encoded latent**: `L_latent = Σ‖p_{t+1} − v_{t+1}‖₂` | ⛔ no — never measures predicted-vs-real latent divergence | ⭐⭐ **yes — planning L2 0.26/0.57/1.01 m, avg 0.61. And THE TRAJECTORY HEAD READS THE OBSERVED LATENT (`E = V + H`); predicted latents are auxiliary supervision ONLY** |
| ⭐ **Epona** ([2506.24113](https://arxiv.org/abs/2506.24113)) | DCAE encoder **frozen** | ⛔ no latent-source table | ⭐ **yes — TrajDiT reads the OBSERVATION-derived historical latent.** And it **names our pathology**: *"during training, the model predicts the next frame using ground-truth historical context, whereas during inference, it relies on its own past predictions. This domain gap… leads to error accumulation"* — fixed by a *chain-of-forward* strategy |
| **HorizonDrive** ([2605.11596](https://arxiv.org/abs/2605.11596)) | VAE decoder **pretrained and frozen, never retrained** | ⚠️ asserted, not cleanly demonstrated (no GT-vs-generated-conditioning ablation) | no; poses recovered **post-hoc from pixels** via VGGT. Names *"exposure bias"* explicitly — but **corrects the DYNAMICS MODEL, not the decoder** |

## 8.3 The two families that DID measure something, and why neither is comparable to our 9.4×

**(1) Occupancy world models — I²-World Table 1** ([arXiv:2507.09144](https://arxiv.org/abs/2507.09144)):
tokenizer **reconstruction** vs **forecasting** on the same benchmark — OccWorld-O 66.38 → 17.14 mIoU
(**3.9×**), OccLLaMA-O 75.20 → 19.93 (**3.8×**), DOME 83.08 → 27.10 (**3.1×**), I²-World-O 81.22 → 39.73
(**2.0×**). Cross-checked three ways by the delegated search (OccWorld's own Tab. 1; OccSora's Tab. 1;
DOME's own Tab. 1).

> ⛔ **These are NOT comparable to our number and must not be quoted as if they were.** They are measured
> **across a time gap** — they confound the decoder being off-distribution with the future being genuinely
> uncertain. **Ours is measured at a MATCHED TIMESTEP**, which is what makes it a clean decoder-transfer
> measurement. ⭐ **No paper in that family disentangles the two, and none of I²-World / DOME / OccWorld /
> OccLLaMA attributes the gap, mentions exposure bias, or ablates it. That is an open lane, and our
> instrument is the one that opens it.**

⚠️ **Verification:** I could not reach I²-World's table myself (abstract page only). The numbers are
**`INHERITED`** from the delegated search's three-probe cross-check. Do not let them decide anything.

**(2) Off-distribution rendering — NeuRAD Table 3** ([arXiv:2311.15260](https://arxiv.org/abs/2311.15260)),
*"FID scores when shifting pose of ego vehicle or actors"*, **with a no-shift baseline column** so the
degradation is computable: PandaSet FC NeuRAD **25.0 → 72.3 @2 m lane shift (2.9×) → 93.9 @3 m (3.8×)**;
UniSim* 41.7 → 79.6 (1.9×) → 102.0 (2.4×). UniSim's own paper reports the same protocol and notes *"the
gap is more significant in extrapolation settings."*

⇒ **This is the closest published analogue IN KIND — a decoder asked to operate off the distribution it
was fit on — and it lands at 2–4×, the same order as our 9.4×.** ⚠️ FID and ADE are **not commensurable**;
the agreement is in order of magnitude only, and I say so rather than implying a comparison.

**(3) The one readout-head Enc-vs-WM comparison found anywhere** —
*"Reconstruction or Semantics?"* ([arXiv:2605.06388](https://arxiv.org/abs/2605.06388)) tabulates an IDM
readout and a success classifier on **encoder latents vs world-model latents**, which is exactly the right
experiment. ⛔ **But it is robotics (BridgeV2 manipulation), not driving, and the delegated search's two
independent transcription probes DISAGREED on the Enc/WM column alignment**, so no per-row number is
admissible. Only the prose direction is supported: readouts degrade on WM latents, and semantic encoders
degrade less than reconstruction-aligned ones. **`PROVISIONAL` at best.**

## 8.4 ⭐ The design conclusion the driving field has already converged on, and it is not ours

**Of every driving world model with a metric head whose wiring could be verified — OccWorld, Drive-WM,
LAW, Epona — EVERY ONE READS THE OBSERVED LATENT for that head.** LAW is explicit that predicted latents
are *supervision targets only*. Epona is explicit that TrajDiT conditions on the observation-derived
latent.

> ## 🔴 **A metric SE(2) head trained EXCLUSIVELY on imagination-rollout pairs, as in TanitAD, has no published precedent in driving. We are not doing a known thing slightly wrong; we are doing an unattested thing. That deserves either a defence or a change — and §5's C1 is the cheapest defence available.**

---

# 9. LIMITATIONS, STATED PLAINLY

1. ⚠️ **§2.2's headline ratio compares two differently-aggregated statistics** (endpoint-at-k averaged over
   {1,2,4} vs ADE over 1…4). Corrected generously the ratio is ~17×, not 33×. **The trend is
   aggregation-invariant; the absolute ratio is not.** A matched-aggregation version costs one extra logged
   line in the next run and I have not computed one.
2. ⚠️ **§2.2 and §2.3 are TRAIN-batch numbers.** They are the strongest possible statement about *fitting*
   (the head cannot fit data it has seen) and say nothing directly about generalisation. The val-side
   equivalent is the (c2) arm at 3.6093, which agrees in kind.
3. ⚠️ **§2.3 is a 3-variable bundle** (speed channel + aux_accel + jerk), not a surgical ablation. **X2 is
   the surgical version.** I have not run it.
4. ⚠️ **The mean-speed baseline in §2.2 is `ESTIMATED`** — a straight-line approximation on a
   **val episode-initial subsample** (n = 599, itself ~6–12 % easier than the full window set per the
   source report's own §6.1), compared against a **train** number. Order-of-magnitude only.
5. ⚠️ **M2 (the shortcut reading) is a `HYPOTHESIS`.** It has three MEASURED supports and one pre-existing
   ablation; it does not have a surgical test. Do not let it into the registry until X2 runs.
6. ⚠️ **The "nobody reports our direction" claim is a negative established by one search strategy.**
   Per `CLAUDE.md` rule 2, absence at one location is not absence. I give a *structural* reason in §4 for
   why it should be rare, which is stronger than the search, but it remains **`UNVERIFIED` as an exhaustive
   negative.**
7. ⚠️ **V-JEPA 2-AC's planning-cost mechanism is `UNVERIFIED`.** I confirmed the paper and its nature but
   could not reach the energy-function text. The §4 row is marked accordingly.
8. **Everything about our model is v1** (`flagship4b-speedjerk-30k` @ 29999) plus the no-speed control.
   **v4 has not been checked and must be before it inherits any of this** — that is X4, and it is free.
9. **No experiment was run by this stream**, per `PRE_REGISTRATION.md` §5. Every number here is read from a
   committed artifact or the source code.
10. ⚠️ **§7 and §8 are DELEGATED and therefore `INHERITED`.** Two parallel search agents produced them under
    the same evidence discipline; I re-verified Valdi's identity myself and **failed** to reach I²-World's
    table. Per `CLAUDE.md` rule 1 nothing in those sections may decide a GPU-day. Every delegated
    verification tag — including the failures — is preserved in `CITATIONS.md` §F.
11. ⚠️ **Three specific literature claims are UNVERIFIED AT SOURCE and are flagged where they appear:**
    GraphCast's Fig. 30 **numeric values** (PDF over the fetch size limit; the sentences were confirmed
    three ways, the magnitudes were not); **Bhansali (1997)**, which is the strongest pro-DMS theory claim
    encountered and whose PDF would not render; and **GenAD's entire internals** (six independent retrieval
    routes failed). ⛔ **Do not quote any of the three.**
12. ⚠️ **The occupancy-family recon-vs-forecast gaps (2.0–3.9×) are NOT comparable to our 9.4×** — they are
    measured across a **time gap** and confound decoder off-distribution error with genuine future
    uncertainty. Ours is measured at a **matched timestep**. The report says this three times because the
    numbers are superficially similar and will otherwise be equated.

---

# 10. AMENDMENTS

| # | what changed vs the pre-registration | why, and what it can bias |
|---|---|---|
| **A1** | The pre-registration framed F4 as *"a zero-GPU probe on our own checkpoint"*. It turned out to need **no probe at all** — the quantity is already logged 620× per run as `g_op_mid_de_m`. | Strictly cheaper and strictly more direct than what was registered. It cannot bias the direction: the statistic was written to disk by the trainer months before this stream existed and was not chosen after seeing a result. |
| **A2** | The no-speed **attributing ablation** (§2.3) was not in the pre-registration. | It was found while verifying A1 across arms, **before** any fix ranking was written. It is a *mechanism* result, not one of the four falsifiers, and no falsifier's verdict depends on it. ⚠️ It is the one place a comparison was constructed after the fact and the reader should discount it until X2 confirms it surgically. |
| **A3** | §6 grew from "the cheapest experiment" (singular, as briefed) to five, with one named as the answer if only one runs. | The brief asked for the cheapest confirming experiment; four of the five are free or near-free and pruning them would have hidden cheap information. §6.1 gives the singular answer the brief asked for. |
| **A4** | **Candidates C8 (horizon-banked readout selection) and C9 (the LAW/Epona pattern) were added to §5 AFTER the §7 and §8 surveys returned**, and C8 was ranked above several pre-existing candidates. | ⚠️ **This is a ranking changed by evidence arriving mid-stream, and the reader should see it as such.** C8 is not a new mechanism — it is the pre-existing horizon lever (C2) given a **better shape** by four independent published systems. It could not have been proposed before the survey because the *bank* framing is the survey's contribution. It cannot bias the report's central finding (§2/F4), which was complete before either survey returned and does not depend on any literature. |
| **A5** | §7 and §8 were produced by **two delegated parallel search agents**, not by me directly. | ⚠️ Declared rather than laundered. I re-verified the two most load-bearing identifiers myself (Valdi's identity/authors/environment; and I attempted and **failed** to reach I²-World's table). Per `CLAUDE.md` rule 1 the relayed material is **`INHERITED`** and **must not decide a GPU-day**; it is used here to *frame* and *rank*, never to justify. Every delegated verification tag is preserved verbatim in `CITATIONS.md` §F. |

---

# 11. CITATION TABLE

See `CITATIONS.md` in this folder.

---

# 12. DELIVERABLE MANIFEST

See `MANIFEST.md` in this folder.
