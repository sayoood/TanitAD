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

<!-- PENDING: horizon-mismatch literature (direct vs iterated multi-step forecasting; training-horizon
ablations in MBRL). See §11 for what lands. -->

---

# 8. DRIVING-SPECIFIC WORLD MODELS

<!-- PENDING: GAIA-1/2, Vista, GenAD, DriveDreamer, iVideoGPT, occupancy and Gaussian/NeRF lines —
does any of them report a decode-from-observed vs decode-from-imagined gap? -->

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

---

# 10. AMENDMENTS

| # | what changed vs the pre-registration | why, and what it can bias |
|---|---|---|
| **A1** | The pre-registration framed F4 as *"a zero-GPU probe on our own checkpoint"*. It turned out to need **no probe at all** — the quantity is already logged 620× per run as `g_op_mid_de_m`. | Strictly cheaper and strictly more direct than what was registered. It cannot bias the direction: the statistic was written to disk by the trainer months before this stream existed and was not chosen after seeing a result. |
| **A2** | The no-speed **attributing ablation** (§2.3) was not in the pre-registration. | It was found while verifying A1 across arms, **before** any fix ranking was written. It is a *mechanism* result, not one of the four falsifiers, and no falsifier's verdict depends on it. ⚠️ It is the one place a comparison was constructed after the fact and the reader should discount it until X2 confirms it surgically. |
| **A3** | §6 grew from "the cheapest experiment" (singular, as briefed) to five, with one named as the answer if only one runs. | The brief asked for the cheapest confirming experiment; four of the five are free or near-free and pruning them would have hidden cheap information. §6.1 gives the singular answer the brief asked for. |

---

# 11. CITATION TABLE

See `CITATIONS.md` in this folder.

---

# 12. DELIVERABLE MANIFEST

See `MANIFEST.md` in this folder.
