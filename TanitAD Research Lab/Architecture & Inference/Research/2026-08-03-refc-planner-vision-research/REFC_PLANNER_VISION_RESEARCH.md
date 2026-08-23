# REF-C — diffusion planner + vision: what to actually change, and what to refuse

**Date:** 2026-08-03 (Europe/Berlin, UTC+2) · **Type:** research direction, evidence-graded
**Author:** REF-C research stream (DIRECTION 2 of 3) · **Compute spent: none.** No GPU launched, no pod
written to, no `stack/` file edited (the CODE and DATA streams own those).
**Method:** primary sources only — REF-C read from `stack/tanitad/refs/refc.py` at HEAD, our numbers from
`Project Steering/MODEL_REGISTRY.md` and banked run artifacts, literature read from the papers themselves.

**Evidence class on every number:** `MEASURED (ours + artifact)` · `PUBLISHED (cited)` ·
`INHERITED (not re-verified here)` · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. The finding in one paragraph

Two published ablations — each with a proper *no-navigation* control — independently reproduce our own
LAN E0 result: **a categorical driving command is worth approximately nothing to a trajectory planner,
and a geometric route/goal is worth a lot.** Our "the route pathway is live but anti-compliant" is
therefore not a TanitAD defect; it is a known property of the categorical-command interface, and the
published fix is a representation change, not a head change. Separately, reading REF-C from source turned
up a structural asymmetry that no document in this repo states: **the decoder cross-attends 64 spatial
tokens from ONE frame, while the entire temporal history reaches it only as a 64-d GRU vector through a
ZERO-INIT linear that FiLMs the MLP branch.** A single RGB frame contains no relative-velocity
information — which is exactly the longitudinal family that holds 88.7 % of our oracle gap and in which
`long_accel` was measured unrecoverable. Finally, the vision lever is **not scale**: our own anchor
ladder measured the encoder over-provisioned, and the published backbone-scaling evidence is null too.
The one axis nobody — us or the cited nulls — has varied is **pretraining**, and REF-C is the only
planner in this comparison whose trunk is randomly initialised.

**Ranked shortlist:** §3. **What I refuse to recommend and why:** §4.

---

## 1. What REF-C actually is — read from source, not from prose

All rows MEASURED (source read 2026-08-03, `stack/tanitad/refs/refc.py` at HEAD; param counts and run
configs from `MODEL_REGISTRY.md` §4).

| aspect | DiffusionDrive (PUBLISHED, CVPR 2025) | **REF-C (MEASURED, ours)** | delta |
|---|---|---|---|
| vision trunk | ResNet-34, **ImageNet-pretrained** (Appendix A) | ResNet-34-*style*, torchvision-free, **randomly initialised — no pretraining of any kind** | ⚠️ **the only unpretrained trunk in this comparison** |
| decoder KV | scene tokens | `fmap_all[…][:, -1]` — **last frame only**, 8×8 = 64 tokens (`refc.py:1112-1117`) | single-frame |
| temporal path | ego status vector | `StrategicCtx` GRU → **d_ctx 64** (base) / 96 (XL) → `ctx_to_cond` **ZERO-INIT** → additive into cond | ⚠️ severe bottleneck |
| anchors | **20**, k-means on training set | **128** (base) / 256 (XL) / 64 (small), FPS over a pool | 6–13× wider |
| inference noise | anchored Gaussian, **N_infer = 20 samples** | `noise = torch.zeros_like(x) if not self.training` → **fully deterministic** | ⚠️ no sampling at all |
| confidence | predicted **at each denoising step**; top-1 at the final step | conf from the **t=0 classifier pass only**; denoise-pass confidences **discarded** | documented in registry as "the selection flaw" |
| route / goal | (not discussed in the paper) | 4-way `nav_cmd` one-hot + `v0` → 2-layer MLP → **d_out 16** (base) / 128 (XL) → additive into cond; FiLM on the MLP branch of each cross-attn layer | categorical |

Two of these — the discarded denoise confidences and the anchor width — are already documented in
`MODEL_REGISTRY.md` §4.1. **The single-frame KV / zero-init temporal bottleneck is not documented
anywhere I could find, and it is the most consequential of the seven.**

### 1.1 The measured constraints any proposal must respect

| # | constraint | class | source |
|---|---|---|---|
| K1 | **Selection is a closed lever.** A learned re-scorer recovers ≤ 8.4 % of the 0.3075 m oracle gap across **47 trained arms**; v1.2 got **+2.9 %, NOT significant** (paired Δ +0.00893 [−0.0062, +0.0250]). Hand-written cost re-rank recovered **0.0 %**. | MEASURED | `MODEL_REGISTRY.md` §4.1 |
| K2 | **The oracle gap is ~92 % irreducible** — it is a minimum over 256 candidates scored against ONE realised future, i.e. mostly aleatoric statistics, not learnable signal. | MEASURED | ibid. |
| K3 | **Encoder scale is not the fan lever.** small (48 M encoder) proposes **at least as tightly per-anchor** as base (90 M); at matched-64 vocabulary small − base = **−0.0620 [−0.0801, −0.0435] SEPARATED, small BETTER**. Registry verdict, verbatim: *"REF-C's encoder is over-provisioned even at base."* | MEASURED | `MODEL_REGISTRY.md` §4.2 |
| K4 | **More anchors ⇒ better fan but worse ranking, netting flat.** oracle-in-fan 0.2213 (64) / 0.1914 (128) / 0.1640 (256); `frac_sel_2x_worse` 0.3825 / 0.4109 / 0.4540. base vs XL selected ADE@2s +0.0013, **not separated**. | MEASURED | ibid. |
| K5 | **The route pathway is live but anti-compliant.** `nav_cmd` sweep displaces the decode by **0.2416 m** (bit-identical control exactly 0.0); feeding the **oracle** route: ADE@2s **+0.0024 [−0.0107, +0.0147] not separated**, cross-track **+0.0031 [+0.0001, +0.0063] separably WORSE**, curvature **+0.0013 [+0.0003, +0.0024] separably WORSE**. | MEASURED | `…/incoming/2026-08-03-lan-refc-e0/LAN_E0_RESULTS.md` |
| K6 | **The oracle-goal advantage is almost entirely ALONG-track.** oracle along-track + learned cross-track → **+83.7 % recovery (separated)**; oracle cross-track + learned along-track → **+2.9 % (NOT separated)**. | MEASURED | `GOAL_INPUT.md` 2026-07-27, quoted in `stack/tanitad/data/lan.py` docstring |
| K7 | **`long_accel` is unrecoverable from the frozen v1 latents** (17 arms, each against a shuffled-latent control). **2049-param ridge beat a 2.17 M head; capacity sweep peaked at 129 params.** | MEASURED (brief; not re-derived here) | REF-C brief |
| K8 | **88.7 % of the oracle gap is longitudinal.** | MEASURED | `CLAUDE.md` (four-families rule) |

**K1 + K2 + K7 together are the program's sharpest prior: the head and the ranker are exhausted; the
representation is not.** Every item below is scored against that prior, and three of the six are marked
as items the prior predicts will underdeliver — including one the literature is enthusiastic about.

---

## 2. Protocol scepticism — read this before any PUBLISHED number below

⚠️ **NAVSIM PDMS is not closed-loop, whatever the papers call it.** NAVSIM is a *non-reactive* simulator:
the policy plans once from recorded observations and the plan is unrolled; the scene does not react and
the policy gets no feedback. The NAVSIM authors state that a high PDMS **does not always imply a high
closed-loop score**, and explicitly recommend CARLA-class simulators as a complementary benchmark
(PUBLISHED, [NAVSIM](https://arxiv.org/pdf/2406.15349)). Treat every PDMS delta below as **open-loop with
simulation-derived subscores** — closer to our open-loop panel than to our AlpaSim runs.

⚠️ **None of DiffusionDrive, DiffusionDriveV2, GoalFlow, or the navigation-understanding paper reports a
confidence interval, a standard deviation, or a significance test — anywhere.** Every number is a single
point estimate on one benchmark split. Our estimator is the paired episode-cluster bootstrap over
episodes. **A PDMS delta is therefore not comparable in kind to any of our Δ's**, and a +0.2 PDMS
ablation row is inside the noise band of a re-run for all anyone can tell.

⚠️ **ADE-class metrics cannot see encoder quality.** PUBLISHED
([Is Ego Status All You Need?, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Is_Ego_Status_All_You_Need_for_Open-Loop_End-to-End_Autonomous_Driving_CVPR_2024_paper.pdf)):
blanking **all** camera input in VAD destroys perception but degrades open-loop planning only slightly
when ego status is present; 73.9 % of nuScenes is straight driving, and ego status is a shortcut that fits
the planning task. AD-MLP (ego-only) scores well open-loop and **badly closed-loop on Bench2Drive**. Our
corpus is 74 % straight and REF-C receives `v0`. ⇒ **Every falsifier below is stated on the four metric
families, never on ADE alone** — an encoder or route change can be real and still move ADE by zero.

---

## 3. The ranked shortlist — 6 changes

Ranked by (evidence quality × expected effect) ÷ cost, with the K1/K2/K7 prior applied.

---

### ⭐ 1 — Give the decoder TEMPORAL tokens, not a zero-init 64-d summary

**Status vs our prior: PREDICTED TO SUCCEED.** This is a representation change of exactly the class
K3/K7 point at, and it is the only item that addresses K8 mechanistically.

**Mechanism.** `refc.py:1112-1117` computes `fmap_all` for all W frames and then keeps **only the last**:
the 64 tokens the anchor queries cross-attend come from a single instant. The other W−1 frames survive
only as `StrategicCtx`'s GRU output — a **d_ctx 64** vector (base) pushed through `ctx_to_cond`, which is
**zero-initialised**, added to `cond`, and applied as a FiLM scale/shift on the MLP branch of each layer.
So the information budget is *64 spatial tokens with full attention* versus *64 scalars through a
bottleneck that starts at exactly zero*. A single RGB frame contains **no relative-velocity, no closing
rate, no time-to-collision** — those are differences between frames. That is precisely the longitudinal
family (K8: 88.7 % of the oracle gap) and precisely the quantity K7 found unrecoverable.

**Evidence.** MEASURED (ours, source read + K7 + K8). PUBLISHED support is *indirect and I will not
overstate it*: NAVSIM planners feed a single-timestep scene representation too — but they pair it with
**LiDAR geometry and an ego-status history**, neither of which REF-C's tokens carry. So this is not a
"copy the SOTA" item; it is a bet that we are strictly poorer than the published baselines on temporal
evidence and that our largest documented defect is the visible consequence. Class: **HYPOTHESIS**, with a
MEASURED mechanism and a 0-GPU test.

**Cost.** Probe: **0 GPU** (banked window dumps). Architecture change: KV becomes `W × 64` tokens with a
learned frame embedding — ~20 lines in `AnchoredDiffusionDecoder`, attention cost ×W. One 30 k arm at
REF-C-small is **~7 h 10 m on an A40** (MEASURED, §4.2).

**Cheapest discriminating experiment (0 GPU, runs on the eval pod's banked dumps).**
Ridge probe with a shuffled-latent control, predicting **lead-vehicle closing rate, time-gap/TTC, and
`long_accel`** from (a) REF-C's single-frame `fmap`, versus (b) the concatenated W-frame `fmap` stack, on
the canonical 40-ep / 881-window val. Same probe capacity for both (K7 says keep it tiny — the 129-param
peak is the design point, not a 2 M head).

**Pre-registered falsifier — both outcomes committed now.**
- **REFUTE:** if (b) does not beat (a) on probe R² for all three longitudinal targets beyond the
  shuffled-latent control, the temporal-token hypothesis is **dead** and item 1 is dropped without a
  training arm. In that case the longitudinal information is absent from the *pixels we feed*, and the
  next question is sensor/FOV, not architecture.
- **CONFIRM:** if (b) beats (a) beyond control, fund one REF-C-small temporal-KV arm. Gate on the
  **four families**, paired episode-cluster bootstrap, same 881 windows: LONGITUDINAL (target-speed
  accuracy, headway/time-gap/TTC) is the primary read; LATERAL (heading, curvature, yaw-rate,
  cross-track) and ADE@2s are non-regression guards; TACTICAL reports the lat/lon confusion.

---

### ⭐ 2 — Replace the categorical route with a PREDICTED GEOMETRIC GOAL POINT (two-stage factorisation)

**Status vs our prior: PARTIALLY PREDICTED TO FAIL — and the failure mode is nameable.** See the ⚠️ below.
This is the item with the best published evidence in the whole shortlist, and it is also the item where
our own K6/K7 findings bite hardest. **It should be funded *after* item 1, not before.**

**Mechanism.** REF-C's route input is a 4-way one-hot that is `follow` on 74–79 % of windows, squeezed
with `v0` through a 2-layer MLP into **16 dimensions** (base), added to `cond`, and delivered as a FiLM
scale/shift. There is no mechanism anywhere in training that *forces* compliance: the loss is imitation
of the realised future, and where the command is predictable from the image the model correctly learns to
read the image and ignore the command. GoalFlow's alternative is structural — decompose planning into
(i) score a **k-means vocabulary of 4096–8192 trajectory endpoints** and pick one, (ii) sinusoidally
encode that 2-D goal point and concatenate it into the conditioning feature, constraining the generated
endpoint.

**Evidence — two independent ablations, each with a no-condition control.**

| source | protocol | control | result |
|---|---|---|---|
| [Navigation understanding, 2604.12208](https://arxiv.org/html/2604.12208) Table V | NAVSIM (non-reactive), no CI | **ID 0 = no navigation, 85.9 PDMS** | command only **86.1 (+0.2)** · TBT only 86.4 · **4×10 m route path + TBT 88.2 (+2.3)** |
| ibid. Table IV | NAVSIM, TransFuser | command perturbed to None / Random / Left / Right / Forward | **PDMS 84.0–84.7 — flat.** Paper's words: *"removal of navigation information paradoxically yields superior results"* |
| [GoalFlow, CVPR 2025](https://arxiv.org/html/2503.05689v1) Table 2 | NAVSIM, no CI | **ℳ₀ = no goal point, 85.6 PDMS** | ℳ₁ +distance-score goal **88.5 (+2.9)** · ℳ₃ full **90.3 (+4.7)** |

Class: **PUBLISHED**, with the §2 caveats — non-reactive benchmark, no intervals, single splits. The two
papers are independent and agree on both halves (categorical ≈ 0, geometric ≈ +2.3 to +4.7 PDMS), which
is worth more than either alone.

⭐ **Our K5 is a replication, not an anomaly.** The published categorical-command null (Table IV: flat
under *randomised* commands) is the same phenomenon as our oracle-route result (K5: +0.0024 not
separated, cross-track and curvature separably worse). We should stop treating REF-C's route behaviour as
a TanitAD bug to be found and start treating it as a known interface defect with a published fix.

⚠️ **Two adaptations are mandatory, and they are not cosmetic.**

1. **A supplied route is not evaluable on our corpus, and may not be admissible at inference.** LAN's
   only working supplier on PhysicalAI is **S1 `ego_future`** — the ego's own future path — because the
   corpus has no map (settled at five probes, `CLAUDE.md`). Any eval of a *supplied* route is therefore
   optimistic by construction. Independently, the 2026-08-03 binding rule states inference is
   **vision-only, no privileged channel**. ⇒ **The deployable form is GoalFlow's *predicted* goal point —
   a two-stage factorisation of the model's own output — not a route input.** That is also the form the
   published +4.7 measures (ℳ₃ predicts the goal from BEV features; the ground-truth-endpoint variant is
   a separate 92.1 row against a 94.8 human score).
2. **Only the distance-score half is buildable.** GoalFlow's scorer is
   `w₁ log δ_dis + w₂ log δ_dac`; **δ_dac needs a drivable-area polygon we do not have.** δ_dis needs only
   the GT future endpoint, which is a label we do have. The buildable subset is therefore **ℳ₁ = +2.9
   PDMS**, not ℳ₃ = +4.7. Quote the number we can actually reach.

⚠️ **WHY OUR OWN FINDINGS PREDICT THIS TO UNDERDELIVER ON ITS OWN.** K6 measured that the oracle-goal
advantage is **+83.7 % along-track / +2.9 % cross-track (not separated)** — i.e. a 2-D goal point is
valuable almost entirely because of its *along-track* coordinate. GoalFlow's oracle row (92.1) is
therefore **our K6 finding in NAVSIM units, not new information**. A *predicted* goal point must predict
along-track distance at 2 s from the same latents in which K7 found `long_accel` unrecoverable and in
which a 129-parameter probe was optimal. **If the along-track quantity is not in the representation, a
goal-point head cannot invent it, and item 2 degenerates into a re-parameterisation of the incumbent —
which K1 already priced at ≈ 0.** This is exactly the class the brief asked me to flag: *the literature is
enthusiastic and our own evidence says the mechanism is upstream.* **Item 2's payoff is conditional on
item 1 landing.**

**Cost.** Endpoint vocabulary by k-means over the parity corpus (offline, minutes, no episode
re-selection — parity-safe). Scorer decoder ≈ a small transformer head over existing tokens. Sinusoidal
goal encoding into `cond` ≈ 15 lines. One 30 k arm.

**Cheapest discriminating experiment (0 GPU first).** Before any training: run the **item-1 probe on the
goal-point target itself** — can a tiny ridge predict the 2 s along-track endpoint from REF-C's `fmap`
above a shuffled-latent control, and does the W-frame stack beat the single frame? That single probe
prices item 2 *and* item 1 at once.

**Pre-registered falsifier.**
- **REFUTE:** if the along-track endpoint is not predictable above the shuffled control from either
  single-frame or multi-frame latents, item 2 is **dropped** — the goal-point decomposition has nothing to
  condition on, and we say so publicly rather than shipping a null arm.
- **CONFIRM:** train ℳ₁-equivalent. Gate on the four families; the primary read is **LONGITUDINAL**
  (target-speed + headway/TTC), because that is where K6/K8 say the goal point must act. A win that shows
  up only in ADE is **not** accepted as confirmation.

---

### 3 — Pretrain the vision trunk. Do NOT scale it.

**Status vs our prior: PREDICTED TO SUCCEED (modestly).** Representation change, not head change.

**Mechanism.** REF-C's trunk is a torchvision-free ResNet-34-style CNN with **no pretrained weights**,
learning general visual structure from 2376 episodes simultaneously with learning to plan. Every planner
in the cited literature initialises from pretrained weights — DiffusionDrive uses **ImageNet-pretrained
ResNet-34** (PUBLISHED, Appendix A); V2-99 backbones are depth-pretrained.

⛔ **The size lever is dead, and I am recommending against it explicitly.** MEASURED (K3): our own ladder
shows a 48 M encoder proposes at least as tightly per-anchor as 90 M, with the registry verdict *"the
encoder is over-provisioned even at base."* PUBLISHED (weak, cross-paper and therefore **confounded** —
different methods, not a controlled swap): DiffusionDriveV2 and SparseDriveV2 with ResNet-34 (21.8 M)
outscore GoalFlow and Hydra-MDP with V2-99 (96.9 M). **Two nulls agree, so do not spend a GPU-day making
the trunk bigger.**

⭐ **But both nulls vary SIZE, and neither varies PRETRAINING.** That is the open axis, and REF-C is the
only model in the comparison sitting on the wrong side of it. Class: **HYPOTHESIS** built on a PUBLISHED
universal practice plus a MEASURED gap in what has been tested.

**Cost.** Low, and unusually well-controlled: **REF-C-small already exists as a matched from-scratch
control** — `refc-diffusion-small-v21-30k`, 54.7 M, 30 k steps, **7 h 10 m on an A40** (MEASURED), same
`--labels v21`, anchors bit-exactly nested. Swapping the stem/stages for an ImageNet ResNet-34 at the same
config gives a clean single-variable A/B. Requires reconciling channel widths (our `in_channels` is 9, not
3 — inflate the pretrained stem kernel across channel groups; this is a known, ~10-line operation, and it
must be documented as the one impurity in the control).

**Cheapest discriminating experiment.** One REF-C-**small** arm, ImageNet-initialised trunk, everything
else held. Compare paired against the banked from-scratch small arm (**ADE@2s 0.5261 [0.4295, 0.6262]**),
same 881 windows.

**Pre-registered falsifier.**
- **REFUTE:** if pretrained-small does not beat from-scratch-small on the paired episode-cluster bootstrap
  in **either** the LATERAL family (heading, curvature, yaw-rate, cross-track) **or** ADE@2s, pretraining
  is not our lever and the trunk question is closed for Phase 0. Per §2, ADE alone failing is *not*
  sufficient to refute — the lateral family is the sensitive read.
- **CONFIRM:** promote to base and re-run the panel.

---

### 4 — Supervise the confidence head at denoise timesteps (DiffusionDrive parity)

**⚠️ Status vs our prior: PREDICTED TO UNDERDELIVER (K1/K2).** Listed because it is nearly free and it
closes a documented deviation from the published recipe — **not** because I expect it to move the panel.

**Mechanism.** PUBLISHED: DiffusionDrive's decoder emits classification scores **at each denoising step**
and selects top-1 at the final step. MEASURED (ours): `refc_train` never supervises the conf head at
denoise timesteps, and consequently selecting on the discarded refined-pass confidence scores **1.36593 —
2.9× WORSE than baseline**, because that signal is unsupervised noise (`MODEL_REGISTRY.md` §4.1). So the
published architecture and ours differ in *training*, not just in which score we read.

**Why I still rank it 4th.** K1 caps *selection-side* recovery at ≤ 8.4 % of a 0.3075 m gap ≈ **0.026 m**,
measured across 47 arms, and K2 says most of that gap is aleatoric. The one honest caveat in our favour:
K1 was established with a **frozen decoder + learned re-scorer**, whereas supervising the conf head during
training also changes the decoder's learned representation — so the 47-arm bound is not exactly binding.
That is a real distinction, and it is still not enough to fund this ahead of items 1–3.

**Cost.** Very low — a loss term in `refc_train` plus one 30 k arm.

**Cheapest discriminating experiment.** Add per-timestep conf supervision to one REF-C-small arm; read
`frac_sel_2x_worse` (currently **0.3825** at small) and `sel_gap` alongside the four families.

**Pre-registered falsifier.**
- **REFUTE:** if paired ADE@2s improvement is < 0.026 m (the K1 ceiling) **or** its CI includes 0, this is
  recorded as *published-parity restored, no measurable benefit*, and the selection question is closed
  permanently with a third independent result.
- **CONFIRM:** improvement beyond the K1 ceiling would mean the 47-arm bound does not transfer to
  training-time supervision — which is itself a finding worth banking and would reopen the ranker.

---

### 5 — Decide the inference-time determinism deliberately (currently an undocumented deviation)

**⚠️ Status vs our prior: PREDICTED TO FAIL, possibly to HARM.** Ranked 5th because the experiment costs
**no training at all** and it converts an accidental-looking deviation into a documented decision.

**Mechanism.** MEASURED: `noise = torch.randn_like(x) * noise_std if self.training else
torch.zeros_like(x)` — at eval REF-C adds no noise and is a **deterministic 2-pass residual refiner**, not
a sampler. PUBLISHED: DiffusionDrive samples from an anchored Gaussian at inference with **N_infer = 20**
(Table 6 sweeps 10/20/40). We are therefore running "truncated diffusion" with the diffusion switched off
at test time.

**Why I expect it to fail anyway.** We already denoise **all** 128/256 anchors, so our fan is 6–13× wider
than DiffusionDrive's 20 samples — we do not lack candidate diversity. Added noise would produce
*within-anchor* diversity that must then be **ranked**, and ranking is our measured weak axis (K4:
`frac_sel_2x_worse` rises 0.3825 → 0.4109 → 0.4540 with anchor count; K1: ranking is capped). **More
candidates into a capped ranker is the one intervention our data most clearly predicts to be neutral or
harmful.**

**Cost.** ~0 — an eval-only sweep of `noise_std` at inference on the banked val cache. Minutes.

**Cheapest discriminating experiment.** Sweep inference `noise_std ∈ {0, 0.02, 0.05, 0.1}` (train-time
value is 0.1) on the canonical 881 windows; paired episode-cluster bootstrap vs the deterministic
incumbent. Report the four families.

**Pre-registered falsifier.**
- **REFUTE (expected):** no setting improves the paired ADE@2s **and** the lateral family ⇒ record the
  determinism as a **deliberate, measured design decision**, add the one-line justification to `refc.py`,
  and stop the question recurring.
- **CONFIRM:** if any noise_std > 0 separably improves, the determinism is a real defect — and, more
  interestingly, it would contradict K1/K4 and demand that the ranking cap be re-derived.

---

### 6 — Trajectory-pattern loss reweighting for the straight-driving imbalance

**Status vs our prior: NEUTRAL.** Weakest evidence in the shortlist; included because the imbalance is
severe and MEASURED on our own corpus.

**Mechanism.** Common manoeuvres dominate; rare ones are sparse; the planner fits the mode. PUBLISHED
([FlowDrive](https://arxiv.org/pdf/2509.21961)): reweighting **by trajectory pattern** yields their largest
single improvement — **and reweighting by scenario type performs *worse* than no weighting at all.**
MEASURED (ours): the corpus is ~74 % straight and v21 label coverage is
**[0.121 / 0.5645 / 0.115 / UNKNOWN 0.1995]**.

**Why the caveat matters more than the headline.** That FlowDrive's own comparison shows one grouping
choice helping and another actively hurting means the effect is **fragile to the choice of pattern**, not
a general law. One paper, no CI, and an internally inconsistent direction ⇒ this is a **try-cheaply**, not
a **fund-confidently**.

⚠️ **Parity guard.** This must be implemented as a **per-window loss weight**, never as a sampler that
re-selects or drops episodes. `CLAUDE.md`: *"Anything that re-selects episodes breaks cross-arm
comparability and must be refused."* Loss reweighting inside the fixed 2376-episode / skip-hash `f09e44db`
set is parity-safe; a rebalanced sampler is not.

**Cost.** Low — a weight vector in the trainer's loss, one 30 k arm.

**Cheapest discriminating experiment.** One REF-C-small arm with inverse-frequency weights over the
trajectory-pattern clusters (reuse item 2's endpoint k-means, so the two items share one offline artifact).

**Pre-registered falsifier.**
- **REFUTE:** if the **gentle** and **sharp** curvature strata (small: 0.813 / 0.848) do not improve on the
  paired bootstrap, or the **straight** stratum (0.408) regresses separably, drop it.
- **CONFIRM:** improvement concentrated in the rare strata with straight held ⇒ promote.

---

## 4. What I deliberately do NOT recommend, and why

A shortlist is only defensible if the exclusions are argued.

| considered | why refused | class |
|---|---|---|
| **Bigger / stronger vision backbone** (V2-99-class, larger ViT) | **Two independent nulls.** MEASURED K3 (our own ladder, separated): the small encoder proposes at least as tightly per-anchor; registry verdict *"over-provisioned even at base."* PUBLISHED (confounded, cross-paper): ResNet-34 methods outscore V2-99 methods on NAVSIM. Item 3 keeps the *pretraining* axis and explicitly discards the *size* axis. | MEASURED + PUBLISHED |
| **More anchors / wider vocabulary** | MEASURED K4: raising 64→128→256 improves oracle-in-fan (0.2213→0.1914→0.1640) but worsens ranking (`frac_sel_2x_worse` 0.3825→0.4109→0.4540); base vs XL selected ADE@2s **+0.0013, not separated**. The two effects cancel. | MEASURED |
| **Fewer anchors, to match DiffusionDrive's 20** | Symmetrically refused: K4 says our fan *degrades* as the vocabulary narrows. The published recipe's 20 anchors is **not** transferable evidence — it was tuned with a scorer supervised at every denoise step (item 4), which we do not have. | MEASURED |
| **Another learned re-scorer / selection head** | **Settled across 47 arms** (K1): ≤ 8.4 % of the gap, v1.2 got +2.9 % not significant, hand-written cost 0.0 %, GT-perfect speed-matcher *worse* than baseline. Do not re-open. | MEASURED |
| **RL fine-tuning of the planner (DiffusionDriveV2 / PlannerRFT)** | PUBLISHED gains are real but modest per component (+0.9 intra-anchor GRPO, +0.6 inter-anchor, +0.2 selector, +0.2 ranking loss) and the whole method depends on a **collision/progress reward from a simulator inside the training loop**. Our only such simulator is AlpaSim at **~3.2× reconstruction-OOD** (RETRACTION_LOG C6), so the reward would be measured on a distribution we have already retracted numbers from. Revisit when the closed-loop renderer question is settled — not now. | PUBLISHED |
| **Replace truncated diffusion with flow matching** (GoalFlow / FlowDrive / GuideFlow) | The published case rests on leaderboard position, and where the ablation *is* isolated (GoalFlow Table 2) the gain is attributed to the **goal point**, not the flow parameterisation — ℳ₀ (flow, no goal) 85.6 vs TransFuser 84.0 is +1.6, versus +4.7 for the goal mechanism. **Item 2 takes the part with the ablation behind it and leaves the rewrite.** | PUBLISHED |
| **Classifier-free guidance on the route** | Attractive and it is the textbook answer to "make a condition matter". Refused *for now* on our data: CFG amplifies a conditional-vs-unconditional difference, and K5 measured our conditional direction to be **anti-compliant** (oracle route makes cross-track and curvature separably worse). **Amplifying a wrong direction makes it wronger.** CFG becomes admissible only after item 2 replaces the categorical condition with a geometric one — at which point it is a one-line inference change worth re-testing. | MEASURED + reasoning |

---

## 5. Sequencing — what this costs to actually do

The shortlist collapses to **one 0-GPU probe that prices the top two items**, then a small number of
REF-C-small arms at ~7 h each.

1. **Now, 0 GPU:** the ridge probe of §3.1 — longitudinal targets (closing rate, time-gap/TTC,
   `long_accel`) **and** the 2 s along-track endpoint, from single-frame vs W-frame `fmap`, each against a
   shuffled-latent control, tiny capacity per K7. **This single probe decides items 1 and 2.**
2. **Now, 0 GPU:** the item-5 inference-noise sweep (eval-only, minutes) — closes a standing deviation.
3. **Then, ~7 h each on one A40**, in this order, contingent on the probe: item 1 (temporal KV) → item 3
   (pretrained trunk, the cleanest control we have) → item 2 (goal point, only if the probe confirmed) →
   items 4 and 6 as cheap riders.

⚠️ **Every arm above reports all four metric families with the paired episode-cluster bootstrap on the
same 881 windows.** Per §2, ADE is structurally unable to see most of what these changes do, and per
`CLAUDE.md` an ADE-only table is an incomplete result.

⚠️ **One data hygiene item for the sibling streams:** `ep_00028.pt` on Thor is a **truncated transfer**
(92,299,264 B vs 117,383,256 B), so the LAN E0 run sits on **859 windows / 39 episodes**, not the 881 / 40
the published REF-C rows use. Re-pulling that file is a prerequisite for any window-for-window comparison
against the registry.

---

## 6. Escalations (not "please merge" notes)

1. **PI decision required — is a route/goal input admissible at inference at all?** The 2026-08-03 binding
   rule says inference is vision-only, no privileged channel. A navigation route is genuinely available to
   a deployed car but **not** to us on PhysicalAI, where LAN's only supplier is the ego's own future path.
   My recommendation is item 2's *predicted* goal point, which needs no route at inference and so sidesteps
   the rule entirely — but the PI should rule on whether supplied-route arms are worth running at all.
2. **`graft_lan` is implemented and has never been trained.** `LanConfig`, `lan_to_cond`, `lan_gate`, the
   param-free bearing compatibility and `stack/tanitad/data/lan.py` all exist at HEAD with
   `graft_lan=False` by default, and LAN E0 explicitly launched no training. This is the "implemented,
   unmerged 12 days" failure class in a new costume — either an arm is scheduled or the seam is documented
   as parked, with a date.
3. **The single-frame KV bottleneck (§1) is not recorded in `MODEL_REGISTRY.md` §4.** The registry
   documents the selection flaw and the anchor ladder but not this. The CODE stream owns `refc.py`; this
   note is the handoff.

---

## 7. Sources

Ours (MEASURED): `stack/tanitad/refs/refc.py`, `stack/tanitad/data/lan.py`,
`Project Steering/MODEL_REGISTRY.md` §4.1–4.3,
`…/Implementation/incoming/2026-08-03-lan-refc-e0/LAN_E0_RESULTS.md`, `GOAL_INPUT.md` (2026-07-27),
`Project Steering/RETRACTION_LOG.md` (C5/C6).

Published:
[DiffusionDrive (CVPR 2025)](https://arxiv.org/html/2411.15139) ·
[DiffusionDriveV2](https://arxiv.org/html/2512.07745) ·
[GoalFlow (CVPR 2025)](https://arxiv.org/html/2503.05689v1) ·
[Unveiling the Surprising Efficacy of Navigation Understanding in E2E AD](https://arxiv.org/html/2604.12208) ·
[FlowDrive](https://arxiv.org/pdf/2509.21961) ·
[NAVSIM](https://arxiv.org/pdf/2406.15349) ·
[Is Ego Status All You Need? (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Is_Ego_Status_All_You_Need_for_Open-Loop_End-to-End_Autonomous_Driving_CVPR_2024_paper.pdf) ·
[AnchDrive](https://arxiv.org/pdf/2509.20253) ·
[GuideFlow](https://arxiv.org/html/2511.18729v1)
