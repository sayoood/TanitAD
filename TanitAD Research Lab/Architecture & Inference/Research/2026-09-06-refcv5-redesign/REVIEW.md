# REVIEW — what refcv5 ACTUALLY includes, and what it does not

**Stream:** Architecture & Inference · **2026-09-06** · branch `agent/arch-inf-20260803`
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv5-redesign/`
**Ordered by:** the PI, who stopped `refcv5-ddim-b1-v72-40k` mid-training.

> ⛔⛔ **EVIDENCE CLASS AND TIER, STATED ONCE.** Every "in refcv5 / not in refcv5" verdict
> below is **MEASURED (ours)** and read from **`run_config.json`** — the config the trainer
> itself stamped at launch — banked at
> `…/2026-09-06-refcv5-launch/raw/run_config.json`. **Not from memory, not from the brief,
> not from any prose summary.** ⛔ **NO EVAL TIER APPLIES TO THIS DOCUMENT.** It contains
> **no capability number of refcv5's**, because **refcv5 produced none** — it was stopped
> before its first landing eval. Every *comparison* number quoted (refcv4b's arms, the
> effect sizes in Table B) carries its own tier and artifact inline and is
> **INHERITED unless marked MEASURED-HERE**.

---

## 0. The one-line answer to the PI's question

**refcv5 is refcv4b's argv byte-for-byte, plus `--sampler ddim --w-u0 0.5`, plus an
explicit `--agents off`.** MEASURED: the recorded `argv` is **62 tokens**, and the only
tokens refcv4b did not carry are those three flags.

Against the programme's own **43-item DiffusionDrive audit** (`D-REFC-DDAUDIT-1…6`,
2026-09-05 — *the audit the PI is referring to when he says "you identified them"*),
refcv5 turns on the **sampler row and nothing else**:

| block of the DD design | items | in refcv5? |
|---|---|---|
| Truncated/anchored diffusion (the noise schedule, the anchored prior, the DDIM update, sampling at inference) | #1, #2, #5, #6, #9, #11, #15 | ⭐ **IN** — this is the one lever |
| The **three grounded attentions** (waypoint-indexed spatial · agent-query · ego-query) | #19, #20, #21 | ⛔ **0 of 3** |
| Scoring the **emitted** fan instead of the `t=0` fan | #28, #31 | ⛔ **OUT** |
| Per-layer AdaLN timestep modulation, per-layer loss | #7, #8 | ⛔ **OUT** |
| V2 (GRPO, truncated advantage, scale-adaptive noise, PDM selector) | #37–#42 | ⛔ **OUT** |
| TransFuser / LiDAR / metric BEV | #32, #33 | ⛔ **OUT — by the vision-only programme rule** |

And against the programme's own **ranked list of validated refcv4b improvements**
(`D-REFCV5-LEVERS`, six items ranked *by measured effect size*), refcv5 includes
**none of the top five**. The lever it does carry is not on that list at all.

⇒ **The PI's reading is correct.** refcv5 is a single-mechanism ablation of the DD
sampler. It is not the capability the redesign brief asked for.

### 0.1 ⭐⭐ AND DD-v1'S OWN ABLATIONS RANK THE ELEMENTS — THE SAMPLER IS THE CHEAP HALF

**PUBLISHED (2411.15139v3, Tab. 3 and Tab. 4), read from the banked PDF:**

| DD ablation | PDMS | what it says |
|---|---|---|
| full DiffusionDrive | **88.1** | — |
| ⛔ **spatial cross-attention REMOVED** | **55.1** | ⭐⭐ **−33.0 PDMS. The waypoint-indexed BEV grid-sample is the single load-bearing component of the whole decoder.** |
| **1 denoising step** instead of 2 | **87.9** | ⚠️ **−0.2 PDMS. The sampler's second step is worth almost nothing.** |
| 3 denoising steps | 88.1 | saturated |

⛔⛔ **This is the finding that settles the redesign, and it is PUBLISHED, not ours.**
The element refcv5 spent **~45 h of A40** on is the one DD's own ablation prices at
**≈0.2 PDMS**; the elements refcv5 left out are the ones DD prices at **33 PDMS**.
⚠️ Read honestly: these are *DD's* ablations on *NAVSIM with LiDAR*, so the magnitudes do
not transfer to our vision-only surface. **The RANKING is what transfers**, and it agrees
with our own independently-measured ranking (`D-REFCV5-LEVERS`, `D-REFCV5-PLAN-10`) —
which is why two independent sources pointing the same way is worth more than either.

⚠️ **A correction to the brief that ordered this review.** It stated *"DD-v1 must be
banked too"*. **It was already banked, on 2026-08-21** — library key **`2411.15139`**,
sha256 `6ad4f8a379494eeb099cf7d489b75bec1c0ec3321428bcd0b0ca0f05662b8600`, 26,370,647 B,
`%PDF-1.5` header verified, `--verify` clean at **487 entries, 0 orphans, 0 problems**. The
entry now also carries the `diffusion-planner` tag and a citation to this document.

---

## 1. PROVENANCE — how each verdict was established

⛔ **"In refcv5" is a claim about a run, and the only admissible source is the run's own
record.** Three artifacts, all banked:

| artifact | what it settles |
|---|---|
| `raw/run_config.json` → `argv` (62 tokens) | which flags were **passed** |
| `raw/run_config.json` → `seams` (46 keys) | which mechanisms were **effective**, with their weights |
| `raw/run_config.json` → `anchors`, `ego`, `selection`, `goal_provenance`, `agent_join*` | the vocabulary, the ego block, the selection band, the goal path, the agent seam |

⚠️ **The `seams` block is the load-bearing one, not `argv`** — because this trainer has
**five zero-weight defaults** whose terms are *skipped entirely* rather than merely scaled
to zero. A flag absent from `argv` and a flag present with weight `0.0` are the same thing
here, and only `seams` distinguishes "off" from "on but inert".

**The recorded `argv`, verbatim:**

```
--arm hier --size base --v2-cache /root/data/train
--v7-labels …/s2_labels_v7.2_train.jsonl.gz --eval-cache /root/data/eval
--eval-labels …/s2_labels_v7.2_eval.jsonl.gz --eval-every 500 --eval-batches 8
--image-hw 256 640 --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
--lr 1e-4 --warmup 2000 --seed 0 --log-every 50 --save-every 500
--nav-from-v7 --u8-batches
--anchors …/anchors.pt --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat
--sel-accel-max 2.0 --goal-str --ego-state-inject --ego-dropout 0.5
--sampler ddim --w-u0 0.5 --agents off --out …
```

---

## 2. ⛔ TABLE A — every element the diffusion papers specify × refcv5

Source for the "DD specifies" column: `D-REFC-DDAUDIT-1…6` and its RESULT
(`…/2026-09-05-refc-vs-diffusiondrive-audit/RESULT.md`, 43 numbered rows), read from
**arXiv 2411.15139v3 + `hustvl/DiffusionDrive@9b52ed0`** and **arXiv 2512.07745v1 +
`hustvl/DiffusionDriveV2@1cd12a1`**. Evidence class **PUBLISHED (cited)** for the DD
column, **MEASURED** for the refcv5 column.

### 2.1 The truncated diffusion policy — ⭐ THE ONE THING refcv5 DID INCLUDE

| # | DD v1 specifies | in refcv5? | stamped value / why |
|---|---|---|---|
| 1 | Anchored Gaussian prior `x_t = √ᾱ·a_k + √(1−ᾱ)·ε` | ⭐ **IN** | `sampler: ddim` |
| 2a | Truncation at inference: from **t=8** | ⭐ **IN** | `sampler_infer_t: 8` |
| 2b | Truncation at **train**: `t ~ U[0,50)` of T=1000 | ⛔ **NOT IN — the flag is INERT** | ⛔ MEASURED this turn: **`--sampler-train-t-max` has NO CONSUMER.** `_sample` has no `self.training` branch and uses `sampler_infer_t` at train **and** eval. refcv5 stamped `sampler_train_t_max: 50` and **trained at t=8**. |
| 3 | 0 denoising iterations at train (one noised sample, one call) | ⚠️ **DIVERGED** — we iterate at train | pre-existing, unchanged |
| 4 | **2** DDIM steps at inference | ⭐ **IN** | `sampler_steps: 2` |
| 5 | `DDIMScheduler(1000, scaled_linear, prediction_type="sample")` | ⭐ **IN (re-implemented)** | ⚠️ see §2.5 — the diffusers **pin never certified it** (`D-REFCV5-DIFFUSERS-PIN-CANNOT-PASS`) |
| 6 | x0 ("sample") prediction target | ⭐ **IN** | `w_u0: 0.5` — the x0 term, measured live (`u0` 0.27576 / 0.18663 / 0.32519 at steps 50/100/150) |
| 7 | Timestep injected **per layer** as **AdaLN** scale/shift | ⛔ **OUT** | ours adds it **once**, as a bias on the input query |
| 8 | Loss per **cascade layer**, layer-to-layer coords **detached** | ⛔ **OUT** | ours: CE on the `t=0` confidence, L1 on the final fan |
| 9 | DDIM sampler, η=0 | ⭐ **IN** | `sampler: ddim` |
| 11 | Coordinate normalisation to [−1,1] | ⭐ **IN, in CONTROL space** | `sampler_space: control`, `control_norm: [4.0, 3.0]` ⇒ ⚠️ **a deliberate deviation from DD, which normalises WAYPOINTS** |
| 15 | Noise creates diversity: each anchor independently noised σ≈0.9/0.7 m ⇒ a **sampled** fan | ⭐ **IN** | the arm is now **stochastic at inference** — which is why an **inference-seed replicate is mandatory** |
| 16 | Mode-collapse prevention by truncation | ⭐ **IN** (follows from #2/#4) | — |
| 13/14 | 20 anchors, k-means over GT waypoints, fixed in metres | ⚠️ **BEYOND DD** | 117 anchors, `(a_lon, a_lat)` 13×9 kinematic grid, **v0-conditioned**, units `alat` |

⇒ **The sampler block is genuinely IN, and it is the DD element with the clearest
mechanism.** ⛔ But it is also the *only* one; #7 and #8 — two of the four things that make
DD's cascade a cascade — are still out; **and #2's TRAINING half is stamped but inert.**

⛔⛔ **TWO SAMPLER FLAGS ARE DEAD, MEASURED THIS TURN — and one of them is a DD-fidelity
defect the launch record does not disclose.** `--sampler-train-t-max` **is read by nothing**
(whole-tree byte probe, readability gap closed to zero: the dataclass field, the assignment,
the argparse line — and no consumer), so DD's `t ~ U[0, 50)` training draw **is not
implemented**: refcv5 trained and inferred at the **same** `t = 8`, while DD deliberately
trains across `t ∈ [0, 49]`. And `--sampler-steps` is **shadowed** by the forward's own
`steps` argument, reachable only when `steps == 0`. ⇒ **refcv5's `config.json` stamps
`sampler_train_t_max: 50` for a mechanism that never ran** — which is the M18 dead-flag
defect, in the very arm whose launch record celebrated passing the M18 check on `w_u0`.

### 2.2 ⛔ The three grounded attentions — **0 of 3**, and this is the PI's specific question

The PI asks about the **environment link**. DD conditions on scene *and agent* tokens.
MEASURED: refcv5 has **neither**.

| # | DD v1 specifies | in refcv5? | what refcv5 has instead |
|---|---|---|---|
| 19 | **Waypoint-indexed spatial attention** — `GridSampleCrossBEVAttention` bilinearly samples the BEV value map **at the 8 waypoints of each noisy trajectory**, re-sampled per cascade layer | ⛔ **OUT** | content-based MHA from anchor queries to **8×20 = 160 perspective-view tokens** of the last frame. **The attention never looks where the candidate goes.** |
| 20 | **Agent cross-attention** to 30 detection queries | ⛔ **OUT — entirely** | `agents: null` · `w_agent: 0.0` · `cross_agent: false` · `agent_join: null`. ⛔ **`AGENT_WEIGHT_DEFAULT = 0.0` means the term is SKIPPED, not scaled — no graph is built, no tokens are embedded.** |
| 21 | **Ego/status cross-attention** to a planning query carrying command + velocity + acceleration | ⚠️ **PARTIAL** | a **FiLM** on the MLP input; `ego_state_inject: true` with channels `[v0, a_long, yaw_rate, curvature, keep]` at t0, `ego_dropout 0.5`. Conditioning exists; **the mechanism is not attention.** |
| 25 | Per-layer AdaLN timestep modulation | ⛔ **OUT** | additive bias on the input query only |
| 26 | Map cross-attention | **N/A** | no map exists in PhysicalAI-AV (settled at five probes: no map, lane graph, junction, roundabout, traffic-light feature or route signal) |

⭐ **The precise answer to "what does refcv5 have instead of scene/agent tokens":**
it has **ungrounded perspective-view conv tokens under a FiLM condition**. Of DD's three
grounded attentions — *at the candidate's own waypoints in a metric frame*, *on detected
agents*, *on the ego planning query* — refcv5 implements **zero**.

### 2.3 ⛔ Selection — the object that is SCORED is not the object that is EMITTED

| # | DD specifies | in refcv5? | evidence |
|---|---|---|---|
| 28 | The **last** cascade layer scores the fan it emits: `argmax(poses_cls[-1])` on `poses_reg[-1]` | ⛔ **OUT** | ours ranks on the **`t=0`** confidence. MEASURED: `sel_idx_base` identical to the `steps=0` winner on **201/201** windows for every `steps ∈ {0,1,2,3,4,8}` |
| 27 | Focal BCE (γ=2, α=0.25), cls 10 / reg 8 | ⚠️ **DIVERGED** | softmax CE over N on the `t=0` pass, traj 1.0 / cls 1.0 |
| 31 | V2's two-stage sub-metric selector (coarse 1-layer → top-32 → fine 3-layer), 5 PDM heads, BCE + MarginRanking(0.05) | ⛔ **OUT** | nothing equivalent |

⚠️ **AND THE FIX IS NOT A FLAG.** `D-REFCV4B-SELREF-1`: enabling `--ablate sel_refined`
(rank the refined fan by the refined confidence, **0 params**) is **separated WORSE** —
`os` 0.3055 → **0.3314 m**, paired **−0.0259 [−0.0505, −0.0033], p 0.017** — flipping
**51/171 = 29.82 %** of picks and collapsing the fan 18 → 16 distinct anchors, because
`sel_refined` is `seen_in_training: no` and no training objective ever scores the refined
estimate. ⇒ **The load-bearing part is selector TRAINING (WP-7), not the wiring.**

⛔⛔ **AND THE SAMPLER MADE THIS DEFECT WORSE, WHICH IS THE SHARPEST SINGLE FINDING IN THIS
REVIEW.** MEASURED: `refc.py` stamps `"sampler_ranks_the_fan": bool(self.sel.refined)` —
and `sel_refined` defaults **False** and **has no CLI flag in the refcv5 trainer** (it is
registered in the *legacy* `refc_train.py`, and implemented in `RefCConfig`, so *the refcv5
trainer cannot arm a lever its own core implements*). ⇒ **refcv5 spent ~45 h adding
stochastic sampling to a fan, and then ranked that fan with a score computed before any
sample was drawn.** The sampler's entire product is diversity; the selector is structurally
blind to it. `refc.py` prices ranking with an unrefined score at **>2× worse than the
fan-best pick on 41.09 % of windows (base)** / **45.4 % (REF-C-XL)**.
⚠️ ⛔ **This is not an argument for flipping the flag** — arming an *untrained* readout is
the measured regression above. It is an argument that **DD's per-layer deep supervision
(#8) and scoring-the-emitted-fan (#28) are ONE work item**, and that item was the natural
companion to the sampler and was not done.

### 2.4 ⛔ V2's additions — all missing, and refcv5 built exactly their PREREQUISITE

| # | V2 specifies | in refcv5? |
|---|---|---|
| 37 | Denoising policy **with a log-probability** (`DDIMScheduler_with_logprob`, `step_num=10` at train, 2 at inference, η 1/0) | ⚠️ **PREREQUISITE NOW EXISTS** — refcv5's sampler is the first REF-C decoder that is stochastic at inference. The **log-prob** itself: ⛔ OUT |
| 38 | Scale-adaptive multiplicative noise, 2 scalars per trajectory | ⛔ **OUT** |
| 39 | Intra-anchor GRPO, G=4 | ⛔ **OUT of the arm**; the *estimator* exists in `stack/tanitad/rl/advantage.py` (sibling-owned) |
| 40 | Inter-anchor truncated advantage, collision → −1, `clamp(min=0)·(r > r_GT)` | ⛔ **OUT of the arm**; estimator present |
| 41 | Reward = NAVSIM PDM sub-scores | ⛔ **OUT** — no PDM simulator on PhysicalAI; rule-based proxies only |
| 42 | Mode selector | ⛔ **OUT** |

### 2.5 ⚠️ TWO HONEST QUALIFICATIONS ON THE ONE LEVER refcv5 DID CARRY

1. ⛔ **The schedule math was never independently certified.**
   `D-REFCV5-DIFFUSERS-PIN-CANNOT-PASS`: `refc_sampler.assert_matches_diffusers` exists
   *"because a re-implementation that is never checked is just an unverified copy"* — and
   it **has never certified anything**. The sampler's soundest-looking half rests on an
   uncalled check.
2. ⚠️ **The sampler runs in CONTROL space, DD's runs in WAYPOINT space** (`sampler_space:
   control`, `control_norm: [4.0, 3.0]`). This is defensible — our vocabulary *is*
   kinematic — but it is **a deviation from the published mechanism**, and it must be
   stated whenever refcv5's sampler is called "DD-faithful". `H-DDA-3` pre-registered the
   *diffusers* form; refcv5 ran a re-implementation of a different parameterisation.

---

## 2.6 ⭐ DD-v1'S MECHANISM, READ FROM SOURCE — the standard DD-v2's reward set

Banked key **`2411.15139`** · sha256
`6ad4f8a379494eeb099cf7d489b75bec1c0ec3321428bcd0b0ca0f05662b8600` ·
`TanitAD Research Lab/Library/papers/2411.15139_DiffusionDrive-*.pdf`.
Code read at **`hustvl/DiffusionDrive@9b52ed0`** and **`hustvl/DiffusionDriveV2@1cd12a1`**
(all 11 v1 files vendored in the v2 repo are **blob-identical**, so one reading covers
both). Evidence class **PUBLISHED (cited)** throughout; σ values marked DERIVED.

**(a) Anchors.** K-means over GT ego futures of NAVSIM navtrain. **20** anchors (18 for
nuScenes), **8 poses × 0.5 s = 4.0 s**, **raw waypoints in metres**, ego frame `(x fwd,
y lat)` — **no heading, no controls, no curvature**. `requires_grad=False`, **frozen**.
MEASURED from the shipped `.npy`: `(20, 8, 2) float64`, x ∈ [0.560, 47.658] m, y ∈
[−10.619, +12.378]; a **speed fan** (mean speed 0.80 → 11.91 m/s), 9 of 20 near-straight,
**no stationary anchor**.
⚠️ **DD-v1 has its own "undefined in the paper" hole**: the clustering script is not
released and the config points at an absolute path on an author's machine. A second probe
found the artifact — it ships in the **V2** repo root, not v1's.
⇒ ⭐ **Ours goes BEYOND DD here**: 117 anchors on a **(a_lon, a_lat) 13×9 kinematic grid**,
**v0-conditioned**, rolled per window. DD has no per-window vocabulary.

**(b) Truncated diffusion.** `DDIMScheduler(1000, "scaled_linear",
prediction_type="sample")` ⇒ the net predicts **x₀ directly**. Train `t ~ randint(0, 50)`
(the paper's "50/1000"); inference starts at **t = 8** — *hardcoded, and never stated in
the paper* — with **2** DDIM steps, `roll_timesteps = [10, 0]`. The starting distribution
is **anchors + reduced noise in normalised space, then `clamp(−1, 1)`**.
**DERIVED σ** at t=8: `√(1−ᾱ) = 0.0316` normalised ⇒ **≈0.90 m in x, 0.73 m in y** — i.e.
*the "diffusion" is a ≈1 m jitter around a frozen 20-point waypoint fan.*
⚠️ **Two quirks worth carrying:** the sample is noised to **t=8** but the first DDIM step
is taken **as if at t=10** (an off-by-two); and the **inference start is far cleaner than
the training upper end (t=49)** — train and inference are *not* matched.
⚠️ **Normalisation is load-bearing, not cosmetic**: with `clamp(−1,1)` it is a **hard box
constraint** at x ∈ [−1.2, 55.7] m, y ∈ [−20, 26] m (asymmetric in y).

**(c) Conditioning — the PI's question, answered from the layer body.** Per
`CustomTransformerDecoderLayer`, in order: **(1)** `GridSampleCrossBEVAttention` — BEV
features sampled at **the current trajectory's own 8 waypoints**; ⚠️ *the paper calls this
"deformable"; the released code is **not** — it is a plain `F.grid_sample` with a learned
softmax over the 8 points*. **(2)** `cross_agent_attention` = plain MHA to **30 agent
queries**. ⭐⭐ **Those queries come from the model's OWN auxiliary detector — not an
external detector or tracker** — a 3-layer `TransformerDecoder` over BEV+status supervised
by a **Hungarian-matched 3D box + class loss**. **(3)** `cross_ego_attention` to 1 ego
token. **(4)** FFN, then **`ModulationLayer`** — FiLM scale/shift from the sinusoidal
timestep embedding.
⚠️ **`status_encoding` (driving command 4 + velocity 2 + acceleration 2 = 8-d) reaches the
diffusion decoder only INDIRECTLY**, via `ego_query`/`agents_query`; it is a **dead
parameter in the layer body** (positively verified: every occurrence is signature or
plumbing, none inside the layer).
⇒ ⭐ **This vindicates `--agents head` over `--agents oracle` as the DD-faithful choice:
DD's agent tokens are a LEARNED detection head trained on GT boxes, exactly what `head`
is.** The oracle feeds labels at inference and is a ceiling, not a port of DD.

**(d) Losses.** Positive assignment is **NEAREST-GT to the RAW FROZEN ANCHOR** (mean L2
over the 8 poses, `argmin`) — ⭐ *not Hungarian, and independent of the model*.
**Classification: sigmoid FOCAL (γ=2.0, α=0.25)** over all 20 with a one-hot target —
⚠️ *the paper's Eq. 6 says BCE; the code uses focal.* **Regression: L1 on the MATCHED
anchor only**, over (x, y, heading). Weights: inner cls **10.0** / reg **8.0**, summed over
both cascade layers, × `trajectory_weight` **12.0** ⇒ effectively **120 × focal + 96 × L1
per layer**.
⛔⛔ **AND THERE IS NO ε-PREDICTION / DENOISING MSE TERM AT ALL.** `diff_loss_weight = 20.0`
is **dead code** — the model never emits `diffusion_loss`. The *only* trajectory
supervision is the matched-anchor L1 and the focal score. ⇒ ⭐ **our `--w-u0` (an x₀ loss
in control space) is BEYOND DD, not a port of it** — a fact refcv5's record does not state.

**(e) Cascade.** **2 layers**, ablated to 4 in Tab. 5 (88.2 vs 88.1 — saturated).
⛔ **Layers do NOT share weights** (`_get_clones` = `deepcopy`); what *is* shared is the
whole 2-layer stack **across the 2 denoising timesteps** — that is what the paper's
"parameters shared across denoising timesteps" means, and it is **not** a statement about
the cascade layers. ⭐ **Every layer re-scores**: each emits its own `(poses_reg,
poses_cls)`, **the loss is applied to every layer and summed — deep supervision** — and the
cascade is **gradient-detached between layers**.
⇒ ⛔ **This is DD element #8/#28 and REF-C has neither half**: we score the `t=0`
confidence once, and gradient flows through the whole path.

**(f) Reported numbers.** NAVSIM navtest, ResNet-34, camera+LiDAR: **PDMS 88.1** (NC 98.2 /
DAC 96.2 / TTC 94.7 / Comf 100 / EP 82.2) with 20 anchors, vs Transfuser **84.0** (+4.1),
VADv2-V8192 80.9 (+7.2 with **400× fewer anchors**), Hydra-MDP-V8192-W-EP 86.5 (+1.6).
**Cost: 2 steps × 3.8 ms = 7.6 ms planner, 60 M params, 45 FPS on a 4090** (vanilla 20-step
DDIM: 130 ms, 7 FPS). Mode-diversity **74 % vs 11 %**. Anchor transfer to CARLA Longest6:
DS **64.27** vs Transfuser 47.30 — *the anchors are not a leak.*

**(g) v1 → v2.** ⭐ **Architecture, anchors and truncation are UNCHANGED**; v2 cold-starts
from v1's weights and adds a **post-training stage**. The brief's three claims all
**CONFIRM verbatim** — reward = NAVSIM PDM (`PDMScorer`, *and genuinely undefined in the
paper*: 4 occurrences of "reward" in 906 lines); the gradient is
`-torch.exp(per_token_logps - per_token_logps.detach()) * advantages` ⇒ **ratio identically
1, no clip, no KL, no reference policy**; `std_dev_t_add = 0.0` in **both** branches ⇒ the
policy is **two multiplicative scalars per trajectory**.
⭐ **Three sharpenings the brief did not carry:** the truncation in code is **stricter than
the paper** — `mask_positive = reward_group > reward_gt`, so a sample must **beat the human
GT's PDM score**, not merely the group mean; the **−1 penalty covers drivable-area
violations**, not just collisions; and **λ is state-dependent** (1.0 when a scene has no
positive advantage, else 0.1), where Tab. 7 lists only "0.1". Hyperparameters: **G = 4**
per anchor (80 rollouts/scene), **γ = 0.8**, **10 denoising steps at RL rollout** (not 2).
⚠️ A real bug-shaped detail: the log-prob is evaluated under `clip(std_dev_t, min=0.1)`
while sampling uses `min=0.04` — **the density is evaluated under a wider Gaussian than the
one sampled from.**

---

## 3. ⛔ TABLE B — every validated refcv4b finding × refcv5

Source: `D-REFCV5-LEVERS` — the programme's own **ranked** list, ordered *by measured
effect size*, plus the claims each row cites. ⛔ **refcv5 includes NONE of the top five.**

| rank | validated finding | measured effect | in refcv5? | why not |
|---|---|---|---|---|
| **1** | ⭐ **Encode CLOSING RATE.** Lead **position** is decodable (**+0.4145 [+0.2018, +0.6120]** paired vs pixels; constant control **exactly +0.000000**); **closing rate is a clean null (+0.0061)** on every arm and a temporal difference recovers nothing. Corroborated 3 ways: `os − ha` speed MAE **+0.0368 separated worse**; LON κ **0.5178 < `ha` 0.6071**; **472 of 1,225** distance-keeping windows are actually closing | representation-scale | ⛔ **OUT** | **A REPRESENTATION requirement — no cost re-weighting substitutes.** Nothing in refcv5's argv touches the representation. |
| **2** | ⭐ **A predicted GEOMETRIC GOAL POINT** in place of the categorical route token (published +4.7 PDMS vs +0.2). Route-removal margin **0.1054 m separated** (`os_navzero − ha0_ext`); route head already predicts from vision at κ 0.4852 and passes its anti-echo control | see §4 F1 | ⛔ **OUT** | `--goal-point-inject` **not in argv**; `GOAL_POINT_WEIGHT_DEFAULT = 0.0` ⇒ the term is skipped entirely |
| **3** | ⭐ **TRAIN the selector (WP-7)** — and do **NOT** enable `--sel-refined` | `--sel-refined` **−0.0259 m separated WORSE**; ranking with an unrefined score costs **>2× worse-than-fan-best on 41.09 %** of base windows | ⛔ **OUT — and the sampler made it bite harder** | ⚠️ the *prohibition* is satisfied **structurally** — the flag is absent from the refcv5 CLI though the mechanism **is** implemented in `RefCConfig` and the flag **is** registered in the legacy `refc_train.py`. ⛔ So refcv5 **ranks a SAMPLED fan with a score computed before any sample was drawn** (`sampler_ranks_the_fan: False`). The *lever* — training the ranking head — was never built |
| **4** | ⭐ **Shape the path's CURVATURE, not its position** (R4b). An **inference-time** ridge on the offset only — `anchor + argmin_u‖u−off‖² + λ‖D²u‖²`, λ=3 chosen on a disjoint 71-episode half, scored on a held-out 70-episode / 2,392-window split: **curvature −0.001903 [−0.003359, −0.000587] separated better (0.007200 → 0.005330, −26 %)**, **ADE −0.000211 [−0.000690, +0.000266] NOT separated — the ADE gain is KEPT**, speed MAE **−0.008266 separated better**, cost **+0.000763 m cross-track (0.76 mm)**. Its deliberate-regression control `ROUGHEN_hf_x2` fires at **+0.005626 separated worse** | −26 % curvature | ⛔ **OUT** | ⛔ **no λ knob exists ANYWHERE** — `refine_lambda`/`lambda_refine`/`refine_w`/`w_refine`/`r4b` = **0 files repo-wide**; refinement is a **boolean**. `feasible_decode: false` |
| **5** | ⭐ **Repair or remove the longitudinal kin3 aux head** — eval-set prior floor **0.8421 nats** vs the trainer's eval CE **1.0090** ⇒ **0.1669 nats WORSE than predicting the class marginal**. (Lateral floor 0.5199, and the lateral head *does* beat it) | a real defect | ⛔ **OUT** | untouched |
| **6** | **H19 anchor prior — DE-PRIORITISE** (0.0034 m, **NOT** separated; 5.85 % of picks) | null | n/a | correctly not pursued |
| — | ⛔ **NOT on the list: "more ego dropout."** refcv4b already ran `--ego-dropout 0.5` for 40,284 steps and its vision-only arm is still **3.8× worse** | — | ⚠️ **refcv5 carries it anyway** (inherited from refcv4b's argv) | a vision-derived longitudinal state is the lever; more of the same knob is not |

### 3.1 Findings refcv5 inherited correctly from refcv4b (stated so the table is not one-sided)

These were **validated and ARE in refcv5**, because they were already in refcv4b's argv:

* the **v0-conditioned 117-anchor kinematic vocabulary** with declared `alat` units
  (`anchor_v0_conditioned: true`; `control_units_source: cli-override-legacy-file`);
* the **measured ego block at t0** (`ego_state_inject`, 5 channels, no future read, no
  finite differencing) — the PI's `D-VELOCITY-AT-CYCLE-TIME` ruling;
* the **reach-clamped selection band** (`sel_reach_clamp: true`, `sel_accel_max 2.0`);
* the **goal-provenance guarantee** — `contains_situation_classifier_output: false`,
  `situation_classifier_in_graph: false`, and the refused edges
  (`lan → inference`, `situation classifier output → any goal node`,
  `future_poses/future_actions → any goal node`) — the PI's 2026-08-03 binding rule,
  stamped and machine-checkable;
* `goal_str: true`, `lan_enable: true`, the hierarchy grafts.

---

## 4. ⛔ WHY refcv5 ended up with one lever — the argument that was made, and why it fails

The launch record states the reason plainly: WP-6 was blocked at preflight, and

> *"running it alone makes attribution cleaner, not worse — exactly one mechanism moves
> against refcv4b."*

**That reasoning is internally valid and was the wrong call**, for three reasons that are
themselves MEASURED:

1. ⛔ **The one-variable rule optimises for ATTRIBUTION; the PI asked for CAPABILITY.**
   `RULE ZERO` is explicit that a refutation is a waypoint, not a deliverable. A clean
   null on one lever is a diagnosis.
2. ⛔ **The lever chosen is not on the validated list at all.** `D-REFCV5-LEVERS` ranks six
   improvements *by measured effect size*; the DDIM sampler is none of them. refcv5 spent
   ~45 h of A40 on the mechanism with **no measured effect size**, while the top-ranked
   lever (**+0.4145 decodability**, constant control exactly 0.000000) went unbuilt.
3. ⚠️ **A one-seed separated CI could not have settled it anyway.** `H-ESTIM-SEED-1`:
   two arms differing in **nothing** read "separated" on **6 of 42 cells = 14.3 %**. And
   refcv5's own lever makes the arm **stochastic at inference**, adding a *third* variance
   the episode-cluster bootstrap is blind to. A single refcv5 arm was never going to
   produce a sufficient claim about the sampler.

⚠️ **What the one-lever choice did buy, and it is not nothing:** the arm proved the
sampler is **live rather than stamped** (`u0` non-zero at every logged step — not the M18
dead-flag defect), and it costs only **+5.3 %** wall-clock (4.046 vs refcv4b's 3.844
s/step). ⇒ **The sampler is cheap and works; it should be carried forward, not dropped.**

---

## 5. What the redesigned arm must add — the bridge to the SPEC

Ranked by the measured effect sizes above, the redesign must carry:

1. the **sampler** (keep — proven live, +5.3 % cost);
2. the **geometric goal point** (rank 2, and the only DD-adjacent item with a published
   effect size of its own);
3. the **curvature-penalised refinement** (rank 4, −26 % curvature at 0.76 mm).
   ⚠️ **A correction that travels with it:** the claim *"the refinement halves ADE"* —
   in the brief, in `H-R4B-1`'s register row, in `…/goal-point/RESULT.md` §5 and in
   `PREREG_R4B.md` §0 — **overstates it**. MEASURED: anchor **0.4281** → emitted
   **0.2965** = **−30.7 %**, not −50 %. The *"doubles curvature"* half is correct
   (0.004019 → 0.008150 = **2.03×**). ⛔ Four artifacts carry the overstatement;
4. the **22-token tactical vocabulary** — ⛔ **not "plumbing":** MEASURED this turn,
   `goal_head_tac` appears **0 times** anywhere in the REF-C line (it is a **v6** head,
   `models/v6.py` ×25), `TACTICAL_GOAL_TOKENS_V7` is not even defined in `v7_labels.py`,
   and the goal set is **multi-label**, so it is not a softmax head like the other four.
   ⇒ **for REF-C this is a plumbing gap AND an architecture gap** — five missing pieces,
   listed in `SPEC.md` item 6;
5. **agent/environment conditioning** (WP-6, DD #20) — the PI's specific question;
6. the **distance-keeping cost** (a GAP, never a rate);
7. **selector training** (WP-7) — gated on `a_star`.

Every one of these is dispositioned IN (with flag and weight) or OUT (with a named
blocker) in `SPEC.md` §3.
