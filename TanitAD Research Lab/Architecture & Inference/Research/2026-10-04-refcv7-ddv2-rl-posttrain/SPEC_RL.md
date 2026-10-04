# SPEC_RL — DiffusionDriveV2's RL stage as a post-training of refcv7-r101-s0 (WP-RL)

**Status: PRE-REGISTRATION, written before any RL number exists.** Package
`TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-ddv2-rl-posttrain/`.
Written 2026-10-04 by the WP-RL agent. No RL arm has run. The only numbers below are inherited
from banked artifacts (each tagged) or are design constants. The measured cost (s/step) is added
in §11 as an AMENDMENT before any training step runs. That amendment carries cost only, never an
outcome.

**PI, 2026-10-04:** *"At parallel we should plan a post training with RL exactly as stated in the
DiffusionDrive paper 2 as extension to see the effect. Run it on the Thor and evaluate the results."*

**Object:** refcv7-r101-s0 at step 50,400 (final). Model-only `D:/refcv7_eval_kit/ckpt/ckpt_50400.pt`,
md5 `b418d0fc4a92a6848c246a6a7c50207b` (MODEL_REGISTRY: 1131/1131 tensors equal to the rolling
`ckpt.pt` md5 `d5f104ee…`). Config md5 `e6512a01b9c70f0e4a4dac581621984a`. Launch tree `fec3a0d`.
**The RL code runs on a COPY of the launch tree** (`/home/nvidia/refcv7_post/rl/tree`, from
`/home/nvidia/refcv7_run/fec3a0dccf`) with three new files overlaid. The model is therefore built
by the code that trained it. MEASURED 2026-10-04: `_roll_state`, `_decode_ctrl` and `_sample` are
byte-identical between `fec3a0d` and the tip `50efa52`. `ddv2_rl.py`, `pdm_proxy.py`, `ddv2_il.py`
and `ddv2_refc_chain.py` are blob-identical too.

**Reference spec:** `…/2026-09-15-ddv2-rl-prep/SPEC_DDV2_RL_PAPER.md`. It is cited as `SPEC §x`.
Paper = arXiv 2512.07745v1 (P). Release = `hustvl/DiffusionDriveV2@1cd12a1` (C). Both are banked.

---

## 1. What the programme learned the hard way, and what is different now

| earlier result (MEASURED, registry rows) | what it means for this run |
|---|---|
| **H-DDV2RL-1 FAILED** and **H-DDV2RL-2 FAIL-HARM** on refcv5-v2 (2026-09-15). The RL-off control was as harmful as RL: T1 ADE +0.081 vs +0.083 m. The release's **all-modes IL** collapsed the 117-anchor fan by 93 % (37.5 → 2.45 m). | The IL **form** decides most of the outcome. RLOFF is mandatory. Fan spread is a canary, read during training. |
| **E-DDV2RL-L1-FAIL-HARM** (2026-09-17). Matched IL plus clip 100: T1 ADE 0.2994 → 0.5502 / 0.5561, 43× the two-seed floor. | A preserved fan does not suffice when the reward is broken. |
| **D9 attribution.** Seed 0 doubled the off-road rate: **DAC was ≡ 1**, because no caller passed a map. Seed 1 over-sped by +0.37 m/s, because **EP saturates**, so speed is a flat direction. | DAC is now live: the 10 cm SAM3 map covers all 4,369 train clips (D3: `/home/nvidia/data/sam3_gt_v3`, 4,369/4,369 readable). EP saturation is **the paper's own reward property** and is kept (§4). It is logged through the weight-0 SPD term. |
| **D-DDV1-CLAMP-1.** The release's x̂0 clamp kills the final-step gradient on clamped coordinates. | The clamp is a waypoint-box operation. Our state is a control residual, so the clamp is not applied. This is deviation D-4. Its would-be binding rate is measured (D2). |
| refcv5-v2's sampler was trained only at labels {10, 0}, so the RL chain's labels 18…2 were off-support (6.1 m fan shift). | **refcv7 was trained with F1 random t ∈ [0, 50)** (argv `--f1-random-t`). The chain's labels are on-support **by construction**. That earlier confound does not exist here. |
| refcv7 route diagnosis (2026-10-04). The fan contains a correct turn on 100 % of turn windows; the pick turns correctly on 84 %. Speed-profile choice is the largest planner lever (−0.606 m ADE). | RL (stage I) trains the GENERATOR. The selector is not trained (§3.3), so a gain may show in the fan before the pick. The T0 fan readouts are secondary for that reason. |

---

## 2. Arms

All arms share one cold start, one train split, one window order per seed, one step count, one
batch, and one optimiser. Each arm differs from `RL` in exactly ONE variable, and the argv audit
must show it (§8 I-11).

| arm | the one difference vs `RL` | why |
|---|---|---|
| **BASE** | no training; the cold start | the total-effect reference |
| **RL** | — (seed 0) | the paper's stage I |
| **RLOFF** | the policy-gradient coefficient × **0.0** (`--rl-weight 0`). The release's advantage-derived IL weights are kept (0.1 / 1.0 per row). | **The control that decided H-DDV2RL-2.** The IL continuation alone. RL − RLOFF is the RL effect. |
| **RL-SHUF** | each step's per-scene reward blocks (PDMS-proxy, constraint flags, human bar) are **deranged across the step's windows** before the advantage | the deliberate regression. The RL signal is decoupled from the scene, and the IL is unchanged. It **must not beat RLOFF**. |
| **RL-s1** | seed 1: window order, chain noise, shuffle stream | the training-seed replicate (H-ESTIM-SEED-1). Its floor bounds every RL claim. |

**Run order and drop order:** RL → RLOFF → RL-SHUF → RL-s1. If the budget forces a cut, RL-s1 is
dropped first, and every RL claim is then labelled *single training seed*. RL-SHUF is dropped
next. Without it, RL − RLOFF can be reported but **cannot be called valid**.

---

## 3. The method, item by item

### 3.1 The rule for paper versus code

1. **Where the paper STATES an object or a number, we implement the paper.**
2. **Where the paper is SILENT, we implement the release.** The release is the code that produced
   the paper's tables.
3. **Where the two DISAGREE**, the conflict is listed in §3.2 with our choice and its reason. A
   release-exact variant stays one flag away.
4. **Where OUR setting makes an item impossible, or changes its physical meaning, that is a
   DEVIATION** (§3.4). Each deviation states its reason and its expected direction. No deviation is
   taken that is not forced.

### 3.2 The PAPER-vs-CODE register (each is a conflict in the sources, not our choice of convenience)

| # | item | paper (P) | release (C) | **ours** | why |
|---|---|---|---|---|---|
| PC-1 | **IL loss form** | Eq. 9 `L = L_RL + λ L_IL`. The only IL objective the paper defines is **Eq. 4**: L_rec on **the anchor nearest the GT** (*"only a single mode receives supervision"*, §3). §4.4 says IL is the analogue of GRPO's KL to the reference model. | L1 of **every chain** (G·N) at every step and layer to the single GT (`rl.py:1104-1112`, SPEC A12) | **Paper: Eq. 4's single-mode L_rec.** L1 in metres of the G chains of the matched anchor `a*` (nearest GT over the DECODED bank, the trainer's own match `refc_v3_train.py:2383-2386`). It covers all 10 steps, averaged over steps and cascade stages as the release averages. | (i) The paper defines its IL, and the release replaces it. (ii) The reference model refcv7 was trained with exactly this matched-anchor L_rec, so it **is** the "KL-to-reference" analogue the paper names. (iii) The release form was MEASURED on our 117-anchor fan to collapse it by 93 % and to cause the whole T1 harm (H-DDV2RL-2). The paper reports −28 % diversity (P Tab. 3). In our setting the release form does not do what the paper says IL is for. The release form stays available as `--il-form release_all_modes`. It is **the named next lever** if RL fails for want of regularisation (§10). |
| PC-2 | RL loss normalisation | Eq. 7: `1/(N_anchor · G · T)` over all samples | mean over **non-zero-advantage** samples per (row, step), then over steps and rows (`rl.py:1099-1102`, A11) | **release** | Tab. 7's λ = 0.1 is the value the authors ran under the release normalisation. Eq. 7's average at the same λ would shrink the RL term by the non-zero fraction (cold start ≈ 3–7 %, inherited D6). That is a ~15–30× change in effective λ, which is a hyper-parameter change and not the paper's setting. |
| PC-3 | the −1 branch | Eq. 10: −1 *"if collision"* | −1 if `NC ≠ 1` **or** `DAC ≠ 1` (`rl.py:896-904`, A8) | **release** | The paper never defines the reward (SPEC §5.1). Its sub-scores, and the branch that reads them, exist only in the release. The off-road veto also addresses D9's seed-0 failure directly. |
| PC-4 | ≥GT positive mask | absent | positive advantage only where `r ≥ r_human − 1e-6` (A7) | **release** (paper silent) | rule 2 |
| PC-5 | per-row λ | λ = 0.1 (Tab. 7) | 1.0 on rows with no positive advantage, 0.1 otherwise (A13); IL is a batch-global scalar (A14) | **release** | numerics of the λ the paper reports. At the cold start ~every row has a positive (inherited D6), so λ ≈ 0.1. |
| PC-6 | LR schedule | "cosine, 10 % linear warmup" | per-EPOCH stepping, which makes the warmup a no-op (A15) | **paper**: warmup + cosine over **steps** (min LR 1e-6, the release's `min_lr`) | per-epoch stepping is meaningless below one epoch (§3.4 D-11) |
| PC-7 | exploration and likelihood | Eq. 5 `N(μ, η(1−α_t) I)`; §4.3 two multiplicative scalars; std floors 0.04 / 0.10 | additive DDPM term × 0; isotropic likelihood σ = max(σ_t, 0.1) at the detached sample (A3, A4) | **release** (= paper §4.3 + suppl. §7) | consistent once §4.3 replaces the additive noise |
| PC-8 | x̂0 clamp inside the step | absent | diffusers `clip_sample=True` default, box ±1 (A18) | **not applied** | deviation D-4 |

### 3.3 Everything else — implemented as the release (pinned bitwise in `test_ddv2_rl.py`)

η = 1 in the training rollout and η = 0 at inference. The inference is the deployed 2-step `[10, 0]`
ladder with F2 DD step pairs, exactly the sampler refcv7 ships. Other release elements, unchanged:

* The truncated start: noise at t = 8 on G = 4 tiled anchor copies, group-major.
* The 10-label chain [18 … 0] of one-unit transitions, with ᾱ(−1) = 1.
* The multiplicative exploration: two scalars per trajectory with floor 0.04. The additive draw is
  multiplied by 0 (it keeps the RNG stream).
* The likelihood floor of 0.10.
* Intra-anchor GRPO `(r − mean_G)/(std_G + 1e-4)` with the unbiased std.
* Per-sample truncation at 0, with the ≥GT mask and then the −1 branch.
* The γ = 0.8 discount, with weight 1.0 on the final step.
* REINFORCE with ratio ≡ 1, one update per rollout.
* AdamW at lr 2e-4 and weight decay 1e-4. Trainable: **the trajectory generator only.**

The trainable set is `traj_proj`, `time_mlp`, `layers` (incl. agent cross-attention and the BEV
waypoint sampler), `adaln`, `cascade`, `control_head`. `control_head` is grad-unreachable under F3,
as config.json `declared_vs_built` records.

Under F3 the query is detached between cascade stages. **The policy gradient therefore reaches only
the LAST stage** (layer 3, its AdaLN and its cascade heads). That is the analogue of the release's
1-layer decoder (A1). Stages 0–2 receive only the IL term on their own outputs. This follows the
release, which averages IL over "every decoder layer".

Frozen: the trunk, perception (map, box, BEV), tactical heads, selection grafts and E9. **The
selector (paper stage II) is NOT part of this work.** The ranked score is the last cascade stage's
emitting conf (F5). It sits on the same `layers.3` features, so RL moves it as a side effect. The
release's cls branch on its shared decoder layer behaves the same way (SPEC §7.2).

### 3.4 DEVIATIONS (forced by our setting)

| id | item | paper / release | ours | why forced | expected direction of effect |
|---|---|---|---|---|---|
| **D-1** | anchors | 20 K-means, fixed | refcv7's **117 v0-conditioned control anchors** → 468 chains/window | the model is refcv7 | more chains per scene: finer GRPO groups and higher cost per window |
| **D-2** | diffusion state | normalised waypoints x/50, y/20 | normalised **residual control** Δ/control_norm on the `ha0_ext_pose` prior (NEW-1), integrated by the unicycle | the model's sampler state | multiplicative noise scales Δ: an anchor at the prior (Δ = 0) explores only through the t = 8 start noise. FAITHFUL arithmetic, DIFFERENT effect (D6 reads the within-group reward spread). |
| **D-3** | decoder | 1 layer | 4 cascade layers (F3, query detached), AdaLN (F4), agent cross-attention, BEV coupling | the model | RL reaches only the last stage (§3.3) |
| **D-4** | x̂0 clamp (A18) and decoder-input clamp (`rl.py:826`) | ±1 on x/50, y/20: a ±50 m / ±20 m waypoint box | **not applied** | on a control residual, ±1 is a ±control_norm bound on **residual acceleration**, a different physical claim. The deployed refcv7 sampler (the policy being post-trained) clamps nothing, so a clamped chain would be a different policy. D-DDV1-CLAMP-1: a clamped coordinate gets zero final-step gradient. | the release-exact clamp would bind on some fraction of coordinates (MEASURED in D2 and reported). Without it, no gradient is silently zeroed. |
| **D-5** | gradient clipping | none (`gradient_clip_val 0.0`) | **max-norm 100** (a spike guard) | the batch is forced ≤ 64 vs 512 (D-11): ≥ 8× more per-step variance. MEASURED: at batch 4 without clipping, 1 of 2 seeds diverged (grad norm 15,712). Max-norm 100 bound 1/600 steps on the stable arms (AMENDMENT A-1, 2026-09-16). | none on a stable run; prevents the measured divergence mode. `grad_clipped` is logged per step. |
| **D-6** | reward horizon | 4 s / 40 ticks, the release's whole plan | **6 s / 60 ticks**: refcv7's whole plan | our plan is 6 s. A 4 s reward would leave slots 7–8 unscored: a flat direction for REINFORCE (the D9 mechanism). | more NC/TTC zeros from non-reactive replayed agents at long range. The human's own rates are the control (§8 I-4). |
| **D-7** | DAC | NAVSIM drivable polygons, ego corners, any tick | **SAM3 10 cm map** (`map_fine`, the window's NOW frame, the map head's own target). A corner is off-road iff its 0.5 m block has ≥ 13/25 seen cells and < 50 % of them are road-like {1,2,3,4,6} (D3's literal set). Unseen or off-grid counts as compliant. | PhysicalAI publishes no map (K9) | an UPPER bound on compliance: unseen ⇒ compliant. Label noise is mitigated by the 0.5 m block. |
| **D-8** | EP reference | pairwise vs **PDM-Closed** (SPEC §13.1) along the route centreline | pairwise vs **the human's own future** along the human's path | no route centreline and no PDM-Closed planner in PhysicalAI (K9) | the human's EP is always 1. The ≥GT bar is harder to clear than vs PDM-Closed. EP saturates above the reference **exactly as in NAVSIM**, so over-speed is a flat direction (§4). |
| **D-9** | NC / TTC map clauses | at-fault lateral contacts and TTC in intersections use lanes | no lane clauses; ahead = ±30° cone; agents = the 2-D obstacle join (`b1_train_plus_eval_agents.jsonl.xz`), replayed non-reactively | no lane graph | lateral contacts are never at-fault: an upper bound on NC |
| **D-10** | corpus | navtrain (~103 k samples, INHERITED from the NAVSIM docs; SPEC §7.8 marks the count UNVERIFIED) | refcv7's **train split**: 4,369 clips, eligible windows only (full 69-tick future and every agent frame labelled) | our data | — |
| **D-11** | scale | 10 epochs × navtrain, batch 512 (8 × L20) | **batch B = 32 windows** (micro-batches of 4–8), K steps from the MEASURED s/step (§11) under the budget. **< 1 epoch** of our corpus: no window is seen twice. | one Thor, ≤ ~24 GPU-h for the RL arm | stated as a fraction of the paper's sample count and step count in §11 |
| **D-12** | precision | 16-mixed | the frozen trunk runs bf16 as trained; the generator runs **fp32** | — | the more exact arithmetic |

---

## 4. The reward — the paper's PDMS shape, and every term's status

`PDMS = NC × DAC × (5·EP + 5·TTC + 2·C)/12` (P Eq. 14). It is computed per candidate over 61 ticks
(t0 + 60 × 0.1 s) by `pdm_proxy.score_candidates`, with DAC from `ddv2_refcv7.dac_fine`. The ego
states come from the candidate's OWN integration (`ddv2_refcv7.tick_states`). That uses the
decoder's `compose_ticks` + `rollout_unicycle`, pinned bitwise to the emitted fan at the slot
ticks. The human is proposal 0 in every call. ⛔ **It is a PROXY. No number from it is PDMS.**

| term | status |
|---|---|
| NC | {0 dynamic, 0.5 static class, 1}; first contact per track decides; at-fault = ego moving ∧ (track stopped ∨ front-edge contact, not behind) |
| DAC | LIVE (D-7). ⛔ The driver REFUSES a train run whose dataset has no 10 cm map store. `n_dac_dead` (windows whose map frame is missing) is logged per step. |
| EP | pairwise vs the human (D-8). **Saturates at 1.0 for any candidate at least as far along as the reference.** This is NAVSIM's rule, so the paper's reward is blind to over-speed above PDM-Closed in the same way. We **keep it**: the PI asked for the paper's reward. The weight-0 SPD term is logged on every candidate, so any drift is visible. If over-speed appears, the pre-registered SPD repair (`PREREG_D9_REWARD_REPAIR.md` §13, `v_tol ∈ {0.75, 1.5, 3.0}` selected on the FIT split) is the next lever (§10). |
| TTC | footprint at +0 / 0.3 / 0.6 / 0.9 s at constant velocity, against tracks inside the ±30° cone, while the ego moves |
| C | NAVSIM's six thresholds on one least-squares polynomial over the window (= savgol with window n_time) |
| weights | 5 / 5 / 2; w_spd = 0 |

**Known-value controls on the reward.** Unit tests in `stack/tests/test_ddv2_refcv7.py`, plus D5
on real windows:

* A path that leaves the road reads DAC = 0. A path whose CORNER, but not its centre, leaves the
  road reads DAC = 0. A lane line under a wheel reads 1. An unseen area reads 1. A left-only road
  is read with the right sign.
* A dynamic agent on the path reads NC = 0 and is a constraint failure. A static-class agent reads
  NC = 0.5. An agent 10 m to the side reads NC = 1 and TTC = 1. An ego that brakes 0.3 m short of a
  stopped agent reads NC = 1 and TTC = 0.
* A path 1.5× faster than the human reads EP = 1.0 exactly (the saturation, stated). A path at
  0.5× reads EP ≈ 0.5. SPD is logged, and PDMS is identical at w_spd = 0.
* The human as a candidate scores exactly the human. Chunked scoring equals unchunked. The track
  filter is exact for NC and TTC.

**Mutations that must go RED (each one is in the test file):**

* sidewalk counted as road;
* DAC from the centre point instead of the corners;
* the lateral axis flipped;
* agents dropped;
* static and dynamic NC values swapped;
* the TTC projection removed;
* a binding that drops agents, drops BEV, or rolls on v0 instead of the prior's speed;
* a decoder pass that mixes queries;
* `tick_states` rolled from the wrong speed.

---

## 5. Splits

* **RL-train:** the 4,369 refcv7 train clips. Each window must have t0 + 69 inside the clip and
  every one of its 70 agent frames labelled (`ddv2_rl_refcv7.py windows`). The list is banked with
  its sha256 digest. The order per seed is `randperm(seed 10,000 + s)`, with no wrap.
* **Held-out:** eval139, scored only after training. The T1 battery uses the S2 grid
  (4,754 windows / 139 episodes) and S6. The T0 proxy uses the eligible eval windows.
* ⛔ No eval clip trains. No hyper-parameter is chosen on eval139.

---

## 6. Endpoints and bars (committed before any data)

**Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`: n_boot 2000, seed 0, cluster =
episode, 95 % percentile. Point estimates are FULL-SET means.

**Floors.**
* **Inference floor:** `|os_s0 − os_s1|`, the same checkpoint at inference seeds 0 and 1.
* **Training floor:** `|RL-s1 − RL|`, when RL-s1 ran.

**Which variance each interval answers.**
* The bootstrap answers "another draw of EPISODES".
* The two inference seeds answer "another INFERENCE run".
* RL-s1 answers "another TRAINING run". Without RL-s1 that question is unanswered and is said so.

### 6.1 PRIMARY — T1 (self-action OPEN LOOP; never closed loop), all four families

**Instrument.** The refcv7 battery's own roll (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-
refcv7-standard-tests/battery/code/roll_seed_r7.py --os-only`). It runs on each arm's export (§7),
at inference seeds 0 and 1, on **S2**. Families come from `refcv3_arm.analyze_refcv3`, plus the
battery's yaw-rate cell and distance keeping:

* **LONGITUDINAL:** speed MAE, target-speed accuracy, along-track MAE, accel MAE, and distance
  keeping (min headway / time gap / min TTC where a lead exists, with n).
* **LATERAL:** heading, curvature, yaw-rate, cross-track.
* **TACTICAL:** trajectory-implied lateral and longitudinal decision correctness. The tactical
  heads are frozen, so their declared rows are an identity control.
* **STRATEGIC:** NOT APPLICABLE, n = 0. The strategic layer is OFF in refcv7 (`--no-strategic`,
  PI ruling, SPEC_REFCV7 §1).

**Decision metric:** S2 `ade_m`.

| hypothesis | contrast | SUCCESS / PASS iff | FAILURE / FAIL-HARM iff | otherwise |
|---|---|---|---|---|
| **H-WPRL-1 (the RL effect)** | RL − RLOFF | CI upper < 0 at **both** inference seeds, **and** \|Δ\| > 2 × the inference floor, **and** RL-s1 − RLOFF also has CI upper < 0 at both seeds, **and** \|Δ\| > 2 × the training floor | both seeds' CIs contain 0, or either upper bound > 0 is separated worse | **NOT PROVEN.** This includes "separated but within 2× a floor" and "RL-s1 not run": the result is then reported as *single training seed* |
| **H-WPRL-2 (the total effect)** | RL − BASE | CI upper < 0 at both seeds and \|Δ\| > 2 × the inference floor (and the training floor if measured) = **IMPROVED** | CI lower > 0 at both seeds = **FAIL-HARM** | NO CHANGE DETECTED (never "no harm") |
| **H-WPRL-3 (validity)** | RL-SHUF − RLOFF | — | RL-SHUF better than RLOFF, separated at both seeds ⇒ **the whole RL panel is VOID** | valid |
| family harm guard | RL − BASE, every family metric | — | any metric separated worse at both seeds, beyond 2× its inference floor ⇒ **reported as a family harm**, and the arm cannot be called IMPROVED on ADE alone | — |

S6 (0–6 s) is reported in the same table as a pre-registered **non-regression**. An RL arm
separated worse than BASE at 6 s on both seeds is a FAIL-HARM at 6 s.

### 6.2 SECONDARY (reported with a direction, no new verdict)

1. **The objective's own read (T0, eval139 eligible windows, recorded future).** The selected-plan
   PDMS-proxy and its sub-scores (NC, DAC, EP, TTC, C, SPD) under the deployed sampler with paired
   inference noise. Also: fan PDMS-proxy mean, fan best (**ORACLE**, labelled), fan endpoint spread
   and best-of-117 ADE, the selected plan's off-road rate, its collision rate, and its signed speed
   bias vs the human (the D9 instruments). Contrasts RL − RLOFF and RL − BASE.
2. **Fan non-collapse gate (from PREREG_DDV2_RL_VALIDATION §13.1).** Each trained arm must retain
   **≥ 60 %** of BASE's deployed fan endpoint spread (paired CI lower bound of Δspread > −40 % of
   BASE). A collapsed arm is reported as such before any endpoint.
3. **Route following** (the route package's instrument, on its 107 eval turn windows): pick turns
   in the correct direction, and heading within 15° of GT. Pick and fan oracle are both reported.
4. **NavSim zero-shot** (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/
   navsim/`, dev box, later): navtest PDMS and navhard EPDMS for RL vs BASE (and RLOFF if compute
   allows), against STOP.
   **External reference only, never our bar:** the paper's RL stage is worth ≈ **+2.1 PDMS** on
   NAVSIM v1 navtest (P Tab. 9: DiffusionDrive 88.1 → +selector 89.1 → full V2 91.2).

### 6.3 Both outcomes, committed

* **H-WPRL-1 SUCCESS:** RL is a lever on refcv7. Next is the paper's stage II (the mode selector)
  on the RL generator, plus the NavSim read.
* **H-WPRL-1 FAILURE / NOT PROVEN** with RLOFF ≈ BASE: the RL term adds nothing at this scale. The
  next lever is chosen by the measured diagnostics (§10), not argued.
* **H-WPRL-2 FAIL-HARM:** check first whether RLOFF also harms (as on refcv5-v2) or only RL does.
  Then read the D9 signatures (off-road rate, speed bias) on the T0 secondary. The repair follows
  §10.

---

## 7. Export and evaluation protocol

`ddv2_rl_refcv7.py export` writes `{model, step: 50400, ddv2_rl}` with refcv7's own keys. Only the
generator subset is replaced. The file loads STRICTLY through `refcv7_loader.build_model`, so the
battery's roll needs no change.

* **BASE rolls:** the battery's own 50,400 rolls are reused when their checkpoint md5, config md5
  and S2 digest re-verify. Otherwise BASE is rolled with the same tool.
* **Timing:** each arm is evaluated after its last segment, on Thor under the lock (or on the dev
  box under its lock). The tool and the windows are the same either way.

---

## 8. Integrity — all must hold, or the run is VOID (not negative)

| id | check |
|---|---|
| I-1 | every step finite; RL arms' parameters moved (`param_delta_norm > 0`) |
| I-2 | RLOFF: `coef_rl_abs_sum == 0.0` exactly on every step |
| I-3 | RL arms: `frac_positive_after_bar > 0` on ≥ 90 % of steps |
| I-4 | human NC = 1 and human DAC = 1, each on ≥ 95 % of training windows (D5, then per step) |
| I-5 | DAC live: `cand_dac_mean < 1` on ≥ 50 % of steps; `n_dac_dead` ≤ 1 % of windows |
| I-6 | identity: an export of a `--lr 0 --rl-weight 0` run equals the cold start **bitwise** (every tensor) |
| I-7 | binding parity (D1): `native_sample` reproduces the deployed sampler **bitwise** on ≥ 12 real windows, and the trainer's own `compute_losses_v3` forward captures the SAME sampler inputs |
| I-8 | resume: a segmented run reads the same windows in the same order and continues the same RNG streams (state restored from the checkpoint). Numerics may differ only by CUDA non-determinism, measured on the smoke. |
| I-9 | RL-SHUF: every step's permutation is a derangement (logged) |
| I-10 | `tick_states` equals the decoder's roll at the slot ticks bitwise (D3) |
| I-11 | argv audit: each arm's `run.json` differs from RL's only in its named variable |

---

## 9. Compute protocol (Thor)

* Every GPU job runs under `flock /home/nvidia/refcv7_post/thor_gpu.lock`, detached
  (`setsid nohup`), one at a time.
* **Segments of ≤ 45 min.** A checkpoint is written every 25 steps and at each segment end:
  `ckpt_latest.pt`, holding the generator subset, AdamW state, RNG generators, step and run hash.
  ≤ 4 checkpoints are kept on disk.
* **Yield detection.** Between steps the trainer scans `/proc/*/fd` for any process OUTSIDE its own
  session that holds or waits on the lock file (`other_lock_users`). Once ≥ 5 min of the segment
  have run and such a process exists, the trainer checkpoints and exits with code 75. The segment
  loop then waits until no other process holds the lock before re-acquiring it.
* **Off-switches:** a `STOP` file in the run dir checkpoints and exits. `DONE.json` marks
  completion. No stray python is left behind, and processes are killed by explicit PID only.
* **⛔ No training step runs without the Master Mind's launch-gate PASS token** (PI 2026-09-26).

---

## 10. Rule Zero — the next lever for each failure, ranked by measured size (filled in RESULT)

| observed | next lever | cost |
|---|---|---|
| fan collapses (< 60 % spread) | IL weight or the release-form check; inspect `il_m` against `chain_endpoint_spread_m` | 0 GPU (diagnostic) + 1 arm |
| RL ≈ RLOFF, `frac_positive_after_bar` small, or within-group reward std ≈ 0 | exploration on the ABSOLUTE trajectory (a declared deviation from D-2) or a larger G | 1 arm |
| RL over-speeds (signed bias separated > 0) | the pre-registered SPD term (`PREREG_D9_REWARD_REPAIR` §13) | 1 arm (+ FIT-split v_tol selection) |
| RL drives off the road more (despite live DAC) | audit the DAC rule against the human control; a stricter block rule | 0 GPU + 1 arm |
| the fan improves (oracle / mean) but the pick does not | **the paper's stage II selector** (PLAN_REFCV8 X1: a trained listwise selector) | stage-II run |
| RLOFF harms on its own | lower lr for the IL continuation, or IL on the last stage only | 1 arm |

---

## 11. AMENDMENT A-1 — measured cost and the step budget (written 2026-10-04, before any training step; cost only)

**MEASURED** on Thor (`raw/diagnose_train.json` D4): one full RL step (capture, 10-label rollout
of 468 chains per window, reward, grad pass; no optimiser update) on real TRAIN windows.

| micro-batch | s / step | s / window | peak GPU memory (`max_memory_allocated`) | capture / rollout / reward / grad (s) |
|---|---|---|---|---|
| 1 | 2.32 | 2.32 | 1.14 GB | 0.25 / 0.15 / 0.54 / 1.35 |
| 2 | 3.27 | 1.64 | 1.82 GB | 0.49 / 0.21 / 1.10 / 1.44 |
| 4 | 4.13 | 1.03 | 3.01 GB | 0.84 / 0.32 / 1.22 / 1.73 |
| **8** | **6.76** | **0.845** | **5.53 GB** | 1.55 / 0.53 / 2.34 / 2.31 |

**The budget rule, fixed here:**

* **B = 32 windows per optimiser step, as 4 micro-batches of 8.** ESTIMATED from the micro-8 row:
  27.0 s/step plus the optimiser and the data stream. `smoke/timing32` MEASURES it.
* **K = 2,000 steps per arm.** That is the paper's optimiser-step count: 10 epochs × navtrain
  (≈ 103 k samples, INHERITED) / 512 ≈ 2,000. It applies on the condition that the measured
  timing32 s/step is ≤ 43.2 s, i.e. the RL arm stays ≤ 24 h of Thor GPU. If that condition fails,
  K = floor(0.9 × 86,400 / s_step).
* **Scale, stated:**
  * 2,000 × 32 = **64,000 windows per arm**. That is 12.6 % of the 507,588 eligible TRAIN windows,
    and no window is seen twice.
  * It is ≈ 6 % of the paper's ≈ 1.03 M sample-visits, at the paper's step count with 1/16 of its
    batch.
* **Schedule:** warmup 200 steps, then cosine to 1e-6 at step 2,000.
* **Checkpoints:** every 25 steps (≈ 12 min) and at every segment end.
* **Disk:** `ckpt_latest.pt` ≈ 0.2 GB (13,804,924 trainable params + AdamW + the base copy), one
  rolling file per arm. Exports are 0.4 GB each.
* **Order (§2):** RL → RLOFF → RL-SHUF → RL-s1 ≈ 4 × 15 h of GPU at the estimated rate, plus
  waits on the shared lock.

The reward (0.29 s/window) is the largest single term at micro 8. It is per-window Python over
`pdm_proxy`. Speeding it up is a cost lever only. It may land only with a bitwise-equality test
on the scores, and is not done here.

---

## 12. AMENDMENT A-0 — the DAC rule is calibrated on the HUMAN control only (written 2026-10-04, before any RL number)

**Why.** A CPU dry run of `diagnose` on two eval windows found the human's own trajectory scoring
**DAC = 0** on one window under the registered rule (D-7, here called **R1**). A 40-window eval
census (`raw/census_eval40.json`, model-free, human and synthetic candidates only) located the
cause. Every failing human corner lies on code **0, "seen, no map class"**: 62 corner-ticks, and no
code 5 or 7. This is an INSTRUMENT problem, a human control failing, not an outcome. No RL arm
exists and no candidate of the model has been scored on eval.

**The ladder, fixed here, strict → lenient:**
* **R1** (registered): code 0 counts as NOT road.
* **R2**: only the D3 audit's "contradict" codes, {5 non-drivable edge, 7 sidewalk / verge}, are
  off-road evidence. Code 0 and 255 carry none. A corner is off-road iff its 0.5 m block has ≥ 13
  road-or-contradict cells and more contradict than road cells.

**Decision rule (fixed before the TRAIN census is read):** use the **strictest rule whose human
DAC = 1 rate is ≥ 95 % on a 300-window TRAIN census** (seed 0, `census --split train`). If neither
rule reaches 95 %, STOP: the DAC instrument is inadmissible and is reported as such. Eval windows
are never used for this choice.

The chosen rule, its census and its eval-40 reading are reported together. ⚠️ Under R2 an ego
driving onto unlabelled ground (code 0) is not penalised, so R2 is the more lenient bound. The
6 m-shift control (the human path moved 6 m sideways) must still read DAC = 0 on most windows
(eval-40: R1 0.913, R2 0.888); otherwise the rule cannot see a road edge at all.

**A-0 RESULT (MEASURED 2026-10-04, `raw/census_train300.json`, 300 TRAIN windows, seed 0, model-free).**

| rule | human DAC = 1 | human NC = 1 | human TTC = 1 | human C = 1 | 6 m shift reads DAC 0 | codes under the failing human corners |
|---|---|---|---|---|---|---|
| R1 | **0.890** (< 0.95) | 0.997 | 0.983 | 0.950 | 0.915 | 0: 777, 7: 45, 5: 26, 255: 12, 1: 7 |
| **R2** | **0.983** (≥ 0.95) | 0.997 | 0.983 | 0.950 | 0.808 | 7: 43, 5: 23, 1: 2, 0: 1 |

⇒ **R2 is the rule** (the strictest rung that passes). It is the driver's default (`--dac-rule R2`)
and is stamped into every `run.json`. R1 failed on unclassified ground: 777 of 867 failing
corner-ticks are code 0. R2 keeps 81 % sensitivity to a 6 m lateral shift. The other known values
hold on TRAIN:
* a 1.5× faster human reads EP = 1 on 0.67 of windows; the rest are collisions with the lead,
  which is correct;
* a 0.5× slower human reads EP 0.49.

**A-1 measured (2026-10-04 13:57Z, before any training arm; `raw/smoke/smoke_report.json`):**
`smoke/timing32` ran 3 steps at B = 32, micro 8, on real TRAIN windows: **30.85 s/step** (wall,
steps 1–2; step 0 took 32.1 s including the stream start). Mean step components: capture 6.9 s,
rollout 2.2 s, **reward 12.4 s**, grad pass 9.5 s.
⇒ 30.85 ≤ 43.2, so the condition holds and **K = 2,000 is fixed**: ≈ **17.1 h of Thor GPU per
arm**, ≈ 69 h for the four arms, plus the evaluation reads and any wait on the shared lock.
