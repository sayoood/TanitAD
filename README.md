# TanitAD

**Mission: build the AD stack that beats the best — a sub-300 M hierarchical latent world model
(the "4-brain"), orders of magnitude less data, inference-efficient on Orin/Thor-class hardware,
inherently safe and aligned with the 2026 UN ADS regulation.**

> Constitution: [`Project Steering/Mission Plan.md`](Project%20Steering/Mission%20Plan.md) — owned by
> Sayed (PI), agents never edit it. First hard external evaluation: **2026-10-05**.
>
> **This file last brought to true state: 2026-08-03.** Every load-bearing number below carries its
> **evidence class** and the **run directory** it lives in. If a number here disagrees with
> [`Project Steering/MODEL_REGISTRY.md`](Project%20Steering/MODEL_REGISTRY.md), **the registry wins
> and this file is the bug.**

---

## Start here (any session, human or agent)

Read in this order. The first three are binding, not background.

| # | File | Why |
|---|---|---|
| 1 | [`CLAUDE.md`](CLAUDE.md) | **Binding working agreements.** Source-of-truth rule, the traps preflight, git hygiene, the never-idle rule, the four-family rule. |
| 2 | [`Project Steering/RETRACTION_LOG.md`](Project%20Steering/RETRACTION_LOG.md) | **Append-only, and must be read before asserting in a known class.** It records the *root-cause class*, which is the part that recurs. |
| 3 | [`Project Steering/MODEL_REGISTRY.md`](Project%20Steering/MODEL_REGISTRY.md) | The **only** quotable source for model facts — architecture, params, training args, parity key, results. |
| 4 | [`Project Steering/PROGRAM_OVERVIEW.md`](Project%20Steering/PROGRAM_OVERVIEW.md) | The living whole-program strategic briefing (vision, edges, hypotheses, honest position, critical path). |
| 5 | [`PROJECT_STATE.md`](PROJECT_STATE.md) · [`DECISIONS.md`](DECISIONS.md) · [`Project Steering/BACKLOG.md`](Project%20Steering/BACKLOG.md) | Live state · the ADR log · the pull-list when the headline item is gated. |
| 6 | [`Paper/TANITAD_PAPER.md`](Paper/TANITAD_PAPER.md) | The living paper (**v0.8**). |

---

## ⛔ The four rules that govern every number in this repo

### 1. Every eval reports **four metric families**, not ADE

Binding since 2026-08-02 (PI, after asking repeatedly). **ADE stays; these are ADDED to it. An eval
that reports ADE alone is INCOMPLETE and must not be presented as a result.**

| family | must report | why it is not optional |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy **and distance-keeping** (headway / time-gap / TTC to the lead) | 88.7 % of the oracle gap is longitudinal; an arm can win ADE while setting the wrong speed |
| **LATERAL** | heading, **curvature, yaw-rate**, cross-track | "lateral is fine" has been asserted from cross-track alone; curvature and yaw are where a smooth-but-wrong path shows up |
| **TACTICAL** | manoeuvre-decision quality and goal/anchor selection (selected vs executed, class confusion) | the 5-way softmax that MIXES lat+lon is the single largest known defect; a scalar ADE cannot see a decision error |
| **STRATEGIC** | strategic decision + route/goal-setting quality | the hierarchy is the programme's thesis; unmeasured means unclaimable |

Per family, **never pooled**. Each carries the **paired episode-cluster bootstrap**
(`taniteval/ci.py`) on the same windows as the ADE beside it. A family with no instrument is a
**work item, not an excuse**; a family that genuinely cannot be computed is reported **with its
reason and its n**, not silently dropped.

### 2. Evidence class on every claim

`MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not re-verified) ·
`ESTIMATED` · `HYPOTHESIS`. **A claim that decides a GPU-day must be MEASURED or PUBLISHED, never
INHERITED.** "X does not exist" needs **two probes at different paths** before it may be written.

### 3. Estimator, or no interval

Decision-grade = **`full_set` mean + episode-cluster bootstrap**, paired for two arms.
⛔ **`overlapping_holdout_se` is deprecated and BIASES THE POINT ESTIMATE**, not just the interval
(27 dumps = 25 distinct arms — C126: headline `ade_0_2s` shifts **−6.67 % to +11.69 %, bidirectionally**; intervals
1.107–3.100× too narrow, median 1.499×). Before trusting any pre-2026-07-25 number, check whether it
is the `heldout` split-mean or the `full_set` mean — the registry publishes both and they differ.

### 4. Prove the metric can discriminate — negative control **first**

A metric that cannot separate the right answer from a wrong one certifies nothing, however
reasonable its value looks. And when you compute per-window components to feed a bootstrap, **assert
they reduce to the family mean printed beside them** — that control caught a **5,347×** curvature
inflation in an agent's own reducer that would have published a lateral separation that does not
exist.

### 5. Labels may use ego; **inference is VISION-ONLY**

Binding since 2026-08-03 (PI, verbatim: *"for ground truth data of scenario classification you can
use both ego and other label, for inference only vision"*).

| stage | what may be used |
|---|---|
| **Ground truth / label derivation** | ego state, other agents, maps, future poses — anything. Labels are built offline; privileged signals are **fine** here. |
| **Inference** | ⛔ **VISION ONLY.** No ego state, no privileged channel. |

⇒ For the scenario classifier the deployable arm is **`head_img` (image-only)** — not `head_img_ego`,
not `head_ego`. ⭐ **Generalised beyond that case: for ANY head, ask whether its inference inputs
include something the label was DERIVED FROM.** If they do, the score measures **leak magnitude**,
not capability. Same family as the C6 confound and the REF-A I-JEPA leak (~80 % of val inside train).
**Guardrail: "vision scores worse" is NEVER a reason to reopen ego at inference** — if the
vision-only arm is weak, the finding is *how much, why, and what would fix it*.

---

## The result that made the four-family rule binding

**Closed loop, on a NuRec neural reconstruction, rendered on the Jetson Thor. 9 rollout starts ×
50 ticks, paired over 437 shared windows.**

`MEASURED` — run dir [`stack/experiments/alpasim-gsplat/results/`](stack/experiments/alpasim-gsplat/results/)
(`metrics_empty.json`, `metrics_objects.json`); code [`stack/experiments/alpasim-gsplat/`](stack/experiments/alpasim-gsplat/);
videos [`TanitAD Research Hub/Evaluation/Videos/alpasim-closedloop/`](TanitAD%20Research%20Hub/Evaluation/Videos/alpasim-closedloop/).

### REF-C base beats flagship v1 closed-loop — and the separation is **entirely lateral**

⛔ **SUPERSEDED 2026-08-03 evening — "entirely lateral" is RETRACTED (R-2026-08-03-C).** Re-measured
on the +23.4 % grad-NCC render the videos actually show, ADE **separates at +7.164 [+5.265, +8.966]**
and so do both longitudinal metrics and strategic corridor departure; the four lateral separations
all survive and widen. The mechanism is that **flagship v1's driven path moves a mean 9.05 m under
the render change while REF-C's moves 0.43 m** — a 21× render-sensitivity ratio. Determinism control
was exactly 0.0 on 450/450 windows, so the attribution is clean. Current numbers:
[`…/closedloop-hq-render/STREAM_C_RENDER_AB.md`](stack/experiments/alpasim-gsplat/results/closedloop-hq-render/STREAM_C_RENDER_AB.md).
**The table below is the OLD render and is kept only as the retracted record.**


> ⛔ **SUPERSEDED 2026-08-03 — the reference video is offset from the rig by a per-scene
> constant** (`+6` on `00040136`, `+5` on `7c72937c`; rule: `video_idx = rig_idx +
> (n_mp4_decodable − n_rig_frames)`, measured by the renderer, unanimous over 12 frames each).
> Re-baselined against the **aligned** reference the improvement is **roughly half the size and
> does not replicate**: `00040136` n=5 **+13.5 %** (was +23.4 %), n=12 **+8.0 %**, and
> `7c72937c` n=12 **+4.4 % — NOT SEPARATED** [−0.0097, +0.0521]. Absolutes move too:
> BEFORE 0.2774 → **0.4228**, AFTER 0.3424 → **0.4800**. The render is still better; the
> magnitude quoted here is not. Corrected table + estimator:
> `TanitAD Research Hub/Evaluation/Implementation/incoming/2026-08-03-render-rebaseline/`;
> `RETRACTION_LOG.md` R-2026-08-03-align. ⚠️ No closed-loop conclusion moves — `cl_metrics.py`
> never opens the reference video.


Paired Δ = flagship v1 − REF-C base (positive = flagship worse), empty-road condition:

| family | metric | paired Δ [CI95] | separated |
|---|---|---|:--:|
| **ADE** | `ade_0_2s` | **+0.7885 [−0.8653, +2.7282]** | ❌ **no** |
| **LONGITUDINAL** | `abs_target_speed_err_ms` | +1.1242 [−0.1008, +2.5657] | ❌ no |
| **LONGITUDINAL** | `along_track_ade_m` | +0.6498 [−1.0166, +2.5903] | ❌ no |
| **LATERAL** | `cross_track_abs_m` (= `dist_to_gt_traj_m`) | **+1.1705 [+0.0296, +2.2438]** | ✅ **yes** |
| **LATERAL** | `heading_err_rad` | **+0.0838 [+0.0278, +0.1750]** | ✅ **yes** |
| **LATERAL** | `curvature_err_1pm` | **+0.0050 [+0.0008, +0.0130]** | ✅ **yes** |
| **LATERAL** | `yawrate_err_rads` | **+0.0378 [+0.0201, +0.0565]** | ✅ **yes** |
| **TACTICAL** | `manoeuvre_plan_eq_logged` | +0.0709 [−0.1241, +0.2600] | ❌ no |
| **STRATEGIC** | `route_corridor_departure_rate` | +0.2037 [−0.0023, +0.3982] | ❌ no |

⭐ **An ADE-only table would have reported nothing** on a comparison where four lateral measures
separate cleanly and all in the same direction. That is the whole argument for the rule, and it
reproduces the 2026-07-23 native-1080 n=12 suite on **different hardware, a different renderer and a
different scene** — so the doctrine is not an artifact of one harness.

### Three defects only the families expose

1. **The arms fail longitudinally in OPPOSITE directions** — flagship `target_speed_err_ms`
   **−2.0412 m/s** (too slow), REF-C **+1.3307 m/s** (too fast). **A pooled score cancels this.**
2. **The flagship does not execute what it selects** — `manoeuvre_exec_eq_plan` **0.4481**, a genuine
   5-class agreement rate (its executed classes span lane_keep 83 / turn_right 30 / accelerate 33 /
   brake_stop 124 of 270). ⛔ **Do NOT read REF-C's 0.8741 as "REF-C executes what it selects".**
   REF-C's executed class is `lane_keep` on **270/270** windows, so the metric collapses to *"how
   often was the plan lane_keep"* = 236/270 = **0.8741 exactly**. It is a **constant predictor tie**
   — the same degeneracy as defect 3, not a competence. **The two numbers are not comparable**;
   `manoeuvre_exec_eq_plan` cannot discriminate on an arm whose execution is single-class.
3. 🔴 **REF-C's 5-way manoeuvre head NEVER emits a longitudinal class.** Closed-loop
   `head_class_share` = lane_keep 0.627 / turn_right 0.373 / **accelerate 0 / brake_stop 0**, while
   **41.9 %** of logged windows are `brake_stop`. **This is the programme's top defect** — the
   documented lat+lon-mixing softmax, observed in closed loop for the first time rather than
   inferred open-loop.

### Honest bounds on this result — read them before quoting it

- ⛔ **Within-sim relative.** REF-C's own open-loop ADE is **1.5157** on these reconstructions vs
  **0.4728** on real footage — **3.21× OOD**. **Orderings survive; absolute rates do not.** Never
  quote a sim rate as a real-world rate.
- ⚠️ **The clusters are disjoint segments of ONE clip**, not independent episodes. It is the right
  estimator for the resampling unit available, and the unit is stated in the JSON so it is never
  mistaken for the 40-episode val bootstrap.
- ⚠️ **STRATEGIC is degenerate on a junction-free 20 s clip.** Flagship `route_head_eq_logged` reads
  **1.0000** only because the logged route takes one value and the head predicts one value — a
  constant predictor ties. **A junction scene is required before any strategic-accuracy claim.**
  `route_label_valid_rate` is 0.3778/0.3867; the rest is `gray_zone`. TTC is reported as **n = 0
  with its reason**, not dropped.
- **With-objects vs empty-road is NULL for both arms** (19 of 20 paired CIs contain zero), and it is
  bounded honestly to *distant* traffic — agents cover 0.02–0.4 % of frame at 40–45 m (~2.8 s gap).
  A close-following / cut-in scene is the discriminating follow-up, not a claim that vision is
  ignored.
- 🔴 **The renderer is a STEP FUNCTION of pose.** Identical pose is bit-exact, but 1e-9 → 1e-4 rad
  all give the same mean pixel delta with no growth (discrete blend-order ties among 3.1 M
  semi-transparent gaussians — not float precision). Decision-level cost on one identical window:
  **0.0000 m** in-process, **4.59 m** through the gRPC float32 pose round-trip, **6.65 m** under a
  0.1 px camera rotation. ⇒ **All production numbers must come from ONE numerical path, and
  bit-identical is the wrong acceptance criterion for a splat renderer.**

### And the open-loop reproduction, on real footage

`MEASURED` — run dir [`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-lan-refc-e0/`](TanitAD%20Research%20Hub/Architecture%20&%20Inference/Implementation/incoming/2026-08-03-lan-refc-e0/)
(REF-C-base 104.192 M, **859 windows / 39 val episodes**, 256 px square raster **asserted before any
forward pass**, paired episode-cluster bootstrap n_boot 2000).

- The 5-way defect reproduces open-loop: **accelerate 0/93 predicted, brake_stop 7/78** — and it is
  **INVARIANT to `nav_cmd`**, because both aux heads read the pooled feature only.
- **REF-C's route pathway is LIVE, not inert.** Sweeping `nav_cmd` over the label-reachable commands
  {follow, left, right} moves the trajectory **0.2416 m**; the bit-identical-input control is
  **exactly 0.0** (tol 1e-6). Verdict **RESPONSIVE**.
- ⛔ And supplying the **oracle** route makes lateral *worse* (`cross_mae` +0.0031 [+0.0001,
  +0.0063], separated) while buying nothing anywhere. ⇒ **REF-C's defect is the ARCHITECTURE, not
  the label.** Do not re-score the REF-C rows expecting a gain.
- ⚠️ `ep_00028.pt` was a truncated transfer, so this is **39/40 episodes** and **not
  window-comparable** to the published 881-window rows.

---

## The Thor story

**`tanitad-thor` — Jetson AGX Thor, aarch64, Blackwell sm_110.** It is a working compute node, an
eval node, and now a simulation node.

### The planning tick is UNDER budget on real trained weights

`MEASURED` — artifact [`TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-08-03-thor-batch9-engine/thor_d6_tick_intent_K20.json`](TanitAD%20Research%20Hub/Production%20&%20Optimization/Implementation/incoming/2026-08-03-thor-batch9-engine/)
— first tick measured **end-to-end** (encoder + strategic head + tactical head + 9-candidate fan +
`step_readout` decode + SE(2) + scoring) rather than composed arithmetically. 60 real held-out
windows / 12 episodes, **real step-29999 weights**, K = 20, budget 100 ms.

| configuration | p50 | p95 | vs budget |
|---|---:|---:|---|
| fp32 eager, serialised fan | 764.1 ms | 768.4 ms | 764 % |
| bf16 + engine, **caller not fixed** | 372.0 ms | 380.1 ms | 372 % |
| bf16 + eager batched, no engine | 204.4 ms | 205.7 ms | 204 % |
| **bf16 + dynamic 1..9 engine, BATCHED fan** | **60.3 ms** | **63.1 ms** | **60 % / 63 % — PASSES** |

**6.17× from the batching fix alone** (`speedup_B_to_C` = 6.169 in the artifact), 12.7× over the
fp32 eager baseline. Engines live at `thor:~/trt_deploy/` with a committed rebuild recipe.

⚠️ **Scope, stated plainly: this is a LATENCY + tactical-decision result, not a four-family accuracy
claim.** Two probes confirm `TacticalSelector` has **no production caller today** — the only
closed-loop driver is heads-only (~24 ms) — so this makes the *designed* path affordable rather than
speeding up what runs now.

### The renderer blocker is gone — and here is exactly how far it goes

`MEASURED` — code [`stack/experiments/nurec-gsplat/`](stack/experiments/nurec-gsplat/) and
[`stack/experiments/alpasim-gsplat/`](stack/experiments/alpasim-gsplat/).

- NVIDIA's NRE renderer is amd64-only, **but we do not need it**: `volume.nurec` is
  **gzip + MessagePack**, and **gsplat 1.5.3 renders it natively on aarch64 including the f-theta
  camera model**, at **16–28 ms per 1920×1080 frame** with the scene GPU-resident. Whole closed loop
  **0.09–0.21 s/step = 5–11 Hz** on Thor.
- Validation is by **gradient-NCC** against the scene's own shipped reference video (see the
  retraction below — PSNR and plain NCC are inadmissible on this clip).
- Also extracted from the USDZ: **`map.xodr`, `clipgt/lane.parquet`, `clipgt/obstacle.parquet`** —
  the strategic-map material the programme has been missing, since PhysicalAI-AV ships none.

⛔ **THE SCOPE LIMIT, and it must travel with every sim number.** This is **AlpaSim's renderer WIRE
CONTRACT satisfied by our renderer, driven by a TanitAD closed-loop harness. It is NOT
`alpasim_runtime.simulate`.** MEASURED on Thor: `alpasim_grpc`, `alpasim_utils` and `alpasim_wizard`
import; **`alpasim_runtime`, `alpasim_controller`, `alpasim_physics` and `utils_rs` do not**, and
`uv` is absent. ⇒ **There is NO AlpaSim collision / offroad / scene score for these runs.** The four
TanitAD metric families are what is measured instead. `cargo` is present, so finishing the runtime is
**bounded, not blocked**.

---

## ⛔ RETRACTED — never re-quote any of these

Read [`Project Steering/RETRACTION_LOG.md`](Project%20Steering/RETRACTION_LOG.md) before asserting in
a known class. These are the entries most likely to be re-quoted from an old summary:

| retracted | what is true instead | class |
|---|---|---|
| **PSNR / plain NCC on the NuRec night clip** — *"render validated at PSNR 16.758 dB"* | ⛔ A **WRONG** reference frame beats the correct one on both (PSNR 17.457 > 16.758; NCC 0.782 > 0.704) — every frame is a dark night street, so ~17 dB measures "both images are dark". ✅ **grad-NCC discriminates** (argmax = frame 0). The mapping is validated by **STRUCTURE, not photometry**. On the corrected `wxyz` quaternion layout: grad-NCC **0.3802** vs best wrong **0.2110**, margin **+0.1692**. | quoted a metric before checking it could discriminate |
| **The ISP / per-frame-photometry lead** | ⛔ **Dead.** The PPISP parameters were found (`.post_processings.0.ppisp.*`, 3594 views) and measured: exposure **exactly 0 for all 3594**, colour **identical** across views (std == 0), vignetting max \|α\| 0.0047 ⇒ **combined effect 0.18 %**, because `per_frame_ppisp_enabled: false`. **The scene ships no per-frame photometry.** ⭐ The real residual is **COVERAGE, not colour**: 79–81 % of the absolute error lives in pixels **no gaussian covers**. ⚠️ Turning the sky env-map on made the render **worse**. ⚠️ Open risk the agent raised against itself: the `f0` appearance-basis choice was selected **on PSNR**, so it rests on a retracted metric. | "the obvious missing piece" assumed to be an improvement |
| **The Thor precision / quantisation table** — *"precision gate PASS, error does not compound"* | ⛔ **Measured on a RANDOMLY-INITIALISED model fed `torch.randn`.** Quantisation error is a function of the *trained* weight/activation distribution; a random network has no outlier channels. 🔴 **We measured the OPPOSITE on real weights** (paper §7.10): post-pool `readout_head` collapses to cosine 0.566 under W+A INT8, costing **+0.0215 m ADE@2s** — past the pre-registered 0.02 m falsifier. ✅ **Latency survives** (weight-independent). Also: the "shipped" `thor:~/trt/predictor_fp16.plan` was **itself built from random weights** and was never deployable — superseded by `thor:~/trt_deploy/`. | numerics measured on random weights |
| **The ego-only `sitclf` swap** — *and, since 2026-08-03, **score-level image+ego fusion too*** | ⛔ **The ego-only swap was REJECTED by the PI ("no ego heads") and must not be proposed again.** ⛔ **Then a THIRD position superseded both candidates: labels may use ego, inference is VISION-ONLY** (rule 5 above) — so `late_fuse_scores` is **also** out, because score-level fusion is still ego-at-inference. **The deployable arm is `head_img` (image-only).** ⚠️ And the finding that started it is now itself suspect: the situation labels are derived from **ego dynamics** (`stack/tanitad/data/situations.py`), so the banked ranking `head_ego` 0.0697 > `head_img_ego` 0.0525 > `head_img` 0.0376 may measure **leak magnitude, not capability** — *flagged to verify from source, not assumed*. The separate `sc_train.py:143` scale bug (a 16-d PCA image block normalised by its own mean-abs against a 3-d ego block on a hand-set `EGO_SCALE`) is real but is no longer the fix. | first: dropping a modality because a fusion bug made it look useless. Then: **a head whose inference inputs include what the label was derived from** |
| **"REF-C's route pathway is INERT"** | ⛔ **REFUTED by the experiment written to test it** — it is **RESPONSIVE** (0.2416 m, control 0.0). Every published REF-C number held a **LIVE** input constant, not a dead one. Every premise was individually MEASURED; the *conclusion* was an **inference** that travelled through six documents wearing its premises' evidence class. **A chain of MEASURED links does not make the conclusion MEASURED.** | inference carried as measurement |
| **Every four-family ABSOLUTE RATE published before 2026-08-03** | ⛔ Wrong by **5×–25×**: `taniteval/four_families.py` hard-coded `DT_S = 0.1` while the windows it reads are the **sparse 4-waypoint 0.5 s grid**. Corrections: `speed_*` **/5.00**, `accel_*` **/25.00**, `yaw_rate_*` /6.48, `curvature_*` **/8.36** and `heading_*` /1.90 (those two via the `MIN_DS` mask alone — they are dt-invariant and were *still* wrong). ✅ **Every cross-arm comparison, rank and paired delta survives** (common factor). **FIXED + tested** (`infer_dt`, `prefer_dense`, `MIN_DS_MPS`, a `_grid` provenance block, 12 tests). | a hazard documented next to one caller instead of fixed at the shared function |
| **`overlapping_holdout_se` / the "8-split episode-disjoint jackknife"** | ⛔ Neither a jackknife nor a valid SE — **and it biases the point estimate too.** It manufactured the programme's one "load-bearing" hierarchy seam: `ctx→tactical` +0.0439 → true **+0.0148**. | a metric NAME is not a metric DEFINITION |
| **`flagship4b-phase0-30k` is "the deployed v1"** | ⛔ It is the **no-speed ablation control** (2.918 m). The deployed v1 is **`flagship4b-speedjerk-30k`**. The HF repo name invites this inversion. | inherited from prose |

Standing corollaries: **REF-A I-JEPA's val number is unusable** (~80 % of val leaked into train);
**never quote a learning-curve exponent bare** (window + R² + n, or it is inadmissible; below
R² 0.80 there is no quotable exponent at all).

---

## Where the program stands

**Open-loop** — canonical val `physicalai-val-0c5f7dac3b11`, 881 windows / 40 episodes, `full_set`
mean + episode-cluster bootstrap. `MEASURED`, provenance in
[`MODEL_REGISTRY.md`](Project%20Steering/MODEL_REGISTRY.md) §6:

| arm | params | ADE@2s `full_set` [bootstrap] |
|---|---:|---|
| Flagship v1 (`flagship4b-speedjerk-30k`, 4-brain WM) | 263.4 M | **0.4271** [0.3675, 0.4871] |
| REF-C-XL (anchored diffusion) | 251.9 M | 0.4714 [0.3896, 0.5556] |
| REF-C-base (anchored diffusion) | **104.2 M** | 0.4728 [0.3835, 0.5699] |
| *constant velocity — the floor* | 0 | *0.8377* |

Rank 1 is a **genuine three-way tie no paired test can order** (base − XL Δ +0.0013 [−0.0281,
+0.0316]) — held by a 263 M world model, a 252 M diffusion arm **and a 104 M diffusion arm**. Scale
bought nothing above 104 M on this corpus.

⚠️ **`0.4271` is a world-model-fidelity number, not a planning bar** — see the registry's own warning
before using it as a threshold.

**The honest position.** The world model is real and beats every trivial floor open-loop. What it
has not got is closed-loop competence: measured properly, the deployed flagship head **loses to a
104 M reference arm**, and the binding constraints are, in order — **longitudinal control**;
**closed-loop competence** (open-loop does not predict it); **generalization**; the **safety-metric
instrument gap** (off-road and collision rates need a map + reactive agents); and **data**
(13.13 h / 4.73 epochs, 42.6 % of clips with no turn, 0 % semantic scenarios). Full treatment in
[`PROGRAM_OVERVIEW.md`](Project%20Steering/PROGRAM_OVERVIEW.md) §7.

---

## The edge, in one table

| Axis | Incumbents | TanitAD target |
|---|---|---|
| Params | 0.3–10 B (Alpamayo-2 32 B, GAIA-3 15 B) | **sub-300 M** |
| Training data | 1000s h + labels / internet-scale pretraining | **tens of hours, zero perception labels** |
| Planning | pixels / diffusion / CEM (seconds–minutes) | **latent imagine-and-select (milliseconds)** |
| Hierarchy | flat or 2-level | **strategic / tactical / operative / fallback (4-brain)** |
| Self-knowledge | none | **imagination-error monitoring, regulation-ready (ISMR/DSSAD)** |

---

## Map

| Path | Content |
|---|---|
| `CLAUDE.md` | **Binding working agreements** — read first |
| `Project Steering/` | Constitution, model registry, retraction log, gate protocol, program overview, pre-registrations, reports, backlog |
| `stack/` | The runnable implementation — the `tanitad` package (4-brain world model, training, instruments), `scripts/`, `tests/`, and `experiments/` (incl. `alpasim-gsplat/`, `nurec-gsplat/`) |
| `taniteval/` | The evaluation harness — canonical `runner.py` CLI, `ci.py` (episode-cluster bootstrap), `four_families.py` |
| `TanitAD Research Hub/` | Per-discipline knowledge bases, the hypothesis ledger, and every run's `Implementation/incoming/<date>-<slug>/` **run directory** |
| `Benchmarks & Eval/` | Leaderboard, diagnostic framework, regulation trace, gate results |
| `Paper/` | The living paper (`TANITAD_PAPER.md`, v0.8) |
| `Ressources/` | UN ADS regulation, Deep Think analyses, reference designs |
| `tools/` | Repo/fleet utilities — `fleet_probe.py`, `registry_lint.py`, `ci_gate.py` |

**Quoting convention: cite a RUN DIRECTORY, not a bare number**, and verify by **loading** the
artifact — never by an exit code.

---

## Fleet

| Host | Role | Note |
|---|---|---|
| `tanitad-thor` | Jetson AGX Thor (aarch64, Blackwell sm_110) — edge inference, four-family evals, **and now the renderer** | Two venvs, **never mixed**: `tanitad-edge` (inference/AlpaSim) vs `tanitad-train`. Export **both** `PATH` and `CPATH` or you get misleading errors. |
| `tanitad-new` | v5f training | ⛔ **DO NOT TOUCH** |
| `tanitad-pod4` | `flagship-v1arch-v2bal-30k` training | ⛔ **DO NOT TOUCH** |
| `tanitad-pod2` | evacuated, idle | rescue payload banked at `_pod_backup/pod2-2026-08-03/` |

`INHERITED` (from `Project Steering/LOOP_STATE.md`, not re-verified in this pass). Operational traps
— MooseFS quota vs `df`, cgroup RAM vs `free`, `pkill -f` self-matching, `PYTHONPATH`, accumulated
`step_s`, pod-stack drift — are all in [`CLAUDE.md`](CLAUDE.md); read it before touching a pod.

---

## Quick start (dev machine)

```powershell
C:\Users\Admin\venvs\tanitad\Scripts\Activate.ps1
cd stack
pip install -e .[dev]
pytest -q          # must stay green before any commit
```

**Parity is sacred.** The canonical train corpus is `physicalai-train-e438721ae894` (2376 episodes)
with skip-hash `f09e44db`. Anything that re-selects episodes breaks cross-arm comparability and must
be refused.

**Agents:** stage with `git add`, **never commit and never push** — the orchestrator commits. End
with a **deliverable manifest** naming every artifact and where it lives. Full contract in
[`Project Steering/AGENT_OPERATING_STANDARD.md`](Project%20Steering/AGENT_OPERATING_STANDARD.md).
