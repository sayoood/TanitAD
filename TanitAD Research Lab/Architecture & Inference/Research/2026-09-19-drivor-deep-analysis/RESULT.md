<title>DrivoR deep analysis</title>

# DrivoR (*Driving on Registers*, valeo.ai, CVPR 2026) — deep analysis, successors, and what it means for TanitAD

`Architecture & Inference · 2026-09-19 · PI request. Primary read in FULL from the banked PDF (lib 2601.05083 v2, 17 pp, local pypdf).`
`Successors read in full the same way: TOAD 2606.07170 (15 pp), CLOVER 2605.15120 (34 pp), DriveZero 2609.06055 (32 pp). Repo README github.com/valeoai/DrivoR read 2026-09-19 (Apache-2.0).`
`One 0-GPU measurement (params + MACs, CPU; the dev-box GPU was busy with a live training run and was not touched): raw/encoder_budget.json.`

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **1** | ⛔ **DrivoR itself does NOT score 56.3 EPDMS — RE-CONFIRMED, not newly found.** DrivoR's navhard-two-stage scores are **48.3** (navtrain only, ViT-S), **52.3** (+65k SimScale; README says 52.2) and **54.6** (+134k SimScale). **56.3 is DrivoR + TOAD** (TOAD Table 2: *"DrivoR 54.6 → + TOAD 56.3"*). ⚠️ **The programme already knew this:** `Opponent Analysis/Research/2026-09-02-cw1-resolution-and-evidence-integrity/` (*"56.3 is DrivoR + TOAD"*), register guideline **T-4** (*"never quote … the DrivoR 56.3"*), and `Benchmarks & Evals/Research/2026-09-15-epdms-stage1-fingerprint/` F2 (54.6 is the +134k variant). ⛔ **What is still wrong: the LIVE page `Benchmarks & Eval/LEADERBOARD.md:1583` still reads "DrivoR 56.3 EPDMS navhard"** (inside its INHERITED/orientation block), as do several KB lines and weekly reports. *(First draft of this package presented this as a new correction; a truncated `grep … \| head -30` had hidden the 09-02 package. Same re-find class as `RETR-2026-09-19-POSTED-LIMIT-REFIND`, second instance today.)* | MEASURED against primaries (DrivoR Tab. 3/14, TOAD Tab. 2/6, README) |
| **2** | ⭐⭐⭐ **The largest measured lever in DrivoR is ENCODER PRETRAINING, and our pre-refcv6 REF-C trunks had none.** DrivoR Tab. 4a (navval PDMS): random init **70.1**, ImageNet-21k **87.5**, DINOv2 **90.0**, i.e. **+19.9** for pretraining. The next largest are **+9.9** (64 trajectory queries vs 1), **+6.1** (one token per trajectory), **+5.6** (LoRA vs frozen) and **+5.3** (disentangled scorer); compression moves ≤ 0.9 and register count ≤ 0.3. REF-C's in-repo `ResNetEncoder` (V2-99 class, **90,458,632 params**, measured today) is randomly initialised. refcv6's PI-directed switch to an ImageNet `resnet101.a1_in1k` (2026-09-16) goes in exactly this direction | PUBLISHED (full text) + MEASURED (our param count) |
| **3** | ⭐⭐⭐ **The scorer is the part that transfers, and TOAD proves WHY.** DrivoR's scorer is a 4-layer transformer decoder that sees each trajectory **re-embedded from its decoded poses with a stop-gradient** ("disentanglement"). It predicts the **6 PDMS sub-scores** separately. Ablation: shared branch 84.7 → separate 86.8 → disentangled 90.0; single total score 88.2 vs 6 sub-scores 90.0. ⭐ TOAD then shows **only this kind of scorer survives being used as a search reward**: re-used as a CEM objective, a fixed-vocabulary scorer (GTRS) drives iPad **34.7 → 23.9** EPDMS while DrivoR's scorer drives it to **49.8** | PUBLISHED (full text, both) |
| **4** | ⭐⭐⭐ **Our REF-C scorer is exactly the type TOAD shows failing.** refcv5-v2: scorer **1,145 parameters**, selecting among a **fixed 117-anchor vocabulary**; anchor-selection accuracy **0.5271**; `--sel-refined` measured **0.0259 m WORSE** on refcv4b (*"the ranking head was never trained to rank"*); and the arm's T1 `os` ADE **0.3079 m loses to the kinematic echo control `ha0_ext` 0.2874 m** (+0.0205 [+0.0043, +0.0390], FAILED its bar) | MEASURED (MODEL_REGISTRY §4.8, T1 open loop) |
| **5** | ⭐⭐ **DrivoR's encoder recipe fits our geometry and costs about what refcv6's trunk costs.** At one front camera, 252×644, 3 frames: DINOv2 ViT-S/14 + 16 registers = **21.9 M params, 54.4 GMAC** (linear + conv) + **≈ 19.7 GMAC** attention, **48 scene tokens** out. REF-C in-repo trunk: **90.5 M, 44.9 GMAC**. refcv6 primary `resnet101` ×3 frames: **42.5 M, 76.4 GMAC**; `resnet34` ×3: **21.3 M, 35.9 GMAC** | MEASURED (params; linear+conv MACs, `torch.utils.flop_counter`) + ESTIMATED (attention, analytic) |
| **6** | ⭐⭐ **DrivoR-Scale's documented failure is LONGITUDINAL, which is our family.** DriveZero Fig. 5: *"the lead vehicle brakes suddenly. DrivoR-Scale keeps its speed and collides with it."* And SimScale data **destroys extended comfort**: Stage-2 EC **76.2 → 38.8 / 44.7** (+65k / +134k), while EPDMS rises | PUBLISHED (DriveZero full text; DrivoR Tab. 3) |
| **7** | ⭐⭐ **No common benchmark exists between us and DrivoR.** It is NAVSIM (PDMS/EPDMS) + HUGSIM; we are PhysicalAI T1 open loop, four families. ⛔ No level comparison is admissible (V-5); only structural comparisons are made below | analysis |

---

## 1 · What DrivoR is (from the primary)

**Architecture (Fig. 1–2, §3).** Three plain transformer blocks, no BEV, no large trajectory dictionary:
1. **Perception encoder.** DINOv2 **ViT-S/14** applied per camera, **LoRA rank 32**. **R = 16 learnable registers per camera** are appended to the patch tokens; only the registers' final-layer outputs are kept ⇒ **4 cams × 16 = 64 scene tokens** (vs **16k** patch tokens uncompressed, a **250×** reduction). Registers are *per camera*, so tokens are camera-aware. The 4 original DINOv2 registers are discarded.
2. **Trajectory decoder.** 4-layer decoder, d = 256, FFN 1024, **64 learned trajectory queries**. Ego status (poses, velocities, accelerations, **driving command**) is encoded and **added to every query**. Each query decodes to a **whole trajectory from ONE token** (x, y, θ per pose). Loss: **winner-takes-all L1** (min over the 64), plus an optional second "accelerated" target resampled from a longer horizon T′ > T.
3. **Scoring decoder.** Same shape. Each decoded trajectory is **re-embedded by an MLP from its poses** and enters as a query, with a **stop-gradient to the trajectory decoder**. The scorer's gradients still train the encoder. **6 MLP heads** predict the PDMS sub-scores (NC, DAC, DDC, TTC, EP, Comf.) with BCE against the NAVSIM oracle. At inference the sub-scores are **re-weighted without retraining** (behaviour control: a "safety-oriented agent" tuned for NAVSIM-v2).

**Size.** **~40 M total** (Table 11: **41 M = 24 M backbone ∥ 17 M rest**; 351 GFLOPs = 350 ∥ 1; 0.5 GB peak; **110 ms/forward** on an A100, batch 1, no quantisation, 4 cams at 672×1148). GTRS with ViT-L: 321 M / 400 ms.

**Training.** 4× A100; AdamW, lr 2e-4, batch 16, cosine; ~1 h/epoch (navtrain), ~1.5 h/epoch (navtrain+navval). **25 epochs** for NAVSIM-v1 on train+val; **10 epochs** for NAVSIM-v2, because EPDMS on `warmup-two-stage` *falls* with more epochs and with more data.

### 1.1 Data — samples and hours

| corpus | amount | driving time | class |
|---|---|---|---|
| OpenScene (nuPlan re-release) | pool for NAVSIM | **120 h at 2 Hz** | PUBLISHED (NAVSIM `2406.15349`) |
| `navtrain` | **103k** samples (NAVSIM paper) — ⚠️ DrivoR README says **"Train 85k / Trainval 103k"**; the two sources disagree and neither explains it | **≈ 11.8–14.3 h** of distinct 2 Hz keyframes (85k–103k × 0.5 s), each with 1.5 s history + 4 s future | ESTIMATED from PUBLISHED sample counts |
| SimScale add-on | **65k / 134k** used by DrivoR (of 147k recovery-based + 237k planner-based scenes generated) — 3DGS re-renders of navtrain with **pseudo-expert labels from PDM-Closed** | ≈ +9 / +18.6 h-equivalent at 0.5 s/sample, **synthetic, not new driving** | PUBLISHED (SimScale `2511.23369`) + ESTIMATED |
| **ours (B1, refcv3–v5)** | **4,572** train clips / 141 eval (non-parity) | **≈ 25.4 h** at 10 Hz (20.0 s/clip from the PhysicalAI-derived card: 4,800 clips ≈ 26.7 h) | INHERITED card ratio + ESTIMATED |
| **ours (parity)** | **2,376** episodes | **≈ 13.2 h** | ESTIMATED, same ratio |

⇒ **DrivoR's real-data diet is the same order as ours (≈ 12–14 h vs ≈ 13–25 h).** Its lead comes from pretraining, 4 cameras and an oracle-scored target, **not from more driving**.

### 1.2 Results (primary, all single-run, no CIs)

| benchmark | DrivoR | notes |
|---|---|---|
| NAVSIM-v1 navtest PDMS | **93.1** (train) · 93.7 (trainval) · 94.0 (+65k) · **94.6** (+134k) | human 94.8; PDM-Closed 89.1; RAP-DINO (10× data) 93.8 |
| NAVSIM-v2 navhard-two-stage EPDMS (post-#151 fix) | **48.3** · 52.3 (+65k) · **54.6** (+134k) | PDM-Closed 56.6 (v3 snapshot); pre-fix DrivoR 45.3 |
| HUGSIM zero-shot (their **patched** simulator) | RC **49.8**, HD-Score **35.7** | UniAD 32.7, LTF 23.7. ⚠️ Scores come from their corrected code and are not comparable with official-code numbers |

### 1.3 Ablations that carry the paper (navval PDMS, one run each)

| lever | range | effect |
|---|---|---|
| encoder init (random → IN-21k → DINOv2) | 70.1 → 87.5 → **90.0** | **+19.9** ⭐ largest |
| scorer: shared → separate → disentangled; 1 score → 6 sub-scores | 84.7 → 86.8 → **90.0**; 88.2 → 90.0 | +5.3; +1.8 |
| trajectory queries 1 → 64 | 80.1 → **90.0** (plateau at 64) | +9.9 (multi-hypothesis WTA) |
| single-token vs multi-token trajectory | 83.9 → **90.0** | +6.1 |
| compression: pooling / decoder / **registers** / none (16k tokens) | 89.7 / 89.3 / **90.0** / 90.2 | ≤ 0.9 — **registers ≈ no compression at 250× fewer tokens** |
| fine-tuning: frozen / full / **LoRA** | 84.4 / 88.4 / **90.0** | LoRA > full FT |
| registers per camera 5 / 8 / 16 / 32 | 89.7 / 89.7 / **90.0** / 89.8 | flat, inside noise |
| extra accelerated target (T′) | v1 +0.6, **v2 −1.6** | helps progress, hurts OOD safety |

---

## 2 · Successors (all read in full)

| work | relation | headline | what it teaches |
|---|---|---|---|
| **TOAD** `2606.07170` (valeo.ai, 2026-06) | DrivoR's scorer used as a **reward**; control-space CEM (bicycle model: accel, yaw-rate; needs **v0**), warm-started at the planner's pick, trust-region anchor penalty, closed-form comfort penalty, returns best of {mean, base} | DrivoR 54.6 → **56.3** EPDMS; lifts all 6 base planners on navhard (iPad +43.6 %); **+1.9 ms** at K=5, M=64 | ⭐ **A scorer is only a usable reward if it generalises OFF its training proposals.** Vocabulary-fit scorers collapse (GTRS search 23.9). Build the scorer into the planner, and search is nearly free |
| **CLOVER** `2605.15120` (Tsinghua AIR) | DrivoR-style generator + scorer; evaluator-filtered pseudo-experts, set-coverage supervision, conservative scorer-mediated self-distillation | 94.5 PDMS, 90.4 EPDMS (navtest), **48.3 navhard = ties DrivoR** | proposal-set diversity is the lever; its navhard gain is **zero** |
| **DriveZero** `2609.06055` | *"Following DrivoR, camera features are compressed using 16 register tokens per camera"*; multi-VFM consolidated encoder + 5.70 M privileged PPO teacher | **57.1** navhard (Scale) / 51.5 (base) | the register compressor is now a community default; the headroom above DrivoR came from the **teacher and the encoder** |
| GTRS-DrivoR (in the DrivoR paper) | DrivoR's ViT-S + registers under GTRS's vocabulary scorer | 45.8 vs GTRS-V2-99 45.4, **3× throughput** | the compressor transfers; the scorer is what separates 45.8 from 48.3 |
| COMPACT-VA `2606.07464` (NVIDIA) | *not* a DrivoR descendant — long-context compression, conditional VQ-VAE on trajectory + intent | 3.3× speed, 2.7× memory | addresses DrivoR's admitted gap: **no history** (scan only) |

---

## 3 · DrivoR vs TanitAD — structural comparison (no shared benchmark)

| axis | DrivoR | TanitAD (REF-C line, refcv5-v2) | consequence |
|---|---|---|---|
| params | **41 M** (24 ∥ 17) | **108.3 M** (core 106.1 M) | we are 2.6× larger, and not better-pretrained |
| encoder | DINOv2 ViT-S, **pretrained**, LoRA | in-repo ResNet, **random init** (refcv6: ImageNet ResNet-101) | **the +19.9-PDMS lever** |
| cameras / res. | **4** (F, FL, FR, B), 672×1148 | **1** front, 256×640 cylindrical | a data limit, not a design choice (PhysicalAI front-wide) |
| history | **none** (their admitted failure case) | **3 frames** (refcv6: history required, PI 09-16) | ✅ our advantage |
| ego inputs at inference | poses, **velocities, accelerations**, driving command | **v0 only** + nav command (PI 2026-08-03 vision-only rule; v0 ruling 09-02) | ⚠️ DrivoR carries the NAVSIM ego-status shortcut and **never ablates it** |
| proposals | 64 learned queries, WTA, 1 token per trajectory | 117 fixed anchors + diffusion refinement | vocabulary ⇒ TOAD's failure mode |
| scorer | 4-layer decoder, stop-grad re-embedding, **6 oracle sub-scores** | **1,145-param** head, no oracle sub-scores | ⭐ the part to adopt |
| supervision | human trajectory + **oracle scorer** (needs map + agents) | human trajectory + v7.2 labels (no map oracle on PhysicalAI) | now partly available: `obstacle.offline` (NC, TTC) + **SAM3 maps** (DAC) |
| world model | none | the programme's thesis | DrivoR has no imagination at all |
| hierarchy | none (flat) | strategic / tactical / operative | route acc 0.7708 from a 771-param head |
| evaluation | NAVSIM PDMS/EPDMS + HUGSIM (closed loop) | T1 open loop, **four families**, paired episode-cluster CIs, replicate seeds | our instrument is stricter; theirs has closed loop |

---

## 4 · Weaknesses of DrivoR (and where we are already better)

1. ⛔ **No variance anywhere.** Every ablation is one training run on navval, and most differences (registers 89.7–90.0, compression 89.3–90.2) sit inside what our own replicate floor would call noise (`H-ESTIM-SEED-1`: 14.3 % false-separated on a same-seed replicate).
2. ⛔ **NAVSIM-v2 model selection on 7 scenes that intersect the test split.** Epochs, data mix and the inference sub-score weights (Tab. 10: NC 10, DAC 13, DDC 6, TTC 14, EP 15) were chosen on `warmup-two-stage` (7 scenes). The authors state it intersects `navhard-two-stage` and was cleared by the benchmark authors. It is still **test-adjacent tuning**.
3. ⚠️ **Ego-status dependence never measured.** Velocities, accelerations and the command enter every query. On NAVSIM an ego-status MLP alone scores 65.6 PDMS. We cannot tell how much of 93.1 is the camera.
4. ⚠️ **No temporal context.** The paper's own failure case (Fig. 10: undercutting a turn toward wrong-way driving) is attributed to single-frame input.
5. ⚠️ **The headline requires distilled PDM-Closed.** 52.3 / 54.6 need 65k / 134k SimScale scenes labelled by PDM-Closed, so part of the gain is imitation of the privileged planner. It also **costs extended comfort** (76.2 → 38.8 / 44.7).
6. ⚠️ **Longitudinal failure** under sudden lead braking (DriveZero Fig. 5). ADE and EPDMS both under-weight exactly what our longitudinal family measures.
7. ⚠️ **The oracle scorer needs an HD map and GT agents.** It exists on NAVSIM, not on PhysicalAI, and not on a car.
8. ⚠️ **HUGSIM numbers come from a patched simulator** (comfort bounds, heading fix). Correct in spirit, but not comparable with anyone's official-code number.
9. **Deployment:** 110 ms on an A100 unquantised; nothing on an edge SoC.

---

## 5 · Developing it for TanitAD — the design, and what to test first

**"DrivoR-T"**: keep what DrivoR measured as load-bearing, and add what it lacks and we need.

| component | DrivoR | DrivoR-T (proposed) | why |
|---|---|---|---|
| trunk | DINOv2 ViT-S + 16 regs/cam, LoRA | **pretrained** trunk (DINOv2/v3 ViT-S + 16 regs **per frame**, or refcv6's ResNet-101), 3-frame history, LoRA | Tab. 4a's +19.9; history fixes their Fig. 10 |
| scene tokens | 64 | 48 (16 × 3 frames) → the **world-model state** | registers ≈ uncompressed at 250× fewer tokens |
| world model | — | predict future **register tokens** against a frozen-DINO / SAM3-BEV target, with **exclusive routing** (the action head reads only WM-shaped tokens) | FS19-1 / FS19-3 (World Tokens, DeepSight) |
| proposals | 64 WTA queries | 64 WTA queries **+** the 117 anchors as extra candidates | removes the vocabulary as the only source |
| scorer | 6 NAVSIM sub-scores | **disentangled** (stop-grad re-embed) scorer predicting **PhysicalAI oracle sub-scores** — NC + TTC from `obstacle.offline`, DAC from **SAM3 maps**, longitudinal (speed error vs posted limit from the nuPlan-trained head), lateral comfort from kinematics | our four families become the scorer's heads; behaviour weights become the PI's knobs |
| inference | argmax score | **TOAD** control-space CEM (a, ω; v0 is admissible), K = 5, M = 64, anchor + comfort penalties | +1.9 ms; never worse than the base under the scorer |
| inputs | ego pose/vel/acc + command | **v0 + nav only** | binding rules |

### 5.1 Pre-registered experiments (proposed, unranked — the Master Mind ranks)

| id | experiment | committed read | cost |
|---|---|---|---|
| **DR-1** | **Scorer-generalisation instrument (TOAD's diagnostic, on our data).** For the eval-139 clips that have SAM3 maps + `obstacle.offline`, compute oracle NC / TTC / DAC for (a) REF-C's own 117 anchors and (b) control-space perturbations of them. Score the refcv5-v2 selector on both | selector rank correlation with the oracle drops by ≥ 50 % from (a) to (b) ⇒ our scorer is vocabulary-bound (TOAD's failure mode) and **must be replaced before any test-time search**; < 20 % drop ⇒ TOAD is directly usable on REF-C | 0 GPU (oracle) + a CPU/4060 scorer pass |
| **DR-2** | **Disentangled oracle-sub-score scorer on refcv6** (stop-grad re-embed, 4-layer decoder, sub-scores from DR-1's oracle), 2 arms + replicate | anchor/selection accuracy > 0.5271 **and** T1 `os` beats `ha0_ext` (the bar refcv5-v2 failed) beyond the replicate floor, with no family regressing | 2 arms + replicate on the PI pod |
| **DR-3** | **TOAD on REF-C** with DR-2's scorer, K ∈ {5, 20}, 5 inference seeds | T1 ADE and the longitudinal family improve beyond the **inference** seed floor; the returned plan never scores below the base (guaranteed by construction, so assert it) | eval only |
| **DR-4** | **Encoder A/B at matched budget**: refcv6 ResNet-101 (76.4 GMAC, 42.5 M) vs DINOv2 ViT-S + 16 regs/frame (≈ 74 GMAC, 21.9 M), both pretrained, both LoRA/fine-tuned per their best recipe | ViT+registers matches ResNet-101 within the replicate floor on all four families at ≤ ½ the params ⇒ adopt registers (they also give the 48-token WM state) | 2 arms + replicate |
| **DR-5** | **G3 external baseline**: reproduce DrivoR (released code + GitHub-Release weights, Apache-2.0) on NAVSIM, then an **ego-ablated** DrivoR (v0 + command only) | the ego-ablated drop is the size of NAVSIM's ego shortcut for this model class. It sets our admissible target on navhard | ⛔ **needs NAVSIM sensor data (large download) — PI decision** |

**Order (Rule Zero rule 5, largest measured effect first):** DR-4's trunk half is already under way as refcv6. **DR-1 is the cheapest discriminating experiment and blocks DR-3**, so it runs first.

---

## 6 · Provenance
* Primaries (Library keys): `2601.05083` DrivoR v2 · `2606.07170` TOAD · `2605.15120` CLOVER · `2609.06055` DriveZero · `2511.23369` SimScale · `2406.15349` NAVSIM · `2606.07464` COMPACT-VA (abstract-level, banked today).
* README: `github.com/valeoai/DrivoR`, read 2026-09-19 via fetch (CVPR 2026 acceptance; results table; **"The model weights are provided in GitHub Releases"**; Apache-2.0). Not banked (not a PDF).
* Measurement: `stack/scripts/drivor_encoder_budget.py` → `raw/encoder_budget.json`. **Control:** the paper geometry (4 × 672×1148, ViT-S + 16 regs) reads **339.3 G** multiply-adds for linear + conv, vs the paper's fvcore **350** (3.1 % apart). The counter recorded only `aten.addmm` + `aten.convolution`, i.e. it **excludes attention exactly as fvcore does**. **Independent check:** REF-C trunk **90,458,632** params = its docstring's figure. Attention MACs are analytic, `2·N²·d·L` with N = 845 tokens, d = 384, L = 12, ×3 frames ≈ **19.7 G** (ESTIMATED). ⚠️ Latency **not** measured: the 4060 was running a live training (`refc_v3_train.py`, a8-occupancy, PID 21724), and the Lab does not load a GPU in use.
* Our numbers: `Project Steering/MODEL_REGISTRY.md` §4.6 (refcv4b), §4.8 (refcv5-v2, T1 open loop, 4,823 windows / 141 episodes, paired episode-cluster bootstrap); `stack/tanitad/refs/refc.py:304–349` (trunk config).
