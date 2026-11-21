# DRAFT AMENDMENT 7 to PREREG_V8F_STOPPERC: Q5, the 416x1024 source. Probe first, both outcomes committed

**Status: DRAFT for the Master Mind to freeze** (2026-11-15). It must be frozen, with a `.sha256`, before any 416x1024 token of a training clip exists and before any 416 number is scored. Card `PREREG_V8F_STOPPERC.md` (sha256 `70c75176…8877`) and Amendments 1-6 stay unchanged. Authority: the PI's Q5 answer, *"Approve reencoding"* (`stopperc/PI_Q5_APPROVAL_2026-11-15.md`). That approves the frames and tokens. It does not approve this design or any spend beyond the re-encode. Facts behind every path and rate: `F:/TanitAD-scratch/q5_416/SCOPE_416.md` (cited as SCOPE §n).

**Data state at drafting.** No 416 token of any training clip has been computed. The only 416 tokens ever computed are FULLRES's eval-139 PCA-128 int8 store, deleted 2026-11-13 (`dagland/LOOP_STATE_V8PROOF_2026-11-12.md:167`). The drafter has read FULLRES's eval-only results (SCOPE §6). The bar and the arms below come from the card's RES-ADDS (§4) and the brief, not from those numbers.

## 1. Why this step comes first (MEASURED)
* **Three perception levers each failed their committed read on this card:**
  * S0-P **FLAT** (`VERDICT_S0P.md`): V160+E − V40+E −0.0063 [−0.0276, +0.0157] (linear); V40+E − E +0.0091 [−0.0315, +0.0477].
  * Arm A **PERC-NULL** (`stage1_A/read/VERDICT_STOPPERC_S1.md`): G-AUROC +0.0054 / +0.0034 against a floor of 0.0398.
  * S0-L **NO-GAIN**: dAUC +0.0095 [−0.0289, +0.0525] (`PI_Q5_APPROVAL_2026-11-15.md`).
* **The card's next step.** Its PERC-NULL row says: *"otherwise the 416x1024 source (Q5)"*.
* **FULLRES is not this test.** It cross-fitted within the 139 eval episodes, never pooled 416 tokens to the trainer's interface, and used 416-resampled frames for its low arm (SCOPE §6). It sets the prior and decides nothing here.
* **Cheapest discriminating step.** An S0-P-style probe costs the frame build plus ~1 h of pod GPU (SCOPE §5a). A training arm costs ~10.7 h of pod GPU for the dense cache plus box-2 runs (SCOPE §5b). The probe goes first.

## 2. Stage Q5-0: the probe (zero training of T)
### 2.1 Inputs, fixed here
* **Frames, 416.** The production builder `build_v2ep_wide.py --height 416 --width 1024` (`calib.cylindrical_rectify` unmodified, per-clip intrinsics), from HF `Sayood/tanitad-v7-training-corpus` at revision `a0cf20df…` (SCOPE §2). The pod builds them shard by shard: it stages the mp4s, builds, verifies and deletes. Eval: the pod builds all 141 eval clips, and E:'s eval139 copy is the cross-check (C6a).
* **Frames, 256 (the comparator).** The pod's existing native 256 v2ep (`/root/data/v2ep`, ledger-checked by sha256, SCOPE §2).
* **Encoder.** DINOv3 ViT-L/16, bf16, on the pod's RTX 4000 Ada, with the B-prime steps: PNG decode, /255, the processor's mean / std, batches of 8, CLS + 4 registers dropped. A sibling of `encode_episode` (`stack/scripts/dinov3_fp8_encode_ship.py:108-130`) is identical except the token-count assert (640 or 1,664). Pinned by C8.
* **Both resolutions run through the same sibling with the same sparse NOW-frame batching.** Only the input pixels and the pooling cells differ between the 416 and 256p arms.
* **Pooling.** P-A, the angle-matched area pool (SCOPE §4). Cells are 6.4 x 6.4 patches (4x10) and 3.2 x 3.2 (8x20) inside patch rows 0.2-25.8, pooled in fp32 and then quantised to fp8 e4m3fn as the builder does (`pull_encode_pooled.py:11-17`). At 256 the same function runs with integer cells. `mean` is the area-weighted mean over the same field, stored fp16.
* **NOW frame.** Cache step t = the window start (README_S0P reading 5), which is raw frame 2t (`pull_encode_pooled.py:82`).
* **Samples.**
  * **S1** is S0-P's own sample: `default_rng(20261111)`, sha256 `843e4017…` (`C:/Users/Admin/s0p_state/sample_sha.txt`).
  * **S2** is the same rule (`s0p_lib.draw_sample`, `s0p_lib.py:37`) with seed `20261115`. It is the replicate of the training sample that S0-P did not have (VERDICT_S0P, "Not measured").
  * **S3** (seed `20261116`) is built only on SAMPLES-DISAGREE.
* **E.** S0-P's 18 inputs (`p10f` 8, `nav` 5, `limit` 4, `limit_ok` 1), with `limit` / `limit_ok` the **causal** bucket (Amendment 1), taken from C0's re-extraction. E−limit is the first 13.
* **Eval windows.** STOPCUE's: 2,365 windows, 131 positives, 139 episodes. The 139 episodes are exactly E:'s eval139 clips (SCOPE §0.1).
* **Models.** `s0p_fit.py` unchanged: per-token PCA dims {16, 32, 64}, linear C ∈ {0.001, 0.01, 0.1, 1}, MLP hidden ∈ {64, 256} with alpha ∈ {1e-4, 1e-2}, 5-fold `GroupKFold` by clip inside the training sample, refit, then score eval once (card §8).

### 2.2 Arms
| arm | committed or beside | per sample |
|---|---|---|
| E | committed (reference) | S1, S2 |
| **V40@256p+E**, **V160@256p+E** (pod 256 tokens, through the sibling) | committed (comparators) | S1, S2 |
| **V40@416+E**, **V160@416+E** | committed | S1, S2 |
| V40@256b+E, V160@256b+E (box-2 tokens, S0-P's banked scores for S1) | beside: the route term (SCOPE §7 measured rel-L2 0.022-0.026 between pod and box-2 tokens) | S1 (S2 from C0's run) |
| M@416+E vs M@256p+E | beside | S1 |
| V40@416 under P-B (full field) vs P-A | beside | S1 |
| V40@416+(E−limit) vs E−limit | beside (Q7: the vision gain without the `limit` channel) | S1 |
| V416tok+E (13x32, P-E, linear only) vs V40@416+E | beside: does the pooled interface hide a gain? | S1 |
| HEAD-UNDERUSES at 416: V40@416+E − T (BAL-NULL_s0 / _s1 `goal_prob`, Amendment 2 rule) | beside | S1 |

### 2.3 Estimator
`taniteval.ci.stat_cluster_bootstrap` over the 139 eval episodes, B 2,000, seed 0. Each sample takes **one** call over all its score arrays, as `s0p_fit.read_all` does (`s0p_fit.py:340-383`), so every paired difference is taken on the same resample. That interval answers *"another draw of episodes"*. S2 answers *"another draw of the training sample"*. The probes are deterministic (lbfgs; MLP `random_state` 0), so there is no inference-seed term. T's training-run floor is not a probe quantity.

### 2.4 Committed outcomes
For grid g ∈ {40, 160}, model family f ∈ {linear, MLP} and sample s, define two predicates. Both arms in each difference come from the same family and sample.
* **R(g,f,s)** (resolution): paired dAUC(Vg@416+E − Vg@256p+E) has point **≥ +0.020** and CI lower bound **> 0**. This is the card's RES-ADDS bar.
* **A(g,f,s)** (above ego): paired dAUC(Vg@416+E − E) has CI lower bound **> 0**. The point is printed beside the card's AUROC floor scale, 0.037.

| outcome | rule | licenses (committed now) | does NOT license |
|---|---|---|---|
| **RES416-ADDS-40** | for some f: R and A at g = 40 hold at **both** S1 and S2 (or at 2 of S1, S2, S3) | **Stage Q5-1** (§3): the dense 416 cache and arms Q5_s0, Q5_s1 with no architecture change. Box-2 time needs the PI's go (QUESTIONS Q1) | "resolution moves T": only Q5-1 decides that |
| **RES416-ADDS-160** | not ADDS-40, and the same rule holds at g = 160 | the card's R arm at 416 (`n_tok` 160 from M2b, `--init-pos-upsample`, NEW). That is an architecture change, approved in card §8 Q1 only on the 256 RES-ADDS, so it **goes back to the PI** with its cost: 2 x 60-120 min box 2 (card §5) plus the 8x20 cache (~89 GB) on box 2's USB HDD. Nothing runs before that go | a 4x10 arm |
| **SAMPLES-DISAGREE** | for some (g, f), R ∧ A holds at exactly one of S1, S2, and no ADDS row holds | build S3. Each ADDS row is then read as "2 of 3". There is no fourth sample | anything at 2 samples |
| **RES416-FLAT** | otherwise | **The stop-perception line ends with this row.** Four eliminations, each logged: the 256 grids (S0-P), a trained adapter with a stop auxiliary (A), a trained encoder (S0-L) and 2.6x the pixels on the same field (Q5). **Next lever, by the card's own logic (§1, §3):** the gap is coverage of stop points, and the cue a readout reaches is already in T's non-vision inputs. Weighting moves the operating point along an unchanged ROC, and hardening recovers 3-6 %. What would move coverage is an **input that carries the stop location**. On PhysicalAI-AV that exists only as an ego-future label (`build_v9_labels.py:937-942`): admissible as a label, never as an input. A map stop line exists on nuPlan / NAVSIM (REFe `AMENDMENT_6_STOP_LINE.md`, card §3). The next arm is therefore a **map-carried stop-line input on that rig, under its own card**. It must pass the goal-input admissibility check (not computed from the situation classifier). The corpus and rig choice is the PI's (QUESTIONS Q2) | "vision cannot carry the stop cue": these are frozen, pooled features, and the unpooled fine-token reader FULLRES named (`LOOP_STATE_V8PROOF_2026-11-12.md:168`) stays outside this card |

The beside reads decide nothing. The prior is a HYPOTHESIS: **LOW** for RES416-ADDS-40. At 256, more tokens added nothing (S0-P, V160 − V40), vision did not separate above E (+0.0091), and FULLRES's eval-only resolution pair read P1 −0.0012 [−0.0066, +0.0043].

### 2.5 Steps and cost (ESTIMATED, SCOPE §5a)
1. **Code + tests** (§6), then a smoke on 3 eval clips (pod GPU, minutes).
2. **C0, box-2 CPU, ~0.5 h:** S0-P's extractors re-run on box 2's 256 caches for S1, and the same extractor with S2's seed through a wrapper. `s0p_lib` stays unedited.
3. **Pod, ~5.5 h wall:**
   * 416 frames for the sample clips (~4,200-4,512) and eval-141: CPU-bound, 4.3-5.4 h.
   * The NOW-frame encode at 416 and at 256p, 34,365 frames each: ~1 h of GPU, overlapped with the build.
   * Output: S0-P-format dirs (`s0p_lib.FIELDS`). E, y, clip, ws and eid are copied from C0 bitwise, and v40, v160 and mean are replaced.
4. **Fits, dev-box CPU, ~4-5 h at 2 threads:** `s0p_fit.py prep` and `fit` per dir, unchanged. The E arms are fitted once per sample.
5. **Read:** `q5_read.py`, on one bootstrap per sample, applies the controls first, then the outcomes.
* **Cost by budget.** The pod work is inside the PI's re-encode approval. Box-2 and dev-box CPU are analysis, as in S0-P.

## 3. Stage Q5-1: the training arm (only on RES416-ADDS-40)
* **Cache.** The dense 416 cache for train-4,572 and eval-141, built on the pod with the B-prime pipeline: `cache_pod.py`'s steps, with the sibling encoder and P-A pooling. Keys, shapes and the manifest-row schema equal box 2's `cache_full`. Kinematics must equal the 256 cache's bit for bit (C6c), as they did between pod and box 2 at 256 (SCOPE §7). ~10.7 h of pod GPU, with the build overlapped (SCOPE §5b).
* **Ship.** Box 2 gets a 4x10 + mean + kinematics copy, ~20 GB. Box 2's free disk is UNVERIFIED.
* **Arms.**
  * **Q5_s0, Q5_s1:** init M2b s{k}. The argv is stage-1 NULL_s{k}'s verbatim, at code of record `3e264bbf` with `--stop-aux off`; the **only** change is `--cache-dir <416 cache>` (`v8f_mini_train.py:232`). 3,000 steps on box 2.
  * **NULLs reused:** stage 1's NULL_s0, NULL_s1, NULL_rep and NULL_rep2, at the same commit and argv except the cache.
  * **Floors** are stage 1's: P 0.7436 m, NR 0.1590 m, AUROC goal 0.0398, fpr 0.0116, hit 0.0427 (`VERDICT_STOPPERC_S1.md`).
* **Rules.** Card §5 unchanged: P, G-AUROC, G-FP and NR at both seeds; outcomes PERC-ADOPT, PERC-AUROC-ONLY, PERC-P-OTHER-PATH, PERC-FP-COSTS, PERC-NULL or PERC-REGRESSES. The estimator is `paired_episode_cluster_bootstrap` / `paired_stat_cluster_bootstrap`, B 2,000, seed 0. Tier T1-kin, through the 416 eval-141 cache. Four families per band. STRATEGIC PAUSED.
* **PERC-NULL here** leads to RES416-FLAT's next lever (§2.4).
* **Launch.**
  * A launch-gate PASS token per run, with the 416 cache's data-manifest sha256 bound.
  * No new trainer flag, so no new G-DVB entry.
  * G-LIVE smoke on the 416 cache.
  * G-EVAL builds the same model from the 416 eval cache.
  * Deliberate regression: a cache whose `pool4x10` rows are shifted one clip must fail the gate's data-manifest binding.
* **Cost (ESTIMATED).** Pod ~11 h wall (inside the re-encode approval). Box 2: 2 x ~40 min (R8) plus T1-kin dumps and gates ≈ 2 h, plus the ~20 GB ship. **The box-2 time is spend beyond the re-encode, and it goes to the PI** (QUESTIONS Q1).

## 4. Controls, each with a known value (evaluated first; any miss refuses the read)
| id | control | known value |
|---|---|---|
| **C0** harness: the 256 route reproduces S0-P bit for bit | S0-P's extractors on box 2's caches, S1: the sample sha256 equals `843e4017…`; shard manifests equal `C:/Users/Admin/s0p_state/shard_{0..3}.verify.json` (shard 0 `ac1a2e03…`); eval npz sha256 equals `0c25fe72…fb39`. The refit of E\|linear and V40+E\|linear on the dev box (same venv, `OMP_NUM_THREADS` 2) gives eval scores bitwise equal to the banked `s0p/run/scores/fit_E_linear.npy` and `fit_V40+E_linear.npy`, so AUROC is 0.8525… and 0.8615… exactly (`RESULT_S0P.json` sha256 `050d8b1f…`) | 0 differing bytes |
| **C1** deliberate regression, identity | "V40@416" is filled with the V40@256p arrays and fitted | paired dAUC **exactly 0.0, CI [0.0, 0.0]**; the outcome code returns not-ADDS |
| **C2** planted positive | a fixed random direction (seed 5) at the median token norm, added to one 4x10 cell of the 416 arrays on positive windows only, train and eval | V40@416+E AUROC ≥ 0.99, and the outcome code returns RES416-ADDS-40 |
| **C3** leaks | (a) the GT label appended to E | AUROC ≥ 0.999 |
| | (b) `limit` / `limit_ok` are the causal bucket (Amendment 1), against a second route | max \|diff\| 0.0 |
| | (c) **mutation:** E's limit block replaced by the logged sidecar `limit`, the ego's own realised speed ceiling over [t0+2 s, +6 s] (`build_v8_speed_max_labels.py:16-19`, card Q7). It is never fitted | E changes on ≥ 1 window; a check that cannot see the substitution is blind |
| | (d) clip ids shared between train and eval | 0 |
| | (e) per sample, E, y, clip, ws and eid equal between the 416 and 256p dirs (the 416 arm reads only pixels) | bitwise |
| **C4** E of S1 equals C0's | | bitwise |
| **C5** permutation null | label shuffle over 20 seeds on V40@416+E linear (the method of `s0p/posthoc_shuffle_seed_spread.py`) | the real fit sits above the null maximum. The single-draw "CI covers 0.5" is printed beside (the VERDICT_S0P lesson) |
| **C6** frame route | (a) pod-built 416 eval v2ep against E:'s eval139, decoded uint8, all 139 clips | max \|diff\| 0 (sha256 equality beside) |
| | (b) `resize416_to_256` of the pod's 416 NOW frames against the pod's native 256 frames | MAE ≤ 1.0/255 (reference 0.59/255, `pull_encode_pooled.py:357`); above it, the routes see different video |
| | (c) per clip, 416 v2ep `n_frames`, `poses` and `actions` against the 256 v2ep's | bitwise (the time grid does not depend on size) |
| **C7** pooling | P-A on a 16x40 field with integer cells, against `pool_field` (`pull_encode_pooled.py:255-260`) | ≤ 1e-6 fp32 |
| | a constant field | the constant, exactly |
| | every 4x10 cell's azimuth span, at both resolutions | 12.0° to 1e-9 |
| | **mutation:** a cell map from the pinhole formula | must fail the 12.0° check |
| **C8** encoder sibling | 3 eval clips at 256, full every-2nd-frame order, batches of 8 | bitwise equal to `encode_episode`. The sparse-vs-full rel-L2 is printed beside: it is common to both arms |
| **C9** counts | eval; train per sample | 2,365 / 131 / 139 eval; 16,000 / 4,000 per sample; the S1 ∩ S2 overlap is printed |
| **C10** decision logic (tests) | the unpaired reading (probe CI lower bound vs comparator point) | goes red |
| | dropping A | flips a constructed case |
| | R ∧ A at S1 only | SAMPLES-DISAGREE |
| | R from the MLP with A from the linear model | does not count (same family required) |

## 5. Families, tier, STRATEGIC
* **Q5-0:**
  * **TACTICAL** only: the stop-point decision's AUROC, n 2,365 / 131 positives / 139 episodes.
  * **LONGITUDINAL n = 0** and **LATERAL n = 0**: a probe emits no plan. The longitudinal and lateral reads of a forced stop are S0-C's (`VERDICT_S0.md`).
  * **Tier:** a frozen-feature probe, a diagnostic of what the inputs carry and never driving performance. T's comparators are T1-kin.
* **Q5-1:** T1-kin, four families per band (card §5).
* **STRATEGIC: PAUSED** (PI item 25) in both stages. It is not a pass and it is not omitted.

## 6. Code
**Reused unchanged, by path.** Dagland working copy: `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p/`. Repo copy: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-11-v8f-stopperc/s0p/` (landed `fa67c63f`; results and scores `02cac966`).
* `s0p_lib.py`: `FIELDS` :25, `draw_sample` :37, `decode_fp8` :87, `auroc` :94, `setup_tree` :105, `LazyFeats` :128.
* `s0p_extract_train.py`: grid read :146-186, NO-GRID guard :174-176. Used for C0 and the S2 E block.
* `s0p_extract_eval.py`: used for C0.
* `s0p_fit.py`: `ARMS` :25, `config_grid` :40, `decide_outcomes` :82, `fit_pca` :167, `project` :190, `cv_select` :231, `fit_one` :248, `prep` :275, `t_scores` :289, `controls` :308, `read_all` / bootstrap :340-383.
* `posthoc_shuffle_seed_spread.py` (C5).
* `test_s0p.py` (pattern for the new tests).
* `chain/s0p_chain.sh`, `s0p_common.sh`, `s0p_chain_tool.py` (box-2 chain for C0).
* Frames: `build_v2ep_wide.py`, plus the shard pattern of `run_b1_416_build.py` (SCOPE §2).
* Encoder and pool references: `dinov3_fp8_encode_ship.py` (`build_model`, `encode_episode`), `pull_encode_pooled.py` (`pool_field`, the fp8 path). Dense cache: `C:/Users/Admin/lorapod/cache_pod.py`.

**New, each with a test and a regression arm that must go red:**
* `q5_frames_pod.py`: the shard driver. It refuses an unpinned revision and checks disk before each shard.
* `q5_now_encode.py`: the sibling encoder, the P-A pool, fp8 and the S0-P-format writer, bound to the sample sha256.
* `q5_sample.py`: S2 / S3 through `draw_sample(seed=…)`.
* `q5_read.py`: the controls, the paired reads and §2.4's outcomes.
* `test_q5.py`: C1, C2, C7, C8, C10.
