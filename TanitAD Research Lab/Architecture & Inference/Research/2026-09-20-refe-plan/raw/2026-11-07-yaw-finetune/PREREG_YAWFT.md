# PREREG_YAWFT: does u400 use a trained yaw-rate input? (WP YAW item 1, stream YF2, 2026-11-06)

**Status: FROZEN when its sha256 is recorded in `PREREG_YAWFT.sha256` (written by YF2 BEFORE any trained arm is evaluated; MM lands the blob). No threshold below moves after the first B token is scored.**
Design: `../YF/FINETUNE_PLAN.md` (stream YF). Evidence classes: MEASURED (ours, artifact named) / ESTIMATED / HYPOTHESIS. Nothing here has been trained on the real cache yet.

## 0. The question and the tier
Does u400 with a trained yaw-rate column in `ego_enc`, fed the measured yaw rate, plan with lower LATERAL error than the same model with the input dead, without harming PDMS?
**Tier.** PDMS and families6 on the 923 confirmation tokens are NAVSIM v1.1 pseudo-simulation of an OPEN-LOOP plan against logged agents (RESULT_SFT4 says so): **not** the EVAL_DOCTRINE action-closed-loop tier. Cached-frame VAL diagnostics are T0 (teacher-forced, logged ego). Nothing here may be quoted as "drives better"; that needs the closed-loop sim A/B (ADOPT step b).
**Open-loop caveat (HYPOTHESIS).** The logged omega at t0 is 0.92-correlated with the ay/v the model already sees (MEASURED, YF F4: corr 0.924, R-squared 0.854 over 751,455 rows), so an open-loop gain can be partly an ego-status shortcut that feedback removes.

## 1. Frozen inputs (every one carries its hash; a mismatch REFUSES the run)
| input | value | pin |
|---|---|---|
| base A0 | `E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt` | sha256 `8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d` (MEASURED, g3_fingerprint.json) |
| frozen-encoder identity (G3) | u400 `frozen_in_proxy_sha256` = model_final's = the smoke cache manifest's | `160b61bc8cb0b3a0c4b21e3eac01cfed7c893f63687326c775ce620cf8c7933f` (MEASURED, `g3_fingerprint.json`); the ep015 cache carries `e155e527...` (the G3 RED arm: a cache from another snapshot differs) |
| yaw input | `ego[4] = YAW_SCALE * angular_rate_z`, **YAW_SCALE = 10.0** (MM; defined once in YT's `refe/ego_yaw_table.py`, imported by `yaw_common.import_yt_scale()`; a disagreement raises). Slot 5 stays 0. Table values are RAW rad/s. | |
| TRAIN/VAL table | `pinned/ego_yaw_navtrain_devbox.json.gz` (copy of YT's, 44,304 tokens / 539 logs; all 10,000 proxy tokens present) | sha256 `517c6413fe3569534ec6ca05dff558b0ea7c2266d5ac7deef308a0bf5ceb528a` |
| TEST table | `pinned/ego_yaw_navtest.json.gz` (12,146 tokens / 136 logs, every navtest token) | sha256 `f5ebe24fd05a8067215c22f08f873808a7020a97eca6940fdcc568f8b39e29a5` |
| cache manifest (frames) | `manifest_yaw_4k.json`: 4,000 frames of the 10k proxy manifest, per-log stratified (largest remainder), seed 20261106, `onpolicy` stripped, 395 logs split 316 TRAIN / 79 VAL by sha256(seed\|log): 3,244 / 756 frames | sha256 recorded in `HANDOFF.md` (regenerated deterministically by `make_manifest_yaw4k.py`) |
| cache | `cache_u400_4k/` built by `proxy_cache_yf2.py` (u400's own encoder, bf16 autocast, `--visual-for sets`, zero visual frames) | its `manifest.json` sha256 is written into every arm's `config.json`; the training agent records it before the first arm |
| TEST tokens | the 923 A5 confirmation tokens (43 navtest logs, `eval/raw/a5_confirm/a5_confirm_tokens.json`) | disjointness from the cache logs is ASSERTED by `make_manifest_yaw4k.py` (it refuses overlap) |
| code | `yaw_ft.py`, `proxy_cache_yf2.py` (+`proxy_cache_yf2.diff` against PK), `build_yaw_ckpt.py`, `gpu_lock.py`, `run_cache_job.py` | sha256 of `yaw_ft.py` and of PK `model.py/train.py/ckpt_io.py/measures.py` are in each arm's `config.json` |

Known caveat: the cache stores bf16-autocast `scene_ctx` (the live trainer's numerics, plan section 4); the deployed planner is fp32. u400 itself was trained under the same bf16 path; the only trained object here is a 256-vector, so the mismatch is inherited, not introduced (UNVERIFIED as an effect).

## 2. Column LR and scale: settled numerically before freezing
**Scale (MM decision, fixed): `YAW_SCALE = 10.0`.** MEASURED here: rms of `10 x omega` over the 10,000 proxy frames = 1.263 (rms of omega 0.1263 rad/s, mean 0.0216, no value exactly 0), against rms 0.72 / 0.70 for ax / ay (YF `ego_rms.out`); at x1 the yaw was 0.18 % of the pre-activation spread (YF F2). So the scale removes the input-size mismatch; what remains is the column's travel.
**Column optimiser: AdamW, wd 0, cosine to 0, no warm-up, a separate parameter group for `col4`.** Adam moves a weight by at most `lr` per step, so under the cosine `|col4| <= sqrt(256) x lr x steps / 2 = 16 x lr x 200`. ESTIMATED norm needed to match ay's pre-activation contribution (YF F2: 0.081 rms at norm 1.84, input rms 0.70) with an input of rms 1.263: about 1.0.
**Probe (`probe_col_lr.py` -> `probe_col_lr.json`, MEASURED, CPU, 400 steps, batch 32, accum 1, seed 0, B R1, TRAIN/VAL of `manifest_yaw_4k.json`).** MECHANICS ONLY: u400's decoder reads the ep015 cache's scene_ctx (a different encoder, YF `cmp_ckpt.out`), so the loss levels are not quotable and no accuracy is claimed.
| lr_col | Adam bound | realised `|col4|` | realised / bound | VAL loss (yaw fed) | VAL loss (yaw zeroed) | gap |
|---|---|---|---|---|---|---|
| 1e-4 | 0.32 | 0.039 | 12.2 % | 0.1876 | 0.1878 | 0.0002 |
| 3e-4 | 0.96 | 0.108 | 11.3 % | 0.1871 | 0.1878 | 0.0007 |
| **1e-3** | 3.2 | 0.259 | 8.1 % | **0.1866** | 0.1878 | **0.0012 (0.64 %)** |
| 3e-3 | 9.6 | 0.588 | 6.1 % | 0.1874 | 0.1878 | 0.0004 |
**Choice: `lr_col = 1e-3`.** Reasons: (1) it is the VAL minimum of the four-point grid and both neighbours are worse (3e-4: 0.1871, 3e-3: 0.1874); (2) it is the same number the analytic argument gave: YF's x1-scale recommendation of 1e-2 is 1e-3 at x10 (the column's effect on the pre-activation scales as lr x scale); (3) the realised/bound ratio falls from 12 % to 6 % as lr rises (gradient sign is noisy), so a larger lr buys little norm and, at 3e-3, loses VAL. The realised norm at the chosen lr (0.26) is below the ESTIMATED 1.0 that would match ay; the registered reading is that more norm did not buy VAL gain here, i.e. the yaw signal in this cache is small, which is what R-squared = 0.854 against (ay, v) predicts.
**Honest limits (UNVERIFIED).** The three best lrs differ by 0.0003-0.0005 in loss, which a seed could move (probe seed noise not measured), so (1) is a tie-break among near-equals, not a finding; the choice rests on (2) and (3) together. The rung-2/3 base lrs (1e-4, 5e-5) are YF's, not probed. **`lr_col` is NOT re-tuned on the real cache**; if R1 is NOT PROVEN the next levers are the capacity rungs, not a bigger column step (the probe shows 3x worse at 3e-3). Cosine drives the last 100 steps' lr to ~0 (column norm moves < 2 % there at every lr), so "still rising at the end" is not a usable lr-limited test and is not used.

## 3. Arms
| arm | what | yaw at train | yaw at eval |
|---|---|---|---|
| **A0** | u400 unchanged. Banked: PDMS 88.65, curvature MAE 0.0188 (923 tokens, `raw/2026-10-05-sft-stage2/sft4_B_u400/result.json`, `sft4_B_u400_families.json`) | none | 0 (as deployed) |
| **B (R1)** s0, s1 | column 4 of `ego_enc.0.weight` ZERO-initialised (separate leaf `col4`), only that column trains; wta_loss (P2 live loss, yaw_wrap on), AdamW wd 0, 400 steps, batch 32 x accum 8, cosine, lr_col per section 2 | table x 10 | table x 10 (YT's shim, scale from meta) |
| **C (R1)** | B's exact run (same code, argv, seed, steps), yaw = 0. Gradient on `col4` is exactly 0.0, so C == A0 bit for bit (G4/K3); evaluated only if a gate fails | 0 | 0 |
| R2 B2 / C2 | + the rest of `ego_enc` (77,056), lr_base 1e-4, `col4` group at lr_col, 400 steps | table x 10 / 0 | table x 10 / 0 |
| R3 B3 / C3 | + `queries`, `dec`, `traj_head` (4,388,412), lr_base 5e-5, `col4` group at lr_col, 200 steps | table x 10 / 0 | table x 10 / 0 |

R2 and R3 run **automatically** if the previous rung's verdict is NOT PROVEN (Rule Zero: a flat rung leaves the next one behind it). Primary contrast in every rung: **B - C** (isolates the yaw input from more fine-tuning). Ranks 0 and 1 (the augmented twin) both train, as the live trainer; both share the same real ego (MEASURED: ego identical across ranks on 2,811 paired ep015 rows), so one table value serves both.

## 4. Gates (all before any statistic; a red gate voids the read). Each has a deliberate-regression arm that must be RED
| gate | check | RED arm | status (MEASURED, `test_yaw_ft.out`, 19/19 CPU checks) |
|---|---|---|---|
| G1 table | YT's control: DB vx/vy/ax/ay equal the bank's ego[0..3]; every TRAIN/VAL/TEST token present, a missing token REFUSES | drop 2 % of tokens -> `TableRefused` | RED arm shown; YT's control is YT's |
| G2 sign | corr(table omega, ay/v) >= 0.85 on TRAIN rows (v > 2) and corr(omega, target yaw@4s) > 0 | negate omega | UNRUN on the real cache; first step of the training agent |
| G3 cache | `proxy_cache_yf2 verify` ok; C1 = 0.0 on every frame (the tool now ABORTS on the first non-zero C1); frozen fingerprint = `160b61bc...` | the ep015 cache (`e155e527...`) | fingerprint equality and RED arm MEASURED; the real build pending (job in HANDOFF) |
| G4 identity | at col4 = 0, decode with omega = 0 AND omega != 0 both `torch.equal` the original u400 decode (>= 512 frames on the real cache) | leave u400's random column (`--regress default_col`) | green; RED arm rc 3 |
| G5 reach | `abs(grad col4).sum() > 0` with the table; EXACTLY 0.0 with omega = 0 | table loaded, not applied (`--regress table_not_applied`) | green (synthetic 2.135 / 0.0); RED arm rc 3, grad_table 0.0 |
| G-CKPT | train-time decode == decode of the composed checkpoint's own `ego_enc` (`torch.equal`); K1 untouched tensors equal; K2 R1: only col 4 of `ego_enc.0.weight` differs; K3 C == u400 state_dict; K4 col4 == final | tampered extra tensor (`queries`) | green; K2 RED arm shown |
| G6 A0 eval | A0 re-run on the 923 tokens reproduces the banked per-token plans and PDMS bit for bit | wrong checkpoint | pending (eval agent) |
| G7 eval reach | at eval B's planner ego slot 4 is non-zero on >= 90 % of tokens and equals float32(10 x table) | shim refuses a token not in the table / a non-zero incoming ego[4] | shim arithmetic equals `make_batch`'s: green; both REFUSALS shown RED |
| G8 determinism | B rerun bit-identical (CPU: col4 `torch.equal`; eval: 10 % of tokens) | | CPU green; eval pending |

## 5. Readouts, estimator, variance
Every number carries arm, tier, estimator, n. Family table = families6 (the SFT-4 table).
* **PRIMARY (LATERAL): curvature MAE (1/m)** of the executed pick vs the human future, paired difference B - C (R1: B - A0). Secondary lateral: heading MAE, cross-track MAE, yaw-rate MAE.
* LONGITUDINAL (speed MAE, along-track MAE, biases): guard, no adverse separation. TACTICAL: n/a as a test (REFe has no tactical layer; families6 derives decisions from executed geometry; reported descriptively). **STRATEGIC: PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25): not a pass, not omitted. PDMS + 6 sub-scores and the share of tokens whose pick changed: guard. Strata TURN (|omega at t0| > 0.05 rad/s) vs STRAIGHT: descriptive.
* T0 on VAL (from `metrics.jsonl`): wta loss, pos L1, yaw L1, winner heading error at 1 s / 2 s, each with the yaw fed and with yaw zeroed. Diagnostics only.
* **Variance (three questions, named).** (1) Another draw of logs: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927 (the SFT-4 / M6 estimator), same resample for all arms. Answers that and nothing else. (2) Another training run: replicate seeds s0, s1 of B (and of C from R2 on); seed floor F = |Delta(s0) - Delta(s1)| on the primary and on PDMS; a separated CI from one seed is necessary, **not sufficient** (`H-ESTIM-SEED-1`). (3) Another inference run: the planner is deterministic (ego probe C1: A0 replicate bit-identical on 265 / 265 tokens), no inference-seed arm. Not measured: another BASE training run; the claim is scoped to this checkpoint.

## 6. Decision text (committed in advance)
* **ADOPT (the rung's B)** iff ALL: G1-G8 green; primary Delta(B - C) paired-CI upper bound < 0 **and** relative improvement >= 5 % (>= 0.00094 1/m on A0's 0.0188); both seeds agree in sign and F < |mean Delta| / 2; PDMS Delta CI lower bound >= -1.0 (M6 non-inferiority margin); `families_adverse` empty on every longitudinal and lateral component. Then: (a) the confirmatory read on the full 12,146 navtest tokens (A0 = 88.25 banked), same bars; (b) hand B to the closed-loop sim as an A/B with live omega on identical scenarios (the capability-tier read); (c) rerun this PREREG on P6 final. Claim after ADOPT: *on NAVSIM open-loop the yaw input is usable and lowers lateral error*, nothing about closed loop.
* **NOT PROVEN** if the primary CI straddles 0, or the seeds disagree, or F >= |mean Delta| / 2. **Do not stop:** advance the rung R1 -> R2 -> R3 with the same bars. If R3 is NOT PROVEN, log the elimination *omega is redundant with (ay, v) at fine-tune scale (R-squared 0.854 on the DB)*; next levers in order of expected effect: (i) P7 from scratch with the YAW_SCALE input, (ii) a sim-side check that ay is trustworthy there, (iii) scorer refresh on B's fan (blocked on PDM relabel compute).
* **REFUTED** if the PDMS upper bound < -1.0, or the primary CI lies entirely above 0 (worse), or a lateral / longitudinal component separates adversely. R1's identity (G4) means harm comes from *using* omega, not from drift: diagnose in order the eval shim's frame (G7), the lr (VAL curve), the sign (G2). Log as an elimination and continue to the next rung only if the harm is not an instrument fault.
* Thresholds (5 %, -1.0, 0.85, 90 %) are the YF proposals, justified in FINETUNE_PLAN section 6. No goalpost moves after the first B token is scored.
