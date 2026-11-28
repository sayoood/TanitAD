# VERDICT STOPPERC S0-L (Master Mind, 2026-11-15): **NO-GAIN**: fine-tuning the encoder with LoRA does not raise the stop cue

**Question (card `PREREG_V8F_STOPPERC.md` section 4, line S0-L):** does L_s0, which trains LoRA adapters (r 32, q and v of all 24 blocks) on the DINOv3-L encoder end to end, read the stop cue better than B-prime s0, its frozen-encoder twin, when both are dumped T1-kin through their own eval caches?

**Rule (`AMENDMENT_5_S0L.md`, frozen 2026-11-13 before any L data):** GAIN iff the paired dAUC point is above 0.037 AND the paired CI's lower bound is above 0.

## Result (MEASURED, `RESULT_S0L.json`; tier T1-kin, first tick, eval-141: a decision-head diagnostic, never driving performance)

| quantity | L_s0 | B-prime s0 | L minus B-prime, paired CI | p(delta > 0) |
|---|---|---|---|---|
| **STOPCUE AUROC**, p(STOP_POINT) vs stop_hit, 2,365 windows, 131 positives, 139 episodes | 0.8656 | 0.8561 | **+0.0095 [-0.0289, +0.0525]** | 0.673 |
| beside: lon-STOP vs regime stop | 0.7706 | 0.7595 | +0.0110 [-0.0198, +0.0429] | 0.773 |

* **NO-GAIN.** The point (+0.0095) is a quarter of the 0.037 bar and the CI includes 0. The point-only reading is also false.
* **Estimator:** `taniteval.ci.paired_stat_cluster_bootstrap`, B 2,000, seed 0, by episode, the same resample for both arms (ci.py sha256 528276c5). It answers *another draw of episodes* only. One training seed per arm, so it cannot speak for *another training run* (H-ESTIM-SEED-1); the 0.037 bar stands in for that.
* **B-prime reproduces its 11-13 value** (0.8561) through the same STOPDEC.

## Controls (all MEASURED)
* **Bank:** the L_s0 checkpoint is step 19,164 (done), sha256 be8a63c8...87ab3 equal to the HF ledger (remote size and sha256 equal). `check_l_ckpt.py` PASS: 96 LoRA tensors, max |B| 0.271, 0 zero modules.
* **Cache route (Amendment 6, before any L dump):** the trained encoder module, adapters unmerged, through B-prime's cache pipeline unedited.
  * B = 0 on that route reproduces B-prime's cache bit for bit (3 clips).
  * The cache's first 8 rows equal the trained module's pooled fp8 output bit for bit.
  * 141 of 141 clips verified by md5 and size after the pull.
* **Dumps:** identity checks 0.0 for both arms (`own_all_vs_ident`, `top1_vs_dump_ol`); counts 2,365 / 131 / 139 equal the expected STOPCUE set.
* **`s0l_read.py controls`:** all pass, all mutants red (`CONTROLS_S0L.json`).
* **Route sensitivity, beside:** the folded-weights cache (bf16 fold, 5.0 % rel-L2 from the trained path; the fp32 fold agrees to 4.8e-6) reads +0.0094 [-0.0288, +0.0521], the same answer. The fold's bf16 rounding is a deployment detail and did not decide this read.
* `merged_weights_sha256` in both caches' meta is the digest of the folded twin (provenance). The record route did not use folded weights.

## Families
* **TACTICAL:** covered, stop / not-stop decision quality as above.
* **LONGITUDINAL and LATERAL:** not covered by S0-L (no speed, distance-keeping, heading, curvature or cross-track cell). L_s0's T1-kin dumps now exist beside B-prime's, in one environment, so a four-family paired read L_s0 vs B-prime is a zero-training CPU job (next, below).
* **STRATEGIC:** PAUSED (PI item 25, 2026-09-19), not a pass.

## What it means, and the next lever (Rule Zero)
* The LoRA moved the encoder's features a lot (pooled 4x10 rel-L2 1.40 against the frozen encoder on the same frames), and the stop decision did not follow. Together with S0-P FLAT (resolution 160 vs 40 cells adds nothing) and arm A PERC-NULL (the auxiliary), three perception levers on the stop cue are now eliminated at this rig's bar.
* **Card next step on NO-GAIN (outcome PERC-NULL):** the 416x1024 source, which is PI question **Q5** (compute for a re-encode). Q5 was deferred until R is read, and R never ran because S0-P read FLAT. **Blocked on a PI decision.** The evidence so far (S0-P FLAT, FULLRES PROOF-14: no stop cue above ego even at full resolution) gives it a low prior.
* **Unblocked now, zero training:** the four-family paired read L_s0 vs B-prime on the existing dumps (does the encoder fine-tune change longitudinal or lateral quality even though the stop cue did not move?).
