# PREREG_V8F_DECINFO: where on the vision chain do speed and yaw rate get lost? (Master Mind, 2026-11-09, written before any DECINFO code or run)

## Why
`PREREG_V8F_PLANTEST3.md` (`1f9609b4`) read NO-GAIN on every decoded arm with DECODER-WEAK at both M2b seeds and KIN-CEILING not beating HOLD (`VERDICT_PLANTEST3.md`). The card's committed next lever is a nonlinear readout and multi-frame decoding on the ladder latents. Two measurements narrow what that lever can find. (1) `DIAG_DECODER.md`: on M2b, an MLP reaches speed R2 0.420 / 0.387 and stacks of 1 to 8 past latents reach at most 0.331 linearly, against a bar of 0.70. (2) `COMPARE_LADDER_ADAPTER.json`: all four finished ladder checkpoints keep M2b's adapter unchanged and retrain only `wm.proj` and the stack, so a ladder z1 is a learned re-projection of the same 256-D mean-pooled token. A readout on ladder z1 alone cannot tell whether the information was lost at the pooling, at the adapter, or was never in the frozen encoder's features. This card measures each stage on the same split, so the next training lever is chosen by where the loss is.

## Objects
Eval-141 windows, the PLANTEST3 TUNE inner split: 53 fit episodes, 17 held-out episodes, 238 held-out windows, salt `plantest2-v1`, 16 frames per window. No SCORE window is read. Targets as the PLANTEST3 gate: speed (m/s) at each frame, and yaw rate (the logged action channel at each frame, A[t]), each divided by its TUNE fit standard deviation. Zero world-model training; only readouts are fitted.

## Rungs (one change per rung: the input to the readout)
* **R1 ladder z1:** the 128-D level-1 latent of each finished ladder checkpoint (at this card's freeze: R35_s0, S35_s0, C0_s0, R35H_s0; later checkpoints are added by the same code and marked as added). This is the lever PLANTEST3 commits.
* **R0 M2b z1:** the same read on M2b s0 and s1, for reference.
* **R2 pooled adapter token:** the 256-D mean over the adapter tokens, before `wm.proj`, M2b s0 and s1.
* **R3 adapter grid:** the 4 x 10 grid of adapter tokens (256 channels), one frame, M2b s0 and s1.
* **R4 adapter grid, four frames:** R3 stacked over the frames t-3 to t.
* **R5 encoder grid:** the cached frozen-encoder tokens before the adapter (no training seed involved), one frame.
* **R6 encoder grid, four frames.**

## Readers (fixed before any read; the same for every rung of its input shape)
* **Vector rungs (R0, R1, R2):** ridge (lambda 1e-3 times the sample count, the gate's form); an MLP as `diag_mlp.py`; each at one frame and at a stack of four frames.
* **Grid rungs (R3 to R6):** a per-channel PCA to 32 channels fitted on fit episodes only, then (a) ridge on the flattened grid, and (b) a three-layer convolution over the 4 x 10 grid (32 channels per layer, frames concatenated along the channel axis), Adam, early stopping on 20 % of the fit episodes, three initialisation seeds, reporting the mean held-out R2 of the three.
* Implementation details the card leaves open go in `theory/decinfo/IMPL_CHOICES_DECINFO.md`, with its sha256, before any real window is read.

## Read (no interval; pooled held-out R2 over (window, frame) samples, 17 episodes; differences under 0.05 carry no weight)
Deciding quantity per rung: the best held-out R2 over that rung's readers (the max over a small fixed set; optimistic, stated as such), for speed and for yaw rate. Bars, those of the PLANTEST3 gate: speed 0.70, yaw rate 0.50. A rung clears only if it clears the bar for that target at every training seed it has. Tier T0, a readout diagnostic, never a driving result. No family table applies: this card has no plan or trajectory; LONGITUDINAL, LATERAL, TACTICAL have no metric here for a stated reason (a readout R2 only), and STRATEGIC is PAUSED (PI item 25).

## Committed outcomes (per target)
* **LADDER-READS:** an R1 rung clears the bar. That checkpoint's z1 and readout become the PLANTEST3 decoder, and PLANTEST3 is rerun on it.
* **POOL-LOSS:** R0, R1 and R2 are below the bar, and R3 or R4 clears it. The information is in the token grid and the mean pool loses it.
* **ADAPTER-LOSS:** R3 and R4 are below the bar, and R5 or R6 clears it. The adapter loses it.
* **NOT-READ-FROM-VISION:** every rung is below the bar. This reader at this sample size does not recover it from any stage of the vision chain. As `A NEGATIVE FROM A LINEAR PROBE IS NOT A NEGATIVE ABOUT LEARNABILITY` warns, this bounds the reader at n = 53 fit episodes and proves no absence.

## What each outcome triggers (committed now, Rule Zero)
* **POOL-LOSS:** the next world-model arm replaces the mean pool in z1 with a spatial or attention pool of the adapter tokens, on the ladder rig.
* **ADAPTER-LOSS:** the next arm unfreezes the adapter or adds a speed and yaw-rate auxiliary loss on it.
* **NOT-READ-FROM-VISION:** the same probe at full n on box 2 (the 4,572 train clips are cached only there; one job, after the ladder), before any conclusion about the encoder.
* **LADDER-READS:** PLANTEST3 on that checkpoint.
* **Every outcome:** a card for measured ego state as part of the world-model state is written, because speed and yaw rate are measured quantities and the PI's rulings admit them as inputs (D-PI-YAWRATE-1, R-EGO). That arm is not conditional on this read; this read decides whether a vision-side repair is also worth its GPU time.

## Controls (the read is INVALID if one fails)
1. **Shuffled targets:** targets permuted across episodes read held-out R2 at or below 0.05 at every rung and reader.
2. **Planted signal:** a target constructed as a fixed random linear map of each rung's input plus noise is recovered at held-out R2 at or above 0.80 by that rung's ridge reader (deliberate regression: zeroing the features must read at or below 0.02).
3. **Split:** the fit reads no held-out episode and no SCORE window (asserted; a mutation goes red).
4. **Reproduction:** rung R0's ridge at one frame reproduces the PLANTEST3 gate's R2 for speed and yaw rate within 0.01 at both seeds (`runs/pre_M2b_s*.json`).
5. **Chain identity:** `wm.z1` of the adapter tokens equals `wm.proj` of rung R2's pooled token bit for bit on a 16-window subset.

## Log
* 2026-11-09 07:11:12 frozen: prefix 6152 bytes, sha256 7e2e8aa66dcb04a275de09df26cd0c5a6f63e2325176e82530e557d2dd7668f7 (everything above this section). No DECINFO readout code or run existed at this time; `decinfo/compare_ladder_adapter.py` (checkpoint tensors only, no window read) existed and is cited in the Why.
* 2026-11-09 amendment 1, written after the first run and before its re-read. Control 2 failed on the four-frame grid rungs (planted-signal R2: R4 M2b_s0 0.332, R4 M2b_s1 0.568, R6 0.743; bar 0.80), so the read as first run is INVALID, as the card says. Cause: the ridge reader there has 5,120 features for 9,672 samples of overlapping windows. Change, ridge reader of R4 and R6 only: the top 8 PCA channels per frame (1,280 features, the count of the one-frame rungs). Control 2 and its bars are unchanged. Conv rows, every other rung and every bar are unchanged. If control 2 still fails on a rung, its ridge row is not quoted and that rung stays INVALID. This was chosen after seeing the failing control and the first-run R2 values; the first-run table stays published beside it.
