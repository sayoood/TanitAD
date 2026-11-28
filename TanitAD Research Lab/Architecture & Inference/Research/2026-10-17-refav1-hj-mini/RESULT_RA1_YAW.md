# RA-1 (refav1-HJ mini), compliant yaw-rate run: the A1x instrument gate FAILED, so nothing else is read

**2026-10-18. MEASURED on box 2 (RTX 4060, SAFE MODE). Tier: T1-class open-loop trajectory** (readout from the t0
observation plus t0 kinematics, no recorded future action). Eval-141 = 2,399 windows over 141 episodes. Estimator: the
**paired episode-cluster bootstrap, B = 2000**. It answers EPISODE variance only. Training-seed variance is the
2-seed floor; there is no inference variance, because the readout is deterministic.

* Representation: `kin_mode = yaw`. Action channel 1 is the MEASURED yaw rate (`D-PI-YAWRATE-1/-2`), and the prior is
  **P10-yaw-540** (eval-141 ADE@6s **2.5627**).
* Pre-registration: `Project Steering/PREREG_HJ_STAGE0.md` (H-HJ-SENS-1 / -K-1 / -COST-1) plus Amendments 1–3.
* Raw: `raw/ra1_yaw/`, 119 files, no checkpoints:
  * `verdict.json` and `stage1_verdict.json`;
  * per run: `eval/eval.json`, `eval/perwindow.npz`, `train.log`, `train_log.jsonl`, `eval.log`, `launch_check.json`
    and `meta.json`.
* Runs on box 2: `D:\tanitad\runs\ra1_yaw\` (checkpoints stay there).
* ⚠️ **Eval set, parity:** RA-1's train clips and eval-141 are disjoint (`eval.json` → `disjointness`: n_train 600,
  n_eval 141, overlap 0), and P10-yaw-540 is fit on the same train clips. So K against P10 is clean. **However, 11 of
  the 141 eval clips lie inside the canonical refcv train corpus `physicalai-train-e438721ae894`.** The box 2 encode log
  says so: *"DROPPED 11 of 141 … The corpus is 130 clips"*. ⇒ No number here may be compared with a refcv-chassis model
  except on the 130-clip subset.

## The verdict, in the pre-registered order

**1. A1x gate: INSTRUMENT BROKEN.** The deliberate regression (time-shuffled targets) must lose to A1. It wins:
A1x − A1 = **−0.1268 [−0.2281, −0.0276]**, separated (eval-141 ADE@6s, cell b0.08g50, seed 0).
PREREG_HJ_STAGE0: *"A1x not losing to A1 ⇒ the instrument is broken; nothing else is read until it is fixed."*
⇒ **H-HJ-SENS-1, H-HJ-K-1 and H-HJ-COST-1 are NOT READ.** The numbers below are reported for the record, and they
decide nothing.

**Cell selection (Amendment 1, on the 60-clip holdout, never on eval-141):** b0.08g50. The seed-0 holdout ADE@6s by cell
was b0.08g50 **3.9129**, b0.08g100 3.9452, b0.04g100 3.9971 and b0.04g50 4.0045.

### Reported, not read

| arm | seed | eval-141 ADE@6s | − P10 [95 % CI] | holdout ADE@6s |
|---|---|---|---|---|
| R0 | 0 | 3.0931 | +0.5303 [+0.3684, +0.6910] | 3.4215 |
| R0 | 1 | 3.0765 | +0.5137 [+0.3594, +0.6629] | 3.5498 |
| A1 b0.08g50 | 0 | 3.5103 | +0.9476 [+0.7190, +1.1856] | 3.9129 |
| A1 b0.08g50 | 1 | 3.3615 | +0.7988 [+0.5797, +1.0260] | 3.8849 |
| A2 b0.08g50 | 0 | 3.5996 | +1.0368 [+0.7774, +1.2933] | 4.0709 |
| A2 b0.08g50 | 1 | 3.5899 | +1.0272 [+0.7859, +1.2941] | 3.9522 |
| A1x b0.08g50 | 0 | — | (A1x − A1 above) | 3.9118 |

* **K (not read):** every arm is WORSE than P10, separated. The training-seed floor is |A1_s0 − A1_s1| = **0.1488 m**.
* **SENS (not read):** A1's level-1 yaw-rate sensitivity (k = 10) is 0.3797 / 0.4976 at seeds 0 / 1, against A0's
  0.1753 / 0.1955. A1 − A0 = +0.2045 [+0.1680, +0.2416] and +0.3021 [+0.2559, +0.3509]. Taken alone this would pass at
  both seeds, but the gate voids it.
* **COST (not read):** L2 − L1 is negative on all six WM runs (A0 −0.1425 / −0.1316, A1 −0.1909 / −0.1881, A2 −0.1826 /
  −0.1978, all separated), so level 2 is LESS monotone than level 1 everywhere.
* **A1h (EXPLORATORY):** eval-141 ADE@6s 3.4901; its yaw-rate sensitivity is −0.0089 [−0.0181, −0.0007] below A1 at the
  same seed.

## Diagnosis: the trajectory readout memorises

* R0's training-batch ADE@6s (`train.log`) falls steadily: **2.093** at step 500, 1.355 at 1,000, 0.8188 at 2,000,
  0.5395 at 3,000 and **0.226** at 6,000. Its eval-141 ADE@6s is 3.0931, and P10 scores 2.5627.
* The readout is a residual on P10 whose output layer starts near zero (std 1e-3). At step 0 it therefore IS P10, and
  training moves it away from P10 in the wrong direction off-sample.
* A1 and A1x memorise alike: training traj 0.4166 vs 0.4277 at step 6,000. A1x's IDM term never learns (`idm_term/L1`
  14.42 at step 6,000, against A1's 0.5862). Yet on the holdout the two sit within 0.0011 m (3.9118 vs 3.9129).
  ⇒ **HYPOTHESIS:** when every arm's error is mostly readout overfit, A1-vs-A1x measures overfit noise and not
  world-model content. That would explain the broken gate without any defect in the A1x construction.
* The world-model arms are worse than R0 on the holdout and on eval-141. Each added level adds parameters the readout
  can memorise through.

## Next lever (Rule Zero): a budget sweep, pre-registered before any of its data

`PREREG_RA1_ES.md` (H-RA1-ES-1):
* **ES-1:** R0 at 250 / 500 / 1,000 / 2,000 steps, at 2 seeds; the budget S\* is chosen on the HOLDOUT.
* **ES-2** (only if some budget beats P10 on the holdout): A1 and A1x at S\*, first re-testing the gate and then K.
* Both outcomes were committed in advance. If no budget beats P10, the lever is data scale (all 4,572 clips, which
  PREREG_HJ_STAGE0 already plans).

It ran on box 2 the same night. Its result is in `RESULT_RA1_ES.md`.
