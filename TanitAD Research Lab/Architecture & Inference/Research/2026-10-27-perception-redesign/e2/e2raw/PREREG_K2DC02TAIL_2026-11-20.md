# PREREG K2DC02-TAIL — the share dose read on the TAIL-EMA weights (zero training)

MM 4a3d05, 2026-11-20. Written AFTER the K2DC02 EMA read (banked 2e4ea640: E2 NO-GO, dose INCONCLUSIVE) and BEFORE any
K2DC02 tail number exists. No K2DC02 tail weights have been built when this file lands.

## 0. What this is, and what it is NOT

* ⛔ **It does not change the K2DC02 verdicts.** PREREG_K2DC02 §4.4 commits INCONCLUSIVE to *"a replicate arm (same flags,
  seed + 1), not a reinterpretation"*. That stands, and the replicate stays a PI decision. This read is a separate question
  with its own rule.
* **Why it is a separate question, MEASURED.** PREREG_K2DC02 put the dose test on `model_ema.pt` (EMA_6000). The E2-EMA and E2-TAIL
  rows show that this weight set keeps 0.301 of the step-0 init (decay 0.9998, no warm-up). The pull toward the init masks
  K2DC's lateral deficit (E2-TAIL M-lat = P2). K2DC − K0 `LAT_cross_mae_m` at seed 0:
  * EMA_6000 (`e2/read_K2DC/read/E2_VERDICT.json`): +0.0032 [−0.0008, +0.0071], not separated;
  * TAIL (`e2/e2raw/e2tail/stageB/read/E2_VERDICT.json`): +0.0126 [+0.0063, +0.0194], separated.

  ⇒ On EMA_6000 the whole deficit the dose could remove is about 3 mm, so the −2.5 mm SUPPORTED bar was nearly out of reach
  by construction. The tail weights carry no step-0 init, like any EMA at E3's 73,280 steps, and that is where the deficit
  lives.
* **Question:** on the E3-representative weights, does cutting the map-hires trunk share from 0.10 to 0.02 reduce the
  lateral deficit?

## 1. Weights and inputs (fixed now)

* **K2DC02 tail:** W_tail = (EMA_6000 − 0.81872 · EMA_5000) / 0.18128, from Thor `runs/r9e2-K2DC02/{model_ema.pt,
  ckpt_ema_5000.pt}`. It is built by the same `ema_tail_weights.py` (e8a4631a) that E2-TAIL used, and written as
  `model_ema_tail5k6k.pt` with its provenance.
* **Comparators, reused and not re-rolled:** the E2-TAIL stage B dumps of K2DC tail (K2's slot) and K0 tail, at the dev-box
  pull `F:/refcv9_e3/e2tail_stageB/s6c/pull/read/dump_{K2,K0}`. Every file must sha256-match that pull's `thor.sha256`, and
  stage A's FIT-threshold EVALs must sha256-match the banked `e2/e2raw/e2tail/stageA/EVAL_{K2,K0}tail_fitthr.json`.
* **Same read conditions as E2-TAIL:**
  * DEV windows: eval139 even half, 70 episodes.
  * Inference seeds 0 and 1.
  * Roll blob 38161299 and env d4bdcf07, the same mapsel / camera-blind / S6c steps, and FIT thresholds per arm.
  * K2DC02's model code is its own read tree (`read_tree_e454e96c6a84_k2dc`, the one its EMA read used).

## 2. Validity (each must hold, or the read is TAIL-VOID and decides nothing)

1. **Tail guard:** every K2DC02 dump records `ckpt_md5` equal to the md5 of its `model_ema_tail5k6k.pt`, and step 6000.
2. **Identity control (exact):** rebuilding K2DC's tail from `runs/r9e2-K2DC/{model_ema.pt, ckpt_ema_5000.pt}` with the
   same tool must reproduce the md5 recorded in E2-TAIL's `K2DC.model_ema_tail5k6k.pt.provenance.json`, bit for bit.
3. **Deliberate regression:** the tail guard must REFUSE a K2DC02 dump made from `model_ema.pt`. The read does not start
   unless this arm goes red.
4. **Comparator integrity:** the sha256 checks in §1 pass.

## 3. Primary: the dose rule on TAIL weights

The PREREG_K2DC_2026-11-12 constants, exactly as PREREG_K2DC02 §3 applied them. Only the weights change, to tail for all three
arms. The rule must hold at both seeds 0 and 1:

| outcome | rule |
|---|---|
| **SUPPORTED** | K2DC02t − K0t `LAT_cross_mae_m` not separated worse, AND K2DC02t − K2DCt `LAT_cross_mae_m` ≤ −2.5 mm, AND the jitter delta K2DC02t − K2DCt < 0 |
| **REFUTED** | K2DC02t − K2DCt `LAT_cross_mae_m` and `LAT_cross_abs_final_m` both within ±1 mm |
| **INCONCLUSIVE** | anything else |

The in-training dose validity is a property of training, not of the read. It is reused: HELD, 3 of 13 late probes in
[0.01, 0.03] (`e2/read_K2DC02/dose/DOSE_VERDICT.json`).

## 4. Secondary: the unchanged E2 rule on tail weights (K2DC02t vs K0t)

Clauses (a)–(e) are applied exactly as E2-TAIL applied them, with (a) at each arm's own FIT thresholds. All four metric
families are reported per seed, with the tier stamped: T0 for (a)/(b), T1 for (c)/(d)/(e) and the dose rule. STRATEGIC is
report-only (`e2_strategic_navcomp.py`), never a pass.

## 5. Committed next steps (each training arm is a PI decision)

1. **SUPPORTED.** The share is a lateral lever on the E3-representative weights. The EMA INCONCLUSIVE was the EMA's masking.
   * Next: K2SG, the map→trunk stop-gradient (dose 0). Its records (REC af27d979), read launcher (c646d129) and verdict
     tool (9d54f890) are landed.
   * It goes to the PI with this evidence, beside the §4.4 replicate.
2. **REFUTED.** The share is eliminated as the lateral lever on the E3-representative read, and K2SG is not recommended.
   * Next: PREREG_K2DC02 §4.3's candidates, either K2 minus `--r9-hist-supervise-geom` or the planner L2-SP anchor
     (`hold_r9_l2sp_planner`). They go to the PI.
3. **INCONCLUSIVE.** No new evidence. The §4.4 replicate stays the committed next step, and its read must include this tail
   read.
4. **The secondary E2 rule reads GO.** Report it. E3 with K2DC02's dose goes to the PI.
5. **TAIL-VOID.** Diagnose. No dose claim is made.

## 6. Estimator, limits, cost

* **Estimator:** paired episode-cluster bootstrap, B 2000, over the 70 DEV episodes. Each arm is ONE training run, so a
  separated interval answers *"another draw of EPISODES"* only (`H-ESTIM-SEED-1`). The inference seeds 0 and 1 are the
  replicate on the inference axis.
* **Cost:** zero training.
  * Thor: K2DC02 only (tail build, stage A FIT + DEV dumps, stage B rolls at 2 seeds, mapsel, camera-blind), about 3–4 h
    of the idle GPU.
  * Dev box: S6c scoring and verdicts.
