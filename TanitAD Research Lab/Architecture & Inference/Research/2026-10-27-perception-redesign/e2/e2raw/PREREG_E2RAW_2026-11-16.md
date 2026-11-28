# PRE-REGISTRATION — E2-RAW: re-read K2DC and K0 on their RAW step-6,000 weights (MM 4a3d05, 2026-11-16 ~16:10Z)

**Written before any raw-weight number exists.** It was also written before the MM read the K2DC EMA fitted-threshold number, which
`runs/E2_READ_K2DC/diag_fitthr/` was still computing. Zero training. The diagnosis behind it is
`dagland/e2/DIAG_K2_FAMILY_2026-11-16.md` (Opus agent).

## 0. What does NOT change

* **The pre-registered E2 verdict stands as written.** K2M1N20 and K2DC are each **NO-GO** on the EMA read the rule specifies
  (`read_K2M1N20/reread_dtfix/`, `read_K2DC/`). E2-RAW is a NEW read with its own rule. It is not a re-scoring of E2.
* ⛔ **A GO here does not fire E3 by itself.** E3's approval (PI 2026-11-09) is tied to "E2 GO", and this read protocol was chosen
  after the EMA data were seen. A GO therefore goes to the PI as a decision, with this file and the numbers.

## 1. Why (MEASURED, MM re-verified 2026-11-16)

* **EMA source:** `stack/tanitad/train/weight_ema.py`.
  * The decay is fixed, with no warm-up.
  * The shadow starts from the weights at step 0 (`refc_v3_train.py` ~11703, "a fresh shadow starts from the weights step 1 will
    actually train").
* **Run configs** (`runs/r9e2-{K0,K2M1N20,K2DC}/config.json` `ema`): decay 0.9998, `num_updates_at_start` 0, init "the weights at
  step 0". Argv `--steps 6000`.
* **Init share still in the shadow:** 0.9998^6000 = **0.301**, against E1's 0.999^4000 = 0.018 and E3's 73,280 steps ≈ 4e-7.
* **Why it hits K2 and not K0:** K2's banded map readouts are new and randomly initialised (`map_banded.py:156-168`). K0's map head
  was pre-trained. The in-run eval (T0, 128 windows, no CI, `metrics.jsonl`) at step 6,000:
  * K2DC edge IoU near: raw **0.219**, EMA **0.166**;
  * K2DC critic BCE: raw 0.402, EMA 0.504;
  * K0's sidewalk IoU has EMA ≥ raw.
* **The raw weights exist:** each run dir's `ckpt.pt` (the end of the WSD decay). `r9_e1_dump.py` records a full `ckpt.pt` as raw
  (`ckpt_kind`, `weights_are_ema: false`).

## 2. The read

* **Arms:** `r9e2-K2DC/ckpt.pt` ("K2DCraw") and `r9e2-K0/ckpt.pt` ("K0raw"). Read tree `read_tree_9697fa4df004_k2dc` (K2DC's frozen
  E2 read tree), DEV = eval139 even half (`DEV_HOLD_SPLIT.json`), 70 episodes, seeds 0 and 1.
* **Guard:** each raw dump's `dump.json` must record `weights_are_ema: false` and step 6,000, or the read aborts.
* **Stage A (map, about 1.5 Thor-h):** for each arm:
  1. a FIT dump on the 1,112 TRAIN-DIAG windows;
  2. fit psw + psw/pedge (f_tol), using the same recipe as `inputs/e2_fitthr_diag.sh`;
  3. a DEV dump with the E2 roll's exact `r9_e1_dump.py` flags (`--split dev --s2-windows`, the roll's remap), minus
     `--require-ema`;
  4. `r9_e1_eval.py`, boot 2000, seed 0, lift-valid M1, gates pinned. Run once at the arm's own FIT thresholds (`--map-thresholds`)
     and once at the default cut, which is reported and decides nothing.
* **Stage B (planner, about 4.4 Thor-h):** the E2 roll (`r9e2_roll.sh`), switched to `ckpt.pt` for both arms, gives (c) S6c, (d) four
  families and (e) camera-blind W4.
  * This needs a raw-weights switch in `r9e2_roll.sh`. It is built as a reviewed hold and runs only after it lands.
  * **Stage B runs only if stage A's clause (a) passes.** If (a) fails, a GO is impossible, so the 4.4 Thor-h go to the next lever
    instead (outcome 2).

## 3. The rule (GO iff ALL hold at both inference seeds; ties fail, as in E2)

| clause | rule |
|---|---|
| (a) map primary | `nf_swedge_prec_lv_5_20`: K2DCraw > K0raw (point), each arm decoded at its **own FIT thresholds** (design 2.1's decision rule; E1's protocol) |
| (b) box | AP@2 m: K2DCraw ≥ K0raw (point), as in E2 |
| (c) S6c | sidewalk rate: K2DCraw < K0raw at seeds 0 and 1 (point), as in E2 |
| (d) four families | no row separated worse (paired episode-cluster bootstrap, B 2000, seed 0, 70 episodes); STRATEGIC reported PAUSED |
| (e) W4 | K2DCraw is separated better than its camera-blind arm, as in E2 |

**Caveat, stated in advance:** one training run per arm (`H-ESTIM-SEED-1`). A separated CI answers only "another draw of
episodes".

## 4. Mechanism prediction (decides the DIAGNOSIS, not the GO)

* **Measure:** the fitted primary, EMA → raw. EMA = `E2_READ_K2DC/diag_fitthr/EVAL_K2_fitthr.json`, and for K0
  `E2_READ_K2M1N20/diag_fitthr/EVAL_K0_fitthr.json`.
* **CONFIRMED** iff K2DC gains ≥ +0.03 and K0 moves ≤ 0.01 in absolute value.
* **REFUTED** if K2DC gains < +0.03.
* **Anything else is MIXED.** If K0 moves > 0.01 alongside a K2DC gain ≥ +0.03, the EMA affects both heads, and the attribution is the
  difference of the two gains.

## 5. Committed outcomes (Rule Zero: each names the next lever)

1. **GO.**
   * Report it to the PI with the case that the E2 screen should be read on raw weights.
   * The PI decides whether E3 fires as approved. E3's own EMA carries ≈ 4e-7 of the init, so nothing in E3 changes.
   * Code lever, either way: every screen of ≤ 10k steps that adds new modules reads raw weights, or the trainer gets an EMA
     warm-up. That lever needs a G-DVB entry.
2. **(a) fails on raw.** EMA is eliminated as the map's carrier, and dose already is (DIAG R3).
   * The next lever is map capacity: M3N20L64 (PI-approved at ~13 Thor-h, 2026-11-09), read on raw weights.
   * Stage B is not run.
3. **(a) passes and (d) fails on raw.** The lateral regression is trained in, not an EMA lag.
   * The next arm is K2DC with share target **0.02** (the design's whole-trunk floor), about 20.5 Thor-h. It is a new training spend,
     so it goes to the PI.
   * Not K2DS: smoothing does not lower the mean dose.
4. **(a) passes and (c) or (e) fails on raw:** diagnose that clause zero-training first, as in E2's knockouts, before proposing an arm.

## 6. Cost and authority

* **Cost:** zero training. Stage A ≈ 1.5 Thor-h; stage B ≈ 4.4 Thor-h (ESTIMATED from K2DC's MEASURED read stages).
* **Authority:** Thor is this stream's resource (PI 2026-11-02), and the read adds no load to any training job. Thor has been idle
  since 15:07Z.
