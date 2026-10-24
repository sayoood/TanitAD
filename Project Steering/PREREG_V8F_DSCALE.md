# PREREG_V8F_DSCALE: half the unique training clips at fixed steps (written 2026-10-25, before any instrument or data)

**Status: PRE-REGISTERED, NOT RUN.** Its box-2 run needs the PI's go. It is v8F-mini's instance of `H-DATA-SCALE-1` (OPEN: *"it decides D3 (data extension) vs more epochs"*). Instruments are built under this file before any run.

## Why this experiment, and why now

* **Four one-lever arms have not moved T.** The arms: WC (loss weight), PGT (O's exposure), GRID (scene resolution) and HIST (scene history). T's slot-pooled latcons RMSE moved −1 % to +1.2 % (`RESULT_V8F_{WC,PGT,GRID,HIST}.md`).
* **Their ADE6 deltas are inside the rig's seed noise.** MEASURED 2026-10-25 (`…/2026-10-17-v8f-mini/raw/2026-10-25-ensemble/`):
  * ADE6: B0 2.175, **B1 2.055**, G0 2.099, H0 2.098, PGT0 2.205, WC0 2.172 m. M2b's seed 1 beats every one-lever arm.
  * The one seed-paired contrast banked at two seeds, M2b − M1, swings 0.102 m on ADE6 between seeds (`paired_swing.out`). G's and H's −0.076 / −0.077 m are inside it.
  * The exception is accel MAE: G −0.123 and H −0.092 m/s² are 6.1× and 4.6× that contrast's accel swing (0.020). That is a candidate longitudinal-smoothness effect of the input levers, not a T effect (T's latcons did not move), and it is one seed each.
* **v8F-mini's error is shared by every trained model, not training variance.** Averaging the 6 s plans of all six banked M2b-family models gives ADE6 2.001 m. That is −0.054 [−0.139, +0.034] m (ns) against the best single model, B1. The GT-code oracle is at 0.971 m (T0). So about 1 m of the error is T's code error, common to all seeds and levers.
* **WC measured that T fits its training set and does not generalise:** train constraint loss fell 18–25 %, eval present-slot error 2.4 %.
* **Two levers remain for a generalisation-limited T: more data (D3) and the encoder (LoRA on DINOv3, the PI's AR-2 go).** We train on 4,719 of PhysicalAI-AV's 306,152 clips, 1.54 % (`D-DATA-COVERAGE-1`). The v9 labels extend from the ego log at near-zero cost, while D3's video download and DINOv3 cache build are the expensive part. This experiment is the cheapest that tells the two levers apart: one box-2 run, no new trainer code.
* **T's latcons RMSE is a precise instrument at one seed.** Its seed floor is |0.6861 − 0.6846| = 0.0016 SD, 0.2 % (`raw/2026-10-23-hist/run/check_inputs.json`, `latcons_rmse_slot_b`). The four one-lever arms stayed within ±1.2 %.

## The arm (ONE lever: the number of unique training clips)

* **B = M2b**, the banked runs: seeds 0 and 1, F = 0.1207 m (2F 0.2415), T's latcons RMSE 0.6861 / 0.6846.
* **Q = M2b + `--data-frac 0.5`.** `select_train_clips` (`stack/tanitad/data/v8f_mini_data.py:114`) takes the first round(0.5 × N) of B's clips in sha12 order, a nested half.
* **The step budget is B's.** `S = ceil(n_windows_all_clips / batch)` (`stack/scripts/v8f_mini_train.py:423`) counts ALL clips, so Q trains 19,164 steps at batch 64. Q therefore sees each of its windows about twice as often as B: B − Q is *"doubling unique data at fixed steps"*.
* Everything else is B's: the eval (eval-141, 2,399 windows, 141 episodes, `--ages cycle`), the losses and weights, the held-age rule, hist 4 / hmax 16, and the parameters.
* **Seed 0 only.** The deciding read is on T, whose seed floor is 0.2 %, and its bar is about 20× that floor. ADE6 at one seed decides nothing (`H-V8F-SEED-1`).

## The reading

* **Tier:** T1-class open loop (eval-141); the T read uses the absmask and GT-oracle evals (T0), as in HIST.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over episodes (`H-ESTIM-SEED-1`: it answers another draw of EPISODES only). The training-seed question is answered by B's measured seed floor.
* **DECIDING READ (D-T):** T's slot-pooled present-slot latcons RMSE, Q_0 against B_0. Instrument: HIST's M-LAT, with the relative change r = RMSE(Q)/RMSE(B) − 1 and the paired CI on the per-window MSE.

| outcome | when (seed 0) | the committed next step |
|---|---|---|
| **DATA-LIMITED** | r ≥ +5 %, MSE Δ separated | **D3**: extend the training clips from the corpus (v9 labels from the ego log, the DINOv3 cache), and run M2b at 2× unique clips under its own PREREG, with the PRIMARY 2F bar at both seeds. D3's predicted effect on T is −r (ESTIMATED, log-linear, ≤ 2× extrapolation). |
| **NOT DATA-LIMITED** | −5 % < r < +5 %, or the MSE Δ not separated | halving the data does not move T in this range, so 2× data is not predicted to: **LoRA on DINOv3** first (the PI's AR-2 go), under its own PREREG, with D3 behind it |
| **EPOCH-LIMITED** | r ≤ −5 %, MSE Δ separated | repeated windows help T more than new ones: **more epochs at full data** (B at 2× steps, the other half of `H-DATA-SCALE-1`), under its own PREREG |

* **Why 5 %:** about 20× B's seed floor (0.2 %) and 4× the largest change any one-lever arm produced (+1.2 %). It is set before any Q data.
* ⚠️ **Sizing D3, stated now so the DATA-LIMITED branch is not oversold.** WC's frozen-O dose curve (`RESULT_V8F_WC.md`, Amendment 2) needs T about 25 % of the way to the GT code to buy ≈ 0.19 m of ADE6. One doubling at r ≈ 5 % is therefore not predicted to clear 2F on its own.
  * D3's PREREG sizes its clip count from the measured r, and its 2× rung tests the slope's prediction first (≤ 2× extrapolation).
  * The same arithmetic binds LoRA: it must move T by tens of percent, not by the 1 % the input levers gave.
* **Reported beside, deciding nothing:**
  * ADE6 Q_0 − B_0 with its CI, against 2F;
  * the four families: longitudinal, lateral, tactical, strategic;
  * T's present-slot RMSE per group (latcons, lonspeed, stoplead) and per latcons slot;
  * T's TRAINING constraint loss over the last 10 % of steps, Q against B. A generalisation-limited T should fit its smaller set better, at worse eval;
  * Λ and the GT-code oracle ADE6.
* ⚠️ **Prediction, ESTIMATED, written down so it can be wrong:** DATA-LIMITED. WC's train/eval split says T overfits.

## Refusals (rc 2)

* Q's argv is not B's plus `--data-frac 0.5`.
* The built clip count is not round(0.5 × N_B), or the clips are not the first half of B's list in sha12 order.
* The step count is not B's 19,164.
* A parameter count or any loss weight (`w_diff`) differs from B's.
* Evals are not on one grid, or the GT codes differ.

## Proofs, before any run

* **Reachability:** `--data-frac 0.5` → `select_train_clips` → the clips the window list is built from → the batches the trainer consumes.
  * MEASURED on the real train list: the built clip count and the first and last sha12, with steps 19,164.
  * A mutation that drops `a.data_frac` from the `select_train_clips` call must turn the test red.
* **G-DVB:** `data_frac` already maps to `data:n_clips` (`v8f_mini_train.py:62`); the gate records the built count.
* **The eval is untouched:** a static check that no eval path reads `data_frac`.
* **Chain and reader:** the HIST pattern, with an identity control (+0 [0, 0]). A synthetic +6 % rise reads DATA-LIMITED, a +1 % rise NOT DATA-LIMITED, and a −6 % fall EPOCH-LIMITED.

## Amendment 1 (2026-10-25, written before any Q data, with the instruments)

* **"MSE Δ separated" means separated in r's direction.**
  * r is read on the slot-pooled RMSE and the CI on the window-weighted per-window MSE, so the two can disagree in sign.
  * A separated MSE Δ whose sign contradicts r counts as NOT separated: the outcome is NOT DATA-LIMITED, and the disagreement is reported.
  * This can only downgrade DATA-LIMITED or EPOCH-LIMITED, never upgrade.
  * Instrument: `PK/code/v8f_dscale_read.py` (docstring, line 17).
* **The clip list was MEASURED before the run** (`raw/2026-10-25-dscale/measure_dscale.{py,json,out}`, the trainer's own `build_data`).
  * Source: `C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt`. Its sorted list and its 60 excluded ids reproduce B's banked sha256s.
  * B: 4,512 clips, 306,566 windows, S 4,791, 19,164 steps.
  * Q: 2,256 clips (`001e7d520dd0` … `83f6618b1584`, the first half in sha12 order, nested in B), 153,271 windows, 19,164 steps. Each Q window is seen 8.0 times against B's 4.0.
  * The run's own gate re-measures these on box 2 from the cache manifest, and the chain asserts them before training.
* **Nothing else changes:** the bar, the estimator, the decision table and the refusals.
