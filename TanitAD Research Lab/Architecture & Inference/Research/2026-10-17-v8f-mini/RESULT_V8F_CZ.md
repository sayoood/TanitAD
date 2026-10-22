# RESULT: PREREG_V8F_CZ stage 1, the absent-slot oracle (2026-10-22)

**Verdict, as committed: GATE FAIL. No training ran** (`raw/2026-10-22-cz/cz_gate.json`, read by `code/v8f_cz_gate_read.py`).
* PI go 2026-10-22 ("Yes").
* Box 2, SAFE MODE, the temperature guard.
* 4 / 4 evals; ZZV8FMINI-DONE 07:07; boot unchanged.
* Code: remote tip `10078b5e`, eight files sha-checked and both v9 releases md5-checked.
* **Evidence:** MEASURED.
* **Tier:** T0 (presence from the GT).
* **Estimator:** paired episode-cluster bootstrap, 141 episodes, 2,399 windows.

| checkpoint | absmask − T's own code, ADE6 (m) | share of the full oracle | LON speed MAE (m/s) | LON along-track (m) | LAT heading (°) | LAT yaw rate (°/s) | LAT cross-track (m) |
|---|---|---|---|---|---|---|---|
| M1 s0 | **+1.107** [+0.931, +1.291] | −0.85 | +0.299 | +1.125 | +0.573 ns | −0.068 ns | +0.084 |
| M1 s1 | **+0.878** [+0.691, +1.068] | −0.83 | +0.275 | +0.915 | +0.006 ns | +0.056 ns | +0.041 |
| M2b s0 | **+1.042** [+0.853, +1.233] | −0.87 | +0.303 | +1.051 | +0.659 | +0.528 | +0.089 |
| M2b s1 | **+1.020** [+0.833, +1.218] | −0.97 | +0.338 | +1.081 | +0.149 ns | −0.005 ns | +0.054 |

* The gate needed absmask − pred < 0, separated, beyond 2F = **0.2415 m** at both M2b seeds (M1's 2F is 0.4444 m). It is positive and separated at all four checkpoints: FAIL at M2b, FAIL at M1 (`cz_gate.json` `DECISION`).
* "ns" means the interval is not separated; every unmarked cell is separated. Positive means absmask is worse.
* The full oracle (all GT) is −1.296 / −1.063 (M1) and −1.205 / −1.051 m (M2b), separated: the headroom stands; this route to it does not.
* 56.8 % of the constraint slots are absent.
* Zeroing them, with every present value T's own, makes driving **worse by about 1 m**, separated, at all four checkpoints. The hypothesis predicted the opposite sign.
* **LON distance keeping:** not instrumented (the open card).
* **TACTICAL:** unchanged by construction.
* **STRATEGIC:** un-paused for v8F-mini; unchanged by construction (S is upstream).

## Why: O has co-adapted to T's own code

* O trains on **T's own code** for most of its training: `p_gt` decays from 1.0 to 0.2, and T's code is used otherwise (`v8f_mini.py:601-602`).
* So O has learned to read T's values in the absent slots. Unsupervised as they are, they are a deterministic function of the scene, and O evidently uses them.
* O has seen "0 in an absent slot" only beside **GT values** in the present slots.
* "T's present values + 0 in absent slots" is a mix O never saw: the same whole-code effect that made single groups hurt in Amendment 1 and pushed `lonspeed`'s necessity above 1 in Amendment 2.

## ⭐ The lesson, which reaches beyond this lever

* **A frozen-O intervention cannot size a TRAINING change.** Retraining T moves the code distribution, and O co-adapts to it.
* Only whole-code oracles are clean, because O was trained on both endpoints: all GT, and all T's own. Every partial oracle in PREREG_V8F_TAC A1 / A2 and this gate is a mix.
* Those reads still RANK what moves O at a frozen O, but they do not predict a trained arm's effect.
* ⇒ The next lever's evidence must come from a **training** read.

**Next (Rule Zero; the committed FAIL branch):** T's PRESENT-slot accuracy, under its own PREREG, gated by a training read rather than a frozen-O proxy.

## Zero-GPU facts for choosing the next lever (MEASURED from the banked runs; they decide nothing)

* **T's constraint loss plateaus by mid-training.** Train Huber `cons`, in 2k-step bins, falls from 0.166 to 0.12–0.13 by about step 8,000, then creeps to 0.116–0.125 at 19,164 (`raw/2026-10-22-cz/cons_curve.out`).
* **The world-model aux does not compete with it.** M1 (no WM) and M2b (WM-b) end within 0.01 of each other at matched steps. That holds even though at M2b's last step the WM term is 87 % of the total loss (0.1 × 162.36 of 18.60) and T's constraint term is 0.4 % (1.0 × 0.5 × 0.140).
* **The eval-set error on PRESENT slots** is 0.45–0.84 SD by group (Amendment 2's `absent_slot_error.out`). T fits neither its training set (ESTIMATED from a Huber of 0.12: an RMSE near 0.5 SD) nor, by more, the eval set.
* ⇒ The next arm must move T's present-slot accuracy **in training** and be read **in training**, under its own PREREG.
