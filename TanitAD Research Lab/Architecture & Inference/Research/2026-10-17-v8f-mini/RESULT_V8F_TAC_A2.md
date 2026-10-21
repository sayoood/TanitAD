# RESULT: PREREG_V8F_TAC Amendment 2 (2026-10-21)

**Verdict, as committed: CONSTRAINT-ACCURACY** (`raw/2026-10-21-tacgrp2/tac_groups2_read.json`, read by `code/v8f_tac_groups2_read.py`).
* Partial accuracy pays: the dose is INTERMEDIATE.
* The longitudinal pair alone does not carry.
* The next arm is a training arm on T's constraint prediction, targeting `latcons + lonspeed + stoplead`. All three are NECESSARY by leave-one-out at M2b.
* **Evidence:** MEASURED.
* **Tier:** T0, every arm reads GT.
* **Estimator:** paired episode-cluster bootstrap, 141 episodes, 2,399 windows, B = 2,000.
* **Seed floor:** 2F = 0.4444 m (M1) and 0.2415 m (M2b).

| read | M1 s0 | M1 s1 | M2b s0 | M2b s1 |
|---|---|---|---|---|
| (A) `lonspeed+londec` vs T's code, m (share) | −0.383 (0.30) | −0.252 (0.24) | −0.351 (0.29) | −0.255 (0.24) |
| (B) drop `lonspeed`'s GT from the full oracle, m (necessity) | +2.112 (1.63) | +1.629 (1.53) | +1.989 (1.65) | +1.743 (1.66) |
| (B) drop `latcons`'s GT | +0.712 (0.55) | +0.698 (0.66) | +0.665 (0.55) | +0.677 (0.64) |
| (B) drop `stoplead`'s GT | +0.325 (0.25) | +0.362 (0.34) | +0.419 (0.35) | +0.390 (0.37) |
| (B) drop `goal` / `londec` / `latdec`'s GT | +0.09 / +0.07 / +0.01 | +0.08 / +0.04 / +0.01 | +0.08 / +0.06 / +0.02 | +0.11 / +0.04 / +0.01 |
| (C) constraints α = 0.25, r | 0.16 | 0.08 | 0.16 | 0.08 |
| (C) α = 0.50, r | 0.35 | 0.20 | 0.36 | 0.21 |
| (C) α = 0.75, r | 0.70 | 0.51 | 0.73 | 0.48 |

**Reading:**
* The response to closing T's constraint error is **convex**: little at a quarter, half the gain at about three-quarters.
* Necessity above 1 for `lonspeed` means the full oracle with T's own speed constraints is **worse than T alone**. That is the whole-code teacher forcing again (`v8f_mini.py:601-602`): a GT plan with T's own speed in it is a mix O never trained on.

**The four families** at M2b s0 / s1, vs T's own code:

| | (A) `lonspeed+londec` | (C) α = 0.5 |
|---|---|---|
| LON, speed MAE (m/s) | −0.212 / −0.141 | −0.213 / −0.085 |
| LON, along-track MAE (m) | −0.359 / −0.264 | −0.360 / −0.162 |
| LAT, heading MAE (°) | −0.252 ns / +0.007 ns | −0.591 / −0.345 |
| LAT, yaw-rate MAE (°/s) | −0.145 ns / +0.104 | −0.700 / −0.182 |
| LAT, curvature MAE (1/m) | −0.002 ns / +0.001 ns | −0.012 / −0.002 |
| LAT, cross-track MAE (m) | −0.032 / −0.029 | −0.084 / −0.075 |

* **LON distance keeping:** not instrumented here (the open session card).
* **TACTICAL:** unchanged by construction; only O's input moves.
* **STRATEGIC:** un-paused for v8F-mini (PI ruling 2026-10-17), and unchanged by construction: S is upstream of every arm.

## ⭐ What the saved codes showed next: most of T's "constraint error" is a train/inference MISMATCH

* **Evidence:** MEASURED, post-hoc diagnostic; `raw/2026-10-21-tacgrp2/absent_slot_error.{py,out}`. It decides nothing; its own gate is pre-registered.
* **The mechanism:**
  * Amendment 2 is the first eval to save the code O read (`tac_code`). It shows T's constraint error is concentrated in the slots where the constraint is **absent**.
  * T's loss **masks** those slots, while `_code_gt` writes **0** there and O is teacher-forced on that 0. So at inference T fills them with unsupervised values.

| group | absent slots | RMSE present | RMSE absent | absent share of T's squared error |
|---|---|---|---|---|
| `latcons` | 76 % | 0.68–0.70 SD | 1.34–1.49 SD | **0.92–0.94** |
| `lonspeed` | 22 % | 0.45–0.47 SD | 0.93–0.95 SD | 0.52–0.54 |
| `stoplead` | 68 % | 0.82–0.84 SD | 1.89–2.43 SD | **0.92–0.95** |

* Worst slot: `stop_x_m` is absent in 94 % of windows, and T writes 3.7–4.6 SD there.
* ⇒ The CONSTRAINT-ACCURACY arm starts with the defect, not with a reweight. **`PREREG_V8F_CZ`** (written before its data):
  * **stage 1:** a zero-training absent-slot oracle gates the run;
  * **stage 2:** arm `M2b-cz` trains T to write 0 in absent slots, one lever moved vs M2b, 2 seeds, under the launch gate.

**The run itself:**
* Box 2, SAFE MODE, the temperature guard, boot unchanged.
* 40 / 40 evals ran (21:07–21:16).
* Code was the remote tip `f2f604f5`, with six key files sha-checked.
* The chain finished cleanly (ZZV8FTAC-DONE 21:20).
