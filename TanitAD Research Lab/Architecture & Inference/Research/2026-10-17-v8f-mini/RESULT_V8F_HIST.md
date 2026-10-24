# RESULT: PREREG_V8F_HIST (10 history frames instead of 4, B's windows, steps and held ages), seed 0

**VERDICT, as committed: FAIL-NO-MECHANISM. STOP at seed 0; seed 1 was not trained.**

* **Evidence class:** MEASURED.
* **Tier:** T1-class open loop (eval-141, 2,399 windows, 141 episodes). The mechanism read uses the absmask and GT-oracle evals (T0).
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over episodes (`H-ESTIM-SEED-1`: it answers another draw of EPISODES only). F = 0.1207 m, 2F = 0.2415 m.
* **Raw:** `raw/2026-10-23-hist/run/hist_read_s0.json` and the rest of `run/`.
* **Arm:** H = M2b + `--hist 10 --window-start 3` (Amendments 1–2).
  * Gate PASS: 306,566 windows, 19,164 steps, built hist / hmax / window_start 10 / 16 / 3, parameters B + 1,536.
  * Trained 07:04–11:06 on 2026-10-23 (14,484 s, 1.323 steps/s against B's 1.217), evaluated to 11:09, read 11:10, chain at tip `6b860b95`.
  * Boot unchanged; no guard stop.

## 1. The committed reads

| read | B = M2b_s0 | H = M2b-h10_s0 | Δ (H − B) [95 %] | bar | met |
|---|---|---|---|---|---|
| **PRIMARY** ADE6 (m) | 2.175 | 2.098 | **−0.077 [−0.154, +0.008], ns** | < −0.2415, separated, both seeds | **no** |
| **M-LAT** latcons RMSE (SD) | 0.686 | 0.689 | **fall −0.4 %**; MSE Δ +0.014 [−0.008, +0.034], ns | fall ≥ 12.5 %, separated | **no** |

Beside, deciding nothing:
* the PRIMARY on the 2,117 windows with t ≥ 11 (fully real history): −0.070 [−0.153, +0.017], ns;
* the latcons leak Λ **−0.155 m [−0.225, −0.091], separated**;
* the S5 tail +0.0 pp [−2.0, +1.7], ns.

## 2. The four families (H − B, paired)

* **LONGITUDINAL:**
  * **speed MAE −0.061 m/s [−0.102, −0.017], separated**;
  * **along-track MAE −0.081 m [−0.153, −0.003], separated**;
  * **accel MAE −0.092 m/s² [−0.138, −0.050], separated**;
  * distance keeping (K = 10) all ns: headway +0.02 m, time-gap −0.001 s, min TTC +0.22 s.
* **LATERAL** (all ns): heading −0.06°, yaw rate +0.16 °/s, curvature +0.001 1/m, cross-track −0.009 m.
* **TACTICAL:**
  * per arm: lateral macro-F1 0.816 → 0.810; longitudinal macro-F1 0.499 → 0.527;
  * T's present-slot RMSE latcons 0.686 → 0.689, lonspeed 0.452 → 0.456, stoplead 0.815 → 0.819 (ns);
  * per latcons slot (beside): the timing slot `turn_t_start_s` 0.808 → 0.856, worse.
* **STRATEGIC** (per arm, within intervals): position error @30 7.02 → 6.96 m, @80 11.21 → 11.19 m; heading @30 4.20° → 4.07°.

## 3. What it shows (EXPLORATORY, one seed; it decides nothing)

1. **Time in the frozen tokens does not make T more accurate**, which is the committed reading. Neither did space (GRID) nor the loss weight (WC).
2. **The two input levers agree on what they DO move.**
   * GRID and HIST each lowered ADE6 by the same amount: −0.076 and −0.077 m, each ns.
   * Each gave separated longitudinal gains: speed MAE −0.052 and −0.061 m/s, accel MAE −0.123 and −0.092 m/s².
   * Two different levers moving the same family the same way is a consistent pattern. ⚠️ It is still one seed each, not a seed replicate (`H-ESTIM-SEED-1`).
3. **The two gains are only partly on the same windows:**
   * per-window correlation of ΔADE6 0.45, per-episode 0.22;
   * both improve on 61 of 141 episodes;
   * the sum of the two per-window deltas averages −0.153 m. This is ESTIMATED as an additive stack, not measured.
4. **With the GT code, H is also slightly better than B** (oracle ADE6 0.938 against 0.971). So part of H's gain is O's own: O reads the longer history, not only T.

## 4. Rule Zero: the next levers (the PI, 2026-10-23: accumulate achievements, not fails)

* **The committed next was D1 (event-stratified batches).** Its mechanism, more exposure to rare events, is close to WC's loss-weight lever, which MEASURED that T fits training but does not generalise. So D1's expected effect is small, and it is deprioritised, not run.
* **Recommended next, largest measured effect (Rule Zero item 5): stack the two levers that moved driving.**
  * G+H = M2b + `--grid pool8x20 --hist 10 --window-start 3`.
  * Both seeds, under its own PREREG.
  * The same 2F bar, plus a pre-registered replication read of the longitudinal family.
* **The lever for T itself:** the encoder (LoRA on DINOv3, the PI's AR-2 go) or more data (D3).
  * T is generalisation-limited, and three input and loss levers have not moved it.

## 5. Provenance

| artifact | where |
|---|---|
| committed read, check-inputs, gate selfcheck | `raw/2026-10-23-hist/run/{hist_read_s0.json, check_inputs.json, gate_M2b-h10/selfcheck.json}` |
| H's four evals | `raw/2026-10-23-hist/run/eval/eval_M2b-h10_s0*` |
| H's run | `raw/2026-10-23-hist/run/runs/M2b_s0_h10/` |
| box-2 logs, guard, telemetry | `raw/2026-10-23-hist/run/box2_logs/`, `run/telemetry_v8fhist.csv` |
| checkpoint (83 MB, not in git) | the external artifacts drive, `<drive>:/Projects/TanitAD-artifacts/v8f_hist/M2b_s0_h10/ckpt_last.pt` (mounted as E: on 2026-10-25; D: in earlier records), sha256 in `run/README.md` |
