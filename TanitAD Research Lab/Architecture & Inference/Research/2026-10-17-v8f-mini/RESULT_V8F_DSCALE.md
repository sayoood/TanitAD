# RESULT: PREREG_V8F_DSCALE (half the unique training clips at fixed steps), seed 0

**OUTCOME, as committed: NOT DATA-LIMITED.**
* The committed next step is **LoRA on DINOv3 first** (`PREREG_V8F_LORA`), with D3 behind it.
* ⚠️ The point estimate is over the bar (r = +6.25 % against ±5 %), but the paired MSE difference is not separated. The committed rule therefore reads NOT DATA-LIMITED (`PREREG_V8F_DSCALE.md` table and Amendment 1).

* **Evidence class:** MEASURED.
* **Tier:** T1-class open loop (eval-141, 2,399 windows, 141 episodes). The T read uses the absmask and GT-oracle evals (T0).
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over episodes (`H-ESTIM-SEED-1`: it answers another draw of EPISODES only). F = 0.1207 m (2F 0.2415). B's T seed floor is 0.2 %.
* **Raw:** `raw/2026-10-25-dscale/run/dscale_read.json` and the rest of `run/`.
* **Arm:** Q = M2b + `--data-frac 0.5` (Amendment 1): 2,256 of 4,512 clips (`001e7d520dd0` … `83f6618b1584`, nested), 153,271 windows, 19,164 steps, 7,214,495 parameters (= B's). The gate PASSED with every value read back.
* **Run:** trained 01:31–04:18 on 2026-10-25 (2.8 h against B's ~4.4 h: half the clips, more cache hits), evaluated to 04:22, read 04:23. Chain at tip `2d85add1`; no guard stop; peak USB temperature 45 °C over the whole run (`v8fdscale_guard.out`).

## 1. The committed read (D-T)

| read | B = M2b_s0 | Q = M2b-df50_s0 | change | bar | met |
|---|---|---|---|---|---|
| **D-T** T's slot-pooled latcons RMSE (SD) | 0.6861 | 0.7290 | **r = +6.25 %** (window-weighted +5.52 %, the definitions agree); MSE Δ **+0.0248 [−0.0053, +0.0569], ns** | r ≥ +5 %, separated in r's direction | **no** (not separated) ⇒ NOT DATA-LIMITED |

Beside, deciding nothing:
* the PRIMARY-shaped ADE6, Q − B: **+0.039 [−0.080, +0.158] m, ns** (2.175 → 2.214);
* the GT-code oracle ADE6: 0.971 → 0.961, ns.

## 2. T, per group (beside)

* Present-slot RMSE, all groups: 0.536 → 0.565, MSE Δ **+0.0321 [+0.0133, +0.0527], separated** (worse).
* Per group:
  * latcons 0.686 → 0.729 (+6.3 %);
  * lonspeed 0.452 → 0.469 (+3.9 %);
  * stoplead 0.815 → 0.856 (+5.0 %).
* ~~**The generalisation signature:** T's TRAINING constraint loss over the last 10 % of steps was LOWER with half the data (0.1209 against 0.1264, −4.3 %), while its eval error rose. T fits its smaller set better and generalises worse. That is the same reading WC gave.~~
  **CORRECTED 2026-10-25 (R70, `raw/2026-10-25-t-attrib/s6_trainval.out`):** the −4.3 % is about one standard error (z ≈ 0.96 over 40 single logged batches), so it is no signature. In the trainer's own units M2b_s0's eval Huber, 0.1355 [0.1178, 0.1548], contains its train Huber, 0.1264: at full data T is FIT-limited, not generalisation-limited.

## 3. The four families (Q − B, paired)

* **LONGITUDINAL:**
  * speed MAE −0.054 [−0.113, +0.005] ns;
  * along-track −0.040 [−0.154, +0.073] ns;
  * accel MAE **−0.183 [−0.239, −0.129], separated** (see §4.2);
  * distance keeping (K = 10): headway +0.069 m ns.
* **LATERAL:**
  * cross-track **+0.094 [+0.044, +0.148] m, separated, worse**;
  * heading +0.08° ns, yaw rate +0.16 °/s ns, curvature −0.002 1/m ns.
* **TACTICAL** (per arm): lateral macro-F1 0.816 → 0.830; longitudinal macro-F1 0.499 → 0.520.
* **STRATEGIC** (per arm, within intervals; NOT PAUSED for v8F-mini): position error @30 7.02 → 7.03 m, @80 11.21 → 11.49 m; heading @30 4.20° → 4.29°.
* Λ (the latcons leak) −0.255 [−0.338, −0.174], separated.

## 4. What it shows (EXPLORATORY, one seed; it decides nothing)

1. **Data is not a large lever for T at this scale, whichever way the CI falls.**
   * Even the point estimate, +6.25 % for HALF the data, predicts about −6 % from doubling it (log-linear, ≤ 2× extrapolation).
   * WC's dose curve needs T about 25 % of the way to the GT code to buy ≈ 0.19 m of ADE6.
   * So D3 at 2× is not predicted to clear 2F. The committed branch (LoRA first) agrees with that arithmetic.
2. **The accel MAE "candidate" of `RESULT_V8F_HIST.md` §6 (R66) is not a lever effect.**
   * Against M2b_s0, every seed-0 arm lowers accel MAE: GRID −0.123, HIST −0.092, PGT −0.019, WC −0.169, and **half the data −0.183** (`raw/2026-10-25-dscale/run/accel_all_arms.out`).
   * A lever that removes half the data cannot plausibly be what improves acceleration. M2b_s0's own accel MAE is the outlier: 1.206 against seed 1's 0.931.
   * R69.
3. ~~**T is generalisation-limited, and**~~ **CORRECTED (R70): T is fit-limited at full data (eval ≈ train error), and the levers that act on its inputs, loss and data volume move it by single-digit percent.** Its error is shared by every trained model (`raw/2026-10-25-ensemble/`). The remaining committed lever is the encoder's representation (LoRA), whose prior for a PASS is LOW (`PREREG_V8F_LORA.md`).

## 5. Provenance

| artifact | where |
|---|---|
| committed read, check-inputs, gate selfcheck | `raw/2026-10-25-dscale/run/{dscale_read.json, check_inputs.json, gate_M2b-df50/selfcheck.json}` |
| Q's four evals | `raw/2026-10-25-dscale/run/eval/eval_M2b-df50_s0*` |
| Q's run | `raw/2026-10-25-dscale/run/runs/M2b_s0_df50/` |
| box 2's production train manifest (R68's evidence) | `raw/2026-10-25-dscale/run/manifest_train.jsonl` |
| box-2 logs, guard, telemetry | `raw/2026-10-25-dscale/run/box2_logs/`, `run/telemetry_v8fdscale.csv` |
| facts scripts | `run/dscale_facts.{py,out}`, `run/accel_all_arms.{py,out}` |
| checkpoint (87 MB, not in git) | the artifacts drive, `<drive>:/Projects/TanitAD-artifacts/v8f_dscale/M2b_s0_df50/ckpt_last.pt` (E: on 2026-10-25), sha256 `242c92cbb5c8a6c0f85a49d2193dc846359a8537661ff66a21fa250484683813` |
