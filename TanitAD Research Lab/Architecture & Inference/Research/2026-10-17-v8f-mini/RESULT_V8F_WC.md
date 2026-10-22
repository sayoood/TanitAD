# RESULT: PREREG_V8F_WC, T's present constraint slots by the loss weight (2026-10-22)

**Verdict, as committed: STOP at seed 0, FAIL-MECHANISM-OK. Seed 1 was not trained** (one failing seed already fails a both-seeds criterion).
* Read by `code/v8f_wc_read.py --stage s0` (Amendment 1's reader); `raw/2026-10-22-wc/run/wc_read_s0.json`.
* PI go 2026-10-22 ("Yes run it"). Box 2, SAFE MODE, the temperature guard (max USB 45 °C), boot unchanged.
* M2b-wc_s0 trained 08:04–12:04, 19,164 steps. The launch gate passed (4,512 clips, the RA-1 holdout withheld, built `w_cons` 4.0).
* **Evidence:** MEASURED. **Tier:** T1-class open loop (eval-141); the mechanism read uses T0 oracle evals to score T. **Estimator:** paired episode-cluster bootstrap, 141 episodes, 2,399 windows; F = 0.1207 m (2F 0.2415) from both banked M2b evals.

| | M2b_s0 | M2b-wc_s0 | M2b-wc − M2b |
|---|---|---|---|
| **ADE6 (m), PRIMARY** | 2.1753 | 2.1718 | **−0.0035 [−0.0795, +0.0850] ns** (bar: < −0.2415, separated) |
| **T present-slot RMSE (SD), MECHANISM** | 0.536 | 0.523 | MSE −0.0137 [−0.0281, −0.0005], separated: **−2.4 % RMSE** |
| latcons / lonspeed / stoplead RMSE | 0.686 / 0.452 / 0.815 | 0.679 / 0.440 / 0.784 | −1 % / −3 % / −4 % |

**The four families**, seed 0, descriptive (they decide nothing; one seed, so a separated cell is necessary, not sufficient):
* **LONGITUDINAL:**
  * speed MAE **−0.066 m/s [−0.103, −0.027], separated (better)**;
  * accel MAE **−0.169 m/s² [−0.214, −0.127], separated (better)**;
  * along-track −0.054 m, ns.
  * Distance keeping is not instrumented here; a stream is wiring the existing instrument in.
* **LATERAL:**
  * cross-track **+0.050 m [+0.008, +0.098], separated (worse)**;
  * heading +0.16°, yaw rate +0.22 °/s and curvature are ns.
* **TACTICAL** (point estimates from the eval json, no CI):
  * lat macro-F1 0.816 → 0.810;
  * lon macro-F1 0.499 → 0.529;
  * goal macro-F1 0.262 → 0.267.
* **STRATEGIC** (un-paused for v8F-mini, PI ruling 2026-10-17):
  * checkpoint position error at 30 m: 7.02 [6.41, 7.71] → 7.05 [6.43, 7.73] m;
  * at 50 m: 8.01 → 8.10 m.
  * Unchanged within its CI. The weight acts on T, which is downstream of S.

## Why: the extra fit did not generalise (MEASURED, zero-GPU, from the two runs' `train_log.jsonl`)

* **T's TRAIN constraint loss fell 18–25 %** under the 8× weight: Huber `cons` in 2k-step bins went 0.130 → 0.100 mid-run and 0.125 → 0.103 at the end.
* **Its EVAL present-slot error fell 2.4 %.** Most of the extra fit is to the training set: T's constraint accuracy is limited by what it generalises from its inputs, not by how hard the loss pushes.
* **The driving result is what that size predicts.** Amendment 2's frozen-O dose curve needs about 25 % of the way to the GT (α = 0.25) to buy 0.16 / 0.08 of the oracle (≈ 0.19 / 0.08 m). A 2.4 % move predicts ≈ 0.01 m, and −0.004 m was measured.
  * ⚠️ That curve is a frozen-O read (R63), so this is a consistency check, not a sizing.

## ⚠️ A flaw in my own pre-registration, and what it does to the committed next step

* **The flaw:** the MECHANISM criterion was "lower and separated" with **no magnitude bar**. A separated 2.4 % move satisfies it. So the committed label reads "the fix took but did not carry", when in substance the fix barely took.
  * The committed next lever for this label is O's use of the constraints. But its premise is that T got meaningfully better and O failed to cash it, and nothing here supports that.
* **The label stands as committed** (FAIL-MECHANISM-OK). ⛔ It is not relabelled after the data.
* **Lesson for every future mechanism check:** give it a size bar derived in advance from the dose curve (here, the accuracy move needed to clear 2F), not only a sign.
* **Measured guide for the next arm:** T's eval error is generalisation-limited ⇒ the strongest lever is **T's input**.
  * The built, PI-gated `pool8x20` grid (160 scene tokens per frame instead of 40) is already cached on box 2, for train and eval.
  * The committed alternative, O trained against T's own code, does not act on T's accuracy at all.
* Choosing between them is the PI's call: both need box 2.

## Amendment 1 (instrument fix, before any number)

* The first s0 read crashed at load: M2b's GT oracles (banked 2026-10-20) predate `tac_code`.
* The fixed reader takes M2b's GT constraints from Amendment 2's `-tacorc-all-goal` oracle (`gt_source` in the read). The criteria are unchanged.
* `--check-inputs` now runs before any training, in the chain.
* See `PREREG_V8F_WC.md` Amendment 1.
