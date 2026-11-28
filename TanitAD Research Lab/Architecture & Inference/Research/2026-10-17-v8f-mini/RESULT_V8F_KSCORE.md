# RESULT_V8F_KSCORE: a trained scorer cannot pick O's best longitudinal hypothesis (2026-10-25, MM)

**Outcome (the `PREREG_V8F_KSCORE.md` table plus Amendment 1, applied verbatim and re-derived by hand): NOT PREDICTABLE.**
PRIMARY-B fails at both seeds.

**Committed next step, quoted:** *"the best hypothesis is not predictable from these inputs at this scale. The longitudinal lever moves to O's speed profile (OACC's line)"*
⇒ `PREREG_V8F_OACC` is next. Its instruments are being built now. Box 2 order: KINDX seed 1 (running since 17:12) → FLAT s0 / s1 → OACC.

* **Evidence class:** MEASURED (ours). Box 2, chain `code/box2_ops/chain_v8f_kscore.sh` at tip `4f561a2c`. Artifacts are in `raw/2026-10-25-kscore/run/`.
* **Tier:**
  * ORACLE (from KHYP) = **T0** diagnostic. It reads the GT, so it gives headroom, never a result.
  * Selectors = open loop on B's own held outputs, with no GT read at inference. The PREREG calls this "T1-class". ⚠ The eval JSON stamps it `T0-open-loop-own-held-outputs (trunk-probe grid)`, the same mismatch as MIXO, and it stays a work item.
  * **No closed-loop claim rides on any number here.**
* **Estimator:** paired episode-cluster bootstrap (`taniteval/ci.py`), B = 2,000, 141 episodes, 2,399 windows. It answers *"would another draw of EPISODES say this?"* only, so both training seeds are shown side by side.
* **F:** 0.1207 m (PREREG). The exact |ADE6(s0) − ADE6(s1)| of B is 0.12074 m.

## The reads (`run/READ.json`, `run/KSCORE_RESULT.md`)

| read | seed 0 | seed 1 | bar | met |
|---|---|---|---|---|
| **PRIMARY-B** ADE6 KSCORE − B | −0.1001 [−0.1463, −0.0583] * | +0.0036 [−0.0273, +0.0318] | ≤ −F = −0.1207 and separated, at BOTH seeds | **no** (s0 is separated but 0.0206 m short; s1 is not separated) |
| **PRIMARY-SEL** ADE6 KSCORE − SEL-CONST | −0.0005 [−0.0188, +0.0170] | −0.0072 [−0.0419, +0.0238] | < 0 and separated, at BOTH seeds | **no** (neither seed) |
| SECONDARY speed MAE KSCORE − B | −0.0820 [−0.1103, −0.0552] * | −0.0046 [−0.0202, +0.0115] | not separated-worse | yes |
| SECONDARY accel MAE KSCORE − B | −0.1193 [−0.1418, −0.0976] * | −0.0235 [−0.0415, −0.0073] * | not separated-worse | yes |
| SECONDARY CONSISTENCY(a) 'any' KSCORE − B | **+0.0246 [+0.0083, +0.0400] *** | −0.0029 [−0.0146, +0.0079] | not separated-worse | **no at s0** |
| SECONDARY LEAD-VIOL (at-risk share) KSCORE − B | **+0.0084 [+0.0016, +0.0176] *** (26 vs 21 of 595) | +0.0050 [−0.0018, +0.0129] (39 vs 36 of 595) | not separated-worse | **no at s0** |

**Hand derivation of the outcome.**
1. PRIMARY-B fails at both seeds, so the table's row **NOT PREDICTABLE** ("PRIMARY-B fails") applies.
2. COSTS SAFETY's condition begins "PRIMARY passes", and Amendment 1's COSTS DYNAMICS needs "PRIMARY-B and PRIMARY-SEL at both seeds". So neither row can fire, even though seed 0's CONSISTENCY and LEAD-VIOL are separated-worse.
3. The reader's state (`pb` false / `ps` false at both seeds) maps to the same row.

## Controls (all PASS, both seeds; `READ.json` → `per_seed.*.controls`)

| control | seed 0 | seed 1 |
|---|---|---|
| 1. A scorer forced to T's argmax gives exactly SEL-ARGMAX | picks equal, plans bit-equal, Δ = 0.0 | same |
| 2. A scorer trained on permuted k\* does not separate from SEL-CONST | Δ = 0.0 exactly. The permuted fit picks class 5 on all 306,566 training windows | same |
| 3. B reproduces the banked ADE6 | 2.175306 vs 2.1753 | 2.054566 vs 2.0546 |
| 4. ORACLE ≤ every vertex selector, window by window | 0 violations | 0 violations |
| alignment / harness identity | GT max abs 7.6e-6 m, n = 2,399 / max abs 0.0 | same |

## What it means

* **The seed-0 gain is the constant class, again. It is not selection.**
  * KSCORE picks class 5 on 2,078 of the 2,399 eval windows (86.6 %).
  * KSCORE − SEL-CONST is −0.0005 m, and CONST − B is −0.0996 [−0.1412, −0.0621] *.
  * This reproduces KHYP's post-hoc finding, now on the **full training set**: 306,566 windows from 4,512 clips, against KHYP's 141 clips. With far more data, the scorer still learns the marginal.
* **At seed 1 there is no constant to exploit:** CONST − B = +0.0108 [−0.0030, +0.0245]. KSCORE is more varied there (class 5 on 1,369 windows, 57 %), but ADE6 does not move.
* **Accuracy against k\* barely clears the majority class:**
  * eval: 0.521 vs 0.504 (s0), 0.393 vs 0.356 (s1);
  * train: 0.529 vs 0.504, 0.437 vs 0.368.
  * The 1.7 / 3.8 pp gain on eval buys no ADE6.
* **The headroom is real and unreachable this way.**
  * KHYP's T0 ORACLE is worth −0.423 / −0.359 m against B.
  * From O's seven plans, T's softmax, B's scene tokens and the ego token, a 205,825-parameter scorer recovers none of it beyond the constant.
  * ⇒ The longitudinal error is not a *choice among O's alternatives* that B's frozen features can make. That is the PREREG's premise for moving the lever to O's own speed profile.
* **The seed-0 constant shift has a safety cost.**
  * CONSISTENCY(a) and LEAD-VIOL are separated-worse.
  * Distance keeping (levels, KSCORE − B): minimum headway −0.154 [−0.299, −0.003] m and minimum time gap −0.066 [−0.156, −0.011] s, both separated; minimum TTC +0.353 [+0.031, +0.687] s, separated. The n is 720 / 663 / 720 lead windows over 73 episodes.
  * So at s0 the class-5 plan follows leads more closely. Had PRIMARY passed, this would have read COSTS SAFETY.

## Four families, KSCORE − B (paired; lower is better except the two tactical-agreement rows)

| family | metric | seed 0 | seed 1 |
|---|---|---|---|
| LONGITUDINAL | speed MAE | −0.0820 [−0.1103, −0.0552] * | −0.0046 [−0.0202, +0.0115] |
| LONGITUDINAL | accel MAE | −0.1193 [−0.1418, −0.0976] * | −0.0235 [−0.0415, −0.0073] * |
| LONGITUDINAL | along-track MAE | −0.1139 [−0.1606, −0.0717] * | +0.0066 [−0.0238, +0.0351] |
| LONGITUDINAL | distance keeping (levels) | headway −0.154 *, time gap −0.066 s *, TTC +0.353 s * | headway −0.065, time gap −0.028 s *, TTC +0.269 s * |
| LONGITUDINAL | LEAD-VIOL (at-risk share) | +0.0084 * | +0.0050 |
| LATERAL | cross-track MAE | +0.0062 [−0.0070, +0.0196] | −0.0003 [−0.0139, +0.0144] |
| LATERAL | heading MAE (deg) | −0.4929 [−0.7815, −0.2503] * | +0.2210 [+0.0610, +0.4188] * |
| LATERAL | yaw-rate MAE (deg/s) | −0.6098 [−0.9727, −0.3215] * | +0.2770 [−0.0610, +0.7801] |
| LATERAL | curvature MAE (1/m) | −0.0125 [−0.0239, −0.0041] * | +0.0053 [−0.0025, +0.0172] |
| TACTICAL (plan-derived) | lateral decision agreement | +0.0050 [−0.0025, +0.0133] | +0.0104 [+0.0000, +0.0233] |
| TACTICAL (plan-derived) | longitudinal decision agreement | +0.0067 [−0.0096, +0.0233] | −0.0054 [−0.0196, +0.0083] |
| STRATEGIC | S's checkpoint error | identical across arms by construction | identical across arms by construction |

* **Strategic.** S is B's own in every arm; the scorer changes only which of O's hypothesis plans is deployed.
  * The family is **not paused** for v8F (`PI-V8F-2026-10-17`).
  * ⚠ The harness's own output (`run/KSCORE_RESULT.md`, `READ.json` → `strategic_family`) printed "PAUSED (PI ruling 2026-09-19)". That is R72's error, re-emitted by the generator after R72 corrected the documents. It is retracted as **R73**, and the generator's label is fixed in this commit (`stack/scripts/v8f_kscore.py`, pinned by `test_v8f_kscore.py`).
  * The banked run output stays as written, as the record of what the harness printed.
* **Descriptive only:**
  * The heading and yaw-rate rows move in opposite directions at the two seeds, so they are no lever reading.
  * The H stack is NOT-RUN (no `--h-weights`), and HS is NOT-INSTRUMENTED. Both are work items, not passes.

## Provenance

* **Box-2 run:** `D:\tanitad\runs\v8f_kscore`, pulled to `run/`. It holds `READ.json`, `KSCORE_RESULT.md`, `kscore_numbers.json`, `telemetry_v8fkscore.csv` and the chain and guard logs.
* **Scorer fits:** `run/fit/{s0,s0_perm,s1,s1_perm}/` hold argv, summary, train log, gate token, selfcheck, SEL-CONST and `ckpt_last.pt`, each 2.5 MB. The checkpoint sha256 are in `run/CKPT_SHA256.txt`.
  * Each scorer has 205,825 parameters and trained for 3,594 steps.
  * The training set is 306,566 windows. SEL-CONST was chosen on train: class 5 at both seeds.
* **Chain history:**
  * One driver death at about 15:30. The log is `run/chain_v8f_kscore.driver_died_1530.out`; the busy marker was set aside and the chain relaunched.
  * One guard-copy stop at 13:44, logged in `run/chain_v8f_kscore.stop1_guardcopy.out`.
  * No number comes from either.
