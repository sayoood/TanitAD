# PREREG_V8F_CZ: T's ABSENT constraint slots (written 2026-10-21, before any of its data)

**Status:** PRE-REGISTERED, NOT RUN. Box 2 awaits the PI's go. Stage 2 is a training run (~9 h) under the launch gate.

## Why

PREREG_V8F_TAC Amendment 2 read **CONSTRAINT-ACCURACY**: the training arm targets T's constraint prediction, all three constraint groups NECESSARY, and partial accuracy pays. Its saved codes then exposed a defect.
* **Evidence:** MEASURED, post-hoc diagnostic; `raw/2026-10-21-tacgrp2/absent_slot_error.out`.
* **The mechanism, from source:**
  * `v8f_mini.py` **masks** an absent constraint out of T's loss (`m = isfinite(tgt) & has`).
  * But `_code_gt` writes **0** in that slot (`nan_to_num(…, nan=0.0)`), and O is teacher-forced on that code (`v8f_mini.py:601-602`).
  * So O learns "0 = absent". At inference, T writes **unsupervised** values there.
* **The size:**
  * absent slots carry **92–95 %** of T's squared constraint error in `latcons` and `stoplead`, and **52–54 %** in `lonspeed`;
  * T writes `stop_x_m` values of 3.7–4.6 SD where there is no stop at all.

This is a train/inference mismatch, not a perception limit. It is the cheapest lever that could make CONSTRAINT-ACCURACY pay.

## The lever (ONE moved)

* **Arm `M2b-cz`:** `v8f_mini_train.py --arm M2b-cz` = M2b with `cons_absent zero`. T's target for an absent constraint becomes the 0 that O is teacher-forced on. Present slots, weights, data, steps and seed are unchanged.
* **Same data:** the 4,512 clips, the RA-1 holdout withheld (sha `f6a1dee0…`), M2b's train list. These are asserted at the gate.
* **Same budget:** M2b's 19,164 steps, asserted at the gate.
* **Seeds 0 and 1.**
* **Admissibility:**
  * T's inputs are unchanged: vision + nav + ego dynamics + the speed limit. The label is used only as a training target.
  * No situation-classifier output enters the goal path.
  * Inference stays vision-only (PI rulings 2026-08-03).

## Stage 1: does the mismatch matter? (eval only, zero training, T0)

* **The run:** `v8f_mini_eval.py run --tac absmask` on M1 / M2b × seeds 0, 1 (queue stage `tacabs`).
  * O reads T's own code, with every constraint slot the GT marks absent set to 0.
  * Only the PRESENCE of each constraint comes from the GT; every value is T's own.
* **The reading:** `code/v8f_cz_gate_read.py`. d_abs = tacabs − tacpred, paired over episodes; share = d_abs / d_all.
* **GATE**, read on M2b: **PASS** iff d_abs < 0, separated, and |d_abs| > 2F at **both** seeds. F_M2b = 0.1207 m, so 2F = 0.2415 m.
  * **PASS** ⇒ stage 2.
  * **FAIL** ⇒ no training on this lever. The next lever is T's PRESENT-slot accuracy, under its own PREREG.

## Stage 2: does the fix make v8F-mini drive better? (training, on GATE PASS only)

* **The run:**
  * The in-process launch gate for M2b-cz (G-DVB + G-LIVE + admissibility), pulled and asserted on the driver side too.
  * Then M2b-cz s0 → s1, each followed by the standard eval and its absent-slot oracle (`tacabsarm`).
* **The reading:** `code/v8f_cz_read.py`, against the banked M2b runs.
* **PRIMARY:** per seed, Δ_s = ADE6(M2b-cz_s) − ADE6(M2b_s), paired over episodes. Tier: **T1-class open loop** (eval-141, 2,399 windows), as stamped by `v8f_mini_full_verdict.py`.
  * **SUCCESS** iff Δ_s < 0, separated, and |Δ_s| > 2F (F = |M2b_s0 − M2b_s1|), at **both** seeds.
* **MECHANISM** decides how a FAIL is read, never the verdict.
  * The absent-slot oracle's code shift on M2b-cz must be ≤ half of M2b's at both seeds, i.e. T now writes ~0 where a constraint is absent.
* **Verdicts:**

| verdict | condition | next |
|---|---|---|
| **SUCCESS** | the primary at both seeds | the T1 closed-loop read, then the remaining present-slot headroom on top of M2b-cz |
| **FAIL-MECHANISM-OK** | no SUCCESS, mechanism in place | the fix took but did not carry: T's PRESENT-slot accuracy, own PREREG |
| **FAIL-NO-MECHANISM** | no SUCCESS, mechanism absent | the loss term is too weak or wrong: diagnose before any other arm |

* **Reported beside it, deciding nothing:**
  * LON / LAT family deltas (paired, same windows), and each run's eval-json families incl. TACTICAL and STRATEGIC. STRATEGIC is **un-paused** for v8F-mini (PI ruling 2026-10-17).
  * Distance keeping (headway / TTC) is not in this instrument; that is the open session card, stated, not dropped.
  * The residual absent-slot headroom on M2b-cz.
* **Refusals (rc 2):**
  * a missing file;
  * runs not on one clip list, not withholding the holdout, not done, or on different step counts;
  * an M2b-cz run not built `cons_absent zero`, or an M2b run not `mask`;
  * evals not on one grid.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, over EPISODES. Training-seed variance is answered by F alone. With 2 seeds, F is itself one draw, and an inference-seed floor does not apply: v8F-mini's planner is deterministic.

## Proofs, before any run

* `stack/tests/test_v8f_cz.py`, 6 tests:
  * **the lever's gradient**, with an EXACTLY-zero control: under `mask`, an absent slot gets 0.0 from T's loss; under `zero`, it gets the sign of the prediction (target 0);
  * declared-vs-built for M2b-cz and its lies (`mask` built, `w_cons` 0, the `FLAG_CHECKS` entry);
  * `absmask` zeroes only the GT-absent constraint slots, and only to 0, with the decisions untouched;
  * both readers' decision tables, incl. one-seed and seed-floor cases, and their refusals;
  * the queue stages `tacabs`, `gatearm`, `tacabsarm` and `train M2b-cz`.
* Mutation proof: `raw/2026-10-21-tacgrp2/mutate_cz.out`.
* Chain: `code/box2_ops/chain_v8f_cz.sh`, derived from `chain_v8f_mini.sh`, with ZZV8FMINI markers in its own log.
  * It refreshes from the tip, with 8 files sha-checked and both v9 releases md5-checked.
  * It runs telemetry and the v8F guard, and stops on a crash, a reboot or a guard stop.

## Outcome (2026-10-22, appended after the data)

* **Stage 1: GATE FAIL, as committed. No training ran.** absmask − pred ADE6 **+1.042 / +1.020 m** at M2b (2F 0.2415), +1.107 / +0.878 at M1, all separated (`…/2026-10-17-v8f-mini/raw/2026-10-22-cz/cz_gate.json`, `RESULT_V8F_CZ.md`).
* ⛔ **The framing above, "a train/inference mismatch, not a perception limit", is refuted (R63).** O co-adapted to T's absent-slot values, so they are its input.
* The committed FAIL branch's next lever is T's PRESENT-slot accuracy, under its own PREREG.
