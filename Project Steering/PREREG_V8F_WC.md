# PREREG_V8F_WC: T's PRESENT constraint slots, by the loss weight (written 2026-10-22, before its data)

**Status: PRE-REGISTERED, NOT RUN. It needs the PI's go for box 2:** about 5 h if seed 0 stops the arm, about 10 h if it continues.

## Why this lever (Rule Zero: the committed FAIL branch of `PREREG_V8F_CZ`)

* **The headroom is real**, T0, MEASURED. If O reads the GT code (the whole-code oracle), ADE6 falls by **1.205 / 1.051 m** at M2b and by 1.296 / 1.063 m at M1, all separated (`…/raw/2026-10-22-cz/cz_gate.json`).
  * The constraints carry **0.88–0.90** of that (`PREREG_V8F_TAC` Amendment 1).
  * Inside them, **latcons, lonspeed and stoplead** are each NECESSARY (Amendment 2).
  * Partial accuracy pays convexly: r(0.25 / 0.5 / 0.75) = 0.16 / 0.36 / 0.73 at s0.
* **The absent-slot route is closed.** Zeroing T's absent slots made driving worse by about 1 m, because O co-adapted to T's values there (R63). What is left is T's accuracy where a constraint IS present. There, T's eval-set error is **0.45–0.84 SD** by group (`…/raw/2026-10-21-tacgrp2/absent_slot_error.out`).
* **Zero-GPU facts about the lever** (`RESULT_V8F_CZ.md`, `…/raw/2026-10-22-cz/cons_curve.out`):
  * T's train constraint loss plateaus at 0.12–0.13 from about step 8,000.
  * M1 (no WM) and M2b reach the same value within 0.01, so the world-model aux does not compete with it.
  * At M2b's last step T's constraint term is **0.4 %** of the total loss, and 11 % of T's own terms: goal 0.120, ½(lat 0.153 + lon 0.771), and 0.5 × cons 0.140.
* ⚠️ **What a loss weight can and cannot do, said before the data.**
  * Adam is invariant to a loss scale on parameters that only the constraint term reaches (the last layer of T's constraint head). There the weight changes ~nothing.
  * It acts through the **shared** parameters: T's trunk, the adapter, and S through the checkpoint T reads. It pulls their representation toward what the constraints need.
  * So this arm asks: **is T's shared representation the binding limit on its constraint accuracy?** The mechanism read below answers it either way.

## The arm (ONE lever moved)

* **`M2b-wc` = `--arm M2b --w-cons 4.0`.** M2b uses the default 0.5, so this is an 8× weight. Everything else is M2b's:
  * the 4,512 clips, with the RA-1 holdout withheld (`--exclude 540:600`);
  * 19,164 steps (`--steps-mult 4`), batch 64, `pool4x10`, `--kin-mode yaw` (D-PI-YAWRATE-1), the P10-yaw-4512 prior.
  * Seeds 0, then 1. Queue stages: `gatewc`, `trainwc SEED`, `evalwc SEED`. The value lives once, in the queue's `WC`.
* **Why 4.0, committed now and as the only dose:**
  * At 4.0 the constraint term is about half of T's own loss. A null mechanism read then says "the weight is not what limits T", not "the dose was timid".
  * T's decision terms are still the other half. Their oracle share is ≤ 0.10 (Amendment 2), so a cost there is bounded.
  * ⛔ No post-hoc dose search: a second dose needs its own PREREG.
* **Launch gate:** the trainer's in-process gate (G-DVB incl. `w_cons`, G-LIVE, admissibility), asserted again by the chain before any training:
  * the built `w_cons` 4.0 and `w_tac` 1.0, read from `selfcheck.json` `levers.w`, which now records them;
  * `w_pen`, `w_pen_self`, `w_wm` and the levers (`wm` b, consistency on, `cons_absent` mask) equal to M2b's;
  * M2b's clip list, the RA-1 holdout, and M2b's step count.
* `launch_gate.py` covers only `refc_v3_train.py`; v8F-mini's gate is the trainer's own, as for M1 / M2b.

## The reading (`code/v8f_wc_read.py`)

* **Tier:** "T1-class open loop" (eval-141: 2,399 windows, 141 episodes), the stamp the v8F-mini verdicts carry. The mechanism read uses T0 oracle evals (the GT code), only to score T.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, over EPISODES.
  * Training-seed variance is answered by **F = |M2b_s0 − M2b_s1| = 0.1207 m (2F 0.2415)**, always taken from both banked M2b evals.
  * v8F-mini's planner is deterministic, so no inference-seed floor applies.
* **PRIMARY**, per seed s: δ_s = ADE6(M2b-wc_s) − ADE6(M2b_s), paired. Seed s PASSES iff δ_s < 0, separated, and |δ_s| > 2F.
* **MECHANISM**, per seed. It decides the reading of a FAIL, never the verdict.
  * The measure: T's present-slot error, the per-window mean of (T − GT)² in SD units, over the constraint slots the GT marks PRESENT, on windows with a v9 row.
  * It must be LOWER on M2b-wc: δ < 0 and separated.
  * T's values come from the `absmask` eval (its present slots are T's own); the GT comes from the GT-oracle eval. The two arms' GT codes must be identical.
* **Stage 1 (seed 0):**
  * CONTINUE iff seed 0 PASSES: only then is seed 1 trained.
  * Otherwise STOP with the seed-0 verdict. One failing seed already fails a criterion that needs both.
* **Stage 2 (seed 1):** the final reading.

| verdict | when | the committed next step |
|---|---|---|
| **SUCCESS** | both seeds pass | the T1 closed-loop read of M2b-wc, then a second dose on top, under its own PREREG |
| **FAIL-MECHANISM-OK** | T's present slots improved at every seed read; driving did not beyond 2F | O's use of the constraints: O trained against T's own code (the `p_gt` floor), under its own PREREG |
| **FAIL-NO-MECHANISM** | an 8× weight did not make T's present slots more accurate | T's INPUT / capacity for the constraints: an architecture change, so a PI go |

* **Reported beside it, deciding nothing:**
  * the LON / LAT family deltas, paired on the same windows: speed, along-track, accel, heading, yaw rate, curvature, cross-track;
  * each run's eval-json families, incl. TACTICAL and STRATEGIC (un-paused for v8F-mini, PI ruling 2026-10-17);
  * the per-group present-slot RMSE (latcons / lonspeed / stoplead).
  * **LON distance keeping is not instrumented** (the open card), and is said so per family.
* **Refusals (rc 2):**
  * a missing file;
  * runs not on one clip list, not withholding the RA-1 holdout, not done, or with different step counts;
  * an M2b-wc run whose argv is not `--arm M2b … --w-cons 4.0`, whose built `w_cons` is not 4.0, or whose `selfcheck` is not PASS;
  * an M2b run with any `--w-cons` in its argv;
  * evals not on one grid;
  * GT codes that differ between the arms.

## Proofs, before any run

* **`stack/tests/test_v8f_wc.py`, 7 tests:**
  * argv → config → loss, as an analytic identity: the total moves by exactly 3.5 × `w_tac` × cons. The control: O's output head gets an EXACTLY unchanged gradient, and T's trainable group a changed one.
  * declared-vs-built for `--w-cons 4.0`, and its lie;
  * the reader's code layout against `v8f_mini.CODE_GROUPS`. This test caught `lonspeed`'s non-contiguous slots before any data.
  * both stages' decision tables, incl. one-seed, seed-floor and no-mechanism cases;
  * every refusal;
  * the queue stages `gatewc`, `trainwc`, `evalwc` (dry run);
  * statically, that each stage's REAL line executes the one command string its dry run prints. The first mutation proof found a real line that could drop the lever under a green dry run.
* **Mutation proof:** `raw/2026-10-22-wc/mutate_wc.out`. The real queue lines were also run once against a recording fake python (`raw/2026-10-22-wc/real_queue_probe.out`): both carry `--arm M2b --w-cons 4.0`.
* **Chain:** `code/box2_ops/chain_v8f_wc.sh`, derived from `chain_v8f_cz.sh`, with ZZV8FMINI markers in its own log.
  * It refreshes from the tip: 8 files sha-checked, both v9 releases md5-checked.
  * It stages the banked M2b side before any training, and runs telemetry and the v8F guard.
  * It stops on a crash, a reboot or a guard stop.
