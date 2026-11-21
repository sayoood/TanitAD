# PREREG_V8F_PGT: train O (and T) against their own upstream outputs, by the scheduled-sampling floor (written 2026-10-22, before its data)

**Status: PRE-REGISTERED, NOT RUN. It needs the PI's go for box 2:** about 5 h if seed 0 stops the arm, about 10 h if it continues. Its instruments (reader, chain stages, tests, mutation proofs) are built before any run, under this file.

## Why this lever (Rule Zero: the committed FAIL branch of `PREREG_V8F_WC`)

* **The committed branch.** `PREREG_V8F_WC` read FAIL-MECHANISM-OK at seed 0 (`RESULT_V8F_WC.md`). Its committed next lever is "O's use of the constraints: O trained against T's own code (the `p_gt` floor), under its own PREREG". This is that PREREG.
* **The measured target the WC result alone lacked** (`RESULT_V8F_DK.md` section 9; EXPLORATORY, T0):
  * T's LATERAL constraint error leaks into O's LONGITUDINAL profile.
  * With T's own `latcons` in place of the GT ones (GT elsewhere), O's plan is **+0.268 / +0.167 m further along at 2 s** on lead windows at M2b s0 / s1, separated, with lateral position and heading flat (`raw/2026-10-22-dk/latcons_mech.json`).
  * The same group carries the 2 s low-TTC tail: putting T's own `latcons` back raises the share of LEAD windows under 5 s TTC by +3.7 / +1.8 pp, separated (`dk_oracle_loo.json`).
  * An O that reads T's code more often in training should learn that T's lateral slots are noisy and stop moving its speed on them.
* **The headroom it can reach** (MEASURED, T0, `…/raw/2026-10-22-cz/cz_gate.json`): with the GT code, ADE6 falls by 1.205 / 1.051 m at M2b. That gap is T's error times O's sensitivity to it. `PREREG_V8F_WC` found T's error hard to move; this arm moves O's sensitivity.

## The arm (ONE lever: the scheduled-sampling schedule)

* **`M2b-pgt` = `--arm M2b --p-gt-start 0.5 --p-gt-end 0.0`.** M2b uses 1.0 → 0.2. Everything else is M2b's:
  * the 4,512 clips, with the RA-1 holdout withheld (`--exclude 540:600`);
  * 19,164 steps (`--steps-mult 4`), batch 64, `pool4x10`, `--kin-mode yaw` (D-PI-YAWRATE-1), the P10-yaw-4512 prior, `w_cons` 0.5.
  * Seeds 0, then 1.
* ⚠️ **One lever, set by two flags, and it acts on THREE reads.** `p_gt` is the probability that a layer reads the GT upper-layer output instead of the predicted one. Two draws at that rate govern three reads (corrected while building the instruments, before any data):
  * S → T and S → O share one draw (`ck_for_t`, `ck_for_o`; `v8f_mini.py:601-602`);
  * T → O has a second draw (`use_t`, `:611-612`).
  * So this arm also trains T and O on S's own checkpoints more often. Its attribution is "the stack trained on its own upstream outputs", never "O alone". This is said here, before the data.
* **Why this dose, committed now as the only one:**
  * M2b's schedule gives the layers their own upstream output on **40 %** of draws over the run, and **70 %** in the last quarter. Moving only the floor to 0.0 gives 50 % / 87.5 %. That dose is too timid: a null at it would teach nothing (the lesson of `PREREG_V8F_WC`).
  * 0.5 → 0.0 gives **75 %** over the run and **about 94 %** in the last quarter.
  * It keeps a warm start: at step 0 the layers still read GT half of the time, so the code channel is learned before it is taken away.
  * ⛔ No post-hoc dose search: a second dose needs its own PREREG.
* **Reachability, proved before any run (the "built, tested, unreachable from its caller" class):**
  * argv → `cfg.p_gt_start` / `cfg.p_gt_end` (`v8f_mini_train.py:161`) → `p_gt_schedule(step, total, …)` (`:461`) → `model.loss(b, p_gt=…)` (`:467`) → the masks (`v8f_mini.py:601`, `:611`).
  * A test must pin it: at p_gt = 0 O's code input equals T's predicted code EXACTLY; at p_gt = 1 it equals the GT code wherever the label exists. Each identity needs a mutation that turns it red.
  * Both flags are already in the trainer's G-DVB map (`v8f_mini_train.py:66`). The chain re-asserts the built values from `selfcheck.json` before training.

## The reading

* **Tier:** "T1-class open loop" (eval-141: 2,399 windows, 141 episodes). The mechanism reads use T0 oracle evals (they read GT code).
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, over EPISODES. Training-seed variance enters through **F = |M2b_s0 − M2b_s1| = 0.1207 m (2F 0.2415)**, from both banked M2b evals. v8F-mini's planner is deterministic, so no inference-seed floor applies.
* **PRIMARY**, per seed s: δ_s = ADE6(M2b-pgt_s) − ADE6(M2b_s), paired. Seed s PASSES iff δ_s < 0, separated, and |δ_s| > 2F.
* **MECHANISM**, per seed, each with a size bar fixed now (the flaw of `PREREG_V8F_WC` was a sign-only bar). It decides the reading of a FAIL, never the verdict.
  * **(M-LEAK) the latcons leak halves.**
    * The leak of an arm is Λ = along-track(2 s) of `eval_<c>-tacorc-all-latcons` minus that of `eval_<c>-tacorc`, on the LEAD windows of the banked K=10 block.
    * Read with `latcons_mech.py`'s definition. M2b reads +0.268 / +0.167 m.
    * PASS iff Λ(M2b-pgt_s) − Λ(M2b_s) < 0, separated, and its size is at least half of Λ(M2b_s).
    * The contrast is a difference of paired differences, bootstrapped over episodes on the same windows.
  * **(M-DK) the 2 s low-TTC tail halves.**
    * The tail is S5 = share of LEAD windows with min-TTC < 5 s at 2 s, from `code/v8f_dk_read.py`'s per-window output.
    * M2b's excess over P10 is +4.4 / +2.5 pp.
    * PASS iff S5(M2b-pgt_s) − S5(M2b_s) < 0, separated, and its size is at least half of M2b_s's excess over P10.
  * The MECHANISM is met at a seed iff both M-LEAK and M-DK pass there.
* **Stage 1 (seed 0):** CONTINUE iff seed 0 PASSES the PRIMARY; only then is seed 1 trained. Otherwise STOP with the seed-0 verdict.
* **Stage 2 (seed 1):** the final reading.

| verdict | when | the committed next step |
|---|---|---|
| **SUCCESS** | both seeds pass | the T1-kin read of M2b-pgt (`stack/scripts/v8f_t1_eval.py`, m ∈ {30, 10}), then the `pool8x20` input arm on top, under its own PREREG and a PI go |
| **FAIL-MECHANISM-OK** | the mechanism is met at every seed read; driving did not move beyond 2F | O's robustness to T is not the ADE bottleneck: T's accuracy is. The `pool8x20` input arm (T's lateral geometry from a 4× finer token grid), PI go |
| **FAIL-NO-MECHANISM** | the leak or the tail did not halve | the coupling is structural, not exposure: separate O's longitudinal head from the lateral code slots (an architecture change, so a PI go) |

* **Reported beside it, deciding nothing:**
  * the four families: LON (speed, along-track, accel, **distance keeping**: headway, time-gap, min-TTC, LEAD-VIOL with its progress control), LAT (heading, yaw rate, curvature, cross-track), TACTICAL and STRATEGIC (un-paused for v8F-mini, PI ruling 2026-10-17);
  * T's present-slot RMSE by group, because T now trains on S's own checkpoints more often and may get worse;
  * the GT-code oracle ADE6 of the arm, because an O trained less on GT may read GT worse.
* **Refusals (rc 2):**
  * a missing file;
  * runs not on one clip list, not withholding the RA-1 holdout, not done, or with different step counts;
  * an M2b-pgt run whose argv is not `--arm M2b … --p-gt-start 0.5 --p-gt-end 0.0`, whose built `p_gt_start` / `p_gt_end` differ, or whose `selfcheck` is not PASS;
  * an M2b run with either flag in its argv;
  * evals not on one grid;
  * GT codes that differ between the arms.

## Instrument conventions (fixed when the instruments were built, before any M2b-pgt data)

* **A symptom M2b does not show cannot halve.** If at a seed M2b's own Λ ≤ 0, or its S5 excess over P10 ≤ 0, that mechanism is NOT-APPLICABLE and NOT met at that seed, and the reader says so. On the banked evals neither triggers: Λ +0.268 / +0.167 m, S5 excess +4.4 / +2.5 pp.
* **M-DK's population.** S5 is paired on the LEAD windows where both arms have a finite min-TTC. An arm whose path never has the lead in its corridor leaves the pair, scored neither at risk nor safe. On the banked M2b evals this drops 38 / 21 of 743 LEAD windows. The contrast over every LEAD window is reported beside the deciding one, and decides nothing.
* **Extra refusals, all in the refusing direction:**
  * any argv lever of an M2b-pgt run other than the two schedule flags differs from M2b's;
  * the loop's own `train_log.jsonl` `p_gt` differs from the declared schedule at a logged step (the measured proof that the flags reached the masks);
  * the built `p_gt_start` / `p_gt_end` are missing from `summary.json` or `selfcheck.json`. `built_levers()` now records them (`v8f_mini.py`, one line).
* **The all-latcons oracle of the arm** is run by `evalpgt` with the same flags as the banked M2b `all-<g>` oracles (`--tac oracle --tac-groups all-latcons --tac-alpha 1`). The existing `:tacgtwo` stage needs evals this arm does not have.

## Proofs, before any run (built under this file, staged with it)

* The reachability test above, with its mutations.
* The reader on the banked M2b evals against themselves: every PRIMARY and MECHANISM contrast reads exactly +0 [0, 0].
* A deliberate-regression arm: a synthetic M2b-pgt whose plans are M2b's plus a forward shift at 2 s must read M-LEAK and M-DK WORSE, separated.
* The chain dry run prints the exact training command it runs, and a static test pins that the printed line is the run line (the WC chain's lesson).
