# PREREG_V8F_HIST: the stack's scene input over 10 history frames instead of 4 (written 2026-10-23, before its data)

**Status: PRE-REGISTERED, NOT RUN. Its box-2 run awaits the PI's go.** Instruments (chain, queue stages, reader, tests, mutation proofs) are built under this file before any run. It runs on box 2, one job at a time, SAFE MODE, under the temperature guard and the 10 h run cap of `PREREG_V8F_MINI` Amendment 4.

## Why this lever

* **It is the committed next step of `PREREG_V8F_GRID`'s FAIL-NO-MECHANISM branch** (`RESULT_V8F_GRID.md`, MEASURED 2026-10-23).
  * Four times the scene tokens moved T's latcons RMSE by −1.2 % (bar 12.5 %).
  * With the GT code, G equals B (oracle ADE6 0.969 against 0.971 m).
  * So spatial resolution is not what limits T.
* **The bottleneck is still T, and its headroom is large.** With the GT code, ADE6 falls from 2.175 to 0.971 m at M2b_s0 (T0, GT-code oracle; `RESULT_V8F_PGT.md` section 3).
  * Three arms aimed at or around T have not moved its latcons accuracy: WC (loss weight), PGT (O's exposure), GRID (resolution).
* **The group that matters is about TIME as much as space.** `latcons` (`v8f_mini_data.py` `LAT_CONSTRAINTS`) holds 12 slots:
  * four are timings or distances along the path: `turn_t_start_s`, `turn_t_end_s`, `turn_d_start_m`, `lc_t_cross_s`;
  * the rest is turn and lane-change geometry.
  * T sees the scene over `hist` = 4 frames at 0.2 s (`DT`), i.e. the last 0.6 s. A manoeuvre's onset is a change over time, which 0.6 s of video barely spans.
* **The plan names it.** `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13.6, row 1: 8×20 pooling → **more history frames** → D1 rebalancing → LoRA.
* **It is free.** The batch already carries `hmax` = 16 frames for M2b (`V8FMiniWindows.item`, `idx = t − hmax + 1 .. t`). `hist` only selects how many of them the readers see.

## The arm (ONE lever: the number of history frames the scene readers see)

* **B = M2b, fixed now:** `PREREG_V8F_PGT` and `PREREG_V8F_GRID` both FAILED, so neither is a base.
  * B is compared against the banked M2b (`raw/full/runs/M2b_s{0,1}`, `raw/full/eval/`).
  * F = |M2b_s0 − M2b_s1| = 0.1207 m; 2F = 0.2415 m.
* **H = B + `--hist 10`.** Everything else is B's:
  * the 4,512 clips, with the RA-1 holdout withheld (`--exclude 540:600`);
  * 19,164 steps (`--steps-mult 4`), batch 64;
  * `--kin-mode yaw` (D-PI-YAWRATE-1), the P10-yaw-4512 prior;
  * `--grid pool4x10`, `w_tac` 1.0 / `w_cons` 0.5, p_gt 1.0 → 0.2;
  * **`hmax` unchanged at its auto value of 16 for M2b**, so the batch content, the window set and the clip cache are B's.
* **Why 10 and not 8:** 10 is the largest `hist` the config allows at `hmax` 16 (`V8FMiniConfig.__post_init__`: `hmax ≥ hist + max_age_s`, with `max_age_s` 6). It does not change `hmax`, so the batch contents stay B's. It spans 1.8 s instead of 0.6 s. The plan's rule prefers the larger measured lever; a dose at 8 is not pre-registered.
* Seeds 0, then 1. A run lives in `<B>_s<SEED>_h10`, so it can never resume or be mistaken for a 4-frame run.
* **Fixed steps, not fixed wall-clock.** 2.5× the scene tokens per reader (400 against 160). ESTIMATED ~4.5–5 h per seed: GRID read 640 tokens at 1.126 steps/s against B's 1.217.

### What else moves with the lever (declared now, part of the lever, each MEASURED in the build)

* **D1, parameters:** the per-frame time table `time` [hist, d] gains 6 rows. Expected exactly **+1,536 parameters, all in the `adapter` group** (`v8f_mini.py:484`, `:714`). Any other delta refuses.
* **D2, initialisation:** `time` is drawn before the conditioning and reader weights, so every later initial weight comes from a different point of the RNG stream. It is like a different init seed.
  * It is part of the lever, and the training-seed floor F already spans init variation.
  * ⚠️ GRID's `pos` table had the same property and it was not stated there. It is stated here.
* **D3, held ages near a clip's start (training only):**
  * `V8FMiniWindows.item` clamps T's and S's held ages by `room = t − (hist − 1)`.
  * At `hist` 10, training windows with t < 15 (against t < 9 at `hist` 4) can have their sampled S age clamped lower.
  * The build reports the exact number of training windows whose clamp changes. Eval windows use ages (0, 0), so the eval is untouched.
* **D4, padded history at eval:**
  * The eval grid's windows sit at t = 3, 7, 11, … up to 71, every 4 steps; 141 windows each at t = 3 and t = 7 (MEASURED on `raw/eval_grid_windows.json`).
  * At `hist` 10, t = 3 sees 4 real frames and 6 padded; t = 7 sees 8 real and 2 padded. At `hist` 4 every eval window sees 4 real frames.
  * The pad mask (`frame_ok`) is B's mechanism, unchanged.
  * The reader reports the PRIMARY on all 2,399 windows (deciding) and, beside it, on the 2,117 windows with t ≥ 11 (fully real history at both `hist` values).
* ⚠️ **The history feeds all three layers.** S, T and O all read `_scene`, so a PASS is "the stack's visual history", never "T alone". The mechanism is read on T, because T is the measured bottleneck.

## The reading (the same instrument and bars as `PREREG_V8F_GRID`, with H in place of G)

* **Tier:** "T1-class open loop" (eval-141: 2,399 windows, 141 episodes). The mechanism read uses the absmask and GT-oracle evals (T0) only to score T.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, over EPISODES. It answers "another draw of episodes" only; training-seed variance enters through F, from the base's two seeds. v8F-mini's planner is deterministic, so no inference-seed floor applies.
* **PRIMARY**, per seed s: δ_s = ADE6(H_s) − ADE6(B_s), paired. Seed s PASSES iff δ_s < 0, separated, and |δ_s| > 2F.
* **MECHANISM (M-LAT)**, per seed. It decides the reading of a FAIL, never the verdict.
  * T's slot-pooled present-slot RMSE on `latcons` falls by at least 12.5 % relative to B_s, separated (paired over episodes on the per-window MSE; `PREREG_V8F_GRID` Amendment 1 F2).
  * The bar's derivation is GRID's: the frozen-O dose curve.
* **Stage 1 (seed 0):** CONTINUE iff seed 0 PASSES the PRIMARY; only then is seed 1 trained. Otherwise STOP with the seed-0 verdict.
* **Stage 2 (seed 1):** the final reading.

| verdict | when | the committed next step |
|---|---|---|
| **SUCCESS** | both seeds pass | the T1-kin read of H (`v8f_t1_eval.py`, m ∈ {30, 10}), then the next §13.6 lever on top (D1 rebalancing), under its own PREREG |
| **FAIL-MECHANISM-OK** | latcons accuracy fell ≥ 12.5 % at every seed read; driving did not move beyond 2F | T is getting more accurate but not enough to convert: the encoder itself, LoRA on DINOv3 (needs the PI's AR-2 go) |
| **FAIL-NO-MECHANISM** | latcons accuracy did not fall ≥ 12.5 % | neither space nor time in the frozen tokens is T's limit: §13.6's D1 rebalancing (event-stratified batches; the clip list unchanged), under its own PREREG, with LoRA (AR-2) behind it |

* **Reported beside it, deciding nothing:**
  * the four families: LON (speed, along-track, accel, distance keeping incl. the 2 s < 5 s TTC share), LAT (heading, yaw rate, curvature, cross-track), TACTICAL and STRATEGIC (un-paused for v8F-mini);
  * T's present-slot RMSE for every group, and **per latcons slot**, with the four timing/along-path slots summarised beside the geometry slots;
  * the latcons leak Λ and the GT-code oracle ADE6;
  * the PRIMARY on the t ≥ 11 subset (D4);
  * both arms' parameter counts and steps per second.
* **Refusals (rc 2):**
  * a missing file;
  * runs not on one clip list, not withholding the RA-1 holdout, not done, or with different step counts;
  * an H run whose argv is not B's plus `--hist 10`;
  * an H run whose built `hist` is not 10, whose checkpoint's `time` table does not have 10 rows, or whose parameter delta over B is not exactly +1,536 in `adapter`;
  * an H run whose `selfcheck` is not PASS, or whose built `hmax` differs from B's;
  * any loss weight or other built lever that differs from B's, compared key by key: `PREREG_V8F_GRID` Amendment 2's `w_diff`, with the same two proven keys;
  * evals not on one grid of windows, or GT codes that differ between the arms.

## Proofs, before any run (built under this file, staged with it)

* **Reachability, train side:**
  * trace `--hist 10` → the config → `_scene`'s gather;
  * on a synthetic clip, the readers' scene tokens are EXACTLY frames t−9..t of the batch's features at `hist` 10, and t−3..t at `hist` 4;
  * all 10 `time` rows get a non-zero gradient, with a control that reads exactly 0.0;
  * a mutation that gathers only 4 frames at `hist` 10 must turn it red.
* **Reachability, eval side (the class "built, tested, unreachable from its caller"):**
  * the eval of an `hist` 10 checkpoint builds the model at `hist` 10 from the checkpoint and gathers 10 frames;
  * an eval that silently builds `hist` 4 must be impossible, or refused, with a mutation.
* **D1–D4 measured:** the parameter delta, the count of training windows whose held-age clamp changes, and the eval padding counts.
* **Chain:** the dry run prints the exact line each stage runs; a static test pins the printed line to the run line; the training wait sits above the guard cap, as in GRID.
* **Reader:**
  * H against itself reads exactly +0 [0, 0];
  * a synthetic regression fails; a synthetic success passes;
  * B's REAL banked record shape (no `w_tac` / `w_cons`) is read (the lesson of `PREREG_V8F_GRID` Amendment 2).

## Amendment 1 (2026-10-23, written before any HIST data: what building the instruments found)

* **F-1: this file's budget sentence was FALSE.** "19,164 steps" and "the window set is B's" do not hold for `--hist 10` as the trainer is written:
  * `train_windows` starts each clip at t = hist − 1 (`v8f_mini_data.py:512`), and the step budget is sized from the window count (`v8f_mini_train.py:338-339`, `:405-406`).
  * MEASURED on box 2's train manifest (`raw/2026-10-23-hist/measure_hist.json`; the same arithmetic at hist 4 reproduces B's 306,566 windows / 19,164 steps exactly): `--hist 10` alone trains 279,494 windows and 17,472 steps (−8.8 %). The t = 3..8 windows of every clip are dropped.
  * The instruments refused on it by design (the chain gate and the reader).
* **The fix, chosen now: H = B + `--hist 10 --window-start 3`, ONE lever set by two flags** (as in `PREREG_V8F_PGT`).
  * `--window-start` is a new trainer flag. Absent, it is hist − 1, today's behaviour, so every existing arm is bit-identical; a test pins that.
  * Set, it fixes BOTH the first training window of each clip AND the held-age clamp: `room = t − window_start`, in place of `t − (hist − 1)` (`V8FMiniWindows.item`).
  * At `--window-start 3`, H's window list, held ages and step budget (306,566 / 19,164) are B's EXACTLY. The readers see 10 frames, padded where a frame precedes the clip, the pad mechanism B already has.
  * **Why, not the alternatives:**
    * "accept 17,472 steps" confounds the lever with 8.8 % less training;
    * "pin `--steps 19164`" keeps the dropped windows dropped.
    * Only this keeps everything but the visual history B's.
  * It also removes the train/eval mismatch F-2 found: H now trains on the padded-history windows the eval scores (t = 3 and t = 7).
* **This changes the "declared with the lever" list:**
  * **D3 is gone:** held ages are B's, window by window.
  * **D4 stands for the eval** (282 windows with padded history), but H now trains on that case too.
  * The PRIMARY on the 2,117 windows with t ≥ 11 is still reported beside it and decides nothing.
  * D1 (+1,536 parameters in `adapter`) and D2 (the init RNG stream) are unchanged.
* **F-3: the gate reads the built values.** `built_levers` records `hist`, `hmax` and the effective `window_start`, so the chain's gate asserts them from `selfcheck.json` instead of inferring hist from the parameter delta.
  * The new flag gets its G-DVB entry: a flag without one is refused by the launch gate itself.
* **F-4: B's record is silent on `--hist`, `--hmax`, `--w-tac`, `--w-cons`.** The reader reuses `PREREG_V8F_GRID` Amendment 2's `w_diff` unchanged.
* **F-6, found by a mutant, outside this PREREG:** `v8f_pgt_read.py`, `v8f_grid_read.py` (and by the same pattern `v8f_wc_read.py`) let an inherited `v8f_dk_read.Refusal` escape as a traceback (rc 1, not rc 2). `v8f_hist_read.py` converts it. The three older readers' runs are finished; their fix is a backlog item.
* **Everything else in this file stands as written:** the bars, the estimator, the stage rule, the decision table and the refusals. The argv check now allows exactly `--hist 10 --window-start 3` (and `--out`).
