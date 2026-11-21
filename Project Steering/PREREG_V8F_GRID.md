# PREREG_V8F_GRID: the stack's scene input at 8×20 tokens per frame instead of 4×10 (written 2026-10-22, before its data)

**Status: PRE-REGISTERED, NOT RUN. PI go given 2026-10-22** ("Then pre-register and run pool8x20"). It runs on box 2 after `PREREG_V8F_PGT`'s chain ends, one job at a time. Its instruments (chain, queue stages, reader, tests, mutation proofs) are built before any run, under this file.

## Why this lever

* **T's accuracy is the bottleneck, and the loss weight cannot move it** (`RESULT_V8F_WC.md`, MEASURED):
  * an 8× constraint weight cut T's TRAIN constraint loss 18–25 % but its EVAL present-slot error only 2.4 %;
  * so the error is generalisation-limited, and T's input is the measured next guide.
* **The group that matters is spatial** (`RESULT_V8F_DK.md` section 9; EXPLORATORY, T0):
  * T's LATERAL constraint group `latcons` (turn timing and geometry, lane-change crossing; `v8f_mini_data.py` `LAT_CONSTRAINTS`) is the one whose error leaks into O's longitudinal profile at all four checkpoints;
  * its eval present-slot RMSE is 0.686 SD at M2b_s0, T's second-worst group.
  * A 4×10 grid of the front camera gives each token a coarse cell. 8×20 halves the cell in both axes.
* **The plan already names it.** `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13.6 lists 8×20 pooling as the first lever when M1 misses K-MINI.
* **The features exist.** Every train and eval clip of box 2's cache carries `pool8x20` [101, 160, 1024] beside `pool4x10` (MEASURED 2026-10-22 on the manifests' first rows; the chain checks every row).
* **The headroom:** with the GT code, ADE6 falls by 1.205 / 1.051 m at M2b (T0, `…/raw/2026-10-22-cz/cz_gate.json`).

## The arm (ONE lever: the scene-token grid)

* **The base B is fixed now, before `PREREG_V8F_PGT` has any data,** because that PREREG commits "pool8x20 on top" for its SUCCESS branch:
  * if `PREREG_V8F_PGT` reads **SUCCESS** (both seeds pass): **B = M2b-pgt** (`--arm M2b --p-gt-start 0.5 --p-gt-end 0.0`), compared against M2b-pgt's two seeds, with F = |B_s0 − B_s1| from them;
  * otherwise (any FAIL, including a seed-0 STOP): **B = M2b**, compared against the banked M2b, F = 0.1207 m (2F 0.2415).
* **G = B + `--grid pool8x20`.** Everything else is B's:
  * the 4,512 clips, with the RA-1 holdout withheld (`--exclude 540:600`);
  * 19,164 steps (`--steps-mult 4`), batch 64, `--kin-mode yaw` (D-PI-YAWRATE-1), the P10-yaw-4512 prior, `w_cons` 0.5.
  * Seeds 0, then 1. A run lives in `<B>_s<SEED>_8x20`, so it can never resume or be mistaken for a 4×10 run.
* **Fixed steps, not fixed wall-clock**, committed now. 4× the tokens per frame costs more per step (ESTIMATED 1.5–2×, so about 6–8 h per seed); the step count stays B's.
* ⚠️ **The grid feeds all three layers.** S, T and O all read the scene tokens (`_scene`), so the attribution of a PASS is "the stack's scene resolution", never "T alone". The mechanism is read on T, because T's accuracy is the measured bottleneck.
* ⚠️ **The parameter count may change** if the token encoder carries per-token parameters. The gate reports both counts. A change counts as part of the lever, and is stated.

## The reading

* **Tier:** "T1-class open loop" (eval-141: 2,399 windows, 141 episodes). The mechanism read uses the absmask and GT-oracle evals (T0) only to score T.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, over EPISODES. Training-seed variance enters through F, from the base's two seeds. v8F-mini's planner is deterministic, so no inference-seed floor applies.
* **PRIMARY**, per seed s: δ_s = ADE6(G_s) − ADE6(B_s), paired. Seed s PASSES iff δ_s < 0, separated, and |δ_s| > 2F.
* **MECHANISM (M-LAT)**, per seed, with a size bar fixed now. It decides the reading of a FAIL, never the verdict.
  * T's present-slot RMSE on `latcons` must fall by at least 12.5 %, relative to B_s, separated (paired over episodes on the per-window MSE, as `v8f_wc_read.py` defines the present-slot error).
  * **Why 12.5 %.** On the frozen-O dose curve (`PREREG_V8F_TAC` Amendment 2; a frozen-O read, R63), moving T's constraints a quarter of the way to the GT (α = 0.25, i.e. RMSE −25 %) buys 0.16 of the oracle, about 0.19 m at s0, still under 2F. Half of that RMSE move is the smallest accuracy change that could matter. A smaller one says the grid is not what limits T.
* **Stage 1 (seed 0):** CONTINUE iff seed 0 PASSES the PRIMARY; only then is seed 1 trained. Otherwise STOP with the seed-0 verdict.
* **Stage 2 (seed 1):** the final reading.

| verdict | when | the committed next step |
|---|---|---|
| **SUCCESS** | both seeds pass | the T1-kin read of G (`v8f_t1_eval.py`, m ∈ {30, 10}), then the next §13.6 lever on top (more history frames), under its own PREREG |
| **FAIL-MECHANISM-OK** | latcons accuracy fell ≥ 12.5 % at every seed read; driving did not move beyond 2F | T is getting more accurate but not enough to convert: the encoder itself, LoRA on DINOv3 (needs the PI's AR-2 go) |
| **FAIL-NO-MECHANISM** | latcons accuracy did not fall ≥ 12.5 % | spatial resolution is not T's limit: the next §13.6 lever, more history frames (temporal context for turn timing), under its own PREREG |

* **Reported beside it, deciding nothing:**
  * the four families: LON (speed, along-track, accel, distance keeping incl. the 2 s < 5 s TTC share), LAT (heading, yaw rate, curvature, cross-track), TACTICAL and STRATEGIC (un-paused for v8F-mini; S reads the grid too);
  * T's present-slot RMSE for every group, and the latcons leak Λ (`PREREG_V8F_PGT`'s M-LEAK);
  * both arms' parameter counts and steps per second.
* **Refusals (rc 2):**
  * a missing file;
  * runs not on one clip list, not withholding the RA-1 holdout, not done, or with different step counts;
  * a G run whose argv is not B's plus `--grid pool8x20`, whose built `n_tok` is not 160, whose checkpoint does not carry `grid: pool8x20`, or whose `selfcheck` is not PASS;
  * a B side whose identity does not match the PGT verdict above;
  * evals not on one grid of windows;
  * GT codes that differ between the arms.

## Proofs, before any run (built under this file, staged with it)

* **Reachability, train side:** `--grid pool8x20` → `n_tok` 160 (`v8f_mini_train.py:160`, checked at `:180`) → `ClipStore(grid=…)` (`:337`) → the batch's `feats` is the clip's `pool8x20` array, EXACTLY, on a synthetic clip whose two grids differ. A mutation that loads `pool4x10` must turn it red.
* **Reachability, eval side (the class "built, tested, unreachable from its caller"):** `v8f_mini_eval.py:268` reads the grid from the checkpoint with a DEFAULT of `pool4x10`.
  * Prove that an 8×20 checkpoint carries `grid: pool8x20`, and that the eval of it reads `pool8x20`.
  * Prove that a checkpoint whose `n_tok` disagrees with the features it is given is REFUSED, not silently scored. If the eval cannot refuse today, the refusal is added, with a mutation.
* **Chain:** every manifest row of box 2's train and eval caches lists `8x20`; the dry run prints the exact line each stage runs, and a static test pins the printed line to the run line.
* **Reader:** B against itself reads exactly +0 [0, 0]; a synthetic regression fails; a synthetic success passes.

## Amendment 1 (2026-10-22, written before any GRID data: what building the instruments found)

* **F1: a derived constant moves with the lever.** The trainer sizes its clip cache from a fixed RAM budget and the token count, then the sampler chunk from that:
  * `v8f_mini_train.py:328`: `lru = max(8, 1.5e9 // (n_max · n_tok · 1024 · 2))`;
  * `:405`: `chunk = max(4, min(48, lru − 2))`.
  * At 8×20 the chunk falls from 48 to about 41 clips, so each batch is drawn from slightly fewer clips at a time.
  * It is NOT pinned: pinning it would need a second flag and more RAM on the PI's machine. It is declared part of the lever, "the stack's scene resolution, with the sampler chunk the trainer derives from it". Both chunk sizes are reported.
* **F2: which RMSE.** M-LAT uses the slot-pooled present-slot RMSE. That is the definition behind the 0.686 SD this file cites, and the one `v8f_wc_read.py` uses.
  * The window-weighted RMSE (0.468 SD at M2b_s0) is reported beside it and decides nothing.
  * The reader flags it when the two disagree on the 12.5 % bar.
* **The parameter count rises by exactly +30,720:** 120 more token-position rows, MEASURED in the build. It is part of the lever, as this file anticipated.
* **The eval refuses a mismatch without an edit.**
  * A checkpoint scored on features of the wrong token count raises a torch `RuntimeError` (`raw/2026-10-22-grid/probe_eval_mismatch.out`); the chain turns that into a STOP.
  * The trainer's own `n_tok` gate (`:180`) compares two values derived from one flag. So reachability rests on the test that the batch's features ARE the clip's `pool8x20` array, exactly.
* **The 10 h run cap of `PREREG_V8F_MINI` Amendment 4 applies unchanged.**
  * If the guard projects a G run above it, that run is a STOP with its reason. Moving the cap is the PI's call.
  * The chain's training wait is 11 h, so the guard ends a slow run, never the wait.

## Amendment 2 (2026-10-23 03:30, written after G s0 trained and was evaluated, BEFORE any GRID number was read)

* **What happened.** The chain trained G s0 (22:22–03:07, rc 0) and ran its four evals (to 03:11, rc 0). Then the reader REFUSED at its input check, so no delta and no verdict was computed:
  * *"M2b-8x20_s0: summary.json built levers ['w'] differ from M2b_s0's"*, and the same for selfcheck.json.
  * The chain stopped (`ZZV8FMINI-STOP grid read s0`). Box 2 is idle; the guard saw max 44 °C.
* **The cause is a record-schema difference, not a second lever.**
  * G's `levers.w` = {w_pen, w_pen_self, w_wm, **w_tac 1.0, w_cons 0.5**}; M2b's banked record has the first three only, at the same values.
  * `built_levers` began recording `w_tac` / `w_cons` at `39e21e5` (2026-10-22 07:45, for `PREREG_V8F_WC`). M2b was trained before that.
  * The reader compared `w` as ONE value, so the two keys missing from B's older record read as a moved lever.
* **Proof that B trained at exactly G's values (MEASURED from source):**
  * Neither M2b's argv (`raw/full/runs/M2b_s{0,1}/argv.json`) nor G's carries `--w-tac` or `--w-cons`, so both train at the trainer's defaults.
  * `v8f_mini_train.py` declares `--w-tac 1.0` and `--w-cons 0.5` at every commit that touched it (`4658d5c`, `09412e4`, `9678721`, `c483a28`).
  * At M2b's era it passes both into the config (`9678721:stack/scripts/v8f_mini_train.py:153-154`), and the loss uses both (`9678721:stack/tanitad/models/v8f_mini.py:618`).
  * The chain's own G gate had already asserted (w_tac, w_cons) = B's 1.0 / 0.5, and it PASSED.
* **The amendment:** the reader compares `w` key by key (`w_diff`).
  * A key in B's record must equal G's.
  * A key only in G is accepted ONLY if it is `w_tac` at 1.0 or `w_cons` at 0.5, and B's argv does not set its flag.
  * Every other difference still refuses, including a key G lost.
  * Nothing else in the reader, the bars, the bootstrap or the decision table changes.
* **Why this is not a goalpost move.**
  * No GRID number had been computed or seen when this was written.
  * The rule admits exactly the values the trainer source proves B trained with.
  * It refuses every other weight difference, as before.
* **The test gap, logged (same class as `D-*` "built, tested, and unreachable"):**
  * The synthetic test world gave B the full `w` dict, so no test had ever shown the reader B's REAL record shape.
  * New tests pin that shape on the real banked M2b records, at both seeds.
  * Six mutants of the amendment all turn the tests RED (`raw/2026-10-22-grid/a2/mutate_a2.out`).
* **Resumption:** the chain is relaunched unchanged from the new tip.
  * The queue skips G s0's finished training (`summary.json` exists) and its four evals (`npz` exist).
  * So the first new computation is the committed seed-0 read itself. Seed 1 trains only if that read says CONTINUE.
  * The first run's chain and guard logs are kept as `chain_v8f_grid_run1.out` / `v8fgrid_guard_run1.out`.
