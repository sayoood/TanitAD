# PREREG_V8F_HIST instruments: build report (2026-10-23)

Built against the export of tip `6bfeced5`. Nothing is committed, staged or pushed; every file below exists ONLY in the build export (see the manifest). CPU only (`CUDA_VISIBLE_DEVICES=-1`), no ssh, no GPU, the chain was never run for real.

## HEADLINE: the PREREG's budget sentence is false for the trainer as written. Amendment 1 is needed BEFORE the chain can be armed (F-1)

`PREREG_V8F_HIST.md` lines 27 and 30 say H trains for **19,164 steps** and that the **window set is B's**. The trainer cannot do both with `--hist 10`:

* `build_data` calls `D.train_windows(store, J, clips, hist=cfg.hist)` (`stack/scripts/v8f_mini_train.py:338-339`) and `train_windows` starts each clip at `t = hist - 1` (`stack/tanitad/data/v8f_mini_data.py:512`).
* The step budget is sized from that count: `S = ceil(n_windows_all_clips / batch)`, `total = S x steps_mult` (`v8f_mini_train.py:405-406`).

MEASURED (`raw/2026-10-23-hist/measure_hist.json`, D3, box 2's own train manifest `manifest_train.jsonl`, sha256 `661f5d45...fe12`, 4,572 rows, 4,512 clips after the RA-1 holdout):

| | windows | S | steps (`--steps-mult 4`) |
|---|---|---|---|
| B = banked M2b (`summary.json`) | 306,566 | 4,791 | 19,164 |
| the same arithmetic at hist 4, from the manifest | 306,566 | 4,791 | 19,164 (**reproduces B exactly**, so the method is validated) |
| H at hist 10 | **279,494** | **4,368** | **17,472** |

H loses windows t = 3..8 of every one of the 4,512 clips (27,072 windows, 8.8 %) and trains **1,692 steps (8.8 %) fewer** than B. Two consequences beyond the PREREG's text:

1. **The chain stops at the gate, on purpose.** `chain_v8f_hist.sh` asserts `n_windows_all_clips == B's n_windows` and `steps_total == 19164` exactly as the PREREG states them, and the reader refuses `H steps != B steps`. The gate's two assertions carry the message "BUILD.md F-1". The test `test_FINDING_hist_10_drops_the_first_six_training_windows_of_every_clip_and_shortens_the_run` pins the measured behaviour and FAILS ON PURPOSE if the trainer ever decouples the window start from `hist`.
2. **H never trains on padded history, but 282 of the 2,399 eval windows (11.8 %) have padded history** (141 at t = 3 with 4 real frames, 141 at t = 7 with 8). B trained on t = 3 windows (4 real frames, none padded at hist 4). The PRIMARY decides on all 2,399 windows, so it contains a train/eval mismatch that only H has. The reader reports the PRIMARY on the 2,117 windows with t >= 11 (fully real at both hist values) beside it, deciding nothing.

### Options for Amendment 1 (a PI / Master Mind decision; the instruments change in the places named)

| option | what it is | what changes |
|---|---|---|
| **A. accept the lever's natural budget** | H = B + `--hist 10`, 17,472 steps over 279,494 windows. The PREREG's "19,164" and "window set is B's" are corrected to the measured numbers. Fewer steps most plausibly works AGAINST H (ESTIMATED, not measured), so a PASS would be conservative and a FAIL weaker evidence. | chain gate: `M2B_STEPS`, the two asserts at lines 186-188 compare with 279,494 / 17,472 (or with `train_windows(hist=10)` of B's manifest); reader: drop `H steps == B steps` for a check against the declared 17,472; `test_FINDING...` stays as is |
| **B. equal steps** | add `--steps 19164` to H's argv (an existing trainer flag, `v8f_mini_train.py:105`, meant as a smoke override). Equal optimiser steps, 279,494 windows. | queue: `--steps 19164` on gatehist/trainhist; reader: `--steps` becomes the one allowed argv difference besides `--hist`/`--out`; chain gate: `steps_total == 19164` stays, the window assert changes |
| **C. B's window set** | give the trainer a window-start flag (`--window-start 3`) so H trains the t = 3..8 windows with padded history, exactly like the eval sees them. Keeps 306,566 windows and 19,164 steps and removes the train/eval pad mismatch. | a TRAINER change (flag + G-DVB entry + `built_levers`), not an instrument change; outside this build's brief |

Recommendation (ESTIMATED, not measured): C is the cleanest lever (one change, B's budget, padded history trained), A is the cheapest and needs no trainer edit. Either way the PREREG must say which, before a GPU-hour is spent.

## What was built (priority order)

| # | deliverable | state |
|---|---|---|
| 1 | `code/v8f_hist_read.py` (571 lines) | built, tested, run on the REAL banked M2b (`real_proofs`) |
| 2 | queue stages `:gatehist :trainhist :evalhist` in `code/v8f_mini_box2_queue.cmd` (CRLF 523/523) | built, dry-run and real-line tests |
| 3 | `code/box2_ops/chain_v8f_hist.sh` (215 lines) | built, `--dry-run` tested against my edited queue |
| 4 | `stack/tests/test_v8f_hist.py` (1,676 lines) | **113 passed** |
| 5 | `raw/2026-10-23-hist/measure_hist.{py,json,out}` | D1-D4 below |
| 6 | `raw/2026-10-23-hist/mutate_hist.{py,out}` | **128 / 128 mutants RED** on the final tree (+ 1 redundant guard, named, not run); round 1 caught 120 / 128 and the 8 survivors were closed (below) |
| 7 | this file | |
| + | `raw/2026-10-23-hist/real_proofs.{py,out}`, `check_inputs_identity.json` | the reader's `--check-inputs` on the REAL banked B side, plus three refusals on real data |

`stack/scripts/v8f_mini_eval.py` was NOT edited: its checkpoint-driven rebuild (`cfg` from the checkpoint, `V8FMiniWindows(hist=cfg.hist, hmax=cfg.hmax)`, strict `load_state_dict`) already reaches `--hist 10` and refuses a time table of another size (pinned by `test_eval_of_a_hist_10_checkpoint_builds_hist_10_and_gathers_ten_frames`, `test_eval_refuses_a_checkpoint_whose_history_disagrees_with_its_time_table`).

## Test results

| suite | result |
|---|---|
| `stack/tests/test_v8f_hist.py` | **113 passed** (54-70 s) |
| `stack/tests/test_v8f_grid.py` | **137 passed** (83 s), re-run on the final tree |
| `stack/tests/test_v8f_pgt.py` | **29 passed** (49 s), re-run on the final tree |
| `stack/tests/test_text_encoding_is_explicit.py` (the encoding lint) | **3 passed** (7 s) |

All with `CUDA_VISIBLE_DEVICES=-1`. The tests cover: **reachability** (argv `--hist` to `make_config`, the G-DVB gate, the built `time` table, `_scene`'s exact gather for S/T/O at both hist values, a NON-ZERO gradient on every one of the ten time rows with exact-0.0 controls, the trainer end to end, the eval hop), **the reader** (decision table, identity control exactly +0 [0, 0], M-LAT, F and the 2F bar, the t >= 11 subset, the committed window counts, every refusal as rc 2 with its reason, B's real record lacking `w_tac`/`w_cons`), **queue and chain** (static, dry-run = run line, real lines execute exactly what they print, failure noticing, skip-when-finished, gate assertions on a good and on every bad selfcheck), and the guarded module-level load.

### The reader's beside-reads (decide nothing)

Four families (inherited from GRID's `famfn`), per-group and per-latcons-slot present-slot RMSE with the four timing slots (`turn_t_start_s`, `turn_t_end_s`, `turn_d_start_m`, `lc_t_cross_s`) pooled beside the eight geometry slots, Lambda, GT-oracle ADE6, the PRIMARY on the t >= 11 subset (2,117 windows) and on its padded complement (282), distance keeping at K = 10 and (when the block exists) K = 30, parameter counts, steps/s, the built hist/hmax/time rows read from the checkpoint. `--check-inputs` validates the B side alone BEFORE any training.

### On the REAL banked M2b (`real_proofs.out`, `check_inputs_identity.json`)

`HIST INPUTS OK: base M2b (b_hist 4, b_hmax 16); F 0.1207; LAMBDA(B) +0.2678 / +0.1667; latcons RMSE(B) 0.6861 / 0.6846 SD; windows 2399 (t>=11 2117, t=3 141, t=7 141); B windows 306566, steps 19164; identity control exactly +0 [0, 0]` (36 identity contrasts x 2 seeds, every one exactly +0 [0, 0]; the 0.6861 SD is the number GRID's real proof recorded). Three further runs each refuse with rc 2 on a broken premise (B's argv naming `--hist 8`; B's two seeds disagreeing about hmax; a banked eval file missing). The K = 10 lead block was not in this export, so the run named it explicitly (`grid/export/.../2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz`, the banked file).

## D1-D4, MEASURED (`measure_hist.json`)

* **D1 parameters: PASS.** A CPU build of `V8FMini` from the trainer's own `make_config` with M2b's flags: hist 4 = 7,214,495 trainable parameters (equal to the banked `summary.json`, group by group), hist 10 = 7,216,031: **+1,536, all in `adapter`**, `cond/S/T/O/WM` +0; hmax 16 at both.
* **D2 initialisation.** Under one torch seed, 95 tensors are identical across hist 4 / 10 and 180 differ. Every identical tensor is explained (drawn before `time`: `adapter`, `pos`; buffers; the P10 prior from its own generator; AdaLN gate biases set to a structured constant, `hjepa.py:357-360`; plain constants). Zero unexplained. The first four `time` rows equal the hist-4 table. Every RANDOM weight built after `time` (cond embeddings, the three readers, heads, WM) comes from a shifted RNG stream: the lever is also an init change, as the PREREG says, and cannot be separated from it by this design.
* **D3 held ages near a clip's start (training only).** The restated clamp equals the real `V8FMiniWindows.item` on 810/810 (t, draw, hist) cells. Windows whose sampled S age CAN clamp differently between hist 4 and 10 are those with 9 <= t <= 14: **27,072 windows** (6 per clip x 4,512). Under the sampler's draw (`ChunkedSampler.epoch`: a_t uniform 0..2, a_s = a_t + uniform 0..4; 15 equiprobable pairs) the clamp changes on 14/12/9/6/3/1 of 15 draws at t = 9..14, i.e. **13,536 changed draws per epoch = 4.8 % of H's windows**. Separately, windows t = 3..8 (27,072) are not trained on at all at hist 10 (F-1).
* **D4 padding at eval: PASS.** `raw/eval_grid_windows.json` (2,399 windows, 141 episodes): hist 10 sees 141 windows with 4 real frames (t = 3), 141 with 8 (t = 7), **2,117 fully real**; hist 4 sees 2,399 fully real. t >= 9 and t >= 11 are the same set (t = 3 mod 4).

## Other findings (candidate Amendment 1, items beyond F-1)

* **F-2 (above):** H's training never contains padded history; the PRIMARY includes 282 windows that do. The t >= 11 beside-read exists for exactly this and decides nothing.
* **F-3, the gate cannot read the built hist/hmax from `selfcheck.json`:** `built_levers` records neither. The chain derives hist 10 from the +1,536 adapter delta (and `(delta)/256 + 4 == 10`) and from the queue's printed line; the trainer's own G-DVB compares `cfg.hist` with argv in-process, and the reader re-reads the BUILT hist, time rows and hmax from `ckpt_last.pt`. Adding `hist`/`hmax` to `built_levers` would close it at the gate (a trainer edit; not made).
* **F-4, B's record is silent on `--hist`, `--hmax`, `--w-tac`, `--w-cons`:** B's argv carries none of the four and `levers.w` only `w_pen/w_pen_self/w_wm`; the reader reuses GRID Amendment 2's `w_diff` (fill only at the proven defaults 1.0 / 0.5 while B's argv stays silent). A test proves a default H reads equal to the real record, and that the function is GRID's own object.
* **F-5, clips shorter than 40 frames:** none (min 96, 4,133 of 4,512 clips have 101); so the PREREG's windows-per-clip arithmetic is exact.
* **F-6, an inherited `Refusal` escaped as a TRACEBACK (rc 1), not rc 2 -- found by the odd-frame mutant, fixed in THIS reader only.** `v8f_pgt_read.build_ctx` calls `DK.join_block(...)` unwrapped, and `v8f_dk_read.Refusal` is an `Exception`, not a `SystemExit`, so a GT window with no row in the lead block ("GRID MISMATCH: 1 of 72 eval windows have no row in the lead block") reached `main` as an uncaught exception. `v8f_hist_read.main` now converts it (`type(e).__name__ == "Refusal"` -> message on stderr, rc 2; pinned by `test_reader_turns_an_inherited_refusal_into_rc2_not_a_traceback`). **`v8f_pgt_read.py`, `v8f_grid_read.py` (and, by the same pattern, `v8f_wc_read.py`) still have it** (their `main` catches only `OSError/KeyError/ValueError/SystemExit`): UNVERIFIED for WC (not read); MEASURED by source for GRID/PGT. The chains' `|| stop` treat any nonzero rc alike, so no chain is wrongly passed; the cost is a traceback instead of a reason. A one-line fix in `PGT.build_ctx` (wrap `join_block` like `dk_read` does) would close all of them.

## Mutation proof (`mutate_hist.{py,out}`)

`python mutate_hist.py <repo root> <python> [--set reach|reader|queue|chain|all] [--check-anchors] [--part k/n] [--only "a||b"]`. Every anchor occurs EXACTLY ONCE (asserted before anything runs); a mutant is tried first on its own group's tests (`-k`), then on the whole file; the mutated file is restored in a `finally` and compared byte for byte at the end (`assert`); the unmutated tree must be GREEN before and after. Nothing is deleted. Because the run mutates source files, it ran only on **byte-identical copies of the export** (`scratchpad/hist_m1..3`), never in the export itself; the six round-1 runs went in parallel (3 copies), and `mutate_hist.out` concatenates every run.

| group | mutants | what they break | RED, round 1 | survivors of round 1 (closed) | RED, final |
|---|---|---|---|---|---|
| reach | 15 | `make_config` ignoring `--hist` (also with a G-DVB gate blind to it: SILENT), the trainer's window set and step budget, the dataset's clamp, `train_windows`' start, the model's gather (4 frames at hist 10, reversed time rows, a frame too old: SILENT), the time table's rows / optimiser group, the eval's rebuild | 14 | the trainer building its DATASET at hist 4 (the clamp) stayed green: `_train` now spies `V8FMiniWindows.__init__` and the trainer test asserts `(hist, hmax)` | 15 |
| reader | 43 | the hist-flag / one-lever / steps / clip-list / launch-gate refusals, the parameter-delta checks (total, per group, the 128-wide expectation, the check not run), every checkpoint fact, the eval-ckpt and eval-grid checks, `w_diff`'s fill, `s0` reading both seeds, the subset / window-count / slot logic, F, the identity control, the three verbatim next-step texts | 37 of 42 | (a) H/B on different eval grids: PGT's own per-arm check had masked mine, so the test now shifts every H file alike; (b) the identity control not run in the real read: the test asserts `identity_control` is present and exactly +0 [0, 0]; (c) B's two seeds disagreeing about hmax: new `check-inputs` test; (d) an odd GT frame read as a cache step: a direct `window_t` test; (e) a slot "present" whenever the window has a row: the world now exposes its own per-slot present counts and the test pins them | 43 (incl. 1 new, below) |
| queue | 28 | `--hist` dropped / 8 / a second lever / the grid / the seed from the stage name, `--selfcheck-only` dropped, the printed line differing from the run line (train AND gate), shared run dir with B, B's checkpoint evaluated, tag collision, E0-E3 each changed, the failure and exit-0-without-summary checks, the skips, the missing-seed refusals, the three dispatch lines | 28 | none | 28 |
| chain | 43 | the dry run's stage list, markers, busy marker, the 1,320-poll wait (vs the guard's cap), the predecessor and its wait, check-inputs ignored, the three job names, the seed-0 STOP, seed 1, the final stage, the pulls, the P10 guard file, the guard-marker check, the gate heredoc's assertion lines (19 mutants, one per check) and its exit code | 41 of 43 | the busy marker left behind by `finish()` (new test pins `finish()`: telemetry stop, marker removed, DONE after both); the gate's "10 rows" line: a REDUNDANT guard (see below) | 42 |
| **total** | **128** (+1 redundant) | | **120 / 128** | 8 | **128 / 128** |

**The redundant guard (named, not run).** The gate's `4 + (adapter delta) // 256 == 10` recomputes from the same two adapter counts what the per-group delta assertion above it already forces (`pg.adapter - pb.adapter == D`, D = 6 x 256, IS "10 rows"), and a missing `adapter` key already fails that loop. It can fire only if D itself is mis-edited, which its own mutant ("the time table row is 128 wide") shows is caught. Round 1 ran it and it stayed GREEN for that reason; it is listed in `EQUIVALENT` in the script, not counted.

**One mutant was added in round 2** ("an inherited Refusal escapes as a traceback, not rc 2"), and it exposed a real defect (F-6 below).


## Deliverable manifest (WHERE EVERYTHING LIVES)

All paths are inside the export `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hist` (PKGDIR). ⛔ **EVERY file below exists in ONE place only (PKGDIR, a scratchpad export), in no repo, no mirror, no hold. Nothing is staged.** Integration (copying them into the tree, a `qland` hold, the commit) is the Master Mind's.

| file | status | note |
|---|---|---|
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_hist_read.py` | NEW, LF | the reader |
| `.../code/v8f_mini_box2_queue.cmd` | EDITED, CRLF 523/523 | three stages added + their dispatch lines + the usage line; every other line byte-identical (a test pins it) |
| `.../code/box2_ops/chain_v8f_hist.sh` | NEW, LF | the chain |
| `stack/tests/test_v8f_hist.py` | NEW, LF | 113 tests |
| `.../raw/2026-10-23-hist/measure_hist.py`, `.json`, `.out` | NEW | D1-D4 |
| `.../raw/2026-10-23-hist/real_proofs.py`, `.out`, `check_inputs_identity.json` | NEW | the reader on the real banked B |
| `.../raw/2026-10-23-hist/mutate_hist.py`, `.out` | NEW | mutation proof |
| `.../raw/2026-10-23-hist/BUILD.md` | NEW | this file |

Not touched: `PREREG_V8F_HIST.md`, every register, `stack/scripts/v8f_mini_eval.py`, `stack/scripts/v8f_mini_train.py`, the model, the data module. `raw/2026-10-23-hist/mk_hold20.py` was already in the export (the Master Mind's hold builder) and is unchanged.

Inputs read from OUTSIDE the export (read-only): box 2's `manifest_train.jsonl` (`C:/Users/Admin/box2_ship/logs/v8f_grid/`, D3), and the banked K = 10 lead block under `scratchpad/grid/export/` (real proof only). Scratch copies of the export used by the mutation run (`scratchpad/hist_m1..3`, ~720 MB each, plus `hist_real_*` staging dirs) were NOT deleted; prune them (PI-only delete) when convenient. C: has ~16 GB free.

## How to arm it (after Amendment 1)

1. Amend the PREREG (F-1) and make the instruments say the same: the chain gate lines 186-188, the reader's `steps` equality, the FINDING test.
2. `V8F_QUEUE=<queue> bash chain_v8f_hist.sh --dry-run` prints the exact line every stage runs.
3. The chain waits for `chain_v8f_grid.out` to print `ZZV8FMINI-DONE`, runs `--check-inputs` on the banked B side, then the gate, then seed 0 and (on CONTINUE) seed 1. Training wait 1,320 x 30 s = 11 h > the guard's 10 h cap.



---

# Amendment 1 implementation (2026-10-23)

H = B + `--hist 10 --window-start 3`: ONE lever set by two flags. Implemented end to end, instruments updated, everything green on the final tree. Nothing is committed, staged or pushed; every file below exists ONLY in the build export (PKGDIR). CPU only (`CUDA_VISIBLE_DEVICES=-1`), no ssh, no GPU, the chain was never run for real.

## Headline

* **F-1 is closed in the trainer.** `--window-start` fixes the first training window of every clip (`train_windows(start=...)`) AND the origin of the held-age clamp (`V8FMiniWindows.item`: `room = max(t - window_start, 0)`). ABSENT it is `hist - 1`, so every arm that does not set it is bit-identical (proved against the saved pre-change modules and against frozen function bodies). At `--hist 10 --window-start 3` the window list, every held age (all keys of every item, 15 sampler draws) and the step budget equal hist 4's EXACTLY.
* **F-3 is closed.** `built_levers` records `hist` (read from the BUILT time table's rows, not echoed from the config), `hmax` and the EFFECTIVE `window_start`; the new flag has its G-DVB entry, and the gate reads the flag back from three places (the model's levers, the measured first `t` of the built window list, the built dataset's clamp origin).
* **ESCALATION (one decision, outside my edit list): `v8f_mini_eval.py:270` and `v8f_t1_eval.py:379` build `V8FMiniWindows` WITHOUT `window_start`.** Effect for an H checkpoint (`ck["cfg"]["window_start"] == 3`): the evaluator's held-age clamp origin is 9 (`hist - 1`) instead of the 3 H trained with. ZERO where ages are fresh (`--ages fresh`: a_t = a_s = 0, `v8f_mini_eval.py:295`; `ages_for(..., "fresh")`, `v8f_t1_eval.py:126-127`). NON-ZERO where ages are not: `v8f_mini_eval.py --ages` DEFAULTS TO `cycle` (`:629`; a_t = i % 3, a_s = a_t + (i // 3) % 5 at `:291-295`, and again in `_ckpt_to_now` `:503-506`), and `v8f_t1_eval.py` uses the periodic schedule (ages 3 / 6, `ages_for` `:123-130`). The ages a window can draw are clamped to `room`, so windows with t < 15 are held harder than H trained (MEASURED on the synthetic world, `a1/measure_a1.json`: the clamp origins 9 vs 3 give different items at exactly t = 4..14). Which eval windows fall in t = 4..14 depends on the eval grid file (UNVERIFIED here: the grid json is not in this export). **The DEFAULT `v8f_mini_eval.py` run is therefore affected, not only the optional ones.** Fix, one keyword each: `window_start=cfg.window_start`. Not made in the first pass (outside the brief) -- **MADE afterwards on the coordinator's approval, see "Eval-side fix" at the end of this section**; before it, treat `v8f_mini_eval.py` (default ages) and `v8f_t1_eval.py` reads of an H checkpoint as carrying a known clamp-origin mismatch on early-clip windows, or run `--ages fresh`.
* The PREREG text itself (Amendment 1 "landed") is not in this export (`Project Steering/PREREG_V8F_HIST.md` has no Amendment); the instruments follow the brief's wording. `e46909f0` is not an object in `E:\Projects\TanitAD` either, so the "pre-change" references are the export's own files (byte-identical to the tip per the brief), saved under `raw/2026-10-23-hist/a1/` with sha256 (below).

## What changed (file:line, final tree)

| file | change |
|---|---|
| `stack/tanitad/data/v8f_mini_data.py` | `WINDOW_START_ABSENT = -1` (:450); `effective_window_start(hist, window_start)` (:453: absent = `hist - 1`, set must lie in `[0, hist - 1]` else `ValueError`); `V8FMiniWindows(..., window_start=None)` stores `self.window_start` (:483) and clamps `room = max(t - self.window_start, 0)` (:498); `train_windows(..., start=None)` starts at `effective_window_start(hist, start)` (:528-538) |
| `stack/tanitad/models/v8f_mini.py` | `V8FMiniConfig.window_start: int = -1` (:144) + `window_start_effective` (:148-150) + range check in `__post_init__` (:161); `built_levers` gains `hist` (`int(self.time.shape[0])`), `hmax`, `window_start` (:741-742) |
| `stack/scripts/v8f_mini_train.py` | argparse `--window-start` (:123, int, default -1); `FLAG_CHECKS["window_start"] = "lever:window_start"` (:66); `make_config` passes it (:164); `declared_vs_built` checks built `hist` / `hmax` / `window_start` against the declared and the data read-backs `window_t_min` / `ds_window_start` against the declared effective start (:201-211; a DECLARED flag with a missing read-back is refused, an absent flag with a missing key is not); `build_data` hands `start=cfg.window_start` to BOTH `train_windows` calls (:354-355) and records `window_start` + the measured `window_t_min` (:357-358); `main` hands the dataset the flag and records `ds_window_start = ds.window_start` (:414-416) |
| `code/v8f_hist_read.py` | argv difference to B is exactly `--hist 10 --window-start 3` + `--out` (`WS_FLAG`, `drop`, :208-218); B naming `--window-start` is refused (:212, and `check_b_side` :521); H steps == B steps (kept) and now H `n_windows` == B's (:196); H's `summary.json` and `selfcheck.json` levers MUST carry built `hist` 10 / `window_start` 3 / `hmax` = B's 16 (strict for H, :225-232); B's levers are compared KEY BY KEY, only where B's record has the key (:241-244; the banked M2b has none of the three: absent is never "different"); `_SKIP_LEVERS_H` excuses only `hist` (:98); the checkpoint's `cfg.window_start` must be 3 (`ckpt_facts`); `selfcheck.info` read-backs (`window_start`, `window_t_min`, `ds_window_start`) must be 3 where present; the beside-read `built` carries `window_start` |
| `code/v8f_mini_box2_queue.cmd` | `:gatehist` and `:trainhist` carry `--hist 10 --window-start 3` (:437, :452); comments updated; **CRLF 525/525** (was 523; verified `b.count(b'\r\n') == b.count(b'\n')`); stages still flat; dry-run line == run line (pinned) |
| `code/box2_ops/chain_v8f_hist.sh` | gate heredoc: `M2B_WINDOWS=306566` passed as a 6th argument (:51, :170); asserts `n_windows_all_clips == B's n_windows == 306,566` and `steps_total == 19,164` again (:188-192), built `hist == 10`, `hmax == 16`, `window_start == 3` from the selfcheck LEVERS (:197-199), `window_start / window_t_min / ds_window_start == (3, 3, 3)` from its info (:200), the queue's printed line carries `--window-start 3` exactly once (:211); two pre-ship greps that the tip's `built_levers` and trainer have the new key / flag (:121-122); the F-1 hint now says "unless --window-start reaches it" |
| `stack/tests/test_v8f_hist.py` | 113 -> **134** tests: the synthetic H run records, queue / chain / gate tests and the reader's refusal table updated; 21 new cases (argv, built levers, checkpoint, selfcheck read-backs, n_windows, B key-by-key in both directions); the FINDING test keeps pinning the trap (hist 10 ALONE drops t = 3..8) AND asserts that `--window-start 3` restores B's set and budget |
| `stack/tests/test_v8f_window_start.py` | NEW, **60 tests** (below) |
| `raw/2026-10-23-hist/mutate_hist.py` | re-anchored on the changed lines; **160 mutants** (was 128) |
| `raw/2026-10-23-hist/real_proofs.py` | +1 refusal (B naming `--window-start`); re-run on the REAL banked M2b side, `a1/real_proofs_a1.out`: still `HIST INPUTS OK ... B windows 306566, steps 19164; identity control exactly +0 [0, 0]`, four refusals rc 2 |

## The new test file (`stack/tests/test_v8f_window_start.py`, 60 passed, 0 skipped)

(a) default bit-identity: parser / config / effective default at hist 1, 4, 7, 10 (a default of 3 is invisible at hist 4, visible at hist 10); `train_windows` against the FROZEN pre-change body at hist 1, 2, 4, 7, 10 x stride 1, 3 on lengths including clips too short for a window; held ages against the frozen clamp on every window and draw; the WHOLE saved pre-change data module key for key (`a1/pre_a1_v8f_mini_data.py.ref`); the saved pre-change model: state_dict, `forward_infer`, and `built_levers` minus exactly the three new keys. (b) hist 10 + window-start 3: the window list equals hist 4's window by window; every `item()` (all keys) equals hist 4's under all 15 draws; the CONTROL (hist 10 alone) differs exactly at t = 4..14; the sampler draws equal batches over equal lists. Validation: 5 out-of-range values refused at five layers each, plus a separate trainer-level case, 5 in-range accepted, the trainer refuses before doing anything. (c) box 2's real manifest (sha256 `661f5d45...`): 306,566 windows / 4,791 / 19,164 at hist 4 AND at hist 10 + window-start 3 (list equal), 279,494 / 4,368 / 17,472 at hist 10 alone; the budget formula proved against the trainer's selfcheck. (d) t = 3: 6 padded frames masked, 4 real; padded content moves NOTHING (exact equality of `traj30`/`traj60` and every prediction under a scramble of the 12 pre-clip features), with a control that a REAL frame does move the output; loss and gradients finite; real frames per layer = `min(tick + 1, 10)` at t = 3, 4, 7, 8, 9, 12. (e) G-DVB: passes the built cases; refuses each hop on its own (model levers, window list's first t, dataset's clamp origin), a declared flag with no read-back, a wrong built hist; the REAL trainer (`TR.main`) refuses a `train_windows`, a dataset and a `make_config` that each drop the flag; the trainer run end to end records the flag in argv, both `train_windows` calls, the dataset, the checkpoint cfg, selfcheck and summary levers.

## Test results (final tree, `CUDA_VISIBLE_DEVICES=-1`; per-file lines in `a1/suites_a1.out`)

`test_v8f_window_start.py` **60** | `test_v8f_hist.py` **134** | `test_v8f_grid.py` **137** | `test_v8f_pgt.py` **29** | `test_text_encoding_is_explicit.py` **3** | and every other file that imports `v8f_mini_train`, `v8f_mini_data` or `v8f_mini`: `test_v8f_mini` 29, `test_v8f_tac` 14, `test_v8f_wc` 8, `test_v8f_cz` 6, `test_v8f_beta` 35, `test_v8f_beta_box2` 2 skipped, `test_v8f_t1` 23, `test_yaw_rate_ruling` 26, `test_ra1_family_paired` 10, `test_ra1_hjepa` 73, `test_launch_gate_v8f` 56, `test_v8f_dk` 25, `test_v8f_smooth_read` 4, `test_v8f_mini_full` 4, `test_v8f_mini_full_report` 9, `test_p10_prior` 33. **TOTAL 718 passed, 2 skipped, 0 failed, 21 files.** The 16 files outside the HIST / GRID / PGT / lint set read 355 passed + 2 skipped BEFORE any edit (`a1/baseline_other_pre_a1.out`) and 355 + 2 after: no regression.

## Mutation proof (nothing was run on the export: copies `hist_m1..3`, byte-restored in a `finally` and asserted)

* `a1/mutate_a1.py` (`a1/mutate_a1.out`): **26 / 26 RED** on `test_v8f_window_start.py`, baseline and after GREEN. The five named: flag never reaches `train_windows` (both calls) RED; room still computed from hist RED; `built_levers` omits `window_start` RED; the G-DVB entry removed RED; default changed to 3 RED. Plus each hop: `make_config`, the dataset construction, `V8FMiniWindows` / `train_windows` ignoring their argument, the absent value 0 instead of `hist - 1`, no range check (data and config), the config default 3, the effective value ignoring the flag, the raw field reported instead of the effective one, `hist` / `hmax` omitted, `hist` one too small, the G-DVB lever check off, each of the two data read-backs off, no read-back required, the absent flag held to 3, the "measured" first `t` faked from the asked-for value (the lie hides), the dataset origin faked from the config, the dataset origin unrecorded.
* `mutate_hist.py` (`a1/mutate_hist_a1.out`): **160 / 160 RED** (128 before; now reach 16, reader 59, queue 33, chain 52; baseline and after GREEN), run as two halves in `hist_m2` / `hist_m3`, and the 16 reach mutants re-run on the FINAL trainer in `hist_m1` (16 / 16). New mutants: the reader's `--window-start` argv / B / `n_windows` / built-key / key-by-key / checkpoint / read-back refusals (16), the queue's `--window-start` drops and wrong values (5), the chain's new assertions and greps (10), and the absent-value mutant (1). Four reach mutants turn red as a FIXTURE ERROR, not an assertion: the trainer's own launch gate refuses the inconsistent build (e.g. "the step budget S is sized on hist 4" now makes the gate see a first window at t = 3 where hist 10 alone declares 9). That is the gate working.

## MEASURED (artifacts in `a1/`)

* `measure_a1.json` / `.out` (box 2's train manifest sha256 `661f5d45...`, 4,512 clips after `--exclude 540:600`): B hist 4 = 306,566 windows, S 4,791, 19,164 steps (equals the banked M2b); H hist 10 + window-start 3 = **306,566 / 4,791 / 19,164 and the same LIST**; H hist 10 alone = 279,494 / 4,368 / 17,472 (27,072 windows and 1,692 steps lost). Held ages, 6 clips x t = 0..40 x 15 draws = 3,690 cells: hist 10 + window-start 3 differs from hist 4 in **0**; hist 10 alone in 690 (t = 4..14).
* The parameter delta is unchanged: +1,536, all in `adapter` (`measure_hist.json` D1); the flag adds no parameter (it is a data lever).

## Unexpected / worth knowing

1. **A probe blind by construction, caught while writing (d).** The residual head is zero-init, so `traj30` equals the frozen prior whatever the scene: a "padded frames cannot move the output" test passes vacuously on a fresh model. The test perturbs `head_o[2]` first and carries a control (a real frame DOES move the output). `forward_infer` also returns `adapt`, the per-frame encoder output of all `hmax` frames before any mask: it legitimately changes when a padded frame does and is excluded from the equality.
2. **`hmax` in the gate is not an independent declared value** (pre-existing): `declared_vs_built` takes `cfg.hmax` as the declared side (`--hmax 0` = auto), so `built.hmax == cfg.hmax` is a self-compare; the chain gate's `hmax == 16` is the real check.
3. **I first left `window_start` in the gate's generic `cfg.` field loop as well as in the lever check; the two guard the same thing and each hid the other's mutant. Removed** (the lever check + two data read-backs remain), then every suite and mutant set was re-run on the final trainer.
4. **`levers` records change shape** (three new keys). The GRID / PGT / HIST readers compare `levers` over the INTERSECTION of keys, so B's banked record (no such keys) reads unchanged against a new run; a re-run B would carry hist 4 / hmax 16 / window_start 3 and is accepted (pinned both ways).
5. The `.cmd` queue is 525 lines now (the earlier text said 523): two comment lines added; every `rem` line avoids `>` and `|`.

## Manifest (PKGDIR = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hist`; PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`)

⛔ EVERY file below exists in ONE place only (PKGDIR), in no repo, mirror or hold. Nothing is staged.

| file | status |
|---|---|
| `stack/tanitad/data/v8f_mini_data.py` | EDITED, LF |
| `stack/tanitad/models/v8f_mini.py` | EDITED, LF |
| `stack/scripts/v8f_mini_train.py` | EDITED, LF |
| `stack/tests/test_v8f_window_start.py` | NEW, LF, 60 tests |
| `stack/tests/test_v8f_hist.py` | EDITED, LF, 134 tests |
| `PK/code/v8f_hist_read.py` | EDITED, LF |
| `PK/code/v8f_mini_box2_queue.cmd` | EDITED, CRLF 525/525 |
| `PK/code/box2_ops/chain_v8f_hist.sh` | EDITED, LF |
| `PK/raw/2026-10-23-hist/mutate_hist.py`, `real_proofs.py` | EDITED, LF |
| `PK/raw/2026-10-23-hist/BUILD.md` | EDITED (this section appended) |
| `PK/raw/2026-10-23-hist/a1/mutate_a1.py`, `mutate_a1.out` | NEW |
| `PK/raw/2026-10-23-hist/a1/mutate_hist_a1.out` | NEW (160 mutants + the reach re-run) |
| `PK/raw/2026-10-23-hist/a1/measure_a1.py`, `.json`, `.out` | NEW |
| `PK/raw/2026-10-23-hist/a1/real_proofs_a1.out` | NEW |
| `PK/raw/2026-10-23-hist/a1/suites_a1.out`, `baseline_other_pre_a1.out`, `run_suites.py` | NEW (per-file test results; the harness) |
| `PK/raw/2026-10-23-hist/a1/pre_a1_*.ref` (8 files) | NEW: the pre-change data / model / trainer / reader / queue / chain / hist test / mutate script, saved before the first edit (sha256 prefixes: data `dfd688ef`, model `f470191f`, trainer `daf4dd9d`, reader `7b401cd4`, queue `eb5c8ce2`, chain `49a0216a`, hist test `5fc4dc7c`, mutate `b0215b4a`); `test_v8f_window_start.py` loads the data and model ones |

Not touched: `PREREG_V8F_HIST.md`, every register, `measure_hist.*`, `mk_hold20.py` (`v8f_mini_eval.py` and `v8f_t1_eval.py` were touched afterwards: "Eval-side fix" below). Scratch left behind (PI-only delete): `scratchpad/hist_m1..3` (now synced to the final tree, except `hist_m2/m3` trainer which differs by one removed tuple entry), `scratchpad/hist_real_a1_051720` (~MB), `scratchpad/a1/` (edit scripts and raw mutation logs).

## Eval-side fix (approved after the first hand-back; closes the escalation above)

**What.** `stack/scripts/v8f_mini_eval.py:270-271` and `stack/scripts/v8f_t1_eval.py:379-380` now pass `window_start=cfg.window_start` to `V8FMiniWindows`. The dataset the evaluator builds therefore clamps held ages from the CHECKPOINT's origin (an H checkpoint, cfg.window_start 3: origin 3; any other: `hist - 1`). A pre-amendment checkpoint (M2b, GRID, PGT: no `window_start` in the saved cfg) loads with the config default -1 = absent and is scored with the origin `hist - 1`, as before. Nothing else in either script changed (`grep -n "hist - 1\|\.hist\b"` finds no other clamp-origin dependence in `stack/scripts/v8f_*.py`). Pre-fix copies, saved before the edit: `a1/pre_a2fix_v8f_mini_eval.py.ref` (sha256 `35584030`), `a1/pre_a2fix_v8f_t1_eval.py.ref` (`80e0e863`); fixed files `7224a140` / `661d7ee7`.

**What the fix prevents (MEASURED, `a1/measure_a2fix.py` / `.json` / `.out`).** On the REAL eval grid (`raw/eval_grid_windows.json`, sha256 `659c3b65...`, 2,399 windows, 141 episodes, t = 3, 7, ..., 71) with `--ages cycle`'s own rule (`v8f_mini_eval.py:291-295`: a_t = i % 3, a_s = a_t + (i // 3) % 5, i = position in `eval_grid_windows`' list) and the real `V8FMiniWindows.item` clamp: **216 of 2,399 windows (9.00 %) would have had different held ages for an H checkpoint WITHOUT the fix** (origin 9 instead of 3). By t: t = 3 **0** of 141 (both rooms are 0), t = 7 **132** of 141, t = 11 **84** of 141, t >= 15 **0** (room >= 6 under both origins). An arithmetic re-derivation of the clamp agrees on every window (0 disagreements); on the synthetic clip all 216 also differ in the tick poses the model reads. Scope: this is the `--ages cycle` (DEFAULT) read of `v8f_mini_eval.py`. `--ages fresh` was never affected. `v8f_t1_eval.py` draws `ages_for(step)` = (step % 3, step % 6) at every rollout step, so its early steps were affected at every early window; that count is not made (the rollout length and the per-step draws are not on the mini grid's list). UNVERIFIED: the numbers assume the evaluator's window order is `eval_grid_windows`' (it is: `v8f_mini_eval.py:269`) and that `wins` is not subsampled (`cmd_run` has no window limit).

**Tests (new section (g) of `stack/tests/test_v8f_window_start.py`: 60 -> 77 tests, +17).** The REAL `EV.main run` and `T1.main run` are driven on the synthetic eval world (14 windows at t = 3, 7, ..., 27), with a spy on `V8FMiniWindows.item` that records the dataset's origin and the ages every call resolved to. (a) hist 10 + window-start 3 vs hist 4, **`--ages cycle` (the default)**: the origin is 3 for both, every call resolves to the same ages, and for the mini evaluator every KEY of every item is equal (2 tests, one per evaluator); the cycle rule itself and the early windows (t = 3, 7, 11) are pinned so the identity is not vacuous; CONTROL: hist 10 without the flag has origin 9 and resolves different ages exactly where the clamp arithmetic says and only at t < 15 (2). (b) a checkpoint whose saved cfg lacks the key (hist 4 and hist 10, made by stripping the key from a real trainer checkpoint) loads with window_start -1 / effective `hist - 1` and every call resolves to the frozen pre-amendment clamp (3 tests); every item equals the SAVED pre-amendment data module's, key for key, built on the same store at the same call (3); and the mini evaluator's `eval_<tag>.npz` arrays and `raw` block are **bit for bit** those of the SAVED pre-fix evaluator (2). (c) a static AST check that every `V8FMiniWindows(` under `stack/scripts/**/*.py` passes `window_start=` (it must SEE the three known callers; `**kwargs` does not count), with its own controls (it flags a missing keyword and a splat, accepts a keyword; it flags both saved pre-fix evaluators) (3).

**Results on the final tree** (`a1/suites_a2fix.out`, the same 21 files, `CUDA_VISIBLE_DEVICES=-1`): **735 passed, 2 skipped (`test_v8f_beta_box2`, as at baseline), 0 failed** (718 + the 17 new). Per file: `test_v8f_window_start` **77**, `test_v8f_hist` 134, `test_v8f_grid` 137, `test_v8f_pgt` 29, `test_text_encoding_is_explicit` 3, `test_v8f_mini` 29, `test_v8f_tac` 14, `test_v8f_wc` 8, `test_v8f_cz` 6, `test_v8f_beta` 35, `test_v8f_beta_box2` 2 skipped, `test_v8f_t1` 23, `test_yaw_rate_ruling` 26, `test_ra1_family_paired` 10, `test_ra1_hjepa` 73, `test_launch_gate_v8f` 56, `test_v8f_dk` 25, `test_v8f_smooth_read` 4, `test_v8f_mini_full` 4, `test_v8f_mini_full_report` 9, `test_p10_prior` 33. By their skip markers, the new tests that need the saved modules / pre-fix evaluators are SKIPPED (not failing) in a `stack/`-only checkout (UNVERIFIED: not run in one).

**Mutants.** `a1/mutate_a1.py` now holds **36** mutants (26 + 10 "EVAL FIX": per evaluator, the keyword dropped, hard-coded 3, `None`, `hist - 1`, and a second unused construction without the keyword that ONLY the static check can see); all anchors occur exactly once; run in three parts on the mutation copies `hist_m1..3`, each part baseline GREEN, every file restored byte for byte and asserted, after GREEN: **36 / 36 RED** (`a1/mutate_a1_a2fix.out`; the earlier 26-mutant run stays in `a1/mutate_a1.out`). The two "unused construction" mutants are killed by `test_every_V8FMiniWindows_construction_under_stack_scripts_passes_window_start` and nothing else; the other eight by the end-to-end evaluator tests. `mutate_hist.py` was not re-run (no instrument it anchors on changed; its group runs inside `test_v8f_hist.py`, 134 passed above).

**Unexpected / worth knowing.** (1) With the fix an H checkpoint's default `v8f_mini_eval.py` read is the one the PREREG describes; any evaluation of an H checkpoint made BEFORE this fix carried the 216-window mismatch. (2) The T1 evaluator builds its dataset on a one-clip proxy store, so its items are model-dependent; the T1 tests therefore compare resolved ages and the origin, and compare full items to the saved pre-amendment dataset only AT CALL TIME on that same store. (3) The static check covers `V8FMiniWindows(` as a bare name or an attribute; an alias (`W = D.V8FMiniWindows; W(...)`) would not be seen.

Files changed by this step (PKGDIR; PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`): `stack/scripts/v8f_mini_eval.py`, `stack/scripts/v8f_t1_eval.py`, `stack/tests/test_v8f_window_start.py`, `PK/raw/2026-10-23-hist/a1/mutate_a1.py`, `PK/raw/2026-10-23-hist/BUILD.md`; new in `PK/raw/2026-10-23-hist/a1/`: `measure_a2fix.py` / `.json` / `.out`, `mutate_a1_a2fix.out`, `suites_a2fix.out`, `pre_a2fix_v8f_mini_eval.py.ref`, `pre_a2fix_v8f_t1_eval.py.ref`.
