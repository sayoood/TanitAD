# PREREG_V8F_GRID: instruments build (no training, no box 2, no GPU)

**Headline: INSTRUMENTS READY** for the pre-registered v8F-mini arm `G = B + --grid pool8x20` (ONE lever). Nothing was trained or run on box 2. The build is a set of files in this package directory; **nothing is on the tip yet. The lander must land them together** (list below). The chain must not be armed before PREREG_V8F_PGT's chain ends.

Base: the export was built on tip `c61f016795e8308cb93b3e868063e0348b42606d`. The only edit to an existing file is `v8f_mini_box2_queue.cmd` (stages added, one usage line changed: `diffs/v8f_mini_box2_queue.cmd.diff`, +87/-1). Re-check that diff against the live tip when landing.

## 1. What was built, in the priority order of the brief

| # | item | where (repo-relative) | status |
|---|---|---|---|
| 1 | reachability + instrument tests, 127 tests | `stack/tests/test_v8f_grid.py` | NEW |
| 2 | the reader (decision rule for B, verdict, M-LAT, identity control) | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_grid_read.py` | NEW |
| 2b | the cache check (every train and eval clip carries 8x20) | same dir `code/check_grid_cache.py` | NEW |
| 3 | box 2 chain | same dir `code/box2_ops/chain_v8f_grid.sh` | NEW (LF) |
| 3 | queue stages `gategrid BASE`, `traingrid SEED BASE`, `evalgrid SEED BASE` | same dir `code/v8f_mini_box2_queue.cmd` | EDIT (CRLF 449/449 kept) |
| 4 | mutation harness + output | `.../raw/2026-10-22-grid/mutate_grid.py`, `mutate_grid.out` | NEW |
| 4 | real-data proofs, eval-mismatch probe, synthetic reads, check-inputs JSON | `.../raw/2026-10-22-grid/` (`real_proofs.py/.out`, `probe_eval_mismatch.py/.out`, `SYNTHETIC_grid_read_*.json` x6, `grid_read_check_inputs_*.json` x2) | NEW |
| - | this file; the queue diff | package root `RESULT_GRID_BUILD.md`, `diffs/v8f_mini_box2_queue.cmd.diff` | NEW |

The trainer, the model, the data module and the eval were **not edited**. The eval needs no refusal edit: a checkpoint/feature size mismatch is already loud (section 4, F3).

## 2. The reachability chain (the "built, tested, unreachable from its caller" class)

Pinned hop by hop on a synthetic world whose two grids DIFFER in shape and content, with real `TR.main` runs:

`argv --grid pool8x20` -> `make_config` n_tok 160 (`v8f_mini_train.py:160`) -> `declared_vs_built` (:180) -> `ClipStore(grid=)` (:337) -> `V8FMiniWindows.item` -> `collate` -> the `feats` the model's loss receives **equal the clip's `pool8x20` array exactly** (and the 4x10 control equals `pool4x10`; both ClipStore branches, yaw and steer) -> the BUILT table `pos` [160, d] -> **a non-zero gradient on every one of the 160 rows, with a prior control that reads exactly 0.0** -> the checkpoint carries `grid`, `cfg.n_tok`, the 160-row table -> `v8f_mini_eval.py:268` reads the grid FROM the checkpoint.

Parameters: G adds exactly (160 - 40) x 256 = **30,720** (the table `pos` is the only per-token parameter), adapter group only. On the banked M2b: adapter 275,712 -> 306,432, total 7,214,495 -> 7,245,215 (MEASURED by the real-size test against the banked summary).

## 3. Verification

**Tests.** `stack/tests/test_v8f_grid.py`: **127 passed** (reachability train+eval 16, reader/instrument 83, cache check 13, queue 10, chain 5), unmutated, clean before and after the mutation run. The v8F neighbours stay green: `test_v8f_pgt/mini/mini_full/wc/dk` 95/95 (94 on the first run; the 95th was the export's CRLF artefact, F7) and `test_v8f_mini_full_report/smooth_read/t1/tac/cz/beta/beta_box2` + `test_launch_gate_v8f` 147 passed, 2 skipped. Tests that load package code tolerate its absence (`skipif`), so a `stack/`-only checkout skips them.

**Reader on real data** (`raw/2026-10-22-grid/real_proofs.out`; zero GPU). B = banked M2b: `--check-inputs` reproduces the PREREG premises (F 0.1207, LAMBDA +0.2678 / +0.1667, latcons RMSE 0.6861 / 0.6846 SD); the paired instrument reads **exactly +0 [0, 0] on 17 contrasts x 2 seeds**, M-LAT included. A SYNTHETIC G built from the banked arrays (names `M2b-8x20`, files `SYNTHETIC_*`, NOT a result of any trained model): a forward-shift regression reads FAIL-NO-MECHANISM, a "60 % to GT, -40 % latcons error" arm reads SUCCESS, "plans = B's, latcons error -30 %" reads FAIL-MECHANISM-OK, on both base branches (B = M2b with the banked evals; B = M2b-pgt from PGT's own proof recipe).

**Queue stages**: dry-run lines equal the real run lines (static + executed with a fake trainer/eval tree on Windows `cmd`), both bases, flat `if errorlevel 1 (... & exit /b 1)`, every failure noticed, finished stages skipped, bad/missing BASE or SEED rc 2, G = B's line with only `--grid` (and `--out`) changed.

**Chain**: `bash -n` clean; `--dry-run` prints, for all 10 stage/base combinations, exactly the queue's own dry-run lines; the gate-assertion heredoc was RUN against synthetic selfchecks (pass + 29 single-field corruptions per base, plus the other base's schedule and the queue's dose, including the parameter delta). The chain itself was **not executed** (needs box 2 and ssh).

## 4. Mutation proof (`raw/2026-10-22-grid/mutate_grid.py`, output `mutate_grid.out`)

Every mutation is one anchored byte-replacement (the anchor occurs exactly once; checked) in a source file or instrument; each must turn `stack/tests/test_v8f_grid.py` RED. The unmutated tree is GREEN before and after; the harness asserts byte-identical restoration.

| group | what is mutated | mutants | RED (final pass, 127 tests) |
|---|---|---|---|
| reach | `make_config` n_tok, the gate, `ClipStore` grid (both kin branches), the checkpoint's grid, the model's position table (size, optimiser group), the eval's grid read, the LRU sizing, **and two silent-scoring models** (tolerates a size mismatch; drops 120 of 160 tokens) | 15 | 15 |
| reader | the base rule (both branches, 4 refusals), the dose constants, verdict logic, PRIMARY (F / 2F / 3F, separation, sign), M-LAT (bar 10/15/0 %, exclusive, separation, sign, definitions), 30 switched-off refusals of the run / checkpoint / eval / PGT-read checks, the identity control, the known values | 76 | 76 |
| queue | `:gridbase` (dose, base validation), `gategrid` / `traingrid` / `evalgrid` (grid, lever, seed, run dir, printed != run, each failure noticed, skip logic, the four eval strings, dispatch) | 43 | 43 |
| chain | the dry-run list, markers, busy flag, training wait, predecessor, base decision, check-inputs / cache-check failure ignored, job arguments, pulls, and 15 gate-assertion lines | 35 | 35 |
| cache | the 8x20 / shape / clip-count / last-line / empty-input checks | 10 | 10 |
| **all** | | **179** | **179** |

**How it got there (kept, not hidden).** First pass against the first 111-test file: **165 of 177 RED, 12 survivors** (a survivor is a gap in the tests, not in the mutant): the steer branch of `ClipStore`; "mechanism = ANY seed"; the M-LAT present-slot-set refusal; the `eid`-vs-B refusal; the latcons-fall and M-LAT-cell identity cells; the "no latcons RMSE" refusal; "G's train log optional"; "known values never checked" in the stage read; `check_grid_cache`'s grids-list check; the "banked GT codes differ" check-inputs refusal; `traingrid` without a seed. Each got a test (127 now) and two mutants were added; the 14 re-ran RED, then **all 179 re-ran against the final file: 179 / 179 RED** (`mutate_grid.out`, sections A, B, C). The `ClipStore` steer-branch survivor was green because the committed config (`--kin-mode yaw`) never reaches it; it is tested because `--kin-mode steer` is one flag away.

## 5. Findings and defects (report, not fixed silently)

- **F1 (derived constant moves with the lever, evidence ESTIMATED from source + MEASURED banked lru).** `v8f_mini_train.py:328`: `lru = max(8, 1.5e9 // (n_max * n_tok * 1024 * 2))` and `:405`: `ChunkedSampler(chunk = max(4, min(48, lru - 2)))`. The banked M2b recorded lru 174, so chunk 48; at 160 tokens lru is ~43 and chunk ~41. The grid lever therefore also changes the sampler's I/O chunking, i.e. the batch composition for the same seed. Not in the PREREG's "one lever". Size of the effect unknown; plausibly inside F (B's two seeds already differ in batch order). The reader REPORTS both lru values (`rows.*.beside.pipeline`); the gate asserts the lru is recorded. Cheap remedy if the PI wants it removed: `--lru 50` on G gives chunk 48 at ~1.7 GB of RAM, but that adds a flag to G's argv and the reader's one-lever check would then (correctly) refuse; the PREREG would need an amendment. PI decision.
- **F2 (PREREG wording).** "latcons present-slot RMSE relative fall >= 12.5 %" does not say whether the RMSE is pooled over slots or over windows. On banked M2b they differ: **0.686 SD (slot-pooled, the PREREG's own number) vs 0.468 SD (window-weighted)**. The reader applies the bar to the slot-pooled RMSE (the PREREG's number), separation to the paired per-window MSE, and reports the other definition and a `definitions_agree` flag, so a disagreement is visible, not silent. Amend the PREREG wording or accept the flag before arming.
- **F3 (no eval edit needed).** A 160-token checkpoint without a `grid` key read as 4x10, and a 40-token checkpoint carrying `grid: pool8x20`, both raise a torch `RuntimeError` ("The size of tensor a (40) must match the size of tensor b (160) at non-singleton dimension 2") and leave no eval file (MEASURED, `probe_eval_mismatch.out`, and three variants in the tests). The mutation harness includes the two ways this could become silent (a model that tolerates the mismatch; one that drops 120 of 160 tokens); both turn tests red.
- **F4 (the trainer's n_tok gate is tautological on the argv path).** `make_config` (:160) and the gate (:180) evaluate the same expression of `a.grid`; the gate cannot catch the flag failing to reach the features. The protection that matters is `ClipStore(grid=)` (:337), the checkpoint's recorded grid/n_tok/pos rows, and the reader's check of all three. Pinned by tests and mutants.
- **F5 (time).** `v8f_guard.sh` stops a run it projects above `V8F_MAX_RUN_H` (default 10 h); G is ESTIMATED at 1.5-2x B's 4.4 h = 6.6-8.8 h. The chain's training wait is 1,200 polls x 30 s = 10 h (a test asserts wait >= guard cap). Set `V8F_MAX_RUN_H` before arming if the PI wants another cap.
- **F6 (banked M2b records no built p_gt).** Its `levers` predate `p_gt_start/end`; for B = M2b the schedule is by construction the trainer's 1.0 -> 0.2. The reader checks G's built schedule against B's; the chain prints a NOTE if the banked train log is missing.
- **F7 (environment).** C: had 2.3 GB free during the build; the test file needs ~300 MB of temp (pytest-of-Admin). The export's copies of the OTHER chains' `.sh` files came out CRLF (autocrlf) although the tip's are LF, which made `test_v8f_pgt.py::test_chain_pgt_static_ordering...` fail in the scratch tree only; normalised there for testing, not part of the deliverables. The longest synthetic-output file name exceeds 260 characters under the scratchpad path (Python could not stat it; `cp` could); the repo path is shorter.

## 6. UNVERIFIED (not measured, do not quote as results)

1. **G's GPU memory and step time at 160 tokens.** ESTIMATE only: S attends over hist x n_tok = 640 tokens per sample (B: 160); one batch of features is 64 x 10 x 160 x 1024 x 2 B = 210 MB (B: 52 MB). Nothing was run on a GPU. The gate (`gategrid`) records the lru; the first seed's wall time is the first measurement.
2. **The chain end to end** (ssh, scp, waits, markers). Only its static properties, `--dry-run`, `bash -n` and the gate heredoc were executed.
3. **The reader on real G outputs** (none exist). Real-data proofs use B's banked evals and SYNTHETIC G arrays.
4. **The cache check proves manifest rows, not bytes.** It requires every effective manifest row to list 8x20 with `shapes.pool8x20 = [n_frames, 160, 1024]`; a truncated `.pt` on box 2's disk would still be found only by the trainer.
5. **A stale `ZZV8FMINI-DONE` in `chain_v8f_pgt.out`** from an earlier PGT run would release the wait early (same hazard as the earlier chains; the chain also checks the busy marker).
6. **Statistical scope.** The deciding interval is the paired episode-cluster bootstrap: it answers "another draw of EPISODES" only (H-ESTIM-SEED-1). The 2F bar uses B's two seeds; there is no replicate arm of G.

## 7. Deliverable manifest (all items live ONLY in this package directory until the lander lands them)

PKGDIR = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid/`. Single-location: every item (nothing is on the tip, nothing on a pod, nothing in a worktree).

| path (under PKGDIR) | bytes | sha256 (first 16) | status |
|---|---:|---|---|
| `stack/tests/test_v8f_grid.py` | 106,786 | `2b63d9ae9dc88d2a` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_grid_read.py` | 35,591 | `28e26d6b7735a555` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/check_grid_cache.py` | 6,135 | `9024504b6ae28fee` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/box2_ops/chain_v8f_grid.sh` | 19,571 | `fcefce988f5aac0e` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_mini_box2_queue.cmd` | 29,590 | `afd96477f383bf28` | EDIT of a tip file (CRLF kept) |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/SYNTHETIC_grid_read_mechanism_only_M2b-pgt.json` | 363,455 | `c7d276c6bf255ebd` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/SYNTHETIC_grid_read_mechanism_only_M2b.json` | 363,596 | `4761c32df7a9e7df` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/SYNTHETIC_grid_read_regression_M2b-pgt.json` | 364,202 | `03c1b24f9fe8dd0f` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/SYNTHETIC_grid_read_regression_M2b.json` | 364,303 | `02523c0b215b5d02` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/SYNTHETIC_grid_read_success_M2b-pgt.json` | 364,438 | `7e0dfc684c3682e4` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/SYNTHETIC_grid_read_success_M2b.json` | 364,560 | `24300bc61fe65e32` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/grid_read_check_inputs_M2b-pgt.json` | 4,617 | `e9251865ec383291` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/grid_read_check_inputs_M2b.json` | 4,801 | `246011021de24bd1` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/mutate_grid.out` | 75,857 | `fee1306ae1bb0b99` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/mutate_grid.py` | 32,717 | `d5aee3aceb599b70` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/probe_eval_mismatch.out` | 1,463 | `e0acd118aa52f0f7` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/probe_eval_mismatch.py` | 2,253 | `e0c5a64dc9edcc4f` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/real_proofs.out` | 5,334 | `109bb619b003a1d8` | NEW |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-grid/real_proofs.py` | 14,922 | `4f6c6cd36069c258` | NEW |
| `diffs/v8f_mini_box2_queue.cmd.diff` | 7,329 | `939e3c96b06e0474` | NEW |
| `RESULT_GRID_BUILD.md` | (this file) | - | NEW |

## 8. Escalation: integration

**The lander must land these files together on the live tip** (the tests, the reader, the checker, the chain and the edited queue are one unit; the chain refuses a tip that lacks any of the stages or reader files, the test skips without the package). Order of operations for the Master Mind: land -> run `v8f_grid_read.py --decide-base` once PREREG_V8F_PGT has its reads -> arm `chain_v8f_grid.sh` only after the PGT chain's DONE marker. Decide F1 and F2 first (they change the PREREG's text, not the instruments).
