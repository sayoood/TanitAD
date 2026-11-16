# BUILD REPORT LON-PLAN (AMEND_HWMPLAN_3, frozen, sha256 7bbdc9fe..., commit 4dc4876c)

Date 2026-11-16. Build only: no window planned, no lambda selected, nothing scored. Tier of every planner row it will produce: T0 (oracle goal), a world-model diagnostic, never driving performance. The code of record is imported unchanged from `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/` (read-only; sha-equal to the repo blobs per section 8.1). Nothing under it was written.

**Integration (escalated here, not in a README).** The MM lands the six files listed in the manifest, then runs the TRAIN selection, writes lambda*_0 and lambda*_1 into the amendment Log, then SCORE. `contrast_l1full_strat.py` needs the strategic-audit reader source at `reader_src/` (a git export of `raw/2026-11-16-v8f-strategic-audit/reader/src/` at 4dc4876c; 7a571fc7 is its ancestor). After landing, point `LONPLAN_READER_SRC` at the repo copy.

## 1. Results (MEASURED, CPU, `TEST_LONPLAN.log`)

`49 passed in 82 s` (serial, `OMP_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=-1`, basetemp on F:). Every green control has at least one deliberate-regression test that must go RED; the mutant tests passed, which means the mutated code made its control read RED.

| control | what the green test proves | mutation (RED arm), result |
|---|---|---|
| **C-L1 reach** | Through the code of record's own run path (`run_hwm.cmd_plan` -> `_init` -> `make_planner` -> `HWMPlanner` -> `plan_batch` -> `plan_topdown2` -> `reg_term`; only model/prior loading is the toy): lambda 1 changes level-1 knots vs lambda 0 at the same seed (max abs diff > 1e-3); REG gradient on the level-1 actions non-zero; lambda 0 equals a run with `reg_term` patched to 0.0 bit for bit; **lambda 0 equals the code-of-record `HWMPlanner` plan bit for bit** | (a) `solver_from` pins `lam` to 0.0: RED, diff exactly 0.0. (b) `plan_batch` builds `reg = None` (hop 5): RED. (c) `reg_term` zeroed: RED, gradient 0 |
| **C-L2 level-1-only identity** | hier + decode, same seed: decoded knots, level-2 and level-3 actions and costs bit-equal between lambda 1 and 0; level-1 knots differ | REG also applied at level 2 (library source exec'd with one added block): RED, level-2/3 actions differ |
| **C-L3 limit** | lambda = `LAMBDA_LIMIT` 1e6 through `HWMPlanner`: level-1 knots equal HOLD_c within 1e-4 prior std, hier and flat | REG reference set to zeros: RED, deviation above tolerance in both modes |
| **C-L4 argv -> built (wrapper)** | refuses: `--lam` or `--seed-mode` not named; declared lam or seed_mode different from the built `SolverCfg`; non-empty folder; a pre-planted `plans_*`; wrong windows sha256; wrong checkpoint sha256; levers off phase-1 defaults; non-oracle arms; `OMP_NUM_THREADS` other than 2; prior seed with lambda != 0. Records in `SOLVER.json`: argv, `str(SolverCfg)`, lambda, checkpoint sha256, window-list sha256, TRAIN-latents sha256, code sha256s, thread counts. **Every `plans_*.npz` is stamped with `lam_planned` and read back** (-1, 0, 1 and 3 tested) | declared/argv mismatch, pre-planted plan file, dropped-`--lam` argv and a `solver_from` that drops lam each REFUSE. The skip trap itself is reproduced without the wrapper: `run_hwm plan` leaves a lambda-0 file untouched when asked for lambda 1 |
| **C-READBACK** | batch 0 (64 windows) of a banked file re-planned from its `SOLVER.json` argv: plan and cost1 bit-equal; refuses a different torch thread count; `readback_all` covers every plan file | file read back at lambda 0: not equal. File with one value changed by 1e-9: not equal |
| **C-REG** | `check_creg`: R - A0 separated-worse and D(R) > D(A0). On the toy run path D(lam -1) > D(0) > D(lam 3). TRAIN sign check C(-1) > C(0) is computed in the selection record | label `none`, D(R) <= D(A0), or `better` each fail the check |
| **C-TRAIN-ISO** | on a synthetic file in the real layout: 4 positions per clip at `unique(round(linspace(0, n_valid-1, 4)))`, windows equal their source frames (0.0), disjointness assertion; the whole selection runs with `hwm_data.eval_windows_from_cache` and `wmact_lib.open_windows` patched to raise; lambda 0 selection plans equal a direct `run_hwm plan` on the same windows bit for bit | a perturbed window gives source diff > 0; an overlapping eval clip raises; a call to an eval loader raises |
| **selection logic (section 3)** | interior optimum, extension up (stop at first C above the running minimum, or D < 0.10; at most 3 extra), extension down from 0.3, tie to the smaller lambda after rounding C to 4 decimals, inadmissible low C skipped, lambda <= 0 never a candidate, no admissible -> down extension then LAMBDA-COLLAPSE | mutants: admissibility dropped, rounding dropped, `EXT_UP` emptied each change lambda* |
| **C-STRAT** | on the phase-1 R35_s0 SCORE windows (sha256 371c753b..., 994 windows, 71 episodes) with banked plan files: all 117 non-STRATEGIC contrast rows, 72 replicate rows, tactical kappa and counts are **bit-equal** (json text equal) to `contrast_l1full.py`'s; the STRATEGIC block is computed ("REPORTED IN FULL"): `s_adh` n 11 windows / 4 episodes, `s_adh_deg` 11 / 4, `s_false` 850 / 69 (MEASURED here; S_ADH is not decision-grade, as ACTRSP section 4.3 states) | the same script with the code-of-record folder as `--eval-src` writes "PAUSED (PI item 25...)", no STRATEGIC rows; `compare()` flags it and fails |

C-STRAT run time: 22 s per script at B = 2000 (about 66 s for the three), far under the 10 min bar. Inputs used: windows sha-equal to phase-1; plan files `C7_R35_s0/win/plans_{flat,hwm}_oracle_s{0,1}.npz`. The flat plans are bit-equal to phase 1 and the hwm plans are not (`c7_result.json`: the C7 run was not bit-equal for hwm). The check compares two scripts on identical inputs, so that does not affect it. The banked phase-1 tar (`hwmplan/hwmplan_phase1_R35.tar`, 103 MB, private HF) was not downloaded.

Also MEASURED, read-only on the real TRAIN latents: `build_selection_windows` gives **564 windows, 141 clips, 5,486 valid NOW positions (min 34 per clip), 11,690 unique frames, repeated-frame max diff 0.0, 0 TRAIN clips among the eval clips, windows equal source 0.0**, both chassis. These equal section 3.1's census.

## 2. Interpretations the MM should confirm (none changes the amendment text)

1. "Running minimum" in the extension rules = the minimum rounded C over the admissible lambda > 0 values already run, before the new value.
2. Extension down is generalised from "chosen is 0.3" to "chosen is the smallest positive value run", which is the same on the base grid.
3. D is measured against HOLD (logged a0 held), as `hold_distance` is called, not HOLD_c.
4. When no lambda > 0 is admissible, the down extension stops at the first admissible value.
5. Selection C is written as the `band_metrics` expression (test asserts equality with `band_metrics` on 30-knot data).

## 3. Not built (outside the brief)

`lonplan_shrink.py` (arm S, C-SHRINK) and the SCORE-side DIR-ACC table. `dir_acc()` exists in `lonplan_select.py` and is used per lambda on TRAIN. The unchanged controls 1-15 and the code-of-record suites (`test_hwm_plan.py`, 50 tests, 523 s; `test_plantest2_controls.py`, 14) were not re-run here. The toy C-L2/C-L3 prove wiring on a synthetic stack; the amendment's R35_s0 16-window smoke with real TRAIN priors is the same code path and needs the real checkpoint, which was not loaded.

## 4. Commands (Git bash on the dev box; set once)

```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH='F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe ; B=F:/TanitAD-scratch/actrsp/lonplan/build ; cd $B
```
Re-run the controls: `$PY -m pytest -q -p no:cacheprovider --basetemp=F:/q/lonplan_tmp/bt test_lonplan.py` (create `F:/q/lonplan_tmp` first, delete after; 82 s).

**TRAIN selection** (one root per chassis, must not exist or be empty; 16-22 runs of 564 windows, 1 worker x 2 threads):
```
$PY lonplan_select.py --chassis s0 --root F:/TanitAD-scratch/actrsp/lonplan/sel_s0
$PY lonplan_select.py --chassis s1 --root F:/TanitAD-scratch/actrsp/lonplan/sel_s1
```
Each writes `<root>/LAMBDA_SELECTION_R35_s{k}.json` plus one `lam_*` folder per value, each with `SOLVER.json`. The runner refuses if the TRAIN latents sha256 differs from the phase-1 record, if a TRAIN clip is an eval clip, or if `OMP_NUM_THREADS` is not 2. Check `sign_check.ok` is true before any SCORE plan at lambda != 0.

**SCORE** (per chassis k; A0 lambda 0 and A1 lambda*_k at inference seeds 0 1; R at lambda -max(lambda*_k, 1), seed 0; one fresh folder each; `--ckpt-sha256` optional but recommended):
```
$PY lonplan_run.py plan --windows-src <windows.npz of chassis k> --windows-sha256 <371c753b... (s0) | 8833e292... (s1)> --declare-lam <LAM> --declare-seed-mode hold -- \
  --tag R35_sK --ckpt C:/Users/Admin/t2v8f_pulls/ladder_R35_sK/ckpt_last.pt --train-latents F:/TanitAD-scratch/hwm_plan/latents_train_R35_sK.pt --train-tag R35_sK \
  --seed-mode hold --lam=<LAM> --arms hwm_oracle --seeds 0 1 --workers 1 --out <NEW FOLDER>
$PY lonplan_run.py readback --out <FOLDER>          # C-READBACK, exit 0 only if every file is bit-equal
$PY contrast_l1full_strat.py --windows <folder holding windows.npz> --a <A1>/plans_hwm_oracle_s0.npz --b <A0>/plans_hwm_oracle_s0.npz --rep-a <A1>/plans_hwm_oracle_s1.npz --rep-b <A0>/plans_hwm_oracle_s1.npz --out <OUT>.json
```
The s0 windows exist locally (`F:/TanitAD-scratch/actsens/read/out/C7_R35_s0/win/windows.npz`, sha-equal to phase 1). **The s1 SCORE windows (sha 8833e292...) are not on disk**: they come from the HF phase-1 tar (a download that needs a go) or from `run_hwm.py windows`. Checkpoint and TRAIN-latents sha256s of both chassis match the phase-1 record (s0 checked here).

**Wall times** (ESTIMATED, INHERITED from AMEND_HWMPLAN_3 section 9, 1 worker x 2 threads; the 16-window smoke should time it first): TRAIN selection 3.6-4.2 h for both chassis, up to +1.6 h with the full extension; SCORE 4.0-4.6 h; readback 0.2-0.35 h; contrasts about 1 min each (MEASURED: 22 s). MEASURED here: control suite 82 s.

## 5. Deliverable manifest

| artifact | where | only copy? |
|---|---|---|
| `test_lonplan.py`, `TEST_LONPLAN.log`, `run_tests.cmd` | `F:/TanitAD-scratch/actrsp/lonplan/build/` | yes (not staged: outside the repo tree) |
| `lonplan_run.py`, `lonplan_select.py`, `lonplan_common.py` | same | yes |
| `contrast_l1full_strat.py`, `lonplan_cstrat.py` | same | yes |
| `reader_src/` (git export of the reader, 12 files) | same | re-creatable from 4dc4876c |
| `BUILD_REPORT_LONPLAN.md` | same | yes |

Suggested repo home: `.../raw/2026-11-08-v8f-bands-build/hwm_plan/lonplan/` (flat, short names; the path-length limit is 200).

## 6. Deletions

`F:/q/lonplan_tmp/` (pytest basetemp and C-STRAT outputs), 11.9 MB, deleted by literal path with PowerShell (bash `rm -rf` was refused by the permission layer). Kept deliberately: `reader_src/` (needed to run C-STRAT; re-creatable).
