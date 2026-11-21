# HANDOFF: PREREG_V8F_SPEEDCAL, built, tested, mutation-checked and RUN (2026-10-25, build agent)

PREREG: `scratchpad/speedavg_res/PREREG_V8F_SPEEDCAL.md`, binding, applied unchanged (the coordinator lands it as `Project Steering/PREREG_V8F_SPEEDCAL.md`). Zero training, dev-box CPU. Nothing staged, committed or pushed; no existing repo file modified (new files only). Box 2 and `qland/` untouched; `box2_ship/` read for `logs/` and the two checkpoints only.

## RESULT (final marker in `C:/Users/Admin/sc_run/logs/bank.out`: `ZZSPEEDCAL-DONE INFORMATION-LIMITED`)

**Outcome row: INFORMATION-LIMITED** (PREREG table applied verbatim by the reader). Tier: T1-class open loop (own held outputs; no recorded future action); CEILING and the `gt` arm are T0. Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes / 2,399 windows, F = 0.1207 m. It answers "another draw of EPISODES" only; the second seed is the replicate (H-ESTIM-SEED-1).

| read | seed 0 | seed 1 |
|---|---|---|
| PRIMARY ADE6, SPEEDCAL - X_s | +0.0374 [-0.0100, +0.0864] not separated | -0.0058 [-0.0156, +0.0036] not separated |
| CEILING (T0): eval-GT affine map - X_s | +0.0296 [-0.0267, +0.0862] | -0.0075 [-0.0174, +0.0025] |
| CAPTURE | -0.052 | +0.009 |
| SECONDARY not separated-worse | NO (speed MAE +0.0226 m/s, accel MAE +0.0192, headway-min +0.041 m, time-gap-min +0.0052 s: all separated worse; TTC, LEAD-VIOL not) | yes |

Controls 1-4 and the guards PASS at both seeds BEFORE any number was read (control 1 max dev 0.0 exactly; control 2 max dev 2.1e-14 / 1.8e-14 against numpy lstsq; control 3 max dev 4.3e-06 / 1.5e-05 m against KINDX's speed oracle; control 4: the permuted-target map is worse than X_s by +5.46 / +5.87 m and not separated-better). Fit set: HBLEND's 141 training clips, 9,582 windows, window list reproduced exactly; fit clips disjoint from eval clips.

Reading, for Rule Zero: both bars fail with the eval-fitted ceiling also above -F, so the table routes to INFORMATION-LIMITED. Descriptive, decides nothing: the affine maps DO cut the speed-cell RMSE to the GT (for example `v_a_ms` at seed 0, eval 1.92 -> 1.33 m/s; the other cells 0-5 %) while ADE6 does not move. The speed error O is sensitive to is therefore not a global affine bias. Committed next step (PREREG): *the speed lever moves to perception (what T sees: temporal context, encoder adaptation); that needs compute beyond box 2, so a PI go.* Blocked on: the PI go (compute beyond box 2). No cheaper lever is licensed by the PREREG.

## Read these

* `<D>/<PK>/raw/2026-10-25-speedcal/run/READ.md` and `READ.json` (full table, per-cell maps and RMSE, SECONDARY rows, B baseline, TACTICAL / STRATEGIC beside), `READ_controls.json`, `OUTCOME.txt`; run log `C:/Users/Admin/sc_run/logs/run.out` (+ `resume.out`, `bank.out`).
* Paired-families: the tool's NATIVE reproduction control ran 119/119 for both arms at both seeds (my evals carry `raw.families`, so the 22-of-79 gap does not arise: INSTRUMENT GAP none) and the coordinator's substitute control PASSES (max |delta| 8e-07 / 7e-07, bar 1e-6).

`<D>` = `scratchpad/speedcal/`, `<PK>` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`.

## Instruments (final shas) and checks

| file | sha256 (first 16) |
|---|---|
| `stack/scripts/v8f_mini_eval_speedcal.py` | 3b184823580e17a2 (used by the run) |
| `<PK>/code/v8f_speedcal_read.py` | cdaca8e637110f95 (used by the run) |
| `<PK>/code/devbox_ops/run_speedcal.sh` | 6a5b348f97b3f227 FINAL; the fits, evals and paired runs used 7e2ff5e69aa16606 |
| `stack/tests/test_v8f_speedcal.py` | 3f18233013aab9a4 (60 tests, CPU, ~25 s) |
| `<PK>/raw/2026-10-25-speedcal/mutate_speedcal.py` | see FILES.txt |

Mutation check on the FINAL tree: `mutate_speedcal.out`, `mutation_results.json`: 61 mutants + 8 end-to-end-only attributed variants, **69 KILLED, 0 crashed / survived / unattributed / bad**, baseline 60 passed before and after, every file restored by sha. Each kill is by an assertion. Earlier passes are kept as history: pass 1 found 4 survivors (the permutation seed at the fit call site, the fit-vs-checkpoint sha in the driver, the limited-fit and limited-eval refusals, which were vacuous because the synthetic grid is refused anyway), 4 crashes (two mutants of mine were mis-built: M01 used the wrong minimum n, M04 tripped the reader's own refusal inside the fixture; one runner test raised FileNotFoundError instead of asserting) and 7 unattributed (my attribution parser misread pytest's indentation). The tests were tightened (message-pinned refusals, permutation-seed pin, fit-sha refusal) and the script was fixed (`mutation_results.pass1/2/3.json` keep the history).

Two runner changes came from the run itself (tests added, mutant M61 for the second):
1. The export has no `raw/` banks, so the reader's lead block was missing and the run STOPPED at its read-input check AFTER all the compute. The runner's `check` now probes the lead block up front (`SPEEDCAL_LEAD_BLOCK`). The lead block used is the repo's banked `2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz` copied unchanged (sha256 c0525943605cfd4a..., identical to the E: repo copy and to `refav1_sweep`'s).
2. The bank copy failed on Windows MAX_PATH (the deliverable tree is deep); it now uses extended-length paths.
The run was resumed at the READ stage with the fits, 12 + 4 evals and the paired runs kept (a re-fit would rewrite the fit json whose sha256 the cal evals record): `C:/Users/Admin/sc_run/run_speedcal_resume.sh`, `run_speedcal_bank.sh`.

## Disclosed gaps

1. **Fit-set GT.** The PREREG says "the v9 train labels". The release file `v9_labels_train.npz` was found only in directories this task may not read, so the GT is the trainer's own `join_train.npz` from HBLEND's fit chain (all four trainings used that join and the same v9 md5); the fit json records its sha256.
2. The existing, unmodified harness `v8f_mini_eval.py` writes a boolean flag key into the `strategic` block of every eval json (value false); it is not in my files. STRATEGIC is in force for v8F: the reader reports the family and the word for the switched-off state never appears in its output (test-pinned).
3. Smoke (300 windows, both seeds) banked under `raw/2026-10-25-speedcal/smoke/` with its label; it is not a result.

## Coordinator to do

Land the new files (staging is yours): the five under `stack/` and `<PK>/code/`, `raw/2026-10-25-speedcal/`, and the PREREG unchanged; record INFORMATION-LIMITED in `GOALS_AND_CLAIMS.md` with the tier stamp; put the perception-lever go to the PI. Temp dirs removed; `C:/Users/Admin/sc_work` (working tree), `sc_run`, `sc_mut_run` and `sc_smoke` can be deleted after landing (C: free ~9 GB).
