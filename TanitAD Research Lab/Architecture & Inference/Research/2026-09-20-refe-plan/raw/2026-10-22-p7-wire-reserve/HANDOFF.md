# HANDOFF -- P7-WIRE-RESERVE (2026-10-22), written because the verification jobs run longer than ~4 min

## State
- CODE + DOCS ARE COMPLETE: `code/refe/` (session_reserve.py, test_session_reserve.py, fixtures/p7_reserve_sessions.json, p6_prep.py, p6_pod.sh, test_p6_chain.py), `RESULT.md`, `raw/` (derived counts, rank-1 command, SPEC_P6 patch, mutation_check.py).
- Per-class runs of the new suite were already green (43 tests); the full-suite / mutation / existing-suite results are being produced by THREE DETACHED jobs (orphaned CPU jobs; they survive their launching shell):
  1. `raw/test_run.out` + `raw/test_run.rc`  (full test_session_reserve.py, expect `ZZSESSRES_PASS 43`, rc=0)
  2. `raw/mutation_check.out`                (controls green, then M1..M13 each RED; last line `MUTATION_CHECK_OK`)
  3. `raw/baseline_summary.txt` / `raw/after_summary.txt` (run_suites.sh: launch_gate, p6_levers, p6_overlap, p6_cpu_probe, p6_planner_command, flock_rw, p6_chain; each ends with a `DONE` line; expect test_p6_chain `ZZP6CHAIN_PASS 432` on both)

## Exact next step (zero model tokens, then ONE read)
`python C:/Users/Admin/par18/p7wire_reserve/finalize_verification.py`
It writes `raw/VERIFICATION.md`, splices it into `RESULT.md` between the VERIFICATION markers and prints `ALL GREEN` or `NOT GREEN: <reasons>` (unfinished jobs count as NOT GREEN, never as green). Re-run it until jobs finish (`bash wait_all.sh` blocks until all four outputs exist; run it in the background).

## If something is not green
- A job died (no `.rc` / no `DONE` / no final `MUTATION_CHECK_*` line and no python process for it): re-run only that job (do NOT re-run `assemble.sh` while any is running; it deletes `work/`):
  - env: `export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 P7WIRE_BASE_DIR="C:/Users/Admin/par18/p7wire_reserve/base_full/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"`
  - 1: `cd work/refe && C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -u test_session_reserve.py > ../../raw/test_run.out 2>&1; echo rc=$? > ../../raw/test_run.rc`
  - 2: `python -u raw/mutation_check.py --work C:/Users/Admin/par18/p7wire_reserve/work --python C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe > raw/mutation_check.out 2>&1` (add `--only M1,M3` to subset)
  - 3: `bash run_suites.sh "$PWD/work" "$PWD/raw" after` and `bash run_suites.sh "$PWD/base_full/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan" "$PWD/raw" baseline`
- A mutation stays GREEN: the printed `NOT CAUGHT` line names it; the mutation table is `raw/mutation_check.py` (M1 = the single exclusion line `session_reserve.py:257`). Add or sharpen the test that should catch it in `code/refe/test_session_reserve.py`, re-copy to `work/refe/`, re-run `--only <id>`.
- An `after` suite differs from `baseline`: the only edits to existing files are `p6_pod.sh` (+66/-0), `p6_prep.py` (+98/-0), `test_p6_chain.py` (+3/-1); diff them against `base_full/.../refe/`.

## Then
Hand back to the Master Mind: the headline of RESULT.md (integration needs) + the verdict line. Do not fetch the 3-C bank (the rank-1 step is `raw/measure_rank1_3c.sh`, PENDING by the brief). Never touch a git tree; never commit or push.
