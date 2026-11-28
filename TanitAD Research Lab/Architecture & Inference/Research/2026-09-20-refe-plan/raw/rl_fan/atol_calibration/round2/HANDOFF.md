# HANDOFF, round 2 (written 2026-10-12 21:26; first arm N0_b4 confirmed finished, exit 0, 21:25:58): the dump is RUNNING; the exact next step is the analysis

## State
* Pre-registration `PREREG_ROUND2.md` is staged (written 21:18, before prep and dump). Rule: statistic m = per-set mean |delta| over x/y; M = 2 x F (F = max m over the NULL arms N0_b4/N1_b1/N2_b2 + N_full) rounded UP to a multiple of 0.005; PASS iff 0 null sets above M and every DECISION arm (A2, B1, B2, C1, C2) has >= 12 of 48 sets above M and worst m >= 5 x M. **A1 is an information arm** (ruling in section 6).
* Keys `rows[14::28][:48]`: overlap with round 1 = **0** (token, log) pairs (3 shared scene names). Prep done in `D:/Projects/TanitAD-artifacts/_rlfan_work/atol_cal2` (3.0 GB; round 1's `atol_cal` untouched). Regenerated perturbed checkpoints and calibration variants are byte-identical to round 1's.
* `python run_calibration2.py --phase dump` started **2026-10-12 21:24:34**, detached. PID **20700** (venv launcher) with the interpreter **49396** (the PID in `DUMP_RUNNING.json`). 16 arms x ~80 s (47 s `wait_idle` + 30-60 s dump): expect it to finish about **21:50-21:55** if the shared GPU stays idle (`wait_idle` blocks while another job runs).
* Log: `round2/dump_phase.log` (this directory; `[<arm>] exit N in Ts` per arm, last line `[dump] DONE in Ns`).
* Completion marker: **`D:/Projects/TanitAD-artifacts/_rlfan_work/atol_cal2/DUMP_DONE.json`** (per-arm exit + wall). `DUMP_ABORT.json` instead means `wait_idle` exited non-zero (exit 7, nothing ran on a busy GPU) or an arm has no `cli.json` (exit 8). A `DUMP_RUNNING.json` whose PID is dead, with neither marker, means the process was killed.
* Expected, not errors: exit 5 on A2/B1/B2/C1/S_cam_reversed (sets beyond the 0.5 catastrophe bound), exit 4 on C2/S_snap027 (no reference at that step; judged offline) and S_wrong_scene (frames never resolve). The CLI exit codes decide nothing.

## Exact next step (a fresh agent; CPU only, ~15 s)
```
export PYTHONIOENCODING=utf-8 TMP=D:/Projects/TanitAD-artifacts/_rlfan_work/tmp TEMP=D:/Projects/TanitAD-artifacts/_rlfan_work/tmp
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/rl_fan/atol_calibration/round2/run_calibration2.py" --phase analyse
```
Outputs land in this directory: `verdict_round2.json` (`state` PASS / FAIL / INCOMPLETE, `M`, `F_max_null_m`, per-arm tables, `CHOSEN_MEAN_ATOL`), `arms_summary_round2.json`, `RESULTS_TABLE_round2.md`, `arms/<arm>/{cli.json,cli.log,manifest}`.
Then: append the verdict line to PREREG section 12 (append only; sections 1-11 never change), `git add` exact paths (check with `git ls-files --stage`), and:
* **PASS** -> build `--compare-mean-atol M` in `refe/rl_fan_dump.py`, pin it beside `--compare-atol 0.5`, extend `check_fancmp`, update runbook D-RLS1-5 (PREREG section 9).
* **FAIL** -> report each non-separating decision arm with the printed M' (PREREG section 9); do not re-tune 25 % / 5x / 2x / 0.005.
* **INCOMPLETE** -> `--phase dump --only <arm,arm>` (finished arms are skipped), re-analyse.
* If `DUMP_ABORT.json` exists: read it, then re-run `--phase dump` (it resumes).

## Do not
* Edit `run_calibration2.py` after the dump (its sha256 is in PREREG section 12; a bug fix is a new commit that names itself). Touch `atol_cal` (round 1), `data/refe_proxy/pixels` (read-only), `rl_fan_dump.py`, `rls1_pod.sh` or the tests before the verdict. SSH to any pod.
* Quote the self-test PASS on round 1's data as evidence: it is in-sample.

## Staged vs not
Staged: `PREREG_ROUND2.md`, `run_calibration2.py`, `keys48_round2.tsv`, `prep_manifest_round2.json`, this file. NOT staged (still being written or not yet produced): `dump_phase.log`, `verdict_round2.json`, `arms_summary_round2.json`, `RESULTS_TABLE_round2.md`, `arms/`. The dumped sets (6 MB per arm) and perturbed copies (two 1.3 GB checkpoints, byte-reproducible with `--phase prep`) stay in `atol_cal2`; delete the two checkpoints after the analysis (D: has ~11 GB free).
