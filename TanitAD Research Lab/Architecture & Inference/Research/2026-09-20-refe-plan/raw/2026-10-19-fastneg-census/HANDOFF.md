# HANDOFF (FASTNEG-CENSUS), written 2026-10-19 ~08:25 by the implementing agent. Read STATE.md too.

## State
Registered, coded, tested and gated. The CPU fits are RUNNING detached (started 08:14, ~43 s per fold-fit, ~215 s per task): `fit --groups primary,placebo,cold` (14 tasks, ~50 min, log `run/fit_main.log`) and `fit --groups null` (20 tasks, ~72 min, log `run/fit_null.log`). Expected end ~09:30. A detached bash watcher (`code/finish.sh`, pids 35648 cmd / 38444 / 41504 bash) waits for both `FIT_GROUPS_DONE` lines, then runs `analyze` (reads HEADROW-PROBE's `oof_proposals_f10.npz` only) and `gen_tables.py`.

## Done-markers (poll with a script, not turns)
`run/FINISH_DONE.txt` = success. `run/FINISH_FAILED.txt` = a fit raised (Traceback in a fit log), both logs went stale for 25 min, or analyze failed (see `run/analyze.log`). Fits are resumable: re-issue the same launch (see STATE.md) and finished tasks (`run/fits/<id>.json` + `.npy`) are skipped.

## Pins (do not change; the gate token binds the code)
prereg `b6cece23...`; amendment 1 `f53ae097...` (`PREREG_FASTNEG_CENSUS_AMENDMENT1.md` in this dir); strata.json `67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6` (label-free proof in `run/strata/strata_label_free_proof.json`). DO NOT edit `code/fastneg_lib.py`, `fastneg_census.py`, `fastneg_strata.py`, `test_*.py` (gate token). `fastneg_analyze.py`, `gen_tables.py` are not bound.

## Exact next steps (about 20 minutes of work after FINISH_DONE)
1. Read `run/results/RESULT_tables.md` (all numbers; T1-T11) and `run/results/census.json`. Do NOT read the probe's outcome files; the probe's OOF npz was read by `analyze` only.
2. Check the controls before the outcome: gate PASS 4/4 (`run/gate_token.json`), join proof 0 mismatches (`run/inputs/inputs_report.json`), label-free proof identical, C-REPRO (our L100 vs the probe's L OOF), C-AUC instrument (random brackets 0.5, oracle = 1.0), the probe export's speeds vs ours (`b_ii_probe.cross_checks`), the null (VOID if either row fires on >= 2 of 5).
3. Write `C:/Users/Admin/par18/fastneg/RESULT.md`: headline (outcome + committed next arm, mechanical per the registered rule as amended by A1.6), the per-row per-seed table with CIs (T5/T6), (a) T2 and (b) T3/T4 per stratum, the null T7 with its fire rate, the descriptive D-PLACEBO / D-COLD (no authority; say if they disagree with the registered L arm, which starts at P2's row and may mute the slope), limits (SEEN scenes, selection tier, P2 in-sample, one rig, two refit seeds, estimator = paired log-cluster bootstrap answering "another draw of LOGS" only), evidence class MEASURED, and the proposed GAC / BACKLOG text for `H-REFE-FASTNEG-1` (supported / falsified per its stated falsifier: FEATURE or NO DEFICIT falsifies; COVERAGE supports) and the BACKLOG next-arm row. Include the disclosure: two format probes printed 3-4 `labels6` entries before stratification (all 1.0).
4. `python code/build_package.py` (runs the 3 synthetic test files, stages `out/<repo path>/` with `paths.json` and `MANIFEST.md`). Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -W ignore -I`, `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8`.
5. Hand back <= 300 words: outcome, per-row table with CIs, the null, the next arm, manifest. Stage never push; no git add/commit.

## Hard limits (unchanged)
CPU only, <= 2 Python processes, psutil available >= 3 GB before each fit (built in), never run the full stack suite, do not touch `srttc_s0/`, `srttc_r1/`, `epr1/`, `par18/{probe,l1run,r0eb,hnfp}`, locks, GPU, pods. Kill by explicit PID only. LF files.
