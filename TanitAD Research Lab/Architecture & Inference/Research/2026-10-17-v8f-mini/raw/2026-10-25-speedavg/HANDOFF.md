# HANDOFF -- PREREG_V8F_SPEEDAVG instruments (built + tested, NOT run on the real arm)

Export (all new files, nothing existing edited): `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/speedavg/`. `FILES.txt` = every new repo path + bytes. Nothing staged, committed or pushed; box 2, box2_ship, qland untouched.
PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`.

## State (MEASURED here)
- `stack/tests/test_v8f_speedavg.py`: **53 passed**, ~30 s CPU. Covers pure pieces, hook on tiny models, end-to-end driver+reader on a synthetic world (controls 1-3 + guards + cross-seed, each with a deliberate-regression arm), the committed outcome table (13 cases, equals the PREREG text cell for cell), refusals, CLI, runner.
- Mutation check `PK/raw/2026-10-25-speedavg/mutate_speedavg.py` (+ `.out`, `mutation_results.json`): **38 mutants, 38 KILLED by an assertion**, 0 survived/crashed; unmutated tree green before and after (53 passed). Includes the brief's four (average one cell only -> control 2; seeds swapped in `self` -> control 1; outcome rows swapped; separation test dropped); six run on the end-to-end tests ALONE, so the controls themselves caught them (red controls named in the json). First pass found 6 survivors + 4 non-assertion reds: test gaps, closed (limited-run refusal, control-3 gating, control-1 exactness, KINDX-condition STOP line, same-file stop, cell-set pin, per-mode restore).
- Real-data SMOKE (stand-in checkpoints, NOT the PREREG's arm; `PK/raw/2026-10-25-speedavg/smoke/SMOKE_LABEL.txt`): first 300 windows, all four modes x two seeds: control 1 = 0.0 exactly; control 2 4.97e-06 m (seed "0"; seed "1" NOT RUN, no banked speed-only oracle for the stand-in); control 3 <= 6.4e-06 m; no-v9-row fallback exercised on 17 windows. No outcome read from a smoke.
- `--dry-run` banked: `PK/raw/2026-10-25-speedavg/dry_run.out` (writes nothing, imports no torch).
- Constraints held: inference vision-only, no steering input; only T's five speed cells (`lon_v_target_ms, v_a_ms, v_end_ms, v_lo_ms, v_hi_ms`) are replaced by the mean of X_0's and X_1's T predictions on the same window; S, the nav/goal path and every other cell are X_s's own (guard `tactical_and_strategic_equal_to_X_s`). S is X_s's own, so it is identical to X_s by construction; STRATEGIC is NOT paused here. Tier stamp: T1-class open loop (own held outputs; no recorded future action).

## Land (coordinator)
The 19 paths in FILES.txt (driver, reader, runner, mutation script, 1 test, 14 banked outputs). `pytest -q stack/tests/test_v8f_speedavg.py` must stay green (it needs Git bash for the runner tests). Mutation check (~13 min single-thread CPU, do not run beside a torch job): `python "<PK>/raw/2026-10-25-speedavg/mutate_speedavg.py" --work C:/Users/Admin/sa_mut --clean` (SHORT --work dir; refuses < 2 GB free; exit 0 only if all killed).

## Run steps for the real arm (only if KINDX's seed-1 read is OUTCOME CARRIER with `speed` among the carriers, i.e. gain_speed(X_s1) >= 2F = 0.2415 m and separated; the runner enforces it and STOPs otherwise)
1. `SPEEDAVG_OUT=<short dir on C:, >= 2 GB free>`, `mkdir -p $SPEEDAVG_OUT/logs`.
2. Set the inputs the defaults get wrong off this scratchpad: `SPEEDAVG_SF=<eval-141 inputs: cache/, inputs/native_tracks.pt, inputs/p10/p10_prior.pt, inputs/v9/{v9_labels_eval139.npz,lead_eval139.npz,refcv6_speed_max_v8_eval.jsonl}>`; `SPEEDAVG_PY=<tanitad venv python>`. Defaults already cover the box2_ship layout for X_s0 (`logs/v8f_mixo/runs/M2b_s0_mixo`, sha c6a62876...) and the KINDX seed-1 chain (`logs/v8f_kindx_s1`: X_s1, `READ_s1.json`, evals, `ckpt_sha256.json` key `M2b-mixo_s1_trained`); a missing checkpoint is pulled READ-ONLY from box 2 by scp and sha-checked; set `SPEEDAVG_X1_SHA` if the json is absent.
3. `--dry-run` first: `"C:/Program Files/Git/bin/bash.exe" "<PK>/code/devbox_ops/run_speedavg.sh" --dry-run`.
4. Launch DETACHED (WMI `Win32_Process.Create`, `cmd /c "... run_speedavg.sh all < NUL > %OUT%/logs/run.out 2>&1"`; CPU only, 8 threads, ~15-20 min). Poll `run.out` by script for `ZZSPEEDAVG-DONE <OUTCOME>` or `ZZSPEEDAVG-STOP <reason>`.
5. Outputs: `$OUT/READ.md`, `READ.json`, `READ_controls.json`, `eval/`. Controls 1-3 run BEFORE any number is read; a failed control or any refusal is rc 2 and reads no outcome. Bank to `PK/raw/<date>-speedavg/`, quote the outcome row and its committed next step verbatim, update `GOALS_AND_CLAIMS.md` in the same turn.

## Watch-outs
- PARTIAL has two literal readings (differ only when one seed meets -F and the other does not): `READ.json` carries `partial_reading_dependent`; report it, do not pick.
- Interval = paired episode-cluster bootstrap (B=2000, 141 episodes, 2,399 windows): blind to training variance (H-ESTIM-SEED-1); the second seed is the replicate, so both seeds must clear PRIMARY.
- Seed-1 control 2 has not run on real data (smoke had no oracle eval for the stand-in); on the real pair `check` requires KINDX's s1 own + only-speed evals, so it will run.
- The synthetic test world has no window without a v9 row; the gt fallback is pinned by the pure test and the smoke (17 windows), not by the end-to-end tests.
