# HANDOFF: TF-N3 jerk-limit DOSE check, arms RJ3 (3.0) and RJ8 (8.0) vs banked RJ5 (5.0). navtrain re-roll, EXPLORATORY. Nothing staged, committed or pushed. No navtest token rolled, scored or opened. No ledger entry.

## BARS (written here and into `tfn3d_analyze.py` BEFORE any RJ3/RJ8 row existed; mechanical, never changed after data)
Arms: RJ3 = `--pos-jerk-limit 3.0`, RJ8 = `--pos-jerk-limit 8.0` (nothing else moved; policy interval stays registered 0.0), same 469 tokens / 13 logs as the TF-N2 reroll (`tokens.txt`), forms C and R, 1 thread, 2 processes.
For each x in {3, 8}, RJx "beats RJ5" iff ALL of:
- **A:** CI lower bound of dPDMS (RJx.R - RJ5.R) > 0 (paired log-cluster bootstrap, B 10,000, seed 20260927);
- **B:** for DAC, NC, C (form R, tracked): point delta vs RJ5.R >= 0 AND worse-flips <= better-flips;
- **C (longitudinal guard):** speed-bias diff vs RJ5.R <= +0.25 m/s AND speed-MAE diff <= +0.15 m/s (point values from the families, `executed_tracked`, form R; paired CIs from `fam_ci.py` reported too).
- (Added fail-safe, mechanical: every RJx row's built-planner jerk read-back must equal the request; a missing/raised family block leaves bar C unverified and the verdict fails safe to "5.0 stays".)
Otherwise **5.0 stays**. Also reported, descriptive only: EP, DAC, TTC, speed bias at jerk 1 (R0) / 3 / 5 / 8.
Controls: R0 identity 30 tokens at 1 thread vs banked R0 rows, max |dxy| <= 1e-6 m (FAIL stops before any RJx roll); knob reachability test + mutation arms (override dropped must go RED).
Estimator answers "another draw of LOGS" only (CPU rollout, labels, tracker, scorer deterministic, argmax policy). Evidence class MEASURED (ours), navtrain, EXPLORATORY. No harness cache (navtrain has none): levels not harness-identical, contrasts are. STRATEGIC family: PAUSED (PI 2026-09-19), never a pass.

(State, markers and read instructions are appended below by the launch step.)

## State (launched 2026-11-12 16:30 local)
Detached chain RUNNING (CIM Win32_Process Create 0x08000200, stdin NUL; launcher pid 20712, Python chain is its child). 2 roll/score processes, 1 thread, CPU only.
- **Run dir:** `F:/tanitad_scratch/tfn3_combo/run_dose/` (reads banked R0/RJ5 rows READ-ONLY from `F:/tanitad_scratch/tfn3_combo/run/banked/`; code export `.../tfn3_combo/w/` unchanged).
- **Markers** in `run_dose/markers/`: `preflight`, `ctl30_R0` (STOPS on FAIL), `roll_RJ3`, `score_RJ3`, `roll_RJ8`, `score_RJ8`, `fam_{RJ3,RJ8,RJ5}_{C,R}` (`--per-window`; RJ5 regenerated from the banked RJ5 scores, JSON-level values verified equal to the banked fam), `analyze`, final **`ALL.done`**. Log: `run_dose/chain.log`.
- **Result:** `run_dose/result/TFN3_DOSE_RESULT.json` (keys `BAR` per arm incl. `BEATS_RJ5`, `DECISION`, `paired`, `token_flips_vs_RJ5.R_tracked`, `fam_paired_vs_RJ5.R`, `mechanism_longitudinal_minus_RJ5.R`, `monotone_picture_R_descriptive_only`, `knob_reach`); paired family CIs `result/fam_ci_RJxR_vs_RJ5R.json`; control `result/ctl30_R0.json`.
- **Expected wall:** ~30 min (ESTIMATED: combo arm took 249 s roll + 395 s score; two arms + ctl30 + fams).
- **Resume:** re-run the launch body (`launch_tfn3_dose.ps1`, pasted into a PowerShell call; the .ps1 file is blocked by execution policy; no Bypass). Skips done stages/rows. Needed after a Windows-Update restart (rc 1073807364).
- **Read:** `python -c "import json;d=json.load(open('F:/tanitad_scratch/tfn3_combo/run_dose/result/TFN3_DOSE_RESULT.json'));print(d['DECISION']);print(json.dumps(d['BAR'],indent=1));print(d['monotone_picture_R_descriptive_only'])"`
- Bar C needs both family blocks; if `fam_RJx_R` is RAISED on the full set (axis guard), the verdict fails safe ("5.0 stays", note says UNVERIFIED), and the mechanism must be read from `levels`/rolls instead.

## Verified before launch (raw in `raw/smoke/`)
- `test_tfn3d_reach.py` 4/4 PASS (`raw/smoke/test_reach.log`): RJ3/RJ8 map to exactly one override (jerk), policy interval untouched; wrapper installs on the real built planner only when requested; unit mutation (override dropped) RED for both values, value mix-up RED.
- Real-roll mutation arm: `TFN3_MUTATE_DROP=pos_jerk_limit` on RJ8, 1 token -> error row `KNOB-NOT-REACHED ... built 1.0 != requested 8.0` (`raw/smoke/mut_jerk.jsonl`).
- 2-token smoke end to end (`raw/smoke/smoke_chain.log`): ctl30 R0 max |dxy| 0.0, both arms rolled/scored, knob read-back 3.0 / 8.0 in all rows, all stages ran. **UNVERIFIED on the 2-token sample:** RJx family blocks RAISED (axis-convention guard, low-speed 2-token artefact, same as the combo smoke) and the paired family CI refused on token-set mismatch (RJ5 regen is full 469); the analyzer failed safe as designed. The paired-CI path itself was exercised on RJ5 vs RJ5 (469 tokens; speed bias/MAE/accel/progress keys found). One analyzer fix after the smoke, before any real row: exact family-key match (`progress` had matched `lead_progress_ratio`).
- Dev note: `tfn3_roll.py`, `tfn3_score.py`, `build_teacher_rollouts.py` are the combo files unchanged (sha256 equal to the `w/` copies, checked by the chain preflight). `raw/instr/` = `families_run.py` (--per-window), `fam_ci.py`, `FAM_CI.md` exported from mirror tip 2382559e with `-c core.autocrlf=false`.

## Deletions (CLAUDE.md rule 4)
- `F:/tanitad_scratch/tfn3_combo/smoke_dose` (4.3 MB, 2-token smoke; banked in `raw/smoke/` first). Deleted by literal path, rc 0.
- Left for the MM after it has read the result: `F:/tanitad_scratch/tfn3_combo/run_dose/` (bank `result/`, `fam/`, `chain.log` first), plus the combo's `run/` and `w/`.

## Deliverable manifest (hold = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n3_dose/`; nothing in the repo, nothing staged)
| file | content |
|---|---|
| `tfn3d_chain.py`, `launch_tfn3_dose.ps1` | chain (ctl30 gate, RJ3 + RJ8, per-window families incl. RJ5 regen), detached launcher |
| `tfn3_roll.py`, `tfn3_score.py`, `build_teacher_rollouts.py` | combo files, unchanged |
| `tfn3d_analyze.py` | analyzer with the bars in code |
| `test_tfn3d_reach.py` | reach test + mutation arms |
| `raw/instr/`, `raw/smoke/` | instruments copy; test log, mutation row, smoke chain log/result |
| `HANDOFF.md` | this file |
Run outputs: `F:/tanitad_scratch/tfn3_combo/run_dose/` only (ONE place until the MM banks them).
