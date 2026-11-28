# HANDOFF: TF-N3 combo arm RJ5H5 (navtrain re-roll, EXPLORATORY). Nothing staged, committed or pushed. No navtest token rolled, scored or opened. No ledger entry.

## BARS (committed here BEFORE any RJ5H5 row existed; not to be changed after seeing data)
Arm RJ5H5 = `--pos-jerk-limit 5.0 --policy-interval-s 0.2`, same 469 tokens / 13 logs as the TF-N2 reroll, 1 thread.
1. **R0 identity control (30 tokens, 15 per shard, 1 thread) vs the banked R0 rows: max |dxy| <= 1e-6 m.** FAIL stops the chain before the RJ5H5 roll.
2. **RJ5H5 is "worth carrying to navtest" iff BOTH:**
   - A: CI lower bound of dPDMS (RJ5H5.R minus RJ5.R, paired log-cluster bootstrap, B 10,000, seed 20260927) is > 0; AND
   - B: for each of DAC, NC and C (form R, tracked): the point delta vs RJ5.R is >= 0 AND worse-flips <= better-flips (a CI upper bound of the drop >= 0 is NOT enough; flips are reported with the tokens).
   Otherwise RJ5 alone stays the lever. Either outcome is a result.
3. Also reported (no bar): RJ5H5.R vs R0.R, vs R5HZ.R, RJ5H5.C vs R0.C; sub-scores NC, DAC, EP, TTC, C, PDMS with CIs; longitudinal mechanism (speed bias, speed MAE, accel MAE from the family output, executed_tracked vs the logged human). STRATEGIC family: PAUSED (PI 2026-09-19), never a pass.
Estimator answers "another draw of LOGS" only (CPU rollout, labels, tracker, scorer deterministic; argmax policy). Evidence class: MEASURED (ours), navtrain, EXPLORATORY. Harness cache is not used (navtrain has none): levels are not harness-identical, contrasts are.

(State, run dir, markers and read instructions are appended below by the launch step.)

## State (launched 2026-11-12 16:07 local)
Detached chain RUNNING (CIM Win32_Process Create 0x08000200, stdin NUL; pid 2968 at launch, Python chain is its child). 2 roll/score processes, 1 thread each, CPU only.
- **Run dir:** `F:/tanitad_scratch/tfn3_combo/run/` (banked N2 rows in `run/banked/`: rolls, scores, fam of R0/RJ5/RG8/R5HZ, extracted from `n2/reroll/run/rolls_scores.tar.xz`). Code export `F:/tanitad_scratch/tfn3_combo/w/` (tip 5896f046, `refe/ eval/ code/ splits/ p5s/ repo/`) with the patched `build_teacher_rollouts.py` and `tfn3_roll.py` copied in.
- **Stages / markers** in `run/markers/`: `preflight`, `ctl30_R0` (STOPS the chain on FAIL), `roll_RJ5H5`, `score_RJ5H5`, `fam_RJ5H5_{C,R}`, `analyze`, `ALL.done`. Progress: `run/chain.log`.
- **Result:** `run/result/TFN3_COMBO_RESULT.json` (key `BAR` has the mechanical verdict on the bars above; `paired`, `levels`, `token_flips_RJ5H5.R_vs_RJ5.R_tracked`, `mechanism_rows`, `families_executed_tracked`, `mechanism_longitudinal_RJ5H5.R_minus_RJ5.R`). Control: `run/result/ctl30_R0.json`.
- **Expected wall:** ~1 to 1.5 h (ESTIMATED from the smoke: 3 tokens rolled in 9 s and scored in 13 s; the N2 reroll took 1.2 to 3.0 s/token warm).
- **Resume:** re-run the launch command (`launch_tfn3_combo.ps1` body, pasted into a PowerShell tool call: the script file itself is blocked by the machine's execution policy; no Bypass used). Skips `.done` stages and rows already written. Needed after a Windows-Update restart (rc 1073807364).
- **How to read:** `python -c "import json;d=json.load(open('F:/tanitad_scratch/tfn3_combo/run/result/TFN3_COMBO_RESULT.json'));print(d['BAR']);print(d['paired']['RJ5H5.R_vs_RJ5.R'])"`.

## Verified before launch (raw in `raw/smoke/`)
- Reachability test `test_tfn3_combo_reach.py` 4/4 PASS (`raw/smoke/test_reach.log`): RJ5H5 maps to exactly two overrides; built planner shows policy_interval_s 0.2 and the post-initialize jerk wrapper installed; default build un-wrapped and registered.
- **Mutation arms RED (real roll, 1 token):** `TFN3_MUTATE_DROP=pos_jerk_limit` -> error row `KNOB-NOT-REACHED positive_jerk_limit_when_nonnegative_acc: built 1.0 != requested 5.0` (chain would FAIL the roll stage); `TFN3_MUTATE_DROP=policy_interval_s` -> rc 3 at build (`raw/smoke/mut_policy_interval_s.log`). Unit mutations A/B in the test also RED.
- Every roll row stores `knobs` read back from the BUILT planner; analyze counts them (`knobs_in_rows`).
- 3-token smoke end to end (ctl30 identity max |dxy| 0.0 on 2 tokens, roll, score, fam, analyze): all stages ran. **UNVERIFIED on the tiny sample:** the family stage raised the instrument's axis-convention guard (a 3-token artefact of low-speed tokens, `fam` block `status RAISED`); the same instrument ran on the 469 banked rows. Analyze records a RAISED family block instead of crashing. If `fam_RJ5H5_*` is RAISED on the full set, the longitudinal mechanism comes from `mechanism_rows` (rows' t_v4/t_amax) only; say so.
- Dev note: `tfn3_score.py` is the N2 score script unchanged (imports `n2/tfn2_probe.py` from E:, read only). The tip's `refe/build_teacher_rollouts.py` is the UNPATCHED file; the hold's patched copy (identical to `n2/reroll/`) was copied into `w/refe/` (chain preflight compares sha256).

## Deletions (CLAUDE.md rule 4)
- `F:/tanitad_scratch/tfn3_combo/smoke/` (3-token smoke tree, small; banked in `raw/smoke/` first). Deleted by literal path.
- Left for the MM after it has read the result: `F:/tanitad_scratch/tfn3_combo/run/` (bank `result/`, `fam/`, `chain.log`, `scores/`, `rolls/` first) and `F:/tanitad_scratch/tfn3_combo/w/` (~roughly 30 MB code + 325 MB repo, re-creatable from tip 5896f046).

## Deliverable manifest (hold = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n3_combo/`; nothing in the repo)
| file | content |
|---|---|
| `tfn3_combo_chain.py`, `launch_tfn3_combo.ps1` | chain (ctl30 gate, one arm), detached launcher |
| `tfn3_roll.py`, `tfn3_score.py`, `tfn3_combo_analyze.py` | roll (adds single-override mutation env), score (N2 unchanged), paired read vs banked arms |
| `test_tfn3_combo_reach.py` | RJ5H5 reachability + mutation arms |
| `build_teacher_rollouts.py` | the N2-patched engine-knob hook (identical to n2/reroll copy) |
| `raw/smoke/` | test log, mutation logs, 3-token smoke chain log/result |
| `HANDOFF.md` | this file |
Run outputs (not in the hold yet): `F:/tanitad_scratch/tfn3_combo/run/` only (ONE place until the MM banks them).
