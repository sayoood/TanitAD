# HANDOFF: TF-N2 teacher re-roll (launched 2026-11-12 14:09 local). Nothing staged, committed or pushed. No navtest token rolled, scored or opened.

## State
Detached chain RUNNING (CIM pid 32436 at launch; the Python chain is its child). Stage 1 (`roll_R0`) confirmed started: rows were appearing in `run/rolls/R0_s{0,1}.jsonl`.

- **Run dir:** `F:/tanitad_scratch/tfn2_reroll/run/`. Code export: `F:/tanitad_scratch/tfn2_reroll/w/` (tip `be6567aa`, the N2 probe's export).
- **Markers** in `run/markers/`: `preflight`, `roll_{R0,RJ5,RG8,R5HZ}`, `score_{...}`, `ctl_R0`, `fam_{arm}_{C,R}` (8), `analyze`, then **`ALL.done`**. Each is `<stage>.done` or `<stage>.FAIL` (reason inside). Progress: `run/chain.log`.
- **Result JSON:** `run/result/TFN2_REROLL_RESULT.json`. Early R0 control: `run/result/ctl_R0.json`. Families: `run/fam/fam_{arm}_{C,R}.json`. Per-token rows: `run/scores/`, rolls: `run/rolls/`.
- **Expected wall:** ~1 to 2.5 h (EVIDENCE: smoke rolled 1.2 to 3.0 s/token warm, the TF-2b record is 9.4 s/token, so 4 arms x 469 tokens / 2 processes = 0.6 to 2.5 h; scoring ~1.5 to 3 s/token). CPU only; 2 processes.
- **Resume:** re-run `launch_tfn2_reroll.ps1` (same args). It skips `.done` stages and rows already written. Needed after a Windows-Update restart or kill (rc 1073807364).

## Arms (population = the 469 scored tokens of the N2 probe, 13 logs, `run/tokens.txt`, so every contrast is paired with N2)
| arm | flags | engine path |
|---|---|---|
| R0 control | none | defaults |
| RJ5 | `--pos-jerk-limit 5.0` | `positive_jerk_limit_when_nonnegative_acc` 1.0 -> 5.0 (= `max_jerk_long` 5.0) |
| RG8 | `--route-goal-min-speed 8.0 --route-goal-horizon 12.0` | feature-builder goal distance (horizon is already 12.0 in the protocol) |
| R5HZ | `--policy-interval-s 0.2` | policy forwarded every 0.2 s instead of every 0.1 s step |

Each arm: rollout -> forms **T** (t20 as is), **C** (Tc20s), **R** (P5-R, lambda 3.0) -> W-TRANSP scoring -> families (C and R forms).

## Findings while building (MEASURED, smoke on 2 tokens; raw in `raw/smoke/`)
1. **R5HZ is `policy_interval_s`, not `target_sample_rate_hz`.** The release yaml's `target_sample_rate_hz: 5.0` is already in force; it only sets the policy's input-history spacing (`agent_loader.py:139-148`). The planner's `policy_interval_s` default is **0.0 = the policy runs every 0.1 s sim step (10 Hz)**; the comment in `build_teacher_rollouts._reset_planner_state` ("forwards its policy at 5 Hz") does not match the code. R5HZ sets 0.2, which also makes the controller hold the cached acceleration target (`trajectory.py:117-123,161`).
2. **The jerk knob is lost if set at build time.** `planner.initialize` overwrites `positive_jerk_limit_when_nonnegative_acc` from the release yaml on its first call (`planner.py:136-147`). The override is therefore re-applied by a wrapper after every `initialize`.
3. **Identity.** R0 reproduces the banked rank-0 t20 with max |dxy| **0.0 m** and the banked probe scores with max abs diff **0.0** (2 tokens). Each knob moved the plan: max |dxy| vs R0 0.07/3.99 (RJ5), 0.45/3.33 (RG8), 0.51/2.11 (R5HZ) m.

## Reachability (built, tested; `test_tfn2_reroll_reach.py`, 4 tests PASS under `run_p4.py`, CPU)
- Defaults: `TEACHER_OVERRIDES == {}`, the built planner equals the registered knobs and `initialize` is not wrapped.
- Every roll row stores `knobs` read back from the BUILT planner after the roll; a mismatch is written as an error row `KNOB-NOT-REACHED`, and the chain FAILS a roll stage unless all 469 tokens have an ok row.
- **Mutations, each RED:** `TFN2_MUTATE_DROP_OVERRIDE=1` with `--route-goal-min-speed 8.0` -> `ZZTFN2_KNOB_FAIL ... built 5.0 != requested 8.0` (rc 3); with `--policy-interval-s 0.2` -> built 0.0 != 0.2 (rc 3); with `--pos-jerk-limit 5.0` -> error rows `built 1.0 != requested 5.0`. Unit mutations (wrapper not installed, `_PLANNER_CFG_KEYS = ()`) also go red.

## Controls and bars (committed here, before any full-run row)
- **R0 identity:** roll vs banked t20 max |dxy| <= 1e-6 m over all 469 tokens (CPU rollouts are deterministic; fallback reading is the PREREG_P4 C3 bar), and R0 sub-scores vs `n2/raw/probe_rows.jsonl.xz` max abs diff <= 1e-9. A FAIL does not stop the chain (paired deltas against R0 stay internally valid) but voids "R0 = the banked teacher".
- **Estimator:** paired log-cluster bootstrap over 13 logs, B 10,000, seed 20260927 (`tfn2_analyze.boot`). Answers "another draw of LOGS" only; the policy is argmax and CPU rollouts are deterministic, so there is no inference variance to speak of. A navtrain-held-out result is EXPLORATORY: the knob values were fixed before the run, but N2's navtest read is the confirmation.
- **Scorer route:** the nuPlan-DB rebuild of the metric-cache parts, as in the N2 probe. The harness-cache repair `../n1/a1/transp_score_a1fix.py --metric-cache-root` does **NOT** apply (navtrain has no harness cache), so levels are not harness-identical; contrasts are.
- **Families:** `families_run.py` unchanged, executed_tracked vs the logged human, STRATEGIC PAUSED (PI 2026-09-19). Per-family values are descriptive; no paired family CI exists (the N1 work item).

## How to read the result
`python -c "import json;d=json.load(open('F:/tanitad_scratch/tfn2_reroll/run/result/TFN2_REROLL_RESULT.json'));print(d['R0_identity']['PASS'], d['levels'], d['paired_vs_R0_same_form'], d['best_arm_form_by_pdms_EXPLORATORY'])"`. Sections: `R0_identity`, `knobs_in_rows`, `levels` (arm.form: PDMS with CI, EP, DAC...), `paired_vs_R0_same_form`, `paired_vs_R0_C` (the N2 baseline), `mechanism` (teacher 4 s path over PDM-Closed's, speed at 4 s, max accel), `token_flips_vs_R0_C_tracked`, `families_executed_tracked`.
Judge against N2's table (`../TFN2_EVIDENCE.md` §2): C 93.88, R +0.55, the warp +0.62 to +1.18. A knob arm is worth carrying only if its paired PDMS delta CI excludes 0 **and** comfort (`c`), DAC and NC do not drop (see `token_flips_vs_R0_C_tracked`). Rule Zero: if no arm clears, the next lever is the combination of the winners plus CERT/R, run on the same tokens.

## Untested / caveats
- Only 2 tokens were smoke-run end to end; the 469-token rows, the family stage on full rows, and the analyze output on full data are first exercised by this run. `families_run` and `analyze` ran on the 2-token smoke only.
- The R0 identity at scale (469 tokens, 13 logs) is unproven until `ctl_R0.json`.
- R5HZ changes the controller's cached-acceleration path as well as the planner's rate; it is a protocol knob of the engine, but it is not the paper's stated protocol (not checked).
- `build_teacher_rollouts.py` here is a patched copy of the tip's file (diff `build_teacher_rollouts.tfn2.diff`, +33 lines, empty-by-default); the MM lands it. The roll/test scripts are copied into `w/eval/` for the run (the chain's preflight compares their sha256 with this folder).
- No `FreeVirtualMemory` probe was run (2 light processes; smoke showed no pressure).

## Deletions (CLAUDE.md rule 4, by literal path)
- `F:/tanitad_scratch/tfn2_reroll/w/` deleted twice and rebuilt (two export attempts with the wrong tar strip depth; ~29 MB then ~0 B; re-creatable from `be6567aa`).
- `F:/tanitad_scratch/tfn2_reroll/run/smoke/` 138,658 B, banked first in `raw/smoke/`.
- **Left for the MM after it has read the result:** `F:/tanitad_scratch/tfn2_reroll/run/` (rolls, scores, fam; bank the result JSON, `ctl_R0.json`, `fam/` and `chain.log` first) and `F:/tanitad_scratch/tfn2_reroll/w/` (~75 MB, re-creatable).

## Deliverable manifest (all under `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2/reroll/`; nothing in the repo)
| file | content |
|---|---|
| `build_teacher_rollouts.py`, `build_teacher_rollouts.tfn2.diff` | patched engine-knob hook (target: the tip's `refe/build_teacher_rollouts.py`) |
| `tfn2_reroll_roll.py` | roll stage with explicit knob flags (goes in `eval/`) |
| `tfn2_reroll_score.py`, `tfn2_reroll_analyze.py`, `tfn2_reroll_chain.py`, `launch_tfn2_reroll.ps1` | score, read, chain, detached launcher |
| `test_tfn2_reroll_reach.py` | reachability, default identity, mutation arms |
| `raw/smoke/` | the 2-token smoke rows, mutation logs, smoke analyze |
| `HANDOFF.md` | this file |
