# HANDOFF_ACTSENS_READ (agent, 2026-11-15 ~02:50): read package BUILT; C7 and C8 RUNNING detached; nothing staged, committed or pushed; box 2 never written to

Tier T0 (oracle goal: a world-model diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25). Everything lives ONLY on `F:/TanitAD-scratch/actsens/read/` (one disk) except this file: bank before anything is deleted.

## One command per banked arm
```
bash F:/TanitAD-scratch/actsens/read/read_actsens.sh <tag> <chassis-seed>
   IDM_s0 0 | NULL_s2 0 | DIV_s0 0 | IDM_s1 1 | DIV_s1 1 | NULL_s3 1      (arm run dir C:/Users/Admin/t2v8f_pulls/actsens_<tag>, needs ckpt_last.pt, argv.txt, BANK.done or RUN.done)
python F:/TanitAD-scratch/actsens/read/verdict_actsens.py --out <verdict.json> --md <verdict.md>      (after all six arms; section 6 mechanically)
```
Long job (about 2 to 3 h per arm, ESTIMATED from C7's first steps; HWM plans 8 units of about 15-25 min, the dev box is busy with other sessions): launch detached with the WMI recipe, template `c7_run.cmd` / `c7_run.sh` (cmd calls Git `usr\bin\bash.exe`, `< NUL`, CreateFlags 0x08000200; script puts `/usr/bin:/mingw64/bin` first). The driver: fails closed (writes `out/<tag>/ARM_FAILED_<tag>.txt`), refuses a finished arm (`ARM_DONE_<tag>.txt` or `out/arm_<tag>.json` exists), resumes after a kill (banked windows, plans, latents are skipped), waits up to 2 h for 6 GB free RAM, needs 1.5 GB on F:. Output `out/arm_<tag>.json` + `out/<tag>/` intermediates; done marker `out/<tag>/ARM_DONE_<tag>.txt`.
Steps: check_tree (ACTSENS tree, `wm_gamma_levels` field, contrast_l1full `b7661635`, decode_check `027da9ee`, 16 R35 reference files equal to HF_BANK sha256) -> IA read (card 4.1: IA-3, BAND-MET; MEASURED 892 s on a busy box, not the 341 s of the card) -> windows + SCORE subset + C6 (994 windows, 71 episodes, equal to R35's) -> TRAIN latents of the arm (`collect_latents_arm.py` = the R35 collector with only paths/tag replaced; 141 clips by READ-ONLY sftp from box 2, about 7 min; run BEFORE or AFTER the chain, it adds sftp reads only) -> 4 plan runs (hwm_oracle, flat_oracle x inference seed 0, 1; phase-1 levers: `--seed-mode prior --lam 0`, T3 2 = default K2, plan-ctx 1, stitch decode, vocab off, constraint weight 0 = run_hwm defaults) -> contrast_l1full (arm vs R35_k plans, arm vs hold, replicates; all four families per band; STRATEGIC PAUSED) -> decode_check (6 batches) -> `assemble_arm.py`.
Chassis note: NULL_s2 reads against R35_s0 and NULL_s3 against R35_s1 (card section 4.4). R35 phase-1 plans and windows come from the HF tar (rev aa2d8014, sha256 abff3a38, verified) extracted under `read/hf_phase1/R35/plan_runs/s{0,1}/`.

## Controls
| control | state |
|---|---|
| C6 windows equal | built into the driver (stops the arm if unequal); R35_s0 through the driver: PASS, all equal, z diff 0.0 (MEASURED, `out/C7_R35_s0/c6.json`) |
| C7 REPRO + SELF | RUNNING detached since 02:28 (`out/C7_R35_s0/`, launch cmd `c7_run.cmd`, WMI PID 4164). Marker `out/C7_R35_s0/ARM_DONE_C7_R35_s0.txt` (or `ARM_FAILED_*`), result `out/C7_R35_s0/c7_result.json` (`C7_PASS`: four plan files bit-equal to the banked phase-1 plans, `contrast --check` reproduces the banked rows, arm minus R35 = 0.0 [0, 0] on every cell, IA read equal to the banked ladder-read arm, decode step3 equal to the banked one). Done so far: the IA read of R35_s0 through this driver equals the banked `ladder_read/results/arm_R35_s0.json` on L1 2.0 s gain_lon and on `passes` (MEASURED). The latents of this run are the banked `latents_train_R35_s0.pt` (sha 76a2d4f5); the collector itself is NOT re-run for R35 (UNVERIFIED: that it reproduces those latents; the pull is 7 min, run `COL_*` env as in the driver if wanted). |
| C8 ZERO-ACT | PASS (MEASURED, `out/C8/c8_result.json`): R35_s0 with `stack.act_embed` zeroed (1,152 parameters): every L1 IA-3 gain (lon and lat, all horizons) reads exactly 0.0 (L1 2.0 s: 0.0 / 0.0); the intact control reads gain_lon 0.3842, gain_lat -0.0385 at L1 2.0 s; Part A (build) both true. IA read took about 22 min on the busy box. |
| verdict | `verdict_actsens.py` tested on synthetic inputs, `test_verdict_actsens.py`: 9 passed, each of CEILING, NARROWS, MECH-ONLY, NULL, HARM, SEEDS-DISAGREE fires once; deliberate-regression cases: training floor larger than the effect must not read NARROWS, sign flip must change the label, thin gain gap must not be MECH-MOVED, missing NULL or chassis gives INCOMPLETE, a lever that did not run cannot be NULL |

## Rule readings the verdict script commits (check them against the card before the first real read)
* Floor_k(q) = max(|NULL_k - R35_k|, max(|seed1 - seed0| of the lever arm k, of R35_k)); BETTER needs both chassis, CI wholly below 0 and |delta| > Floor_k; WORSE any chassis.
* MECH-MOVED: the same gain (lon or lat) at L1 2.0 s above R35's at both chassis, arm lo > R35 hi, gap > |NULL - R35|. R35 IA-3 references = the banked `ladder_read/results/arm_R35_s{0,1}.json`.
* BAND-MET lost = R35 met a level (beats, healthy, not worse) and the arm does not, at either chassis. HARM is an either-chassis rule, so it never reads SEEDS-DISAGREE. CEILING = the same quantity beats HOLD at both chassis and the other quantity is not WORSE.
* `--lever-ran IDM=0` withholds NULL until C2 (gamma log) and C4 (DIVRS) hold; the script cannot see C2/C4/C5.
* Card items NOT in the script (reported beside, no bar, from the arm JSON/logs): probe_actsens Part A per arm, WMCEIL, WMACT ACTION-USE (UNVERIFIED on ladder checkpoints), tactical kappa is in `hwm.tactical_kappa`.

## UNVERIFIED
C7 and C8 IA-3 results until their markers exist; the full-driver wall time per arm; collector reproduction of R35's latents; `decode_check` with the arm's own latents (the same code with env paths: `make_decode_check_arm.py` asserts the original sha and replaces 7 path lines); IDM/DIV arms have never been through the driver (only R35 has: no ACTSENS checkpoint exists yet).

## Deletions (own temp, literal paths, none the only copy)
`read/tree/taniteval/results` (about 288 MB) and `read/tree/stack/experiments` (about 89 MB): git-export parts, re-creatable; `read/hf_phase1/dl/hwmplan/hwmplan_phase1_R35.tar` (103,516,160 B): re-creatable from HF rev aa2d8014, extracted copy kept; `read/tmp/ft`, `read/tmp/ft2` (278 B, 3 B): fail-closed test outputs. Exact sizes of the two tree dirs were not logged before deletion (du before: 450 MB tree).
Left on F: `read/tree` (about 70 MB code export), `read/hf_phase1/R35` (about 100 MB), `read/out/`. F: free was about 4.4 GB; keep above 3 GB, delete `read/hf_phase1/R35/plan_runs/s*/windows_all.npz` (65 MB) first if needed (not read by any step).

## Files (all `F:/TanitAD-scratch/actsens/read/`, one disk)
read_actsens.sh, env.sh, check_tree.py, ia_read.py, c6_windows.py, collect_latents_arm.py (+ make_collector_arm.py), decode_check_arm.py (+ make_decode_check_arm.py), assemble_arm.py, c7_checks.py, c7_run.sh/.cmd, c8_run.sh/.cmd, c8_check.py, verdict_actsens.py, test_verdict_actsens.py, hf_fetch.py, publish_context_labels.py (copy from bb3825f8, token read in place, never printed), hf_bank_phase1.json, ts/clips141.json.
Integration (Master Mind): land these into the repo with the lander (one folder `actsens_read/`); nothing here edits the code of record.
