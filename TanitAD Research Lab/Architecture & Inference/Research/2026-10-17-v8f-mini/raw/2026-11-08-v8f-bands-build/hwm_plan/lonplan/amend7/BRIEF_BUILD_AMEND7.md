# BRIEF_BUILD_AMEND7: build brief for a Sonnet build agent (written 2026-11-17 by the pre-registration agent)

**Spec:** `F:/TanitAD-scratch/actrsp/lonplan/amend7/AMEND_HWMPLAN_7.md`, sha256 `95c97b39f2692aeb9bff67383e8f8ba58771aeee18739d8693130e80297f7be9` (`AMEND_HWMPLAN_7.sha256`). Read it in full first. It is frozen: never edit it, and every runner refuses on a sha256 mismatch.

**Scope:** build and test the AMEND_7 chain. Do NOT launch it; the MM launches after review. Do not fit RA7 on the real 1,000-clip targets beyond the plumbing smoke of section 4.

## 0 Rules (binding)
* Never commit or push. Never edit `E:/Projects/TanitAD/stack`, any frozen card, or any file under `amend5/` or `amend6/`. Import from them; fork into `amend7/build/` where a change is needed.
* Never print or write tokens, pod or box IPs or ports.
* In `.md` files, follow the MM's list of words to leave out (the MM restates it in your brief).
* Dev box: `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (bare `python3` is a Store stub), `PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"` (`;`-joined Windows paths), `WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree`, as in `amend6/build/run_amend6.sh`.
* One pytest at a time; `--basetemp` and `TMP` under `amend7/work/tmp` (create the parent first). Wait for at least 5.5 GB of commit headroom before a heavy step (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300`).
* Every `subprocess.run(..., text=True)` names `encoding=`. Scripts with apostrophes or backslashes go through the Write tool, not a heredoc.
* Delete your own temporary data at the end (literal absolute paths; list each with its size in the hand-back). Never delete `lat6/`, readouts, results or another agent's files.
* If the job runs past about 4 minutes of waiting, write `amend7/build/HANDOFF.md` (at most 2k tokens) and end. Hand back at most 300 words plus file paths.

## 1 Inputs (read, never modify)
* `amend6/build/fit_ra6.py`: `pairs` (22-49), `ridge_fit` / `ridge_pred` (52-59), `fold_map` (62-63), `choose_lam` (66-72), `merge_latents` (98-109), `chunk_parts` (112-137), `curve_pure` (149-172), `_scale_fn`, `_ra5`, `_load141`, the curve's order and fold map (216-221).
* `amend6/build/amend6_env.py` (the env-patching pattern), `amend6_run.py`, `amend6_read.py` (`outcome6`, `delta_r`), `amend6_controls.py` (structure, C-FIT6 and the ISO checks), `test_amend6.py` (the `synth` fixture), `run_amend6.sh` (`gate`, `pair`, `fail`, `say`, sentinels), `amend6.cmd`, `launch6.ps1`.
* `amend5/build/ra_l1.py`: `Readout` (45-61), `load_readout` (64-76; the SHUF permutation at 75), `roll` (161-164), `ra_pass` (167-183), `plan_arrays` (186-256), `parse`, `check_declared`, `check_built`, `run_arm`. Also `amend5_stage.py`, `amend5_run.py`, `amend5_read.py` (`run_contrast`, `cell`, `sep_better`, `worse`, `score`), `amend5_controls.py`.
* amend1 `amend1_kstep.py` (branch `agent/arch-inf-20260803`, path `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-16-v8f-actrsp-prereg/amend1/amend1_kstep.py`; read with `git -C C:/Users/Admin/tanitad-push show <branch>:<path>`): `mlp_fit` / `mlp_pred` (112-132), the swap partner rule (71-80), the anchor rule (195), the contrast (198-202).
* Data: `F:/TanitAD-scratch/hwm_plan/latents_train_R35_s{k}.pt`, `amend6/lat6/` (chunks + `CHUNKS.json`), `amend6/clips6.json`, `F:/TanitAD-scratch/hwm_plan/ts/clips141.json`, `amend6/readout/FIT_CURVE_s{k}.json`, `amend6/RA6_FIT_s{k}.json`, `amend6/readout_in/` (RA6), `amend5/readout_in/` (RA5), R35 `C:/Users/Admin/t2v8f_pulls/ladder_R35_s{k}/ckpt_last.pt`.

## 2 Files to produce, all in `F:/TanitAD-scratch/actrsp/lonplan/amend7/build/`

| file | what it does | reuse |
|---|---|---|
| `amend7_env.py` | points AMEND_5's modules at the amend7 folders (`AMEND5_ROOT`, readout dir `amend7/readout_in`), sets `ST.SPEC` / `ST.SPEC_SHA` to the AMEND_7 card, `ST.RA_SHA` from `RA7_FIT_s{k}.json` (else `'0'*64`), and patches `ra_l1.load_readout` and `ra_l1.plan_arrays` with the `ra7_l1` versions. Never falls back to RA5 or RA6 | copy of `amend6_env.py` plus the two patches |
| `pairs7.py` | builds `amend7/pairs/pairs7_s{k}.npz` per chassis: z₀, z₁₀, D, y, clip index, frame, fold (curve order), swap partner, the 10 scaled actions, F_true and F_alt (card 3.3) through `ra_l1.roll`, GAIN7 anchor contrasts (card 3.7). Writes `PAIRS7_s{k}.json` (sha256, n, the clip order's sha256, ā). Asserts C-PAIRS7 and C-ISO7 on build | `fit_ra6.merge_latents`, `chunk_parts`, `pairs` (call it, then assert equality with the extended table); `ra_l1.load_ctx` / `roll` for R35 |
| `mlp7.py` | pure functions, no file I/O: the RA7 net (card 2.1), `train_mlp(X, y, cfg, seed)`, the grid and `select` (card 2.3, recording every inner error and the validation clip ids), `crossfit(arm, ...)` returning pooled held-out predictions and per-fold models and the fold record, `LinearD` with float64 L-BFGS (card 3.5), the planted target (card 3.4), the shuffled target, the ZERO and CONST arms, `r2`, `r_wm`, `r_swap` / `s_swap`, the paired clip bootstrap (card 3.6), `gate7` (card section 4) and `fit_outcome` | RY net from `amend1_kstep.py:112-132` with the two changes of card 2.1; R² as `fit_ra6.py:166` |
| `fit7.py` | CLI: `crossfit --chassis k --arm RA|MLPA|MLPB|KCTL|PLANT|SHUFT|ZERO|CONST` (writes `amend7/fit/<arm>_<k>.npz` + `.json`), `final --chassis k --seed-set A|B` (writes `amend7/readout_in/readout7_MLP_R35_<k>[_B].pt` and `amend7/RA7_FIT_<k>.json` with sha256), `gain --chassis k` (GAIN7 and RA5's r_wm on the 859 new clips), `gate` (reads both chassis, writes `amend7/readout/G7_FIT.json` + `.md` with the section 4 label and every number stamped T0 / MEASURED / estimator) | `fit_ra6.main`'s structure |
| `ra7_l1.py` | `MLPReadout` (frozen, eval; `read(Dz, z0, shuf=False)` = RA7([z₀, z₀ + Dz]), z₀ broadcast to Dz's shape; SHUF permutes D by `RandomState(0).permutation(128)`), `load_readout7` (sha256 check before parsing, C-RA7-SHA), and a fork of `ra_l1.plan_arrays` whose only change is that every readout call passes the batch's z₀ (RAL1, NEG, SHUF and ORACLE paths) and that `built` records kind `mlp`. Everything else is imported from `ra_l1` unchanged | `ra_l1.py:45-76` and `:186-256` |
| `amend7_controls.py` | `fit --chassis k`: the 8.1 controls and their regression arms; `run --chassis k`: AMEND_5's controls on the amend7 folders plus C-GRAD7, C-PLANT7, C-RA7-SHA; `merge`: `amend7/controls/CONTROLS_FIT.json` and `CONTROLS.json`, exit 0 only if every control is green and every regression arm RED | `amend6_controls.py` + `amend5_controls.py` |
| `amend7_run.py` | AMEND_5's `run_stage` on the amend7 folders for arms RAL1 (readout A), RAL1B (readout B), SHUF, NEG, ORACLE, OFF | `amend6_run.py`; add the readout-B arm |
| `amend7_read.py` | `train --chassis k` (AMEND_5's TRAIN-stage report; C-SHUF7 sign, C-PLANT7 ii); `score`: AMEND_5's `score()` on the amend7 folders, then Δ_7 against `amend6/score/RAL1_A0_s{k}`, F_T,score from RAL1B, Floor_k = max(F_I, F_T), the separated-better / WORSE rules recomputed with Floor_k and the readout-B point (card 6.3), CEIL-ORACLE7, RA7-ORACLE − RA6-ORACLE, and the section 7 outcome into `amend7/readout/RESULT_AMEND7.json` + `.md`; AMEND_5's full output kept as `RESULT_AMEND7_full.*` | `amend6_read.py` (`outcome6` → `outcome7`, `delta_r` → `delta_7`) |
| `test_amend7.py` | see section 3 | `test_amend6.py`'s `synth` |
| `run_amend7.sh`, `amend7.cmd`, `launch7.ps1` | the chain (section 5) | `run_amend6.sh`, `amend6.cmd`, `launch6.ps1` |
| `BUILD_AMEND7.md`, `MANIFEST_sha256.txt` | build report (reachability trace with file:line, test counts, smoke result without metrics, deletions) and the sha256 of every build file | |

## 3 Tests (`test_amend7.py`, synthetic data only; all must pass)
1. Every 8.1 regression arm reads RED and every green reads green: C-PAIRS7 (flipped chunk byte), C-ISO7 (planted SCORE eid), C-CURVE7 (sorted-id fold map), K-control (misaligned targets), C-FOLD7, C-SEL7, C-CONST7, C-ZERO7, C-WM7 (exchanged rollouts), PLANT P3, SHUF-T (identity shuffle).
2. The K-control on synthetic data: LinearD reproduces `fit_ra6.ridge_fit` predictions within 1e-8 at a given λ.
3. `gate7` reads PASS on a planted nonlinear synthetic truth and FIT-FAIL on a linear one; the outcome table of card section 4 (PASS, STATE-ONLY, FIT-FAIL with TRANS-ONLY, SEEDS-DISAGREE at one chassis, FIT-INVALID) and of section 7 (`outcome7`) row by row.
4. Seed rule: same seed and config give bit-identical predictions; seed sets A and B differ; batch order changes with the seed.
5. Selection never reads the outer test fold (assert on the recorded validation clip ids); ties go to the earlier config.
6. C-GRAD7 on a toy stack: non-zero gradient on knots 1-10, exactly 0.0 on knots 11-12 and yaw; RED when RA7 is detached in the cost and when it reads after 12 actions. C-PLANT7 (i) and its mutation.
7. `ra7_l1` with the readout file's sha256 mismatched refuses; `amend7_env` with no `RA7_FIT_s{k}.json` gives `'0'*64` and every readout use refuses (no RA5 / RA6 fallback).
8. The `plan_arrays` fork with a readout whose MLP is exactly linear in D reproduces `ra_l1.plan_arrays` with the equivalent RA on a toy batch within 1e-6.

## 4 Smoke on real data (plumbing only)
* Build `pairs7_s0.npz` and check C-PAIRS7, C-ISO7 and C-WM7's rollout equality. Rerun RA's cross-fit and check C-CURVE7 against `FIT_CURVE_s0.json` (this reproduces a number AMEND_6 already published).
* One MLP fit with one config and 1 epoch on 50 clips, to check shapes and timing only. Do not record, print or read its metrics. Record the wall time per epoch in `BUILD_AMEND7.md`.
* Delete the smoke's outputs afterwards except `pairs7_s0.npz` (the chain reuses it if its sha256 matches; otherwise it rebuilds).

## 5 The chain (`run_amend7.sh`; sentinels on the last line of `amend7/read.log`)
0. spec sha256 equals `AMEND_HWMPLAN_7.sha256`, else `ZZ AMEND7 FAIL ZZ spec`.
1. tests (`test_amend7.py`).
2. per chassis: `pairs7.py`; `fit7.py crossfit` for RA, KCTL, CONST, then MLPA, MLPB, PLANT, SHUFT, ZERO; `fit7.py final` A and B; `fit7.py gain`. Chassis side by side only at 11 GB commit headroom, else serially (the `pair` function of `run_amend6.sh`).
3. `amend7_controls.py fit` for both chassis and `merge`; any RED ends with `ZZ AMEND7 FIT INVALID ZZ <control>`.
4. `fit7.py gate`. If the label is not R7-FIT-PASS: write `ZZ AMEND7 FIT DONE ZZ <label>` and stop.
5. R7-FIT-PASS only: `amend7_controls.py run` (both chassis), `merge`; TRAIN stage and `amend7_read.py train`; SCORE arms; `amend7_read.py score`; `ZZ AMEND7 READ DONE ZZ <outcome>`.
6. Delete `amend7/pairs/` and `amend7/work/` at the end of a finished chain (re-creatable), and log both deletions with sizes in `read.log`.
* Launch: `launch7.ps1` with WMI `Win32_Process.Create`, CreateFlags `0x08000200`, stdin from NUL; `amend7.cmd` calls `C:\Program Files\Git\usr\bin\bash.exe` (not `Git\bin\bash.exe`). Write it, do not run it.
* Repair notes never go into `read.log` (watchers match its text).

## 6 Done
* All tests green (count recorded), the smoke of section 4 done, `MANIFEST_sha256.txt` written, `BUILD_AMEND7.md` with the reachability trace (argv or arm name, then `amend7_env` patch, then `ra7_l1.plan_arrays`, then `MLPReadout.read` inside `ra_pass`'s cost, then `st._optimize` and `backward`, then the knots written to `plan_raw`; file:line at each hop), the chain written and not launched.
* Hand-back: test count, smoke wall time per epoch, any deviation from the card (there should be none; if the card cannot be built as written, stop and say why instead of changing it), the deliverable manifest with paths, and the deletions.
