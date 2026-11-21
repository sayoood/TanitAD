# READY: SH5-DAC-PROX queue (CPU work done; NO GPU job has been launched by the building agent)

Everything below is for the **Master Mind** to launch. Sealed prereg: `PREREG_SH5_DAC_PROX.md` (sha256 in `PREREG_SH5_DAC_PROX.sha256`) **plus `PREREG_SH5_DAC_PROX_AMENDMENT1.md`, `PREREG_SH5_DAC_PROX_AMENDMENT2.md` and `PREREG_SH5_DAC_PROX_AMENDMENT3.md`** (Amendment 2, MM ruling 2026-10-19: **CTX-REACHABLE iff, on BOTH seeds, `Delta_F` lower CI > 0 AND point >= +0.02** (was: lower CI > +0.02, whose MDE +0.050 LOCAL / +0.035 FULL sat above the +0.03 SH5 was sized for); NOT-REACHABLE, PICK-GAIN, UNDERPOWERED and the VOID COUNT rule unchanged; GAIN-NOT-FROM-CTX reads S-shuf with the new rule; power in `raw/amend2_power.json`); **Amendment 3** (MM ruling 2026-10-19 15:45, code `sh5_lib.null_label_fires`): **a C-NULL-LABEL replicate FIRES iff it is CTX-REACHABLE by the Amendment-2 rule on its own seed (lower CI > 0 AND point >= +0.02; `null_label_fires` calls `seed_call`)**, VOID = any of the 3 replicates fires (count unchanged); price and power in `raw/amend3_power.json`: false VOID on a clean machinery 0.5-1.3 % (FULL) / 2.5-7.3 % (LOCAL), and a machinery that manufactures +0.03 now voids FULL with P 0.90-1.00 and LOCAL with P 0.49-0.87 (sealed rule 0.26-0.59 / 0.09-0.26) and, for Amendment 1, (code only: the G-LIVE probe under the trainer's autocast, the ctx load without a host-side stack, one analysis-set definition; found by a post-seal CPU dry run of the trainer on the real inputs with a random ctx). Every stage that can produce a number from real data refuses unless the prereg, the amendments and the 16 pinned code files still hash (`sh5_guard.py`); a changed pinned file needs a written Amendment first.

Conventions
- `PY` = `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore`
- `ROOT` = `C:/Users/Admin/par18/sh5` (the working copy that holds the seal; the repo copy under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-sh5-dac-prox/` has the same layout, so the seal verifies there too)
- `DD` = `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19` (D: is exFAT; large files only there; C: has no room)
- run from `ROOT/code`; GPU stages must NOT set `CUDA_VISIBLE_DEVICES=-1`; CPU stages must (`CUDA_VISIBLE_DEVICES=-1`) and `OMP_NUM_THREADS=2`.
- Detached GPU jobs: `powershell -File ROOT/code/launch.ps1 -Cmd "<command>" -Log <log>` (WMI launch, `< NUL`; any other launch dies or loses stdin on this box). The job takes the dev-box GPU lock ITSELF (`sh5_gpulock`: RAM gate before the lock, type-C compute check, fail closed, never removes a foreign lock, releases only its own bytes, exit 75 if the GPU never frees). Do not touch `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`.
- Markers a watcher can poll: `ZZGPUWAIT` `ZZGPULOCKED` `ZZCTX_OK|ZZCTX_FAIL|ZZCTX_FID_EARLYSTOP` `ZZGPURELEASED` `ZZTHROUGHPUT` `ZZGLIVE ok=` `ZZPREFLIGHT` `ZZSH5FIT_OK|ZZSH5FIT_FAIL` `ZZRAMWAIT`. Exit codes: 0 ok, 75 GPU busy, 4 not locked, 5 ctx fidelity early stop, 3 G-LIVE failed.
- The ctx build must run AFTER L1 step 3 (`C:/Users/Admin/par18/l1run/STEP3_RC` exists, rc 0 -> that job is done) and only while the dev-box GPU lock is free; the lock wrapper enforces it.

## Q0 (DONE, CPU)
- `sh5_train_eval.py prep` -> `DD/data/sh5_inputs.npz` (label-free; sha256 in `raw/ARTIFACTS_D.json`); `sh5_ctx_build.py plan` -> `work/plan/{plan,coverage,needed_frames,blocked_tokens}.json`: 816 / 2,782 tokens (1,444 / 4,941 sets, 38 / 41 logs) have local frames.
- After the seal (13:39:48): `prep --labels` -> `DD/data/sh5_labels.npz` (sha256 `5662515167f37076a53a66de030b1ab27493ee983230760be0de1404c9959925`). The rebuild reproduced `sh5_inputs.npz` byte for byte. Real-data alignment check (`raw/postseal_alignment_check.txt`): 53,221 unsafe of 316,224; P(label unsafe | a key-time clearance < 0) = 1.0; AUC_F of P2's row 0.826694; the LOCAL restriction gives 1,444 sets / 816 tokens / 38 logs.

## Q1 (DECISION, optional, buys the FULL variant): stage frames for the other 1,966 tokens
`work/plan/needed_frames.json` lists the exact jpg names (S0 stager format), about 1.5 GB (ESTIMATED). Stage them from the pod into a new folder `IMG`, then re-run the plan with it:
```
CUDA_VISIBLE_DEVICES=-1 PY sh5_ctx_build.py plan --inputs DD/data/sh5_inputs.npz --out-dir DD/plan_full --images-root IMG
```
The FULL variant needs its own out-dirs below (`ctx_full`, `fits_full`), because the ctx rows follow the plan's token order (no migrate feature). The decision thresholds do not change; the achievable calls do (prereg s8).

## Q2 ctx build (GPU, locked; resumable; ~23 min LOCAL / ~77 min FULL; 3.0 / 10.2 GiB)
```
PY sh5_ctx_build.py run --inputs DD/data/sh5_inputs.npz --plan ROOT/work/plan/plan.json --out-dir DD/ctx_local --max-wait 21600
```
Expect `ZZCTX_OK <n>/<N> tokens; C-FID {... 'pass': True}`. `ZZCTX_FID_EARLYSTOP` (rc 5) = the cached ctx does not reproduce the census logits: STOP and report, do not train. The result is `DD/ctx_local/{ctx.npy,ctx_done.npy,ctx_manifest.json,fid.jsonl}`.
FULL: same with `--plan DD/plan_full/plan.json --out-dir DD/ctx_full`.

## Q3 pre-flights (GPU, minutes; both are cheap and run BEFORE the 20-learner stage)
C-ID (E = 0, no optimizer step; OOF logits must be exactly 0.0), both seeds:
```
PY sh5_train_eval.py fit --data DD/data --ctx DD/ctx_local/ctx.npy --ctx-manifest DD/ctx_local/ctx_manifest.json --seed 0 --stage A1 --epochs 0 --out-dir DD/fits_local/cid
PY sh5_train_eval.py fit --data DD/data --ctx DD/ctx_local/ctx.npy --ctx-manifest DD/ctx_local/ctx_manifest.json --seed 1 --stage A1 --epochs 0 --out-dir DD/fits_local/cid
```
C-REACH pre-flight (planted arm, ONE fold, the real recipe), seed 0:
```
PY sh5_train_eval.py fit --data DD/data --ctx DD/ctx_local/ctx.npy --ctx-manifest DD/ctx_local/ctx_manifest.json --seed 0 --stage PRE --folds 0 --out-dir DD/fits_local/pre
```
Read `ZZPREFLIGHT`: planted-arm DAC AUC (all strata) near 1 and `r2_reg` >= 0.5 on fold 0. If it is far below, stop and report (the branch cannot read even a planted channel; the 20-learner stage would be NO-CALL(C-REACH) anyway).

## Q4 main stage A1 (GPU; 20 learners per seed = S, Snoprox, Sshuf, plant x 5 folds; ESTIMATED 0.17-0.56 h per seed LOCAL)
```
PY sh5_train_eval.py fit --data DD/data --ctx DD/ctx_local/ctx.npy --ctx-manifest DD/ctx_local/ctx_manifest.json --seed 0 --stage A1 --out-dir DD/fits_local/main
PY sh5_train_eval.py fit --data DD/data --ctx DD/ctx_local/ctx.npy --ctx-manifest DD/ctx_local/ctx_manifest.json --seed 1 --stage A1 --out-dir DD/fits_local/main
```
`ZZGLIVE ok=True` precedes the first step (a failed G-LIVE aborts with rc 3 and no step is taken); `ZZTHROUGHPUT` prints the measured rate after the first chunks (re-estimate the rest from it). A killed job loses at most the running cycle; resubmit the same command.

## Q5 CPU: gates and the registered analysis (CPU only)
```
CUDA_VISIBLE_DEVICES=-1 PY run_tests.py --out ROOT/work/tests_summary_post.json          # optional re-run (the evidence behind the `red_table` gate is work/tests_summary_amend3.json = the full run on the Amendment-3 code, staged as raw/tests_summary_amend2.json; the earlier records stay for provenance)
CUDA_VISIBLE_DEVICES=-1 PY sh5_gates.py --cid-dir DD/fits_local/cid --glive DD/fits_local/main/g_live_A1_s0.json DD/fits_local/main/g_live_A1_s1.json --ctx-manifest DD/ctx_local/ctx_manifest.json --tests ROOT/work/tests_summary_amend3.json --out DD/fits_local/gates.json
CUDA_VISIBLE_DEVICES=-1 PY sh5_train_eval.py analyze --data DD/data --fits DD/fits_local/main --gates DD/fits_local/gates.json --keep-tokens DD/ctx_local/ctx_manifest.json --out-dir DD/results_local
```
`analyze` prints `<final outcome> <raw call> <next arm>` and writes `results.json` + `RESULT_tables.md`. A gate that is false removes the call (`NO-CALL(GATE)`); the gates are read from the evidence files, never typed.

## Q6 conditional stage A2 (GPU; `plantweak`, C-POWER at +0.03; 5 learners per seed; 0.04-0.14 h)
Run iff the A1 raw call is NOT-REACHABLE, or INCONCLUSIVE with an upper CI < +0.03 (otherwise the weak plant cannot change the outcome).
```
PY sh5_train_eval.py fit ... --seed 0 --stage A2 --out-dir DD/fits_local/main      (and --seed 1)
```
then re-run `analyze` (same command; it picks the arm up from the fits dir).

## Q7 stage B: nulls (GPU; C-NULL x5 on clearance permutation, C-NULL-LABEL x3; 25 learners seed 0, 15 seed 1; 0.13-0.70 h per seed)
Run iff A1 is CTX-REACHABLE or INCONCLUSIVE (they read the attribution and decide VOID):
```
PY sh5_train_eval.py fit ... --seed 0 --stage B --out-dir DD/fits_local/main      (and --seed 1)
```
then re-run `analyze`. `R["null_complete"]` says whether all 5 + 3 replicates exist; an incomplete null set is flagged, never silently accepted.

## What each outcome triggers (prereg s9)
CTX-REACHABLE + PICK-GAIN -> integrate SH5 into REFe's scorer (launch-gate profile, dev gate, shared navtest look, EP-ADMISSION gate). CTX-REACHABLE, no PICK-GAIN -> stop at the diagnostic and report why. NOT-REACHABLE -> the trunk must learn it (P6): the proximity auxiliary loss becomes the A40 arm draft. INCONCLUSIVE / UNDERPOWERED / NO-CALL / VOID / GAIN-NOT-FROM-CTX -> named by the report; the first lever is FULL (Q1).

## Cleanup
The building agent left no process, no lock and no job; its scratch directory on D: is deleted. Tests that need a temp dir take it from `SH5_TMP` (set it to a folder on D:, C: has little room).
The GPU is the dev box's single RTX 4060: at the last probe (14:09) the lock was held by `refcv8-trainfit` (since 13:34:10); every GPU stage here waits for it (`--max-wait`) and never touches it.
