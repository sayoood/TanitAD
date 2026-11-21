# L1 Step A run: STATE (2026-10-19 06:20). Resume here; do NOT wait on the GPU job with model turns.

**Done (steps 0-2, MEASURED):** run dir `D:/Projects/TanitAD-artifacts/l1_stepA_run` (W). PREREG `a76f62d8...` + amendment `45f5c791...` copied unchanged (`STEP0_SHA256.txt`). `sample.json` = 400 tokens / 24 logs, token-list sha `211e2640...` = banked. G0 re-run `g0_rerun.json` = banked JSON modulo paths: M-PDM PASS (0 logs, 0 tokens), M-SCENE overlap 24 logs / 400 tokens (DECLARED exposure, A1.3); control FAILS as required.

**Step 3 is WAITING (launched 06:17:47, WMI detached):** cmd 70556 > launcher 59044/51552 (kill by explicit PID only; max-wait 21600 s, expires ~12:17). Log `C:/Users/Admin/par18/l1run/step3.log`. Patched launcher `code/launch_stepA_ctype.py` (compute-type guard + 6000 MiB backstop + RAM >= 8 GB gate before the lock; 41 tests, RED mutations). Bundle forward needs `L1_TREE=C:/Users/Admin/par18/l1run/tree_hybrid` (the p2heldout tree has NO ctx_cache.py; hybrid = its exact code + tip `ctx_cache.py`); already set in `step3.cmd`.
* First wait reason is the RAM gate (box had 2-3 GB available at 06:19; foreign lock `refcv8-kdraw-step30000ema_unc` + type-C pid 62136 also hold the GPU). If RAM never reaches 8 GB while the GPU is free, relaunch `step3.cmd` with `--min-avail-gb 5` only on a Master Mind decision.
* Markers in `step3.log`: `ZZGPUWAIT` (reason) -> `ZZGPULOCKED` -> per-log lines -> `ZZBUNDLE_OK` or `ZZBUNDLE_FAIL` -> `ZZGPURELEASED`; file `STEP3_RC` (0 = ok). `ZZGPU_NOT_RUN` / rc 75 = max-wait expired, nothing ran.

**G2b HARD GATE:** only `ZZBUNDLE_OK` + `STEP3_RC`=0 + `bundle/bundle.npz` pass it (the forward reproduced the banked fan, logits and pick bit for bit on all 400 tokens). Anything else (`ZZBUNDLE_FAIL`, rc != 0, exit 4/2): STOP, report, read NO A-quantity, run no step 4-7. Plumbing fixes go in copies, never in landed files. `steps4to7.cmd` refuses itself (exit 9) without the gate (tested RED).

**Steps 4-7 (CPU, about 1 h; launch detached like step 3, one at a time):**
`Invoke-CimMethod Win32_Process Create` with `cmd.exe /c "C:\Users\Admin\par18\l1run\steps4to7.cmd < NUL >> C:\Users\Admin\par18\l1run\steps4to7.log 2>&1"`, CreateFlags `0x08000200`. Markers: `STEP4_DONE`, `STEP5_DONE` (G1 label identity runs first; exit 3 = G1 failed), `STEP6_DONE`, `STEP7_DONE ALL_DONE`; failure = `STEP<n>_FAIL`. Output `W/RESULT.json` (outcome by `interval_aware` and `outcome_if_literal`).
* RAM: step 5 is RAM-fragile; launch it with psutil available >= 8 GB, and do not start a stack suite beside it.

**Report:** outcome by BOTH rules, A1-A5 with log-cluster CIs (24 logs, 10,000, seed 20260927), stamped SEEN-SCENE, the four families (STRATEGIC PAUSED). O1 -> Step B (own amendment for C8); O2 -> L2; O3 -> re-run on all 1,705 tokens. Staged under `C:/Users/Admin/par18/l1run/out/` (`paths.json`); add RESULT.json there. Limits: at most 2 Python processes, no pod, never touch another session's lock.

## UPDATE 2026-10-19 06:45 (Master Mind): step 3 attempt 1 FAILED on plumbing; RELAUNCHED
* Attempt 1 took the lock at ~06:30 and died at the first goal: `planner._goal_for` imports `augment_routes` from `<tree>/code/`, and `tree_hybrid` had NO `code/` dir (`ModuleNotFoundError`, rc=1, lock released). No forward output was produced; nothing was read.
* Fix: `tree_hybrid/code/` = the p2heldout tree's `code/` (59 files, byte-identical, sha256 list compared; `augment_routes.py` = tip blob `b4a7b78a...`). All 7 lazy imports of planner + bundle now resolve (checked).
* Relaunched 06:44 (cmd pid 58056, same step3.cmd, max-wait 21600 s). It waits again: the refcv8 chain re-took the lock at 06:37.

## UPDATE 2026-10-19 11:32 (Master Mind): attempt 2 FAILED on plumbing at 11:30; attempt 3 RUNNING
* Attempt 2 took the lock at about 11:25, then died at the first token: `ctx_cache.to_u16` requires a bf16 tensor ("the trainer's autocast dtype"), but `planner.infer` runs fp32 (the planner has NO autocast; the banked P2 fan was made the same way). So `visual_ctx` arrived fp32. Nothing was read.
* Fix (in the run's COPY `code/make_stepA_bundle.py`; the pre-patch copy is `.pre_bf16`): `visual_ctx` is rounded fp32 → bf16 explicitly before `to_u16`, so the bundle format (bf16 bits) and `real_scorer.py` are unchanged.
  * G2b (fan / logits / pick bit for bit) does not touch ctx and is unaffected.
  * The CPU scoring phase is already declared NOT bit-comparable (amendment C9: `G2b_cross_device` is reported; if pick agreement < 0.99, the pre-committed contingency re-scores on the GPU).
  * Anchor and perturbations share the same ctx, so every comparison is like for like.
* Attempt 3: launched 11:32 (cmd pid 43488), took the GPU lock immediately (the refcv7 chain had released it).

## UPDATE 2026-10-19 11:47 (Master Mind): step 3 PASSED G2b; steps 4-7 RUNNING
* Attempt 3: `ZZBUNDLE_OK 400 664 s`, 0 bad. The bundle forward reproduced the banked fan, logits and pick bit for bit on all 400 tokens. STEP3_RC = 0; GPU released 11:44. Rate: about 1.66 s per token incl. scenario setup.
* Bundle: `D:/Projects/TanitAD-artifacts/l1_stepA_run/bundle/` (bundle.npz 8.3 MB, ctx.npy 1.57 GB as bf16 bits, rounded from fp32 per the 11:32 fix).
* Steps 4-7 launched 11:47 (cmd pid 66352, `steps4to7.log`), with 10.7 GB RAM available at launch. The report is due after `STEP7_DONE ALL_DONE`.

## UPDATE 2026-10-19 13:30 (RESULT agent, Sonnet): steps 4-7 DONE; RESULT written and staged; nothing is running
* `steps4to7.log`: `STEP7_DONE 13:08:29 ALL_DONE`; G1 PASS (20 tokens, max abs diff 0.0, control 17/20 changed). `W/RESULT.json`: **O2 NOT ADMITTED by both rules** (A1 0.685 [0.614, 0.761] vs gate 0.80; A4 0.015, A5 0.008 pass). G2b cross-device: pick agreement 1.0000, so the C9 GPU re-score contingency did NOT fire.
* Written: `out/RESULT.md`, `out/MANIFEST.md`, `out/derived_tables_stepA.py` + `out/raw/derived_tables_stepA.json` (post-hoc descriptive A1 table; control reproduces RESULT.json exactly). `out/paths.json` extended (staging for landing; NOT in git yet).
* **Open decision for the Master Mind:** prereg O2 row hands the Step A pairs to L2 as training data and says search is retried after L2's scorer "re-passes this Step A". Re-scoring on the same 400 tokens after training on them is in-sample; say which tokens/logs the re-pass uses before L2 trains.
* D: only: `bundle/bundle.npz`, `bundle/ctx.npy` (1.57 GB), `perts.npz` (not staged, >5 MB); sha256 in `out/MANIFEST.md`.
