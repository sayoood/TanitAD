# HANDOFF RL-1 (written 2026-10-18 ~22:30 Berlin by the RL-1 agent; fresh agent continues here)

## State
* Steps 1-3 DONE and written up: `C:/Users/Admin/par18/rl/out/RESULT.md` (interim), pre-registrations + hashes in `out/raw/` (`PREREG_RL1_A_MECHANISM.md`, `PREREG_RL1_B_COMPOSITES.md`, each with `.sha256`; verify with `sha256sum -c`), results `out/raw/mechanism_r0.json`, `mechanism_r0b.json`, `composites_rlval.json`.
* Step 4 (the ONE navtest open) is RUNNING detached: composite **V1** = RL{EP,C} from R0B + BASE{NC,DAC,TTC,DDC,lane}, W3 harness on `nav/seam_V1.npz` (built, gates in `nav/build_V1.json` all true). Launched 22:24 Berlin, cmd PID 71612 (python waiter 66696 -> harness chain). Own output dir `nav/score/refe_rl1_V1/`; the banked score store was NOT written. Expected 25-60 min.
* DONE marker: `C:/Users/Admin/par18/rl/nav/DONE_V1` (first line `OK` or `FAIL`; status JSON `nav/score_logs/refe_rl1_V1.status.json`, logs `nav/score_logs/refe_rl1_V1.attempt*.log`). The runner retries up to 3 times.

## Rules that still bind
CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=8`), at most 2 concurrent Python processes (the harness chain + waiter count as 2 until DONE_V1 exists), never touch `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`, no pod, no ssh, kill only by explicit PID (no `/T`, no pkill), no git commit/push/add, no register edits. Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`. navtest is opened ONCE: do NOT re-score or score another variant.

## Exact next steps
1. Wait with a script, not turns: `until [ -f C:/Users/Admin/par18/rl/nav/DONE_V1 ]; do sleep 60; done` as a background command or Monitor. If it reads FAIL, read the attempt logs and STOP with the blocker named (do not relaunch another scoring run without the caller).
2. Dry known-value test of `nt_read.py` (it was written, not yet run): make `nav_dry/` with DONE_T=`OK`, `score_logs/refe_rl1_T.status.json` = `{"status":"PASS"}`, `seam_T.npz` = copy of `C:/Users/Admin/freept/work/seam_R0B.npz`, `score/refe_rl1_T/refe_rl1_T.csv` = copy of `E:/Projects/TanitAD/data/refe_navtest/score/refe_fp_R0B/refe_fp_R0B.csv`, and `points_T.npz` = the freept `points.npz` plus `pick_T` = its `pick_R0B`; run `nt_read.py --variant T --work nav_dry --out nav_dry/result`. Required: V_vs_BASE equals FREEPOINT R0B vs BASE (PDMS +0.55 [+0.24, +0.84], NC -0.25, TTC -0.40) and identity_a max diff 0.0. Fix any bug before the real read (this is not a second navtest open: identical banked rows).
3. Real read: `python nt_read.py --variant V1 --work C:/Users/Admin/par18/rl/nav --out C:/Users/Admin/par18/rl/nav/result` (families6 recomputation takes minutes: launch detached per the WMI recipe in the brief). Read `OUTCOME` (N-PASS / N-GAIN-LOST / N-SAFETY-LOST / N-BOTH) and the gates (`harness_PASS`, `rows_12146`, `identity_a_same_pick_rows_equal` must all be true, else INVALID).
4. Update `out/RESULT.md` headline item 5 and section 6 with the navtest numbers (levels x100, V1 - BASE with CIs for PDMS and the six sub-scores, lever view V1 - R0B and V1 - C0A, families6, DEPLOYABLE-SAFE), apply the PREREG_B section 5 row for the observed outcome (next arm text is already there), mark my RL-val forecasts as checked. Report every number with its tier (T1 NAVSIM v1.1 navtest) and the estimator (paired log-cluster bootstrap, 10,000, seed 20260927, "another draw of LOGS" only, one seed).
5. Stage: copy `rlh.py mechanism.py composites.py nt_build.py nt_run.py nt_read.py` to `out/raw/code/`, `nav/build_V1.json nav/points_V1.npz nav/seam_V1.npz nav/result/* nav/score_logs/*.json` to `out/raw/navtest/` (the harness CSV `nav/score/refe_rl1_V1/refe_rl1_V1.csv` is small; the `*_hooks.json` / `*.log` there are 10+ MB: copy the CSV, status and counts only), then regenerate `out/paths.json` as `[repo path, absolute source, "raw"]` rows with repo prefix `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-rl-headwise/`. Write a deliverable manifest in RESULT.md. Hand back <= 300 words.

## Open blocker (name it again in the hand-back)
R0d dumps are not on the dev box (pod + private HF `Sayood/tanitad-refe-rls1-r0d` only; 9 files, 103 MB). A download needs the caller's go-ahead. Once local, `mechanism.py --arm r0d --dl <dir with r0d/final/>` and the composite loop run unchanged.


## UPDATE 2026-10-18 23:25 Berlin (Master Mind) -- supersedes "State" step 4 and the R0d blocker above
* **The first V1 scoring run FAILED and scored nothing usable.**
  * Attempt 1 (22:24-22:38) streamed 4,433 of 12,146 tokens, then W3's RAM guard aborted the devkit: available memory fell under 3,000 MB, because two other sessions' full stack test suites held 7.3 GB.
  * The harness reconstructed a PARTIAL CSV from those rows, then crashed at `run_v1.py:459` on `os.path.relpath` across drives (C: vs E:) before its C2/C3 row checks ran.
  * Attempts 2-3 aborted at once on RAM, and the glob fallback re-copied that partial CSV. All of it is quarantined, NOT read, in `nav/quarantine_ram_abort_20261018/` (sha256 d2539c5f... for both copies).
  * **Disclosure for RESULT.md:** the Master Mind saw that partial CSV's average row while diagnosing: 4,433/12,146 tokens, PDMS 0.8844. V1 and its bar were frozen before that (PREREG_B), so nothing is selected on it. State this in section 6.
* **Relaunch RUNNING:** `nav/nt_run2.py` (PID 42560, WMI-detached), wrapper `nav/score_e2.py`.
  * Three guards: a cross-drive relpath fallback; a memory gate (psutil available >= 6,500 MB for 3 consecutive minutes); stale CSVs moved aside before each attempt.
  * OK additionally requires 12,146 token rows. Trail: `nav/score_logs/refe_rl1_V1.nt_run2.log`. The marker is still `nav/DONE_V1`.
  * Stage `nt_run2.py`, `score_e2.py` and the trail log with the package.
* **R0d is now LOCAL** (PI go-ahead): `r0d_dl/r0d/` holds 43 files, 344.8 MB, sha256-verified against the hub. The 2 x 1.29 GB SFT model checkpoints were skipped.
  * The Master Mind ran `mechanism.py --arm r0d --dl C:/Users/Admin/par18/rl/r0d_dl --controls` -> `out/raw/mechanism_r0d.json` (rc 0, all controls pass). Label it an EXTENSION: PREREG_A excluded R0d. Result: lane 0.49-0.62 vs own A at u250-u1000, MECH-PROGRESS at all four states; it replicates R0.
  * Add it to RESULT.md step 1.
* **R0d composites and the lane-dose family are now owned by a separate stream (RL-2, `C:/Users/Admin/par18/rl2/`).** Do not duplicate them here.
* **When landing RL-1, also land a TRAPS.md entry:** "W3 navtest: a RAM-guard abort leaves a reconstructed PARTIAL CSV, which the glob fallback re-copies; a cross-drive relpath crash skipped the C2/C3 row checks". Fix pattern: `nav/nt_run2.py`.
* **UPDATE 23:58 (Master Mind):** nt_run2 attempt 1 (23:30-23:51) was RAM-aborted again: wrapper rc 3 at 8,756/12,146 rows. The likely cause is a sibling agent's 3 GB stream scan.
  * This time the C2/C3 checks RAN and REJECTED the partial (status FAIL, missing 3,390 tokens). The guards work.
  * **Second partial view, disclose it:** the status json printed the partial summary, PDMS 88.47 over 8,756 tokens, and the Master Mind read it.
  * **Disclose this too:** `rl2 MM_DECISION_NAVTEST2.md` (landed bd37576) was written AFTER the first partial view (88.44 over 4,433) and before this one. Its rule is outcome-category-based and selects nothing, but the timing must be stated in RESULT.md section 6.
  * Attempts 2-3 follow automatically.

## UPDATE 2026-10-19 00:45 Berlin (RL-1 continuation agent) -- RL-1 IS COMPLETE; nothing is running

* Steps 2-5 done. Dry known-value test of `nt_read.py` PASSED (no bug, file unchanged); the real read ran on `nav/` (V1 attempt 2 of `nt_run2`, 12,146/12,146, harness PASS). **OUTCOME N-BOTH**: V1 - BASE PDMS +0.10 [-0.07, +0.26], NC -0.12 [-0.24, -0.02] ADV, DAC -0.12, EP +0.53, TTC -0.13 [-0.32, +0.06]; full tables in `out/RESULT.md` section 6.
* Triggered: PREREG_B section 5 row N-SAFETY-LOST/N-BOTH -> **R0e-pick** (A40, BLOCKED); RL-2 `MM_DECISION_NAVTEST2.md` row N-BOTH -> **look #2 for L0.25 = NO**. The two documents disagree on the next arm (R0e-pick vs the anchored arm): Master Mind / PI to choose.
* Everything is staged under `out/` (see `out/RESULT.md` section 11 and `out/raw/MANIFEST_RL1.sha256`); landing list `out/paths.json`. Not in git. TRAPS entry text is in RESULT.md section 10.
* Binding rules unchanged: navtest is spent (no re-score, no second variant from this package); CPU only; no git add/commit/push; no register edits.
* Open blockers: A40 compute + PI grant + launch-gate token for R0e-pick / the anchored arm. R0d dumps are now local (`r0d_dl/`), read as an extension (RESULT.md 2b).
