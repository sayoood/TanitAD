# RECLAB pod scripts at the code of record cf27dbb7 (MM, 2026-11-22)

Card: `PREREG_V8F_RECLAB.md` (frozen 2026-11-22T03:53:32Z, sha256 154d7ad9...). Code: `cf27dbb7` on `agent/v8f-rollhist-cdb-20261120` (97fe0a56 plus the `--dagger-label` flag). Nothing here has run on the pod; no GPU, no spend.

## Why these files exist

The build staged the RL chain (`../rh_rl_chain.sh`, `../rl_arm.sh`, `../chk_arm_rl.sh`) at 97fe0a56, the RH code of record. The flag exists only at cf27dbb7, so the RL arms need their own tree and a fresh G-SUITE. Preparing that found one defect in the build's scripts:

* **The RL arm would have been refused at the gate record.** `rl_arm.sh` called the pod's `rh_gate_record.py`, which takes its RH branch only when the tag starts with "RH" (`rh_gate_record.py:63`, `:98`). `RL_s0` failed that test and went down the B0 branch ("B0 --dagger-p is not 0", "B0 argv names --dagger-bank"). MEASURED from source. The chain would have failed closed after the GPU selfcheck, under the lock. The RHp3 arms passed because "RHp3" starts with "RH"; RHK had its own `rhk_gate_record.py`.

## The files (sha256, first 16)

| file | what | sha256 |
|---|---|---|
| `rl_env.sh` | sourced by the RL scripts instead of `rh_env.sh`: sources it unchanged, then sets `TREE=$R/tree_cf27`, `COMMIT=cf27dbb7...`, `BLOBS`, `GSUITE_RL` | 8e907088803b9e88 |
| `rl_tree_prep.sh` | MM, once, CPU: `tree_cf27` = `tree_cdb` (blob-checked first) + the 8 blobs of `rl_delta_cf27.tar`, then a blob-by-blob check against `all_blobs_cf27dbb7.txt` and a file count | see `SHA256_CF27.txt` |
| `all_blobs_cf27dbb7.txt` | `git ls-tree -r cf27dbb7 stack taniteval`, blob and path, plus the row for `raw/eval_grid_windows.json` (blob c489b553, the same at both commits) at the end, as the pod's `all_blobs_97fe0a56.txt` has it: 3,413 rows | 3fca0151af391e17 |
| `rl_gsuite.sh` | copy of `rh_gsuite.sh` (477082d7): env, blob list, junit and temp dirs, log and result names, sentinel | 2a2c2d0177b252ae |
| `rl_gate_record.py` | made by `make_rl_gate_record.py` from `rh_gate_record.py` (847019579b8ae271): COMMIT cf27dbb7, `gsuite_result_rl.json`, `done_rl/tree.done`, RL takes the RH branch plus three label checks, any other tag refused | a9a0a5fa76642870 |
| `rh_rl_chain.sh`, `rl_arm.sh`, `chk_arm_rl.sh` | made by `make_rl_cf27.py` from the build's forms (6e2934ab, b2648f4a, 7b6e4409): `rl_env.sh`, the new blob list, a `gsuite` step after `tree` (kept when a prep run already passed), `rl_gate_record.py`, `gsuite_result_rl.json` | d8a660055627f77c, 0ee7c8a08e254125, 784d0162eac05642 |
| `test_rl_gate_record.py`, `test_log_rl_gate_record.txt` | planted files: a well-formed RL_s0 mints and verifies; 10 broken forms are refused | 3f6b78e950a579bb, 720e4f8f243e7784 |

`rl_delta_cf27.tar` (sha256 9418e753a6465d36) is not landed: rebuild it on the mirror with
`git -c core.autocrlf=false archive --format=tar -o rl_delta_cf27.tar cf27dbb79f2491a6db1197860bc50ca7c8cdeff4 -- stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_dagger.py stack/tanitad/data/v8f_recovery.py stack/tests/golden_v8f_dagger_97fe0a56.py stack/tests/test_v8f_dagger_train.py stack/tests/test_v8f_recovery.py stack/tests/test_v8f_recovery_labeller.py stack/tests/test_v8f_recovery_wiring.py`, and check the sha256 above before shipping.

## Checks on the dev box (MEASURED)

* A pod probe before shipping (07:07Z, read only) found the pod's `all_blobs_97fe0a56.txt` at sha256 53e0333783446b75 with 3,408 rows, not the 95d6bbb6 / 3,407 of a plain `ls-tree` of `stack taniteval`: its last row is `raw/eval_grid_windows.json`, which `tree_cdb` holds. The first `rl_tree_prep.sh` would have refused at its sha check, and a list without the row would have failed the file count in `rl_gsuite.sh` and in the chain's tree step. The pod file is reproduced on the mirror byte for byte (ls-tree in `blob path` form plus that row, sha256 53e0333783446b75), and the cf27 list is built the same way (3,413 rows).
* The cf27dbb7 blob list differs from 97fe0a56's in exactly 8 rows: 5 added (`v8f_recovery.py`, three recovery test files, the golden), 3 changed (`v8f_mini_train.py`, `v8f_dagger.py`, `test_v8f_dagger_train.py`). This is the land guard's A 5 / M 3.
* The six changed test files at cf27dbb7, whole files, dagger-first, CPU only (`../../test_log_cf27dbb7_cpu.txt`, `../../run_cf27_tests.sh`): **234 passed, 2 skipped, 1 failed**. The failure is `test_v8f_ego_dagger_compose.py::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit`, which also fails on the pod at the base cdbff971 and at 97fe0a56 (`junit_rh` and `/root/strat1/junit/base`, shards 5 and 6; the RH G-SUITE counts 635 tests failing on both). This run repeats the full order after the build's last one-parameter test edit.
* A first run (`../../test_log_cf27dbb7.txt`) left the dev-box GPU visible: on Windows an empty `CUDA_VISIBLE_DEVICES=` is dropped, so the trainer ran on CUDA and four CPU-written tests failed on device mismatches and GPU non-determinism. That run is not a result about the code. It also used the shared dev-box GPU without its lock, for about 9 minutes in that run and about 30 s in a second try that was stopped; no lock was held when checked afterwards. The script now sets `-1` and refuses unless torch reports no CUDA.
* `rl_gate_record.py`: well-formed MINTED and VERIFY OK; M1-M10 RED (label flag missing, `human`, levers without speed_v1, probe rows 0, probe key missing, G-SUITE at 97fe0a56, tag RH_s0, no RL tree marker, wrong bank sha256, a bound file changed after the mint).

## Order on the pod

Steps 1 and 2 are done (2026-11-22, 07:07-07:09Z, CPU, seconds, no GPU or lock): the 13 files are on the pod (`sha256sum -c SHA256_CF27.txt` OK there), and `rl_tree_prep.sh` printed `ZZ RL TREE cf27dbb7 OK ZZ` (tree_cdb 3,408 of 3,408 blobs, tree_cf27 3,413 of 3,413, log `/root/rollhist/logs/rl_tree_prep.log`). PI go "Run both" (2026-11-22 about 07:12Z). Step 3: `rl_gsuite_locked.sh` (b1859777, this folder) first ran the suite under the lock from 07:40:19Z; the peer then said the suite needs no lock and does not hold its CPU guard, so that run was ended at 07:45:50Z (about 5 min in) and `rl_gsuite.sh` ran standalone 07:46:07Z to 08:23:54Z: **PASS, 0 regressions, 0 new-test failures, 0 missing, 106 new-only tests** (`pod_records/gsuite_result_rl.json`, logs beside it, sha256 in `pod_records/SHA256_POD.txt`). Five tests read skip in the RH suite and fail here (3 in `test_v8f_wc.py`, 2 in `test_refcv7_vis1.py`), none touched by the 8 changed files: they skip on a tree with no Research Lab folder and fail on one that has only part of it. `tree_cdb` received the eval-141 grid at 2026-11-21 04:47:40Z, after the RH suite ran (02:17-02:44Z), and `tree_cf27` carries it, so these tests now run and miss the package files (messages in the junit, MEASURED). Step 4: dry mints of RL_s0 and RL_s1 08:28Z, TOKEN-WRITTEN, all 21 checks of each `checks.json` pass (C-ARGV ... G-SUITE); `dry_rl/` (768 KB) deleted. The label read-back (G-DAGGER on the RL selfcheck) and `rl_gate_record.py` run only in the real chain, after the GPU selfcheck. Step 5: `launch_rl.sh` 08:28:44Z (`ZZ RL LAUNCHED ZZ`); chain start 08:28:39Z, waiting on the lock; the marker watcher set `/root/N1DW_READY` at 08:28:38Z.

1. Ship: this folder's `rl_env.sh rl_tree_prep.sh all_blobs_cf27dbb7.txt rl_gsuite.sh rl_gate_record.py rh_rl_chain.sh rl_arm.sh chk_arm_rl.sh`, the rebuilt `rl_delta_cf27.tar`, and from `../`: `rl_argv.py rh_rl_marker_watch.sh launch_rl.sh`, `dscale10k_check_rl.py` into `study/`. Verify sha256 on the pod.
2. `bash /root/rollhist/rl_tree_prep.sh` (expect `ZZ RL TREE cf27dbb7 OK ZZ`).
3. G-SUITE at cf27dbb7: `bash /root/rollhist/rl_gsuite.sh` as a standalone CPU job when no training runs on the pod (about 36 min; the chain keeps the pass), or let the chain run it under the lock.
4. Render and dry mint as `../../DRY_MINT_PLAN.md` B (needs the G-SUITE result for the evidence step), then delete `dry_rl/`.
5. Launch through `launch_rl.sh` (marker watcher and chain; the chain waits on `/root/gpu_queue.lock`).
