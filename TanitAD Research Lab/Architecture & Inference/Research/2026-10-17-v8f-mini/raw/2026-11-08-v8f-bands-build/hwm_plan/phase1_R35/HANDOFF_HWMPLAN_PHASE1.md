# HANDOFF HWMPLAN phase 1 on R35 (2026-11-14, supersedes the R35H version): prepared, job WAITING for the RAM floor

Tier T0 (world-model diagnostic, never driving performance). Frozen code of record untouched (blob-equal check from the previous agent stands; the hwm_plan.py:66 assert PASSES on R35, MEASURED by the smoke).

## Done (MEASURED)
1. R35 checkpoints, sha256 equal to each pull's BANK_MANIFEST.json (45,266,310 B each): s0 `d920d62c6c391a61291b966a6af1783156b4351d25c7408913e4191daa0215cf`, s1 `a97e6017cffb13830dacd04e1ea108fa8534bc3ab3007b8f536f01ac0846f881` (`C:/Users/Admin/t2v8f_pulls/ladder_R35_s{0,1}/ckpt_last.pt`).
2. R35 TRAIN latents: 141 clips x 17 starts = 2,397 windows, Z1 (2397,16,128), A_raw (2397,16,2); control 4 max_abs_diff 0.0 for both seeds (`collect_R35.log`, 405 s, clips by read-only sftp from box 2, md5-checked, nothing written there).
   `F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt` sha256 `76a2d4f5482bd98670bab40ebbf4b3c85fde8ebf222596947fa16f0c7e82754c`
   `F:/TanitAD-scratch/hwm_plan/latents_train_R35_s1.pt` sha256 `692eb0021a4b39295a52589ab2b7f2de0a45615345bca7a7acea62101d32708a`
   Collector `collect_latents_train_R35.py` (R35H copy; only checkpoint tags/paths and the partial-file name changed).
3. SCORE subset (`subset_score.py`): 994 windows, 71 episodes (70 TUNE episodes excluded) = RESULT_LADDER_PLANTEST.json R35 c1 `n_windows_score` 994 / `n_episodes_score` 71. Second route `chk_split.py`: sha256-rank split rebuilt from scratch on the 141 episode ids; the expected (eid, now) set equals the subset set (True). Geometry read from the built model: s2=3, s3=5, K1=4, K2=2, T3=2, n_traj=30, n_op=10, stitch decode.
4. 16-window smoke, both seeds, hwm_oracle + flat_oracle, inference seed 0, prior seed-mode, lam 0 (`smoke_R35.sh`, `smoke_R35_s{0,1}.log`, outputs `x/smokeR35_*`): all plans and costs finite; hwm_oracle repeat run bit-equal on every array (`chk_smoke.py`). No metric read. 26-37 s per 16-window unit at 1 worker.
5. s0 windows already built: `phase1_R35/s0/windows_all.npz`, SCORE `windows.npz` sha256 `371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee` (s1 is built by the job).

## Running
Free RAM was 5.26 GB at launch (below the 6 GB floor; other sessions' jobs, not stopped). A WAITER runs detached: WINPID cmd 29136 (bash 47020). It polls every 5 min, starts `phase1_R35.sh` when RAM >= 6 GB and C: >= 6 GB (2 workers if RAM >= 8 GB, else 1), gives up after 12 h. State: `F:/TanitAD-scratch/hwm_plan/phase1_R35/WAITER_STATUS.txt` (WAITING / LAUNCHING / JOB_EXITED / GAVE_UP), `waiter.log`. The job logs to `phase1_R35/phase1_R35.log`, banks one plans file per (ckpt, arm, seed) at `phase1_R35/s{0,1}/plans_<arm>_s<seed>.npz` (a re-run skips banked ones) and writes `phase1_R35/PHASE1_R35.done` = `rc=N <time>`. If GAVE_UP or rc!=0: read the log tail, fix the cause, relaunch `phase1_R35_waiter.cmd` through the WMI recipe (body of `launch_waiter.ps1`, run inline; ExecutionPolicy blocks .ps1 files).
ETA (ESTIMATED: 16-window smoke scaled to 994 windows, 64 units per checkpoint, not measured at full size): about 2.5 h at 2 workers, about 5 h at 1, plus the wait.

## Next step once `PHASE1_R35.done` reads rc=0
Git-bash, `source /f/TanitAD-scratch/hwm_plan/common.sh; cd $H`, then per seed tag:
`for S in s0 s1; do $PY -u run_hwm.py report --tag R35_$S --ckpt C:/Users/Admin/t2v8f_pulls/ladder_R35_$S/ckpt_last.pt --train-latents F:/TanitAD-scratch/hwm_plan/latents_train_R35_$S.pt --train-tag R35_$S --seed-mode prior --lam 0 --arms hwm_oracle flat_oracle --seeds 0 1 --out F:/TanitAD-scratch/hwm_plan/phase1_R35/$S; done`
It writes RESULT_HWMPLAN_R35_s*.json/.md (hold and prior refs, paired episode-cluster bootstrap, inference-seed replicate). Read against the card's pre-registered outcomes (PLANNER-LIMITED / GOAL-LIMITED, HIER rows) and its controls, then copy the results into this folder.

## Deletions (own temp data)
`F:/TanitAD-scratch/hwm_plan/ts/cache/train/clips/_current.pt` (last pulled clip copy, ~20.9 MB, re-creatable from box 2); `F:/TanitAD-scratch/hwm_plan/latents_train_R35_partial.pt` (~40 MB, mid-run checkpoint of the finished collection; exact sizes not logged before deletion). Kept: R35H latents (inputs for phase 1b), smoke dirs (1.1 MB).
