# HANDOFF -- kdraw (K-draw ensemble E1 read), written 2026-10-19 00:16 +0200

## State
* **PRE-REGISTERED** (before any new roll): `raw/kdraw/PREREG_KDRAW_E1.md` (+ Addendum A). Arms ENS-K / MEDOID-K, K=4 on disjoint sets A={0,1,2,3} B={4,5,6,7}, K=8; bar = E1 separated < 0 on BOTH sets (+ the 2x-floor NOT PROVEN rule); PASS / PARTIAL / FAIL committed.
* **Queue ALIVE, detached, nothing rolled yet** -- it is waiting its turn: pid file `/e/refcv8_eval_kit/chain/kdraw_queue.pid`, log `/e/refcv8_eval_kit/chain/kdraw_queue.log` (markers `ZZKDQ...ZZ`).
  Gate order: (Y) yield to force_s2 (ready, waiting for 12 GB commit) and to the float64 probe `/c/lgs/f64_after_fs2.sh` -> (M) free commit >= 10 GB twice 60 s apart -> (L) GPU lock (foreign holder `srttc_s0` since 23:14; never broken).
  Jobs: `ctrl` (3 prefix re-rolls vs the banked dumps, ~10 min), then seeds 2,3,4,5,6,7 (os-only, ~45.5 min each on an idle GPU, MEASURED 2,730.6 s in `roll_s1.json`). Up to 4 jobs per lock hold.
* Expected GPU time once running: ~5 h. ETA of the start is NOT knowable (foreign GPU job + force_s2 + fp64 + commit < 10 GB); the Master Mind can read the log.
* **RELEASE_R7** is written by the queue's EXIT trap (success OR failure) AFTER the lock is released: `echo "released by kdraw <ts>" > /c/lgs/RELEASE_R7`. It therefore appears only when the queue ends (~5 h of rolls after the gates open). The deferred refcv7 chains
  (`/c/lgs/f64_after_fs2.sh` step 3) wait for it. To release them EARLIER, write that file by hand -- the queue keeps going and rewrites it harmlessly. Dry-run flag `KD_DRY=1` never writes it.

## Check
* alive: `ps -ef | grep -c '[k]draw_queue_r8'` (>=1) and `tail /e/refcv8_eval_kit/chain/kdraw_queue.log` (YIELD / MEMWAIT / LOCKED / JOBSTART / JOBDONE / JOBFAIL / SEGEND / RELEASE_R7).
* rolled seeds: `python kdraw_roll_r8.py status` (needs the env of the run script) or `ls /e/refcv8_eval_kit/kdraw/step30000ema_unc/dump_s*/KDRAW_VERIFY.json`. Roll logs: `/e/refcv8_eval_kit/kdraw/step30000ema_unc/roll_s<k>.log`; python-side log `/e/refcv8_eval_kit/chain/kdraw_queue_py.log`.
* determinism control: `/e/refcv8_eval_kit/kdraw/step30000ema_unc/KDRAW_CONTROL.json` (`verdict.deterministic_per_seed`; case C1 = seed 1 os-only prefix vs banked `dump_s1`, C3 = seed 0 full vs banked `dump_s0`, C4 = seed 0 os-only vs `dump_s0`).
* stop WITHOUT releasing R7: `kill -9 $(cat /e/refcv8_eval_kit/chain/kdraw_queue.pid)` (a plain `kill` runs the EXIT trap and writes RELEASE_R7). If it is inside the lock wait, ALSO kill its child `gpu_lock.py acquire --job refcv8-kdraw-*` by explicit PID (an orphaned acquire could still take the card later). Restart = the same launch command (resumable; done jobs are skipped).

## Exact next steps
1. (any time, works on partial seeds, 3-4 s CPU) **`sh C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/run_kdraw_analysis_r8.sh`** -> `raw/kdraw/KDRAW_E1.json` + `.md` (judge by the JSON; `overall` = PENDING until sets A and B are both rolled; any failed control prints VOID, never a verdict).
2. Read `KDRAW_E1.md`: controls first (C-K1/C-K2 reproduce -0.0085/-0.0076/-0.0138; C-MASK sha 4ebc6b86...), then E1 cells, verdicts, mode disagreement, NR. Report PASS/PARTIAL/FAIL exactly as PREREG section 3, with the sentence "bar of record = single-draw os; exploratory at 30k".
3. Final checkpoint (registered read, ~2026-10-20 19:00): after the final battery banks `dump_s0/dump_s1` for `step48852ema_unc`, run `KD_STEP=48852 nohup sh code/kdraw_queue_r8.sh ...` (needs `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt` and `battery_summary.json`/`windows_s0.json` under `refcv8_eval_archive/battery/step48852ema_unc` -- UNVERIFIED names) then `sh code/run_kdraw_analysis_r8.sh step48852ema_unc` (writes `raw/kdraw/step48852ema_unc/`). The mask sha must again be 4ebc6b86....

## Files (all new, in PKG; nothing edited, nothing committed)
`raw/kdraw/PREREG_KDRAW_E1.md`, `raw/kdraw/HANDOFF.md`, `raw/kdraw/pytest_test_kdraw_e1_r8.log` (16 passed); `code/kdraw_roll_r8.py`, `code/kdraw_queue_r8.sh`, `code/kdraw_e1_r8.py`, `code/run_kdraw_analysis_r8.sh`, `code/test_kdraw_e1_r8.py`.
Off-PKG artifacts (by design): dumps and control under `E:/refcv8_eval_kit/kdraw/step30000ema_unc/` (small; bank `KDRAW_VERIFY.json` + `KDRAW_CONTROL.json` with the result), queue logs under `E:/refcv8_eval_kit/chain/kdraw_queue*`.

## Known limits
Single training seed. T1 open loop. S2 only (no E3). The 30k read is exploratory. Memory/commit on the box is the binding constraint (free commit 5-7 GB at 00:15). Env knobs: `KD_NEED_GB`, `KD_FS2_YIELD_S` (21600 set), `KD_F64_YIELD_S` (14400), `KD_LOCK_MAXWAIT_S` (28800), `KD_SEG_MAX`, `KD_JOBS`.
