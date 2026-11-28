# HANDOFF -- refcv8 QUEUE behind the step-5,000 battery chain (armed 2026-10-16 10:56), EvalFlyWheel

NOT staged in git, NOT committed. The Master Mind lands (qland). Single copy of this package: `C:/Users/Admin/qland/hold_r8batt/stage/.../2026-10-15-refcv8-milestone-battery/`.

## STAGE 3 -- the queue (newest section; the two sections below are unchanged and still true)
* ARMED: `nohup sh code/queue_after_5000.sh > E:/refcv8_eval_kit/chain/queue_after_5000.out` -- MSYS pid **233019**. Log with opaque markers:
  `E:/refcv8_eval_kit/chain/queue_after_5000.log` (first line `ZZQQUEUESTARTZZ`, terminal `ZZQQUEUEENDZZ`); timings `chain/queue_timings.txt`; one line per step in `KIT.log`.
  GPU-lock wait chatter goes to `chain/queue_lockwait_<job>.log`, not the marker log. Do not edit `queue_after_5000.sh` while it runs (bash reads scripts lazily).
* FLOW (each stage independent: a failure is logged, the next runs; every GPU step takes the shared O_EXCL lock, never breaks a foreign one, the stage's EXIT trap releases it):
  **W** wait for the NEWEST segment of `chain/battery_step5000.log` to reach a terminal marker (pass 2 of the 5,000 chain, `ZZR8BATTCHAINENDZZ`, expected ~11:50-12:30) -> 
  **1 wrapfix** `wrapper_clause_r8fix_gpu.py` on `ckpt_5000.pt` (job `refcv8-wrapfix-step5000`, ~7 min: dataset build + one collate + two P3 probes) -> `raw/step5000/wrapper_clause_r8fix_gpu.json`
  (`CLAUSE` = the amended W-R8-1 reading; also `control_unfixed` (must reproduce G0's `cascade` 1.235e-01 / `r8_subscore` 1.642e-02 -> `control_reproduces_g0`), `fixed_as_returned`, `bars`) ->
  **2 ema** CPU gate (`ema_check_r8.py`, md5-pinned; STOPS the stage if the EMA is identical to raw or the strict load is not clean) -> battery PASS 1 on `ckpt_ema_5000.pt`
  (job `refcv8-battery-step5000ema`; `run_battery_r8.py --no-g0 --tag step5000ema_unc --infer-seeds 0,1 --skip-roll --lock-held`, env `R8_NO_POSE_SYNC=1`, `R8_SEAM_TELE_DIR=E:/refcv8_eval_kit/battery/seam_tele_step5000ema`;
  one retry) -> `pair_read_r8.py --clock standalone` (paired vs refcv7-50,400) -> `NO_G0_REASON_step5000ema.json` + `bank_battery_r8.py` -> `raw/step5000ema/` (~3.3 h) ->
  **3 milestone 10,500** `thor_pull_r8.sh` (idempotent: already pulled, see below) -> `ema_check` (info) -> `chain_g0_r8.sh 10500` (own lock/markers; ~2 h) -> wrapper-fix probe at 10,500 (non-blocking extra, `Q_WRAPFIX_MS=0` to drop) ->
  `chain_battery_r8.sh 10500` with `R8_CORRECTED=0` (pass 1 only, ~3.3 h; `Q_CORRECTED=1` adds pass 2) -> `raw/step10500/BATTERY_SUMMARY_step10500.json`.
  Resumable: re-run the same command; a stage whose md5-pinned artifact says done is skipped (`Q_FORCE=1` reruns). `Q_STAGES="ema"` runs a subset; `Q_MILESTONE=15000` retargets stage 3.
* ⛔ **DEVIATION FROM THE BRIEF -- there is NO step-10,000 pair.** The trainer saves `ckpt_<N>.pt` + `ckpt_ema_<N>.pt` only at `MILESTONES=(5000,15000,20000,30000)` (`refc_v3_train.py:232`); MEASURED on Thor 10:17: run dir =
  `ckpt_5000.pt, ckpt_ema_5000.pt, ckpt.pt` (rolling, every 500 steps, 1,560,403,737 B, was step 10,500). I pulled that rolling file (md5 `ad55ca34b0f8aadb0a38b6a7331662b1` on BOTH sides, Thor mtime unchanged across the
  copy) at 10:24 and derived **`E:/refcv8_eval_kit/ckpt/ckpt_10500.pt`** (md5 a08a6231...) and **`ckpt_ema_10500.pt`** (md5 8863ae3f...; 1165 keys, 821 float tensors differ from live, decay 0.9998, num_updates 10500) with
  `slim_rolling_ckpt_r8.py`; sidecar `ckpt_10500.provenance.json` + `ckpt_10500.md5`. The rolling copy was deleted after the slim pair was written (free space). Metrics copy with exactly one eval row at 10,500
  (`eval_loss` 51.56521, `eval_traj` 0.61754): `E:/refcv8_eval_kit/thor_reads/metrics_20261016T102601.jsonl`. The derived pair is NOT trainer-written; the derivation is pinned against the real `WeightEMA` by a test.
  If you would rather wait for the real step-15,000 pair (~tonight), kill the queue before stage 3 and run `Q_MILESTONE=15000 Q_STAGES=milestone sh code/queue_after_5000.sh` later.
* CPU gates already run (artifacts): `raw/step5000ema/EMA_CHECK_step5000.json` -- strict load clean (missing 0 / unexpected 0), steps equal (5000), **821 of 1,061 float tensors differ, rel L2 distance 1.37 %, max abs 0.226** => the EMA arm is not pointless;
  `raw/step10500/EMA_CHECK_step10500.json` -- same, for the derived pair. CPU smokes: (a) `wrapper_clause_r8fix_gpu.py --device cpu` on the diag batch cache (`raw/step5000/wrapper_clause_r8fix_CPU_SMOKE.json`): the UNFIXED control reproduces the earlier CPU diagnosis bit-for-bit (max wrapper 5.174e-02, floor 2.406e-01), the fixed+amended reading is PASS (366 terms, `r8_spd_crc` dropped, max wrapper 5.2e-06 on `traj` vs the diagnosis' 3.4e-06 -- thread-count numerics, bar 1e-5, UNVERIFIED which), W1/W2 detected; (b) the EMA battery path end to end on CPU (`run_battery_r8.py --smoke-cpu --no-g0` on `ckpt_ema_5000.pt`, 2 clips x 3 windows -> `pair_read_r8.py` without `--g0-json` -> `bank_battery_r8.py --passes unc`): all three produced their artifacts (code-path check only, no number is a result); (c) `test_queue_after_5000.py`: 9 passed (`raw/pytest_test_queue_after_5000.log`), 4 mutations of the orchestrator went RED.
* HOW TO CHECK: `grep -c . E:/refcv8_eval_kit/chain/queue_after_5000.log; tail E:/refcv8_eval_kit/chain/queue_after_5000.log`. Per-stage logs: `chain/wrapfix_step5000.log`, `chain/battery_run_step5000ema_unc.log`, `chain/pair_read_step5000ema_unc.log`,
  `chain/queue_pull_10500.out`, `chain/thor_pull_step10500.log`, `chain/queue_g0_10500.out`, `chain/chain_step10500.log`, `chain/battery_step10500.log`. Done = `raw/step5000/wrapper_clause_r8fix_gpu.json` (`ok` true),
  `raw/step5000ema/BATTERY_SUMMARY_step5000ema.json`, `raw/step10500/BATTERY_SUMMARY_step10500.json` with `passes.unc.status == COMPLETE`.
* KILL SAFELY: `kill <queue pid>` stops only the parent loop; the running stage is a child subshell -- find it with `ps -o pid,ppid,args` (ppid = queue pid) and kill THAT pid (its EXIT trap releases the GPU lock), then its python
  children by explicit PID only. Never `pkill -f`.
* UNVERIFIED (nothing below ran on the GPU): the wrapper-fix script on CUDA (CPU smoke only; its control block can only be checked against the banked G0 terms on the GPU), GPU memory / wall-clock of the EMA battery (same code path
  and size as the raw pass: 3.04 GiB peak for refcv7, the raw refcv8 pass ran), the G0 / battery at 10,500, `--no-g0` stamping in `pair_read` (it reads `G0 NOT AVAILABLE, PROVISIONAL, no g0.json`; the real reason is in `NO_G0_REASON_step5000ema.json`).
* Single training seed; EARLY reads of a 48,852-step run; the EMA arm and the 10,500 read are exploratory: nothing here is a bar verdict.


---
## (earlier) HANDOFF -- refcv8 step-5,000 BATTERY chain (armed 2026-10-15 23:31) + G0 (below), EvalFlyWheel

NOT staged in git, NOT committed. The Master Mind lands (qland). Single copy of this package: `C:/Users/Admin/qland/hold_r8batt/stage/.../2026-10-15-refcv8-milestone-battery/`.

## STAGE 2 -- the unattended battery chain (this section is newest; the G0 section below is unchanged and still true)
* ARMED: `nohup sh code/chain_battery_r8.sh 5000 > E:/refcv8_eval_kit/chain/battery_5000.out` -- MSYS pid **45849**, started 23:31:06. Log with markers:
  `E:/refcv8_eval_kit/chain/battery_step5000.log` (first line `ZZR8BATTSTARTZZ`). It polls the G0 chain's log (G0 chain = MSYS pid 2699, still
  `running:PRECONDOK`: G0 is STILL QUEUED behind NavSim's `aux_navtest_forward` lock; no GPU work has happened). Do not edit `chain_*.sh` while they run.
* FLOW: wait for a G0 terminal marker -> **gate on artifacts** (`g0.json` present, for THIS ckpt md5, `M1_detected` true; else `ZZR8BATTSKIPZZ <reason>` +
  `raw/step5000/BATTERY_SUMMARY_step5000.json` status SKIPPED, exit) -> wait_commit -> GPU lock (never broken; EXIT trap releases) -> **PASS 1 `unc`**
  (standalone pose clock, `R8_NO_POSE_SYNC=1`, banked baselines incl. refcv7-50,400; seed 0 all arms, seed 1 os + controls; S2 + S6; panels; four families)
  -> `pair_read_r8.py` -> sanitised bank -> **`ZZR8BATTDONEZZ`** -> **PASS 2 `corr`** (corrected clock = the run's own, os-only both seeds, NO baselines:
  E1/E3 only) -> `ZZR8BATTCORRDONEZZ` -> `ZZR8BATTCHAINENDZZ`. Other markers: `...G0ENDEDZZ, GATEOKZZ, LOCKEDZZ, PASSDONEZZ, PAIREDZZ, RETRYZZ, FAILZZ, G0DEADZZ, NOLOCKZZ, NOCOMMITZZ`.
  **DONE = the primary (paired) read is banked, the lock is still held for pass 2; CHAINEND = everything finished, lock released.** Resumable: re-run the same
  command; `--skip-roll` reuses only complete seed dumps; one automatic retry per pass.
* POLICY (stamped, MM may change): G0 FAIL-as-coded with M1 detected still runs the battery (`--g0-nonblocking`), every number then PROVISIONAL
  (`BATTERY_SUMMARY.g0_gate`, `PAIRED_READ.g0.stamp`); `R8_REQUIRE_G0_PASS=1` turns that into a skip. Off by default: `R8_PERC=1` (raw perception pass after
  the four families), `R8_YIELD=1` (refcv7 evening-yield), `R8_CORRECTED=0` (skip pass 2). Env must be set when the chain is LAUNCHED.
* WHY TWO CLOCKS: the corrected clock (pose-sync, what G0 reproduces) shifts GT / `v0` / the model-free arms by up to 0.046 m / 0.034 m/s (MEASURED in the CPU smoke), so a
  paired difference against the banked refcv7 dumps (uncorrected clock) is only valid on the standalone clock, where GT and `ha*`/`stop` are BIT-IDENTICAL to refcv7's
  (MEASURED in the smoke: `g_max_abs_diff` 0.0, 6/6 windows; SPEC section 5 control). `build_s6_dump` RAISES on a GT mismatch, hence pass 2 has no baselines.
* EXPECTED WALL-CLOCK once the lock is free (ESTIMATED from refcv7-50,400, MEASURED there: roll s0 7,199 s, roll s1 2,086 s, whole battery 3 h 35 min): G0 ~20-40 min,
  pass 1 ~3.3 h, pass 2 ~1.6 h. First numbers (`ZZR8BATTDONEZZ`) ~4 h after the card is free. Timings: `E:/refcv8_eval_kit/chain/battery_timings_step5000.txt`.
* CHECK: `tail E:/refcv8_eval_kit/chain/battery_step5000.log`; roll progress `E:/refcv8_eval_kit/chain/battery_run_step5000_unc.log` and
  `E:/refcv8_eval_kit/battery/step5000_unc/roll_s*.log`. Kill safely: `kill 45849` (EXIT trap releases the lock), then children by explicit PID only.
* RESULTS WILL BE IN: `raw/step5000/BATTERY_SUMMARY_step5000.json` (+ `battery_unc/`, `battery_corr/` sanitised small artifacts incl. `PAIRED_READ_step5000_unc.{json,md}`);
  work dirs (dumps, panels, raw ids -- never land) `E:/refcv8_eval_kit/battery/step5000_{unc,corr}/`; windows `E:/refcv8_eval_kit/windows/{s2,s6}_windows.json`
  (sha12 twins + `windows_equality.json` in `raw/windows/`: S2 **4,754 / 139**, S6 **3,668 / 139**, set-equal to refcv7-50,400's dumps, panels, s6 dumps and window file).
* UNVERIFIED (CPU smoke = code path only, 2 clips x 3 windows; ZERO GPU): GPU memory of the refcv8 roll on the 8 GB card (refcv7 peak 3.04 GiB) and its wall-clock; the full-size
  analysis/bootstrap time; behaviour on the REAL g0.json of refcv8 (the G0 block was tested on the banked refcv7 one); the perception pass on GPU (`perc_pass_r8.py`: the CPU 4-window smoke RAN to `perc.json` on the refcv8 ckpt md5 a32e3b70... incl. the LAW-frame control PASS, 894 s; the arm label inside perc.json is the r7 name `refcv7` for the refcv8 weights; OFF in the chain unless `R8_PERC=1`);
  `--g0-nonblocking` stamping on a real FAIL; refcv6@38k baselines are not used (absent -> reported missing).
* Wrapper defects found+fixed here (details in README): moved baseline paths, `TIERS` KeyError, roll child without the pose policy, G0 hard stop, corrected-clock/baseline GT mismatch.

---
## (earlier) HANDOFF -- refcv8 step-5,000 G0 (detached job), EvalFlyWheel 2026-10-15

NOT staged in git, NOT committed. The Master Mind lands. Single copy of this package: `C:/Users/Admin/qland/hold_r8batt/stage/.../2026-10-15-refcv8-milestone-battery/`.

## The detached job
* What: `sh code/chain_g0_r8.sh 5000` -- preconditions -> wait_commit (>= 6 GB free commit) -> GPU lock (waits; a foreign lock is never broken) ->
  1-batch PROBE -> 24-seed G0 (A7 rules, M1/M2/M4/M5, A6 numerics arm, wrapper clause) on `E:/refcv8_eval_kit/ckpt/ckpt_5000.pt` vs the in-run row at step 5000 ->
  slim sanitised bank to `raw/step5000/` of this package.
* Started 2026-10-15T21:46:36+0200 by `nohup sh .../code/chain_g0_r8.sh 5000 > E:/refcv8_eval_kit/chain/chain_5000.out`, shell pid 2699 (MSYS pid; Windows tools may not see it).
* At hand-off it was WAITING in `gpu_lock.acquire` behind `aux_navtest_forward` (pid 30716, another session's NavSim job). Expected G0 wall-clock once it runs: ~20-40 min.
* Re-launch: `sh code/chain_g0_r8.sh 5000` again. The lock is O_EXCL, so a second chain only queues behind the first; check for a live `chain_g0_r8` first so you do not queue two.

## How to check (assert on ARTIFACTS, not exit codes)
* Markers in `E:/refcv8_eval_kit/chain/chain_step5000.log`: `ZZR8PRECONDOKZZ` (done) -> `ZZR8LOCKEDZZ` -> `ZZR8PROBEOKZZ` (or `ZZR8PROBEFAILZZ`: read
  `E:/refcv8_eval_kit/battery/step5000/probe/probe.log`) -> `ZZR8G0DONEZZ` (or `ZZR8G0NOJSONZZ`: read `.../step5000/g0.log`) -> `ZZR8BANKEDZZ` -> `ZZR8CHAINENDZZ`.
  Other failure markers: `ZZR8PRECONDFAILZZ`, `ZZR8NOCOMMITZZ`, `ZZR8NOLOCKZZ`.
* The lock: `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (and `devbox_gpu.chain_active` while our chain holds it). The chain's EXIT trap releases both.
* Done = `E:/refcv8_eval_kit/battery/step5000/g0.json` non-empty AND `raw/step5000/G0_SUMMARY_step5000.json` here. Read `GATE_VERDICT`, `M1_detected`
  (M1 NOT detected => G0 is VOID whatever else reads), `verdicts_by_rule`, `n_terms_not_ok_by_class`, `pose_sync_patch.stats` (must be 139/139).
* `KIT.log` (`E:/refcv8_eval_kit/KIT.log`) has one line per step.

## If something fails
* PROBE fail: the usual suspects are a Thor path not in `code/refcv8_eval_remap_devbox.json` (the loader names the file), `OutOfMemory` on the 8 GB card
  (add `--micro 1,1,2,2,2,2` to the `g0_refcv8.py` lines in the chain), or host commit (`wait_commit`).
* G0 FAIL as coded is NOT automatically a loader defect (refcv7's own G0 was FAIL as coded at 5,000 and at 50,400). Classify by `by_class_counts` /
  `terms_not_ok_by_class`; the pose-sync patch is already in, so a remaining mismatch is not that.
* To see what the missing patch would do (diagnostic only): `R8_NO_POSE_SYNC=1` before `g0_refcv8.py`; it is the deliberate-regression arm of the instrument.

## What next (in order; none started)
1. G0 PASS => battery numbers may count. Seeds 0/1 battery: `python code/run_battery_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_5000.pt
   --g0-json E:/refcv8_eval_kit/battery/step5000/g0.json --metrics <metrics copy> --infer-seeds 0,1 --lock-held ...` (UNVERIFIED stages: smoke with
   `--max-clips 2 --max-windows-per-clip 3` first; build the S2 windows file `E:/refcv8_eval_kit/windows/s2_windows.json` with `code/make_s2_windows.py` of the refcv7 battery).
2. The paired four-family read (S2 / S6, both seeds) of refcv8 `os` vs the refcv7-50,400 baseline in
   `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/` (ckpt md5 b418d0fc4a92a6848c246a6a7c50207b; `raw/analysis_s{2,6}_s{0,1}.json`,
   `raw/family_rows.json`); BAR-R8-E1 / E3 / NR per SPEC_REFCV8_DRAFT section 5. Single training seed + one inference seed floor: say so.
3. Master Mind: land the durable pose-sync fix in `refcv7_loader.py` (README headline 1) and add `--r8-v9-labels-eval`, `--pose-sync-sidecar`,
   `--join-defect-masks` to a repo-side dev-box remap (the launch gate's `refcv8_eval_remap_a2_thor.json` is Thor-only).
4. Later milestones: pull `ckpt_<N>.pt` read-only from Thor (`/home/nvidia/refcv8_run/runs/refcv8-r101-s0`, md5 on both sides), refresh a metrics copy,
   `sh code/chain_g0_r8.sh <N>` (the md5 constant in the chain is step-5000 only; the precondition is skipped for other steps).
* Thor is a training box: read-only `ssh -n` / `scp` / `md5sum` only; never python/GPU/eval there.
