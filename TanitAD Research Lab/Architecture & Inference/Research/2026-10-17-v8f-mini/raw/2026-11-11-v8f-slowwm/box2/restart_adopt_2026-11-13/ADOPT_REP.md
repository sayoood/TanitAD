# SLOWWM SLOW_rep adopted after the driver chain died (MM, 2026-11-13 00:05)

**What happened (MEASURED).**
* 2026-11-12 20:27:06: the SLOWWM chain (dev-box msys pid 834, commit `7296df50`) started SLOW_rep and launched its box-2 stage job, gid `20261112_202710`. It then polled for the job's marker.
* About 20:34: the chain died. Its console log (`chain_console.log`, last write 20:34) ends in `^C`. No STOP.txt was written and its EXIT trap did not run, so `BOX2_BUSY.txt` kept `slowwm chain 834 2026-11-12 07:02:31`. The dev box did not restart (last boot 2026-11-12 01:33:20).
* Cause: not established. HYPOTHESIS: an interrupt from a detached launch made in the MM session at about 20:34-20:42. The stage-1 waiter's first launch, a plain WMI launch, died after its first line at that time. The relaunch used CREATE_NEW_PROCESS_GROUP and survived.
* Box 2 ran the job to the end: `STAGE_20261112_202710.done`, `SLOWWM-OK SLOW_rep`, `SLOWWM-JOB-END rc=0` at 23:46:53. Its GPU then sat idle.
* Found by the MM at 00:02 on 2026-11-13: no chain process, no new state file since 20:27. The stage-1 waiter would have waited on the dead marker until its 14:00 deadline.

**What was done.** The job was **adopted, not run a second time** (a rerun would have repeated about 3.3 h of GPU for the same record).
1. `adopt_rep_run.sh` (sha256 `9e61ac68...572b`) is `stages/slowwm_run.sh` with three lines changed, and nothing else. The changes match the 2026-11-12 pilot adopt:
   * the `source` path made absolute;
   * `GID` fixed to `20261112_202710`;
   * the `b2_put` / `b2_launch` line replaced by a no-op.
2. `adopt_rep_and_relaunch.sh` (sha256 `c4d3cf62...1acb`):
   * refuses if `done/SLOW_rep.done`, a SLOWWM STOP.txt or DONE.txt exists, if any of the four earlier runs lacks its done marker, or if a `slowwm_chain` process runs;
   * moves the dead chain's busy marker aside only if it is exactly that chain's mark;
   * claims box 2 with noclobber and runs (1): stage marker DONE at once, pull with sha256 checks, `verify-token`, bank manifest against RUN.done's evidence, cleanup job;
   * writes `done/SLOW_rep.done` with `adopted-gid=20261112_202710`;
   * releases box 2, then execs the normal `launch.sh`.
3. The chain then logged `skip` for ship, basis, pilot and all five runs, `permit: projected 12.5 h <= 72 h`, `all runs banked`, and wrote `ZZ SLOWWM-DONE 2026-11-13 00:05:33`.
4. The STOPPERC stage-1 waiter saw SLOWWM's DONE and a free box 2, and launched stage 1 at 00:06:32.

**Evidence in this folder.**
* `adopt_rep.log`: the adopt run.
* `chain_console_rep.log`: the relaunch.
* `chain_console_died_2034.log`: the dead chain's console log, as found.
* The two scripts and `adopt_rep.cmd`.

The bank `C:/Users/Admin/t2v8f_pulls/slowwm_SLOW_rep/` holds BANK.done and BANK_MANIFEST.json, and its ckpt is 46,847,094 B, like the other four arms. Its `CLEANUP.json` lists no deleted temp (the gate dir did not exist; the run script checks that it is absent).

**Deletions.** None by the MM. The dead marker was moved into `C:/Users/Admin/box2_ship/busy_aside/`, not deleted.

**Lesson (also in the MM's watches from now on).** A driver chain can die without a sentinel while its box-2 job runs on. A watch that waits only for STOP/DONE files never fires. The stage-1 watch now also checks that a `stopperc_chain` process exists every 5 min, and wakes the MM after two misses in a row.
