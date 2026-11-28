# SLOWWM pilot adopted after the dev-box restart (MM, 2026-11-12 07:01)

**What happened (MEASURED).**
* 01:11:18: the SLOWWM chain (dev-box pid 146542, commit `7296df50`) claimed box 2 and shipped. At 01:11:56 it launched the pilot stage job `PILOT_SLOW_s0`, gid `20261112_011153`, and began polling for its marker.
* 01:29: the dev-box shells logged `fork: retry: Resource temporarily unavailable`.
* 01:32:46: Windows Update restarted the dev box (event 1074, TrustedInstaller.exe; boot 01:33:20). The chain died. It never ran its EXIT trap, so `BOX2_BUSY.txt` kept its mark, and it never saw the job's marker.
* Box 2 did not restart (last boot 2026-11-11 06:59). The job ran to the end: `STAGE_20261112_011153.done` at 01:58:43, `SLOWWM-OK PILOT_SLOW_s0 mean 0.4704 s/step, 12.5 h planned`. Box 2's GPU then sat idle (1,276 MiB, 0 %) until 07:02.

**What was done.** The job was **adopted, not run a second time** (a rerun would have repeated about 47 min of GPU for the same record).
1. `adopt_pilot_run.sh` is `stages/slowwm_run.sh` with three lines changed, and nothing else (`diff` printed in the MM session):
   * the `source` path made absolute;
   * `GID` fixed to `20261112_011153`;
   * the `b2_put` / `b2_launch` line replaced by a no-op.
   Everything after the launch is the chain's own code: wait for the marker (DONE at once), pull the five pilot files with sha256 checks, `slowwm_check.py project` (re-derives the timing from the pulled log), `verify-token` (binds the token to this commit and argv), then the cleanup job and its post-checks.
2. `adopt_and_relaunch.sh`:
   * refuses if `done/pilot.done` or a SLOWWM `STOP.txt` exists;
   * moves the dead chain's busy marker aside only if it is exactly `slowwm chain 146542 2026-11-12 01:11:18` and that pid is gone;
   * claims box 2 with noclobber and runs (1);
   * writes `done/pilot.done` in the chain's own format, plus `adopted-gid=20261112_011153`;
   * releases box 2, then execs the normal `launch.sh`.
3. The chain then logged `skip ship (done)`, `skip basis (done)`, `skip pilot (done)`, `permit: projected 12.5 h <= 72 h`, `start SLOW_s0` (box-2 job launched 07:02:33).

**Evidence in this folder.**
* `adopt.log`: the adopt run.
* `PILOT.verified.json`: pulled and re-derived on the dev box at 07:01:22.
* `CLEANUP.json`: the box-2 temp dirs deleted by literal path, with sizes.
* The two scripts.

**Deletions (logged).** The cleanup job deleted only the run's own temporary gate and pilot out dirs on box 2, listed with bytes in `CLEANUP.json`, after the bank was verified on the dev box. The two dead markers were moved into `C:/Users/Admin/box2_ship/busy_aside/`, never deleted.

**Launcher change.** The relaunches use `C:\Program Files\Git\usr\bin\bash.exe -l`, not the `Git\bin\bash.exe` wrapper, which ^C-kills a WMI batch at exit. The logs now append (`>>`).
