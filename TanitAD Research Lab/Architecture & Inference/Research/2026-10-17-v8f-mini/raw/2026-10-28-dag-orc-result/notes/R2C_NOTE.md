
## r2's A s1 refused at its run record (2026-10-28 06:04): the box-2 queue was replaced mid-run by a self-test; A s1 re-run with the code pinned

Written 2026-10-28 ~06:15 by the Master Mind. A s1's refused artifacts are never rolled out or read.

**What happened.**
* 05:56:16: a dry-run self-test, run by the agent building the LAUNCH-INTENT box-2 chain, reached the REAL box 2. Git bash put `/usr/bin` ahead
  of its stub `ssh` / `scp` on PATH, so the stubs were never called. Its code-refresh step extracted a scratch tar over `D:\tanitad\repo`.
  Of 65 paths, 62 were byte-identical to r2's. Three were not: `stack/scripts/v8f_mini_train.py` (`0d708b3c…`, the unlanded launch-intent
  trainer), `stack/tanitad/models/v8f_mini.py` (`b337ccec…`), and the box-2 queue (`b921d0af…`, r2's queue plus a `:launchintentarm` stage).
  BOX2_BUSY was not taken.
* A s1 was training (launched 05:20; its modules were imported at 05:20, from r2's code). It finished at about 06:01 (3,000 steps, 2,262 s,
  `train.log`). cmd.exe then resumed the queue `.cmd` by byte offset in the replaced file and started the trainer again. That run was the
  launch-intent build: `[resume] step 3000 … DONE 3.2 s`. It rewrote `summary.json`, `selfcheck.json` and the checkpoint. Its records carry
  `launch_*` keys and `init_ckpt.applied = false, skipped = "resume"`.
* The A s1 T0 eval (06:01) imported the replaced model file.
* The run-record check refused A s1: *"summary.json init_ckpt not applied / read back: False / None"*. The r2 chain, the T1-kin chain and
  the SPEEDSUB waiter all stopped as designed.

**What was done.**
* 06:04:09: r2's own tar (`repo_v8fdagoaccorc_r2_tip.tar`) was re-extracted on box 2. The three files read r2's sha256 again: trainer
  `1c888280…`, model `8ec5e190…`, queue `d8aa4f83…`.
* A s1's box-2 items were renamed `*_r2b_queue_edit_stop`: run dir, gate dir + log, three argv pins, four eval files. The stray
  `TIP_V8FLAUNCHINTENT.txt` became `*_selftest_incident_3bb7c5c2`. The dev-box pull and r2's stopped logs went to `*_r2b_queue_edit_stop`.
  Nothing was deleted.
* The remote tip had moved to `94a47f4b` (SPEEDSUB; no trainer, model, data or queue change). The chain refreshes box 2 from the remote tip
  of its branch. A s0 and REP s0 were gated at `8c24823f`, and the reader requires one commit for all four runs. So the relaunch reads the
  branch `agent/dagorc-r2-pin`, created at `8c24823f`, through `OAF_BR` (`run_chain_v8f_dagoaccorc_r2c_pin.cmd`). The chain code is
  unchanged.
* The first relaunch stopped on the E: disk floor before touching anything. Two aside dirs and two dead lander holds (414 MB) moved to F:.
  r2c relaunched at 06:09: box 2 code = `8c24823f`, its own aside marker recognised, A s0 and REP s0 skipped by the queue, A s1 and REP s1
  from their gates.

**Unaffected.** A s0 and REP s0 trained and evaluated before 05:56. Their run records, K-REACH and T1-kin rollouts all PASS, and the
queue skips them on the relaunch.

**Cost.** Box 2, 05:56–06:09 lost, plus A s1 trained twice (0.7 h). The box-2 total is now about 4.8 h, against the ~3 h the PI's go was
asked as.

**The classes.**
1. *A batch file edited while cmd.exe runs it executes the new bytes from the old offset.* Never rewrite a `.cmd` that a live job may be
   executing. Ship a new file name, and switch only between jobs.
2. *A stub that relies on PATH order is not a stub on Git bash.* `/usr/bin` wins. A self-test that must not reach a host fails closed:
   check that the stub resolves first and that the real host alias does not resolve, before any step runs.
