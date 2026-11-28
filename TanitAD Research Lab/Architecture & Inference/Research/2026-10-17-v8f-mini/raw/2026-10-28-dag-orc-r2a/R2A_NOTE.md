
## r2's first launch (2026-10-28 03:27–03:40): refused at its first run record; relaunched with a chain-specific aside marker

Written 2026-10-28 ~03:50 by the Master Mind. Nothing of the refused launch is rolled out or read.

**What happened.**
* r2's predecessor check said GO, the dev-box aside ran (r1's pulls to `v8f-dag-orc-oacc_r1_kreach_stop`), and the box-2 aside job printed
  `ASIDE-ALREADY-DONE ... marker D:\tanitad\runs\v8f_mini_full\r2_aside_done.txt` and moved nothing.
* That marker was **DAG-OACC r2's**, written 2026-10-27 20:21 (`logs/v8fdagoacc10271620_r2aside.log`). The DAG-ORC r2 lib copied DAG-OACC r2's
  aside mechanism, marker file name included, and both chains use the same runs root.
* The four gates then passed at `8c24823f` (tokens bound to it). The queue SKIPPED A s0's training, because r1's `summary.json` was still in
  `M2b_s0_dagoaccorc`, and the eval stage reused r1's eval files. The pulled checkpoint was r1's, byte for byte (`6ea4d6baf3f806b3…`, the same
  sha256 as `v8f-dag-orc-oacc_r1_kreach_stop`), with r1's counts (19,161 / 76,898).
* The run-record check refused it: *"the run record carries no `levers.ego_perturb_stream`"*. That field exists only on the fixed trainer, so
  its absence is exactly the signature of a stale run. The chain stopped (`ZZV8FMINI-STOP run ... record M2b_s0_dagoaccorc`).

**The fix.** The marker is now this chain's own name, `r2_aside_done_dagoaccorc.txt`, and it says `DAG-ORC r2 aside done`. The self-test's r1
fixture now carries DAG-OACC r2's `r2_aside_done.txt` as a decoy in the same runs root, and a new mutant restores the shared name; it must be
killed. Lib `e330241a…`, chain `239c9010…` (`gen/build_r2_lib.py`, `gen/make_chain_r2.py`; pre-fix copies in `logs/pre_marker_fix/`).

**Moved aside, never deleted.** Dev box: `v8f-dag-orc-oacc` → `v8f-dag-orc-oacc_r2a_marker_stop`; the r2 log, guard log, gates-seen marker,
current-job file and job-log dir → `*_r2a_marker_stop`. Box 2: the relaunch's aside renames every DAG-ORC item to `*_r1_kreach_stop`. Those
items hold r1's run plus the four gate records the refused launch rewrote in place.

**Cost.** Box 2 03:27–03:40 (0.2 h). The total is now about 3.9 h, against the ~3 h the PI's go was asked as.

**The class.** *Two chains that write the same marker into one directory share one marker.* It is the same class as the composed
perturbation's stream (Amendment 2): an identity (a seed, a file name) copied from a sibling is shared with that sibling. A positive
content check caught it on the first artifact: here, a field that only the fixed build writes.
