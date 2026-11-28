# RL-S1 kill monitor: why it died, the fix, and the chain's queue rule (report)

Written by the monitor-fix agent (Sonnet) on 2026-10-14. The agent's harness refused the file write, so the Master Mind banked it from its
hand-back. Measurements: `probes/RESULTS.txt` and the scripts beside it. Next-step notes: `HANDOFF.md`.

## Root cause (MEASURED, dev box, real banked inputs from the private HF repos)

`rl_val_eval.load_scores` decodes a WHOLE `.npz` member on every row access, so retained memory grows as **N²** (about 3,584 B x N² per file).

| loader | N (sets) | peak RSS |
|---|---|---|
| old | 300 | 341 MB |
| old | 600 | 1,275 MB |
| old | 1,000 | 3,465 MB |
| old | 1,500 | 7,735 MB |
| old | 3,000 (one whole file) | **21,395 MB** (31.6 GB peak commit) |
| new | 3,000 | **53 MB** |

The pod's R0 monitor loaded several such files beside a training trainer in a 71 GB cgroup. Its rc 137 and empty log are consistent with an OOM
kill (`../2026-10-14-rls1-r0-kill/`); the kernel's OOM line itself is UNVERIFIED (not readable in the container). The post-training replay
survived because nothing else held the memory.

## Fix (`refe/rls1_checks.py`, `refe/rls1_pod.sh`)

* **Light loader:** `rls1_checks.load_scores_light`. The monitor peaks at **95.8 MB** on the full 3,000-set RL-val.
* **Replay unchanged:** the replay over R0's banked inputs is **byte-identical** to the pod's `kill_replay.json` (2,010 B, sha256 `d292d6fd...`, 6.7 s).
* **New markers:** the monitor prints `ZZRLS1_MONITOR_UP` before loading anything and `ZZRLS1_MONITOR_READY` after.
* **Fail closed:** `supervise_arm` stops a trainer whose monitor died or never reached READY, by explicit process-group id. It writes
  `KILLED.json` with code **K-MON** and the chain refuses.
* **Queue:** the chain holds only `chain.queue` while anything is ahead, and takes `chain.lock` only when clear. It hands the lock back if the
  P2 replicate passes its wait inside the settle window. It needs three idle GPU polls before any trainer.
* **Arm names:** an unregistered name in `RLS1_ARMS` is refused at precheck.
* **RL-S1b wiring (`RLS1_ARMS="R0b C1 R1b"`):**
  * R0b is read against C0's dumps and R1b against C1's, both under profile `refe_rls1b`;
  * a K3 or K-MON kill of R0b stops the chain;
  * a missing control is refused;
  * R2b is not registered (it needs a PI decision).

## Tests (agent; the Master Mind's re-run is recorded in the landing commit)

| suite | result |
|---|---|
| `test_rls1_queue.py` (new, about 7.5 min) | 101 |
| `test_rls1_chain.py` | 392 (was 355) |
| `test_flock_rw.py` | 59 |
| `test_s1aux_chain.py` | 361 |
| `test_p2rep_chain.py` | 350 |
| `test_p6_chain.py` | 341 |
| `test_rl_tlane_label.py` | 36 |
| `test_rl_s1b.py` | 18 |
| `test_p2r_chain.py` | 139 |
| `test_sft4r_chain.py` | 20 |
| `test_launch_gate.py --static` | 637 |

These mutations go RED:
* the fail-closed branch removed;
* the READY timeout removed;
* a killed R0b not stopping the chain.

## Open (reported, not fixed)

* **P6 race:** P6's first GPU stage reads only `chain.lock`, not `chain.queue`, so it does not wait for a QUEUED RL-S1. The race closes only
  when RL-S1 takes its lock within one 5 s poll. `p6_pod.sh` was not edited.
* **Arm order:** `rls1b_arms.json` lists C1, R0b, R1b. The chain follows `RLS1_ARMS`; the Master Mind's order is R0b -> C1 -> R1b.
* **Monitor robustness:** a monitor that hangs AFTER READY is not detected, and the monitor has no `oom_score_adj`.
* **Pod behaviour of the queue:** UNVERIFIED.
* **Other callers of the old loader:** `rl_val_eval.load_scores` itself still has the N² access pattern for every caller other than the
  monitor. That is a backlog item.

## Master Mind re-verification (2026-10-14 ~22:15 Berlin, dev box)

| suite | result |
|---|---|
| `test_rls1_chain.py` | 392 |
| `test_flock_rw.py` | 59 |
| `test_launch_gate.py --static` | 637 |
| `test_rl_s1b.py` | 18 |
| `test_rls1_queue.py` alone | **101** |
| `test_rls1_queue.py` beside four other suites | **FAILED** once, at `assert time.time() - t_start >= 3.5, "returned before S1 was over"` |

**Why that one failure is a timing assumption, not a chain bug.** The test starts the chain about 1 s after the S1 holder takes its 4 s lock. Under
load the start slips, so a chain that waited exactly until S1 was over can return in under 3.5 s. The robust invariant, that RL-S1 never held
`chain.lock` while S1 was alive, is sampled separately and is not what failed. The suite is in no G-SUITE. Follow-up: compare against the
holder's release time instead of a fixed 3.5 s.
