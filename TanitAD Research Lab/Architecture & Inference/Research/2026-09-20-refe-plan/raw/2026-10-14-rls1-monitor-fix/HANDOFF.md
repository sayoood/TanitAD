# HANDOFF (rls1 monitor fix) 2026-10-14 -- FINAL state. All 6 tasks + the RL-S1b wiring are DONE; nothing is staged or committed.

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. The Master Mind lands the files below through the mirror (project override: no `git add`/`commit`).
`raw/2026-10-14-rls1-monitor-fix/REPORT.md` could NOT be written (the harness blocked a subagent writing a report file); its content went to the hand-back message instead.

## Files (single-location on the dev box E: until landed)
MODIFIED `refe/rls1_checks.py`, `refe/rls1_pod.sh`, `refe/test_rls1_chain.py`, `eval/RLS1_POD_RUNBOOK.md`; NEW `refe/test_rls1_queue.py`; NEW `raw/2026-10-14-rls1-monitor-fix/{HANDOFF.md,probes/*}`.
Regenerate `SHIP_MANIFEST.json` when shipping (code files changed). Python for the tests: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (has numpy; the system pythons do not). Backups of the originals: session scratchpad `*.orig`, `*.before_b`.

## What was established (MEASURED unless stated)
* Root cause: `rl_val_eval.load_scores` retains N x N (about 3,584 B x N^2 per file): old loader at N = 3,000 = 21.2 GB RSS / 31.6 GB peak commit per file; new = 53 MB. New checker peak 95.8 MB on the full RL-val; replay over R0's banked inputs is BYTE-IDENTICAL to the pod's `kill_replay.json` (2,010 B, sha256 `d292d6fd6ae2a782`). The pod's rc 137 / oom_kill / 0-byte log are INHERITED (the brief's README does not exist in the repo); the kernel OOM line is UNVERIFIED.
* Chain: queue (`ahead`/`queue_wait`/`chain_past`/`lock_held`, fd 199 chain.queue guard, chain.lock fd 200 yielded while anything is ahead), `gpu_idle_wait` before every trainer, `supervise_arm` K-MON (stop by explicit pgid, KILLED.json code K-MON), RL-S1b arms R0b/R1b registered (profile `refe_rls1b`), `RLS1_STOP_AFTER=queue`.
* P6 finding (not edited): P6's first GPU stage does NOT wait for a QUEUED RL-S1 (it reads only `chain.lock`); only RL-S1 taking the lock within one 5 s poll closes the race.

## Tests (last full run)
see the hand-back message / `eval/RLS1_POD_RUNBOOK.md` header: `test_rls1_chain.py` 392, `test_rls1_queue.py` 101 (about 7.5 min), `test_flock_rw.py` 59.

## Open items for the Master Mind
1. P6 guard (above): should `p6_pod.sh` `wait_clear` also read `chain.queue`?
2. JSON order `[C1, R0b, R1b]` vs the coordinator's R0b -> C1 -> R1b: chain accepts both; the operator sets `RLS1_ARMS`.
3. Residual: a hung-but-alive monitor after READY is not detected; no `oom_score_adj` on the monitor; pod behaviour of the queue is UNVERIFIED.
