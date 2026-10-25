# PROBEGATE handoff (refcv9 L2-P) -- written 2026-10-25 ~20:10 (Europe/Berlin)

State at hand-off: PREREG sealed (sha256 `20a643303207f36dc3c90b2db0f0e59c6d3d8a251ca307c674eb9241dd3503b3`, 2026-10-25T20:01:52+0100); probe scores + labels built and independently verified; the detached chain is WAITING for the dev-box GPU lock (held by `refcv8-p4-gtmap`; `refcv9-l2b-locus` and `refcv9-l1c-ztac` queued; the lock is non-FIFO).

## Read the result
* `result/VERDICT.md` -- mechanical verdict (PASS / FAIL(clauses) / INVALID / PENDING), the four rates, the ceiling share, the G rows, the estimator statement and the pre-registered NEXT LEVER.
* `result/PG_RESULT.md` / `PG_RESULT.json` -- all cells, controls K0-K8, descriptive tables.
* Done marker: the opaque token `ZZPGENDZZ` in `chain.log` (written by the EXIT trap on every path; its line carries `rc=`, `stage=` and the first line of the verdict). If the chain dies early the VERDICT.md says `NO VERDICT: ... stage <name>`.

## Chain facts
* Script `chain_pg.sh` (sha-pinned). Detached by `launch_detached.py` (the first launch, via WMI, was killed when its tool call ended: `chain_attempt1_killed.log`; it never reached the lock). Log `chain.log`; stdout `logs/chain.nohup`; lock `logs/lock.log`, `logs/lock.json`; rolls `logs/roll_probegate.log`, `logs/roll_randgate.log`; score `logs/score.log`.
* Rolls write to `F:/refcv9_probegate/` (`trace_{probegate,randgate}_s0.npz`, `roll_*_s0.json`, dumps deleted after a final trace); small traces are copied to `bank/`.
* Cap: RANDGATE is rolled only if the lock-held time after PROBEGATE x 2.05 <= 5,760 s; otherwise `ZZPGCAPZZ` and P3 reads PENDING.
* Stop it: kill the chain bash by explicit PID (the EXIT trap releases the lock and kills its own child). Never `rm` the lock file by hand while the chain holds it.

## Deliverable manifest (all under C:/Users/Admin/refcv8_pf/refcv9_l2/probegate/ unless noted; nothing staged in git, nothing on G:)
PREREG_PROBEGATE.md/.sha256, CODE_PINS.txt, pg_policy.py, pg_policy_test.py, pg_labels.py, pg_verify_labels.py, pg_score.py, chain_pg.sh, launch_detached.py, probe_scores_s0.npz, labels_probegate_s0.npz, labels_randgate_s0.npz, labels.sha256, LABELS_CHECK.json, LABELS_VERIFY.json, chain.log, chain_attempt1_killed.log, logs/, HANDOFF.md, explore1-4.py (+ .out; the 0-GPU reads quoted in the prereg section 0). Produced by the chain: result/, bank/, and F:/refcv9_probegate/.
Integration is the Master Mind's: copy `probegate/` (without explore*/npz dumps if desired) into `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-25-refcv9-l2/probegate/` and stage it; register the verdict in GOALS_AND_CLAIMS.md the same turn.
