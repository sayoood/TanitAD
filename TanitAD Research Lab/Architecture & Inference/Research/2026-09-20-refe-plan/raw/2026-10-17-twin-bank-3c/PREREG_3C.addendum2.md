# ADDENDUM 2 to PREREG_3C.md -- written 2026-10-17 by the build agent, AFTER the identity controls and BEFORE any production number (disclosed)

Scope: the PREREG's **Cost** line only ("~20 s per 10 s rollout per process ... 44,409 x 20 s / 90 processes ~ 2.7 h of pod CPU"). It was labelled ESTIMATED; no bar or
committed outcome depends on it. `PREREG_3C.md` is not edited (sha256 `e8365f34...`).

MEASURED 2026-10-17 on the pod (raw: `raw/pod_cpu_quota.txt`, `results/timing/*.json`):
* The container's CPU quota is **11.05 CPUs** (cgroup `cfs_quota_us` 1,105,000 / `cfs_period_us` 100,000), although `nproc` reports 96. A "90-process" run gets no more
  throughput than ~11-12 processes.
* Same 74-twin control: probe at 10 concurrent processes 13.4 s mean per rollout; 32 concurrent processes 49.2 s (median ratio 3.18x).
* Production-shaped sample (the real shard plan, 11 and 20 evenly spaced shards of 720, 1,040 twins in all): **1.08 twins/s at 11 processes, 1.10 at 20** (20 processes throttled
  2,973 periods). Full re-roll = 44,409 / 1.1 per s = **~11.2 h**, i.e. ~4x the PREREG line. The twin-parts build is a separate ESTIMATED 2-3.5 h.
* The brief's ~90 processes were therefore not used; `run_full.sh` defaults to 12.
No keep rule, horizon, goal horizon, control or acceptance bar changes. Whether ~13-15 h of pod CPU is acceptable before the safety-objective A40 arm is a Master Mind / PI decision.
