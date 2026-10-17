# PREREG_3C — ADDENDUM 5 (Master Mind, 2026-10-17 ~20:50 CEST): the pod was lost mid-run and MIGRATED; the re-roll resumes on a smaller container. Disclosed BEFORE any merged number exists; no bar changed.

**What happened.**
* ~15:47 UTC (17:47 CEST) the pod stopped answering on every port in the middle of the full re-roll (`full/`, 720 shards). No process on it could be
  reached or stopped; nothing was deleted by us.
* The PI migrated the pod to a new machine (RunPod pod migration, the same `/workspace` volume copied across). The new container came up at
  ~17:39 UTC with the volume copy still running: directories not yet copied appear as `nobody`-owned 0700 placeholders, and volume usage grew
  ~0.6-0.8 GB/min while we watched (MEASURED 18:46-18:49 UTC). Nothing is resumed until no placeholder remains at the levels the re-roll reads
  and two polls 5 minutes apart agree.

**What changes, and only this.**
1. **The container is smaller.** MEASURED on the new pod: cgroup memory limit **54,999,998,464 B (55.0 GB)** (was 71.0 GB); CFS quota
   **850,000 / 100,000 = 8.5 CPUs** (was 11.05). `nproc` and `free` still report the host, not the container.
2. **`NPROC=12 -> 9`** for the resumed re-roll, by `run_full.sh`'s own rule (quota + 1 for I/O wait; it measured that throughput does not grow past
   the quota). `run_full.sh` honours `NPROC` from the environment. This changes throughput only: each shard is the same command on the same input
   slice (`NSHARDS=720`, `CHUNK=32` unchanged), and finished shards are skipped while partial shards resume from their own audits, exactly as
   after addendum 4's relaunch.
3. **Wall time** grows by roughly 11.05/8.5 = 1.3x on what remains (ESTIMATED; the CPU-bound rollouts scale with the quota).

**Unchanged:** the frozen `code/` directory (the post-merge orchestrator's step 0 re-hashes its five files against `MANIFEST.json` and refuses on
any difference); every input bank; the merge and its provenance checks; the yield guard; (c'1)-(c'3) and every bar. (c'2) keeps its 12 shards: it
runs only when the pod is otherwise idle, and its per-process footprint is the re-roll's own, which is MEASURED on the resumed run before (c'2)
starts. If (c'2) is OOM-killed, the orchestrator reports it INCOMPLETE (exit 7) and the step resumes its shards; a re-run, not a different test.

**Integrity of what crossed the migration.** We do not assume the copy is faithful. The merge refuses unfinished shards and broken provenance, and the
orchestrator's step (a) re-hashes the input and output banks, re-checks the PREREG sha, and re-reads the yield independently from `audit_all.jsonl`
before anything reaches P6.
