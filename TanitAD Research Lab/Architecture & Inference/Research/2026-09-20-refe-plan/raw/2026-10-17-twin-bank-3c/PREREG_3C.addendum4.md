# PREREG_3C — ADDENDUM 4 (Master Mind, 2026-10-17 ~12:10 CEST): the K=2 parts build was OOM-killed, then the volume hit its quota. Disclosed; no bar changed.

**What happened (MEASURED on the pod, 2026-10-17):**
* The continuation (`build_parts_continue.sh`, addendum 3) passed the K=1 distinct check (37/40), then started the K=2 build at the default 12 workers.
  The build stopped with no Python traceback: the log's last line is the route-rank patch notice
  (kept as `full/parts_build_k2.OOM_KILLED_12workers.out`; the continuation's own log as `full/build_parts_continue.OOM_KILLED.out`).
* The container's cgroup shows the cause. `memory.limit_in_bytes` = **70,999,998,464** (71.0 GB), and `memory.max_usage_in_bytes` reached
  **exactly that limit**. `memory.oom_control` reads **oom_kill 2**. `free` on the same pod reports 503 GB (the host), so `free` cannot see this cap
  (TRAPS entry added the same day).
* **6,807 of 8,090 K=2 parts** were on disk at the kill. Part writes are atomic (temp file + `os.replace`), and **0 `.tmp` files** remained, so no
  partial part exists. `build_metric_cache.py build` skips existing parts, so the rebuild resumes rather than repeats.
* The relaunch then failed at once: `Disk quota exceeded` while opening its log. The 373 GB volume was full. The K=1 + K=2 parts take 15.7 GB
  (`du`, apparent size), which matches the ~19 GiB headroom the earlier write test had measured.

**Fix (no bar, no code in the frozen `code/` dir, no input changed):**
1. Space, deletions logged in pod `/dev/shm/pod_deletions.log`:
   * `hf_dl/queue/onpolicy_queue.tar.zst` (1.82 GB) was a **verified copy**: its local sha256 `c62a121c…` equals the LFS sha256 on
     `Sayood/tanitad-refe-navtrain-banks` main.
   * `metric_cache_twins_rejected` (18.2 GB) held the PRE-FIX twin parts, unusable (twin route == logged on 74/74; addendum 1). It was deleted with
     the PI's approval in chat, 2026-10-17.
   * The volume accepted writes again on the first try after the delete.
2. Memory: the continuation is relaunched with **`WORKERS=6`** (the script honours the env var). This is ESTIMATED at ~35 GB peak from the 12-worker
   peak at the 71 GB cap. The K=3 build uses the same setting.

**Unchanged:** the K=2/K=3 parts are built by the same command on the same inputs. Only the worker count changes, which changes throughput, not the
per-part computation (each part is built independently). The D7 distinct check (`--n 40 --min-frac 0.9`), the yield guard, and (c'1)-(c'3) are untouched.
