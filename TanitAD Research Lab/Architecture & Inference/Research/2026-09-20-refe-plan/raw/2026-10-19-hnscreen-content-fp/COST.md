# COST of the content fingerprint (HNSCREEN-CACHE-CONTENT-FP) -- MEASURED on this box, 2026-10-19

Evidence class: **MEASURED** unless marked. Raw: `logs/bench_cost.json`, `logs/bench_cold_d.txt`; scripts `scripts/bench_cost.py`, `scripts/bench_cold.py`. Box: 24 logical CPUs, 31 GB RAM,
C: = 1 TB NVMe (SSDPR-PX600), D: = 4 TB **USB** SSD (SanDisk Extreme Pro). Both measured while other agents' jobs were running (the box was busy), so every number is a conditioned sample, not a ceiling.

## What is read, and how often
A content fingerprint **re-reads every byte** of the cache. There is no way to bind bytes without reading them, so this is the price of the check, not an inefficiency.

| entry point | full reads of the cache per call | pinned by |
|---|---|---|
| `ctx_cache.finalize` / `ctx_cache.py seal` (the build, once) | 1 | `B_finalize_leaves_a_sealed_cache...` |
| `hn_fingerprints.py` (the fill, once) | 1 | `B_hn_fingerprints_py...` |
| `launch_gate.py refusals` | 1 | `G_cost_one_full_read_per_check_refusals_1_verify_1_run_2_finalize_2` |
| `launch_gate.py run` (`run_checks`: refusals + data manifest) | 2 | same |
| `launch_gate.finalize` (the mint: refusals + data manifest) | 2 | same |
| `launch_gate.py verify` (the supervisor, before each trainer start) | 1 | same |

Per seed: mint = `run` 2 + `finalize` 2, then `verify` 1 = **5 reads**; three seeds = 15; plus the seal and the fill = **17 reads** of the cache for the whole screen.

## Throughput (this box)
| quantity | value |
|---|---|
| sha256, in memory, 1 thread | **1.29 GB/s** |
| `content_fingerprint` of a 160 MiB cache, **page cache warm**, threads 1 / 2 / 4 / 8 / 16 | 1.16 / 1.90 / 2.68 / 3.49 / 3.61 GB/s (default is 4 threads, env `TANITAD_GATE_HASH_THREADS`) |
| seal of the same 160 MiB (default threads) | 0.096 s |
| the old listing fingerprint of the same directory | 1.2 ms |
| **D: (USB SSD), cold, sequential read + sha256**, two ~1.3 GB checkpoints | 29.5 s and 30.2 s = **44 and 43 MB/s** (first read) |
| the same file read again, page-cache warm | 0.93 s = 1.4 GB/s |
| C: NVMe, cold | **UNMEASURED** (no large cold file found on C:; do not infer) |

## The screen's cache and what that costs
Size: 19,000 scenes x 3.75 MiB = **69.6 GiB** + fan files (< 1 %) -- **INHERITED** (`PREREG_P6HN_SCREEN.md` section 3 / S2), the cache is not built yet.

| if the cache is read at | one full read | 5 reads (one seed's mint + verify) | all 17 reads |
|---|---|---|---|
| hash-bound, 3.6 GB/s (8+ threads, fast local disk) | 21 s | 1.7 min | 5.9 min |
| default 4 threads, 2.68 GB/s | 28 s | 2.3 min | 7.9 min |
| **D: as measured today, 44 MB/s** | **28 min** | **2.4 h** | **8 h** |

**Reading it.** CPU is never the limit (1.3 GB/s per thread). The disk is. A 70 GiB cache cannot live in the page cache of a 31 GB box, so every pass is a cold read at the drive's rate. If the cache is built on D:
(the USB SSD, shared with other jobs) the gate adds hours; on a local NVMe it adds minutes (UNMEASURED there). **Put the cache on the fast local disk, or measure the cache drive before the screen is queued.**

## Levers NOT pulled (deliberately), for the Master Mind
1. **Fewer reads per command.** `run` and `finalize` each hash the cache twice (the refusal rule and the data manifest compute the same fingerprint). A per-process memo keyed on (path, size, mtime) would make it 1 + 1 and the per-seed total 3 instead of 5. Not done: a memo
   trusts mtime, which is exactly the kind of cached claim this check exists to avoid, and it would need plumbing through `profile_refusals`' signature (many callers). Cheap to add if the drive turns out to be slow.
2. **Hash only the changed part.** Not possible without trusting something other than the bytes.
3. The check is never skipped: there is no env var, flag or profile key that disables it (a default that disables the check is the class this fix exists to close).
