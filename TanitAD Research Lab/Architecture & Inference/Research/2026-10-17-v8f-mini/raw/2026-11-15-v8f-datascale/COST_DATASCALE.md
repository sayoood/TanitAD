# COST_DATASCALE (pilot-derived)

MEASURED on the 200-clip pilot (`PILOT_REPORT.json`): stage 3856.0 clips/h, build 1816.0 clips/h (13 workers), encode 1354.0 clips/h; failure rate 0.0%.
Per clip (MB): mp4 12.76, egomotion 0.416, obstacle 0.508, v2ep (transient) 37.41, cache (kept) 20.88.

| N clips | pod-h sequential (RTX 4000 Ada) | pod-h pipelined (slowest stage) | pod-h pipelined, L40 encode (ESTIMATED) | transient disk if v2ep kept (GB) | kept cache = HF private (GB) | obstacle coverage (ESTIMATED from pool) |
|---|---|---|---|---|---|---|
| 10000 | 15.5 | 7.4 | 5.5 | 374 | 209 | 97.98 % (131,219 of 133,929 in the pool, MEASURED metadata) |
| 45000 | 69.7 | 33.2 | 24.8 | 1683 | 940 | 97.98 % (131,219 of 133,929 in the pool, MEASURED metadata) |
| 133929 (admissible ceiling under the frozen scope) | 207.4 | 98.9 | 73.7 | 5010 | 2796 | 97.98 % (131,219 of 133,929 in the pool, MEASURED metadata) |
| 306151 (NOT admissible under the frozen scope; extrapolation) | 474.1 | 226.1 | 168.6 | 11453 | 6392 | 97.98 % (131,219 of 133,929 in the pool, MEASURED metadata) |

Evidence: rates and sizes MEASURED (pilot, one run, one seed of the selection; no replicate, so the rate carries unknown run-to-run noise). Extrapolation to N beyond 200 is ESTIMATED: linear in N,
network and CPU contention at 8 threads / 13 workers assumed unchanged. The earlier corpus build measured 1,703 clips/h build and 2.76 s/clip encode (`/root/logs/v2ep_run.log`, `/root/logs/cache_pod.log`): INHERITED cross-check.
