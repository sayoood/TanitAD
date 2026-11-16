# P5-C label run: TB-0 FAIL on the control pass (R-A only). MM, 2026-11-16

Evidence class: **MEASURED**. Artifacts are in this directory. The runner is `a31/a1/p5c_pod_label.sh` (blob `727e72d4`); join `4670f88a`; A3.1 `cda1567b`.

## What failed
- **Verdict:** `ZZP5CL_FAIL control_gates`. TB-0 FAIL, TB-1 PASS, on both runs.
- **Pre-registered consequence (A3 §4):** no C1 launch; fix and rebuild. **C0 proceeds.**
- **TB-0 = R-B (P5-R EVAL labels) AND R-A (Tc20s warm start == dev-box Tc20s build within 1e-9).**
  - **R-B passes every criterion** on both ranks: sim8 identical 1,200/1,200 per rank; rank-0 row max-diff median 8.0e-6, p99 1.2e-3 (bar 1e-3 / 5e-3); TB-6 ratios within 5e-7 of the dev box; sim8-fail rate 0.25 % on both.
  - **R-A misses:** max abs diff **1.4901395672950457e-09** against tol 1e-9, over 2,400 rows / 318 logs, with 0 traj-sha mismatches.

## Distribution (`ra_probe.py`, pod, run 1)
| rows of 2,400 with max \|Δ\| over | count |
|---|---|
| 0 | 2,232 |
| 1e-12 | 61 |
| 1e-10 | 22 |
| 5e-10 | 8 |
| **1e-9** | **2, both rank 1** (1.49e-9 on x, 1.23e-9 on y; the labels' max \|value\| is about 34 m) |

- Median 1.8e-15; p99 9.6e-11; p99.9 9.3e-10.
- **Rank-0 maximum: 9.36e-10, inside the bar.**
- The worst relative error is about 4e-11. Yaw max is 6e-11.

## Hypothesis tested and REFUTED: the OpenBLAS kernel
- **Why it was the suspect.** The dev box (i9-12900F) selects OpenBLAS **Haswell**; the pod (EPYC 7452) selects **Zen**. Both run numpy 1.26.4 and OpenBLAS 0.3.23.dev DYNAMIC_ARCH, with the same SIMD set (AVX2 / FMA3, no AVX-512).
- **Run 2** used `OPENBLAS_CORETYPE=Haswell` (verified `Core: Haswell`; the runner passes its environment to the shards unchanged). It reproduced run 1's R-A value **bit-for-bit**.
- ⇒ BLAS is not the source. The residual is deterministic per platform: [H] libm or compiler differences between Linux/glibc and Windows/UCRT, with no clean override.

## Status
- **C1 is BLOCKED on a PI ruling.** The MM does not widen or rescope a frozen bar after seeing the data.
- **C0 is unaffected:** its profile carries only the base pin.
- Options, all post-data and to be disclosed as such:
  1. **Rescope R-A to rank-0 rows**, the only rows whose Tc20s/P5-R labels enter C1 under A3 §3.3's forced `rank0` policy. Rank-0 max 9.36e-10 ⇒ PASS. **MM recommendation.**
  2. Calibrate R-A from the measured cross-platform floor, e.g. 1e-8 with this distribution disclosed.
  3. Keep R-A as written. C1 then needs the label build on the dev-box platform, which needs the ~170 GB of navtrain DBs locally; not feasible today.
