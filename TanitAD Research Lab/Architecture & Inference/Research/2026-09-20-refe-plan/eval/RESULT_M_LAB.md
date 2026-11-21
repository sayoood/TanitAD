# RESULT M-LAB: the online PDM labeller costs 0.34–0.83 CPU-s per 64-proposal set — a 24-vCPU pod has ≥ 2× margin

> **Purpose.** Condition C4 of `eval/P6_GO_SIGNAL.md`. It decides P6-C, the pod's vCPU count, using the sizing formula in `SPEC_P6_PAPER_FAITHFUL.md` §3.
>
> **Run.** 2026-10-11, 19:08–20:15 Berlin, on the dev box: i9-12900F, 24 logical CPUs, P- and E-cores with SMT.
>
> **Evidence class.** Every number is **MEASURED** unless marked.
> - c is **process CPU time** (user+sys over the whole labelling loop, divided by sets). It is never wall time.
> - Raw data: `raw/2026-10-11-m-lab/`.
> - The all-arms table (`all_arms_table.txt`) is computed by the Master Mind directly from each arm's `runs/<arm>/worker*.json`.
> - The agent's draft (`RESULT_M_LAB.draft.md`) aggregated only 3 of the 21 arms, so its §1 and §4 are incomplete. This file supersedes it.

## Verdict

**C4 is GREEN.** At every measured c, P6's labelling fits a ≥ 24-vCPU pod with a margin of at least 2×. The regime the spec marks infeasible (c ≈ 18) is 22–53× away.

## 1 · c, the cost per 64-proposal set (all 21 arms, 0 errors)

| design | condition | CPU-s/set | arms |
|---|---|---|---|
| **cached parts** (D9 parts on disk, loaded per set, then `executed_plans8` + `score_plans`): **P6's design** | pinned to one P-core, low background load (≈ 8 of 24 logical CPUs busy) | **0.342 / 0.356 / 0.361 / 0.363** | `cached_r4_n1`, `cached_r5_n1`, `cached_r6_n1`, `cached_r1p_n1` |
| cached parts | unpinned, background load 10 / 17 / 21 logical CPUs | 0.472 / 0.668 / 0.833 | `cached_r1_n1`, `cached_r2_n1`, `cached_r3_n1` |
| cached parts, parts kept in RAM | unpinned | 0.408 | `cached_r2_retain_n1` |
| **build + label** (parts rebuilt from the log for every set: the fallback if parts are not pre-built) | pinned, low load | 0.804 | `build_r1p_n1` |
| build + label | unpinned, low load / heavy load | 0.84–0.87 / **1.968** | `build_r2_n1`, `build_r3_n1` / `build_r1_n1` |
| single-set build | unpinned | 0.98–1.04 | `single_r{1,2,3}_n1` |

**Load sensitivity.** On this hybrid CPU, c roughly doubles when the box is saturated: SMT sharing, and placement on E-cores. That explains the spread; it is not noise.

**Correctness control.** In every cached arm the labels equal the NAVSIM census to 1e-4 on **all** proposals: 15,168 of 15,168 per run. The build arms agree on 95–98.5 %, which is expected by construction: a rebuilt cache is not NAVSIM's file (`eval/validate_pdm_targets.py`). A deliberate wrong-proposals control must not match, and it does not.

## 2 · Scaling with workers (cached design, same tokens)

| N workers | CPU-s/set per worker | sets/s (wall, all workers) |
|---|---|---|
| 1 | 0.472 | 2.00 |
| 2 | 0.549 | 3.18 |
| 4 | 0.680 | 5.67 |
| 8 | 0.714 | **9.62** |

P6 needs **4.49 sets/s**. Four dev-box workers already exceed that, and eight give 2.1×. Per-worker cost rises with N, because workers share cores and SMT siblings, but throughput keeps scaling.

## 3 · Memory and disk

- **RSS per cached-parts worker** at steady state is about 250 MB; peak working set 260 MB. No city maps are loaded.
- **Labeller RAM** for 12 workers × 0.31 GB (RSS plus a small LRU) × 1.25 ≈ **4.7 GB**. Trainer and loader RAM are outside M-LAB (UNVERIFIED).
- **Parts on disk:** 262 kB per scene (lzma) ⇒ about **27 GB** for 103,037 scenes. This is ESTIMATED from the navtest mean size.

## 4 · Sizing (spec formula: CPUs = 4.49 × c; workers = 1.5 × that; pod vCPU = workers + 2 loader + 2 trainer/IPC)

| basis | c | sustained CPUs | workers | **pod vCPU** |
|---|---|---|---|---|
| cached, pinned, low load (typical) | 0.36 | 1.6 | 3 | **7** |
| cached, worst measured (saturated box) | 0.833 | 3.7 | 6 | **10** |
| cached, worst × 2 for a slower pod vCPU (ESTIMATED; ratio NOT measured) | 1.67 | 7.5 | 12 | **16** |
| build + label, worst measured (fallback design) | 1.968 | 8.8 | 14 | **18** |
| *spec row, INHERITED 1.78* | 1.78 | 8.0 | 12 | 16 |

## 5 · Recommendation (P6-C)

1. **Pod: ≥ 24 vCPU, ≥ 64 GB RAM, ≥ 400 GB disk.**
   - **vCPU.** 24 vCPU leaves 20 for labellers, which sustains c up to about **2.97 CPU-s/set**. That is ≥ 3.5× the worst cached measurement and 1.5× the worst build measurement. It absorbs a pod core up to ~3.5× slower than this dev box's P-core under load.
   - **RAM.** The labellers need about 5 GB. The rest is for the trainer and loader, whose host RAM is UNVERIFIED (see G-LIVE-MEM).
   - **Disk.** It covers the navtrain DBs and pixels (164–449 GB, INHERITED; P2's runbook asks ≥ 400 GB proven with `dd`) plus about 27 GB of parts.
2. **Pre-build the parts once, before step 0.** That keeps P6 on the cached design. 103,037 scenes × ~0.85 CPU-s ≈ 24 CPU-h, about 1–1.5 h wall on 20 workers (ESTIMATED).
3. **In-situ check during the first 100 steps:** labeller queue depth > 0 and GPU stall ≈ 0. This is G-LIVE-LAB in SPEC_P6 §5; it is not built yet.

## 6 · Limits (stated, not hidden)

- **The dev-box P-core is a lower bound** for a pod vCPU. The ratio is NOT measured; §4 brackets it at 2×.
- **Data are the navtest metric cache.** No navtrain cache exists on the dev box. The proposals are the banked `navtest_final` set of `model_final.pt`, not u400's.
- **Not included:** IPC between trainer and worker, queueing, and the trainer's own CPU.
- **Disturbance during some runs.** The agent's stray `find /` (killed ~19:20) ran during `cached_r1`–`r3` and part of `build_r1`. That inflates exactly the three unpinned arms quoted as worst case, so the worst case is conservative.
