
## Amendment 1 (`A1-2026-10-27`): the outcome table had four unplaced cells; written before any DAGGER2 row exists

**When.** Written 2026-10-27, after the four box-2 arms were trained and before any DAGGER2 row exists. No T1-kin rollout of DAG2 or DAG2-0
has run, so R-CRAWL, R-MOVE, R-TLAT, R-TLON and R-ADE have never been computed.
* The box-2 pulls carry each arm's T0 eval file (`eval_M2b-dag2*_s*.json`). None was opened to write this. The amendment moves no guard,
  so R-T0 reads exactly as written above.
* What has been read: the reader's preflight read the iteration-2 banks' COVERAGE (s0 0.773, s1 0.764) and the gate and run records. COVERAGE
  decides BLOCKED only, and this amendment does not touch BLOCKED.

**What it moves, and why.** The reader (`dag2_read.py`) codes the table cell by cell. It found that the table above is not complete,
although it says it is. Over the 16 (R-CRAWL, R-MOVE) × 2-seed cells with the guards held:
* FIXED takes 1 cell;
* PARTIAL takes 7;
* NO EFFECT takes 4;
* **4 cells match no row**: R-MOVE holds at exactly one seed and R-CRAWL fails at that seed, so neither PARTIAL clause applies (the first
  needs R-MOVE at both seeds, the second needs R-CRAWL and R-MOVE together at one seed) and NO EFFECT does not apply (it needs R-MOVE to
  fail at both).

The fix partitions the guarded cells on R-MOVE alone: **at both seeds** with R-CRAWL at both is FIXED, **at one or two seeds** otherwise is
PARTIAL, **at neither** is NO EFFECT. Each of the 12 cells that were placed keeps its outcome. The 4 unplaced cells become PARTIAL.

### The outcome table, as read from now on

| outcome | condition | committed next |
|---|---|---|
| **FIXED** | R-CRAWL and R-MOVE at both seeds; guards held | adopt DAG2 as the base. Read A is re-run on DAG2 and on DAG2 with DAG-OACC's lever when it exists; a DOMINANT read there is the dominance claim |
| **PARTIAL** | R-MOVE at both seeds but R-CRAWL fails somewhere; or R-MOVE at one seed only; guards held | adopt DAG2 as the base if R-MOVE holds at both seeds, else DAG stays the base; report. The residual crawl is then a launch-intent defect (O's v0 at tick 1, `R76`). Next: the launch-recovery target of `PREREG_V8F_HACC_MC` §7 on the base's states, under its own PREREG |
| **NO EFFECT** | R-MOVE fails at both seeds; guards held | the residual crawl is not an own-state-distribution defect. Next: as PARTIAL, on DAG states |
| **HARMS** | a guard fails at ≥ 1 seed | DAG2 is not adopted; DAG stays the base. Next: as PARTIAL, on DAG states |
| **BLOCKED** | COVERAGE < 0.5 | the rejoin-target form (DAGGER §7) |
| **REFUSED** | any control fails | fix and re-run |

Two more readings change with it. Neither moves a decision.
* **PARTIAL's committed next ran "on DAG2 states" even when DAG2 is not adopted.** Four of the 7 PARTIAL cells already had R-MOVE at one seed
  only, so DAG2 was not adopted in them. The launch-recovery work now runs on the base's states: DAG2's if PARTIAL adopted it, DAG's
  otherwise.
* **The change-3 bars enter no outcome.** The guards R-TLAT and R-TLON read *not separated-worse*, and that reading needs no bar. The bars
  are reported beside, as DAGGER's *clears* reading (separated-better AND point ≥ bar, inclusive).

**What R-MOVE can and cannot claim (`H-ESTIM-SEED-1`).** No crawl row has a training-replicate floor: RF is measured on the lat/lon F1 heads
only. A separated R-MOVE at one seed is therefore an episode-level result about one training run.
* What stands in for the replicate is the second seed. DAG2 and DAG2-0 are trained independently at each seed, so R-MOVE at both seeds
  means two independent run pairs each separated.
* Only FIXED, and PARTIAL with R-MOVE at both seeds, adopt DAG2.
* A one-seed R-MOVE is reported as PARTIAL (one seed). It is never called an effect of the lever, and it adopts nothing.

**What does not change.** The arms, the banks, the crawl set C, R-CRAWL, R-MOVE, every guard, F, 2F, RF, the bars' values, the controls,
COVERAGE, the estimator, the tiers, the beside rows and the queue.
