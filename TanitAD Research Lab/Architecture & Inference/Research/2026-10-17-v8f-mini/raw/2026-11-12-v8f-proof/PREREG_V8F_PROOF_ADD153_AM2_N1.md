# PREREG_V8F_PROOF_ADD153_AM2_N1 — note N1 to amendment 2 of ACCHIST10K: the four-head read is the last ensemble step

Frozen before any output of amendment 2 exists (no run folder `D:/tanitad/leadacc/K25D_s150`, `K25DS_s149` or `ens4` on box 2; the sequence is still waiting for box 2). Amendment 2 (`PREREG_V8F_PROOF_ADD153_AM2.md`, sha256 cda2a800) stands unchanged; this note adds one stopping rule to its section 3. Row 230, hold 170 (box-2 and row order as set by the coordinator, 2026-11-28).

## The rule

* `ENS4` is the **last ensemble step** read against the eval-141 bar. If `ENS4` reads eval-141 A below 0.30, or its eval-141 dA_floor lower bound is not above 0, **no further members are added** to reach that bar. Adding heads until the bar is crossed would turn the bar into a search over ensemble sizes.
* `ENS5` (with `K25D_s150`) is read beside, as amendment 2 says. It names nothing and can never satisfy the consequence. That holds even if `ENS5` alone reads eval-141 A >= 0.30.
* The held-out read stays primary for the name (card ADD153 section 5). The eval-141 read decides only the consequence.
* If `ENS4` misses, the next lever is a different one, not more heads: a longer temporal window or the 8x20 token grid. The grid needs a disk for a token store: box 2's D: has 1.93 GB free against about 13 GB for a crop store. Each needs its own card.
