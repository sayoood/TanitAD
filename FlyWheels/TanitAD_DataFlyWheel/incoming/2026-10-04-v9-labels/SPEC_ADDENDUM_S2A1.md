# SPEC addendum S2-A1 — E3's reference distorts turns; a geometry-faithful sub-population is added (E3 as registered stays the REPORTED bar)

*Written 2026-10-04 by the Data FlyWheel after the Master Mind's GO and BEFORE any Stage-2 number exists (no E1–E4
code has run). Hashed into `raw/SPEC_SHA256.txt` before the study runs.*

## Why
E3 measures the lateral leak as δ = the offset of RC_v's point from `S_25` (the path smoothed with σ = 25 m) at the same
arc length, with the bar p90 |δ| ≤ 1.0 m over ALL valid windows. A Gaussian of σ = 25 m (support ±75 m) cuts the
corner of a junction turn by metres (a 90° turn at R 15 m has a 24 m arc, far inside the kernel), so on windows whose
reference support contains a turn δ measures **the reference's own geometric distortion**, not lane-level information.
Measured this way, the bar can fail for a reason unrelated to the leak it is meant to detect. (ESTIMATED from the
kernel geometry; no data looked at.)

## What is added (nothing is removed or changed)
* **E3 as registered is still computed and REPORTED as the bar** (PASS / FAIL as written).
* **E3-A1 (added):** the same δ restricted to **straight-support windows** — the heading of `S_25` changes by ≤ **10°**
  over the reference's whole support `[s_now − 75 m, s_now + L + 75 m]` (where a σ = 25 m route is geometrically
  faithful, so `S_8 − S_25` is within-lane / lane-choice information). Bar: p90 |δ| ≤ **1.0 m** on that population,
  with its n and share of windows printed.
* **E3-A1 diagnostics (no bar):** p90 |δ| on the complementary curved-support windows; median |δ| on straight-support
  windows with vs without a lane-change-scale lateral excursion (D2-style detrended ≥ 1.5 m) within [NOW, NOW+8].
* **Decision rule E4, amended for this case only:** if a variant FAILS E3 as registered but PASSES E3-A1 while its
  curved-support p90 carries the failure, it is reported as "E3 FAILED as registered — failure localised on
  curved-support windows (reference distortion)", and E3-A1 is used in place of E3 in the variant ranking. A variant
  that fails E3-A1 fails E4 outright. Both readings are printed side by side.

## Also fixed now (implementation detail E1 needs, no bar touched)
* E1's GT on all eval139 windows is the egomotion log at the 8 slot times in the NOW frame (raw log position and
  quaternion yaw at NOW), valid when the log covers NOW+6 s with gaps ≤ 1.0 s. On EVAL-DIAG it is cross-checked
  against the bank's `gt` (max and median |difference| printed); if the median exceeds 0.10 m, the bank's `gt` is used
  for every EVAL-DIAG comparison and the discrepancy is reported.
* E2's derangement control: clips are deranged with seed 20261004; a recipient window receives the donor clip's RC at
  the same provider index `t` (clipped to the donor's last window).
