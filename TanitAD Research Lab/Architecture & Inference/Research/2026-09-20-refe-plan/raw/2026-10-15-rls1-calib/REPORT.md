# RL-S1 outcome O2, lever (i): TTC/NC calibration read (report)

Written by the O2 calibration agent (Opus) on 2026-10-15. The agent's harness refused the file write, so the Master Mind banked it from the hand-back.
Pre-registration: `eval/PREREG_RL_S1.md` section 11, appended before any calibrated pick existed. Sections 0-10 are byte-identical to blob `0a71d19e`.

**Evidence class.** Every number is MEASURED on RL-val, the selection tier, with one training seed per arm. Intervals are the paired log-cluster
bootstrap (10,000 resamples, seed 20260927), so they answer only *"would another draw of logs say this?"*. The inputs are sha256-verified against the
HF repos (`inputs_sha256_hf_verified.json`, `hf_get_verify.py`).

## Verdict: FAIL. Calibration does not rescue TTC, and the two follow-on zero-A40 gates also FAIL as registered

## Feasibility

The pick is `argmax NC * DAC * (5 EP + 5 TTC + 2 C) / 12 * LANE` (`refe/planner.py:552-557`). That is a multi-head composite, so a per-head monotone map
can change the pick. Both controls passed:
- **CTRL-ID:** identity maps change 0 picks in all 14 dumps.
- **Reach:** a constant map changes 413 to 1,528 picks per dump.

## Task 1: R0b against C0 (reproduces `../2026-10-15-rls1b-r0b-kill/kill_checks.jsonl` to 1e-9)

| read | PDMS | TTC | NC | EP | DDC |
|---|---|---|---|---|---|
| u250 | +0.53 [+0.11, +0.94] | -0.60 [-1.14, -0.10] | -0.03 | +2.23 | -0.67 |
| u500 | +0.90 [+0.39, +1.43] | -0.73 [-1.39, -0.17] | -0.17 | +2.74 | -0.75 |

- DAC is -0.27 at u250 and +0.10 at u500. C is 0 at both reads.
- **Outcome O5 fired by its letter at u250.** R0b's own BCE-only arm reads EP -0.08 [-0.18, -0.01] against C0. Against that arm, R0b also fails g5
  (TTC -0.57 / -0.70), so the kill stands on both pairs.

## Section 11 primary: per-head Platt calibration, 5-fold cross-fitted over logs, the same procedure on both sides of every pair

TTC and PDMS of calibrated R against the calibrated control at the same update:

| read | TTC | PDMS |
|---|---|---|
| R0 u500 vs C0 | -0.80 | +0.99 [+0.48, +1.52] |
| R0 u1000 vs C0 | -0.87 | +1.23 [+0.64, +1.82] |
| R0b u250 / u500 vs C0 | -0.70 / -0.80 | +0.52 / +0.85 |
| R0b u250 / u500 vs R0b-A | -0.67 / -0.80 | +0.59 / +0.79 |

- B2 (the PDMS lower bound above 0) holds at every deciding read.
- B1 (TTC >= -0.5) fails at every deciding read.
- The FAIL holds under all three alternative fold seeds, all three secondary maps, and even an in-sample fit.
- **Why it fails:** the heads were already calibrated, with held-out error 0.004 to 0.026 and fitted slope 0.90 to 1.02. A monotone map also cannot
  change a head's within-scene ranking.

## Two further zero-A40 levers, each pre-registered in this folder before it ran

| lever | pre-registration | result |
|---|---|---|
| the same TTC gate on both sides | `PREREG_O2_NEXT_TTCGATE.md` | **FAIL.** The gate raises C0's TTC too, so the gap widens to -0.87 to -1.27. |
| the gate on R only, against C0 as deployed | `PREREG_O2_NEXT_GATEDR.md` | **FAIL as registered.** Each arm passes only at its final state (R0 u1000: PDMS +0.76 [+0.19, +1.33], TTC -0.47; R0b u500: +0.77 [+0.25, +1.32], TTC -0.33). Both fail the PDMS lower bound at their earlier read. |

The R-only gate's final-state pass is **post hoc**. It is not promoted: the registered rule required both reads.

## Master Mind re-verification (2026-10-15 ~01:55 Berlin, dev box)

- `calib_read.py` was re-run from scratch with `C:/Users/Admin/venvs/tanitad`. The result is **identical to the banked `calib_read.json` in every
  top-level key except `runtime_s`**, ending `ZZCALIB_READ_OK {'outcome': 'FAIL', ...}`.
- The `PREREG_RL_S1.md` prefix is byte-identical to the tip blob (36,667 bytes). Only section 11 was appended.

## Next lever (Rule Zero)

Every zero-A40 lever on the existing dumps has now failed. The remaining levers are each blocked on a PI decision:

| lever | why it is blocked |
|---|---|
| **O2 (ii): a separate safety head** that EPO cannot move (the TTC/NC heads' gradient isolated from the progress outputs) | an architecture change plus A40 time |
| **R2b**, the lane-change arm | a 4th arm outside RL-S1's authorised three |
| a navtest read of "R0 u1000 + R-only gate" (dev box, about 5.3 h) | it changes the deployed selection rule, and the RL-val reads above do not justify it |

## Files

| file | content |
|---|---|
| `calib_common.py` | shared loader, folds and bootstrap |
| `r0b_deltas.py`, `.json` | task 1 |
| `calib_read.py`, `.json`, `.log` | section 11, primary plus secondaries and controls |
| `PREREG_O2_NEXT_TTCGATE.md`, `ttcgate_read.py`, `.json`, `.log` | the same-gate lever |
| `PREREG_O2_NEXT_GATEDR.md`, `gatedr_read.py`, `.json`, `.log` | the R-only gate lever |
| `hf_get_verify.py`, `inputs_sha256_hf_verified.json` | input provenance |

The local input copies are `C:/Users/Admin/rls1b_dl/r0b/` (HF `Sayood/tanitad-refe-rls1-r0b`) and `C:/Users/Admin/rls1_calib_x/` (re-extracted RL-val
labels). Both are re-fetchable and sha256-pinned.
