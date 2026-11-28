# RESULT: free-point navtest read. **VALID; row FP-NONE.** No zero-training scorer point is deployable-safe. R0B gains PDMS but pays in NC and TTC.

Master Mind, 2026-10-18. Prereg `PREREG_FREEPOINT.md` (landed before any number), ADDENDUM_1 (2026-10-17), ADDENDUM_2 (2026-10-18, sha256 `b72e345a…`, written before D3's result).

**Tier and estimator.**
- T1 NAVSIM v1.1 PDMS on the full navtest: 12,146 tokens, 136 logs.
- Paired log-cluster bootstrap, 10,000 resamples, seed 20260927. It answers *another draw of LOGS* only: one training seed per state, and selection is deterministic, so there is no inference variance.
- Evidence class: **MEASURED**. Raw files: `raw/result/freepoint_result.json` and `raw/result/RESULT_TABLES.md`.

## 1. Gates: VALID
| gate | result |
|---|---|
| IC1 (inputs) | 0 mismatches over 3,410 tokens compared |
| IC2, IC5 (forward identity) | max abs 0.0; C0A scored twice bit-identical (`ic5_forward.bit_identical`) |
| IC3 / IC4 (BASE pose agreement vs banked) | 12,132 / 12,146 equal = 0.9988 (bar 0.97) |
| BASE vs banked u400 PDMS | −0.01 [−0.02, +0.00] (identity) |
| D1 (shuffled picks must disagree) | agreement 0.013: teeth |
| D2 (state fingerprints) | match expected: teeth |
| **D3 (swap two camera slots, 8 tokens: pose agreement must FAIL)** | **`frac_equal_with_swap` 0.0 < 0.5: teeth** (ADDENDUM_2; CPU forward) |

The chain's first verdict was `ZZCHAIN_DONE INVALID`, with D3 **absent**, not failed:
- no code wrote `<work>/D3.json`;
- the swap smoke had been staged but never run.

ADDENDUM_2 committed both outcomes before running D3 as ADDENDUM_1 defines it. Only the `checks` and `read` stages were re-run. The pre-D3 outputs are kept in `raw/result_before_d3/`.
- `RESULT_TABLES.md` and the families JSON are byte-identical to the final ones.
- `freepoint_result.json` differs only in its `identity` block, which now carries D3.
- The gate summary differs.

## 2. Levels and deltas (x100)
| point | PDMS | NC | DAC | EP | TTC | DDC | zero-PDMS picks |
|---|---|---|---|---|---|---|---|
| BASE (deployed, `r0 step0`) | 88.24 | 98.94 | 96.57 | 81.19 | 95.63 | 97.66 | 530 |
| C0A | 87.36 | 98.98 | 96.35 | 79.35 | 95.64 | 98.69 | 549 |
| **R0B** | **88.79** | 98.69 | 96.53 | 83.00 | 95.23 | 97.55 | 566 |
| M25 | 87.77 | 98.91 | 96.45 | 80.39 | 95.48 | 98.59 | 545 |

**Against BASE (the committed comparison):**

| point | PDMS | NC | DAC | EP | TTC | DDC |
|---|---|---|---|---|---|---|
| C0A | −0.88 [−1.47, −0.14] | +0.04 [−0.18, +0.24] | −0.21 [−0.76, +0.48] | −1.84 ADV | +0.02 [−0.44, +0.45] | +1.03 [+0.72, +1.37] |
| **R0B** | **+0.55 [+0.24, +0.84]** | **−0.25 [−0.45, −0.06] ADV** | −0.04 [−0.26, +0.15] | +1.81 [+1.50, +2.13] | **−0.40 [−0.71, −0.09] ADV** | −0.12 [−0.21, −0.03] ADV |
| M25 | −0.47 [−1.03, +0.24] | −0.03 [−0.22, +0.15] | −0.12 [−0.62, +0.57] | −0.80 ADV | −0.15 [−0.58, +0.26] | +0.93 [+0.63, +1.25] |

**DEPLOYABLE-SAFE:**
- R0B: **no**. PDMS separated above 0, but NC and TTC are separated below 0.
- M25: **no**. PDMS is not separated.
- ⇒ **Row FP-NONE.**
- Prereg consequence: *the safety-objective arm proceeds exactly as pre-registered*. It did: R0d read O4 (`raw/2026-10-18-r0d-outcome/`, `D-REFE-SAFEOBJ-R0D`), and its next lever is running (BACKLOG `SO-CARVE`).

**Lever view (no row, reported as committed):**
- R0B − C0A: PDMS +1.43 [+0.63, +2.05]; EP +3.64 [+2.81, +4.40]; NC −0.29 [−0.53, −0.06] ADV; TTC −0.41 [−0.86, +0.03]; DDC −1.14 [−1.49, −0.82] ADV.
- M25 − C0A: PDMS +0.41 [+0.26, +0.57]; EP +1.04; TTC −0.16 [−0.34, +0.00]; DDC −0.10 ADV.

## 3. Four families (`families6`, episode-cluster intervals, unpaired, conservative non-overlap)
- **LONGITUDINAL.**
  - R0B has **no adverse component of 21** and removes most of the deployed scorer's slow bias:
    - `along_final_bias_m` −0.195 [−0.549, +0.153] vs BASE −0.826 [−1.19, −0.478]. ⚠️ **CORRECTED 2026-10-18 (R49):** these intervals OVERLAP on [−0.549, −0.478], so the shift is a point estimate, NOT separated. This line first read "intervals non-overlapping";
    - `speed_bias_mps` −0.064 vs −0.230.
  - C0A has 5 adverse components: `along_mae_m`, `speed_mae_mps`, `speed_rmse_mps`, `target_speed_acc.within_2.0_mps`, and lateral `curvature_mae_1pm`.
  - M25 has none.
  - The EP and TTC sub-scores are in section 2.
- **LATERAL.**
  - No point is adverse. R0B: heading MAE 2.48 vs 2.53 deg; curvature MAE 0.0189 vs 0.0199 1/m; cross-track MAE 0.374 vs 0.378 m.
  - DAC and DDC are in section 2.
- **TACTICAL** (rows without an interval, reported, not tested):
  - longitudinal decision accuracy: R0B 0.687 / kappa 0.529 vs BASE 0.679 / 0.516;
  - lateral decision: 0.927 vs 0.925;
  - 5-way manoeuvre: 0.788 vs 0.783.
- **STRATEGIC: PAUSED** (PI 2026-09-19, PI_DECISION_QUEUE item 25) and unavailable in NAVSIM. Not a pass.

## 4. Reading
1. RL-S1's EPO lever has the same anatomy on navtest (T1, log-disjoint from RL-val) as on RL-val:
   - **it buys progress** (EP +1.81 vs BASE, and the slow bias mostly removed);
   - **it pays in the safety terms** (NC −0.25, TTC −0.40, +36 zero-PDMS picks).
   - On RL-val (ESTIMATED, selection tier) R0-B vs the deployed base read PDMS +0.59 [+0.14, +1.02] and TTC −0.17, not separated. The navtest point is almost the same PDMS, and the TTC cost is larger and separated.
2. That trade is what the SAFEOBJ constraint exists to remove. R0B's longitudinal gain is the part worth keeping: no adverse family component. The cost is entirely in NC/TTC.
3. Rule Zero: FP-NONE eliminates "deploy a free point". The next lever was pre-committed, and it ran and is running: SAFEOBJ R0d → O4 → SO-CARVE.

## 5. Instrument finding (not a retraction)
A gate that reads a control file **no code writes** reports the control *missing*. The chain correctly refused (INVALID, not VALID). But the defect went unseen until the read, because no stage asserted that every control it gates on has a producer.

Same family as the "built, tested, and unreachable from its caller" class in CLAUDE.md:
- the D3 check existed and was correct;
- its input was never produced.

Fix for any successor chain: the `checks` stage must refuse at *startup*, not at the read, when a gated control has no producer stage.

## 6. Deliverable manifest
| artifact | where |
|---|---|
| this RESULT.md, ADDENDUM_1, ADDENDUM_2 (+sha256), PLAN.md, HANDOFF.md, EXPECTED_STATES.json, `code/*.py` (10) | repo, this package |
| `raw/result/` (RESULT_TABLES.md, freepoint_result.json, families JSON + logs), `raw/result_before_d3/` (same, pre-D3), gate JSONs, chain and score logs | repo, this package `raw/` |
| 9 chunk npz, 4+1 seam npz, points.npz, calib.npz, inputs_full.json, `cap/` (bulk, ~311 MB) | dev box `C:/Users/Admin/freept/work/` **and** `D:/Projects/TanitAD-artifacts/freept_navtest_2026-10-17/`, sha256 in this package's `MANIFEST.sha256` |
| `ship.log`, `ship.before_outage_2255.log` | dev box only. **Not committed: they contain the old pod's address.** |
| pod-side `/workspace/ops/freept` | pod. Delete only after the D: copy is verified (HANDOFF step 6). Not done. |
