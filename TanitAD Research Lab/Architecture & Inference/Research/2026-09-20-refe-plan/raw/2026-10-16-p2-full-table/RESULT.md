# RESULT: P2's own 64-proposal table over the full navtest. **Built and proven to belong to P2. Oracle selection would reach 98.23 PDMS; P2's pick reaches 88.82. The scorer, not the proposal fan, holds the gap.**

Master Mind, 2026-10-18. Plan and gates are in `PLAN.md`, written before launch on 2026-10-16. The job ran with interruptions:
- low RAM, 2026-10-17 (`P2FULL_FAILED.rc3_20261017T1734_lowram.txt`);
- disk full, 2026-10-18 01:39 (`P2FULL_FAILED.rc3_20261018T0139_diskfull.txt`);
- resumed 06:19 (`P2FULL_FAILED.superseded_by_resume_20261018`);
- finished at `P2FULL_DONE`, 2026-10-18 06:36.

Finished chunks were never recomputed; each resume skipped them by state file.

**Tier and estimator.** T1 NAVSIM v1.1, 12,146 tokens / 136 logs. Log-cluster bootstrap (`zero_tail.Boot`), 10,000 resamples, seed 20260927. It answers *another draw of LOGS* only. P2 is one training seed (H-ESTIM-SEED-1), and its inference is deterministic. **MEASURED, descriptive, nothing fitted.** This table is the confirmation set of the SELRULE-DAC study.

## Gates (all PASS)
- **Same-pick proof on all 12,146 tokens** (`same_pick_proof.json`): pose max abs 0.0 m, 12,146 bit-identical; fingerprints equal; PDMS at the pick equals the banked csv exactly on 12,146 tokens; 509 zero tokens on both sides.
- **Per-chunk fidelity gates** c00 to c13 (`state/gate_fidelity_c*.json`).
- **Cross-check** (`crosscheck_509.json`): the 509 zero-token rows are bit-identical, on every field, to the landed 509-token table.
- `zero_subset_vs_landed.json` reproduces the landed reading key for key: SEL 445 / PROP 64, SEL share 0.874 [0.810, 0.932], SEL ceiling 3.54 [2.90, 4.23].

## Anatomy (`anatomy.json`, x100)
| quantity | point [95 %] |
|---|---|
| P2 pick | **88.82** [87.86, 89.77] |
| best-of-64 oracle | **98.23** [97.82, 98.62] |
| random selector (mean over 64) | 73.00 [71.49, 74.57] |
| oracle − pick | **9.41** [8.68, 10.18] |
| ... from tokens whose pick is non-zero | **5.88** [5.54, 6.21] |
| ... from DAC-zero picks | 2.50 [1.88, 3.20] |
| ... from NC-zero picks | 1.01 [0.77, 1.26] |
| ... from both-zero picks | 0.02 [0.00, 0.07] |

Other counts:
- tokens where the oracle beats the pick: 8,292 of 12,146;
- zero picks: 509;
- tokens whose best proposal is still zero: 64. These are the PROP cases; the fan has no non-zero option.

## Reading
1. **The fan is good enough; the selection is not.** The best-of-64 oracle (98.23) sits above the human reference in `D-REFE-GAP-CI`.
   - That reference is 94.55 (INHERITED from the registry; not re-measured here).
   - Only 64 of 12,146 tokens have no non-zero proposal.
   - The selection gap is the programme's largest single headroom at T1.
2. **Most of that headroom is NOT the zero tail.** 5.88 of the 9.41 points come from picking a weaker non-zero proposal (EP/TTC/C ranking). The zero tail's selection ceiling is 3.54.
   - A scorer lever aimed only at DAC/NC zeros addresses at most about 38 % of the selection headroom.
   - The ranking among non-zero proposals is the larger pool. This is ESTIMATED from the decomposition, not from an arm.
3. Rule Zero: the table's first consumer was the SELRULE-DAC confirm (`raw/2026-10-16-selrule-dac/RESULT_CONFIRM.md`). It reached +0.02 of the 3.18 DAC oracle. Re-weighting the scorer's own confident heads cannot move the zeros, so the zero lever stays P6 (trunk DAC gradient). The 5.88-point non-zero ranking pool is unassigned: named as backlog row `SEL-RANK`.

## Deliverable manifest
| artifact | where |
|---|---|
| this package (code, PLAN, HANDOFF, STATUS, gates, state, logs, anatomy, measured JSON, csv) | repo |
| `table_p2_full.npz` (95.2 MB, sha256 `33441082b7044c95...`), `dumps/dump_main_c00..c13.npz` | `D:/Projects/TanitAD-artifacts/p2_full_table/` (MANIFEST.sha256.json, 24 files) and second copy of the table at `C:/Users/Admin/p2full_work/final/` |
