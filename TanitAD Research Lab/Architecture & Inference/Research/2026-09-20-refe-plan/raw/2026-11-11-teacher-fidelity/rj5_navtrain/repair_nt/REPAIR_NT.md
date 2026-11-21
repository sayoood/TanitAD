# REPAIR-D on the RJ5 navtrain labels (198 dev-box logs): 90 zero-DAC labels, 73 repaired, 17 left over as the navtrain development set

Master Mind, 2026-11-13. This is descriptive label construction on navtrain only; no navtest token was used and there is no ledger entry.

- **Evidence class:** MEASURED.
- **Tier:** builder-side tracked score. This is the scorer REPAIR-D certifies with: the LQR tracker on the nuPlan-DB scene. It is not the harness.

## What ran
- **Inputs** (`mk_inputs.py`):
  - the label plan is `p5r`, from the RJ5 labels of the 198 local logs: 16,133 rows, banked on HF at `evidence/2026-11-13-rj5-navtrain-labels-local`;
  - the target is the RJ5 roll `traj`, from the merged RJ5 rows on HF at `evidence/2026-11-13-rj5-navtrain-local`.
- **Command:** `n4_nt_build.py derive --route navtrain`, the pinned TF-N4 A1 code (run tree `F:/tanitad_scratch/n4nt/run_a1`, read only), 2 shards. It ran 13:38 to 16:42 local, CPU only (`run_repair_nt.sh`).
- **Smoke, 4 tokens (`smoke/`):** with target `tc20s`, C-DERIV read 0.60 to 0.94 m; with the RJ5 roll `traj`, it read 0.0.
  - `tc20s` is the tracker-conditioned target, not the RJ5 roll.
  - The full run uses `traj`.

## Results (`derive_rows.tar.xz`)
| | count |
|---|---|
| labels scored | 16,133 (0 errors) |
| C-DERIV (200 tokens: R(traj) vs the banked `p5r`) | max deviation 0.0 m |
| builder DAC = 0 | **90 (0.56 %)** (navtest N2-R: 116 of 12,146 = 0.96 %) |
| REPAIR-D (N4D) repaired | 73 of 90 |
| REPAIR-D-M (N4DM) repaired | 74 of 90 |
| deliberate regression N4X (sign-flipped) repaired | 5 of 90 |
| **residual after REPAIR-D** | **17 tokens on 7 logs** (`navtrain_residual17.json`; N4DM repairs 1 of them) |

- The labels' builder mean PDMS is 95.77 and builder DAC is 99.44.
- That is over all 16,133, before any repair, x100.

## Use
- **The 17 tokens are the navtrain development set** for the REPAIR-D successor.
  - On navtest, 15 of the 20 residual tokens had **no** DAC = 1 teacher iterate (TF-N6 A1, `n6_tol/run/RESULT_N6.md`). That generator limit is what a wider candidate generator must close.
  - Its parameters are set here, never on the navtest 20.
- **Student labels:** applying the TF-N6 rule (tolerance teacher, then human) to these residuals gives a navtrain label set with no zero-DAC target.
  - The card asks for a tracker-aware check of human-sourced labels first.
- The 877 other logs of the bank are being rolled now (`../bulk/RJ5_BULK_FETCH.md`). The same pass on them grows the set.
