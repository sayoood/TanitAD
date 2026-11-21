# N1DW build: the seed-matched R2 repeat for the s1 arms (coordinator review R-N1DW-MINT, item 1), 2026-11-19

**Question.** The first R2 comparison (`r2_compare.out`, `r2_result.json`) read every N1DW arm against two repeats of N1D_s0's argv only (`../r2_run.py` line 13). The s1 arms moved by about the same amount at both weights (ckpt term 0.4475 and 0.4478). The MM's reading, that this was a seed mismatch with the repeat, was a HYPOTHESIS until a repeat with the matching seed was compared.

**Run** (MEASURED, pod GPU, 18:19:51Z to 18:22:04Z, under the GPU queue lock):
- `r2s1_run.py` is `r2_run.py` with N1D_s0 replaced by N1D_s1, writing to its own folder.
- `r2s1_chain.sh` ran two selfcheck-only repeats, A and B, then compared N1DW4_s1 and N1DW2_s1 against them with the unchanged `r2_compare.py`.
- Outputs: `r2s1_compare.out` (sha256 bdf12072), `r2s1_result.json` (sha256 bd5d29cb), log `n1dw_r2s1.log`.
- A first try failed before running anything, because its output folder did not exist (`n1dw_r2s1_try1_nodir.log`). It was rerun unchanged after creating the folder.

**Result**
| term | N1DW4_s1 against the s0 repeat (first comparison) | N1DW4_s1 against the s1 repeat | N1DW2_s1 against the s1 repeat |
|---|---|---|---|
| ckpt | 0.4475 | **0.0009** | **0.0005** |
| wm | 8.285 | **0.0674** | **0.0067** |
| ade | 0.2143 | 0.1464 | 0.0535 |
| fde | 0.3435 | 0.4295 | 0.1605 |

The repeat is bit-identical: every term differs by 0.0 between A and B.

**Reading**
* **The HYPOTHESIS holds.** Against a seed-matched repeat, the s1 arms' shift on the terms the extra loss barely touches falls by a factor of 75 to 1,200 (ckpt and wm).
* What remains scales with (w - 1), as at s0:
  * ade: 0.1464 / 0.0535 = 2.7;
  * fde: 0.4295 / 0.1605 = 2.7;
  * at s0 the ade ratio was 2.9;
  * the ratio of (w - 1) is 3.
* This matches a gradient lever acting through G-LIVE's optimizer steps: G-LIVE records the terms after its steps (see `../../MM_BUILD_N1DW.md`).
* The comparator still reports `pass: false`, as it must for any gradient lever under this design. Its rule compares post-step terms.
* The binding step-0 evidence stays B1's CPU Z0 / Z1 / Z_ID (`../../b1/REPORT_B1.json`, sha256 dc212041).
