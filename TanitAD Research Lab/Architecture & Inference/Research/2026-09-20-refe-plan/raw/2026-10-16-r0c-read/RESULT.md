# R0c read — O3 "breached with a small KL": the EPO ascent itself lowers TTC; the reference hypothesis is REFUTED

**Master Mind, 2026-10-16.** The procedure is exactly `eval/PREREG_R0C_A1.md` §1:
- the dev box read R0c's HF egress (`Sayood/tanitad-refe-rls1-r0c` `final/`, 19 files, LFS sha256 verified);
- `r0c_ba_read.py` was run per check;
- `r0c_read.py --policy eval/r0c_policy_A1.json`, with policy sha256 `34a8e169c2248c12d9dd2315c4028035a99c9c0e5498ace6b089e926b5b0a0eb` (equal to the landed one).

Script: `r0c_do_read.py`; log: `do_read.log`.

## The arm
R0 is the base; the only change is `--epo-ref armA`, anchoring EPO's KL to arm A's own policy instead of the frozen base. It ran from 11:48Z on the A40, was **KILLED by K3 at 13:43:01Z** ("g3/g5 bound breached at update250 and update500"), and its chain refused any further arm.

## Result (MEASURED; RL-val selection tier, 3,000 sets / 75 logs; paired log-cluster bootstrap, 10,000 resamples, seed 20260927)
| check | PDMS B − A (×100) | TTC B − A | NC B − A | EP A → B | g5 |
|---|---|---|---|---|---|
| u250 | **+0.69 [+0.31, +1.07]** | **−0.53** | −0.05 | 83.18 → 85.68 | TTC breached |
| u500 | **+0.62 [+0.16, +1.07]** | **−1.10** | −0.28 | 83.19 → 85.89 | TTC breached |

- **The outcome is O3:**
  - g5 was breached at two consecutive checks (250, 500);
  - the mean `epo_kl` (A → B) over the second breach window (the A1 Q2 reading) is **0.0665 ≤ 0.2**;
  - every other reading of that window is also ≤ 0.2 (0.034 / 0.050 / 0.057 / 0.064), so the outcome does not depend on the Q2 choice.
- **The prereg's meaning:** B gains PDMS through progress (EP +2.5 to +2.7) and pays for it in TTC. The anchor is not the defect, because B stayed close to A (KL 0.07, vs 0.15–0.26 base-to-B in R0). That is the third arm with the same signature: R0 −1.00, R0b −0.57/−0.70, R0c −0.53/−1.10.
- **The PDMS gain is separated, but single seed:** "separated, single seed, not yet a lever claim" (H-ESTIM-SEED-1). It cannot be adopted, because g5 is a binding guard.

## Four metric families
This is a scorer-only arm read at the RL-val selection tier: the NAVSIM PDM-score of the trajectory the scorer selects.
- **LONGITUDINAL:** partial. TTC and EP above are its only RL-val members. Speed and headway are not computed at RL-val (n = 0).
- **LATERAL:** not computed. The scorer selects among fixed proposals and has no lateral output of its own (n = 0).
- **TACTICAL:** not computed at RL-val (n = 0).
- **STRATEGIC:** PAUSED (PI ruling 2026-09-19, item 25).

Full families are read at the full-navtest endpoint, which only O1 reaches.

## Next lever (committed in the prereg's O3 row, copied verbatim by the reader)
> EPO restricted to the progress heads' parameters with a frozen trunk. The banked displacement lives 99 % in `score_dec` and 0.0 % in `score_head`, so this is a different regime, and its fraction of R0's J gain on cached trunk features is a CPU job to run FIRST (PI decision for the parameter-group change)

- The **CPU job runs first**. It is mine, 0 A40.
- The A40 arm is a **PI decision**.
