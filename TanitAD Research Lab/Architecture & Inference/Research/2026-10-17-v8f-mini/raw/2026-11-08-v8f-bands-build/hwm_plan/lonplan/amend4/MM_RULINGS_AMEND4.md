# MM rulings on AMEND_HWMPLAN_4 (L2-IMG), at the freeze, 2026-11-17

**The amendment is FROZEN as drafted (`AMEND_HWMPLAN_4.md`, sha256 in `AMEND_HWMPLAN_4.sha256`), with the fills of section 10 set as below.** No SCORE window has been planned under `prim` or `prim23`; every number in the draft is TRAIN, in-sample (128 windows, 32 clips per chassis), T0.

## Fills (section 10 / QUESTIONS_AMEND4 M0-M5)
* **M0, run it: YES.** The expected label is NULL (ESTIMATED from TRAIN). The register row still needs the programme's estimator on held-out windows, not an in-sample read. It also sets the RA read's base solver mechanically (section 6).
  * Cost: about 4.7 h of dev-box CPU (1 process × 2 threads), no GPU, no spend.
  * **AMEND_HWMPLAN_5 (the level-1 cost read through RA) is drafted NOW, in parallel**, with its base solver as its one open fill. Its draft names which level's actions RA reads: the TRAIN hint is that level 2's own primitives as knots 1-10 read 0.06-0.15 m/s² better at 0-2 s than level 1's plan.
* **M1, code location: (b),** the study-local module `lonplan/amend4/l2img.py` installed around `HWMPlanner.plan_batch`. The code of record (`hwm_plan.py` 5a072272…) is untouched. `off` equals the code of record by construction, and C-OFF checks it.
* **M2, A0: reuse** LON-PLAN's banked SCORE plans, on the condition that C-OFF re-plans batch 0 of each chassis bit-equal under the new build.
* **M3, the RS rule:** the in-batch cyclic donor shift. A batch without a cross-clip shift is refused.
* **M5, P23: REGISTERED** with its own label (L23IMG-…) and the multiplicity guard of section 6.
  * P23 was added after the first TRAIN read showed level 3 off the image. Its constants were fixed before it ran, and SCORE is untouched. This is stated wherever its label is quoted.

## For the PI (reported in the MM's next report; no approval is needed for CPU-only work)
* **P1:** the read is dev-box CPU only, and it runs after the build unless the PI says no (as LON-PLAN's Q1).
* **P2:** unchanged. World-model action-sensitivity training on box 2 is not pre-approved, and it is asked with this read's result. The zero-training RA read comes first in every branch.

## Register rows landed with the freeze
`H-L2IMG` (HYPOTHESIS, OPEN), `D-HWM-UPPER-OFFIMAGE` (MEASURED defect), `D-HWM-PRIORBOX-IMAGE` (MEASURED defect), as drafted in section 12.
