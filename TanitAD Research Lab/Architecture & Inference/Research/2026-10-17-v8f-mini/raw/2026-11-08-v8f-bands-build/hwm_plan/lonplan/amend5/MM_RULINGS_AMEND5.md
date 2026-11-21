# MM rulings on AMEND_HWMPLAN_5 (RA-L1), at the freeze, 2026-11-17

**The amendment is FROZEN as drafted (`AMEND_HWMPLAN_5.md`, sha256 in `AMEND_HWMPLAN_5.sha256`), with the fills of section 10 set as below.** No SCORE window has had an RA pass. Every number in the draft is TRAIN, in-sample for R35 and for RA (564 windows, 141 clips per chassis), T0.

MM spot check at the freeze: `ra5_ceiling_s0.json` `rows.RA_ORACLE.minus_HOLD` reads 0.541 [0.4816, 0.601], as section 1 quotes it.

## Fills (section 10 / QUESTIONS_AMEND5 M0-M7)
* **M0, the base solver: set mechanically** from AMEND_HWMPLAN_4's labels (its section 6). The branch is B1, B2 or B3 of section 4. It is written into `BASE_SOLVER_AMEND5.md` when AMEND_4's result lands, before any RA pass on SCORE. The frozen file stays byte-equal. The expected branch is B3, ESTIMATED from AMEND_4's TRAIN record.
* **M1, which level's actions RA reads: as recommended.**
  * Under B1 / B2, RA-L2 is the committed lever. RA-L1 is a second registered arm with its own label and AMEND_4's multiplicity guard.
  * Under B3, RA-L1 is the committed lever. RA-L2 on P's banked SCORE files is computed beside, and no outcome is decided on it.
* **M2, code location: (b),** the study-local module `lonplan/amend5/ra_l1.py`, applied post-hoc to the base's banked plan files. The code of record (`hwm_plan.py` 5a072272…) is untouched.
* **M3, RA-L2 on A0: not run.** It would read primitives that AMEND_4 measured as off-image and bound-hitting (`D-HWM-UPPER-OFFIMAGE`).
* **M4, `l2_prim_raw` on the P / P23 files:** required of AMEND_4's build. The MM sent the requirement to its build agent on 2026-11-17. If the TRAIN-stage files lack the key, this read's TRAIN stage re-plans P / P23 on TRAIN, about 10 min per arm and chassis, as section 9 says.
* **M5, RA-ORACLE on SCORE: REGISTERED, beside,** as the ceiling that sorts NULL from READ-LIMITED. Its interpretation is the one committed in section 6.
* **M6, run order:** right after AMEND_4's result lands.
* **M7, run it: YES.** It costs about 1 h of dev-box CPU and is held out for RA. It puts the expected READ-LIMITED on the programme's estimator rather than an in-sample record, and its numbers size the two blocked levers (Q2b, P3).

## Build
One Sonnet agent builds `ra_l1.py`, the wrapper and flag, MEAN, the section 8.2 controls with their regression arms, and the C-PLANT toy. The prototype is `ra5_ceiling.py`. The agent starts when an agent slot frees (cap 4), so the build is ready before AMEND_4's read ends.

## For the PI (in the MM's next report; nothing here needs approval)
* **P1:** the read is dev-box CPU only, about 1 h, zero training, zero spend.
* **P2 (unchanged):** world-model action-sensitivity training on box 2 is not pre-approved (`QUESTIONS_LONPLAN.md` Q2b). It is asked with this read's result, where READ-LIMITED ranks it as the larger term.
* **P3 (new):** a 2 s acceleration readout fit on more TRAIN clips than RA's 141 needs the level-1 latents of more TRAIN clips from box 2's `cache_full/train/clips`. It is asked with this read's result.

## Register rows landed with the freeze
`H-RAL1` (HYPOTHESIS, OPEN) and `D-RA-CEIL-TRAIN` (MEASURED, a TRAIN feasibility record), as drafted in section 12.
