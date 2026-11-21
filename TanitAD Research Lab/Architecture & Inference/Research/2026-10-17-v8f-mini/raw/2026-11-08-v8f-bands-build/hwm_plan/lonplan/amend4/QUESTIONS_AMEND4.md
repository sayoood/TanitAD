# QUESTIONS_AMEND4: decisions raised by AMEND_HWMPLAN_4 (L2-IMG, the level-2 image constraint at λ = 0)

Draft, 2026-11-17, Arch+Inference agent, for the Master Mind (MM). Each question has a recommendation. Section numbers refer to `AMEND_HWMPLAN_4_DRAFT.md`.

## For the MM (freeze fills, section 10)

### M0. Run this read, or go straight to the RA read?
* **What the TRAIN probe says (in-sample, 128 windows, ESTIMATED for SCORE).**
  * The constraint works as built: in the image, level 3 identical, level 1 closer to its sub-goals (cost −30 % / −21 %).
  * 0-2 s `acc_mae` barely moves: P − A0 = −0.0225 / +0.0069 and P23 − A0 = −0.0659 / −0.0174, against a gap to HOLD of about 0.9 m/s².
  * The expected labels are NULL, and RA is the next lever in every branch.
* **Recommendation: run it, and start the RA amendment now, in parallel (M4).**
  * It is about 4.7 h of dev-box CPU, it runs in the background, and it blocks nothing.
  * It settles `H-L2IMG` on held-out windows with the programme's estimator, instead of leaving an in-sample TRAIN read as the record.
  * Its labels set the RA read's base solver mechanically (section 6).
* **The alternative:** skip SCORE, record `H-L2IMG` as "expected NULL, not read", and draft RA on A0's upper levels. This saves 4.7 h of CPU, but it leaves a register row resting on in-sample evidence.

### M1. Where does the new code live?
* **(a)** In the code of record: `SolverCfg.l2_image` in `hwm_plan.py` and `--l2-image` in `run_hwm.py`, as AMEND_HWMPLAN_3 section 7 sketched. This needs a land, and `test_hwm_plan.py` (50 tests, about 9 min) re-run.
* **(b)** A study-local module `lonplan/amend4/l2img.py` that installs the section 2.2 shadow around `HWMPlanner.plan_batch` (instance attributes for one call, as `shadow_solver` does), called by the run wrapper. The code of record is untouched.
* **Recommendation: (b).**
  * The code-of-record sha256 stays the one that LON-PLAN's `SOLVER.json` files and C-IN record (`hwm_plan.py` `5a072272…`). A0's banked SCORE plans then stay readable back with no new land.
  * `l2img_probe.py` already implements the shadow this way, and with nothing installed it reproduced A0's TRAIN plans bit for bit.

### M2. Reuse LON-PLAN's banked A0 SCORE plans?
* **Recommendation: reuse them**, on the condition that C-OFF re-plans batch 0 of `score/A0_s{k}/plans_hwm_oracle_s{0,1}.npz` bit-equal under the new build.
* Re-running A0 costs about 0.7 h per chassis and answers nothing new. Under M1 (b) the condition is close to an identity, and it is still checked.

### M3. The deliberate-regression rule for RS
* **Recommendation:** the in-batch cyclic donor shift (section 4). It is one bijective shift per batch, the smallest that moves every window to another clip, and a batch with none is refused.
  * On TRAIN it read RS − A0 = +0.3245 / +0.3074 (section 3).
* The alternative, `windows.npz`'s `donor` list, can point outside the batch. It would need a cross-batch level-2 rollout, which is new plumbing for no added question.

### M4. Draft the RA amendment (AMEND_HWMPLAN_5) now, in parallel?
* Every outcome row of section 6 names the level-1 cost read through RA as the next lever. Its base solver is set mechanically by this read's labels, and is A0's upper levels if both arms read NULL.
* **Recommendation: yes, draft it now** (zero compute), with the base solver as its one open fill. It stays a separate amendment, as RESULT_LONPLAN section 4 requires.
* A hint for that draft (TRAIN, in-sample): level 2's own primitives as knots 1-10 read 0.06-0.15 m/s² better at 0-2 s than level 1's plan (section 3). So which level's actions RA reads is a question worth naming there.

### M5. P23 as a registered arm
* P23 was added **after** the first TRAIN read showed level 3 off the image as well (ratio 2.23 / 2.23). Its constants were fixed before it ran, and SCORE is untouched.
* **Recommendation: register it**, with its own label and the multiplicity guard of section 6. It is the cheapest extension that could still make the image route work, at about 1.2 h of CPU on P's code.
* The alternative is to report it beside only, with no label.

## For the PI

### P1. Compute
* The read is dev-box CPU only: about 4.7 h of wall time at 1 process × 2 threads (3.3-6.1 h at ±30 %), ESTIMATED from the MEASURED TRAIN probe rates and LON-PLAN's SCORE logs (section 9). No GPU, no box 2, no training, no spend.
* **Recommendation:** tell the PI that it runs after the build unless the PI says no, as was done for LON-PLAN (`QUESTIONS_LONPLAN.md` Q1).

### P2. The training-gated lever behind a non-CEILING outcome (the question is unchanged; its timing is new)
* `QUESTIONS_LONPLAN.md` Q2b: world-model action-sensitivity training on box 2 is not pre-approved, and the PI is to be asked **with this read's result**. This draft keeps it that way. The zero-training RA read (M4) comes first in every branch.
* Q2a (the goal head for the T1-kin read on CEILING) stays with the PI, unchanged.
