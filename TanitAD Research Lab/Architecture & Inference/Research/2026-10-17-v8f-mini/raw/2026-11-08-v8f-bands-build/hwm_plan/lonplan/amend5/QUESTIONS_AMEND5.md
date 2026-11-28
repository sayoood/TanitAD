# QUESTIONS_AMEND5: decisions raised by AMEND_HWMPLAN_5 (RA-L1, the level-1 acceleration cost read through RA)

Draft, 2026-11-17, Arch+Inference agent, for the Master Mind (MM). Each question has a recommendation. Section numbers refer to `AMEND_HWMPLAN_5_DRAFT.md`. Nothing here needs GPU, box 2, training or spend.

## For the MM (freeze fills, section 10)

### M0. The base solver (the one open fill)
* It is set mechanically by AMEND_HWMPLAN_4's labels (its section 6), and this draft writes all three branches out (section 4):
  * **B1:** an AMEND_4 lever arm reads CEILING. Base = that arm (both: P23 if its Δ_H point is lower at both chassis, else P).
  * **B2:** else an arm reads NARROWS. Base = that arm (both: the same tie rule on Δ_0).
  * **B3:** else A0's upper levels (LON-PLAN's banked `score/A0_s{k}`).
* **Recommendation:** write the branch that AMEND_4's readout names into the Log at freeze, before any RA plan exists on SCORE. The expected branch is **B3** (AMEND_4 expected L2IMG-NULL and L23IMG-NULL from TRAIN).

### M1. Which level's actions RA reads (section 2.2)
* **RA-L2:** the target is RA's read of level 2's own primitives, knots 1-10. They exist only under P and P23, which bank `l2_prim_raw`. The TRAIN hint (AMEND_4 section 3) favours this source: level 2's primitives as knots 1-10 read 0.06-0.15 m/s² better at 0-2 s than level 1's plan.
* **RA-L1:** the target is RA's read of the base's own level-1 knots 1-10. It exists in every branch.
* **Recommendation:** in B1 and B2, RA-L2 is the committed lever and RA-L1 a registered second arm with its own label and AMEND_4's multiplicity guard. In B3, RA-L1 is the committed lever. RA-L2 on P's banked SCORE plans is computed beside at zero planner cost, and no outcome is decided on it. A0 itself has no level-2 primitives for knots 1-10: its decode starts at macro 3 (`hwm_plan.stitch`, j0 = 3), and decoding macros 0-3 would inherit the off-image defect (median residual 0.90-0.91, `D-HWM-UPPER-OFFIMAGE`).

### M2. Code location
* **(a)** The code of record (`hwm_plan.py`, `run_hwm.py`).
* **(b) Recommended:** a study-local module `lonplan/amend5/ra_l1.py`, applied POST-HOC to the base's banked plan files. The RA pass needs only `z_now` (the SCORE `windows.npz`), HOLD_c, the base's `a1_raw` and, for RA-L2, `l2_prim_raw`. It draws no random number. The code-of-record sha256 (`hwm_plan.py` `5a072272…`) stays the one every banked `SOLVER.json` records.

### M3. A0's level-2 macros are not banked
* LON-PLAN's A0 files hold `plan_raw`, `cost1`, `iters1`, `idx`, `a1_raw` and `decode_resid` only (MEASURED, `score/A0_s0/plans_hwm_oracle_s0.npz`).
* So an RA-L2 arm on A0 would need a re-plan of A0 with the macros banked (about 2,594 s per chassis for 2 seeds, MEASURED rate) plus a decode of macros 0-3.
* **Recommendation:** do not run it. It reads off-image primitives that AMEND_4 measured as bound-hitting on 94-97 % of rows. The beside row on P's primitives answers the source question at zero cost.

### M4. Does AMEND_4's build bank `l2_prim_raw` on the TRAIN stage too?
* AMEND_4 section 4 lists `l2_prim_raw` among the keys of the P / P23 plan files. This draft's TRAIN stage (section 5) needs them on the 564 TRAIN windows as well as on SCORE.
* **Recommendation:** confirm it with AMEND_4's build agent now. If the TRAIN-stage files lack the key, the TRAIN stage of this read re-plans P / P23 on TRAIN (about 10 min per arm and chassis at the MEASURED 1.0-1.1 s per window).

### M5. Register the RA-ORACLE ceiling row on SCORE
* RA-ORACLE hands the RA pass the TRUE frame-10 latent as its target: a stronger oracle than HWM-ORACLE's frame-30 goal. It is a T0 ceiling in the sense of PT3M's ORACLE-END and READ-END, never a plan or an arm.
* It is what tells a NULL caused by the source (the hierarchy's 2 s intent) apart from a NULL caused by the read itself (the world model along RA's direction, or a flat 2 s profile).
* **Recommendation:** register it, beside, with its committed interpretation in section 6. It costs about 2 minutes of CPU per chassis. RA was fit on TRAIN clips only, so on SCORE it is held out.

### M7. Run the SCORE read, or record the expectation?
* **What the TRAIN record says (in-sample for R35 and for RA, ESTIMATED for SCORE; section 3).** Even with the TRUE frame-10 latent as its target, the RA pass lands 0-2 s `acc_mae` +0.541 [+0.482, +0.601] / +0.586 [+0.525, +0.651] m/s² above HOLD. No REG dose (λ 0.003-1000), no gain gate (q50-q90) and no candidate count brings it below HOLD. The expected label is READ-LIMITED.
* **Recommendation: run it.** It costs about 1 h of CPU and is held out for RA. Its RA-ORACLE row puts READ-LIMITED on the programme's estimator instead of an in-sample TRAIN record. Its numbers also size the two blocked levers that the PI is asked about (Q2b, P3).
* **The alternative:** record `H-RAL1` as "expected READ-LIMITED, not read" and take Q2b / P3 to the PI now with the TRAIN numbers. That saves about 1 h of CPU and leaves the register row on in-sample evidence.

### M6. Run order
* **Recommendation:** run this read right after AMEND_4's readout lands. It is post-hoc on banked files and costs about 1 h of CPU in total (section 9). It can run beside any other dev-box CPU job that leaves ≥ 5.5 GB of commit free.

## For the PI (reported in the MM's next report; nothing here needs approval)
* **P1.** The read is dev-box CPU only, about 1 h, zero training, zero spend. It runs after the build unless the PI says no, as LON-PLAN's and AMEND_4's reads did.
* **P2.** Unchanged from `QUESTIONS_LONPLAN.md` Q2b and `MM_RULINGS_AMEND4.md` P2: world-model action-sensitivity training on box 2 is not pre-approved. Section 6 names it with that blocker in the branches where it is the next lever, and this read's result goes with that question.
* **P3 (new, only if section 6 lands on READ-LIMITED).** The next lever there is a 2 s acceleration readout fit on more TRAIN clips than RA's 141. Fitting it is CPU only, but it needs the level-1 latents of more TRAIN clips from box 2's `cache_full/train/clips` (the `collect_latents_train_R35.py` pull, 3-10 s per clip, MEASURED in `collect_R35.log`). The pull uses box 2 only for reading files. It is asked with this read's result, not now.
