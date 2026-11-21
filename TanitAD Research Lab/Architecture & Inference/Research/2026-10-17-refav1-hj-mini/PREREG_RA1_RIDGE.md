# PREREG_RA1_RIDGE — do the pooled frozen DINOv3 features carry ANY readable 6 s signal over P10 at 540 clips?

**Written 2026-10-18, after ES-1 of `PREREG_RA1_ES.md` and before the probe has read any data.** Exploratory. It amends
nothing and moves no bar of PREREG_HJ_STAGE0 or PREREG_RA1_ES.

## Why this is the next lever

ES-1 (MEASURED, T1-class open loop, holdout = the 60-clip Amendment-1 set; eval-141 reported, not selecting) found:

* R0's MLP readout is best at 250 steps. There it is still ≈ P10: eval-141 +0.0144 / +0.0020, holdout −0.0608 / −0.0210,
  none separated.
* It then gets monotonically WORSE: seed-mean holdout delta +0.0628 at 500, +0.3007 at 1,000, +0.3805 at 2,000 and
  +0.4612 at 6,000.

There is no early phase in which it improves, so this is not classical overfitting with a sweet spot. The MLP takes
nothing transferable out of the features at this scale. Two explanations remain, and they lead to different levers:

* **(a)** The features carry transferable signal, but this readout (40 × 1024 tokens → adapter → MLP) memorises 540
  clips before it can find that signal. ⇒ **Lever:** a constrained readout.
* **(b)** At 540 clips, there is nothing a readout can use. ⇒ **Lever:** data scale (all 4,572 clips, already planned by
  PREREG_HJ_STAGE0) or non-frozen features (LoRA).

A holdout-selected ridge map is the lowest-capacity readout available, and it costs CPU minutes. So it separates (a) from
(b) more cheaply than any training run.

## Design

`stack/scripts/ra1_ridge_probe.py`, run once on box 2 under SAFE MODE (CPU, 4 threads). Every data path is RA-1's own,
imported.

* Target: the residual `gt − P10-yaw-540` [30 × 2].
* Feature sets: **f8** (the prior's 8 standardised features: the no-vision control), **mean** (the 1024-d pooled patch
  mean, i.e. the 4 × 10 cells averaged), **mean+f8**, and **bias** (the mean train residual alone).
* λ · n on the grid {1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 100}. Each feature set's λ is chosen on the HOLDOUT
  ADE@6s. eval-141 scores each set ONCE, at its chosen λ.
* **Controls (all must hold, or nothing is read):**
  * P10's ADE@6s reproduces 2.5627 (eval-141) and 3.0245 (holdout) within 1e-3 m;
  * the holdout and train clip lists reproduce RA-1's sha256s;
  * train, holdout and eval-141 are pairwise disjoint.
* Estimator: paired episode-cluster bootstrap, B = 2,000. It covers episode variance only; the probe is deterministic,
  so there is no seed floor. The four metric families are reported for the vision probe and for P10, with STRATEGIC
  PAUSED.

## Readings, committed before any data

Bar for "beats": eval-141 ADE@6s, `mean+f8 − P10` < 0, separated.

1. **SIGNAL** — `mean+f8` beats P10 AND beats `f8`, both separated. ⇒ The features carry transferable signal at 540
   clips, and explanation (a) holds: the MLP readout is the defect. **Next:** constrain R0's readout (low-rank /
   ridge-initialised / strong weight decay), re-run ES at that readout, then the WM arms and the A1x gate.
2. **PRIOR CORRECTION ONLY** — `mean+f8` beats P10 but not `f8`. ⇒ The gain is a correction of P10 from its own inputs,
   not vision, and it counts as (b) for vision. **Next:** a refit of P10's own head (a prior question, not an RA-1
   question), plus the data-scale lever for vision.
3. **NO LINEAR SIGNAL** — `mean+f8` does not beat P10. ⇒ Read together with ES-1, explanation (b) holds. **Next:** data
   scale. Its cost is stated with its blocker: the 256-px frames of the other 3,972 train clips are not on box 2, and the
   dev box C: has about 8 GB free.
   ⚠️ A linear negative is not a learnability negative (`TRAPS.md`), and it is never reported as one.

⚠️ The probe also reports `hold_ade6_by_lambda` for every set. If the chosen λ sits at the grid's edge, that is
disclosed, and the edge is not extended after the fact.
