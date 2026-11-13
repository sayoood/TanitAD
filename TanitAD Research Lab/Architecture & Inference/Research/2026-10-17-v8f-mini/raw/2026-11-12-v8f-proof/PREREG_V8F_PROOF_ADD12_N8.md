# Note N8 to PREREG_V8F_PROOF_ADD12 (SPATIAL): the lead reads on TRAINED states: what is extracted, the tool, the order, what is quoted (2026-11-13)

Written before any trained state is extracted, and while the stage 0e read runs (no stage 0e number exists yet).

## 1. What is read

The level-1 world-model state of a checkpoint, `WMAux.z1(adapter(tokens) + position term)`, per frame. A frame's state depends on that frame alone (no temporal context enters `z1`).
Extracted on the pod by `pod/probe/extract_states.py` (CPU, one thread, idle priority, one clip in memory; sha256 `f7cfb17e72a822a4`) for
(a) the 1,200 train probe clips of note N7, row for row with the probe set, and (b) the eval clips of the pod cache.

## 2. The reads

The stage 0e tool with the principal-component states replaced by the trained state ("state-file mode"): no state is fitted; folds, probe, subsample rule, pair rule, bootstrap and the ego vector are those of `S0E_DEFS.md`.
* r-a: lead gap R2 and lead presence AUROC from the state, and the difference to the ego vector.
* Change read (the form note N5 assigns to Q2): arms H (state at g and g-5, plus the ego block), N (state at g, plus the ego block), E (ego block alone); skill over persistence on continuous-lead pairs.
* Where: the train probe set (in-sample for the model, and labelled so wherever a number is quoted); r-a also on the eval bank frames (139 episodes; out-of-sample, wide intervals).

## 3. Arms and order

* Every HIER seed of HIERREP as it finishes: the trained mean state, 128 numbers. This is the baseline of stage 1.
* The G and W seeds of stage 1 when they exist (grid state 640, wide mean state 640).

## 4. What is quoted, and what is not

* For HIER seeds alone: values with intervals under the label "baseline (trained mean state)". No outcome name of the card is assigned from them.
* Trained mean state beside the untrained states of stage 0e: two numbers side by side. They have different widths (128 against 64 and 640), so no paired claim is made across them.
* The card's Q1 and Q2 names need both G seeds, as the card writes. The contrasts fixed now for stage 1: G minus HIER at the same seed, and G minus W at the same seed, on gap R2, presence AUROC and change skill (paired clip-cluster bootstrap).
  A contrast is called separated only if it holds at both seeds AND is larger than the spread of the HIER seeds read by this tool (the replicate floor of this read; rule H-ESTIM-SEED-1: an interval over clips is blind to training variance).

## 5. Controls of the state-file mode (each must read its known value, or nothing is quoted)

* t1 alignment: the `clip_idx` and `t` arrays of a state file equal those of the probe set exactly; the checkpoint sha256 in the state file's record equals the sha256 of the arm's `ckpt_last.pt`.
* t2 identity of the two code paths: state-file mode fed with the mean over the 40 projected cells (64 numbers, a fixed linear function of the probe tokens) reproduces the stage 0e r-a values of the mean state (gap R2 and presence AUROC) within 0.005.
* t3 labels permuted within fold: every R2 and skill at or below 0.02.
* t4 the state is what the arm declares: 128 numbers for HIER, 640 for G and W; for G the gate record of the arm shows `wm_state=grid` built (the arm's gate token), for HIER and W `mean`.

## 6. Both outcomes of the baseline read, committed

* The trained mean state reads the lead gap no better than the ego vector, or no better than the untrained mean state of stage 0e: with power, training under the present auxiliary loss does not put the lead into a mean state. This is the measured form of the pooling lemma for a trained model and supports the lever of the card (keep the cells).
* The trained mean state reads the lead gap above the ego vector and above the untrained mean state: the auxiliary loss already organises the mean state around the lead as far as a mean allows. The bar for G is then the trained HIER state, as the card's `Z-ABOVE-MEAN` contrast already requires, and the size of that margin is the quantity of interest.
* In both cases the change read of the HIER state is the baseline for Q2: a mean state of one frame cannot carry motion (lemma group P; control e2: 0.50), two frames can only through the change of totals.

## 7. Limits

T0, a world-model diagnostic, never driving performance. Train probe clips are in-sample for every trained arm. One checkpoint per seed (the last step). Linear and kernel reads.
LONGITUDINAL, LATERAL, TACTICAL: not measured by these reads; STRATEGIC: PAUSED (PI item 25).
