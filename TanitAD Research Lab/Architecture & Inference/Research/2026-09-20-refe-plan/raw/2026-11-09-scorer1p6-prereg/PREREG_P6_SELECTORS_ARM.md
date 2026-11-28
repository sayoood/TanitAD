# PREREG_P6_SELECTORS_ARM: the same selectors on a trained arm's logits (FROZEN 2026-11-09)

**Status.** Frozen by the Master Mind, by sha256, before any arm of `PREREG_SCORER1P6.md` finished training and before any arm logit on FIT logs exists. It follows `PREREG_P6_SELECTORS.md`, which ended at nomination in outcome S2, and does not change that card or `PREREG_SCORER1P6.md`.

**Question.** On logits from a scorer that never trained on the fit logs, can a selector raise PDMS? `H-P6-SELECT-1` could not answer this, because P6's logits on FIT logs are in-sample.

**What is out of sample here, and what is not.** The arms' fine-tune excludes the FIT logs, so the arm's scorer weights did not train on them. The trunk and the proposal generator are P6's, frozen, and did see these logs. The test is partly out of sample, and it is reported that way.

## 1 · Arm and data
- **Arm:** the one adopted under the adopted-arm rule of `PREREG_SCORER1P6.md`. If none is adopted: C0. One arm only.
- **Fit data:** the same 8,489 FIT sets, the same PDM truth, the same FIT-A/FIT-B split file (`split.json`, sha256 of that file in `result/split.sha256`, split digest `8a22f2e0...`). Only the six logits are replaced by the arm's.
- **Proposal identity:** the arm's proposals on these sets must be bit-equal to the labelled ones (or differ only by the sets' rounding to 0.01 m and 0.001 rad), or the fit is refused.
- ⛔ navtest is never used for fitting or for choosing between variants.

## 2 · Candidates, nomination, reads
- **Candidates:** the nine variants of `PREREG_P6_SELECTORS.md` section 2, with its hyper-parameters, seeds and Amendment 1's speed proxy. Nothing is added after a read.
- **Comparator:** the arm's own logits through the benchmark rule. That replaces "P6's rule" in sections 3 and 4 of the card.
- **Nomination:** card section 3, on FIT-B.
- **Reads:** card section 4, on E2 then at most one per family on E3, using the arm's `e3_logits.npz` and the truth tables. Reads count in the same K ledger as the first card.

## 3 · Bars, on E3, against the arm's own rule
As card section 4: P, Bonferroni-adjusted CI-lo > 0 · G1, TTC, NC, DAC each point ≥ −0.5 · G2, CRUISE overspeed CI-hi < +0.10 · goal status as `PREREG_SCORER1P6.md`. Four metric families with CIs, STRATEGIC PAUSED and not a pass.

## 4 · Committed outcomes
- **A1 GAIN:** a candidate clears P and the guards. The arm plus selector is the new best selector stack, stated against P6 as well as against the arm.
- **A2 NO GAIN:** none nominates or none clears. Selection on outputs is closed as a lever on this model, in-sample and out. The remaining levers are Batch 2's, in the order of the Shapley split.
- **A3 CONTRADICTION:** a candidate gains on FIT-B and loses on E2 or E3. The rest of the out-of-sample gap lies in the trunk's in-sample reading. Next: fit on logs the trunk never saw, which needs a model trained without them.
- ⛔ A miss is reported as written. Bars do not move after a read.

## 5 · Cost
About 20 min of pod GPU (ESTIMATED: the label job's 0.142 s per set, MEASURED, on 8,489 sets). It runs after the arms and only if P7's resume is not delayed by more than that.
