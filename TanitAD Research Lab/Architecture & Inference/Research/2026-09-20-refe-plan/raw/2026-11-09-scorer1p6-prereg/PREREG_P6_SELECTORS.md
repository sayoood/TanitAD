# PREREG_P6_SELECTORS: can a selector on P6's own outputs raise PDMS? (FROZEN 2026-11-09)

**Status.** Frozen by the Master Mind, by sha256, before any selector was fitted on navtrain data and before any selector was read on E2 or E3. Batch 0 of `PROGRAMME_P6_SCORER.md`. It does not change `PREREG_SCORER1P6.md`.

**Question.** With no retraining, how much of the gap between P6's pick and the best of its 64 proposals can a selector close, using only what is available at inference: P6's six logits per proposal, the model's own proposal geometry, and measured ego speed? No map, no agents, no truth at inference.

## 1 · Data
- **Fit data:** the FIT sets of the pod label job (24 held-out logs, 1,705 sets, plus 6,784 sets of further whole logs), `fit_pack.npz`: geometry, P6 logits, PDM truth. Whole logs, never in the fine-tune arms' training.
- **Split, by log, seed 20261109, before any fit:** FIT-A about 70 % of sets (fit), FIT-B the rest (nominate). The split list is written with its sha256 before the first fit.
- ⚠️ P6 almost certainly trained on these logs, so its logits there are in-sample. The labels here come from the cached route; on 3 of 120 checked sets EP differs from the database route. The navtest read below is the only judge.
- ⛔ navtest is never used for fitting or for choosing between variants.

## 2 · Candidates (the full list; nothing is added after a read)
| family | variants |
|---|---|
| R rule shape | exponents on NC and DAC from {0.25, 0.5, 1, 2, 4}, weights on EP, TTC from {1, 2, 5, 8, 12}, comfort from {0, 1, 2}; 400 random draws at seed 1 plus P6's own rule; objective: mean true PDMS at the pick on FIT-A |
| C calibration | per-head Platt; per-head isotonic; per-head Platt in three bins of plan v4/v0 (below 0.85, 0.85 to 1.15, above 1.15); each then through the benchmark rule |
| S stackers | gradient-boosted trees on true PDMS; gradient-boosted trees per head (NC, DAC, TTC, EP) through the rule; an MLP per head through the rule. Features: as `probe_families.py`. Hyper-parameters fixed as in that file; MLP: two layers of 128, 60 epochs |
| P policy gradient | full-information expected true PDMS under a softmax over the 64 proposals, residual on P6's rule, temperature 0.05, the MLP of `probe_families.py` |

## 3 · Nomination (FIT-B only)
Each family nominates its best one or two variants by mean true PDMS at the pick on FIT-B, only if its paired log-cluster bootstrap against P6's rule on FIT-B has CI-lo > 0 (10,000 draws, seed 20260927). A family with no such variant nominates none, and that is reported.

## 4 · Reads on navtest and bars
- Nominees are read on **E2** (2,000 tokens, `e2_tokens.json`, sha256 `bd368d91...`), then at most one per family on **E3** (12,146 tokens, by lookup in the full 64-proposal table once it exists, otherwise harness on changed tokens).
- Every navtest read of a selector is counted as K across this card, with the running count printed. Any best-of-K statement uses a Bonferroni-adjusted interval.
- **Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, over the set's own logs. Answers "another draw of logs". **Tier:** T1-class open loop.

| id | endpoint, against P6's rule, on E3 | bar |
|---|---|---|
| P | PDMS × 100 | Bonferroni-adjusted CI-lo > 0 |
| G1 | TTC, NC, DAC sub-score × 100 | each point ≥ −0.5 |
| G2 | CRUISE overspeed share (plan v4/v0 > 1.15) | CI-hi of the difference < +0.10 |
| goal status | PDMS level | as `PREREG_SCORER1P6.md` |

The four metric families are reported with CIs as in `PREREG_SCORER1P6.md` section 4, with STRATEGIC **PAUSED** (PI 2026-09-19), not a pass. A cost in a family is named, and the verdict is ADOPT or PDMS GAIN WITH COST by that card's clause.

## 5 · Committed outcomes
- **S1 GAIN:** one or more candidates clear P and the guards. The best is carried forward as a selector, and refitted on the logits of each trained arm as a counted layered candidate.
- **S2 NO SELECTOR GAIN:** none clears. Selection on P6's existing outputs is not a lever: the remaining gap needs better scene reading, so the trained arms of Batch 1 and Batch 2 are the levers, in the order of the Shapley split.
- **S3 CONTRADICTION:** a candidate gains on FIT-B and loses on E2 or E3: the in-sample logits did not transfer. Next: the same fit on the logits of a trained arm on FIT logs it never trained on, where that fit is out of sample.
- ⛔ A miss is reported as written. Bars do not move after a read.
