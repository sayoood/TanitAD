# P6 scorer optimisation: the method list and how candidates are judged (Master Mind, 2026-11-09)

**PI, 2026-11-09:** *"Let focus on optimizing the P6 scorerer to achieve the possible 97.9. Let try all possible ways incl. rule based, machine learning based, RL based, etc..."*

**What 97.9 is.** MEASURED on E1 only (1,058 navtest tokens, stratum-weighted, no interval on the level): the best of P6's own 64 proposals scores 97.88, and P6's pick 91.92. It uses each proposal's true score, which depends on how other agents really moved. It is an upper bound on any selector, not a reachable target. Every method is reported as the share of the gap it closes. The bar that counts first is DriveZero Table A13, 94.55. Full-navtest PDMS is 91.20; the ceiling on the wider sets will be measured on E2.

## First read: which families have headroom (`probe_families/`)

E1 only, fit on 4 folds of logs and scored on the 5th, nothing carried forward. At pick time the methods see only P6's six logits, the model's own proposal geometry and measured ego speed. Paired log-cluster bootstrap, 2,000 draws, unadjusted for the seven methods. DESCRIPTIVE; P6's rule is 91.92.

| method | family | PDMS | delta [95 % CI] | share of the gap |
|---|---|---|---|---|
| rule exponents and weights | rule | 92.71 | +0.79 [+0.08, +1.65] | 13 % |
| per-head Platt calibration | calibration | 92.33 | +0.41 [−0.19, +1.12] | 7 % |
| hard vetoes on NC, DAC, TTC | rule | 91.81 | −0.11 [−0.83, +0.66] | −2 % |
| speed term in the rule | rule | 91.58 | −0.34 [−0.94, +0.09] | −6 % |
| boosted trees, per-head stack | ML | 91.23 | −0.70 [−1.58, +0.31] | −12 % |
| full-information policy gradient | RL without a simulator | 90.71 | −1.21 [−2.42, +0.03] | −20 % |
| boosted trees, true PDMS | ML | 90.63 | −1.29 [−2.54, −0.08] | −22 % |

**How to read it.**
- The rule-shape interval would not survive adjustment for seven methods.
- The ML and policy-gradient losses are from fitting on about 850 tokens. This says those families cannot be judged here, not that they fail. They get about ten times more sets from the labelling job.
- A global speed term loses, consistent with `H-SEL-RECAL-1`. P6's own overspeed share is already 0.535.
- A selector that sees only logits and geometry has no new scene information. The large part of the gap has to come from a scorer that reads the scene better, which is why training comes next.

## The list, in order

Data for everything below: PDM truth on P6's proposals from the pod label job. FIT logs (about 8,000 sets) are never used to train the fine-tune arms.

**Batch 0, no GPU, runs as soon as the labels exist.** Fitted on FIT logs, on P6's logits and again on each trained arm's logits.
1. Rule shape.
2. Calibration: Platt, isotonic, speed-binned.
3. Stackers: boosted trees, and an MLP on logits and geometry.
4. Full-information policy gradient on the same inputs.

**Batch 1, L40, frozen (`PREREG_SCORER1P6.md`, about 12 L40-h, ESTIMATED).** C0, C0r, S1E, then ENS3. Conditional S1.

**Batch 2, L40, order set by Batch 1's Shapley split of the remaining regret (about 25 to 30 L40-h, ESTIMATED, each under its own freeze).**
5. Dose: C0 at 1,600 updates, and all 189,888 sets labelled.
6. RL with a safety constraint on P6: the `--epo-con g5` dual machinery. It was K3-killed on u400 three times, so it runs only with the TTC guard as its stop.
7. Set-aware scorer: attention across the 64 proposals, so each is scored against the others. ⚠️ **CORRECTION 2026-11-09: already built, as written.** P6's `score_dec` blocks apply self-attention across the set (`refe/model.py:430`, `:436`; the docstring at `:703` says so), so this lever as worded adds nothing. If it returns it needs a different mechanism under its own freeze, for example explicit relative features between proposals.
8. Privileged-teacher distillation (`SCORER_GYM_DESIGN.md` section 2.7): a teacher with true agents and map trains the scorer. Inference stays vision and ego motion only.
9. Larger ensemble, and flip test-time augmentation (latency is not a gate now, PI ruling).

**Blocked, named.** Closed-loop RL in the gym: needs the pace term under its own freeze (M-GYM-6 FAILED B3). P7's drivable head addresses DAC (1.32 of the regret) and is read when P7 ends.

**Not admissible at inference.** HD-map or agent-truth rules. The vision-only and ego-motion rulings bind; such information may be used for labels and teachers only.

## How candidates are judged, so that many tries do not manufacture a result

1. ⛔ navtest is never used for fitting. All fitting and choice among a family's variants happens on navtrain FIT logs.
2. Each family nominates at most two candidates, chosen on FIT logs held back from that fit. The nomination rule is written before any navtest read.
3. Nominees are read on E2, then at most one per family on E3 (the full navtest).
4. Every navtest read is counted. Any best-of-K statement uses a Bonferroni-adjusted interval over all K reads so far, and K is printed beside it.
5. Each new lever has its own frozen card, replicate floor and four-family report. `PREREG_SCORER1P6.md` is unchanged by this list.
6. Layering is allowed (a calibrator on a trained arm), and is itself a counted candidate.

## Stopping

Rule Zero applies. First stop: an arm proven at or above 94.55 on E3, then replicate and closed loop. Work then continues up the list toward the ceiling while levers remain, or stops when every remaining lever is named as blocked. A FAILED bar is reported as written and the next lever runs.

## Compute

Batch 1 is authorised with the P7 pause. With Batch 2, the total is about 40 L40-h (ESTIMATED), which moves P7's end from about 14 Nov to about 16 Nov if run in one block. Pod time beyond Batch 1 is checked with the PI before it starts.
