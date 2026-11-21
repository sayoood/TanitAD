# QUESTIONS_WEATHERHEAD: the PI's decisions for the weather head card

Card: `PREREG_V8F_WEATHERHEAD_DRAFT.md` (DRAFT, not frozen, nothing run). Only decisions that are the PI's are listed. Each carries a recommended answer. MM rulings (gate route, fit cache, base commit, the all-five output rule) are in the card's section 0 and are not repeated here.

## Background in three lines
* STRAT1's shipped head did not detect weather at both seeds (seed 0 +0.0463 [-0.0108, +0.1014], not separated; seed 1 +0.0717 [+0.0254, +0.1204]; MEASURED, T0, `STRAT1_CTX.json`).
* A zero-training probe that pools over 4 frames x 40 tokens on S1's frozen adapter does detect it at both seeds (+0.0784 [+0.0225, +0.1305] and +0.0813 [+0.0283, +0.1326]; MEASURED, T0, `weatherprobe.json`).
* The same probe on a RANDOM adapter also detects (+0.0713 [+0.0171, +0.1247]; MEASURED): the weather signal is in the cached image features; S1's training is not shown to have put it there.

## Q1. Fallback pod time for form (b), approved in advance and conditional on its trigger?

Form (b) retrains S1 with the pooled head inside the 3,000-step fine-tune (3 LoRA-pod runs, about 2.3 pod-h, ESTIMATED from the STRAT1 card's budget, plus dev-box reads). It runs ONLY if the frozen-model head and its weather-only variant both fail to detect weather AND a head with its own trainable projection does detect it (the card's FEAT rows), which would mean the frozen S1 adapter is the bottleneck. It goes into the pod queue as one job between the data-scale blocks.
**Recommended: YES, conditional on that trigger.** An advance go keeps the fallback from idling on a question; the trigger is measured and pre-registered, and a projected chain above the card's estimate by more than 50 % comes back to you.

## Q2. Is "weather is readable from the model's vision input features by the shipped head" an acceptable detected-weather deliverable?

If the random-adapter control matches the trained adapter, the card's committed wording is: weather is readable from the model's vision input features (cached DINOv3 tokens over 4 frames) by the shipped pooled head, and S1's training is not shown to add to it. The output is the same either way (a vision-only model output, isolated from the planner and every goal path). The alternative is to require the driving model's trained representation to carry weather, which makes form (b) the primary arm: pod time, a driving read at two seeds, and a coupling with STRAT1's still-open SEEDS-DISAGREE driving outcome.
**Recommended: YES, acceptable.** The detected-information product is the output; whether the shared adapter carries it matters only for a later consumer that reads adapter tokens instead of the head (the derived-goals card can decide that with its own pre-registration).

## Q3. Will you label about 40 more frames, stratified on the minority weather classes (fog, rain, snow)?

The weather label agrees with your own frames 36/40 = 90.0 % [76.9, 96.0] overall (INHERITED, `H-CTX-LABELS-V1`), with no per-class check. The probe already reads 0.936-0.939 against the label (MEASURED). Once a head sits at or above the label's own agreement with you, further accuracy against the label cannot be told from label noise, so the next weather lever is the label, mainly on the rare classes (train windows: fog 831, rain 1,647, snow 2,585 of 36,051, MEASURED). Frames would be drawn blind (no label shown), stratified by the label's class, and scored as the first audit.
**Recommended: YES, after the card's read**, about 20-30 minutes of your time (ESTIMATED).

## Q4. (Only if the MM rules that form (a) is not covered by existing authorisations.) Go for the frozen-model head fit?

Form (a) trains NO parameter of the driving model. It fits a 1.33 M-parameter output head on cached features with S1 frozen, checks that every planner output stays bit-identical, and needs about 3-6 h of the dev-box GPU (ESTIMATED), no pod time and no box-2 time. The MM may rule it covered by "Build the arm" (2026-11-16) and the retrain authorisation of 2026-11-08; this question reaches you only if the MM rules otherwise.
**Recommended: YES.**
