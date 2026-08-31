<title>RESOLUTION — Orbis 2 read. The hold-action novelty claim NARROWS, it does not fall</title>

# RESOLUTION — `2607.15898` (Orbis 2), the flagged read

**Read 2026-08-31 by the Master Mind** — assigned off `LAB_DIGEST.md`, which surfaced this
as the one open item with a deadline attached to *a publication* rather than to a gate.
⭐ It was the **first item the new digest routed**, and it had been sitting unread in a
`COMMS.md` nobody opened.

**Source:** the banked PDF, 19 pages, full text extracted. Not the abstract, not a relay.

## What Orbis 2 actually does — verbatim

> *"To evaluate the responsiveness of our model to steering commands, we consider two
> settings. 1) **Original trajectories**: using the unaltered ground truth steering signal
> corresponding to each validation sample (as done in [12, 18, 35]) and 2) **Counterfactual
> trajectories**: obtained by altering the original odometry … scaling the original speeds
> and yaw rates by a factor 0.5 and 1.5."*

⭐⭐⭐ **AND THEIR STATED RATIONALE IS OUR ECHO ARGUMENT, PUBLISHED:**

> *"It is essential to evaluate steering in a counterfactual setting, because the original
> trajectories are **partly leaked by the context frames**, i.e. **a model that cannot steer
> could still score well by relying on context only**."*

Evaluated on 100 six-second nuPlan-turns samples (500 generations); Table 3 shows Orbis 2
best on the original trajectory and on three of four counterfactual modes.

## ⇒ The verdict, and it is a NARROWING not a retraction

⚠️ **Our claim as WORDED survives.** They run no hold-action floor: their control **varies**
the action across five modes; ours **holds** it and requires the live arm to beat it. Those
are different instruments — perturbation versus ablation.

⛔ **But the claim as READ would mislead, and must not ship unqualified.** *"A hold-action
floor the model must beat — not found in any driving world model"* invites the reader to
think we originated the concern. **We did not.** Orbis 2 states the rationale explicitly,
and cites [12, 18, 35] for the real-only steering evaluation it is correcting — so the
*problem* is established literature, and the counterfactual *remedy* is published.

⇒ **Any publication of the hold-action floor MUST cite Orbis 2** and position ours as an
**ablation-form control** against their **perturbation-form** one. What remains genuinely
ours is the specific instrument and its floor semantics, not the insight that context
leakage flatters an action-blind model.

## ⭐ Two things in our favour, both worth keeping

1. **Their counterfactuals are still realised motion.** They are built by *"altering the
   original odometry"* — scaling speeds and yaw rates. That is the same limitation as our
   P2(b): a perturbed realised-motion channel is not a command channel. **Orbis 2 does not
   introduce one either**, so P2(b) remains open ground rather than settled elsewhere.
2. **Ablation and perturbation fail differently.** A model can track scaled odometry (their
   test) by amplifying a context-derived trajectory, without the action being *necessary*.
   Holding the action tests necessity. ⚠️ Neither instrument dominates; the honest framing
   is complementary, and that is a stronger paper position than a novelty claim.

⚠️ **Scope:** this resolves the flagged claim only. It says nothing about Orbis 2's other
results, which were not the question asked.
