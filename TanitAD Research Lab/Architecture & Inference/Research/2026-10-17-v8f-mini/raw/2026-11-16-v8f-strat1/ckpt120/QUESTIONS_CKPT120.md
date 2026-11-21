# QUESTIONS_CKPT120: the PI's decisions for the 120 m strategic checkpoint card

Card: `PREREG_V8F_CKPT120_DRAFT.md` (this folder; draft, not frozen, nothing built or run). Only the PI's decisions are listed: architecture changes, the training go and pod time, and the base. The MM's design rulings (the deciding contrast's form, the floor, the gate route, the label build) are in card section 0.

## Why this card has two rungs (one paragraph)
STRAT1 read S-PARTIAL at 80 m. The design measurement shows the shortfall sits where the route text's side never reaches S. The nav token carries the side only within max(30 m, 6 s x speed). At 80 m that leaves 35 of 244 scorable windows without a side. On those windows S1 is on the announced side 26-40 % of the time against a constant of 80 %. On the other 209 windows it is 93-94 % against 76 %. At 120 m, 122 of 341 scorable windows give S no side. An ideal follower of S's current input cannot beat the constant there (−0.126 [−0.301, +0.080], MEASURED from labels). A 120 m output alone (H1) is therefore predicted to fall short. The input lever (H2) is what can pass. Each rung is one lever.

## Q1. Architecture change: does S get a 120 m checkpoint? (H1)
* **What changes.** S gains a fourth checkpoint at 120 m of arc, from one new 1,539-parameter layer on S's existing head, trained against the ego's own driven path at 120 m. The 30/50/80 m checkpoints, their loss and their weight are unchanged.
* **Isolation.** The 120 m point goes to the loss and the evaluator only. T and O never read it (pinned by a test). Feeding it to T is a later card with its own go.
* **Where the point lies.** 120 m lies in the 6-30 s band on 304 of the 341 scorable eval windows; none lie below 6 s.
* **Recommended: YES.**

## Q2. Architecture change: does S read the route text's next turn at any distance? (H2)
* **What changes.** S gains one extra input token: the route text's next turn side and its counted-down distance, with no announcement cut. T and O keep today's nav exactly.
* **Admissibility.** It is the same NVIDIA route text as STRAT1's nav (an input at training and inference). It carries no situation-classifier output. It is optimistic by construction (the route is supplied along the driven path), and every number is worded so.
* **Recommended: YES, approved now to run in the same chain right after H1.** The measurement already predicts H1's shortfall (RULE ZERO), so waiting for H1's read would cost a round-trip for a predicted outcome.

## Q3. Training go and pod time
* **Arms.** 7 arms of 3,000 steps on the LoRA pod, one job at a time:
  * H1 at seeds 0 and 1, plus its replicate (the training-run floor);
  * one label-permuted arm that must fail;
  * H2 at seeds 0 and 1;
  * one donor-route arm that must fail.
* **Time.** About 1.75 h of GPU at 0.19 s/step (INHERITED rate from the STRAT1 pilots), plus a 200-step pilot and G-SUITE on the pod CPU. The chain stops and returns to the MM if its projection exceeds 4 h.
* **Conditional arms.** About 15 min each, only on their committed outcomes: a COSTS branch, a SEEDS branch, a separate-trunk branch.
* **Queue.** The pod also carries the data-scale blocks (about 15-26 GPU-h) and STRAT1's four SEEDS arms (in flight).
* **Recommended: YES, slotted between two data-scale blocks after the SEEDS arms**, so the card reads within a day.

## Q4. Base of record
* **Proposal.** The base is STRAT1's S1, already trained at seeds 0 and 1 on the pod: text nav, ctx heads, the OBD-CTL recipe, commit `a4357e79`. The base is not retrained.
* **Floor.** S1 at seed 10 (in flight) adds a second replicate pair.
* **Two amendments, fixed now.**
  * If STRAT1's SEEDS extension finds the ctx heads cost driving at 3 of 4 seeds, the base becomes N1 (text nav, no ctx heads; banked).
  * If the data-scale card reads DATA-HELPS and you rebase (data-scale Q3), H1 and H2 re-run on the union base. That base needs 120 m labels on the new clips, which need v9 rows first.
* **Recommended: S1 now, with both amendments as written.**
