<title>Injected row I-1, quality half: a cheap pre-screen before any value-head arm (2026-09-09)</title>

# I-1's quality half now has a 0-GPU pre-screen, because latent divergence provably bounds planner cost

`TanitAD Research Lab - Deployment & Optimization - daily pass 2026-09-09 (LAB-RUN-010).`
`⭐ SERVES THE TOP INJECTED ROW I-1 (MM, 2026-08-29) - the QUALITY half, which is the row's open part.`
`Also serves backlog L-3, L-4, FS5-1, and the P-7 rollout-latency finding.`

---

## Why this package exists

I-1 asks: *does a cheap learned terminal value beat fan-scoring at equal rollout budget?* Its **cost
half was MEASURED 2026-09-01** - breadth is **5.94x cheaper** than depth at equal step-budget (73.33 to
12.34 ms at B=960), the value head costs **0.150 ms (1.2 %)**, and breadth is flat to N about 32. The
row's own state says: *"the QUALITY half - does it produce BETTER plans - is UNTOUCHED and is the real
experiment. Row STAYS OPEN."*

Backlog **L-3** sharpened it: compare at **matched LATENCY**, not matched step-count, because they
differ about 6x. **That is a correct but expensive design** - it needs trained value heads on the tiny
ladder before anything can be read.

**Today's finding makes a cheaper first step available.**

---

## Findings

### F1 - A published proof connects a latent diagnostic to PLANNER COST

`lib 2608.12939 (ACPC). FULL TEXT. Class PUBLISHED.`

They prove that action-conditioned rollout divergence *"bounds the perturbation-induced change in
multi-step prediction error **and planner cost**"*, and they measure the planner half directly:
**15.2 +/- 2.0 % reduction in CEM selection regret**, alongside 55.9 +/- 4.7 % reduction in
prediction-error MAE.

⭐ **The consequence for I-1: planner-side quality is partly predictable from a latent statistic
computed without running the planner.** I-1's quality half has been blocked on cost - you had to train
a value head and run two planners to learn anything. **A bound gives a pre-screen.**

⚠️ **Bound, not estimate.** A bound tells you when a latent is *too poor* to support good planning; it
does **not** rank two adequate latents. So this pre-screens *out*, it does not decide *for*. Stated
because the temptation to over-read it is obvious.

### F2 - Their planner is CEM, ours is iCEM - the transfer is close but the variance is not

Their regret result is on **CEM selection**. Our refav1 planner is **iCEM**, and `CLAUDE.md` records
that it **samples**, so *"the same checkpoint evaluated twice does not give the same answer"* with a
**seed floor of about 0.30 m ADE**.

⛔ **Therefore any I-1 quality comparison must vary the INFERENCE seed, not only the episode draw.**
`CLAUDE.md` names three different questions riding on one interval - episodes, training runs,
inference runs - and a fan-vs-value comparison on a sampling planner is squarely the third.
**An I-1 quality delta below roughly 0.30 m is not an effect on this rig.**

### F3 - The scorer is the family's largest published lever, and our ranking is still blind to the refined fan

From 2026-09-05: GuideFlow measures the **scorer** at **+15.9 EPDMS** against **+4.0** for all three
generative constraint mechanisms combined; and our own audit found *"the ranking never sees the refined
fan (201/201)"* (FS5-1).

⚠️ **This bears on I-1's framing.** I-1 contrasts a *learned terminal value* with *fan-scoring* as
alternatives. **The published evidence says the scoring stage dominates the generative stage** - so the
sharper question is not *value head OR fan-scoring* but **what the scorer sees**. ⛔ FS5-1 is 0 GPU and
already ready; **running I-1's expensive quality arm while the ranking is wired to the unrefined fan
would compare a value head against a deliberately handicapped baseline.**

### F4 - The deployment constraint that frames all of it

P-7 (2026-08-31, MEASURED): latency is **linear in K**, so K=300 costs about 350 ms to 1,015 ms against
a **100 ms** budget, and CUDA graphs' 2.57x does not close it. **Temporal abstraction is a deployment
requirement, not an efficiency preference.** Any I-1 outcome that only wins at large K is undeployable
regardless of plan quality.

---

## Five-dimension analysis - the I-1 transfer

| dimension | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ direct - it is the top injected row, and injected rows are exempt from production-relevance by the lane's own charter. It also touches L-3, L-4, FS5-1 and the 100 ms deployment budget. |
| **CONSEQUENCE** | Converts I-1's quality half from *"train value heads, then compare two planners"* into a **staged** experiment whose first stage is 0 GPU and can refute the line before any arm is spent. Changes the ORDER of three live backlog rows (FS5-1 before L-3). |
| **COMBINATION** | ⭐ Combines the day's A2 result (a latent diagnostic bounding planner cost) with our own 2026-09-01 cost measurement (breadth 5.94x cheaper, value head 1.2 %) and with 2026-09-05's scorer dominance. **Neither paper knows about our 100 ms budget or our 0.30 m inference-seed floor; both are ours and both bind the design.** The composition is: pre-screen with a latent bound, compare at matched latency, and only after the scorer is wired to the refined fan. |
| **CHANCES / RISKS** | **Upside:** a cheap gate that can close or advance a standing injected row without GPU spend. **Risks:** (a) a bound pre-screens out, it cannot rank - over-reading it would be the "true but wrong for the reader" class; (b) CEM is not iCEM and their regret number is not ours; (c) **the pre-screen shares a mechanism with the thing it screens** if the value head is trained on the same latent the diagnostic reads - `CLAUDE.md`'s "a check that shares the defect it checks for is green forever"; (d) their diagnostic ships with **no chance controls**. |
| **EXPERIMENT** | **Staged, both outcomes committed in advance.** **Stage 0 (0 GPU, banked latents and dumps):** compute the ACPC-style rollout-divergence statistic for the v7 arms, plus a **constant-predictor control that must read the no-information value** and a **raw-input floor**. ⛔ **Committed: if the divergence bound for our best arm already admits a planner-cost gap larger than the entire fan-vs-value effect we could hope for, the latent is the binding constraint and NO value-head arm is run this cycle - the finding is that I-1 is blocked on representation, not on scoring.** **Stage 1 (only if Stage 0 passes):** L-3 as written - value head vs fan-scoring **at matched LATENCY**, with **inference-seed replicates** (at least 3) so the delta is read against the 0.30 m sampling floor, and **only after FS5-1 has reconnected the ranking to the refined fan.** |

---

## What this changes for TanitAD - at most three recommendations

1. ⭐⭐⭐ **Re-order the queue: FS5-1 (reconnect the ranking to the refined fan, 0 GPU) BEFORE L-3.**
   Comparing a value head against a fan-scorer that never sees the refined fan measures our wiring
   defect, not the design question. This is the cheapest change on this page and it protects the
   expensive arm.
2. ⭐⭐ **Insert Stage 0 as I-1's gate.** A 0-GPU latent pre-screen that can refute the line before any
   arm is spent, with its refuting outcome committed in advance.
3. ⛔ **Bind I-1's quality comparison to INFERENCE-seed replicates and the 100 ms budget.** A delta
   under about 0.30 m ADE is not an effect on a sampling planner, and a win that only appears at large
   K is undeployable under P-7.

## Limits of this package

`Literature plus prior in-programme measurements - no new measurement was taken this pass.` The 15.2 %
and 55.9 % figures are the paper's, on their visual-control tasks with **no chance baseline reported**,
and neither is a TanitAD number. The 5.94x, 0.150 ms, 0.30 m and 100 ms figures are ours and carry
their existing artifact paths. **I-1's quality question remains OPEN** - this package changes what the
cheapest next step is, and does not answer the row. **The primary IS banked** (`kb_add`, rc=0); debt D-12 discharged.
