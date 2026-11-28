# THEORY_CRAWL, Amendment 1: P1 is refuted; T1 is restated and P1′ is pre-registered (Master Mind, 2026-11-06, dated amendment beside the frozen note `6e83451c`, written before any P1′ measurement)

## 1. What P1 measured (`wsp_read/P1_RESULT.json`, `P1B_RESULT.json`, `P1C_RESULT.json`; SPEC `52d74aa6`, recorded before any aggregate; [M], T2, 48 scenes, 93 starts)
* **P1 FAILS as written.** C0C's planned speed at +2 s lies ABOVE its executed speed in every occupied 3–8 m/s bin. mean(g − v) is:
  * +0.428 [0.112, 0.805] at s0;
  * +1.059 [0.405, 1.967] at s1.
* **No contraction below the identity**, and no fixed point below the human's median in range.
* **Leg C fails too.** P10 is not nearer the identity: 28 of its 279 rows run away above 40 m/s at tick 2.
* **Scope.** The T2 rollouts carry 3 ticks per start, so the map was read at an iteration depth of at most 2.
* **Root-cause class of the refuted mechanism:** T1 assumed the bias acts on the speed LEVEL (g(v) < v). The data say the plan accelerates, but less.

## 2. Post-hoc descriptives (not findings; each needs the pre-registration below)
* **The human's own band map** is +1.68 [1.37, 2.02] m/s per 2 s. C0C sits below it by:
  * −1.25 [−1.69, −0.83] at s0;
  * −0.62 [−1.30, +0.20] at s1.
* **T1-kin with the human's history:** plan minus logged is −0.46 / −0.49 m/s per 2 s, separated at both seeds.

## 3. T1′ (restated; [H])
* The squared-error planner outputs E[a | z], the conditional mean of the ACCELERATION.
* Where intent is unresolved by z, that mean mixes "accelerate as the human did" with "hold or brake", so the planned acceleration shrinks toward zero:
  * â ≈ π(z) · a_go + (1 − π(z)) · a_stay, with π the go-probability given z.
* The car still accelerates, just less than the human: under-acceleration, not decay. Over a 2 s horizon and repeated starts, this is a crawl relative to the human and to P10.
* **T2 (the frontier) and T3 (LoRA's scope) are unchanged by this.** They never depended on the sign of g(v) − v.

## 4. P1′ (pre-registered now; it can fail)
* **The window set.** On the T1-kin eval-141 windows with v0 in 3–8 m/s, take the shrinkage ratio ρ = â_plan / a_human over a 2 s horizon, restricted to windows with a_human > 0.5 m/s².
* **The confidence split.** Split the windows in two by the out-of-fold posterior of the banked go/stay probe on V+E (§7.31's probe, its own folds, never refit on these windows):
  * CONFIDENT: posterior ≥ 0.8;
  * UNSURE: posterior in [0.35, 0.65].
* **P1′ holds iff**, at both seeds:
  * median ρ(CONFIDENT) − median ρ(UNSURE) > 0;
  * separated, by an episode-cluster bootstrap with B 2,000, seed 0.
* **P1′ fails if** the difference is not separated at either seed, or is negative. Then the under-acceleration is not a mean over unresolved intent, and T1′ is wrong too.
* **NOT EVALUABLE** if either group holds fewer than 30 windows at a seed. That is named, never padded.
