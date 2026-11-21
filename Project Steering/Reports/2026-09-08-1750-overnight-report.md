# Overnight report — 2026-09-07 evening to 2026-09-08 17:50 UTC

**For the PI.** You asked to be told when the implementation is finished and to get a report. This
is the report; **the implementation is not finished, and §5 says exactly what is left and what is
waiting on you.** 16 commits, `cb84c07 → bdc2693`. ⛔ Nothing was pushed; nothing touched `main`.

## 1. What is running right now

| arm | where | state | lands |
|---|---|---|---|
| **refcv5-v2** (40 k) | A40 `tanitad-refcv3` | **step 37,050 / 40,284**, stderr **1458 B unchanged for 20 h**, 7 procs | **~21:30 UTC / 23:30 Berlin** |
| **WP-D panel** D0/D1/D2 | Thor | ✅ **ALL THREE COMPLETE**, 4,000 steps each, **stderr 0 B on every arm**, 1 launch each, 0 relaunches | done 11:54 UTC |
| **WP-D D0b** (replicate) | Thor | launching now — required before criterion A3 can be read at all | ~4.7 h |
| **WP-D probe** | dev-box 4060 | running — the pre-registered A1–A4 evaluation | ~50 GPU-min |
| **WP-B** (waypoint index) | dev box, no GPU | implementing | — |

## 2. The three results that matter

### 2.1 ✅ The landing comparison is safe from sampler noise — 155×

refcv5-v2 samples at eval (`--sampler ddim`) and refcv4b does not, so inference noise enters the
paired delta **from one side only** and nobody had measured it. Three rolls of refcv5-v2's own
mid-training checkpoint, one variable:
* ✅ the model-free controls read **EXACTLY 0.00000** across all three seeds — that is a passing
  control, not a close one — while every model arm moved, which also proves the sampler really is
  reached at eval;
* ⭐ **the floor is METRIC-SPECIFIC and scales with how many stochastic arms a statistic
  differences**: 0 → 0.00000 · 1 → 0.00073 · 2 → 0.00223–0.00364;
* ✅ the bar (`os − ha0_ext`) clears its floor **155×**, separated at all three seeds.
⛔ The nav effect does **not** clear it — **0.64× its own floor, with a sign that flips between
seeds** ⇒ `NOT MEASURED`, not "nav does nothing". Only a bound survives: |effect| < ~0.005 m.
⭐ Banked at one seed it would have published a direction that does not exist.

### 2.2 ⛔ refcv5-v2 is training 11,286 parameters that receive NO gradient

The 22-token tactical head is **built, rollable, and never learns** (`grad_abs_sum` **0**, 2/2
grads `None`). ⭐ The arm is **not** tactically blind — the factored `tac_goal_head` trains
normally (`grad_abs_sum` 14.368); it is the **token** head that is inert.
⇒ ⛔ **No landing result may be attributed to "the 22-token tactical vocabulary."** Its real levers
are `--sel-refined`, `--sampler ddim`, P14 fan ranking, `--anchor-v0-conditioned`, `--nav-from-v7`.
⭐ Why two guards stayed green: one asks *"is it built?"* (it is); the other enumerates **declared
loss weights**, and this head **has no weight flag**, so it produced no row at all. Neither guard
was wrong — both were asked a question narrower than the claim resting on them.

### 2.3 ✅ WP-D was built, and it went from design to a complete 3-arm panel in one night

A polar 24×20 agent-occupancy auxiliary loss, training-only, removability proven. Its census is
the substantive finding: **27.958 % of GT-occupied cells are agent-occluded**, so the two-state
target anyone would write by default asserts that **more than a quarter of the agents in the grid
are free road**. No occlusion label existed anywhere — the join's `occ` column is the FOV mask —
so it had to be derived.

## 3. What I got wrong, and who caught it

⭐ Recorded because the pattern is the point: **every one of these returned something PASS-SHAPED.**
* The landing harness was **blind** to `n_anchors` and would have printed a complete-looking panel
  with the tactical family's anchor half **silently absent**.
* The pre-launch currency gate reported **no drift after scanning nothing**.
* A wrapper reported `GATE_EXIT=0` for a **timed-out, output-less** gate — `$?` after a pipeline is
  `tail`'s status. I made the same mistake myself hours earlier on a different tool.
* WP-A's all-zero control was published as reading its known value **"EXACTLY"**; it reads
  **+0.881 %** high. Retracted (`R-2026-09-07-ap-ties`). Class: **a control biased by the same
  estimator it validates**. ⭐ The ladder and every conclusion stand; only the word is withdrawn.
* **Twice I passed an agent a stale premise** — "refav1 is training on Thor" (it finished
  2026-09-04, freeing a GPU the programme thought it did not have) and "Thor already holds what
  the arm needs" (the join it held shares only **182 clips** with v7.2). ⚠️ The second would have
  failed **quietly**: a 4 %-coverage run reads as *"the aux loss does not help."*
* Five steering files carried an index blob in **neither HEAD nor the worktree** — one ordinary
  commit from reverting **397 lines**, including both open decision items.

## 4. What is NOT finished

**WP-B** — DiffusionDrive's waypoint-indexed cross-attention, into the **sparse** agent tokens, not
a dense raster. Being implemented now. This is the actual remaining build.
**WP-C** — ⛔ **waiting on you**, see below.
**Item 10's wiring** — buildable defaulted OFF; the weight budget is yours.

## 5. ⛔ WHAT NEEDS YOUR WORD — `PI_DECISION_QUEUE.md`, 11 items, each with a default

The two that actually gate work:

**Item 11 — does "finish the implementation" authorise WP-C?** It reached me as a **relay**, and a
relay cannot authorise, so I did not act on it. ⭐ WP-B needed no new authorisation and is running
under your own earlier instruction. ⛔ But WP-C (agent seam ON at 108 M / 40 k) is **compute spend
on a design that already returned a verdict**: its two-seed gate FAILED, and the deliberate-
regression arm reproduced the whole degradation ⇒ the cost was the **auxiliary task competing for
a 17 M trunk**, not the agent information. Whether that vanishes at 108 M is a **hypothesis**.
*Default if silent: not launched.*

**Item 10 — the `MANEUVER_WEIGHT` budget.** Wiring the dead tactical head costs loss weight taken
from somewhere. *Default if silent: wiring built, left off; nothing spends against it.*

*Every commit message carries its own evidence and artifact paths; the queue carries the
decisions. Both survive a lost session, which is why they are the channel.*
