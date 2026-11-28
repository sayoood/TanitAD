# What relaxing Ga would actually cost: the open-loop regression is EVENLY SPLIT lateral/longitudinal, and TAIL-HEAVY laterally

**MEASURED 2026-07-28**, extracted from the four committed frontier artifacts. **No GPU, no new run** —
this decomposition was computed by `e1c_eval.py` on **every arm** and has been sitting in
`points[step].M1_lateral_split.openloop.ego` since E1c.
**Estimator: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000), 967 windows / 44 held-out
episodes, ego frame. `overlapping_holdout_se` used nowhere.

⚠️ **This is a reporting gap of mine, not a new measurement.** The standing discipline requires
lateral/longitudinal decomposition; I reported scalar `ADE@2s` for four consecutive experiments while
the split was already in the artifacts. Recording that plainly.

## Why it matters now

Four levers have failed to reach **Ga** (open-loop ADE@2s not separated-higher), and the open question
put to the PI is whether Ga is the right guardrail at all. **That judgement needs to know WHAT is
being traded, not just how much.** A regression concentrated in one axis is a different decision from
one that degrades both.

## The answer (paired delta, ft − base, step 4000 of each arm)

| arm | `cross_abs@2s` (lateral) | `along_abs@2s` (longitudinal) | `ade_over_knots` |
|---|---|---|---|
| **E1c** λ=1, full buffer | +0.1611 [+0.113, +0.210] ✅sep | +0.1838 [+0.109, +0.270] ✅sep | +0.1947 |
| **E1e-A** λ=3, full | +0.0971 [+0.056, +0.142] ✅sep | +0.0926 [+0.032, +0.162] ✅sep | +0.0990 |
| **E1e-B** λ=8, full | +0.0500 [+0.022, +0.084] ✅sep | +0.0520 [+0.010, +0.104] ✅sep | +0.0500 |
| **E1f** λ=3, junction-only | +0.0460 [+0.019, +0.078] ✅sep | +0.0702 [+0.019, +0.139] ✅sep | +0.0555 |

⇒ **The regression is essentially EVENLY SPLIT between lateral and longitudinal in every arm**, and
**both are CI-separated in every arm**. There is no cheap one-sided trade available: relaxing Ga means
accepting degradation on **both** axes.

## ⭐ And the lateral damage is TAIL-HEAVY

| arm | `cross_abs@2s` (mean) | `cross_p90@2s` (tail) | amplification |
|---|---|---|---|
| E1c | +0.1611 | **+0.3945** | **2.45×** |
| E1e-A | +0.0971 | +0.1849 | 1.90× |
| E1e-B | +0.0500 | +0.0756 | 1.51× |
| **E1f** | +0.0460 | **+0.0422 [−0.0050, +0.1142] — NOT separated** | 0.92× |

The artifact's own guidance says to gate on p90/p95/max rather than the mean
(*"MEASURED: mean 0.25 m vs p90 1.40 m on the same windows"*). By that reading **E1c's true lateral
cost is ~0.39 m at the 90th percentile, not the ~0.16 m the mean suggests** — and a lane half-width
is ~1.75 m, so a 0.39 m p90 shift is a meaningful fraction of the corridor.

⭐ **The amplification shrinks monotonically as the regression shrinks — and E1f alone shows none.**
E1f is the only arm whose lateral tail is **not** separated from base: it degrades the lateral *mean*
slightly while leaving the *worst-case* lateral behaviour intact.

## What this contributes to the PI's decision

- **Relaxing Ga is a two-axis concession**, not a lateral-only or longitudinal-only one.
- **The honest cost of the strongest closed-loop arm (E1c, −0.4407 departure) is ~0.39 m at the
  lateral p90**, which is where a corridor metric would feel it.
- **The cheapest arms (E1e-B, E1f) cost ~0.05 m mean on both axes**, and E1f additionally leaves the
  lateral tail unseparated — the mildest open-loop footprint measured, though it also delivered no
  overall-corridor gain (E1F_RESULT §2).

⚠️ **What this does NOT do:** it does not argue for or against relaxing Ga. It supplies the missing
axis-level cost so the judgement is made on the trade's actual shape. **No arm should be run against
Ga until that judgement is made.**

## Bounds

- 967 windows / 44 episodes; `cross_p90` is a tail statistic and its CIs are correspondingly wide
  (E1c: [+0.188, +0.558]).
- All four arms are **fine-tunes of one base checkpoint**; none is a from-scratch result.
- Step 4000 chosen for every arm for comparability, **not** each arm's best point — E1c's best
  closed-loop point is step 2750, whose open-loop cost is +0.2158 rather than +0.1947.
