# PRE-REGISTRATION — MM-E16: does a TEMPORALLY-STRIDED predictor reach the tactical band?

**Written 2026-08-30, BEFORE any arm runs** · Master Mind · **Tier** T0 primary,
T1 escalation pre-committed · **Depends on** MM-E15 (D-HORIZON-LADDER).

```yaml
hypothesis: MM-E16
question: MM-E15 measured that no rollout in the v7 recipe reaches even the NEAR
          EDGE of the tactical label band (2.0 s), and that effective imagination
          is ~0.2 s against a strategic referent of 12.4 s median. Does a predictor
          that steps at a COARSER dt reach the band without more rollout compute?
one_variable: the predictor's effective step, dt_eff 0.1 s -> 0.5 s at matched k
held_constant: [encoder, window, o5_k, o1_k, batch, lr, steps, seed, corpus,
                w_o1_ctrl, in_channels, cond_param]
```

## 1. Why this arm, and why NOT simply a longer rollout

The naive reading of MM-E15 is *"extend `o5_k`"*. ⛔ Two reasons that is the wrong
first move:

1. **It extends a predictor that already collapses.** MM-E10: `max|h1−h2| = 1.211`
   vs `max|h2−h4| = 0.00179`, ~680× smaller. Rolling a collapsed predictor further
   buys more of the same guess.
2. **Cost.** Reaching the strategic band at `dt 0.1` needs `o5_k 80` — 10× the
   rollout compute, on the arm we already know is degenerate past one tick.

⭐ **The bands argue for temporal abstraction instead.** A strategic predictor
stepping at ~1 s reaches 12.4 s in ~12 steps — the same rollout budget the operative
level spends on 1.2 s. The label bands even supply the ratio: **1 : ~3 : ~15**
(operative 0–2 / tactical 2–6 / strategic 8–30), read off the corpus rather than
chosen. This arm tests the cheapest rung of that ladder.

⚠️ **But `--o5-k 20` MUST RUN FIRST AND SEPARATELY.** It is the code's own default,
it exactly spans the operative band, and it is one flag. Confounding the stride test
with a rollout-length change would repeat the `--v2` conflation failure (ten levers,
two axes, non-attributable result). **Order: MM-E11 completes → `o5k20` arm →
MM-E16.**

## 2. Arms

| arm | change from `postrain30k`'s recorded line |
|---|---|
| `o5k20` | `--o5-k 20` only (the code default; operative coverage 40 % → 100 %) |
| `stride5` | `dt_eff 0.5 s` at matched k (tactical reach 4.0 s), everything else held |
| **deliberate-regression** | `dt_eff 0.5` with the **context window unstrided** — the arm that looks like the fix but leaves the input at 0.6 s |

⭐ The third arm is the one that makes a PASS mean something: if the gate cannot fail
a stride applied to the output while the input still sees 0.6 s, then a pass on
`stride5` tells us nothing about whether the *representation* reached further.

## 3. Reads and outcomes, COMMITTED IN ADVANCE

**Primary (T0):** predictor separation across horizons — the MM-E10 collapse
statistic, same instrument, same corpus, same n. Incumbent:
`max|h1−h2| = 1.211`, `max|h2−h4| = 0.00179`.

| outcome | criterion | consequence |
|---|---|---|
| **STRIDE-WORKS** | `max\|h2−h4\|` rises **≥10×** off the floor AND the deliberate-regression arm does NOT | temporal abstraction enters the v7 recipe; design the 3-rung ladder at 1 : 3 : 15 |
| **STRIDE-INSUFFICIENT** | rises but **<10×** | report the number; do not adopt alone |
| **STRIDE-INERT** | unchanged within noise | ⛔ the collapse is NOT a step-size problem — it is the predictor's capacity or objective, and the next lever is neither stride nor rollout length |
| 🔶 **VOID** | the regression arm rises too | the statistic is measuring the stride, not the reach — instrument fault, no verdict (C160) |

⛔ **ANTI-GATE, committed before any number exists.** Per the banked precedent (an arm
**10.7× worse** on open-loop next-action MSE was **2.3× better** closed-loop), and per
MM-E11's own anti-gate: **a T0 prediction regression on a coarser-stride arm is
EXPECTED and is not grounds to reject it.** A strided predictor is solving a
different, harder one-step problem. If drift or nrmse worsen while horizon separation
improves, that is the intended trade and it escalates to **T1**.
⚠️ **The limit:** a T0 regression with **no** separation gain is a dead arm.

## 4. Controls that must read known values, or the panel is VOID

* **C0 identity** — same inputs twice ⇒ spread exactly 0.0 (the forward is
  deterministic; without this the small numbers are noise, not absence).
* **C1 scene reference** — the denominator must stay LARGE. If it collapses too, the
  ratio is 0/0 and the result is VOID, not a finding. *(This is what refuted the
  "degenerate rollout" account in MM-E10.)*
* **Deliberate regression** — §2 arm 3 must FAIL.
* **n and d printed.**

## 5. What this cannot show

Nothing about driving (T0), and nothing about whether reach converts into skill —
that needs T1 with the hold-action control. ⚠️ Single seed: deltas smaller than the
(unmeasured at 30k) seed band are **unresolved, not null** (the MM-E8 lesson). The
≥10× criterion is set far outside plausible seed noise precisely because the
incumbent's `h2−h4` sits at 1e-3.

⚠️ **And a scope limit that must travel with any result:** these are ~19 M v7-TINY
arms on stage S-W with every planner objective at zero. They are a world-model trunk
plus a readout and were never trained to drive. A stride result here is evidence about
the **predictor's reach**, never about the hierarchy's performance.
