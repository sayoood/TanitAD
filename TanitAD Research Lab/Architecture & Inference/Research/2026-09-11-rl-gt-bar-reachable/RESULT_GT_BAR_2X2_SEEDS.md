# The ≥GT bar's effect is WITHIN THIS RIG'S OWN NOISE at n = 2 — and the seed that made it look like a lever was an outlier

**Master Mind, 2026-09-11.** Dev-box RTX 4060, no spend. ⛔ **TIER T0, training-side.**
n = 2 per cell, no paired CI, non-parity corpus. ⛔ **Nothing here is a capability claim.**

---

## 1. The 2×2

| setting | seed | `frac_above_bar` | ΔR1 | ΔR2 (pp) | ΔR3 (%) | audit |
|---|---|---|---|---|---|---|
| bar OFF | 0 | null | +0.0264 | −0.410 | **+221.18** | clean |
| bar OFF | 1 | null | +0.0991 | +0.625 | **+76.02** | clean |
| bar ON | 0 | **0.1505** | +0.1238 | +0.990 | **+42.26** | clean |
| bar ON | 1 | **0.1371** | +0.0814 | +1.224 | **+38.61** | clean |

⛔ **The bar-OFF cell spans 145 percentage points of ΔR3 across two seeds that differ in
NOTHING but the seed.**

| metric | lever (on − off) | noise, OFF | noise, ON | verdict |
|---|---|---|---|---|
| ΔR1 | +0.0398 | 0.0727 | 0.0424 | ⛔ **within noise (0.55×)** |
| ΔR2 pp | +0.9993 | 1.0352 | 0.2344 | ⛔ **within noise (0.97×)** |
| ΔR3 % | −108.17 | 145.15 | 3.66 | ⛔ **within noise (0.75×)** |

---

## 2. ⛔ THE CORRECTION — I reported a seed-0 outlier as a lever

I wrote that the bar cut drift **5.2×**, from +221 % to +42 %, and called two seeds agreeing
*"the first encouraging sign this stage has produced."* ⛔ **That was reading one draw.**

At seed 1 the **same bar-OFF setting** produces **+76.02 %**. The +221 % was not the baseline's
behaviour; it was one sample from a cell whose own spread is **145 points**. Against that, a
−108-point "improvement" is **0.75× its own noise**.

⭐ This is exactly what `H-ESTIM-SEED-1` exists to catch, arriving from the direction nobody
watches: not a separated CI on a one-seed arm, but a **large, clean, replicated-looking effect**
that dissolves the moment the control gets a second seed. And it is the WP-D lesson again — an
arm with **zero levers moved** once read "separably worse" on **5 of 9** family metrics.

⚠️ **The comparison I ran is the right one and the interval would not have caught this.** The
paired episode-cluster bootstrap resamples **episodes with the models held fixed**; it answers
*"would another draw of episodes say this?"* and is **structurally blind** to *"would another
training run say this?"*. Only the replicate sees it.

---

## 3. ⭐ WHAT SURVIVES, AND THE HYPOTHESIS IT CREATES

**Solid.** `frac_above_bar_mean` reads **0.1505** and **0.1371** — the mask genuinely admits only
about **14 %** of sampled anchors, consistently across seeds, where every prior arm read `null`.
The mechanism is real, reached, and doing what it says.

⭐ **The asymmetry the mean test throws away.** Bar-ON's two seeds span **3.66** points of ΔR3;
bar-OFF's span **145.15**. That is a **40× difference in spread**, in the opposite direction from
the noise, and a test built to compare *means* discards it.

⇒ ⚠️ **HYPOTHESIS, not a result: the ≥GT bar may STABILISE the arm rather than shift it.** That
is a coherent mechanism — a truncation that admits only candidates beating the human demonstration
removes exactly the low-quality positive gradients that make an unconstrained run wander. ⛔ **n = 2
cannot distinguish a real variance reduction from two lucky draws**, and a spread computed from two
points is not a spread.

⇒ **Seeds 2, 3 and 4 are running for BOTH settings**, taking each cell to **n = 5**. ~20 minutes,
zero spend. ⛔ **Until they land, no claim is made in either direction.**

---

## 4. What this does NOT change

* ⭐ The bar is **reachable and mutation-proven** — bit-identical gradient below the bar, exactly
  `0.0` above it, partial masking in between. That result stands on identities, not on seeds.
* The **separation check** on the reward audit stands: `hackable` FLAGGED with `winners == []`,
  `default` clean, watched in both directions.
* The two defects found in passing stand: a guard that existed **only inside its own test** while
  the register called it closed, and `--noise-mode` being unreachable from the caller.

⛔ And the standing blocker is unchanged, only sharper: **seeds, not plumbing** — and now with a
measured reason, because this rig's bar-OFF cell moves 145 points on nothing at all.
