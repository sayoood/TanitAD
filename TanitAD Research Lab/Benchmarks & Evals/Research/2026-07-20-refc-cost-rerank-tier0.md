# REF-C v1.0 — cost re-ranking, training-free: **0 % of the selection gap is recoverable**

**Date:** 2026-07-20 · **Pod:** `tanitad-eval` (A40, idle) · **Corpus:** canonical val
`physicalai-val-0c5f7dac3b11`, **881 windows**, window 8 / stride 8 / nav=follow — the same
harness (`bench.run`, 8-split episode-disjoint jackknife) as every leaderboard row.
**Cost:** one 27 s GPU decode pass + CPU sweeps. **Zero training.**

---

## 0. The answer, in one line

> **Zero.** Re-ranking REF-C's own 256-proposal fan with the P2 planner cost recovers
> **0.0 % of the 0.307 m selection gap**. The best point on the entire confidence↔cost blend
> curve is **λ = 0 — i.e. the unmodified baseline**. Pure cost re-ranking is **−171 %**
> (ADE@2s 0.471 → 0.999). The best result found anywhere across every cost variant tried is
> **0.73 %** of the gap (0.4714 → 0.4692), which is ~1/40 of the metric's own CI95.
>
> **This is a clean negative, and it is decisive: the hand-built cost is a strictly weaker
> ranker than the confidence it would replace.** It is *not* noise — it beats chance
> comfortably — it is simply beaten by the model's own logits.

---

## 1. Identity — why this comparison is interpretable

| | |
|---|---|
| Registry key | **`refc-v10`** |
| Display name | **REF-C v1.0 (cost re-rank, training-free)** |
| Weights | **`refc-xl-30k`, byte-for-byte** — `/root/models/refc-xl-30k/ckpt.pt`, md5 `966d4eff1ea5ddf86efba01b8344e198`, step 29 999 |
| Parameters added | **0** · GPU-hours of training | **0** |
| What changed | **only** the decode-time selection rule: `argmax(anchor_logits)` → cost / blended re-rank over the same 256 already-refined proposals |

Because v1.0 and `refc-xl-30k` share weights byte-for-byte, **every delta measured here is
attributable purely to the selection policy.** That is exactly what makes v1.0 the clean
control for **REF-C v1.2** (frozen decoder + a *learned* re-scorer): v1.2's delta over v1.0
is the value of *learning* the ranker, with the proposal set and the trunk held fixed.

**Baseline parity check (must pass or the experiment is void):** the λ = 0 row reproduces the
published row **exactly** — heldout **0.4577** (published 0.4577), full-set **0.471439**
(published 0.471439), and `sel_idx == argmax(anchor_logits)` asserted on every batch.

---

## 2. ⚠️ Two numbers in the brief were single-clip, not canonical — corrected here

The brief carried `oracle-in-fan 0.295` and `65 % of frames >2× worse`. Those are the **ep11
stride-1 clip-wide** figures from `plan_fan.py` (a deliberately selected failure clip). On the
canonical 881-window val the same quantities are:

| Quantity | Brief (ep11, stride-1) | **Canonical val (881 windows)** |
|---|---|---|
| ADE@2s selected | 1.110 | **0.4714** full / 0.4577 heldout |
| ADE@2s oracle-in-fan | 0.295 | **0.1640** full / 0.1642 heldout |
| Selection gap | 0.815 | **0.3075** |
| `frac_sel_2x_worse` | 0.65 | **0.454** |

All percentages below use the canonical denominator **0.3075 m**.

---

## 3. The λ curve — the whole thing, not just the best point

`select = argmin_i [ −log p_i + λ · z(J_i) ]`, with `J` from `planner_p2.cost_fn` **verbatim**
and `z(·)` the per-window z-score across the 256 candidates (so λ is scale-free: "cost-sigmas
per nat of log-probability").

| λ | ADE@2s full | ADE@2s heldout | `frac_sel_2x_worse` | mean #candidates ranked above the pick by confidence |
|---:|---:|---:|---:|---:|
| **0 (baseline)** | **0.4714** | **0.4577** | 0.454 | 0.00 |
| 0.02 – 0.5 | 0.4714 | 0.4577 | 0.454 | 0.00 |
| 0.75 | 0.4716 | 0.4569 | 0.454 | 0.00 |
| 1.0 | 0.4716 | 0.4569 | 0.454 | 0.00 |
| 1.5 | 0.4744 | 0.4626 | 0.459 | 0.01 |
| 2.0 | 0.4759 | 0.4664 | 0.462 | 0.01 |
| 3.0 | 0.4773 | 0.4660 | 0.464 | 0.02 |
| 5.0 | 0.4893 | 0.4923 | 0.477 | 0.04 |
| 10 | 0.4983 | 0.5104 | 0.482 | 0.05 |
| 30 | 0.5349 | 0.5789 | 0.493 | 0.15 |
| 100 | 0.5745 | 0.6608 | 0.495 | 0.33 |
| ∞ (**pure cost**) | **0.9989** | 1.2259 | **0.647** | 9.28 |

The curve is **monotone-degrading**. There is no interior optimum: the blend is a no-op until
λ ≈ 1.5 (REF-C's logits are sharp — mean top-1 prob 0.654, top1−top2 logit gap 1.56), and the
moment λ is large enough to flip a pick, the flip is harmful on average.

**Top-K variant** (restrict to the K most-confident, then take min-cost) — same verdict:

| K | 1 | 2 | 3 | 4 | 8 | 16 | 32 | 64 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| ADE@2s full | **0.4714** | 0.4862 | 0.5852 | 0.6280 | 0.7836 | 0.9235 | 0.9755 | 1.0011 | 1.0067 |

**Sensitivity — the negative is not an artefact of the densifier or the weights.**
`natural`-start spline instead of the v0-clamped one: identical to 4 d.p. through λ = 0.75.
Term ablation `w_c = w_s = 0` (speed + progress only, needs no derivative of the interpolant):
best point **0.4692 at λ = 3** = **0.73 % of the gap**, still inside the noise.

---

## 4. Per-stratum — the mechanism is **confirmed as a diagnosis and refuted as a remedy**

Longitudinal strata are cut on the GT's own accel over the scored horizon
(`a_gt = (v[t+2s] − v[t]) / 2`; braking ≤ −0.5 m/s², accelerating ≥ +0.5).

| Stratum | n | baseline | oracle-in-fan | **pure-cost re-rank** | Δ |
|---|---:|---:|---:|---:|---:|
| **braking** | 110 | **1.0626** | 0.2065 | 1.5757 | **+0.513** |
| accelerating | 160 | 0.7194 | 0.2177 | 1.4930 | +0.774 |
| steady | 611 | 0.3001 | 0.1422 | 0.7657 | +0.466 |
| high speed | 294 | 0.3243 | 0.1461 | 0.4573 | +0.133 |
| med speed | 293 | 0.4989 | 0.1673 | 0.8003 | +0.301 |
| low speed | 294 | 0.5912 | 0.1785 | 1.7386 | +1.147 |
| high / braking | 29 | 1.0306 | 0.2099 | 1.2211 | +0.191 |
| low / braking | 46 | 1.0517 | 0.2142 | 2.1229 | +1.071 |

*(At the best blend λ = 0 every Δ is exactly 0.000 by construction — the table above is the
pure-cost arm, which is the only one that actually moves picks in every window.)*

**Confirmed:** the error really does concentrate where the brief predicted. Braking windows are
**3.5× worse than steady** (1.063 vs 0.300), and the Frenet decomposition of the *selected*
plan is **0.420 m along-track vs 0.120 m cross-track — 78 % longitudinal**. REF-C is indeed
defaulting to something constant-velocity-shaped when the ego decelerates.

**Refuted:** the VTARGET-tracking cost makes those very windows **worse, not better**
(+0.513 m on braking). It is directionally wrong there, and §5 says why.

---

## 5. Why it fails — three measurements, not three opinions

**(a) VTARGET is a 10–20 s set-speed, and this is a 2 s selection problem.**
Mean VTARGET is **+1.42 m/s above v0**. As a predictor of the next-2 s mean speed it is
*worse than simply holding v0*: corr 0.957 / MAE **1.65 m/s** vs v0's corr 0.997 / MAE
**0.475 m/s**. So `w_v (v̂ − v_target)²` systematically rewards *faster* plans on exactly the
braking windows the cost was supposed to rescue. (The fixed mint was used as instructed —
`tanitad.lake.vtarget.vtarget_v2`, 5 s lookahead floor, explicit `valid` mask. **18.5 %** of
windows fell back to v0-hold; mean realised lookahead 107 steps = 10.7 s. The defect is not in
the mint, it is in using a set-speed as a 2 s reference.)

**(b) Even a *perfect* speed target would not help — speed is a scalar, and the fan is not.**
A **GT-cheating** ranker that picks the candidate whose mean speed exactly matches the GT's
scores **1.1236 m — far worse than the 0.4714 baseline.** Dozens of the 256 anchors have the
right arc length and the wrong shape; a speed-only criterion cannot tell them apart, and the
remaining P2 terms (comfort, progress) carry no lateral information at all. This is P2's own
stated honest scope — "no lateral / route / goal term" — turning fatal when the proposal set
is a wide multimodal fan rather than a locally-perturbed CEM population.

**(c) Longitudinal information alone caps out at 34 %.**
A GT-cheating **along-track-only** ranker reaches 0.3669 → only **34 %** of the gap.
A GT-cheating **cross-track-only** ranker reaches 1.5199 — *worse than baseline*. Neither axis
alone is the lever; the recoverable signal is **joint** geometry.

**(d) The cost is not noise — it is simply beaten.**
Spearman(cost, ADE) = **0.881** across the fan; Spearman(−confidence, ADE) = **0.907**.
Inside the top-K confidence set the cost beats a coin flip by a wide margin, and still loses to
just taking the top-1:

| K | oracle | **cost pick** | chance (mean of set) | anti-oracle | conf pick (= deployed) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.3421 | 0.4862 | 0.5638 | 0.7855 | **0.4714** |
| 4 | 0.2506 | 0.6280 | 0.7444 | 1.3293 | **0.4714** |
| 8 | 0.2026 | 0.7836 | 1.0336 | 2.2789 | **0.4714** |
| 32 | 0.1713 | 0.9755 | 2.5033 | 8.3635 | **0.4714** |

At K = 8 the cost closes **30 %** of the chance→oracle span; the model's own confidence closes
**68 %**. **The confidence head is ~2.3× the ranker the hand cost is.** Replacing it was always
going to lose; the only question was by how much, and the answer is 0 % gained at best.

---

## 6. `frac_sel_2x_worse`, before vs after

| | baseline | best blend (λ = 0) | pure cost |
|---|---:|---:|---:|
| `frac_sel_2x_worse` | **0.454** | 0.454 | **0.647** |

45.4 % of canonical-val windows pick a plan >2× worse than one already in the fan. Cost
re-ranking does not reduce that; pure cost raises it to 64.7 %.

---

## 7. Honest ceiling — and what it says about spending GPU on a learned ranker

**The 0.164 oracle is unreachable, and part of it is a lottery.** It is a *min over 256 draws*
from a proposal distribution whose *typical* member is **13.9 m** off. Min-of-K over random
subsets: K=1 → 13.92, K=8 → 2.63, K=32 → 0.864, K=128 → 0.325, K=256 → 0.225 (with
replacement; the exhaustive min is 0.164). A large slice of "oracle-in-fan" is the statistics
of holding 256 tickets, not evidence of a systematically better plan waiting to be found.

**But the reachable part sits right at the top of the confidence ranking, and that is the
number that matters for v1.2.** Oracle *restricted to the K most-confident candidates*:

| K | 1 | **2** | **4** | **8** | 16 | 32 | 64 | 256 |
|---|---|---|---|---|---|---|---|---|
| best-in-set ADE@2s | 0.4714 | **0.3421** | **0.2506** | **0.2026** | 0.1819 | 0.1713 | 0.1679 | 0.1640 |

**A perfect re-scorer over only the top-8 would reach 0.203 — 87 % of the whole selection gap,
and it would beat flagship v1 (0.452) by 2.2×.** The information is present, it is dense at the
head of the ranking (top-4 already yields 72 % of the gap), and it is reachable with an
8-candidate forward pass.

**Verdict on Tier 1/2 GPU spend: YES, but the ranker must be LEARNED and must see geometry.**
- The **best GT-free ranker we could build by hand recovered 0.0 %** — so "just write a better
  cost" is not the answer, and neither is a weight sweep (three cost variants, two densifiers,
  17 λ values, 10 top-K values: nothing crossed the noise floor).
- The ceiling for the v1.2 design (frozen decoder + learned re-scorer over the top-K) is
  **0.203 @ K=8 / 0.251 @ K=4**, and the gap between "confidence ranks it 68 % of the way" and
  "perfect ranks it 100 %" is exactly what a learned head has to close.
- Any such head must be trained on **joint** along+cross error, not on a speed target: §5(b)/(c)
  show that speed-only supervision — even with ground truth — is *negative*.
- This also independently corroborates flagship v1.5's finding that fixing the ranking **input**
  is insufficient. Here we fixed the ranking **rule** and it was also insufficient. What is left
  is the ranking **objective**.

---

## 8. Reproduction

```bash
ssh tanitad-eval
cd /root/taniteval && PYTHONPATH=/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
  python3 -m taniteval.refc_rerank dump      # 27 s, one GPU pass, keeps the full fan
  python3 -m taniteval.refc_rerank analyze   # CPU: λ sweep, top-K, strata -> results/refc-v10.json
  python3 -m taniteval.refc_rerank diag      # CPU: ceilings/controls -> results/diag_refc-v10.json
```

Reused verbatim, never reimplemented: `planner_p2.cost_fn` + `planner_p2.W`,
`pathspeed.step_speed/heading_deg/arclength/frenet_residual`, `closedloop.WHEELBASE`,
`refc_eval`'s exact decode call, `bench.run`, and `tanitad.lake.vtarget.vtarget_v2`
(md5 `a1bd0ca813ab0a0578d08c4337a9e569`, identical to the repo copy — the module **refuses** to
fall back to the defective `planner_p2.vtarget_for` mint).

**The one modelling assumption, stated:** REF-C's proposals are 4 waypoints while `cost_fn`'s
comfort terms live at the 10 Hz tick, so each proposal is densified to a 20-step dt=0.1 path by
a cubic spline through (0,0) + the 4 waypoints, **clamped at t=0 to the observed entry speed
v0** (so first-step braking is information, not artefact), natural at t=2 s. The densifier is a
fixed linear operator, self-tested against a direct solve at import. The `natural`-start
variant and the derivative-free `w_c = w_s = 0` ablation both reproduce the negative (§3).
