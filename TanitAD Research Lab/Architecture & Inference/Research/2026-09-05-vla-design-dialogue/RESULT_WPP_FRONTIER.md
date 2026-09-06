# WP-P — the influence/consistency frontier, measured: raising authority alone cannot deliver both

**TanitAD_TrainingFlyWheel · 2026-09-06 · Tier T0, NON-PARITY pilot.**
Evidence class **MEASURED (ours)**. `n = 1,360` windows · 34 episodes · K = 128 ·
`refc-base-30k` · verified join · **zero GPU** (closed-form sweep on banked tensors).
Code `code/wpp_frontier.py` · raw `raw/wpp_frontier.json`, `raw/wpp2.log`.
Answers the quantitative half of `DIALOGUE_07` Amendment 5 §A5.3.

---

## 0. The question, and why it is answerable at all

The PI's requirement is a CoT that **influences** behaviour while staying **consistent**
with the trajectory finally selected. The frontier literature's verdict on that pair is
verbatim *"Neither design satisfies both desiderata"* — but it compares **two discrete
models**. The Energy Bridge has a **continuous dial**, `β` in `S1 = S0 − β·E`, so the
trade-off can be **traced** rather than argued about.

⭐ And on a real trained model it costs nothing: `β = 1` reproduces the shipped
`refc-base-30k` exactly (asserted in the run), `β = 0` is the port-ablated base policy, and
everything between and beyond is arithmetic on tensors already on disk.

---

## 1. The frontier

`sd[S0] = 17.907` · `sd_a[E] = 1.457` · **shipped authority = 8.1 %** · chance rank = 64.5.

| β | authority | INF flip | INF kl | **CON rank** | CON top-1 | **ADE (m)** | ADE cross-ep |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 % | 0.0000 | 0.0000 | 75.80 | 0.0000 | 0.6636 | 0.6636 |
| **1.0** | **8.1 %** | 0.2324 | 0.188 | 64.40 | 0.0000 | **0.6521** | 0.6774 |
| **2.0** | **16.3 %** | 0.4044 | 0.722 | **53.66** | 0.0037 | **0.6583** | 0.7325 |
| 5.0 | 40.7 % | 0.8287 | 6.56 | 14.33 | 0.2890 | **1.4864** | 1.3288 |
| 10.0 | 81.4 % | 0.9596 | 11.47 | 4.00 | 0.5044 | 2.2410 | 2.3904 |
| 20.0 | 162.7 % | 0.9963 | 17.13 | 1.67 | 0.6713 | 3.6993 | 3.6024 |
| 40.0 | 325.5 % | 0.9985 | 19.34 | 1.42 | 0.7287 | 4.6373 | 5.0664 |
| 80.0 | 651.0 % | 1.0000 | 24.23 | 1.18 | 0.8515 | 7.3042 | 7.1542 |
| 160.0 | 1301.9 % | 1.0000 | 24.91 | 1.15 | 0.8647 | 7.6221 | 7.6612 |

---

## 2. What the curve says

### ⭐ 2a. Consistency becomes measurable at β = 2, and the operating window is narrow

At the **shipped** authority (8.1 %) `CON rank` is **64.40** against a chance of **64.50** —
literally nothing, exactly as A5.3 predicted from arithmetic. Doubling the authority to
**16.3 %** moves it to **53.66**, and the ADE is **0.6583 m**, still **better than the base
policy's 0.6636 m**.

> ⇒ **The usable window is β ∈ [1, 2] — authority 8–16 %.** Inside it, consistency goes from
> unmeasurable to measurable *without paying in performance*. That is a concrete number for
> Stage A, replacing A5.3's qualitative "β must be comparable to sd[S0]".

### ⛔ 2b. Beyond that, performance collapses long before consistency becomes good

| | |
|---|---|
| β = 5 (40 % authority) | ADE **1.4864 m** — **2.3×** the base, and CON rank is still 14.33 |
| β = 20 | ADE **3.70 m** |
| β = 160 | CON rank **1.15** (near-perfect consistency) at ADE **7.62 m** — **11.5×** the base |

**Near-perfect consistency is purchasable, and it costs the car.** This is a measured
instance of the published *"neither design satisfies both desiderata"* — now a curve rather
than two points, which is the contribution Amendment 3 §A2 claimed and this is its first
data.

### ⭐ 2c. The control that decides whether climbing is worthwhile

The real-vs-cross-episode gap **must grow with authority**; if a matched and a mismatched
prior converge, the extra weight is buying obedience rather than understanding.

```
  beta 1.0  gap +0.0252 m        <- usable regime
  beta 2.0  gap +0.0742 m        <- usable regime, ~3x larger
  beta 5.0  gap -0.1576 m        <- sign flips; ADE already 2.3x degraded
  beta 10   gap +0.1494 m
  beta 20   gap -0.0970 m
  beta 40   gap +0.4290 m
  beta 80   gap -0.1500 m
```

**In the usable regime the gap grows ~3× (+0.0252 → +0.0742).** Above β = 5 it **oscillates
in sign**, because both arms are wrecked and the difference is noise.

---

## 3. ⛔ CORRECTION — the run's first printed verdict was wrong

Version 1 announced *"THE GAP GROWS WITH AUTHORITY … the frontier is worth climbing"* by
taking `max(gap)` over the **whole** sweep (+0.4290 at β = 40) and comparing it to β = 1.

**That is a max over a sign-oscillating sequence — it selects the largest positive
fluctuation and calls it a trend.** Same class as selecting a hyper-parameter on the scored
split: the maximum of noise is not an estimate of anything.

⇒ Fixed in code: the trend is read **only in the usable regime** (`ADE ≤ base`), the sign
oscillation above β = 5 is detected and printed, and the verdict now states the real
finding:

> **In the usable regime the gap grows with authority, but ADE collapses before consistency
> becomes good — so raising β ALONE cannot deliver both. The energy must get BETTER, not
> LOUDER.**

---

## 4. What this decides for TanitLang

| | |
|---|---|
| ⭐ **Stage A's β** | Start at **β ≈ 2× the natural port scale (~16 % authority)** — the measured point where consistency becomes readable at no ADE cost. Not a guess; the smallest authority at which the design's own consistency metric returns a signal. |
| ⛔ **The failure mode to avoid** | Turning β up to force consistency. At 40 % authority this energy already doubles ADE. A design that reports high `CON_rank` without its ADE is reporting obedience. |
| ⭐⭐ **The real requirement on the reasoner** | **The energy must RANK WELL, not merely be loud.** A 5-class manoeuvre prior given authority destroys driving because its ranking is poor. The headroom is the target: the fan holds **0.2315 m** while the model ships **0.65 m**, so a good energy has somewhere to go. |
| ⚠️ **Reporting rule** | Any future `CON_rank` is quoted **with its β, its authority %, and its ADE**. All three, or the number is not interpretable — this table is why. |

---

## 5. Scope

* T0, NON-PARITY pilot, one checkpoint, one port, deterministic eval forward.
* **No interval on the sweep rows.** Each β is a point estimate on the same 1,360 windows;
  the contrast that *does* carry a paired episode-cluster CI is `real − cross-episode` at
  β = 1, and it replicates under a second window draw (WP-N, WP-O). The **shape** of the
  curve is the finding; individual rows are not separated from each other.
* The energy here is REF-C's own manoeuvre prior, **not** a reasoner. The curve says what
  authority does to *this* energy; a better energy would move the whole frontier, which is
  precisely the hypothesis TanitLang exists to test.
* `β = 0` reads `CON rank` **75.80**, i.e. *worse than chance* — the base policy's selection
  is mildly anti-correlated with the port's energy. Expected: the base ignores that energy
  by construction.


---

# ADDENDUM - WP-Q: every row now carries an interval, and the recommendation MOVES

Code `code/wpq_frontier_ci.py`, `code/wpq2_con_ci.py` - raw `raw/wpq_frontier_ci.json`,
`raw/wpq2_con_ci.json`, `raw/wpq.log`, `raw/wpq2.log`. Zero GPU. Paired episode-cluster
bootstrap, 4,000 iterations, 34 episodes.

## Q1. What WP-P asserted without intervals, and what survives

| WP-P claim | measured | verdict |
|---|---|---|
| ADE(beta=2) is better than base | **-0.0052** [-0.0445, +0.0330] | ⛔ **overlaps 0 - NOT established** |
| ADE(beta=2) is worse than beta=1 | **+0.0061** [-0.0173, +0.0279] | ⛔ **overlaps 0 - NOT established** |
| the real-vs-cross gap grows 1 -> 2 | **+0.0491** [+0.0307, +0.0686] | ✅ **SEPARATED** |

=> **Across beta in [0.5, 2.5], ADE is statistically indistinguishable from the base
policy.** WP-P's "beta=2 still better than base" was a point estimate and is withdrawn as a
claim; what stands is the weaker and sufficient statement that ADE does not degrade there.

## Q2. The gap grows MONOTONICALLY and is separated at every safe beta

| beta | 0.50 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 2.50 | 3.00 |
|---|---|---|---|---|---|---|---|---|
| gap (m) | +0.0115 | +0.0251 | +0.0387 | +0.0520 | +0.0608 | +0.0742 | +0.0845 | +0.0461 |
| separated | * | * | * | * | * | * | * | no |

Seven consecutive separated points, monotone increasing to beta=2.5. **Scene-specificity
grows with authority across the whole safe band** - far stronger than WP-P's two points.

## Q3. The KNEE, located

**beta = 3.00 (authority 24.4 %)** is the first beta whose ADE is **separated worse** than
base: **+0.1524** [+0.0608, +0.2547]. It is also where the gap stops being separated. Both
signals agree on the same grid point.

## Q4. ⭐ CON_rank gets its own interval - and this decides Stage A

WP-Q still judged "consistency is measurable" by a point estimate against an **arbitrary
0.9 x chance threshold** - the same shape as the control gates this campaign has already had
to repair twice: a decision rule taken from a remembered constant instead of from the
statistic's own distribution. Fixed: CON_rank bootstrapped against its own chance value.

| beta | auth % | CON rank | CI95 (chance = 64.5) | |
|---|---:|---:|---|---|
| 0.00 | 0.0 | 75.80 | [70.76, 80.03] | SEPARATED **worse than chance** |
| 1.00 | 8.1 | 64.40 | [59.19, 69.20] | overlaps chance |
| 1.25 | 10.2 | 61.96 | [56.58, 67.10] | overlaps chance |
| **1.50** | **12.2** | **59.06** | **[53.79, 64.27]** | **SEPARATED** |
| 2.00 | 16.3 | 53.66 | [48.41, 58.92] | SEPARATED |
| 2.50 | 20.3 | 47.42 | [42.69, 52.35] | SEPARATED |

⭐ **Consistency separates from chance from beta = 1.50 (authority 12.2 %)** - not 2.0 as
WP-P's threshold suggested.

## Q5. ⛔ The recommendation MOVED TWICE, and the final one is beta = 1.50

* WP-P said **beta = 2.0** - from a point estimate against an arbitrary threshold.
* WP-Q's script said **beta = 2.5** - "largest authority not separated-worse than base".
  ⛔ **That rule is wrong**: it maximises consistency subject to a *weak* constraint and
  lands on a point whose ADE point-estimate is 5 % worse, buying consistency that was
  already available more cheaply.
* ⭐ **beta = 1.50 is the answer.** It is the **smallest** authority at which consistency
  separates from chance, and it carries the **best ADE in the entire sweep (0.6479 m)** -
  better than the shipped beta=1 (0.6521) and than base (0.6636), though that ADE difference
  is not itself separated.

> **STAGE A: beta = 1.5, authority ~12 %.** Beyond it you pay ADE for consistency you
> already have; below it consistency is unmeasurable.

!! **And one more finding worth keeping:** at **beta = 0** CON_rank is **75.80, separated
from chance in the WORSE direction**. The base policy's selection is significantly
ANTI-correlated with the port's energy - expected, since the base ignores that energy by
construction, but it means "chance" is not where an unguided arm sits, and a CON_rank near
64.5 is already evidence of *some* alignment.
