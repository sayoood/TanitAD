# The ≥GT bar's real effect is on VARIANCE, not on the mean — n = 5, exact permutation test

**Master Mind, 2026-09-11.** Dev-box RTX 4060, ten 2,000-step arms, no spend.
⛔ **TIER T0, training-side, non-parity corpus.** ⛔ **Not a capability claim; that requires T1 and
a four-family panel.** ⛔ **A separation on a training-side statistic is not evidence the car drives
better.**

---

## 1. The data — five seeds per cell

| | seeds 0–4 (sel-ADE drift, %) | mean | sample stdev | span |
|---|---|---|---|---|
| **bar OFF** | 221.18, 76.02, 159.51, 200.24, **28.35** | 137.06 | **82.33** | **192.83** |
| **bar ON** | 42.26, 38.61, 44.10, 35.89, 40.52 | 40.28 | **3.19** | **8.21** |

`frac_above_bar_mean` across the five bar-ON arms: **0.1505, 0.1371, 0.1545, 0.1508, 0.1260** —
mean **0.1438**, range **0.0285**. The mask admits about **14 %** of sampled anchors, consistently.

## 2. ⭐ The test that settles it — exact, distribution-free, all 252 label assignments

⛔ **Not an F-test** (it assumes normality, and this bar-OFF cell is visibly heavy-tailed). The
labels ON/OFF are exchangeable under the null, and with n = 5 each **every one of the C(10,5) = 252
splits can be enumerated**, so the p-value is exact rather than asymptotic.

| effect | observed | splits at least this extreme | exact one-sided p |
|---|---|---|---|
| **variance ratio** | **667.4×** | **1 of 252** | **0.0040** |
| mean difference | +96.78 points | 6 of 252 | 0.0238 |

⭐ **Only ONE of the 252 possible assignments produces a variance ratio this extreme — and it is the
observed one.** ⚠️ **0.0040 is the FLOOR of this design**: with n = 5 per cell, 1/252 is the smallest
p obtainable, so the test is at its limit and a larger effect could not register as more significant.

⇒ **The bar's primary effect is that it makes the arm PREDICTABLE.** The mean moves too, but less
convincingly.

## 3. ⛔ THREE READINGS OF THE SAME EXPERIMENT — and the first two were mine, reported early

| n | what I reported | why it was wrong |
|---|---|---|
| 2 | *"within noise"* | a crude test comparing means against the larger within-cell **range** — extremely conservative, and blind to spread |
| 3–4 | *"complete separation, p = 1/35"* | a **small-sample artifact**. Seed 4's bar-OFF arm reads **28.35**, below every bar-ON value, and the separation vanished |
| **5** | **variance p = 0.0040, mean p = 0.0238** | the exact permutation test on the **complete** sample |

⛔ **The lesson is not that the crude tests were wrong — it is that I REPORTED THEM.** Twice I put a
partial-sample pattern in front of the PI and had to withdraw it. ⭐ **A sweep that is still running
has no result. Wait for the sample to close, then use the instrument that fits the question.**

⚠️ And note which instrument fits: the range-overlap test says "no separation" at n = 5 and is
**technically correct and useless** — the effect is not in the location of the ranges, it is in
their widths, and a test built to compare centres discards exactly that.

## 4. ⛔ WHAT IS NOT YET SETTLED — the weakness control

The bar admits only **14 %** of sampled anchors. ⛔ **The sceptical reading is trivial and must be
excluded, not argued away:** fewer positive gradients means a smaller total update, and a model that
changes less drifts less — which would make this an artifact of update **magnitude**, not update
**quality**.

⚠️ One piece of evidence already cuts against it: bar-ON produces **more** fan-reward improvement
(mean **+0.1087** vs **+0.0461**), which a merely-weaker update should not. But those distributions
**overlap**, so it is suggestive and not a discriminator.

⇒ **`offshort-s0/1/2` are running**: bar-OFF at **300 steps**, which is 15 % of 2,000 and therefore
matches the bar's admitted fraction. ⛔ **One-directional by design**: an arm that still drifts badly
**rules out** the weakness explanation; one that drifts little does **not** confirm it, because 300
full steps is not the same as 2,000 steps at 14 % admission — the optimiser state and schedule
differ.

## 5. What else the five seeds say

* **Fan reward (ΔR1)** — bar-ON mean +0.1087 vs +0.0461. ⛔ **Overlapping; no claim.**
* **Fan collision (ΔR2)** — bar-ON −0.047 vs +0.620, with bar-ON the **noisier** cell here
  (stdev 2.40 vs 1.22). ⛔ **Overlapping; no claim.** ⚠️ This retracts my earlier *"collision moves
  1.4 pp the wrong way"* — that was two seeds, and at five the sign is not stable.
* ⭐ **The mask's admitted fraction is the most reproducible number in the whole panel**: range
  0.0285 across five independent runs.
