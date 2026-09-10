# ⭐ What survived every control: the ≥GT bar is 2.24× more REWARD-EFFICIENT than an equally small update

**Master Mind, 2026-09-11.** Fifteen arms, dev-box RTX 4060, no spend.
⛔ **T0 training-side, non-parity corpus. Not a capability claim; that needs T1.**

## The complete sample — n = 5 against n = 5

| arm | reward gained (ΔR1) | drift (ΔR3) | **reward per drift-point** |
|---|---|---|---|
| **bar ON**, 2,000 steps | **+0.1087** | 40.28 % | **0.00270** |
| bar OFF, 300 steps (length-matched) | +0.0343 | 28.49 % | 0.00120 |
| bar OFF, 2,000 steps | +0.0471 | 137.06 % | 0.00034 |

**Exact permutation on reward, all 252 splits:** observed **+0.0744**, **2 of 252** at least this
extreme ⇒ **one-sided p = 0.0079**.

⭐ **The test is NOT at its floor this time.** With n = 5 per cell the minimum obtainable p is
1/252 = 0.0040, and the observed value is 2/252 — so the design had room to report something more
extreme and did not. ⚠️ Contrast the retracted variance claim, which sat **exactly at** the floor.

⭐ **It also STRENGTHENED as the sample closed**: p = 0.0357 at n = 3, **0.0079** at n = 5. A
small-sample artifact moves the other way.

## ⛔ What this does and does not say

* ✅ **Against an equally small update, the bar returns 2.24× more reward per unit of drift.** That
  is the comparison that matters, because the length-matched control was built to reproduce the
  bar's own side effect — its 14.4 % admitted fraction — and it does reproduce the drift and the
  variance. It does **not** reproduce the reward.
* ⛔ **The ranges OVERLAP** (`max(short) = 0.0753`, `min(on) = 0.0535`). This is a **mean-shift**
  claim, not a separation claim.
* ⛔ **The variance claim stays RETRACTED.** Training 15 % as long reproduces both the low drift and
  the collapsed spread.
* ⛔ **300 full steps is not 2,000 steps at 14 % admission** — optimiser state, momentum and
  schedule all differ. This limit was declared before the control ran and still binds.
* ⛔ **Nothing here says the car drives better.** T1 and a four-family panel decide that.

## ⇒ The mechanism reading

The bar admits only the candidates that **beat the human demonstration's own score**. The
length-matched control removes the same *volume* of gradient **arbitrarily**. ⭐ Both arrive at
similar drift; only the selected one converts it into reward. ⇒ **the benefit is in WHICH gradients
survive, not how many** — which is exactly what DiffusionDrive-V2's stricter truncation claims to do,
and the first evidence on our rig that it does something the volume alone does not.
