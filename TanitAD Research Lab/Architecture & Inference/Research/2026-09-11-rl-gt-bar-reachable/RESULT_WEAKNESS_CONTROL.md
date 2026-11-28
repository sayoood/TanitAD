# ⛔ The control KILLED the variance headline — and revealed a different, better-supported one

**Master Mind, 2026-09-11.** Thirteen arms on the dev-box RTX 4060, no spend.
⛔ **TIER T0, training-side, non-parity corpus. Not a capability claim.**

---

## 1. What the control asked

The ≥GT bar admits **14.4 %** of sampled anchors. ⛔ **The trivial reading:** fewer positive
gradients ⇒ smaller total update ⇒ less drift. That would make *"the bar stabilises the arm"* an
artifact of update **magnitude**, not update **quality**.

⇒ **`offshort`: bar OFF at 300 steps** — 15 % of 2,000, matching the admitted fraction.

## 2. ⛔ IT DID NOT CLEAR. The weakness explanation SURVIVES for drift.

| arm | n | drift mean | drift stdev | drift span |
|---|---|---|---|---|
| bar OFF, 2,000 steps | 5 | **137.06 %** | **82.33** | 192.83 |
| **bar ON**, 2,000 steps | 5 | 40.28 % | **3.19** | 8.21 |
| **bar OFF, 300 steps** (control) | 3 | **28.73 %** | **5.08** | 9.24 |

⛔ **Training for 15 % as long reproduces BOTH halves of the finding** — the low mean (28.7, if
anything *lower* than the bar's 40.3) **and** the collapsed spread (5.08 against 3.19, versus
82.33 for the full-length control).

⇒ ⛔ **I RETRACT the headline.** *"The ≥GT bar's real effect is on variance — it makes the arm
predictable"* is **NOT ESTABLISHED**. The variance collapse is fully explained by a smaller total
update, and the exact permutation p = 0.0040 I reported measures a real difference **against the
wrong comparator**. ⭐ The statistic was sound; the **control** was missing, and I designed it only
after publishing the claim.

## 3. ⭐ WHAT THE CONTROL REVEALED INSTEAD — the bar is not merely "less training"

At **comparable drift**, the bar extracts far more from the same budget:

| arm | drift | fan-reward improvement | **reward per drift-point** |
|---|---|---|---|
| **bar ON**, 2,000 | 40.28 % | **+0.1087** | **0.00270** |
| bar OFF, 300 (control) | 28.73 % | +0.0302 | 0.00105 |
| bar OFF, 2,000 | 137.06 % | +0.0471 | 0.00034 |

**Exact permutation, bar-ON versus the length-matched control, on reward:** observed difference
**+0.0785**, **2 of 56** splits at least that extreme ⇒ **one-sided p = 0.0357**.

⇒ ⭐ **The bar buys REWARD EFFICIENCY, not stability.** It is **2.6×** better than the
length-matched control per unit of drift paid, and **7.9×** better than the full-length one.

⚠️ **Scope it honestly.** The ranges **overlap** at the boundary (`max(short) = 0.0523` against
`min(on) = 0.0500`), n is **3 against 5**, and p = 0.0357 is not far from the 1/56 = 0.018 floor of
this design. ⛔ **This is a supported direction, not a settled result.**

## 4. ⚠️ The control's own limit, stated when it was designed and still binding

⛔ **300 full steps is NOT the same as 2,000 steps at 14 % admission** — the optimiser state,
momentum and schedule all differ. The control was pre-declared **one-directional**: an arm that
still drifted badly would have **ruled out** the weakness explanation; one that drifts little
**does not confirm it**. ⇒ Formally the drift comparison is **inconclusive**, and I report it as
*"not excluded"* rather than *"confirmed"*, which is the weaker and correct phrasing.

⭐ **But the reward comparison does not share that limit**, because it asks a different question:
not *"is the update smaller?"* but *"what did the same drift buy?"*

## 5. What now stands, in order of confidence

1. ⭐ **The mechanism is reached and correct** — mutation-proven by analytic identity: bit-identical
   gradient below the bar, **exactly 0.0** above it, partial in between. This rests on identities
   and is untouched by any of the above.
2. ⭐ **The mask admits 14.4 % of anchors**, range **0.0285** across five seeds — the most
   reproducible number in the panel.
3. ⚠️ **The bar is more reward-efficient than an equivalently small update** (p = 0.0357, ranges
   overlapping) — **supported, not settled**.
4. ⛔ **The bar reduces drift variance** — **RETRACTED**; reproduced by training less.
5. ⛔ **The bar improves driving** — **untested**. That is T1 and a four-family panel.

## 6. ⭐ The methodological lesson, which is the durable part

**I published a p = 0.0040 result against a comparator that could not distinguish the hypothesis
from its trivial alternative.** The permutation test was exact and correct; it answered a question
whose answer did not mean what I said it meant.

⇒ ⭐ **A control that matches the INTERVENTION'S SIDE EFFECT — not just its absence — is what
converts a difference into a mechanism.** Ablating the bar gives "with vs without". Matching its
14 % admitted fraction gives **"is it the selection, or merely the volume?"** ⛔ Only the second is
a mechanism claim, and it is the one I had to be argued into running.

⚠️ This is the same family as the programme's constant-only and raw-input floors: *a probe without
a control that must read a known value manufactures results.* Here the missing floor was not
"no information" but **"the same amount of information, chosen at random."**
