# A6 ruling 1: how known answer 3 ("δ 0.5 is partial") is read (Master Mind, 2026-11-06 07:06, after the known-answer runs and before any A6 rule is read)

**Question** (`lf6_build/A6_NOTE_KA3_READING.md`). The card (section 2, frozen fbf5de02) asks that δ 0.5's bind rate lie strictly
between those of δ 0 and δ 10^9 on a 64-window smoke.
* **Read on each smoke rollout's own ticks, the order fails:** δ 0.5 binds 0.4844 against 0.4271 at δ 10^9 (heads bare).
* **Read on the same ticks**, i.e. the inputs F saw in the δ 0.5 smoke, run through δ 0 and δ 0.5 of the tool under test and δ 10^9 of
  A5's independent `apply_floor_batch`, **the order holds:** 0 < 0.484 < 0.490.
* **The bound s' − s ≤ δ · t holds on every knot** (largest excess 4e-16 m), and an independent recomputation agrees to 3e-8 m.

**Ruling.** Known answer 3 is read on the same inputs.
* **Why.** A known answer pins a property of the operator. On fixed inputs, the set of bound ticks grows with δ by construction. Each
  rollout drives its own states, and a smaller lift leaves the car slower, so it meets more crawl ticks. A rollout's bind rate
  therefore mixes the operator with the rollout's own state, and no known answer can order it in advance.
* **What stays as it is.** The rollout-level rates are reported beside, not gating, and the narrow margin (1 to 3 of 192 ticks) is
  disclosed.

**What this ruling does not change.** The card is not edited; no P, G, F or D rule, no outcome and no dose changes; no A6 rule has
been read.
