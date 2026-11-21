# ⭐ AN OFFER: the seam pair is PRICED, READY, and STARVED here — run it on whatever rig is free

*From the dev-box longitudinal stream, 2026-09-06T00:30Z. This is a
work-transfer offer, not a request for a slot.*

## Why this exists

The Thor stream's `2f494aa` concluded, correctly, that
**"P4's first named successor is BLOCKED BY CONSTRUCTION at these weights"** —
`--jerk-seam` is inert while `W_JERK = 0` in the `wk15` triple. Both streams
reached that independently, on two rigs.

**The experiment that unblocks it is designed, priced and queued here — and it
has been starved for ~2.5 h** because the dev-box GPU is shared and both slots
have been continuously occupied. The gate is behaving correctly (a third arm is
a measured OOM risk on the 8 GB 4060), so this is not fixable locally.

⇒ **If Thor or any other rig has a slot, please run it there.** Getting the
lever measured matters more than which stream measures it.

## It is priced — first-order, not marginal

`raw/lon_jerk_scale.txt`, zero GPU, computed from the banked plans exactly as
`_cost_chunk` computes the terms, at `W_JERK = 0.02`:

| arm | jerk term | **seam delta** | kappa term | goal term | **seam / (goal+kappa)** |
|---|---|---|---|---|---|
| `wk15` | 3.552e-02 | 7.403e-02 | 2.314e-02 | 2.646e-01 | **0.494** |
| `lonshift` | 2.606e-02 | 2.786e-02 | 4.067e-03 | 6.975e-03 | **4.05** |

The seam is **half** the size of everything it trades against on `wk15` and
**4× larger** on `lonshift`. This same instrument **declined D4** (≈4.6 %
realised after the measured 1.71× oracle discount) and **refuted D5**
(`goal_reach_s`, worse than D2 on every column) — so it is discriminating, not
permissive.

## The exact arms — one variable between them

```
# BASELINE of the pair (NOT wk15: W_JERK must be live on BOTH sides)
--cost-metric ccos --cost-weights 0.02,15.11245,64.29715042415070 --plan-seed 0

# THE ARM: the seam is the ONLY difference
--cost-metric ccos --cost-weights 0.02,15.11245,64.29715042415070 --plan-seed 0 \
  --jerk-seam a0
```

⛔ `refav1_arm.py` now **REFUSES** `--jerk-seam` with `W_JERK == 0.0` before the
rollout (commit `32698bd`), so the inert configuration cannot be re-run by
accident. Guard exercised in both directions.

## Committed outcomes, before any numbers exist

* **If `seamon − seambase` moves the longitudinal family past that rig's own
  seed floor** — the unpriced seam was a real blocker and the cost side is live.
* **If it is a null with `W_JERK` live** — the seam is genuinely not the
  longitudinal blocker, and the cost side closes. That is a real result and must
  be reported as one; it is exactly what the inert arm could NOT tell us.
* **If `loncomb3` (D2 + seam) is WORSE than `lonshift`** — at ratio 4.05 the
  seam may dominate D2's objective. **Report that as the finding; do not retune
  the weight to rescue it.**

## What is already known, so nobody re-pays for it

| lever | status |
|---|---|
| D2 `a_shift` | ⭐ **landed, the win** — replicated across two rigs |
| D1 `a_sustain` | indistinguishable from D2 at arm level (Thor) |
| **`goal_reach_s`** | ⛔ **REFUTED 0-GPU** — worse than D2 on every headline column (`D-REFAV1-LON-D5-REFUTED`) |
| raise `GOAL_A_MAX` | ⛔ declined — ≈4.6 % realised, and bounded (clip 2.5 ≡ 4.0) |
| arm `W_VEND` | ⛔ PI decision — `test_C1` pins the call site |
