# The collision term could not see a car it drove through — 289 candidates, 27.7 % of lead windows

**Evidence class: MEASURED (ours).** Tier: **corpus-kinematic / instrument** — ⛔ **this is not a
T0 or T1 driving number and no arm is ranked here.** Artifacts: `raw/swept_vs_point.json`;
code `stack/tanitad/rl/rewards.py`; pin `stack/tests/test_collision_swept_segment.py` (**11 passed**).

## The defect

`stack/tanitad/rl/rewards.py::_collision` tested the **sampled waypoints**, not the path between
them. Pre-fix, static branch, verbatim:

```python
d = (traj.unsqueeze(-2) - obs.unsqueeze(-3)).norm(dim=-1)
hit = (d < r).any(dim=-1).any(dim=-1)
```

Point-to-point. A candidate whose consecutive waypoints **straddle** an obstacle — both endpoints
farther than `r`, the segment between them passing through it — was scored **safe**.

Raised by the refcv5 build stream as its escalation #1 (*"shared with the RL stream, so a
swept-segment fix is a cross-stream decision"*) and confirmed by reading the body, not a summary.

## The hole has a size, and on our own grid it is metres

MEASURED on the banked base fan (`fan_bank_base_240w.npz`, 240 windows × 128 candidates × 5 steps),
with the arm's own `r = ego 1.0 + obs 1.0 = 2.0 m`:

| quantity | value |
|---|---|
| waypoint spacing, median | **6.628 m** |
| waypoint spacing, p95 | **13.381 m** |
| waypoint spacing, max | 46.160 m |
| `v0` median / max | 9.94 / 36.25 m/s |
| **undetected corridor at p95** (`spacing − 2r`) | **+9.381 m** |

⚠️ A 9 m blind corridor is not a rounding concern; it is longer than the vehicle.

## What changes when the path is actually swept

MEASURED on the **8,320** (window, candidate) pairs that have a lead (65 of 240 windows):

| test | fires | rate |
|---|---|---|
| point (pre-fix) | 764 | 9.1827 % |
| **swept (fixed)** | **1,053** | **12.6562 %** |
| **newly caught (straddling)** | **289** | **3.4736 %** |
| lost | **0** | — |

⇒ a **+37.8 % relative increase** in detected contacts, and **18 of 65 lead windows (27.7 %)**
contained at least one candidate that drove through the lead undetected.

⭐ **`lost = 0` is the load-bearing check, not a formality.** The swept form is a strict superset of
the point form — every waypoint is an endpoint of some segment and the clamp includes `t ∈ {0, 1}`.
That is what makes previously banked numbers **LOWER BOUNDS** rather than incomparable.

## What this does and does not say about banked numbers

⚠️ **This is a STRICTER INSTRUMENT, not a corrected one.** The RL panel's contact readouts — the
human's selected-path contact **0.0000**, refcv3's **0.0007**, and the `sel_contact` / `top8_contact`
columns that read identically zero in every arm — were all computed with the point test. They are
**lower bounds**. ⛔ **A swept re-read is NOT a regression against them and must never be reported as
one**; it is a different measurement and must be labelled with which predicate produced it.

⭐ **Direction of the correction for the veto arm:** a veto keyed on this predicate was letting
straddling candidates through, so the veto-only arm's measured feasibility gain is a **lower bound**
too. The fix can only reveal more to veto, never less.

## The fix

Point-to-**segment** distance (`segment_point_distance`), clamped to the segment, OR-ed with the
point test (`_swept_hit`). ⛔ **Not a finer time grid** — that costs compute forever and still leaves
a smaller hole.

The moving-lead branch is swept in the **RELATIVE frame** (`lead_s − traj_s` against the origin), so
the lead's own motion over the step is accounted for and the time alignment survives. Sweeping the
ego path against a **static** lead would have re-introduced the `H-RL-THRESH-1` failure the branch's
own docstring exists to prevent — *a safety term that fires on the demonstration*. Pinned by a test
that a 2 s-gap follower is **not** flagged, with a same-breath positive control (a stopped lead
**is** flagged) so the branch cannot pass by being inert.

A degenerate zero-length segment (a stopped ego) falls back to the point test by construction.

## Controls

1. ⛔ **Deliberate regression:** a straddling fixture where the **pre-fix expression, verbatim**, must
   PASS it and the swept test must CATCH it. A gate never shown to fail the defect certifies nothing.
2. **Monotonicity** over 5 random seeds and on the real bank: swept never detects less. `lost = 0`.
3. **Time-alignment guard** + its positive control, above.
4. **Absent obstacles still score 0** — honest no-information, not a free pass.

## Open

The **static** branch's fire-rate change is unmeasured — the banked fan carries `lead5` but no static
obstacle set. It will be readable once the agent-track join is wired into the batch
(`2026-09-05-agent-join-into-batch`), and the same before/after must be reported there.
