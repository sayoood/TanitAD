# PREREG_VDAC_P6 — Addendum A4: the G20 alignment bar follows the card ("at the banked rate"), 0.99

2026-11-12, Master Mind.
- **Base card:** `PREREG_VDAC_P6.md` (sha256 `5594c2ae...`), with A1 (`8f121bdf...`), A2 (`c3d56358...`) and A3. None is edited.
- **Scope:** this addendum changes ONE implementation constant, the G20 bar, and nothing else in the design, the arms or the read.

## What the card says
- §7, G-LIVE alignment: *"on >= 64 real frames, P6's rank-0 targets lie inside the label at the banked rate"*.
- Deliberate regression: *"flip the label's y axis; shift the lookup by one frame"*.
- The banked rate is the label store's own QC: the teacher's rank-0 target points lie inside the label on **0.99999** of points over 103,037 rows (`FEASIBILITY.md:79`; `vdac_probes.py` docstring: 99.9989 %).

## What the code did, and what it cost
- `refe/vdac_probes.py:24` set `BAR = 0.95`, carried over from SPEC_P7's own alignment bar, not from this card.
- `judge_g20` requires the real fraction ≥ BAR and **both mutants < BAR**: a check whose mutants also pass is not a check.

**MEASURED on the pod (S1a HA-s0 preflight, 2026-11-12T09:08Z, 256 real frames, 5,120 points):**

| reading | value |
|---|---|
| `frac_real` | 1.0000 |
| `frac_yflip` | 0.9572 |
| `frac_shift1` | 0.9582 |

- At 0.95 both mutants pass, so G20 is RED by its own design and the preflight refused. P7 resumed at once, and no training step ran.
- The cause is the bar: most target points sit far inside the drivable area, so a flipped or shifted label still covers about 96 % of them. A 0.95 bar has no power to see that misalignment.

## The ruling
- **G20's bar becomes 0.99: the card's "banked rate" (0.99999) read to two nines.** Everything else in `judge_g20` is unchanged: ≥ 64 frames, real ≥ bar, each mutant < bar, finite fields.
- **Why 0.99, and why it is not fitted to the data above:**
  1. The card's own criterion is the banked rate, 0.99999, so the faithful bar sits near it, not at 0.95. Restoring it makes the gate **stricter**, not looser.
  2. 0.99 tolerates 1 % misses on 5,120 points, about 900× the banked miss rate (1.1e-5). It therefore does not refuse a correct lookup for sampling reasons.
  3. Any bar in (0.9582, 1.0] would separate these readings. The MM picks the card's rate to two nines, a round value fixed by the card's wording. It is not chosen as a point between the observed numbers.
- **Disclosure:** this addendum is written AFTER the readings above were seen. It changes an instrument constant to match the frozen card. It does not touch any outcome bar, arm, estimator or read of PREREG_VDAC_P6.

## Implementation (landed with this addendum)
- `refe/vdac_probes.py`: `BAR = 0.99`. The docstring states the card's banked rate and this addendum.
- `refe/test_vdac_gate.py` V11 stays valid as written: its synthetic real frames read ≥ 0.999 and its mutants < 0.5. A new check asserts that the pod's measured mutant readings (0.9572 / 0.9582) are refused at the new bar while a real 1.0 passes.

## Not changed
- SPEC_P7's own alignment gate is out of scope here. The MM checks separately whether P7's check has the same lack of power.
