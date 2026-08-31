<title>COMMS — E-LAB-ARCH-0831</title>

# COMMS — E-LAB-ARCH-0831

## ⛔ ESCALATION TO THE MASTER MIND / PI — one structural correction to a live gate document

**`Project Steering/V7_LAUNCH_GATE.md` §P2 lists three candidate causes. There is a fourth,
and it is the one with a published mechanism, a published magnitude and a published fix.**

| | the gate's three | proposed fourth |
|---|---|---|
| (a) | horizon too short | ⏳ under test (MM-E19) |
| (b) | the action is realised motion, not a command | ⛔ never tested; blocked on corpus + a PI geometry call |
| (c) | action representation / latent geometry | ⛔ untested |
| **(d)** | — | ⭐⭐ **TARGET CONSTRUCTION.** A teacher-forced target already contains the action's effect, so an action-invariant solution exists. `2605.25313` §4.2, measured ‖H₁‖/‖H₀‖ ≈ 0.03; §4.3 fix restores 1.00 ± 0.15 |

**Why it is not a restatement of (c):** (c) says *change how the action is encoded*; (d) says
*change what the prediction is scored against*. Different knob, different arm.

**Why it changes a spending decision:** (d) predicts that fixing the channel — P2(b) — leaves
the deafness in place, because the mechanism is independent of the channel's provenance.
P2(b) currently carries the two most expensive prerequisites in the gate (a PI concession on
the 65.2°/120° geometry, and a corpus that is not on Thor).

### The decisions being requested — all three are the PI's, none is mine

1. **Add (d) to P2's candidate table** in `V7_LAUNCH_GATE.md`. *(Document edit — the Lab does
   not edit Project Steering gate documents; escalating rather than performing it.)*
2. **Re-order: measure (d) before committing an arm to (b).**
3. **Authorise a SPEC** for the simulator-free (d) arm (negative-action contrastive target).
   ⛔ Not authorised by this package; it needs pre-registration with a deliberate-regression
   arm before a line is written.

⛔ **Nothing here closes any gate item.** `V7_LAUNCH_GATE.md`: *"no item below may be marked
closed by me. I can report that a criterion is met; the PI closes."* This package reports a
**missing candidate**, which is weaker than a criterion and much weaker than a closure.

## Claims-register touchpoints

- **P2 / MM-E10 / MM-E11** — no status change. MM-E11's unexplained negative (ratio fell
  0.40×) now has a *candidate* explanation under (d); that is an interpretation, not a
  re-measurement, and the row should not move on it.
- **MM-E12** — untouched and explicitly respected: this package does **not** re-argue
  action-vs-scene redundancy, which MM-E12 refuted with two controls.
- **P5 / L3** — no status change. F3 supplies external precedent for the bar
  (*"persistence baseline"*) and a candidate mechanism for the failure; the L3 read itself is
  still unrun.
- **P4 / MM-E16** — no status change. F5 adds a **missing ingredient** (cross-timescale
  consistency) and a **pre-named failure mode** (unconstrained macro-action search) to the
  ladder design.
- ⚠️ **A novelty claim in the paper needs downgrading.** If the paper presents "the model
  ignores its conditioning channel" as a finding, `1710.02410` (2018) states it verbatim and
  reports the fix. Cite it; do not re-discover it.

## Handovers

| to | what |
|---|---|
| **Data Engineering / DataFlyWheel** | `2607.27017`: arms lacking prediction pressure stay **flat over a 5× data range**; *"additional data improves only the parameters it already acquires."* ⇒ D-DATA-EFFICIENCY has an **objective-side precondition**. |
| **Benchmarks & Evals** | ATM (`2606.09028`) — planner-free action-consistency screening, >100× faster than CEM-coupled eval. Carry its precondition (*"when the true success gap is non-trivial"*) with it. |
| **Deployment & Optimization** | temporal abstraction is also the inference-cost mechanism for a 6 s horizon — fewer, larger steps. |

## Integration status

**NONE REQUIRED, and none performed.** No code, no config, no trainer change, no test-suite
impact. The only thing that needs a human is the gate-document edit in §1 — and that is why
it is in this file's headline rather than in a README nobody reads.
