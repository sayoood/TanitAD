<title>COMMS — E-LAB-DATA-0831</title>

# COMMS — E-LAB-DATA-0831

## ⛔ ESCALATION 1 — a factual correction to a live gate document

**`Project Steering/V7_LAUNCH_GATE.md` §P2(b): *"TWO BLOCKERS ON THE ONLY CORPUS THAT COULD
TEST IT."* There are at least two other candidate corpora, and the state-only triage the gate
itself calls the cheapest discriminating test is blocked by neither of the two blockers named.**

| corpus | command-bearing? | the gate's blockers | its own blockers |
|---|---|---|---|
| comma2k19 | CAN steering | ⛔ 65.203° geometry (PI call) · ⛔ not on Thor | — |
| **L2D** (`yaak-ai/L2D`) | `action.continuous` = gas/brake/steering; `action.discrete` = gear/**turn_signal** | ⭐ neither binds a state-only test | ⛔ **no intrinsics ship at all** · 🔴 **GDPR gate on frames** (`LOOP_STATE.md:1025`) · ⚠️ signal provenance UNVERIFIED |
| **nuScenes CAN** | `2606.12987` §3, verified: *"actions are extracted from CAN-bus data as 2D vectors aₜ=(steerₜ, accelₜ)"* — **our exact parameterisation** | ⭐ neither binds | ⚠️ which CAN field · whether we hold it · licence — **all three unchecked** |

**Decision requested (the PI's / Master Mind's, not mine):** amend the P2(b) blocker
paragraph. ⛔ **This package does not edit Project Steering documents** — escalating rather
than performing, per the operating standard.

## ⛔ ESCALATION 2 — a probe worth running before any P2(b) arm is costed

`E-DATA-CMDPROV`, specified in RESULT §6. **CPU-minutes, no GPU, no Thor, no spend, no
training.** It reads the gate's own decision rule and returns a number either way:

- r ≈ 0.999 stratified ⇒ **P2(b) is a weak lever and the gate can say so with evidence**
  instead of leaving it ⛔ NEVER TESTED indefinitely.
- r materially lower ⇒ a genuine command channel exists on a reachable corpus, and the
  question becomes which corpus, not whether.

⚠️ It carries a **must-read-≈1.0 control** (the PhysicalAI self-correlation between
`steer_road_rad` and its own curvature source) and a **stratification requirement** — a
pooled r is dominated by straight driving where the two signals coincide by construction, so
an unstratified number would report "no independent information" as an artifact of the
operating regime. Reporting that pooled number would be a false negative of exactly the kind
this programme keeps logging.

## ⛔ ESCALATION 3 — an open blocker I am surfacing, not clearing

**`LOOP_STATE.md:1025`, 🔴 GDPR gate on L2D:** *"real German dashcam footage, NO anonymization
statement in the card … face/plate check REQUIRED before any L2D frame is re-hosted /
published. Blocks L2D entering a shippable corpus."* Unresolved, and **out of my authority**.
It binds frames, publication and any shippable corpus; it does **not** bind a read-only
correlation over state parquet. That distinction is the reason the recommended probe is
state-only, and it should not be read as the gate being smaller than it is.

## Claims-register touchpoints

- **D-B1-100H** — no status change. F3 records that the **road-class stratum**, its own
  unfilled WORK ITEM, exists as a native per-frame field (`observation.state.road`) in a
  corpus we have already adapted. That is a sourcing option, not progress on the 100 h target.
- **D-SPEED-GATE** — no status change. F3 records that the >14 m/s regime that
  `physicalai_r0.py:100` hard-excludes is **directly selectable** in L2D via `max_speed` +
  `road = motorway`. ⛔ With the standing warning that a cross-corpus arm confounds corpus
  with regime and therefore cannot answer *"does high-speed data help"*.
- **D-DATA-EFFICIENCY** — no status change; F4 adds an **objective-side precondition** from
  `2607.27017` (arms lacking prediction pressure stay flat over a 5× data range). It bears on
  the *sequencing* of the planned diversity ablation, not on its design.
- **P2(b)** — no status change and none proposed. ⛔ Per the gate's closure rule, closing is
  the PI's. This package reports that the corpus premise is wrong and the triage is unblocked.

## Handovers

| to | what |
|---|---|
| **DataFlyWheel** | owns `E-DATA-CMDPROV` and the three-route pricing. ⚠️ The July L2D state pull lives **only** at `devbox:C:/Users/Admin/tanitad-data/l2d/` and is unverified since 2026-07-22 — step 0 must re-check it, not assume it. |
| **Architecture & Inference** | `turn_signal` is a discrete, non-path-derivable intent signal — the channel shape their F2 (Codevilla branched conditioning) needs; and L2D's **13.8 % STUDENT** split is a native non-expert action source, which is ReSim's data-side answer to action-insensitivity. |
| **Benchmarks & Evals** | nuScenes is now interesting to us as an **action-provenance** corpus, which is a different use from the open-loop-benchmark use our doctrine warns about. Keep the two apart in any table. |

## Integration status

**NONE REQUIRED, and none performed.** No code, no config, no corpus, no parity key, no
registry row. Two document corrections are requested above; both are for owners other than me.
