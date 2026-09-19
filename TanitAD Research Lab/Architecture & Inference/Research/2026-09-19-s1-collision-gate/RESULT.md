# S1 — the collision gate at dev-box scale (`H-SEL-GATE-1`)

**Owner:** TanitAD_TrainingFlyWheel · **assigned by:** the Master Mind, 2026-09-19 ·
**pre-registration:** `Project Steering/PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md` — the
**S1 AMENDMENT** plus **ERRATUM-1** (the fan is 128, not 117), **ERRATUM-2** (`collided` is
`nc != 1`), **S1A.10** (nav = v1) and **S1A.11** (the box read), all landed or staged **before
any S1 data**.

> **STATUS 2026-09-20 01:10: the pass is RUNNING on CPU** (A8 `ckpt_5000.pt`, 736 eligible
> windows, `--replicate 50`, ~10.7 s/window ⇒ ~2.2 h). The verdict section is written from
> `verdict_A8.json` when it lands, whatever it says.

## What is already measured

| read | result |
|---|---|
| **window identity** | `len(ds)` = **10,600** = the trainer's own printed held-out count. The 1,000 are its own seed-12345 draw |
| **eligibility** (item 19's rules) | **736** windows over **59** episodes; dropped: 205 with no 6 s route future, 59 missing an agent-label frame. ⛔ Every S1 number is over **736 windows / 59 clusters**, never "1,000" |
| **fetch equivalence** | 9/9 fields equal to item 19's own `Ctx.fetch`, called unbound, on 3 windows |
| **S1A.6 round-trip** | **PASS, thin: 0.9905** vs the 0.99 bar (NC 0.9946, DAC 0.9959, 7/736 flips). Re-expansion error median 0.013 m, p95 0.039 m |
| **S1A.3 determinism** | the 2-window smoke replicate reproduced **every** selection and nc flag (0 diffs) |
| **agent GT** | the trainer's own join attach: **62/62 episodes, 9,929/10,600 windows labelled (93.7 %)**, 367,351 target boxes, pad 32 |
| **smoke** | A3's finished checkpoint, 2 windows, CPU: the whole chain runs — load → perception → forward → fan → gates → box read → analyze |

## ⛔ A finding that outlives S1: under a LIVE SAM3 map, DAC zeroes the HUMAN on 44.6 % of windows

**MEASURED** (`raw/roundtrip_halfB_A8cfg.json`, 736 held-out windows): the recorded human's own
`dac` is **0 on 328/736 = 44.6 %**, so the human's `pdms` is 0 there too (`pdms` multiplies by
`dac`). The rule is strict by construction: DAC is 0 if **any** of the ego box's 4 corners, at
**any** of 41 ticks, lands on a **seen** cell whose drivable fraction is below 0.5
(`pdm_proxy.dac_from_drivable`) — 164 samples per trajectory, so one noisy cell at range zeroes
the window.

⚠️ **The banked ddv2 dumps are NOT a counterexample, and this is the check that matters.** They
read `human_pdms` **0.986 with 0/493 zeros** — but those runs had **DAC DEAD**: nothing passed
the drivable map, so `score_candidates` defaulted the multiplier to ONES. The
*"DAC IS LIVE / DAC IS DEAD"* stamp postdates them and **no ddv2 log carries it**. ⇒ The two
numbers measure different things, and 44.6 % is the first reading of human DAC compliance with
the map **live**.

**Consequences, stated rather than discovered later:**
1. The S1 **primary** statistic is the collided-selection rate, which is `no_at_fault_collision`
   and does **not** involve DAC. It is unaffected.
2. Every **PDMS / EP / DAC** family number on this corpus is dominated by a term that fails the
   human on nearly half the windows, so the S1 harm checks on those three are **weak where DAC
   is 0**. The geometric families (along, cross, speed, heading, curvature) and the tactical
   agreement are unaffected.
3. ⚠️ **Beyond S1:** the ddv2 RL reward multiplies progress by this same DAC term with the
   **human as the reference**. If the map is live there, the reference's own progress is zeroed
   on ~45 % of windows. That is a question for `PREREG_D9_REWARD_REPAIR`, raised here with its
   artifact, not resolved here.

## ⚠️ Process note — a trigger must wake on the EVENT, never on a resource precondition

MEASURED the same night, and it cost the GPU ~3.5 idle hours: I armed the overnight trigger on
*"A8's done-marker **AND** GPU ≤ 2,500 MiB **AND** host ≥ 8 GB"*. A8's marker was written at
21:10, but the PI's own reconstruction-studio servers held 3,945 MiB, so the **wake** condition
never became true and nothing started; the trigger also only reported. ⇒ **Wake on the event
(the done-marker). Read resource state as INFORMATION at wake time, and let the launch step's
own gate decide whether to run.** Same class as a wait-loop that matches only success markers
(`[[wait-loops-must-match-failure]]`): the condition that ends the wait must be the thing that
happened, not the thing you hope is also true.

## Verdict

*(pending `verdict_A8.json`; the pre-registered reading is S1A.7, with the three sentences it
must carry — internal-vs-external validity, nav = v1, and the ~1 % representation floor)*
