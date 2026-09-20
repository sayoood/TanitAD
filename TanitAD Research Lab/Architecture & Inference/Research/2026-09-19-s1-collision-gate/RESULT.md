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

## Verdict — ⛔ `S1-GATE-PRED` is **NOT QUOTABLE**, and the ceiling it aimed at is **real**

**MEASURED 2026-09-20** on A8 `ckpt_5000.pt`, **736 windows / 59 episode clusters**, CPU,
7.80 s/window (`raw/verdict_A8.json`, rows in `raw/rows_A8.sha12.jsonl.gz`). Estimator:
`paired_episode_cluster_bootstrap`, 2,000 draws. Tier: **T1\*** for BASE/RANDOM/CONST/PRED,
**T0** for the two oracle arms (they read the recorded future).

| arm | collided-selection rate | vs BASE (paired CI) |
|---|---|---|
| `S1-RANDOM` | 0.3637 | **+0.1463** [+0.0973, +0.1968] separated |
| **`S1-BASE`** | **0.2174** | — |
| `S1-GATE-CONST` | 0.2174 | **+0.00000** [0, 0] — ⭐ the deliberate-regression arm reads **EXACTLY 0** |
| `S1-GATE-PRED` | 0.2391 | +0.0217 [−0.0053, +0.0482] **not separated** |
| `S1-ORACLE-CV` | 0.1196 | −0.0978 [−0.1410, −0.0586] separated **(T0)** |
| `S1-GATE-ORACLE` | **0.0299** | **−0.1875** [−0.2454, −0.1365] separated **(T0)** |

**Share of the ceiling reached: −11.6 %** [−28.9 %, +2.5 %] ⇒ below 10 %, which is
`PREREG_S1` §10's **pre-committed** reading: *the headroom is real and the predicted occupancy is
not good enough to reach it; the next work is perception quality, not a better gate.*

⭐ **The decomposition S1A.5 was pre-registered for, and it is the result worth keeping:**
* `ORACLE − ORACLE-CV` = **0.0897**, i.e. **48 % of the ceiling is lost to having no motion
  forecast** — with **perfect t0 detection** and constant-velocity extrapolation.
* `ORACLE-CV − PRED` = **0.1195**, the detection-and-velocity error, which takes the gate past
  BASE into harm.

⛔ **And it harms.** PRED is **separated-worse than BASE on 13 of 14 families**: `pdms` −0.0690,
`dac` −0.0720, `ttc` −0.0625, `ep` −0.0615, `comfort` −0.1821, `tac_lat_agree` −0.1821,
`along_2s` +1.35 m, `cross_2s` +0.69 m, `along_4s` +2.55 m, `cross_4s` +3.38 m, `speed_err_4s`
+1.13 m/s, `heading_err_4s` +0.177 rad, `curv_err_2s` +0.713 (n = 437 masked windows). Only
`tac_lon_agree` is not separated. By S1A.7 that is **FAIL-HARM**, reported here beside the
NOT-QUOTABLE verdict rather than instead of it. STRATEGIC (**PAUSED**, PI item 25): nav
compliance 0.359 (PRED) vs 0.436 (BASE) over 39 turn windows; evaluated, never counted.

### The box read (S1A.4 / S1A.11) — the dependency that failed

| | |
|---|---|
| detection | **AP(BEV, 2 m) = 0.00383** [0.0025, 0.0056] over **15,206 GT** in 736 labelled windows, against the pre-registered base rate **0.1241** ⇒ ⛔ **FAIL** |
| velocity | MAE **5.43 m/s** vs the zero-velocity floor **7.44 m/s**, gain **+2.01** [+0.87, +3.03] ⇒ **PASS** |

⇒ `S1-GATE-PRED` is **not quotable** (S1A.4's own rule). The head's *velocities* beat a zero
predictor; its *boxes* are not where the agents are.

⛔ **CORRECTION to S1A.11's base rate — found after the data, so the CRITERION is not changed,
the NUMBER is retracted.** I pre-registered the sampling extent as the head's decode scales
(60 × 32 m). MEASURED (`raw/box_geometry_probe.txt`, 4 windows): GT centres actually span
x ∈ [−108, +95] m and y ∈ [−69, +103] m, so 0.1241 is not this corpus's chance level and must not
be quoted as one. ⭐ **The conclusion does not rest on it.** The extent-free discriminator: the
nearest-GT distance is within 2 m for **6.0 %** of the head's boxes against **12.9 %** for
uniform-random placement, and the head's boxes spread **±198 m** in x (std 69 m) where the GT
spreads 46 m. Predictions and GT share one convention and both centre near zero, so this is
**not** a frame or unit error: at 5,000 steps the head is **not localising**.

### What this says, and what it does not

1. ⭐ **The headroom is real at 416 × 1024 on 124 clips:** a gate on recorded agents removes
   **86 %** of colliding selections (0.2174 → 0.0299, separated). `PREREG_S1`'s ceiling
   reproduces on a different fan, a different corpus and a different model from item 19's.
2. ⭐ **The selector is not noise:** BASE beats RANDOM by 0.1463, separated — worth stating
   next to the weak-planner caveat.
3. ⛔ **A gate is only as good as its occupancy.** Driven by a head at chance, the identical
   gate that recovers 86 % on recorded agents loses 11.6 % and harms 13 of 14 families.
4. **Internal validity holds** (all arms share one fan, one checker, one scoring rule);
   **external validity to a trained planner does NOT** — A8's selector is weak
   (`anchor_acc` 0.092 vs chance 1/128 = 0.0078).
5. **nav = v1**, the trained derivation; on PhysicalAI it is derived from the ego's own future
   path, so it is **optimistic by construction**.
6. The waypoint representation's own floor (S1A.6) is ~1 % of windows: absolute rates carry
   ±1 %, between-arm deltas do not.
7. ⚠️ The fan's collision-free share here is **0.6363** — a NEW measurement on this fan
   (ERRATUM-1), never a check against item 19's 0.553.

### Next levers, in the order the evidence ranks them

1. **Perception quality is the binding constraint** (§10's own commitment). The box head must
   localise before any predicted gate can be read: today AP is below a uniform predictor.
2. ⭐ **Even perfect detection is not enough:** 48 % of the ceiling needs a **motion forecast**,
   measured, not argued (`ORACLE` vs `ORACLE-CV`). A gate on static t0 boxes tops out near half.
3. The agent-seam arm (`H-SEL-AGENT-1`) is untouched by this and remains open.
