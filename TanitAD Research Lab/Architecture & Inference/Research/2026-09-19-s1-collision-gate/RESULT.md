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

### ⭐ WHERE the human's DAC fails — near and early, at borderline cells, NOT at far range

I expected far-range map noise. **It is not** (`raw/dac_horizon_probe.json`, 400 held-out
windows, no model, no GPU):

| | |
|---|---|
| human DAC by horizon | 1 s **0.740** · 2 s **0.670** · 3 s **0.603** · 4 s **0.553** — a quarter of windows already fail within ONE second |
| first offending corner, time | p10 **0.0 s** · median **0.7 s** · p90 3.1 s |
| first offending corner, range | p10 **2.6 m** · median **7.8 m** · p90 33.1 m |
| the offending cell's drivable fraction | median **0.40** against the 0.5 threshold; 35.8 % of failures sit above 0.4 |

⇒ **A range cap on the DAC horizon would NOT repair this** — the failures start beside the car.
The two live candidates are the **threshold** (0.5 on a *fractional* coverage map, where the
offending cells cluster just under it) and the **near-field map quality** itself. Both are
`PREREG_D9_REWARD_REPAIR`'s to decide; this file supplies the measurement and the discriminator,
not the repair.

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

### ⭐ WHY the head misses, and the bar the next arm has to clear

The head is **wired and training** — it is **undertrained**, and that is a different next lever
from "fix the perception plumbing". MEASURED on A8's own banked `metrics.jsonl` (500 train rows,
0 GPU):

| steps | `box3d` loss | `box3d_centre` | matched boxes / batch |
|---|---|---|---|
| 0–500 | 50.40 | 36.39 | 37.7 |
| 500–1,500 | 26.19 | 15.54 | 38.3 |
| 2,000–3,000 | 22.85 | 14.37 | 39.2 |
| 4,000–5,000 | **18.58** | **9.93** | 36.4 |

`box3d_centre` is a **summed L1 in metres** over (x, y) for matched pairs
(`agent_slots.py:577`), so at step 5,000 the head's matched boxes are still **~10 m off**, and it
is still falling. The held-out probe agrees independently: nearest-GT distance **p50 9.54 m**.

⇒ **The AP of 0.0038 is the arithmetic consequence of a ~10 m centre error against a 2 m matching
threshold, not of a broken wiring.** ⭐ **The next arm's bar, stated as a number:** `box3d_centre`
must fall **below ~2 m** (the matching threshold) before any predicted gate can be read at all;
until then `S1-GATE-PRED` is untestable rather than refuted-in-principle. Everything else in the
S1 rig is in place and proven, so that arm is a training question, not an instrument question.

### ⛔ Is the head SCORED on targets it is never TRAINED on? No — asked from source, then measured

The absence-vs-defect question (a head scored on targets it cannot represent looks exactly like a
head that cannot localise). **Answered from SOURCE, so no GPU was spent re-answering it:**
* the dataset emits targets **RAW** — *"Targets are emitted RAW (no visibility filter here). The
  filter lives in `refc_agents.agent_losses`…"* (`refc_v3_train.py` ~2676), and that filter
  belongs to the 2-D agent head's loss, not to the box3d path;
* the box3d loss matches through `agent_slots.match_slots` (`:481-517`), which Hungarian-matches
  **every** valid target and drops only the **farthest** when targets exceed queries — here 32
  padded targets against 100 queries, so it **never drops**;
* the decode is a **SCALE, not a clamp** (`cx = raw · x_fwd_m`, raw unbounded), so "outside the
  extent" never meant "unrepresentable".
⇒ **There is no set the head is trained on but not scored on.** AP 0.00383 is not an artifact of
scoring unrepresentable targets, and the "not localising **to the 2 m threshold**" reading stands.

**Context, MEASURED** (`raw/gt_extent_probe.json`, 736 labelled windows, 15,206 GT boxes, no
model): only **22.3 % (3,394)** lie inside the decode extent (x ∈ [0, 60], |y| ≤ 16); **49 % are
BEHIND the ego**; median range **38.9 m**, 72 % within 60 m, and the field spans ±190 m. ⚠️ The
box read's GT population is therefore *all 32 nearest agents, 360°* — **not** the gate's
population, which is only agents that can intersect a 4 s path ahead. A future detection read for
a gate should state that population explicitly; this one is reported as measured.

### ⭐ How far is 5,000 steps from the 2 m bar? — the dev box CANNOT say, and here is the arithmetic

`raw/box_centre_curve.json`, A8's own banked rows, 0 GPU, under CLAUDE.md's fit discipline.

| | |
|---|---|
| last 500 steps | `box3d_centre` **10.32 m** = **5.16×** the 2 m bar |
| OLS log-log, full window | exponent **not quotable**: **R² 0.4100**, n = 498, window [10, 5000] |
| OLS log-log, last half | **R² 0.0931**, n = 251, window [2500, 5000] — worse, not better |
| matched-step ratio (the prescribed fallback) | 15.11 m (1k–3k) → **10.42 m** (3k–5k) = **×0.6896 per 2,000 steps** |
| what that ratio implies | ~**13,885 total steps** to reach 2 m — ⛔ **beyond the 2× extrapolation bound (10,000)** |

⛔ **The honest sentence, for the pod request's §5 `D-S1-DEP-BOX`:** *at 5,000 steps the head is
5.16× above the bar; the error is still falling; and **this run cannot say how many steps close
that gap** — both fits are below R² 0.80 so no exponent is quotable, and the fallback ratio's
projection lands past the 2× bound.* Nothing here bounds the **corpus** scale either: A8 is one
corpus at one size, so a scale claim would have no second point to rest on.

**Two one-liners the reading needed:**
1. **The stable matched count (36–39) says NOTHING about the head.** `match_slots` matches
   `min(n_target, n_query)` pairs, and with 100 queries against ≤ 32 padded targets **every**
   valid target is matched by construction — confirmed in the data: `n_matched == n_target` on
   **100 %** of 498 rows (37.44 each). It measures **label density**.
2. **The x-vs-y split cannot be read from banked rows:** `box3d_centre` is a **summed** L1 over
   both axes (`agent_slots.py:577`) and the trainer logs no split. **MEASURED on held-out
   predictions instead** (`raw/box_axis_probe.json`, 24 windows, **623 matched pairs**, through
   the **training** matcher — the AP's 2 m greedy matcher would pair only the lucky hits and
   flatter the head):

| population | mean \|dx\| | mean \|dy\| | L1 sum | x-share |
|---|---|---|---|---|
| every matched target (623) | 5.36 m | 6.68 m | **12.04 m** | 0.445 |
| ⭐ **near-forward — the gate's OWN population** (129) | **3.09 m** | **2.97 m** | **6.06 m** | **0.51** |
| far or behind (494) | 5.96 m | 7.65 m | 13.60 m | 0.438 |

⇒ **The error is ISOTROPIC, not longitudinal** (x-share 0.445 overall, 0.51 near-forward), so the
lever is general localisation — not a depth- or range-specific fix. ⭐ **And the near field is
2.2× better than the average**, which the far/behind 79 % of pairs dominates: on the population a
collision gate actually acts on, the head is ~3× from the 2 m bar rather than ~6×. ⛔ This is the
centre error per population, **not** a re-measured AP — the Master Mind's decision not to spend a
pass on that stands, and no AP number here is restated.

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

## E9 — does the T1 harness run on a refcv6 checkpoint at 416 × 1024, and at what cost?

**Answered, MEASURED 2026-09-20** (`raw/e9_refcv3arm_cpu.json` — clip ids rewritten as `sha12`;
`raw/e9_refcv3arm_cpu.log`).

1. ⛔ **`taniteval/tools/t1_eval.py` itself does NOT run one, by design.** It carries **0**
   `refc` references across 28 functions (the control read non-zero, so the file was read), and
   its rollout `roll_closed` (`:772`) drives the **flagship's** action-feedback loop, which
   `refc_v3` has no action to feed. **Declared: unwired for refcv6, and correctly so.**
2. ⭐ **The route that works is `taniteval/tools/refcv3_arm.py` → `t1_eval.analyze`** (8 call
   sites). It ran end to end on **A8 `ckpt_5000.pt` at 416 × 1024 on CPU**: 7 windows, 1 episode,
   128 anchors, arms `os` / `ha` / `ha0` / `ha0_ext`, and it emitted **all four binding families**
   (`longitudinal` speed MAE/bias/RMSE + target-speed accuracy; `lateral` heading, yaw-rate,
   curvature and cross MAE/bias; `tactical`; `strategic`), stamped **T1 self-action open loop**.
3. **Cost on CPU: 52 s wall for 7 windows = 7.4 s/window end to end**, model load, corpus build
   and analysis included; the forward alone is **~5.5 s/window** (its own `[cost]` line). ⚠️ From
   `n = 7` on ONE episode: an extrapolation to the 736-window read is **~1.1 h forward-only**, and
   it is an extrapolation, not a measurement. For scale, this night's S1 harness — decode, forward,
   128 candidate splines, the checker and the box read — measured **7.80 s/window** over 736
   windows on the same CPU.

## ⭐ Does the EVAL path supervise `box3d_z` / `box3d_h`? — YES in the A3/A8-era runs

The standing caveat (*"`n_z = n_h = 0` over 900 windows on BOTH halves, so no refcv6 arm may quote
an eval-side z or h number"*) **no longer describes this code or these runs.** It was a **WIRING
defect, now fixed** — not a design choice. Three probes, two of them independent of each other
(`raw/eval_zh_supervision.json`):

1. **SOURCE, train reader** — `refc_v3_train.py:6615` passes
   `with_track_ids=bool(getattr(args, "join3d", None))`.
2. **SOURCE, eval reader** — `:6814` passes the **same** flag, and `:6799` documents the defect
   and its history in the trainer's own words (*"`with_track_ids` MUST MATCH THE TRAIN READER … the
   eval reports `box3d_z` / `box3d_h` = 0.0 with n_z = n_h = 0: a zero that reads as PERFECT"*).
   Two construction sites, one flag; only one site used to carry it.
3. ⭐ **DATA, a different mechanism — the runs' OWN artifacts.** Every A8 eval row (steps 1,000
   through 5,000) and A3's read **`eval_box3d_n_z = eval_box3d_n_h = eval_box3d_n_matched =
   38.012`**, with `eval_box3d_z` 0.7449 and `eval_box3d_h` 0.2354 — the **same identity** the
   train side shows, and emphatically not zero.

**Fix cost: already paid** — one flag at the second construction site; nothing further is owed.
The 900-window `n_z = 0` measurement **predates** it.

⛔ **The durable rule, because a code-version belief is not evidence:** an eval-side z/h number is
admissible when **that run's own eval row** shows `n_z > 0` **and** `n_z == n_matched`. That is a
per-run **artifact** check. A3 and A8 pass it; runs predating 2026-09-18 stay capped.

### Next levers, in the order the evidence ranks them

1. **Perception quality is the binding constraint** (§10's own commitment). The box head must
   localise before any predicted gate can be read: today AP is below a uniform predictor.
2. ⭐ **Even perfect detection is not enough:** 48 % of the ceiling needs a **motion forecast**,
   measured, not argued (`ORACLE` vs `ORACLE-CV`). A gate on static t0 boxes tops out near half.
3. The agent-seam arm (`H-SEL-AGENT-1`) is untouched by this and remains open.
