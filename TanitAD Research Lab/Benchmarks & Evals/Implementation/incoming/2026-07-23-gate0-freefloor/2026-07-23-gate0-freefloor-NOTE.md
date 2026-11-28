# Gate-0 — does a FREE inference-time floor collapse REF-C's off-road? — MEASURED 2026-07-23

**The pre-registered cheapest experiment** from `Architecture & Inference/Research/2026-07-23-closed-loop-wm-training-verdict.md`
(rungs 1-2 of the ladder): add a **training-free drivable-area floor** to REF-C-base's existing
anchored-diffusion planner and measure **off-road departure** WITH vs WITHOUT it on the balanced
38-scene AlpaSim suite. **If off-road collapses toward ~0 (esp. intersections) -> the free floor works,
RL is unnecessary for this failure, Gate-0 PASS, decided with ZERO training.** A surviving residual is the
Gate-1 target. Prerequisite (drivable geometry accessible at inference) was CLEARED in
`gate0_prerequisite_NOTE.md` (`…/incoming/2026-07-22-alpasim-closedloop-evalpod/`).

## ⚠️ FRAMING (mandatory)
WITHIN-SIM RELATIVE, on NuRec reconstructions (~3.2× OOD, RUN_RECIPE §13). Relative ON-vs-OFF ranking on
the SAME reconstructions is trustworthy; absolute rates are not real-world. Host = eval pod `tanitad-eval`
(A40, free); pod2/pod3/pod1 untouched. Lock `gate0-run` (released at end).

## The floor (no model change — hooks the existing rescorer)
REF-C's `forward(steps=2)` already returns `anchor_traj [1,128,4,2]` (denoised rig-frame anchors) +
`anchor_logits [1,128]` (conf). The floor re-selects over those 128 already-computed anchors:
1. **Cost-guided SELECTION (robust core):** `argmax(conf − λ·offroad_cost − μ·collision_cost)` instead of
   `argmax(conf)`. `offroad_cost` = per-waypoint distance-outside the drivable lane union (0 inside),
   later-waypoint-weighted. λ=5.0 logits/m, μ=0.0 (off-road-only for the pre-registered off-road reading;
   collision term implemented, held at 0 for the primary run).
2. **Road-boundary SAFETY CLAMP:** if the selected plan's max waypoint still exits the lanes by >0.75 m,
   override to the most-on-road anchor (`argmin offroad_cost`).
Implementation: `refc_floor_driver.py` (extends the validated M4 `refc_driver.py`; same f-theta canon +
gate-2 timing). Drivable surface = `shapely.unary_union` over AlpaSim's OWN `_get_lane_polygon`
(eval/scorers/offroad.py) so the floor's geometry == the scored metric's geometry.

## ⚠️ COST GEOMETRY VALIDATED FIRST (the flagged frame/sign risk) — `gate0_cost_validation.json`
A frame bug WAS caught and resolved before any suite run:
- `ds.rig.trajectory.positions` is **ego-local** (starts [0,0,0]); the naive first probe (50 of 385 lanes)
  gave a false 192 m gap. The FULL lane union spans the ego-path region.
- **Driver-received `current.pose` is in the MAP (lane) frame under IDENTITY.** `world_to_nre` (a pure
  translation) must NOT be applied — applying it moves the on-road GT to cost 21.5 m (off-road).
- **Gold-standard check:** the REAL force-GT driver-received poses (`refc_openloop_preds.jsonl`, 4 scenes,
  300 poses) score off-road cost **mean 0.00, 100% on-road** against their scene lanes; clean off-road
  points score **90–202 m**. Correct sign + magnitude.
- **Production `_select` unit test:** on an intersection scene, an on-road anchor (cost 0) is chosen over a
  higher-confidence (conf 12) off-road anchor (cost 17.8) with floor ON; floor OFF picks the off-road one.
  VERDICT: COST TRUSTWORTHY.

## 🎯 RESULT — off-road WITH floor (ON) vs WITHOUT (OFF), REF-C-base, n=37 paired (same session)
Paired control `gate0_off` reproduces the prior baseline EXACTLY (offroad 0.49, pass 15/37, score 0.229)
-> the deltas below are the floor's effect, not run-to-run noise. Canon f_eff=266.1 OK both arms.
| category | n | OFF offroad | **ON offroad** | **ΔOFFROAD (ON−OFF)** | OFF→ON pass | OFF→ON caf | OFF→ON score |
|---|---|---|---|---|---|---|---|
| **intersection** | 7 | 0.71 | **0.71** | **+0.00** [0,0] | 1→1 | 0.71→**0.43** | 0.063→0.063 |
| **roundabout** | 8 | 0.75 | **0.88** | **+0.125** [0,0.375] | 2→1 | 0.00→0.00 | 0.101→0.024 |
| traffic_light | 6 | 0.17 | 0.17 | +0.00 [0,0] | 4→4 | 0.33→0.33 | 0.337→0.337 |
| highway | 8 | 0.38 | 0.25 | −0.125 [−0.375,0] | 3→3 | 0.38→0.38 | 0.221→0.221 |
| straight_other | 8 | 0.38 | 0.38 | +0.00 [0,0] | 5→5 | 0.12→0.12 | 0.430→0.430 |
| **JUNCTION (int+rbt)** | 15 | 0.73 | **0.80** | **+0.067** [0,0.2] | 3→2 | 0.33→**0.20** | 0.083→0.043 |
| **OVERALL** | 37 | 0.49 | **0.49** | **+0.000** [−0.081,+0.081] | 15→14 | 0.30→**0.24** | 0.229→0.212 |

Overall off-road paired: **Δ+0.000, boot95 [−0.081,+0.081]** (CI includes 0); scenes fixed off→on-road **0**,
broke on→off **1** (roundabout `fd3a49fa`, score 0.61→0.00 — naive lane-keeping steered it off), McNemar p=1.0.
**Side effects:** at-fault collisions DOWN (0.30→0.24 overall; intersection 0.71→0.43 — 2 scenes' collisions
removed) despite μ=0; plan-deviation UP (0.33→0.76) and score marginally DOWN (0.229→0.212).

## MECHANISM (MEASURED, `gate0_floor_ON.jsonl`, 1665 drives) — why the closed-loop metric barely moves
The floor works **per-plan** but hits a hard ceiling closed-loop:
- **Per-step it fixes most off-road plans:** of 96 drives where the base `argmax(conf)` plan was off-road
  (>0.5 m), the floor pulled **76 back on-road** (intersection 29→6, roundabout 26→1). Clamp fired 61×,
  reached on-road 54×. Selected-plan off-road cost mean **0.123 → 0.039** (−68 %).
- **But ~21 % of off-road moments have NO on-road anchor** (`base_off>0.5 & sel_off>0.5`, 20/96): the
  fixed 128-anchor FPS vocabulary (comma2k19 is 74 % straight) lacks the sharp-curve modes a junction
  needs — `base_off` reaches **11.8 m** while the best available anchor is still **3.3 m** off-road.
- **`offroad` is MAX-over-the-20 s-rollout**, so a single un-fixable junction moment flags the whole scene.
  Fixing 76 of 96 per-step plans therefore does **not** reduce the number of scenes with ≥1 off-road
  excursion. A pure SELECTION floor cannot synthesize a trajectory the anchor set does not already contain,
  and it has no route-direction awareness (at roundabouts "nearest lane" ≠ "correct lane").

## VERDICT — ❌ NOT A PASS. RESIDUAL SURVIVES → Gate-1 warranted (reported plainly)
**The free cost-guided SELECTION floor does NOT collapse off-road.** Overall off-road **0.49 → 0.49**
(Δ+0.000 [−0.081,+0.081], CI includes 0); **intersection 0.71 → 0.71 (unchanged)**; roundabout **0.75 → 0.88
(worse)**. **Zero** junction scenes moved off-road→on-road; one roundabout scene got worse. The pre-registered
PASS condition ("off-road collapses toward ~0, esp. intersections") is **not met** — so **the cheapest free
experiment does NOT rule RL/Gate-1 out for this failure.** It does not close the question in the "free" direction.

**Why (MEASURED, not asserted):** the floor fixes 76/96 per-step off-road plans, but **`offroad` is
MAX-over-the-20 s rollout** and **~21 % of off-road moments have no on-road anchor** in the fixed 128-anchor
FPS vocabulary (straight-biased; `base_off` 11.8 m, best anchor still 3.3 m off) — so pure re-selection
cannot keep the ego on-road through a junction it has no anchor for. This is the **anchor-vocabulary ceiling**,
and it is exactly what a selection-only floor cannot beat.

**Not worthless, but not the lever:** the floor is a mild net-safety change — at-fault collisions
0.30 → 0.24 (intersection 0.71 → 0.43) even at μ=0 — at the cost of higher plan-deviation (0.33 → 0.76) and a
hair of score. Worth keeping as a cheap safety override; insufficient as the off-road fix.

**The residual = the Gate-1 target = the JUNCTION off-road failure** (intersection + roundabout, offroad
0.73–0.88 in BOTH arms; both architectures already fail here, `scenario_stratified_scaled_NOTE.md`). The
next levers, cheapest first:
1. **Cost-guided DIFFUSION proper (the mission's "optional refinement"):** the per-denoise-step gradient nudge
   modifies the trajectory DURING denoising — it can escape the fixed anchor vocabulary the selection floor is
   trapped in. Still training-free; the natural next ~free experiment. (Implemented hooks are in place:
   `RefCModel` returns the anchors; the differentiable off-road energy is the same lane geometry.)
2. **Route-direction-aware cost** (roundabouts broke because "nearest lane" ≠ "correct-direction lane").
3. **A richer anchor vocabulary** with sharp-junction modes (FPS over a junction-inclusive pool).
4. If a residual still survives → **Gate-1 closed-loop-aware training** (RoaD/CAT-K or analytic-gradient
   through the diff-WM), per the verdict doc — now NOT ruled out by Gate-0.

## Caveats
1. WITHIN-SIM RELATIVE / ~3.2× OOD (§13) — relative ON-vs-OFF only. 480×854. NuRec reconstructions.
2. n per category 6-8 -> wide binomial CIs; the OVERALL + JUNCTION deltas are the powered signals.
3. Floor = off-road-only (μ=0). Collision term implemented but held at 0 (the pre-registered metric is
   off-road; collision is a separate potential Gate-1 lever).
4. AlpaSim's `offroad` = not-fully-in-a-lane AND touching-road-edge — a forgiving metric; the floor
   targets the same lane geometry so it is aligned.

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `gate0_cost_validation.json` | repo (staged) · pod `/workspace/` | frame+sign validation (MEASURED) |
| `gate0_freefloor_results.json` | repo (staged) · pod `/workspace/` | per-category ON-vs-OFF paired (MEASURED) |
| `refc_floor_driver.py` | repo (staged) · pod `/workspace/` | the free floor (cost-select + clamp) |
| `gate0_run.sh`,`gate0_master.sh`,`gate0_aggregate.py` | repo (staged) · pod | reusable run + aggregate pipeline |
| `gate0_api_probe.py`,`gate0_frame_resolve.py`,`gate0_frame_confirm.py`,`gate0_unit_test.py` | repo (staged) · pod | validation harness |
| `gate0_floor_ON.jsonl` (per-drive base-vs-floor selection + costs + clamp, 1665 drives) | repo (staged) · pod `/workspace/` | mechanism diagnostics (MEASURED) |
| rollouts (38×2), USDZs (38) | **pod only** `/workspace/gate0_{on,off}`, sceneset 986fec83 | regenerable via the pipeline |

**Pod left CLEAN:** services killed by port, `gpu_lock` released, GPU idle.
