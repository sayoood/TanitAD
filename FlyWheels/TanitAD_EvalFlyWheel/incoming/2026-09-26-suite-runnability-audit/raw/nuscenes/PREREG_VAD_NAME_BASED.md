# PRE-REGISTRATION — name-based VAD collision selection (written BEFORE any code change or run)

**Registered** 2026-09-26 by the EvalFlyWheel orchestrator. Its sha256 is recorded in
`PREREG_VAD_NAME_BASED.sha256` in the same commit it is staged in, before the run exists.
**Asked by** the Master Mind (RULE ZERO: the cheapest lever first — no download, no PI decision).

## The change

`occupancy_vad` currently picks colliding agents by raw `category.json` INDEX (`{2..8}` pedestrian,
`{14..23}` vehicle), which is correct only for the 32-entry lidarseg ordering. Replace the INDEX test with a
NAME test — `vehicle.*` → vehicle map, `human.pedestrian.*` → pedestrian map — keeping every other filter
unchanged (`NAME_MAPPING[cat] ∈ DET_CLASSES`, `num_lidar_pts > 0`, the BEV range).

**Derived effective set (the literal the tests pin):** the detection-class filter passes 14 categories;
intersected with VAD's intended sets that is **12**:
* pedestrian (4): `human.pedestrian.{adult, child, construction_worker, police_officer}`
* vehicle (8): `vehicle.{car, truck, bus.bendy, bus.rigid, trailer, construction, motorcycle, bicycle}`
* ⛔ excluded: `movable_object.barrier`, `movable_object.trafficcone` (detection classes, but in neither
  intended set — under the lidarseg ordering they sit at indices 9–13).

⚠️ **This rests on an INHERITED claim** — W6's F10, that the lidarseg ordering puts exactly the 7
`human.pedestrian.*` at 2..8 and the 10 `vehicle.*` at 14..23. It is not re-verified here (the lidarseg
`category.json` is not on this box). **The external gate below is what tests it.**

## Acceptance — committed in advance

**A. Equivalence (unit test).** On a SYNTHETIC 32-slot ordering built to match VAD's assumption, the
name rule and the index rule must assign the **identical** target to every slot. (Proves the name rule
reproduces verbatim VAD wherever the index rule is correct.)

**B. The real 23-entry file (unit test).** The name rule must select exactly the 12 names above, and
must NOT select barrier or traffic cone.

**C. ⭐ THE EXTERNAL GATE.** On nuScenes val under `nuScenes_OL_L2_stp3` (VAD pipeline, 5,119 samples), the
GT-collision floor must match **PARA-Drive Table 8's VAD-protocol GT row** (banked primary
`paradrive-cvpr2024`, p.8): **1.02 / 0.96 / 0.91 % at 1 / 2 / 3 s, avg 0.96 %**:
* each of 1 s, 2 s, 3 s and the average within **±25 %** of its published value
  (avg band **[0.72, 1.20] %**), **and**
* strictly decreasing **1 s > 2 s > 3 s**, the published shape.

**Why ±25 %, stated rather than chosen after the fact:** our UniAD-protocol GT floor — a path that already
selects by NAME and is not under suspicion — sits **+10 / +12 / +23 / +15 %** above PARA-Drive's UniAD GT row
(0.35 / 0.38 / 0.35 / 0.36). So ±25 % means *"as close as our already-correct path gets"*. ⚠️ The gate
therefore tests **fixed vs broken, not bit-exact reproduction** — and it has the power for that: the
pre-fix value **0.359 %** lies far outside the band and would be REJECTED.

**D. L2 untouched.** CV and STOP L2 bit-identical to the pre-fix runs (CV `0.8181846342993767`,
STOP `6.590993453736307`) — the change must move collision and nothing else.

## Outcomes, written down now

| outcome | reading | action |
|---|---|---|
| **PASS** (C holds) | the name rule restores VAD's object set | lift the refusal: VAD collision computed by name, GT floor quotable WITH this band caveat |
| **FAIL-LOW** (avg < 0.72 %) | the object set is still too small | investigate the other filters; the refusal stays |
| **FAIL-HIGH** (avg > 1.20 %) | over-inclusion (e.g. barriers / cones) | investigate; the refusal stays |
| **FAIL-SHAPE** (not 1 s > 2 s > 3 s) | a reduction / convention mismatch, not the object set | investigate; the refusal stays |

⛔ On any FAIL the refusal from `55aa747` stays in force, and the fallback is the 32-entry lidarseg
`category.json` — a new download, the PI's call. ⛔ The band is not widened after seeing the number.
