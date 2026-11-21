# The perception pipeline runs on the 139 eval clips at 256 × 1024 — 16/16, including the guards

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: dev box, CPU** · **T1 not applicable**
(this is a data-and-geometry validation, not a driving eval)

## What the PI asked for

> *"You validate the pipeline wit the 139 eval clips."* — PI, 2026-09-16

and, in the same breath, *"so let increase the azimut resolution and use 256×1024, we will do this
for all our future trainings."*

## What was actually open

`tests/test_refcv6_geometry_agnostic.py` says it in its own docstring:

> at the time of writing only the 256 x 640 cache exists on this box (the 256 x 1024 rebuild of
> the 139 eval clips is another agent's job). The real-data checks in
> `test_refcv6_perception_realdata.py` therefore run at **640**, and this file is what says the
> code is ready for **1024**.

So the real-data gates had only ever run at **640**; 1024 was covered by synthetic tensors alone.
The 1024 cache now exists (140 files, 7.6 GB), so the gates were pointed at it.

## Result — 16 passed, 0 failed, 6:36

`TANITAD_V2EP_DIR` → the 256 × 1024 cache · `TANITAD_ORIENT_CLIPS=0` (score **every** scorable
clip, which is what makes the orientation guard powered rather than convenient).

| check | at 1024 |
|---|---|
| every map opens as its own clip | PASS |
| a map renamed to another clip is refused | PASS (mutation) |
| coverage on the real windows, and the floor bites | PASS |
| frame alignment, and an off-by-`n_stack` goes red | PASS (mutation) |
| collate returns the label grid with a plausible seen share | PASS |
| first six columns are `agents_at_time` | PASS |
| cuboids are ground-standing as the spec says | PASS |
| a 2-D obstacle table is refused, not imputed | PASS (mutation) |
| **orientation guard, mirror goes red** | PASS |
| **the guard itself fails when the lift is mirrored** | PASS (mutation) |
| the power constant is computed, not chosen | PASS |
| **the whole map branch runs on real frames and the loss reaches the trunk** | PASS |
| the lift puts left on the left, for every clip | PASS |
| **the 3-D join reaches the height targets** | PASS |
| the join covers every agent-frame it ships | PASS |

⭐ The two that matter most are the **orientation guard and its mirror mutation**. The lift's
azimuth mapping is the one thing a geometry change could silently break — the columns move from
0.1875 °/px to 0.1172 °/px — and the guard both passes on the real lift and goes RED when the lift
is mirrored, at the new geometry, over every scorable clip.

## ⛔ What this does NOT validate

This is the **data and geometry** path. It is **not** a forward pass of the SPEC-v2 model.
`lift_orientation.lift_image_features` **average-pools the image by stride** as a stand-in for the
trunk, so **no timm trunk, no tactical layer and no planner ran here**. The map branch and the box
decoder that do run are built at toy widths on pooled pixels.

⇒ The other half — a real `resnet101` forward, both perception heads, the tactical decoder and the
diffusion planner, on this cache — is a separate deliverable and is **not** claimed by this note.

## ⚠️ And the geometry itself is an open PI decision

At 256 × 1024 the vertical field falls **45.456° → 29.341°** and the nearest visible road moves
**3.14 m → 5.02 m** ahead — a **1.88 m blind strip** in front of the ego (MEASURED camera height
1.3158 m). The 408 × 1024 cache keeps 99.6 % of today's field and is built. **PI decision queue
item 15**; the default is the PI's own instruction, 256 × 1024. ⛔ Nothing here says 256 × 1024 is
the right geometry — only that the pipeline runs correctly at it.

## Files

| path | what |
|---|---|
| `raw/pytest_realdata_256x1024.log` | the full run, verbatim |
