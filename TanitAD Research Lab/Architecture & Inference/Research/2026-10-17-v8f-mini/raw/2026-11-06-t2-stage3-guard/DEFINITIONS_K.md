# DEFINITIONS_K: the open choices of the guard-start draw (AGENT K, 2026-11-06), written BEFORE the stage-3 draw is run

Evidence class: the choices below are DEFINITIONS, not measurements. Nothing in this file was derived from a draw value: when it was written
nobody on this task had printed or read a guard start (only the screening table's `n_ticks` was read: 200 for all 20 guard scenes).

## What the frozen card fixes (`t2_prereg/PREREG_T2_CRAWL.md`, sha256 `2c4c6557...b413`, never edited)
* Section 3: "Guard scenes: two starts drawn by `numpy.random.default_rng(1)`, uniform over the renderable span, leaving 6 s."
* Section 6b: "**The guard starts.** Two per guard scene from `default_rng(1)`, as section 3 says. (The first build drew one per guard scene.)"
* Section 6b, the start clamp, for every start: "A start needs f0 >= 18 logged frames of history."

## What the card leaves open, and the choice made (each is the choice `t2v8f_kit.all_starts()` already implements)
`t2v8f/code/t2v8f_kit.py` (sha256 `20cf3284bf8fb201eaa1bfdf3eed1bc469397bf0aa6131acb0206e5068f1627d` at writing) carries the draw since before the freeze. Stage 1's and stage 2's builders assert their
plan equals `all_starts()`. This agent changes none of it; stage 3's builder asserts the same equality, and the test re-derives the draw by hand.

| # | open choice | the choice | why this one |
|---|---|---|---|
| 1 | one stream or one per scene | **ONE `numpy.random.default_rng(1)`, never re-seeded**, consumed by the guard scenes in order | the card says "drawn by `default_rng(1)`", a single generator; a per-scene re-seed would give every scene the same two draws |
| 2 | order of the scenes | **`order_index` ascending in `SCREEN_TABLE.json`**, which is the slot order `guard_1 .. guard_20` (screening order). Launch scenes consume NO draw, so their position changes nothing | the card's only order is the screening order (`SCREEN_RULE.md`) |
| 3 | units | **ticks of the 10 Hz rig** (an integer frame index f0), not seconds | the cell takes `--starts` as 10 Hz tick indices; a start in seconds would need a rounding rule the card does not give |
| 4 | the draw call | **`rng.integers(lo, hi + 1)`, twice per scene, in sequence** (int64, then `int(...)`); not `rng.uniform`, not `rng.choice`, not a sorted pair | "uniform over the renderable span" on an integer grid; `integers` is the uniform integer draw. The first draw is the first start |
| 5 | inclusive bounds | **both ends inclusive**: `lo = 18`, `hi = n_ticks - 61` | `lo`: card 6b (f0 >= 18 logged frames). `hi`: "leaving 6 s" = 60 ticks after f0, so `f0 + 60 <= n_ticks - 1`; this is the dispatcher's own rule `f0 + 60 < n_ticks` |
| 6 | the "renderable span" | **ticks 0 .. n_ticks - 1 of the front-wide camera** (`SCREEN_RULE.md`: first to last 10 Hz tick), cut by 5's two bounds. `n_ticks` is the screening table's, and the builder re-checks it against each scene's own rig | the span is defined in `SCREEN_RULE.md` |
| 7 | two equal draws | **collapse to one start**, as 6b rules for the launch starts | 6b's collapse rule, applied unchanged; the kit's `_collapse` |
| 8 | scene index in the records | **stage-3 `idx` = 42 .. 61** in guard order (stage 2's launch scenes are 0 .. 41) | so stage-2 and stage-3 rollouts pool without an index clash; `idx` is only a record key |

## Start class
By the stage-2 rule, `make_stage2.start_class`: **launch** = the logged speed at f0 (backward difference over the logged rig rows, `|p[f0] - p[f0-1]| / 0.1`) below 0.5 m/s, otherwise **moving**.
A guard scene is a scene with no launch event and v >= 5.2 m/s on at least half its span, so most guard starts will read `moving`; the table in the hand-back reports what the draw gives.

## What is NOT chosen here
No guard start was moved, re-drawn or filtered after seeing it. If a start lands on a class or a position the Master Mind dislikes, that is a dated amendment beside the frozen card.
