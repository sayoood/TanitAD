# SPEC addendum 2 — crosswalk-synonym pruning arm (pre-registered 2026-09-14, after spdP3a FAILED, before this arm's result)

Chosen AFTER seeing `spdP3a` fail N2 on the edge class (IoU 0.681 night / 0.840 day) while its crosswalk class held (0.962 / 0.980):
the edge failure points at the dropped ROAD synonym ("asphalt road surface" feeds the drivable boundary the edge class is built
from). This is a new arm chosen on that diagnosis — stated here so it is not read as a pre-planned one.

Bars: UNCHANGED (N1–N4 vs `spd0`, both clips). Base: `spdF4a` flags.

| arm | prompts dropped | expected speed effect |
|---|---|---|
| `spdP2a` | zebra crossing, pedestrian crossing (both road prompts kept) | 19 → 17 prompts, ≈ −0.09 s/frame |

Committed in advance, as in addendum 1: a pass is reported as "passes on the two test clips — needs PI sign-off and validation on
more scenes" (two clips hold 4 crossings); a failure is not carried and closes prompt pruning for this map.
