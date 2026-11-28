# search log — E-DE-SIGN-3 (2026-09-18)

| # | probe | hits | note |
|---|---|---|---|
| 1 | repo: `31` in `…/2026-09-17-kitscenes-map-speed-values/` (RESULT + raw JSON) | the JSON holds `speed_limit_flagged: 69` and no valued count | ⛔ the quoted "31" has **no artifact path** in the package that produced 69 |
| 2 | label record fields | `cot_tokens.speed_limit` = **bool**; `cot_tokens.evidence` truncated to ~700 chars; `semantics.cot` = full CoT | parser reads `semantics.cot` |
| 3 | web (standing question, Band C): "posted speed limit map dataset open speed limit per lane nuPlan Argoverse OpenStreetMap vision traffic sign speed limit 2026" | 9 | OSM `maxspeed` / `maxspeed:lanes` exists but is crowd-sourced and sparse; ⛔ unusable for us without GNSS (settled). **No new vision-derivable posted-limit corpus found at this probe.** Named as EMPTY for a new supplier |
