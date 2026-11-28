# SPEC addendum — prompt pruning arms (pre-registered 2026-09-14, after the redundancy screen, before any pruning-arm result)

Bars: UNCHANGED from `SPEC_sam3_speed_prereg.md` (N1–N4 vs `spd0`, both clips). Base: `spdF4a` flags (fp16 fusion encoder, fp32
decoder and mask head, prompts batched, async CPU).

Screen (MEASURED, `prompt_redundancy.json`, 192 front frames): unique share of each prompt's accepted pixels not covered by the other
prompts of its family — asphalt road surface 0.0054, pedestrian crossing 0.0084, zebra crossing 0.0160, crosswalk stripe 0.0157
(vs white stripe on road), lane marking 0.0212 (vs road marking 0.2788).

| arm | prompts dropped | expected effect |
|---|---|---|
| `spdP3a` | asphalt road surface, zebra crossing, pedestrian crossing | 19 → 16 prompts |
| `spdP5a` | P3 + lane marking + crosswalk stripe | 19 → 14 prompts |

**Excluded from pruning whatever the result:** `stop line` and `diagonal stripes on road` returned ZERO instances on both clips — the
test clips contain no stop line and no hatched area, so their absence here is not evidence of redundancy.

**Committed in advance:** a pruning arm that passes N1–N4 on both clips is reported as **"passes on the two test clips — needs PI
sign-off and validation on more scenes"**, NOT as approved: the prompt set was tuned with the PI's video reviews, and two clips cover
no stop line, no hatched area, no rain. A pruning arm that fails any bar is not carried.
