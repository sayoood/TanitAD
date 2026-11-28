# Strategic context labels aligned to the v8f-mini cache steps (Data-Eng agent, 2026-11-16)

Layer: HF private release `Sayood/tanitad-context-labels-v1` rev `100f03da5c068e841b6eb7b6fc768f0f39714b27`. All 13 files match the release's own `manifest.json` sha256 (MEASURED, 0 mismatches). Numbers below come from `LABEL_QA.json` and `strat_labels_table_report.json` in this folder.

## Headline
1. Table built: `strat_labels_table.npz` (sha256 `8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff`, 1,682,384 bytes). It matches `navtok_table.npz` row for row (asserted on clip, off, n for both splits; 461,518 train and 14,237 eval steps). 5 tests pass, including a shifted-time-base arm that must break alignment.
2. Fit for a head: **country, weather, illumination, period_of_day**. Surface is usable with care. **Visibility is unfit.** Twilight is unfit as a class.
3. Decision for the MM: the time base is ASSUMED, not measured (see Limits). It moves labels by under 1.7 % of steps at +-0.4 s, so it is not decision-critical.

## Schema
* Per FRAME, not per window: one row per present frame at 0.2 s (`frame_idx` 0..100, `t_s = 0.2 * frame_idx`, `video_frame = 6 * frame_idx`, nominal 30 fps video clock). 470,475 rows, 4,659 clips, every frame present.
* Country, month, hour_of_day are per clip (constant over a clip) and come from NVIDIA `data_collection.parquet`. Country agrees with that file on 100 % of our clips. Period_of_day is the clip hour. Weather, surface, visibility, illumination vary within a clip.
* Time base of the cache steps is the join's label clock (egomotion origin). Cache step j sits at video time `j * dt_c`, `dt_c` = clip median of the `now_s` spacing (0.20133 s). Step 0 is the first camera row: implied `now_s(0)` median 0.1131 s against the clock sidecar's `grid_start_s` median 0.1135 s.

## Alignment rule
Nearest present label frame in absolute time; ties go to the earlier frame; valid only within 0.25 s; the last label is held past the end up to 0.25 s. `unknown` is invalid for that variable. Flags: 0 ok, 1 clip absent from the release, 2 step further than 0.25 s from every label frame. Arrays per split: `*_frame_idx`, `*_off_s`, `*_flag`, and per variable `*_<var>_id` (int8, -1 invalid) and `*_<var>_valid`. Class lists are in `strat_labels_vocab.json` and `vocab_<var>` in the npz. Variables: country (25), weather (clear, fog, rain, snow), surface (dry, snow_ice, wet), visibility (good, reduced), illumination (dark, day, lit_night, twilight), period_of_day (afternoon, evening, midday, morning, night).

## Coverage (MEASURED)
| split | clips in release | steps ok | steps, clip absent | steps out of tolerance |
|---|---|---|---|---|
| train4572 | 4,512 of 4,572 (60 absent) | 455,342 of 461,518 (98.66 %) | 6,056 | 120 |
| eval141 | 141 of 141 | 14,231 of 14,237 (99.96 %) | 0 | 6 |

The 60 absent train clips are the release's train4512 split gap; their steps are flagged 1 and must be masked in the loss. Within clips present, coverage is 99.97 % (train) and 99.96 % (eval); the out-of-tolerance steps are the 9 short clips (96 or 97 frames) plus clips with more than 101 steps. Unknown labels remove a further 43 weather, 13 surface and 13 illumination train steps, and 1 eval surface step.

## Constant baseline on eval (always the majority class, MEASURED)
| variable | classes | majority | accuracy |
|---|---|---|---|
| country | 25 | United States | 0.092 |
| weather | 4 | clear | 0.846 |
| surface | 3 | dry | 0.785 |
| visibility | 2 | good | 0.646 |
| illumination | 4 | day | 0.595 |
| period_of_day | 5 | night | 0.241 |

Eval has 141 clips; weather rain 7 clips, fog 3, snow 9; twilight 1 clip (240 steps). Per-class eval accuracy for rare classes rests on a handful of episodes.

## Fitness for a head
* **country**: fit as a metadata head (25 classes, 71 to 300 train clips each; every eval country has train clips, the smallest has 71). It is a clip constant, so it tests memorisation of place, not perception. Country is confounded with the capture platform: Cramer's V country by `platform_class` is 0.288 (bias-corrected 0.279) on train4572, 0.284 (0.275) on train plus eval; eval alone 0.430 raw and 0.167 bias-corrected, n = 141 over 23 countries. Bulgaria, Estonia, Greece, Hungary, Latvia, Lithuania and Romania are 82 to 93 % hyperion_8.1; Austria, Belgium, Denmark, France, Germany, Italy, Sweden and the US are near 50/50. A head can partly read the platform instead of the country.
* **weather**: fit. Card audit 90.0 % on 40 random frames against 82.5 % always-clear (optimistic, n small). Rare classes: fog is 2.2 % of train steps.
* **illumination and period_of_day**: fit for day, lit_night and dark. Period_of_day is the clip hour, exact but a function of metadata, not of the image. **Twilight is not fit**: the card records 0 of 9 recalled and the share is 1.48 % of train steps; train a head on 3 classes or mask twilight.
* **surface**: borderline. The card ties the always-dry score on random frames (80.0 % each) and wet recall there is 2 of 5. Use with a loss weight below weather and report per class.
* **visibility: unfit.** The card shows 62.5 % against 77.5 % always-good on random frames, and `poor` is never output (0 of 5). Here `reduced` is 35.6 % of train steps and 35.4 % of eval steps against 8 of 40 in the PI's frames, so the label is mostly noise on top of a high constant share; the eval constant baseline is 0.646. Do not train on it.

## Eval-only countries
None. All 23 eval countries have 71 or more train clips. Two train countries have no eval clip (Slovakia, Slovenia). No country leak or out-of-distribution country in eval.

## Limits
* ASSUMED: cache row 0 is video frame 0 and video time is `frame / 30` s. Check: shifting the time base by 0.2 s changes 1.0 % (weather), 0.6 % (surface), 0.5 % (visibility), 0.12 % (illumination) of eval steps, and 0 % for country and period_of_day; at 0.4 s, 1.6 / 1.0 / 0.9 / 0.24 %. The labels are smoothed (switch rates 0.002 to 0.05), so alignment error under 0.4 s is small.
* The cache grid is 0.67 % slower than the 0.2 s label grid, so step 100 sits 0.13 s past the last label frame and is held.
* Frames without a VLM answer (95.5 %) carry a probe label; audit scores are optimistic (same 60 frames guided the choices).
* Only the top class is stored; label distributions (`*_p`) are in the release if soft targets are wanted.
* Cramer's V uses clips in both splits; platform_class has 2 values (hyperion_8, hyperion_8.1).
