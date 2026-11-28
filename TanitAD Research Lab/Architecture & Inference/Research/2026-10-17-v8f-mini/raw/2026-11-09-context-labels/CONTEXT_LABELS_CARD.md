---
pretty_name: TanitAD context labels v1
tags:
- autonomous-driving
- labels
---
# tanitad-context-labels-v1

Per-frame strategic context labels for the TanitAD programme (PI ruling 2026-11-08, `REQ_HORIZONS.md`, `SPEC_HWM.md` section 6). Private. Parquet only: no image or camera frame is in this repository. These are labels, built offline. Period of day and illumination at inference come from images and past frames only; `country`, `month` and `hour_of_day` here are source metadata and are never model inputs.

All numbers are MEASURED (2026-11-09) unless marked. They describe label quality, not driving.

## Contents
* **470,475 rows, one per present frame at 0.2 s, 4,659 clips** (eval: 147, train4512: 4,512), 25 countries, a country on every row.
* `frames/frames_*.parquet`: per-frame geometry and context columns. `clips/clips.parquet`: one row per clip. `manifest.json`: sha256 of every file.
* 4,606 clips have 101 frames, 44 have 100, and 9 have 96 or 97. Those 9 really lack the frames in the source video (101 geometry rows, 96 or 97 present). Every present frame is labelled.

## Fields
| field | classes | source | temporal treatment (switch rate) |
|---|---|---|---|
| `country`, `month`, `hour_of_day` | | collection metadata, every frame | none |
| `weather` | clear, rain, snow, fog | Qwen2.5-VL-3B at 1 Hz on 1,000 clips; a logistic probe on SigLIP 2 embeddings on the other frames | sticky forward-backward, 0.05 |
| `surface` | dry, wet, snow_ice | as weather | 0.01 |
| `visibility` | good, reduced, poor | as weather; a not-good verdict needs two Qwen questions to agree | 0.002 |
| `illumination` | day, twilight, lit_night, dark | SigLIP 2 zero-shot | 0.05 |
| `period_of_day` | night, morning, midday, afternoon, evening | clock hour (metadata) | none |
| `sun_el_deg_est`, `sun_band_est` | | estimated from location and time metadata | none |

Each field has `<field>_p` (distribution) and `<field>_src` (`vlm`, `probe`, `zeroshot`) columns. A frame whose top probability is below 0.4 reads `unknown`. `vlm_frame` marks the 20,986 frames (4.5 %) with Qwen answers, and `vlm_clip` the 1,000 clips with them. The clips were selected to oversample rare conditions; per-clip weights undo this in probe training. Model and prompt identities are in `model_sha_siglip`, `model_sha_qwen` and `prompt_sha`.

## Share of frames
| field | shares (%) |
|---|---|
| weather | clear 85.96, snow 7.16, rain 4.58, fog 2.28, unknown 0.01 |
| surface | dry 79.30, wet 12.87, snow_ice 7.83 |
| visibility | good 64.41, reduced 35.59, poor 0.00 |
| illumination | day 57.04, lit_night 24.77, dark 16.68, twilight 1.50 |

## Audit against the PI's blind answers
60 frames: 40 random, 20 chosen rare; answers exported before any reveal. Wilson 95 % intervals on frame counts. "Always commonest" is the score of always answering the commonest class in the same frames. `SCORE_RELEASE.json` holds the full read.

| field | 40 random frames | always commonest | all 60 | always commonest | recall on all 60 |
|---|---|---|---|---|---|
| weather | 36/40 = 90.0 % [76.9, 96.0] | 82.5 % | 52/59 = 88.1 % [77.5, 94.1] | 67.8 % | clear 37/40, rain 7/9, snow 7/9, fog 1/1 |
| surface | 32/40 = 80.0 % [65.2, 89.5] | 80.0 % | 50/60 = 83.3 % [72.0, 90.7] | 65.0 % | dry 32/39, wet 11/14, snow_ice 7/7 |
| visibility | 25/40 = 62.5 % [47.0, 75.8] | **77.5 %** | 39/59 = 66.1 % [53.4, 76.9] | 59.3 % | good 24/35, reduced 15/19, poor 0/5 |
| illumination (glare counted as day) | 30/39 = 76.9 % [61.7, 87.4] | 41.0 % | 42/59 = 71.2 % [58.6, 81.2] | 42.4 % | day 21/25, lit_night 12/14, dark 9/11, twilight 0/9 |
| rough time of day (from illumination) | 32/39 = 82.1 % [67.3, 91.0] | 41.0 % | 48/59 = 81.4 % [69.6, 89.3] | 47.5 % | day 21/21, night 27/28, dawn or dusk 0/10 |

**These scores are optimistic.** The same 60 frames guided the prompt, probe and smoothing choices below, and n is small.

## How far to trust each field
* **Weather and illumination:** above the always-commonest score. Usable, with the interval in mind.
* **Surface:** ties the always-commonest score on the random frames; wet recall there is 2/5. Better on the rare frames (all 60). Use with care.
* **Visibility: below the always-commonest score on the random frames, and it never outputs `poor` (0 of 5 PI frames).** The 35.59 % `reduced` share looks too high against the PI's 8 of 40. Do not use it as a training target before it is improved.
* **Twilight and dawn or dusk: not detected (0 of 9, 0 of 10).** The sun-band estimate does not recover them (0 of 10). The `twilight` share of 1.50 % should not be read as a corpus share.
* **Frames without Qwen (95.5 %) carry the probe's label.** In the 187-clip pilot, the out-of-fold probe matched Qwen on 84.8 % (weather), 83.3 % (surface) and 90.7 % (visibility) of 1 Hz frames.

## Choices made after seeing data
1. **Visibility:** the frozen selection rule kept one question (20/40 on the report frames, below always-good at 31/40). Combinations were then read under the same selection score, four tied, and the tie was broken on the report frames, which makes the choice post hoc: a not-good verdict needs two questions to agree (26/40 in that read). The raw answers of both questions are kept in the per-clip parts of the pipeline.
2. **Surface smoothing:** the written rule picked raw (83.31 % against 82.70 % at switch 0.01). The rule scores against Qwen's 1 Hz answers, which flicker themselves. Raw flips 3.33 % per step against 0.134 % at 0.01; the PI's random frames tie (34/37 both). Switch 0.01 is used.
3. **Probe:** C = 5, unweighted, for every field. The selector picked class-balanced for weather; on the PI's frames that read 45/55 against 50/55 unweighted, and the release publishes distributions, which class weighting biases.
4. **Glare** is folded into day: 0 of the PI's 4 glare frames matched.
5. **View quality is not released** (not credible in the pilot and not among the PI's fields), and the camera-view question was dropped from the 1,000-clip run.
6. **One non-finite Qwen answer** (clip `e8e57d1b`, 1 Hz frame 3, of the 20,986) counts as no answer; that frame takes the probe's label.

## Alpamayo text as a source (tested, not used)
The PI asked to search the included Alpamayo text for this vocabulary. Over 4,729 clips, only 146 (3.1 %) mention weather, surface or visibility, 275 (5.8 %) mention light, and 381 (8.1 %) mention either. On the PI's audited clips, the text found 0 of 9 rain and 1 of 9 snow frames. The text describes driving decisions and is too sparse for per-frame labels. No Alpamayo text enters this release.

## Next levers
* Visibility and twilight need more audited examples of those classes, a stronger prompt or both. With 19 reduced, 5 poor and 9 twilight frames audited, there is too little to choose a better one without overfitting.
* Qwen on more clips is possible at about 11 s per clip (MEASURED on this run), but it would not by itself fix visibility, whose Qwen-only score against the PI was 20/40.

## Reproduce
Scripts: `merge_full.py`, `extract_context.py`, `chain_full.py`, `score_release.py`, `audit_prompts.py`, `PROMPT_DECISION.md`, in `raw/2026-11-09-context-labels/` of the TanitAD repository.
