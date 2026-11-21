# PREREG_Q3RAIN: second blind draw of the rain label (follows PREREG_Q3.md, PI go 2026-11-17)

Written before any frame was drawn or seen. Its sha256 is in `PREREG_Q3RAIN.sha256`, written before the build ran. Tier: none (label quality). Evidence class of the result: MEASURED (label vs the PI's blind answers).

## Why
`RESULT_Q3.md`: the rain label read 8/11 = 0.727 (Wilson 95 % [0.434, 0.90]), which `PREREG_Q3.md` classes as borderline and for which it commits "a second blind draw of that class (about 20 frames) before any relabel". This is that draw, with the same instrument.

## Fixed inputs (same as PREREG_Q3.md)
Label `train_weather_id` in `F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz` (sha256 `8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff`), TRAIN split, windows and image convention as in PREREG_Q3.md (8 NOW steps per clip, `train_flag == 0`, `train_weather_valid`; image = video frame 6k of the label frame index k; optional second image 1 s earlier).

## The draw (deterministic)
* 20 frames, label class rain only, one per clip, 20 distinct TRAIN clips.
* Seed string `q3rain-weather-v1` (new). Every eligible window gets `h = sha256("q3rain-weather-v1:" + clip + ":" + t)`; sort by `h`, keep the first window of each clip, take the first 20 clips.
* Excluded clips: all 60 clips of the PI's earlier pilot answers (`pilot_audit_answers_PI_60.json`) and all 40 clips of the Q3 sheet (`q3sheet/key.json`). Zero overlap is a control.
* Presentation order: `numpy.random.default_rng(20261118).permutation(20)`. Frame ids are opaque (8 hex of `sha256("q3rain-weather-v1-id:" + clip + ":" + k)`). The sheet shows no label, model output, class count or clip id. All five answers are offered (clear, fog, rain, snow, can't tell), so a rain frame can be called anything. Same answer format as Q3 (`q3-answers-v1`).
* Same sampling note as Q3: this measures PRECISION of the rain label, not recall.

## Scoring (what `score_q3rain.py` computes)
Reads BOTH answer files (`answers_q3.json` with `q3sheet/key.json`, and the new answers with `key_q3rain.json`). For label rain: pooled agreement a / n over the 11 Q3 rain frames plus the 20 new ones (31 frames; n counts frames not answered can't tell), Wilson 95 % interval, can't tell count, the answer distribution (clear, fog, snow, rain), and the second draw alone for reference (not decision-bearing).

## Bar (Q3's bar applied to the pooled count)
| pooled rain agreement | reading |
|---|---|
| >= 0.80 | rain label sound at this n; no relabel for rain |
| < 0.70 | relabel candidate for rain |
| 0.70 to 0.80 | stays borderline. Reason: at n = 31 the Wilson interval is about 0.17 wide, so a point value inside the 0.10 band cannot be separated from either line; no third draw is registered, the rain relabel is then left to a PI decision on cost (the fog and rain relabel can share one card) |
With n answered < 25 of the 31 (more than 6 can't tell) the pooled reading is reported as NOT AUDITABLE at this n. In counts: 25 of 31 = 0.806 (sound), 24 of 31 = 0.774 (borderline), 21 of 31 = 0.677 (candidate); 22 of 31 = 0.710 is the lowest borderline count.

Not allowed after the answers are seen: moving the lines, dropping or re-drawing frames, excluding the 11 Q3 rain frames from the pool, changing can't tell handling. A post-hoc cut is labelled post-hoc.

## Controls
Disjointness of the 20 clips from the 100 excluded clips (reads 0), no duplicate clips, the sheet holds none of the clip ids or label-file words, the scorer on planted answers (all rain gives 1.0 pooled over 31; one planted mismatch moves exactly one count).
