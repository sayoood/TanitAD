# PREREG_Q3: blind audit of the weather label on the minority classes (Q3, PI go 2026-11-17)

Written before any frame was drawn or seen. Its sha256 is in `PREREG_Q3.sha256`, written before `build_q3.py` was run. Tier: none (label quality, no model output, no driving claim). Evidence class of the result: MEASURED (label vs the PI's blind answers).

## Question
How often does the weather label (vocab clear / fog / rain / snow) agree with the PI's own reading of the frame, per label class? The earlier audit (`H-CTX-LABELS-V1`, 40 random frames, 36/40 = 90.0 % [76.9, 96.0]) had no per-class check, and a random draw holds almost no fog, rain or snow. The weather head reads 0.936 to 0.939 against this label (MEASURED, `RESULT_WEATHERPROBE.md`), at or above that agreement, so the label is the next lever.

## Label source and frame population (fixed)
* Label: `F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz` (sha256 `8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff`), field `train_weather_id`, vocab `clear, fog, rain, snow`. This is the label the weather head trained on.
* Population: the TRAIN split (4,572 clips). NOW frames of a train window are the 8 steps `unique(round(linspace(3, n-1, 8)))` of each clip (as in `wprobe_fit.py`), kept when `train_flag == 0` and `train_weather_valid`. That gives 36,051 windows, class counts clear/fog/rain/snow 30,988 / 831 / 1,647 / 2,585 (MEASURED, `RESULT_WEATHERPROBE.md`; re-counted by the build).
* Image shown: the front-wide camera frame of the label's own frame index k (`train_frame_idx`), video frame 6k at 30 fps (the context-label convention). It lies within 0.14 s of the cache step. The optional second image is frame k-5 (1 s earlier), shown only when k >= 5.
* Excluded: every clip of the PI's earlier pilot answers (60 frames, 60 clips, 56 of them train clips; the 40 random-stratum frames are the `H-CTX-LABELS-V1` set) from `raw/2026-11-09-context-labels/pilot_audit_answers_PI_60.json`. All 60 are excluded, which is stricter than the 40.

## The draw (deterministic)
* 40 frames, one per clip, 40 distinct clips. Quotas by the label's class: **fog 11, rain 11, snow 11, clear 7**.
* Seed string `q3-weather-v1`. Every eligible window (clip, step t) gets `h = sha256("q3-weather-v1:" + clip + ":" + t)`. Classes are processed in the order fog, rain, snow, clear. For a class, take the eligible windows of that class on clips not excluded and not yet used, sort by `h`, keep the first window of each clip, and take the first quota clips.
* Presentation order: `numpy.random.default_rng(20261117).permutation(40)`. Frame ids are opaque (first 8 hex of `sha256("q3-weather-v1-id:" + clip + ":" + k)`). The sheet shows no label, no model output, no class count and no clip id.
* Sampling note: the draw is stratified on the label's class, so it measures PRECISION of each label class (of frames labelled c, how many does the PI call c). It does not measure recall (labelled clear but truly fog) and it is not an unbiased estimate of overall label accuracy.

## Answers
Per frame the PI picks clear, fog, rain, snow or can't tell, plus an optional note. A "can't tell" frame is dropped from that class's denominator and counted separately.

## Scoring (exactly what `score_q3.py` computes)
1. Per label class c: n_c (frames answered other than can't tell), a_c (PI answer equals c), agreement a_c / n_c, Wilson 95 % interval (z = 1.959964), and the count of can't tell.
2. Overall: pooled a / n over the 40 (stratified, NOT comparable to the 90.0 % of the random draw), and the prevalence-weighted agreement `sum_c w_c * (a_c / n_c)` with w = train window shares 30,988 / 831 / 1,647 / 2,585 over 36,051 (ESTIMATED, a reweighting of 40 frames, reported with no interval).
3. The 4 x 4 confusion, rows = label class, columns = PI answer (clear, fog, rain, snow), with a fifth column for can't tell.

## Bar and what each outcome means (committed before any frame is seen)
The bar is on the point agreement of each class, read together with its interval. A 0.70 line is chosen because a label class that agrees under 0.70 with the PI has more than 30 % wrong positives, which puts a ceiling of 0.70 on the head's precision for that class against the PI's truth no matter how well it fits the label, and the head's 0.94 overall already sits above the 0.90 label agreement. With n = 11 per class, 8/11 = 0.727 has Wilson 95 % [0.43, 0.90], so one class cannot be separated from the line by its interval alone; the decision therefore rests on the point value and is stated that way.

| per-class outcome (n_c >= 8 answered) | meaning for the lever |
|---|---|
| agreement >= 0.80 | the label class is sound at this n; no relabel for that class; further head accuracy against it is still a real signal for that class |
| 0.70 <= agreement < 0.80 | borderline; a second blind draw of that class (about 20 frames) before any relabel |
| agreement < 0.70 | **relabel candidate**: the label is the bottleneck for that class; the next arm is a relabel (PI-audited classes, relabel from the PI answers plus a stronger labeller), as a new card with its own go |
| n_c < 8 (more than 3 can't tell) | the PI cannot resolve that class from a single front frame; the class is not auditable by this sheet; the sheet design (more frames of the clip, other camera) is the next step |

Whole-sheet reading: if all three minority classes read >= 0.80, the label is not the weather lever and the head's next lever is the head or the features. If one or more minority classes read < 0.70, the label is the lever for those classes. The clear class (7 frames) is a sanity check only (no bar; the lower bound of its interval is reported); a clear label that the PI calls fog, rain or snow is recorded as a recall problem of the label for that class and goes in the confusion.

Not allowed after the answers are seen: moving the 0.70 or 0.80 lines, dropping a class, re-drawing frames, or choosing "can't tell" handling differently. Any post-hoc cut is labelled post-hoc and gets its own registration.

## Controls
* `score_q3.py` on a planted all-match file must give agreement 1.0 for every class, and one planted mismatch must move exactly one cell of the confusion and exactly one class count.
* Disjointness (must read 0): drawn clips vs the 60 excluded clips; duplicate clips in the draw (must read 0).
* The sheet is built without reading any label into it: `grep` of the sheet for the words of the label file and the clip ids must give 0 apart from the class words in the radio buttons.
