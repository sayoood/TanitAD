# FOG_PROVENANCE: where the "fog" weather label comes from (read-only, no relabel)

Evidence class: MEASURED (files named per hop; counts re-computed 2026-11-17 from `labels/rel/frames/*.parquet` and `strat_labels_table.npz`). Context: `q3sheet/RESULT_Q3.md`, fog 2/10 = 0.20 against the PI's blind reading (snow 4, clear 3, rain 1, fog 2, one can't tell).

## Short answer
The fog label is **not** a chain-of-thought keyword. The Alpamayo text was tested as a source and not used (release README, "Alpamayo text as a source": 0 of 9 rain frames found). Weather comes from **two image models**: Qwen2.5-VL-3B answering a one-letter multiple-choice question (option D = "fog or mist") on 1 Hz frames of 1,000 clips, and, on every other frame, a logistic probe on SigLIP 2 embeddings trained against those Qwen answers. Both are smoothed over time by a sticky forward-backward pass. So the "source text" of a fog label is a prompt option plus a probability, not a sentence.

## The chain, hop by hop
Paths: TanitAD repo branch `origin/agent/arch-inf-20260803` (mirror tip 5e7083d8), folder `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-09-context-labels/` (written `RAW/`); labels folder `F:/TanitAD-scratch/strat1/labels/` (written `L/`).
1. `RAW/extract_context.py:33` `WEATHER = ("clear","rain","snow","fog")` (the column order of every `weather_p`). Lines 38-43: SigLIP zero-shot prompts, fog = "a dashcam photo in fog or mist". Line 50: the Qwen question, "...D. fog or mist. Answer with one letter." Lines 55-56: `VLM_NCLS`, `PROMPT_SHA`. `Qwen.score` (about lines 149-170) returns the softmax over the answer letters (constrained choice), so the output is a 4-way probability, never generated text.
2. `RAW/merge_full.py:29` `FIELDS` binds weather to `v_weather` (Qwen) and `z_weather` (zero-shot). Line 41 `PROBE_C = 5.0`; line 42 `VLM_FRAMES_REPRESENTED = 5`. `clip_context` (lines 118-175): a Qwen answer on every 5th frame (`np.arange(0, n, 5)`) of a Qwen clip; a non-finite answer counts as no answer (line 134); every other frame takes the probe (lines 138-143, `other = "probe"`); the Qwen probability is raised to the power 5; `M.sticky_posterior(E, switch)` smooths (switch 0.05 for weather); label = argmax, "unknown" when the top probability is below `UNKNOWN_BELOW` (0.4). `<field>_src` records vlm or probe.
3. Release `Sayood/tanitad-context-labels-v1@100f03da`: `frames/frames_train4512_*.parquet`, columns `weather`, `weather_p`, `weather_src`, `z_weather_p`, `vlm_clip`, `vlm_frame` (README table: "Qwen2.5-VL-3B at 1 Hz on 1,000 clips; a logistic probe on SigLIP 2 embeddings on the other frames", sticky 0.05). Pulled by `L/pull.py:9`.
4. `L/build_strat_labels.py:17` `VARS` includes weather; `load_release` (lines 68-83) builds the vocab sorted without "unknown", which gives clear, fog, rain, snow (the table's order, different from step 1's), and one int8 id per present frame; `build` (lines 40-60) aligns each cache step to the nearest label frame within 0.25 s and writes `weather_id` and `weather_valid`; `main` (lines 99-115) prefixes the split, so the field is `train_weather_id` in `L/strat_labels_table.npz` (sha256 8055a321...cf184eff).
5. Consumers: the weather head and `q3sheet/build_q3.py` read `train_weather_id`.

Second path probed before concluding that no raw text exists: the release parquet has no text column (67 columns, all numeric, categorical or sha); the Alpamayo experiment lives in `RAW/alpamayo_text/` and is explicitly not merged; a name search of `F:/TanitAD-scratch` found a folder `ctxrel`, not opened. UNVERIFIED: whether the pipeline's raw per-clip Qwen parts survive on another disk; they would hold the four Qwen probabilities per 1 Hz frame, still no text.

## How many train clips and windows carry fog
* Release train4512 frames (0.2 s grid): **10,361 fog frames in 334 clips**, of which **878 (8.5 %) are Qwen frames** and **9,483 (91.5 %) probe frames**. 3,743 of the 10,361 lie in clips that have Qwen answers somewhere.
* Cache steps of the train table (`train_weather_id == 1`, `train_flag == 0`, valid): **10,362**.
* The head's train windows (8 NOW steps per clip): **831 fog windows in 260 clips**, **92 Qwen (11.1 %) and 739 probe (88.9 %)**.
* Eligible clips for a fog draw after the Q3 exclusion: 252 (`BUILD_Q3.md`).

## The PI's 10 answered fog frames
Per frame, what produced the label. Probabilities are from the release row for that frame (the smoothed `weather_p`). "Zero-shot" is the raw SigLIP 2 reading with the prompt "a dashcam photo in fog or mist" against three others, kept in the release for comparison; it is not what trained the head. The table has 11 rows: 10 answered plus the one can't tell.

| frame | PI | src | smoothed p(fog) | zero-shot argmax (p), fog p | Qwen answer |
|---|---|---|---|---|---|
| ba9b8423 | snow | vlm | 0.999 | snow (0.889), fog 0.016 | D (fog or mist), 1 Hz answer on this frame |
| 195a8e19 | clear | probe | 1.000 | fog (0.576), fog 0.576 | none on this frame |
| 90861e82 | rain | probe | 1.000 | fog (0.904), fog 0.904 | none on this frame |
| c17dde20 | clear | probe | 0.658 | fog (0.690), fog 0.690 | none on this frame |
| 0dcf8c92 | clear | probe | 0.999 | fog (0.863), fog 0.863 | none on this frame |
| f555c33d | can't tell | probe | 0.900 | fog (0.921), fog 0.921 | none on this frame |
| 32d39ba6 | snow | probe | 1.000 | fog (0.891), fog 0.891 | none on this frame (clip has Qwen frames elsewhere) |
| fb59bc3b | fog | probe | 0.740 | clear (0.396), fog 0.366 | none on this frame |
| 11d9a0a7 | snow | probe | 0.748 | fog (0.555), fog 0.555 | none on this frame (clip has Qwen frames elsewhere) |
| d287888e | snow | probe | 0.814 | snow (0.553), fog 0.280 | none on this frame (clip has Qwen frames elsewhere) |
| 84d1bd8b | fog | probe | 1.000 | fog (0.977), fog 0.977 | none on this frame |

Reading (HYPOTHESIS where marked, the rest MEASURED from the table):
* 9 of the 10 answered (10 of 11 rows) are **probe** frames; only `ba9b8423` is a Qwen frame, so the label is mostly the probe's extrapolation of Qwen's behaviour. Qwen's own letter on that frame is not stored; smoothed p(fog) 0.999 with src vlm means its top letter was D. The PI read snow, and the zero-shot reading of the same frame is snow 0.889, fog 0.016.
* The probe reads fog at 0.66 to 1.00 on every row; the PI answers (snow 4, clear 3, rain 1) say the probe's fog class catches scenes that are snowy, bright or wet. Zero-shot also reads fog on 8 of 11 rows (HYPOTHESIS: the probe inherits a whiteness or low-contrast cue shared by snow, haze and overcast; not tested here).
* On the two frames the PI calls fog (`fb59bc3b`, `84d1bd8b`) the label is probe-sourced too, at 0.74 and 1.00.
* The "source text" for each is therefore: the prompt option "D. fog or mist" (Qwen, 1 frame) or no text at all (probe, 10 frames); the signal is a SigLIP 2 embedding plus logistic weights. There is no per-frame sentence to quote.

## What a stronger labeller could be (options only, nothing built)
All of these are a new card and need the PI's own go.
1. **Qwen on every fog candidate frame**: re-score the 334 clips that hold probe-fog frames at 5 Hz-equivalent coverage; about 11 s per clip (README measurement) is about 1 GPU-hour. Weakness: Qwen-3B produced the fog letter on the one Qwen frame the PI read as snow.
2. **A larger VLM** (a 7B or larger open VLM licensed for research) on the same candidates, with a question that separates snow in the air or on the ground from fog or haze, checked on the PI's answered frames before use.
3. **A PI-audited supervised probe**: train the probe on the PI's answered frames plus a further audited fog and snow batch instead of Qwen, split by clip. With 2 true fog frames of 10 the fog class is thin, so a fog-focused audit would come first.
4. **Merge fog into a precipitation-or-haze class**, or drop fog from the weather vocabulary until a labeller reaches the bar; fog would move to the visibility field (itself unfit per the release README).
5. **Cheap physical cues as a veto**: the release already stores `luma_*`, `sky_luma`, `sat_mean`, `lit_frac` and a sun-elevation estimate; weak alone, usable to flag low-contrast frames against Qwen.
Each option is scored the same way: a blind sheet on per-class label precision (this instrument), with the bar of `PREREG_Q3.md`.
