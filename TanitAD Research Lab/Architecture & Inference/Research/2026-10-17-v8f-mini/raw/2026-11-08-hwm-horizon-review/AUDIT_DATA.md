# AUDIT_DATA: data audit against REQ_HORIZONS (R-STR, R-VIS, R-DATA) and per-frame extraction design

Author: data-audit subagent, 2026-11-08. Read-only work: no GPU, no launch, no download above 50 MB, no commit, no edit of code or registers. Evidence classes: MEASURED (with artifact or command), PUBLISHED (cited), INHERITED (not re-verified), ESTIMATED, HYPOTHESIS.

Scratch scripts and outputs were kept in the session scratchpad (listed for deletion at the end).

---------------------------------------------------------------------------------------------

## 1. THE BLOCKING QUESTION: do camera frames exist more than 20 s after a clip start?

### 1.1 Answer

**No. Video stops at about 20.17 s. Egomotion does not: it runs 20 to 141 s (median about 140 s) from the same start.** The two register rows describe different layers and both are right for their layer:

| layer | object | duration | register row that describes it |
|---|---|---|---|
| provider camera file | `camera_front_wide_120fov/<clip>.mp4` plus `.timestamps.parquet` | **20.13 to 20.17 s**, 604 to 605 frames at 30 fps | `D-DATA-HORIZON` (line 3293: "clips are 19.9 s") describes this layer (and the 20 s episode cache) |
| provider egomotion | `labels/egomotion/*.zip` (per clip parquet), `labels/egomotion_alpamayo/<clip>.parquet` | **median 140 s, min 20.2 s, max 141 s** | `C141` (line 3377) describes this layer |
| episode cache | `_epcache/physicalai-*/ep_*.pt` | 20 s window | the thing C141 blames |

So the strategic band 6 to 30 s is **observable in egomotion-derived geometry for most clips, and observable in real camera frames for none beyond 20.17 s**.

### 1.2 Two independent probes per quantity

**(a) Video duration.**
* Probe 1 (MEASURED, local files): `ffprobe` on 4 mp4 in `C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/` (every 1500th file): duration 20.1333 to 20.1667 s, 604 or 605 frames, 1920x1080, 30 fps.
* Probe 2 (MEASURED, timestamp tables): 967 clips that have both a front-wide `.timestamps.parquet` and a local egomotion table: video span median **20.133 s** (join script output, `scratchpad/join.json`).
* Probe 3 (PUBLISHED): the HF dataset card (`nvidia/PhysicalAI-Autonomous-Vehicles`, README fetched anonymously, 35 kB): "306,151 clips that are each 20 seconds long"; lidar "approximately 200 lidar spins (10 Hz capture rate for a 20 sec clip)". The card header says 1700 hours, 25 countries.

**(b) Egomotion duration.**
* Probe 1 (MEASURED, local provider zips): 197 chunk zips under `labels/egomotion/`, **18,987 clips**: duration min 20.21 s, p10 43.0 s, median **140.05 s**, p90 140.37 s, max 140.40 s. Median 2,902 rows per clip (about 20.7 Hz).
* Probe 2 (MEASURED, second artifact family): `labels/egomotion_alpamayo/<clip>.parquet` for the **4,512 train clips**: min 20.21 s, p5 30.95 s, median **139.69 s**, max 141.25 s (`scratchpad/alp_dur4512.json`).
* Probe 3 (MEASURED, HF side, anonymous tree API): the listing of `labels/egomotion/` has 1,000 chunk zips, mean 405,885 bytes per clip if 100 clips per chunk. 103 of our 197 local zips are byte-size identical to the HF listing entries (the other 94 differ in size; I did not chase why, so "local = HF" is shown for 103 of 197 only). Size per clip is consistent with 2,900 rows of 15 float columns, not with 400 rows (a 20 s table).
* Alignment (MEASURED, 967 joined clips): egomotion starts **0.115 s (median) before the video starts** (range -0.16 to +0.23 s) and ends **119.7 s (median) after the video ends**. Same clock, same origin (card: local frame with origin at timestamp 0).

I did not read an egomotion header over the network (the dataset is gated, a token would be needed); the size comparison above is the HF-side probe.

### 1.3 Counts that support a key frame with 30 s of egomotion-derived checkpoints

Source: `egomotion_alpamayo` durations, key frame at time `t_k` after clip start needs egomotion span at least `t_k + 30` (MEASURED, `scratchpad/alp_dur4512.json`, `scratchpad/p10.py`).

| set | n | key at 0 s (span >= 30) | key at 7.8 s, the label anchor (span >= 37.8) | key at 20 s, the last video frame (span >= 50) |
|---|---|---|---|---|
| `TRAIN_CLIPS_4512.json` | 4,512 | **4,293 (95.1 %)** | **4,211 (93.3 %)** | **3,950 (87.5 %)** |
| `clips141.json` / `.txt` | 141 | **137 (97.2 %)** | **133 (94.3 %)** | **124 (87.9 %)** |

Notes:
* `clips141.json` is the 141-clip training stream `wmact/train_stream`, and all 141 ids are the first 141 ids of `TRAIN_CLIPS_4512.json` (overlap 141 of 141). It is **not** the eval-141 set. The eval label file `s2_labels_v7.2_eval.jsonl.gz` holds 147 records (0 overlap with train 4,512), the census says eval141 has 141 clips; I did not reconcile 147 and 141.
* The register's own figure for the 801 labelled clips was 94.5 % with the full band (INHERITED, C141); the 93.3 % at the 7.8 s anchor here is on 4,512 clips and is in line.
* Real future frames: **0 of 4,512 and 0 of 141** clips have a real frame 30 s after any key frame (video ends at 20.17 s, so `t_k + 30 > 20.17` for every `t_k >= 0`). A key at 0 s has real frames to 20.17 s, which covers 6 to 20 s of the 6 to 30 s band (14.2 of 24 s, 59 %). A key at 7.8 s sees 12.4 s ahead, so 6 to 12.4 s (about 6.4 of 24 s, 27 %, counted from key + 6 s to key + 20.17 - 7.8). A key at 14.2 s or later sees nothing in the band.

### 1.4 What can and cannot be supervised in the 6 to 30 s band

Can be supervised, on observed data:
* AV checkpoints (ego route points at 6 to 30 s) from egomotion: poses at every 0.2 s up to key + 30 s, for the clip counts above. Geometry only; privileged label, allowed by the standing rule.
* Strategic goals derived from the geometry (the existing `g_str`/`a_str` tokens, for example `TURN_RIGHT_FOLLOW_ROUTE`, `by_time_s` 11.1) over the observed band; `horizon.available_s` in the v7.2 label file records it per clip (first record: 35.0 s, recording span 132.2 s).
* Image-conditioned prediction of the above from the key frame (and past frames): input is visual, target is geometric.

Cannot be supervised from this corpus alone:
* Any target that needs a real image later than 20.17 s after clip start (future-frame or latent prediction in the band beyond 20.17 s minus the key time; weather change, a sign or junction first seen at 25 s).
* Anything about the scene beyond the front camera at those times (lidar and radar are also 20 s; the card says about 200 lidar spins).
* Maps, lane graph, traffic-light features (already settled in the standing rules, not re-derived).

### 1.5 Can consecutive clips be chained? (the Data Strategy doc says chunks are not sessions)

* `DataEng/DATA_STRATEGY.md` line 677: 1,418 chunks for 4,729 clips, median 2 clips per chunk, max 76 (INHERITED, for our subset). In the whole corpus (`clip_index.parquet`, MEASURED): 3,146 chunks, median 99 clips per chunk. Chunks are 90.5 % country-pure and span a median 20 distinct hours of day, so **a chunk is not a drive**.
* **But sessions do exist inside chunks, and the egomotion shows it.** Test (MEASURED, `scratchpad/p6.py`): for the 18,988 clips of the 197 local chunks, compare clip B's first 20 s speed profile (0.2 s grid, std at least 1 m/s) with every 20 s window of clip A's egomotion from 20 s onward, within a chunk. **509 (A, B) pairs matched with RMSE below 0.05 m/s** over 100 samples (a chance match is not plausible). Offsets: many at exactly 20, 60, 80, 100 s, others at 21.2, 23.8, 104.0, 116.4, 119.4 s, so B is a later, sometimes overlapping, window of A's own drive (`scratchpad/chain_hits.json`). Rate: 509 / 18,988 = **2.7 % of clips have a detected continuation in their own chunk** (lower bound: same-chunk only, moving clips only, one threshold).
* For the training set (MEASURED, `scratchpad/p9.py`, 4,800 alpamayo clips, 1,415 chunks): **22 pairs; 21 of the 4,512 train clips have a successor clip inside the 4,800; 20 pairs have both ends in train**. Chaining is therefore not a usable source of real frames beyond 20 s for our corpus today (0.5 % of clips), and it needs successor clips pulled from HF at about 12 MB each if wanted.
* Cross-chunk continuation was not tested (HYPOTHESIS: rare, because a chunk is a country-pure block). Clip B would only help if its video is local or pulled.
* Consequence for splits: **same-drive clips sit in the same chunk, so a clip-level split leaks**; see section 4.

---------------------------------------------------------------------------------------------

## 2. Inventory per signal, and the proposed frame-level schema

Layers are stated per row, never "our ingest" in general.

| signal | exists where | granularity | provenance | verdict |
|---|---|---|---|---|
| country | `metadata/data_collection.parquet` column `country`, 306,152 rows, 0 nulls, 25 countries (MEASURED, census `CENSUS_STRAT0.json`; `strata.country` also in v7.2 labels) | per clip | provider metadata | **complete** |
| month, hour of day | `data_collection.parquet` `month`, `hour_of_day`, 0 nulls | per clip (integer hour; constant over 20 s) | provider metadata; hour semantics (local vs UTC) not documented; the census diurnal check treats it as local clock | complete, semantics HYPOTHESIS |
| period of day | v7.2 labels `strata.daynight_clock` (day 2,452 / night 2,120 of 4,572, MEASURED) | per clip, 2 classes | derived from hour | coarse, clock based |
| illumination | census `illum_all_ESTIMATED`: night 117,015 / day 110,359 / low_sun 45,668 / twilight 33,110 (306,152 clips); train 4,512: night 2,002, day 1,380, low_sun 623, twilight 507 | per clip, 4 bands | **ESTIMATED**: country-centroid latitude, mid-month declination, hour read as local solar time | no image-derived value anywhere. Clock and sun disagree for 22,754 + 4,360 = 27,114 clips (8.9 %, ESTIMATED) |
| weather | **absent** in the provider metadata (columns are only `country, month, hour_of_day, platform_class, radar_config`, MEASURED). The card text lists "clear, rain, snow, fog" as dataset diversity (PUBLISHED), but no per-clip field exists. `lake/enrich.py` `weather`/`time_of_day` axes are stubs (INHERITED). CoT mentions: snow 65 of 4,572, wet road 11 (INHERITED) | none | none | **missing** |
| road surface, visibility | absent | none | none | **missing** |
| road class | v7.2 labels `strata.road_class` (urban 2,965 / intersection 861 / highway 746 of 4,572, MEASURED) | per clip (4,572 labelled) | geometry/heuristic | partial, not a weather signal |
| scene visibility flags | v7.2 `scene.traffic_light_visible`, `vru_visible`, `vehicle_visible` | per labelled clip, at the anchor | VLM grounding, disputed | not weather |
| AV checkpoints at 6 to 30 s | egomotion tables (see section 1), poses at 20.7 Hz | per frame derivable | provider egomotion | **derivable, not stored**. Today's S head uses 30, 50, 80 m checkpoints (REQ_HORIZONS reading 1) |
| strategic goals | v7.2 `g_str`/`a_str`, 4,572 records, `t0_s` = 8.0 for all, tokens FOLLOW_ROUTE 2,948, TURN_RIGHT 596, TURN_LEFT 574, STOP_AT 454 (MEASURED, `raw/s2_labels_v7.2_train.jsonl.gz`) | **one anchor per clip** (t0 8.0 s), band 8 to 30 s, `bands.strategic_s` = [8, 30] | geometry | single anchor only; band is 8 to 30, not 6 to 30 |
| detected strategic information (R-STR) | none beyond the above | none | none | **missing** |

### 2.1 Missing

1. Weather, road surface, visibility (image-derived).
2. Illumination from images (R-VIS requires image-only at inference, so an image label is needed for the training target and the evaluation).
3. Per-frame (0.2 s grid) versions of everything; today everything is per clip or one anchor.
4. Checkpoints and goals at every key frame, in the 6 to 30 s band (today: one anchor, 8 to 30 s).
5. A session or drive key (section 4).

### 2.2 Proposed schema (Parquet, one row per clip and 0.2 s grid frame)

Grid: `frame_idx` k = 0..100, `t_s` = 0.2 k, video frame number = round(30 * t_s) = 6k (30 fps source).

| field | type | provenance | confidence |
|---|---|---|---|
| `clip_id`, `frame_idx`, `t_s`, `video_frame` | str, int, float, int | grid | exact |
| `country`, `month`, `hour_of_day` | copied | `metadata` | exact; `meta_source` = data_collection |
| `sun_el_deg_est`, `sun_band_est` | float, enum (night, twilight, low_sun, day) | ESTIMATED from country centroid, month, hour; **weak cross-check only, never a label** | `sun_conf` = low (US spans several time zones) |
| `luma_mean`, `luma_p05`, `luma_p95`, `sky_luma`, `sat_mean`, `headlight_frac`, `lit_frac` | float | image statistics | exact |
| `illum_img` (dark, lit_night, twilight, day, glare) + `illum_img_p` (probabilities) | enum + 5 floats | image only (stats + image model), calibrated on the PI audit; the primary illumination label | calibrated probability |
| `illum_agree_sun` | bool | cross-check only | derived |
| `weather` (clear, rain, snow, fog, unknown) + `weather_p` | enum + floats | VLM / probe | probability; `weather_src` = vlm|probe|smoothed |
| `surface` (dry, wet, snow_ice, unknown) + `surface_p` | enum + floats | same | probability |
| `visibility` (good, reduced, poor) + `visibility_p` | enum + floats | same | probability |
| `view_quality` (clean, wiper, glare, occluded, tunnel) | enum | VLM + stats | probability |
| `ckpt_xy_6..30s` | list of (x, y) in ego frame at +6, 10, 14, 18, 22, 26, 30 s (or the 30/50/80 m route points) | egomotion | `ckpt_valid_mask` per entry (span too short = false) |
| `g_tac6_*` / `g_str_*` : `band_lo_s`, `band_hi_s`, `token`, `by_time_s`, `within_m`, `valid` | float, float, str, float, float, bool | geometry, recomputed per key frame | `valid` |
| `g_gap_6_8s_token`, `g_gap_6_8s_valid` | str, bool | geometry, 6 to 8 s window | explicit owner of the 6 to 8 s window |
| `session_key`, `chunk` | str, int | section 4 | exact / derived |
| `label_version`, `model_sha`, `prompt_sha` | str | audit trail | exact |

**The 6 to 8 s window.** v7.2 `bands` are operative [0,2], tactical [2,6], strategic [8,30] (MEASURED, record 0), while R-STR starts at 6 s: the 6 to 8 s window has no owner in v7.2. The per-frame schema therefore stores the strategic band as [6, 30] with the 6 to 8 s window carried as its own field (`g_gap_6_8s_*`) so that the old [8, 30] label can be reproduced and the new one is not silently a different object. Which level (tactical vocabulary stops at 6 s) owns the window is a Master Mind / PI decision recorded in the label version.

Clip-level table (one row per clip) carries the clip aggregates (mode and confidence of each field) and the split. Labels are build-time only; none of them is an inference input (`R-VIS`, standing rule of 2026-08-03). The goal path stays information-disjoint from the situation classifier: `g_str_*` are computed from egomotion, not from classifier output.

---------------------------------------------------------------------------------------------

## 3. Extraction design for the missing fields

### 3.1 How many frames

Frames = clips x frames per clip, at 0.2 s: 101 per 20.17 s clip.

| scope | clips | frames at 5 Hz | frames at 1 Hz |
|---|---|---|---|
| train 4,512 | 4,512 | 455,712 | 91,000 (20 per clip, rounded) |
| train + eval 147 | 4,659 | about 470,600 | 93,200 |
| whole corpus (front wide only) | 306,152 | 30.9 M | 6.1 M |

### 3.2 Which signal needs which rate

Weather, surface and country are constant over 20 s except for occlusion events; period of day is constant at clip level (integer hour). Only illumination (tunnels, underpasses, headlight glare, sun at the horizon) and view quality vary inside a clip. HYPOTHESIS, to be tested on the audit sample: the VLM needs to run at 1 Hz and the 0.2 s grid is filled by a cheap per-frame head plus temporal smoothing. This is the reason the design below has three tiers instead of a VLM on every frame.

### 3.3 Options compared

| option | what it gives | VRAM | ESTIMATED throughput on RTX 4060 8 GB | strengths | weaknesses |
|---|---|---|---|---|---|
| A. Small open VLM, for example Qwen2.5-VL-3B-Instruct or a 2 to 4B Qwen3-VL class model, 4-bit or int8, scored by constrained choice (log-probabilities over fixed answers) | weather, surface, visibility, view quality in one pass; free text for the audit | 3 to 5 GB (4-bit), 7 GB (bf16, too tight with the lock) | 2 to 4 frames/s at 448 px, short prompt, single-token answers | best semantic accuracy on rain/snow/fog/wet; research licence acceptable | slowest; label noise from prompt wording; must be calibrated on the audit |
| B. Zero-shot image-text model (SigLIP2 or a CLIP class model) | weather / illumination / surface by text prompts, and a 1 to 2 k-dim embedding per frame | 1.5 to 2.5 GB | 100 to 150 frames/s at 384 px fp16 | fast, runs on the 5 Hz grid | weak on thin weather (light rain, wet road); prompts are fragile; needs the audit to calibrate |
| C. Image statistics for illumination | luma percentiles, sky-region luma, saturation, headlight mask, histogram | none (CPU) | decode-bound | exact, transparent, per frame | cannot tell night from a tunnel or dusk from overcast without context; no weather |
| D. Sun position from country, month, hour | cross-check only | none | none | independent of pixels | ESTIMATED, coarse (centroid latitude, US time zones, DST, unknown hour semantics); 8.9 % clock/sun disagreement; must never be a label on its own or an inference input |
| E. Linear or small MLP probe on features already cached | per-frame labels at 5 Hz from the DINOv3 features that box 2 already holds (`D:\tanitad\cache_full\train`, 89 GB for 4,512 clips, pipeline `ra1a-pooled-v1`, model `dinov3-vitl16`, MEASURED by ssh listing and `meta.json`) | none extra | thousands per second on CPU | no new decode, no frames on box 2 | trained on A/B labels; needs the cache for the same clip ids; the cache grid / pooling must be checked before relying on it (HYPOTHESIS: pooled grids keep enough global appearance for illumination and weather) |

### 3.4 Recommendation

Three tiers, each independently useful, banked incrementally:

1. **Tier 1, every 0.2 s frame, CPU**: image statistics (C) and the sun-band cross-check fields (D). Cost is the decode.
2. **Tier 2, every 0.2 s frame, small GPU job**: B (SigLIP2-class zero-shot, with its embedding stored) to produce `illum_img`, `weather_p`, `surface_p`, `view_quality`; calibrated on the audit sample (section 4).
3. **Tier 3, 1 Hz frame subset, VLM (A)**: recommended model **Qwen2.5-VL-3B-Instruct, 4-bit (about 4 GB VRAM)** with constrained-choice scoring; its answers are the reference label for weather, surface and visibility, and they supervise a probe (E or B head) that fills the other four frames per second. Temporal smoothing (median over 1 s) with the VLM label winning on disagreement.

Why not a VLM on all frames: 455 k frames at 2 to 4 frames/s is 32 to 63 h on one 4060; weather does not change at 5 Hz.

### 3.5 Time (all ESTIMATED; assumptions listed)

Assumptions: decode of 1080p H.264 plus downscale to 448 px about 250 frames/s per CPU process (the clip has 605 frames, all decoded); GPU stage rates from the table above; 4,512 clips.

| stage | work | dev box RTX 4060 (shared by lock, so assume one job at a time) | box 2 RTX 4060 (safe mode: 4 threads, BelowNormal, about 1.4x slower per its measured 5.0 vs 3.6 s/clip) | rented pod |
|---|---|---|---|---|
| decode + stats (Tier 1) | 4,512 x 605 = 2.73 M frames decoded | 2.73 M / 250 = 3.0 h one process; about 1.5 h with 2 | about 2.1 h with 4 threads; **needs the mp4 on D: (see 3.6)** | not needed |
| B at 5 Hz | 455 k frames | 455 k / 120 = 1.05 h | about 1.5 h | about 0.4 h (A40 class, HYPOTHESIS) |
| A at 1 Hz | 91 k frames | 91 k / 3 = 8.4 h (range 6 to 13 h) | 12 h, one job at a time (not recommended) | about 2 to 3 h |
| E probe fit and apply | 455 k rows | minutes on CPU | minutes | minutes |
| **total, train 4,512** | | **about 10 to 13 h GPU-lock time, split into 30 min jobs per 500 clips** | **only B, on the 600 + 141 clips that already have 256 px caches (about 25 GB there)** | **blocked: training is running, no load may be added now; after it ends A + B in about 3 h** |
| **whole corpus (306 k clips)** | 30.9 M frames | not feasible: 3.7 TB of front-wide video to pull (306 k x 12 MB, ESTIMATED) and about 8 to 9 days of B alone at 120 frames/s | no | pod only, after a PI decision on pulling about 3.7 TB |

Scope proposal: do the 4,512 + eval (about 4,660 clips) first; extend with the larger context held-out set (section 4.4, about 1,000 clips) because R-DATA says "each frame" of the dataset the programme uses, not of all 306 k clips (PI to confirm).

### 3.6 Where the source frames live, and what must be pulled

* **Dev box, in place (MEASURED)**: `C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/` holds **4,719 clips** (9,438 files = mp4 + timestamps; directory total about 57.5 GiB by `ls` block count 60,242,580 KiB). **All 4,512 train clips are present** (video local 4,512 of 4,512; 141 of 141). The other six camera directories hold 1 file each (only the front wide camera is local). C: has 36 GB free, E: 12 GB, F: 55 GB. Nothing to pull for the first pass; read in place, write only parquet (tens of MB).
* **Eval clips (MEASURED)**: all 147 eval label ids have a local front-wide mp4 (147 of 147). Nothing to pull for them.
* **Box 2 (read-only ssh probe, MEASURED)**: D: free 13.3 GB, C: free 11.6 GB. `D:\tanitad\cache_full\train` is 89.0 GB for 4,575 files (DINOv3 pooled features, not frames); `eval` 2.7 GB; `D:\tanitad\data\train600-256` 20.7 GB (256 px episodes of 600 clips), `eval141-256rb` 4.95 GB. No raw 1080p video. **Plan for box 2: do not stage frames there.** Run Tier 1 and Tier 2 on the dev box, and ship only the final parquet (about 50 MB) to box 2 if a probe-fit is wanted there. If a box 2 job is ever wanted, process in batches of 150 clips (about 1.8 GB mp4), delete each batch after its parquet is written, stay under 12 GB, keep `%RUNSAFE%` and one job at a time, check `BOX2_BUSY.txt` first (it is absent now, MEASURED).
* **Pod**: after training ends, pull the same mp4 from the dev box? 57 GB over the link is large. Preferred: pull from HF directly on the pod (gated token needed) only the chunks that hold the needed clips; 4,512 clips lie in 1,378 chunks (MEASURED), whole-chunk zips would be about 1.2 GB each (ESTIMATED), so per-clip range reads are the right route.
* **Never all at once on a low-disk machine**: Tier 1/2 stream clip by clip (decode to stdout from ffmpeg, no frame files written); only the 1 Hz VLM subset needs images, and those can be decoded on the fly too.

---------------------------------------------------------------------------------------------

## 4. Label validity plan

### 4.1 Human audit sample, and how the PI sees it

* **Size**: 600 frames. Basis: for an accuracy of about 0.8 per field, 95 % half-width of 5 points needs about 250 frames per field (Wilson); 600 gives 5 fields (weather, surface, visibility, illumination, view quality) roughly 4 points half-width on the common classes. Rare classes (snow, fog, heavy rain) cannot be estimated from a random sample: 65 snow-mentioning and 11 wet-road-mentioning clips of 4,572 (INHERITED) are far too few, so reserve **300 frames of the 600 for targeted rare-class candidates** (VLM or zero-shot score above threshold, plus all CoT snow and wet clips) and report rare-class precision only, with recall flagged unmeasured.
* The other 300 are stratified random: 25 countries (at least 6 each, Bulgaria, Hungary and Romania included) crossed with the four sun bands; at most 1 frame per clip, at most 1 clip per chunk.
* **How the PI sees it (suggested)**: one private HTML artifact, a contact sheet of 4 images per row with the model's labels and probabilities hidden until the PI taps a choice (blind first, then reveal), one-tap options per field, a skip button, and a result table (per field accuracy with Wilson interval and confusion matrix) updated in the page. 600 frames at about 6 s each is about 1 h of PI time; it can be split in sessions of 100. Use 1080p crops downscaled to 640 px only for this private page; raw frames are not published anywhere else.
* An agent-made second labelling pass (VLM prompt variants) is NOT a substitute for the PI audit; the VLM disagreeing with itself is reported as a separate diagnostic.

### 4.2 Illumination: image first, hour and sun band as a weak cross-check only

Rule (coordinator, 2026-11-08): `hour_of_day`, `daynight_clock` and the sun-band estimate are NOT per-frame illumination labels. The illumination label comes from the image. Clock and sun fields stay in the table as metadata and as a weak cross-check.

**D3's brightness definition** (read from `Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility/scripts/c5_frames.py` and `raw/c5b_daynight_vs_luma.json`, MEASURED by D3): luma = mean over pixels of the three RGB channels (0 to 255), rows above the bottom strip, averaged over the sampled frames of a clip; random 300 train and 139 eval clips; "dark" = clip mean luma below 30; the label compared is the clock label `daynight_clock`, not the census sun band. Results: night-labelled clips with mean luma below 30: 14.6 % (train), 12.5 % (eval); day-labelled below 30: 2.9 %; 60 to 67 % of night-labelled clips are brighter than 56 (the day 5th percentile); AUC 0.74 (train), 0.81 (eval). So "8 % actually dark against 46 % night" is the share of clips with mean luma below 30, against the clock share.

**Is `hour_of_day` local time? Yes, with high confidence (two independent probes).**
* Probe 1 (MEASURED, census `diurnal`): the 5-hour trough of clip counts starts at hour 0 or 1 for the United States, Germany, France, Italy, Sweden, i.e. the same local night in countries 6 to 9 h apart. A single UTC clock would move the US trough by 6 to 9 h. Peaks differ (US 10, Germany 19) because of collection schedules, not of the clock.
* Probe 2 (MEASURED, this audit, `scratchpad/p13.py`, `scratchpad/luma.json`): 110 non-US train clips, 3 frames each (3, 10, 17 s), downsampled gray, rows 0 to 25 of 36, median luma by season and hour: winter hour 16 to 18 median **13.9** (n=2, both under 30); summer hour 16 to 18 **96.3** (n=4); winter hour 19 to 21 **26.8** (n=9, 56 % under 30); summer hour 19 to 21 **75.8** (n=7, none under 30); winter 7 to 8 **97.0** (n=2). The brightness follows the local solar day by season, which a UTC or a fixed clock would not give. Small n; direction only.

**Do lit streets explain the gap? Largely yes, together with auto-exposure (HYPOTHESIS for the mechanism, MEASURED for the gap).** My sample reproduces D3: 9 of 110 clips (8 %) have mean luma below 30; for clips at hour 0 to 5 (n=12) the median mean luma is **57.8** and none is below 30, p95 luma 111 against 162 for hours 8 to 15 (n=45, median mean luma 96.2). So clock-night frames are mostly not black: the camera's exposure and street lighting lift the mean, and a mean-luma threshold of 30 selects only unlit rural or winter-evening scenes. Consequence: neither "46 % night" (clock) nor "8 % dark" (mean luma below 30) is a definition of illumination. The census sun-band "night" (38 %, ESTIMATED) sits between them and is not checked by D3.

**What to derive from the image** (per frame, cheap, Tier 1/2): p05, p50, p95 luma on the road-and-sky rows, sky-region luma (upper third), fraction of saturated bright blobs (headlights, street lamps, `lit_frac`), colour cast, and the zero-shot or VLM class. Calibrate the class to PI audit labels (4.1), not to the hour. Target classes: dark (unlit night), lit_night (street-lit night), twilight, day, bright_sun/glare. A "lit_night" class is needed precisely because lit streets are common in this corpus.

**Cross-check use only**: per clip, report the confusion of image class against sun band and against the clock label, and Spearman of `sky_luma` against estimated solar elevation by country group. Flag, do not correct: a clip is flagged when the image class is at least two classes away from the sun band. Disagreements go into the audit sample. No hour, month or sun-band value enters any per-frame label or any inference input (R-VIS).

### 4.3 Split rule that keeps one drive out of both sides

Measured reasons: 123 of the 147 eval clips share a chunk with at least one train4512 clip (98 of 122 eval chunks contain train clips, `scratchpad/p10.py`), and same-drive pairs are found inside chunks (section 1.5).

**Grouping key**: `group = chunk` (from `clip_index.parquet`), merged with detected continuation pairs (egomotion speed-fingerprint matches, section 1.5) by union-find, so a drive that crosses a chunk border (not tested) would only be caught if both chunks are loaded. Rule: **all clips of one group go to the same side.** A chunk is a conservative superset of a session (country-pure 90.5 %, but spans about 20 hours), so the rule over-excludes but cannot leak within a chunk.

### 4.4 Larger held-out set for context heads, and its effect on parity

* Problem (MEASURED, census): eval-141 has 24 countries, min 2 clips per country (Hungary 2, Romania 2, Bulgaria 2), and Sweden 16 at most. Per-country accuracy cannot be reported from that.
* **Proposal `CTX-HO`**: 40 clips per country x 25 countries = **1,000 clips**, drawn **outside the 4,512 train clips and outside any chunk that contains a train4512 clip**, chosen by seeded hash of clip id (hash order, same style as the canonical corpus), two clips per chunk at most. Feasibility (MEASURED, `scratchpad/p12.py`): clips in chunks with no train4512 clip: Luxembourg **8** (4 chunks), Hungary 47 (1 chunk), Bulgaria 66 (2 chunks), Belgium 71, Latvia 98, Poland 103; total 173,509 clips in 1,768 chunks. So 40 per country is feasible everywhere except Luxembourg (8): use 8 there and report it, rather than relax the chunk rule.
* Pull cost: about 1,000 x 12 MB = 12 GB (ESTIMATED) by per-clip range reads, with the dev box doing the pulls (C: 36 GB free); delete the mp4 after the labels are written (the programme delete rule, the labels are banked on HF).
* **Parity**: this does not re-select the canonical corpus (`physicalai-train-e438721ae894`, 2376 episodes, skip-hash `f09e44db`) and does not change the 4,512 / 40-episode val deployments: `CTX-HO` is a new, additional held-out set for the context heads only, disjoint by construction from the 4,512 and from every chunk they touch. Stating it explicitly: **no existing selection changes; one new disjoint set is added.** Any context head trained on labels from the 4,512 must be evaluated on `CTX-HO` (and on eval-141 for the usual four metric families, which remain unchanged). The v7.2 labels' `split` values (`train/val/test` in `clip_index.parquet`: train4512 holds train 2,651 / val 1,199 / test 662, MEASURED) are the provider split and are not used as the grouping rule.

---------------------------------------------------------------------------------------------

## 5. HF dataset: proposal (nothing created, nothing uploaded)

* **Repo (PRIVATE, PI account)**: `<PI-account>/tanitad-context-labels-v1` (type dataset). Not named after the corpus so that an unrelated NVIDIA licence statement is not implied; the card states "labels derived from nvidia/PhysicalAI-Autonomous-Vehicles (licence: other, research use), no raw frames".
* **Contents (parquet of labels only)**:
  * `frames/part-0000N.parquet`: the schema of 2.2, one row per clip and 0.2 s frame, sharded by country group, about 100 to 150 MB per shard max.
  * `clips/clips.parquet`: clip-level aggregates, group key, split (`train4512`, `ctx_ho`, `eval141`).
  * `audit/audit_v1.parquet`: the PI audit results (frame id, PI label, model label, disagreement).
  * `embeddings/siglip_*.npy` optional (float16, 1,152 dims x 455 k = about 1.0 GB); skip in v1 unless the probe needs them.
  * `MANIFEST.json` (sha256 per file, row counts, model and prompt sha), `README.md` (data card).
* **Raw frames are not needed on HF**: the frames are redistributable only under the NVIDIA licence (gated dataset, `license:other`); the labels join back by `clip_id` and frame index; the dev box already holds the mp4; the programme rule is that raw camera frames stay off the public repo, and the audit images are shown only in the PI's private page, not stored in the dataset. Exception that needs a PI yes: a 600-frame audit thumbnail pack (about 20 MB) for reproducibility, private repo only.
* **Size (ESTIMATED)**: 455 k rows x about 60 columns; with lists of probabilities and checkpoints, Parquet zstd about 100 to 200 bytes per column-row group, so **about 40 to 90 MB** for train 4,512 (HYPOTHESIS on compression); with `CTX-HO` (about 101 k rows) add about 10 to 20 MB. Embeddings, if included, add about 1 GB.
* **Upload settings (dev box)**: `HF_HUB_DISABLE_XET=1` (required; hf_xet hangs in CloseWait, MEMORY). Token read in place from `Keys.txt` into an environment variable (`grep -oE 'hf_[A-Za-z0-9]+'`), never printed or written. Use `huggingface_hub.HfApi.upload_folder` with `repo_type="dataset"`, `private=True`, `create_repo(..., private=True, exist_ok=True)`; one commit per shard group so a stall loses one shard; set `HF_HUB_DOWNLOAD_TIMEOUT` and retry; check free space (C: 36 GB, but write the shards to F: or E: as the ledger allows).
* **Verification**: (1) sha256 of every local shard written to `MANIFEST.json` before upload; (2) after upload, `HfApi.list_repo_tree` sizes equal local sizes; (3) a fresh `snapshot_download` into an empty directory (anonymous download must FAIL, authenticated must succeed, proving privacy); (4) recompute sha256 of the re-downloaded files and compare with `MANIFEST.json`; (5) row counts equal clips x 101 (minus grid rows documented as missing) and key columns have no nulls where `*_valid` is true. Only then delete the local staging copy (programme rule: delete only a banked copy, log path and size).

---------------------------------------------------------------------------------------------

## 6. Open points for the Master Mind / PI

1. The strategic band: the v7.2 labels use 8 to 30 s with one anchor at 8.0 s; REQ_HORIZONS says 6 to 30 s per key frame. Per-frame regeneration of `g_str` and checkpoints is a new label job (egomotion only, no GPU), 93.3 % of the 4,512 clips support the anchor at 7.8 s.
2. Visual supervision of the strategic band is limited to the video window (to 20.17 s). PI decision: accept geometry-only targets in the band, or pull successor clips for the 2.7 % of drives that have one.
3. Scope of R-DATA: the 4,512 + eval + `CTX-HO`, or the whole 306 k corpus (not feasible on local compute, section 3.5).
4. Hour semantics are settled as local clock (section 4.2, two probes); the sun band is still a coarse estimate and stays a cross-check. D3 (`D3_raw_plausibility/RESULT.md` D-1) independently found recording-level leakage between train and eval (3 image-confirmed eval clips), which agrees with the chain test of section 1.5.
6. The 6 to 8 s window has no label owner in v7.2 (section 2.2).
5. The 94 local egomotion zips whose size differs from the HF listing: a version difference was not investigated.

## 7. Artifacts and temp files

* Deliverable: `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/horizon_review/AUDIT_DATA.md`
* Scratch (session scratchpad `C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/`): `p1.py` to `p13.py`, `luma.json`, `ego_dur.json`, `join.json`, `alp_dur4512.json`, `chain_hits.json`, `chain_hits_alp.json`, `hf_ego.json`, `readme.md`, `p6.out`. Deleted at the end of the job (see hand-back).
