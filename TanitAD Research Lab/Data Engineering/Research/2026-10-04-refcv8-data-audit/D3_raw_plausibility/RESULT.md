# D3 — raw-data plausibility of the refcv7 training corpus (RESULT)

*Stream D3 of the refcv8 data audit, 2026-10-04. The stream's agent could not write report files (harness block), so
this RESULT.md is the Master Mind's transcription of its final report, verdicts and numbers unchanged. Every row is
MEASURED from the artifact named under `raw/` (scripts under `scripts/`, logs under `raw/logs/`, md5s in
`../LANDING_READY_D3.txt`). CPU only; Thor jobs ended 12:40:17 Berlin; ids are sha12.*

**Verdict:** the raw data (poses, frames, boxes, maps) is physically sound. Two small DEFECTS (split leakage across
recordings; the ego as an agent box in 18 clips) and a set of MINOR items, none of which explains the refcv7 planner
failures. The data problems that matter are in the LABELS and how they are applied (streams D1, D2, D4).

## DEFECT

| id | finding | n / evidence | consequence | fix |
|---|---|---|---|---|
| **D-1 split not recording-disjoint** | No clip id is shared (4,369 train / 139 eval), but clips are cut from the same ~140 s recordings. Matched by each clip's 20 s speed + yaw-rate profile against every clip's full egomotion log (4,126 of 4,508 clips testable; 120 of 139 eval). Image-confirmed tier (score < 0.01): **3 eval clips** share a recording with a train clip; looser tier (< 0.1): **5 eval / 6 train**. Eval `547d1d68c1bd` starts 0.1 s after train `f0a6ab615da8` ends (image continuity NCC 0.998). No eval clip shares wall-clock time or frames with train. Lower bound (logs reach ~120 s; 19 eval clips untestable). Train duplication: **135 recording groups (2–5 clips) cover 296 train clips (6.8 %)**; two train pairs are the same video under two ids (`26f668b218f6`/`a8f0a8b2476a`, `1aca93cd9155`/`0d6f01ac5244`). A naive score < 1 rule would have claimed 21 % — chance matching, refuted by the image channel. | `raw/c1_*` | ~2–4 % of eval clips mildly optimistic (route/scene seen); train reweighted < 7 % | drop the 6 train partner clips (or the 5 eval clips) and one clip of each duplicate pair (`raw/c1_pairs_lt6.npz`, `raw/c1_series_summary.npz`) |
| **D-2 ego as an agent box** | 18 train clips (0 eval), 693 frames; 84 % of those boxes sit within 0.5 m of (1.4, 0) with ego dimensions (~4.6 × 2.0 m); up to 121 frames per clip | `raw/c3b_agents_detail.json` | teaches "a car at the ego position"; a permanent collision for an oracle no-collision scorer | mask those frames |

## MINOR

* **Eval map GT**: 2 eval clips have no map GT (`081b986f8888`, `2aa810802777`, both NO_V2; 1.4 % of eval); all train clips have it. Confirm the eval masks them.
* **Eval nav mix skewed** (permutation p = 0.0062): TURN_R 27.3 % (38 clips), TURN_L 9.4 % (13) vs train 19.2 % / 17.8 %. Left-turn route following is evaluated on **13 clips**.
* Eval is stratified on road class and day/night; country mix fine (p = 0.52); Slovenia and Slovakia absent from eval; 1–13 eval clips per country — too few for per-country numbers.
* The 16-bit legacy `episode_id` collides 123× inside train and 9× train↔eval — harmless only while the join runs with `allow_legacy_ids False`.
* **Day/night is a clock label, not visual**: 60 % of "night" train clips are brighter than the 5th percentile of "day"; luma AUC 0.74; only ~8 % of clips are actually dark (mean luma < 30) vs 46 % labelled night; one "day" clip (`76fe124aa725`) has mean luma 12.6.
* **Pose vs image time**: poses sit on the uniform grid, the image is the next camera frame 0–34 ms later (mean 16.5 ms) → a longitudinal offset of 0.19 m mean, up to ~0.9 m at p95 clip speed.
* Clock sidecar missing for 22 train + 3 eval clips (nominal dt 0.1007).
* Poses vs actions disagree on curvature by up to ±10 % per clip against the fixed 2.9 m wheelbase (global slope 1.029, corr 0.997).
* 22.8 % of windows lack the tail of the 6 s future (masked by horizon, not zero-filled): valid 77 % at 6 s, 95 % at 3 s.
* 15 clips genuinely reverse (14 train, 1 eval); 31 rows in 4 train clips brake harder than 8 m/s² (−8.3 to −11.2, consistent with the provider's accelerations); one clip (`f33569756edd`) has a 2-row speed glitch.
* **Boxes**: near-duplicate same-class boxes in 4.89 % of train frames but only 0.34 % of boxes, no repeated track ids ⇒ **the GT does NOT carry ~2.1 boxes per object; the refcv7 reel's duplicates are prediction-side** (supports inference NMS). Track jumps (> 5 m per 0.1 s for vehicles) in 0.028 % of track steps (6,410 frames, 887 clips; 91 % single-track id switches, up to 276 m) — rate targets are unclipped differences, bounded by the L1 rate loss. Box base height off the ground plane by > 1.5 m: 0.9 % < 30 m, 11.7 % at 30–100 m, 25.9 % > 100 m — **treat z as near-field only**. Agent records start ≥ 9 rows late in 235 clips, end ≥ 3 s early in 174; 540 clips have empty frames (18,347) — this is the 3.64 % of windows without agent labels. 59.8 % of boxes flagged occluded, 50.1 % behind the ego, 3.4 % beyond 150 m.
* **Map GT**: 3 train clips with a low ego-path-on-map fraction (`d2a8b3c54f04` 0.547, `cbae3039d2bc` 0.682, `3e6bb6e60ef6` 0.705; 9 below 0.9). 585 train clips (13.4 %) have no lane-line cell in any frame — follows country (Estonia 51 %, Greece 41 %, Latvia 33 %, US 4.5 %) and road class (highway 2.4 %, urban 15 %), not day/night (13.7 vs 13.1 %) ⇒ real missing markings, not labeller dropout. Lane-line cells are 0.92 % and non-drivable-edge 0.36 % of seen cells (class weights matter).
* Rig B's black bottom strip is 27–37 rows (the training equalizer uses 43); its height varies within 14 train + 5 eval clips.

## OK

| check | result |
|---|---|
| ids | 0 shared clip ids, episode_uids or 8-char prefixes; both caches carry exactly the trainer's windows: 746,946 train, 23,772 eval |
| strata | country p = 0.52, strata cell p = 0.91, day/night and road class within 0.2 pp (control: a random 139-of-train split gives median p = 0.49) |
| poses (4,369 + 139 clips, 897k rows) | 0 non-finite; 0 fully stationary clips; max speed 40.1 m/s; 0 rows with \|yaw rate\| > 1.2 rad/s at v > 5 m/s or \|a_lat\| > 8 m/s²; heading-vs-velocity rule fires on 29 rows in one clip; pose-implied dt vs sidecar median 1.000004, 0 clips off > 2 % |
| position jumps | the brief's literal "> 3 m per step" rule is a false alarm (any speed > 29.8 m/s exceeds it; 3.37 % of rows); the speed-consistent rule flags 9 rows in 4 clips, all at clip edges |
| agent boxes | 27.6 M train boxes, 100 % carry z and h, 0 non-finite; car l 3.5–7.0, w 1.5–2.6, h 1.4–3.0 m (0.8 % "out of range" are tall vans); persons h 1.3–1.9 m |
| SAM3 map GT on the ego path (2.98 M points) | 99.92 % of the ego's next-6-s path is road-like, 96.9 % code 1 "drivable"; 0.013 % sidewalk/edge; 0 clips empty |
| frames (60,282 train + 27,942 eval decoded) | 0 black, 0 over-exposed, 0 frozen, 0 near-frozen in 74 k moving pairs |
| rigs | rig B 58.05 % train / 53.24 % eval (χ² 1.28, n.s.); eval camera height and pitch inside the train range; 46 of 139 eval clips share an exact calibration with a train clip |

**Balance (window level, train 746,946 vs eval 23,772):** turns (\|Δyaw\| ≥ 30° over 6 s) 15.31 % vs 14.18 %; stopped
now 6.00 % vs 5.30 %; stop within 6 s 11.02 % vs 11.11 %; > 25 m/s 6.30 % vs 7.08 %; night (clock) 46.13 % vs 46.03 %.
Across the 25 train countries: turns 6.6–24.6 %, high speed 0.8–21.8 %, night 19–57 %.

## Controls (each read its known value)
* Poses: a synthetic circle read yaw rate 0.2000 rad/s and a_lat 2.000 m/s² with 0 flags; an injected 5 m jump, a reversed circle and 70 m/s were each caught. The first sustained-jerk rule failed its own one-sample-spike control and was fixed before any data was scored.
* Recording finder: self-match 4,126/4,126 at zero offset; 40/40 shifted-and-noisy windows recovered at 37.3 s.
* Path shape: median residual 0.018 m for score < 0.01 pairs vs 6.3 m for the same pairs shifted 5 s and 45 m for random pairs.
* Image channel: 5/5 time-overlapping pairs with score ≤ 0.0092 matched at the predicted frame row (NCC 0.98–1.00); 0/8 with score ≥ 0.047; all 13 adjacent pairs NCC 0.81–1.00 vs a null maximum 0.42.
* Box registration: median track speed in turning segments 0.48 m/s; with the yaw sign flipped 26.5 m/s; with ego translation frozen 5.9 m/s (certifies the frame convention only; a constant box-vs-pose lag is invisible to it and was not tested).
* SAM3 ego-path: shifting the path 6 m sideways drops road-like 99.9 % → 44 %.
* Frames: all-zero / all-white / gradient images read black / over-exposed / mean 127.0.

## Not done
* The leak's effect on eval scores was not quantified (the refcv7 per-window dumps are not on the dev box).
* The SAM3 ego-path test would be partly circular if the labeller seeds "drivable" from the ego path; the 6 m shift control shows the region is not a thin painted corridor.
* Frames: a random 300 of 4,369 train clips (all eval), not the full corpus.
* Recording overlap at gaps > 20 s has no image check (motion profile + path shape only; ~6 expected false matches below score 0.03 of 4,126 windows).

Not for git (single copies): the dev-box scratchpad copy of the Thor `_v2manifest.pt` (24 MB, regenerable) and
`physicalai_front_wide_intrinsics.csv` (gated PhysicalAI metadata — do not commit). Thor: `/home/nvidia/refcv8_audit/D3/`
(6.7 MB, all copied into `raw/`, disposable).
