# TanitAD Media

**The single central folder for every video asset the programme has produced.**
Consolidated 2026-08-28 on the PI's instruction ("put the mp4s in a central folder in our project folder", Sayed 2026-08-28).

| | |
|---|---|
| Unique assets (deduped by sha256) | **132** |
| Total size | **597.4 MB** |
| Campaigns | **25** |
| Tracked in git | 45 |
| Untracked (gitignored `*.mp4`) | 87 |
| Sole-copy (existed in exactly one place before consolidation) | **0** |
| Had no local-backup copy (G:-mount only) | 17 |
| Unreadable / UNVERIFIED | 0 |
| On a pushed remote ref | 45 |
| **Has an off-device copy (local disk OR pushed)** | **132 / 132** |
| **Single-device risk (would die with the G: mount)** | **0** |

## How to trust this folder

`MEDIA_MANIFEST.json` is **git-tracked**; the `.mp4` binaries are **not** (`.gitignore` line 24 is `*.mp4`). That is deliberate: the manifest is the durable record, so even where a binary is absent we still know it existed, what it showed, and what its bytes hashed to.

```
python tools/media_verify.py            # re-hash every row: MISSING / MISMATCH / ORPHAN
python tools/media_verify.py --json     # machine-readable
```

`media_verify.py` probes **absence from both sides**: it walks the manifest to find MISSING/MISMATCH, and independently walks `Media/` on disk to find ORPHANs (a file nobody indexed). Iterating manifest rows alone structurally cannot reveal an unindexed file.

## Rules

1. **Every asset was COPIED, never moved.** All original locations still hold their copies, and the local backup at `C:\Users\Admin\tanitad-media-backup\` is untouched and remains the backup.
2. **Identity is sha256, never filename.** Four filenames in this corpus carry *different content* in different campaigns (the two AlpaSim open-loop scenes), and four assets appear under *two names*. Both are recorded per row.
3. **New videos go into `Media/<YYYY-MM-DD>-<campaign-slug>/`** and get a manifest row in the same turn. An asset outside the manifest is invisible to the programme.

## Is anything at risk?

**No.** All 132 assets have a copy that survives the G: Drive mount failing.

`tracked_in_git` is *not* the safety property on its own -- `.git` also lives on G:. The property that matters is `off_device_copy`, true when either:

* `on_local_backup` -- a copy exists in `C:\Users\Admin\tanitad-media-backup\` (115 assets), or
* `on_remote_ref` -- the blob is reachable from a remote ref, i.e. really pushed to GitHub (45 assets).

The 17 assets with **no** local-backup copy are all confirmed present on a remote ref (checked for all 17, not sampled), so they survive too. `on_remote_ref` is a point-in-time fact as of `generated_utc`; re-derive it rather than quoting it after a history rewrite.

## Campaigns

### `2026-07-19-flagship-v1-ood-overlays` - 5 files, 4.1 MB

Flagship v1 (speedjerk-30k) open-loop overlays on the same OOD comma.ai + Cosmos clips as REF-A: the matched comparison set.

| file | MB | shows | git |
|---|---|---|---|
| `flagship-30k_comma_curve_overlay.mp4` | 1.31 | flagship v1 (speedjerk-30k) | - |
| `flagship-30k_comma_highspeed_overlay.mp4` | 1.05 | flagship v1 (speedjerk-30k) | - |
| `flagship-30k_comma_straightcruise_overlay.mp4` | 1.50 | flagship v1 (speedjerk-30k) - straight cruise | - |
| `flagship-30k_cosmos_foggycurve_overlay.mp4` | 0.10 | flagship v1 (speedjerk-30k) - foggy curve | - |
| `flagship-30k_cosmos_sunnyhighway_overlay.mp4` | 0.11 | flagship v1 (speedjerk-30k) - sunny highway | - |

### `2026-07-19-flagship30k-gtpred-overlays` - 14 files, 14.8 MB

Ground-truth-vs-prediction overlays: 8 comma/Cosmos gtpred clips plus 6 flagship-30k PhysicalAI val-episode overlays.

| file | MB | shows | git |
|---|---|---|---|
| `comma_curve_gtpred.mp4` | 1.32 | GT-vs-prediction | - |
| `comma_highspeed-curve_gtpred.mp4` | 1.10 | GT-vs-prediction - high-speed curve | - |
| `comma_highspeed_gtpred.mp4` | 1.06 | GT-vs-prediction | - |
| `comma_straightcruise_gtpred.mp4` | 1.51 | GT-vs-prediction - straight cruise | - |
| `cosmos_dream-curve_gtpred.mp4` | 0.13 | GT-vs-prediction - synthetic curve | - |
| `cosmos_goldenhour-highspeed_gtpred.mp4` | 0.10 | GT-vs-prediction - golden-hour high speed | - |
| `cosmos_night-urban_gtpred.mp4` | 0.16 | GT-vs-prediction - night urban | - |
| `cosmos_sunny-highway_gtpred.mp4` | 0.12 | GT-vs-prediction | - |
| `flagship30k_overlay_braking_ep27.mp4` | 1.41 | flagship v1 (speedjerk-30k) - braking, episode 27 | - |
| `flagship30k_overlay_gentleturn_ep38.mp4` | 2.24 | flagship v1 (speedjerk-30k) - gentle turn, episode 38 | - |
| `flagship30k_overlay_highspeed-curve_ep28.mp4` | 1.41 | flagship v1 (speedjerk-30k) - high-speed curve, episode 28 | - |
| `flagship30k_overlay_highspeed-wrong_ep31.mp4` | 0.77 | flagship v1 (speedjerk-30k) - high-speed failure case, episode 31 | - |
| `flagship30k_overlay_sharpturn_ep03.mp4` | 1.90 | flagship v1 (speedjerk-30k) - sharp turn, episode 3 | - |
| `flagship30k_overlay_straightcruise_ep17.mp4` | 1.56 | flagship v1 (speedjerk-30k) - straight cruise, episode 17 | - |

### `2026-07-19-refa-ood-overlays` - 5 files, 4.2 MB

REF-A (frozen DINOv2 4-brain, dyn-in, 30k) open-loop overlays on OOD footage (comma.ai + Cosmos synthetic).

| file | MB | shows | git |
|---|---|---|---|
| `refa-dynin-30k_comma_curve_overlay.mp4` | 1.34 | REF-A dyn-in 30k | - |
| `refa-dynin-30k_comma_highspeed_overlay.mp4` | 1.10 | REF-A dyn-in 30k | - |
| `refa-dynin-30k_comma_straightcruise_overlay.mp4` | 1.54 | REF-A dyn-in 30k - straight cruise | - |
| `refa-dynin-30k_cosmos_foggycurve_overlay.mp4` | 0.10 | REF-A dyn-in 30k - foggy curve | - |
| `refa-dynin-30k_cosmos_sunnyhighway_overlay.mp4` | 0.11 | REF-A dyn-in 30k - sunny highway | - |

### `2026-07-20-label-qa-clips` - 19 files, 12.0 MB

Label-QA clips: per-episode trajectory-label renders (turn angle / radius / speed tags) used to validate the situation labels.

| file | MB | shows | git |
|---|---|---|---|
| `labels_ep_00006_turn-170deg-R7.mp4` | 0.63 | label-QA render - episode 6 | - |
| `labels_ep_00006_turn-170deg.mp4` | 0.10 | label-QA render - episode 6 | - |
| `labels_ep_00012_turn-right-104deg.mp4` | 0.82 | label-QA render - episode 12 | - |
| `labels_ep_00015_sharpturn-225deg-R13.mp4` | 0.85 | label-QA render - sharp turn, episode 15 | - |
| `labels_ep_00017_sharpturn-right-173deg.mp4` | 0.55 | label-QA render - sharp turn, episode 17 | - |
| `labels_ep_00018_turn-137deg.mp4` | 1.03 | label-QA render - episode 18 | - |
| `labels_ep_00029_turn-right-110deg-15ms.mp4` | 0.71 | label-QA render - episode 29 | - |
| `labels_ep_00031_turn-120deg-R6.mp4` | 0.74 | label-QA render - episode 31 | - |
| `labels_ep_00036_turn-147deg-R24.mp4` | 0.79 | label-QA render - episode 36 | - |
| `labels_ep_00038_straight-22ms.mp4` | 0.50 | label-QA render - episode 38 | - |
| `labels_ep_00045_straight-highspeed-35ms.mp4` | 0.61 | label-QA render - episode 45 | - |
| `labels_ep_00056_stopped-noarc.mp4` | 0.30 | label-QA render - episode 56 | - |
| `labels_ep_00064_straight-16ms.mp4` | 0.61 | label-QA render - episode 64 | - |
| `labels_ep_00068_straight-slow.mp4` | 0.43 | label-QA render - episode 68 | - |
| `labels_ep_00069_widedrift-479m-D3case.mp4` | 0.65 | label-QA render - episode 69 | - |
| `labels_ep_00071_widedrift-87deg.mp4` | 0.77 | label-QA render - episode 71 | - |
| `labels_ep_00073_turn-108deg.mp4` | 0.81 | label-QA render - episode 73 | - |
| `labels_ep_00076_straight-23ms.mp4` | 0.52 | label-QA render - episode 76 | - |
| `labels_ep_00078_turn-153deg.mp4` | 0.63 | label-QA render - episode 78 | - |

### `2026-07-20-refb-v2-30k-overlays` - 5 files, 5.7 MB

REF-B v2 at step 29999 on PhysicalAI val episodes (sharp turn, worst window, straight cruise, high-speed curve/straight).

| file | MB | shows | git |
|---|---|---|---|
| `refb-v2-30k_step29999_physicalai_ep03_sharpturn.mp4` | 1.52 | REF-B v2 step-29999 - sharp turn, episode 29999 | - |
| `refb-v2-30k_step29999_physicalai_ep11_failure-worstwindow.mp4` | 1.14 | REF-B v2 step-29999 - worst-ADE window, episode 29999 | - |
| `refb-v2-30k_step29999_physicalai_ep17_straightcruise.mp4` | 1.19 | REF-B v2 step-29999 - straight cruise, episode 29999 | - |
| `refb-v2-30k_step29999_physicalai_ep28_highspeed-curve.mp4` | 1.22 | REF-B v2 step-29999 - high-speed curve, episode 29999 | - |
| `refb-v2-30k_step29999_physicalai_ep31_highspeed-straight.mp4` | 0.60 | REF-B v2 step-29999 - high-speed straight, episode 29999 | - |

### `2026-07-20-refc-planfan-xl-overlays` - 9 files, 12.5 MB

REF-C plan-fan (step 29999) and REF-C XL live (step 28000) overlays on the same PhysicalAI val episodes.

| file | MB | shows | git |
|---|---|---|---|
| `refc-planfan_step29999_ep03_sharpturn.mp4` | 2.21 | REF-C plan-fan step-29999 - sharp turn, episode 29999 | - |
| `refc-planfan_step29999_ep11_failure-worstwindow.mp4` | 2.08 | REF-C plan-fan step-29999 - worst-ADE window, episode 29999 | - |
| `refc-planfan_step29999_ep28_highspeed-curve.mp4` | 1.73 | REF-C plan-fan step-29999 - high-speed curve, episode 29999 | - |
| `refc-planfan_step29999_ep31_highspeed-straight.mp4` | 0.76 | REF-C plan-fan step-29999 - high-speed straight, episode 29999 | - |
| `refc-xl-live_step28000_physicalai_ep03_sharpturn.mp4` | 1.52 | REF-C XL live step-28000 - sharp turn, episode 28000 | - |
| `refc-xl-live_step28000_physicalai_ep11_failure-worstwindow.mp4` | 1.14 | REF-C XL live step-28000 - worst-ADE window, episode 28000 | - |
| `refc-xl-live_step28000_physicalai_ep17_straightcruise.mp4` | 1.21 | REF-C XL live step-28000 - straight cruise, episode 28000 | - |
| `refc-xl-live_step28000_physicalai_ep28_highspeed-curve.mp4` | 1.23 | REF-C XL live step-28000 - high-speed curve, episode 28000 | - |
| `refc-xl-live_step28000_physicalai_ep31_highspeed-straight.mp4` | 0.61 | REF-C XL live step-28000 - high-speed straight, episode 28000 | - |

### `2026-07-21-refc-planfan-clips` - 18 files, 8.6 MB

Plan-fan visualisations; _base vs _xl are the two REF-C capacities on the identical window. Scenario-tagged (junction/braking/cruise/high-speed).

| file | MB | shows | git |
|---|---|---|---|
| `planfan_bad_selection_good_fan_ep09_f167_base.mp4` | 0.40 | REF-C plan-fan - good fan / bad selection, episode 9, frame 167, base capacity | - |
| `planfan_bad_selection_good_fan_ep09_f167_xl.mp4` | 0.44 | REF-C plan-fan - good fan / bad selection, episode 9, frame 167, XL capacity | - |
| `planfan_bad_selection_good_fan_ep19_f119_base.mp4` | 0.61 | REF-C plan-fan - good fan / bad selection, episode 19, frame 119, base capacity | - |
| `planfan_bad_selection_good_fan_ep19_f119_xl.mp4` | 0.62 | REF-C plan-fan - good fan / bad selection, episode 19, frame 119, XL capacity | - |
| `planfan_braking_longitudinal_ep27_f159_base.mp4` | 0.46 | REF-C plan-fan - braking, longitudinal braking, episode 27, frame 159, base capacity | - |
| `planfan_braking_longitudinal_ep27_f159_xl.mp4` | 0.48 | REF-C plan-fan - braking, longitudinal braking, episode 27, frame 159, XL capacity | - |
| `planfan_cruise_steady_ep05_f079_base.mp4` | 0.59 | REF-C plan-fan - steady cruise, episode 5, frame 79, base capacity | - |
| `planfan_cruise_steady_ep05_f079_xl.mp4` | 0.61 | REF-C plan-fan - steady cruise, episode 5, frame 79, XL capacity | - |
| `planfan_cruise_steady_ep15_f135_base.mp4` | 0.42 | REF-C plan-fan - steady cruise, episode 15, frame 135, base capacity | - |
| `planfan_cruise_steady_ep15_f135_xl.mp4` | 0.46 | REF-C plan-fan - steady cruise, episode 15, frame 135, XL capacity | - |
| `planfan_good_selection_ep13_f031_base.mp4` | 0.42 | REF-C plan-fan - good selection, episode 13, frame 31, base capacity | - |
| `planfan_good_selection_ep13_f031_xl.mp4` | 0.42 | REF-C plan-fan - good selection, episode 13, frame 31, XL capacity | - |
| `planfan_high_speed_ep31_f031_base.mp4` | 0.23 | REF-C plan-fan - high speed, episode 31, frame 31, base capacity | - |
| `planfan_high_speed_ep31_f031_xl.mp4` | 0.23 | REF-C plan-fan - high speed, episode 31, frame 31, XL capacity | - |
| `planfan_multimodal_junction_ep34_f095_base.mp4` | 0.50 | REF-C plan-fan - multimodal junction, episode 34, frame 95, base capacity | - |
| `planfan_multimodal_junction_ep34_f095_xl.mp4` | 0.52 | REF-C plan-fan - multimodal junction, episode 34, frame 95, XL capacity | - |
| `planfan_multimodal_junction_ep36_f119_base.mp4` | 0.58 | REF-C plan-fan - multimodal junction, episode 36, frame 119, base capacity | - |
| `planfan_multimodal_junction_ep36_f119_xl.mp4` | 0.62 | REF-C plan-fan - multimodal junction, episode 36, frame 119, XL capacity | - |

### `2026-07-22-alpasim-closedloop-archive` - 4 files, 3.0 MB

SUPERSEDED 10.4s AlpaSim CLOSED-loop clips from the now-terminated eval pod. Archive only.

| file | MB | shows | git |
|---|---|---|---|
| `flagship-v1_alpasim-closedloop_10s.mp4`<br>*alias:* `Flagship_v1_video.mp4` | 0.77 | flagship v1 - CLOSED loop | tracked |
| `refc-base_alpasim-closedloop_10s.mp4`<br>*alias:* `REFC_base_video.mp4` | 0.64 | REF-C base - CLOSED loop | tracked |
| `refc-small_alpasim-closedloop_10s.mp4`<br>*alias:* `REFC_small_video.mp4` | 0.82 | REF-C small - CLOSED loop | tracked |
| `refc-xl_alpasim-closedloop_10s.mp4`<br>*alias:* `REFC_xl_video.mp4` | 0.77 | REF-C XL - CLOSED loop | tracked |

### `2026-07-25-idm-youtube-validation` - 1 file, 0.6 MB

IDM reconstruction validation against YouTube footage, episode 20.

| file | MB | shows | git |
|---|---|---|---|
| `idm_recon_ep00020.mp4` | 0.61 | IDM reconstruction - episode 20 | tracked |

### `2026-08-02-v2corpus-vs-v1-overlays` - 12 files, 26.3 MB

v1 vs v2corpus overlays on PhysicalAI, split INTRAIN (seen in training) vs LEAKFREE (held out); the pairing is the point.

| file | MB | shows | git |
|---|---|---|---|
| `v1_physicalai_INTRAIN-ep01_overlay.mp4` | 2.53 | v1 - episode SEEN in training, episode 1 | - |
| `v1_physicalai_INTRAIN-ep11_overlay.mp4` | 1.96 | v1 - episode SEEN in training, episode 11 | - |
| `v1_physicalai_LEAKFREE-ep00_overlay.mp4` | 2.50 | v1 - held-out episode, episode 0 | - |
| `v1_physicalai_LEAKFREE-ep03_overlay.mp4` | 2.81 | v1 - held-out episode, episode 3 | - |
| `v1_physicalai_LEAKFREE-ep08_overlay.mp4` | 2.08 | v1 - held-out episode, episode 8 | - |
| `v1_physicalai_LEAKFREE-ep31_overlay.mp4` | 1.21 | v1 - held-out episode, episode 31 | - |
| `v2corpus_physicalai_INTRAIN-ep01_overlay.mp4` | 2.53 | v2corpus - episode SEEN in training, episode 1 | - |
| `v2corpus_physicalai_INTRAIN-ep11_overlay.mp4` | 1.98 | v2corpus - episode SEEN in training, episode 11 | - |
| `v2corpus_physicalai_LEAKFREE-ep00_overlay.mp4` | 2.50 | v2corpus - held-out episode, episode 0 | - |
| `v2corpus_physicalai_LEAKFREE-ep03_overlay.mp4` | 2.83 | v2corpus - held-out episode, episode 3 | - |
| `v2corpus_physicalai_LEAKFREE-ep08_overlay.mp4` | 2.14 | v2corpus - held-out episode, episode 8 | - |
| `v2corpus_physicalai_LEAKFREE-ep31_overlay.mp4` | 1.20 | v2corpus - held-out episode, episode 31 | - |

### `2026-08-03-alpasim-closedloop-thor` - 4 files, 41.5 MB

AlpaSim/NuRec CLOSED-loop on Thor: the model DRIVES, each frame rendered from where it actually went. 4 x 18.0s. WITHIN-SIM RELATIVE (3.21x OOD).

| file | MB | shows | git |
|---|---|---|---|
| `flagship-v1_empty_road.mp4` | 9.31 | flagship v1 - empty road | tracked |
| `flagship-v1_with_objects.mp4` | 9.64 | flagship v1 - with traffic | tracked |
| `refc-base_empty_road.mp4` | 11.17 | REF-C base - empty road | tracked |
| `refc-base_with_objects.mp4` | 11.36 | REF-C base - with traffic | tracked |

### `2026-08-03-alpasim-gsplat-cutin` - 4 files, 31.3 MB

AlpaSim gsplat scripted scenarios: cut-in and lead-vehicle-at-8m, flagship-v1 vs refc-base.

| file | MB | shows | git |
|---|---|---|---|
| `flagship-v1_cutin.mp4` | 8.14 | flagship v1 - cut-in scenario | tracked |
| `flagship-v1_lead8.mp4` | 7.13 | flagship v1 - lead vehicle at 8 m | tracked |
| `refc-base_cutin.mp4` | 8.09 | REF-C base - cut-in scenario | tracked |
| `refc-base_lead8.mp4` | 7.91 | REF-C base - lead vehicle at 8 m | tracked |

### `2026-08-03-alpasim-gsplat-scene2-realclose` - 2 files, 16.3 MB

AlpaSim gsplat scene2 real-close-objects scenario, flagship-v1 vs refc-base.

| file | MB | shows | git |
|---|---|---|---|
| `flagship-v1_scene2_objects.mp4` | 8.45 | flagship v1 - scene2 with close objects | tracked |
| `refc-base_scene2_objects.mp4` | 7.83 | REF-C base - scene2 with close objects | tracked |

### `2026-08-03-alpasim-openloop-thor` - 4 files, 45.2 MB

AlpaSim/NuRec OPEN-loop on Thor, scene 00040136: ego follows the LOGGED trajectory; the model predicts but never drives. WITHIN-SIM RELATIVE (3.21x OOD).

| file | MB | shows | git |
|---|---|---|---|
| `flagship-v1_openloop_empty_road.mp4` | 11.28 | flagship v1 - empty road, OPEN loop | tracked |
| `flagship-v1_openloop_with_objects.mp4` | 11.45 | flagship v1 - with traffic, OPEN loop | tracked |
| `refc-base_openloop_empty_road.mp4` | 11.13 | REF-C base - empty road, OPEN loop | tracked |
| `refc-base_openloop_with_objects.mp4` | 11.29 | REF-C base - with traffic, OPEN loop | tracked |

### `2026-08-03-alpasim-openloop-thor-junction-7c72937c` - 4 files, 42.5 MB

AlpaSim/NuRec OPEN-loop on Thor, junction scene 7c72937c. NOTE: filenames COLLIDE with the sibling scene campaign but the content differs, verified distinct by sha256.

| file | MB | shows | git |
|---|---|---|---|
| `flagship-v1_openloop_empty_road.mp4` | 10.89 | flagship v1 - empty road, OPEN loop | tracked |
| `flagship-v1_openloop_with_objects.mp4` | 10.34 | flagship v1 - with traffic, OPEN loop | tracked |
| `refc-base_openloop_empty_road.mp4` | 10.89 | REF-C base - empty road, OPEN loop | tracked |
| `refc-base_openloop_with_objects.mp4` | 10.41 | REF-C base - with traffic, OPEN loop | tracked |

### `2026-08-05-v1arch-oodval-openloop` - 3 files, 111.8 MB

Long open-loop reels on PhysicalAI's OWN official eval split (290 clips, zero training overlap). WORLD-MODEL FIDELITY, not driving: the rollout decodes the expert's true future actions.

| file | MB | shows | git |
|---|---|---|---|
| `v1arch_oodval_openloop_best12.mp4` | 23.71 | v1-arch - best 12 by ADE (CHERRY-PICKED), OPEN loop | tracked |
| `v1arch_oodval_openloop_representative.mp4` | 68.40 | v1-arch - spread selection, 30 eps, OPEN loop | tracked |
| `v1arch_oodval_openloop_worst12.mp4` | 19.70 | v1-arch - worst 12 by ADE, OPEN loop | tracked |

### `2026-08-06-alpamayo2-vs-flagship` - 1 file, 2.0 MB

Alpamayo-2 vs TanitAD flagship side-by-side comparison reel.

| file | MB | shows | git |
|---|---|---|---|
| `alpamayo_vs_flagship.mp4` | 2.04 | Alpamayo-2 vs flagship | tracked |

### `2026-08-06-v16-vs-v1arch` - 2 files, 78.3 MB

v1.6 (unfrozen trunk) vs v1-arch: a continuous reel plus the paired comparison reel.

| file | MB | shows | git |
|---|---|---|---|
| `v16_continuous.mp4` | 69.49 | v1.6 - continuous reel | tracked |
| `v16_vs_v1arch_reel.mp4` | 8.82 | v1.6 | tracked |

### `2026-08-06-v17` - 1 file, 69.3 MB

v1.7 continuous open-loop reel.

| file | MB | shows | git |
|---|---|---|---|
| `v17_continuous.mp4` | 69.26 | v1.7 - continuous reel | tracked |

### `2026-08-09-v5f-fan` - 1 file, 19.4 MB

v5f trajectory-fan compact visualisation.

| file | MB | shows | git |
|---|---|---|---|
| `v5f_fan_compact.mp4` | 19.43 | v5f | tracked |

### `2026-08-10-v5f-bev` - 1 file, 19.8 MB

v5f metric-BEV compact visualisation.

| file | MB | shows | git |
|---|---|---|---|
| `v5f_bev_compact.mp4` | 19.83 | v5f | tracked |

### `2026-08-10-v5f-planfan` - 1 file, 20.1 MB

v5f plan-fan compact visualisation.

| file | MB | shows | git |
|---|---|---|---|
| `v5f_planfan_compact.mp4` | 20.06 | v5f | tracked |

### `2026-08-12-ph0-vlm-overlay` - 2 files, 2.4 MB

Phase-0 rich overlay reels: the VLM-annotation overlay and the full pipeline reel.

| file | MB | shows | git |
|---|---|---|---|
| `ph0_full_pipeline_reel.mp4` | 1.55 | Phase-0 pipeline | tracked |
| `ph0_vlm_overlay_reel.mp4` | 0.90 | Phase-0 pipeline | tracked |

### `2026-08-16-sam3-dtype-fix` - 8 files, 4.8 MB

SAM3 segmentation rich overlays after the dtype fix; one clip per PhysicalAI clip hash.

| file | MB | shows | git |
|---|---|---|---|
| `0089a096_rich.mp4` | 0.72 | SAM3 rich overlay | tracked |
| `093bfa29_rich.mp4` | 0.52 | SAM3 rich overlay | tracked |
| `15a65b76_rich.mp4` | 0.51 | SAM3 rich overlay | tracked |
| `38aac500_rich.mp4` | 0.31 | SAM3 rich overlay | tracked |
| `42745b48_rich.mp4` | 0.49 | SAM3 rich overlay | tracked |
| `814c2f74_rich.mp4` | 0.79 | SAM3 rich overlay | tracked |
| `8f5df500_rich.mp4` | 0.78 | SAM3 rich overlay | tracked |
| `bb41e3b8_rich.mp4` | 0.64 | SAM3 rich overlay | tracked |

### `2026-08-17-sam3-extraction-v2` - 2 files, 1.0 MB

SAM3 extraction v2 rich overlays.

| file | MB | shows | git |
|---|---|---|---|
| `v2_aa291a17.mp4` | 0.48 | SAM3 extraction v2 | tracked |
| `v2_e084c7c3.mp4` | 0.48 | SAM3 extraction v2 | tracked |
