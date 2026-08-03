# TanitAD — evaluation overlay videos

Pulled from `tanitad-pod3:/workspace/idmretrain/taniteval/results/videos` on **2026-08-01**, the only pod holding rendered overlays. (pod2's 600 `.mp4` are RAW PhysicalAI `camera_front_wide` clips, not ours.)

**Visualisation standard** (Sayed's standing preference): camera projection + metric BEV inset together, plus a text overlay of the decoded tactical manoeuvre and strategic route/goal, plus ADE.

⚠️ These are the videos that existed BEFORE the v2corpus run. Fresh long overlays for v2corpus vs v1 on the canonical val are rendering separately.

---

## ⭐ AlpaSim / NuRec sim videos — OPEN loop and CLOSED loop are SEPARATE folders

⛔ **They answer different questions and must never be confused.** The mode is burned into
every frame as well as into the folder name.

| folder | mode | what it means | files |
|---|---|---|---|
| [`alpasim-openloop-thor-2026-08-03/`](alpasim-openloop-thor-2026-08-03/README.md) | **OPEN** | the ego follows the **LOGGED** trajectory; the model predicts and is scored, but **never drives**. Isolates perception + prediction. | 8 × 19.0 s (2 scenes × 2 arms × 2 traffic conditions) |
| [`alpasim-closedloop-thor-2026-08-03/`](alpasim-closedloop-thor-2026-08-03/README.md) | **CLOSED** | the model **drives**; each frame is rendered from where it actually went. Perception error and control drift are confounded here — that is the point of also having the open-loop set. | 4 × 18.0 s |
| `alpasim-closedloop-archive-2026-07-22/` | CLOSED | superseded 10.4 s clips from the terminated eval pod. | archive |

Both sets use the identical 2026-08-03 render (4 layers + scale-cull 0.95 + gated sky 0.3,
grad-NCC 0.3424). ⛔ All AlpaSim numbers are **WITHIN-SIM RELATIVE**: REF-C's open-loop ADE
is 1.5157 on these reconstructions vs 0.4728 on real footage — **3.21× OOD**. Orderings
survive; absolute rates do not.

---

**56 videos, 49.8 MB total** *(the pod3 set below; the AlpaSim folders above are counted separately).*


## REF-A — 5 clips, 4.2 MB

| file | MB |
|---|---|
| `refa-dynin-30k_comma_curve_overlay.mp4` | 1.3 |
| `refa-dynin-30k_comma_highspeed_overlay.mp4` | 1.1 |
| `refa-dynin-30k_comma_straightcruise_overlay.mp4` | 1.5 |
| `refa-dynin-30k_cosmos_foggycurve_overlay.mp4` | 0.1 |
| `refa-dynin-30k_cosmos_sunnyhighway_overlay.mp4` | 0.1 |

## REF-B — 5 clips, 5.7 MB

| file | MB |
|---|---|
| `refb-v2-30k_step29999_physicalai_ep03_sharpturn.mp4` | 1.5 |
| `refb-v2-30k_step29999_physicalai_ep11_failure-worstwindow.mp4` | 1.1 |
| `refb-v2-30k_step29999_physicalai_ep17_straightcruise.mp4` | 1.2 |
| `refb-v2-30k_step29999_physicalai_ep28_highspeed-curve.mp4` | 1.2 |
| `refb-v2-30k_step29999_physicalai_ep31_highspeed-straight.mp4` | 0.6 |

## REF-C — 9 clips, 12.5 MB

| file | MB |
|---|---|
| `refc-planfan_step29999_ep03_sharpturn.mp4` | 2.2 |
| `refc-planfan_step29999_ep11_failure-worstwindow.mp4` | 2.1 |
| `refc-planfan_step29999_ep28_highspeed-curve.mp4` | 1.7 |
| `refc-planfan_step29999_ep31_highspeed-straight.mp4` | 0.8 |
| `refc-xl-live_step28000_physicalai_ep03_sharpturn.mp4` | 1.5 |
| `refc-xl-live_step28000_physicalai_ep11_failure-worstwindow.mp4` | 1.1 |
| `refc-xl-live_step28000_physicalai_ep17_straightcruise.mp4` | 1.2 |
| `refc-xl-live_step28000_physicalai_ep28_highspeed-curve.mp4` | 1.2 |
| `refc-xl-live_step28000_physicalai_ep31_highspeed-straight.mp4` | 0.6 |

## flagship-v1 — 5 clips, 4.1 MB

| file | MB |
|---|---|
| `flagship-30k_comma_curve_overlay.mp4` | 1.3 |
| `flagship-30k_comma_highspeed_overlay.mp4` | 1.0 |
| `flagship-30k_comma_straightcruise_overlay.mp4` | 1.5 |
| `flagship-30k_cosmos_foggycurve_overlay.mp4` | 0.1 |
| `flagship-30k_cosmos_sunnyhighway_overlay.mp4` | 0.1 |

## other — 14 clips, 14.8 MB

| file | MB |
|---|---|
| `comma_curve_gtpred.mp4` | 1.3 |
| `comma_highspeed-curve_gtpred.mp4` | 1.1 |
| `comma_highspeed_gtpred.mp4` | 1.1 |
| `comma_straightcruise_gtpred.mp4` | 1.5 |
| `cosmos_dream-curve_gtpred.mp4` | 0.1 |
| `cosmos_goldenhour-highspeed_gtpred.mp4` | 0.1 |
| `cosmos_night-urban_gtpred.mp4` | 0.2 |
| `cosmos_sunny-highway_gtpred.mp4` | 0.1 |
| `flagship30k_overlay_braking_ep27.mp4` | 1.4 |
| `flagship30k_overlay_gentleturn_ep38.mp4` | 2.2 |
| `flagship30k_overlay_highspeed-curve_ep28.mp4` | 1.4 |
| `flagship30k_overlay_highspeed-wrong_ep31.mp4` | 0.8 |
| `flagship30k_overlay_sharpturn_ep03.mp4` | 1.9 |
| `flagship30k_overlay_straightcruise_ep17.mp4` | 1.6 |

## planfan-clips — 18 clips, 8.6 MB

| file | MB |
|---|---|
| `planfan_bad_selection_good_fan_ep09_f167_base.mp4` | 0.4 |
| `planfan_bad_selection_good_fan_ep09_f167_xl.mp4` | 0.4 |
| `planfan_bad_selection_good_fan_ep19_f119_base.mp4` | 0.6 |
| `planfan_bad_selection_good_fan_ep19_f119_xl.mp4` | 0.6 |
| `planfan_braking_longitudinal_ep27_f159_base.mp4` | 0.5 |
| `planfan_braking_longitudinal_ep27_f159_xl.mp4` | 0.5 |
| `planfan_cruise_steady_ep05_f079_base.mp4` | 0.6 |
| `planfan_cruise_steady_ep05_f079_xl.mp4` | 0.6 |
| `planfan_cruise_steady_ep15_f135_base.mp4` | 0.4 |
| `planfan_cruise_steady_ep15_f135_xl.mp4` | 0.5 |
| `planfan_good_selection_ep13_f031_base.mp4` | 0.4 |
| `planfan_good_selection_ep13_f031_xl.mp4` | 0.4 |
| `planfan_high_speed_ep31_f031_base.mp4` | 0.2 |
| `planfan_high_speed_ep31_f031_xl.mp4` | 0.2 |
| `planfan_multimodal_junction_ep34_f095_base.mp4` | 0.5 |
| `planfan_multimodal_junction_ep34_f095_xl.mp4` | 0.5 |
| `planfan_multimodal_junction_ep36_f119_base.mp4` | 0.6 |
| `planfan_multimodal_junction_ep36_f119_xl.mp4` | 0.6 |

## Naming

`<model>_<step>_<corpus>_<episode>_<tag>.mp4` — e.g. `refc-xl-live_step28000_physicalai_ep03_sharpturn.mp4` is REF-C XL at step 28,000 on PhysicalAI val episode 3, a sharp-turn window.


`planfan-*` clips are plan-fan visualisations (`_xl` vs `_base` = the two REF-C capacities on the same window). `calibration-probes` are camera-calibration fits, not performance evidence.

