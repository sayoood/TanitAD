# T2 u400 baseline: tier 1 report

**Stamp:** T2 (re-perception closed-loop sim; MM ruling 2026-11-07), T2-class closed-loop sim, not real driving. REFe SFT-4 u400, bf16, 10 scenes, one seed.

**Identity:** checkpoint `8b4dba4818b5cc96...`, scorer `69d13d712ca6e6ed...`, SUITE_TOKENS `f0c49923e147f2bc...`, seed 20261107. Post-hoc script `58afbae0c5fb33d7...`. Amendments 8 and 9 are applied below as reporting only. The full-window verdict is primary.

## Summary

- **Speed is the main failure.** a1 passes on 0 of the 9 scenes where it applies (8 fail, 1 informational) and a2 fails on all 9. On the 3 free-road cruise scenes the plan already slows at the first call on real frames (v4/v0 0.827, 0.809, 0.807), so that is the model and not the render.
- **One at-fault contact in 10 scenes** (f1 FAIL on the junction scene, under 1 m/s into a stopped car). Three scenes have logged cars running into the slow ego from behind. EPDMS 0.0 on two of them (scenes 4 and 7) is an instrument defect, section 2.
- **Lane keeping against the human path is weak.** e1 fails on 9 of 10, and e2 on 6 of 10. e1 already fails before the first redraw on 5 scenes, so render redraws do not explain those.
- **Render:** 23 whole-scene redraws over 10 scenes. Front similarity 1 s apart has a per-scene median of 0.9632 to 0.9794 with no redraw between the frames, and 0.7744 to 0.9304 across a redraw. There is no real reference on thor3, so this is the weaker within-render measure.
- **Limits:** one seed, 10 scenes, closed-loop sim. TACTICAL is N/A (A-14), STRATEGIC is paused.

## 1. Scenes and full-window verdicts

| # | scene | category | EPDMS saved | EPDMS descriptive* | NC | DAC | TTC | EP | C | mean v m/s | a1 | a2 | b1 | b2 | c1 | c2 | c3 | c4 | d1 | d2 | e1 | e2 | f1 | f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 00_cruise_86cad912 | cruise | 0.754 | 0.754 | 1 | 1 | 1 | 0.409 | 1 | 4.7 | F | F | - | - | - | - | - | - | P | P | F | F | P | P |
| 2 | 01_cruise_59714859 | cruise | 0.660 | 0.660 | 1 | 1 | 1 | 0.585 | 0 | 6.5 | F | F | - | - | - | - | - | - | P | P | F | P | P | P |
| 3 | 02_cruise_13fa82d6 | cruise | 0.658 | 0.658 | 1 | 1 | 1 | 0.580 | 0 | 6.5 | F | F | - | - | - | - | - | - | P | P | F | F | P | P |
| 4 | 06_steady_lead_024d89a3 | steady_lead | 0.000 | 0.440 | 0 | 1 | 0 | 0.656 | 1 | 5.2 | F | F | I | F | P | - | - | - | P | P | F | P | P | F |
| 5 | 07_steady_lead_adfdcd3c | steady_lead | 0.603 | 0.603 | 1 | 1 | 1 | 0.446 | 0 | 2.3 | F | F | - | P | - | - | - | - | P | F | F | F | P | P |
| 6 | 08_steady_lead_87751a07 | steady_lead | 0.876 | 0.876 | 1 | 1 | 1 | 0.702 | 1 | 8.1 | F | F | F | F | - | P | - | - | P | P | F | F | P | P |
| 7 | 12_lead_braking_390d9e3b | lead_braking | 0.000 | 0.315 | 0 | 1 | 0 | 0.757 | 0 | 5.6 | F | F | I | F | - | - | - | - | P | P | P | P | P | P |
| 8 | 14_stop_go_95472243 | stop_go | 0.931 | 0.931 | 1 | 1 | 1 | 0.833 | 1 | 3.9 | - | - | I | P | - | P | - | - | P | P | F | F | P | P |
| 9 | 16_merge_cutin_afd632f0 | merge_cutin | 0.803 | 0.803 | 1 | 1 | 1 | 0.528 | 1 | 5.6 | F | F | I | I | - | - | - | - | P | P | F | P | P | P |
| 10 | 20_J-UL_52f7393a | junction | 0.000 | 0.000 | 0 | 1 | 0 | 0.557 | 0 | 2.1 | I | F | - | F | - | - | F | - | P | P | F | F | F | P |

P pass, F fail, I informational (the human fails too), - not applicable. Mean EPDMS over the 10 scenes: saved **0.529**, descriptive 0.604.

\* **Descriptive, post hoc, not used for any gate** (MM ruling, 2026-11-09). n = 10. Only the NC term is replaced, by the scorer's at-fault events; TTC stays as saved. The value changes on 2 scenes: 06_steady_lead_024d89a3, 12_lead_braking_390d9e3b. Every gate and bar reads the saved EPDMS.

TTC is 0 on 3 scenes, each from a car in its contact list: 06_steady_lead_024d89a3 (car_44), 12_lead_braking_390d9e3b (car_9), 20_J-UL_52f7393a (car_14).

Rule counts over tier 1 (scorer's aggregate on the 10 files): `analysis/tier1_aggregate_20261107.md`. Gate H: no FAIL on any of the 10 scenes.

**TACTICAL:** N/A for the executed-decision confusion table in this read, per A-14 (AMENDMENT 1): the event-detection definition is still a work item. A missing family is a work item, not a pass. The c1 to c4 columns above are the scorer's rule statuses. **STRATEGIC:** paused (PI ruling 2026-09-19), never a pass, never omitted.

## 2. Contacts and an instrument defect

Scorer's f1 events (NAVSIM classification). `behind human` is how far the ego's station is behind the human's at the same time. All contacts come after the scene's first redraw.

| scene | car | type | at fault | t s | ego v | car v | human v | behind human m | car centre lon, lat m |
|---|---|---|---|---|---|---|---|---|---|
| 06_steady_lead_024d89a3 | car_44 | ACTIVE_REAR | no | 9.4 | 3.16 | 8.04 | 8.82 | 35.02 | [-4.41, -0.21] |
| 06_steady_lead_024d89a3 | car_51 | ACTIVE_LATERAL | no | 13.7 | 6.99 | 6.48 | 7.88 | 46.21 | [-3.82, 2.19] |
| 12_lead_braking_390d9e3b | car_9 | ACTIVE_REAR | no | 10.6 | 3.72 | 8.18 | 6.73 | 21.22 | [-4.52, -0.03] |
| 20_J-UL_52f7393a | car_14 | ACTIVE_REAR | no | 13.9 | 0.36 | 7.23 | 2.91 | 26.47 | [-4.84, 0.57] |
| 20_J-UL_52f7393a | car_1 | STOPPED_TRACK | yes | 14.6 | 0.97 | 0.0 | 2.76 | 27.86 | [5.06, -0.71] |

**Only one at-fault contact in tier 1:** scene 10 (J-UL), at 0.97 m/s into a stopped car about 5 m ahead, right after a logged car had run into the ego from behind.

**Instrument defect (logged, not patched, so all 28 scenes stay comparable).** The NC term saved in `rollout.npz` comes from `engine.py`, which skips a replay car only while its centre is more than 150 deg behind the ego. A replay car that rear-ends the ego drives through it, the bearing drops below 150 deg, and it then counts as a collision. NAVSIM exempts the track after the first not-at-fault contact. That is why scenes 4 and 7 have EPDMS 0.0 with f1 PASS. The fix goes in with the tick patch or as its own change, with a baseline pre-registered first.

**Mechanism, HYPOTHESIS:** on the three contact scenes the ego is 21 to 35 m behind where the human was and far slower than the logged car behind it, which cannot react. It is not tested here. A read with n, contact against no-contact scenes, follows after the 28.

## 3. G-R: the same checkpoint on real seed frames (AMENDMENT 8 item 4)

First-call plan_v4 / plan_v0. `sim` is the first call in the closed loop, `real f-theta` and `real pinhole` are the real logged seed frames. T0-class for the real-frame forward. Limits: pinhole frames warped to f-theta with masks; the real frame is from t = 0 and the sim's first call sees about t = 0.53 s.

| scene | category | v m/s | sim | real f-theta | real pinhole | control (rerun of sim frames) |
|---|---|---|---|---|---|---|
| 00_cruise_86cad912 | cruise | 12.28 | 0.825 | 0.827 | 0.829 | 0.83 |
| 01_cruise_59714859 | cruise | 12.50 | 0.809 | 0.809 | 0.809 | 0.705 |
| 02_cruise_13fa82d6 | cruise | 11.02 | 0.801 | 0.807 | 0.802 | 0.802 |
| 06_steady_lead_024d89a3 | steady_lead | 9.12 | 0.611 | 0.551 | 0.685 | 0.613 |
| 07_steady_lead_adfdcd3c | steady_lead | 5.17 | 0.403 | 0.41 | 0.417 | 0.4 |
| 08_steady_lead_87751a07 | steady_lead | 9.41 | 1.05 | 1.067 | 1.145 | 1.029 |
| 12_lead_braking_390d9e3b | lead_braking | 11.53 | 0.967 | 0.86 | 0.865 | 0.967 |
| 14_stop_go_95472243 | stop_go | 10.41 | 0.601 | 0.727 | 0.602 | 0.722 |
| 16_merge_cutin_afd632f0 | merge_cutin | 9.79 | 0.897 | 0.904 | 0.904 | 0.907 |
| 20_J-UL_52f7393a | junction | 4.94 | 0.64 | 0.71 | 0.698 | 0.671 |

The control re-runs the saved sim frames through the G-R path. It picks the same proposal with plan_v4 within 0.05 m/s on 4 of 10 scenes. Not on: 00_cruise_86cad912 (same pick: False, plan_v4 difference 0.0 m/s); 01_cruise_59714859 (same pick: True, plan_v4 difference -1.242 m/s); 08_steady_lead_87751a07 (same pick: True, plan_v4 difference -0.197 m/s); 14_stop_go_95472243 (same pick: True, plan_v4 difference 1.25 m/s); 16_merge_cutin_afd632f0 (same pick: False, plan_v4 difference 0.0 m/s); 20_J-UL_52f7393a (same pick: True, plan_v4 difference 0.126 m/s). REFe's trajectory is bf16, so plan_v4 sits on a grid of about 1.25 m/s at 32 to 64 m, and one different pick moves the ratio by a grid step.

On real frames v4/v0 is below 0.95 on 9 of 10 first calls (f-theta warp), and the free-road scenes (3 cruise) read 0.827, 0.809, 0.807. Median difference sim against real f-theta: 0.012, largest 0.126. On the cruise scenes the slowdown belongs to the model at the first call: rendering cannot explain it. On lead, stop-go, merge and junction scenes a lower planned speed at the first call can be right, so G-R alone does not show a defect there. It says nothing about later calls.

## 4. Windows, descriptive (AMENDMENT 8 item 3, AMENDMENT 9 item 2)

Rules a1, a2, c4, e1, e2 in that order, using the pinned scorer's rule code with only the valid ticks and free-road intervals narrowed. These windows decide no rule. `to_first_redraw` differs per scene and must not be compared across checkpoints as a fixed window. Limited by `human_ref_valid_until_s`.

| scene | full | pre 0-6 s | post > 6 s | to_first_redraw | ends at s | human ref valid until s |
|---|---|---|---|---|---|---|
| 00_cruise_86cad912 | F F - F F | F F - P P | F F - F F | F P - P P | 4.3 | whole |
| 01_cruise_59714859 | F F - F P | F F - P P | F F - F P | - - - P P | 3.75 | whole |
| 02_cruise_13fa82d6 | F F - F F | F F - F P | F F - F F | F P - F P | 4.8 | whole |
| 06_steady_lead_024d89a3 | F F - F P | F P - F P | F P - P P | - - - F P | 4.3 | whole |
| 07_steady_lead_adfdcd3c | F F - F F | F F - F F | F F - F F | F F - F F | 13.35 | 11.75 |
| 08_steady_lead_87751a07 | F F - F F | - - - F F | F F - F P | - - - F F | 4.3 | whole |
| 12_lead_braking_390d9e3b | F F - P P | - - - P P | F F - P P | - - - P P | 3.75 | whole |
| 14_stop_go_95472243 | - - - F F | - - - P P | - - - P F | - - - P P | 5.9 | whole |
| 16_merge_cutin_afd632f0 | F F - F P | F F - P P | F F - F P | - - - P P | 3.75 | whole |
| 20_J-UL_52f7393a | I F - F F | I F - F F | F F - F F | - - - F F | 3.75 | 15.15 |

| rule | pre: P / F / other | post: P / F / other | to_first_redraw: P / F / other |
|---|---|---|---|
| a1 | 0 / 6 / 4 | 0 / 9 / 1 | 0 / 3 / 7 |
| a2 | 1 / 6 / 3 | 1 / 8 / 1 | 2 / 1 / 7 |
| c4 | 0 / 0 / 10 | 0 / 0 / 10 | 0 / 0 / 10 |
| e1 | 5 / 5 / 0 | 3 / 7 / 0 | 5 / 5 / 0 |
| e2 | 7 / 3 / 0 | 5 / 5 / 0 | 7 / 3 / 0 |

Sanity: with no narrowing the same code reproduces the saved a1, a2, c4, e1, e2 status on all 10 scenes.

## 5. Render fidelity (AMENDMENT 8 item 2, AMENDMENT 9 item 1)

Embedding pinned before any value was read: torchvision fasterrcnn_resnet50_fpn_v2 COCO, ResNet-50 backbone, layer4 global average, 480x270 RGB, ImageNet normalisation, weights sha256 `dd69338a24b8d738...`. **Every scene is `no_real_reference`:** real frames exist on thor3 only at t0 and t0 - 2 s per token, so only the weaker within-render similarity between frames 1 s apart is given. No threshold is set. Per-frame series against arc length: `analysis/render_fidelity/<scene>.npz`.

| scene | frames | median | p05 | min | median, no redraw between | median across a redraw | min across a redraw | median before first redraw |
|---|---|---|---|---|---|---|---|---|
| 00_cruise_86cad912 | 527 | 0.9739 | 0.907 | 0.8806 | 0.9761 | 0.9182 | 0.8806 | 0.9715 |
| 01_cruise_59714859 | 527 | 0.9629 | 0.8751 | 0.8526 | 0.9661 | 0.912 | 0.8526 | 0.9684 |
| 02_cruise_13fa82d6 | 527 | 0.962 | 0.8894 | 0.8589 | 0.9632 | 0.8932 | 0.8589 | 0.9648 |
| 06_steady_lead_024d89a3 | 527 | 0.9685 | 0.9201 | 0.9045 | 0.9697 | 0.9208 | 0.9045 | 0.9668 |
| 07_steady_lead_adfdcd3c | 527 | 0.9708 | 0.7816 | 0.7475 | 0.9717 | 0.7744 | 0.7475 | 0.9698 |
| 08_steady_lead_87751a07 | 527 | 0.9723 | 0.9285 | 0.889 | 0.9751 | 0.9304 | 0.889 | 0.9567 |
| 12_lead_braking_390d9e3b | 527 | 0.9692 | 0.9168 | 0.8595 | 0.9707 | 0.9225 | 0.8595 | 0.9699 |
| 14_stop_go_95472243 | 527 | 0.9775 | 0.919 | 0.9065 | 0.9794 | 0.926 | 0.9065 | 0.9686 |
| 16_merge_cutin_afd632f0 | 527 | 0.9694 | 0.8658 | 0.8522 | 0.9726 | 0.9284 | 0.8522 | 0.9783 |
| 20_J-UL_52f7393a | 527 | 0.9703 | 0.9027 | 0.8885 | 0.9724 | 0.9251 | 0.8885 | 0.9655 |

### Redraws per scene and their triggers (chunk stamps are in `analysis/tier1_posthoc.json`)

| scene | redraws | at t s (trigger, camera that fired) |
|---|---|---|
| 00_cruise_86cad912 | 3 | 4.8 (veil on front_wide) / 9.1 (detail on front_wide) / 14.95 (marks on front_wide) |
| 01_cruise_59714859 | 3 | 4.3 (veil+marks on front_wide) / 8.55 (marks on front_wide) / 15.5 (detail+marks on front_wide) |
| 02_cruise_13fa82d6 | 2 | 5.35 (veil on front_wide) / 13.9 (marks on front_wide) |
| 06_steady_lead_024d89a3 | 2 | 4.8 (marks on front_wide) / 11.2 (marks on front_wide) |
| 07_steady_lead_adfdcd3c | 1 | 13.9 (veil on front_wide) |
| 08_steady_lead_87751a07 | 2 | 4.8 (veil on front_wide) / 9.1 (marks on front_wide) |
| 12_lead_braking_390d9e3b | 2 | 4.3 (marks on front_wide) / 14.4 (marks on front_wide) |
| 14_stop_go_95472243 | 3 | 6.4 (marks on front_wide) / 11.75 (marks on front_wide) / 16.55 (marks on front_wide) |
| 16_merge_cutin_afd632f0 | 3 | 4.3 (veil+detail on front_wide) / 9.1 (detail+marks on front_wide) / 13.35 (detail+marks on front_wide) |
| 20_J-UL_52f7393a | 2 | 4.3 (veil on front_wide) / 13.35 (marks on front_wide) |

23 redraws over 10 scenes, first between 3.75 and 13.35 s. Each is a whole-scene redraw with no image condition. Guard settings: {"reanchor_below": 0.3, "veil_below": 0.75, "detail_below": 0.65, "marks_below": 0.6, "min_gap": 8}.

## 6. How far the verdicts can be read

- Render drift is not cited against any failed rule here unless G-R or a window separates it (AMENDMENT 8 item 5).
- A "drives well" statement is limited to what `to_first_redraw` supports (AMENDMENT 9 item 4). On a1 and a2 that window is short: warm-up 2 s plus a 2 s minimum interval leaves many scenes not applicable.
- **Frames inspected:** render statements in this report rest on the embedding numbers and the guard's stamps. By eye, so far: 3 front frames of 86cad912, and the 8 s contact sheets of the 3 cruise scenes in the render proof (8 frames per arm). No frame of scenes 4 to 10 has been inspected by eye. "Follows the caption over the seed" stays a withdrawn hypothesis, never a scene filter.
- One seed. Second seed 20261108 is planned after tier 1.

## 7. Goal rounding (descriptive, no causal claim)

Goal points rounded to 2 decimals change the picked proposal on 9 of 93 calls on the 3 scored cruise scenes; largest plan_v4 change 1.95 m/s, median under 0.02 m/s (`analysis/goal_round_sensitivity.json`). Covers those 3 scenes only.

## 8. Run record

Pause 1: suite held 2026-11-08 15:58:21 CET, between scene 4 and scene 5, render proof runs ended 2026-11-08 19:16:42 CET, Phone2Data window 2026-11-08 19:17:19 CET to 2026-11-09 00:02 CET, resumed 2026-11-09 00:05:36 CET.

Pause 2 (Phone2Data longer-request test, approved by the PI in that session): stop signal 2026-11-09 00:10:22 CET, scene 5's run ended 2026-11-09 00:53:04 CET, Cosmos handed over 2026-11-09 00:53:55 CET, window ended by 2026-11-09 01:56 CET (container gone and claim released at the 01:56:25 check), resumed 2026-11-09 01:57:47 CET. The log's run times for scene 4 (529 min) and scene 5 (110 min) include the pauses: the runs took about 48 and 46 min.

Config compared at each resume (`analysis/suite_resume_config.json`, `suite_resume2_config.json`): image `sha256:3ebd890bc438...`, yaml `6eb709afaf4af982...`, pipeline patch `4e10d6eb27fe7aad...`, env unchanged, `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` not set. Difference between the two resume dumps: none (time and container id aside). Limit: the container of scenes 1 to 4 no longer exists, so the comparison is same file versions and the env list read before its removal. Scene 1 ran before the slot label filter (`rollout.py` 2026-11-08 13:40 CET) and was re-scored after a hand fix. Nothing from the render proof was adopted.
