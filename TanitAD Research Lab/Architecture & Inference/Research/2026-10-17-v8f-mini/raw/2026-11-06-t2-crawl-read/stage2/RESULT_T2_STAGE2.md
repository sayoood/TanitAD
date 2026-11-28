# RESULT_T2_STAGE2: the T2 crawl read (PREREG_T2_CRAWL, frozen 2c4c6557), Master Mind, 2026-11-06

**Tier T2 re-perception** (camera rendered at the executed pose and time, NuRec agents replayed at log time, playback executor). Never pooled with T1-kin.

## Outcome, as the pinned read wrote it (`READ_STAGE2.json`, read sha256 `f2136b39`, rule R-SCN)
**CRAWL-PERSISTS, sub-branch (a).** On M1, the crawl share, over 35 launch scenes and 67 starts:
* X1 = C0(T2) - P10(T2): **+0.3388 [+0.2555, +0.4291] sep** at s0 and **+0.4256 [+0.3231, +0.5224] sep** at s1; h = 0.0267 / 0.0298. Seed gap 0.0868.
* X2 = C0(T2) - C0(RR): -0.0021 [-0.0400, +0.0375] at s0 and +0.0208 [-0.0067, +0.0525] at s1. The camera clock does not explain the crawl.
* BL = BLIND - C0(T2): +0.0111 [-0.0704, +0.0917] at s0 and -0.0757 [-0.1599, +0.0143] at s1. Replacing the scene tokens by their training mean does not change it.

The committed lever is the card's 2(a): the perception lever (the A40 LoRA run, `D-V8F-SPEED-PERCEPTION` (a)) goes to the PI, with T2 as its tier and this read as the case.

## Scenes the renderer refused (rule R-SCN, `AMENDMENT_T2_SCENE_REFUSAL.md` frozen 2167a7eb, written before any row was computed)
7 of 42 launch scenes, all 14 cells each, before any planner ran (`EXCLUSIONS2.json`, ATTRIBUTED):
* 6 ACTOR_REFUSED: `dffd4c04`, `00169207`, `1aa48b8a`, `c67bdfef`, `aa094af8`, `42a44314`. Each refusal's track count equals its own `sequence_tracks.json`.
* 1 NO_DYNAMIC_LAYER: `90a3d349`.
* 14 of the 81 starts are lost. They are not replaced.

## Beside, NOT deciding: every row of the T1 reader, C0_s(T2) - P10(T2), paired scene-cluster bootstrap B 2000 seed 0, 35 scenes
Error rows: lower is better, so a positive delta means C0 is worse. TAC and S-POS rows take Read A's direction, which the dominance read applies.
STRATEGIC is PAUSED by PI ruling ITEM 25 (2026-09-19); its rows are shown and are not a pass.

| row | family | P10 mean | s0 delta [CI] | s1 delta [CI] | n windows/scenes |
|---|---|---|---|---|---|
| ADE6 | LONGITUDINAL+LATERAL | 6.082 | +1.4845 [+1.0622, +1.9001] sep | +1.2200 [+0.6300, +1.7351] sep | 67/35 |
| speed MAE | LONGITUDINAL | 3.317 | -0.6364 [-1.1399, -0.1462] sep | -0.7322 [-1.2104, -0.2829] sep | 67/35 |
| along-track MAE | LONGITUDINAL | 4.716 | +2.4368 [+2.0458, +2.8107] sep | +2.2227 [+1.6466, +2.7433] sep | 67/35 |
| accel MAE | LONGITUDINAL | 6.699 | -2.6961 [-3.7573, -1.6240] sep | -4.0461 [-5.4084, -2.7634] sep | 67/35 |
| heading MAE | LATERAL | 33.026 | +10.7286 [+5.9367, +15.5984] sep | +5.7704 [+1.3532, +10.4046] sep | 67/35 |
| yaw-rate MAE | LATERAL | 117.364 | +20.4621 [+7.3489, +34.4045] sep | +10.6545 [-11.4006, +29.2237] | 67/35 |
| curvature MAE | LATERAL | 0.423 | +0.6071 [+0.4382, +0.8007] sep | +0.7418 [+0.5353, +0.9643] sep | 67/35 |
| cross-track MAE | LATERAL | 2.806 | -1.1576 [-1.4599, -0.8689] sep | -1.4590 [-1.7797, -1.1680] sep | 67/35 |
| L1 lateral-accel MAE | LATERAL | 23.027 | -11.1400 [-15.1846, -7.2678] sep | -15.4340 [-20.6899, -10.5104] sep | 67/35 |
| L2 curvature MAE (GT>=5.2) | LATERAL | 0.236 | +0.6881 [+0.4592, +0.9686] sep | +0.8622 [+0.5793, +1.1905] sep | 67/35 |
| M1 crawl share | LONGITUDINAL | 0.287 | +0.3388 [+0.2555, +0.4291] sep | +0.4256 [+0.3231, +0.5224] sep | 67/35 |
| TAC-LAT | TACTICAL | 0.313 | +0.0299 [-0.0882, +0.1493] | +0.0299 [-0.0735, +0.1231] | 67/35 |
| TAC-LON | TACTICAL | 0.985 | -0.3582 [-0.4783, -0.2388] sep | -0.3284 [-0.4412, -0.2206] sep | 67/35 |
| DK-HW | LONGITUDINAL | 2.436 | -0.2227 [-1.0179, +0.6281] | -0.2134 [-1.2180, +0.7619] | 37/21 |
| DK-TG | LONGITUDINAL | 0.891 | +0.0884 [-0.2592, +0.6387] | +0.0407 [-0.4786, +0.6730] | 24/20 |
| DK-VIOL | LONGITUDINAL | 0.000 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 25/21 |
| S-POS@30 | STRATEGIC | 1.864 | -3.2517 [-5.4559, -1.0540] sep | -1.6301 [-4.6448, +1.3778] | 64/34 |
| S-POS@50 | STRATEGIC | -2.995 | -4.4233 [-9.6645, +1.0194] | -1.3507 [-7.4658, +5.1771] | 50/27 |
| S-POS@80 | STRATEGIC | -16.310 | -6.4248 [-19.8905, +6.5739] | -4.2690 [-18.0485, +10.8870] | 21/11 |

* **Population:** launch scenes are selected (t_c - 4 s and t_c - 2 s around a launch). At T1-kin on eval-141 the same contrast is +0.0583 / +0.0586 (`cl_read/RESULTS.md`), on a different population.
* **Per start class** (launch: v0 < 0.5 m/s, 24 starts; moving: 43 starts): `SCORE2.md`.
