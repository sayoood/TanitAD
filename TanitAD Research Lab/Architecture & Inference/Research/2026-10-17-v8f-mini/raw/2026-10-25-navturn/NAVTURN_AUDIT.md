# NAVTURN_AUDIT: why v8F-mini's nav token and its turn labels disagree on eval-141 (2026-10-25)

**EXPLORATORY. Decides nothing; it corrects the premise of T_ATTRIB rank 1 and ranks what to pre-register next.**
* **Evidence class:** MEASURED (ours, scripts + `.out` in this directory) unless a line says ESTIMATED / HYPOTHESIS / UNVERIFIED. The source line quotes are from the mirror at `c7d40467`.
* **Data:** eval-141 (2,399 windows, 141 episodes; 2,365 windows / 139 episodes carry a v9 row). Population: the banked 339 = A 210 (turn-label cells present at the T-tick row, nav token FOLLOW at the NOW row) + B 129 (nav token TURN, no turn-label cell).
* **Tier:** T0 diagnostic of T's own code (T-vs-GT error); the ADE bounds are the T_ATTRIB (E) outer bounds (T0, O fed the GT code).
* **Estimator:** counts are exact. Every RMSE / share carries an episode-cluster bootstrap (B = 2,000, 141 episodes). It answers *another draw of EPISODES* only. Single-seed arms: M2b s0 and s1 are the pair (`H-ESTIM-SEED-1`). Window counts are not independent: 156 windows are 20 events (see section 3).

## HEADLINE

1. **The two annotations answer different questions, built from the same turn table.**
   * **Nav token:** "is the NEXT announced junction-type turn within `max(30 m, 6 s x v_now)` of arc, or already in progress?" Announced = `is_turn` (|dyaw| >= 15 deg, R_arc <= 140 m, **vmin <= 8 m/s**) and not suppressed. Spatial, whole recording ahead.
   * **Turn-label cells:** "what is the dominant >= 20 deg sustained-yaw segment overlapping the time band [NOW+2 s, NOW+8 s]?" **Any** segment qualifies: no `is_turn` gate, no suppression gate, no 30-deg gate.
2. **A (the 210) is 74 % a speed-gate artefact.**
   * **156 windows (74.3 %), 20 events in 15 episodes:** fast bends / ramps / roundabouts (median |dyaw| 92 deg, R_arc 63 m, vmin 11.2 m/s). The only reason they fail `is_turn` is vmin > 8 m/s (smallest margin 8.40 m/s). The class label calls all 156 LANE_KEEP (133) or none (23), **zero TURN**.
   * 21 windows (10.0 %, **2 episodes**) are an A7 turn-suppression that silences the nav but not the cells.
   * 16 windows (7.6 %) are a real horizon strip: the turn starts 6.0 to 7.9 s ahead, past the nav's 6 s x v.
   * 17 windows (8.1 %) are UNVERIFIED (event beyond the pose track); 13 of 17 convert when the gate is lifted, so they are probably curves too.
   * **Timing offset: 0.**
3. **B (the 129) is a horizon-and-band mismatch, not a defect.**
   * 50 (38.8 %): the nav announces a turn that starts after NOW+8 s (30 m floor at low speed; v_now overstates the approach speed when the ego brakes for the turn).
   * 40 (31.0 %): the ego is already mid-turn and the turn ends in a median 1.0 s; the label band starts at +2 s.
   * 13 (10.1 %): label block masked (reversing); 4 grazing the band edge; 1 sign/theta rule; 17 (13.2 %) UNVERIFIED (tail).
   * **4 (3.1 %) are an artefact of the banked row pairing** (the nav token is taken at NOW, T reads it at the T-tick row).
4. **Consequence for T_ATTRIB rank 1 ("goal / route input for route-silent turn windows"): its proof of existence is confounded.**
   * The 0.95 SD (silent) vs 0.40 SD (speaks) contrast compares fast bends with slow tight junction turns. **No speaking window can have vmin > 8 m/s**, because the nav gate forbids it.
   * **Like-for-like** (silent AND `is_turn`, 37 windows, 11 events) vs speaking (266): RMSE **0.651 [0.502, 0.752] vs 0.410 [0.347, 0.472]** (s0); 0.757 vs 0.396 (s1). That silent set holds phi 0.015 / 0.020 of T's error, not 0.189 / 0.181.
5. **What a consistent definition could move (turn-cell squared error of the 210; s0 / s1):**
   * Label-side (stop describing non-announced events as turn cells): **81 % / 80 %** (phi 0.154 / 0.144; up to 97 % / 97 % if the 17 unresolved are curves too). **A metric / supervision change, 0 m of driving by itself.**
   * Nav-side like-for-like (let the nav speak for the 37 silent junction events): 4.9 % / 8.2 % (phi 0.009 / 0.015). ESTIMATED; it supplies the ego's own future path.
   * Nav-side announcing curves: HYPOTHESIS 62 % / 59 % (phi 0.117 / 0.106). Not admissible as built (optimistic route; and it would turn the nav token into the situation label).
   * **Driving ceiling of everything in the 210 is the banked (E): 0.161 / 0.142 m, below 2F = 0.2415 m.** CURVE alone is 0.114 / 0.105 m.
   * **The 210's turn-cell error is therefore mostly a metric / label-definition artefact plus a perception gap on 20 bend events, not a route-input lever.**
6. **Two corrections to the banked T_ATTRIB:**
   * "Route silent" is token-silent only. 51.8 % of FOLLOW windows (1,024 / 1,976 at the T-tick row; 5,729 / 11,462 join-wide) still carry `args_valid = 1` and a distance to the next announced turn. In the 210, 16 windows carry the label's own event distance (RMSE 0.555 / 0.643), 92 a different later turn, 102 nothing (RMSE 1.010 / 1.020). The comment in `s2c_routesilent.out` ("FOLLOW carries args_valid only for TURN tokens") is false.
   * T reads its nav at the T-tick row (`rowT = t - a_T`), the banked script reads it at `rowN = t`. Properly paired the count is 214 + 127 = 341 (overlap 210 / 125).

## 1. Both derivations, from source (Q1)

All paths are in the mirror at `c7d40467`. Both annotations come from `build_v9_labels.py`'s turn table, which comes from `s2_geom_emit_v7.py`.

| | **Nav token** | **Turn-label cells** |
|---|---|---|
| inputs | the ego's own 10 Hz track of the **whole recording**, arc length, current speed only (`build_v9_labels.py:233-257, 263-283`) | the ego's future track in [NOW+2, NOW+h] (`:806-847`) |
| event set | `ann = [u for u in turns if u.announced and u.s_end > s_now]`, announced = `is_turn and not suppressed` (`:264`, `:229-230`) | **all** turns in the table: `for u_ in turns:` no `is_turn`, no suppression test (`:819-826`) |
| horizon | **distance**: `d_ann = max(30 m, 6 s * max(v_now, 0))` (`:265`; literals `:44-45`); token = TURN iff `in_prog or d_next <= d_ann` (`:279-280`) | **time**: `b0, b1 = now + 2.0, now + min(h, 8.0)` (`:817`; `band_s (2.0, 8.0)` `:34`); needs `h_obs >= 3 s` and `< 0.3 s` reversing (`:806-808`; `:41, :453`) |
| which event | the nearest by `s_start` (`:272`) | the one with the largest yaw change inside the band whose sign equals the sign of theta, theta = signed max yaw change after NOW+2 (`:813-826`) |
| threshold | `is_turn(seg)`: `dyaw >= 15 and R <= 140 and vmin <= 8.0` (`s2_geom_emit_v7.py:156-176`; `:87-89`) | none beyond the segmenter: smoothed yaw rate >= 6 deg/s for >= 0.8 s and \|dyaw\| >= 20 deg (`s2_geom_emit_v7.py:60-62, :117, :127`) |
| suppression | A7 clip-level, segments with `t_start` in raw [8, 43] s (`build_v9_labels.py:245-251`, literal `:46`) | ignored |
| what the model reads | one-hot token + `d_next/100` clipped to [0, 3] + `args_valid` (`v8f_mini_data.py:199-209`). **`args_valid = 1` and `d_next` are set for the nearest announced turn even when the token stays FOLLOW** (`build_v9_labels.py:272-278`) | the 26-cell regression target `cons` (`v8f_mini_data.py:218-219`), cells 1..9 = the turn cells |
| row | T reads nav at `i = last_t = last_now - a_T` (`v8f_mini.py:563, :603`; `v8f_mini_data.py:506, :510`) | `lab_t` = row `t - a_T` (`v8f_mini_data.py:506`) |

The class label shares the segmentation but not the cells' freedom: `big = abs(theta) >= 30`, `is_junc = big and dom is not None and dom.is_turn`, and `lat_cls_a` is TURN only if `is_junc` (`:828-830`, `:874-882`). **The cells are populated for `dom` whether or not the class is TURN** (`:832-847`).
* MEASURED (`a3_error.out`): of the 476 windows with turn cells, 197 have class TURN, **225 LANE_KEEP and 54 none**. 59 % of turn-cell windows describe an event the class label does not call a turn.

## 2. The reconstruction, and how far to trust it

`r1_repro.py` re-runs the mirror's own `clip_geo` / `clip_labels` (unmodified) on the eval-141 pose tracks, with the clip clock recovered from the join. Both suppression settings are run per clip.
* **Interior rows** (>= 8.5 s and >= `d_ann` + 5 m of reconstructed track left; 7,037 of 13,896 rows, 49.4 %): nav token **7,036 / 7,037**, turn-label presence **7,036 / 7,037**, `lat_cls_a` 7,034 / 7,037. Values within tolerance: t_start, d_start, R_arc 100 %; t_end 94.5 %; dyaw 94.6 % (`r1_repro.out`).
* **All rows:** 12,999 / 13,896 = 93.5 %. The rest are tail rows where the **real ego log is longer than the 20 s pose track** (label block present in 13,539 rows, re-run 11,755). On those rows I do not claim the re-run; evidence tiers per window are in `navturn_windows.csv` (`ev` = REPRO / TRACK / JOIN / UNVERIFIED).
* **Controls** (`a7_controls.out`): suppression ON changes the re-run token in 23 of 141 clips and the join follows ON in only 2 (`dbad28c2`, `eb96a48b`), OFF in 21. For all 156 CURVE windows the re-run token equals the join's FOLLOW, and the smallest vmin over the gate is 8.40 m/s (speed noise 0.022 m/s). A (the 210): re-run token = join token on 210 of 210. B: 67 of 129 (the rest are tail rows).
* **Mutation test** (`a4_sweep.out`): lifting the speed gate converts 146 of the 156 CURVE windows to agreement; the other 10 stay silent because they are also past the horizon. The attribution is causal, not a correlation.

## 3. A: turn label present, nav token FOLLOW (n = 210) (Q2)

Banked count reproduced (`n1_counts_and_tickrow.out`). Turn-cell RMSE in SD units, M2b s0 / s1 (`a3_error.out`); phi = share of all T squared error.

| cause | class | n | share | episodes / events | RMSE s0 [CI] | phi s0 / s1 | share of 210's SSE s0 / s1 | (E) m s0 / s1 |
|---|---|---|---|---|---|---|---|---|
| **CURVE**: `is_turn` fails the vmin <= 8 gate | threshold mismatch | 156 | 74.3 % | 15 / 20 | 0.968 [0.843, 1.084] | 0.143 / 0.130 | 75.8 % / 71.7 % | 0.114 / 0.105 |
| **SUPPRESSED**: A7 flag silences nav, cells stay | label / policy defect | 21 | 10.0 % | 2 / 2 | 0.711 [0.000, 0.791] | 0.010 / 0.014 | 5.5 % / 7.9 % | 0.018 / 0.020 |
| **HORIZON**: `d_next > d_ann` (ratio median 1.13) | horizon mismatch | 16 | 7.6 % | 9 / 9 | 0.564 [0.434, 0.685] | 0.005 / 0.006 | 2.6 % / 3.5 % | 0.014 / 0.006 |
| UNRESOLVED: event beyond the pose track | unverified | 17 | 8.1 % | 4 / 4 | 1.347 [0.409, 1.725] | 0.030 / 0.031 | 16.0 % / 16.9 % | 0.015 / 0.011 |
| timing offset | | 0 | 0 % | | | | | |
| **all** | | **210** | | 27 / 35 | 0.958 [0.843, 1.061] | 0.189 / 0.181 | | 0.161 / 0.142 |

* Banked phi 0.189 / 0.181 and (E) 0.161 / 0.142 reproduce.
* **CURVE** detail (`a5_detail.out`):
  * |dyaw| p25 / median / p75 = 47 / 92 / 151 deg; R_arc 48 / 63 / 76 m; vmin 8.7 / 11.2 / 12.5 m/s; the segment lasts a median 7.9 s.
  * vmin bands: (8, 10] 59, (10, 12] 41, (12, 15] 37, > 15 19. The 59 within 25 % of the gate are the genuinely ambiguous ones; the other 97 are clearly fast.
  * Top-1 / 3 / 5 episodes carry 17 / 44 / 64 % of the CURVE error (s0): a few bends drive it.
* **SUPPRESSED:** both clips have an `is_turn` junction turn at t0 = 11.8 / 11.9 s, inside the suppression window [8, 43] s. The CI is degenerate (2 episodes).
* **HORIZON:** label t_start is 6.0 to 7.9 s (the strip between the nav's 6 s and the label's 8 s); 13 of 16 would be announced by an 8 s x v_now horizon. T sees the event distance in 14 of them (nav distance = label d_start), which is why its RMSE (0.56) is already near the speaking level.

**Examples** (ep = episode prefix; t = cache step, 0.2 s; `d_next`, `d_ann` in m; label cells at the T-tick row; theta = lat_theta_deg; full rows in `a2_classify.out`, `navturn_windows.csv`):

| cause | window | ego yaw | nav state | label values |
|---|---|---|---|---|
| CURVE | ep `3971a8c6` t = 3 | segment dyaw +91.6 deg, theta +43.1, dpsi6 +44.5 | FOLLOW, d_next 166 m, d_ann 73.5, v 12.3 m/s | t_start -0.5 s, t_end 11.7 s, R 92 m, vmin 11.80, class LANE_KEEP |
| CURVE | ep `262b900f` t = 47 | dyaw -21.6, theta -13.2 | FOLLOW, d_next 170 m, d_ann 98, v 16.3 | t_start 5.8 s, R 125 m, vmin 12.41, class none |
| CURVE | ep `316f3c56` t = 39 | dyaw +50.2, theta +7.1 | FOLLOW, d_next 244 m, d_ann 30, v 3.6 | t_start 7.2 s, R 63 m, vmin 8.41 (near gate), class none |
| CURVE | ep `32f9b6b2` t = 23 | dyaw +30.8, theta +7.0 | FOLLOW, d_next 525 m, d_ann 116, v 19.4 | t_start 6.7 s, R 116 m, vmin 15.72, class none |
| SUPPRESSED | ep `dbad28c2` t = 23 | dyaw -90.1, theta -9.4 | FOLLOW, d_next 453 m (a different, far turn), d_ann 88 | t_start 7.3 s, R 19.6 m, vmin 5.75, `is_turn` true |
| SUPPRESSED | ep `eb96a48b` t = 27 | dyaw -23.6, theta -19.8 | FOLLOW, d_next 180 m, d_ann 52 | t_start 6.7 s, R 26 m, vmin 5.68 |
| HORIZON | ep `01be5919` t = 15 | dyaw -64.9, theta -10.2 | FOLLOW, d_next 57.9 m = label d_start, d_ann 49.3, v 8.2 | t_start 7.4 s, R 21 m, vmin 5.34 |
| HORIZON | ep `18b72291` t = 15 | dyaw -97.1 | FOLLOW, d_next 57.8, d_ann 42.8, v 7.1 (ratio 1.35) | t_start 7.3 s, R 17 m |
| HORIZON | ep `79e30ac2` t = 11 | dyaw -89.5 | FOLLOW, d_next 64.2, d_ann 60.6, v 10.1 (ratio 1.06) | t_start 6.7 s, vmin 6.64 |
| UNRESOLVED | ep `a459164a` t = 67 | dyaw -29.9, theta -41.2 | FOLLOW, no valid distance | t_start 6.0 s, R 102 m, vmin inside the track 14.3 |

## 4. B: nav token TURN, no turn-label cell (n = 129) (Q2)

| cause | class | n | share | episodes / events | evidence |
|---|---|---|---|---|---|
| **HORIZON**: announced turn starts after NOW+8 s | horizon mismatch | 50 | 38.8 % | 9 / 10 | 44 by pose-track end (turn is beyond a track that still has >= 8 s), 6 by re-run table |
| **BAND_START**: announced turn ends before NOW+2 s | horizon / definition (label excludes the first 2 s) | 40 | 31.0 % | 19 / 22 | re-run table |
| UNRESOLVED: turn beyond the track, < 8 s of track left | unverified | 17 | 13.2 % | 8 / 8 | none |
| **LABEL_MASKED**: no label block at all (theta NaN) | label censoring | 13 | 10.1 % | 2 / 2 | join; all 13 re-run as reversing (ESTIMATED sub-cause, `h < 3 s` not excluded) |
| BAND_EDGE: turn overlaps the band by < 0.5 s | horizon (band edge) | 4 | 3.1 % | 4 / 4 | re-run table |
| **TIMING**: the T-tick row's token is FOLLOW | row-pairing artefact | 4 | 3.1 % | 4 / 4 | join |
| SIGN_THETA: overlaps >= 0.5 s but the dominant-segment sign rule rejects it | label rule | 1 | 0.8 % | 1 / 1 | re-run table |

* **HORIZON:** v_now median 8.1 m/s, nav `d_next` median 30.4 m, `d_ann` median 48.8 m. The nav distance is spatial and v_now is instantaneous: an ego braking for the turn needs 8+ s for 30 m (example `24fee8a5`).
* **BAND_START:** the ego is mid-turn; the turn ends in a median 1.0 s (p10 0.5, p90 1.7), started 2.7 s ago, median |dyaw| 65 deg. The label band is [NOW+2, NOW+8] by design (the operative layer owns 0 to 2 s), so no cell exists.
* **No B window carries T error:** absent cells have no error (R63). Their (E) mass is 0.049 / 0.063 m in total, with a per-window gap (0.91 / 1.17 m) at the all-window mean (1.08 / 0.92). B is a coverage question (T gets no supervision for a turn the nav announces), not an error source.

**Examples:**

| cause | window | ego yaw | nav state | label |
|---|---|---|---|---|
| BAND_START | ep `18b72291` t = 3 | segment dyaw +38.5, dpsi6 +30.3 | TURN_L in progress (d_next 0), v 5.7 | t0 -0.7 s, t1 +2.0 s, theta +4.9, class LANE_KEEP, cells absent |
| BAND_START | ep `1bd040a8` t = 19 | dyaw +84.2, theta -4.4 | TURN_L in progress, v 6.0 | t0 -3.3 s, t1 +0.8 s |
| BAND_START | ep `24fee8a5` t = 59 | dyaw +83.4, dpsi6 +18.7 | TURN_L in progress, v 7.4 | t0 -3.0 s, t1 +1.2 s |
| HORIZON | ep `24fee8a5` t = 3 | dyaw +83.4, theta +0.9, dpsi6 -2.9 | TURN_L, d_next 29.6 m, d_ann 64.2, v 10.7 | turn starts 8.2 s ahead, vmin 1.82 |
| HORIZON | ep `c08bebe5` t = 3 | dyaw +56.9, theta -3.2 | TURN_L, d_next 13.5 m, d_ann 30 (floor), v 0.0 | turn starts 9.4 s ahead |
| HORIZON | ep `fe83e06e` t = 35 | dyaw -81.9, theta -16.9 | TURN_R, d_next 23.9, d_ann 42.1, v 7.0 | turn starts 8.1 s ahead, vmin 0.84 |
| LABEL_MASKED | ep `af6f5964` t = 3 | theta NaN, dpsi6 -18.7 | TURN_R, d_next 3.6 m, v 0.55 | block masked (reversing) |
| TIMING | ep `9d3d1cd1` t = 27 (a_T = 1) | theta -9.6 | NOW row TURN_R d_next 77.7 vs d_ann 77.9; T-tick row FOLLOW | cells absent |
| BAND_EDGE | ep `5d39a1de` t = 47 | dyaw -63.2, theta +5.8 | TURN_R, d_next 27.6, v 4.5 | turn starts 7.6 s ahead, overlap 0.44 s |

## 5. What fixes what, and what it does to admissibility (Q3)

Mutation sweeps on the mirror's builder (`a4_sweep.out`; ESTIMATED where a tail window keeps the join token, so changes there are under-counted):

| lever | A / B / total (baseline 210 / 129 / 339) | side effect |
|---|---|---|
| is_turn gate 8 to 10 m/s | 152 / 132 / 284 | class TURN windows 194 to 240 |
| gate 12 | 113 / 134 / 247 | 275 |
| gate 15 | 66 / 134 / 200 | 315 |
| gate unbounded | 51 / 135 / 186 | 319 (+125, +64 %) |
| nav horizon 8 s (gate 8) | 197 / 155 / 352 | fixes 11 of 16 HORIZON, creates 26 new B |
| nav horizon 12 s | 189 / 184 / 373 | worse |
| gate unbounded + horizon 8 s | 32 / 170 / 202 | best single alignment still leaves 202 |

| cause | change nav? | change label? | neither? | admissibility |
|---|---|---|---|---|
| CURVE (156) | lifting the gate fixes 146, but moves the TURN class by +125 windows (the tactical family's base rates change) and puts the ego's own future bend path into the input | **stop writing turn cells for non-announced events, or add an event-kind flag and its own absent-mask**: pure supervision / metric change; consistent with the class (LANE_KEEP) | the curve geometry stays unpredicted by vision (T at chance, 20 events) | label side is admissible (labels may use ego future). Nav side is optimistic by construction (the route supplier is the ego's own path) and it would make the nav token a function of the same table that yields the situation label. |
| SUPPRESSED (21) | drop the suppression for nav | drop the cells under suppression | | either; 2 episodes. A label-release inconsistency. |
| HORIZON A (16) | horizon 8 s fixes 11, but creates 26 new B | cells only for events inside the nav horizon | | a single multiplier cannot fix both directions: nav is in distance with a 30 m floor, the label in time, and they cross at 5 m/s. A time-based nav equals the label band (an echo of the label). |
| HORIZON B (50), BAND_START (40), BAND_EDGE (4) | shorten the horizon / drop in-progress tokens: not computable without the future | extend the band to [0, 8] s, or leave the cells absent by design | **neither is a defect**: the label's [2, 8] band follows the three-layer split | no action needed beyond documenting. |
| LABEL_MASKED (13), SIGN_THETA (1) | | | by design / rare | none |
| TIMING (4) | | | fix the banked script to pair rows | none |

**Admissibility, measured** (`a3_error.out`): the nav token already tracks the class: of the 197 TURN-class windows, 177 (89.8 %) carry a same-side nav TURN token. Making the nav consistent with the label would push this to 100 % and make the "goal" input a function of the situation label. The binding rule (a goal input must not carry the situation classification) is the reason the nav-side options above are not clean levers; a PREDICTED goal point remains the admissible route.

**Share of the 210-window turn error a consistent definition could move** (turn-cell SSE, `a3_error.out`):

| definition change | s0 | s1 | what it is |
|---|---|---|---|
| label-side: no turn cells for CURVE + SUPPRESSED (177 windows) | 81.3 % (phi 0.154) | 79.6 % (phi 0.144) | **metric / supervision artefact**, 0 m ADE by itself. Up to ~97 % if the 17 unresolved are curves too (ESTIMATED). |
| nav-side like-for-like: nav speaks for the 37 silent `is_turn` windows, error to the 0.41 / 0.40 speaking level | 4.9 % (phi 0.009) | 8.2 % (phi 0.015) | ESTIMATED; needs the ego's own future path |
| nav-side announcing curves, error to the junction level | 62.2 % (phi 0.117) | 58.7 % (phi 0.106) | **HYPOTHESIS**: 0.40 was measured on slow, tight junction turns only, and it is an oracle input |

**Plain answer:** the 210-window turn error is mostly a metric artefact, in the sense that about four fifths of it sits in cells that describe events the programme's own tactical class calls LANE_KEEP. Removing it improves T's score and removes supervision of bend geometry; it does not move driving. The part that could be a driving lever is a perception gap (T is at chance, 0.97 SD, on 20 bend events in 15 episodes), outer bound 0.114 / 0.105 m, below 2F = 0.2415 m. The legitimate route-input part (HORIZON + SUPPRESSED, 37 windows) is worth at most 0.032 / 0.026 m.

## 6. Rule-Zero next step (cheapest experiments that could still make this lever work)

1. **0 GPU, now:** emit `turn_kind` (junction / curve) and the segment id into the join, and score every banked arm by kind. For M2b s0 / s1 the split is in section 3. The T_ATTRIB rank-1 should be re-stated as "bend geometry from vision" (CURVE) plus "horizon strip + suppression" (small), not "route input".
2. **Box 2, ~4 evals (extends T_ATTRIB rank 0):** the per-kind GT oracle with a **curve-only** and a **junction-only** turn-cell arm. Pre-register both outcomes: curve-only worth >= 0.10 m means bend geometry matters to O and is a perception lever; below 0.03 m means the cells are supervision noise and the label-side null is free.
3. **If a nav change is wanted at all:** a predicted goal point, information-disjoint from the situation head; do not lift the gate in the shipped nav (it moves the TURN class by +125 windows and ties nav to the label).

## 7. UNVERIFIED and caveats

* The pose track (20 s) is shorter than the real ego log: 34 A / B windows (17 + 17) are UNVERIFIED, and 44 B-HORIZON windows rest on the track-end argument (the turn starts beyond a track that still has >= 8 s), not on the re-run table.
* The A7 suppression flag is not in the join; "SUPPRESSED" is inferred from join-vs-re-run agreement (2 clips, 2 events).
* LABEL_MASKED's "reversing" sub-cause uses a pose-gradient reversing test; `h_obs < 3 s` is not excluded.
* Scope: eval-141 only. The builder is the same for train, so the definitional causes apply there; the rates there are unmeasured (INHERITED assumption).
* Window counts overstate the evidence: CURVE is 20 events, SUPPRESSED 2, A-HORIZON 9; B-HORIZON 10, BAND_START 22.
* The N1 / N2 estimates assume the speaking level (0.41 / 0.40 SD, 266 windows) transfers; the (E) bounds are T0 outer bounds on the whole constraint block, not on the turn cells.
* The nav-horizon sweep keeps the join token where the baseline re-run does not reproduce it (tail rows), so changes there are under-counted.
* T_ATTRIB's rank-1 figures (phi, (E)) were reproduced exactly; its interpretation, not its arithmetic, is what changes.

## 8. Deliverable manifest

Directory: `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/navturn/` (to be banked at `PK/raw/2026-10-25-navturn/`). Nothing staged, committed or pushed; every input was read from the mirror at `c7d40467` or the banked t-attrib files.

| file | what |
|---|---|
| `NAVTURN_AUDIT.md` | this report |
| `n1_counts_and_tickrow.py` / `.out` | reproduces 210 / 129, re-pairs on the T-tick row (341), nav-feature validity |
| `r1_repro.py` / `.out` | re-runs the mirror's v9 builder on the eval-141 pose tracks; validation table |
| `a2_classify.py` / `.out` | per-window cause classification, examples |
| `a3_error.py` / `.out` | T error, (E) bounds and counterfactual shares by cause; nav-vs-class echo |
| `a4_sweep.py` / `.out` | gate / horizon mutation sweep |
| `a5_detail.py` / `.out` | events, episode concentration, parameter distributions |
| `a6_navfeat.py` / `.out` | what the nav FEATURE vector says in the 210 |
| `a7_controls.py` / `.out` | instrument controls |
| `nt_common.py` | shared loaders |
| `navturn_windows.csv` | one row per disagreement window (cause, evidence tier, facts) |
| `repro_eval141.npz`, `repro_turns.json`, `navturn_cause.npy` | re-run per-row fields, per-clip turn tables, cause array |
| `src/` | read-only export of the mirror's builder and the banked t-attrib inputs (34 MB; re-derivable from `c7d40467`, do not bank) |
