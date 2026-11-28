# D1 — Which channel carries a value, on which window? The exact coverage census of refcv7-r101-s0

Stream D1 of the refcv8 data audit · Data Engineering · 2026-10-04 · every number below is MEASURED from the launch tree
`fec3a0d` and the run's own `config.json` / `metrics.jsonl`, unless a line says otherwise. Artifacts: `tables/` (the per-window truth
tables), `raw/` (the records, controls, numbers), `code/` (everything that produced them).

**Scope stamp.** This is a census of what the TRAINER'S OWN dataset objects hand to the loss for every window of the corpus refcv7 trained
on (4,369 clips, 746,946 windows) and of the held-out eval139 set (139 clips, 23,772 windows). It is a statement about LABEL/INPUT PRESENCE
and TIMING, not about model skill; no model, no GPU, no frame was used (frames are only decoded for 24 control windows). The train and eval
numbers are populations (every window counted), not samples, so they carry no sampling interval; for EVAL139 an episode-cluster bootstrap
over its 139 clips is given for the headline rows because that set is a sample of a wider population.

## 0. The answer to the PI's question, in six numbers

*"Why does the video say 'no GT label at this instant'?"* — because for **76.8 % of all training windows (and 77.3 % of eval139 windows)
the trainer's label block returns IGNORE for the tactical lateral/longitudinal labels and for all 22 goal cells.** That is not noise or
a missing file: every clip has exactly ONE label record anchored at raw 8.0 s, the trainer admits a window only when its NOW lies within
±2.0 s of the anchor (`v7_labels.window_in_band`), and a clip's NOW sweeps 0.7 → 19.0 s. Every one of the 4,369 clips contributes exactly
39–41 labelled windows of ~171, none contributes 0, and none contributes outside t_now ∈ [6.0, 10.0] s.

| # | headline | TRAIN (4,369 clips, 746,946 windows) | EVAL139 (139 clips, 23,772 windows) |
|---|---|---|---|
| 1 | windows carrying tactical lat/lon GT (and the 22-token goal set) | **173,409 = 23.22 %** | **5,404 = 22.73 %** (bootstrap 95 % CI 22.1–23.2 %; 3 stationary clips have none at all) |
| 2 | turn windows (\|Δyaw over the next 6 s\| ≥ 30°, full 6 s in the clip) that carry lateral GT | **30,931 of 88,238 = 35.1 %** (L 35.2 %, R 34.9 %); where present the label's side equals the realised side on 65.2 % | **814 of 2,602 = 31.3 %**; side agrees on 53.7 % |
| 3 | windows whose fed nav token is LEFT/RIGHT but **no turn starts within the next 6 s** (labels) | **205,292 = 74.3 % of the 276,434 L/R windows = 27.5 % of all windows** | 6,996 = 80.2 % of 8,726 L/R windows = 29.4 % of all |
| 4 | … and **none within the next 10 s** | **165,174 = 59.8 % of L/R windows = 22.1 % of all windows** | 5,620 = 64.4 % of L/R = 23.6 % of all |
| 5 | windows whose OWN realised max speed over [+2,+6] s **exceeds** the fed per-clip ceiling (windows with the full 6 s) | **57,535 of 576,848 = 10.0 %** (fed 30 km/h: 14.9 %; fed 120: 39.7 %) | 8.7 % |
| 6 | … whose realised max is **below the next-lower bin** (i.e. a lower bin would have contained it) | **41,985 = 7.3 %** (fed 50: 14.3 %; fed 100: 8.7 %) | 7.5 % |

Reading rules for rows 3–6: the nav token and the ceiling are each ONE value per clip fed on EVERY window (`window_ceiling_frac 1.0`), but both are
computed from the ego's future relative to the 8.0 s anchor. They are accurate near the anchor and decay with distance from it (T8): the
fed token matches a realised ≥30° turn on 27 % of turn windows at t_now < 2 s and on 73 % at t_now ∈ [6,8); the ceiling is exceeded on 15.6 % of
windows at t_now < 2 s and on 4.4 % at [6,8). Row 3/4 denominators are the L/R windows; the "% of all windows" figures divide by all windows.
Row 5/6 are over windows that have the full 6 s of future in the clip (77.2 % of windows; the last ~40 windows of every clip do not).

Other channels (T1): nav token 100 % · ceiling 100 % · SAM3 10 cm map **100.00 %** (TRAIN) / 98.56 % (EVAL139: the two eval clips with no GT file) · agent boxes
**96.36 %** (TRAIN; 2.27 % of windows are labelled-CLEAR) / 95.33 % (EVAL139) · 3-D cuboid on ≥1 agent 94.09 % / 92.89 % · full 6 s of future 77.23 % / 77.23 %.
The only channel with a structural, by-design hole is the TACTICAL one.

## 1. What was built, how, and what was NOT done

* **Tree.** Thor `/home/nvidia/refcv7_run/fec3a0dccf` — the launch tree. Blob check (raw bytes, `git hash-object` rule, recorded in `raw/thor_*/record_*.json`):
  `stack/scripts/refc_v3_train.py` `203b437f…` = `fec3a0d`; `stack/tanitad/data/v7_labels.py` `86f0c46e…` = `fec3a0d`; `stack/tanitad/data/v2_dataset.py` `ee61ea93…` = `fec3a0d`.
  The dev-box extract `C:/Users/Admin/ev7` is byte-identical to these after CRLF normalisation (checked: trainer bytes identical; the other two LF-normalised md5 equal Thor's).
  `config.json` md5 `e6512a01…` (Thor run dir == `D:/refcv7_eval_kit/ckpt/config.json`).
* **Dataset objects.** TRAIN: `build_v2_providers` (lazy: poses resident, payloads never read) → `V3Dataset` → `enable_clip_clock` + `assert_label_clock_true` (G3) →
  `enable_max_speed_v6` → `enable_nav_from_v7` → `enable_agent_join` → `enable_map_hires` → `enable_join3d`, in the order and with the arguments of `train()` (lines cited in `code/d1_census.py`).
  EVAL: the launch tree's own `refcv7_loader.build_eval_dataset`. The per-window label block of `V3Dataset.__getitem__` is then evaluated **without the frame decode**, calling the
  same functions (`ds._now_s`, `v7_labels.tactical_class_ids`, `v7_labels.tactical_goal_targets`, the dataset's nav/ceiling tables, `agent_join.lookup`, `zh_for_frame`, the map store's `n_frames`).
  Control **C2** compares that direct evaluation with the FULL `ds[i]` (frames decoded) on 24 windows over 6 clips per split: **0 mismatches** on every label field (TRAIN and EVAL).
* **Departures from `train()` (recorded in the records):** (1) `JoinFileReader(with_rates=False)` — rates are loss targets; the config's `n_windows_labelled` (719,739) is reproduced exactly, so presence is unaffected;
  (2) no model attributes, no VIS-1 sidecar, no calibration loader (none touches label presence); (3) the label-module negative policy is evaluated under **two states** (see F2) — an addition, not a departure.
* **Timing / ETA.** I did NOT run a separate 20-clip timing job: the build is dominated by streaming the two agent-join `.xz` files (the same bytes for any clip count). The eval139 run (139 clips) served as the timing run:
  build 121.8 s + census 0.5 s + C2 6.2 s = 130.9 s wall on Thor → predicted full-train wall ≈ 122 s + 4,369/139 × 0.5 s ≈ 140 s + the larger map/join loads. **Measured full train run: 199.4 s wall**
  (build 171.8 s, census 15.5 s, C2 7.6 s), started 10:41:23Z, **last Thor job ended 10:44:41Z (12:44 Berlin)**; no D1 process was left alive (checked by explicit `ps`). The 1,000-clip sample stage was not needed.
* **Not done / not claimed.** The map-GT flag is the trainer's own census state (file present and NOW frame in range); I did not measure how many of those frames are mostly "not seen" (255) cells.
  The `lat_peak_m` units question belongs to D2. No claim is made about what the video overlay code itself prints; only about the labels the trainer sees.
  The CPU lock was queued behind D3 for more than 30 min (`flock` is not FIFO); the lock was held only for my three jobs (eval 131 s, a first train attempt that died within seconds on a `v2_cache`-is-a-list bug in MY driver (fixed and smoke-tested on the dev box before re-queueing), the train run 199 s).

## 2. Controls — every one had to read a known value, and did

| id | control | reading | known value | verdict |
|---|---|---|---|---|
| **A** | route-following package (`2026-10-04-refcv7-route-following`): EVAL grid = 139 eps × 8 windows at (j+0.5)·n/8, episodes sorted by sha12; GT-turn = terminal heading of the slot-50→60 segment ≥ 30°, path ≥ 5 m | window-set digest **92e36a1a…92a1e == published**; GT-turn **107 (L 40 / R 67)**, straight 588, gentle 105, unclassified 312; `lat_v7` **IGNORE on 72** (turn-L 30, turn-R 42); label side L/S/R on present turn-L 5/5/0, turn-R 1/12/12; side agrees 17 of 35 = 48.6 % | 107 / 72 / 49 % (RESULT.md §1) | **REPRODUCED EXACTLY** |
| A′ | same grid, the literal "\|Δyaw over 6 s\| ≥ 30°" definition | 124 windows (L 43 / R 81), IGNORE on 83 (66.9 %), 41 with a label; overlap with A: 106 | — | **explains the difference**: A uses the heading of the LAST (5→6 s) segment, not the net 6-s heading change; 18 windows turn ≥30° net but end straighter |
| **B** | `config.json` stamps | TRAIN n_windows 746,946 · agent labelled 719,739 · labelled-clear 16,966 · map ok 746,946 · ceiling fed 746,946 · nav clips follow/left/right 2,752/779/838 · 4,369 clips · 22 clock-unverified clips · EVAL: 23,772 / 22,663 / 582 / 23,430 / 88-13-38 | identical values in `config.json` | **all 14 equal** (`raw/analysis/controls_summary.json`) |
| **C** | record anchor 8.0 s ⇒ in band exactly for t_now ∈ [6.0, 10.0] | the 4,369 train-cache records (and the 139 eval records) have `t0_s` = 8.0, distinct values [8.0] (the MM measured all 4,572); trainer in-band == literal `6.0 ≤ t_now ≤ 10.0` on **all 746,946 / 23,772 windows, 0 mismatches**; TRAIN in-band t_now ∈ [6.0, 10.0] exactly (largest outside-below 5.9999981, smallest outside-above 10.0000372); 19 windows lie within 1e-6 s of a band edge and the trainer agrees with the INCLUSIVE literal on all of them; lat/lon IGNORE ⇔ not in band | 40 windows/clip | **PASS** |
| **D** | sign convention + clock + units, against the LABEL FILE (independent builder, 100 Hz egomotion) | 980 TRAIN nav_30s turn entries lie fully inside the cached poses: label `dyaw_deg` vs the heading change of the CACHED poses between the entry's raw start and end — **Pearson 0.99999, sign agree 980/980, median \|diff\| 0.57°, p90 0.64°**; NAV_TURN_L has dyaw_deg > 0 on 100 %, NAV_TURN_R < 0 on 100 %; mean pose Δyaw for L +71.9°, R −59.3°; pose speed channel / finite-difference of xy = **1.0000** median (4,362 clips; p5–p95 0.9996–1.0004) ⇒ m/s | left = positive, m/s | **PASS** (left is POSITIVE: yaw CCW-positive, ego +y = left, `refb_labels.py:14-15`) |
| **D′** | a named left turn | eval clip `c45d6a86ff06`, window t = 71 (t_now 8.22 s): token NAV_TURN_L, Δyaw(6 s) **+87.1°**, lateral offset at +6 s **+29.8 m** (left), forward 14.8 m, heading of last segment +86.6°, `lat_v7` = TURN_L; the right-turn mirror `7e5dc4ad4e52`: −101.5°, −21.7 m | left ⇒ + | **PASS** |
| **E** | the ceiling's own definition: record `SPEED_BAND.v_hi_ms` vs MY max pose speed over [+2,+6] s at the window nearest t_now = 8.0 | 4,369 clips: median \|diff\| **0.0099 m/s**, p90 0.058, Pearson 0.99999, 100 % within 0.5 m/s (EVAL: 0.0101 / 0.048 / 0.99998) | ≈ 0 | **PASS** (the pose speed channel, clock and window are right) |
| **K1** | the run's own in-run eval log: the 128 windows `torch.randperm(N, seed 12345)[:8·16]` | census ⇒ goal cells (17 trainable tokens) **271 = 33.875/batch**, tactical rows **29 = 3.625/batch**, agent-labelled 121 (= 15.125·8), map-labelled 127 (= 15.875·8); the log carries (33.875, 3.625) at **all 101 evals** | exact | **EXACT MATCH** |
| **K2** | the run's own TRAIN log (1,008 logged steps, batch 16) | tactical rows/window logged **0.2357 [0.2295, 0.2426]** vs census 0.2322; agent-labelled logged 0.9621 vs census 0.9636; map 1.0000 vs 1.0000 | within CI | **PASS** |
| **X** | cross-environment mirror: the same driver on the dev box (Windows, local data copies) | EVAL139 table, 44 columns × 23,772 rows | bit-identical | **0 differing columns** |
| **C2** | direct label evaluation == full `ds[i]` | 24 windows / 6 clips (12 in-band) per split; lat, lon, goal y/w, nav, ceiling, agent label+count, map label, cuboid mask | 0 mismatches | **PASS** (train + eval) |

## 3. Tables (generated from `raw/analysis/numbers_{train,eval139}.json` by `code/d1_report.py`; no hand transcription)

Definitions used throughout: **in-band** = the trainer's `window_in_band`; **turn window** = the window's clip has the full 6 s of future and \|wrap(yaw[+6 s] − yaw[NOW])\| ≥ 30° (left = positive);
**fed ceiling** = the clip's `SPEED_BAND.v_hi_ms` binned by the containing-window rule onto {30,50,100,120} km/h; **exceeds** = realised max over [+2,+6] s (m/s·3.6) > the bin's km/h value;
**below the next-lower bin** = the realised max falls in a lower bin than the fed one (so `below` is 0 by construction for the 30 km/h bin); the **+k s rows** are `round(k/dt)` pose rows with `dt` the clip's own clock step (0.1007 s median).
"Labels" in T6 means `nav_30s.entries` of token NAV_TURN_L/R (the file also carries **NAV_FOLLOW_ROAD pseudo-entries**, `dyaw_deg` null, t 0→30 — 2,833 of them in the train blob; a naive reader that treats every entry as a turn finds a "turn start" at the anchor on every clip).
**Caveat on every label-based turn statistic (T6 rows "labels", T8 columns "turn start <= 6 s"):** the blob lists only turns that start AT OR AFTER the anchor (minimum `t_start_s` over all 2,419 train turn entries = 0.0, none negative; same for `manoeuvre_sequence`). A turn that starts before raw 8.0 s is invisible to the labels AND to the fed token, so for windows with t_now < 8 s the label-based "no turn starts within H" share is an upper bound; the pose-based rows ("realised", from the clip's own poses, windows with the full 6 s) are not affected and agree to within 2.4 points on TRAIN (74.3 % vs 71.9 %).

### T1. Channel coverage per window

| channel | TRAIN n | TRAIN % | EVAL139 n | EVAL139 % |
|---|---|---|---|---|
| windows | 746,946 | 100.00 % | 23,772 | 100.00 % |
| joined to a v8 record | 746,946 | 100.00 % | 23,260 | 97.85 % |
| tactical lat/lon GT (in band) | 173,409 | 23.22 % | 5,404 | 22.73 % |
| >=1 goal token SCORED (w>0), as the workers saw it | 173,409 | 23.22 % | 5,404 | 22.73 % |
| >=1 goal token POSITIVE | 173,409 | 23.22 % | 5,404 | 22.73 % |
| nav token fed (nav_valid) | 746,946 | 100.00 % | 23,772 | 100.00 % |
| nav token = left or right | 276,434 | 37.01 % | 8,726 | 36.71 % |
| max-speed ceiling fed | 746,946 | 100.00 % | 23,772 | 100.00 % |
| agent boxes labelled | 719,739 | 96.36 % | 22,663 | 95.33 % |
| agent labelled AND >=1 agent | 702,773 | 94.09 % | 22,081 | 92.89 % |
| agent labelled and EMPTY (clear) | 16,966 | 2.27 % | 582 | 2.45 % |
| 3-D cuboid on >=1 agent | 702,773 | 94.09 % | 22,081 | 92.89 % |
| SAM3 10 cm map label (trainer census 'ok') | 746,946 | 100.00 % | 23,430 | 98.56 % |
| full 6 s of future poses in the clip | 576,848 | 77.23 % | 18,359 | 77.23 % |
| tactical GT AND full 6 s future | 173,409 | 23.22 % | 5,404 | 22.73 % |

EVAL139 episode-cluster bootstrap 95 % CI (139 clips, B=2000): tactical GT 22.1-23.2 %; agent labelled 92.5-97.9 %; map label 96.4-100.0 %; full6 77.2-77.3 %

TRAIN: 746,946 windows over 4,369 clips (windows per clip min/median/max 161/171/179); clips with >=1 tactical window 4,369; tactical windows per clip min/median/max 39/40/41. EVAL139: 23,772 windows over 139 clips; clips with >=1 tactical window 136.

### T2. Tactical lateral / longitudinal classes (windows carrying the class)

**lateral (`lat_v7`)**

| class | TRAIN windows | % of all | % of in-band | TRAIN clips | EVAL windows | % of all | % of in-band |
|---|---|---|---|---|---|---|---|
| LANE_KEEP | 112,092 | 15.01 % | 64.6 % | 2,824 | 3,616 | 15.21 % | 66.9 % |
| LANE_CHANGE_L | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| LANE_CHANGE_R | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| ABORT_LC | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| NUDGE_L | 18,409 | 2.46 % | 10.6 % | 464 | 517 | 2.17 % | 9.6 % |
| NUDGE_R | 22,377 | 3.00 % | 12.9 % | 564 | 794 | 3.34 % | 14.7 % |
| TURN_L | 10,716 | 1.43 % | 6.2 % | 270 | 160 | 0.67 % | 3.0 % |
| TURN_R | 9,815 | 1.31 % | 5.7 % | 247 | 317 | 1.33 % | 5.9 % |

**longitudinal (`lon_v7`)**

| class | TRAIN windows | % of all | % of in-band | TRAIN clips | EVAL windows | % of all | % of in-band |
|---|---|---|---|---|---|---|---|
| FOLLOW | 6,232 | 0.83 % | 3.6 % | 157 | 357 | 1.50 % | 6.6 % |
| CRUISE | 47,484 | 6.36 % | 27.4 % | 1,197 | 1,668 | 7.02 % | 30.9 % |
| YIELD_MERGE | 0 | 0.00 % | 0.0 % | 0 | 0 | 0.00 % | 0.0 % |
| BRAKE_TO | 34,568 | 4.63 % | 19.9 % | 871 | 876 | 3.69 % | 16.2 % |
| CREEP | 4,729 | 0.63 % | 2.7 % | 119 | 119 | 0.50 % | 2.2 % |
| HOLD | 4,007 | 0.54 % | 2.3 % | 101 | 80 | 0.34 % | 1.5 % |
| ADAPT_SPEED_FOR_CURVE | 37,721 | 5.05 % | 21.8 % | 950 | 874 | 3.68 % | 16.2 % |
| ACCELERATE | 38,668 | 5.18 % | 22.3 % | 974 | 1,430 | 6.02 % | 26.5 % |

### T3. The 22 goal tokens: positives and scored cells (windows)

| token | TRAIN positive windows | % of all | pos. clips | TRAIN scored (as the workers saw it) | % of all | TRAIN scored (train-blob census state) | EVAL positive | EVAL scored |
|---|---|---|---|---|---|---|---|---|
| FOLLOW_LANE | 138,161 | 18.497 % | 3,481 | 173,409 | 23.22 % | 173,409 | 4,408 | 5,404 |
| TURN_L | 10,716 | 1.435 % | 270 | 173,409 | 23.22 % | 173,409 | 160 | 5,404 |
| TURN_R | 9,815 | 1.314 % | 247 | 173,409 | 23.22 % | 173,409 | 317 | 5,404 |
| YIELD_FOR_TURN_L | 753 | 0.101 % | 19 | 173,409 | 23.22 % | 173,409 | 40 | 5,404 |
| YIELD_FOR_TURN_R | 753 | 0.101 % | 19 | 173,409 | 23.22 % | 173,409 | 0 | 5,404 |
| YIELD | 23,468 | 3.142 % | 591 | 23,468 | 3.14 % | 23,468 | 678 | 678 |
| STOP_POINT | 11,542 | 1.545 % | 291 | 173,409 | 23.22 % | 173,409 | 279 | 5,404 |
| SPEED_BAND | 173,409 | 23.216 % | 4,369 | 173,409 | 23.22 % | 173,409 | 5,404 | 5,404 |
| CORRIDOR_OFFSET | 32,532 | 4.355 % | 820 | 32,532 | 4.36 % | 32,532 | 913 | 913 |
| EVADE_IN_CORRIDOR | 9,414 | 1.260 % | 237 | 10,168 | 1.36 % | 10,168 | 239 | 279 |
| OVERTAKE_VEHICLE | 754 | 0.101 % | 19 | 139,152 | 18.63 % | 139,152 | 40 | 4,448 |
| MERGE | 3,097 | 0.415 % | 78 | 141,258 | 18.91 % | 141,258 | 200 | 4,608 |
| GAP_TARGET | 14,322 | 1.917 % | 361 | 14,322 | 1.92 % | 14,322 | 436 | 436 |
| REACT_ON_ONCOMING | 12,949 | 1.734 % | 326 | 12,949 | 1.73 % | 12,949 | 397 | 397 |
| TAKE_EXIT_L | 833 | 0.112 % | 21 | 14,897 | 1.99 % | 14,897 | 0 | 515 |
| TAKE_EXIT_R | 4,964 | 0.665 % | 125 | 16,276 | 2.18 % | 16,276 | 198 | 358 |
| TRAFFIC_LIGHT_REACT | 673 | 0.090 % | 17 | 29,701 | 3.98 % | 29,701 | 0 | 1,033 |
| TRAFFIC_LIGHT_REACT_RED | 14,022 | 1.877 % | 353 | 29,701 | 3.98 % | 29,701 | 317 | 1,033 |
| TRAFFIC_LIGHT_REACT_YELLOW | 876 | 0.117 % | 22 | 29,701 | 3.98 % | 29,701 | 120 | 1,033 |
| TRAFFIC_LIGHT_REACT_GREEN | 14,130 | 1.892 % | 356 | 29,701 | 3.98 % | 29,701 | 596 | 1,033 |
| LANE_CHANGE_L | 872 | 0.117 % | 22 | 173,409 | 23.22 % | 12,145 | 0 | 5,404 |
| LANE_CHANGE_R | 557 | 0.075 % | 14 | 11,244 | 1.51 % | 11,244 | 40 | 357 |

Two label-module states are tabulated for TRAIN (see FINDINGS): windows whose scored-cell count differs between them: 161,264; cells differing by token: {"LANE_CHANGE_L": 161264}.

### T4. In-band fraction as a function of t_now (1-s bins over the clip)

| t_now bin (s) | TRAIN windows | TRAIN in-band | TRAIN % | EVAL windows | EVAL in-band | EVAL % |
|---|---|---|---|---|---|---|
| [0,1) | 1,760 | 0 | 0.0 % | 58 | 0 | 0.0 % |
| [1,2) | 43,097 | 0 | 0.0 % | 1,374 | 0 | 0.0 % |
| [2,3) | 43,318 | 0 | 0.0 % | 1,380 | 0 | 0.0 % |
| [3,4) | 43,387 | 0 | 0.0 % | 1,378 | 0 | 0.0 % |
| [4,5) | 43,387 | 0 | 0.0 % | 1,376 | 0 | 0.0 % |
| [5,6) | 43,313 | 0 | 0.0 % | 1,381 | 0 | 0.0 % |
| [6,7) | 43,339 | 43,339 | 100.0 % | 1,382 | 1,352 | 97.8 % |
| [7,8) | 43,362 | 43,362 | 100.0 % | 1,381 | 1,351 | 97.8 % |
| [8,9) | 43,336 | 43,336 | 100.0 % | 1,384 | 1,354 | 97.8 % |
| [9,10) | 43,363 | 43,363 | 100.0 % | 1,377 | 1,347 | 97.8 % |
| [10,11) | 43,369 | 9 | 0.0 % | 1,382 | 0 | 0.0 % |
| [11,12) | 43,494 | 0 | 0.0 % | 1,382 | 0 | 0.0 % |
| [12,13) | 43,512 | 0 | 0.0 % | 1,381 | 0 | 0.0 % |
| [13,14) | 43,531 | 0 | 0.0 % | 1,386 | 0 | 0.0 % |
| [14,15) | 43,521 | 0 | 0.0 % | 1,383 | 0 | 0.0 % |
| [15,16) | 43,476 | 0 | 0.0 % | 1,385 | 0 | 0.0 % |
| [16,17) | 43,386 | 0 | 0.0 % | 1,381 | 0 | 0.0 % |
| [17,18) | 43,271 | 0 | 0.0 % | 1,378 | 0 | 0.0 % |
| [18,19) | 7,723 | 0 | 0.0 % | 243 | 0 | 0.0 % |
| [19,20) | 1 | 0 | 0.0 % | 0 | 0 | n/a % |

### T5. Turn windows (|dyaw over [0,6] s| >= 30 deg, windows with the full 6 s in the clip) and the tactical lateral label

"Side agrees where present": the label's side (NUDGE_x / TURN_x / LANE_CHANGE_x -> x, LANE_KEEP -> straight) equals the sign of the realised 6-s heading change; LANE_KEEP on a turn window counts as DISagreement.

| turn windows | TRAIN n | lat GT present | % present | side agrees where present | EVAL n | lat GT present | % present | side agrees where present |
|---|---|---|---|---|---|---|---|---|
| all_turn | 88,238 | 30,931 | 35.1 % | 65.2 % | 2,602 | 814 | 31.3 % | 53.7 % |
| left | 43,544 | 15,329 | 35.2 % | 65.5 % | 929 | 252 | 27.1 % | 49.6 % |
| right | 44,694 | 15,602 | 34.9 % | 64.9 % | 1,673 | 562 | 33.6 % | 55.5 % |

10-30 deg: TRAIN n 86,122, lat GT present 27.7 %; EVAL n 2,573, 27.9 %

< 10 deg: TRAIN n 402,488, lat GT present 29.5 %; EVAL n 13,184, 29.4 %

Route-following package definition (terminal heading of the slot-50 -> 60 segment, >= 30 deg, path >= 5 m) on ALL windows: TRAIN GT-turn 81,703 (L 40,109, R 41,594), lat IGNORE on 64.4 %; EVAL139 GT-turn 2,317 (L 849, R 1,468), lat IGNORE on 69.1 %.

### T6. The fed nav token against where a turn actually is

| quantity | TRAIN | EVAL139 |
|---|---|---|
| windows with nav token left / right / follow | 133,190 / 143,244 / 470,512 | 2,229 / 6,497 / 15,046 |
| clips with nav token left / right / follow | 779 / 838 / 2752 | 13 / 38 / 88 |
| L/R windows (denominator) | 276,434 | 8,726 |
| L/R windows, NO turn START within the next 6 s (labels) | 205,292 (74.3 %) | 6,996 (80.2 %) |
| ... and none IN PROGRESS either (6 s) | 172,389 (62.4 %) | 6,329 (72.5 %) |
| L/R windows, NO turn START within the next 10 s (labels) | 165,174 (59.8 %) | 5,620 (64.4 %) |
| ... and none IN PROGRESS either (10 s) | 134,509 (48.7 %) | 5,013 (57.4 %) |
| L/R windows with 6 s of future in the clip: NO realised >= 30 deg turn in 6 s (poses) | 153,466 of 213,494 (71.9 %) | 5,434 of 6,738 (80.6 %) |
| ... and not even 10 deg | 54.3 % | 62.6 % |
| realised >= 30 deg turn in the next 6 s: windows | 88,238 | 2,602 |
| ... of which the fed token is FOLLOW | 28,210 (32.0 %) | 1,298 (49.9 %) |
| ... of which the fed token matches the turn side | 53,866 (61.0 %) | 1,241 (47.7 %) |
| ... of which the fed token is the WRONG side | 6,162 | 63 |
| median / p5-p95 seconds to the next labelled turn start (L/R windows that have one) | 9.8 (1.0-27.7) | 11.7 (1.4-28.6) |

### T7. The per-clip speed ceiling against the window's own realised speed ([+2,+6] s ahead; windows with the full 6 s)

| fed ceiling | TRAIN windows | realised max EXCEEDS ceiling | realised bin BELOW fed bin | same bin | EVAL windows | exceeds | below | same |
|---|---|---|---|---|---|---|---|---|
| all fed bins | 576,848 | 57,535 (10.0 %) | 41,985 (7.3 %) | 488,458 (84.7 %) | 18,359 | 8.7 % | 7.5 % | 84.6 % |
| fed 30 km/h | 218,540 | 32,583 (14.9 %) | 0 (0.0 %) | 185,957 (85.1 %) | 6,210 | 15.7 % | 0.0 % | 84.3 % |
| fed 50 km/h | 204,826 | 12,801 (6.2 %) | 29,369 (14.3 %) | 162,656 (79.4 %) | 6,475 | 5.7 % | 13.8 % | 80.6 % |
| fed 100 km/h | 125,418 | 1,021 (0.8 %) | 10,961 (8.7 %) | 113,436 (90.4 %) | 4,752 | 2.3 % | 8.8 % | 88.8 % |
| fed 120 km/h | 28,064 | 11,130 (39.7 %) | 1,655 (5.9 %) | 26,409 (94.1 %) | 922 | 15.1 % | 7.3 % | 92.7 % |

Margin: realised max exceeds the fed ceiling by > 5 km/h on 4.9 % of TRAIN windows (28,400), by > 10 km/h on 2.1 % (11,893); it is more than 20 km/h BELOW the fed ceiling on 29.0 % (167,369); median realised-minus-ceiling -12.4 km/h (EVAL139: >5 km/h 3.4 %, >10 km/h 1.4 %).

In-band windows only (the span the ceiling's own definition describes): TRAIN n 173,409: exceeds 4.7 %, below 2.7 %, same 94.5 %; EVAL n 5,404: exceeds 3.5 %, below 4.0 %, same 93.2 %. Current speed v0 above the fed ceiling: TRAIN 7.6 %, EVAL 6.8 % of windows.


### T8. The per-clip constants against t_now (2-s bins; the nav token and the ceiling are ONE value per clip on every window)


| t_now bin | TRAIN windows | tactical GT | nav = L/R | L/R with a turn start <= 6 s (labels) | L/R with NO >=30 deg turn in 6 s (poses) | realised turns: token matches | realised turns: token FOLLOW | ceiling exceeded | realised bin below fed | same bin | EVAL L/R with turn start <= 6 s | EVAL ceiling exceeded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [0,2) | 44,857 | 0.0 % | 36.9 % | 0.0 % | 85.3 % | 26.9 % | 52.9 % | 15.6 % | 13.0 % | 73.2 % | 0.0 % | 14.2 % |
| [2,4) | 86,705 | 0.0 % | 37.0 % | 24.8 % | 79.6 % | 42.2 % | 43.2 % | 13.0 % | 11.5 % | 77.3 % | 10.6 % | 11.9 % |
| [4,6) | 86,700 | 0.0 % | 37.0 % | 36.4 % | 71.2 % | 62.7 % | 31.4 % | 8.8 % | 8.5 % | 84.6 % | 22.6 % | 8.7 % |
| [6,8) | 86,701 | 100.0 % | 37.0 % | 44.0 % | 64.5 % | 73.0 % | 25.2 % | 4.4 % | 3.2 % | 94.3 % | 30.4 % | 3.0 % |
| [8,10) | 86,699 | 100.0 % | 37.0 % | 27.7 % | 64.5 % | 70.4 % | 27.4 % | 5.1 % | 2.2 % | 94.7 % | 28.4 % | 3.8 % |
| [10,12) | 86,863 | 0.0 % | 37.0 % | 23.6 % | 70.1 % | 64.1 % | 30.7 % | 10.4 % | 5.9 % | 85.7 % | 21.7 % | 7.5 % |
| [12,14) | 87,043 | 0.0 % | 37.0 % | 22.0 % | 73.9 % | 61.2 % | 29.1 % | 14.4 % | 9.0 % | 78.5 % | 15.4 % | 13.2 % |
| [14,16) | 86,997 | 0.0 % | 37.0 % | 21.4 % | 76.0 % | 59.8 % | 29.8 % | 15.8 % | 10.5 % | 75.5 % | 16.4 % | 16.5 % |
| [16,18) | 86,657 | 0.0 % | 37.0 % | 19.8 % | n/a % | n/a % | n/a % | n/a % | n/a % | n/a % | 22.6 % | n/a % |


**In-band windows, lateral label against the window's own realised heading, by distance from the anchor**


| window distance from the 8.0 s anchor | TRAIN in-band | TRAIN turn windows | label = TURN_x | label = LANE_KEEP | label side == realised side | EVAL turn windows | EVAL side agree |
|---|---|---|---|---|---|---|---|
| abs(t_now-8) in [0,1] | 86,689 | 15,789 | 57.7 % | 34.6 % | 64.8 % | 410 | 53.9 % |
| abs(t_now-8) in [1,2] | 86,720 | 15,142 | 55.2 % | 32.0 % | 65.6 % | 404 | 53.5 % |

## 4. Findings (read these before the tables are quoted)

**F1 — The tactical label is a 4-second spotlight, not a channel.** One record per clip, anchor 8.0 s, ±2.0 s admission ⇒ exactly the windows with t_now ∈ [6.0, 10.0] s (39–41 per clip, 23.2 % of the corpus,
identical in every clip; T4). The goal set (22 cells) and `lat_v7`/`lon_v7` are IGNORE everywhere else, so **76.8 % of training windows teach the tactical and goal heads nothing**, and the head is
trained on one narrow slice of every clip (the 8.0 ± 2 s part), a slice that is also where the fed nav token and ceiling are most accurate (T8). Inside the band the label's own interval is
a fixed raw [10,14] s while the window's own tactical horizon is [t_now+2, t_now+6] s, so the temporal overlap is 100 % at t_now = 8 and 50 % at the edges (mean 75 %, analytic from the rule).
The label's agreement with what the ego does is modest even inside the band: on in-band turn windows (|Δyaw(6 s)| ≥ 30°) the lateral label is TURN_x on 57.7 % / 55.2 % (|t_now−8| < 1 / 1–2 s), LANE_KEEP on 34.6 % / 32.0 %, and its side equals the realised side on 64.8 % / 65.6 % (EVAL139: 53.9 % / 53.5 %, n = 410 / 404).

**F2 — NEW (MEASURED, with the run's own log as the proof): the 22-token goal head trained LANE_CHANGE_L as a supervised negative on 100 % of tactical windows, while its `pos_weight` and `config.json` census assumed 7.0 %.**
`v7_labels._MEASURED_GEOMETRY_TOKENS` is MODULE STATE refilled by every `load_v7_labels` call. `train()` loads the TRAIN blob (line 8393; the config census/mask/`pos_weight` are computed here),
then the EVAL blob (line 8736) — which has no CoT-sourced `LANE_CHANGE_L` — and only then creates the training `DataLoader` (line 8957, default fork). So the workers see the eval-blob set
(8 tokens, `LANE_CHANGE_L` included as "geometry-emitted"), under which every absent `LANE_CHANGE_L` cell on an in-band window is a weight-1 negative. Census state: 12,145 scored windows (7.0 % of in-band);
as the workers saw it: 173,409 (100 %); positives 872 windows / 22 clips (0.50 % of in-band windows). `pos_weight[LANE_CHANGE_L]` = 12.6 (= 290 entailed negatives / 23 positives, config census) while the supervised
negatives are ≈ 4,350 clips: the weight is ~16× too small for the cells actually supervised. **Proof which state the workers had:** trainable goal cells per tactical row — run log (1,008 steps) **9.643 [9.594, 9.692]**
vs census state B (eval-blob) **9.605** vs state A (train-blob) **8.675**; the log excludes A. The eval-side numbers are unaffected (eval139 always ran in state B). Cost on the model is probably small
(a rare token trained as "never") but it is an undeclared difference between the stamped recipe and the trained one, of exactly the `declared-vs-built` family. Two other tokens are CoT in the train blob and absent in the eval blob's CoT set
(`TAKE_EXIT_L`, `TRAFFIC_LIGHT_REACT`); by the set arithmetic (geometry set = tokens that are neither CoT-backed nor in the frozen `NEEDS_PERCEPTION` set) they stay out of the geometry set — only `LANE_CHANGE_L` flips (measured: the two sets differ by exactly that one token). (Columns `n_goal_scored` vs `n_goal_scored_censusstate` in the truth table carry both.)

**F3 — The nav token is an oracle that is correct only near the anchor.** One token per clip (2,752 follow / 779 left / 838 right clips) is fed on all 171 windows. On TRAIN L/R windows, no labelled turn starts within 6 s on 74.3 %
and none within 10 s on 59.8 % (T6); from the poses, 71.9 % of L/R windows (with full future) do not even realise a 30° turn in the next 6 s, 54.3 % not 10° (EVAL139: 80.2 % / 64.4 % / 80.6 % / 62.6 %).
The median time from a window to the next labelled turn start is 9.8 s (p5–p95 1.0–27.7 s). The reverse failure: of 88,238 realised ≥30° turns, 32.0 % are fed FOLLOW and 7.0 % (6,162) the WRONG side;
only 61.0 % get the matching token (EVAL139: 49.9 % FOLLOW, 47.7 % matching). By t_now: the matching share is 26.9 % at t_now < 2 s, 62.7 % at [4,6), 73.0 % at [6,8) (T8). Part of the mismatch is structural: the token (and every `nav_30s` entry) is defined from the anchor onward, so a window before 8.0 s whose own next 6 s contain a turn that begins before 8.0 s is fed a token about a LATER turn (or FOLLOW). The time-localised nav IS in the file
(`nav_30s.entries`: `t_start_s`, `t_end_s`, `dyaw_deg`, arc-length `distance_m`) — refcv7 fed only the bare token. `ttn_s` / `ttn_side` in the truth table are that quantity per window.

**F4 — The ceiling is one number per clip and its error is a function of distance from the anchor.** 10.0 % of full-future windows realise a max speed above the fed ceiling (4.9 % by > 5 km/h, 2.1 % by > 10 km/h), 7.3 % fall a bin below it,
and the median realised-minus-ceiling is −12.4 km/h (T7). It is 4.7 % exceed / 2.7 % below inside the band, 15.6 % exceed at t_now < 2 s and 15.8 % at [14,16) (T8). The 120 km/h bin exceeds on 39.7 % of its windows, and 94 % of those exceedances (10,470 of 11,130) come from the 85 train clips whose own `v_hi` is above 120 km/h
and is clamped to that bin (the config's `n_over_ceiling` is 86 over the full 4,572 blob). 7.6 % of ALL windows have a current speed v0 already above the fed ceiling; 7.3 % have v0 < 1 m/s.

**F5 — Dead classes.** Zero windows in TRAIN or EVAL139 carry `LANE_CHANGE_L`, `LANE_CHANGE_R`, `ABORT_LC` (3 of the 8 lateral classes) or `YIELD_MERGE` (1 of the 8 longitudinal classes): the CE heads have permanently empty classes. Of the lateral GT that exists, 64.6 % is LANE_KEEP, 10.6 + 12.9 % NUDGE_L/R, 6.2 + 5.7 % TURN_L/R (T2).

**F6 — EVAL139 cannot score several goal tokens.** Positives in eval139 (windows / clips): `YIELD_FOR_TURN_R` 0 / 0, `TAKE_EXIT_L` 0 / 0, `TRAFFIC_LIGHT_REACT` 0 / 0, `LANE_CHANGE_L` 0 / 0; one clip each for `YIELD_FOR_TURN_L`, `OVERTAKE_VEHICLE`, `LANE_CHANGE_R`; 3–5 clips for `TURN_L` (4), `MERGE` (5), `TRAFFIC_LIGHT_REACT_YELLOW` (3), `EVADE_IN_CORRIDOR` (6), `STOP_POINT` (7), `TURN_R` (8). A per-token metric on eval139 is a few clips wide (T3).

**F7 — EVAL139 differs from TRAIN in coverage only through three stationary clips and the clock fallback.** Three eval clips (`081b986f8888`, `2aa810802777`, `3db625a5f941`) have no measured clock and are excluded from tactical scoring by G3 (512 windows); two of them also have no SAM3 map file (342 windows,
the whole map shortfall 23,772 → 23,430). In TRAIN, 22 clips (0.50 %) are clocked on the pose-dt/nominal fallback (grid_start 0.0 instead of the measured median 0.113 s) — their t_now is shifted by up to ~0.11 s relative to a measured clock (18 of the 19 band-edge windows, those within 1e-6 s of 6.0/10.0, sit in `nominal_dt` fallback clips, 1 in a sidecar clip).

**F8 — Agent boxes.** 96.36 % of windows are labelled (719,739), 2.27 % labelled-clear (16,966), 3.64 % NO_LABEL (27,207: the join's label span ends before the clip does). EVAL139: 95.33 % / 2.45 %. They are NOT a trailing block: 344 of 4,369 clips (7.9 %) hold all of them — 12 clips have no agent label on any window, 124 clips lose only a trailing run, 149 only a leading run, the rest have interior gaps; the unlabelled windows' t_now spreads over 0.9–18.2 s (median 9.3 s). 19.8 % of the unlabelled windows are in the tactical band (23.2 % of all windows are), i.e. the two holes are almost independent.

**What this implies for refcv8 (ESTIMATED reasoning, not a result):** the cheapest levers sit in the data path, not the model: (i) label EVERY window with a window-anchored tactical/goal record (the geometric emitter already exists and the
poses are all there — the truth table's `dyaw6_deg`, `lat6_m`, `vmax26`, `stop_0_6` columns are the first-order version), which multiplies tactical supervision ~4.3× and removes the 8.0 ± 2 s concentration; (ii) feed a time-localised nav
(`ttn_s`, side, distance) instead of the bare per-clip token; (iii) replace the per-clip ceiling by a per-window one or restrict its feed to in-band windows; (iv) pin the label-module state (F2) so the stamped and trained negative policies are one thing.
Each is a separate arm with its own pre-registered bar (D2–D5 own the label-quality side).

## 5. Truth-table schema (`tables/label_truth_{train,eval139}.npz`, one row per dataset window, row == `ds` index; `tables/label_truth_columns.json` has the same text)

`window_index` · `clip_sha12` (S12) · `clip_ix` (→ `tables/label_truth_*_clips.json`: per clip sha12, T, dt, grid_start, clock source, nav token/args, strata, SAM3 n_frames, fed ceiling) · `t` · `t_now_s` · `has_record` · `in_band` ·
`lat_v7` · `lon_v7` (−100 = IGNORE; class names in `raw/thor_*/record_*.json → vocab`) · `n_goal_scored` / `n_goal_scored_censusstate` · `n_goal_pos` · `goal_y_bits` / `goal_w_bits` / `goal_w_bits_censusstate` (bit i = goal token i) ·
`nav_cmd` (0 follow / 1 left / 2 right) · `nav_valid` · `vmax_ms` · `vmax_valid` · `vmax_bin` · `agent_labelled` · `n_agents` · `box3d_labelled` · `n_box3d` · `map_label` · `n_future_valid` (≤ 60) · `full6` ·
ego-future geometry (label side): `v0 v2 v4 v6` · `dyaw6_deg` (LEFT = POSITIVE) · `lat6_m` (+ = left, NOW ego frame) · `fwd6_m` · `vmax26` · `vmin26` · `vmin06` · `stop_0_6` ·
route-package geometry: `theta_slot_deg` · `path_len_slot_m` · labelled-turn timing: `ttn_s` · `ttn_dyaw_deg` · `ttn_side` · `in_turn_now` · `turn_now_dyaw_deg` · `n_turns_in_labels`.
Conventions: geometry columns are NaN unless the clip has the full 6 s after NOW (`full6`); a "+k s" row is `round(k/dt)` rows with `dt` the clip's own clock step; `stop_0_6` = min speed over [0,+6] s < 0.5 m/s.

## 6. Reproduce

```
# Thor (venv /home/nvidia/venvs/tanitad-train, flock /home/nvidia/refcv8_audit/cpu.lock, OMP_NUM_THREADS=4)
python d1_census.py --split train --repo /home/nvidia/refcv7_run/fec3a0dccf --kit /home/nvidia \
       --config /home/nvidia/refcv7_run/runs/refcv7-r101-s0/config.json --out out --tag train      # 199 s
python d1_census.py --split eval  ... --tag eval139                                                  # 131 s
# dev box (numpy): code/d1_analyze.py --dir raw/thor_train --tags train ; code/d1_report.py raw/analysis ;
#                  code/d1_controls_vs_log.py ; code/d1_controls_summary.py ; code/d1_finalize_tables.py .
```
Dev-box mirror (eval139, same driver, local data): `raw/devbox_eval/` (bit-identical, control X). Smoke of the train path on the 139-clip dev-box train subset: `raw/devbox_train_smoke/` (map flag forced 0 there — no train SAM3 GT on the dev box; not a result).

## 7. Evidence classes

MEASURED (artifact paths above): every count, fraction, control and the F2 proof. INHERITED and NOT re-verified: the claim that the DataLoader forks on Linux (the worker state in F2 is *proven* by the log, not by that claim);
the MM's "all 4,572 records have t0 = 8.0" (I re-derived it for the 4,369 train-cache records: distinct `t0_s` = [8.0]). ESTIMATED: the "what this implies" paragraph. UNVERIFIED: how much of each map frame is "seen" cells.
