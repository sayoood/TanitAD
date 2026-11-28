# D2 / RESULT — are the labels refcv7 trained on CORRECT?

Stream D2, refcv8 data audit, 2026-10-04.  Companion: `SEMANTICS.md` (field-by-field definition audit, file:line).  All artifacts in `raw/`.
Evidence class everywhere: **MEASURED** (D2, artifact named), unless marked READ / UNVERIFIED.  Tier: these are **label-validity** numbers, not driving performance (no T0/T1 claim).
Label files: train md5 `b45377a1f25263b5c0f3d318c126b1ac` (4,572 records), eval md5 `eefc38d1453bd1c73802d44d45affced` (147 records).  Poses: the exact `_v2manifest.pt` the trainer's loader builds (`v2_dataset._scan_meta`) — train md5 `3c9f8bc8583938ff60c57d4d97fcd8c0` (identical on Thor and locally, 4,369 clips), eval139 (139 clips).  Join: `stable_episode_id(clip_id)` (asserted equal to the manifest's `episode_uid` for every clip).  Records without a pose track: **203 train, 8 eval** (clips outside the refcv7 cache — D1's coverage domain); they are not scored.

## HEADLINE

| channel | verdict | the number (n, source) |
|---|---|---|
| `a_tac.lat` TURN_L / TURN_R | **TRUSTWORTHY** | 87.0 % / 85.4 % corroborated by an independent ≥ 30° heading change (train 235/270, 211/247); side never inverted (0 TURN_L→R, 0 TURN_R→L); eval139 4/5, 7/8 |
| `a_tac.lat` LANE_KEEP | **TRUSTWORTHY** | 91.6 % (train 2,587/2,824), 92.5 % eval139 (86/93); the misses are 109 independent turns ≥ 30° at wide radius (road bends the `is_turn` gate refuses) |
| `a_tac.lat` NUDGE_L / NUDGE_R | **WRONG (definition)** | only 10.3 % / 12.1 % corroborated (train 48/464, 68/564); 89.7 % have a heading that never returns (a curve); eval139 3/13, 2/20.  1,028 labels = **23.5 % of the tactical lateral mass** |
| `a_tac.lon` CRUISE, BRAKE_TO, FOLLOW | **TRUSTWORTHY on [0,6]** | 100 % / 96.2 % / 100 % (train, on the span the builder used) |
| `a_tac.lon` ACCELERATE | **NOISY** | 81.6 % on [0,6], 58.9 % on the record's own band [2,6]; threshold is +1.0 not the documented +1.5 (18.0 % of records) |
| `a_tac.lon` HOLD, CREEP | **NOISY, window-dependent** | HOLD 98.0 % / CREEP 68.1 % on [2,6]; 44.5 % / 23.5 % on [0,6] |
| `a_tac.lon` ADAPT_SPEED_FOR_CURVE | valid curve flag, **hides the speed class** | 99.2 % have ≥ 20° heading change; underneath 272 cruise / 486 accelerate / 192 brake (train, [0,6]) |
| `lat_peak_m` | **settled: mis-NAMED, not broken** | the 20-s clip-start-frame peak lateral displacement in metres; reproduces at corr 0.9995; not the NUDGE quantity; refcv7 never read it |
| `g_tac.goals` TURN_*, STOP_POINT | **TRUSTWORTHY** | TURN 87/85 % (above); STOP_POINT 291/291 positives have an independent stop |
| `g_tac.goals` VLM tokens (15 tokens, 3,472 annotations) | **UNVERIFIABLE by geometry; noisy by provenance** | all `vlm-cot`; 77–100 % untimed; 30–100 % `disputed`; the VLM's lateral side is right 69.6 % in its own window (§3) |
| `nav_command` token | **a clip-level route label, NOT a per-window command** | turn starts median 7.3 s after the anchor; only 24.7 % of TURN-token windows see a same-side turn begin in their own next 6 s (§5) |
| `speed_max_input` / fed 4-way ceiling | **exact near the anchor, stale elsewhere** | R² of the window's own realised max: 0.988 for \|NOW−t0\| ≤ 2 s (23.4 % of windows), 0.80 at ≥ 6 s; over all windows `v_now` alone beats the fed ceiling (R² 0.899 vs 0.889, train) (§4) |
| clock / `t0_s` / `bands` / `within_m` / `SPEED_BAND` values | **EXACT** | speed reproduced to 0.0017 m/s median on the label clock; `v_hi` to 0.005 m/s on 4,369 clips |
| VLM-vs-geometry (the 36 % "conflicts") | **neither source is the noisier one on the lateral axis** | 1,519 conflicts: VLM right 51.1 %, ours right 51.0 % (each on its own window); ours right on TURN rows 90–94 %, VLM right on NUDGE rows 57–58 % (§3) |

## 1. `lat_peak_m` verdict

`a_tac.lat_args.lat_peak_m` is the **signed peak lateral displacement, in metres (left = +), over the clip's first 20 s, measured in the heading frame of the clip's first sample** — written by the v7.1 patch (`build_v71.py:68-86,128-134`), not by the emitter.  It is **not** a lateral offset of a nudge, it is **not** anchor-relative, and the data is **not broken**: recomputing it from cache poses reproduces the stored value at corr 0.99953 (train n=4,369) / 0.99959 (eval139 n=139), sign equal 98.9 % / 100 %, median |err| 0.59 / 0.53 m; the anchor-frame [0,6] s and band-start [2,6] s hypotheses reproduce it at corr only 0.59/0.44 (train).  Median |x| 19.6 m, max 305.3 m, 70.2 % > 5 m are the road's own geometry over 20 s.  Applying the documented 1.0 m rule to it re-labels 85.0 % of LANE_KEEP and 0.3 % of NUDGE are below 1 m.  Defects are in the **name and docstrings** (`ego_manoeuvre.py:103-112`, `build_v71.py:20-22`; a different quantity under the same name exists in `ego_manoeuvre.Manoeuvre`).  The claim was already withdrawn in `V72_MANIFEST.field_semantics`; the "ship the true deciding quantity" work item is still open.  Exposure in refcv7: **zero** (no consumer in the trainer path).  (`raw/agree_train.json → recompute_checks.lat_peak_m`)

## 2. Independent agreement of `a_tac.lat` / `a_tac.lon`

**Independent derivation (`raw/d2_lib.py`, literals stated before any data was read).**  Poses are interpolated on the trainer's clock; window = the record's own tactical band [t0+2, t0+6] with t0 = 8.0 (NOW = anchor).  Lateral: *turn* = |heading change| ≥ 30° inside the window (variant B: ≥ 30° inside any 3 s); otherwise the lateral excursion ε = peak |lateral offset| after removing the constant-curvature arc through the window's net heading change; |net dyaw| ≥ 15° → BEND; ε < 0.3 m → LANE_KEEP; 0.3 ≤ ε ≤ 1.5 m **and returning** (|ε_end| < ½ε) → NUDGE; ε > 1.5 m → SHIFT (lane-change scale); else LANE_KEEP.  Longitudinal: stop = v < 0.5 m/s, crawl ≤ 2 m/s, brake Δv ≤ −1.5, accelerate Δv ≥ +1.5 (Δv over the same window).  Scoring maps builder NUDGE_x to {NUDGE_x, SHIFT_x}, LANE_KEEP to {LANE_KEEP, BEND_x}, FOLLOW to {BRAKE, CRUISE}; ADAPT_SPEED_FOR_CURVE is scored separately (a curve flag).

**Disclosure — my classifier was revised once.**  The first run (rule "A0": every non-returning 0.3–1.5 m excursion and every ≥ 15° net-yaw excursion called SHIFT) gave **61.2 % on eval139 (85/139) and 62.9 % on train**; the disagreements were road-curvature residuals of my own model, not label errors.  The BEND class and the 0.3–1.5 m-not-returning → LANE_KEEP rule were added after looking at eval139 disagreements (so the eval139 figure is partly tuned; **the train figure was computed only after the final rule and is the cleaner number**).  A0 is kept in the code (`variant="A0"`) and its matrices are in `raw/agree_*.json`.

### 2a. Analytic controls (`raw/controls_result.json`; synthetic 10 Hz tracks, fine-integrated)

| track (known geometry) | expected | independent | **builder** `tactical_actions` (object under test) |
|---|---|---|---|
| arc L, R=20 m, v=5 m/s (band dyaw 57.296°) / mirror R | TURN_L / TURN_R | TURN_L / TURN_R, dyaw error 0.000° | TURN_L / TURN_R (+ADAPT_SPEED_FOR_CURVE) |
| straight, 14 m/s | LANE_KEEP, CRUISE | LANE_KEEP, CRUISE | LANE_KEEP, CRUISE |
| gentle bend R=300 m (10.69° in band) | LANE_KEEP | LANE_KEEP | **NUDGE_L**  ← a bend is called a nudge |
| bend R=160 m (20.05°) | BEND_L (builder: LANE_KEEP) | BEND_L (caught a rule-order error in my first spec; fixed, re-run) | LANE_KEEP |
| bend R=100 m (32.09°) | TURN_L under rule A, LANE_KEEP under B | TURN_L / BEND_L | LANE_KEEP (R-gate) |
| 1.0 m out-and-back at 14 m/s (L and R) | NUDGE_L / NUDGE_R | NUDGE_L / NUDGE_R | **LANE_KEEP** ← a real nudge is missed (needs peak yaw ≥ 5°) |
| 3.5 m pulse / 3.5 m lane change | SHIFT | SHIFT | NUDGE (no upper bound) |
| 0.1 m wobble | LANE_KEEP | LANE_KEEP | LANE_KEEP |
| Δv +0.5 m/s² (Δv = +2.0) / −0.5 / decel-to-stop / v=0 / v=1.0 | ACCEL / BRAKE / BRAKE / HOLD / CREEP | all 17/17 correct | ACCELERATE / BRAKE_TO / BRAKE_TO / HOLD / CREEP |
| Δv = +1.0 m/s over the band | CRUISE (documented ACCEL bar is +1.5) | CRUISE | **ACCELERATE** ← the +1.0 effective threshold |

Independent classifier: **17/17 lateral, 17/17 longitudinal** on the controls.  The builder reproduces turns and the speed classes but reads a gentle curve as NUDGE, misses a genuine 1 m nudge, and accelerates at +1.0 m/s.

### 2b. Lateral agreement (strict, band [2,6], rule A)

| split | n | agreement | 95 % Wilson | rule B | span [0,6] | first-run A0 | worst confusion |
|---|---|---|---|---|---|---|---|
| eval139 | 139 | **102/139 = 73.4 %** | 65.5–80.0 | 103/139 = 74.1 % | 101/139 = 72.7 % | 85/139 = 61.2 % | NUDGE_L→LANE_KEEP 10/13 (77 %); NUDGE_R→LANE_KEEP 12/20 (60 %) |
| train | 4,369 | **3,149/4,369 = 72.1 %** | 70.7–73.4 | 3,164 = 72.4 % | 3,017 = 69.1 % | 2,748 = 62.9 % | NUDGE_L→LANE_KEEP 331/464 (71.3 %); NUDGE_R→LANE_KEEP 386/564 (68.4 %) |

Collapsed to keep / move / turn the agreement is 74.1 % (eval139) and 73.2 % (train); where both sources name a side it matches 21/24 (eval139) and 745/827 = 90.1 % (train).  Per class (train): LANE_KEEP 91.6 %, TURN_L 87.0 %, TURN_R 85.4 %, **NUDGE_L 10.3 %, NUDGE_R 12.1 %**.  The independent derivation finds out-and-back nudges (NUDGE, 0.3–1.5 m, returning) in only **20 of 4,369 clips (0.46 %)**, lane-change-scale SHIFTs in 281 (6.4 %), bends in 294 (6.7 %), turns in 609 (13.9 %; builder TURN: 517).

```
[train] LATERAL A: indep over the record's own band [t0+2,t0+6]  n=4369  strict-agree 3149/4369 = 0.721
                         LANE_KEEP    BEND_L    BEND_R   NUDGE_L   NUDGE_R   SHIFT_L   SHIFT_R    TURN_L    TURN_R   | n
LANE_KEEP                     2447        64        76         4         2        64        58        55        54   | 2824
NUDGE_L                        331        31         8         2         6        46        17        23         0   | 464
NUDGE_R                        386        22        32         2         4        23        64         0        31   | 564
TURN_L                           1        29         0         0         0         4         1       235         0   | 270
TURN_R                           0         0        32         0         0         3         1         0       211   | 247

[eval139] LATERAL A, band [t0+2,t0+6]  n=139  strict-agree 102/139 = 0.734
                         LANE_KEEP    BEND_L    BEND_R   NUDGE_L   NUDGE_R   SHIFT_L   SHIFT_R    TURN_L    TURN_R   | n
LANE_KEEP                       83         2         1         0         0         0         2         2         3   | 93
NUDGE_L                         10         0         0         0         0         3         0         0         0   | 13
NUDGE_R                         12         1         3         0         0         1         2         0         1   | 20
TURN_L                           0         0         0         0         0         0         1         4         0   | 5
TURN_R                           0         0         1         0         0         0         0         0         7   | 8
```

**What the NUDGE class is** (train, n=1,028 scored): over [0,6] the heading does *not* return in **89.7 %** (|net yaw| ≥ ½|peak yaw|); median net yaw 9.3°, median key-frame lateral peak 5.9 m; only 21 are independent nudges, 397 are lane-change-scale shifts, 199 are bends (net yaw ≥ 15°), 356 show no lane-change-scale excursion once curvature is removed, 55 are independent ≥ 30° turns (1,028 = 21+397+199+356+55) (the 66 suppressed turns that the PI rule relabels NUDGE).  Within the tactical band the median detrended excursion is 0.63 m (0.09–0.11 m in the operative band [0,2], so the NUDGE signal is not an operative-band artefact; 19 % have ≥ 0.3 m there).  I cannot separate "gentle bend" from "slow lateral drift" with ego poses alone (this is an identifiability limit of any pose-only derivation, mine included), so the claim is: **NUDGE does not identify a deliberate nudge**, not that every NUDGE is a bend.

### 2c. Longitudinal agreement (speed-defined classes; ADAPT and FOLLOW handled as stated)

| split | n speed-defined | band [2,6] | span [0,6] (builder's actual) | worst confusion on [0,6] |
|---|---|---|---|---|
| eval139 | 117 | 92/117 = **78.6 %** (70.4–85.1) | 107/117 = **91.5 %** (85.0–95.3) | ACCELERATE→CRUISE 5/36 |
| train | 3,419 | 2,744 = **80.3 %** (78.9–81.6) | 3,060 = **89.5 %** (88.4–90.5) | ACCELERATE→CRUISE 178/974 (18.3 %); on band [2,6] 396/974 (40.7 %) |

```
[train] LONGITUDINAL indep over [t0,t0+6] (builder's actual plan span)   agree 3060/3419 = 0.895
                           HOLD  CREEP CRUISE  ACCEL  BRAKE   | n
HOLD                         45     37      2      0     17   | 101
CREEP                         0     28     31      7     53   | 119
CRUISE                        0      0   1197      0      0   | 1197
ACCELERATE                    0      0    178    795      1   | 974
BRAKE_TO                      0      0     33      0    838   | 871
FOLLOW                        0      0    157      0      0   | 157
ADAPT_SPEED_FOR_CURVE         0      0    272    486    192   | 950
```
The CREEP/HOLD rows are scored on [0,6] here; on the band they define they are 68.1 % and 98.0 %.  The longitudinal label is a **cascade over two windows** ([2,6] for HOLD/CREEP, [0,6] for everything else — SEMANTICS §4).

### 2d. The labels as the trainer applies them (every supervised window)

Each window with |NOW−t0| ≤ 2 s is trained on the record's single label; here the independent test is re-run on **that window's own forward band [NOW+2, NOW+6]**.  Train: 173,409 windows (23.4 % of 741,483 are supervised): lateral strict agreement **70.8 %** (rule B 71.0 %), longitudinal **76.2 %** (135,688 speed-defined windows).  By NOW−t0 (0.5 s bins, −2.0 → +2.0): lateral 69.0, 71.1, 72.7, 72.8, 72.0, 71.1, 69.5, 67.9 %; longitudinal 80.5, 82.2, 82.2, 81.1, 78.0, 73.3, 68.5, 63.8 %.  eval139: 5,527 windows, lateral 72.0 %, longitudinal 77.8 %.  So the lateral label is uniformly as good as at the anchor (its error is class definition, not time) while the longitudinal label **decays 18 points across the supervised band** — it describes [t0, t0+6], not the window's own future.

## 3. Alpamayo (VLM) vs our geometry — which source is noisier?

Both sides are scored against the same independent geometry on **both** time bases (ours: raw [8,14]; the VLM's own: [5.1, 11.1], `alpamayo_records.py:149`).  4,243 train+eval records carry a VLM lateral side (`raw/alpamayo_vs_geometry.json`).

* The stored lateral flag is exactly reproducible from the label file (4,243/4,243).  **1,519 conflicts (35.8 %).**  The geometry is "straight" on 57.9–59.1 % of clips, so a constant "straight" predictor scores that.
* Each source against geometry on its own window: **VLM 69.6 %** (2,952/4,243, 68.2–70.9) vs **our label 72.4 %** (3,074, 71.1–73.8).  Timing test (VLM side vs geometry by window start): peak 70.1 % at start 6.1 s, 69.6 % at the stated 5.1 s — within ≈ 1 s of its stated window; ours peaks at 8.1 s (73.0 %).
* **Among the 1,519 conflicts, each on its own window: VLM right 776 (51.1 %), ours right 775 (51.0 %).**  Quadrants: both right 231 (15.2 % — a *timing* artefact: the two describe different 6 s), both wrong 199, VLM-right/ours-wrong 545, VLM-wrong/ours-right 544.  On the common (our) window the VLM is right on 557 (36.7 %), ours on 775 (51.0 %), both wrong 187.
* **By our label class (conflicts n; VLM right / ours right):** LANE_KEEP 825: 49.3 / 47.2 · NUDGE_L 208: 58.2 / 35.1 · NUDGE_R 232: 57.3 / 34.5 · **TURN_L 149: 49.0 / 89.9 · TURN_R 105: 40.0 / 94.3.**  By road class: highway 189: 63.0 / 19.6 · intersection 253: 43.9 / 75.1 · urban 1,077: 50.7 / 50.9.  Day 813: 51.1 / 52.2 · night 706: 51.1 / 49.7.  Conflict rate: highway 26.5 %, intersection 39.5 %, urban 37.3 %; day 35.4 %, night 36.2 % (no day/night effect); VLM says left 53.4 %, right 58.4 %, straight 20.2 %; VLM channels concordant 25.6 %, self-contradictory 51.7 %.
* **Longitudinal** (4,241 scored): the VLM phrase agrees with the realised Δv **in its own window** on 63.5 % (2,694; 62.1–65.0).  By phrase: Strong Acceleration 95.5 % (n=179), Strong Deceleration 83.8 % (265), Gentle Acceleration 65.5 % (1,106), Maintain Speed 61.3 % (1,166), **Gentle Deceleration 56.5 % (1,525)**.  Our label's class agrees with independent geometry 89.5 % (§2c).  Conflict rate by our class: HOLD 7 %, CREEP 19 %, BRAKE_TO 34 %, ADAPT 33 %, ACCELERATE 39 %, CRUISE 41 %, FOLLOW 42 %; intersection 20.7 % vs urban/highway 39.3 %.  The stored `dv_measured_ms` equals my independent Δv on [5.1, 11.1] at median 0.005 m/s (99.9 % within 0.3).
* **Verdict.**  *Lateral:* neither source is the noisier one — they are coin-flips against each other.  Our geometry is the reliable source where it says **TURN** (90–94 % right in conflicts); the VLM is the better source where our label says **NUDGE** (57–58 % vs 34–35 %), which is exactly the class §2 shows is a curve label; on highways the VLM is right 63 % vs our 20 %.  *Longitudinal:* our geometry is the less noisy source (89.5 % vs 63.5 %), the VLM's weak phrases being "gentle".  The `agree` flags conflate real error with window mismatch (15 % of lateral conflicts are both-right).

## 4. Speed ceiling validity (`raw/speed_eval139.json`, `speed_train.json`)

Setup: the fed value is the clip-constant sidecar `v_hi_ms` (and its 4-way bin {30,50,100,120} km/h), fed on **every** window.  The window's own realised max over [NOW+2, NOW+6] is read from the **100 Hz egomotion log** (independent of the cache; the builder's 10 Hz sampling).  Windows with the log shorter than NOW+6 are excluded: 145 of 23,772 eval, 5,463 of 746,946 train.

| scope | n windows | corr(fed, own max) | R²(own max ← fed) | R²(own max ← `v_now`) | mean abs err | own max > fed by > 3 m/s | fed > own by > 5 m/s | own max > the BIN's km/h limit |
|---|---|---|---|---|---|---|---|---|
| **eval139, \|NOW−t0\| ≤ 2 s** | 5,527 | 0.9926 | **0.985** | 0.887 | 0.47 m/s | 0.6 % | 0.1 % | 3.3 % |
| eval139, all windows | 23,627 | 0.941 | 0.886 | **0.919** | 1.51 | 6.7 % | 5.0 % | 10.5 % |
| **train, \|NOW−t0\| ≤ 2 s** | 173,409 | 0.9938 | **0.988** | 0.877 | 0.44 | 0.6 % | 0.06 % | 4.7 % |
| train, all windows | 741,483 | 0.9426 | 0.889 | **0.899** | 1.53 | 8.4 % | 3.6 % | 11.9 % |

By NOW−t0 (train, R²(fed) / mean abs err m·s⁻¹): [−9,−6) 0.79 / 2.39 · [−6,−4) 0.85 / 1.98 · [−4,−2) 0.93 / 1.27 · [−2,0) 0.987 / 0.45 · [0,2) 0.988 / 0.43 · [2,4) 0.94 / 1.16 · [4,6) 0.88 / 1.75 · [6,12) 0.80 / 2.35.  So the channel is **exactly the window's own max for the 23.4 % of windows the tactical labels also cover, and a stale value for the other 76.6 %**; there it explains less of the window's future than the current speed does.  (Its source stretch is [t0+2, t0+6] = raw [10,14] s; for a window at NOW = 2 s that is 8 s ahead.)

**Bins:** the 4-way ladder compresses real speed: 30 km/h ↔ v_hi 0–8.3 m/s, 50 ↔ 8.3–13.9, **100 ↔ 13.9–27.7 m/s (50–100 km/h is ONE input value, 21.7 % of train clips)**, 120 ↔ 27.8–37.8; at clip level R²(v_hi ← bin only) = 0.81 vs 0.87 from `v0` alone.  The sidecar bin equals an independently written ladder on 4,369/4,369 + 139/139 clips (0 mismatches).

**Documented intersection artefact (stopped ego ⇒ lowest bin):** clips with v0 < 1 m/s at the anchor — train **271 (6.2 %); 232 of them (85.6 %) have bin ≤ 30 km/h**; eval139 8 clips, 7 (87.5 %).  Their realised max over [2,6] has median 4.9 m/s, p95 10.1 m/s, i.e. the value is *correct as a realised maximum* of a slowly pulling-away ego and *wrong as a limit*.  Intersection-class clips: **74.0 % land in ≤ 30 km/h** (train 811; eval 65.4 %, n=26) vs urban 36.2 % and highway 2.7 % — reproducing the record's "75 %".  Only 25.3 % of intersection clips are stopped at the anchor, and moving intersection clips still get ≤ 30 km/h in 69.8 %: the artefact is "slow ego ⇒ low limit" in general, not just "stopped ego".

## 5. `nav_command` as the trainer uses it (`raw/census_nav_train.json`)

Same token on every window; the turn it names starts a median 7.3 s after the anchor (42.0 % > 10 s; 68 tokens name a turn > 30 s away).  Independent test (≥ 30° inside any 3 s, within the window's own next 6 s; 576,553 windows whose next 6 s are inside the cache): of 213,371 windows carrying a TURN token, **24.7 % (52,728)** see a same-side turn begin; 64.9 % of the 81,248 windows that do have one carry the matching token; in the supervised range the L/R tokens hit 35.6 % / 33.2 %.  Consistency defects inside the file: 132 records have `nav_command = FOLLOW` while `nav_30s` lists a turn (66 suppressed turns whose flag `nav_30s` never read — it looks for `"suppressed"`, the emitter writes `"applied"`; 66 not suppressed, the mechanism being a `seq[0]` that is a curve hiding a later turn — 76 clips have that shape).

## 6. What I could not do / caveats

* **Thor not used for compute.**  The brief asked for Thor + `flock`; the train cache's `_v2manifest.pt` (the exact tensor store `v2_dataset._scan_meta` reads) is 23.8 MB, so I copied it (md5 `3c9f8bc8…` identical to Thor's) and ran locally, CPU, ≤ 2 GB.  No Thor python process was started; last Thor contact was one `scp` read + `ls`/`md5sum` at ≈ 12:02 local.
* 203 train / 8 eval records have no pose track (not in the refcv7 cache) — unscored; 3 eval clips and 22 train clips run on the fallback clock (≈ 0.1 s off), kept in.
* Independent geometry cannot establish "nudge" vs "curve" (identifiability); my rule was revised once after seeing eval139 (disclosed in §2).  The VLM comparison uses my geometry as referee for the VLM's *side*, so a gentle bend counts as a side.
* Not audited: the strategic labels (refcv7 ran `--no-strategic`), the correctness of the 15 VLM goal tokens against the images (needs vision), `scene`/`tac_SIT`, `strata` correctness (used only to stratify).  UNVERIFIED.

## 7. Next levers, ranked by measured mass (all label-side, 0 GPU; effects are HYPOTHESES until run)

1. **NUDGE (23.5 % of the lateral loss mass)**: mask `NUDGE_L/R` (and `EVADE_IN_CORRIDOR`, which requires a NUDGE) in the lateral CE for refcv8, or re-derive with the returning/curvature-detrended rule in `raw/d2_lib.py` (≈ 0.5 % of clips are true nudges, 6.4 % lane-change-scale).  Cheapest discriminating experiment: a 3-way lateral head (LANE_KEEP / TURN_L / TURN_R, 91.6 / 87.0 / 85.4 % corroborated) vs the 5-way head, same everything else.
2. **Longitudinal window and threshold**: relabel on the declared [2,6] band (or declare [0,6]) and make ACCELERATE the documented +1.5 (moves 175 of 974); expected gain is label consistency of ≈ 10 points (80.3 → 89.5 % agreement).
3. **Max-speed channel**: recompute the ceiling **per window** from the window's own [NOW+2, NOW+6] (the same quantity; R² 0.988 where it is exact) or feed it only inside \|NOW−t0\| ≤ 2 s with `valid = 0` elsewhere; today 76.6 % of windows carry a stale value.
4. **Nav**: replace the bare token by a time-localised nav from `nav_30s` (turn within horizon with distance); fix the `suppressed`/`applied` key and the 30 s cap mismatch first.
5. Documentation: rename `lat_peak_m` → `lat_disp_peak_20s_m` (or drop it), correct `ego_manoeuvre.py:103-112`, `v_target_ms` → `v_min_ms`.
