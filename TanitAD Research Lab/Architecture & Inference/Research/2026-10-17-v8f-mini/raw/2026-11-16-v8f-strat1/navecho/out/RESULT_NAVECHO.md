# RESULT_NAVECHO -- D-STRAT1-TACLAT-F1: nav echo or skill cost

Evidence class: MEASURED (ours; banked T1-kin dumps in `F:/TanitAD-scratch/strat1/read/t1`, zero training, CPU). Tier of every number: T1-kin tick 0 (lateral head argmax at the window's first tick under the T1 feed: dead-reckoned nav, logged scene) vs the v9 label lat_cls (5-way); the nav-only predictor sees only nav_in at that tick.
Estimator: episode-cluster bootstrap (taniteval.ci, stat / paired stat), B=2000, seed 0, 141 episodes; 'another draw of EPISODES' only; training-run variance = floor F = |C0_rep - C0_s0|. 2,399 windows, 141 episodes. Artifact: `F:/TanitAD-scratch/strat1/navecho/out/RESULT_NAVECHO.json`.

## Reading by the committed rules: **MIXED**

* ECHO needs the nav-only predictor at >= 90 % of C0 head macro-F1 at s0 and s1. It reads 0.689 (s0) and 0.652 (s1): not met.
* SKILL-COST needs the S1 - C0 head gap separated, same sign, at s0 and s1 on the uninformative windows. It reads s0 -0.000 [-0.001, +0.000], s1 -0.001 [-0.001, -0.000] sep: s0 is not separated, so not met.
* Neither rule fires, so the reading is MIXED. **The uninformative split cannot carry the test**: it holds 1,736 labelled windows of which only 10 are turns (TURN_L 1, TURN_R 9), C0 and S1 at s0 and s1 recall no turn there (macro-F1 0.332, the majority-class score over the three present classes), and a gap of -0.0003 to -0.0005 is one or two windows, far below the floor F 0.067. It neither supports nor rules out a skill cost on a turn that no feed announces.
* **Where the gap lives (additive, labelled windows, net correct windows C0 - S1):** s0 total 51: UNINF_BOTH 5, INF_ANY 46; s1 total 57: UNINF_BOTH 5, INF_ANY 52. About 90 % of the gap sits on the 439 windows (371 labelled) where at least one feed names a turn.
* **Why it is not a plain echo either:** a map from the nav token alone reaches macro-F1 0.562 (v9 feed) against the C0 head 0.816 / 0.862; the head uses the image beside the token. On the windows where C0's own token names a turn the head reads 0.736 / 0.805 and the token-only map 0.337, so the head sits far above its nav. What the data cannot separate: the v9 token is built from the ego's own future (section 1), so on those windows C0 holds a cleaner route cue than the text feed holds, and that difference is the cost the text nav carries into the head on announced turns. The text feed names a turn on 223 windows, the v9 feed on 395; the two tokens differ on 261 windows.
* Same sign at s10 / s11 and for N1 / S1d: see the tables. s10 is the weak seed (INF_ANY S1 - C0 -0.094, not separated).

## 1. Provenance (two probes each, commit 0ddfb65)

| object | derivation (file:line) | function of the ego future? |
|---|---|---|
| `lat_cls_a` | `stack/scripts/build_v9_labels.py:802-891, 969`: heading change `dps = degrees(gpsi[20:] - gpsi[20])` of the LOGGED future from NOW + 2 s to the horizon, `theta` its extreme, `is_junc = big and dom.is_turn` where `dom` is a turn segment of the ego's own track overlapping 2 s to 8 s; class = side of theta (1 L, 2 R); else lane change or lane keep. Test pins: `tests/test_v9_labels_builder.py:50-68, 147` | YES, entirely (ego logged poses 2-8 s ahead) |
| v9 `nav_token`, `nav_d_next_m` | `build_v9_labels.py:260-283` (`nav_fields`): the first ANNOUNCED turn `u` of `turns` (segmented from the same ego track, `Turn.announced` :229); token = side of `u` when in progress or `d_next <= d_ann = max(30 m, 6 s x v_now)` (:265, :279); distance = `u.s_start - s_now`. Fed as 5 channels by `stack/tanitad/data/v8f_mini_data.py:219-222` (one-hot token, d/100 clipped to [0,3], valid) | YES, same turn table and track as the label |
| text nav (N1, S1, S1d) | `v8f_mini_data.py:373-396` (`nav_text_arrays`) loads `navtext_nav.npz` (sha256 pinned; report `inputs/navtext_nav_report.json`); built by the BUILD_SPEC section 5 rule from NVIDIA route text (GPS trace matched to OSM routing, `navtok.py`; `PREREG_NAVTEXT_DRAFT.md` sections 1-3): the same announcement cut (`d_ann = max(30, 6 v)`, countdown distance, U-turn excluded) applied to the text turn. 8,235 eval steps differ from the v9 token (`navtext_nav_report.json`, eval `n_steps_text_ne_v9`) | NO ego-future turn table; it follows the driven GPS route (a function of where the car drove) on a different clock (G-CLOCK-NAV median error 5.3 s) |

The v9 token and the label share a source (the ego track's turn table) and differ in time window: the token looks to the next announced turn (about 6 s of travel or 30 m), the label looks 2 s to 8 s ahead. They are near functions of each other, not identical: of the 335 labelled windows where C0's v9 token names a turn, 157 carry LANE_KEEP and 178 a turn label.

## 2. The nav-only predictor (macro-F1 against lat_cls, tier T1-kin tick 0)

Map the nav token (FOLLOW / TURN_L / TURN_R / no row) to the majority label; fit on one half of the EPISODES, score on the other, halves swapped. Majority baseline 0.317.

| feed | fit A score B | fit B score A | mean | 20 random splits (min / median / max) | token x distance bins (sensitivity) | shuffled nav (20 perms, mean) |
|---|---|---|---|---|---|---|
| v9 | 0.577 | 0.548 | 0.562 | 0.414 / 0.523 / 0.650 | 0.477 | 0.317 |
| text | 0.596 | 0.498 | 0.547 | 0.375 / 0.550 / 0.614 | 0.533 | 0.317 |

| arm | feed | head macro-F1 | nav-only (token, mean of halves) | ratio | nav-only (token x distance) | ratio |
|---|---|---|---|---|---|---|
| C0_s0 | v9 | 0.816 | 0.562 | 0.689 | 0.477 | 0.585 |
| C0_s1 | v9 | 0.862 | 0.562 | 0.652 | 0.477 | 0.553 |
| C0_s10 | v9 | 0.818 | 0.562 | 0.687 | 0.477 | 0.583 |
| C0_s11 | v9 | 0.853 | 0.562 | 0.660 | 0.477 | 0.560 |
| N1_s0 | text | 0.669 | 0.547 | 0.818 | 0.533 | 0.797 |
| N1_s1 | text | 0.709 | 0.547 | 0.771 | 0.533 | 0.751 |
| N1_s10 | text | 0.704 | 0.547 | 0.777 | 0.533 | 0.757 |
| N1_s11 | text | 0.704 | 0.547 | 0.777 | 0.533 | 0.757 |
| S1_s0 | text | 0.676 | 0.547 | 0.808 | 0.533 | 0.788 |
| S1_s1 | text | 0.713 | 0.547 | 0.767 | 0.533 | 0.748 |
| S1_s10 | text | 0.718 | 0.547 | 0.761 | 0.533 | 0.742 |
| S1_s11 | text | 0.699 | 0.547 | 0.782 | 0.533 | 0.762 |
| S1d_s0 | text | 0.679 | 0.547 | 0.805 | 0.533 | 0.785 |
| S1d_s1 | text | 0.700 | 0.547 | 0.781 | 0.533 | 0.762 |
| S1d_s10 | text | 0.701 | 0.547 | 0.780 | 0.533 | 0.760 |
| S1d_s11 | text | 0.720 | 0.547 | 0.759 | 0.533 | 0.740 |
| C0_rep | v9 | 0.835 | 0.562 | 0.674 | 0.477 | 0.572 |

Nav-only out-of-fold macro-F1 on each split (its own feed): v9: UNINF_BOTH 0.332, INF_ANY 0.373, UNINF_V9 0.332, INF_V9 0.337; text: UNINF_BOTH 0.332, INF_ANY 0.405, UNINF_V9 0.461, INF_V9 0.421.

## 3. The split

Splits by the nav feeds at tick 0: UNINF_BOTH = neither the v9 nor the text token names a turn (the committed uninformative set, the only one where no arm holds route information); INF_ANY = its complement; UNINF_V9 / INF_V9 = by C0's own token.

| split | n windows | n labelled | n episodes (labelled) | label support |
|---|---|---|---|---|
| UNINF_BOTH | 1960 | 1736 | 131 | LANE_KEEP 1726, TURN_L 1, TURN_R 9 |
| INF_ANY | 439 | 371 | 41 | LANE_KEEP 184, TURN_L 46, TURN_R 141 |
| UNINF_V9 | 2004 | 1772 | 132 | LANE_KEEP 1753, TURN_L 1, TURN_R 18 |
| INF_V9 | 395 | 335 | 37 | LANE_KEEP 157, TURN_L 46, TURN_R 132 |

### UNINF_BOTH  (floor F: macro-F1 0.067; TURN_R recall 0.111)

Head macro-F1 and TURN recall per arm (point [95 % episode-cluster CI]; n/a = class absent):

| arm | macro-F1 | TURN_L recall | TURN_R recall |
|---|---|---|---|
| C0_s0 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C0_s1 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C0_s10 | 0.399 [0.333, 0.624] | 0.000 [0.000, 0.000] | 0.111 [0.000, 0.143] |
| C0_s11 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C0_rep | 0.399 [0.333, 0.624] | 0.000 [0.000, 0.000] | 0.111 [0.000, 0.143] |
| N1_s0 | 0.332 [0.331, 0.499] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| N1_s1 | 0.332 [0.330, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| N1_s10 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| N1_s11 | 0.332 [0.330, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| S1_s0 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| S1_s1 | 0.332 [0.330, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| S1_s10 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| S1_s11 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| S1d_s0 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| S1d_s1 | 0.374 [0.332, 0.603] | 0.000 [0.000, 0.000] | 0.111 [0.000, 0.143] |
| S1d_s10 | 0.332 [0.330, 0.499] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| S1d_s11 | 0.332 [0.331, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

Paired X - C0 (same seed), macro-F1 [paired CI]; sep = CI excludes 0, >F = above the floor; net = net correct windows X - C0:

| X | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| S1 - C0 | -0.000 [-0.001, +0.000]; net -5 | -0.001 [-0.001, -0.000] sep; net -5 | -0.067 [-0.125, -0.000] sep >F; net -4 | -0.000 [-0.001, +0.000]; net -2 |
| N1 - C0 | -0.000 [-0.001, -0.000] sep; net -5 | -0.001 [-0.002, +0.000]; net -5 | -0.067 [-0.125, -0.000] sep >F; net -5 | -0.000 [-0.001, +0.000]; net -1 |
| S1d - C0 | -0.000 [-0.001, +0.000]; net -3 | +0.041 [-0.001, +0.097]; net -5 | -0.067 [-0.126, -0.000] sep >F; net -8 | -0.000 [-0.001, +0.000]; net -2 |

TURN recall gap S1 - C0:

| class | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| TURN_L | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| TURN_R | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.111 [-0.143, +0.000] | +0.000 [+0.000, +0.000] |

### INF_ANY  (floor F: macro-F1 0.009; TURN_R recall 0.078)

Head macro-F1 and TURN recall per arm (point [95 % episode-cluster CI]; n/a = class absent):

| arm | macro-F1 | TURN_L recall | TURN_R recall |
|---|---|---|---|
| C0_s0 | 0.739 [0.655, 0.802] | 0.826 [0.667, 0.946] | 0.823 [0.671, 0.938] |
| C0_s1 | 0.804 [0.727, 0.868] | 0.913 [0.786, 1.000] | 0.872 [0.722, 0.967] |
| C0_s10 | 0.721 [0.631, 0.793] | 0.848 [0.692, 0.958] | 0.894 [0.736, 0.993] |
| C0_s11 | 0.796 [0.720, 0.859] | 0.870 [0.710, 0.980] | 0.830 [0.680, 0.938] |
| C0_rep | 0.748 [0.653, 0.824] | 0.826 [0.727, 0.917] | 0.901 [0.756, 0.988] |
| N1_s0 | 0.575 [0.426, 0.689] | 0.543 [0.125, 0.900] | 0.582 [0.439, 0.698] |
| N1_s1 | 0.624 [0.474, 0.737] | 0.565 [0.200, 0.933] | 0.645 [0.504, 0.764] |
| N1_s10 | 0.610 [0.456, 0.715] | 0.522 [0.120, 0.909] | 0.667 [0.534, 0.778] |
| N1_s11 | 0.613 [0.459, 0.724] | 0.543 [0.171, 0.868] | 0.660 [0.510, 0.785] |
| S1_s0 | 0.585 [0.440, 0.691] | 0.565 [0.174, 0.933] | 0.596 [0.442, 0.729] |
| S1_s1 | 0.628 [0.483, 0.736] | 0.587 [0.232, 0.963] | 0.645 [0.503, 0.769] |
| S1_s10 | 0.627 [0.479, 0.732] | 0.543 [0.154, 0.909] | 0.674 [0.544, 0.781] |
| S1_s11 | 0.605 [0.450, 0.715] | 0.565 [0.208, 0.870] | 0.631 [0.460, 0.772] |
| S1d_s0 | 0.582 [0.429, 0.696] | 0.543 [0.125, 0.900] | 0.596 [0.448, 0.721] |
| S1d_s1 | 0.613 [0.466, 0.723] | 0.522 [0.154, 0.889] | 0.652 [0.512, 0.772] |
| S1d_s10 | 0.611 [0.461, 0.717] | 0.543 [0.151, 0.931] | 0.645 [0.518, 0.750] |
| S1d_s11 | 0.632 [0.477, 0.739] | 0.587 [0.231, 0.872] | 0.652 [0.496, 0.775] |

Paired X - C0 (same seed), macro-F1 [paired CI]; sep = CI excludes 0, >F = above the floor; net = net correct windows X - C0:

| X | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| S1 - C0 | -0.154 [-0.280, -0.044] sep >F; net -46 | -0.176 [-0.318, -0.057] sep >F; net -52 | -0.094 [-0.240, +0.028] >F; net -25 | -0.191 [-0.332, -0.081] sep >F; net -58 |
| N1 - C0 | -0.164 [-0.295, -0.050] sep >F; net -50 | -0.180 [-0.326, -0.060] sep >F; net -53 | -0.111 [-0.260, +0.016] >F; net -30 | -0.183 [-0.323, -0.071] sep >F; net -53 |
| S1d - C0 | -0.157 [-0.288, -0.045] sep >F; net -48 | -0.191 [-0.335, -0.073] sep >F; net -54 | -0.110 [-0.255, +0.016] >F; net -31 | -0.164 [-0.307, -0.058] sep >F; net -50 |

TURN recall gap S1 - C0:

| class | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| TURN_L | -0.261 [-0.538, +0.000] >F | -0.326 [-0.683, +0.000] >F | -0.304 [-0.616, +0.036] >F | -0.304 [-0.654, -0.044] sep >F |
| TURN_R | -0.227 [-0.451, -0.015] sep >F | -0.227 [-0.415, -0.021] sep >F | -0.220 [-0.407, -0.035] sep >F | -0.199 [-0.427, +0.016] >F |

### UNINF_V9  (floor F: macro-F1 0.035; TURN_R recall 0.056)

Head macro-F1 and TURN recall per arm (point [95 % episode-cluster CI]; n/a = class absent):

| arm | macro-F1 | TURN_L recall | TURN_R recall |
|---|---|---|---|
| C0_s0 | 0.332 [0.329, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C0_s1 | 0.332 [0.329, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C0_s10 | 0.367 [0.330, 0.624] | 0.000 [0.000, 0.000] | 0.056 [0.000, 0.143] |
| C0_s11 | 0.331 [0.329, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| C0_rep | 0.367 [0.330, 0.624] | 0.000 [0.000, 0.000] | 0.056 [0.000, 0.143] |
| N1_s0 | 0.413 [0.328, 0.774] | 0.000 [0.000, 0.000] | 0.278 [0.000, 0.500] |
| N1_s1 | 0.446 [0.328, 0.878] | 0.000 [0.000, 0.000] | 0.444 [0.000, 0.800] |
| N1_s10 | 0.436 [0.328, 0.839] | 0.000 [0.000, 0.000] | 0.389 [0.000, 0.700] |
| N1_s11 | 0.441 [0.328, 0.868] | 0.000 [0.000, 0.000] | 0.389 [0.000, 0.700] |
| S1_s0 | 0.411 [0.328, 0.768] | 0.000 [0.000, 0.000] | 0.278 [0.000, 0.500] |
| S1_s1 | 0.446 [0.328, 0.871] | 0.000 [0.000, 0.000] | 0.444 [0.000, 0.800] |
| S1_s10 | 0.436 [0.328, 0.839] | 0.000 [0.000, 0.000] | 0.389 [0.000, 0.700] |
| S1_s11 | 0.425 [0.328, 0.812] | 0.000 [0.000, 0.000] | 0.333 [0.000, 0.600] |
| S1d_s0 | 0.411 [0.328, 0.775] | 0.000 [0.000, 0.000] | 0.278 [0.000, 0.500] |
| S1d_s1 | 0.446 [0.331, 0.846] | 0.000 [0.000, 0.000] | 0.444 [0.000, 0.700] |
| S1d_s10 | 0.431 [0.328, 0.829] | 0.000 [0.000, 0.000] | 0.389 [0.000, 0.700] |
| S1d_s11 | 0.425 [0.328, 0.813] | 0.000 [0.000, 0.000] | 0.333 [0.000, 0.600] |

Paired X - C0 (same seed), macro-F1 [paired CI]; sep = CI excludes 0, >F = above the floor; net = net correct windows X - C0:

| X | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| S1 - C0 | +0.080 [-0.006, +0.233] >F; net -21 | +0.114 [-0.006, +0.319] >F; net -17 | +0.069 [-0.128, +0.285] >F; net -19 | +0.094 [-0.005, +0.273] >F; net -17 |
| N1 - C0 | +0.081 [-0.006, +0.238] >F; net -22 | +0.114 [-0.006, +0.319] >F; net -17 | +0.069 [-0.128, +0.285] >F; net -20 | +0.110 [-0.005, +0.312] >F; net -15 |
| S1d - C0 | +0.080 [-0.006, +0.238] >F; net -20 | +0.114 [-0.005, +0.298] >F; net -17 | +0.064 [-0.128, +0.274] >F; net -23 | +0.094 [-0.005, +0.273] >F; net -16 |

TURN recall gap S1 - C0:

| class | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| TURN_L | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| TURN_R | +0.278 [+0.000, +0.500] >F | +0.444 [+0.000, +0.800] >F | +0.333 [-0.143, +0.700] >F | +0.333 [+0.000, +0.600] >F |

### INF_V9  (floor F: macro-F1 0.005; TURN_R recall 0.083)

Head macro-F1 and TURN recall per arm (point [95 % episode-cluster CI]; n/a = class absent):

| arm | macro-F1 | TURN_L recall | TURN_R recall |
|---|---|---|---|
| C0_s0 | 0.736 [0.657, 0.795] | 0.826 [0.667, 0.946] | 0.879 [0.782, 0.951] |
| C0_s1 | 0.805 [0.734, 0.868] | 0.913 [0.786, 1.000] | 0.932 [0.880, 0.978] |
| C0_s10 | 0.710 [0.622, 0.781] | 0.848 [0.692, 0.958] | 0.955 [0.879, 1.000] |
| C0_s11 | 0.798 [0.724, 0.860] | 0.870 [0.710, 0.980] | 0.886 [0.802, 0.950] |
| C0_rep | 0.741 [0.646, 0.820] | 0.826 [0.727, 0.917] | 0.962 [0.925, 0.993] |
| N1_s0 | 0.611 [0.450, 0.723] | 0.543 [0.125, 0.900] | 0.583 [0.432, 0.709] |
| N1_s1 | 0.650 [0.491, 0.766] | 0.565 [0.200, 0.933] | 0.629 [0.488, 0.752] |
| N1_s10 | 0.643 [0.481, 0.749] | 0.522 [0.120, 0.909] | 0.659 [0.521, 0.778] |
| N1_s11 | 0.644 [0.485, 0.757] | 0.543 [0.171, 0.868] | 0.652 [0.495, 0.786] |
| S1_s0 | 0.619 [0.466, 0.727] | 0.565 [0.174, 0.933] | 0.598 [0.434, 0.734] |
| S1_s1 | 0.655 [0.500, 0.764] | 0.587 [0.232, 0.963] | 0.629 [0.485, 0.758] |
| S1_s10 | 0.662 [0.506, 0.764] | 0.543 [0.154, 0.909] | 0.667 [0.528, 0.781] |
| S1_s11 | 0.638 [0.476, 0.748] | 0.565 [0.208, 0.870] | 0.629 [0.451, 0.778] |
| S1d_s0 | 0.618 [0.459, 0.731] | 0.543 [0.125, 0.900] | 0.598 [0.443, 0.731] |
| S1d_s1 | 0.639 [0.485, 0.751] | 0.522 [0.154, 0.889] | 0.644 [0.500, 0.772] |
| S1d_s10 | 0.645 [0.485, 0.750] | 0.543 [0.151, 0.931] | 0.636 [0.505, 0.748] |
| S1d_s11 | 0.664 [0.505, 0.771] | 0.587 [0.231, 0.872] | 0.652 [0.493, 0.781] |

Paired X - C0 (same seed), macro-F1 [paired CI]; sep = CI excludes 0, >F = above the floor; net = net correct windows X - C0:

| X | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| S1 - C0 | -0.117 [-0.237, -0.017] sep >F; net -30 | -0.151 [-0.289, -0.049] sep >F; net -40 | -0.049 [-0.188, +0.064] >F; net -10 | -0.160 [-0.298, -0.060] sep >F; net -43 |
| N1 - C0 | -0.125 [-0.251, -0.024] sep >F; net -33 | -0.155 [-0.294, -0.050] sep >F; net -41 | -0.067 [-0.214, +0.048] >F; net -15 | -0.154 [-0.288, -0.056] sep >F; net -39 |
| S1d - C0 | -0.118 [-0.243, -0.019] sep >F; net -31 | -0.166 [-0.310, -0.061] sep >F; net -42 | -0.066 [-0.209, +0.050] >F; net -16 | -0.134 [-0.269, -0.043] sep >F; net -36 |

TURN recall gap S1 - C0:

| class | s0 | s1 | s10 | s11 |
|---|---|---|---|---|
| TURN_L | -0.261 [-0.538, +0.000] >F | -0.326 [-0.683, +0.000] >F | -0.304 [-0.616, +0.036] >F | -0.304 [-0.654, -0.044] sep >F |
| TURN_R | -0.280 [-0.494, -0.081] sep >F | -0.303 [-0.447, -0.180] sep >F | -0.288 [-0.440, -0.155] sep >F | -0.258 [-0.460, -0.066] sep >F |

## 4. Controls

* K-UNION: pass True (confusion(split) + confusion(complement) equals the full-set confusion for every arm; full-set macro-F1 equals the landed RESULT_FAM_EXTRA.json to 1e-12).
* K1 self-pair: pass True (the head against itself on UNINF_BOTH reads exactly 0.0 [0.0, 0.0] on every cell).
* K-SHUFFLE: shuffled nav 0.3170 against majority baseline 0.3170: pass True.
* K-LANDED / K-FEED problems list: [] (empty = every arm full-set macro-F1 equals the landed `RESULT_FAM_EXTRA.json` to 1e-12, and every arm of a family carries the identical nav feed).

## 5. Limits

* The pre-registered echo question asks about windows with no route information, and only 10 labelled turn windows exist there; the MIXED reading follows from that as much as from the numbers. A powered version needs a larger eval grid or a train-split read (not run here).
* One fixed episode split (seed 0) carries the committed nav-only figure; 20 random splits range 0.41 to 0.65 (v9 feed), against head 0.816 / 0.862, so the ratio stays below 0.90 at every split.
* A separated interval from a one-seed arm is necessary, not sufficient; the estimator answers 'another draw of EPISODES'. The S1 - C0 gap on INF_ANY is separated at s0, s1 and s11 and has the same sign at s10.
* Provenance rows are read from source at commit 0ddfb65.
