# DSCALE10K DATA-HURTS diagnostics (RESULT_DIAG)

Evidence class MEASURED (ours; banked T1 tick dumps, zero training, CPU). Tier T1-kin tick 0 (heads) / T1-kin trajectory cells (TAC-LAT, TAC-LON), 141 episodes / 2,399 windows. Estimator paired episode-cluster bootstrap, B=2000, seed 0, taniteval.ci resampler (F:/TanitAD-scratch/otband_read/tree/taniteval/taniteval/ci.py); 'another draw of EPISODES' only. Rules frozen in `DIAG_ADDENDUM.md` (sha256 `44dade04f24f`) before any number.

## 1. Per-goal-class F1 by source of supervision (stage B, deciding)

**Label: NOT-DILUTION** (groups that lose at both seeds: {'VLM': False, 'GEO': False}). n = 2399 windows, 141 episodes; a class is scored only on windows where its label exists.

Controls: oracle reads F1 1.0 on every supported class = True; U vs itself reads delta 0 CI [0,0] = True.

| group | seed 0 D (U-B) [95% CI] | seed 1 D (U-B) [95% CI] | floor |B_rep - B_s0| (beside) | classes |
|---|---|---|---|---|
| VLM macro-F1 | -0.0297 [-0.0784, +0.0208] | -0.0046 [-0.0468, +0.0366] | 0.0132 | YIELD, CORRIDOR_OFFSET, EVADE_IN_CORRIDOR, OVERTAKE_VEHICLE, MERGE, GAP_TARGET, REACT_ON_ONCOMING, TAKE_EXIT_R, TRAFFIC_LIGHT_REACT_RED, TRAFFIC_LIGHT_REACT_YELLOW, TRAFFIC_LIGHT_REACT_GREEN, LANE_CHANGE_L, LANE_CHANGE_R |
| GEO macro-F1 | -0.0021 [-0.1223, +0.0574] | +0.0238 [-0.0562, +0.0734] | 0.0118 | FOLLOW_LANE, TURN_L, TURN_R, YIELD_FOR_TURN_L, YIELD_FOR_TURN_R, STOP_POINT |

Per class (stage B; F1 of each arm, D = U - B with CI; support = positive windows / scored windows / positive episodes):

| class | source | pos / scored / ep | B_s0 | U_s0 | D s0 | B_s1 | U_s1 | D s1 |
|---|---|---|---|---|---|---|---|---|
| FOLLOW_LANE | geometry | 1910 / 2107 / 139 | 0.973 | 0.970 | -0.0020 [-0.0071, +0.0029] | 0.974 | 0.970 | -0.0042 [-0.0120, +0.0027] |
| TURN_L | geometry | 47 / 2320 / 9 | 0.808 | 0.757 | -0.0509 [-0.1409, +0.0091] | 0.816 | 0.710 | -0.1053 [-0.2506, -0.0082] sep |
| TURN_R | geometry | 150 / 2320 / 22 | 0.741 | 0.740 | -0.0011 [-0.0460, +0.0420] | 0.760 | 0.732 | -0.0285 [-0.0852, +0.0357] |
| YIELD_FOR_TURN_L | geometry | 11 / 2107 / 3 | 0.533 | 0.308 | -0.2256 [-0.6667, +0.0000] | 0.533 | 0.533 | +0.0000 [+0.0000, +0.0000] |
| YIELD_FOR_TURN_R | geometry | 10 / 2107 / 2 | 0.000 | 0.286 | +0.2857 [+0.0000, +0.4444] | 0.000 | 0.286 | +0.2857 [+0.0000, +0.4444] |
| YIELD | vlm | 85 / 695 / 17 | 0.126 | 0.101 | -0.0253 [-0.0714, +0.0000] | 0.042 | 0.062 | +0.0208 [-0.0941, +0.1663] |
| STOP_POINT | geometry | 131 / 2246 / 25 | 0.125 | 0.107 | -0.0183 [-0.0622, +0.0228] | 0.213 | 0.209 | -0.0047 [-0.1374, +0.1269] |
| CORRIDOR_OFFSET | vlm | 120 / 695 / 24 | 0.304 | 0.228 | -0.0762 [-0.1470, -0.0129] sep | 0.268 | 0.277 | +0.0092 [-0.1795, +0.2104] |
| EVADE_IN_CORRIDOR | vlm | 110 / 695 / 22 | 0.438 | 0.400 | -0.0382 [-0.1700, +0.1062] | 0.438 | 0.514 | +0.0765 [-0.0781, +0.2307] |
| OVERTAKE_VEHICLE | vlm | 20 / 695 / 4 | 0.182 | 0.000 | -0.1818 [-0.5714, +0.0000] | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] |
| MERGE | vlm | 25 / 695 / 5 | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] |
| GAP_TARGET | vlm | 60 / 695 / 12 | 0.114 | 0.256 | +0.1421 [-0.1797, +0.4103] | 0.152 | 0.000 | -0.1515 [-0.4444, +0.0000] |
| REACT_ON_ONCOMING | vlm | 50 / 695 / 10 | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] | 0.038 | 0.000 | -0.0385 [-0.1212, +0.0000] |
| TAKE_EXIT_R | vlm | 30 / 695 / 6 | 0.125 | 0.000 | -0.1250 [-0.3333, +0.0000] | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] |
| TRAFFIC_LIGHT_REACT_RED | vlm | 65 / 720 / 8 | 0.442 | 0.419 | -0.0228 [-0.3031, +0.3118] | 0.424 | 0.557 | +0.1325 [-0.0850, +0.3540] |
| TRAFFIC_LIGHT_REACT_YELLOW | vlm | 15 / 695 / 3 | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] |
| TRAFFIC_LIGHT_REACT_GREEN | vlm | 75 / 695 / 15 | 0.085 | 0.026 | -0.0588 [-0.1770, +0.0597] | 0.233 | 0.123 | -0.1091 [-0.3158, +0.1099] |
| LANE_CHANGE_L | vlm | 15 / 695 / 3 | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] |
| LANE_CHANGE_R | vlm | 10 / 695 / 2 | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] | 0.000 | 0.000 | +0.0000 [+0.0000, +0.0000] |

Excluded from the macro (n_pos = 0): SPEED_BAND (n_pos 0, scored 0), TAKE_EXIT_L (n_pos 0, scored 695), TRAFFIC_LIGHT_REACT (n_pos 0, scored 695)

Stage A, beside (T mechanism stage): VLM D s0 -0.0280 [-0.0758, +0.0156]; s1 -0.0306 [-0.0728, +0.0112]; GEO D s0 -0.0259 [-0.0617, +0.0036]; s1 +0.1239 [-0.0019, +0.1756].

## 2. TAC-LAT / TAC-LON accuracy, U - B, per country and platform (stage B, descriptive only)

Layer: country = clip-level modal of the STRAT labels table eval_country_id (valid, flag 0) [F:/TanitAD-scratch/strat1/read/inputs/strat_labels_table.npz, the STRAT1 read input; a per-frame label release layer]; platform = strat_platform_by_clip.npz eval_platform_id [same folder]. Corpus clip metadata for the 10k set (country, platform_class) lives in datascale/ds_select.py order_all_eligible.parquet.

K-RE (recomputed against the banked read): all equal = True. Strata read: {'country': 23, 'platform': 2}, cells 100 (multiplicity; no stratum is promoted).

### country

| stratum | n win | n ep | LAT s0 | LAT s1 | LON s0 | LON s1 | flag |
|---|---|---|---|---|---|---|---|
| Austria | 153 | 9 | +0.000 | +0.000 | +0.007 | -0.065 |  |
| Belgium | 136 | 8 | -0.022 | -0.029* | +0.007 | -0.059 |  |
| Bulgaria | 17 | 1 | n/a | n/a | n/a | n/a |  |
| Croatia | 51 | 3 | +0.000 | +0.000 | +0.451 | -0.294 |  |
| Czechia | 17 | 1 | n/a | n/a | n/a | n/a |  |
| Denmark | 119 | 7 | +0.025 | -0.017 | -0.025 | -0.269 |  |
| Estonia | 68 | 4 | -0.059 | -0.059 | +0.324* | +0.118 |  |
| Finland | 119 | 7 | -0.017 | -0.067 | +0.067 | +0.000 |  |
| France | 136 | 8 | -0.029 | -0.096 | -0.066 | -0.066 |  |
| Germany | 153 | 9 | -0.059* | -0.013 | -0.072 | -0.072 |  |
| Greece | 85 | 5 | +0.000 | +0.035 | -0.188* | -0.306* | same-dir TAC-LON |
| Hungary | 34 | 2 | +0.000 | +0.029 | +0.088* | -0.235 |  |
| Italy | 136 | 8 | -0.081 | -0.037 | -0.088 | -0.125 |  |
| Latvia | 102 | 6 | -0.225* | -0.039 | -0.059 | -0.284* |  |
| Lithuania | 51 | 3 | +0.020 | +0.000 | -0.118 | -0.314* |  |
| Luxembourg | 17 | 1 | n/a | n/a | n/a | n/a |  |
| Netherlands | 189 | 11 | -0.026 | -0.085* | -0.042 | -0.069 |  |
| Poland | 119 | 7 | -0.067 | -0.008 | -0.008 | +0.059 |  |
| Portugal | 102 | 6 | -0.098 | +0.010 | +0.000 | +0.049 |  |
| Romania | 17 | 1 | n/a | n/a | n/a | n/a |  |
| Spain | 170 | 10 | -0.124* | -0.065* | -0.129* | -0.047 | same-dir TAC-LAT |
| Sweden | 187 | 11 | -0.102* | -0.059* | -0.064 | -0.160* | same-dir TAC-LAT |
| United States | 221 | 13 | -0.068 | +0.009 | -0.113* | -0.090 |  |

### platform

| stratum | n win | n ep | LAT s0 | LAT s1 | LON s0 | LON s1 | flag |
|---|---|---|---|---|---|---|---|
| hyperion_8 | 1141 | 67 | -0.041* | -0.024 | -0.056* | -0.078* | same-dir TAC-LON |
| hyperion_8.1 | 1258 | 74 | -0.059* | -0.034* | -0.021 | -0.112* | same-dir TAC-LAT |

`*` = the paired episode-cluster CI excludes 0 inside the stratum. Strata under 5 episodes cannot be flagged.

## Reading (MM-facing, added after the numbers; the label above is the pre-committed rule's output)
* F1 read: **NOT-DILUTION** as written (neither group loses at both seeds). Neither group's macro-F1 difference is separated at any seed; the VLM group's point is negative at both seeds (-0.030, -0.005), the geometry group's is -0.002 and +0.024. The test is power-limited: VLM classes have 2 to 24 positive episodes each, 4 of 13 VLM classes read F1 0.000 in all four stage-B arms shown (MERGE, TRAFFIC_LIGHT_REACT_YELLOW, LANE_CHANGE_L, LANE_CHANGE_R), and the CI half-widths (about 0.05) are four times the replicate floor (0.013 / 0.012). So this read does not confirm dilution and does not exclude it; it shows no head-level loss large enough to see.
* Per class, separated at one seed only: CORRIDOR_OFFSET (vlm) -0.076 at seed 0, TURN_L (geometry) -0.105 at seed 1; none at both seeds.
* Country / platform: no stratum is FLAGGED (no stratum improves at both seeds). Same-direction (cost at both seeds, separated inside the stratum): Greece on TAC-LON, Spain and Sweden on TAC-LAT, platform hyperion_8 on TAC-LON and hyperion_8.1 on TAC-LAT. 100 stratum cells were read (multiplicity), strata under 5 episodes cannot be flagged (6 countries have 1 to 4 episodes), and no label decides from this.
* Evidence class MEASURED; tier and estimator as stamped at the top; each seed's difference answers "another draw of episodes" only, with the replicate floor beside.
