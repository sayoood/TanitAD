### TACTICAL under both label clocks (SPEC A5): step5000, step 5000. PRIMARY clock: **OLD**

Clock and dt source: CORRECTED = grid_start + (t + w - 1 + n_stack - 1) * dt; 136/139 eval clips on the sidecar clock (dt median 0.10066662995966351), 3 on nominal dt 0.1 s with grid_start 0, 0 on pose dt; OLD = (t + w - 1) * 0.1 s.

Tier T1 (UNRULED for an action-free model). Declared-head accuracy [episode-cluster CI], κ full-set; in-band n per clock. Label tables sha256 `83b47aae4aff…`; controls C1–C4 PASS (`raw/label_clock/label_clock_record.json`).

**Inference seed 0**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7169 [0.6427, 0.7857] · 0.2418 (1141) ⭐ | 0.7134 [0.6392, 0.7821] · 0.2300 (1106) |
| LAT | z_tac_v7_heads | 0.7038 [0.6279, 0.7759] · 0.1975 (1141) ⭐ | 0.7016 [0.6268, 0.7749] · 0.1937 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.6713 [0.5925, 0.7483] · 0.0066 (1141) ⭐ | 0.6718 [0.5928, 0.7489] · 0.0068 (1106) |
| LON | v6_behaviour_decoder | 0.3707 [0.3003, 0.4432] · 0.1821 (1141) ⭐ | 0.3680 [0.2965, 0.4418] · 0.1786 (1106) |
| LON | z_tac_v7_heads | 0.3883 [0.3173, 0.4618] · 0.2131 (1141) ⭐ | 0.3843 [0.3140, 0.4587] · 0.2083 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.3655 [0.2948, 0.4378] · 0.1790 (1141) ⭐ | 0.3644 [0.2924, 0.4370] · 0.1769 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7372 / SCOREABLE | 882 / 0.7369 / SCOREABLE |
| TURN_L | 40 / 0.9751 / UNSCOREABLE (n_pos < 200) | 40 / 0.9769 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9571 / UNSCOREABLE (n_pos < 200) | 63 / 0.9532 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9939 / UNSCOREABLE (n_pos < 200) | 16 / 0.9942 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8648 / UNSCOREABLE (n_pos < 200) | 73 / 0.8620 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.0952 / UNSCOREABLE (n_pos < 200) | 48 / 0.1042 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.6914 / UNSCOREABLE (n_pos < 200) | 8 / 0.6937 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.4499 / UNSCOREABLE (n_pos < 200) | 40 / 0.4520 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 1.0000 / UNSCOREABLE (n_pos < 200) | 40 / 1.0000 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.5646 / UNSCOREABLE (n_pos < 200) | 63 / 0.5544 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6526 / UNSCOREABLE (n_pos < 200) | 24 / 0.6485 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7104 / UNSCOREABLE (n_pos < 200) | 118 / 0.7155 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

**Inference seed 1**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7169 [0.6427, 0.7857] · 0.2418 (1141) ⭐ | 0.7134 [0.6392, 0.7821] · 0.2300 (1106) |
| LAT | z_tac_v7_heads | 0.7038 [0.6279, 0.7759] · 0.1975 (1141) ⭐ | 0.7016 [0.6268, 0.7749] · 0.1937 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.6713 [0.5925, 0.7483] · 0.0066 (1141) ⭐ | 0.6718 [0.5928, 0.7489] · 0.0068 (1106) |
| LON | v6_behaviour_decoder | 0.3707 [0.3003, 0.4432] · 0.1821 (1141) ⭐ | 0.3680 [0.2965, 0.4418] · 0.1786 (1106) |
| LON | z_tac_v7_heads | 0.3883 [0.3173, 0.4618] · 0.2131 (1141) ⭐ | 0.3843 [0.3140, 0.4587] · 0.2083 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.3655 [0.2948, 0.4378] · 0.1790 (1141) ⭐ | 0.3644 [0.2924, 0.4370] · 0.1769 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7372 / SCOREABLE | 882 / 0.7369 / SCOREABLE |
| TURN_L | 40 / 0.9751 / UNSCOREABLE (n_pos < 200) | 40 / 0.9769 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9571 / UNSCOREABLE (n_pos < 200) | 63 / 0.9532 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9939 / UNSCOREABLE (n_pos < 200) | 16 / 0.9942 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8648 / UNSCOREABLE (n_pos < 200) | 73 / 0.8620 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.0952 / UNSCOREABLE (n_pos < 200) | 48 / 0.1042 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.6914 / UNSCOREABLE (n_pos < 200) | 8 / 0.6937 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.4499 / UNSCOREABLE (n_pos < 200) | 40 / 0.4520 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 1.0000 / UNSCOREABLE (n_pos < 200) | 40 / 1.0000 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.5646 / UNSCOREABLE (n_pos < 200) | 63 / 0.5544 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6526 / UNSCOREABLE (n_pos < 200) | 24 / 0.6485 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7104 / UNSCOREABLE (n_pos < 200) | 118 / 0.7155 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

⭐ = the PRIMARY clock for this checkpoint. A tactical comparison across step 34,500 must show both clocks, and it mixes training time with the fix.
