### TACTICAL under both label clocks (SPEC A5): step30000, step 30000. PRIMARY clock: **OLD**

Clock and dt source: CORRECTED = grid_start + (t + w - 1 + n_stack - 1) * dt; 136/139 eval clips on the sidecar clock (dt median 0.10066662995966351), 3 on nominal dt 0.1 s with grid_start 0, 0 on pose dt; OLD = (t + w - 1) * 0.1 s.

Tier T1 (UNRULED for an action-free model). Declared-head accuracy [episode-cluster CI], κ full-set; in-band n per clock. Label tables sha256 `83b47aae4aff…`; controls C1–C4 PASS (`raw/label_clock/label_clock_record.json`).

**Inference seed 0**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7546 [0.7002, 0.8121] · 0.5181 (1141) ⭐ | 0.7514 [0.6959, 0.8105] · 0.5083 (1106) |
| LAT | z_tac_v7_heads | 0.7117 [0.6427, 0.7791] · 0.3242 (1141) ⭐ | 0.7107 [0.6417, 0.7796] · 0.3143 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.7187 [0.6541, 0.7764] · 0.4081 (1141) ⭐ | 0.7134 [0.6492, 0.7729] · 0.3964 (1106) |
| LON | v6_behaviour_decoder | 0.4943 [0.4252, 0.5615] · 0.3400 (1141) ⭐ | 0.4901 [0.4224, 0.5575] · 0.3354 (1106) |
| LON | z_tac_v7_heads | 0.5320 [0.4592, 0.6009] · 0.3828 (1141) ⭐ | 0.5289 [0.4551, 0.5969] · 0.3791 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.4514 [0.3817, 0.5184] · 0.2925 (1141) ⭐ | 0.4458 [0.3736, 0.5118] · 0.2853 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7999 / SCOREABLE | 882 / 0.7980 / SCOREABLE |
| TURN_L | 40 / 0.9881 / UNSCOREABLE (n_pos < 200) | 40 / 0.9883 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9695 / UNSCOREABLE (n_pos < 200) | 63 / 0.9684 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9983 / UNSCOREABLE (n_pos < 200) | 16 / 0.9975 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8598 / UNSCOREABLE (n_pos < 200) | 73 / 0.8547 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.9546 / UNSCOREABLE (n_pos < 200) | 48 / 0.9505 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.9889 / UNSCOREABLE (n_pos < 200) | 8 / 0.9940 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.6168 / UNSCOREABLE (n_pos < 200) | 40 / 0.6092 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 0.8915 / UNSCOREABLE (n_pos < 200) | 40 / 0.8919 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.8759 / UNSCOREABLE (n_pos < 200) | 63 / 0.8831 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6340 / UNSCOREABLE (n_pos < 200) | 24 / 0.6335 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7087 / UNSCOREABLE (n_pos < 200) | 118 / 0.6974 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

**Inference seed 1**

| head | surface | OLD clock: acc [CI] · κ (n in band) | CORRECTED clock: acc [CI] · κ (n in band) |
|---|---|---|---|
| LAT | v6_behaviour_decoder | 0.7546 [0.7002, 0.8121] · 0.5181 (1141) ⭐ | 0.7514 [0.6959, 0.8105] · 0.5083 (1106) |
| LAT | z_tac_v7_heads | 0.7117 [0.6427, 0.7791] · 0.3242 (1141) ⭐ | 0.7107 [0.6417, 0.7796] · 0.3143 (1106) |
| LAT | v6_behaviour_decoder_NAVZERO | 0.7187 [0.6541, 0.7764] · 0.4081 (1141) ⭐ | 0.7134 [0.6492, 0.7729] · 0.3964 (1106) |
| LON | v6_behaviour_decoder | 0.4943 [0.4252, 0.5615] · 0.3400 (1141) ⭐ | 0.4901 [0.4224, 0.5575] · 0.3354 (1106) |
| LON | z_tac_v7_heads | 0.5320 [0.4592, 0.6009] · 0.3828 (1141) ⭐ | 0.5289 [0.4551, 0.5969] · 0.3791 (1106) |
| LON | v6_behaviour_decoder_NAVZERO | 0.4514 [0.3817, 0.5184] · 0.2925 (1141) ⭐ | 0.4458 [0.3736, 0.5118] · 0.2853 (1106) |

| goal token (nav true) | OLD: n_pos / AUROC / status | CORRECTED: n_pos / AUROC / status |
|---|---|---|
| FOLLOW_LANE | 911 / 0.7999 / SCOREABLE | 882 / 0.7980 / SCOREABLE |
| TURN_L | 40 / 0.9881 / UNSCOREABLE (n_pos < 200) | 40 / 0.9883 / UNSCOREABLE (n_pos < 200) |
| TURN_R | 67 / 0.9695 / UNSCOREABLE (n_pos < 200) | 63 / 0.9684 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_L | 16 / 0.9983 / UNSCOREABLE (n_pos < 200) | 16 / 0.9975 / UNSCOREABLE (n_pos < 200) |
| YIELD_FOR_TURN_R | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| YIELD | 139 / — / UNSCOREABLE (n_pos < 200) | 136 / — / UNSCOREABLE (n_pos < 200) |
| STOP_POINT | 74 / 0.8598 / UNSCOREABLE (n_pos < 200) | 73 / 0.8547 / UNSCOREABLE (n_pos < 200) |
| SPEED_BAND | 1141 / — / SCOREABLE | 1106 / — / SCOREABLE |
| CORRIDOR_OFFSET | 200 / — / SCOREABLE | 193 / — / UNSCOREABLE (n_pos < 200) |
| EVADE_IN_CORRIDOR | 49 / 0.9546 / UNSCOREABLE (n_pos < 200) | 48 / 0.9505 / UNSCOREABLE (n_pos < 200) |
| OVERTAKE_VEHICLE | 9 / 0.9889 / UNSCOREABLE (n_pos < 200) | 8 / 0.9940 / UNSCOREABLE (n_pos < 200) |
| MERGE | 40 / 0.6168 / UNSCOREABLE (n_pos < 200) | 40 / 0.6092 / UNSCOREABLE (n_pos < 200) |
| GAP_TARGET | 97 / — / UNSCOREABLE (n_pos < 200) | 95 / — / UNSCOREABLE (n_pos < 200) |
| REACT_ON_ONCOMING | 82 / — / UNSCOREABLE (n_pos < 200) | 80 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TAKE_EXIT_R | 41 / 0.8915 / UNSCOREABLE (n_pos < 200) | 40 / 0.8919 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_RED | 65 / 0.8759 / UNSCOREABLE (n_pos < 200) | 63 / 0.8831 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_YELLOW | 25 / 0.6340 / UNSCOREABLE (n_pos < 200) | 24 / 0.6335 / UNSCOREABLE (n_pos < 200) |
| TRAFFIC_LIGHT_REACT_GREEN | 122 / 0.7087 / UNSCOREABLE (n_pos < 200) | 118 / 0.6974 / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_L | 0 / — / UNSCOREABLE (n_pos < 200) | 0 / — / UNSCOREABLE (n_pos < 200) |
| LANE_CHANGE_R | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) | 8 / 1.0000 / UNSCOREABLE (n_pos < 200) |

⚠ Nav-echo caveat (Master Mind register REFCV6-BATTERY-5K): on PhysicalAI the nav input is derived from the ego's own future path, so every nav-true LAT κ here is optimistic by construction (the nav-echo family); the NAVZERO rows are the nav-free reading.

⭐ = the PRIMARY clock for this checkpoint. A tactical comparison across step 34,500 must show both clocks, and it mixes training time with the fix.
