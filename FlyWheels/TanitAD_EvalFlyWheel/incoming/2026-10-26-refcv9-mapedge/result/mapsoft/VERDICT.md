FAIL (M1, M2, M3, M4, M5)

# MAPSOFT (mechanical; PREREG_MAPSOFT sha256 38c77b174122)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); estimator: paired episode-cluster bootstrap B 2000 seed 0 over EPISODES only; one training seed; one inference seed

| arm (S6c) | sidewalk | edge contact | lane cross |
|---|---|---|---|
| BASE | 0.1792 | 0.2855 | 0.2939 |
| HUMAN | 0.0125 | 0.1485 | 0.1941 |
| GT-T1 | 0.0166 | 0.2148 | 0.2721 |
| OWN-T1 | 0.1657 | 0.2708 | 0.2908 |
| MAPSOFT-XF | 0.1789 | 0.2842 | 0.2942 |
| DER-MAPSOFT-XF | 0.1757 | 0.2824 | 0.2933 |

M1 sidewalk vs BASE -0.0003 [-0.0015, +0.0006] -> False
M2 edge vs BASE -0.0013 [-0.0031, +0.0000] -> False
M3 no four-family row worse -> False ['S6c_stage:ade_m', 'S6c_stage:fde_m', 'S6c_stage:LON_speed_mae_mps', 'S6c_stage:LON_along_mae_m', 'S6c_stage:LON_along_abs_final_m', 'S6c_stage:LON_accel_mae_mps2', 'S2c_stage:ade_m', 'S2c_stage:fde_m']
M4 deranged not below BASE (sidewalk -0.0034 [-0.0066, -0.0006], edge -0.0031 [-0.0057, -0.0006]) -> False
M5 STOP guard -> False {"path_len_m": {"mean": 67.70822581402862, "n": 3251, "vs_BASE": {"delta": -0.0183, "lo": -0.0395, "hi": -0.0012, "separated": true, "n_windows": 3251, "n_episodes": 138, "mean_new": 67.70822581402862, "mean_ref": 67.72654502790304}}, "stop_share": {"mean": 0.026760996616425715, "n": 3251, "vs_BASE"

fitted lambda per fold (own): {"0": [0.25, 0.0], "1": [0.25, 0.0], "2": [0.0, 0.0], "3": [0.0, 0.0], "4": [0.0, 0.0]}
fitted lambda per fold (deranged): {"0": [0.5, 0.0], "1": [0.5, 0.0], "2": [0.0, 0.0], "3": [0.5, 0.0], "4": [0.5, 0.0]}
share of GT-T1's sidewalk gain: 0.0019230769230769204
human-relative path length (DESCRIPTIVE): {"MAPSOFT-XF": {"mean_abs_dev_m": 5.399264802466929, "vs_BASE": {"delta": 0.0237, "lo": 0.0071, "hi": 0.0432, "separated": true, "n_windows": 3198, "n_episodes": 136, "mean_new": 5.39926480246693, "mean_ref": 5.375562324353096}}, "GT-T1": {"mean_abs_dev_m": 5.404592520964943, "vs_BASE": {"delta": 0.029, "lo": -0.1141, "hi": 0.1752, "separated": false, "n_windows": 3198, "n_episodes": 136, "mean_ne

controls failed: none

NEXT: own-map precision binds (the fit returned (0, 0) in >= 3 folds): perception training of the map head (sidewalk / boundary precision) -- compute, a PI decision.

ZZMSOFTENDZZ