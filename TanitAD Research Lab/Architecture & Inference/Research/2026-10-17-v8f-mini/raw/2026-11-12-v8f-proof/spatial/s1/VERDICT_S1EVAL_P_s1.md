# VERDICT_S1EVAL_P_s1: note N8, r-a read of a trained state on the eval bank
Tier T0 (probes on a trained state; a world-model diagnostic, never driving performance). Where: eval bank frames, out-of-sample for the model. Estimator: episode-cluster bootstrap, B 2000, seed 0, one draw shared; folds by episode (5, seed 0).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass)

## CONTROLS (first)
- v1: PASS :: {"bank_rows": 9321, "missing": 0, "duplicate_state_keys": 0, "pass": true}
- v4: PASS :: {"width": 640, "expected": 640, "record_state_mode": "grid", "record_ckpt_sha256": "64930fa330e938f7d1db07e8d07169750fe92ca3ff0fe69a80633c782f95864e", "ckpt_checked": true, "pass": true}
- v2: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.1327, "chosen": "linear", "bar": "equal at 4 decimals", "pass": true}
- v3: PASS :: {"state_gap_r2_permuted": -0.025995722060436544, "bar": 0.02, "pass": true}
- controls_pass: True

## Reads (trained state, width 640; 139 episodes, 9321 frames, lead gap on 2485); label: trained state, tag P_s1; no outcome name of the card
- lead gap R2: state {"r2": 0.5419475029094372, "ci95": [0.41284953933435314, 0.6318682006236878], "n": 2485}; ego floor (linear) {"r2": 0.13274532858514854, "ci95": [-0.20927207753577948, 0.36662808755865817], "n": 2485}
- state minus ego, gap R2: {"diff": 0.4092021743242886, "ci95": [0.19129353208833574, 0.7245697920134766], "separated_pos": true, "separated_neg": false}
- lead presence AUROC: state {"auc": 0.8862473756437571, "ci95": [0.8402024786705138, 0.9222140459861421], "n": 8838}; ego floor (linear) {"auc": 0.5958643724459143, "ci95": [0.5052374011198562, 0.685234724127315], "n": 8838}
- state minus ego, presence AUROC: {"diff": 0.2903830031978428, "ci95": [0.1960698264960354, 0.3854319383831433], "separated_pos": true, "separated_neg": false}
- beside, not paired (different widths), the untrained states of stage 0b on the same frames: {"grid_gap_r2": 0.3981, "mean_gap_r2": 0.2577, "ego_gap_r2": 0.1327, "grid_presence_auc": 0.761, "mean_presence_auc": 0.742, "ego_presence_auc": 0.596}
