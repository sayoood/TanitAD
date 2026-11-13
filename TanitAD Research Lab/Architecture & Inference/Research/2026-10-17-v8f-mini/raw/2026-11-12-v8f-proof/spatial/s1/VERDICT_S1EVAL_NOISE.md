# VERDICT_S1EVAL_NOISE: note N8, r-a read of a trained state on the eval bank
Tier T0 (probes on a trained state; a world-model diagnostic, never driving performance). Where: eval bank frames, out-of-sample for the model. Estimator: episode-cluster bootstrap, B 2000, seed 0, one draw shared; folds by episode (5, seed 0).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass)

## CONTROLS (first)
- v1: PASS :: {"bank_rows": 9321, "missing": 0, "duplicate_state_keys": 0, "pass": true}
- v4: PASS :: {"width": 128, "expected": 128, "record_state_mode": null, "record_ckpt_sha256": null, "ckpt_checked": false, "pass": true}
- v2: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.1327, "chosen": "linear", "bar": "equal at 4 decimals", "pass": true}
- v3: PASS :: {"state_gap_r2_permuted": -0.014532125144943508, "bar": 0.02, "pass": true}
- controls_pass: True

## Reads (trained state, width 128; 139 episodes, 9321 frames, lead gap on 2485); label: trained state, tag NOISE; no outcome name of the card
- lead gap R2: state {"r2": -0.013390159542899571, "ci95": [-0.09406504762447361, 2.5245544390807545e-05], "n": 2485}; ego floor (linear) {"r2": 0.13274532858514854, "ci95": [-0.20927207753577948, 0.36662808755865817], "n": 2485}
- state minus ego, gap R2: {"diff": -0.1461354881280481, "ci95": [-0.39484380252570356, 0.18542693401689714], "separated_pos": false, "separated_neg": false}
- lead presence AUROC: state {"auc": 0.4565171605740218, "ci95": [0.40047269655576667, 0.5197120290650594], "n": 8838}; ego floor (linear) {"auc": 0.5958643724459143, "ci95": [0.5052374011198562, 0.685234724127315], "n": 8838}
- state minus ego, presence AUROC: {"diff": -0.13934721187189247, "ci95": [-0.23515012118254427, -0.04546517357308474], "separated_pos": false, "separated_neg": true}
- beside, not paired (different widths), the untrained states of stage 0b on the same frames: {"grid_gap_r2": 0.3981, "mean_gap_r2": 0.2577, "ego_gap_r2": 0.1327, "grid_presence_auc": 0.761, "mean_presence_auc": 0.742, "ego_presence_auc": 0.596}
