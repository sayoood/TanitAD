# VERDICT_S1EVAL_G_s0: note N8, r-a read of a trained state on the eval bank
Tier T0 (probes on a trained state; a world-model diagnostic, never driving performance). Where: eval bank frames, out-of-sample for the model. Estimator: episode-cluster bootstrap, B 2000, seed 0, one draw shared; folds by episode (5, seed 0).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass)

## CONTROLS (first)
- v1: PASS :: {"bank_rows": 9321, "missing": 0, "duplicate_state_keys": 0, "pass": true}
- v4: PASS :: {"width": 640, "expected": 640, "record_state_mode": "grid", "record_ckpt_sha256": "40481794b113835f747eb2c2263a0ff2a2689d1de55ce9c627ae77c8ccec31b1", "ckpt_checked": true, "pass": true}
- v2: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.1327, "chosen": "linear", "bar": "equal at 4 decimals", "pass": true}
- v3: PASS :: {"state_gap_r2_permuted": -0.020634308103322008, "bar": 0.02, "pass": true}
- controls_pass: True

## Reads (trained state, width 640; 139 episodes, 9321 frames, lead gap on 2485); label: trained state, tag G_s0; no outcome name of the card
- lead gap R2: state {"r2": 0.5521115458333795, "ci95": [0.4190226679713025, 0.651330387625021], "n": 2485}; ego floor (linear) {"r2": 0.13274532858514854, "ci95": [-0.20927207753577948, 0.36662808755865817], "n": 2485}
- state minus ego, gap R2: {"diff": 0.419366217248231, "ci95": [0.21234448859099905, 0.7076453535471556], "separated_pos": true, "separated_neg": false}
- lead presence AUROC: state {"auc": 0.8595531001212691, "ci95": [0.8017086897702029, 0.9058500358958934], "n": 8838}; ego floor (linear) {"auc": 0.5958643724459143, "ci95": [0.5052374011198562, 0.685234724127315], "n": 8838}
- state minus ego, presence AUROC: {"diff": 0.2636887276753548, "ci95": [0.1694098576268541, 0.3583567165296474], "separated_pos": true, "separated_neg": false}
- beside, not paired (different widths), the untrained states of stage 0b on the same frames: {"grid_gap_r2": 0.3981, "mean_gap_r2": 0.2577, "ego_gap_r2": 0.1327, "grid_presence_auc": 0.761, "mean_presence_auc": 0.742, "ego_presence_auc": 0.596}
