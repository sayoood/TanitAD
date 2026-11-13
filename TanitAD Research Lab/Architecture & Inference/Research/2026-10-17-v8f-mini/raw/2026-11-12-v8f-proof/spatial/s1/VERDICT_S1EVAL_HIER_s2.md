# VERDICT_S1EVAL_HIER_s2: note N8, r-a read of a trained state on the eval bank
Tier T0 (probes on a trained state; a world-model diagnostic, never driving performance). Where: eval bank frames, out-of-sample for the model. Estimator: episode-cluster bootstrap, B 2000, seed 0, one draw shared; folds by episode (5, seed 0).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass)

## CONTROLS (first)
- v1: PASS :: {"bank_rows": 9321, "missing": 0, "duplicate_state_keys": 0, "pass": true}
- v4: PASS :: {"width": 128, "expected": 128, "record_state_mode": "mean", "record_ckpt_sha256": "e1d6317081c0ead5d79ca4dedaadf9fdd80c6140510642c0790a36d221c33bab", "ckpt_checked": true, "pass": true}
- v2: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.1327, "chosen": "linear", "bar": "equal at 4 decimals", "pass": true}
- v3: PASS :: {"state_gap_r2_permuted": -0.022875972639652353, "bar": 0.02, "pass": true}
- controls_pass: True

## Reads (trained state, width 128; 139 episodes, 9321 frames, lead gap on 2485); label: trained state, tag HIER_s2; no outcome name of the card
- lead gap R2: state {"r2": 0.14274631052691222, "ci95": [-0.19114147351953684, 0.3935701490284838], "n": 2485}; ego floor (linear) {"r2": 0.13274532858514854, "ci95": [-0.20927207753577948, 0.36662808755865817], "n": 2485}
- state minus ego, gap R2: {"diff": 0.010000981941763687, "ci95": [-0.26704892460843027, 0.29690185743032477], "separated_pos": false, "separated_neg": false}
- lead presence AUROC: state {"auc": 0.7917989916517838, "ci95": [0.7122978693762353, 0.8581183356787949], "n": 8838}; ego floor (linear) {"auc": 0.5958643724459143, "ci95": [0.5052374011198562, 0.685234724127315], "n": 8838}
- state minus ego, presence AUROC: {"diff": 0.19593461920586952, "ci95": [0.09686805616908259, 0.29423243643249003], "separated_pos": true, "separated_neg": false}
- beside, not paired (different widths), the untrained states of stage 0b on the same frames: {"grid_gap_r2": 0.3981, "mean_gap_r2": 0.2577, "ego_gap_r2": 0.1327, "grid_presence_auc": 0.761, "mean_presence_auc": 0.742, "ego_presence_auc": 0.596}
