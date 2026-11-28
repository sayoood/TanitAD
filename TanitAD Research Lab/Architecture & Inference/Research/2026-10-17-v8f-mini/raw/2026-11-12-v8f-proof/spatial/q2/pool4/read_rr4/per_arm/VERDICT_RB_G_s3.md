# VERDICT r-b on the trained model: G_s3

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1809, "windows_dropped": 139, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.02841157422329954, "ii": -0.03799356974874457, "iii": -0.03888988445636832, "iv": -0.03817151404316421, "ivb": -0.026933820737470526}}
- t4_ceiling: PASS :: {"r2_i": 0.48117035157857546, "r2_ii": 0.4979961535151304, "r2_iii": 0.5494787002525778}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999338248751094, "corr_start": 0.9999346227911998, "bar": 0.999, "max_abs_state_diff_future": 0.24895519018173218, "max_abs_state_diff_start": 0.24646544456481934}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.051482546737447454}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 441 of 1809 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.481 [+0.279, +0.630]
- (ii) present state (copy): +0.498 [+0.325, +0.626]
- (iii) true future state (ceiling): +0.549 [+0.393, +0.659]
- (iv) ego only, start ego vector plus actions: +0.081 [-0.357, +0.346]
- (iv-b) ego only, bank probe on the future ego vector: +0.179 [-0.211, +0.421]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: -0.017 [-0.091, +0.052]
- (i) minus the ego floor: +0.302 [+0.081, +0.605] (separated above 0)
- (ii) minus the ego floor: +0.319 [+0.092, +0.639] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.068 [-0.002, +0.143]
- (iii) minus (ii), what the future state adds over the copy: +0.051 [-0.019, +0.122]

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
