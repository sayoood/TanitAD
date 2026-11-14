# VERDICT r-b on the trained model: W_s0

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1809, "windows_dropped": 139, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121578202445703, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.03860153933041577, "ii": -0.0485009255962896, "iii": -0.03612899043252149, "iv": -0.0381715140431631, "ivb": -0.026933820737489844}}
- t4_ceiling: PASS :: {"r2_i": 0.2388675773130935, "r2_ii": 0.3183617076868728, "r2_iii": 0.34009181471677974}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999061243286539, "corr_start": 0.9999123990448371, "bar": 0.999, "max_abs_state_diff_future": 0.105712890625, "max_abs_state_diff_start": 0.09375}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.021730107029906964}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 441 of 1809 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.239 [-0.071, +0.451]
- (ii) present state (copy): +0.318 [+0.014, +0.526]
- (iii) true future state (ceiling): +0.340 [+0.073, +0.535]
- (iv) ego only, start ego vector plus actions: +0.081 [-0.357, +0.346]
- (iv-b) ego only, bank probe on the future ego vector: +0.179 [-0.211, +0.421]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: -0.079 [-0.226, +0.052]
- (i) minus the ego floor: +0.060 [-0.161, +0.338]
- (ii) minus the ego floor: +0.140 [-0.095, +0.440]
- (iii) minus (i), headroom of the rollout: +0.101 [-0.045, +0.259]
- (iii) minus (ii), what the future state adds over the copy: +0.022 [-0.062, +0.112]

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
