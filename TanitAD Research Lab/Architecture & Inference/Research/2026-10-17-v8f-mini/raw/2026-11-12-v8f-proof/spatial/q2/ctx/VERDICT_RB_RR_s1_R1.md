# VERDICT r-b on the trained model: RR_s1_R1

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1948, "windows_dropped": 0, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.027064214891074023, "ii": -0.027213258705986343, "iii": -0.03569790534592587, "iv": -0.020292262480175793, "ivb": -0.06145995107337576}}
- t4_ceiling: PASS :: {"r2_i": 0.4906928648741772, "r2_ii": 0.45022499760907886, "r2_iii": 0.5260294975439679}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999098600791013, "corr_start": 0.999921566094253, "bar": 0.999, "max_abs_state_diff_future": 0.24508410692214966, "max_abs_state_diff_start": 0.2691798210144043}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.07580449993488902}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 471 of 1948 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.491 [+0.352, +0.600]
- (ii) present state (copy): +0.450 [+0.306, +0.565]
- (iii) true future state (ceiling): +0.526 [+0.392, +0.629]
- (iv) ego only, start ego vector plus actions: +0.038 [-0.388, +0.299]
- (iv-b) ego only, bank probe on the future ego vector: +0.186 [-0.198, +0.419]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: +0.040 [-0.007, +0.088]
- (i) minus the ego floor: +0.305 [+0.074, +0.650] (separated above 0)
- (ii) minus the ego floor: +0.265 [+0.025, +0.607] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.035 [-0.011, +0.083]
- (iii) minus (ii), what the future state adds over the copy: +0.076 [+0.008, +0.143] (separated above 0)

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
