# VERDICT r-b on the trained model: RR_s3

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1809, "windows_dropped": 139, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.03632014495785141, "ii": -0.04104745960486422, "iii": -0.04202317739891348, "iv": -0.03817151404316421, "ivb": -0.026933820737470526}}
- t4_ceiling: PASS :: {"r2_i": 0.4655033295614599, "r2_ii": 0.4616991954662715, "r2_iii": 0.5279995759890763}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999057728720295, "corr_start": 0.999909329512483, "bar": 0.999, "max_abs_state_diff_future": 0.22075331211090088, "max_abs_state_diff_start": 0.22521024383604527}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.06630038052280485}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 441 of 1809 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.466 [+0.290, +0.599]
- (ii) present state (copy): +0.462 [+0.291, +0.590]
- (iii) true future state (ceiling): +0.528 [+0.366, +0.645]
- (iv) ego only, start ego vector plus actions: +0.081 [-0.357, +0.346]
- (iv-b) ego only, bank probe on the future ego vector: +0.179 [-0.211, +0.421]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: +0.004 [-0.034, +0.041]
- (i) minus the ego floor: +0.287 [+0.058, +0.607] (separated above 0)
- (ii) minus the ego floor: +0.283 [+0.058, +0.608] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.062 [+0.011, +0.118] (separated above 0)
- (iii) minus (ii), what the future state adds over the copy: +0.066 [+0.008, +0.126] (separated above 0)

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
