# VERDICT r-b on the trained model: PF_s1

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1809, "windows_dropped": 139, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.04190878053548208, "ii": -0.03456750605901804, "iii": -0.03947736339399932, "iv": -0.03817151404316421, "ivb": -0.026933820737470526}}
- t4_ceiling: PASS :: {"r2_i": 0.40941751300810525, "r2_ii": 0.5212284323801413, "r2_iii": 0.5962042423926599}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999378331148249, "corr_start": 0.9999399137181915, "bar": 0.999, "max_abs_state_diff_future": 0.2560698986053467, "max_abs_state_diff_start": 0.2519431710243225}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.07497581001251863}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 441 of 1809 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.409 [+0.180, +0.571]
- (ii) present state (copy): +0.521 [+0.342, +0.646]
- (iii) true future state (ceiling): +0.596 [+0.436, +0.705]
- (iv) ego only, start ego vector plus actions: +0.081 [-0.357, +0.346]
- (iv-b) ego only, bank probe on the future ego vector: +0.179 [-0.211, +0.421]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: -0.112 [-0.220, -0.027] (separated below 0)
- (i) minus the ego floor: +0.231 [+0.005, +0.529] (separated above 0)
- (ii) minus the ego floor: +0.343 [+0.121, +0.673] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.187 [+0.094, +0.302] (separated above 0)
- (iii) minus (ii), what the future state adds over the copy: +0.075 [+0.010, +0.141] (separated above 0)

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
