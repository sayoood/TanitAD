# VERDICT r-b on the trained model: RES_s1

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1809, "windows_dropped": 139, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.0573520481943246, "ii": -0.06142804407939595, "iii": -0.06184323726513563, "iv": -0.03817151404316421, "ivb": -0.026933820737470526}}
- t4_ceiling: PASS :: {"r2_i": 0.48646607956781385, "r2_ii": 0.4747462380763636, "r2_iii": 0.5401998714355105}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999347247607271, "corr_start": 0.9999398343105651, "bar": 0.999, "max_abs_state_diff_future": 0.2294139564037323, "max_abs_state_diff_start": 0.2461256980895996}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.06545363335914689}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 441 of 1809 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.486 [+0.328, +0.600]
- (ii) present state (copy): +0.475 [+0.327, +0.589]
- (iii) true future state (ceiling): +0.540 [+0.399, +0.640]
- (iv) ego only, start ego vector plus actions: +0.081 [-0.357, +0.346]
- (iv-b) ego only, bank probe on the future ego vector: +0.179 [-0.211, +0.421]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: +0.012 [-0.037, +0.061]
- (i) minus the ego floor: +0.308 [+0.100, +0.608] (separated above 0)
- (ii) minus the ego floor: +0.296 [+0.078, +0.612] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.054 [-0.000, +0.107]
- (iii) minus (ii), what the future state adds over the copy: +0.065 [+0.003, +0.126] (separated above 0)

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
