# VERDICT r-b on the trained model: P_s1_R3g

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1948, "windows_dropped": 0, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.020979829534147898, "ii": -0.02337016534919889, "iii": -0.03352391802807264, "iv": -0.020292262480175793, "ivb": -0.06145995107337576}}
- t4_ceiling: PASS :: {"r2_i": 0.51201448611729, "r2_ii": 0.4783835756345851, "r2_iii": 0.5478591454733045}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999392387472592, "corr_start": 0.9999403385800572, "bar": 0.999, "max_abs_state_diff_future": 0.23835766315460205, "max_abs_state_diff_start": 0.21168527007102966}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.06947556983871939}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 471 of 1948 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.512 [+0.368, +0.622]
- (ii) present state (copy): +0.478 [+0.320, +0.602]
- (iii) true future state (ceiling): +0.548 [+0.409, +0.646]
- (iv) ego only, start ego vector plus actions: +0.038 [-0.388, +0.299]
- (iv-b) ego only, bank probe on the future ego vector: +0.186 [-0.198, +0.419]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: +0.034 [-0.006, +0.078]
- (i) minus the ego floor: +0.326 [+0.104, +0.646] (separated above 0)
- (ii) minus the ego floor: +0.293 [+0.067, +0.617] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.036 [-0.014, +0.092]
- (iii) minus (ii), what the future state adds over the copy: +0.069 [+0.006, +0.138] (separated above 0)

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
