# VERDICT r-b on the trained model: ROLL_s0

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1809, "windows_dropped": 139, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.03160421456863882, "ii": -0.039690205817580315, "iii": -0.03965335959892036, "iv": -0.03817151404316421, "ivb": -0.026933820737470526}}
- t4_ceiling: PASS :: {"r2_i": 0.4307798744757654, "r2_ii": 0.44968241788246766, "r2_iii": 0.47454607940198124}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999131970366922, "corr_start": 0.9999105580067315, "bar": 0.999, "max_abs_state_diff_future": 0.2817414402961731, "max_abs_state_diff_start": 0.23271560668945312}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.02486366151951358}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 441 of 1809 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.431 [+0.217, +0.580]
- (ii) present state (copy): +0.450 [+0.296, +0.567]
- (iii) true future state (ceiling): +0.475 [+0.317, +0.600]
- (iv) ego only, start ego vector plus actions: +0.081 [-0.357, +0.346]
- (iv-b) ego only, bank probe on the future ego vector: +0.179 [-0.211, +0.421]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: -0.019 [-0.123, +0.068]
- (i) minus the ego floor: +0.252 [+0.052, +0.519] (separated above 0)
- (ii) minus the ego floor: +0.271 [+0.055, +0.591] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.044 [-0.032, +0.134]
- (iii) minus (ii), what the future state adds over the copy: +0.025 [-0.041, +0.086]

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
