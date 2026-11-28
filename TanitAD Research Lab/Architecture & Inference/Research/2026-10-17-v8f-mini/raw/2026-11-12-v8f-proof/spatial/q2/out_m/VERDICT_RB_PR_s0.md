# VERDICT r-b on the trained model: PR_s0

Tier T0 (a world-model diagnostic on logged frames, never driving performance). Card PREREG_V8F_PROOF_ADD12.md (Q2, stage 1), definitions of stage 0 (S0_DEFS R8, R19).
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Controls (first)

- t1_bank_rows_have_states: PASS :: {"bank_rows": 9321, "bank_rows_without_state": 0, "episodes": 139, "windows_in_latents": 1948, "windows_used": 1809, "windows_dropped": 139, "state_width": 640}
- t2_ego_floor_bank: PASS :: {"ego_gap_r2": 0.13274532858514854, "reference": 0.133, "tol": 0.03, "chosen": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- t3_permuted_labels: PASS :: {"bar": 0.02, "values": {"i": -0.03170218914385181, "ii": -0.03684440167546876, "iii": -0.0376708973967288, "iv": -0.03817151404316421, "ivb": -0.026933820737470526}}
- t4_ceiling: PASS :: {"r2_i": 0.4415436622188693, "r2_ii": 0.4396467512603144, "r2_iii": 0.49759655410302084}
- t5_state_sources_agree: PASS :: {"corr_future": 0.9999271222592387, "corr_start": 0.999930341926314, "bar": 0.999, "max_abs_state_diff_future": 0.26433420181274414, "max_abs_state_diff_start": 0.24816083908081055}
- t6_known_values: PASS :: {"copy_as_rollout_diff": 0.0, "true_as_rollout_diff": 0.057949802842706455}
- controls_pass: True

## Lead gap at +2.0 s (windows with a lead at both times, n 441 of 1809 windows, 139 episodes; R2 with episode-cluster bootstrap, B 2000)

- (i) rolled-out state: +0.442 [+0.255, +0.581]
- (ii) present state (copy): +0.440 [+0.254, +0.580]
- (iii) true future state (ceiling): +0.498 [+0.326, +0.617]
- (iv) ego only, start ego vector plus actions: +0.081 [-0.357, +0.346]
- (iv-b) ego only, bank probe on the future ego vector: +0.179 [-0.211, +0.421]; floor used: ivb

Paired over episodes:
- (i) minus (ii), rollout against copy: +0.002 [-0.062, +0.065]
- (i) minus the ego floor: +0.263 [+0.068, +0.554] (separated above 0)
- (ii) minus the ego floor: +0.261 [+0.050, +0.559] (separated above 0)
- (iii) minus (i), headroom of the rollout: +0.056 [-0.023, +0.142]
- (iii) minus (ii), what the future state adds over the copy: +0.058 [-0.013, +0.130]

**Outcome at this seed (the card's name needs both seeds): IMAGINE-FLAT**

Limits: one training run; the probe is fitted on bank frames of the training folds (139 episodes); a separated interval is necessary and not sufficient.
