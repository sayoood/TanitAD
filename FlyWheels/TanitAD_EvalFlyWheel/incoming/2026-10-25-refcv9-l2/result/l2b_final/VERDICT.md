# L2-B VERDICT: SELECTION

PREREG_L2B sha256 089d7ab67a9c9f9600d84399e30d8a49e2e48f710bea7dc9e941e4e41382a7a0
tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: 'another draw of episodes' only; NOT another training run, NOT another inference run (seed 1 = replicate, never pooled)

* seed 0 (primary): **SELECTION** -- basis R2 (GOOD-cruise existence; the class-level existence is saturated in FB and DR); R1 SELECTION, R2 SELECTION, existence saturated True
* seed 1 (replicate): **SELECTION**

| set (seed 0) | n | EXIST target-class cand [CI] | GOOD (ADE<=tau) [CI] | pre-blend pick is target class |
|---|---|---|---|---|
| FB | 259 | 1.000 [1.000, 1.000] | 0.981 [0.956, 0.997] | 0.147 |
| CT | 259 | 1.000 [1.000, 1.000] | 0.973 [0.945, 0.993] | 0.958 |
| DR | 339 | 1.000 [1.000, 1.000] | 0.434 [0.346, 0.527] | 0.015 |
| MB (target decel) | 300 | 0.993 [0.983, 1.000] | 0.970 [0.942, 0.992] | 0.067 |

FD now 0.1395 (n GT-cruise 1856); floor if every SELECTION window were fixed: class-existence 0.0000, good-existence 0.0027
descriptive headroom: {"ADE6_best_cruise_cand_minus_emitted": {"mean": -1.4658, "lo": -1.8424, "hi": -1.0889, "n_windows": 259, "n_episodes": 55}, "ADE6_gt_oracle_cand_minus_emitted": {"mean": -2.7066, "lo": -3.1316, "hi": -2.2727, "n_windows": 259, "n_episodes": 55}, "ADE6_emitted_FB_mean": 3.400013249271678, "ADE6_emitted_CT_mean": 1.6120588301092103}
pre-blend (refcv6) pick class shares on FB (decel, cruise, accel): {'FB_base_pick_class_shares(decel,cruise,accel)': [0.8532818532818532, 0.14671814671814673, 0.0]}
controls hard pass (seed 0): True; {"K_G_fields": true, "K_B_emitted_equals_fan_at_sel_idx": true, "K_C_sel_idx_is_argmax_over_e9_eligible": true, "K_A_reroll_equals_L1_BASEos": true, "K_D_prior_path_equals_banked_prior8": true, "K_H_ade_definition_matches_battery": true, "K_F_set_sizes": true, "K_I_CT_identity": true, "K_J_emitted_candidate_class_matches": true, "K_J_gap_nonnegative_when_emitted_is_argmax": true}

**Next lever:** TRAIN-fitted selector / score correction over the fan (the L1-type lever); the FD it can remove is the FB share with a cruise candidate (see floors)
Not computed: distance-keeping (no lead track; n=0); STRATEGIC (--no-strategic; n=0)