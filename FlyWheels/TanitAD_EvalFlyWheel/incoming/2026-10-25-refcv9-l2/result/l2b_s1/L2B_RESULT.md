# L2-B brake locus (seed 1) -- SELECTION

PREREG_L2B sha256 089d7ab67a9c  |  tier T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: 'another draw of episodes' only; NOT another training run, NOT another inference run (seed 1 = replicate, never pooled)

**label SELECTION** (basis: R2 (GOOD-cruise existence; the class-level existence is saturated in FB and DR)); R1 (class existence) SELECTION, R2 (good-cruise existence) SELECTION; existence saturated: True

| set | n | EXIST target-class cand [CI] | GOOD (ADE<=tau) [CI] | refcv6 pre-blend pick is target class | median #target cands |
|---|---|---|---|---|---|
| FB | 255 | 1.000 [1.000, 1.000] | 0.984 [0.966, 0.997] | 0.137 | 38.0 |
| CT | 255 | 1.000 [1.000, 1.000] | 0.977 [0.955, 0.993] | 0.965 | 47.0 |
| DR | 338 | 0.997 [0.990, 1.000] | 0.417 [0.332, 0.513] | 0.021 | 28.0 |
| MB (target=decel) | 301 | 0.997 [0.989, 1.000] | 0.970 [0.940, 0.994] | 0.070 | 74.0 |

FD now 0.1374; floor if every SELECTION window were fixed: class-existence 0.0000, good-existence 0.0022 (n GT-cruise 1856)
tau (GOOD tolerance) 2.781 m; headroom (descriptive): {"ADE6_best_cruise_cand_minus_emitted": {"mean": -1.6809, "lo": -2.0781, "hi": -1.2779, "n_windows": 255, "n_episodes": 55}, "ADE6_gt_oracle_cand_minus_emitted": {"mean": -2.8766, "lo": -3.3194, "hi": -2.4235, "n_windows": 255, "n_episodes": 55}, "ADE6_emitted_FB_mean": 3.5240951130104134, "ADE6_emitted_CT_mean": 1.5324995372884997}
FB score gap quantiles: {'q10_50_90': [0.07568359375000001, 0.6222639083862305, 1.411807918548584], 'n': 255}
base pick: {'FB_base_pick_class_shares(decel,cruise,accel)': [0.8627450980392157, 0.13725490196078433, 0.0]}

controls: {"K_G_fields": true, "K_B_emitted_equals_fan_at_sel_idx": true, "K_C_sel_idx_is_argmax_over_e9_eligible": true, "K_A_reroll_equals_L1_BASEos": true, "K_D_prior_path_equals_banked_prior8": true, "K_H_ade_definition_matches_battery": true, "K_F_set_sizes": true, "K_I_CT_identity": true, "K_J_emitted_candidate_class_matches": true, "K_J_gap_nonnegative_when_emitted_is_argmax": true}