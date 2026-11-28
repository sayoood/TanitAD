# VERDICT LEADSUP T0 N1 (card PREREG_V8F_PROOF_ADD24, section 3a, with note PREREG_V8F_PROOF_ADD24_N1)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Evidence class: MEASURED once the arms exist. Estimator: episode-cluster bootstrap, B 2000, rng 0 (episodes of the scored windows; clips on the train probe set).
Definitions and every reader decision: LEADSUP_READ_DEFS.md.

## Controls (first)

- k1_label_path: PASS :: {"r2_skill_this_path": 0.7499068112986798, "r2_skill_objslot_label_pipeline_now": 0.7499068112986798, "r2_skill_banked_ADD20": 0.7499068112986798, "n": 413, "tol": 1e-06, "card_value": 0.75}
- a2_pack_counts: PASS :: {"expected": {"windows": 1948, "scored": 429, "episodes_scored": 48}, "got": {"windows": 1948, "scored": 429, "episodes_scored": 48}}
- a1_records|LS_s0: PASS :: {"eval_states": {"sha256": "6797fbde6b852398994ac55c82b98f90b40737837303e8b8f272ef38321adfb4", "record_sha256": "6797fbde6b852398994ac55c82b98f90b40737837303e8b8f272ef38321adfb4", "match": true, "ckpt_sha256": "df7f6b8094ad515f0eec8b393d31e70dd6024a90f5a1ea1e231b343ff1f9ec35", "pass": true}, "train_states": {"sha256": "c399848660bfe585863b51a1433580b1eba3715cbe56e0a693ab0db4b7a8ad54", "record_sha256": "c399848660bfe585863b51a1433580b1eba3715cbe56e0a693ab0db4b7a8ad54", "match": true, "ckpt_sha256": "df7f6b8094ad515f0eec8b393d31e70dd6024a90f5a1ea1e231b343ff1f9ec35", "pass": true}, "eval_lead": {"sha256": "b597a4707bd919c2f6de99133e35009aac38adab820488a71fa035e937c54971", "record_sha256": "b597a4707bd919c2f6de99133e35009aac38adab820488a71fa035e937c54971", "match": true, "head_sha256": "e406b0219bcf60b89f0629c1c3eae0cd62dd42073d08b96213f9018811dd8cf2", "head_match": true, "states_sha256_in_record": "6797fbde6b852398994ac55c82b98f90b40737837303e8b8f272ef38321adfb4", "ckpt_sha256": "df7f6b8094ad515f0eec8b393d31e70dd6024a90f5a1ea1e231b343ff1f9ec35", "pass": true, "applied_to_pulled_states": true}, "train_lead": {"sha256": "193cf77fd543926f2213b7873ccd24e0300b86d614877423255e633bb170c380", "record_sha256": "193cf77fd543926f2213b7873ccd24e0300b86d614877423255e633bb170c380", "match": true, "head_sha256": "e406b0219bcf60b89f0629c1c3eae0cd62dd42073d08b96213f9018811dd8cf2", "head_match": true, "states_sha256_in_record": "c399848660bfe585863b51a1433580b1eba3715cbe56e0a693ab0db4b7a8ad54", "ckpt_sha256": "df7f6b8094ad515f0eec8b393d31e70dd6024a90f5a1ea1e231b343ff1f9ec35", "pass": true, "applied_to_pulled_states": true}, "ckpt_sha256_identical_across_the_four_files": true}
- a3_same_head_eval_train|LS_s0: PASS :: {}
- a4_window_states|LS_s0: PASS :: {"window_frames_0to4_without_state": 0, "window_states_0to4_finite": true, "future_frames_without_state": 0}
- k2_head_forward_eval|LS_s0: PASS :: {"max_abs_gap_m": 4.506111145019531e-05, "max_abs_closing_ms": 5.379319190979004e-06, "max_abs_presence_logit": 3.337860107421875e-06, "tol": 0.0001, "n_rows_compared": 13673, "n_windows_compared": 1948, "has_history_equal": true}
- k3_closing|LS_s0: PASS :: {"closing_r2_shuffled": -0.2300492824668774, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_permutation_over_all_windows_closing_r2": -0.6280961480278737}
- k3_card_form_slot|LS_s0: PASS :: {"r3_skill_shuffled": 0.01929208505205937, "r3_skill_shuffled_raw_copy_info_only": 0.020866540734338912, "bar": 0.02, "gates_under_note_n1": false, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "shuffled_head_gap_mean_minus_label_gap_mean_m": 1.7153368238675384, "info_only_r3_skill_other_permutations": [0.0518252546527308, 0.050787118611673954, 0.057690764775308434], "info_only_permutation_over_all_windows": {"closing_r2": -0.6280961480278737, "r3_skill": 0.018194187214340074, "r3_skill_raw_copy": 0.5250508225304211, "gap_mean_minus_label_mean_m": 22.77313581513146}}
- k3_slot_n1|LS_s0: PASS :: {"delta_permuted": 0.0014183245505601239, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_delta_other_permutations": [-0.0007204976965913268, 0.0012478078151589855, -0.018464702180212678, 0.0009398524810712017]}
- k4_no_future_input|LS_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0, "end_frame_read": 4}
- a5_train_layout|LS_s0: PASS :: {"clips_equal": true, "clip_idx_equal": true, "t_equal": true, "n_rows": 121166}
- k2_head_forward_train|LS_s0: PASS :: {"max_abs_gap_m": 5.7220458984375e-05, "max_abs_closing_ms": 6.094574928283691e-06, "max_abs_presence_logit": 4.291534423828125e-06, "tol": 0.0001, "n_rows_compared": 116366, "n_windows_compared": 116366, "has_history_equal": true}
- a1_records|LS_s1: PASS :: {"eval_states": {"sha256": "e4c88cfe071cf649e34b79f9639f80ba5940f253bd1980ae4406c8845eb204d4", "record_sha256": "e4c88cfe071cf649e34b79f9639f80ba5940f253bd1980ae4406c8845eb204d4", "match": true, "ckpt_sha256": "055e57bda3def2ca823056178307c9b53b60639fd811c241706fc580a02cdc93", "pass": true}, "train_states": {"sha256": "dd6b0b1e2c92118692b50194a5b6859ada70abcecdfe27fd25a529a4a1d5a0e3", "record_sha256": "dd6b0b1e2c92118692b50194a5b6859ada70abcecdfe27fd25a529a4a1d5a0e3", "match": true, "ckpt_sha256": "055e57bda3def2ca823056178307c9b53b60639fd811c241706fc580a02cdc93", "pass": true}, "eval_lead": {"sha256": "07dcb3e7175594071d1a8dd1b4e08b436246ade4b121b093c9e303f436434fce", "record_sha256": "07dcb3e7175594071d1a8dd1b4e08b436246ade4b121b093c9e303f436434fce", "match": true, "head_sha256": "c9c0b7a95210543da09b8fda8c02dafb932c37ed98b335f850bd4e6fcee9ab58", "head_match": true, "states_sha256_in_record": "e4c88cfe071cf649e34b79f9639f80ba5940f253bd1980ae4406c8845eb204d4", "ckpt_sha256": "055e57bda3def2ca823056178307c9b53b60639fd811c241706fc580a02cdc93", "pass": true, "applied_to_pulled_states": true}, "train_lead": {"sha256": "ea99dfcfd7f5c3e28095e2518e9af9817ae7dfd1c4beb331e7242f3d1af0a384", "record_sha256": "ea99dfcfd7f5c3e28095e2518e9af9817ae7dfd1c4beb331e7242f3d1af0a384", "match": true, "head_sha256": "c9c0b7a95210543da09b8fda8c02dafb932c37ed98b335f850bd4e6fcee9ab58", "head_match": true, "states_sha256_in_record": "dd6b0b1e2c92118692b50194a5b6859ada70abcecdfe27fd25a529a4a1d5a0e3", "ckpt_sha256": "055e57bda3def2ca823056178307c9b53b60639fd811c241706fc580a02cdc93", "pass": true, "applied_to_pulled_states": true}, "ckpt_sha256_identical_across_the_four_files": true}
- a3_same_head_eval_train|LS_s1: PASS :: {}
- a4_window_states|LS_s1: PASS :: {"window_frames_0to4_without_state": 0, "window_states_0to4_finite": true, "future_frames_without_state": 0}
- k2_head_forward_eval|LS_s1: PASS :: {"max_abs_gap_m": 4.4345855712890625e-05, "max_abs_closing_ms": 7.867813110351562e-06, "max_abs_presence_logit": 4.76837158203125e-06, "tol": 0.0001, "n_rows_compared": 13673, "n_windows_compared": 1948, "has_history_equal": true}
- k3_closing|LS_s1: PASS :: {"closing_r2_shuffled": -0.3629504263486065, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_permutation_over_all_windows_closing_r2": -0.7974023988673309}
- k3_card_form_slot|LS_s1: PASS :: {"r3_skill_shuffled": 0.01646143716871895, "r3_skill_shuffled_raw_copy_info_only": 0.017480616283442107, "bar": 0.02, "gates_under_note_n1": false, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "shuffled_head_gap_mean_minus_label_gap_mean_m": -0.12400912769481565, "info_only_r3_skill_other_permutations": [0.03945010876660371, 0.03689927300664386, 0.07623565710098512], "info_only_permutation_over_all_windows": {"closing_r2": -0.7974023988673309, "r3_skill": 0.2344732488535166, "r3_skill_raw_copy": 0.6309673075382165, "gap_mean_minus_label_mean_m": 23.899094330124356}}
- k3_slot_n1|LS_s1: PASS :: {"delta_permuted": -0.00568237199588717, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_delta_other_permutations": [-0.002582839412989646, -0.0007363255594139417, 0.009249109806137357, -0.004294613407456405]}
- k4_no_future_input|LS_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0, "end_frame_read": 4}
- a5_train_layout|LS_s1: PASS :: {"clips_equal": true, "clip_idx_equal": true, "t_equal": true, "n_rows": 121166}
- k2_head_forward_train|LS_s1: PASS :: {"max_abs_gap_m": 5.91278076171875e-05, "max_abs_closing_ms": 6.67572021484375e-06, "max_abs_presence_logit": 5.7220458984375e-06, "tol": 0.0001, "n_rows_compared": 116366, "n_windows_compared": 116366, "has_history_equal": true}
- a1_records|LSH_s0: PASS :: {"eval_states": {"sha256": "81006c0d437ee767208650f82d122ec1ddec7e42b50a8dbfe31a1e589980093a", "record_sha256": "81006c0d437ee767208650f82d122ec1ddec7e42b50a8dbfe31a1e589980093a", "match": true, "ckpt_sha256": "9decbe2b356f759ca422443795832446e15cf5bb0d6a37d9bc21b4bad064984b", "pass": true}, "train_states": {"sha256": "6ae34783bea11b57ef59d56b4f8d7585eb77158c19eb157a79d6cffd1476d175", "record_sha256": "6ae34783bea11b57ef59d56b4f8d7585eb77158c19eb157a79d6cffd1476d175", "match": true, "ckpt_sha256": "9decbe2b356f759ca422443795832446e15cf5bb0d6a37d9bc21b4bad064984b", "pass": true}, "eval_lead": {"sha256": "254ffad728ee01a71f9698958268932f053270641da5d4f60184067098d86bd2", "record_sha256": "254ffad728ee01a71f9698958268932f053270641da5d4f60184067098d86bd2", "match": true, "head_sha256": "d89f5d2f551670ee99908857d221848c571e62771ca634bd61ba1eb2332be2fd", "head_match": true, "states_sha256_in_record": "81006c0d437ee767208650f82d122ec1ddec7e42b50a8dbfe31a1e589980093a", "ckpt_sha256": "9decbe2b356f759ca422443795832446e15cf5bb0d6a37d9bc21b4bad064984b", "pass": true, "applied_to_pulled_states": true}, "train_lead": {"sha256": "765159dad40c95c7f6927a06bd379459d065f65ca653fac9b86c37710be6f4e4", "record_sha256": "765159dad40c95c7f6927a06bd379459d065f65ca653fac9b86c37710be6f4e4", "match": true, "head_sha256": "d89f5d2f551670ee99908857d221848c571e62771ca634bd61ba1eb2332be2fd", "head_match": true, "states_sha256_in_record": "6ae34783bea11b57ef59d56b4f8d7585eb77158c19eb157a79d6cffd1476d175", "ckpt_sha256": "9decbe2b356f759ca422443795832446e15cf5bb0d6a37d9bc21b4bad064984b", "pass": true, "applied_to_pulled_states": true}, "ckpt_sha256_identical_across_the_four_files": true}
- a3_same_head_eval_train|LSH_s0: PASS :: {}
- a4_window_states|LSH_s0: PASS :: {"window_frames_0to4_without_state": 0, "window_states_0to4_finite": true, "future_frames_without_state": 0}
- k2_head_forward_eval|LSH_s0: PASS :: {"max_abs_gap_m": 5.1975250244140625e-05, "max_abs_closing_ms": 6.288290023803711e-06, "max_abs_presence_logit": 4.76837158203125e-06, "tol": 0.0001, "n_rows_compared": 13673, "n_windows_compared": 1948, "has_history_equal": true}
- k3_closing|LSH_s0: PASS :: {"closing_r2_shuffled": -0.31971829072152835, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_permutation_over_all_windows_closing_r2": -0.5765810548710746}
- k3_card_form_slot|LSH_s0: FAIL :: {"r3_skill_shuffled": 0.09494346839064594, "r3_skill_shuffled_raw_copy_info_only": 0.1325162196089955, "bar": 0.02, "gates_under_note_n1": false, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "shuffled_head_gap_mean_minus_label_gap_mean_m": 3.915630664288564, "info_only_r3_skill_other_permutations": [0.10529293903760562, 0.11404993712039435, 0.14516593986072834], "info_only_permutation_over_all_windows": {"closing_r2": -0.5765810548710746, "r3_skill": 0.2595142180150518, "r3_skill_raw_copy": 0.6848410525476011, "gap_mean_minus_label_mean_m": 24.808446319679078}}
- k3_slot_n1|LSH_s0: PASS :: {"delta_permuted": -0.006829429222471317, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_delta_other_permutations": [-0.0007579498428504916, -0.0014880964369183847, -0.004436198378130918, -0.016916141035004717]}
- k4_no_future_input|LSH_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0, "end_frame_read": 4}
- a5_train_layout|LSH_s0: PASS :: {"clips_equal": true, "clip_idx_equal": true, "t_equal": true, "n_rows": 121166}
- k2_head_forward_train|LSH_s0: PASS :: {"max_abs_gap_m": 6.699562072753906e-05, "max_abs_closing_ms": 7.867813110351562e-06, "max_abs_presence_logit": 5.245208740234375e-06, "tol": 0.0001, "n_rows_compared": 116366, "n_windows_compared": 116366, "has_history_equal": true}
- a1_records|LSH_s1: PASS :: {"eval_states": {"sha256": "0d3bf41ae273ef7247e1c4bc7ebf7d80aeb478a60d2505e1b7334f0775873ca1", "record_sha256": "0d3bf41ae273ef7247e1c4bc7ebf7d80aeb478a60d2505e1b7334f0775873ca1", "match": true, "ckpt_sha256": "27c58b683f78996511961c4f3fbc1b8b001d4b77acd4c97183d0a7a5a279827e", "pass": true}, "train_states": {"sha256": "7b8203cd2f155229958930b1bd326dfc1e4818a99f66922a689bdf3511658179", "record_sha256": "7b8203cd2f155229958930b1bd326dfc1e4818a99f66922a689bdf3511658179", "match": true, "ckpt_sha256": "27c58b683f78996511961c4f3fbc1b8b001d4b77acd4c97183d0a7a5a279827e", "pass": true}, "eval_lead": {"sha256": "dd495d8cdbcfb96f5559a296aed4ffcb653dea166dd9d5bf768638b6c364b202", "record_sha256": "dd495d8cdbcfb96f5559a296aed4ffcb653dea166dd9d5bf768638b6c364b202", "match": true, "head_sha256": "97d494e23f9d870f3f5b57a097b05c0d2832a8553fca7e0d3e29199dd214e9dd", "head_match": true, "states_sha256_in_record": "0d3bf41ae273ef7247e1c4bc7ebf7d80aeb478a60d2505e1b7334f0775873ca1", "ckpt_sha256": "27c58b683f78996511961c4f3fbc1b8b001d4b77acd4c97183d0a7a5a279827e", "pass": true, "applied_to_pulled_states": true}, "train_lead": {"sha256": "4b185353608ee2ab800c10c17f88e85b12da034d6e1fdccad018b5698ed48b53", "record_sha256": "4b185353608ee2ab800c10c17f88e85b12da034d6e1fdccad018b5698ed48b53", "match": true, "head_sha256": "97d494e23f9d870f3f5b57a097b05c0d2832a8553fca7e0d3e29199dd214e9dd", "head_match": true, "states_sha256_in_record": "7b8203cd2f155229958930b1bd326dfc1e4818a99f66922a689bdf3511658179", "ckpt_sha256": "27c58b683f78996511961c4f3fbc1b8b001d4b77acd4c97183d0a7a5a279827e", "pass": true, "applied_to_pulled_states": true}, "ckpt_sha256_identical_across_the_four_files": true}
- a3_same_head_eval_train|LSH_s1: PASS :: {}
- a4_window_states|LSH_s1: PASS :: {"window_frames_0to4_without_state": 0, "window_states_0to4_finite": true, "future_frames_without_state": 0}
- k2_head_forward_eval|LSH_s1: PASS :: {"max_abs_gap_m": 4.220008850097656e-05, "max_abs_closing_ms": 5.066394805908203e-06, "max_abs_presence_logit": 4.76837158203125e-06, "tol": 0.0001, "n_rows_compared": 13673, "n_windows_compared": 1948, "has_history_equal": true}
- k3_closing|LSH_s1: PASS :: {"closing_r2_shuffled": -0.20906056640466253, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_permutation_over_all_windows_closing_r2": -0.3959933119210708}
- k3_card_form_slot|LSH_s1: FAIL :: {"r3_skill_shuffled": 0.027900221644908907, "r3_skill_shuffled_raw_copy_info_only": 0.026144678309702685, "bar": 0.02, "gates_under_note_n1": false, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "shuffled_head_gap_mean_minus_label_gap_mean_m": 0.581244493800849, "info_only_r3_skill_other_permutations": [0.03264614900795526, 0.053006544609975226, 0.07567798868321796], "info_only_permutation_over_all_windows": {"closing_r2": -0.3959933119210708, "r3_skill": 0.11780291957382072, "r3_skill_raw_copy": 0.5876829016618681, "gap_mean_minus_label_mean_m": 20.149784889622403}}
- k3_slot_n1|LSH_s1: PASS :: {"delta_permuted": -0.010918223047636344, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_delta_other_permutations": [0.0017537840654939796, 0.00016861858065431168, -0.002928791595544933, -0.007148451664880318]}
- k4_no_future_input|LSH_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0, "end_frame_read": 4}
- a5_train_layout|LSH_s1: PASS :: {"clips_equal": true, "clip_idx_equal": true, "t_equal": true, "n_rows": 121166}
- k2_head_forward_train|LSH_s1: PASS :: {"max_abs_gap_m": 4.9114227294921875e-05, "max_abs_closing_ms": 7.510185241699219e-06, "max_abs_presence_logit": 4.76837158203125e-06, "tol": 0.0001, "n_rows_compared": 116366, "n_windows_compared": 116366, "has_history_equal": true}
- a1_records|LSD_s0: PASS :: {"eval_states": {"sha256": "79d301aaa7baf0d94c24969890053505429f9c591c6d1a845e9a2ee586c5150c", "record_sha256": "79d301aaa7baf0d94c24969890053505429f9c591c6d1a845e9a2ee586c5150c", "match": true, "ckpt_sha256": "1b617e679ecf44bced995c205544baaabc4f3cc13bd583de454b5f96d8f28e3d", "pass": true}, "train_states": {"sha256": "951d14515011146ddc3df232eecd62c06c21fe1cc2486833f6f30970404b8c27", "record_sha256": "951d14515011146ddc3df232eecd62c06c21fe1cc2486833f6f30970404b8c27", "match": true, "ckpt_sha256": "1b617e679ecf44bced995c205544baaabc4f3cc13bd583de454b5f96d8f28e3d", "pass": true}, "eval_lead": {"sha256": "e5182ce773db00e08cd61f51961add5ca0639f5f2b5c9bc8953a7ae37adc6a37", "record_sha256": "e5182ce773db00e08cd61f51961add5ca0639f5f2b5c9bc8953a7ae37adc6a37", "match": true, "head_sha256": "0668e2c99590a1e687663578bbe98be76c5ba5637ffd9d29134b4584825a91ac", "head_match": true, "states_sha256_in_record": "79d301aaa7baf0d94c24969890053505429f9c591c6d1a845e9a2ee586c5150c", "ckpt_sha256": "1b617e679ecf44bced995c205544baaabc4f3cc13bd583de454b5f96d8f28e3d", "pass": true, "applied_to_pulled_states": true}, "train_lead": {"sha256": "e05902f7164189fc09690366753f0d6be17080508a08ba64b367885c454c0b05", "record_sha256": "e05902f7164189fc09690366753f0d6be17080508a08ba64b367885c454c0b05", "match": true, "head_sha256": "0668e2c99590a1e687663578bbe98be76c5ba5637ffd9d29134b4584825a91ac", "head_match": true, "states_sha256_in_record": "951d14515011146ddc3df232eecd62c06c21fe1cc2486833f6f30970404b8c27", "ckpt_sha256": "1b617e679ecf44bced995c205544baaabc4f3cc13bd583de454b5f96d8f28e3d", "pass": true, "applied_to_pulled_states": true}, "ckpt_sha256_identical_across_the_four_files": true}
- a3_same_head_eval_train|LSD_s0: PASS :: {}
- a4_window_states|LSD_s0: PASS :: {"window_frames_0to4_without_state": 0, "window_states_0to4_finite": true, "future_frames_without_state": 0}
- k2_head_forward_eval|LSD_s0: PASS :: {"max_abs_gap_m": 5.173683166503906e-05, "max_abs_closing_ms": 5.081295967102051e-06, "max_abs_presence_logit": 3.814697265625e-06, "tol": 0.0001, "n_rows_compared": 13673, "n_windows_compared": 1948, "has_history_equal": true}
- k3_closing|LSD_s0: PASS :: {"closing_r2_shuffled": -0.3361509767695463, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_permutation_over_all_windows_closing_r2": -0.4556821250393961}
- k3_card_form_slot|LSD_s0: FAIL :: {"r3_skill_shuffled": 0.054953020270925856, "r3_skill_shuffled_raw_copy_info_only": 0.07324278164390308, "bar": 0.02, "gates_under_note_n1": false, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "shuffled_head_gap_mean_minus_label_gap_mean_m": 2.7719758712206257, "info_only_r3_skill_other_permutations": [0.08685643788379016, 0.1075515941088, 0.07042769063090148], "info_only_permutation_over_all_windows": {"closing_r2": -0.4556821250393961, "r3_skill": -0.006002393200220446, "r3_skill_raw_copy": 0.507677355892344, "gap_mean_minus_label_mean_m": 18.73999984962888}}
- k3_slot_n1|LSD_s0: PASS :: {"delta_permuted": -0.0025839579080171315, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_delta_other_permutations": [-0.004182488781165006, -0.0008938887526768191, -0.008302872849297982, -0.0027878239923251336]}
- k4_no_future_input|LSD_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0, "end_frame_read": 4}
- a5_train_layout|LSD_s0: PASS :: {"clips_equal": true, "clip_idx_equal": true, "t_equal": true, "n_rows": 121166}
- k2_head_forward_train|LSD_s0: PASS :: {"max_abs_gap_m": 4.38690185546875e-05, "max_abs_closing_ms": 5.230307579040527e-06, "max_abs_presence_logit": 4.76837158203125e-06, "tol": 0.0001, "n_rows_compared": 116366, "n_windows_compared": 116366, "has_history_equal": true}
- a1_records|LSD_s1: PASS :: {"eval_states": {"sha256": "3f24136a3aa643e49295f6cba6f2be3eb397ae4fe76c9940e31c1dec35b43985", "record_sha256": "3f24136a3aa643e49295f6cba6f2be3eb397ae4fe76c9940e31c1dec35b43985", "match": true, "ckpt_sha256": "c7b954dd88574edab2b79f798f44abf2e0f25f6b55a63b4bf4aba40bcd552865", "pass": true}, "train_states": {"sha256": "07ae8328cf9ab75025fed5c9f18c3c6a1a9d23c7e1f430c5326d4ad9db77b346", "record_sha256": "07ae8328cf9ab75025fed5c9f18c3c6a1a9d23c7e1f430c5326d4ad9db77b346", "match": true, "ckpt_sha256": "c7b954dd88574edab2b79f798f44abf2e0f25f6b55a63b4bf4aba40bcd552865", "pass": true}, "eval_lead": {"sha256": "6559bb376b35f9ffb728e474525003b3eb10451cd571a0bb45abb6b84bd3b933", "record_sha256": "6559bb376b35f9ffb728e474525003b3eb10451cd571a0bb45abb6b84bd3b933", "match": true, "head_sha256": "7eff4816292382ff3d18379132959c3ca4ab919191796206000f621cf2ba354b", "head_match": true, "states_sha256_in_record": "3f24136a3aa643e49295f6cba6f2be3eb397ae4fe76c9940e31c1dec35b43985", "ckpt_sha256": "c7b954dd88574edab2b79f798f44abf2e0f25f6b55a63b4bf4aba40bcd552865", "pass": true, "applied_to_pulled_states": true}, "train_lead": {"sha256": "7ea2b975433d81d463b78aef71bc5ece5e19c16233cf1f724f5d9ab723f448cc", "record_sha256": "7ea2b975433d81d463b78aef71bc5ece5e19c16233cf1f724f5d9ab723f448cc", "match": true, "head_sha256": "7eff4816292382ff3d18379132959c3ca4ab919191796206000f621cf2ba354b", "head_match": true, "states_sha256_in_record": "07ae8328cf9ab75025fed5c9f18c3c6a1a9d23c7e1f430c5326d4ad9db77b346", "ckpt_sha256": "c7b954dd88574edab2b79f798f44abf2e0f25f6b55a63b4bf4aba40bcd552865", "pass": true, "applied_to_pulled_states": true}, "ckpt_sha256_identical_across_the_four_files": true}
- a3_same_head_eval_train|LSD_s1: PASS :: {}
- a4_window_states|LSD_s1: PASS :: {"window_frames_0to4_without_state": 0, "window_states_0to4_finite": true, "future_frames_without_state": 0}
- k2_head_forward_eval|LSD_s1: PASS :: {"max_abs_gap_m": 3.886222839355469e-05, "max_abs_closing_ms": 4.842877388000488e-06, "max_abs_presence_logit": 4.76837158203125e-06, "tol": 0.0001, "n_rows_compared": 13673, "n_windows_compared": 1948, "has_history_equal": true}
- k3_closing|LSD_s1: PASS :: {"closing_r2_shuffled": -0.2895380484320451, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_permutation_over_all_windows_closing_r2": -0.4316331645704896}
- k3_card_form_slot|LSD_s1: FAIL :: {"r3_skill_shuffled": 0.06221960049815878, "r3_skill_shuffled_raw_copy_info_only": 0.07748814625673917, "bar": 0.02, "gates_under_note_n1": false, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "shuffled_head_gap_mean_minus_label_gap_mean_m": 2.693584055285765, "info_only_r3_skill_other_permutations": [0.08043426725671898, 0.06687028506307746, 0.08823439666439392], "info_only_permutation_over_all_windows": {"closing_r2": -0.4316331645704896, "r3_skill": 0.002341022899856382, "r3_skill_raw_copy": 0.5489490680442206, "gap_mean_minus_label_mean_m": 20.86842417370609}}
- k3_slot_n1|LSD_s1: PASS :: {"delta_permuted": -0.008687326649576832, "bar": 0.02, "n_windows_permuted": 428, "permutation_is_identity": false, "seed": 0, "info_only_delta_other_permutations": [-0.005746259628209405, -0.002530977837546411, -0.0002584004967223219, -0.00036251171617229794]}
- k4_no_future_input|LSD_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0, "end_frame_read": 4}
- a5_train_layout|LSD_s1: PASS :: {"clips_equal": true, "clip_idx_equal": true, "t_equal": true, "n_rows": 121166}
- k2_head_forward_train|LSD_s1: PASS :: {"max_abs_gap_m": 4.506111145019531e-05, "max_abs_closing_ms": 6.07222318649292e-06, "max_abs_presence_logit": 6.198883056640625e-06, "tol": 0.0001, "n_rows_compared": 116366, "n_windows_compared": 116366, "has_history_equal": true}
- k5 (trained, not only built), k6 (checkpoint identity with the T1-kin dump), k7 (q3_read controls): the lead's, not run here.
- controls_pass (gate for the slot names under note N1): True

- gate for the closing-speed names and the gradient question (k1, k2, k3_closing, k4, a-checks): PASS
- gate for the slot names (those plus k3_slot_n1): PASS

## Card form (as written)

Control k3 of the card, slot part (states permuted among the scored windows, R3 skill over the bias-corrected copy at most +0.02), per arm; it does not gate under note N1:

- LS_s0: k3 slot part PASSED (R3 skill permuted +0.019; bar +0.02)
- LS_s1: k3 slot part PASSED (R3 skill permuted +0.016; bar +0.02)
- LSH_s0: k3 slot part FAILED (R3 skill permuted +0.095; bar +0.02)
- LSH_s1: k3 slot part FAILED (R3 skill permuted +0.028; bar +0.02)
- LSD_s0: k3 slot part FAILED (R3 skill permuted +0.055; bar +0.02)
- LSD_s1: k3 slot part FAILED (R3 skill permuted +0.062; bar +0.02)

The card names no outcome for an arm type with a failed control. Arms whose k3 slot part failed: LSH_s0, LSH_s1, LSD_s0, LSD_s1.
Arm types concerned in this run: LSD, LSH.

## Names under note N1 (both seeds needed; every slot name carries the N1 mark)

- Arm type LS: closing **SUP-CLOSING-FLAT**; slot (N1) **SUP-SLOT-FLAT-N1**
- Arm type LSH: closing **SUP-CLOSING-FLAT**; slot (N1) **SUP-SLOT-FLAT-N1**
- Arm type LSD: closing **SUP-CLOSING-FLAT**; slot (N1) **SUP-SLOT-FLAT-N1**
- Gradient question, LS against LSD (paired over episodes, floor |M1(LSD_s0) - M1(LSD_s1)| = 0.089): **SUP-GRADIENT-EQUAL**
- Gradient question, LSH against LSD (paired over episodes, floor |M1(LSD_s0) - M1(LSD_s1)| = 0.089): **SUP-GRADIENT-EQUAL**
- Reading with the earlier cards: flat in all three: the pooled 4 by 10 grid of frozen camera tokens does not carry the motion of the lead at this precision
- Cost: named by the lead's T1-kin cost read (q3_read.py, controls k5..k7), not by this reader

Four families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25).

## Windows

Windows 1948; scored 429 (episodes 48); with the label history rule 413; episodes over all windows 139.

## Per arm, eval windows (T0)

### LS_s0

checkpoint sha256 df7f6b8094ad515f0eec8b393d31e70dd6024a90f5a1ea1e231b343ff1f9ec35

- M1 closing speed at S0 (label closing speed): R2 +0.155 [-0.043, +0.283]; RMSE 2.10 m/s [+1.821, +2.416]; n=413
- M2 gap at S0, head: R2 +0.724 [+0.592, +0.801]; RMSE 5.39 m [+4.584, +6.293]; n=413
- M2 gap at S0, one-frame probe of the same arm: R2 +0.604 [+0.455, +0.694]; RMSE 6.46 m [+5.575, +7.300]; n=413; head minus probe in R2 +0.120 [+0.002, +0.253]
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.016 [-0.127, +0.152] n=429; level R2 +0.503 [+0.276, +0.637]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.105 [-0.214, +0.001] n=429; level R2 +0.442 [+0.196, +0.590]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.065 [-0.052, +0.180] n=429
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.071 [-0.082, +0.211] n=429; level R2 +0.531 [+0.328, +0.663]
- M3 DELTA (note N1; names the slot): -0.004 [-0.026, +0.016] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +1.34 m
- M3 R0 (head copy): level R2 +0.495 [+0.276, +0.632] (n scored 429)
- M3 R3 level R2 +0.531 minus banked rollout read +0.508: +0.024 [-0.131, +0.172] (n=429)
- M4 presence AUROC at S0: 0.914 [+0.877, +0.942]; n=1844 (positives 534, negatives 1310); names nothing

### LS_s1

checkpoint sha256 055e57bda3def2ca823056178307c9b53b60639fd811c241706fc580a02cdc93

- M1 closing speed at S0 (label closing speed): R2 +0.014 [-0.278, +0.192]; RMSE 2.27 m/s [+1.938, +2.611]; n=413
- M2 gap at S0, head: R2 +0.580 [+0.384, +0.702]; RMSE 6.65 m [+5.622, +7.691]; n=413
- M2 gap at S0, one-frame probe of the same arm: R2 +0.628 [+0.484, +0.723]; RMSE 6.26 m [+5.312, +7.159]; n=413; head minus probe in R2 -0.048 [-0.214, +0.099]
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.025 [-0.089, +0.141] n=429; level R2 +0.219 [-0.066, +0.415]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.097 [-0.217, +0.025] n=429; level R2 +0.121 [-0.202, +0.345]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.143 [+0.020, +0.264] n=429 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.140 [+0.022, +0.252] n=429 (separated above 0); level R2 +0.311 [+0.040, +0.492]
- M3 DELTA (note N1; names the slot): -0.004 [-0.059, +0.051] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): -0.52 m
- M3 R0 (head copy): level R2 +0.199 [-0.083, +0.381] (n scored 429)
- M3 R3 level R2 +0.311 minus banked rollout read +0.484: -0.173 [-0.387, +0.011] (n=429)
- M4 presence AUROC at S0: 0.913 [+0.877, +0.941]; n=1844 (positives 534, negatives 1310); names nothing

### LSH_s0

checkpoint sha256 9decbe2b356f759ca422443795832446e15cf5bb0d6a37d9bc21b4bad064984b

- M1 closing speed at S0 (label closing speed): R2 +0.092 [-0.138, +0.231]; RMSE 2.18 m/s [+1.891, +2.488]; n=413
- M2 gap at S0, head: R2 +0.595 [+0.357, +0.734]; RMSE 6.53 m [+5.469, +7.583]; n=413
- M2 gap at S0, one-frame probe of the same arm: R2 +0.622 [+0.476, +0.711]; RMSE 6.31 m [+5.452, +7.184]; n=413; head minus probe in R2 -0.027 [-0.234, +0.129]
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.069 [-0.070, +0.180] n=429; level R2 +0.398 [+0.121, +0.563]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.034 [-0.152, +0.071] n=429; level R2 +0.332 [+0.021, +0.516]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.077 [-0.054, +0.203] n=429
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.213 [-0.002, +0.380] n=429; level R2 +0.491 [+0.267, +0.634]
- M3 DELTA (note N1; names the slot): +0.019 [-0.027, +0.067] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +3.58 m
- M3 R0 (head copy): level R2 +0.354 [+0.048, +0.545] (n scored 429)
- M3 R3 level R2 +0.491 minus banked rollout read +0.508: -0.016 [-0.172, +0.135] (n=429)
- M4 presence AUROC at S0: 0.922 [+0.890, +0.949]; n=1844 (positives 534, negatives 1310); names nothing

### LSH_s1

checkpoint sha256 27c58b683f78996511961c4f3fbc1b8b001d4b77acd4c97183d0a7a5a279827e

- M1 closing speed at S0 (label closing speed): R2 +0.026 [-0.192, +0.163]; RMSE 2.26 m/s [+1.934, +2.603]; n=413
- M2 gap at S0, head: R2 +0.737 [+0.589, +0.829]; RMSE 5.26 m [+4.346, +6.192]; n=413
- M2 gap at S0, one-frame probe of the same arm: R2 +0.616 [+0.478, +0.705]; RMSE 6.36 m [+5.464, +7.218]; n=413; head minus probe in R2 +0.121 [-0.022, +0.259]
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.044 [-0.074, +0.146] n=429; level R2 +0.441 [+0.206, +0.591]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.108 [-0.228, -0.002] n=429 (separated below 0); level R2 +0.352 [+0.079, +0.519]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.136 [+0.004, +0.271] n=429 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.116 [-0.011, +0.249] n=429; level R2 +0.483 [+0.265, +0.637]
- M3 DELTA (note N1; names the slot): +0.001 [-0.085, +0.060] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +0.21 m
- M3 R0 (head copy): level R2 +0.415 [+0.170, +0.572] (n scored 429)
- M3 R3 level R2 +0.483 minus banked rollout read +0.484: -0.000 [-0.179, +0.164] (n=429)
- M4 presence AUROC at S0: 0.917 [+0.882, +0.945]; n=1844 (positives 534, negatives 1310); names nothing

### LSD_s0

checkpoint sha256 1b617e679ecf44bced995c205544baaabc4f3cc13bd583de454b5f96d8f28e3d

- M1 closing speed at S0 (label closing speed): R2 +0.178 [-0.059, +0.335]; RMSE 2.07 m/s [+1.836, +2.335]; n=413
- M2 gap at S0, head: R2 +0.626 [+0.400, +0.752]; RMSE 6.27 m [+5.273, +7.340]; n=413
- M2 gap at S0, one-frame probe of the same arm: R2 +0.588 [+0.436, +0.675]; RMSE 6.59 m [+5.879, +7.287]; n=413; head minus probe in R2 +0.038 [-0.139, +0.185]
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.060 [-0.099, +0.197] n=429; level R2 +0.448 [+0.171, +0.606]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.032 [-0.164, +0.075] n=429; level R2 +0.394 [+0.096, +0.560]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.041 [-0.066, +0.148] n=429
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.086 [-0.075, +0.226] n=429; level R2 +0.463 [+0.211, +0.619]
- M3 DELTA (note N1; names the slot): +0.001 [-0.044, +0.045] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.43 m
- M3 R0 (head copy): level R2 +0.413 [+0.097, +0.598] (n scored 429)
- M3 R3 level R2 +0.463 minus banked rollout read +0.508: -0.044 [-0.222, +0.109] (n=429)
- M4 presence AUROC at S0: 0.904 [+0.863, +0.936]; n=1844 (positives 534, negatives 1310); names nothing

### LSD_s1

checkpoint sha256 c7b954dd88574edab2b79f798f44abf2e0f25f6b55a63b4bf4aba40bcd552865

- M1 closing speed at S0 (label closing speed): R2 +0.089 [-0.127, +0.234]; RMSE 2.18 m/s [+1.851, +2.554]; n=413
- M2 gap at S0, head: R2 +0.544 [+0.217, +0.719]; RMSE 6.93 m [+5.378, +8.747]; n=413
- M2 gap at S0, one-frame probe of the same arm: R2 +0.655 [+0.533, +0.726]; RMSE 6.02 m [+5.287, +6.768]; n=413; head minus probe in R2 -0.111 [-0.404, +0.070]
- M3 R1: skill over the raw copy of the same head (card form, names nothing) -0.018 [-0.157, +0.127] n=429; level R2 +0.256 [-0.249, +0.535]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.108 [-0.213, +0.007] n=429; level R2 +0.190 [-0.327, +0.484]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): -0.006 [-0.117, +0.113] n=429
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.042 [-0.141, +0.197] n=429; level R2 +0.300 [-0.125, +0.535]
- M3 DELTA (note N1; names the slot): -0.032 [-0.066, +0.004] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.41 m
- M3 R0 (head copy): level R2 +0.269 [-0.163, +0.526] (n scored 429)
- M3 R3 level R2 +0.300 minus banked rollout read +0.484: -0.184 [-0.537, +0.069] (n=429)
- M4 presence AUROC at S0: 0.909 [+0.871, +0.942]; n=1844 (positives 534, negatives 1310); names nothing

## Per arm, train probe clips (held in for the model and the head; reported only, names nothing)

### LS_s0

rows 121166, with a 5-frame history 116366, gap rows 32025, closing rows 30865, scored rows (M3) 24533, clips 1200

- M1 closing speed at S0 (train labels): R2 +0.209 [+0.167, +0.249]; RMSE 2.65 m/s [+2.507, +2.794]; n=30865
- M2 gap at S0, head: R2 +0.604 [+0.559, +0.647]; RMSE 9.79 m [+9.186, +10.390]; n=32025
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.075 [+0.040, +0.107] n=24533 (separated above 0); level R2 +0.576 [+0.519, +0.623]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.006 [-0.037, +0.024] n=24533; level R2 +0.539 [+0.478, +0.589]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.108 [+0.067, +0.144] n=24533 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.106 [+0.066, +0.142] n=24533 (separated above 0); level R2 +0.590 [+0.534, +0.636]
- M3 DELTA (note N1; names the slot): +0.033 [+0.007, +0.058] n=24533 (separated above 0) (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): -0.04 m
- M3 R0 (head copy): level R2 +0.541 [+0.484, +0.591] (n scored 24533)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.878 [+0.867, +0.888]; n=116366 (positives 45364, negatives 71002); names nothing

### LS_s1

rows 121166, with a 5-frame history 116366, gap rows 32025, closing rows 30865, scored rows (M3) 24533, clips 1200

- M1 closing speed at S0 (train labels): R2 +0.182 [+0.129, +0.227]; RMSE 2.69 m/s [+2.560, +2.832]; n=30865
- M2 gap at S0, head: R2 +0.600 [+0.556, +0.641]; RMSE 9.85 m [+9.275, +10.422]; n=32025
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.057 [+0.021, +0.094] n=24533 (separated above 0); level R2 +0.534 [+0.476, +0.586]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.017 [-0.051, +0.019] n=24533; level R2 +0.498 [+0.435, +0.553]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.091 [+0.054, +0.127] n=24533 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.090 [+0.055, +0.127] n=24533 (separated above 0); level R2 +0.551 [+0.494, +0.601]
- M3 DELTA (note N1; names the slot): +0.021 [-0.002, +0.046] n=24533 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +0.01 m
- M3 R0 (head copy): level R2 +0.506 [+0.443, +0.561] (n scored 24533)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.871 [+0.860, +0.882]; n=116366 (positives 45364, negatives 71002); names nothing

### LSH_s0

rows 121166, with a 5-frame history 116366, gap rows 32025, closing rows 30865, scored rows (M3) 24533, clips 1200

- M1 closing speed at S0 (train labels): R2 +0.201 [+0.159, +0.239]; RMSE 2.66 m/s [+2.522, +2.812]; n=30865
- M2 gap at S0, head: R2 +0.643 [+0.599, +0.684]; RMSE 9.30 m [+8.715, +9.897]; n=32025
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.076 [+0.042, +0.109] n=24533 (separated above 0); level R2 +0.597 [+0.538, +0.646]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.004 [-0.029, +0.033] n=24533; level R2 +0.566 [+0.503, +0.617]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.100 [+0.064, +0.132] n=24533 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.137 [+0.092, +0.181] n=24533 (separated above 0); level R2 +0.624 [+0.570, +0.669]
- M3 DELTA (note N1; names the slot): +0.042 [+0.015, +0.068] n=24533 (separated above 0) (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +1.82 m
- M3 R0 (head copy): level R2 +0.564 [+0.498, +0.618] (n scored 24533)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.884 [+0.874, +0.894]; n=116366 (positives 45364, negatives 71002); names nothing

### LSH_s1

rows 121166, with a 5-frame history 116366, gap rows 32025, closing rows 30865, scored rows (M3) 24533, clips 1200

- M1 closing speed at S0 (train labels): R2 +0.186 [+0.140, +0.227]; RMSE 2.69 m/s [+2.546, +2.841]; n=30865
- M2 gap at S0, head: R2 +0.636 [+0.587, +0.677]; RMSE 9.40 m [+8.786, +9.978]; n=32025
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.050 [+0.013, +0.084] n=24533 (separated above 0); level R2 +0.589 [+0.532, +0.638]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.030 [-0.065, +0.002] n=24533; level R2 +0.554 [+0.493, +0.606]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.090 [+0.054, +0.123] n=24533 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.089 [+0.054, +0.122] n=24533 (separated above 0); level R2 +0.606 [+0.547, +0.654]
- M3 DELTA (note N1; names the slot): +0.019 [-0.003, +0.039] n=24533 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): -0.35 m
- M3 R0 (head copy): level R2 +0.567 [+0.506, +0.619] (n scored 24533)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.880 [+0.869, +0.890]; n=116366 (positives 45364, negatives 71002); names nothing

### LSD_s0

rows 121166, with a 5-frame history 116366, gap rows 32025, closing rows 30865, scored rows (M3) 24533, clips 1200

- M1 closing speed at S0 (train labels): R2 +0.203 [+0.161, +0.241]; RMSE 2.66 m/s [+2.518, +2.801]; n=30865
- M2 gap at S0, head: R2 +0.591 [+0.549, +0.629]; RMSE 9.96 m [+9.416, +10.492]; n=32025
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.043 [+0.007, +0.077] n=24533 (separated above 0); level R2 +0.557 [+0.507, +0.597]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.038 [-0.071, -0.005] n=24533 (separated below 0); level R2 +0.519 [+0.467, +0.563]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.085 [+0.052, +0.118] n=24533 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.091 [+0.054, +0.126] n=24533 (separated above 0); level R2 +0.579 [+0.531, +0.620]
- M3 DELTA (note N1; names the slot): +0.010 [-0.007, +0.026] n=24533 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +0.75 m
- M3 R0 (head copy): level R2 +0.537 [+0.485, +0.582] (n scored 24533)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.861 [+0.849, +0.873]; n=116366 (positives 45364, negatives 71002); names nothing

### LSD_s1

rows 121166, with a 5-frame history 116366, gap rows 32025, closing rows 30865, scored rows (M3) 24533, clips 1200

- M1 closing speed at S0 (train labels): R2 +0.207 [+0.167, +0.247]; RMSE 2.65 m/s [+2.507, +2.802]; n=30865
- M2 gap at S0, head: R2 +0.564 [+0.518, +0.607]; RMSE 10.27 m [+9.694, +10.829]; n=32025
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.026 [-0.012, +0.061] n=24533; level R2 +0.512 [+0.448, +0.563]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.053 [-0.087, -0.020] n=24533 (separated below 0); level R2 +0.472 [+0.406, +0.525]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.083 [+0.051, +0.116] n=24533 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.089 [+0.055, +0.123] n=24533 (separated above 0); level R2 +0.543 [+0.486, +0.589]
- M3 DELTA (note N1; names the slot): +0.006 [-0.010, +0.022] n=24533 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +0.78 m
- M3 R0 (head copy): level R2 +0.499 [+0.437, +0.548] (n scored 24533)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.855 [+0.843, +0.868]; n=116366 (positives 45364, negatives 71002); names nothing

## Paired differences of M1 (R2, episode-cluster bootstrap of the difference; floor above)

| pair | seed | diff | 95% interval | n |
|---|---|---|---|---|
| LS minus LSD | s0 | -0.024 | [-0.164, +0.132] | 413 |
| LSH minus LSD | s0 | -0.087 | [-0.235, +0.070] | 413 |
| LSH minus LS | s0 | -0.063 | [-0.233, +0.090] | 413 |
| LS minus LSD | s1 | -0.075 | [-0.328, +0.156] | 413 |
| LSH minus LSD | s1 | -0.063 | [-0.254, +0.132] | 413 |
| LSH minus LS | s1 | +0.013 | [-0.167, +0.216] | 413 |

## Limits (card section 7 and the reader)

- Two training runs per arm type: a name from two seeds is stated with that limit; the interval answers the question about episodes, not about training runs. The gradient question carries its floor from the two LSD seeds.
- One corpus, one evaluation set: 429 scored windows of 48 episodes. Labels come from the offline obstacle tracks; the eval presence label is the bank lead_present, the train presence label is lead_valid (LEADSUP_READ_DEFS.md).
- The states are the float16 state files of the extraction, not the float32 states the head saw in training; the pod and this box apply the head to the same float16 states (control k2).
- The one-frame probe is fitted on eval-bank rows (cross-fitted by episode); the head is fitted on the train clips. The two are different protocols and are reported beside, not as a paired test of the same fit.
- DELTA reads the error of imagining the gap 2.0 s ahead from the present gap and the logged ego action; R3n has the same folds, the same ego term, and no closing speed of the head.
- R3 coefficients are cross-fitted over eval folds; the ego speeds of the rule are the logged ones (the action input of a rollout); no goal input, nothing of the situation classifier.
- The cost read (T1-kin) and controls k5 to k7 belong to the lead.
