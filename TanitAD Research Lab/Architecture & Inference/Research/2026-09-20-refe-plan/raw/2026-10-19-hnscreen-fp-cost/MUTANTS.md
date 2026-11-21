# MUTANTS -- every mutant of the (b') change, generated from the final logs by scripts/mutant_table.py

Each row is a copy of the real source with ONE exact-once snippet edit (the single enforcing line deleted / inverted); the named arms must turn RED (MEASURED, artifact: the log named in the last section).
A mutant that stays green is a PROBLEM line in the log and the matrix fails.

| file | mutant (what the edit does) | arms RED (listed, up to 2) | verdict |
|---|---|---|---|
| gate_token.py / launch_gate.py | anchor_listing_problems: the SIZE comparison is deleted | R2_a_size_change_refuses_every_intermediate_entry_point_and_voids_at_the_end | RED |
| gate_token.py / launch_gate.py | anchor_listing_problems: the MTIME comparison is deleted | R3_an_mtime_ns_change_with_identical_bytes_refuses_and_restoring_it_matches_again, R3e_an_mtime_change_after_t | RED |
| gate_token.py / launch_gate.py | anchor_listing_problems: an EXTRA file is not reported | R4_an_extra_file_refuses_every_intermediate_entry_point, R4_a_renamed_file_is_both_a_missing_and_an_extra_file | RED |
| gate_token.py / launch_gate.py | anchor_listing_problems: a MISSING file is not reported | R4_a_missing_file_refuses_every_intermediate_entry_point, R4_a_renamed_file_is_both_a_missing_and_an_extra_fil | RED |
| gate_token.py / launch_gate.py | _stat_listing: a symlinked directory is silently skipped | R4_a_symlinked_directory_is_a_problem_not_a_silent_skip | RED |
| gate_token.py / launch_gate.py | end_verify: the CONTENT comparison with the anchor is deleted (the end verify can never void on byte | R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate | RED |
| gate_token.py / launch_gate.py | end_verify: the final STAT comparison is deleted | R3e_an_mtime_change_after_the_last_intermediate_call_voids_at_the_end_though_the_bytes_are_equal | RED |
| gate_token.py / launch_gate.py | end_verify: the record is always END_VERIFY_OK | R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate | RED |
| gate_token.py / launch_gate.py | anchor_binding_refusals: the cache-directory comparison is deleted | R6_an_anchor_minted_from_another_cache_directory_is_refused_by_every_entry_point | RED |
| gate_token.py / launch_gate.py | anchor_binding_refusals: the chain-id comparison is deleted | R6_an_anchor_of_another_chain_is_refused | RED |
| gate_token.py / launch_gate.py | anchor_binding_refusals: the 'older than the chain' comparison is deleted | R6_an_anchor_minted_before_the_chain_started_is_refused_by_every_entry_point | RED |
| gate_token.py / launch_gate.py | load_anchor: the signature check is deleted | R6_an_edited_anchor_and_an_anchor_signed_with_another_key_are_refused | RED |
| gate_token.py / launch_gate.py | mint_anchor: the comparison with the registered pin is deleted | R6_the_anchor_is_minted_only_for_the_registered_sealed_untampered_cache, R6_a_resealed_self_consistent_cache_i | RED |
| gate_token.py / launch_gate.py | mint_anchor: the 'changed while being anchored' comparison is deleted | R6_a_cache_edited_while_it_is_being_anchored_is_not_anchored | RED |
| gate_token.py / launch_gate.py | mint_anchor: the content check of the cache at the start is ignored | R6_the_anchor_is_minted_only_for_the_registered_sealed_untampered_cache | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: a missing END record is admitted | R5_the_end_verify_skipped_the_reader_gate_refuses, R5_the_end_record_deleted_after_it_was_written_the_reader_g | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: a VOID record is admitted | R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: the END record's signature is not checked | R5_a_VOID_record_refuses_and_a_hand_edited_one_is_refused_by_its_signature | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: the binding of the END record to THE SAME anchor is deleted | R5_an_END_VERIFY_OK_signed_for_another_anchor_is_refused | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: the seed-coverage comparison is deleted | R5_an_END_record_that_does_not_cover_every_seed_read_is_refused | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: a pass without a token is skipped silently | R5_a_pass_without_a_token_is_refused | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: the single-anchor requirement is deleted | R5_tokens_bound_to_two_different_anchors_are_refused | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: a listing-only token is admitted as a content-bound one | R5_a_token_that_binds_only_a_size_listing_is_refused | RED |
| gate_token.py / launch_gate.py | ctx_chain_refusals: the token signature is not checked | R5_an_edited_token_is_refused_by_its_signature | RED |
| gate_token.py / launch_gate.py | launch_gate.cmd_run: the anchor of the call is not applied | R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1, W_cmd_ru | RED |
| gate_token.py / launch_gate.py | launch_gate.cmd_verify: the anchor is not handed to verify_token | R0_clean_control_anchor_bound_calls_MATCH_the_key_is_the_registered_pin_and_the_mode_is_recorded, W_cmd_verify | RED |
| gate_token.py / launch_gate.py | launch_gate.finalize: the anchor is not handed to the data manifest | R0_clean_control_anchor_bound_calls_MATCH_the_key_is_the_registered_pin_and_the_mode_is_recorded, R7_cost_anch | RED |
| gate_token.py / launch_gate.py | launch_gate.run_checks: the anchor is not handed to the data manifest | R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1 | RED |
| gate_token.py / launch_gate.py | launch_gate.hnscreen_refusals: the anchor is ignored (the cache is re-read in full) | R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1, W_hnscre | RED |
| gate_token.py / launch_gate.py | launch_gate: the --ctx-mode DEFAULT is anchor-bound | R7_default_mode_is_content_it_matches_the_clean_cache_and_refuses_the_same_size_same_mtime_flip | RED |
| gate_token.py / launch_gate.py | launch_gate._with_ctx_anchor: an anchor named under content mode is silently ignored | R7_an_anchor_named_under_content_mode_is_refused_never_silently_ignored | RED |
| gate_token.py / launch_gate.py | launch_gate._with_ctx_anchor: anchor-bound is allowed for any profile | R7_anchor_bound_needs_all_three_flags_and_exists_only_for_the_hn_screen_profile | RED |
| gate_token.py / launch_gate.py | launch_gate._with_ctx_anchor: the three flags are not required together | R7_anchor_bound_needs_all_three_flags_and_exists_only_for_the_hn_screen_profile | RED |
| gate_token.py / launch_gate.py | gate_token.fp_key AND anchor_fingerprint: BOTH layers of the invalid-key guard deleted (an INVALID a | R2_a_size_change_refuses_every_intermediate_entry_point_and_voids_at_the_end | RED |
| hnscreen_devbox.py (the chain) | stage_list: ENDV is not a stage (the end verify is skipped) | C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_ | RED |
| hnscreen_devbox.py (the chain) | stage_ENDV: a failed end verify does not void the chain | C_R1_a_same_size_same_mtime_edit_mid_chain_voids_the_chain_no_output_is_read_and_a_relaunch_cannot_launder_it | RED |
| hnscreen_devbox.py (the chain) | stage_ENDV: a VOID chain is re-verified on a relaunch (a hand-restored cache launders it) | C_R1_a_same_size_same_mtime_edit_mid_chain_voids_the_chain_no_output_is_read_and_a_relaunch_cannot_launder_it | RED |
| hnscreen_devbox.py (the chain) | ctx_args: the intermediate gate calls get no anchor flags (every call re-reads in full) | C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_ | RED |
| hnscreen_devbox.py (the chain) | read_argv: the reader is not given --chain-root | C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_ | RED |
| hnscreen_devbox.py (the chain) | stage_list: ANCHOR is not a stage (the chain has no start anchor) | C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_ | RED |
| hnscreen_devbox.py (the chain) | preflight: any gate.ctx_mode is accepted | C_an_unknown_ctx_mode_is_a_config_error_before_anything_runs | RED |
| hnscreen_devbox.py (the chain) | DEFAULTS: the chain's default gate.ctx_mode is content | C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_ | RED |
| hnscreen_devbox.py (the chain) | stage_ANCHOR: a refused anchor does not stop the chain | C_a_tampered_cache_is_refused_at_the_anchor_and_nothing_is_trained | RED |
| hnscreen_devbox.py (the chain) | verify_pass_token: the start verify of a pass is not run with the anchor flags (a stale-token-only c | C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_ | RED |
| hn_screen_read.py (the reader) | the reader's cache-binding gate is not called | G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read, G_R1_a_VOID_chain_is_refused_before_an | RED |
| hn_screen_read.py (the reader) | the reader ignores a refusal of the gate | G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read, G_R1_a_VOID_chain_is_refused_before_an | RED |
| hn_screen_read.py (the reader) | a gate that cannot run is a pass | G_a_missing_gate_key_is_a_refusal_never_a_pass | RED |
| hn_screen_read.py (the reader) | the reader checks no seed (an empty pass list) | G_R5_an_anchored_chain_without_an_end_record_is_refused_before_any_score_is_read, G_a_chain_root_with_no_tokens_is_refus | RED |
| hn_screen_read.py (the reader) | --chain-root is optional | G_the_chain_root_is_a_required_argument | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.content_check: the per-file byte comparison is deleted | U_invalid_flip, G_refused_flip_pin_registered | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.hnscreen_refusals: the comparison with the registered pin is deleted | G_refused_resealed_pin_registered, G_refused_a_wrong_dirc_pin_on_the_sealed_cache | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.hnscreen_refusals: BOTH content comparisons are deleted (the same-size flip must now PAS | G_refused_flip_pin_registered, G_refused_flip_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refus | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.verify_token: the data fingerprint comparison is deleted (the supervisor trusts the token | G_verify_REFUSES_flip_after_the_token_was_minted_in_process_and_as_subprocess, G_verify_REFUSES_add_after_the_ | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.cmd_verify: the content flags are not passed to verify_token (the profile -> caller wiri | G_verify_clean_control_MATCHES_in_process_and_as_the_chains_subprocess | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.finalize AND cmd_verify both drop the content flags (the token binds a size listing and | G_mint_finalize_PASSES_on_the_sealed_cache_and_binds_its_CONTENT_fingerprint, G_verify_REFUSES_flip_after_the_ | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.finalize: the content flags are not passed to data_manifest (the token binds a size list | G_mint_finalize_PASSES_on_the_sealed_cache_and_binds_its_CONTENT_fingerprint, G_mint_finalize_FAILS_on_a_same_ | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.run_checks: the content flags are not passed to data_manifest | G_run_checks_fails_every_check_on_a_same_size_flip_and_records_the_content_fingerprint | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.content_flags_of: the profile no longer fingerprints by content | U_only_the_hn_screen_profile_fingerprints_by_content_and_only_ctx_cache, G_verify_REFUSES_flip_after_the_token | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | launch_gate.hnscreen_refusals: --ctx-cache is fingerprinted by size listing again (the defect) | G_clean_control_the_sealed_cache_passes_the_refusals_cli | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.content_check: a missing manifest is silent (no problem, nothing listed) | U_invalid_nomanifest, G_refused_nomanifest_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.content_check: an extra file is not reported | U_invalid_add, G_refused_add_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.content_check: a missing file is not reported | U_invalid_delete, G_refused_delete_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.content_check: a stale manifest (a resized file) is not reported | U_invalid_stale, G_refused_stale_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.content_check: the canonical-form check is deleted | U_invalid_noncanonical, G_refused_noncanonical_pin_registered | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token._content_listing: a symlinked directory is silently skipped | U_a_symlinked_directory_is_a_problem_not_a_silent_skip | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | gate_token.fp_key: an INVALID content fingerprint is keyed like a valid one (problems ignored) | U_defect_reproduced_listing_fingerprint_blind_to_a_same_size_flip | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | ctx_cache.finalize: the cache is not sealed at build | B_finalize_leaves_a_sealed_cache_covering_manifest_json_and_every_unit | RED |
| gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code) | hn_fingerprints: --ctx-cache is described by the size listing | B_hn_fingerprints_py_prints_the_dirc_key_and_refuses_an_unsealed_or_tampered_cache | RED |

Mutants listed: 68 (68 RED, 0 not RED).  The reader's table lists only the 5 NEW mutants; the 30 older ones of `test_hn_screen_read.py` are unchanged and still RED (see its log).

## Logs

* gate_token.py / launch_gate.py: `test_hn_ctx_anchor_matrix_run3.log` -> ZZTEST_PASS 72  (38 arms on the real tree + 34 RED mutations)
* hnscreen_devbox.py (the chain): `test_hnscreen_ctx_chain_matrix_run2.log` -> ZZTEST_PASS 19  (9 chain arms + 10 RED mutations of hnscreen_devbox.py)
* hn_screen_read.py (the reader): `test_hn_screen_read_run1.log` -> ZZTEST_PASS 36  (1 unmutated suite (incl. items 5, 6, CLI, paired families) + 35 RED mutations)
* gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code): `test_hn_content_fp_run2.log` -> ZZTEST_PASS 81  (62 arms on the real tree + 19 RED mutations)
