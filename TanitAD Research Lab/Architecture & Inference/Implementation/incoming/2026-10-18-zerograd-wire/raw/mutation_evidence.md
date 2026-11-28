# Mutation evidence for test_sup_refcv7_zerograd.py (MEASURED 2026-10-18, this box)

## M1: the hook line 'zerograd_check "$fs"   # ZG-HOOK' replaced by 'ZG_STATUS=PASS'
```
FAILED test_clean_run_writes_the_done_marker_with_a_zerograd_pass_record
FAILED test_a_checkpoint_older_than_the_requested_steps_is_flagged_loudly_not_silently_gated
FAILED test_untrained_head_refuses_the_marker_writes_the_fail_marker_and_never_relaunches[True]
FAILED test_untrained_head_refuses_the_marker_writes_the_fail_marker_and_never_relaunches[False]
FAILED test_a_trainer_written_summary_is_gated_by_a_fresh_supervisor_without_launching
FAILED test_a_checkpoint_without_optimizer_state_is_recorded_not_applicable_never_silently_a_pass
FAILED test_a_gate_that_cannot_look_is_unevaluated_and_fails_closed_like_a_fail[garbage_ckpt]
FAILED test_a_gate_that_cannot_look_is_unevaluated_and_fails_closed_like_a_fail[no_ckpt]
FAILED test_a_gate_that_cannot_look_is_unevaluated_and_fails_closed_like_a_fail[gate_missing]
FAILED test_a_gate_that_cannot_look_is_unevaluated_and_fails_closed_like_a_fail[crash_exit1]
FAILED test_a_gate_that_cannot_look_is_unevaluated_and_fails_closed_like_a_fail[json_disagrees_with_rc]
FAILED test_allow_list_exempts_frozen_by_design_parameters_and_is_logged_and_recorded
FAILED test_a_failed_positive_control_fails_the_run
FAILED test_gate_off_is_loud_and_recorded_and_a_typo_refuses_instead_of_disabling
FAILED test_manifest_zerograd_vars_are_not_inherited_from_the_callers_environment
FAILED test_MUTATION_without_the_hook_the_untrained_head_writes_the_done_marker
FAILED test_the_hook_line_sits_in_finalize_run_and_both_completion_paths_call_it
17 failed in 73.03s (0:01:13)
```

## M2: startup test restored to the OLD 'done-marker present -> FINISHED, exit 0' (condition replaced by 'if true; then')
```
FAILED test_untrained_head_refuses_the_marker_writes_the_fail_marker_and_never_relaunches[True]
FAILED test_a_trainer_written_summary_is_gated_by_a_fresh_supervisor_without_launching
2 failed, 15 passed in 189.17s (0:03:09)
```

## Trainer mutant: _opt_param_names replaced by [n for n,_ in model.named_parameters()] (test_zerograd_trainer_payload.py)
```
3 failed, 2 passed: test_param_names_follow_the_optimizer_id_order..., test_the_trainers_checkpoint_carries_names...[dd], test_the_gate_prefers_the_embedded_names...
(the adam parametrisation and the error_kind test pass, as they should: for plain Adam the two lists coincide)
```

## The same 5 payload tests on the tip export (no param_names, no error_kind): 5 failed
