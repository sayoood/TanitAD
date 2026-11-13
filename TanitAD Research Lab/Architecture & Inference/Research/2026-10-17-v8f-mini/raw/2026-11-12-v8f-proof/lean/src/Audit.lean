import V8proofs.Pool

/-! Axiom audit of the pooling lemma: every line below must list only `propext`, `Classical.choice`, `Quot.sound`
(the standard axioms of Lean 4 with Mathlib) and never `sorryAx`. -/

open V8Proofs.Pool

#print axioms pair_bound
#print axioms pair_bound_attained
#print axioms pair_bound_vec
#print axioms stronger_pair_bound_false
#print axioms two_position_floor_tight
#print axioms four_position_floor_tight
#print axioms variance_is_floor
#print axioms zero_skill_of_equal_states
#print axioms pool_scene
#print axioms pool_move_inside
#print axioms pool_move_across
#print axioms grid_scene_diff
#print axioms grid_separates
#print axioms grid_blind_without_live_adapter
#print axioms blockState_move_inside
#print axioms blockState_move_across
#print axioms location_zero_skill_blocks
#print axioms grid_sees_motion_enter
#print axioms grid_sees_motion_leave
#print axioms mean_move
#print axioms mean_scene
#print axioms mean_blind_to_motion
#print axioms location_zero_skill_mean
#print axioms location_skill_le_zero_mean
#print axioms mean_sceneList
#print axioms mean_depends_on_count_only
#print axioms mean_count_diff
#print axioms mean_separates_counts
#print axioms mean_confuses_two_with_double
#print axioms sum_perm_invariant
#print axioms readout_perm_invariant
