import V8proofs.Hier

/-! Axiom audit of the hierarchy theorems (T1, T2): every line must list only `propext`, `Classical.choice`, `Quot.sound`
and never `sorryAx`. -/

open V8Proofs.Hier

#print axioms geom_succ
#print axioms compose_bound
#print axioms compose_bound_attained
#print axioms jump_bound
#print axioms geom_factor
#print axioms jump_sum_pos
#print axioms jump_beats_compose_iff
#print axioms jump_beats_compose_iff_one
#print axioms risk_split_one
#print axioms mean_is_optimal
#print axioms risk_split
#print axioms mean_plan_misses_mode_a
#print axioms mean_plan_misses_mode_b
#print axioms decided_hits_mean_misses
#print axioms single_plan_cannot_hit_both
#print axioms flat_hits_one_mode
#print axioms flat_hit_count_le
#print axioms decided_hit_count_ge
#print axioms hierarchy_dominates_flat
