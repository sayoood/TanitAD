import V8proofs.Rule

/-! Axiom audit of the rule, imagination and weight-averaging theorems (T3, T7, T9): every line must list only
`propext`, `Classical.choice`, `Quot.sound` and never `sorryAx`. -/

open V8Proofs.Rule

#print axioms vb_nonneg
#print axioms fallback_feasible
#print axioms shield_step
#print axioms shield_forever
#print axioms shield_keeps_safe_proposal
#print axioms penalty_bound
#print axioms masked_argmax
#print axioms violations_le_misses
#print axioms imagined_gain_is_real
#print axioms imagined_choice_regret
#print axioms exists_choice_within_two_gaps
#print axioms two_gaps_attained
#print axioms one_gap_not_enough
#print axioms return_gap_le
#print axioms model_gap_from_step_error
#print axioms ensemble_identity
#print axioms ensemble_le_member_sum
#print axioms linear_head_soup
