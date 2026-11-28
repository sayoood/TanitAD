# HARDENING (eps) : APPLIED

The final deliverables are the swept tree + one extra check in refe/test_rl_safeobj.py: `U_eps_shifts_the_violation_and_the_term_by_exactly_eps_through_epo_con_term_and_through_train_losses` and mutation M27 (the call site passes a literal 0.0 for the slack); M16 (eps ignored by the term) now also goes RED behaviourally, not only on the pin.
Reason: eps = 0 in every registered arm, so no earlier behavioural check read the slack (M16 was '[PIN ONLY]' in the first sweep).
Evidence: logs/sweep/logs_t3 (test_rl_safeobj on the hardened tree) and logs/sweep/logs_t4 (test_rl_r0c, which imports it).
No trainer, gate or chain line changed; the trainer blob is the one the whole sweep ran on.
