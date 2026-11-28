# Mutation record (ROLLHIST gate build B3, 2026-11-20)
Mutation: in `rh_bank_generator_problems` delete the generator-record comparison (the two `if tk != token` / `if kp != kappa` blocks). File: `mutants_dscale10k_check_rh_NOGENCMP.py`
(= dscale10k_check_rh.py minus exactly those 4 lines). Run: `DS_CHECK_RH_PATH=<mutant> pytest test_dscale10k_check_rh.py` (log `pytest_log_mutant.txt`).
Unmutated: 37 passed (`pytest_log_new.txt`). Mutant: 35 passed, **2 FAILED**:
- `test_bank_header_without_generator_record_refused`  RED (required control: the header without a generator record mints once the comparison is gone)
- `test_wrong_declared_token_refused`  RED (its last case: argv + CLI say causal_g1x, bank generated as causal_g1)
Every other control stays green on the mutant, so each of them is held by a different line (argv, sha, pair, init). Evidence class: MEASURED (logs above).
