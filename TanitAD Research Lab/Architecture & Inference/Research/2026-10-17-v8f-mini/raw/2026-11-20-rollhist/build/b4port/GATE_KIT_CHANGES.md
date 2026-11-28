# Gate kit port (b3gate -> b3gate_cdb), Amendment 1
Constants/paths that bound 2bda9e1a, each changed:
1. test_dscale10k_check_rh.py:35 `COMMIT` was "2bda9e1a382986cf..." -> `os.environ.get("RH_NEWCOMMIT", "PLACEHOLDER_NEWCOMMIT_MM_FILLS")` (MM fills with the port commit; it is only a value passed through to mint/run/token_problems).
2. ref/tree (the stub tree the tests load rollhist_pair_check.py / v8f_dagger.py / v8f_mini_data.py from): v8f_mini_data.py replaced by cdbff971's blob (was 2bda9e1a's 4b89f122..); rollhist_pair_check.py and v8f_dagger.py are byte-identical to 2bda9e1a's (same sha256, pair-check 0dc30d9e...); the two empty __init__.py stubs kept empty.
3. Comment/docstring text only: dscale10k_check_rh.py:803 (+ the NOGENCMP mutant copy), rh_evidence.py:6-7 (INHERITED evidence now valid on the port commit per Amendment 1), test line 289 comment.
Unchanged: CODE_COMMIT = "648d4ca3" (UM_s0 code of record, a different binding), the fixtures (UE_s0.A / U_sK.B templates), the 37 tests, the UE identity test.
No sha or commit constant in dscale10k_check_rh.py itself bound 2bda9e1a (grep). rh_arm.sh / chk_arm_rh.sh: no match.
