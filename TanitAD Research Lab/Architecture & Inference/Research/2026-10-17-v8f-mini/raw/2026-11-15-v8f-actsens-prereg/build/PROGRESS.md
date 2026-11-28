step1 export+baseline PASS trainer sha 510b1e61 files 3397
step2 IDM lever+test_v8f_actsens.py PASS 15 tests; mutations M1-M8 RED (see mutation_results_*idm*.json)
step3a DIVRS ported (trainer hunks hand-fixed), test_v8f_divrs 18 passed + 3 ladder-path tests; 7 mutations running
step5a package unit tests UNIT ok=464 bad=0
step4 C8 PartA PASS zero=exactly 0.0, control 8.72%/10.32% sha c351c088d0182984
step4 C1 synthetic two-tree parity PASS (m2b + wmonly state sha, live.terms, groups grad EQUAL); actsens.diff sha 0c66c201
step3 DIVRS 7 mutations D1-D7 all RED (mutation_results_divrs_ALL.json); IDM M1-M8 RED
step2/3 final tests on committed tree: test_v8f_actsens+divrs+ladder 65 passed
step5 box2 package: UNIT 464/0, DRYRUN 110/0, shipcheck PASS; manifest built
