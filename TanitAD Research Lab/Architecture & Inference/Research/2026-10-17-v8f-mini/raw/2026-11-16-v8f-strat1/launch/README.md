# STRAT1 launch kit (pod), 2026-11-16

* Code of record: branch `agent/v8f-strat1-20261116` = `22999538` (b9250158 + the launch checker `strat1_check.py` mint / run / suite and `test_strat1_token.py`, LF). Tokens bind that commit and the pod tree, which `strat1_pilot2.sh` checks equal to it before anything runs.
* `gsuite.sh`: G-SUITE, base cdbff971 vs new b9250158, 8 file shards per tree. `strat1_pilot2.sh`: re-runs the three test files the launch commit touches on the 22999538 tree (their junit overrides the b9250158 rows), compares per test id, and only with 0 regressions mints and runs the three 200-step timing pilots. `strat1_pilot.sh` is the first version: it was stopped while waiting, before any mint, because it bound b9250158 to a tree holding the newer checker.
* `strat1_chain.sh`: the 7-arm chain, NOT launched (the card is frozen; the chain waits for the pilot's s/step projection, 10 h stop).
* `inputs.sha256`: the 21 pod inputs, verified against their box-2 / dev-box sources. G-EVAL here checks a strict load, built levers equal the selfcheck's, and refuses a cfg that drops ctx_heads; the 8-window forward_infer parity of BUILD_SPEC section 7 is not built.
