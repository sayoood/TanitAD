# `refe/test_scorerft_profile.py` went RED once `SCORERFT_RANK_W` is filled (MM, 2026-11-21)

**Why it matters.** `test_scorerft_profile.py --fast` is in `SCORERFT_SUITE_EXTRA` (`launch_gate.py:719-720`), so it runs inside T's G-SUITE on the pod. T is launched from package B,
which is package A plus the registered weight. With the test as landed in `8dfc1e7b`, T's launch gate would have refused T after the weight rule (due 2026-11-21 ~17:00Z).
The control arm C and the weight rule run from package A, where the pin is still `None`, so they were not affected.

**How it was found.** A rehearsal, not an incident: on a scratch copy of `8dfc1e7b`'s `refe/` (`F:/mm_e07efd49/wrdry`), `SCORERFT_RANK_W = None` was set to `0.5`, and the
profile tests were run before the real weight existed.

**Two defects, both in the test's fixtures (the gate code is correct):**
1. `targv()` and `filled()` used the fixture `W_FIX = 0.25`. Once the pin is filled, the profile REQUIRES `--rank-w repr(W)`, so the fixture argv was refused:
   `AssertionError: T: the template argv with the pins filled must be clean: refe_scorerft_t requires --rank-w 0.5` (`original_file_w0.5_fast.log`).
2. The refusal arm planted `--rank-w 0.5` as the wrong weight. 0.5 is one of the weight rule's own candidates (`weight_rule_rank.CANDIDATES` = 2^-6 ... 2^2), so at W = 0.5
   the planted value was the right one and was not refused: `AssertionError: T must refuse: rank-w` (`fix_v1_w0.5_fast.log`, after fixing 1 only).

**Fix.** `W_ARGV = float(L.SCORERFT_RANK_W) if L.SCORERFT_RANK_W is not None else W_FIX` is used for the template argv, the filled profile and the T-rule check. The planted
wrong weight is `repr(W_ARGV + 0.5)`. `W_FIX` stays the number used by the numeric gradient captures (`reach_cap`, `dvb_cap`).

**Evidence.** `matrix_fixed_all_candidates.tsv`: the fixed file at `None` and at all nine candidates passes `--fast` (242 checks). It also passes full (272 checks) at None, 0.25,
0.5 and 4.0. Those are the same counts the file gave before the fix. `filled_w0.5_pin_and_chain_tests.tsv`: the profile-pin tests and the P5-C / P7 chain tests with the pin filled at 0.5.
The deliberate-regression arm is the original file: RED at 0.5 (above).

**Class.** Same as "built, tested, and unreachable from its caller": a test that had only ever run with the pin unfilled. The state it guards (pin filled) had never been executed
before it would gate a launch.
