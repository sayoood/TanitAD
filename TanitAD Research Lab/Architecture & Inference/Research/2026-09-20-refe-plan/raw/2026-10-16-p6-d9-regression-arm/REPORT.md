# P6 D9 refusal: the cache was bit-identical; the gate's regression arm was blind on saturated sets

**Master Mind, 2026-10-16 ~03:10 Berlin.** P6's chain `REFUSED: D9: cached parts are NOT bit-identical to fresh parts` at 00:42:59Z, at the end of its ~24 CPU-h metric-cache build.

## What actually failed (MEASURED on the pod; `pod_d9_verify.out`, `d9_regression_diag.json`)
- **Identity: cached = fresh on 96 of 96 sets**, over 24 tokens × 4 random proposal sets. The refusal message was wrong about what failed.
- The failure was the **regression arm**. It shifts the cached ego rear axle +5 m and must make the scores differ. It left **22 of 96** sets equal, and the old rule demanded `regression_equal == 0`.
- The diagnostic replays `exactness` exactly (same tokens, rng seed 0, same proposals, same shift):
  - **all 22 undetected sets are SATURATED**: every proposal's fresh PDMS is exactly 0 (17 also have DAC = 0 on all eight);
  - **0 of 20 non-saturated sets were undetected**;
  - 54 further saturated sets were still detected, through TTC, comfort or DDC.
  - ⇒ a 5 m ego shift cannot be required to move a set already scoring zero everywhere. The rule asked for something impossible on 23 % of the random sets.
- Per token: 21 of 24 tokens have at least one detected set. The other 3 drew four saturated sets each.

## Root cause class
**A gate's deliberate-regression arm that was never run on real data.** `test_p6_chain.py` stubs `ZZD9_PASS`, so the arm's behaviour on real scenes had never been observed before the 24 CPU-h build finished. This is the same family as CLAUDE.md's "built, tested, and unreachable from its caller", from the other side: built, tested on a stub, and wrong on the real input distribution.

## The fix (`refe/build_metric_cache.py`, `refe/test_d9_accept.py`)
`d9_accept` PASSES only if all three hold:
1. **IDENTITY:** cached = fresh on every set (unchanged).
2. **SENSITIVITY:** the regression arm moved **every non-saturated set**. A non-zero set it leaves equal means the comparator is blind there, so the gate FAILS.
3. **NON-VACUITY:** it moved at least half of all sets. A blind comparator, or an all-saturated sample, can never pass on saturation alone.

Supporting changes:
- `exactness` now classifies each set (saturated, detected) and counts the tokens with a detected set.
- A failing gate prints its reason (`ZZD9_FAIL <why>`). `p6_pod.sh` greps only `^ZZD9_PASS`, so the chain is unchanged.
- `build_metric_cache.d9fix.diff`: +36 / −4 lines. Nothing else in P6 changes.

**Tests:**
- `test_d9_accept.py` **ZZTEST_PASS 16**, both standalone and inside the archived P6 tree (963638a + the fix). It covers the measured pod sample, each clause's deliberate-regression arm, and the OLD rule reproduced refusing the bit-identical pod sample.
- **Three source mutations go RED:** dropping the sensitivity clause, dropping the non-vacuity clause, and classifying every set as saturated.
- `test_p6_chain.py` **ZZP6CHAIN_PASS 341** on 963638a + the fix.

## ⚠️ What this does NOT relax
- Identity stays exact on every set.
- A set that CAN move must move.
- The numbers above were known before the rule was written. That is acceptable only because this is an **instrument** gate, not a scientific outcome, and the rule is stricter than "count anything". It is stated here so nobody reads it as a pre-registration.

## Shipping
- P6's shipped tree stays **963638a**, so the field-loss wiring of `134be4f` is NOT pulled in.
- The ship commit is a side commit **on top of 963638a** that changes exactly these two files (branch `p6-ship/963638a-d9fix`).
- The same two files land on `agent/arch-inf-20260803` (this commit), so nothing is stranded.
- On restart the chain re-runs the cache stage. The build skips existing parts, then verify runs, then lanes, lab, mem, egr and the launch gate, all beside P2-s1 / R0c (overlap mode).
