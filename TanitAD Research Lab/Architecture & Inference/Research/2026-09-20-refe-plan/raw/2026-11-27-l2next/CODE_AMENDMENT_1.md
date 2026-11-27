# CODE AMENDMENT 1 to the L2-NEXT build (SH-MLP + EP-HEAD), 2026-11-27, MM

**Timing.** Made after the cards were frozen (18:05Z) and BEFORE any SH-MLP or EP-HEAD head was trained on real data, before any gate token, and before any read. The cards (`PREREG_SHMLP.md`, `PREREG_EPHEAD.md`) and their sha256 are unchanged, and `CODE.sha256` stays as frozen. This file and `CODE_AMENDMENT_1.sha256` record the two files whose hashes moved. No bar, outcome rule, estimator, recipe value or selection rule changed.

**Why.** The regression run of `test_safehead_next.py` (full mode, after the agent's `--fast` 13/13) failed because two deliberate-regression arms were NO-OPs for the new families, so their gate checks could not go RED. A mutation that cannot fail proves nothing (CLAUDE.md "BUILT, TESTED, AND UNREACHABLE": mutate, do not inspect).
1. **G-CKPT, family shmlp (`sh_ckpt_bias_dropped` -> PASS, expected FAIL).** In a schema /2 head, `W` and `b` are C1's (frozen); the training lives in the residual MLP. Replacing `b` with `b0` changed nothing, and the round trip compared only `W` and `b`. So **G-CKPT was blind to every MLP array**: a corrupt or missing MLP in the saved file would have passed.
   - Fix (gate): the round trip also compares every `MLP_KEYS` array of the loaded head with the in-memory probe.
   - Fix (trainer, mutation path only): for schema /2 the arm zeroes the trained MLP output bias `mlp_b2` in the saved file. Schema /1 is unchanged.
2. **G-LIVE, family ephead (`sh_live_term_dropped` -> PASS, expected FAIL).** The arm dropped the TTC term (row 3), which EP-HEAD (row 2 only) never declares.
   - Fix (trainer, mutation path only): drop TTC's term when TTC is trained (SAFE-HEAD and SH-MLP: unchanged), else the last trained row's (EP-HEAD: `qfl_ep`).

**Non-mutation code paths are unchanged**, for schema /1 and /2. The diffs are `code_amendment_1_train.diff` (28 lines) and `code_amendment_1_gate.diff` (11 lines).

**Verification** (`regression/fix2.tsv`, serial, commit gate >= 8,000 MB):
- `test_safehead_next.py` full: **ZZSHNEXT_PASS 17**. Every deliberate-regression arm of both families turns its check RED.
- `test_safehead.py`: **ZZSAFEHEAD_PASS 14** (SAFE-HEAD unchanged).
- `test_hnscreen_chain.py`: **ZZTEST_PASS 31**.
