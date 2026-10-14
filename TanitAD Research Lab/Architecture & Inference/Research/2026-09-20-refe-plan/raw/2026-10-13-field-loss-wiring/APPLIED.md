# Field-loss patches APPLIED (Master Mind, 2026-10-14 ~21:40 Berlin)

Applied after P6 shipped (`/workspace/refe-plan-p6` = 963638a), as `RESULT.md` step 1 requires. The shipped trees on the pod (P6, S1, P2-s1, RL-S1)
carry their own copies and are untouched.

## How it was applied

| file | how | evidence |
|---|---|---|
| `refe/train.py` | `patch -p1` clean (`PATCH_train.diff`) | the pre-patch tip blob md5 `cb2f0f3f...` equals `patch_dry_run.txt`'s original, so the result is byte-identical to the copy this folder's proofs ran on, including (a) default path bit-identical |
| `refe/gate_checks.py` | clean | the pre-patch md5 `77395f08...` equals the tested original |
| `refe/test_launch_gate.py` | clean | the pre-patch md5 `826e151b...` equals the tested original |
| `refe/launch_gate.py` | hunks 1 and 3 clean; **hunks 2 and 4 rejected** | the original had changed since the dry run: the `refe_rls1b` profile, 5a2517b. Both rejected hunks are pure insertions; they were inserted by exact anchor (before `_BASE = dict(` and before the `trainer_src` read) |
| `refe/weight_rule.py` | copied (new file, step 3) | |

**The `launch_gate.py` merge equals the patch exactly.** Against the tip blob, the merged file adds **33 lines, the same multiset as the patch's 33
`+` lines**, and removes 0, the patch's 0 (`difflib` comparison).

## Suites after applying (dev box)

| suite | result | note |
|---|---|---|
| `test_launch_gate.py --static` | **ZZLG_STATIC_PASS 637** | = 552 + 85, `RESULT.md`'s prediction |
| `test_launch_gate.py --static --no-banked` | **623** | = 538 + 85 |
| `test_field_loss.py` | ZZTEST_PASS 42 | |
| `test_rl_s1b.py` | ZZTEST_PASS 18 | |
| `test_rl_epo.py` | ZZTEST_PASS 42 | |
| `test_p6_chain.py` | ZZP6CHAIN_PASS 341 | |

`test_field_wiring.py` (this folder) can no longer run as written: it applies the patches to a copy of the UNPATCHED originals and now finds them
already applied ("Reversed (or previously applied) patch detected"). Its proofs stand for `train.py` / `gate_checks.py` / `test_launch_gate.py` by
the md5 identity above, and for `launch_gate.py` by the exact-lines check and the 637 / 623 counts.
