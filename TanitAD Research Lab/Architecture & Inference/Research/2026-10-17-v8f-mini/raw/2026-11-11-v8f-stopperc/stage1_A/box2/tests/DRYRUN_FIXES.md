# STOPPERC dry run: changes to the package under test (2026-11-12)

The dry run changed `stopperc_chain.sh` and `stopperc_check.py`. Old shas are those of the package as received.

| file | old sha256 | new sha256 |
|---|---|---|
| `stopperc_chain.sh` | a62733d59614bc84738952773c8736ac6b396acbf32f6a21692c5bf2e868e4f9 | f69d10973c2c263b074f0ff90c24129a89b4693fddea4bd10434cbe7b033a51f |
| `stopperc_check.py` | a135a0eae000ef799fea83e0716d0b22732f9d008d32dc9ad865fd19af665004 | 4e3dca4197e1c963a25c74bf630bbe5987b1c6daa83078d35903c6bedab54f3f |

Unchanged: argvs_stopperc.txt (b005ea0b...), make_argvs_stopperc.py, both job .cmd files, and every STOPBAL file (stopbal_check.py 70e260a6...5212 is still the pin).

## 1. REAL BUG (fixed): a W that rounds to 0.0000 passed the W smoke
`wsmoke_stage` accepted any `^[0-9]+\.[0-9]{4}$` W, including `0.0000`. The checker refuses a non-positive `STOPPERC_W` (`stopperc_check._w_final`), but only when the first A argv is built, i.e. AFTER NULL_s0 and NULL_rep2 (about 80 min of box 2). Found by dry-run scenario 3 (goal term 1e-5 gives W 0.0000: the chain ran on to A_s0).
Fix (chain, line 86): after the regex, require `float(W) > 0` or `die 20` (STOP rc 14 at the W smoke, no run spent). The dry run now stops there (rc 14, no `done/wsmoke.done`, no run on the box).

## 2. TEST HOOKS (not bugs; both inert on a real launch)
A throw-away repo cannot carry the code-of-record sha and a fake BAL checkpoint cannot hash to the pinned 5169ea95, so two hooks were added, following the base checker's own pattern (`STOPBAL_FAKE_*`, gated on the fake-box switch):
* `stopperc_chain.sh` line 11: `COMMIT` is replaced by `$STOPPERC_TEST_COMMIT` ONLY when `STOPPERC_DRY=1` (stub transport; the sourced common script refuses DRY unless both ssh and sftp carry the stub marker). A real launch has DRY unset. The dry run proves the default commit is 3e264bbf (scenario 6, `nocommit`: ship folder keyed by it, refusal rc 8) and that a stub without DRY refuses (rc 16).
* `stopperc_check.py` `bal_pin()`: `STOPPERC_TEST_BAL_SHA` replaces the pinned BAL-NULL_s0 sha ONLY while `STOPBAL_ROOT` (the base's fake switch) is set. Without the hook the fake BAL checkpoint is UNKNOWN and the chain stops rc 13 (scenario 2, `unknown`): the real pin is exercised too.

Unit tests re-run after both edits: `36 passed`.
