# PREREG_3C — ADDENDUM 3 (Master Mind, 2026-10-17 ~10:20Z): the parts build's D7 distinct check could not import its module. Disclosed; no bar changed.

`code/build_twin_parts_3c.sh` built the K=1 twin parts cleanly (`ZZMETRIC_CACHE_DONE built=32929 skipped_5s_guard=0 failed=0 logs=980 6708s`),
then its acceptance check `check_twin_parts_distinct.py` died on `ModuleNotFoundError: No module named 'online_labeller'` (log kept as
`full/parts_distinct_k1.IMPORT_FAILED.out`). Cause: the script `cd`s to `$PKG/refe` but runs the check BY PATH, so Python's `sys.path[0]` is
the check's own directory; `$PKG/refe` (where `online_labeller.py` lives) was never on `PYTHONPATH`, which the check's docstring requires.
The rehearsal could not see it: it ran on the mini bank's existing parts and never executed the parts build. Class: built, tested, UNREACHABLE path.

**Fix without touching the frozen code dir** (`run_full.sh`'s `shard_meta` pins it): `build_parts_continue.sh` (this package; pod copy
`/workspace/ops/bank3c/build_parts_continue.sh`) repeats the build script's commands verbatim, adding `$PKG/refe` to `PYTHONPATH` for the check
only: the K=1 check on the parts already built, then K=2 and K=3 build + check, then `PARTS_DONE`.
**K=1 check (MEASURED):** `ZZD7_DISTINCT_OK k=1 sampled=40 differ=37 frac=0.925 min_frac=0.9` — 3 of 40 sampled twins share the logged route,
consistent with the 5.9 % of K=1 twins whose fixed route equals the logged one (those read f = 0.5 and are dropped by the commit filter).
