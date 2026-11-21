# READER_FIX_RREAD1 — eval-tree model module swapped to the trained tree's copy (2026-11-03, before any R rollout)

**What failed.** The first chain launch (2026-11-03T03:48:11) stopped at gate 3 (pre-flight `PRE_R_s0_kbox`, rc 7):
`V8FMiniConfig.__init__() got an unexpected keyword argument 'launch_intent_w'` (`logs/smoke_PRE_R_s0_kbox.out`).
The box-2 runs were trained from the `ad0bcb72` tree, whose `stack/tanitad/models/v8f_mini.py` carries the
`launch_intent_w` field another session added for `PREREG_V8F_LAUNCH_INTENT`. The dev-box eval tree
(`F:/TanitAD-scratch/recover_read/tree_dag`, a copy of the FRAMEFREEZE tree) predates it. The pre-flight did its job:
no rollout and no number existed when it refused.

**What the field is (MEASURED, diff of the two modules, 45 diff lines, all additions):** a config float (default `0.0`),
its finiteness check, two module-level functions (`launch_mask`, `launch_along_term`) that only the trainer calls, a
loss term computed only when `launch_intent_w > 0`, one total line gated the same way, and one `built_levers` key.
No forward-path line changes. All six RECOVER-FT checkpoints carry `launch_intent_w = 0.0` (read from `ck["cfg"]`).

**The fix.** Only `stack/tanitad/models/v8f_mini.py` in `tree_dag` is replaced by the `ad0bcb72` blob (the original is kept
beside it as `v8f_mini.py.pre_launch_intent`). Nothing else in the eval tree changes: the other seven `stack/tanitad`
files that differ from `ad0bcb72` (`v8f_mini_data.py`'s training-side ego-perturbation stream, and modules of other
programmes) stay as they are, so the eval tree is the one that produced the banked P10 / DAG-OACC rollouts this read
compares against. The tool (`v8f_t1_eval_framefreeze.py`, pinned sha `0721fc01…`), the reader, the tests and both
frozen cards are untouched.

**Control (bit-identity, MEASURED):** DAG-OACC s0, 6 windows, `--smooth sg72 --scene logged --feasible kbox`, 16 threads,
run with the original module and again with the swapped one; every array of the two ticks files must be byte-equal.
Result (`raw/SWAPCTL.txt`, logs `logs/swapctl_{BEFORE,AFTER}.out`): the 35 data arrays (8,625 elements) are byte-equal. The comparator as
written printed FAIL because it also compared the `tag` array, which holds each run's own name (`BEFORE` / `AFTER`); disclosed there.
Relaunched 2026-11-03 after this control.

Class: instrument environment (a stale module in a copied eval tree), not an analysis change; disclosed per the
post-data amendment rule although no data existed yet.
