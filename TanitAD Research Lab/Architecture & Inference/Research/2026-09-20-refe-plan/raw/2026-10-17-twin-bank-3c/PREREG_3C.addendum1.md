# ADDENDUM 1 to PREREG_3C.md -- written 2026-10-17 by the build agent, BEFORE any control or production number (disclosed)

`PREREG_3C.md` (sha256 `e8365f34fd5f11a23a1b5e0fceda14dc14426277ae09db3e5335526c1bd6cde3`) is NOT edited. These items resolve points the frozen text leaves
open or gets wrong about the pod, each decided before a number existed. None of them moves the keep rule, the horizon, the goal horizon, or any acceptance bar.

1. **`aug.divergence_m` is recomputed** for a kept row (max per-step displacement of the new 4 s `traj` against the frame's rank-0 `traj`,
   `augment_search.divergence_m` verbatim, exactly as `reroll_twins.py` did for the probe's `rr_h6` rows). PREREG step 3 lists the `aug` additions but is silent
   on this field; leaving the banked value would put the PRE-reroll trajectory's divergence on the new trajectory (a stale field that D7's verify selects on).
   A kept candidate with no rank-0 reference row is DROPPED, reason `no_rank0_ref` (never shipped stale).
2. **Commit rule reads the instrument's own value**: `f_at_s[9]` = `round(f(10 s), 3)`, the number `probe/long_rollout.py` writes and `probe/analyze_long.py`
   judged (`>= 0.75`). The unrounded `f10_raw` is recorded too; any twin where the two readings disagree is counted in `merge_report.json` (expected 0).
3. **Sharding**: a log can hold up to 642 twins (996 logs, 44,409 twins, MEASURED on the pod), so one-shard-per-log would floor wall time at ~3.6 h. Shards are
   `(log, chunk-of-32-twins)` units assigned by LPT (deterministic); a unit never spans logs. `--log-shard i/N` keeps its meaning of "shard i of N". N is recorded in
   every shard's `shard_meta.json` together with the input sha256, and a resume with different N / input / arm is refused.
4. **Yield guard denominator**: the expected K=1 share 0.37 is 17/46, i.e. over the twins whose f(10 s) was MEASURED. The guard (`< 0.20` stops the build) is
   evaluated on that EVALUATED basis (needs K=1 evaluated n >= 30, else NOT_EVALUATED); the share over ALL input twins is reported next to it, and if the two
   readings disagree on tripping, `merge_report.json` says so and the Master Mind decides (merge exits 4 on the evaluated-basis trip only).
5. **(c'2) baseline** = the probe's "held, 12 s" arm: `reroll_twins.py run --arm held` (goal horizon 12 s, route held), 4 s, same twin; path-length and end-speed
   math copied verbatim from `probe/progress_check.py`. Judged per rank where n >= 5.
6. **Parts coverage (MEASURED 2026-10-17 on the pod, before the controls)**: the D7 parts the PREREG points to (`mc_final`) hold the 74 mini-bank twins only.
   The 44,409 production twin parts that exist (`metric_cache_twins_rejected`) are the PRE-FIX ones (centreline equal to the logged one on 74/74 mini twins
   compared with `mc_final`), so f(t) = 0.5 everywhere on them: unusable. The full run therefore needs a fixed-tree D7 twin build first (command and cost in HANDOFF.md).
   This blocks only priority 3 (the launch); the controls on the mini bank are unaffected.
