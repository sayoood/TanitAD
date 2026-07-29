# Gate secondary #2 — "NO EMITTER EXISTS" is FALSE. The emitter exists; the ARM's LOG is empty.

**Executed 2026-07-29 08:3x UTC** against the PI's instruction *"build the emitters"* (task #42).
⭐ **Nothing needed building. The correct action is different, and cheaper.**

## What the gate result says vs what is true

`V4_30K_GATE_RESULTS.md:310` records:

| # | metric | measured | verdict as written |
|---|---|---|---|
| 2 | `speed_benefit_recovered_frac` | null | ⚪ *"**NO EMITTER EXISTS** anywhere in the codebase"* |

⛔ **MEASURED — that is false on three counts:**

1. **`stack/tanitad/eval/speed_benefit.py` EXISTS** — 209 lines, with `recovered_frac()` and a
   gate-ready `emit()` whose docstring opens *"``speed_benefit_recovered_frac`` — the v4 gate's
   quiet-plateau KILL secondary"*.
2. **It is WIRED** — `stack/scripts/gate_emitters.py:240` carries `speed_benefit_emit()` as a thin
   adapter, and lists it in its own header table.
3. **It is VALIDATED** — `gate_emitters.py:44` records `speed_benefit_frac = 0.8184` on v1
   (8–10 k bucket, **PASS ≥ 0.70**).

**I ran it against the v4-30k arm.** It executed and returned a complete, gate-shaped record.

## ⭐ THE REAL CAUSE — a DATA gap, not a CODE gap

```
n_arm_rows                  : 0        ← the arm contributes NOTHING to the (8000,10000] bucket
n_nospeed_rows              : 40       ← the control side is healthy
nospeed_control_bucket_mean : 0.5794
arm_bucket_mean             : null
```

**MEASURED on `v4fs_train_log.jsonl`:** the log spans steps **0 → 29,999** and contains
**ZERO occurrences of `g_op_fwd_ade_m`** (`grep -c` = 0). The emitter's input metric was **never
written by that launch**.

⇒ `LOOP_STATE` had already recorded this cause verbatim: *"the from-scratch log has
`g_op_fwd_ade_m` = 0 matches — the `a9dfe223` log-fix did NOT ride this launch."*
**The gate result attributed to a missing emitter what was in fact a missing log field.**

## Corrected verdict for secondary #2

> **The emitter exists, is wired, and is validated (0.8184 PASS on v1). The v4-30k arm cannot supply
> its input because its own training log never recorded `g_op_fwd_ade_m`.**
> ⇒ **UNPRODUCIBLE-FROM-THIS-LOG**, not NO-EMITTER.

## What this changes for the PI's decision (task #42)

The PI chose *"build the emitters"* from the three options in `V4_30K_GATE_RESULTS.md:390-391`.
**For #2 there is nothing to build.** The producible routes are:

1. ⭐ **Re-emit `g_op_fwd_ade_m` from the v4-30k CHECKPOINT** rather than the log — the metric is a
   rollout readout, so it can be computed offline from `v4fs_ckpt.pt` on the 40-episode surface.
   **This is an eval, not a retrain.**
2. **Accept #2 as unproducible for this arm** and adjudicate on the measured secondaries.
3. ⛔ **NOT: re-run the 30 k training with the log-fix** — 4 GPU-days to recover one secondary.

## ⭐ SECONDARY #7 — AUDITED 2026-07-29 09:0x UTC. ITS EMITTER EXISTS TOO.

`V4_30K_GATE_RESULTS.md:311` says *"NO v4-AWARE PANEL (`efficiency.py` has zero v4 awareness)"*.
**MEASURED — the EMITTER is not the gap:**

- **`gate_emitters.deploy_tick_from_eff_json()` EXISTS** (line 117) with a testable logic core
  (`deploy_tick_from_eff_json_dict`), and reads *"the composed deployed tick's p99, NOT the eager
  baseline (which is the un-optimised ~100 ms tick, not deployed)"*.
- **It is PINNED by two tests** in `stack/tests/test_gate_emitters.py:89-101`, including
  `test_deploy_tick_rejects_a_fast_but_WRONG_lever` — i.e. it already refuses a lever that is fast
  but incorrect.
- **`efficiency.py` already computes `p99_ms`** (line 158).

⛔ **THE REAL GAP IS AN ARCH BRANCH, NOT AN EMITTER.** `efficiency.build_case()` dispatches on
`entry["arch"]` and handles exactly:
`flagship-worldmodel` · `flagship-worldmodel-v2` · `refa-plus` (line 321) · `refc` (352) ·
`refb` (412). **There is no v4 branch**, so no LEVER PANEL can be produced for a v4 arm — and the
emitter consumes a lever panel.

⇒ **Corrected verdict for #7: BUILD THE ARCH BRANCH in `efficiency.build_case`, not an emitter.**
That is a real, bounded piece of work (a `plan_step` + stage callables for the v4 head), and it is
the ONLY one of the two secondaries that needs code.

⚠️ **NOT ATTEMPTED HERE.** Writing a v4 branch requires deciding what the *deployed tick* means for
a v4 arm (which stages compose the tick, and which levers are admissible), and that is a
specification question, not a typing exercise. It should be its own scoped task.

## Root-cause class

**An absence-claim written from one observation and never re-probed** — the same class as C59
(searched by one name), C64 (a constraint never stated) and the `pseudosim` join blocker retracted
earlier today. The gate author saw `null` and inferred "no emitter"; the emitter was two files away.

⇒ **RULE (already in force, re-earned): before writing "X does not exist" into a decision document,
run the thing.** A `null` is a symptom, not a diagnosis.

## Evidence class

| claim | class |
|---|---|
| emitter exists / wired / validated at 0.8184 | **MEASURED** — read from source this session |
| `n_arm_rows = 0`, control mean 0.5794 | **MEASURED** — emitter executed this session |
| v4fs log has 0 × `g_op_fwd_ade_m` over steps 0–29,999 | **MEASURED** — `grep -c` |
| "re-emitting from the checkpoint would work" | **HYPOTHESIS** — the readout is available offline, but not attempted |
| #7 `deploy_tick_p99_ms` | ⛔ **NOT AUDITED** |
