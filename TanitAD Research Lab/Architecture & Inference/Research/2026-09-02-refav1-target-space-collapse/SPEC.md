# SPEC — does `--target-space frozen` remove refav1's collapse minimum?

`E-ARCH-TSC-1` · 2026-09-02 · Architecture & Inference · Thor · **PI directive: "stop it, fix it,
and do experiments to validate this before we spend many days"**

```yaml
hypothesis: H-TSC-1
one_variable: FALSE — B carries TWO changes; see the amendment in section 4a
held_constant: [seed, steps, bs, lru, cache, episodes, labels, nav, lr, clip, corpus]
success: "frozen holds adapter_std within 5 % of its start over 250 steps AND tgt_std_op stays ~1.0"
failure: "frozen's adapter_std falls monotonically like adapter's (>15 % over 250 steps)"
controls: [deliberate_regression, target_scale_instrument, known_value]
```

## 1. What we are testing and why

**MEASURED 2026-09-02, first refav1 launch (450 steps, `--target-space adapter`, the default):**

| | step 1 | step 450 | |
|---|---|---|---|
| `adapter_std` | 0.4763 | **0.3385** | 9 of 9 deltas negative, never once up |
| `loss_feat_str` | 0.0052 | **0.0003** | |
| `loss_feat_tac` | 0.0022 | **0.0005** | |
| `loss` | 0.600 | **0.165** | *looked like the best run of the day* |

`refa_v1.py:263-270` documents the mechanism: in `"adapter"` space the target is
`adapter(std(future))` and **the adapter is TRAINED**, so *"mapping everything to a constant
zeroes the loss … nothing REMOVES the minimum."* ⇒ a falling loss is **indistinguishable from
progress** unless the target's own scale is visible.

⛔ **This is not a bug report about a crash.** The run was stable, GPU-saturated, and improving
by every number in the log. That is what makes it dangerous: five days of Thor would have
produced a checkpoint with an excellent loss curve and no learning.

## 2. ⭐ THE INSTRUMENT ADDED FIRST — the mechanism must be MEASURABLE, not inferred

`refa_v1.py` now emits `tgt_std_op` · `tgt_std_tac` · `tgt_std_str` beside every loss.
**A loss is only interpretable against the variance of the thing it predicts.** Without this the
collapse and genuine learning produce the same curve, and my own first two readings of this run
("diverging", then "recovered, best of the run") were both wrong for exactly that reason.

⭐ **Known-value control, and it is what makes the panel admissible:** in `"frozen"` space the
target is `std(future)` with **frozen buffers**, so `tgt_std_op` **must read ~1.0 and cannot
fall**. If it does fall, the instrument or the wiring is broken and no verdict may be drawn.
Smoke-tested on CPU before launch: adapter `tgt_std_op` 1.0000, frozen 1.022 ✅

## 3. ⚠️ A FINDING FROM THE INSTRUMENT, BEFORE ANY ARM RAN — THE FIX IS PARTIAL

`--target-space` governs **only the operative term** (`refa_v1.py:262`: *"target space for the
PRIMARY (operative) term"*). The tactical and strategic targets are `_tac_field(tgt)` and
`strategic.subspace(tgt)` — **both derived from the trained adapter in BOTH modes**. Confirmed by
the instrument: `tgt_std_tac` and `tgt_std_str` read the same in adapter and frozen.

⇒ **`frozen` removes the collapse minimum from the operative term only.** The hypothesis under
test is therefore *indirect*: the operative term carries `w_feat_op = 1.0` against 0.5 / 0.25, so
anchoring it to a frozen target should hold the adapter up — **and the other two targets with
it**. That is plausible, not established, and it is precisely what the arms measure.

## 4. The arms — one variable, both instrumented

| arm | `--target-space` | role |
|---|---|---|
| **A** | `adapter` | ⛔ **DELIBERATE REGRESSION** — re-runs the defect. If A does not collapse, the diagnosis is wrong and B proves nothing |
| **B** | `frozen` | the proposed fix |

250 steps each, `--seed 0 --bs 8 --lru 64`, identical data. ~89 min per arm at the MEASURED
21.4 s/step ⇒ **~3 h total against the 5 days it protects.**

## 4a. ⚠️ AMENDMENT — B IS NOT ONE-VARIABLE, AND SAYING SO WOULD HAVE BEEN THE CONFLATION ERROR

Written into this SPEC as `one_variable: target_space`. **That is no longer true.** After the
audit the defaults changed, so Arm B carries **two** differences from A:

| | `target_space` | `detach_aux_targets` |
|---|---|---|
| **A** | `adapter` | **off** |
| **B** | `frozen` | **on** |

⇒ **If B holds, it establishes that THE SHIPPED FIX WORKS — not which half did it.** That is the
decision-relevant question (may refav1 relaunch for a full epoch?) and it is answered in 90
minutes. Attribution between the two halves is a *separate* question needing a third arm:

* **C** = `frozen` + `--no-detach-aux-targets` ⇒ isolates `target_space`.

⛔ **Recorded rather than quietly tolerated.** Declaring one variable while running two is exactly
the error that made MM-E19's k=60 arm un-attributable for a week, and the correction cost there
was a whole extra 8.7 h control run. Naming it now costs nothing; discovering it in the write-up
would cost the arm.

⚠️ **The verdict wording must match:** a passing B licenses *"the shipped anti-collapse
configuration prevents the collapse"*, **never** *"frozen targets prevent the collapse"*.

⚠️ Arm A is re-run rather than reusing the banked 450 steps, because those predate the instrument
and carry no `tgt_std_*`. Reusing them would leave the mechanism inferred — the thing this SPEC
exists to stop.

## 5. Reads, COMMITTED IN ADVANCE

| # | read | fix CONFIRMED | fix REFUTED |
|---|---|---|---|
| R1 | `adapter_std` over 250 steps | B holds within **5 %** of start | B falls **>15 %**, like A |
| R2 | `tgt_std_op` | B pinned **~1.0** (frozen buffers) | B drifts ⇒ ⛔ instrument fault, **VOID** |
| R3 | `tgt_std_tac` / `tgt_std_str` | hold in B | fall in B ⇒ **fix is PARTIAL** — the indirect anchor is not enough and those terms need their own frozen target |
| R4 | A reproduces the collapse | A's `adapter_std` falls, targets shrink | ⛔ A does not collapse ⇒ **the diagnosis is wrong**, no verdict on B |

⚠️ **R4 is the one that can invalidate everything else.** A gate that does not fail the broken
arm says nothing about a pass on the fixed one.

## 6. What this does NOT settle

* **The 30-step untruncated BPTT chain** (`refa_v1.py:525/529`, no `detach`) is a *separate*
  known-bad config — the Lab's ASK-1 found no published recipe back-propagates that far, and the
  gnorm excursions (3.6e3, 5.7e7, `inf`) are its signature. Neither arm addresses it.
* **Nothing here is a capability claim.** Tier **T0**; 250 steps is a mechanism probe, not a
  result, and no driving metric is produced.
