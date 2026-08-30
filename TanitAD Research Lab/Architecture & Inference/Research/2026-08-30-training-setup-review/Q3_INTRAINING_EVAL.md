# PI Q3 — "is there an eval step INSIDE training?" → ⛔ **NO.** Design + implementation

**Date:** 2026-08-30 · **Owner:** TanitAD_TrainingFlyWheel · **Evidence class:**
MEASURED (file:line from source) · **Tier:** T0 trainer-side.

---

## 0. The answer, at file:line

⛔ **There is NO val-side eval anywhere inside a training run.**

| probe | finding |
|---|---|
| the save-interval block | `train_v6_staged.py:6131` — contains **only** the default-off X2 seam dump (`--dump-seam-plan`) and `_save_ckpt`. No forward pass on held-out data. |
| every `.eval()` in the trainer | `:957` (VLM) and `:1063` (EMA target deepcopy). **Neither is a validation pass.** |
| the val corpus | `--v2-val-cache` is **refused at startup** (P4-5) because the `ds_val` it built was constructed, printed, and never read. There is not even a val dataset in scope. |

⇒ Every real read — drift, meanpred, absorption, T1 — is a **post-hoc script on a
pulled checkpoint**. We cannot see a checkpoint degrade during a run, and
**checkpoint selection is blind** (§5 of the review: no criterion beyond "final").

## 1. ⭐ THE DESIGN DECISION: it is NOT an ADE watcher

A distance metric is **structurally blind to the programme's largest known
failure**. MEASURED: v1.x reproduced an S-curve **97.9 %** open-loop and **0.0 %**
with the action held — it echoed the action channel instead of driving. The
current v7 line shows the same signature: closed-loop vs hold-action gap ~1 %
(H-ARCH-ACTINS). **An arm that ignores its action input can post an excellent ADE.**

⇒ The primary quantity is **`action_sensitivity`** — how far the prediction moves
when the action input is changed — and ADE is reported *beside* it, never alone.

| quadrant | reading |
|---|---|
| high sensitivity + low ADE | ✅ healthy |
| **low sensitivity + low ADE** | ⛔ **the echo trap — and ADE alone calls it success** |
| high sensitivity + high ADE | unstable |

⚠️ Sensitivity is **not** a quality metric: a large value is not automatically
good, since an unstable model also moves a lot. It is read *with* ADE.

## 2. What it emits per save-interval

`val_ade_m` · `action_sensitivity` (+ `action_delta_m`, `scale_m`) ·
`hold_action_ade_m` · `beats_hold_action` · `eval_mode` · `n_batches` · `_tier`.

⛔ **The hold-action control is not a courtesy baseline.** MEASURED (H-CTRL-1, T1,
6,844 windows): the do-nothing control beat a released arm on **every** metric
family, including a better chance-corrected tactical kappa (0.6449 vs 0.3795). An
eval without it can watch a model improve against itself while losing to nothing.

## 3. Guards that came from this programme's own retractions

* ⛔ **`.eval()` is set explicitly and the mode is recorded**, and the previous
  mode is **restored on exit** (TRAIN-C5: a readout that forgot `.eval()`
  overstated ADE by **+175.7 %**, and its run-to-run wobble was dropout, not
  noise). A monitor must not leave the model in eval for the next training step.
* ⛔ **Absence is reported, never averaged.** An empty val set returns
  `status: ABSENT`, not `0.0`. Same for a missing `alt_action_fn`: the primary
  quantity reports `ABSENT` with a reason rather than vanishing.
* ⛔ **Every value is stamped `T0 trainer-side`** and names `t1_eval.py` as the
  route to a capability claim — because *"v1.6 is best-in-program"* came from a
  trainer log and was ~10 % optimistic against `eval_*.py`.

## 4. Cost

**~2 forward passes × `max_batches`, no backward, bounded by construction.** At
`max_batches=8` and batch 8 that is 128 windows ≈ **10 s at the measured 12–14
windows/s** — against a `--save-every 1000` interval of ~16 min on Thor, i.e.
**~1 %**. Pinned by `test_max_batches_bounds_the_cost`.

## 5. ⚠️ A bug I shipped into this module and caught

The hold-action block gated on `if base_sum:` — the **truthiness of the sum** — so
a control that scored a **perfect 0.0 m** was reported as *"no baseline computed"*.
**Absence and a real zero collapsed into one branch**, which is the exact
confusion this module's own empty-val-set guard exists to prevent, reproduced one
function further down. Fixed to a count (`n_base`), pinned by
`test_a_PERFECT_baseline_is_reported_not_treated_as_absent`.

⚠️ The test fixture also needed correcting: a straight constant-velocity ground
truth makes the hold-action control **perfect**, so `beats_hold_action` was
vacuous. The fixture now curves, and both cases are tested.

## 6. Status and what is NOT done

✅ `stack/tanitad/train/intrain_eval.py` + `stack/tests/test_intrain_eval.py` —
**13 tests, all passing.** Model-agnostic (takes `forward_fn`), so it is testable
without a GPU and reusable by all four B1 consumers.

⛔ **NOT WIRED INTO THE TRAINER YET, deliberately.** Wiring touches
`train_v6_staged.py` at the save-interval and must land **default-OFF** behind a
flag so no live run changes behaviour. That is the next step and it needs the
trainer's real `forward_fn` adapter (the batch carries `future_actions2`, which is
the channel `alt_action_fn` perturbs).

⛔ **NOT VALIDATED ON A REAL MODEL.** Every number above is design + unit tests. A
mock validates plumbing; only the real model validates wiring (TRAIN-C1).

