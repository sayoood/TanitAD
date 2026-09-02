# SPEC — does the EMA teacher pin the tactical/strategic TARGET SCALE that the live recipe lets inflate?

`E-ARCH-TSC-2` · 2026-09-02 · Architecture & Inference · dev-box RTX 4060 (0 load on any training box) ·
**pre-registered by the Master Mind before any arm ran**

```yaml
hypothesis: H-TSC-2   # "an EMA target path cannot inflate (or collapse) faster than its decay allows"
one_variable: ema_targets          # arm E vs arm B' differ in THIS flag only; A' is the sensitivity control
held_constant: [seed 0, steps 250, bs 8, lru 16, the SAME 20-clip slice, labels/nav blob, lr, clip 1.0,
                target_space frozen, detach_aux_targets on, bptt_truncate 15, log_every 10]
success: "E: tgt_std_tac at step 250 within 2x of its step-1 value AND tactical loss share < 10 %;
          B' reproduces the inflation (tgt_std_tac > 10x by step 250); A' collapses (adapter_std falls > 15 %)"
failure: "E inflates like B' (tgt_std_tac > 5x by step 250) — the EMA does not pin the scale"
controls: [deliberate_regression (B' = the LIVE recipe, must inflate), sensitivity (A' = adapter space, must
           collapse), known_value (tgt_std_op ~ 1.0 in every frozen arm), instrument (per-channel tgt_std_*)]
splits: {fit: "20 EVAL clips as TRAINING data for a MECHANISM probe only — the models are DISCARDED",
         val: "none", test: "none — no number from this rig may enter MODEL_REGISTRY.md"}
```

## 1. What was measured on the live run, and why a 250-step probe answers it

MEASURED 2026-09-02 19:00Z on `/home/nvidia/experiments/refav1-b1-v72-1ep-21109/train_log.jsonl`
(register row C-REFAV1-TAC-INFLATION):

| | step 1 | step 500 | ratio |
|---|---|---|---|
| `tgt_std_tac` | 0.0571 | **5.951** | ×104 |
| `loss_feat_tac` | 0.0022 | **0.263** | ×119 |
| tactical share of the loss (w 0.5) | 0.27 % | **18.4 %** | |
| `tgt_std_str` | 0.0987 | 1.297 | ×13 |
| `tgt_std_op` (known value) | 0.9995 | 0.990 | pinned |
| `loss_feat_op` | 1.383 | 0.476 | healthy |

The collapse fix (`target_space=frozen`) anchored the OPERATIVE target only. The tactical and strategic
targets still pass through the trained adapter (`tq = tac_pool(tac_queries, adapter(...))`,
`st = strategic.read(adapter(...))`), so their scale is free — it shrank in the adapter-space run and it
inflates in this one. Inflation was ×20 by step 250 on the live run, so **250 steps is enough to see it**.

## 2. Arms (all: `refa_v1_train.py`, dev box, `--device cuda`, identical flags except the named one)

| arm | flags | role | MUST read |
|---|---|---|---|
| **B′** | `--target-space frozen --detach-aux-targets --bptt-truncate 15` | ⛔ **deliberate regression = the LIVE recipe** | `tgt_std_tac` **inflates > 10×** by 250; if it does not, the slice rig is insensitive and E proves nothing |
| **E** | B′ + `--ema-targets` (decay 0.996 → 0.999 over the 250 steps) | the proposed fix | `tgt_std_tac` **within 2× of step 1**, tactical share < 10 % |
| **A′** | `--target-space adapter --no-detach-aux-targets` | sensitivity control (the ORIGINAL defect) | `adapter_std` **falls > 15 %** — proves the rig sees BOTH directions |

Order of execution: **B′ first** (if it does not inflate, stop and report — E is uninterpretable), then E, then A′.
`--log-every 10` so 25 rows per arm. Batch 8 (Thor's live batch); if 8 does not fit in 8.6 GB, use the
largest that fits for ALL THREE arms and state it — the arms compare to each other, not to Thor.

## 3. Reads, COMMITTED IN ADVANCE

| # | read | CONFIRMED | REFUTED |
|---|---|---|---|
| R1 | B′ `tgt_std_tac(250)/tgt_std_tac(1)` | > 10 (rig reproduces the live inflation) | < 3 ⇒ rig insensitive, **VOID** |
| R2 | E `tgt_std_tac(250)/tgt_std_tac(1)` | ≤ 2 | > 5 ⇒ EMA does not pin the scale |
| R3 | E tactical loss share at 250 | < 10 % | ≥ 25 % |
| R4 | A′ `adapter_std` | falls > 15 % (collapse reproduced) | flat/rising ⇒ rig insensitive to collapse, **caveat on R1/R2** |
| R5 | `tgt_std_op` in B′ and E | ≈ 1.0 throughout | drifts ⇒ ⛔ instrument fault, **VOID** |
| R6 | E `loss_feat_op(250)` vs B′ | within 10 % of B′ (EMA does not hurt operative learning) | > 25 % worse ⇒ cost to weigh |

## 4. Also to be settled, because the live-run decision needs it

**R7 — resume tolerance.** Load the live recipe's checkpoint format (an OFF checkpoint, no `ema.*` keys) into
a model with `ema_targets=True`: does `--resume` accept it and initialise the EMA copies from the student, or
refuse on missing keys? MEASURED with the B′ checkpoint from this rig. This decides whether option (b) in
C-REFAV1-TAC-INFLATION (switch the live run at a checkpoint) is a 5-minute operation or a code change.

## 5. What this does NOT settle

* Nothing here is a capability claim. Tier **T0**, mechanism only, discarded models, 20 EVAL clips used as
  training data (admissible ONLY because no model from this rig is ever evaluated or registered).
* Whether inflation HURTS the final model — that is what the live run's operative term and the step-1000 /
  later T1 reads answer. This SPEC answers only whether the EMA path pins the scale.
