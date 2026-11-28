# Why OBD-NULL_s0 fails NULL parity: float drift from the first steps, not a changed training path (2026-11-11 ~09:20)

**Question.** The card (`PREREG_V8F_OBD.md:15`) expected OBD-NULL_s0 (DAG-OACC s0's argv, only `--out` changed) to reproduce DAG-OACC s0's checkpoint bit for bit, because box-2 training at a fixed seed had read as deterministic. The chain's tensor check (`C:/Users/Admin/obd_state/NULL_PARITY.json`, 07:50) found **252 of 276 tensors differ**. That triggers the card's named fallback (`PREREG_V8F_OBD.md:54`). This note says why.

**MEASURED** (`null_drift.py` → `NULL_DRIFT.log`; inputs: the banked `train_log.jsonl` of `C:/Users/Admin/t2v8f_pulls/obd_OBD-NULL_s0/` and `E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/`, sha256 in the log):

* **Same data.** Both summaries carry the same `train_sha256` (`37f4f9ee…`), `n_windows` 306,566, `S` 4,791, `n_clips` 4,512, `exclude` 540:600, seed 0, 3,000 steps.
* **Same path, different floats.** At step 50 the logged values agree to 6 or 7 significant digits but are not equal: `grad_norm` 13.334982 vs 13.335000, `ckpt` 0.53473705 vs 0.53473717. **None of the 60 logged steps has all 8 logged losses bit-equal.**
* **The drift grows.** Batch ADE, relative difference over the 60 logged steps: median 2.06 %, p90 4.04 %, max 9.28 %. At step 3,000, batch ADE is 3.6626 (NULL) against 3.6568 (DAG-OACC).

**Reading.** A changed training path, such as a different batch order, a consumed RNG draw or a different loss term, would show at step 50 as a different value, not as a 7th-digit difference. What the logs show instead is the same computation with non-associative GPU float sums (kernel or algorithm choice, atomics) that amplify over 3,000 steps. The +2 lines between the OTBAND tree and `cdbff971` (the launch probe's acceleration key) are not needed to explain it.
⇒ **The note `same-seed-replicate-is-bit-identical` (wm-only ladder training, 2026-11-09) does not carry over to the full `v8f_mini_train.py` trainer.**

**Consequences for the OBD read, under the card's existing rules (no rule changes):**

1. The fallback applies exactly as written: OBD-NULL_s1 is trained, every "against DAG-OACC" row is read against OBD-NULL of the same seed, and NULL minus DAG-OACC is the code-drift row.
2. On this evidence that code-drift row measures a **same-seed, same-code rerun**. It is an extra estimate of the "another training run" variance, beside floor G (seed). It is reported as a beside row, never as a gain or a cost.
3. Any future card whose control depends on bit parity of the full trainer must read parity on metrics, against a floor, or else make the trainer deterministic first (`torch.use_deterministic_algorithms`, fixed cuBLAS workspace). That is a work item, not a change to this card.

Evidence class: MEASURED (the two banked logs). That GPU nondeterminism is the mechanism is an inference from the 7th-digit drift at step 50. No deterministic-mode rerun was done (ESTIMATED cost: one 38-minute box-2 run).
