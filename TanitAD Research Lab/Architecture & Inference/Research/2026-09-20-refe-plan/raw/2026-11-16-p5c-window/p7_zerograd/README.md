# P7 post-training zero-gradient gate: PASS (MM, 2026-11-16)

Evidence class: MEASURED. This is the INVALID criterion of `PREREG_P5C.md` §2.1: *"P7's post-training zero-gradient gate is RED on a planner parameter"*.

## Setup
- **Instrument:** `stack/scripts/zerograd_gate.py` (gate v1, sha256 `614ea07c…`), run on the P5-C pod on CPU with `/workspace/venv-teacher/bin/python`.
- **Checkpoint:** P7's `ckpt_last.pt`, sha256 `7f5ac13b2a0c07ab46f9ce349f3c07ecf3cd27e918ca05bc5d6980971b981415`. It was saved at 06:07:27Z, at the same step as `model_final.pt` (`2613fc17…`).
- **Names:** the format is `refe-partial-v1`, with no `model` key. So `param_names.json` = `list(model_partial)` (277) + `["aux_drv." + k for k in aux_drv]` (43) = 320.
  - That equals the optimizer's 320 ids (sorted) with **0 shape mismatches** against `exp_avg`.
  - The gate reports `naming: param_names unique=True n_alignments=1`.

## Result (`zg_named.out`, `zg_named.json`)
- 320 registered, 320 with state, **320 trained**.
- Case (a) no-state 0. Case (b) zero-state 0.
- opt-step-max 9,075, which is P7's final step.
- **VERDICT: PASS.**

The optimizer is AdamW (decoupled weight decay 0.01), so the gate's case (b) is valid; see its blind-spot notes.

## Deliberate regression (`zg_mutant.out`, `zg_mutant.json`)
- **The mutant copy has two defects:**
  - `traj_head.0.weight` (a planner parameter): `exp_avg` and `exp_avg_sq` zeroed;
  - `aux_drv.head.out.bias`: optimizer state entry deleted.
- **Result:** **FAIL, rc 1**, with no-state 1 and zero-state 1. Both cases are caught, so the PASS above is meaningful.
- The 257,651,244 B mutant checkpoint was deleted on the pod.

`zg_plain.out` is the first attempt without names: rc 2, `no_names`, the expected refusal for this format.

⇒ P7 is not INVALID on this criterion. With the read's NOT PROVEN vs P6, the base is **B-P7** (PREREG_P5C §2.1).
