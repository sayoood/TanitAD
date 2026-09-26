# PARTIAL — refcv6 box/agent head PRESENCE signal (handed over to the box-head audit, 2026-09-26 ~23:20)

**Scope.** This is partial work only. The task moved to the dedicated box-head audit
(`…/2026-09-26-box-head-audit/`) before the per-slot measurement.
- Items 1–2 below are complete.
- Item 3 is a single operating point computed from the video agent's counts.
- The AUROC and the precision = recall gate were NOT measured; see "what is missing".
- The numbers are in `raw/box_presence_partial.json`.

## 1. The presence loss, traced

Code is at tip `2c510fb`; it is unchanged since refcv6's launch `284393c`
(`git diff --stat` is empty for `agent_slots.py`, `box3d_head.py` and `refc_agents.py`).

| hop | file:line | what |
|---|---|---|
| channel | `agent_slots.py:168` | `presence` = slot field 0, one logit: DETR's ∅ / no-object |
| init prior | `agent_slots.py:563-573, 616-617` | head bias = `logit(0.05)`, so every slot starts at p = 0.05 |
| targets | `box3d_head.py:376-398` → `refc_agents.visible_target_filter` | visible-field filter BEFORE matching (default ON since 2026-09-23) |
| matching | `agent_slots.py:787-823` (`match_slots`), cost `:764-785` | Hungarian per frame, under `no_grad`. Cost = 1.0·L1 centre (m) + 0.5·L1 size (m) + 1.0·(−p[class]) + **1.0·(−sigmoid(presence))** (`MATCH_COST_W`, `:216-221`). A confident slot is cheaper to match. If targets > 100, the farthest are dropped (counted). |
| presence target | `agent_slots.py:864-870` | matched slot → 1 (weight 1.0); **every unmatched slot → 0 (weight `NO_OBJECT_W` = 0.1, `:233`)** |
| presence loss | `agent_slots.py:871-873` | `binary_cross_entropy_with_logits(logit, tgt, weight=w)`, `reduction="mean"`, i.e. divided by B·100 (NOT by Σw) |
| term weight | `SLOT_LOSS_W["presence"]` = 1.0 (`:226-228`); `--w-box3d 1.0` / `--w-agent 1.0` (argv) | in `total` |
| trainer | box3d `refc_v3_train.py:4591-4594`; agent `:4166-4170` (`refc_agents.agent_losses`, `filter_visible=True`) | both heads use the same `slot_set_loss` |
| consumer of the gate | `refc_agents.py:199` `presence_gate = 0.5`; `:317-334` | the planner's agent tokens are SOFT-scaled by p (hard mask off). The video agent's "confident" = p ≥ 0.5. |

## 2. The analytic optimum under the 0.1 weighting

Take a slot that the model believes is matched with probability π (conditional on its features). The expected loss is
π·(−log p) + 0.1·(1−π)·(−log(1−p)). It is minimised at

**p\* = π / (π + 0.1·(1 − π)), i.e. logit(p\*) = logit(π) + ln 10 = logit(π) + 2.303.**

- **The gate 0.5 is crossed at π = 1/11 = 0.091.** A slot the model thinks has a 9 % chance of being matched is "confident".
- A calibrated 50 % belief needs **p ≥ 0.909**, i.e. a logit ≥ +2.303.
- **Exchangeable slots** (the head cannot tell which slot will be matched), K targets per 100:

  | K | p\* |
  |---:|---:|
  | 4 | 0.294 |
  | 5 | 0.345 |
  | 10 | **0.526** |
  | 21 | 0.727 |
  | 24.5 | 0.764 |
  | 50 | 0.909 |
  | 80 | 0.976 |

  So with ≥ 10 targets, ALL 100 exchangeable slots are "confident". The tilt only needs π ≥ 0.091 per slot.
- For ~4–5 targets per 100 (the training mean): if all slots were exchangeable, none would be confident (0.29–0.34). The inflation therefore comes from partial specialisation plus the ×10 odds tilt. For example, 10 candidate slots competing for 4 targets gives π = 0.4 each, p\* = 0.87, so 10 confident for 4 targets.
- **Loss check.** Exchangeable at K = 5: 0.0934 per slot. The prior init (p = 0.05 everywhere) gives 0.1547.
  - MEASURED (`metrics.jsonl`, `box3d_presence` median): 0.103 at ≤ 1k steps, **0.066** at 34.5–38.25k, eval 0.0625 at 38k.
  - So the head is between exchangeable and perfect: partially specialised.
- **Training K is small; the video's clips are crowded.** MEASURED (`metrics.jsonl`): visible targets per labelled window, median **4.75–5.5** over the run; eval 4.21 at 38k; 0 dropped. The video's three clips carry **80.3 / 24.5 / 21.0** targets per window. The two "all-confident" clips are therefore 5–16× denser than the training average. On 0191487845ef, 98.7 confident against 80.3 targets is NOT a large over-count; on 9f8bedcfb9de, 98.3 against 24.5 is a 4× over-count.

## 3. MEASURED, one operating point (p ≥ 0.5)

This is arithmetic on `C:/Users/Admin/qland/work/mapvid/boxes_summary.json`, the video agent's counts.

| clip | head | targets/win | confident/win | matched slots confident (TPR) | UNmatched slots confident (FPR) |
|---|---|---:|---:|---:|---:|
| 0191487845ef | box3d | 80.3 | 98.7 | 0.997 | **0.948** |
| 9f8bedcfb9de | box3d | 24.5 | 98.3 | 0.983 | **0.982** |
| 34765c024267 | box3d | 21.0 | 71.1 | 0.903 | 0.660 |
| 0191487845ef | agent | 80.3 | 99.2 | 0.999 | 0.963 |
| 9f8bedcfb9de | agent | 24.5 | 99.6 | 0.997 | 0.996 |
| 34765c024267 | agent | 21.0 | 80.9 | 0.928 | 0.778 |

**At the gate the presence probability barely separates matched from unmatched slots on the two crowded
clips: TPR ≈ FPR.** Only 34765c024267 separates, at 0.90 vs 0.66.

⚠️ **Not measured here.** The AUROC, the per-slot presence distribution and the precision = recall gate all
need per-slot presence probabilities with their matched/TP labels. The video agent's dump
`C:/Users/Admin/qland/work/mapvid/mapdump_step38000/` holds the **MAP only** (`probs`/`frac`/`seen`/`valid`
`[N,9,120,64]`, `meta.jsonl`), with **no box-slot tensors**. Its `per_frame.jsonl` holds per-window counts only.
Getting them needs a GPU forward over the 513 box windows that saves `box_slots["presence_logit"]`, the visible
targets and the Hungarian and greedy-2 m assignments. That is ~4 min on the RTX 4060 under the GPU gate; the
video agent's renderer already builds all of it.

## 4. Pointers for refcv7 (NOT a recommendation; the box-head audit owns it)

- **Threshold.** Under `NO_OBJECT_W` = 0.1 the calibrated gate is p ≥ 10/11 (logit ≥ +2.30), not 0.5.
  This is the same "tilted posterior" mechanism as the map head's median-frequency weights
  (map audit RESULT §2c, SPEC_REFCV7 A4 item 1).
- **Loss options.**
  - (a) Keep 0.1 and correct the decision: `logit − ln 10 ≥ 0`.
  - (b) Raise `NO_OBJECT_W` toward 1 once the head is past the cold start. This can be scheduled.
  - (c) A focal presence loss, which damps easy negatives without tilting the optimum.
- **Candidate check.** On a real-config smoke, confident slots per window (at the declared gate) vs visible
  targets per window. For example, pass iff the ratio is ≤ 1.5 on the train-density windows. The regression
  arm is the undeclared 0.5 gate with w0 = 0.1, which should inflate the ratio on crowded windows.
