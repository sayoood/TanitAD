# LOGGING SPEC: refcv7 detector heads (box3d and agent). The box quality refcv6 never logged

**Written:** 2026-09-27, by the box-head audit.

**Why.** MEASURED on refcv6's own `metrics.jsonl` (4,621 rows, md5 `1f195e3b1b897e4255bd1b8f3984514d`; `raw/analysis/traj.json`):
- Every box key is a LOSS TERM or a COUNT: `box3d_presence`, `box3d_cls`, `box3d_centre`, … , `box3d_n_target`, `box3d_n_matched`, …
- **No precision, recall, AP, confident-slot count or calibration was ever logged, for either head.** The quality keys that do exist (`cd_conflict`, `tacv6_goal_conf_bce`, …) belong to other heads.
- `box3d_n_matched == box3d_n_target` by construction: the Hungarian matcher assigns every target.
- **0 `ga_*` keys** (`grep -c`): the gradient reach was never logged (D-REFCV6-GRAD-REACH-DEAD).
- The agent head's `rates` and `occ` terms are in its total but are NOT logged (`refc_v3_train.py:4370-4373` lists presence / cls / centre / size / yaw / project / ground only).

**Consequence.** The presence BCE fell monotonically (train median 0.103 → 0.065; eval 0.0838 → 0.0619). Meanwhile, at the model's own gate, confident slots outnumbered targets 3.4× on representative eval windows (`raw/analysis/analyze_clipgrid.json`). A falling loss and an unusable detector coexisted for 38,000 steps, and nothing in the record could show it.

## Keys (per head h ∈ {box3d, agent})

"POS" means the VIS-1 POSITIVES. "gate" means the launch config's DECLARED presence gate.

### 1. Every train log row (cheap, per batch)

- `{h}_n_pos`, `{h}_n_ignore`, `{h}_n_dropped_hidden`: the VIS-1 census of the batch.
- `{h}_n_ignore_masked_slots`: slots whose presence weight the IGNORE rule zeroed.
- `{h}_n_conf`: slots at or above the gate, labelled rows.
- `{h}_conf_ratio` = `n_conf / max(n_pos, 1)`.
- `{h}_sum_phat` = Σ p̂ over slots, where p̂ is the probability the declared objective makes calibrated. **With NO_OBJECT_W w0 ≠ 1 that is σ(logit − ln(1/w0)), NOT σ(logit).**
- `{h}_tp@gate`: the greedy 2 m BEV TP count at the gate (`box3d_head.box3d_match_rows`, score = presence), so that precision and recall can be POOLED later.
- `ga_box_decoder`, `ga_box_memory`, `ga_agent_head`: the gradient abs-sum after backward, before `zero_grad` (`refcv6_perception_branch.grad_reach_report`).

### 2. Every in-run eval row (pooled over the eval batches; never a mean of per-batch ratios)

| key | definition |
|---|---|
| `eval_{h}_prec@gate` | Σ tp / Σ n_conf |
| `eval_{h}_rec@gate` | Σ tp / Σ n_pos |
| `eval_{h}_conf_ratio` | Σ n_conf / Σ n_pos |
| `eval_{h}_ap2m` | AP@2 m BEV over all eval windows (`ap_from_rows`) |
| `eval_{h}_auroc_matched` | presence AUROC, Hungarian-matched vs unmatched, tie-aware |
| `eval_{h}_auroc_objectness` | the same, with the label "a POS within 2 m of the slot's centre" |
| `eval_{h}_ece_matched` | 10-bin ECE of p̂ vs matched |
| `eval_{h}_sumphat_over_pos` | Σ p̂ / Σ n_pos |
| `eval_{h}_rec@gate_<cls>`, `eval_{h}_npos_<cls>` | per class, all 10 classes, count beside each |
| `eval_{h}_cls_acc_tp` | argmax class accuracy on greedy TPs |
| `eval_{h}_centre_err_p50` | Hungarian-pair median centre error, in metres |

- Also log the `{h}_n_*` census keys of section 1 on eval rows.

### 3. Alert (printed, and stamped into the row)

- `eval_{h}_conf_ratio` outside [0.5, 1.5] (the G-LIVE-GATE band, RESULT §7.2) for 3 consecutive eval rows after step 2,000.
- OR `eval_{h}_sumphat_over_pos` outside [0.7, 1.4].

refcv6 would have fired: its conf_ratio at step 38,000 was 3.40 (box3d) and 3.75 (agent) on representative windows.

## Tests (each must go RED under a mutation)

1. A pinned test runs one synthetic batch and asserts that every key above is present with a finite value. **Deleting any one logging call must turn it red.**
2. A known-value test: GT written into the slot format reads prec = rec = AP = 1.0, conf_ratio = 1.0, and auroc = 1.0 exactly. A constant presence reads auroc = 0.5 exactly.
3. The pooled-vs-mean test: two eval batches with (tp, n_conf) = (1, 1) and (0, 9) must log prec = 0.10, not 0.5.
