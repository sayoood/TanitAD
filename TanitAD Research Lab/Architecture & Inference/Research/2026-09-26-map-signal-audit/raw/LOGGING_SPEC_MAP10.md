# LOGGING SPEC — refcv7's 10 cm map head: per-class, per-band, every eval

**Registered:** 2026-09-26, ~22:55 Berlin, by the map-signal audit.

**Binding source:** `SPEC_REFCV7.md` §8, item 3, "G-DVB logging".

**Spelling.** The key names are the NEW-2 builder's, taken from ONE place: `map_head_hires.per_class_key`, read at 22:50. The audit adopts them, so there is one spelling, not two. This spec adds the requirements the builder's version does not yet meet (§4).

**Why this exists.** refcv6 logged map quality for ONE class. MEASURED on `D:/refcv6_eval_kit/ckpt_final/metrics.jsonl`:
- 4,621 rows;
- the map keys are only `map`, `map_iou_drivable`, `map_{gt,pred}_drivable_frac`, `map_pred_drivable_prob_mean` and `n_map_cells*`, plus their `eval_` twins.

The per-class path existed and was never called:
- `map_loss_row(..., with_metrics=True)` exists at `refcv6_perception_branch.py:566-572`;
- the trainer's call at `refc_v3_train.py:4477-4481` passes no `with_metrics`;
- the Training Watch plotted drivable IoU and the CE only (`build_watch_refcv6.py:648-656`).

The thin-class collapse (lane 0.009, crosswalk 0.001 at 35k) was invisible for 38,000 steps.

## 1. Names

**Classes** (`CLASS_KEYS`, code order 0–7): `nocls`, `drivable`, `lane`, `crosswalk`, `arrow`, `edge`, `hatched`, `sidewalk`.

**Bands** (`BAND_KEYS`): `0_20`, `20_40`, `40_60`. These are rig-x metres, i.e. fine rows 0–199, 200–399 and 400–599.

**Key form:** `map_hires_<stat>_<class>_<band>`. The eval rows add the `eval_` prefix.

## 2. Required keys

| key | where | count | meaning |
|---|---|---:|---|
| `map_hires` | train + eval | 1 | the 10 cm loss as optimised |
| `n_map_hires_cells` | train + eval | 1 | supervised cells (`fine_codes != 255` ∧ lift valid) |
| `n_map_hires_cells_seen` | train + eval | 1 | `fine_codes != 255` before the lift narrowing |
| `map_hires_n_<c>_<b>` | train + eval | 24 | labelled cells |
| `map_hires_lc_<c>_<b>` | train + eval | 24 | loss contribution; Σ over (c, b) = `map_hires` |
| `map_hires_gn_<c>_<b>` | train | 24 | ‖dL/dz‖ on logit channel c (all supervised cells of the band) |
| `map_hires_gno_<c>_<b>` | train | 24 | the same, restricted to cells labelled c (the pull UP; 0 ⇔ the class gets no signal) |
| `map_hires_inter_<c>_<b>` / `map_hires_union_<c>_<b>` | train + eval | 48 | argmax counts under the DECLARED decision rule |
| `map_hires_interraw_<c>_<b>` / `map_hires_unionraw_<c>_<b>` | train + eval | 48 | the same under the RAW argmax. Required whenever the loss carries class weights (§3). |
| `eval_map_hires_iou_<c>_<b>`, `eval_map_hires_lshare_<c>_<b>` | eval | 48 | POOLED IoU = mean(inter)/mean(union) = Σ/Σ, and the loss share. `null` when undefined, never 0. This is `derived_per_class`. |
| `ga_mh_lift`, `ga_mh_encoder`, `ga_mh_refine`, `ga_mh_trunk_s8_stage` (+ `_n`) | train | 8 | per-part `grad_abs_sum` |

**Counts, never per-batch IoUs.** The eval row is a MEAN over batches (`refc_v3_train.py:8401`; the builder's `_eval_row_from_acc`), so mean(inter)/mean(union) is the pooled IoU. A mean of per-batch IoUs is biased for rare classes.

## 3. The decision rule must be declared, and both rules logged when weighted

Under a class-weighted CE the softmax learns `q_c ∝ w_c · P(c | x)`, so the raw argmax is not the calibrated decision. With median-frequency weights:
- lane beats drivable at a 4.6–5.0 % posterior;
- arrow/text beats it at 0.2–0.7 %.

The analytic ceilings are in `raw/analytic_class_signal.json` and RESULT.md §2. `config.json["map_hires"]["decision_rule"]` must say `raw` or `prior_corrected` (`argmax_c (z_c − log w_c)`). The in-run counts and every eval use it.

## 4. Changes this requires in the builder's `code/fix/` trainer (routed to the Master Mind)

1. **The gradient-reach logging is dead, for refcv6 AND for NEW-2.**
   - `_pr_row` is filled on `step % max(1, args.log_every) == 0` BEFORE `step += 1`. Tip `refc_v3_train.py:8243-8247`; builder `code/fix/.../refc_v3_train.py:8604-8616`.
   - The row is written on `step % args.log_every == 0` AFTER it (tip `:8250-8251`).
   - With `--log-every 50` the two never coincide: MEASURED, **0 `ga_` keys in refcv6's 4,621 rows**. The new `ga_mh_*` keys inherit the defect.
   - Fix: trigger on `(step + 1) % args.log_every == 0 or step + 1 >= args.steps`, the rule `_grad_probe_row` uses at tip `:8220-8224`.
2. `interraw` / `unionraw` and the `decision_rule` stamp (§3).
3. **The 0.5 m auxiliary's keys stay `map`, `map_iou_drivable`, … in the builder's version.** They are stamped `map_lowres_role: "internal 0.5 m auxiliary (keys map*, NOT the map)"`. That meets A3 only if no report or Watch panel labels them "the map". The audit's recommendation is to rename them for refcv7 runs to `aux05_*`: no refcv6 reader reads a refcv7 file. This is a Master Mind decision.

## 5. The refcv7 Training Watch must show

1. **"Map (10 cm) — per class":** 8 × 3 panels of `eval_map_hires_iou_<c>_<b>` over steps, with `eval_map_hires_n_<c>_<b>` printed.
2. **"Map (10 cm) — signal":** the train EMA of `lshare` per class, with the class's label-mass share as a dashed reference, and `gno` per class on a log axis. `gno` = 0 is the "class gets no signal" alarm.
3. **A thin-class alarm tile.** It goes RED if any class's `0_20` eval IoU is ≤ 0.05 at any eval at or after step 5,000, or if an eval row lacks any required key.
4. Both decision rules, when weighted.

## 6. The gate check: G-DVB/map-logging

**Input.** The `metrics.jsonl` of the G-LIVE smoke on the REAL config, which must include ≥ 1 in-run eval, plus the Watch built from it.

⛔ **It reads ARTIFACTS, never the trainer source.** The refcv6 lesson is that an AST census read 0 on both the fixed and the broken trainer.

**PASS iff ALL of:**
1. `map_head_hires.missing_per_class_keys(first_eval_row, prefix="eval_") == []`, meaning all 24 `iou` and all 24 `lshare` keys are present. Also all 48 `eval_map_hires_{inter,union}_*` are present and finite.
2. On every train row, |Σ `map_hires_lc_*` − `map_hires`| ≤ 1e-5 · |`map_hires`|.
3. `eval_map_hires_n_<c>_0_20` > 0 for all 8 classes. If the smoke's eval windows lack a class, extend the eval set; the check is not waived.
4. The Watch output carries all 24 per-class IoU series and the alarm tile.
5. `ga_mh_lift`, `ga_mh_encoder`, `ga_mh_refine` and `ga_mh_trunk_s8_stage` appear in ≥ 1 train row, each > 0.
6. When weighted: the `interraw` / `unionraw` keys and `config.json["map_hires"]["decision_rule"]` are present.

**Regression arms (each MUST FAIL):**

| arm | the change | must fail on |
|---|---|---|
| RL1 drivable-only | only `map_hires_*_drivable_*` keys logged (the refcv6 pattern) | 1 (`missing_per_class_keys` returns 42 of 48) |
| RL2 IoU-not-counts | per-batch `map_hires_iou_*` logged instead of `inter`/`union` | 1 |
| RL3 no Watch panel | the Watch built without the per-class panel | 4 |
| RL4 reach off-by-one | the refcv6 `ga_*` trigger kept (as in `code/fix/` today) | 5 (0 `ga_mh_*` rows at `--log-every` > 1) |
