# RULEOWNL1_DEFS: what `--rule-own-form {sq,l1}` does, and every decision with its file:line (note PREREG_V8F_PROOF_ADD34_N1 sections 2 and 4)

Tree: `F:/TanitAD-scratch/v8proof/ruletrain3/work/tree_patch` (tree of card ADD34 `ruletrain2/work/tree_patch` plus this change; base copy `work/tree_base`, trainer sha256 `a5806960...`, `rule_own.py` `e4ad14ac...` verified before the work). Line numbers are of that tree. Tier of any later read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react). Evidence class of this file: MEASURED by reading source, plus the checks in `checks/`.

## Files

* changed: `stack/scripts/v8f_mini_train.py` (`a5806960...` -> `2ad71c9f41328ae99a502b8d800d40b5011c6fb1549aefa9e1619116f4cb5a77`), `stack/tanitad/data/rule_own.py` (`e4ad14ac...` -> `e72d8854591e32c5aa67f54bc5ccfa979ead6854542e205b9e1d4fc5e3892797`), `stack/tests/test_v8f_rule_own.py` (`a3364480...` -> `3ff1293680e46fd20bf4bf551e4bb43636b68eb0a6b937753b87bce47857e8fd`; one pinned source string of the loop call now carries the form argument).
* new: `stack/tests/test_v8f_rule_own_l1.py` (`68db2307cbe49cf5ce8ff5f241e96eef1ea4763b51b7e4a2a66e4c8fede5e11f`, 13 tests).
* unchanged byte for byte: model files, `v8f_t1_eval.py`, data module, `rule_relabel.py`, `dist_rule.py` (see `PINS.txt`). No model-config key: a checkpoint's `cfg` equals its base's (o5).

## The flag

`--rule-own-form {sq,l1}`, default `sq` (`v8f_mini_train.py:291`). Everything else of `--w-rule-own` is unchanged: mask and inputs (`rule_own.own_inputs`), the detached plan, the float64 rule call (`cap_of_plan`), the cast of the cap back to the prediction dtype, where the term enters the loss.

* `sq`: `per = ((pl - cap) ** 2).sum(-1).mean(1)` (`rule_own.py:107`), the line of card ADD34 as it was. No extra RNG draw, no new log key, no new key in the checkpoint or `cfg`, no new key in `info.rule_own` or `levers` (see the gate below). `own_term(..., "sq")` equals the explicit old expression with `torch.equal` (unit test `test_sq_default_is_the_earlier_term_bit_for_bit`), and o1 reads max abs difference 0.0 over 100 steps.
* `l1`: `per = knot_distance(pl - cap).mean(1)` (`rule_own.py:106`): per sample the mean over the 30 knots of the Euclidean distance in METRES between the predicted knot and its cap. Batch term = `per.mean()` over the masked samples, exactly 0.0 (a plain zero tensor, no graph) when none are masked (`rule_own.py` early return, unchanged), times `--w-rule-own` in `add_own_term` (`rule_own.py:146-150`). `n_active` = samples whose per-sample term is above 0.
* **Exact zero at the cap.** `knot_distance` (`rule_own.py:83`): `sq = (diff**2).sum(-1)`; `pos = sq > 0`; the sqrt is taken on `where(pos, sq, 1.0)` and the result is `where(pos, sqrt(...), 0.0)`. A knot with a zero squared norm gets value 0 and gradient exactly 0 (the selected-away branch carries a zero cotangent into a finite slope), every other knot keeps the exact norm: no epsilon. A plain `sqrt` gives NaN gradients there (o3 part 5 prints this for the record: `plain_sqrt_gradient_all_finite_for_the_record` false). A plan that keeps the rule equals its cap bit for bit (the rule returns the original knots for unchanged steps), so term and gradient are exactly 0 for it, as before.
* A knot with a very small but non-zero distance keeps its true slope (unit vector over `n x 30`): the gradient of the linear form has magnitude `W / (n_masked x 30)` per knot whatever the violation, versus `2 W d / (n_masked x 30)` for `sq`.

## Decisions with file:line

| decision | where |
|---|---|
| form validation; `l1` requires `w_own > 0` (ValueError in the constructor, rc 1) | `rule_own.py:35-40` |
| the set of forms | `rule_own.py:27` |
| `own_term(traj30, inp, form="sq")`; unknown form raises | `rule_own.py:91`, `:104-110` |
| `add_own_term(..., form="sq")` passes the form to the term | `rule_own.py:146-150` |
| flag, choices, default | `v8f_mini_train.py:291` |
| G-DVB entry | `v8f_mini_train.py:127` (`FLAG_CHECKS["rule_own_form"] = "data:rule_own_form"`) |
| G-DVB clauses: form must be sq or l1; `l1` without `--w-rule-own > 0` refused; declared form must equal the BUILT form (`rec.get("form", "sq")`) | `v8f_mini_train.py:1187-1190`, `:1199-1200` |
| the built record names its form only for `l1` (`info.rule_own.form`), so the `sq` record keeps its key set | `v8f_mini_train.py:2999-3000` |
| `RuleOwn` built with the declared form | `v8f_mini_train.py:2993` |
| G-LIVE `rule_own_probe` computes the term, the second call, the add helper and the all-False control with the BUILT form and reports `form` | `v8f_mini_train.py:2939-2948` |
| selfcheck string `levers.rule_own_form=l1` only when the built form is `l1`; absent for `sq` | `v8f_mini_train.py:3172-3173` |
| loop call passes the built form (the add line itself is unchanged, guard `ZZRULE-OWN-NOT-APPLIEDZZ` unchanged) | `v8f_mini_train.py:3362` |

Train log keys (`rule_own_term`, `rule_own_n_lead`, `rule_own_n_active`), summary keys and the checkpoint total key are unchanged for both forms (o4: the key sets of the `sq` and `l1` runs are equal). `rule_own_probe` of a default-form arm now carries `"form": "sq"` (the one added key of the default path, inside the selfcheck probe record only, as the note asks the probe to report the form); `levers` and `info.rule_own` of the default form have no form key.

## Reading the scale

On the tiny synthetic world the `sq` term reads about 6.2 and the `l1` term about 1.16 on the same planted plan (o2 b); in the real-data selfcheck read `rule_own_probe.term`, `grad_norm_term_on_traj30` and `grad_norm_traj_loss_on_traj30` for `l1` before the arms start. The tiny-world probe of GOL4 reads gradient norm 4.29 for the term against 0.51 for the trajectory loss (untrained plans; not a prediction for the real arms).

## Limits

* Real data was not read: nothing ran on a real cache, the pod or box 2.
* The linear form has a constant-magnitude pull per active knot; the card note fixes the form, so it was built as written.
