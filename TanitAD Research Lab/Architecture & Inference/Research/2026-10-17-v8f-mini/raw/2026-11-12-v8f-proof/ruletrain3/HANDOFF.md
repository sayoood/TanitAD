# HANDOFF: RULEOWNL1, trainer flag `--rule-own-form {sq,l1}` (note PREREG_V8F_PROOF_ADD34_N1), 2026-11-15

State: DONE on the CPU. Nothing trained on real data, no git, no pod, no box 2. Folder `F:/TanitAD-scratch/v8proof/ruletrain3/`. Base = tree of card ADD34 (`ruletrain2/work/tree_patch`, trainer `a5806960...` and `rule_own.py` `e4ad14ac...` verified; untouched). Work trees `work/tree_base` (copy), `work/tree_patch` (result). Definition and each decision with file:line: `RULEOWNL1_DEFS.md`.

## What changed
* `stack/scripts/v8f_mini_train.py` `2ad71c9f41328ae99a502b8d800d40b5011c6fb1549aefa9e1619116f4cb5a77`: flag, G-DVB entry and clauses (declared equals built; `l1` needs `--w-rule-own > 0`), probe reports the form, selfcheck string `levers.rule_own_form=l1`, the loop passes the built form.
* `stack/tanitad/data/rule_own.py` `e72d8854591e32c5aa67f54bc5ccfa979ead6854542e205b9e1d4fc5e3892797`: `knot_distance` (exact 0 value and gradient where the distance is 0, no epsilon), `own_term` / `add_own_term` / `RuleOwn` take the form.
* `stack/tests/test_v8f_rule_own.py` `3ff1293680e46fd20bf4bf551e4bb43636b68eb0a6b937753b87bce47857e8fd` (one pinned loop-call string updated); new `stack/tests/test_v8f_rule_own_l1.py` `68db2307cbe49cf5ce8ff5f241e96eef1ea4763b51b7e4a2a66e4c8fede5e11f` (13 tests).
* Default `sq`: no new log key, no model-config key, no key in `levers` or `info.rule_own`; the only added key on that path is `form: "sq"` inside the `rule_own_probe` record (the note asks the probe to report the form).

## Selfcheck strings
`levers.rule_own_form=l1` (with `levers.w_rule_own=1` / `=4`); absent for the default form. Also `info.rule_own.form` = `l1` (absent for sq) and `rule_own_probe.form`.

## Controls (all PASS; JSON and script in `checks/`)
* o1 `o1_identity.json`: default form, tree_patch vs tree_base, 100 steps, shapes G, GOW (`--w-rule-own 1.0`), GOWFT (fine-tune shape): every logged field and final weights max abs diff 0.0.
* o2 / o3 `o2_o3.json` (script `o2_o3_l1.py`): cap equals `distance_rule` by hand (49 of 64 rows masked); l1 term equals the float64 hand value (rel 2.3e-8); no-lead and NaN-speed rows give exactly 0 term and gradient; into the lead term above 0 with a finite non-zero gradient; inside the rule exactly 0 and 0; no lead exactly 0; partial mix 24 of 49 lead samples active; model gradient reached; a constructed plan with 4 knots on the cap has a finite gradient, exactly 0 on those knots (a plain sqrt gives NaN, printed).
* Mutation `o3_l1_mutation.json`: the line selecting the linear form replaced by the squared computation; control green, mutation red in all three reads (in-process, trainer totals equal the sq run, pytest). Unit test also holds the mutation case.
* o4 / o5 `o4_o5.json` (script `o4_o5_l1.py`): default selfcheck has no own key; GOW has no form key; GOL / GOL4 selfchecks carry the strings, probe form l1 (tiny world: term 15.4, 50 of 50 active at W 4), gate_extra accepts the right expectation and refuses `rule_own_form=sq` / wrong weight; refusals rc 1 for l1 without weight, with weight 0, with `--ego-perturb-p`, rc 2 for an unknown form; train-log key sets equal for sq and l1; checkpoint `cfg` equals its base's and the sq run's, strict load, pinned eval tool sha.
* Fine-tune smoke `t_ft_smoke.json`: GOL_s0/s1, GOL4_s0/s1 selfcheck with the card argv (2,000 steps) and 20 real steps from the tiny base: finite, cfg equals base.
* Tests `tests.log`: 145 passed, 0 failed, rc 0 (132 of the base list + 13 new).

## Build the pod tree
`cp -r /workspace/v8proof/tree_ruleown /workspace/v8proof/tree_ruleown2; cd tree_ruleown2; patch -p1 --binary < RULEOWNL1.patch` (replay verified here: the four files reproduce the sha256 above), or copy `patched/` over `stack/`. Verify the `PIN|` lines of `ARMS_RULEOWNL1.txt`, run the GOL_s0 selfcheck first and read `rule_own_probe` (term, n_active, both gradient norms).

## For the lead to decide
* The pod tree must be a copy of the card ADD34 tree (`tree_ruleown`), as the PIN lines assume (the `rule_relabel.py`, `dist_rule.py` and model pins are unchanged from ADD34).
* Scale of the real-data term is not known until the first pod selfcheck; l1 gives a constant-size pull per active knot.

## Temp folders and stray files (delete refused here; for you to delete)
`ruletrain3/work/tmp` 1.6 GB, `work/pt` 1.5 GB, `work/verify_patch` 356 KB. Trees `work/tree_base`, `work/tree_patch` 39 MB each (kept). Unused template copies in `checks/`: `o1_w1_identity.py`, `o2_o3.py`, `o3_w3_mutation.py`, `o4_o5.py` (copied from ruletrain2, never run; the scripts that ran are `o1_identity.py`, `o2_o3_l1.py`, `o3_l1_mutation.py`, `o4_o5_l1.py`, `t_ft_smoke.py`, `t_worker.py`). `apply_edits_l1.py` (the edit script) is in `work/`.
