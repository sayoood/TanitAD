# HANDOFF: RULEOWN, trainer flags `--w-rule-own` and `--rule-relabel-weight` (cards ADD34 and ADD32_N1 stage 2), 2026-11-15

State: DONE on the CPU. Nothing trained on real data, no git, no pod, no box 2. Folder `F:/TanitAD-scratch/v8proof/ruletrain2/`. Base = tree of card ADD32 (`C:/Users/Admin/v8proof_c/ruletrain/tree_patch`, sha256 of trainer `51b0f057...` and `rule_relabel.py` `8fd91e20...` verified on the copy; that tree is untouched). Work trees: `work/tree_base` (copy), `work/tree_patch` (result). Definitions and every decision with file:line: `RULEOWN_DEFS.md`. Pins `PINS.txt`, arms `ARMS_RULEOWN.txt` (tree `/workspace/v8proof/tree_ruleown`), diff `RULEOWN.patch`, files `patched/`.

## What changed
* `stack/scripts/v8f_mini_train.py` (changed, sha256 `a5806960d09625895970a7591dc2901151f36968e011619ed01d0423fbbd370b`): two flags, G-DVB entries and clauses (`bad_rule_own`), G-LIVE probe `rule_own_probe`, the loop lines, guards `ZZRULE-OWN-NOT-APPLIEDZZ` and `ZZRULE-WEIGHT-NOT-APPLIEDZZ`, log / summary / checkpoint-total keys only when on.
* new `stack/tanitad/data/rule_own.py` (`e4ad14ac2c43bc2308e325f9888821c73e56cafbda102ca8ed4e0cf827130480`) and `stack/tests/test_v8f_rule_own.py` (`a3364480ae61f5d9563a4316360082a94a7090fec2248dad2be59bca14ad09be`, 18 tests).
* Unchanged byte for byte: model files, `v8f_t1_eval.py` (`61e140ed...`), data module, `rule_relabel.py`, `dist_rule.py`. No model-config key: a checkpoint's `cfg` equals its base's and the pinned eval tool strict-loads it (o5).
* A: term = W x mean over lead samples of the mean over 30 knots of the squared distance between `out["traj30"]` and its detached cap (`distance_rule`, m 10, `floor_lifts=False`); mask and inputs as `--rule-relabel`. B: extra loss `(W - 1) x` (ade, fde, plus acc / launch_along when on) over the samples the relabel CHANGED, not renormalised.

## Selfcheck strings (gate_extra prints whole floats as ints)
`levers.w_rule_own=1` (GOW), `levers.w_rule_own=4` (GOW4), `levers.rule_relabel=on levers.rule_relabel_weight=4` (GRW6). Also `info.rule_own` and `rule_own_probe` (term, n_lead, n_active, gradient norms; only when on). Train log when on: `rule_own_term`, `rule_own_n_lead`, `rule_own_n_active`; weight: `rule_relabel_weighted_n`.

## Controls (each in `checks/`, script beside it; all PASS)
* o1 / w1 `o1_w1_identity.json`: both flags absent, 100 steps, shapes G, LS, GFT, GRT: all logged fields and final weights max abs diff 0.0 against the base tree.
* o2 / o3 `o2_o3.json`: cap equals `distance_rule` by hand, exact (49 masked rows of 64; 39 changed on a planted fast plan); no-lead rows and NaN-speed rows give exactly 0 term and gradient; reach in three parts plus partial (24 of 49 lead samples active on a half-planted batch); gradient reaches the model.
* o3 / w3 mutation `o3_w3_mutation.json`: control green; deleting the add line (MA1) goes red in all three checks, the loop call (MA2) goes red in the trainer run, the weight add line (MB1) goes red in the in-process check and the trainer guard.
* o4 / w4 / o5 `o4_o5.json`: gate, refusals, log keys, cfg equal to base, strict load. w2 / w4 `w2_w4.json`: weighted loss equals the hand computation (relative 3.8e-7), only changed rows get a gradient. Fine-tune smoke `t_ft_smoke.json`: GOW_s0/s1, GOW4_s0, GRW6_s0/s1 selfcheck with the card argv (2,000 / 6,000 steps) and 20 real steps, finite.
* Tests `tests.log`: 132 passed, 0 failed (new 18 + the 114 of the earlier builder's list).

## Build the pod tree
`cp -r /workspace/v8proof/tree_ruletrain /workspace/v8proof/tree_ruleown; cd tree_ruleown; patch -p1 --binary < RULEOWN.patch` (or copy `patched/` over `stack/`), verify the `PIN|` lines of `ARMS_RULEOWN.txt`, then run the selfcheck of GOW_s0 first and read `rule_own_probe`.

## Known limits (read before the arms start)
* Scale of the term: m^2 at weight W against position terms in m / 3 and / 6. On the tiny synthetic world it reads 300 to 600; the first real-data selfcheck prints `term`, `n_active` and the two gradient norms: read them. The card fixes the form, so it was built as written.
* Rule applied twice: 0.0 on the float64 path the term uses (bar 1e-5 met); 6.1e-5 for a float32 call, for the record.
* Real data not read; `info.rule_own` and the probe counts are measured on the first pod selfcheck.
* The o4 refusal run for `--dagger-p` stops earlier on the DAgger bank requirement; the clause itself is tested in `test_gdvb_declared_equals_built_and_a_wrong_expectation_is_refused`. `--w-rule-own -1` stops in the constructor (ValueError, rc 1).
* Temp folders left (delete refused in this session): `ruletrain2/work/tmp` 2.3 GB, `ruletrain2/work/pt` 1.7 GB, `ruletrain2/work/verify_patch` 336 KB. Trees `work/tree_base`, `work/tree_patch` 39 MB each (kept as reference / result).
