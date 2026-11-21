# HANDOFF: RULETRAIN, trainer flag `--rule-relabel` (card ADD32), 2026-11-15

State: DONE on the CPU. Nothing trained, no git, no pod, no box 2. Folder `F:/TanitAD-scratch/v8proof/ruletrain/`. Base = LEADSUP tree (`C:/Users/Admin/v8proof_c/leadsup/tree_patch`; its sha256 verified on the copy before the edit). Work trees: `C:/Users/Admin/v8proof_c/ruletrain/tree_base` (clean copy), `tree_patch` (result).
Definitions and every loss-term decision: `RULETRAIN_DEFS.md`. Pins: `PINS.txt`. Arms: `ARMS_RULETRAIN.txt` (GFT_s0/s1, GFTb_s0/s1, GRT_s0/s1; tree `/workspace/v8proof/tree_ruletrain`).

## What changed
* `stack/scripts/v8f_mini_train.py` (only changed file): flag `--rule-relabel {off,on}` (default off), `bad_rule_relabel` (G-DVB), `rule_relabel_probe` (G-LIVE), RuleRelabel stage in the loop, loop guard `ZZRULE-NOT-APPLIEDZZ`, log / summary / checkpoint-totals keys only when on.
* new `stack/tanitad/data/rule_relabel.py` (stage + `relabel_gt`), new `stack/tanitad/data/dist_rule.py` (byte-identical to `distrule/dist_rule.py`, sha256 `40b5dd91...`), new `stack/tests/test_v8f_rule_relabel.py` (10 tests).
* UNCHANGED byte for byte: `v8f_mini.py`, `hjepa.py`, `v8f_beta.py`, `v8f_t1_eval.py` (`61e140ed...`), `v8f_mini_data.py`. No model-config key is added: a GRT checkpoint's `cfg` equals its base's.
* The relabel replaces `batch["gt"]` (the one target array of ade / fde / acc / launch_along) with `distance_rule(gt, v0=raw p10f[t][0], present, gap, gap, v_L=max(0, raw v[t] + closing), m=10, floor_lifts=False)` for rows with a label lead; labels from the join via `lead_sup_label_arrays` (no lead head, no `--lead-train`).

## Fine-tune path (card section 3): WORKS
`--init-ckpt <ckpt> --steps 2000 --lr 1e-4 --warmup 100` is accepted by the gate for both flag values (`checks/t_ft.json`, `ft_sc_*.selfcheck.json`): config check passes, 0 tensors differ after the strict load, schedule built 100 / 1e-4 / 2000. `--seed` is not in the config check, so base seed + 100 is possible: it changes the data order (`ChunkedSampler(seed=a.seed)`) and the per-step RNG, while the weights come from the checkpoint. The replicate GFTb uses `--seed` base + 100 (arms file).

## Selfcheck strings
`levers.rule_relabel=on` and `info.rule_relabel_n=<N>` (read from the first real-data selfcheck; synthetic 7 to 50). Also in selfcheck: `info.rule_relabel_changed`, `info.rule_relabel.{labels_sha256,n_pres,n_gap,n_clos,dist_rule_sha256}`, `rule_relabel_probe`. A flag-off arm has none of these keys. Train log when on: `rule_relabel_n`, `rule_relabel_changed`, `rule_relabel_rows` (per log interval).

## Results (each read from a file in `checks/`)
t1 identity, 100 steps, base vs patch, argv G / LS / GFT (fine-tune shape, flag absent): all logged fields and final weights max abs diff 0.0, PASS (`t1_identity.json`). t2 + t3 (`t2_t3.json`): relabelled rows equal a by-hand `distance_rule` exactly (49 relabelled of 64, 16 changed), other rows exact; with a lead the loss term and gradient differ, without a lead exactly equal (0.0). Mutations (`t3_mutation.json`): deleting the loop line is caught by the loop guard (rc 1; the selfcheck stays green, as it calls the stage directly); removing the stage's output is caught by t2/t3 and by the selfcheck probe; control green. t4 + t5 (`t4_t5.json`): selfcheck default / on PASS, gate_extra right rc 0 and wrong rc 1, refusals (ego-perturb, recover, no lead rows) rc 1, count above 0, the log carries the count; a GRT checkpoint loads strictly in the eval builder like its base. Tests (`tests.log`, full rerun after the non-finite-speed change): 114 passed, 0 failed (new 11 + LEADSUP / TARGET / CELL / MOTION / SPATIAL). The patch applies to a clean copy and the edit script replays to the same sha256 (`checks/verify_patch.log`).

## Build the pod tree
`cp -r /workspace/v8proof/tree_leadsup /workspace/v8proof/tree_ruletrain; cd tree_ruletrain; patch -p1 --binary < RULETRAIN.patch` (or copy `patched/` over `stack/`), verify the `PIN|` lines of the arms file, then run the selfcheck of GRT_s0 first and read `info.rule_relabel_n`.

## Known limits (not verified)
* Real data: nothing run on a real cache or the pod; the share of real logged futures the cap-only rule moves is unknown (synthetic: 13 to 16 of 64).
* Non-finite speed: read RAW from the clip store (`RuleRelabel(mode, J, store)`); a row with non-finite `v[t]` or `p10f[t][0]` keeps its target and is not counted, as the wrapper does (t2b in `t2_t3.json`, unit test). How often it occurs on real data is not measured.
* Refused with the flag: `--encoder lora`, `--ego-perturb-p`, `--dagger-p`, `--recover-p`.
* The LEAD hinge keeps the human arc `s_h` (DEFS, open question).
* t1 identity used the tiny base checkpoint `tmp/ft/base`; t_ft / t4_t5 use `tmp/ft_v2` (same recipe, bytes differ).
