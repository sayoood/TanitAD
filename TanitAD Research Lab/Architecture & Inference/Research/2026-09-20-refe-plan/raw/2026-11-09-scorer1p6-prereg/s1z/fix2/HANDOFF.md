# HANDOFF: S1Z G18 fix 2 (probe batch + noise floor), 2026-11-10. CPU only; nothing committed, pushed, or on a pod/GPU.

Base: tip 9e1efffe. Changed (3 files, LF, in `files/`, list in CHANGED.txt; patch `g18_fix2_vs_9e1efffe.patch` passes `git apply --check`), all under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`:

## The change
- `scorer_finetune.py` 279-280: constants `TWIN_PROBE_SEARCH_MAX = 400`, `TWIN_NOISE_FACTOR = 10.0`.
- `scorer_finetune.py` 2538-2580 (G18): (1) no-grad search of the training loader (`dl`, cycled) for the first micro-batch with `sets_with_pair > 0` and `pairs > 0` on EVERY `--twin-heads` head under the live top-K rule (via `twin_term` + `twiner.last`); none within 400 -> `probe_batch.found false`, `reason` names "the search limit of 400 micro-batches", G18 not ok. (2) On that batch: g0a, g0b (w=0 twice), g1 (w=1), each through the real `train_losses`. noise_m = sum|g0a-g0b|, delta_m = sum|g1-g0a|. PASS needs, per SCORER module, finite, delta>0 and delta > 10*noise. (3) The exact-0.0 control (term alone, w=0) now runs on the probe batch (was `batch`; same code, stronger input).
- `g18` dict (2614-2625): `ok` adds `pb_batch is not None and noise_ok`. `loss_grad_delta_w1_minus_w0` now holds delta_m (same key, judge_g18 reads it). New keys: `loss_grad_noise_w0_vs_w0`, `noise_factor`, `noise_rule_ok`, `delta_over_noise` (None when noise is 0), `probe_batch` {found, index, searched, search_limit, pairs_per_head, reason}. `micro_batch_pairs` = the w=1 pass on the probe batch. Share checks, built facts, mutation-env checks unchanged.
- `s1z_layer.py`: the two stored `new` texts (constants entry, G18 entry) regenerated from the trainer; `peel()` returns git blob f5c8b769b612e4871f8022e69c05dc4076597556 exactly (MEASURED, sha1 of the peeled bytes).
- `test_s1z_preflight.py`: injection anchor/indent moved to the new block (uses `pb_batch`, `pb_rows`); stub model gets seeded logit noise; new section H (line 312).

## Test proof (MEASURED, CPU, venv tanitad, one at a time, free RAM >= 3 GB; `test_results.json`)
All 11 PASS (rc 0): test_s1z_layer, test_s1z_loss, test_s1z_gate, test_s1z_preflight (A..H), test_scorer1p6_gate, test_scorer1p6_preflight, test_scorer1_layer, test_pair_loss, test_pair_loss_p6, test_rl_epo (ZZTEST_PASS 42), test_dump_scores (ZZTEST_PASS 16).
Section H: H1 sparse bank (1 set in 8): probe index > 0 (pairless micro-batches skipped), pairs > 0 on every head, G18 ok. H2 bank with no pair: searched 400, found false, reason has "search limit of 400", no PREFLIGHT_PASS. H3 tiny noise (1e-6): ok, noise > 0, delta/noise > 10 all modules. H4 noise sigma 1.0, correct trainer, no env: noise_rule_ok false, G18 not ok. H5 DROP_LOSS + noise 1e-3: delta/noise < 10 on all modules (fails by the rule alone). Section C (NORADIUS, NOTOPK, DROP_LOSS, deleted line) still RED.

## Unverified / for the Master Mind
- The L40 GPU noise floor (bf16/TF32, two forward passes) is UNMEASURED. The 10x factor is the card of this task, not a measured margin; the real noise_m on the pod may exceed delta/10 on score_q_mlp or score_dec (earlier pod reading: delta 0.372 / 1.052 was noise on a pairless batch, so the true delta is unknown). If the preflight fails with a ratio near 10, read `delta_over_noise` before touching anything.
- `gate_checks.judge_g18` was NOT changed (instructed): it still only demands delta > 0, so on the pod it does not re-apply the 10x rule (it does read `mutate_*` flags and the shares). The 10x rule lives in the trainer's G18 `ok`; the chain's "mutant must fail preflight AND G18 ok false" check covers it. Consider making judge_g18 read `noise_rule_ok` / `probe_batch.found` in a later, separate change (moves no profile digest by itself but test_s1z_gate needs extending).
- Search cost: up to 400 no-grad forwards of the real trunk on the pod (batch-size dependent); at the measured pair-share (NC 0.031, DAC 0.016, TTC 0.125 per set) a joint-pair micro-batch should appear well inside 400, but this is ESTIMATED.
- The search iterates `dl`, advancing its shuffle generator; preflight exits before training so this should not matter, not checked for a combined preflight+run process.
- test_launch_gate was not in the required list and not run.

## Cleanup
`F:/tanitad_scratch/g18fix/` (tree, tmp, a/ b/ chk/, scripts, ~29 MB) deleted by literal path at the end (self-created temp, source = repo tip + this folder).
