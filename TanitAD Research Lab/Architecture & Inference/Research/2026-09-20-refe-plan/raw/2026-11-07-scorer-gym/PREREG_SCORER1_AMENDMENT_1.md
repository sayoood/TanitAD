# AMENDMENT 1 to PREREG_SCORER1.md (sha256 1f839af2...): rulings on the implementation gaps

**Date:** 2026-11-08 ~09:30Z, Master Mind.
**Written before** any arm trained and before any held-out number was read. The arms, endpoints and bars do not change.

Item numbers follow `scorer1_impl_20261108/SPEC_GAPS.md`.

1. **"Speed direction zero by construction" is corrected to "count-balanced".** The balancing equalises the kept pair COUNTS (F = S, tested). Gradient mass can still differ, because pairs are weighted by the sigmoid and carry different speed contrasts. The preflight reports F vs S gradient mass as a descriptive number. It does not gate.
2. **"Mean over the kept pairs" is pooled over all kept pairs of the micro-batch** (sets and heads together). Accepted.
3. **Per-proposal validity.** `L_pair` sees only the set mask. Before launch it must be shown whether an invalid proposal's label can enter a pair. If it can, those proposals are excluded from `L_pair`. An unresolved answer is a launch blocker.
4. **λ_p.** Measured in the trainer by `--pair-calibrate` (50 micro-batches, gradients pooled over micro-batches and then normed, as in `weight_rule.py`; L_BCE = arm B's pointwise BCE including the lane term, without the expected-reward term). `--pair-calibrate` is a measuring mode and is forbidden in a launch argv.
5. **Seeds.** `--seed` is the data-order seed: C0 = 0, C0r = 1, S1 = 0. `L_pair` has its own generator (20261107), independent of `--seed`.
6. **"400 further updates" = `--max-updates 400`** with the trainer's cosine schedule over those 400 updates, the same for C0, C0r and S1. u400's own schedule covered a longer run, so "same recipe" means the same losses, learning rate, data and flags, not the same schedule position. The `refe_scorer1` profile must allow this flag.
7. **Checkpoint identity is bound by sha, and there is a discrepancy to resolve first.** The dev-box `model_sft4_B_u400.pt` hashes to `8b4dba48...` (MEASURED 2026-11-08), which is the sha this card and every dev-box u400 eval use. `MODEL_REGISTRY.md` lists `8e382eeb...` for the same name. Before launch: compare the two files' state dicts tensor by tensor, and record which file the pod holds. If the state dicts are identical, the difference is serialisation: bind to the pod file's sha and fix the registry note. If they differ, stop and escalate: the arms would not start from the checkpoint that was evaluated.
