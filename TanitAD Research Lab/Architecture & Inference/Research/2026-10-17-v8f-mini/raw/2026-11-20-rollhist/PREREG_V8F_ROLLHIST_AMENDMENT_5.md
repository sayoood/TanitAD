# PREREG_V8F_ROLLHIST, Amendment 5 (FROZEN 2026-11-21 04:53Z by the MM, before any RH or B0 training step or number)

Card `PREREG_V8F_ROLLHIST.md` (781eea87) and Amendments 1 to 4 all stand. Amendment 4 is 192f1cff. The commit of record stays **97fe0a56**, and no `stack/` file changes.

This amendment fixes one study-side gate check, `C-RH-BANKGEN` in `/root/rollhist/study/dscale10k_check_rh.py`, so that it reads the bank's generator record from the place the trainer reads it. No arm, argv, step, read or bar changes.

## What happened (MEASURED, pod)
* After Amendment 4 the chain relaunched at 04:48:04Z.
* The RH_s0 selfcheck passed:
  * K-B3: 0 of 141 eval clips in the bank;
  * `declared_vs_built` PASS;
  * `dagger_probe` and `acc_probe` pass.
* The study-side gate-token mint then refused, on C-RH-BANKGEN only:
  * "the bank's generator record reads readout 'pinned', the argv declares 'causal_g1x'";
  * "... kappa 'none', the argv declares '0.2'";
  * "shard k: its provenance generator record differs from bank.json's", for k = 0 to 3.
* **The bank is correct.** These records all read `{readout causal_g, acc_gain 1.0, path_exec true, kappa_max "0.2"}`:
  * `bank/s0/bank.json` `common.generator`;
  * `PROVENANCE.json` `generator`;
  * all 4 shard provenance files.
* **The trainer reads that same record.** `DaggerBank` builds its header as `{**bank.json["common"], ...}` (`stack/tanitad/data/v8f_dagger.py:366`). `DaggerSwap.built_levers` reads `header["generator"]` (line 653).
* **The check read the wrong level.** It read `bank.json["generator"]` (top level), which the merge never writes (`stack/scripts/v8f_dagger_bank.py:329, 411`), so the absent record read as `pinned` / `none`.
* Its unit-test fixture (`build/b3gate/test_dscale10k_check_rh.py:155-157`) wrote the record **both** in `common` and at the top level. That is a shape the real merge does not produce, so the check passed its own tests. This is the "built, tested, and unreachable from its caller" class again.

## The change
* Check file sha256 94e6e147... → **69ff1a0f962123fd74c66cf6a1f3cb0e016799146ca75665c58fcb3e71e8cfc6**. The old file is kept as `dscale10k_check_rh.py.bgen`.
* The patch tool is `fix_bankgen.py` (sha256 10927bf6). It makes three anchored edits:
  1. the record is `gen_b = bank.json["common"]["generator"]`;
  2. each shard's provenance record is compared with `gen_b`;
  3. a top-level record that differs from `common` is refused.
* No gate token is pinned to the old file's sha (grep: none). The gate token binds this file, so the mint after the fix binds the new sha.

## Arms (MEASURED, pod, `/root/rollhist/logs/bankgen_arms.json`, sha256 33f5f7ce; ALL_AS_EXPECTED true)
| arm | input | result |
|---|---|---|
| K0 | the real s0 bank, fixed check | PASS, 0 problems |
| K0-old | the real s0 bank, old check | reproduces the refusal |
| R1 | `common.generator` removed | refused (reads pinned) |
| R2 | `common` κ set to none | refused (kappa) |
| R3 | `common` readout set to pinned | refused |
| R4 | one shard's acc_gain set to 0.5 | refused (shard 1 only) |
| R5 | record only at top level | refused: the trainer would read pinned |
| R6 | top-level record differs from `common` | refused |
| R7 | wrong declared sha | refused |
| R8 | top-level copy equal to `common` | PASS |

The mutated copies were own temp under `/root/rollhist/tmp_bankgen` (214 KiB). They were deleted by the script, and the deletion is logged in `deletions.log`.

## Relaunch
The same command (`relaunch_rh.sh`, causal_g1x). The selfcheck re-runs, and the partial dir is moved aside by `arm()`. This is a first read under the failed-control rule: the control form is unchanged, and the check now reads what its own docstring names ("as `DaggerSwap.built_levers` reads it").

## Follow-up (not blocking)
* Fix the b3gate fixture so that it writes the record only in `common`, the merge's shape.
* Add a test that the check passes the shape the merge writes.
* The lesson, added to Amendment 4's: before a launch, run the study-side mint, not only the trainer's selfcheck, on a real bank on the pod.
