# Brief: ROLLHIST B1 fix round (two card deviations plus the regression the full disk cut off; dev box only, no commit)

You are a TanitAD build agent working for the Master Mind (MM).

- Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
  `git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`
- The B1 build is in `F:/TanitAD-scratch/rollhist/build/`: the tree at `build/tree/stack`, the records at `build/b1/`. Start with `HANDOFF_B1.md`, `REPORT_B1.json` and `mutation_table_rollhist.json`.
- The card is `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_DRAFT.md`. Its arms table in section 2 is binding for the argv spelling.

The rules of the first B1 brief (`F:/TanitAD-scratch/rollhist/BRIEF_BUILD_ROLLHIST_B1.md`) all still apply: stage only, no commit or push, no pod, no GPU, no edits to the protected files, one pytest at a time, no secrets, no IPs.

## ⚠ DISK
F: ran OUT OF SPACE (ENOSPC) at 11:52Z during the B1 regression. It has about 7.7 GB free now, and other sessions use it too.
- Before EVERY pytest run, check `(Get-PSDrive F).Free`. If it is under 4 GB, wait (re-check every 60 s, cap 20 min), then report instead of running.
- Keep `TMP`/`TEMP` under `F:/TanitAD-scratch/rollhist/build/tmp/<run>`. Delete each one right after its run, by literal path, and log it in `F:/TanitAD-scratch/rollhist/DELETIONS.log`.

## Fix 1: the generator record only when it is non-default (card N2: the defaults reproduce today's generator BIT FOR BIT)
B1 writes the `generator` record into every bank header, so a new default bank's `bank.json` sha differs from today's.
- Change it: write the record (header, shard provenance, `bank.json`) ONLY when the settings differ from `pinned` / `none` (gain and path-exec at their defaults).
- A default run then produces files byte-identical to the unpatched generator's. Prove it with a test on the same small input: every output file is byte-equal to the unpatched generator's, `bank.json` included.
- `built_levers` already reads an absent record as `pinned` / `none`. Keep that.
- Adapt mutant M5 so it means "a NON-default bank carries no generator record", and make it go RED.

## Fix 2: the declaration names the WHOLE readout, as the card's arm spells it (`--dagger-bank-readout causal_g1x`)
B1 compares only the readout name and kappa; gain and path-exec are bound only through the bank sha. Per CLAUDE.md, the argv must NAME what the history is.
- Define ONE function, `readout_token(readout, acc_gain, path_exec)`:
  - `pinned` gives `pinned`;
  - `causal_g` gives `causal_g{gain:g}` plus `x` when path-exec is on.
  - Examples: gain 1 with path-exec is `causal_g1x`; gain 0.5 without it is `causal_g0.5`.
- Use it everywhere:
  - `built_levers` reports `dagger_bank_readout` as that token, derived from the loaded header;
  - the trainer flag accepts the tokens;
  - `bad_dagger` compares declared with built on the token;
  - `rollhist_pair_check.py` requires RH to name the token and the kappa.
- Tests:
  - `causal_g1x` declared for a `causal_g` bank with gain 1 and path-exec off is REFUSED;
  - the same for gain 0.5 with path-exec on;
  - the matching declaration is accepted.
- Add a mutant: compare only the readout base name, which must turn a named test RED.

## Regression rerun (B1's run lost two results to ENOSPC)
- Run, on the patched tree AND on an unpatched tree:
  - `tests/test_v8f_orc_chain.py`. B1 saw 13 errors there. Find out whether that is environmental or real.
  - `tests/test_v8f_ego_perturb.py`. B1 saw 3 failures on the patched tree, and its unpatched log is empty.
- The unpatched tree is an export of the same tip `7b26a8ac`. Delete it after the run.
- Rerun the three new test files (`test_v8f_t1_readout.py`, `test_v8f_rollhist.py`, `test_rollhist_pair_check.py`) after the fixes.
- Rerun ALL mutants, M1-M13 plus the new one.
- Write `regression_rollhist.json` properly (it is 0 bytes now).

## Deliverables (update in place under `build/b1/`)
- `rollhist.patch` (regenerated, LF);
- `TREE_FILES_SHA256.txt`;
- `mutation_table_rollhist.json`;
- `regression_rollhist.json`;
- `REPORT_B1.json`, with a `fix_round` section;
- `HANDOFF_B1.md`: append a "Fix round" section of at most 400 words;
- `MANIFEST_SHA256.txt`.

Hand back in at most 250 words. Do not wait on anything longer than about 4 minutes. Each test file takes at most about 5 min, so run them in the foreground one at a time.
