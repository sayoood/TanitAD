# HANDOFF_ACTSENS_BUILD (agent, 2026-11-15): code and box-2 package BUILT, nothing launched, nothing committed or pushed, box 2 never written to

Everything below lives ONLY on `F:/TanitAD-scratch/actsens/` (one disk): bank it before anything is deleted. Progress lines: `F:/TanitAD-scratch/actsens/PROGRESS.md`.

## What passed (all dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`, MEASURED)
| step | result |
|---|---|
| 1 tree | `F:/TanitAD-scratch/actsens/tree`: 3,397 files = `git ls-tree` count, trainer sha256 `510b1e61`; local throwaway repo, baseline `8c28ba8`, ACTSENS commit `11b4ce47` (never pushed) |
| 2 IDM | `--wm-idm-gamma G1 G2 G3`, `V8FMiniConfig.wm_gamma_levels`, per-level build, `FLAG_CHECKS` entry, `bad_wm_idm_gamma` reading `HJEPAStack.built_levers()`. `test_v8f_actsens.py`: 18 passed (L1 grad 10.0x, L2/L3 and predictor bit-equal, `--wm a` exactly 0.0, C3 identity 900 x idm/L1, default bit-equal). Mutations M1-M8 all RED (`mutation_results_idm_ALL.json`) |
| 3 DIVRS | `2_divrs.diff` ported (3 hunks fixed by hand; the argparse hunk had landed mid-call and the sampler call in the wrong function, both repaired). `test_v8f_divrs.py` 18 passed; D1-D7 all RED (`mutation_results_divrs_ALL.json`); C4 regression = D1/D7. Both levers on the ladder path (`--wm-only --wm-windows wmonly`) gate PASS; uniform default epoch order bit-equal to `ChunkedSampler` |
| 4 parity | C1 (synthetic, two trees): M2b and WM-only state_dict sha, `live.terms`, grad groups all EQUAL ladder tree vs ACTSENS tree. C8 part A: R35_s0 with `stack.act_embed` zeroed reads forward response exactly 0.0; control 8.72 % / 10.32 % (the card's 8.7 / 10.3) |
| 5 package | `box2_build/`: UNIT 464/0, DRYRUN 110/0 (stub transport, poison ssh/sftp on PATH, no BREACH), ship build against the real source passes the needles (42 members) |
Final regression: `test_v8f_actsens + test_v8f_divrs + test_v8f_ladder` = 65 passed.

## Artifacts (sha256 prefix)
`actsens.diff` `0c66c201` (vs `8c28ba8`, reverse-applies clean; 5 files: trainer, model, data module, 2 tests) · trainer `5fe7296b` · `argvs_actsens.txt` `2d94ee1a` · `box2_build/frozen/argvs_actsens.json` `b0dd0af8` · `frozen/r35_live_terms.json` `3391b3b9` · `actsens_check.py` `8536b93f` · `PACKAGE_MANIFEST.sha256` lists all 23 files.

## Exact launch (Master Mind; needs the diff landed, a PASS token per run is made by the chain itself, and the box-2 go)
```
export ACTSENS_COMMIT=<40-hex of the landed ACTSENS commit>
export ACTSENS_SSH="$(command -v ssh)" ACTSENS_SFTP="$(command -v sftp)"
bash F:/TanitAD-scratch/actsens/box2_build/actsens_chain.sh     # optional ACTSENS_PRED_NAME / ACTSENS_PRED_STATE, ACTSENS_AMENDMENT
```
Order PILOT_DIV_s0, IDM_s0, NULL_s2, DIV_s0, IDM_s1, DIV_s1, NULL_s3 (card section 2); banks `C:/Users/Admin/t2v8f_pulls/actsens_<tag>/`.

## UNVERIFIED
* The real selfcheck's key names (read from source; the stand-in writes the same). Real-data C1 (`live.terms` of R35_s0's argv vs its banked `selfcheck.json`): the dev box has no cache_full / zcache / v9 labels, so only the synthetic two-tree form ran; the chain's C3 check enforces the real-data form on box 2 (bit-equal; a GPU mismatch REFUSES the IDM arm).
* IA-3 gains under C8 (only part A, the forward response, ran).
* DIVRS cell counts and start-up on 91-frame ladder windows (the pilot measures them). Which chain precedes ACTSENS on box 2. Pinned-suite lists for the two new test files (lander work).
* `test_v8f_divrs.py` mutation expectations match by test-name prefix (two names in `mutations_divrs.json` are abbreviated).

## Deleted (own temp, literal paths; none was the only copy)
`F:/TanitAD-scratch/actsens/bt/{d1,final1,t1,t2,t3}`, `c1/{s_actsens,s_ladder}`, `unit_tmp/{u1,u2,u3}`, `divrs_work`, `shipcheck/actsens_tree.tar` (sha in its manifest), `C:/Users/Admin/AppData/Local/Temp/actsens_dry_run_6718`. Left: `tree` (the only copy of the tested state), `divrs_apply`, `gac_tip.md` (4 MB), earlier attempt's files.
