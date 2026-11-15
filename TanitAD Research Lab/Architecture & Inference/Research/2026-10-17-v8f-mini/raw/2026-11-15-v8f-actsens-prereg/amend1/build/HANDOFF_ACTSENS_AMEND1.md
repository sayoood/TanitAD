# HANDOFF_ACTSENS_AMEND1: AMEND_ACTSENS_1 implemented in the trainer and the box-2 package (nothing launched, nothing written to box 2)

Date 2026-11-15. Spec: `AMEND_ACTSENS_1.md` in this folder. Card, doses, arms, run order and outcomes are untouched.

## Headline: integration the Master Mind must do
1. **Land the trainer change.** `F:/TanitAD-scratch/actsens/amend1/actsens_amend1.diff` (92 lines, two files: `stack/scripts/v8f_mini_train.py`, `stack/tests/test_v8f_actsens.py`) applies on top of the ACTSENS code `11b4ce47` (content-equal to the code of record `6b2bca1f`). Local throwaway commit `d0303505a0b569e0aaf03413ee9eee195e67206c` in `F:/TanitAD-scratch/actsens/tree` (never pushed). The Master Mind builds the real commit.
2. **Replace the landed package** `C:/Users/Admin/actsens_pkg` (not edited by me) with `F:/TanitAD-scratch/actsens/amend1/box2_build/`. Then export `ACTSENS_COMMIT=<the real landed commit>`. The ship stage refuses any commit whose trainer lacks the string `"live_first"`.
3. The pinned `frozen/argvs_actsens.json` sha256 changed to `c111dc0372dd35b973adadd7a0bcd0d514df708b03aea03f3d7ad86796dc94e3` (the six run argvs are byte-identical to the old file; two gate-only entries were added). `actsens_check.py` pins it.

## Part 1, the trainer
* `wm_only_live_check` keeps `terms_first`: the terms of step 0, taken before any `opt.step()`. `live.terms` (the last of three AdamW steps) is unchanged.
* `main()` pops `terms_first` out of `live`, so the `live` key set of every selfcheck is unchanged, and writes `sc["live_first"]` only when `a.wm_only and a.wm_idm_gamma is not None` (the flag is named in the argv). A flag-absent selfcheck has no `live_first` key.
* Tests (`stack/tests/test_v8f_actsens.py`, ladder-path synthetic gate, `--wm-only` as the real arms use):
  * `test_amend1_live_first_identity_dose_against_default_same_seed`: `--wm-idm-gamma 1000 100 100` against `100 100 100`, same seed. Every unweighted `live_first` term equal bit for bit, `level/L2` and `level/L3` equal, `level/L1` difference = 900 x `idm/L1` within 1e-5 relative. It also asserts the dose gate's `live.terms` differ from its `live_first` (step 3 against step 0).
  * `test_amend1_flag_absent_has_no_live_first_and_its_live_terms_equal_the_default_named`: no `live_first`, same `live` key set, `live.terms` equal to those of the `100 100 100` gate bit for bit.
* Mutation MA1 (record `live_first` from the last step, `if s == steps - 1`): the identity test went RED, restored byte for byte (sha256 re-checked). Record: `F:/TanitAD-scratch/actsens/amend1/mutation_amend1.json` (the 19 other tests of the file stayed green).
* `test_v8f_actsens.py` + `test_v8f_divrs.py` + `test_v8f_ladder.py`: **67 passed** (`trainer_tests.log`).
* Evidence class: MEASURED on CPU, synthetic. The real-data claim (REF live.terms equal R35's on all 25 keys) is the Master Mind's box-2 measurement; this package checks it on the real gate.

## Part 2, the box-2 package (`F:/TanitAD-scratch/actsens/amend1/box2_build/`)
* `frozen/argvs_actsens.json`: `REF_s0`, `REF_s1` = R35_s{k}'s exact argv (read from `C:/Users/Admin/t2v8f_pulls/ladder_R35_s{k}/argv.txt`), `--out D:\tanitad\actsens\gate_out\REF_s{k}`, `--wm-idm-gamma 100 100 100` appended. Text copy: `argvs_actsens_amend1.txt`. Generator: `make_argvs_amend1.py`.
* `actsens_check.py`:
  * `TABLE` gains `REF_s0/REF_s1` (lever `ref`, order 0, so not in `RUN_ORDER`; `cmd_stage` refuses them: gate only, no run token).
  * `GATE_AT_PILOT = ["REF_s0", "IDM_s0", "DIV_s0"]`. A non-pilot IDM stage (IDM_s1) gates REF_s{k} first (cached if already passed).
  * REF checks (`ref_problems`): the IDM checks (ladder, G-DVB, G-LIVE) with built gamma [100, 100, 100]; `live.terms` equal to `frozen/r35_live_terms.json` on every key (key sets equal too); `live_first` present and finite.
  * C3 (`c3_problems(init, live_first, ref_first)`): IDM `live_first` against REF `live_first` read from REF's gate output on box 2: unweighted terms and level/L2, level/L3 equal, level/L1 difference = 900 x idm/L1 (1e-5). REF missing, token invalid or its selfcheck sha differing from its token: the IDM arm is refused. The IDM token records `c3_ref_selfcheck_sha256`.
  * DIV and NULL selfchecks carrying `live_first` are refused.
  * NEEDLES (also used by `actsens_ship_tree.py`) gain `"live_first"`.
  * Cleanup also removes the REF gate dir (own temp, literal path, logged). The pilot and each IDM bank carry `ref_selfcheck.json` (`stages/actsens_run.sh` pulls and sha-checks it).
* Tests: stand-in trainer `tests/plant_trainer.py` emits `live_first` (step-0 semantics: IDM `live.terms` drift away from R35's as in the 02:38 case) and modes `drifted_first`, `other_batch_first`, `ref_term_changed`, `no_first`.
  * UNIT: **ok=633 bad=0** (`tests/UNIT_RESULT.log`). A2 regressions covered: C3 reading `live.terms`; a REF with one term changed (flipped last bits); an IDM `live_first` from another batch; plus missing REF, missing `live_first`, other REF batch.
  * DRYRUN: **ok=122 failed=0** (`tests/DRYRUN_RESULT.log`), a new section 11 runs the three regressions over the real chain (pilot stops rc 4; a refused REF_s1 refuses IDM_s1 only). A first dry run was contaminated because I edited `actsens_run.sh` while it ran (bash reads scripts as it goes); its log is `dryrun_contaminated_run1.log` and is not evidence. Sections 2 to 11 of it had 0 failures, and the clean rerun passed all 122.
  * Shipcheck (`tests/SHIPCHECK_RESULT.log`), against the throwaway repo, not the mirror: commit `d030350...` builds (42 members, tar sha256 `f2c92f68f984e209d219df2b419742de811cdb8d52cb088450d69c186f5ccbf2`); the previous throwaway commit `11b4ce47` is REFUSED for lacking `"live_first"`.
* `PACKAGE_MANIFEST.sha256` regenerated (23 files). README gate section updated.

## Deliverable manifest (all single-copy unless noted; none is in a repo)
| artifact | where |
|---|---|
| trainer + test change | `F:/TanitAD-scratch/actsens/tree` commit `d030350`; diff `F:/TanitAD-scratch/actsens/amend1/actsens_amend1.diff` (single copy) |
| mutation record | `F:/TanitAD-scratch/actsens/amend1/mutation_amend1.json` (spec: `mut_amend1.json`) |
| trainer test log | `F:/TanitAD-scratch/actsens/amend1/trainer_tests.log` |
| amended package | `F:/TanitAD-scratch/actsens/amend1/box2_build/` (single copy) |
| argvs (text) | `F:/TanitAD-scratch/actsens/amend1/argvs_actsens_amend1.txt` |
| shipcheck tar and manifest | `F:/TanitAD-scratch/actsens/amend1/shipcheck/` (built from the throwaway commit; rebuild from the real commit) |
| patch scripts used | `patch_check.py`, `patch_plant.py`, `patch_unit.py`, `make_argvs_amend1.py` in `F:/TanitAD-scratch/actsens/amend1/` |
| deletions log | `F:/TanitAD-scratch/actsens/amend1/deletions.log` |

## Not done, and risks
* Not run on box 2 or on real data. UNVERIFIED until the first real gate: that the real REF selfcheck carries `live_first` with the key names the checker reads (read from the trainer source; the stand-in uses the same names), and that R35_s1 reproduces like R35_s0 did (25 of 25 was measured for s0 only).
* The package's ship stage reads the code from a commit in the mirror `C:/Users/Admin/tanitad-push`; that commit does not exist yet.
* F: free space: 2.2 GB at the start (below the 3 GB floor, other sessions' data), 1.7 GB at the low point while a test run held about 230 MB, 8.7 GB at the end. Temp dirs deleted by literal path, logged in `deletions.log` (some sizes not measured before deletion).
