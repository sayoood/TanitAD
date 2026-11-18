# BUILD_MIX_IMPL: the per-source T-block row weight (card section 3.3) and its reachability proof (section 3.4)

Card: `F:/TanitAD-scratch/datascale/card_mix/CARD_DSCALE_MIX.md` (sha256 `0bc241f5...`, read, not edited) + `MM_RULINGS_DSCALE_MIX.md` (ruling 2: identity 1 is a required test with its mutation).
Base: `cdbff971` of `C:/Users/Admin/tanitad-push`, exported to `F:/TanitAD-scratch/datascale/mix_impl/tree/` (`stack`, `taniteval`). Nothing committed, pushed, landed or run on a pod / box 2 / GPU. Evidence classes: all MEASURED here (logs under `mix_impl/logs/`) unless marked.

## 1. What changed (file, lines in the changed file; pristine copies in `mix_impl/base/*.base`)

| file | change | size |
|---|---|---|
| `stack/tanitad/models/v8f_mini.py` | `V8FMini.loss` `:1168` `rw = batch.get("tac_w")`; `:1170-1171` goal mask x `rw`; `:1174-1175` `_partial_ce(..., rw)` for lat / lon; `:1183-1187` cons with `mf = m.float() * rw`; `_partial_ce` `:1277-1285` optional `rw`. Absent key runs the original expressions. No config field. base sha256 `388f453a...` = `read3/READ_PINS.json`'s pin (the base copy matches it). | +9 lines |
| `stack/scripts/v8f_mini_train.py` | three flags `:315-320` (`--tac-src-w` 1.0, `--tac-src-clips` "", `--tac-src-clips-sha256` ""); `FLAG_CHECKS` `:143-144`; `TacSrc`, `make_tac_src`, `tac_src_stage`, `tac_src_count`, `tac_src_share`, `bad_tac_src`, `tac_src_probe`, `tac_src_probe_batches` (new block `:1652-1900`, before `declared_vs_built`); main build `:3573`; G-DVB `:3594`; G-LIVE `:3653-3663`; selfcheck `:3752`; resume/ckpt `:3802, :3820`; loop stage `:3853`, attach `:3893`, guard `:3903`; summary `:3992`. | +300 lines |
| `stack/tests/test_v8f_dscale_mix.py` (new) | 32 tests (card 3.4 checks 1-7, identity 1, the refusals, G-DVB, G-CKPT) | 442 lines |

Deviations from the card's wording (none changes behaviour):
* `bad_tac_src` is called in `main` beside `bad_perf_resume` (`:3594`), **not inside `declared_vs_built`**: `test_v8f_t2bank.py::test_reachability_from_the_caller_source` pins the literal text of the `declared_vs_built(...)` call and went RED when I added a parameter (found by the whole-file run, fixed, re-run GREEN). `FLAG_CHECKS` completeness is still enforced inside `declared_vs_built`.
* `--tac-src-clips-sha256` is REQUIRED with a list (a list with no declared sha is refused), stricter than the card's "sha != declared".
* `tac_src` is refused beside `--wm-only`, `--encoder lora`, `--t2-bank-p > 0` (the weight is defined on labelled cache windows only). UE_s0's argv uses none.
* Probe sampling: the three probe batches come from the trainer's own epoch-0 sampler (first 64 listed + 64 other windows; mixed = 32 + 32), not a separate fixed generator.
* The stage reads `ci` BEFORE `ego.apply` / `dag.apply` (a swapped sample keeps the human's labels) and `tac_w` is attached AFTER the stages (they return their own dict).
* The card's 7-check G-LIVE list names a 200-step union smoke (check 6): not runnable here (no union data, no GPU). Here check 6 is a 6-step `--synthetic` run (`test_check6_*`); the 200-step union smoke is the MM's gate run. UNVERIFIED on real union data.

## 2. Reach trace (argparse -> namespace -> batch key -> loss term -> backward), `file:line` at each hop

| hop | where |
|---|---|
| argparse | `v8f_mini_train.py:315-320` (`--tac-src-w/-clips/-clips-sha256`, defaults 1.0 / "" / "" = off) |
| namespace -> stage | `main` `:3573` `make_tac_src(a, J, clips, wins, info)` (`:1665`); refusals `:1670-1696`; `info["tac_src"]` `:1699` |
| G-DVB | `FLAG_CHECKS` `:143-144`; `bad_tac_src(a, tsrc, info)` `:3594` (declared lambda / list / sha == BUILT stage's) |
| G-LIVE | `tac_src_probe(...)` `:3663` on the BUILT `model.loss` (checks 1-5), record in `selfcheck.json["tac_src_probe"]` `:3752` |
| stage | loop `:3853` `tac_w = tac_src_stage(b, tsrc)` (from the LOGGED `ci`, before ego/dagger), counts `:3855` |
| batch key | loop `:3893` `b["tac_w"] = tac_w` (after the stages); guard `:3903` `ZZTACSRC-NOT-APPLIEDZZ` if the key is missing at the loss |
| loss term | `v8f_mini.py:1168` `rw = batch.get("tac_w")` -> goal `:1170-1171`, lat/lon `:1174-1175` -> `_partial_ce` `:1283-1285`, cons `:1183-1187`; total `:1244` (`cfg.w_tac * (goal + 0.5*(lat+lon) + w_cons*cons)`, unchanged) |
| backward | `v8f_mini_train.py` `T["total"].backward()` (unchanged line, in the loop after the guard) |
| run record | summary `:3992-3995`: `tac_src_w`, `tac_src_list_sha256`, `tac_src_rows_listed`, `tac_src_rows_other`, `tac_src_weight_share_listed`, `tac_src_empty_batches`; checkpoint carries `tac_src` + `tac_src_totals` (`:3820`), restored on resume (`:3802`) |

## 3. Default-off identity evidence (check 7, and identity 1)

* **Identity 1, absent key == the tip, bit for bit (MM ruling 2):** `test_identity1_*` (5 tests): (a) the original T-block lines re-spelled in the test from the model's own outputs equal `loss()` terms `torch.equal` on mixed / listed-only / other-only batches; (b) the changed model vs the pristine `cdbff971` model file (`base/v8f_mini.py.base`, loaded as a second module, same seed/weights): every term, the total and every parameter gradient `torch.equal`. Mutation (an absent key gets default weight 0.999) -> RED. (b) skips if the base file is absent (it is on this machine, it ran).
* **Whole-trainer default off:** `tools/default_off_identity.py` runs the SAME synthetic argv (8 steps, no `--tac-src-*`) with the pristine `cdbff971` trainer+model and with the changed ones: `train_log.jsonl` rows, model tensors, optimiser state, checkpoint key set, `summary.json` key set and values, `selfcheck.json` key set and all sections are identical (only `argv_sha` differs, because `--out` is in the argv). Log: `logs/default_off_identity.log`. Also `test_check7_default_off_*`: no `tac_src*` key in summary / selfcheck / checkpoint.
* Not verified: the "UE_s0's dry build gives the same `selfcheck.json` term values as at `cdbff971`" on the REAL union data (no union data here); the synthetic equivalent is the above.

## 4. Tests and mutations (`logs/test_v8f_dscale_mix_verbose.log`: 32 passed; also `logs/final_test_v8f_dscale_mix.log`)

Each RED arm: a mutated COPY (model: a copy of `v8f_mini.py` loaded under another module name; trainer: a copy laid out as `<tmp>/stack/scripts/`), the check's own GREEN control on the same batches. The tree is never mutated.

| card 3.4 check | GREEN on the change | RED mutation (test) | fires at |
|---|---|---|---|
| 1 upper identity (ones == absent, terms, total, all grads) | diff exactly 0.0 | `w * (rw[:, None] + 1e-3)` (`c1_upper_identity_epsilon`) | check 1 |
| 2 lower identity (lambda 0: T block exactly 0.0, backward exactly 0.0 everywhere; control absent > 0 on h_goal/h_lat/h_lon/h_cons) | block 0.0, grad mass 0.0, control > 0 | delete `rw` at the lat call; the same at lon; goal mask loses `rw`; cons loses `rw` (4 tests) | check 2 (4 of 4) |
| 3 partial, not a switch (lambda 0.5: analytic weighted mean rtol 1e-6; differs from lambda 0 and 1) | lat and lon match | `rw = (rw > 0.5).float()` (`c3_switch_not_weight`) | check 3 |
| 4 non-T invariance (every non-T term + grad of their sum bit-equal; control: a T term moves) | diff 0.0, control > 0 | `rw` multiplied into S's `ok` mask (`c4_rw_reaches_S_mask`) | check 4 |
| 5 corpus rows untouched (listed-only batch: weight all ones, loss bit-identical to absent) | diff 0.0 | `tac_src_stage` inverted (`torch.where(full, lambda, 1.0)`) (`test_check5_mutation_inverted_source_turns_RED`) | check 5 alone, and check 2 |
| 6 loop reach (summary: `tac_src_w` 0.0, `rows_other` > 0, `weight_share_listed` == 1.0) | `test_check6_loop_reach_*` (6 synthetic steps; listed + other == 96 rows; selfcheck probe pass) | delete `b["tac_w"] = tac_w` -> `SystemExit ZZTACSRC-NOT-APPLIEDZZ`; also delete the count call -> `rows_other` 0, share None (K-TACSRC would refuse) | the guard / K-TACSRC fields |
| 7 default off (no stage, no key; no key added to summary / selfcheck / checkpoint) | `test_check7_default_off_*` + the whole-trainer identity above | default of `--tac-src-w` set to 0.0 -> `SystemExit ZZDVB-REFUSEDZZ` (lambda != 1 with no list) | the refusal |
| identity 1 (ruling 2) | 5 tests, incl. vs the pristine model file | absent key gets weight 0.999 -> oracle RED | `test_identity1_RED_*` |

Count: 12 mutation arms (7 model-source, 1 stage monkeypatch, 3 trainer-source, 1 identity-1 model-source), all RED on the intended check; every one has a GREEN control run in the same test. Further tests: all refusals (lambda out of [0,1] / nan, lambda != 1 no list, list with lambda 1, sha mismatch = G-RM4, missing sha, missing file, foreign clip, all / none of the clips, duplicate id), `bad_tac_src` read-back of each flag (+ refusal beside `--wm-only` / `--encoder lora`), the stage's values, a probe that refuses `w_tac` 0, and G-CKPT (crash at step 4, resume: counts and final model tensors bit-equal to the uninterrupted run).
Measured on the synthetic rig (3 listed + 3 other clips): at lambda 0 the probe's check 2 gives T block exactly 0.0 and gradient mass exactly 0.0 on an other-only batch, control (key absent) > 0 on all four heads.

## 5. Whole-file runs of the existing tests that touch the changed modules (47 files + the new one)

Selection: every `stack/tests/*.py` that imports `v8f_mini_train`, `tanitad.models.v8f_mini`, or `v8f_mini_data` (list: `logs/touched_test_files.txt`). Run WHOLE, one at a time (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, `--basetemp` under `mix_impl/tmp`, `wait_commit.py 5.5` before each). Logs: `logs/new_<file>.log` (first full pass) and, for the 34 files that import the trainer (plus the new file), `logs/final_<file>.log` (second pass after the t2bank fix); summaries `logs/new_SUMMARY.txt`, `logs/final_SUMMARY.txt`.
* GREEN in both passes: 36 files (e.g. dagger 57, dagger_train 95, ego_perturb 80, recover_wiring 53, mini 28, otband 49, lead_head 45, level_head 27, window_start 66, t1_speedsub 33, beta 35).
* **One regression found and fixed:** `test_v8f_t2bank.py::test_reachability_from_the_caller_source` (a source pin on the `declared_vs_built(...)` call text) -> moved the `bad_tac_src` call out of that call; now `11 passed`.
* **11 files RED, every failure SET identical to the pristine `cdbff971` tree** (the same files run in an untouched export, `logs/base_<file>.log`, `logs/base_SUMMARY.txt`; compared by failing test id): `test_ra1_hjepa` 4, `test_v8f_cz` 1, `test_v8f_ego_dagger_compose` 1, `test_v8f_flat` 1, `test_v8f_kindx_s1` 149, `test_v8f_kscore` 4, `test_v8f_launch_intent` 1, `test_v8f_lora` 14, `test_v8f_mixo` 1, `test_v8f_oacc` 2, `test_v8f_tac` 4 (11 files, 182 failures, all pre-existing; not caused by this change). (ra1_hjepa and cz do not import the trainer; run once.) The causes were not investigated (queue-stage files / snapshot pins absent from this export).
* Not run: the other ~14,000 tests of the suite (G-SUITE in the pod environment is the MM's).

## 6. What I could NOT verify
* Any run on the real union data / the pod / a GPU (none was allowed): the 200-step union smoke, UM_s0's real `selfcheck.json`, K-TACSRC on 61,568 steps, C-INIT / C-ORDER (the stage draws no random number and reads `ci` only; synthetic bit-identity above is the evidence).
* `bset_clips.txt` itself (4,512 ids; the card says it is written from B_s0's dry record clip list) was not built here; the code takes any list whose byte sha256 equals the declared one, and `train_sha256` equality is the checker's C-TACSRC-LIST.
* The pod-environment G-SUITE and the 0-regression comparison against `cdbff971` there (differences from this dev box: BLAS threads, real files under `/workspace`).
* `launch_gate.py` row text: not applied (binding), as the card says.
* Gradient-variance / loss-scale effects at lambda 0 on the real data (about 19.9 corpus rows per batch, card section 3.2): by design, not tested here.

## 7. Deliverable manifest

| artifact | where | single copy? |
|---|---|---|
| changed trainer / model / new test (exported tree) | `F:/TanitAD-scratch/datascale/mix_impl/tree/stack/{scripts/v8f_mini_train.py, tanitad/models/v8f_mini.py, tests/test_v8f_dscale_mix.py}` | **yes, F: scratch only, NOT staged / NOT in git** (the brief forbids touching `E:/Projects/TanitAD/stack`; the MM lands it as a child commit of `cdbff971` on its own branch) |
| pristine copies | `mix_impl/base/v8f_mini.py.base`, `mix_impl/base/v8f_mini_train.py.base` (git blobs of `cdbff971`; the pristine tree is also re-creatable by `git archive cdbff971`) | re-creatable |
| blob shas for the pod G-SUITE | `mix_impl/expect.tsv` (`git hash-object --no-filters`) | yes |
| this doc | `mix_impl/BUILD_MIX_IMPL.md` | yes |
| logs (new test file verbose + quiet, 47 touched files x 2 passes, 12 pristine-tree controls, default-off identity) | `mix_impl/logs/` | yes |
| tooling (default-off identity script, run scripts, env) | `mix_impl/tools/` | yes |

expect.tsv (blob sha, path):
```
1879d27fb4985c7a4cfc7ae26b9faef1852472ca	stack/scripts/v8f_mini_train.py
87750e5b0fa5d6a7ced59adb1f30e27ff3bb632d	stack/tanitad/models/v8f_mini.py
1e98fa28d1f950fa9e27c112cbc005d5166e564e	stack/tests/test_v8f_dscale_mix.py
```
(the pristine `v8f_mini_train.py` blob at `cdbff971` is `10cca1339e3b36afc3a0aff7e2892d078a664b04`.) File sha256: trainer `016cd146...`, model `c981d644...`, test `ebc5fea9...` (model change moves `v8f_mini.py` from `388f453a...`: any pin of that sha sees it, as card section 0 item 2 says).

## 8. Deletions by this agent (own temporary data, literal paths; nothing was the only copy: the exports are re-creatable from git, the logs and scripts are kept)
Logged below after deletion.

| path (literal) | size | why it was safe |
|---|---|---|
| `F:/TanitAD-scratch/datascale/mix_impl/tmp/tree_base` | 460,755 KB | pristine `cdbff971` export, re-creatable by `git archive cdbff971` (the two touched files are kept as `base/*.base`) |
| `F:/TanitAD-scratch/datascale/mix_impl/tmp/run` | 1,089,679 KB | pytest `--basetemp` folders and synthetic-world run dirs; every log is kept under `logs/` |
| `mix_impl/tmp/*.py`, `*.sh`, `run_all.out` (patch scripts, env, run scripts) | < 60 KB | the reusable ones are copied to `tools/`; the patch scripts only produced the files in `tree/` |
| `mix_impl/tree/**/__pycache__` (4 dirs + `stack/scripts/__pycache__`) | < 2 MB | bytecode |
Kept: `tree/`, `base/`, `logs/`, `tools/`, `expect.tsv`, this file. `tmp/` is left as an empty directory.
