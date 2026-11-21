# NOTES: PREREG_V8F_DAGGER, the INTEGRATION build (data worker + trainer worker + the fourth flag)

Built 2026-10-26 in `E:/Projects/TanitAD-artifacts/di/tree` from the tip `2a78f059cd4dcd29c279467ccca84581123d73b5` of `agent/arch-inf-20260803` (`di/TIP.txt`). STAGED, nothing committed or pushed.
Evidence classes: MEASURED (this build; artifact path) / UNVERIFIED. CPU only; no GPU, no box 2, no real corpus.
**The proofs on the final bytes are in `FINAL.txt` (written by `di/chain.py`; copy it and the logs `p1b_mutate_dagger.log`, `p2b_mutate_dagger_train.log` next to this file when landing). This file states what was built; FINAL.txt states what was proved.**

## 1. What was integrated

| piece | from | note |
|---|---|---|
| `stack/tanitad/data/v8f_dagger.py` | data worker (`dg/tree`) | byte-identical, sha256 `831f64f8...` |
| `stack/scripts/v8f_dagger_bank.py` | data worker + **E1 patch** | the patch (`PATCH_v8f_dagger_bank_window_start.diff`) spells `window_start=cfg.window_start` at both `V8FMiniWindows` calls; behaviour identical |
| `stack/tests/test_v8f_dagger.py` | data worker | the 40-item version, sha256 `a4f53cd1...`, identical to the trainer worker's copy |
| `stack/scripts/v8f_mini_train.py` | trainer worker (`dt/tree`) + **the fourth flag** | see section 3 |
| `stack/tests/test_v8f_dagger_train.py` | trainer worker + 7 new test functions (13 new items), 6 edited | see section 3 |
| `raw/2026-10-26-dagger-build/`, `raw/2026-10-26-dagger-train/` | the workers | `mutate_dagger_train.py` extended by 14 mutants (85 -> 99); `smoke/smoke.sh` extended; worker NOTES carry an INTEGRATION UPDATE banner |

**Rebase check (MEASURED).** `dt/tree`'s `v8f_mini_train.py` minus its DAgger changes equals the tip's: the tip blob and the worker's saved pre-change reference (`pre_dagger_v8f_mini_train.py.ref`) are both sha256 `65ea1e3a5c05...`; the only non-DAgger lines the worker's diff removes are the `declared_vs_built` signature, its call in `main`, and the closing of two dict lines. The tip moved `07dcb513 -> 2a78f059` between the two workers' builds, and `git diff --name-only 07dcb513 2a78f059 -- stack taniteval` is EMPTY: no rebase was needed and none was done.

## 2. E1 (applied)
The patch applies with `git apply --check` clean. `test_v8f_window_start`'s static guard now sees `window_start=` at both calls. No data-library mutant is anchored on the changed line (the data mutator's `--check`: 57 mutants, each anchor found exactly once, each mutated text compiles), so no re-anchoring was needed.

## 3. The fourth flag, `--dagger-bank-sha256 <64 hex>` (PREREG section 4: "DAG names all four"; escalation E2 of the trainer worker)
| rule | where (`v8f_mini_train.py`) | test | mutant |
|---|---|---|---|
| default "" = off (a default run is the pre-change run) | parser `--dagger-bank-sha256` | `test_the_four_flags_exist_default_to_off...` | S01 |
| has a G-DVB `FLAG_CHECKS` entry (`dagger:dagger_bank_sha256`) | `FLAG_CHECKS` | `test_the_four_flags...`, `test_a_flag_without_an_entry_is_refused[dagger_bank_sha256]` | S02 |
| declared with no swap built is refused | `bad_dagger` `dfl` tuple | `test_a_declared_flag_with_no_swap_built_is_refused[extra3]` | S03 |
| REQUIRED when `--dagger-p > 0` | `bad_dagger` | `test_g_dvb_refuses_an_illegal_or_empty_declaration[...is required...]`, `test_required_at_p_above_zero...`, `test_a_refused_declaration_stops_the_real_trainer...[extra4]` | S04 |
| REFUSED when `--dagger-p` is 0 (with or without a bank) | `bad_dagger` | the same tests, `[extra5/extra6]` | S05 |
| declared with no `--dagger-bank` at p > 0 is refused | `bad_dagger` | `...illegal_or_empty_declaration[...names a bank to hash]` | S06 |
| the GATE re-hashes `bank.json` from disk and refuses a declared hash that differs | `bad_dagger` | `test_the_gate_rehashes_bank_json...` (also after the file is changed on disk, post-build) | S07, S08 (case) |
| the declared value must be a full sha256 (64 hex digits, either case) | `is_sha256_hex`, `bad_dagger` | `test_a_declared_value_that_is_not_a_full_sha256_is_refused[x6]`, `test_the_sha256_shape_check...`, `test_a_matching_declared_sha256...` (capitals) | S09, S10, S11, S12 |
| the bank is LOADED against the declared hash (`DaggerBank.load(expect_sha256=)`): another manifest is refused before any control is read | `load_dagger_bank` | `test_a_declared_sha256_that_is_not_the_banks_is_refused_at_load`, `test_a_wrong_declared_sha256_stops_the_real_trainer...` | S13 |
| recorded: `levers.dagger_bank_sha256_declared` (beside the built `levers.dagger_bank_sha256`; None when the lever is off), in `selfcheck.json`, `summary.json` | `main()` | `test_flags_to_config...`, `test_a_default_run...`, `test_dag_0...` | S14 |

* The two layers are deliberate and tested separately: the load refuses a bank whose manifest hashes otherwise; the gate (`declared_vs_built`) independently re-hashes the file at gate time, so a `bank.json` that changed after the swap was built is still seen.
* `launch_gate.py` was NOT touched. Its "data-manifest sha256s (including the bank's)" must take the bank's sha from `selfcheck.json` `levers.dagger_bank_sha256` (now also declared in argv, hence inside the token's argv sha256) or hash `bank.json`. **ESCALATION E7 (Master Mind):** decide whether `launch_gate.py` gets an explicit bank-manifest entry.
* Edited existing tests: the `DAG(...)` helper now names all four flags (computing the manifest sha), the inline `{bank}` cases carry `{sha}`, the flag-set assertion is "the four flags" (it asserted the absence of the fourth).

## 4. Disk (MEASURED, 2026-10-26)
E: was at 100 % (0 B free) after the archive extraction; the first pytest run died with `ios_base::badbit` while writing a checkpoint. `di/tree` was pruned of `taniteval/results` (1.1 GB) and `stack/experiments` (664 MB); no `v8f` test reads either (grep). Both are restorable from the tip. The manifest lists them under `MISSING_FROM_TREE` and they are NOT deletions to land. E: still holds the workers' disposable mutation copies `dg/mut` (784 MB) and `dt/mut` (1.3 GB).

## 5. Open items (decisions and UNVERIFIED), unchanged from the workers unless stated
* E3 (eval-141 source: the repo grid `raw/eval_grid_windows.json` is REQUIRED by the trainer), A1 (tie set = the PREREG text), A2 (nav progress on route-less frames): the workers' NOTES. The tip's PREREG Amendment 1 already records A1/A2.
* E6 (the bit-identical exemption fires how often on real data): UNVERIFIED.
* No real TRAIN cache, no real bank, no CUDA run, no 64-row gate probe on real data: UNVERIFIED.
* `smoke/smoke_output.txt` is the WORKER's three-flag smoke; `smoke.sh` is updated to name `--dagger-bank-sha256` and to add three refusals (no hash at p > 0, a hash at p = 0, a wrong hash). Re-running it is not part of the proofs of record (`FINAL.txt` lists them); if it was re-run, `smoke/smoke_output_integration.txt` holds the output.


## 6. SECOND INTEGRATION PASS (2026-10-26, after the first proof run): two record gaps closed; the proof of record is RE-RUN on these bytes

The first proof run (`di/out_run1`, kept) was stopped: two things the reader (`C:/Users/Admin/dagread/dag_read.py`) REFUSES on were not written by the build. Both are records, not levers: no loss trace moves.

### 6a. Gap A: the bank generator now WRITES K-B4 and K-B5 (PREREG section 6 lists them as bank controls "read before any training")
| what | where |
|---|---|
| `kb4_cases()` runs the six synthetic cases (straight, stationary_tie, lateral_cut, heading_cut, end_edge, loop) against `match_route` as it is NOW; `kb4_record()` = `{PASS (derived from the cases), cases}`; `kb4_combine()` re-runs them at the merge and AND-s every shard's recorded case | `stack/tanitad/data/v8f_dagger.py`, section "the bank controls K-B4 / K-B5" |
| `kb5_measure()` re-anchors the bank's OWN records exactly as the swap does (`reanchor_rows` onto the float32 `pose_now`, cast to float32) and measures: `max_pose_now_err_m` (re-anchored NOW vs the LOGGED float64 pose at r'), `max_rel_transform_err_m` (EVERY pair of held frames, before = the record's float64 poses, after = the float32 tensor), their headings, `max_reanchor_vs_pose_now_m` (exactly 0.0) and `max_rel_transform_err_float64_m` (not gated: separates a formula defect from the float32 floor); `Kb5Accumulator` feeds it chunk by chunk; `kb5_pass()` = n > 0 and every gated number finite and <= `KB5_TOL_M` = 1e-4 (computed from the numbers, never from a flag); `kb5_combine()` = sums and MAX over shards | same file |
| `snapshot()` adds `log_now` (the logged pose at r'; NOT a bank field, so the shard bytes and `bank.json` are unchanged); `run_windows(kb5=)` feeds the accumulator; `cmd_gen` writes `controls` into each shard's provenance and exits 6 on a failed control; `cmd_merge` refuses a shard with no `controls`, recomputes PASS from the numbers, writes `controls` into `PROVENANCE.json`, and on a failure writes `PROVENANCE.json` with PASS false, `bank_sha256 = null`, `complete = false`, `failed_controls`, writes NO `bank.json` and exits 6 | `stack/scripts/v8f_dagger_bank.py` |
Tests (`stack/tests/test_v8f_dagger.py`): `test_kb4_the_seven_probe_cases` now calls `G.kb4_cases()` (the unit test and the record are ONE function); `test_kb4_each_case_goes_false_when_the_matcher_is_bent[6]`; `test_kb4_a_matcher_that_raises...`; `test_kb5_measure_*` (3) and `test_kb5_pass_*`; `test_kbctl_gen_*` (3) and `test_kbctl_merge_*` run the GENERATOR INSIDE the test body (a generator that fails to write a record fails the test, not a module fixture) and recompute K-B5 independently from the stored records and the store's logged poses. Mutants in `raw/2026-10-26-dagger-build/mutate_dagger.py`: 57 -> 86 (29 new: 6 K-B4 and 13 K-B5 library, 10 generator). The data suite is 40 -> 54 items. The shard's K-B5 record also carries `log_pose_sha256` (the digest of the logged poses it was compared against, in record order): the first pre-run of the new mutants left ONE survivor (`K-B5 reads the logged pose at the wrong row (r' + 1)`: the re-anchoring lands on whatever target it is given, so the numbers alone cannot tell which row was used), so the TEST was strengthened (an independent recomputation of the digest from the store) and the mutant is now killed.
**K-B5 definition (an interpretation to confirm).** The gate is the LITERAL reading of the instruction: the float32 tensors the swap emits vs the logged / recorded FLOAT64 poses. A clip whose poses are > ~1 km from the clip origin cannot be stored in float32 to 1e-4 m whatever the re-anchoring does, so it would fail on the float32 floor alone; the record carries `max_rel_transform_err_float64_m` (~1e-14) beside it to tell the two apart. PhysicalAI poses are clip-local (a 20 s clip is <~ 800 m), so this is expected not to bind; MEASURED on the 12-clip CPU smoke (6d).

### 6b. Gap B: the trainer records the TRAIN window-list digest on EVERY run
`stack/scripts/v8f_mini_train.py`: `build_data` sets `info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)` (every run: default, DAG-0, DAG); `summary.json` carries it at top level beside `train_sha256` (the level the reader's `--key-window-sha` / `--key-train-sha` defaults read, `dig`/`find_key` bare prefix); `selfcheck.json` carries it in `info`. The DAgger control path (`load_dagger_bank`) still computes its own `wsha` for K-B3.2; the new test asserts the two and the bank's `window_list_sha256` are equal. **Default-off identity:** the loss trace, the weights and the optimiser state of a default run are bit-identical to the saved pre-change trainer (`test_defaults_reproduce_the_pre_dagger_trainer_bit_for_bit`, run unchanged); its summary-key assertion now allows exactly ONE new non-dagger key, `train_window_list_sha256`. Test: `test_every_run_records_the_train_window_list_sha256_beside_train_sha256` (hand formula over the list the trainer BUILT, on three runs). Mutants W01-W06 in `mutate_dagger_train.py` (99 -> 105; the six killed in a group pre-run, 6 of 6). The trainer suite is 96 -> 97 items.

### 6c. Reader fixtures
`C:/Users/Admin/dagread/fixtures/dagger_smoke/` must be refreshed from `raw/2026-10-26-dagger-build/smoke_kb45/` (new `PROVENANCE.json` with `controls`, new `shard_000of001.provenance.json` with `controls`; `bank.json` is unchanged, see the smoke result). The trainer-side fixtures need a `summary.json` with `train_window_list_sha256`.

### 6d. The 12-clip CPU smoke of the generator, re-run into a NEW directory (`raw/2026-10-26-dagger-build/smoke_kb45/`; the old `smoke/` is untouched)
Same command as the data worker's (`smoke/shard_000of001.provenance.json` `argv`): the REAL `M2b_s0` checkpoint (sha256 `601cb435...`) on the 12 SYNTHETIC TRAIN clips of `dg/smoke_world`, `--limit-clips 12 --batch 32 --threads 4 --device cpu --replay-frac 0.1`, `smoke_run.log`. MEASURED: **`bank.json` is byte-identical to the old smoke's (sha256 `c23c35b5...`) and so is the shard `.npz` (sha256 `8250fe28...`)**: the records changed nothing about the bank. `PROVENANCE.json` now carries `controls`: **K-B4 PASS true, the six cases True; K-B5 PASS true**, 1,608 records, 410,556 held-frame pairs, `max_pose_now_err_m` 0.0 (the synthetic poses are exactly float32-representable), `max_rel_transform_err_m` 1.50e-05, `max_rel_transform_err_float64_m` 5.1e-14, `max_reanchor_vs_pose_now_m` 0.0. Plumbing only: the clips are synthetic, so the real corpus's coordinate magnitudes (the float32 floor of K-B5) are UNVERIFIED.
