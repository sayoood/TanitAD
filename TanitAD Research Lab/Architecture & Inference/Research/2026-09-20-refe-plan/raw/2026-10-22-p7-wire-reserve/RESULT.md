# P7-WIRE-RESERVE -- the session reserve (R-A) removed from every log-keyed training input, with an audit that proves it

Agent: P7-WIRE-RESERVE, 2026-10-22. 0-GPU, no network, no git tree touched, nothing staged / committed / pushed.
Everything below lives ONLY under `C:/Users/Admin/par18/p7wire_reserve/` (single location, see the manifest).
Source of truth: mirror tip `1664143ba9bfb9ad45aeb095a31a69dd6c5b7c30` of `C:/Users/Admin/tanitad-push`; `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
Evidence classes: MEASURED (ours, artifact path given) / INHERITED (another doc or agent, not re-verified here) / ESTIMATED (arithmetic) / PENDING.

## HEADLINE -- what the Master Mind (and the other agent) must do to integrate

1. **MM lands `code/refe/*` with the scoped lander** (4 files + 1 fixture + 1 new test, table in section 1). Land `session_reserve.py`, `test_session_reserve.py` and `fixtures/p7_reserve_sessions.json` together: the test and the chain read the fixture.
   Export/land with **LF** line endings (this box's `core.autocrlf=true` rewrote the reserve file to CRLF, sha `4f5b1f1d...` instead of the pinned `53f3bcd4...`; `load_reserve` accepts the CRLF form of the same text, a test pins that, but land LF).
2. **Edit outside the brief's owned list, flagged:** `test_p6_chain.py` +3/-1: `SANDBOX_DROP_PREFIXES` gains `"P7_"` (line 568). Without it a P7 chain's own G-SUITE would leak `P7_RESERVE` into the sandboxed default-path chain and refuse it. The tip's test is otherwise untouched (432 passes on the tip, same suite expected after; see the verification block).
3. **Other agent (P7-WIRE-HEAD / `launch_gate.py` owner):** call `load_reserve / filter_bank / audit_inputs` from G-RESERVE (exact names, section 1); add `test_session_reserve.py` (`ZZSESSRES_PASS`) to the `refe_p7` suite; **add a `P6_BANK_TUPLES`-style pin for the bank directory `train_grow_cmd_3c_p7`** (92,896 rank-0 + the rank-1 count from the PENDING step, section 5); regenerate `SHIP_MANIFEST` (`launch_gate.py ship-manifest`) for the new/changed files. `train.py` may call `assert_rows_clear(rows, reserve)` as its consumer-side refusal (not wired: I do not own `train.py`).
4. **Pending measurement (not run, by the brief):** the rank-1 exclusion needs the 3-C bank; the exact command is `raw/measure_rank1_3c.sh` (section 5). Until it runs, the P7 tuple count is ESTIMATED (~150,834), not pinned.
5. P6 default path is untouched and byte-identical when `P7_RESERVE` is unset (section 3, MEASURED by the sandboxed real-chain identity test against the tip's chain).

## 1. What was built

| file (authoring copy under `code/refe/`) | status | what |
|---|---|---|
| `session_reserve.py` | NEW, 767 lines, stdlib only | the module; interface below |
| `test_session_reserve.py` | NEW, 1,147 lines, 43 tests | unit + regression arms + REAL `p6_pod.sh` in the sandbox |
| `fixtures/p7_reserve_sessions.json` | NEW, byte-identical copy of `raw/reserve_sessions.json`, sha256 `53f3bcd4684513f0ee765fbcfbaf583b587f58c97e9c8915f4e4df3ae3e32f29` | makes tests / G-SUITE self-contained on the pod |
| `p6_prep.py` | +98 / -0 | six P7-only subcommands `reserve-*` |
| `p6_pod.sh` | +66 / -0 | dormant section 5a + final audit line |
| `test_p6_chain.py` | +3 / -1 | `"P7_"` added to `SANDBOX_DROP_PREFIXES` |

Module interface (`session_reserve.py`; line numbers in `code/refe/`):
- `session_of(log) -> str` (76): `log.rsplit("_", 2)[0]`, **but REFUSES a non-log string** (`UnresolvedLog`). The bare rsplit turns `"L0"` into the "session" `"L0"` that is in no reserve: a logless key would read as clean. Valid form: `\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+_\d+_\d+` (MEASURED: all 103,037 attributed token-table rows resolve with it, `n_unresolved 0` in `raw/derive_counts.out`; `RealDataTests`).
- `load_reserve(path) -> frozenset[str]` (155): sha256-pinned constants (49-51) for the file, the session list and the log list; refuses on any mismatch, wrong schema, wrong counts (16 / 116), or a log whose session is not listed. One changed byte refuses (`ReserveFileTests.test_one_byte_changed_refuses`).
- `filter_bank(rank_jsonl_in, out_path, reserve, *, token_log=None) -> {n_in, n_out, n_excluded, n_unresolved, sha256_out, ...}` (210): the single exclusion line is **257**; survivors are copied byte-for-byte; a row with no resolvable `log_name` raises `UnresolvedLog` after a full scan and **nothing is written**; write is tmp + `os.replace`. With `token_log` (token -> log from the token table) it also refuses a row whose `log_name` differs from its token's log (`AttributionConflict`): that is the twin cross-check.
- `audit_inputs(reserve, *, bank_files, metric_cache_dir=None, label_store=None, image_index=None, calib_rows=None) -> {ok, leaks:{input:[keys]}, unresolved:{input:n}, counts}` (549): `ok` is False on any leak, any unresolved key, **or an input that examined zero keys** (`counts["empty_inputs"]`; an empty input proves nothing); a missing path raises; lists are capped by `max_report` but `counts["leak_totals"]` is exact.
- Also: `filter_bank_dir` (281: all `targets_rank*.jsonl`, refuses strays like the trainer, writes `reserve_filter_report.json`, `targets_sha256` = the trainer's own identity digest of the outputs, `train.py:1026`), `assert_rows_clear` (311), `filter_calib` (343), `move_reserved` / `restore_moved` (436 / 475, reversible through the manifest `reserve_moved.json`, idempotent, merge-safe), `split_label_store` (511), `count_from_token_table`, `token_log_map`, CLI `verify|counts|filter|audit` (735).

## 2. How each input is keyed -- read from the tip (`1664143`, under `PK/refe/`), MEASURED by reading source; the test `SourceKeyingTests` regex-pins these claims in the shipped source and has a red arm

| input | key | `file:line` |
|---|---|---|
| bank rows rank-0 / rank-1 | `row["log_name"]` + `row["token"]`; scenes grouped by `(log_name, token, step)` | `train.py:664-665` (rows read), `train.py:770` (grouping), `train.py:672` (TargetBank refuses stray `targets_*.jsonl`), `build_metric_cache.py:49-62` (`keys_of_bank`: `by.setdefault(r["log_name"], ...)`) |
| **twins (rank-1) -> parent's log** | built from the **same scenario object** as the parent, so `log_name`/`token` ARE the parent's | `build_teacher_rollouts.py:338` (`make_row(sc, ...)`), `:399-402` (`"token": sc.scenario_name`, `"log_name": sc.log_name`); `augment_search.py:239` (`row = BR.make_row(sc, ..., 1)`), `:253` (`row["aug"] = {...}` added on that row); `slow_twins.py:153-155` (`tw = {k: v for k, v in row.items() ...}`: a copy of the parent row) |
| D9 metric cache | `<root>/<log>/<token>[__lane_rank_K].pkl.xz` (twins live in the parent's log dir) | `online_labeller.py:47` (`twin_suffix`), `:55-56` (`parts_path`), `build_metric_cache.py:49-62` |
| aux label store | `<store>/shards/<log>.aux1` (magic `AUXSDF1\n`, u32 LE header length, header JSON with `log` and `entries[].t`) and manifest `files["shards/<log>.aux1"]` | `aux_label_store.py:14` (key = (log_name, token)), `:23` (layout), `:226` (`write_shard`), `:623-634` (`finalize`) |
| image zips / frames | `<root>/<log>_<CAM>.zip`, loose `<root>/**/<log>/<CAM>/<hash>.jpg` | `train.py:71` (`container_set`), `:125`, `:165` (`f"{log_name}_{c}.zip"`), `build_targets.py:95`, `:134` |
| calibration rows | `table["logs"][<log>]` | `calib_table.py:79-81` (`vectors`), `train.py:743` |
| trainer identity digest | basename + bytes of `targets_rank*.jsonl` (so a filtered bank is a different, honest bank) | `train.py:1026` (`_digest_files`) |

Twin attribution, shown from source: a twin row is produced from the same `sc` as its parent (`make_row` reads `sc.scenario_name` and `sc.log_name`), or is a copy of the parent row, so **a twin carries its parent's log**. `filter_bank` therefore attributes it by its own `log_name`, and (with the token table) additionally proves `log_name == log_of(token)` for every row, rank-0 and rank-1 (`FilterTests.test_token_table_cross_check_catches_a_twin_with_the_wrong_parent_log`).

## 3. The P7-only hook, and why P6 is untouched

Hook: environment variable **`P7_RESERVE=<path to reserve_sessions.json>`** (plus optional `P7_LABEL_STORE`, `P7_TOKEN_TABLE`). All in `code/refe/`:

| `file:line` | what |
|---|---|
| `p6_pod.sh:336-346` | header comment of section 5a; `:346` `if [ -n "${P7_RESERVE:-}" ]; then` ... `:397` `fi`: the whole block is dormant when unset |
| `p6_pod.sh:347-349` | refuses when the module is not shipped, the file is missing, or `reserve-verify` refuses (one byte changed, wrong schema) |
| `p6_pod.sh:352-357` | rebinds `CBANK` to `<bank>_p7` and `BANK` to `$P6/p7_reserve/calib`, so the **ARGS line text is unchanged** and names the filtered copies; a `CBANK` already ending `_p7` is verified, not re-filtered (mode B) |
| `p6_pod.sh:358-359` | **the guard**: an indented `resume_bank_check` on the REBOUND bank before anything moves (a P6 run directory started on another bank refuses, untouched) |
| `p6_pod.sh:360-366` | `p7_audit()` over bank, D9 cache, image root, calibration table, label store |
| `p6_pod.sh:370` | P6-chain-lock guard: a P7 copy refuses to move the shared cache / pixels while `$DATA/refe_p6` holds its `chain.lock` |
| `p6_pod.sh:373-390` | `reserve-filter` (bank, rank-0 AND twins), `reserve-calib`, `reserve-move` (cache, images; reversible), `reserve-labelstore` + `aux_label_store.py finalize` on both stores |
| `p6_pod.sh:392-396` | `invalidate cache lanes lab mem egr launch_gate`, `key_mark reserve`, final `p7_audit`, `stop_after reserve` |
| `p6_pod.sh:482` | **7a**: a final audit one-liner immediately before ARGS, guarded by `P7_RESERVE` |
| `p6_prep.py:288-325` | `reserve_cmd` (imports `session_reserve` lazily: the P6 chain never loads it); subcommands registered `:367-399`; each prints one `ZZRESERVE_*_OK` / `_BAD` marker (audit `_LEAK`) |
| `test_p6_chain.py:566-568` | `"P7_"` in `SANDBOX_DROP_PREFIXES` |

Default-path identity (MEASURED, `ChainBehaviourTests.test_default_path_is_identical_to_the_tips_chain_when_p7_reserve_is_unset`): the REAL `p6_pod.sh` of this deliverable and the REAL `p6_pod.sh` of the tip, both run in the suite's sandbox with `P7_RESERVE` unset, give the **same call log, the same chain log (dd timing lines filtered) and the same data tree**, to `ZZP6_DONE`. The tip's own `check_chain` reports 0 violations and all 79 of its mutations are still caught with block 5a in place (`ChainStaticTests`). `git diff`-style stats: `p6_pod.sh` +66/-0, `p6_prep.py` +98/-0: **nothing existing was removed or rewritten**; no change to `train.py`, `model.py`, `launch_gate.py`.

P7 chain shape (not P6): a P7 chain must be a COPY (`p7_pod.sh`) with its own `$P6`, `$RUN`, `$REPO`, and a `refe_p7` gate profile. The `refe_p6` profile will FAIL G-CLOCK for the `train_grow_cmd_3c_p7` bank until its tuple count is registered (by design: an unregistered bank must not launch).

## 4. Deliberate-regression arms (each goes RED when the guarded behaviour is removed; the mutation run is in the verification block)

planted leaked bank row (`AuditTests.test_planted_leaked_bank_row_rank0`), planted leaked twin (`test_planted_leaked_twin_rank1`), planted D9 cache key (`test_planted_d9_cache_key`), logless key (`test_logless_keys_are_unresolved_never_clean`, `FilterTests.test_a_row_with_no_resolvable_log_is_refused_and_nothing_is_written`), a reserve file with one byte changed (`ReserveFileTests.test_one_byte_changed_refuses`, and in the chain `ChainBehaviourTests.test_refusals_leave_every_input_untouched`), plus label-store shard + stale manifest, image zip / frame dir / refs, calibration row, empty input, capped lists, a leak planted after the fact in the real chain (refused before ARGS).
`raw/mutation_check.py` applies 13 one-line mutations to a COPY of the tree (each asserts its target text occurs exactly once, so "not applied" can never read as "stayed green"); **M1 deletes the single exclusion line (`session_reserve.py:257`) and runs the FULL suite**; M2..M13 cover twins-not-excluded, `session_of` accepting any string, the sha pin unchecked, audit not recording a leak / unresolved / empty input, the move not moving, `p6_prep` audit always OK, the resume guard deleted, the final audit deleted, block 5a unguarded, and the filter result unchecked.

## 5. Derived counts

MEASURED now, from `raw/src/navtrain_token_table.tsv.gz` (the module's own `count_from_token_table` and `filter_bank_dir` over a one-row-per-attributed-token rank-0 pseudo-bank with real tokens and real logs; `raw/derive_counts.py` -> `raw/derive_counts.out`, `raw/derived_counts.json`, `raw/rank0_filter_report.json`):

| quantity | value | class |
|---|---|---|
| rank-0 scenes in / out / excluded | **103,037 / 92,896 / 10,141**; 16 sessions hit; unresolved 0; token cross-check true | MEASURED (reproduces the pin 10,141; `RealDataTests`) |
| P7 steps | 25 x ceil(92,896 / 256) = **9,075** (P6: 10,075; saves 1,000) | arithmetic on MEASURED counts |
| rank-1 exclusion | **PENDING** (needs the 3-C bank) | PENDING |
| rank-1 if twins fall uniformly over scenes | ~6,325 of 64,263 excluded, ~57,938 remain, total ~150,834 tuples | ESTIMATED |
| 3-C bank size | 167,300 = 103,037 + 64,263 | INHERITED (`launch_gate.P6_BANK_TUPLES` and its comment; ruled authoritative by MM_RULINGS conflict (i)) |

**PENDING step (not run: the brief forbids fetching the bank):** `raw/measure_rank1_3c.sh`. Precondition: `bank3c_hf.sh` printed `ZZB3_OK` (HF dataset `Sayood/tanitad-refe-navtrain-banks` holds `refe_train_grow_cmd_3c.tar.zst`). CPU only, ~1 min after the download. Run:
```
HF_TOKEN_FILE=<path to Keys.txt> PKG=<tree holding refe/ and raw/2026-10-22-p7-session-reserve/> bash measure_rank1_3c.sh
```
The token is read in place by the python child, never echoed, never an argument; `HF_HUB_DISABLE_XET=1`. It verifies remote size + LFS sha256 against the downloaded bytes, extracts, checks the rank-file sha256 prefixes `daa23b96` / `2a63acc2` and 167,300 rows (INHERITED expectations from PREREG_3C.addendum6: if they differ it is another bank, STOP), then runs `session_reserve.py filter ... --token-table ...` and prints the report. Success reads: `n_in 167300`, `by_rank_in {"0": 103037, "1": 64263}`, `by_rank_excluded {"0": 10141, "1": <THE RANK-1 NUMBER>}`, `token_log_checked true`, `n_out = 92896 + (64263 - rank-1 excluded)`. Pin that `n_out` as the tuple count of `train_grow_cmd_3c_p7` and the report's `targets_sha256` as its data pin.

## 6. Doc fix, proposed (not applied)

`raw/SPEC_P6_bank_figure.patch`: SPEC_P6's "189,888" is the pre-3C bank's figure; the built bank is **167,300** (`launch_gate.P6_BANK_TUPLES["train_grow_cmd_3c"]`). Edits SPEC_P6 lines 63, 112, 149, 256 (`patch -p1 < raw/SPEC_P6_bank_figure.patch` from the repo root; `--dry-run` applied cleanly to the tip's file, MEASURED). 167,300 is labelled INHERITED / ruled authoritative; 189,888 is labelled MEASURED (the P2 final run's `bank` events) and pre-3C.

## 7. VERIFICATION (generated by `finalize_verification.py` from the raw outputs; do not hand-edit between the markers)

<!-- VERIFICATION:BEGIN -->
**Verification verdict (generated by finalize_verification.py): NOT GREEN: suite test_p6_chain: baseline vs after differ or one is missing**

### 1. test_session_reserve.py, full run (CUDA_VISIBLE_DEVICES=-1, driverl-eval venv, P7WIRE_BASE_DIR = the pristine tip's refe/)

```
Ran 43 tests in 330.436s
OK
ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the real rank-0 shape', 'sandbox chain (P6 default path)', 'sandbox chain (P7 planted leaks)', 'sandbox chain (P7 refusals)', 'sandbox chain (P7 reserve stage)', 'source keying claims', 'source keying claims (red arm)', 'test_p6_chain static checks', 'tip p6_pod.sh comparison', 'token-table rank-0 count']; ABSENT: [])
rc=0
```

### 2. raw/mutation_check.out (verbatim)

```
CONTROL (unmutated)
  ChainStaticTests: rc=0  ZZSESSRES_PASS 3  (optional checks run: ['test_p6_chain static checks']; ABSENT: [])
  FAST/UnitTests: rc=0  ZZSESSRES_PASS 3  (optional checks run: []; ABSENT: [])
  FAST/ReserveFileTests: rc=0  ZZSESSRES_PASS 4  (optional checks run: []; ABSENT: [])
  FAST/FilterTests: rc=0  ZZSESSRES_PASS 7  (optional checks run: []; ABSENT: [])
  FAST/AuditTests: rc=0  ZZSESSRES_PASS 11  (optional checks run: []; ABSENT: [])
  FAST/MoveTests: rc=0  ZZSESSRES_PASS 4  (optional checks run: ['aux_label_store.finalize']; ABSENT: [])
  FAST/RealDataTests: rc=0  ZZSESSRES_PASS 3  (optional checks run: ['filter over the real rank-0 shape', 'token-table rank-0 count']; ABSENT: [])
  FAST/PrepCliTests: rc=0  ZZSESSRES_PASS 2  (optional checks run: []; ABSENT: [])
  FAST/ChainStaticTests: rc=0  ZZSESSRES_PASS 3  (optional checks run: ['test_p6_chain static checks']; ABSENT: [])
  FULL: rc=0  ZZSESSRES_PASS 43  (optional checks run: ['aux_label_store.finalize', 'filter over the real rank-0 shape', 'sandbox chain (P6 default path)', 'sandbox chain (P7 planted leaks)', 'sandbox chain (P7 refusals)', 'sandbox chain (P7 reserve stage)', 'source keying claims', 'source keying claims (red arm)', 'test_p6_chain static checks', 'tip p6_pod.sh comparison', 'token-table rank-0 count']; ABSENT: [])
  test_a_leak_planted: rc=0  ZZSESSRES_PASS 1  (optional checks run: ['sandbox chain (P7 planted leaks)']; ABSENT: [])
  test_refusals_leave_every_input_untouched: rc=0  ZZSESSRES_PASS 1  (optional checks run: ['sandbox chain (P7 refusals)']; ABSENT: [])
MUTATIONS (each must be RED)
M1  RED  [FULL]  filter_bank: the single exclusion line deleted (a reserved row is written to the output)
      8 failing test(s): test_a_leak_planted_after_the_fact_is_refused_before_ARGS, test_every_subcommand_prints_one_marker_and_the_right_rc, test_filter_bank_dir_report_and_the_trainers_identity_digest, test_p7_reserve_leaves_every_input_and_the_arguments_name_the_copies, test_rank0_and_twins_are_excluded_by_their_own_log, test_refusals_leave_every_input_untouched, test_survivors_are_byte_for_byte, test_the_filter_on_a_real_shaped_rank0_bank_excludes_exactly_10141
M2  RED  [FAST]  filter_bank: twins (rank >= 1) are not excluded
      5 failing test(s): test_every_subcommand_prints_one_marker_and_the_right_rc, test_filter_bank_dir_report_and_the_trainers_identity_digest, test_rank0_and_twins_are_excluded_by_their_own_log, test_the_filter_on_a_real_shaped_rank0_bank_excludes_exactly_10141, test_token_table_cross_check_catches_a_twin_with_the_wrong_parent_log
M3  RED  [FAST]  session_of: accepts any string (the bare rsplit; 'L0' becomes a clean session)
      7 failing test(s): test_a_row_with_no_resolvable_log_is_refused_and_nothing_is_written, test_assert_rows_clear_is_the_consumer_side_refusal, test_every_subcommand_prints_one_marker_and_the_right_rc, test_filter_calib, test_is_reserved_needs_a_resolvable_log, test_logless_keys_are_unresolved_never_clean, test_session_of_and_its_refusals
M4  RED  [FAST]  load_reserve: the sha256 pin is not checked
      2 failing test(s): test_every_subcommand_prints_one_marker_and_the_right_rc, test_one_byte_changed_refuses
M5  RED  [FAST]  audit_inputs: a bank leak is never recorded
      12 failing test(s): test_every_subcommand_prints_one_marker_and_the_right_rc, test_leak_lists_are_capped_but_counted, test_move_cache_and_images_then_restore_exactly, test_planted_calibration_row, test_planted_d9_cache_key, test_planted_image_zip_frame_dir_and_references, test_planted_label_store_shard_and_stale_manifest, test_planted_leaked_bank_row_rank0, test_planted_leaked_twin_rank1, test_split_label_store_then_refinalise, test_the_filter_on_a_real_shaped_rank0_bank_excludes_exactly_10141, test_the_unfiltered_input_set_is_flagged_everywhere
M6  RED  [FAST]  audit_inputs: an unresolved key is not counted
      1 failing test(s): test_logless_keys_are_unresolved_never_clean
M7  RED  [FAST]  audit_inputs: an input that examined zero keys counts as clean
      1 failing test(s): test_an_empty_or_missing_input_is_not_a_clean_input
M8  RED  [FAST]  move_reserved: the reserve is not moved (left in the shared root)
      2 failing test(s): test_every_subcommand_prints_one_marker_and_the_right_rc, test_move_cache_and_images_then_restore_exactly
M9  RED  [FAST]  p6_prep reserve-audit: always prints OK
      1 failing test(s): test_every_subcommand_prints_one_marker_and_the_right_rc
M10  RED  [test_refusals_leave_every_input_untouched]  p6_pod.sh: the resume-bank guard before any move is deleted (a P6 run directory is no longer protected)
      1 failing test(s): test_refusals_leave_every_input_untouched
M11  RED  [test_a_leak_planted]  p6_pod.sh: the final audit before ARGS is deleted
      1 failing test(s): test_a_leak_planted_after_the_fact_is_refused_before_ARGS
M12  RED  [ChainStaticTests]  p6_pod.sh: the reserve block is not guarded by P7_RESERVE (P6's default path would run it)
      1 failing test(s): test_static_properties_and_their_mutations
M13  RED  [test_refusals_leave_every_input_untouched]  p6_pod.sh: the bank filter result is not checked (the chain would go on after a refusal)
      1 failing test(s): test_refusals_leave_every_input_untouched
MUTATION_CHECK_OK
```

### 3. the existing refe suites: pristine tip 1664143 (baseline) vs the tree with this deliverable (after)

| suite | baseline rc | baseline marker | after rc | after marker | same |
|---|---|---|---|---|---|
| test_launch_gate | 0 | ZZLG_STATIC_PASS 1168 | 0 | ZZLG_STATIC_PASS 1168 | yes |
| test_p6_levers | 0 | ZZP6_PASS 165 | 0 | ZZP6_PASS 165 | yes |
| test_p6_overlap | 0 | ZZTEST_PASS 54 | 0 | ZZTEST_PASS 54 | yes |
| test_p6_cpu_probe | 0 | ZZTEST_PASS 111 | 0 | ZZTEST_PASS 111 | yes |
| test_p6_planner_command | 0 | ZZP6PLAN_PASS 172 | 0 | ZZP6PLAN_PASS 172 | yes |
| test_flock_rw | 0 | ZZTEST_PASS 59 | 0 | ZZTEST_PASS 59 | yes |
| test_p6_chain | 0 | 2026-10-22T19:35:50Z ZZEGR_FAIL main_ok=False regression_refused=True ; ZZP6CHAIN_PASS 432 | 0 | 2026-10-22T19:34:35Z ZZEGR_FAIL main_ok=False regression_refused=True ; ZZP6CHAIN_PASS 432 | NO |

baseline finished: True; after finished: True

<!-- VERIFICATION:END -->

Measured before the final runs (per-class, `--only`, CPU), already green on the assembled tree: UnitTests 3, ReserveFileTests 4, FilterTests 7, AuditTests 11, MoveTests 4, RealDataTests 3, SourceKeyingTests 2, PrepCliTests 2, ChainStaticTests 3, and all 4 ChainBehaviourTests (43 in all). Baseline on the pristine tip (MEASURED, `raw/baseline_*.out`): `test_p6_chain.py` -> `ZZP6CHAIN_PASS 432` (11m16s idle); `test_launch_gate.py --static --no-banked` -> `ZZLG_STATIC_PASS 1168` on both the tip and the assembled tree (`raw/baseline_test_launch_gate.out`, `raw/after_test_launch_gate.out`).

## 8. NOT covered / caveats (say it plainly)

- **nuPlan DBs are not moved.** They are read only by (log, token) of the filtered bank rows, so a reserve scene is never reached; but the audit does not scan them. INHERITED reasoning, not separately tested.
- **`commands.json`** holds reserve keys but is not read by the trainer (source-read; not wired).
- **Downstream arms** (RL, scorer fine-tunes, ensemble members, hard-negative banks) are NOT wired; `filter_bank` / `audit_inputs` are generic for them.
- **The trainer-level `--reserve-manifest` refusal** belongs to `train.py` (other agent). `assert_rows_clear` is offered.
- **Aux label store:** reserve shards move only when `P7_LABEL_STORE` names the store; then both stores are re-finalised. The store holds 103,139 frames vs the bank's 103,037 (INHERITED from SESSION_RESERVE.md): exclusion is by session, never by bank row, for that reason.
- **Commands stage:** a P7 chain with `P6_CBANK` set needs `stage/commands.ok` pre-marked (the 5a block runs after the commands stage).
- **Shared roots:** the D9 cache and image roots are shared with the P6 chain; P7 MOVES the reserve out (reversible: `p6_prep.py reserve-move --restore`). Never run P7's block from the same `$DATA` while P6 trains; the lock guard enforces it only for a P7 copy whose `$P6` differs from `$DATA/refe_p6`.
- **Not run on the pod**, no GPU, no real 3-C bank, no real cache / zips / label store: the chain behaviour is proven on the suite's sandbox (`fakepy` stub), the counts on the token table.
- **CRLF finding:** on this Windows box `git archive` exported CRLF files; the reserve file then hashes `4f5b1f1d...` and refuses. Landers and pod checkouts must keep LF (a `.gitattributes` `eol=lf` on `fixtures/p7_reserve_sessions.json` is advisable; not added by me).
- Single seed does not apply (no training here); no intervals are quoted.

## 9. Deliverable manifest

| artifact | where | single location? |
|---|---|---|
| `code/refe/session_reserve.py`, `test_session_reserve.py`, `fixtures/p7_reserve_sessions.json`, `p6_prep.py`, `p6_pod.sh`, `test_p6_chain.py` | worktree-free local dir `C:/Users/Admin/par18/p7wire_reserve/code/refe/` | **YES: only here** (not in any repo) |
| `RESULT.md`, `HANDOFF.md`, `finalize_verification.py` | `C:/Users/Admin/par18/p7wire_reserve/` | **YES** |
| `raw/derive_counts.py`, `derived_counts.json`, `rank0_filter_report.json`, `derive_counts.out` | `.../raw/` | **YES** |
| `raw/measure_rank1_3c.sh` (PENDING step) | `.../raw/` | **YES** |
| `raw/SPEC_P6_bank_figure.patch` | `.../raw/` | **YES** |
| `raw/mutation_check.py`, `mutation_check.out`, `test_run.out`, `baseline_*.out`, `after_*.out`, `VERIFICATION.md` | `.../raw/` | **YES** |
| `base/`, `base_full/` (pristine LF export of the tip), `work/` (assembled tree), `assemble.sh`, `run_suites.sh`, `patch*.py` | `.../` | scratch; reproducible from the tip + `code/` |
| the reserve itself `reserve_sessions.json` | repo: `PK/raw/2026-10-22-p7-session-reserve/raw/` at tip `1664143` (sha256 `53f3bcd4...`) | in the repo |
