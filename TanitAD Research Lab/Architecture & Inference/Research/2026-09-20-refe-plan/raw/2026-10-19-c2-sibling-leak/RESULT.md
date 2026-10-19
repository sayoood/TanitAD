# RESULT -- C2-GATE-SIBLING-LEAK (BACKLOG; fix before any reuse of the VOID C2 chain), 2026-10-19

Agent: Lab (Sonnet). Tip read: `d2f4a062783f3fd9757ef10cc3266a13b42d8815` (`git fetch` of `agent/arch-inf-20260803` at the start; the two edited files are byte-equal to the tip blobs I started from; re-checked later: the tip moved to `5944fc6a` and the whole `2026-10-18-sel-rank/code` tree hash is unchanged, `cdb9cdf1c350dd3e07d029b33251ec8619dd3d7c`). Evidence class of every number: **MEASURED** (ours; logs in `raw/`, `raw/logs/`). CPU only (`CUDA_VISIBLE_DEVICES=-1`), one python at a time for the test matrix, nothing written to D:, no GPU / lock / pod / other agent's directory touched. **The C2 chain itself was NOT re-run** (it is VOID); this change only makes the gate reusable.

## Headline

**Poison arm GREEN on the fix / RED on the tip gate; deleted-code_dir arm GREEN / RED; mutation: each of the 3 restored lookups turns its arm RED (table in section 4).  `test_srttc_c2_gate.py`: tip 39 tests, fix 45 tests (+6 new); passed/ran per runner and order -- tip: unittest default: 39/39; unittest first: 39/39; unittest last: 39/39; pytest default: 39/39; pytest first: 39/39; pytest last: 39/39; pytest reversed: 39/39 | fix: unittest default: 45/45; unittest first: 45/45; unittest last: 45/45; pytest default: 45/45; pytest first: 45/45; pytest last: 45/45; pytest reversed: 45/45.**

**Integration the Master Mind must do (escalated here, not buried):**
1. **Land 2 edited blobs + this package directory** (`out/paths.json`; before / after blobs in `EDITED_BLOBS.md`):
* `srttc_c2_gate.py`: tip blob `86abd952b71949718580ad48950c45096d90aa20` -> new `2d16c74b2dbed2cbc54d5ce077114c9e993a8b1f` (sha256 `43ec5fd2362c9acb40a042be7c591d63e273f69f5942450e6f5d4c26c49c5f99`)
* `test_srttc_c2_gate.py`: tip blob `cc3d822a55f0b8ba1f3295e8f2a37c00a3128677` -> new `b2298021118284a85577014108e8b81f08e20739` (sha256 `074fa12403201b98059afd3374ffb76926a1d369ae4f4f7edc5daa6a3f3a94dd`)
   Only `srttc_c2_gate.py` and its test changed. `srttc_c2.py`, `srttc_r1.py`, `srttc_r1_gate.py`, `srttc_c2_run.py`, `test_srttc_r1.py`, `test_srttc_r1_gate.py` are byte-identical to the tip (sha256 compared by `finalize.py`).
2. **Re-mint the C2 gate before any reuse.** `CODE_SHA256_C2.json` and `raw/r1/f10/c2_gate_token.json` describe the pre-fix bytes and are left as written (they record what ran); the C2 token binds `srttc_c2_gate.py` (`BOUND_FILES = LANDED_FILES + NEW_FILES`, `srttc_c2_gate.py:72-73`), so `verify_token` refuses it now. The C2 line is VOID, nothing is blocked today. The "after" hashes of the earlier NOT-landed attempt in `2026-10-19-epr1ub-gate-cache/` (`c2_not_landed/`, `CODE_SHA256_GATEFIX.json`) describe other bytes; this fix supersedes that attempt (it includes its poison guard).
3. **Register text (not edited by me)**: BACKLOG `C2-GATE-SIBLING-LEAK` -> DONE, pointer to this directory; the SR-TTC C2 gate row: "sibling leak fixed 2026-10-19; token must be re-minted before reuse".

## 1. The defect, reproduced on the tip (MEASURED)

`srttc_c2_gate.load_verified_pair` / `exec_verified_module` / `_load_frozen_gate` registered the loaded module in `sys.modules` under a name that carried **only the sha256 of the bytes** and returned it to any later caller with the same bytes. That is sound for the bytes and wrong for everything else a module is: its `__file__` and its state belong to the **first** load. `srttc_c2.run_c2` re-reads `R.__file__` from disk at run time (`srttc_c2.py:110`; `srttc_r1.py:258,858` do the same for their own file), so a second code dir with identical bytes ran the first dir's module, and once the first dir was gone `run_c2` raised `FileNotFoundError`. That is exactly the 42 / 43 failure of the earlier attempt: its new poison test loaded the real bytes from a temp dir that was later deleted, and a later real-run test was handed that module (unittest runs classes by name, so the poison class ran before the real-run class).

```
A: R.__file__ in A: True | sys.path grew: 1 | sys.path[0] = C:\Users\Admin\par18\c2leak\work\c2leak_repro_g0zn6byj\A
B (identical bytes, A deleted): same srttc_r1 object as A: True | same c2 object: True
B: R.__file__ = A\srttc_r1.py | exists: False
B: c2mod.R is rB: True
B: reading R.__file__ (as run_c2 does at srttc_c2.py:110) -> FileNotFoundError
```

Three smaller leaks of the same class, all on the tip: (i) the sibling import `import srttc_r1 as R` (`srttc_c2.py:37`) was answered by a **process-global** `sys.modules['srttc_r1']` that `load_verified_pair` swapped in for the duration of the exec and swapped back (`:147-157`); (ii) `srttc_c2.py:35-36` does `sys.path.insert(0, HERE)` with HERE taken from the verified path, and **nothing removed it** (`sys.path grew: 1` above); (iii) a load whose exec raised stayed registered, so the next call with the same bytes got the half-built module back with no error (`raw/audit_loaders_BASE.json`: second call `RETURNED tuple` for all three loaders).
The earlier audit did see that `srttc_c2.py` imports `srttc_r1` (`raw/audit_loaders_BASE.json`: `sibling_files_imported_at_module_level`); what it did not check was the `__file__` / run-time re-reads of a cache hit.

## 2. Every module load in the C2 chain, on the tip (file:line)

| # | site (tip) | mechanism | status after the fix |
|---|---|---|---|
| 1 | `srttc_c2_run.py:21-23` | `sys.path.insert(0, HERE)`; `import srttc_c2_gate as CG` (registered once per process under its plain name) | unchanged by design: one entry = one code dir = one process |
| 2 | `srttc_c2_gate.py:96-109` `_load_frozen_gate` (called at `:112`) | exec + `sys.modules["srttc_r1_gate__c2dep_<sha12>"]`, hit at `:103-104` | fresh module per load |
| 3 | `srttc_c2_gate.py:117-131` `exec_verified_module` | exec + `sys.modules["<name>__c2g_<sha12>"]`, hit at `:125-126` | fresh module per load |
| 4 | `srttc_c2_gate.py:134-158` `load_verified_pair` | hit at `:144-145` returns BOTH cached modules; `:149-157` swaps `sys.modules["srttc_r1"]` around the exec | fresh pair; sibling import bound per load; nothing global touched |
| 5 | `srttc_c2.py:35-37` | `sys.path.insert(0, HERE)`; `import srttc_r1 as R` | the insert is undone after every exec; the import is answered by THIS load's module |
| 6 | `srttc_c2.py:110`, `srttc_r1.py:258,858` | `__file__` re-read at run time | `__file__` is now always the verified dir's file (see limit L1) |
| 7 | `srttc_r1_gate.py:276-294` `load_verified_module` | `_MODULES` + `sys.modules[name]` | NOT called from the C2 chain (`grep load_verified_module srttc_c2*.py` = 0 hits); landed and frozen; not edited (see L3) |
| 8 | tests: `test_srttc_c2_gate.py:38,70,725-726`; `test_srttc_r1.py:18,30,152-155` | `sys.path.insert`, plain `import srttc_r1`, mutant `exec` / `spec_from_file_location` under unique names | test side only; the new tests restore `sys.modules['srttc_r1']` and `sys.path` |

`importlib` outside `test_srttc_r1.py`, `runpy`, `__import__`: none in the chain; `exec(` is only the four gate sites above. Full grep: `raw/loader_trace_BASE.txt`.

## 3. The fix (`srttc_c2_gate.py`; pattern of `epr1ub_gate.load_verified_ub`: no cache, a fresh module per call)

* `_exec_fresh(stem, path, src, bind=None)` is the single exec site. It gives the module a **unique name** `<stem>_n<k>` (nothing is looked up, nothing is served), sets `__file__` to the verified path, execs the **same bytes that were hashed**, removes the module again **if the exec raises**, and **restores `sys.path`** afterwards.
* `bind={"srttc_r1": <this load's verified module>}`: the exec'd `srttc_c2` module gets its own `__builtins__` whose `__import__` answers `import srttc_r1` with exactly that module (a dotted import of it is refused). `sys.modules['srttc_r1']` and `sys.path` are not consulted for it and not touched. `check_dvb`'s existing `c2mod.R is mod` check therefore holds by construction, and still runs.
* `_load_frozen_gate`, `exec_verified_module`, `load_verified_pair` are thin wrappers over it; a pair that fails removes both halves. The sha256 checks (`expect_sha256`, the `GateError` on mismatch) are unchanged and still happen on the bytes that are then exec'd (read once, hashed, exec'd).
* Cost: a repeat `load_verified_pair` is ~10 ms (first 2247.5 ms; next 15: median 10.1 ms, min 8.7 ms, max 12.7 ms) against ~0.1 ms for the old cache hit (first 1799.1 ms; next 15: median 0.1 ms, min 0.1 ms, max 0.3 ms); the first load of a process is the torch import.
* Residual by design: every successful load keeps its (uniquely named) module in `sys.modules` (dataclasses with `from __future__ import annotations` look their module up there); a gate run makes a few dozen.

## 4. Arms and mutants (`TestEachLoadUsesExactlyTheVerifiedFiles`, 6 tests; every arm asserts a SENTINEL the module returns)

* **(a) poison arm.** Code dirs A, B (a different `srttc_r1`: `SENTINEL = "B"`, `fit_row` returns `('fit_row from', 'B')`) and A2 (the same bytes as A, another dir) are loaded in a process whose `sys.modules['srttc_r1']` already holds an unrelated decoy. Each pair must satisfy `c2.R is r1`, `r1.SENTINEL == tag`, and `srttc_c2.fit_c2_rows` (the module's OWN function, whose default fit is `R.fit_row`) must return that dir's sentinel; A2 must not be A's module; the decoy and `sys.path` must be untouched.
* **(b) deleted-dir arm.** A is loaded, A gets state (`LEFT_BY_A`), A's dir is deleted, B (identical bytes) is loaded: it must not raise, must not carry `LEFT_BY_A`, and `B.R.__file__` must be readable and hash to the verified sha256 (what `run_c2` does at `:110`).
* **failed-load arm** (the earlier attempt's guard, adapted): for each loader a raising exec leaves no module, restores `sys.path`, does not touch `sys.modules['srttc_r1']`, and raises again on the next call.
* **(c) mutation.** Three one-line mutants of the NEW gate, each restoring one process-global lookup: (1) `name = stem; if name in sys.modules: return sys.modules[name]` (the sha-keyed lookup of the old loaders), (2) the sibling binding removed (`import srttc_r1` answered by `sys.modules` / `sys.path` again), (3) the `sys.modules.pop` poison guard removed. The test `test_restoring_each_process_global_lookup_turns_the_arms_red` requires GREEN on the fix and an `AssertionError` on each mutant. Measured (`raw/arm_table_MINE.md`):

| gate | poison (A, B, A2 + a global decoy) | deleted dir (A gone, B identical) | failed load leaves nothing |
|---|---|---|---|
| gate in mine | GREEN | GREEN | GREEN |
| sha-keyed sys.modules lookup (old loaders) | RED | RED | GREEN |
| sibling import via sys.modules / sys.path (old global binding) | RED | RED | GREEN |
| failed load stays registered (no poison guard) | GREEN | GREEN | RED |

- sha-keyed sys.modules lookup (old loaders) / poison (A, B, A2 + a global decoy): unexpectedly identical: <module 'srttc_r1__c2g_73df054d5d71' from 'C:\\Users\\Admin\\AppData\\Local\\Temp\\c2g_sibling_utwdjppz\\pa_A_2064a9340f55\\srttc_r1.py'
- sha-keyed sys.modules lookup (old loaders) / deleted dir (A gone, B identical): True is not false : B was handed A's module object (its state is visible)
- sibling import via sys.modules / sys.path (old global binding) / poison (A, B, A2 + a global decoy): <module 'srttc_r1'> is not <module 'srttc_r1__c2g_7f471c2621a5_n3' from 'C:\\Users\\Admin\\AppData\\Local\\Temp\\c2g_sibling_utwdjppz\\pa_A_e40277f69449\\srttc_
- sibling import via sys.modules / sys.path (old global binding) / deleted dir (A gone, B identical): 'b841347d90c298ec597f10afbbd284d68258df55c98cbac8cbd9a0a444cc5d0b' != 'f6a94984d72c1892ca979d3364f2ca625bc4a326d7f4dc4191ca96a0a0198e7a'
- failed load stays registered (no poison guard) / failed load leaves nothing: Items in the first set but not the second:

* **The same arms on the unfixed tip gate** (`raw/arm_table_BASEGATE.md`; the tip gate under the new test file):

| gate | poison (A, B, A2 + a global decoy) | deleted dir (A gone, B identical) | failed load leaves nothing |
|---|---|---|---|
| gate in mixed | RED | RED | RED |

- gate in mixed / poison (A, B, A2 + a global decoy): unexpectedly identical: <module 'srttc_r1__c2g_b7464f950bc0' from 'C:\\Users\\Admin\\AppData\\Local\\Temp\\c2g_sibling_bbehjb4e\\pa_A_2b0724edb79a\\srttc_r1.py'
- gate in mixed / deleted dir (A gone, B identical): True is not false : B was handed A's module object (its state is visible)
- gate in mixed / failed load leaves nothing: RuntimeError not raised

Also: `test_the_sibling_import_is_answered_by_the_verified_module_only` (the bound `__import__` answers only the verified module, refuses a submodule, leaves `builtins.__import__` untouched) and `test_every_load_is_a_module_of_its_own_with_its_own_file`.

## 5. Test counts, tip (base) vs fix (mine) -- every runner and order

One python at a time, `-B -I`, CPU only, flat code dirs (`base/` = the 36 files of `2026-10-18-sel-rank/code` at the tip, `mine/` = the same with the 2 edited files). pytest 9.1.1 is APPENDED to `sys.path` from a copy of the previous agent's `pytest_site` (it is not installed in the gate's interpreter; no random-order plugin, so explicit node-id orders). Orders (`code/run_order.py`): **default** = the runner's own (unittest: class name then method; pytest: definition order); **first** = the poisoners first, **last** = the poisoners last, **reversed** = the whole file reversed; poisoners = the 6 new tests (mine only) + `test_a_valid_token_writes_the_c2_rows_result_and_attestation_bound_to_the_r1_run` (the real-run test the earlier attempt broke). Counts are passed/ran of the whole file.

| test file | runner | order | BASE (tip files) | MINE (the fix) |
|---|---|---|---|---|
| `test_srttc_c2_gate.py` | unittest | default | 39/39 pass; real-run test: ok | 45/45 pass; real-run test: ok |
| `test_srttc_c2_gate.py` | unittest | first | 39/39 pass; real-run test: ok | 45/45 pass; real-run test: ok |
| `test_srttc_c2_gate.py` | unittest | last | 39/39 pass; real-run test: ok | 45/45 pass; real-run test: ok |
| `test_srttc_c2_gate.py` | pytest | default | 39/39 pass; real-run test: ok | 45/45 pass; real-run test: ok |
| `test_srttc_c2_gate.py` | pytest | first | 39/39 pass; real-run test: ok | 45/45 pass; real-run test: ok |
| `test_srttc_c2_gate.py` | pytest | last | 39/39 pass; real-run test: ok | 45/45 pass; real-run test: ok |
| `test_srttc_c2_gate.py` | pytest | reversed | 39/39 pass; real-run test: ok | 45/45 pass; real-run test: ok |
| `test_srttc_r1.py` | unittest | default | 35/35 pass | 35/35 pass |
| `test_srttc_r1.py` | pytest | default | 35/35 pass | 35/35 pass |

Reproduction of the earlier attempt's failure (the NOT-landed gate AND tests of `c2_not_landed/` on the tip's loaders, unittest default, `nlbase/`): **42/43 pass, 1 FAIL; not ok: ['test_srttc_c2_gate.TestEntry.test_a_valid_token_writes_the_c2_rows_result_and_attestation_bound_to_the_r1_run']** (`raw/logs/nlbase_unittest_test_srttc_c2_gate_default.log`).

Cross-check: the earlier NOT-landed attempt's test file (`c2_not_landed/test_srttc_c2_gate.py`) against THIS gate, unittest default (`nl/`): **39/43 pass, 4 FAIL; not ok: ['test_srttc_c2_gate.TestAFailedLoadIsNotServedLater.test_a_clean_load_is_still_a_cache_hit_for_identical_bytes', 'test_srttc_c2_gate.TestAFailedLoadIsNotServedLater.test_exec_verified_module_does_not_serve_a_failed_load', 'test_srttc_c2_gate.TestAFailedLoadIsNotServedLater.test_load_frozen_gate_does_not_serve_a_failed_load', 'test_srttc_c2_gate.TestAFailedLoadIsNotServedLater.test_load_verified_pair_does_not_serve_a_failed_c2_load']**. Its 39 original tests plus the real-run test no longer fail with `FileNotFoundError`; the remaining failures of that file, if any, are its own tests that assert the OPPOSITE design (`test_a_clean_load_is_still_a_cache_hit_for_identical_bytes` asserts `assertIs(m1, m2)`) or that anchor on its `poison guard:` comments, which this fix does not carry (see `raw/logs/nl_unittest_test_srttc_c2_gate_default.log`). Those tests are superseded by `TestEachLoadUsesExactlyTheVerifiedFiles`.

## 6. Limits and what I did not change (stated plainly)

* **L1 (UNVERIFIED scope, pre-existing).** `srttc_c2.py:110` and `srttc_r1.py:258,858` re-read their own file from disk at run time. The gate hashes and execs the same bytes, but if a file changed on disk between that check and the end of the fit, the sha256 the result records would describe the new bytes. The landed files are frozen, so I did not edit them; `verify_token` re-hashes the deployed code at the start of every run. Not measured here.
* **L2.** The process-wide `builtins.__import__` is not patched; only the exec'd module's own `__builtins__` is a copy (it sees builtins as of its load).
* **L3.** `srttc_r1_gate.load_verified_module` (`:276-294`) has the same `_MODULES` shape and `__file__` quirk. It is not reachable from the C2 chain, it is landed and frozen, and I did not edit it. If the R1 gate is ever run twice in one process from two dirs, it has the same exposure; flagged for the Master Mind, not fixed.
* **L4.** `srttc_c2_gate` itself is imported once per process under its plain name by `srttc_c2_run.py`; the entry is one code dir = one process (the in-process tests are the only place two dirs share a process).
* The earlier audit text said the identical-bytes hit is a "provenance quirk, not a bypass"; the measurement above shows that for the C2 pair it is a live failure (`FileNotFoundError`), not only a quirk.
* No re-mint (it needs the landed commit and a decision to reuse the line); no model number touched; no GPU.

## 7. Deliverable manifest

All paths are relative to `C:/Users/Admin/par18/c2leak/out/` (the repo paths are the same strings under it; `out/paths.json` maps each to its absolute source). Repo prefix: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/`.

| artifact | repo path | state |
|---|---|---|
| EDITED gate | `2026-10-18-sel-rank/code/srttc_c2_gate.py` | edited (tip blob and new blob in `EDITED_BLOBS.md`); ONE place only |
| EDITED tests | `2026-10-18-sel-rank/code/test_srttc_c2_gate.py` | edited (+6 tests); ONE place only |
| this report, diff vs tip, tip sha, blob table, after-fix pins | `2026-10-19-c2-sibling-leak/{RESULT.md, c2_gate_sibling_fix.diff, TIP_SHA.txt, EDITED_BLOBS.md, CODE_SHA256_SIBLINGFIX.json}` | new; ONE place only |
| instruments | `2026-10-19-c2-sibling-leak/code/{run_order.py, run_matrix.py, finalize.py, RESULT_TEMPLATE.md, arm_table.py, repro_leak.py, time_load.py}` (+ `audit_loaders.py` is the unchanged one in `2026-10-19-epr1ub-gate-cache/code/`) | new; ONE place only |
| loader audit, tip vs fix; repro; arm tables; trace; timing; matrix results | `2026-10-19-c2-sibling-leak/raw/{audit_loaders_BASE.json, audit_loaders_MINE.json, repro_BASE.txt, repro_MINE.txt, arm_table_*.md, loader_trace_BASE.txt, time_load_*.txt, matrix_results.jsonl, ids_*.json}` | new; ONE place only |
| every test log of section 5 | `2026-10-19-c2-sibling-leak/raw/logs/*.log` | new; originals also in `C:/Users/Admin/par18/c2leak/evidence/logs/` |
| work trees (`base/`, `mine/`, `mixed/`, `nl/`, `pytest_site/`, `ep_deploy/`, `tip/`) | `C:/Users/Admin/par18/c2leak/` | scratch (C:), reproducible from the tip sha; safe to delete |
