> **UNEXPECTED -- read before trusting this file:**
> * C2: base 39/39 pass; named: not run (300 s) mine 42/43 pass, 1 FAIL; named: not run (201 s) (expected mine = base + 4, all pass)

# RESULT -- EPR1UB-GATE-CACHE (BACKLOG; `D-REFE-EPR1UB-1`), 2026-10-19

Agent: Lab (Sonnet). Tip read: `a61c8d1` (`git fetch` at start; re-checked unchanged). Evidence class of every number below: **MEASURED** (ours; logs in `raw/logs/`), unless stated. CPU only (`CUDA_VISIBLE_DEVICES=-1`), one python at a time, no GPU / lock / pod touched.

## Headline -- integration the Master Mind must do

1. **Land 4 edited blobs + this package directory** (paths in `out/paths.json`, before/after blobs in `EDITED_BLOBS.md`):
   * **asked for:** `epr1ub_gate.py`, `test_epr1ub_gate.py` (`2026-10-19-ep-r1ub/code/`).
   * **adjacent finding of the audit (separable -- drop these two rows if you do not want to touch the VOID C2 line):** `srttc_c2_gate.py`, `test_srttc_c2_gate.py` (`2026-10-18-sel-rank/code/`).
2. **`epr1_gate.py` is NOT edited**, on purpose: it is not affected (section 5) and its bytes are pinned by `PARENT_GATE_SHA256` (`epr1ub_gate.py:47`) and by the landed blob that `G-COMMIT` checks.
3. **The old pins describe the BEFORE bytes and are deliberately left as written** (they record what ran): `CODE_SHA256_EPR1UB.json`, `CODE_SHA256_C2.json`, `raw/f10/epr1ub_gate_token.json`, `raw/r1/f10/c2_gate_token.json`. They no longer match the repo files, so **both gates need a fresh mint before any reuse** (EP-R1ub: `verify_token` re-hashes `BOUND_CODE_FILES`, which includes `epr1ub_gate.py`, `epr1ub_gate.py:111,725`; C2: `BOUND_FILES` includes `srttc_c2_gate.py`, `srttc_c2_gate.py:72-73`). The after-fix hashes are in `CODE_SHA256_GATEFIX.json`. Neither line is being reused (EP-R1 closed at the fit; SR-TTC VOID), so nothing is blocked today.
4. **Register text (not edited by me)**: in `D-REFE-EPR1UB-1` replace the "Disclosed defect ... Fix before any reuse (BACKLOG)" sentence by the one in section 9; close BACKLOG `EPR1UB-GATE-CACHE`.

## 1. The defect, reproduced on the tip (MEASURED)

`epr1ub_gate.load_verified_ub` (tip `epr1ub_gate.py:205-223`) cached the loaded module in `_MODULES` under `epr1ub__gate_<sha256(epr1ub.py)[:12]>` and returned it on a hit **before** `epr1ub.py` was exec'd. What one call certifies is `epr1ub.py` **and** the `epr1.py` beside it, and `epr1.py` is checked only inside `epr1ub._load_parent` (`epr1ub.py:37-54`) while that module body runs. So after any clean load, a code_dir with a byte-identical `epr1ub.py` and a tampered `epr1.py` was handed the cached module and the parent check never ran. The same hole opens when `epr1.py` is edited in the **same** dir after a clean load.

Why the old test hid it: `test_a_tampered_epr1_is_refused_when_the_ub_module_is_loaded` never made the clean load the bug needs; it passed only if no earlier test had loaded `epr1ub.py` through `load_verified_ub`. unittest runs classes in name order (`TestCommit` first: cold, passes); pytest runs them in definition order (`TestDVB` first: warm, **fails**). Baseline on the tip, same 17 tests: unittest 17/17 OK; pytest 16 pass + that one FAIL (16/17 pass, 1 FAIL; named: FAIL (52 s)).

## 2. Fix: NO CACHE (`epr1ub_gate.py`, `load_verified_ub`)

The brief allowed "key on (ub sha, every verified parent sha, code_dir)" or "no cache". I chose **no cache**:
* A key must name every input of the verification (ub bytes, each file `_load_parent` reads, the dir). The bug IS an omitted key input, and the next input added to `_load_parent` would be omitted again. With no cache there is no key to get wrong.
* It is cheap: a repeat load is **~12 ms** (median of 20, range 9.9-13.7 ms; the first load of a process is 7.5 s, which is the torch import and is not repeated -- `D:/eu1/time_load.py`, driverl-eval python 3.11.13, CPU).
* Nothing in the repo depends on module identity across calls: callers are `run_checks` (`epr1ub_gate.py:611`, one call per process) and the test file (grep of the tip for `load_verified_ub` / `_MODULES` / `epr1ub__gate_`: those only).
* Each load registers its own module name (`epr1ub__gate_<sha12>_<n>`; dataclasses with `from __future__ import annotations` look the module up in `sys.modules` while the class bodies run), the name is checked free (a test mutant of the gate may hold the same sha), and **a load that raises removes its module** (the tip left the half-built module registered).

Residual, unchanged and documented: `epr1ub._load_parent` still allows ONE `epr1` per process (`epr1ub.py:45-49`), so a CLEAN second code_dir in the same process is refused with "a different 'epr1' module is already imported", exactly as before. The new arm uses a TAMPERED second dir and asserts the refusal reason (the sha256 text and that dir's name), so it cannot pass on that unrelated error.

## 3. RED arms (all in `test_epr1ub_gate.py`)

The arm is `refuses_a_tampered_parent_after_a_clean_load(Gm)`: (1) a CLEAN load of the real code dir through `Gm` (it warms whatever cache `Gm` has, so the arm never depends on another test), then (2) a load from a second dir with a byte-identical `epr1ub.py` + `epr1_gate.py` and a tampered `epr1.py`, which must raise `RuntimeError` naming the sha256 refusal and that dir; `AssertionError` if accepted.

| arm | on the FIXED gate | on the mutant |
|---|---|---|
| second-dir arm (`test_second_dir_arm_passes_on_the_fixed_loader_and_goes_red_on_the_sha_only_cache`) | passes | the loader put back **verbatim** (`PRE_FIX_LOADER`, the tip text) is ACCEPTED -> the arm raises `AssertionError` ("ACCEPTED") |
| cold + warm (`test_the_arm_does_not_depend_on_what_ran_before_it`) | passes from a never-used loader and after 3 other loads | the old loader is caught even from a cold start |
| same dir, fresh interpreter (`test_a_parent_tampered_after_a_clean_load_in_the_same_dir_is_refused_in_a_fresh_interpreter`) | `REFUSED` | old loader returns `ACCEPTED` |
| a refused load leaves no module (`test_a_refused_load_leaves_no_module_behind`) | 1 module gained (the clean one) | with the `sys.modules.pop` line deleted: 2 (half-built one stays) |
| fresh module per call (`test_every_call_reverifies_and_returns_a_fresh_module`) | 2 loads, 2 objects, one shared verified `E` | -- |
| decoy `epr1_gate` (`test_a_decoy_epr1_gate_already_in_sys_modules_is_refused`, section 6) | `REFUSED` | with the tip's `return cur` put back: `ACCEPTED` |

**External mutation, the literal reading of the brief.** `D:/eu1/mut_ub` = my fixed gate with ONLY a cache keyed on `sha256(epr1ub.py)` put back (`_SHA_CACHE`, 3 lines) and the full new test file. Result: 23 tests, **6 FAIL** (`raw/logs/mut_unittest.log`): `test_a_parent_tampered_after_a_clean_load_in_the_same_dir_is_refused_in_a_fresh_interpreter`, `test_a_refused_load_leaves_no_module_behind`, `test_a_tampered_epr1_is_refused_when_the_ub_module_is_loaded`, `test_every_call_reverifies_and_returns_a_fresh_module`, `test_second_dir_arm_passes_on_the_fixed_loader_and_goes_red_on_the_sha_only_cache`, `test_the_arm_does_not_depend_on_what_ran_before_it`; the decoy arm still passes (that fix is intact), as do all other tests.

And the new test file against the **unmodified tip** gate (`D:/eu1/mixed_ub` = tip `epr1ub_gate.py` + the new test file, `raw/logs/tipgate_newtests_unittest.log`): 23 tests, **7 FAIL**: `test_a_decoy_epr1_gate_already_in_sys_modules_is_refused`, `test_a_parent_tampered_after_a_clean_load_in_the_same_dir_is_refused_in_a_fresh_interpreter`, `test_a_refused_load_leaves_no_module_behind`, `test_a_tampered_epr1_is_refused_when_the_ub_module_is_loaded`, `test_every_call_reverifies_and_returns_a_fresh_module`, `test_second_dir_arm_passes_on_the_fixed_loader_and_goes_red_on_the_sha_only_cache`, `test_the_arm_does_not_depend_on_what_ran_before_it`. The named test reports "a TAMPERED epr1.py in a second code_dir was ACCEPTED after a clean load" even though unittest runs it first.

## 4. The named test is order-independent (MEASURED)

It now makes its own clean load, so the state the old cache needed is present in every order. Orders run, on the tip with the OLD tests ("base") and on mine; counts are passed/failed of the whole file; `named` is the outcome of `test_a_tampered_epr1_is_refused_when_the_ub_module_is_loaded`:

| runner | order | base (tip, old tests: 17) | mine (23 tests) |
|---|---|---|---|
| unittest | default (class order by name) | 17/17 pass; named: ok (41 s) | 23/23 pass; named: ok (62 s) |
| unittest | named test FIRST | 17/17 pass; named: ok (99 s) | 23/23 pass; named: ok (88 s) |
| unittest | named test LAST | 16/17 pass, 1 FAIL; named: FAIL (108 s) | 23/23 pass; named: ok (73 s) |
| pytest | default (definition order) | 16/17 pass, 1 FAIL; named: FAIL (52 s) | 23/23 pass; named: ok (70 s) |
| pytest | named test FIRST | 17/17 pass; named: ok (60 s) | 23/23 pass; named: ok (102 s) |
| pytest | named test LAST | 16/17 pass, 1 FAIL; named: FAIL (63 s) | 23/23 pass; named: ok (68 s) |
| pytest | whole file REVERSED | 16/17 pass, 1 FAIL; named: FAIL (59 s) | 23/23 pass; named: ok (71 s) |

Notes: pytest here is 9.1.1 appended to `sys.path` after the venv (`code/run_pytest.py`; pytest is NOT installed in the gate's interpreter, and no random-order plugin is, so `-p no:randomly` is moot and explicit node-id orders were used instead: **first** = the named test then the rest, **last** = the rest then the named test, **reverse** = the whole file reversed). unittest orders are explicit `Class.method` lists.

## 5. Audit of the same pattern

**`epr1_gate.load_verified_module` (`epr1_gate.py:283-300`): NOT affected.** Key = `epr1__gate_<sha256(epr1.py)[:12]>` and that file is everything the module execs: its module-level imports are stdlib + numpy + torch (`epr1.py:27-40`; the audit lists sibling files imported at module level: `[]`), and `__file__` is read only at run time (`epr1.py:258,979`). Measured (`raw/audit_loaders_TIP.json`): tampered bytes in a second dir get their own module and sha (cache miss), are refused when the verified sha is passed, identical bytes return the cached module (sound; its `__file__` stays the first dir, a provenance quirk, not a bypass). Not edited.

**`srttc_c2_gate` -- the named pattern is NOT present, but a sibling defect IS, and I fixed it.**
* Not present: `_load_frozen_gate` (`:96-107`), `exec_verified_module` (`:115-129`) key their `sys.modules` lookups on the sha256 of the only file they exec (those files import no sibling: audit lists `[]`), and `load_verified_pair` (`:132-156`) keys the C2 module on **both** shas (`:141`) and binds `srttc_c2.py`'s `import srttc_r1 as R` to the verified module for the duration of the exec (`:147-155`). Measured: a tampered `srttc_r1.py` or `srttc_c2.py` in a second dir gets its own module(s); identical bytes are a hit; both are refused when the verified sha is passed.
* **Present (sibling):** all three loaders register the module in `sys.modules` BEFORE exec (`:105,127,146`) and look it up there (`:101,123,142`), but never remove it if the exec raises. So after one failed load the next call with the same bytes gets the half-built module back **as a cache hit, with no error** -- MEASURED on the tip for all three (`raw/audit_loaders_TIP.json`: first call `RuntimeError`, second call `RETURNED tuple`). Fix: `try/except BaseException: sys.modules.pop(name, None); raise` in each (marked `# poison guard: <function>`). `raw/audit_loaders_MINE.json`: second call now `RAISED RuntimeError`. RED arms: `TestAFailedLoadIsNotServedLater` (4 tests; each loader has a failing-load arm paired with a mutant that turns its `pop` line into `pass`, which must serve the stale module; a control that a clean load is still a hit). Against the tip gate the 3 failing-load arms FAIL ("a FAILED load was served from sys.modules"), the control passes.
* Exposure is low (a gate run retries only in-process after an exec failure); it is fixed because the cold path would have refused and the hot path did not, which is the shape of the named bug.

`srttc_r1_gate.load_verified_module` (`srttc_r1_gate.py:276-294`) has the same `_MODULES` shape as `epr1_gate`'s; by reading, same verdict (its file imports no sibling; the audit confirms `srttc_r1_gate.py` / `srttc_r1.py` have none). Not run, not edited.

## 6. A second hole found in the same file: `_load_landed_gate` (fixed)

`epr1ub_gate._load_landed_gate` (tip `:51-65`) hashes `HERE/epr1_gate.py` against `PARENT_GATE_SHA256`, then `return cur` for **any** module already registered as `epr1_gate` (tip `:60`) -- so a decoy or tampered `epr1_gate` (its HMAC primitives sign the token) imported earlier in the process was never looked at. Now it is reused only if its `__file__` is the verified path, else `RuntimeError` (the same identity test `epr1ub._load_parent` uses, `epr1ub.py:46-48`). Arm: a fresh interpreter registers a tampered copy as `sys.modules['epr1_gate']` and imports the gate: fixed -> `REFUSED`; tip's `return cur` -> `ACCEPTED`. Strength is the same as `_load_parent`'s: it removes the accidental case (a stale or other-dir module); an in-process adversary can forge `__file__`.

## 7. Test counts, base (tip) vs mine -- full form, every file that names an edited file

One python at a time, CPU only, `-I`, deploy-layout flat trees on D: (`D:/eu1/base_ub`, `mine_ub`, `base_c2`, `mine_c2`: landed parents byte-exact from `1afc54f6` blobs -- `epr1.py` sha256 `a751c63e...`, `epr1_gate.py` `5bfe2cf9...` as pinned -- plus the tip's staged files; the `sel-rank/code` dir flat for C2). Gate tests use the mirror's `.git` read-only.

| file | runner | base (tip) | mine |
|---|---|---|---|
| `test_epr1ub_gate.py` | unittest | 17/17 pass; named: ok (41 s) | 23/23 pass; named: ok (62 s) |
| `test_epr1ub_gate.py` | pytest 9.1.1 | 16/17 pass, 1 FAIL; named: FAIL (52 s) | 23/23 pass; named: ok (70 s) |
| `test_srttc_c2_gate.py` | unittest | 39/39 pass (300 s) | 42/43 pass (201 s) |

Files that name an edited file: `test_epr1ub_gate.py` (names `epr1ub_gate`), `test_srttc_c2_gate.py` (names `srttc_c2_gate`). `test_epr1ub.py`, `test_epr1.py`, `test_epr1_gate.py` name none of them (grep: 0 hits) and were not re-run.

## 8. Not done / limits

* No re-mint (that needs the landed commit and the PI-side decision to reuse a line; neither line is being reused).
* The pytest order result uses pytest 9.1.1 from another venv's pure-python files on `sys.path` (torch/numpy from the gate's venv); no `pytest-randomly` run exists (not installed). Explicit orders were used.
* One training-free scope: all of this is gate code; no model number is touched.
* The `CODE_SHA256_*.json` pins and the two tokens were not rewritten (section Headline 3).

## 9. Proposed register text

`D-REFE-EPR1UB-1`, replace the disclosed-defect sentence: "**Gate defect FIXED (2026-10-19, `raw/2026-10-19-epr1ub-gate-cache/`)**: `load_verified_ub` no longer caches (a repeat load is ~12 ms); a tampered parent in a second dir or after a clean load is refused in every test order (23 tests, unittest and pytest, 7 of 7 orders green), RED when the sha-only cache is put back. Audit: `epr1_gate` not affected; `srttc_c2_gate` had a sibling defect (a failed load was served from `sys.modules` on the next call), fixed. The EP-R1ub and SR-TTC C2 gate tokens describe the pre-fix bytes: re-mint before any reuse." BACKLOG `EPR1UB-GATE-CACHE`: DONE (same pointer).

## 10. Deliverable manifest

All paths below are relative to the deliverable root `C:/Users/Admin/par18/epr1ubfix/out/` (the repo paths are the same strings under it; `out/paths.json` maps each to its absolute source).

| artifact | repo path (under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/`) | state |
|---|---|---|
| EDITED gate / test | `2026-10-19-ep-r1ub/code/epr1ub_gate.py` | edited; tip blob and new blob in `EDITED_BLOBS.md`; ONE place only |
| EDITED gate / test | `2026-10-19-ep-r1ub/code/test_epr1ub_gate.py` | edited; tip blob and new blob in `EDITED_BLOBS.md`; ONE place only |
| EDITED gate / test | `2026-10-18-sel-rank/code/srttc_c2_gate.py` | edited; tip blob and new blob in `EDITED_BLOBS.md`; ONE place only |
| EDITED gate / test | `2026-10-18-sel-rank/code/test_srttc_c2_gate.py` | edited; tip blob and new blob in `EDITED_BLOBS.md`; ONE place only |
| this report | `2026-10-19-epr1ub-gate-cache/RESULT.md` | new; ONE place only |
| blob table, after-fix pins | `2026-10-19-epr1ub-gate-cache/EDITED_BLOBS.md`, `2026-10-19-epr1ub-gate-cache/CODE_SHA256_GATEFIX.json` (+ `EDITED_BLOBS.md` at the deliverable root) | new; ONE place only |
| instruments | `2026-10-19-epr1ub-gate-cache/code/{audit_loaders.py, time_load.py, run_pytest.py, build_trees.sh}` | new; ONE place only |
| loader audit, tip vs mine | `2026-10-19-epr1ub-gate-cache/raw/audit_loaders_{TIP,MINE}.json` | new; ONE place only |
| every test log of section 4 / 7 | `2026-10-19-epr1ub-gate-cache/raw/logs/*.log` (18 files) | new; originals also in `D:/eu1/logs/` |
| the D: work trees (`D:/eu1/base_ub`, `mine_ub`, `mut_ub`, `mixed_ub`, `base_c2`, `mine_c2`, `pytest_site`) and `D:/eu1/logs/` | `D:/eu1/` | scratch, reproducible with `code/build_trees.sh`; logs banked under `raw/logs/`; safe to delete |
| editing scripts | `C:/Users/Admin/par18/epr1ubfix/{patch_gate,patch_test,patch_c2,audit_loaders,finalize}.py`, `src/*.tip` (tip blobs) | scratch |
