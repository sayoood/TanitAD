# HNSCREEN-FP-REQUIRED -- the P6-HN screen's launch gate can no longer PASS on an unfilled `fingerprints_expected`

Architecture FlyWheel agent, 2026-10-19, 0 GPU, CPU only. Base: tip `9e8d11fb13a72dbd00d5ddb642b8a0ace63ae099` (`git ls-remote` at start). Evidence class of every number below: **MEASURED**, artifact under `raw/2026-10-19-hnscreen-fp/` (names in brackets).

**INTEGRATION NEEDED (headline):** (1) merge 3 small hunks of `refe/launch_gate.py` (below; R0E-B2B5 may be editing the same file for the R0e-pick profile -- all three hunks are inside `hnscreen_refusals`, none touches `PROFILES`); (2) land the new `refe/hn_fingerprints.py`; (3) land the `refe/test_launch_gate.py` edit; (4) **before any screen launch, FILL the profile** (procedure at the end) -- until then `refe_hnscreen` REFUSES by design.

## 1. The defect, reproduced (MM ruling 6)

`PROFILES["refe_hnscreen"]["fingerprints_expected"] == {}` and the check iterated `(prof.get("fingerprints_expected") or {}).items()`, with `continue` on a flag absent from the argv. On the ORIGINAL gate (mutant `mut_ab`, which is the original behaviour, [`out_mutation.log`]) the clean `S1` argv against `{}`, against the key absent, `None`, `[]`, and against a dict holding only `--hn-bank` (or only `--ctx-cache`) all returned **`refusals == []`**: no content check ran, the profile-level refusals were empty. With `--ctx-cache` (or `--hn-bank`) dropped from the argv, or carried twice, the argv was refused only by the older DROPPED / "carried 2 times" rules; the content rule itself was silent (it `continue`d).

## 2. The fix (`refe/launch_gate.py`, `hnscreen_refusals`; exact hunks in [`launch_gate.diff`])

* **(a)** under `hn_rules`, `fingerprints_expected` must be a dict holding BOTH `--hn-bank` and `--ctx-cache`; otherwise a refusal: `"<profile>: fingerprints_expected not filled: the content check cannot run (no expected fingerprint for <flags>); fill it from the bank/cache builder (hn_fingerprints.py), never by hand"`. Placed BEFORE the parser's early return, so it is always reported.
* **(b)** an expected flag absent from the argv, carried twice, or with other than one value is a refusal (was `continue`): `"<flag> is in fingerprints_expected but the argv does not carry it exactly once with one value (<n> found): the content check cannot run"`.
* `GT.fp_key(GT.fingerprint(...))` is untouched. A one-phrase docstring correction (the docstring said "when the profile carries"). The profile's shipped default stays `{}`: an unfilled profile can never PASS.
* Other profiles: none carries `fingerprints_expected` and none sets `hn_rules` (asserted in the test and by `grep` over `refe/`); their behaviour is untouched. The only other repo hits are banked historical artifacts under `raw/2026-10-18-p6hn-prereg/impl3/` (not edited).

Merge hunks (against `9e8d11f`; line numbers are the tip's):
```
@@ 569  docstring: "...must equal the registered ones when the profile carries `fingerprints_expected`;"
                 -> "...must equal the registered ones: `fingerprints_expected` MUST cover --hn-bank and --ctx-cache (an unfilled profile REFUSES, HNSCREEN-FP-REQUIRED);"
@@ after 592 (before `ns, err = C.parse_scorer_argv(...)`)  +5 lines: fp_exp / fp_unfilled / the "not filled" refusal
@@ 597-599  `for f, want in sorted((prof.get("fingerprints_expected") or {}).items()):` -> `sorted(fp_exp.items())`;  `continue` gains an `out.append(...)` before it
```

## 3. Tests (`refe/test_launch_gate.py`, `hnscreen_static`; [`test_launch_gate.diff`]) -- 31 new assertions

* The old clean arms relied on the empty default. They now run against `PF = {**P, "fingerprints_expected": <FP computed from the fixture's own bank and cache>}` (`refs()` defaults to `PF`; the argv-acting regression arms of H5 and the CLI arms use it too). The old `clean(a1,"S1",{... "--hn-bank": fk})` (bank only) is now correctly RED and is replaced.
* Deliberate-regression arms, each with its positive twin (`clean(a1,"S1",PF)`): HNFP-1 empty dict, 1b key absent, 1c `None`, 1d non-dict; HNFP-2 `--ctx-cache` key missing, 2b `--hn-bank` key missing; HNFP-3 expected `--ctx-cache` / `--hn-bank` absent from the argv, 3b flag carried twice, 3c one flag with two values (refused by the older "needs exactly one value" + the parser, because the trainer's parser rejects a 2nd value and the run ends before the fingerprint loop -- still a FAIL), 3d an expected flag no older rule polices (only rule (b) catches it); the shipped profile either is filled or refuses; CLI `refusals` on the shipped profile -> rc 5 with the message.
* **Mutation RED** [`out_mutation.log`, harness `mut_harness.py`, builder `make_mutants.py`]: the same test code with a non-aborting `ok`, run on the fixed tree and on three mutants of `launch_gate.py`:

| tree | what is reverted | assertions failing (of 278 in `hnscreen_static`) |
|---|---|---|
| fixed | -- | **0** |
| `mut_a` | rule (a): `if fp_unfilled:` -> `if False:` | **8**: HNFP-1, 1b, 1c, 1d, 2, 2b, the shipped-profile arm, the CLI rc-5 arm |
| `mut_b` | rule (b): the `out.append` before `continue` removed | **4**: HNFP-3 (both flags), 3b, 3d |
| `mut_ab` | both (== the original behaviour) | **12** = 8 + 4 |

  Restored: the staged `launch_gate.py` is the fixed file (sha256 in the manifest); the mutants live only in `par18/hnfp/mut_*` and are not staged.
* **`test_launch_gate.py`, FULL form (static + the real trainer on the rehearsal bank), CPU, `CUDA_VISIBLE_DEVICES=-1`:**

| | tree | result | log |
|---|---|---|---|
| before | the tip's bytes (blob hashes verified equal to git) | `ZZLG_FULL_PASS 971` (static 940) | `out_base_full.log`, `out_base_static.log` |
| after | fixed | `ZZLG_FULL_PASS 1002` (static 971) | `out_after_full.log`, `out_after_static.log` |

  1002 - 971 = 31 = the new assertions; 0 skips (`ZZTEST_SKIP` count 0 in both static logs), 0 failures.

## 4. `refe/hn_fingerprints.py` -- the fill helper (NEW)

`python refe/hn_fingerprints.py --hn-bank <bank file> --ctx-cache <cache dir> [--out fp.json]` prints `{"--hn-bank": "file:<sha256>", "--ctx-cache": "dir:<n>:<bytes>:<listing sha256>"}` computed by `gate_token.fingerprint` + `fp_key` (the gate's own functions; keys come from `launch_gate.HNSCREEN_FLAGS`). stdout is exactly the JSON (LF); provenance (kind, bytes, files, method) goes to stderr; a missing path, a zero-byte bank or an empty cache directory exits **2** with nothing on stdout (a vacuous fingerprint would match every other empty input). Tests (H3b, in `test_launch_gate.py`): the helper's output (as a process) fills a profile under which S0/S1/S2 are clean; a bank with **one bit flipped at the same size** -> FAIL (`content fingerprint of --hn-bank`); a cache file **one byte longer**, or **one file added** -> FAIL (`content fingerprint of --ctx-cache`); the four refusals; `--out` is LF.

**Caveat, MEASURED** [`out_probe_dir_fp_limit.log`, `probe_dir_fp_limit.py`]: the `--ctx-cache` fingerprint is a directory LISTING (relpath + size; contents not hashed, by `gate_token.fingerprint`'s documented design). A same-size byte flip INSIDE a cache payload file is **not detected** (`detected=False`); longer/shorter, added, removed and renamed files are. The bank (a file) is a full sha256: a same-size flip is detected. The cache's content is bound elsewhere in the gate (G-DVB `ctx_cache.manifest_sha256`, G15 against the live trunk). Not changed (the task said keep the computation unchanged); flagged for the Master Mind.

## 5. Sibling test files that mention `launch_gate` (21 files besides `test_launch_gate.py`)

Static finding: `refe_hnscreen` / `hn_rules` / `fingerprints_expected` are referenced by no test file except `test_launch_gate.py` (and `test_hnscreen_chain.py`, which drives a STUB gate by profile name), so no sibling can reach the changed code path. Measured confirmation is **RUNNING DETACHED at hand-off** (`par18/hnfp/suite_finish.py`, one child at a time): **PENDING -- see `HANDOFF.md`**. Baseline (original tree), finished before hand-off [`suite_base.json`]: `test_aux_g12_bf16` rc 0 (**1 SKIP: no CUDA, the bf16 G12 probe did NOT run; not a pass of that probe**), `test_aux_s1_wiring` rc 0 `ZZTEST_PASS`, `test_carve_feed_gate` rc 0 `ZZTEST_PASS`, `test_gate_gpu_probes` rc 0 (`SKIP_CUDA: the GPU check did NOT run`), `test_gate_suite_prerun` rc 0 `ZZTEST_PASS 144` (977 s).

## 6. Things the Master Mind must know

1. **Filling is a source edit**, by design (the token binds the commit): `PROFILES["refe_hnscreen"]["fingerprints_expected"]` is on `launch_gate.py:338` -- the line most likely to conflict with R0E-B2B5's profile edits. Suggested order: merge R0e's profile first, then fill.
2. **Until it is filled, the dev-box chain's gate step REFUSES** (`hnscreen_devbox.py` T<s> mints `refe_hnscreen`). That is the intended behaviour, not a regression; `test_hnscreen_chain.py` uses a stub gate and is unaffected.
3. **Fill procedure:** build bank + cache -> `python refe/hn_fingerprints.py --hn-bank <bank> --ctx-cache <cache>` -> paste the JSON into the profile (`{...,"fingerprints_expected": {"--hn-bank": "file:...", "--ctx-cache": "dir:..."}}`) -> commit -> gate. The values are path-independent (file sha256; directory listing by relative path), so they survive re-rooting to the run box.
4. A malformed fill (`""`, `None`, a wrong string) is fail-closed: it never equals a real `fp_key`, so it mismatches and refuses. A value `"missing:"` would equal a missing path's key, but the helper never emits it and a missing input is refused elsewhere (G-DVB "not readable").
5. Staged files are byte-exact LF; the tip's `refe/` was exported with `git cat-file --batch` (blob hashes of `launch_gate.py` and `test_launch_gate.py` equal git's).

## 7. Proposed register text (NOT applied -- register files untouched)

* **BACKLOG** `HNSCREEN-FP-REQUIRED` -> **DONE (pending landing)**: "`refe_hnscreen` refuses an unfilled / incomplete `fingerprints_expected` and an expected flag absent from the argv; 31 assertions, 12 of them RED on the original gate (`raw/2026-10-19-hnscreen-fp/out_mutation.log`). Remaining: `HNSCREEN-FP-FILL` -- fill the profile from `hn_fingerprints.py` once the bank and cache are built (precondition of the launch gate PASS)."
* **GAC** note for the P6-HN screen row: "launch-gate content check is now mandatory (MM ruling 6 closed in code, not yet filled); the `--ctx-cache` fingerprint is a size listing, not a content hash (MEASURED limit)."
* Class for the retraction/trap log: *a default that disables the check* -- the profile shipped a check whose input was empty; the test suite's own clean arm relied on that emptiness.

## Manifest

See `paths.json` (repo path, absolute source, "raw"). Single-copy status: everything is in `C:/Users/Admin/par18/hnfp/` (and the mutant/baseline trees there); nothing is on a pod; nothing is staged in git (no `git add`, no commit, no push).

## Integrity

Base tip at start `9e8d11f`; the branch moved to `72b2209` during the run; `refe/launch_gate.py` (blob `1121cc960f`) and `refe/test_launch_gate.py` (blob `c664142412`) are unchanged between the two, and `refe/hn_fingerprints.py` does not exist at either, so the staged files apply to the new tip as they are.

sha256 of the staged code (LF):

```
a40c46be4564b610372f16b94f2da54f1aa9d4931e2f03869a48eba55d7c3d2a *refe/hn_fingerprints.py
5a3b1a21bc2a902786acae181cffd3752e34be210fa0904080d4346436b18f2a *refe/launch_gate.py
e3676b9cc4fd84c722a3dd03fe2fb113ce489e634e0c074978d0e669b2d7b439 *refe/test_launch_gate.py
```
