# LICENCE GUARD: `navpipe_chain.py --mode navtest|c2` now requires the C1 licence, not an EP-ADMISSION record (2026-10-20 ~03:55)

Ruling: `MM_RULING_C2_LICENCE.md`, sha256 `fa7a26046a2fb5f8d8e69b2644c97a6cfce8907201d40d5d3c5be5358560ba6a`. Draft amendment: `AMENDMENT_C2_DRAFT.md` (same dir). Nothing was run on navtest, a GPU or a pod; the guard was exercised on planted data and on the real banked carve-B arrays.

## What the pipeline required before (quoted from the navpipe build, `navpipe/code/`)
- `navpipe_core.py:40` `REQUIRED_PINS` carried `admission_record_sha256`.
- `navpipe_core.py:152-153` `check_c2_seal(amendment, amendment_sha256, admission_record, ...)`; `:172` `G-SEAL-ARGS` also demanded `--admission-record`.
- `navpipe_core.py:208-217`: `G-SEAL-ADMISSION-EXISTS` (file exists), `G-SEAL-ADMISSION-SHA` (sha == the amendment's pin), `G-SEAL-ADMIT` (`verdict == "ADMIT"` and `rules == "r1"`); the field names were an UNVERIFIED placeholder (`DESIGN.md` section 5).
- `navpipe_chain.py:152` called it; `:154` printed `admission <verdict>`; `:674` ended with `ZZNAVPIPE_ADMISSION_NOT_RUN`; `:755` `--admission-record`; `:778` passed it into `Cfg`.
So a C2 run could not start without an EP-ADMISSION ADMIT, which `D-REFE-EPADM-N1-1` showed cannot be earned by a search candidate at any available geometry.

## What it requires now (`c2amend/code/`; same file names)
- `navpipe_core.py:40-41` `REQUIRED_PINS`: `admission_record_sha256` replaced by `c1_prereg_sha256` and `licence_ruling_sha256`.
- `navpipe_core.py:152` `check_c2_seal(amendment, amendment_sha256, c1_result, c1_result_sha256, ...)`; `:156` `check_s0_seal` unchanged in behaviour (no licence).
- `navpipe_core.py:164-175` the ruling's constants: outcomes `O1_CONFIRMED` / `O2_DIRECTIONAL`, safety minimum `-0.25`, moved-share minimum `0.05`, C1 = 905 tokens / 24 logs, the six C1 stages.
- `navpipe_core.py:177-231` `check_c1_licence`: 12 guards `G-LICENCE-{ARGS,EXISTS,SHA,PIN,IS-C1,PREREG,STAGES,OUTCOME,SAFETY-READABLE,SAFETY,MOVED-READABLE,MOVED}`. `:233-` `_check_seal` keeps its guards (`G-SEAL-ARGS` now covers the amendment arguments only; the licence owns its own arguments so no guard shadows another) and adds `G-SEAL-COST-STATED` (`:277`, previously an unmarked and unmutated check) and `G-SEAL-LICENCE-RULING` (`:279`).
- `navpipe_chain.py:87` `Cfg(c1_result=, c1_result_sha256=)`; `:152` the seal-plus-licence call (still guard `C-SEAL-FIRST`, still before any data is opened); `:156` `ZZNAVPIPE_LICENCE_OK <outcome> safety <json> moved <x> c1_result <sha12>`; `:183` banks `licence_c1.json`; `:677` ends with `ZZNAVPIPE_LICENCE_BASIS`; `:758-759` `--c1-result`, `--c1-result-sha256`; `--admission-record` no longer exists (an old command line fails loudly with argparse exit 2).
- The S0-bank and fixture modes are untouched except that `_check_seal` and `Cfg` were generalised (`need_admission` -> `need_licence`, `licence: None` for S0).

## Operator use
`python navpipe_chain.py --mode navtest --amendment <sealed.md> --amendment-sha256 <sha> --c1-result <dir>/RESULT_SPC.json --c1-result-sha256 <sha typed from the register> [--table-ref ...] [--dry-run]`. The `chain_state.json` that the C1 chain wrote must sit beside the result file (copy both, or point at the C1 work dir `D:/Projects/TanitAD-artifacts/spc/`).

## Residual limits (stated, not hidden)
- `RESULT_SPC.json` carries no prereg or config field of its own. The binding to C1's sealed prereg and to a finished C1 chain comes from the sibling `chain_state.json` (written by `spc_chain.py`, source-checked by `test_c1_stages_are_the_c1_chains_own_analysis_preconditions`). Someone who edits both files and types the new sha defeats it; so does anyone who can write the register. It guards against the wrong-but-valid file (carve B, the SEARCH-DOSE arm) and against accident, not against a forger.
- The fixture precondition and the latency status are checked by the operator and the pin, not by this guard (amendment DQ7, DQ3).
- The new `DESIGN.md` (this dir `docs/`) differs from navpipe's only in the four admission statements and a header note.

## Two other changes the verification forced (outside the licence, both in the navpipe package, both tested)
- **`navpipe_core.py` `load_banked_seam` held about 14 GB** (MEASURED 2026-10-20: one test reached a 13.5 GB working set and starved the box's commit memory down to 51 MB free): `{t: z["poses"][i] ...}` re-reads the whole `poses` member of an `NpzFile` on EVERY access and each row view kept its copy alive, 12,146 x 1.17 MB. The chain caches that dict for the whole run, so a real C2 would have held it for the 10-22 h. Fixed (members read once; `S0FanRef` had the same shape and got the same fix). New test `test_loading_the_banked_seam_holds_one_copy_...` (the old line holds 206.8 MiB for a 0.14 MiB array and fails it; the fix passes). The suite fell from 42 s to 15 s. Whether this contributed to the fixture's `RAM_LOW` stop is UNVERIFIED (the fixture does not appear to load the seam).
- **`G-DELETE-STATUS` survived navpipe's own mutation pass** (`G-DELETE-VERIFY` refused the same defect first). New test `test_delete_ctx_refuses_a_chunk_that_is_not_yet_banked_even_when_every_artifact_verifies` isolates it.
- `mutation_check.py` now prints `ZZMUT_START <guard>`, the first failing test per mutant, and reports a suite that died of the BOX (ENOSPC, MemoryError, access denied) as `INFRA_ERROR`, never as CAUGHT; `TESTS` gained `test_navpipe_licence.py` (run first, so licence mutants fail fast).

## Test and mutation results
Full suite (navpipe_licence + navpipe + navpipe_s0 + navpipe_guards): `201 passed in 15.22s` (was 119 passed before the change). Mutation check (`mutation_check.py`, every `@GUARD` line disabled once in a scratch copy, suite must go RED, a catch caused by the box (ENOSPC etc.) is reported INFRA_ERROR, never CAUGHT): **60 of 60 judged guards caught**, survivors / infra errors: none, untestable without the GPU: ['MB-IDENT-REF']; the licence and seal guards (24): 24 caught. The unmutated control read green in every one of the 1 batches. Log with the first failing test per mutant: `raw/mutation_check.log`; merged: `raw/mutations_merged.json`.
