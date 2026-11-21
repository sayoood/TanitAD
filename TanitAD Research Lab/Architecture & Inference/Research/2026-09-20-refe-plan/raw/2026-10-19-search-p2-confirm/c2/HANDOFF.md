# HANDOFF, C2 amendment draft + the navpipe licence guard (FINISHED 2026-10-20 ~04:30; package `C:/Users/Admin/par18/c2amend/`)

## State
DONE and staged in `out/` (manifest `out/paths.json`, summary `out/RESULT.md`). CPU only; nothing committed, pushed or `git add`ed; no navtest/B-REST/S0 label, score or search output read; no GPU, no pod.
1. `AMENDMENT_C2_DRAFT.md` (`Status: **DRAFT**`; the MM seals): object, T1 endpoint, estimator, DQ3 rule verbatim, four families, licence check, Thor cost plan (UNMEASURED + protocol), fixture precondition, cost, gates + RED arms, outcomes + next arms, **DQ1..DQ11 (recommendations, none chosen)**, PINS block (22 pins; parses with the chain's own parser; 9 left `@@SEAL@@`: 7 code shas, `banked_seam_sha256`, `banked_csv_sha256`).
2. Licence guard: `navpipe_chain.py --mode navtest|c2` takes `--c1-result` + `--c1-result-sha256`; `--admission-record` is gone. Record with file:line before/after: `LICENCE_GUARD.md`. Suite **201 passed** (was 119); mutation check **60 of 60 judged guards caught**, 1 untestable without the GPU (`MB-IDENT-REF`), none survived (`logs/mutation_final.log`, merged `logs/mutations_merged.json`).
3. Two defects found and fixed on the way (both tested): `load_banked_seam` held ~14 GB for a whole C2 run (suite 42 s -> 15 s); `G-DELETE-STATUS` survived navpipe's own mutation pass.
4. `code/conftest.py` is LOCAL ONLY (removes each test's tmp dir; C: has ~0.1 GB free). Not staged.

## Exact next step (Master Mind)
- Land navpipe's `out/` first, then this `out/` (it overwrites 6 code files + DESIGN.md of that package, adds `test_navpipe_licence.py`).
- Rule DQ1..DQ11. Then, once the code is final: fill the 9 `@@SEAL@@` pins (DQ9 offers a seal helper), set `Status: **SEALED`, write the sha256 sidecar, and `--dry-run` the chain.
- Re-run `finalize_c2amend.py` only if a file here changes (idempotent).

## Blockers named (not mine to clear)
- Fixture carve B has NOT read IDENTICAL (stopped RAM_LOW, 3 of 5 chunks): re-run on the landed code. The GPU stage has never run. D: ~0 free (C2 needs >= 9.64 GiB); dev-box GPU lock held by another battery.
- No Thor latency exists (UNMEASURED; needs a Thor slot, an agent to write `thor_search_latency.py`, a latency budget: DQ3, DQ4).
- Any pinned-code change after the seal makes the chain refuse (`SEAL_PIN_MISMATCH`): re-pin by a "Revision 1" with the terms byte-identical (DQ8).

## UPDATE 2026-10-20 (C2 FINAL; package `C:/Users/Admin/par18/c2final/`): this handoff's "next step" is superseded
- `MM_RULING_C2AMEND.md` (sha256 `fe4d9ce6…`) ruled DQ1..DQ11 and added the C8/O4 config choice and the one-run marker. Folded into `AMENDMENT_C2_DRAFT.md` (now **FINAL DRAFT**, section 14 RULED, 27 pins (12 filled at the seal) incl. `dose_chain_sha256`, `fixture_record_sha256`, `code:seal_c2_amendment.py`) and into the code (see `LICENCE_GUARD.md`, section "C2 FINAL"). `seal_c2_amendment.py` is the DQ9 seal tool; the Master Mind seals, nobody else.
- Remaining before the seal: land the dose package (`dose/code/dose_chain.py`), run the fixture on the landed code (it writes `FIXTURE_RECORD.json`; it must read IDENTICAL), then `seal_c2_amendment.py --check-only` and the real seal.
