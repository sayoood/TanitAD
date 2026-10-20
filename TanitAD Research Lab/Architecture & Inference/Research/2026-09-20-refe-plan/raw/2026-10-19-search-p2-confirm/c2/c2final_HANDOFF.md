# HANDOFF, C2 FINAL (code to seal-ready), written 2026-10-20 ~04:57 by the C2-final worker

## State
- All 8 brief deliverables are DONE on CPU, in `C:/Users/Admin/par18/c2final/repo/` (tip `6a0cfae`, `TIP.txt`; shell vars in `env.sh`). Nothing committed/pushed/`git add`ed; the seal tool was never run for real.
- Full suite **357 passed** (navpipe_licence + navpipe + navpipe_s0 + navpipe_guards + navpipe_select + seal_c2), baseline 201. Use ONLY `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (the other pythons have no pytest/numpy).
- **Mutation run is RUNNING detached** (started ~04:50, `cmd` PID 7020 via `run_mut.cmd`): about 125 guards at ~25 s each, expect ~55-70 min. Log `mut_out/mutation_check_c2final.log`, result `mut_out/mutations_c2final.json`, DONE MARKER `mut_out/mut.done` (contains `DONE <exitcode>`). Interim at 04:56: 17 caught, 0 survived. A probe of the 9 new override mutations + control already read 9/9 caught, control green.
- A provisional `out/` (19 files + `paths.json`) is staged; section 13 of the draft still reads `MUTATION_RESULT_PENDING`.

## Exact next step (mechanics, Haiku-level)
1. Do NOT edit anything under `repo/.../2026-10-19-search-navtest-pipe/code/` while the run is going (each mutant copies the live files). Wait for `mut_out/mut.done` (poll with a script, not turns).
2. Run `C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/par18/c2final/finalize_c2final.py`. It refuses (exit 2, prints why) on any SURVIVED / INFRA_ERROR / MUTATION_NOT_APPLICABLE guard or a red control; otherwise it fills section 13, copies the json+log into the package `raw/`, re-runs the suite, re-stages `out/` + `out/paths.json` and writes `out/RESULT.md`.
3. If it refuses on a survivor: add a RED test for that guard in `test_navpipe_select.py` / `test_seal_c2.py` (never weaken a guard), re-run `mutation_check.py --only <names>` (env as in `run_mut.cmd`), then re-run step 2 (it needs a full-run json: merge by hand or re-run the whole check detached).
4. Hand back to the Master Mind (<= 250 words): deliverable file:lines (below), counts 357 / mutation, the two findings, the manifest = `out/paths.json`.

## Where each deliverable is (package `.../raw/2026-10-19-search-navtest-pipe/code/` unless noted)
1 DQ6(b): `navpipe_chain.py:456` stage_score_c2, `:406` scores_valid, `:577` merge_all; `navpipe_core.py:980,992,1006`; `COST.md`.  2 DQ7(b): `navpipe_core.py:395,402`; chain `:648`.  3 config: `navpipe_core.py:281,308,345,445-478,547`.  4 portability: `navpipe_core.py:41-58` (siblings verified on the tip; list in `siblings.txt`).  5 DQ9: `seal_c2_amendment.py`.  6 draft: `.../2026-10-19-search-p2-confirm/c2/AMENDMENT_C2_DRAFT.md` (FINAL DRAFT, section 14 RULED, 27 pins, 12 filled at seal).  7 counts above.
New tests: `test_navpipe_select.py` (58), `test_seal_c2.py` (33 incl. the real draft is seal-ready).

## Findings for the Master Mind (also in `LICENCE_GUARD.md` and draft section 14)
- `dose_chain.licensed()` / `c2_config` read `moved_share`; the C2 licence reads `inert.frac_moved` (DQ2 (a)). They can disagree on one file: C2 refuses by name, never chooses. Needs one ruling.
- `c2_config` takes the dose read in every row, so C2 needs the dose result even for C8: C2 cannot start before the dose arm reads.
- `dose/code/dose_chain.py` is not on the tip yet; tests use a test double that agrees with the real file (`par18/doseamend/code/dose_chain.py`) over a 63 x 63 x 5 grid of reads, and the real file drives the seal when present. The seal pin `dose_chain_sha256` binds the landed file.

## Local-only (never stage): `code/conftest.py`, `scratch/`, `pt/`, `npmut/`, `mut_out/probe.json`.
- Known nit (tests only, not a guard): `test_navpipe_select.py` `REAL_DOSE` falls back to the dev-box copy `C:/Users/Admin/par18/doseamend/code/dose_chain.py` when neither `C2_DOSE_CHAIN` nor the repo-relative `dose/code/dose_chain.py` exists; once the dose package lands, drop that fallback (edit only after the mutation run has finished).
