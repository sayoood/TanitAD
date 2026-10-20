# HANDOFF hntok3 (HNSCREEN-TOKEN-NAMES-STATIC), written 2026-10-20

State: code and tests DONE and syntax-checked; check M + L alone pass unmutated (15 s). The three full suites are running detached (about 10-15 min total).

Edits (tip `2f1969a`; fixed tree `C:/Users/Admin/par18/hntok3/fix/`, pristine tip export `.../base/`, both under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`):
- `refe/hnscreen_devbox.py`: `token_names_config_problems` `:949-973`, call in `preflight` `:981-983` (rc 4 `TOKEN-NAMES-CONFIG`, before the freeze and S0), docstring `:8-10`.
- `refe/test_hnscreen_chain.py`: `run_chain(no_cfg_check=)`, L's four name cases bypass the new first line, NEW check M, 10 new mutants, `load_module` keeps `__src__`.
- `refe/test_hnscreen_ctx_chain.py`: `scenario(no_cfg_check=)`, arm C10 uses it.

Jobs (all CPU, detached, no further action needed to start them):
- before (tip tree `base`): `logs/before_*.log`, done marker `logs/before_ALLDONE`.
- after (fixed tree `fix`): `logs/after_*.log`, done marker `logs/after_ALLDONE`.
- `post.sh` waits for both, runs check M against the TIP source (must go RED, `logs/red_on_tip_M.log`), then `finalize.py` (writes `out/` tree, `out/paths.json`, `.../configguard/RESULT.md`, `edits.diff`, logs). Done marker `logs/POST_DONE`; output `logs/finalize.out` (counts as JSON).

Exact next step (fresh Sonnet agent, small):
1. Wait for `logs/POST_DONE` (one background until-loop, not turns). If C: free < 60 MB stop.
2. Read `logs/finalize.out`: `dropped` must be `[]`, `red_on_tip` true, `base_ok` all true. Read the three `ZZTEST_PASS` lines and any `ZZTEST_FAIL`/`PROBLEM` lines in `logs/after_*.log` (expected before 21 / 22 / 23; after 21+10=31 / 22 / 23 for chain / ctx / token_names, chain also gains check M).
3. If a mutant stayed GREEN or a check failed: fix in `fix/`, then rerun with `bash run_suite.sh fix rerun <tests>` and `python finalize.py` (edit the `after_` tag in finalize.py to `rerun`).
4. Skim `out/.../configguard/RESULT.md` (generated), then hand back (<=150 words): `file:line`, counts before/after, manifest = `out/paths.json`.

Never commit or push. Kill only own PIDs. TEMP/TMP under `hntok3/tmp`.
