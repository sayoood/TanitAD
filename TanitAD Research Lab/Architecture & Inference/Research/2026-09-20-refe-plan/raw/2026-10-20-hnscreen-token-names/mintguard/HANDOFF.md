# HANDOFF: hntok2 (HNSCREEN-TOKEN-NAMES mint-stage guard) -- STATE: DONE, nothing running

Base tip `08ff54ed71f08bc86c99378d650bea758cf06460`. Staged in `C:/Users/Admin/par18/hntok2/out/` (`paths.json` = 11 rows; the 3 edits carry `base=<tip blob>`; `out/RESULT.md` = the repo `.../raw/2026-10-20-hnscreen-token-names/mintguard/RESULT.md`).

## What changed
`refe/hnscreen_devbox.py`: new `token_name_problems` (:564-585) and the guard in `verify_pass_token` (:593-596, before the `launch_gate verify` subprocess). A wrong `gate.token_check_names` (a value absent from the token's `checks`, or not PASS) stops the chain at the mint: rc 2, `SCREEN-INVALID: token check '<name>' absent`, outcome row 0. Tests: check L in `test_hnscreen_chain.py` (+4 mutants), one real-mint arm + 1 mutant in `test_hnscreen_ctx_chain.py`.

## Counts (ZZTEST_PASS, before -> after)
chain 17 -> 21; ctx_chain 20 -> 22; token_names 23 -> 23 (file unchanged).

## Next step (Master Mind)
1. Land `out/paths.json` with the base guard (blobs: hnscreen_devbox `47eda339`, test_hnscreen_chain `6015a0ee`, test_hnscreen_ctx_chain `4136ba87`).
2. BACKLOG `HNSCREEN-TOKEN-NAMES` follow-up -> DONE (rows proposed in RESULT.md).
3. Optional, not done: a static pre-check in `Chain.preflight()` against `launch_gate.PROFILES[profile]["checks"]` (the guard cannot precede S2 or T0's own `--preflight`, because the token does not exist until then).

## Re-run (CPU, about 7 min; python = C:/Users/Admin/venvs/tanitad/Scripts/python.exe)
`bash C:/Users/Admin/par18/hntok2/run_suite.sh tree rerun refe/test_hnscreen_chain.py refe/test_hnscreen_ctx_chain.py raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py` (logs `logs/rerun_*.log`, done marker `logs/rerun_ALLDONE`). Tree: `hntok2/tree` (tip export + edits; `taniteval/results` and `eval/raw` data pruned for disk).
