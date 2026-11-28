# MANIFEST A6-IMPL (nothing staged, nothing committed, no GPU)
| artifact | where |
|---|---|
| scorer change: `behaviour_score.py` (HumanPath end extension, `_moving_chord_dir`, params), `test_behaviour_score.py` (+3 tests, golden test strips 2 new params), `mutation_check.py` (+`pathext` site), `make_pin.py` (+amendment_6 sha) | repo-external: E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code/ |
| `CODE_PIN.json` (new), `CODE_PIN.before_a6.json` (previous), `TEST_RESULT.txt` 85/85, `MUTATION_RESULT.txt` ALL KILLED, `mutation_check_result.json` | same folder |
| selector: `select_suite_tokens.py`, `a3_slots.py` (A6 copies), `audit_a6.py`, `per_token_results.pkl` (copy of a3's, input), `rows3_a6.pkl` (7 MB, audit input) | E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/a6/ |
| `SUITE_CANDIDATES_A6.json`, `MAP_FIELDS_A6.json`, `SUITE_PIN_A6.json`, legacy `SUITE_CANDIDATES.json`/`_ALT.json`, `suite_features.csv.gz`, `run_report.json`, `a6_run.log` | same |
| `AUDIT_GAP.json`, `AGREEMENT_A6.json`, `DIFF_VS_A3.json`, `HANDOFF.md`, `MANIFEST.md` | same |
Not in the git repo (artifacts tree only). Raw nuPlan frames/DBs: read only, nothing copied out.
Deletions (own scratch): tsf_bundle2_20261107/a6_scratch/ (3 MB, superseded first-run JSONs), a6/__pycache__ (2 MB), a6/dbg.py, a6/audit.out (1 KB each), code/__pycache__ (regenerable). All succeeded.
Untouched: a3/, bundle_a3 on F:, AMENDMENT_6 files, GPU lock.
