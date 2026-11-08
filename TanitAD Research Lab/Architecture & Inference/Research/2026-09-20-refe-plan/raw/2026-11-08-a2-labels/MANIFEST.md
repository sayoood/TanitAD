# MANIFEST: A2-LABELS (root = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/a2/`). Nothing staged in git, nothing committed (the brief says stage nothing). Every artifact below exists ONLY here, so it is single-copy: bank it by landing per HANDOFF step 1.

| artifact | path | note |
|---|---|---|
| terms | `overlay/refe/behaviour_terms.py` | new |
| navtest gate | `overlay/refe/behaviour_gate.py`, `gate_data/{navtest_logs.txt,navtest_tokens.txt.gz,navtest_gate.json}`, `tools/build_navtest_gate.py` | data re-creatable from `F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test` |
| patched files | `overlay/refe/{navsim_pdm_targets,online_labeller,label_worker}.py`; diffs `patches/*.diff`; generators `tools/make_overlay.py`, `tools/make_merged.sh` | base = origin d5ec25ec |
| tests | `tests/test_behaviour_terms.py`, `tests/test_behaviour_gate.py`, `tests/test_a2_off_identity.py`, `tests/run_tests.py`, `tests/_pytest_shim/pytest.py` | logs in `measure/test_log_*.txt` |
| real-data measurement | `measure/RESULT_A2_ACTIVATION.md`, `a2_real_results.json`, `job_identity_result.json`, `cost_probe_result.json`, `states_used.json`, `build_scenes64.log`, `measure_a2.log`; scripts alongside | |
| plans | `WIRING_PLAN.md`, `A2_PREREG_NOTES.md`, `HANDOFF.md`, `CODE_PIN.sha256` | |

## Scratch NOT deleted (the delete was refused by the harness's safety check; the owner must remove it)
`F:/a2_scratch` (about 41 MB: tip export, merged tree, built scenes with metric-cache parts, fans, npz dumps, intermediate result dirs, logs). Everything worth keeping was copied into `measure/` first. Re-creatable: tip from git d5ec25ec, scenes via `measure/build_scenes.py` (seeded) from the local nuPlan trainval DBs. Nothing in it is a only copy. Note that `tests/test_behaviour_*.py`, `measure/*.py` and `tools/make_merged.sh` default to paths inside it (`A2_TIP_REFE` overrides), so re-export the tip before re-running them.
Not touched: `pilot/scorer_gym/` (read only, no bytecode written), `scorer_finetune.py` (not opened for edit).
