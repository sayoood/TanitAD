# HANDOFF: A2-LABELS (2026-11-08). State: labeller extension BUILT, TESTED and MEASURED. Nothing running. Nothing staged, nothing committed.

Base: `agent/arch-inf-20260803` tip **d5ec25ec** on origin (the mirror's LOCAL branch ref still read 10dbe3f, behind GitHub: used `git fetch` then the remote sha). Re-export it with
`git -c core.autocrlf=false archive d5ec25ec "<Research>/2026-09-20-refe-plan/refe" | tar -x` then `bash tools/make_merged.sh <that refe dir> <out>` to get tip + overlay.

## What exists (all under this dir)
* `overlay/refe/behaviour_terms.py` the five terms; `behaviour_gate.py` the navtest gate (log + token keys, sha-verified lists, fails closed); patched `navsim_pdm_targets.py` (`score_plans(..., behaviour=False)`), `online_labeller.py` (`OnlineLabeller(behaviour=False)`, `_pdm_init(..., behaviour=False)`), `label_worker.py` (`--behaviour`). Diffs vs the tip: `patches/*.diff` (regenerate: `tools/make_overlay.py`).
* `gate_data/` 147 navtest logs, 77,148 tokens, sha256 in `navtest_gate.json` (built by `tools/build_navtest_gate.py`). Lands beside the module, in `refe/gate_data/`.
* Tests: `tests/test_behaviour_terms.py` (10, synthetic scenes, controls read exactly 1; run `python tests/run_tests.py` in venv driverl-eval, which has no pytest), `tests/test_behaviour_gate.py` (7, real pytest in venvs/tanitad), `tests/test_a2_off_identity.py` (real data, env-gated). Logs in `measure/test_log_*.txt`. Mutation checks done by hand: removing the q_osc line or the q_comf exponent turns a test RED; the gate has its own regression arm (`REFE_LAB_MUTATE=no_navtest_gate`).
* `measure/`: `RESULT_A2_ACTIVATION.md` (+ `a2_real_results.json`, `job_identity_result.json`, `cost_probe_result.json`, `states_used.json`, build/measure logs), scripts `build_scenes.py`, `measure_a2.py`, `job_identity_child.py`, `run_identity.sh`, `cost_probe.py`, `make_report.py`.
* `WIRING_PLAN.md` (trainer wiring with anchors, leak test, M-SHUF mutation), `A2_PREREG_NOTES.md`.

## Exact next steps (for whoever continues)
1. Master Mind: integration. The overlay is NOT in the repo. To land: copy `overlay/refe/*.py` + `gate_data/` into `refe/`, land via the coordinated lander (`land.sh`/`qland`), `pytest` the refe suite (`test_online_labeller*`, `test_p6_*`) first: only `online_labeller.py`, `label_worker.py`, `navsim_pdm_targets.py` change existing behaviour-bearing files, and OFF is byte-identical (62 real scenes).
2. When SCORER1-IMPL has landed: execute `WIRING_PLAN.md` section 2 (re-anchor line numbers) and write `behaviour_label.py` (offline side-file writer, ~100 lines, copy of `rl_terms_label.py`).
3. Pod check (minutes of CPU): the label-quality gate of `A2_PREREG_NOTES.md` section 4 on REFe's own fan over >= 500 navtrain sets, then decide the OPEN items in section 2 (q_free speed limit via a D9 parts bump; q_gap bands; q_stall stop signs; drop near-dead terms).
4. Only then `PREREG_A2.md`, PI go, launch-gate PASS.

## Known gaps (flagged, not hidden)
* Trainval states, not the navtrain filter (filter file absent); synthetic fan, not REFe's. 2 of 64 selected states failed to build (`'NoneType' object has no attribute 'id'` in scenario/PDM-Closed route set-up, scenes 18 and 54, same log; not the extension).
* `q_free` uses a reference-relative v* (no map in the parts). `q_stall` has no stop-sign knowledge. `q_gap` bands fire on human driving. See the notes.
* With behaviour OFF the labeller does NOT gate navtest (pre-existing); the gate runs when behaviour is ON, and WIRING_PLAN item 9 closes it in the trainer.
* Timings are one process on a shared box.
* Scratch `F:/a2_scratch` (41 MB) was NOT deleted: the harness refused the removal; see MANIFEST.
