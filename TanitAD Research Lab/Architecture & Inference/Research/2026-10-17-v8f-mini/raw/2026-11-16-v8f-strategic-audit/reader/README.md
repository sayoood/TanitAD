# STRATEGIC reader (Benchmarks agent build, MM-verified 2026-11-16)

Why: the PI ended the STRATEGIC pause on 2026-11-16, so every eval reports the STRATEGIC family in full. This package is the four-family reader with STRATEGIC cells from NVIDIA's navigation text, built to `../STRATEGIC_PLAN.md` section 4 items 1 and 3.

* `src/hwm_eval.py`, `src/assemble_arm.py`, `src/verdict_actsens.py`: copies of the HWM / ACTSENS readers with the pause text taken out and a STRATEGIC block in: `s_adh` (route adherence to an announced L/R turn inside the plan's arc), its signed-degree twin, `s_false` (false turn), each beside the logged path; n/a rows with reason and would-have n for S_CKPT (the HWM planner has no S head), strategic intent 6-30 s, derived goals and detected info. Full 6 s band only.
* `src/strategic_text.py` (new): joins the navtok table (`navtok_table.npz`, sha256 prefix 2836369e, sha-checked on load; default path on F:, or `ws.nav_table`) to windows at the window's NOW. `src/strategic_cells.py`: the cell rules.
* The originals in `raw/2026-11-08-v8f-bands-build/hwm_plan/` and `raw/2026-11-15-v8f-actsens-prereg/read/pkg/` are kept as written: they produced the banked HWM and ACTSENS reads, whose STRATEGIC stamp reads PAUSED because those reads finished before the ruling. Future reads (ACTRSP and later) use this package.
* `src/hwm_plan.py`, `src/hwm_data.py`, `src/goal_head.py`: unchanged copies so the tests run from this folder (env: `env.sh`).

Evidence (MEASURED):
* Tests: `83 passed, 3 errors` re-run by the MM in one process (42 s); the 3 errors are existing `test_hwm_plan.py` tests that need the `../wmact/latents_train.pt` data folder, and passed (3/3) in the agent's run with it (`out/PYTEST_3_DATA_TESTS.log`).
* Mutation: deleting the one TEXT-ATTACH line in `hwm_eval.metrics_of` turns 5 integration tests RED (`out/MUTATION_attach_line_deleted.txt`). A grep test asserts "PAUSED" is absent from the three readers, with a control string present.
* Join offset: the window's NOW maps to table step now + 0 (median |v_track - v0| 0.013 m/s, 84.1 % within 0.05 m/s), not the audit's planned now + 3 (0.150 m/s, 24.4 %), `out/STRATEGIC_TEXT_OFFSET_CHECK.json`. 910 of 994 ACTSENS windows carry a text row.
* Demo on the banked ACTSENS windows (`out/STRATEGIC_ACTSENS_DEMO.md`): T0 HWM-ORACLE diagnostic, the goal is the encoded true future, never a driving number. S_ADH: 8-16 scorable windows in 2-6 episodes per arm and planner, NOT decision-grade. S_FALSE (about 850 windows, 69-70 episodes) is read only as plan minus logged (logged rate 0.131): HWM-ORACLE plans turn less than the logged path where no turn is announced (-0.06 to -0.10, separated), FLAT-ORACLE plans more (+0.12 to +0.25, separated). Neither direction is a pass: below logged can mean plans straighter than the drive on unannounced bends. Intervals answer another draw of episodes only.
* COMPLETE_M 20 m, theta 25 deg (SLIGHT 10 deg) and the 30 m margin are HYPOTHESIS values; any decisive read pre-registers them (the ACTRSP card fixes them in section 4.3).

Not built here: S_CKPT in `v8f_mini_eval.py` (the STRAT1 build owns that file; its frozen card reads S_CKPT through `strategic_cells.ckpt_cell`).
