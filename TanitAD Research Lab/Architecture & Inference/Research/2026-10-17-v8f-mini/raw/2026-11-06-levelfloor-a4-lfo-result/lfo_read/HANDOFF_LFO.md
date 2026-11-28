# HANDOFF_LFO: arm LFO075 (A4, the level floor with the LOGGED speed as the level input, a CEILING diagnostic). Built, validated, chain LAUNCHED 2026-11-06 04:15:55. Nothing committed, nothing landed.

Card `a4_prereg/PREREG_LEVELFLOOR_A4.md` sha 8bfbc386 (frozen, unchanged). Packages (staged, not in git): `E:/Projects/TanitAD-artifacts/dagland/motion_c/lfo_build/` and `.../lfo_read/`. Scratch `F:/TanitAD-scratch/motion_c/lfo_read/` (`tree/` the LFO read tree, `out/` the 4 rollouts, `work/`, `smoke/`) and `F:/TanitAD-scratch/motion_c/lfo_build/` (`ka/` the two full KA rollouts, `mut/`, `smoke/`).

## State
* Launched with WMI `Win32_Process.Create` (`lfo_read/lfo_chain.cmd`); WMI pid 54956; the chain's own pid is in `lfo_read/raw/LFO.running`. Gates 0-2 and both pre-flights passed at 04:16; RILFO075_s0 finished at 04:20 in 243 s and its rollout check (floor keys, logged source, CEILING label, planner sha) passed. ETA ESTIMATED 04:37 (3 more rollouts of ~4 min on 12 CPU threads, then the reader ~3 min).
* Order: RILFO075_s0, RILFO075_s1 (bare), RILFO075C_s0, RILFO075C_s1 (composite), all `reindexed`, CPU. Reader: `code/lfo_read.py` -> `raw/LFO_READ.json`, `RESULTS.md`.
* Check with: `ls raw; tail -n 8 logs/chain.out` (lines `ZZLFO-*`). Markers: `raw/LFO.done` (finished) or `raw/LFO.fail` (the failing step is named in it; a chain killed by TERM, INT or HUP also leaves one). RESUME = launch `lfo_chain.cmd` again (finished rollouts are skipped).

## Built and validated (all MEASURED, logs in the packages)
* Tool: `lfo_build/tool/v8f_t1_eval_ri_lfo.py` sha 6744d3e4, regenerable from the LF tool 6166178a by `code/apply_lfo_patch.py`. Flag `--floor-vhat-source {heads,logged}` (heads = default, logged refused without `--level-floor on`, named explicitly in every LFO argv). Operator `level_floor.py` unchanged (2781e23a).
* KA-O1 PASS (full RILF075_s0 rollout, heads source): 46 common tick fields, 7,197 rows, 141 dump files, the analysis JSON and 44 per-window metric arrays byte-identical to the banked LF rollout. KA-O2 PASS (full RIC0_s0, floor off): same, byte-identical.
* KA-O3 PASS (4 tests): oracle v_hat equals `lf_read.logged_speed` to 1e-9 m/s; gt_ok False gives no floor. Mutations: deleting the oracle assignment fails 3 of 4 KA-O3 tests, dropping the gt_ok guard fails 1 of 4; 12 of 12 tool mutations RED. LF test suite on the LFO tool: 12 of 12 green. Real-data smoke (64 windows, logged): lf_vhat equals the reader's logged speed to 1e-6 m/s.
* Reader: ka-lf 24 of 24 (RILF075 arrays in the LFO slot reproduce LF075's P, G, F and every row, attribution 0.0 on all 19 rows, label CEILING-SHORT), ka-null 17 of 17. 12 + 11 package tests green; 31 of 31 reader / pre-check / chain mutations RED.

## Exact next step
When `raw/LFO.done` exists: read `lfo_read/RESULTS.md` (outcome label exactly one of CEILING-CLEARS / CEILING-HARMS / CEILING-SHORT / F-UNREACHED, every line carries CEILING), then package `lfo_build` + `lfo_read` like `lf_result_pkg` (`PINS.txt`, `MANIFEST.txt` regenerate with `python -I code/make_pins.py`) and hand to the Master Mind to land, with the A4 claim-row update. Act per card section 5: CLEARS -> addendum A5 (noise sweep) and the estimator evidence to the PI; HARMS -> an operator-variant addendum; SHORT -> report T2 / kappa 1 options to the PI.

## Escalation for the Master Mind
`lf_read/lf_read_chain.sh` gate 0 uses `sed -n 's|...\(code/...\|tests/...\)...|\1|p'`: the delimiter `|` is also the alternation character, so the expression matches nothing and its pin loop over code and test files never ran (only the `need` presence list did). The LFO chain uses `#` as the delimiter and a test (`test_chain_refuses_a_pinned_file_that_differs`) catches a changed file. The LF chain already ran; this is a latent gate defect, not a result error.
