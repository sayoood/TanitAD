# HANDOFF ACTLEVEL (build agent, 2026-11-19)

State: DONE. Level term `--w-act-level V` built in `work/tree_level` (copy of `actcons/work/tree_patch`; base sha256 of the three changed files verified before work). Nothing landed, no remote machine touched.

Changed files: `stack/tanitad/models/act_cons.py` (new `level_label_mask`, `act_cons_term(..., level=False)`, `build_record(..., level_w=0.0)`), `stack/scripts/v8f_mini_train.py` (flag, G-DVB entry, `bad_act_cons`, probe level reads, selfcheck, loop, summary), `stack/tests/test_v8f_act_cons.py` (two source-count lines only), new `stack/tests/test_v8f_act_level.py` (11 tests). Copies in `patched/`, diff in `ACTLEVEL.patch`, pins in `PINS_LEVEL.txt` (15 lines: the 14 of `in_pod/arms_actcons_c0.txt` + the new test).

Measured here: tests.log 184 passed, 0 failed (11 files of run_tests.sh, 11 of them new level tests); `checks/v3_identity.json` pass, max_abs_diff 0.0 (4 steps x 37 keys, old = actcons tree_patch); `MUTATIONS_ACTLEVEL.json` 6 of 6 red, unmutated green before and after; `checks/a4_level_smoke.json` PASS (selfcheck `levers.w_act_level` 0.01, probe level keys, cfg and state keys equal to V = 0 run, step-1 total delta 0.235916 vs V x act_level 0.235917, mutation deleting the add reads 0.0).

Arm tokens: argv `--w-act-level 0.001` / `--w-act-level 0.01`; expectation `levers.w_act_level=0.001` / `levers.w_act_level=0.01` (plus `levers.w_act_cons=0.001 levers.act_cons_max=1.5`). The gate parser was not run here.

Next step for the lead: read `ACTLEVEL_DEFS.md` REVIEW items R1 to R6; build the arms file with `PINS_LEVEL.txt` (tree name `/workspace/v8proof/tree_actlevel`); run `checks/v3_identity.py <old_tree> <new_tree> out.json` on the pod tree pair; the pod dry gate reads the probe on real data.

FOOTPRINT 2071 MB, over the 400 MB limit: work/pt 1380 MB (the full 11-file pytest basetemp, test tmp_path dirs), work/tmp 612 MB (a4l 330, v3 264, synthetic worlds); both are temp data the lead can delete (re-creatable by run_tests.sh, checks). Strays (could not delete, delete is refused): all of `work/` (tree_level 39 MB, tree_mut ~39 MB, mut_smoke, tmp, pt, dbg.py, edit1.py, edit2.py, quick.log, mut.log) and `checks/a4_level_*.log`. `work/tmp` holds synthetic worlds and smoke run dirs; nothing else outside actlevel was written.
