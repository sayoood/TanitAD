# RESULTS: PREREG_V8F_LAUNCH_INTENT draft + `--launch-intent-w` (built and tested, NOT run)

Generated 2026-10-28 11:36:19 by `tools/assemble.py` from the run logs. Nothing was trained, launched, staged in git, committed or pushed. Box 2, the queue and every chain were not touched.
Base: tip `d7208328075a06efae321cc16c09ce7a902280ad` (`TIP.txt`). Every number below is MEASURED from the log named beside it. A line that reads PENDING is not a pass.

## 1. The trainer flag against the unmodified tip

| check | result | pass |
|---|---|---|
| GREEN on the build (all 36 tests) | `36 passed in 78.56s (0:01:18)` | **PASS** |
| RED on the unmodified tip d7208328 (same file, same golden) | `35 failed, 1 passed in 36.44s` | **PASS** |

* Files: `logs/final_green.log`, `logs/final_red_on_tip.log`. The RED run is the same test file and golden executed against the unmodified `d7208328` export (the new symbols do not exist there).
* The flag-off golden (`stage/stack/tests/launch_intent_golden.json`) was WRITTEN BY THE UNMODIFIED TIP (`logs/golden_write.log`) on three argv families (a DAG-OACC-style argv with the swap and `--w-acc 49.6969`, an ORC-style argv, plain M2b), 4 CPU steps each, torch 2.11.0+cu128. The build reproduces every logged number, every logged key, the final weights and optimiser state, the counters, and the key sets of `summary.json` / the checkpoint, bit for bit. The only differences are the declared additions (the checkpoint `cfg` gains `launch_intent_w` = 0.0; `levers` gains `launch_intent_w`, `launch_intent`, `launch_n`, `launch_eligible`; `selfcheck.json` gains `launch_intent_probe`).
* `diff.patch`: 189 lines added and 2 removed (the two existing lines it touches are `make_config`'s keyword list and the `cfg.` read-back tuple, each re-added with the new name); `v8f_mini_data.py` and `launch_gate.py` are byte-identical to the tip (`logs/launch_gate_grep.txt`).

## 2. Mutation proof of the trainer flag (`mutation.txt`, `tools/run_mutations.py`)

**30 / 30 mutants killed; survivors: none.** Controls (each selection GREEN on the unmutated tree first): all rc=0.

Each mutant deletes or breaks ONE rule in a copy of the tree (`F:/TanitAD-scratch/launch_intent/mut`, restored and sha256-verified after every mutant) and names the `-k` selection that must go RED. Required mutants include: deleting the loss line that adds the term, flipping the registered `>=`, flipping `<` to `<=`, the chord over the wrong knots, the chord along one axis, the wrong knot window, the cross axis, the wrong denominator, the mask ignored, the mask taken after the ego stages, the attach line deleted, the loop never launching, the FLAG_CHECKS entry deleted, the G-DVB read-back deleted, the probe stubbed, the probe's all-False control deleted, the checkpoint losing the totals, the parser default flipped.

## 3. The existing v8F suites on the build, one file at a time

| suite | rc | result |
|---|---|---|
| `test_v8f_ego_dagger_compose` | 1 | `1 failed, 35 passed in 284.78s (0:04:44)` |
| `test_v8f_ego_perturb` | 0 | `80 passed, 2 skipped in 80.08s (0:01:20)` |
| `test_v8f_dagger_train` | 0 | `95 passed, 2 skipped in 157.99s (0:02:37)` |
| `test_v8f_dagger` | 0 | `57 passed in 72.50s (0:01:12)` |
| `test_v8f_init_ckpt_o_interface` | 0 | `7 passed in 16.17s` |
| `test_launch_gate_v8f` | 1 | `47 passed, 9 errors in 16.14s` |
| `test_v8f_orc_chain` | 0 | `13 passed, 32 skipped in 34.85s` |
| `test_v8f_mini` | 1 | `1 failed, 28 passed in 19.51s` |
| `test_v8f_pgt` | 0 | `7 passed, 22 skipped in 8.94s` |
| `test_v8f_flat` | 1 | `1 failed, 40 passed, 222 skipped in 41.14s` |
| `test_v8f_oacc` | 1 | `1 failed, 38 passed, 94 skipped in 30.56s` |

A suite that is red here and green on the tip would be a regression of this build; compare `logs/suite_*.log` with the tip before reading any red as this build's. (`PENDING` = not finished.)

## 4. The dose, the REP licence and the outcome table

* `f_measure.json` (`tools/measure_f.py`): f = 49,618 / 306,566 = **0.161851**; window list sha256 equals the banked DAG-OACC runs'; trainer mask vs independent numpy spelling: 0 disagreements. **W_A = 1.58926**, W_B = 0.0357048, W_HI = 4.17852.
* `compare_rep_argv.out`: DAG-ORC r2's pinned REP argv equals DAG-OACC's banked argv token for token apart from `--out` and `--seed`, at both seeds (exit 0).
* `enumerate_cells_li.out`: exit 0 (`FAILS: none`), per-row counts equal DAG-ORC's registered counts. `enumerate_cells_li_mutation.txt`: 14 / 14 mutants killed.

## 5. Flags for the Master Mind (also in the PREREG §0)

1. **No standalone "O's along-track loss" exists at the tip.** O's position loss is ADE/FDE over both axes and all 30 knots (`v8f_mini.py:870`) plus `w_acc`'s second difference. The flag adds W x the along-axis, knots 1-12 slice of ADE's own per-knot distance on launch samples (one additive term, one total line). Confirm the reading.
2. **Arm B (one sixth) is a near no-op:** f = 0.1619, so the unweighted share is already 0.1619 and W_B = 0.0357 (+3.6 % weight). Re-fix B's share or drop B. Closed form for any share s: 1 + W = s (1 - f) / ((1 - s) f).
3. **A gate-probe class the build caught:** every probe in `main()` calls `model.loss` on the probe batch, and a lever-ON loss needs `b["launch"]` there too. The first test run failed on exactly this (5 tests); the probe batch now carries the mask. Any future lever that changes the loss's inputs must reach every probe, not only the loop.
4. `launch_gate.py` needs no change; the v8f-mini gate is the trainer's `--selfcheck-only` (`logs/launch_gate_grep.txt`). The chain's one-lever asserter must allow `--launch-intent-w` as the only extra flag.
5. REP reuse saves about 1.4 h of box 2 but needs DAG-ORC r2's REP runs to finish and pass; the golden covers CPU, not box 2's GPU.

