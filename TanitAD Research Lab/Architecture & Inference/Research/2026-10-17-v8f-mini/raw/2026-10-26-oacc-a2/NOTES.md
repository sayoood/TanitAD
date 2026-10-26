# raw/2026-10-26-oacc-a2: PREREG_V8F_OACC Amendment 2, and a mutation-anchor regression repaired

## 1. Amendment 2 (post-data, control only)
* The text is `AMEND_OACC2.md`, appended to the PREREG.
* The reader edit is `apply_oacc_a2.py`, which edits `code/v8f_oacc_read.py`, `stack/tests/test_v8f_oacc.py` (new test
  `test_check_x_run_amendment_2_record_formats`) and `raw/2026-10-25-oacc/mutate_oacc.py` (new mutant 162, "loss weights may differ from B's").
* `out/real_records_check.out` runs the control on the REAL pulled records, which are the argv / summary / selfcheck files only; no row is
  computed. The reader as it ran (98713259) refuses; the fixed reader returns no defect.

## 2. The mutation-anchor regression, introduced by the O-RECOVER landing (98713259)
* The new `ego_perturb_probe` copied two lines of the OACC `acc_probe`, `if res["control_prior_grad_diff"] != 0.0:` and
  `if res["control_wm_grad_diff"] not in (None, 0.0):`. Its block also sat between the acc probe's refusal and `adm = ...`.
* So `mutate_oacc.py --check` failed on the tip: two anchors were found 2×, and one was found 0×.
* **Fix:** the ego probe's two lines are respelled with identical semantics (`0.0 != ...`, `not in (0.0, None)`). Its try-block moves to just
  after `model.to(dev).train()`, so it sees the model state the acc probe saw. At p = 0 the probe returns at once.
* `v8f_mini_train.py`: 10 changed lines vs 98713259 (`out/v8f_mini_train.py.vs98713259.diff`).

## 3. Evidence (CPU)
* `out/pytest_oacc_ego.out`: `test_v8f_oacc.py` + `test_v8f_ego_perturb.py`, 211 passed, 1 skipped.
* `out/mutate_oacc_check.out`: 309 mutants, each anchor found exactly once.
* `out/mutate_orecover.out`: 20/20 RED, files restored byte-identical.
* `out/mutate_oacc_sel.out`: the 21 OACC mutants on the acc probe, the built levers and the new weight check (#35-#50, #150, #151,
  #153, #161, #162). The run was still going at landing (about 5 min a mutant), so it is banked by a follow-up commit, not here.
  Box 2 was idle, and the run tests the tests: the suites and the real-record check above already pass.
* ⚠️ **A cmd.exe trap, seen twice today.** `queue.cmd evalmixo`'s dry-run echo line sits near cmd.exe's 8,191-character limit (8,092 chars
  with a 57-char checkout and a long pytest root). A longer temp path makes `test_queue_oacc_leaves_every_other_stage_untouched` fail with
  "The input line is too long". Keep the checkout and `--basetemp` / `--tmp` short. The first `mutate_oacc` baselines here failed exactly so.
* The superseded attempts (the first test run's 52 basetemp errors, the anchor failures before the de-dup, the `--only` runs whose baseline hit
  the line limit) are not banked here. They stay in `E:/Projects/TanitAD-artifacts/oaccfix/out/` on the dev box.
