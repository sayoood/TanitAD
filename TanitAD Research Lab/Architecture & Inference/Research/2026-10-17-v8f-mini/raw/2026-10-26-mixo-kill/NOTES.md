# raw/2026-10-26-mixo-kill: the MIXO reader's mutation proof made whole, and re-checked on the tip's trainer

## 1. What was missing
`RESULT_V8F_MIXO.md` records that the reader's −2F edge and its O-LEARNED branch had SURVIVED its mutation proof. That is why the s0 verdict
was re-derived by hand from the per-window arrays (identical to 1e-15). Fourteen of the 162 mutants survived in all.

## 2. The fix (two files)
* `stack/tests/test_v8f_mixo.py`: 14 killing tests, added and nothing removed (`out/test_v8f_mixo.diff`). Among them:
  * the probe refusing a code that is neither the GT's nor T's own, an unlabelled window that read the GT, and a unit whose GT share is far
    from p_gt;
  * a failed probe refusing the launch before the gate record is written;
  * the PRIMARY exactly at −2F clearing, and one step short of it not clearing;
  * O-LEARNED only if the D-M read says so at every seed that missed;
  * the committed next-step texts held in full;
  * the dose-identity cells enforced one by one;
  * the distance-keeping lead windows being the lead block's join.
* `raw/2026-10-25-mixo/mutate_mixo.py` (`out/mutate_mixo.diff`):
  * three anchors respelled to the bytes FLAT / OACC left on the tip;
  * the scratch root made overridable through `MIXO_MUT_BASE`. The default is unchanged in behaviour; it moved the run's scratch off a full C:.

## 3. Evidence
* `out/mutation_proof_kill_6df75ad2.out`: tip 6df75ad2 plus the 14 tests gives **162/162 RED**; restored byte-identical; baseline 166 passed.
  `out/coupled_check_6df75ad2.out` holds the two label-pin tests of `test_v8f_flat.py` / `test_v8f_oacc.py`, which load `test_v8f_mixo.py`.
* **The re-check on the tip's trainer.** After 6df75ad2, O-RECOVER (98713259), OACC Amendment 2 (04d56671) and O-RECOVER Amendment 4
  (853f2fc3, the ego probe's at-rest exemption) changed `v8f_mini_train.py`.
  Every other file MIXO mutates is byte-identical on the tip. So `run_mx.sh` builds a tree from the tip and overlays the two edited files:
  * `out/recheck_mutate_mixo_check.out`: all 162 anchors are still found exactly once.
  * `out/recheck_mutate_mixo_train.out`: the trainer's mutants, re-run.
  * `out/recheck_coupled_check.out`: the coupled pin check, run by `run_oc.sh`. The `run_mx.sh` attempt errored before running any test,
    because `pytest --basetemp` needs its parent directory to exist.
* ⛔ **The first re-check is VOID (`R77`).** `run_mx.sh` copied the trainer out of the OACC fix tree at ~11:35, while the OACC mutation proof
  was editing it. The copy carried a mutant: `selfcheck.json` omits `acc_probe`, line 1216. Its outputs are banked under `out/void_r77/` as a
  record and decide nothing.
* **The second re-check is superseded by Amendment 4.** `run_tv.sh` re-ran the check on the tip's own trainer blob (7f06ff00): 15/15 RED,
  restored byte-identical, banked under `out/recheck_on_7f06ff00/`. Amendment 4 then changed the trainer again (12 lines in `ego_perturb_probe`).
  `run_tv2.sh` re-ran it on that trainer (the tip's blob, asserted by hash before the run), and the `recheck_*` files are from that run.
* `run_proof.cmd` / `wait_no_pytest.ps1` / `move_proof_to_e.py` are the as-run proof runners, banked as provenance. ⚠️ `run_proof.cmd` calls
  PowerShell with `-ExecutionPolicy Bypass`. Do not reuse it: later runs wait on `pytest_runs.py` instead.

The banked MIXO verdict does not move.
