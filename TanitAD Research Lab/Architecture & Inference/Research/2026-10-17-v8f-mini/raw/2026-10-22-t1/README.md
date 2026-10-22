# raw/2026-10-22-t1: a T1 (self-action) eval for v8F-mini, built and CPU-tested, not yet run on real cameras

* **The tool:** `stack/scripts/v8f_t1_eval.py`, tested by `stack/tests/test_v8f_t1.py` (23 tests, CPU; re-run by the Master Mind before landing in a clean export of the tip `abccc45`: 23 passed, with `test_v8f_mini.py` 29 passed beside it).
* **Its name is T1-kin, never "closed loop".** `Project Steering/EVAL_DOCTRINE.md` (PI amendment 2026-09-02) makes T1 self-action OPEN loop; only T2 closes the loop. Here the ego state is the model's own executed motion and the cameras stay logged.
* `DESIGN_T1_V8F.md`: written before the code. It covers the inputs the model reads, what is replaced and what stays logged, the leak test per input, the controls, and the real-data floor probe (section 8).
* `mutate.py`, `mutation_results.json`: six source mutations, each turning tests red.
* `probes/`:
  * `floor_real_probe.py` (+ `.json`, `.log`): the camera-free floor arm on the 141 real eval clips, MEASURED by the T1 stream and not re-run by the Master Mind. ADE6 by replan period m: 2.487 (m = 30, equal to the banked `P10_refit__ade6`), 3.44 (10), 5.41 (5), 23.3 (2), 21.7 (1).
  * `floor_sensitivity_probe.py` (+ `floor_sensitivity.json`, `floor_real_fid.json`): state-rebuild fidelity.
  * `nav_probe*.py`: the dead-reckoned nav against the logged `d_next`.
* `smoke_real_ckpt.py`, `smoke_out/*.json`: the real M2b_s0 checkpoint rolling on the synthetic world. **Plumbing only: the numbers mean nothing.** The synthetic dumps are not banked.
* Not done: any model-vs-floor T1-kin number. It needs the pooled eval cache (box 2) and a PI go.
* Not fixed here: `taniteval/tests/test_t1_eval.py::test_cli_help_works_without_a_gpu` fails on a Windows cp1252 console and passes under `PYTHONUTF8=1` (an environment failure, not a code one). `taniteval/tools/t1_eval.py` still prints the old "action-closed-loop" wording; its owner should fix it.
