# SPEEDAVG pair-check fix (2026-10-25 21:45, MM), before any SPEEDAVG number

The real arm's first run (21:40) STOPPED at the driver's pair check, before any eval:
`REFUSED: the two checkpoints' configs differ in ['o_interface']`.

`cfgdiff.py` on the two checkpoints gives exactly one difference: `o_interface` is `<MISSING>` in X_s0 and `'full'` in X_s1.
* X_s0 = `M2b-mixo_s0`, sha c6a62876…. It was trained before PREREG_V8F_FLAT's lever existed.
* X_s1 = `M2b-mixo_s1`, sha 68a0f335…. It was trained after the lever landed.

`v8f_mini_eval.py` builds `M.V8FMiniConfig(**ck["cfg"])`, so X_s0's missing key builds the default `"full"`. That default is "every earlier arm, bit for bit" (`v8f_mini.py:183`). The two are one model, and the raw-dict compare was stricter than the loader.

**Fix:** `pair_violations` compares `built_cfg(...)`, the stored keys over the dataclass defaults.
* Test: a pre-key checkpoint against `full` passes; against `cut` it is refused.
* Mutation: with the raw-dict compare restored, `test_pair_violations_on_checkpoints` goes RED by its assertion.
* `test_v8f_speedavg.py`: 53 passed.

No arm, cell set, bar or outcome changes. The run's logs from the refused attempt are kept beside the rerun's.
