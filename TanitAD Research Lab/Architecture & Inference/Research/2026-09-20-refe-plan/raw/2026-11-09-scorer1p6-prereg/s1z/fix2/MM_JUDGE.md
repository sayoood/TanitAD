# Master Mind addition to fix 2: the launch gate re-judges the new G18 fields (2026-11-10)

The agent's fix put the 10x noise rule only in the trainer's own `ok`. `gate_checks.judge_g18` re-judges G18 from its raw fields and treats `ok` as a claim, so without this addition a record whose `ok` was false on the noise rule could still pass the gate's judge.

Added to `refe/gate_checks.py`:
- `G18_NOISE_FACTOR = 10.0`, pinned equal to `scorer_finetune.TWIN_NOISE_FACTOR` by `test_s1z_gate.py`.
- `judge_g18` now fails when any of these holds:
  - `loss_grad_noise_w0_vs_w0` is missing, not finite, or does not cover every scorer module;
  - `noise_factor` differs from the gate's;
  - the delta is not more than 10x the noise on some module;
  - `probe_batch.found` is not true;
  - some head has 0 pairs in `micro_batch_pairs`, which was the pod's record of 2026-11-10.

Added to `refe/test_s1z_gate.py`, section G5: the fixture carries the new fields. There are six RED cases: noise equal to delta, delta 7.5x noise, the noise field missing, a wrong factor, no probe batch found, and 0 pairs on DAC. There is also one pass case at 15x, which shows the rule is not a switch.

Tests (CPU, venv tanitad, CUDA hidden, on the tip 9e1efffe with fix 2 and this addition overlaid):
- PASS: test_s1z_gate (3 s), test_s1z_preflight (17 s), test_scorer1p6_gate (4 s).
- test_launch_gate stops at `full()` on the missing `D:\Projects\TanitAD\data\refe_consistent\targets_rank1.jsonl`, the known environmental failure recorded in `s1z/HANDOFF.md`. The suite on the pod runs it in its own environment.
- The agent's 11 scripts passed on fix 2 before this addition. The other 8 do not call `judge_g18`, and the change adds one constant and lines inside `judge_g18` only, so they were not re-run.
