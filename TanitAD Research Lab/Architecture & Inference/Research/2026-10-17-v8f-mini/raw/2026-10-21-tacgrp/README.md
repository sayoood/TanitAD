# raw/2026-10-21-tacgrp: PREREG_V8F_TAC Amendment 1 (the per-group oracle) on box 2

| file | what it is |
|---|---|
| `tac_groups_read.json` | the committed reading (`code/v8f_tac_groups_read.py`): MIXED; every group at every checkpoint, with LON / LAT families |
| `eval/eval_<ckpt>-tacorc-<group>.{npz,json}` | the 32 group-oracle evals (4 checkpoints × 6 groups + 2 composites); the json holds the manipulation record |
| `telemetry_v8ftacgrp.csv` | box-2 telemetry over the run (max USB 36 °C) |
| `box2_logs/chain_v8f_tacgrp.out` | the chain: refresh → run (rc 0) → **STOP at the pull** (the `GROUPS` trap) |
| `box2_logs/finish_v8f_tacgrp.out` | the finisher: pull + read, ZZV8FTAC-DONE; no eval re-run |
| `box2_logs/v8ftacgrp_run.log`, `v8ftacgrp_guard.out` | the box-2 queue log (32 `[ok]`), the temperature guard |
| `mutate_tac.py`, `mutate_tac_a2.out` | the mutation harness at Amendment 2 and its output |

The tacpred / tacorc runs these are read against are PREREG_V8F_TAC's, banked in `../2026-10-20-tac/eval/`.
Code at the run: remote tip `742fca0e`, six key files sha-checked on box 2; no v8F file differs from `c7134f8`.
