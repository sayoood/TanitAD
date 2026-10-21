# raw/2026-10-21-tacgrp2: PREREG_V8F_TAC Amendment 2 on box 2, and the absent-slot diagnostic

| file | what it is |
|---|---|
| `tac_groups2_read.json` | the committed reading (`code/v8f_tac_groups2_read.py`): CONSTRAINT-ACCURACY, every row, LON / LAT families, T's code error per group |
| `eval/eval_<ckpt>-tacorc-<spec>.{npz,json}` | the 40 evals: `lonspeed+londec`, `all-<g>` ×6, `constraints-a{25,50,75}`, each with the code O read (`tac_code`) |
| `telemetry_v8ftacgrp2.csv` | box-2 telemetry over the run |
| `box2_logs/` | the chain (ZZV8FTAC-DONE 21:20), the box-2 queue log (40 `[ok]`), the temperature guard |
| `absent_slot_error.{py,out}` | POST-HOC: T's constraint error split into present / absent slots. It motivated PREREG_V8F_CZ and decides nothing |
| `mutate_cz.{py,out}` | the mutation proof for PREREG_V8F_CZ's instruments |
| `mk_hold9.py` | the script that built this landing |

The `tacpred` / `tacorc` / `tacorc-constraints` runs these are read against are banked in `../2026-10-20-tac/eval/` and
`../2026-10-21-tacgrp/eval/`. Code at the run: remote tip `f2f604f5`, six key files sha-checked on box 2.
