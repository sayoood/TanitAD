"""Rerun ONLY the two real-data mutation arms of `run_mutations_pin5.py`, through its own `real_arm` and `REAL_MUT` (imported, not copied).
Why: the full run of 04:02 printed `mutations RED: 21 of 23`; both GREEN lines were `the mutated read failed rc=1` with an empty stderr, a failure of the
mutated READ, not a verdict on it. A manual run of the same mutated file at 04:5x read rc 0. The 21 synthetic arms are not rerun here; their RED lines stand in `logs/mutations_pin5.log`.
Run from t2v8f/:  PIN5_MUT_SCRATCH=<fresh dir> python -I pin5/rerun_real_arms.py   (ends with `real arms RED: N of 2`)."""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_mutations_pin5 as M  # noqa: E402

M.SCRATCH.mkdir(parents=True, exist_ok=True)
red = 0
for name, edits, extra, what in M.REAL_MUT:
    is_red, text = M.real_arm(name, edits, extra, what)
    red += is_red
    print(f"{name}: {'RED' if is_red else 'GREEN  <-- not caught'}  {text}", flush=True)
print(f"real arms RED: {red} of {len(M.REAL_MUT)}", flush=True)
print("ZZ PIN5-REAL-RERUN-DONE ZZ", flush=True)
sys.exit(0 if red == len(M.REAL_MUT) else 1)
