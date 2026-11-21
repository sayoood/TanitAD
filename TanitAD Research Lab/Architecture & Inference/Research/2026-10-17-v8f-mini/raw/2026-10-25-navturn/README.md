# Nav-vs-turn-label audit (2026-10-25, EXPLORATORY, CPU only)

Why the nav token and T's turn-label cells disagree in 339 eval-141 windows (341 when properly paired to the T tick). Written by a Sonnet analysis agent. The Master Mind checked both derivations at `stack/scripts/build_v9_labels.py:229-230, 264-281` (nav) and `:815-848` (turn cells), and the like-for-like contrast in `a3_error.out`, before banking. The report is `NAVTURN_AUDIT.md`, and each script's `.out` sits beside it.

* `r1_repro.py` re-runs the builder's own nav and turn logic, which reproduces 7,036 of 7,037 interior rows. `repro_eval141.npz` and `repro_turns.json` are its outputs.
* `navturn_windows.csv` lists every disagreeing window with its cause; `navturn_cause.npy` holds the same causes as an array.
* Run with the `tanitad` venv and `CUDA_VISIBLE_DEVICES=-1`. A read-only mirror export (`src/`) was used and is not banked.
* It feeds R71 and `PREREG_V8F_KIND.md` Amendment 1.
