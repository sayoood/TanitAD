# T-error attribution (2026-10-25, EXPLORATORY, CPU only)

Where v8F-mini's tactical-layer constraint error lives, and what each lever could close. Written by a Sonnet analysis agent; the Master Mind verified the train-vs-eval units against `stack/tanitad/models/v8f_mini.py:681-682` and the lever table against `s9_lever_table.out` before banking. The report is `T_ATTRIB.md`, and each script's `.out` sits beside it.

* Inputs: the banked per-window eval NPZ/JSON of M2b s0/s1 and the GRID, HIST, PGT, WC and DSCALE arms. They were read from the local box-2 pull (`box2_ship/logs/v8f_*/eval/`), and the same files are banked under this package's `raw/<date>-<arm>/` dirs. `raw_in/` holds byte copies of `raw/join_eval141.npz`, `raw/eval141_kin.npz` and `raw/eval_grid_windows.json`.
* `table.npz`: the per-window table every script reads, built by `s1_build_table.py`. `gap_M2b_s{0,1}.npz`: per-window T-to-GT ADE6 gaps.
* `tcommon.py`: `S` is the directory of the file itself (it was a scratch path when the scripts ran).
* `T_ATTRIB.md` §5 lists the box-2 jobs it asks for; none were run.
* Reproduce with the `tanitad` venv (numpy 2.5.1, `CUDA_VISIBLE_DEVICES=-1`): `s3_tail.py` rerun from this directory matched its `.out` byte for byte (2026-10-25).
