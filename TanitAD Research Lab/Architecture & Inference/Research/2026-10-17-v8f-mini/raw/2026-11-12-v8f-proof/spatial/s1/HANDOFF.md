# HANDOFF spatial/s1 (state-file mode, note N8). Tool: `s1_read.py`; rules: `S1_DEFS.md`. CPU only.
PY = C:/Users/Admin/venvs/tanitad/Scripts/python.exe ; run from F:/TanitAD-scratch/v8proof/spatial/s1 ; CUDA_VISIBLE_DEVICES=-1 ; PYTHONPATH not needed (the tool adds s0e, s0, s0c, res itself).

State file = output of pod/probe/extract_states.py (`<name>.npz` + `<name>.npz.json`, train, --lo 600 --hi 1800).

1. One state file (HIER seed, mean 128):
   `PY s1_read.py run --states <h1.npz> --tag HIER1 --expect-dim 128 --expect-state mean --ckpt-sha256 <sha256 of the arm ckpt_last.pt>`
   G seed (grid 640): `--expect-dim 640 --expect-state grid`; W seed (wide mean 640): `--expect-dim 640 --expect-state mean`.
   -> RESULT_S1_<TAG>.json, VERDICT_S1_<TAG>.md, OOF_S1_<TAG>.npz, stages/<TAG>/ (resumable). Exit 2 = a control missed (no label).
2. Contrast of two runs (A minus B, same draw): `PY s1_read.py pair --a RESULT_S1_G1.json --b RESULT_S1_HIER1.json` -> PAIR_S1_G1_minus_HIER1.json/.md
3. Replicate floor over the HIER seeds: `PY s1_read.py spread --results RESULT_S1_HIER1.json RESULT_S1_HIER2.json RESULT_S1_HIER3.json --name HIER` -> SPREAD_S1_HIER.json/.md
4. Smoke (label blind; swapped label tracks): `PY make_smoke_states.py noise --clips 300 --dim 128` then
   `PY s1_read.py smoke --states smoke_states/noise_D128.npz --expect-dim 128 --expect-state mean --clips 300 --folds 1`
5. Tests (synthetic only, 8 tests, about 25 s): `PY -m pytest -q --basetemp F:/TanitAD-scratch/v8proof/spatial/s1/tmp_test/b test_s1_read.py` (the folder tmp_test must exist first).

Before the first real run
* `RESULT_S0E.json` must exist with `controls_pass` true: control t2 reads its two mean-state values at run time and fails otherwise. The first real run also computes and caches t2 (`cache/t2_<key>.json`, key = sha256 of s0e_read.py, INDEX.json, s1_read.py); later runs reuse it.
* Run one process at a time; the stage 0e job has priority on this box. Peak RAM: see the figure in the hand-back.
* `--ckpt-sha256` is optional but give it: without it that item of t4 prints NOT CHECKED.
* No outcome name is assigned. The label is "baseline (trained mean state), <TAG>" (mean, width 128) or "trained state, <TAG>".
* A contrast (note section 4) is separated only if it holds at both seeds AND exceeds the HIER spread from step 3.
Named gaps: eval-bank reads not in this tool (S1_DEFS G1); intervals do not carry training variance.

Timing (ESTIMATED from the stage 0e job log, per fold: r-a 208 s for 12 probes, change 223 s for 8 fits; NOT measured on this tool, the real-data smoke was skipped because S0E_DONE/STOP did not exist): D=128 about 25 to 30 min per state file (8 r-a + 6 change fits per fold, 5 folds); D=640 about 40 to 50 min. First real run adds the cached t2 (about 8 to 10 min) and a 1 to 2 min load. Run `smoke` once after S0E_DONE to replace these estimates.
