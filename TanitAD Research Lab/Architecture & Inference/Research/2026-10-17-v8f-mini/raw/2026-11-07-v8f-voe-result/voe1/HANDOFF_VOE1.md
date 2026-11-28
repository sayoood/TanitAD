# HANDOFF_VOE1 (agent, 2026-11-07; nothing staged, committed or pushed)

Card `../PREREG_V8F_VOE1.md`, frozen prefix 5,822 bytes `555c5a38...` (re-hashed, card unedited). IMPL_CHOICES_VOE1.md frozen 15:10:20, sha256 `848a2c8c...` (in `IMPL_CHOICES_VOE1.sha256`) BEFORE any statistic. Tier T0.

## State (final)
* **The run is INVALID by the card's own controls.** `RESULT_VOE1.md` prints INVALID and no label table (as IMPL_CHOICES section 7 committed). Control 1 FAIL: RI_s0/L3 real-vs-real accuracy 0.469 [0.444, 0.493], two-part separated below 0.5 (the other 11 cells are not). Control 2 FAIL: RI_s1/ACTION-SWAP/L3 accuracy 0.550 [0.515, 0.582], reference 97.5 % 0.533, rank 0.005 = NOTICES (the other 11 cells are not). Control 3 PASS (ACC 2.9 %, NULL-ACC 1.7 %, DIFF 2.8 %, RVR 3.9 %, AP-CHANCE 1.7 %, AP-NULL 0.8 %). Control 4 PASS (13 tests).
* Run: `run_voe1.py` rc=0 in 216 s (`VOE1_RUN.done`); raw `RESULT_VOE1.json`, `traces_voe1.npz`.
* Post hoc, labelled as such (`diag_controls_voe1.py` -> `DIAG_CONTROLS_VOE1.md/.json`; reads control rows only): the controls as worded ('at any level', 12 cells) fail a clean pipeline often: in the run's own 20 permutation draws at least one of the 12 cells was separated in 7/20 (control 1) and 8/20 (control 2) draws (independent-cell expectation 33 %). The failing control-1 cell does not persist when the later copy is 0.4 to 1.0 s later (0.504, 0.505, 0.493, 0.481, none separated). The failing control-2 cell is not marginal (rank 0.005). Not a verdict; the card's INVALID stands.
* Read-invalid view (NOT admissible): `python report_voe1.py --read-invalid` -> `RESULT_VOE1_INVALID_READ.md` (all tables, banner on top) and `fig/*.png` (title banner). No M2b manipulation row was read by the agent.
* DECISION NEEDED (Master Mind): (i) amend controls 1 and 2 post hoc (the model is `../AMENDMENT_LATSEM_1/2.md`), e.g. a family-wise rule, then re-run `python report_voe1.py` (no recompute: the JSON stays); or (ii) keep INVALID and run the cheap next lever below.
* NEXT LEVER (cheap, not run: context budget): extra random-init seeds 2..9 for tags RI (control rows only; `build_bank_voe1.py` with more tags, then `eval_tag(cal=False)`): it measures how often a random-init world model trips control 2 on the real data, instead of on permutations. Blocker for running it here: none except budget.

## Rebuild commands (re-creatable, ~2.5 min + 5 min)
```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 WMACT_CODE_ROOT="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tip_1246c8dd"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY build_bank_voe1.py   # bank_voe1.pt
$PY counts_voe1.py       # counts_voe1.json
$PY -m pytest -v -p no:cacheprovider test_voe1_synth.py > PYTEST_VOE1.log; sha256sum voe1_core.py test_voe1_synth.py > PYTEST_VOE1.hashes
$PY run_voe1.py --workers 6
```
Code export tip_1246c8dd lies in the session scratchpad (re-create from the mirror, see `../wmact/HANDOFF_WMACT.md`).

## Post-freeze code edits (bug fixes only, no definition changed)
* `test_voe1_synth.py`: the real-stack test needed `torch.no_grad()` around `encode` (test bug, 13/13 pass). `voe1_core.py`: `ap_cell` also returns the bootstrap draws `boot_apc` (an extra output; no statistic changed); core hash at the run `c8211855...`, pytest re-run on that hash.

## Deletions (own temp, re-creatable, none the only copy)
`voe1/bank_voe1.pt` 44,282,983 bytes (rebuild: `build_bank_voe1.py`, 140 s; sha256 was `84a08d6b...`), `voe1/__pycache__` two .pyc (40,639 + 7,483 bytes) and the directory. Nothing outside `voe1/` was touched. `bank_voe1.report.json` (the reproduction check and the re-init report) stays.
