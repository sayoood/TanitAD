# README_PRH_SEED_V2

Card `PREREG_PRH_SEED_V2.md`, sha256 861f0a05e19518903e367da64b73e6ec1333592b6d51f49a9798f01f1a03814d (checked). Code: `prh_seed_v2.py` (modes `calib`, `read`; reuses `prh_seed.py`). Same environment as v1 (CPU, 2 threads).

## Step A (controls only), MEASURED, `CALIB_PRH_SEED_V2.json`, sha256 4a3ca969aa740a4ce0d6f44f031852a2828c1aa9fb99b9a073878ea71e405678 (printed before Step B)
Points (SELF / FLOOR / MIX25 / MIX50 / MIX75 / SHUF), admissible, choice ratio:
* CC8: 1 / 0.9943 / 0.9998 / 0.9988 / 0.9970 / 0.0001, admissible, 6.09
* CC32: 1 / 0.9777 / 0.9989 / 0.9942 / 0.9880 / 0.0004, admissible, 9.85
* PROC: 1 / 0.0801 / 0.4749 / 0.1439 / 0.0590 / -1.1836, admissible, 2.68
* CKA: 1 / 0.9249 / 0.9934 / 0.9331 / 0.7978 / 0.0019, NOT admissible (MIX50 - FLOOR not separated), 0.38
Choice rule as written: largest ratio is CC32 (9.85); PROC (2.68) is not within 10 % of it, so **chosen metric S = CC32**. CIs are in the JSON.

## Step B (once), `RESULT_PRH_SEED_V2.json` / `.md`
Outcome line: **O2 PLATONIC-OTHER**, all controls pass (C-SELF, C-ROT 1.4e-14, C-SHUF S -0.0032 and 10-NN at chance, C-ZC 6 clips max abs diff 1.9e-6 re-pulled this run, D-MIX2 separated: delta +0.00064, CI [0.00015, 0.00114]).
Values are in the result file; P1 +0.0169 [0.0155, 0.0186]; P2a +0.0087 / +0.0084 (separated); P2b -0.0464 / -0.0458 (separated, negative).

## Choices left open (also in the JSON): see `CHOICES_V2` in `prh_seed_v2.py`
R_x maps are `nn.Linear(1024,256)` default init under torch seeds 200/201/300 on the LayerNorm-affine of nm, rescaled to TR0's total fit-half variance; PROC whitening ridge 1e-3; CC ridge 1e-6; SHUF permutes within both halves.

## UNVERIFIED / caveats
* CC32 has a floor of 0.978 for two random maps and D-MIX2's effect is separated but small (0.0006); the rule was applied as written, the dynamic range of the chosen metric is narrow. PROC had far wider range on the controls but a lower ratio.
* Profiles along SLOW were not interpreted.

## Tests
`test_prh_seed_v2.py` + `test_prh_seed.py`: 13 passed. Mutation checks (`tests/mutation_check_v2.py`, record `tests/MUTATION_PRH_SEED_V2.json`): dropping admissibility clause 3, flipping clause 4, flipping clause 2 each make the tests RED.

## Deleted temp data (own; re-creatable or copies of box-2 files)
* `F:/TanitAD-scratch/prh_seed/work/extract.npz` 116,894,913 B
* `F:/TanitAD-scratch/prh_seed/work/zc` 628,002 B
* `F:/TanitAD-scratch/prh_seed/work/sftp_cmds.txt` 960 B
Nothing staged or committed. Box 2: sftp reads only.
