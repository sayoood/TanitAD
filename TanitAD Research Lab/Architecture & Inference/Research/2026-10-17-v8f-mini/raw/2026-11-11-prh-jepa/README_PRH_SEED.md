# README_PRH_SEED

Card: `PREREG_PRH_SEED.md`. sha256 checked before use: **411ddb7215e279213664c88118488a2f448a60f755dbb42d9780713f645388ec** (current). The first frozen wording had `a616e4395ac343f93f70fea3cf0c108f5b1d765cbc5dcf90d5ae1dfdadf07a65`; the coordinator reported the change as two words, with no rule, bar or control changed. Both shas are in `RESULT_PRH_SEED.json` meta.

## What ran
`prh_seed.py` on CPU (2 threads), zero training, nothing run on box 2 (three small files per seed read by sftp only). Wall time MEASURED: extraction 29.7 s (141 eval clips, 14,237 frames, 4 adapters through the model's own `adapter(f)+pos`, mean over tokens), analysis 8.1 s (B = 2000). Environment: read tree 202849da, `tanitad.__file__` asserted under it.

## MEASURED results (artifact `RESULT_PRH_SEED.json` / `.md`, this directory)
Outcome line the harness prints: **O2 PLATONIC-OTHER**, but **a control failed (D-MIX), so the read is unusable and must not be interpreted** (card rule). Numbers for the record, evidence class MEASURED:
* M(TR0,TR1) 0.6773 [0.6718, 0.6809]; M(RI0,RI1) 0.6344 [0.6274, 0.6398]; P1 difference +0.0429 [0.0360, 0.0489].
* M(TR0,SLOW) 0.5756, M(RI0,SLOW) 0.5617, M(TR0,PCA) 0.7563; M(TR1,SLOW) 0.5743, M(RI1,SLOW) 0.5617, M(TR1,PCA) 0.7595. P2a diffs +0.0139 and +0.0126 (separated); P2b diffs -0.1807 and -0.1852 (separated, negative: PCA is closer than SLOW).
* Controls: C-SELF 1.0 pass; C-ROT |diff| 3.9e-14 pass; C-SHUF M -0.0009, 10-NN 0.0069 vs chance 0.0068 pass; C-ZC 6 clips max abs diff 1.9e-6 pass; **D-MIX FAIL**: M(TR0,DMIX) 0.6796 vs M(TR0,TR1) 0.6773, delta +0.0023, paired CI [-0.0044, +0.0092] (not separated). The harness does report the change (the point values differ); it cannot resolve it with the separation test I chose as the operational pass rule.
* Diagnostic (no bar): M(TR0, a pure random affine map of nm) = 0.6464 [0.6405, 0.6510], close to M(TR0,TR1) and above M(RI0,RI1)'s lower edge. The M metric sits on a high floor (about 0.63 to 0.65 for unrelated affine maps of the same features), so its resolution to a partial replacement is small. This is the likely reason D-MIX is not separated.
* Beside rows: Procrustes R2 TR0/TR1 0.240, RI0/RI1 0.073; mutual 10-NN 0.889 vs 0.873 (chance 0.0068); profiles along SLOW in the JSON (256 values each). Ridge 1e-3 sensitivity (point): 0.6728 and 0.6355.

## Choices the card left open
All listed in `RESULT_PRH_SEED.json["choices"]` (and in the `.md`). Main ones: M ridge 1e-6 x mean eigenvalue; "separated" = 95 % paired percentile CI lower bound above 0; D-MIX pass rule = paired CI of the M difference excludes 0 in either sign (my own operationalisation, fixed before the data were read; I did not move it after seeing the failure); C-SHUF permutes TR1 within both halves; SLOW rep = projection of nm on the fit-half x-side canonical directions; profile defined as a least-squares readout of each SLOW direction. `ok` computed exactly as `zcache.py` (get_kin + kin_features).

## UNVERIFIED
* Whether a different D-MIX scaling or a lower-floor metric would pass; not tried (the card's bars are not moved).
* The per-direction profile reading (Fig. 6 analogue) was not interpreted.
* The nm-affine identity was only checked numerically (max abs 5.7e-6, TR0).

## Tests
`test_prh_seed.py`: 7 passed (same subspace M=1, independent subspace low, rotation invariance, bootstrap point equals direct, separation rule, outcome logic for O1/O2/O3 and the all-four P2 rule). Mutation check (`tests/mutation_check.py`, record `tests/MUTATION_PRH_SEED.json`): flipping `is_sep_pos` makes the tests go RED (clean rc 0, mutated rc 1).

## Deleted temp data (own, re-creatable or pulled copies with source on box 2)
* `F:/TanitAD-scratch/prh_seed/work/extract.npz`, 116,894,913 bytes (re-created by `prh_seed.py` in 30 s)
* `F:/TanitAD-scratch/prh_seed/work/zc` (6 box-2 npz copies), 628,002 bytes
* `F:/TanitAD-scratch/prh_seed/work/sftp_cmds.txt`, 960 bytes
(The mutation scratch dir was removed by its runner.)

## Manifest
All in `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/prh/`: `prh_seed.py`, `test_prh_seed.py`, `README_PRH_SEED.md`, `RESULT_PRH_SEED.json`, `RESULT_PRH_SEED.md`, `tests/mutation_check.py`, `tests/MUTATION_PRH_SEED.json`. Nothing staged or committed.
