# HANDOFF_LATSEM2B (agent, 2026-11-07 14:18, dev-box CPU). DONE: the read ran once, all controls pass. Nothing staged, committed or pushed.

Card `../PREREG_V8F_LATSEM2B.md` prefix 4,611 bytes `bf6b3717` and Amendment 1 prefix 2,778 bytes `120072ff` re-verified untouched. Card Log has two new lines by this agent: 13:44:00 (IMPL_CHOICES `a90cd9ca`, before any statistic) and 14:16:58 (result). Tier T0. The previous agent's label-source table and disclosures live in `IMPL_CHOICES_LATSEM2B.md` section 2.

## Result (details `RESULT_LATSEM2B.md`, numbers `RESULT_LATSEM2B.json`)
* Controls 1, 2, 3, 4 pass on TRAIN and eval-141. C3 shares declared separated (TRAIN / eval-141): S1 classification 2.78 / 2.50 %, S1 continuous 0 / 0 %, S2 3.57 / 3.02 %, S3-PARTIAL 0.83 / 0.83 %. The interval alone would declare 58 % of S2 rows on permuted probes.
* **SCENE-AWARE at L1 only**, through agents within 30 m: TRAIN +0.357 / +0.329 (s0 / s1) separated; eval-141 +0.308 / +0.231 same sign (eval-141 s1 interval includes 0). Lead gap is separated at both seeds at no level.
* **No SCENE-IMAGINING** (TRAIN lead gap, longest horizon: at most one seed separated). **No ACTION-SEMANTIC-LEAD**: S3-PARTIAL is positive (+0.72 to +3.30 m TRAIN), the wrong sign, every level and seed.
* Card's next lever: the reliance arm on L1.
* Runtime: timing gate 10.5 s per draw, projected 20 min; actual 1,566 s (26 min) for both datasets, 4 worker processes x 1 thread.

## Notes for the Master Mind
* Reproduction check vs run 1 (eval-141, same probes): binary and continuous rows agree to 1e-14 (S2 continuous 2e-7); the 7-class `lon7` rows differ up to 3.3e-4 (S1) and 2.4e-3 (S2), most likely lbfgs stopping with another BLAS thread layout (not tested). `lon7` is not a card outcome target.
* R2 reference thresholds are negative (-0.94 to -0.15), so the reference adds little for continuous S1; it binds for classification, S2 and S3.
* Disclosed in IMPL_CHOICES: TRAIN label alignment peaks at -1 raw frame; L1 shrink fitted on TRAIN (in-sample); 7 TRAIN clips without labels; 6 gap frames without ego speed.

## Re-run (Git bash, this directory; about 26 min; draws resume from `draws_*.jsonl`, delete them for a fresh draw set)
```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 WMACT_CODE_ROOT="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tip_1246c8dd"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -u run_latsem2b.py < /dev/null > run_latsem2b.run1.log 2>&1   # asserts card, amendment, IMPL hashes
$PY report_latsem2b.py                                           # regenerates RESULT_LATSEM2B.md
```

## Manifest (all in `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/latsem2b/`, unstaged; sha256 first 12)
IMPL_CHOICES_LATSEM2B.md `a90cd9ca9190` (+ .sha256); run_latsem2b.py `1fca3df36a73`; report_latsem2b.py `41266e96053a`; test_synth_latsem2b.py `bc41f5924d9b` (synthetic smoke test of the code paths); RESULT_LATSEM2B.json `6fddb156d006`; RESULT_LATSEM2B.md `3cda1ebb6a3d`; draws_train.jsonl `ebf5377089ae`; draws_eval141.jsonl `bdd7ddfcb887` (4.4 MB each, the 219 permutation draws per dataset); run_latsem2b.run1.log; fig/pca_levels_gap_train.png `41dfb56f9e17` and fig/pca_levels_gap_eval141.png `56db7a9ba964` (no camera frame, landable); labels_latsem2b.py, labels_latsem2b_train.npz / .report.json / .log (previous agent, unchanged). Also edited: `theory/PREREG_V8F_LATSEM2B.md` log lines only.
Nothing in `theory/latsem2/` or `theory/wmact/` touched; no .pyc written (`sys.dont_write_bytecode`). Box 2, pods and busy markers untouched.

## Deletions (own scratch, re-creatable)
`latsem2b/newblock_scratch.txt` (1,538 bytes, a text block spliced into `report_latsem2b.py`). Nothing else created outside this directory; the pinned `tip_1246c8dd` export belongs to others and stays.
