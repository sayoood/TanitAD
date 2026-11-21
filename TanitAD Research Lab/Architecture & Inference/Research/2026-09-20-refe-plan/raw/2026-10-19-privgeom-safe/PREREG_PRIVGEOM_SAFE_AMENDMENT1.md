# PREREG PRIVGEOM-SAFE, AMENDMENT 1: the fit / analysis code pins and the post-extraction stage definitions

Sealed BEFORE the gate stage, the first stage that opens a stored label array. It changes no decision rule, margin, constant, seed, scalar or recipe of `PREREG_PRIVGEOM_SAFE.md` (sha256 `c6d6e1fb228f7e4806d5b0316699f8fef05855684d0c535bd246248217bd7084`, sealed 2026-10-19 10:39:42); it pins the code that implements them and states how the stages are run.

## A1.1 Code pins (sha256)

| file | sha256 |
|---|---|
| `code/pg_geom.py` (as the prereg) | `5b78a694f40b6b2b4030fdeb3bf5be2c65b349c8887f9c11c78d0dfd8d9df7de` |
| `code/pg_extract.py` (as the prereg) | `64a148906329f8aeba5dc3d87d30aba9f2e91d90389ea77c0169614e1bade23b` |
| `code/pg_select.py` (as the prereg) | `51f85a63e5ffcfe4f4355004da17fb2e9fc3f808609532b2d3b01137657e03d6` |
| `code/pg_lib.py` | `61bfe59692987de6e43cf9d2585c2992192a4b9dca6d513688bffc5c77dd7178` |
| `code/pg_run.py` | `7297a1b4a3e51e1cdc784eac340dd6d334f225ff6d2e81ab6529a9d197cae9d0` |
| `code/test_pg_geom.py` (as the prereg) | `3cf48766adc17b4c1ac07bb1b043e3798ffa2a680226196f124fc69f56406b64` |
| `code/test_pg_lib.py` | `fd1d408aed2d870f86c9a02a556b32ce0f54708042c25823666c425f06b89cfa` |
| `code/finish.sh` (orchestration only; reads no label) | `71cd244f9570fb8ee3ec3a2aa88798e6a573884eadb191a7385b0e1199a1799f` |

The gate token binds these hashes, the subset, the join arrays and the labels array it writes; `fit` and `analyze` refuse if any differs.

## A1.2 Stage definitions (as built)

- Interpreter for every post-extraction stage: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe` (shapely 2.1.2 is needed for `pg_geom`'s no-contact variants; torch 2.7.1), `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=8`, at most 2 Python processes.
- **join** (label-free): `run/scalars/<log>.npz` -> arrays aligned to the census indices; sets whose extraction failed are dropped and counted; the census `z6` must equal the pickrule package's `z6` (and `log_idx`) on every retained set, else C-JOIN fails. Reads no `labels6` and no `lab`.
- **gate** (opens `labels6` for the retained sets only; the first label read): C-AGREE (the recomputed verdicts `rl6`, clipped to [0, 1], against the stored NC and DAC columns, exact equality, >= 99 % each); the label-identity rates of prereg 3.4; C-REACH on the real arrays (G := the row's label, arms L+G and G-only, both rows, both seeds: F-AUC >= 0.99, 8 fits); `test_pg_geom.py` and `test_pg_lib.py` (RED mutation table, C-ID-L, C-REACH, a mute-G-block mutation, C-CONV within 0.01 of an oracle, C-NULL false-fire on synthetic noise, C-POWER). A FAIL of C-REACH, a suite, the branch table or C-JOIN is a gate FAIL (no fit starts). A C-AGREE failure does NOT stop the fits: it VOIDs the row (VOID-INPUTS) in `analyze`.
- **fit**: 40 tasks (per row, per seed: L, L+G, G-only, the companion variant L+G(V), the weak plant, 5 null replicates) of 5 folds each; resumable; the planted-label fits are the gate's. Companion variants: DAC-nosign = `|clearance|` of `dac_clear_min` and the 9 key times (10 columns, `dac_frac_out` dropped); NC-floored = `nc_min_dist` and `nc_min_tgap` floored at 0.5, `nc_min_ttc_margin` floored at 0.0 (prereg 3.4).
- **analyze**: per row, per seed: F and Q1-3 AUC with CIs for L, L+G, G-only, L+G(V); `Delta_F` and its call; null fire list and VOID; the weak-plant call; `final_call` = `pg_lib.final_call(raw call, void_null, void_inputs, power)` (VOID-INPUTS over VOID over the UNDERPOWERED conversion of a NO-JUMP; INFO-UPSTREAM is never converted by the power control). The pick-level read of prereg 8: composites P2, EP, EP[L], EP[L+G], EP[L+G nc only], EP[L+G dac only], EP[G-only]; the EP column is EP-R1's c* OOF column of the pickrule package; picks by the pinned `pickrule_lib.rule_pick`; label-lookup PDMS by `pickrule_lib.pdms_from_labels`; bootstrap draws identical to the primary read.
- **Extraction**: two worker processes claim logs through `O_EXCL` lock files in `run/scalars/`; `finish.sh` waits for 41 `.npz` and no locks. Each worker pauses (scripted) below 3 GB available RAM.

## A1.3 Disclosures made before sealing

1. **A plumbing smoke ran before this amendment**: `join`, `gate`, `fit`, `analyze` on the first two finished logs (295 sets, `run_smoke/`), with `PG_SMOKE=1`, which makes the gate use the RECOMPUTED NAVSIM verdicts (`rl6`) as labels and NEVER the stored `labels6`. It tested code paths only; its numbers are tiny-sample numbers, are not part of any result, and no rule, constant, margin or seed was changed after seeing them. I did look at the table it produced (NC INCONCLUSIVE, DAC INFO-UPSTREAM on 295 sets with 1 to 2 batches per epoch); the engineering fixes made after it were the fits directory creation, the interpreter choice (driverl-eval) and the smoke flags.
2. Synthetic results obtained before sealing (`test_pg_lib.py`, 4,000 sets, the registered recipe): the L+G arm reaches F-AUC 0.8810 against an oracle 0.8842 and L 0.7737 (so lr 1e-2 for the G block is not mute); the same arm with the G block at lr 1e-3 FAILS the C-REACH bar (RED); a permuted block's `Delta_F` is -0.0001 to +0.0002 (0 of 3 fires); a weak plant gives `Delta_F` +0.0736 [+0.0704, +0.0771].
3. At sealing, 4 of 41 logs were extracted (about 12 minutes of one worker under contention); no extracted scalar and no recomputed verdict has been compared with a stored label.
