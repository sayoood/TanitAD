# C2 planted positive: tool notes (STAGED, not landed)

Frozen text: AMENDMENT_7 C2 row: a fixed random direction (seed 5) at the median token norm, added to one 4x10 cell of the 416 arrays on positive windows only, train and eval. Bar: V40@416+E AUROC >= 0.99 and outcome RES416-ADDS-40.

## Choices the amendment leaves open (decided BEFORE any planted score exists; MM to confirm)
1. **Cell**: row 2, column 5 of the 4x10 grid, row-major token index 25 (a central cell). Overridable only by editing CELL_ROW/CELL_COL.
2. **Arrays**: v40 only. v160 and mean are untouched; the control tests V40@416+E.
3. **Norm**: median of the L2 norm over every token of S1's train v40@416 (fp8 decoded to fp32). One number for S1, S2 and eval.
4. **Direction**: numpy RandomState(5).standard_normal(1024), unit length, times the norm; added in fp32, re-quantised to fp8 e4m3fn (the builder's to_fp8, finite assert).
5. **Positives**: y == 1, train and eval.
6. **c2_auroc**: min over samples S1, S2 AND over linear and mlp (conservative). `c2_auroc_linear_min` and `c2_auroc_mlp_min` are also in RESULT_C2.json. The amendment names neither family; if the MM wants linear only, copy c2_auroc_linear_min into the key.
7. **Prep**: `prep` PCA-projects only the planted v40 (same s0p_fit.fit_pca / project, same file names); v160 and mean are not planted and not needed by V40+E.
8. **Read**: decide() sees g = 40 only for the planted arm (E and V40@256p+E are the REAL fits in run/work); g = 160 cells are empty, so ADDS-160 cannot occur.

## Disk (F: floor 25 GB, binding until 16:00Z)
Untouched arrays are `os.link` hardlinks (zero bytes); no copy fallback (a failed link raises). Expected extra F: bytes: planted v40 train 655 MB x 2 = 1.31 GB; planted eval npz ~0.49 GB (an npz member cannot be hardlinked); PCA projections ~0.19 GB x 2 + eval 24 MB x 2; fit outputs ~ MB. Total about 2.3 GB. `make` also refuses in Python when free - expected < 25 GB; run_c2.sh refuses below 26 GB before every step. With 31 GB now and ~3 GB for the main fits, the margin is thin (about 26 GB at the end): delete `run/c2/work/*/proj_train_v40.npy` after the fits if needed (re-creatable).

## Time (ESTIMATED, UNVERIFIED)
Only fit_E_linear.json exists (136 s); no V40+E fit time is banked yet. make ~3-6 min, prep 2 x ~2 min (v40 PCA stamped about 1 min at 05:00-05:01 on S1_416), 4 fits maybe 5-25 min each. Plan 30-90 min total at 2 threads.

## Tests
`test_q5_c2.py`: 9 pass. Includes: plant on all windows does not pass; zero amplitude is the identity and reads FLAT; untouched arrays are same-file hardlinks and byte-identical; v40 differs only in the planted cell of positives; link failure never copies; disk floor refuses.
