# HANDOFF: P6 selector runner (PREREG_P6_SELECTORS, frozen; sha256 d4778ca4...c690b checked by the tool)

CPU only. Built and tested on the dev box with synthetic packs of the exact `fit_pack.npz` format. Never run on the real pack (does not exist yet) and never run on the pod. No git writes were made: the Master Mind lands `selectors/`.

## Files (`p6base/selectors/`)
`selcore.py` (features, 9 variants, bootstrap, ledger; no E:/F: dependency) · `fit_selectors.py` · `read_selectors.py` · `packtools.py` (test/smoke pack builders) · `test_selectors.py` (22 tests) · `navtest_tokens_e3.txt` + `.sha256` (the 12,146 E3 tokens; the fit refuses if any pack token is in it).

## Input layout (`fit_pack.npz`, from `labels/p6_labels_tool.py cmd_pack`)
`log token traj_xy[N,64,8,2] yaw[N,64,8] logits[N,64,6] pdm[N,64,7 = nc dac ep ttc c ddc pdms] has_pdm`. Optional `v0[N]` (measured ego speed). Sets without finite labels are dropped (counted; >2 % refuses).

## Commands (dev box; venv `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`)
```
cd E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/selectors
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -m pytest -q test_selectors.py --basetemp F:/tanitad_scratch/p6sel/pt        # 22 pass, about 25 s
# 1. fit (detached, WMI "cmd /c ... < NUL" per the dev-box memory)
$PY fit_selectors.py --pack <fit_pack.npz> --out <OUT> --threads 4 [--min-free-gb 6] [--only R1,C_platt]
#    pod: copy selectors/ (without tests), add --no-ram-wait; needs numpy scipy sklearn torch only
# 2. E2 read, then E3 read (a read of a selector is counted in ledger_K.json)
$PY read_selectors.py --stage e2 --nominees <OUT>/nominees.json --e2-table <p6base/e2/e2_table_p6final.npz> --out <RES>
$PY read_selectors.py --stage e3 --nominees <OUT>/nominees.json --e2-table <same> --e2-result <RES>/selectors_read_e2.json --out <RES> \
      [--full-table F:/TanitAD_p6read_work/tactical/table_p6final.npz] [--changed-csv NAME=csv]
```
Without `--full-table` the E3 read ends `INCOMPLETE` and writes seams plus `needs_scoring.json` (same layout as `read_scorer1p6.py emit-changed`); score them with the harness and pass `--changed-csv NAME=csv`.
Outputs: `OUT/split.json+.sha256` (written before the first fit), `variants/<id>/{params.pkl,meta.json,pick_fit_b.npy}`, `nominees.json+.sha256`, `FIT_REPORT.md`, `R_candidates_fit_a.json`; `RES/selectors_read_{e2,e3}.{json,md}`. Re-running the same command resumes by variant.

## Expected runtime at 8,489 sets (MEASURED on a synthetic 8,484-set pack, 4 threads, dev box; the real pack has the same shape, but the real features may cost differently)
R1+R2 19 s each (one 401-candidate table each) · C 1 to 3 s each · S_gbdt_pdms 5 s, S_gbdt_heads 15 s · S_mlp_heads 119 s · P_pg 133 s. Whole fit 316 s (5.3 min, > 4 min: launch detached). Reads: seconds to a minute (the E3 family tables dominate).

## Choices where the card is silent (all in the code and the read JSON)
R = the two best of the 401 candidates by FIT-A (so FIT-B never picks among 401). S_mlp_heads = one MLP per head NC, DAC, EP, TTC (comfort and DDC kept at P6's value). `rank01` uses tie-correct average ranks, not the probe's argsort-of-argsort. Policy gradient is unweighted (no stratum weights in FIT). S3 "loses" = paired point difference < 0 on E2 or E3. See `AMBIGUITIES` B1 to B9 in `read_selectors.py`.

## UNVERIFIED
1. The pack carries no ego speed. Without `v0` the tool uses a proxy (median over the 64 proposals of the first 0.5 s chord) at fit AND read, recorded in `v0_source`. Feeding `v0` into `cmd_pack` (as in `dump_p6.npz` `ego[:,6]`) would remove this; the speed-binned and ratio features depend on it.
2. `traj_xy` is assumed to be the 8 poses at 0.5 s of `props8` (shape is enforced, the time grid is not provable from the pack).
3. The real E2 table file name (`e2_table_p6final.npz`, from `build_e2_table.py`) and the full 64-proposal table layout (`token pdms sub valid`, as `table_p6.npz`) are assumed; both are gated against P6's banked csv before any read.
4. Never run on the pod or on the real FIT labels. The smoke on the 1,058-token stand-in ran end to end (fit, E2, E3, ledger, control) but no number from it is a result; its outputs are stamped `SMOKE_NOT_A_RESULT`.
5. G2 here is the overspeed-share bar on E3 cruise tokens (the card's table); the old card's G2 (regret) is not used.
