# HANDOFF_SLOWWM (Master Mind, 2026-11-11 ~14:05): code of record built and verified; the box-2 chain package is next

Card: `PREREG_V8F_SLOWWM.md` (sha256 `7c2775e3…94ca`, frozen 13:02:01).

## State (MEASURED)
* **Code of record:** commit `f7e790ab57e10a9004b04b206195a0ae5a5295b8` on branch `agent/v8f-slowwm-20261111`. It is base `1faa1a06` (R35H's ladder code) plus 4 files, +528 / −11, push verified by `ls-remote`:
  * `stack/scripts/v8f_mini_train.py`: the flag `--wm-basis {adapter, slow, pca}` (default `adapter`), the kind/sha refusal, and `wm_basis_readback` written to `selfcheck.json` `levers.wm_basis_readback` when the argv names the flag;
  * `stack/tanitad/data/v8f_wm_windows.py`: serves the basis cache (`zbasis` key) with manifest checks, plus `norm_mean_tokens`;
  * `stack/tanitad/models/v8f_mini.py`: config `wm_basis`, and `built_basis()`;
  * `stack/tests/test_v8f_slowwm.py`.

  Unified diff: landed beside this file as `slowwm.diff`.
* **Tests** (fresh run 13:50–14:00, CPU, tree `F:/TanitAD-scratch/slowwm/tree`):
  * `test_v8f_slowwm.py` + `test_v8f_ladder.py`: **50 passed**;
  * `test_slow_basis.py`: **13 passed**. It needs `stack/scripts` on PYTHONPATH; the first run failed only on that import.
* **Controls** (`controls_slowwm.py` re-run → `CONTROLS_SLOWWM.json`):
  * control 1, off identity: **pass** (722 arrays bit-equal to `1faa1a06`, planted batch, R35H s0's argv);
  * control 2, reachability: **pass**;
  * control 4, echo/hash audit: **pass**;
  * **all 8 mutations red** (M1 default slow, M1b config default slow, M2 config line deleted, M3 loss ignores the basis, M4 proj detached, M5 kind mismatch accepted, M6 trainer refusal removed, M7 mean path not refused).
  * Control 3 on REAL train frames runs on box 2 in the chain; it passed on SMOKE (`smoke/CHECK3_SMOKE.json`, eval clips only).
* **The two Sonnet agents** that built and then tried to finish this: the first ran to 325k context and wrote the code, builder, hook, tests and controls but no docs. The second produced nothing. The MM made the diff, re-ran tests and controls, and committed.

## Interfaces the box-2 chain needs
**Builder** `slow_basis.py` (copy to box 2 with the shipped tree; `ZC_REPO` = the stack tree):
1. `fit --root D:\tanitad\cache_full --split train --tracks D:\tanitad\cache_full\native_tracks.pt --exclude 540:600 --expect-windows 40358 --out D:\tanitad\zbasis\fit --spool D:\tanitad\zbasis\spool`
   * Writes `basis_slow.npz`, `basis_pca.npz`, `SPECTRA.json`, `FIT_RECORD.json`.
   * It streams float64 moments, so RAM is a few MB beyond one clip.
   * `--spool` keeps `nm` per clip as f32: 4,713 clips × T × 1024 × 4 B, ESTIMATED ~1.9 GB (from ~101 frames per clip). It saves a second feature read.
2. `cache --root D:\tanitad\cache_full --split {train,eval} --tracks … --basis D:\tanitad\zbasis\fit\basis_{slow,pca}.npz --out D:\tanitad\zbasis --spool D:\tanitad\zbasis\spool --commit f7e790ab…`
   * `cache` appends `<kind>\<split>` itself, so `--out` is the root `D:\tanitad\zbasis`. Found by the chain builder; an earlier line here named `…\{slow,pca}`, which would double the kind directory.
   * Per clip it writes `zbasis` f32 [T,256], `ok`, `t_s`, `basis_sha`, plus `BASIS_MANIFEST.json` and `basis.npz`.
   * ESTIMATED 2 kinds × ~480 MB each, train plus eval (the token-mean cache is 478 MB per seed for the same clips, MEASURED) ≈ 1.0 GB.
3. `check3` on the 8 clips of `theory/ladder/clips8.json`: control 3 on real frames. A failure stops the chain.

**ESTIMATED box-2 cost of the basis step:** about 20–40 min of CPU (the zcache build of the same clips took ~20 min, MEASURED 0.26 s/clip cold) and ≤ 4 GB on D:. The box has ~9 GB free; STOPBAL and the seed-3 runs need ~5 GB per run while running.

**Arms:** R35H's banked argv (`C:/Users/Admin/t2v8f_pulls/ladder_R35H_s{0,1}/argv.txt`) with exactly these tokens changed:
* `--wm-basis slow|pca`;
* `--wm-zcache D:\tanitad\zbasis\{slow,pca}\train`;
* `--seed` 0 / 1 / 2;
* `--out D:\tanitad\runs\v8f_slowwm\<TAG>`.

`--init-ckpt` stays the M2b checkpoint of the same index: the adapter is unused by z1 under a basis, but the readers and planner are M2b's. SLOW_rep uses seed 2 with the s0 init. Order: SLOW_s0, PCA-B_s0, SLOW_s1, PCA-B_s1, SLOW_rep.

**Gate readback:** `levers.wm_basis_readback` = declared vs built:
* `kind`;
* `basis_sha`;
* `dim` 256;
* `cache_problems []`;
* `planner_identity_probe.pass`.

The checker pattern is `theory/wmpool/box2/wmpool_check.py` `gate_problems`, mirrored for this lever.

**Read side:** `slowwm_hook.py` builds an arm's z1 inputs from frames (basis applied to `nm`) for the planner harness and the IA-0 readers. The read package is still to build.

## Next
1. The box-2 chain package `theory/slowwm/box2/` (template `theory/wmpool/box2/`): wait for the seed-3 chain (`C:/Users/Admin/obd_seed3_state` DONE / non-busy STOP), ship `f7e790ab`, the basis step, pilot 200 steps of SLOW_s0, then the 5 runs, bank and verify.
2. The read package (planner harness + decodability).
