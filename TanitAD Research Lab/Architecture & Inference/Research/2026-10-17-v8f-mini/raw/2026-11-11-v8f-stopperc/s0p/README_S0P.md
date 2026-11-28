# S0-P probe pipeline (card `PREREG_V8F_STOPPERC.md` section 4 "S0-P" + section 8; `AMENDMENT_1_S0.md`)

Zero training of T. Linear / MLP probes on frozen DINOv3 features. Tier: a diagnostic of what T's inputs carry, never driving performance.
`E`'s `limit` / `limit_ok` = the CAUSAL bucket (past speed, `CAUSAL_STEPS` 20) on both sides (Amendment 1).

## Files
| file | role |
|---|---|
| `s0p_lib.py` | sampler (`draw_sample`), E layout (18 = p10f 8, nav 5, limit 4, limit_ok 1; E-limit = first 13), AUROC, tree set-up, lazy-grid loader |
| `s0p_extract_eval.py` | step 1, eval side (DONE, banked `F:/TanitAD-scratch/stopperc_s0p/s0p_eval.npz`, sha256 0c25fe72...fb39) |
| `s0p_extract_train.py` | step 2, train side, for box 2 CPU (BUILT, tested locally, NOT launched) |
| `s0p_fit.py` | step 3: `prep` (PCA) / `fit` (CV + refit + score eval once) / `read` (controls, bootstrap, outcomes) / `all` |
| `s0p_smoke.py` | step 4: end-to-end CODE CHECK on eval arrays only; never a result |
| `test_s0p.py` | 13 tests, every guard with a regression arm |

## Step 1 eval arrays (MEASURED, verified this session)
2,365 windows, 131 positives, 139 episodes = 139 clips (STOPCUE's counts). Controls in the npz meta: E speed vs the dumps' first-tick `v0` max |diff| 0.0 over 2,399/2,399 grid windows (the dumps hold no other E field); second route (ClipStore.get, J["nav"], causal_bucket) max |diff| 0.0 on all four E blocks; NOW pool4x10 vs the stored fp8 row 0.0; `has` / `stop_hit` / (clip index, ws) equal STOPDEC's npz of BAL-NULL_s0 AND _s1; causal `limit_ok` False fraction 0.0 on eval. File complete (sha256 recomputed equals the banked one). Reused, not regenerated.

## Step 2 run (box 2, Master Mind launches; box 2 is not touched from here)
```
set "OMP_NUM_THREADS=2" & set "CUDA_VISIBLE_DEVICES=-1"
D:\tanitad\venv\Scripts\python.exe <copy of s0p>\s0p_extract_train.py --tree D:\tanitad\repo --cache-dir D:\tanitad\cache_full --split train ^
   --join D:\tanitad\runs\v8f_mini_full\join_train.npz --tracks D:\tanitad\cache_full\native_tracks.pt --out-dir <OUT>\s0p_train --eval-cache-dir D:\tanitad\cache_full
```
Launch detached with WMI `cmd /c ... < NUL` (memory note). Copy `s0p_lib.py` and `s0p_extract_train.py` together (the script imports `s0p_lib` from its own folder). Add `--dry-run` first: pool + sample + counts only (writes `sample.json`), reads no grid rows.
* Refuses unless the pool equals the banked prevalence: 286,035 observed / 13,116 positive / 272,919 negative (`--expect-pool`). Refuses an overlap of train and eval clip ids (`D.assert_disjoint`; needs `<eval-cache-dir>/eval/manifest.jsonl`). Stops with `NO-GRID` if `pool8x20` or `mean` is absent from the train cache (card S0-PRE: a PI question, no re-encode approved).
* Resumable: `progress.json` every 100 clips; rerun the same command after a cut. Output: a directory of .npy (E, v40, v160, mean, y, clip, ws, eid) + `meta.json`, `sample.json`, `MANIFEST.sha256` (written last = complete).
* Needs `D:\tanitad\repo` to contain `stack/scripts/v8f_t1_eval.py` with `causal_bucket` and `CAUSAL_STEPS == 20` (asserted).
* ESTIMATED size 3.3 GB (N 16,000 x (18 + 40 + 160 + 2 KB-rows): v40 655 MB, v160 2.6 GB, mean 33 MB). Put `--out-dir` on a drive with >= 4 GB free and not on a D: exFAT 1 MiB-cluster top level (a few big files are fine there).
* ESTIMATED runtime 15-45 min. Basis: MEASURED locally 240 windows over 114 real clips (eval cache, SSD) in 8.5 s = 0.035 s/window, all CPU; box 2 adds USB-HDD random reads of ~200 KB per window (16,000 windows, ~3.3 GB; ESTIMATED at 20-100 MB/s random = 1-3 min of transfer plus ~16,000 x 3 seeks) and runs with other jobs reading the same HDD (card: after the dose-2 and SLOWWM chains).
* MEASURED equivalence: on the 56 windows the real-format smoke shared with the eval arrays, the train extractor's E, v40, v160, mean equal step 1's bit for bit (0 differing bytes).

## Step 3 run (dev box or box 2 CPU; ~2 h ESTIMATED at 2 threads)
```
python s0p_fit.py prep --train <OUT>\s0p_train --eval F:/TanitAD-scratch/stopperc_s0p/s0p_eval.npz --work <WORK>
python s0p_fit.py fit  --train ... --eval ... --work <WORK> --arm V160+E --model mlp      # one detached job per (arm, model); 10 of them; each banks fit_<arm>_<model>.{json,npy}
python s0p_fit.py read --train ... --eval ... --work <WORK> --out RESULT_S0P.json
```
`prep` needs RAM for one decoded chunk only (128 windows, ~0.7 GB); projections are 655 MB (v160) + 164 MB (v40). `read` needs `S0P_TREE` (default the STOPBAL read tree) for `taniteval.ci`, and the STOPDEC npz of BAL-NULL_s0/_s1 in `stopbal/read/out/stopdec/`.
ESTIMATED fit times: the smoke fits (N 1,191) took E 2.2 s / 16.5 s, E-limit 0.1 / 14 s, M+E 0.2 / 53 s, V40+E 3.3 / 81 s, V160+E 13.8 / 219 s (linear / MLP, MEASURED); scaled by N (x13.4): linear fits ~0.5-5 min each, MLP E 4 min, M+E 12 min, V40+E 18 min, V160+E ~50 min (every MLP fit hit max_iter 300, so cost is linear in N and in features). Total ~2 h, V160 MLP the long pole; the 10 jobs are independent.

## Readings fixed here where the card is silent (report, do not change the card)
1. Sampler: ONE generator, positives drawn first, negatives second; clip id = the clip-id string sorted lexicographically; `rng.choice(n_pool, k, replace=False)` over the sorted pool.
2. PCA: one label-free fit per grid on the whole train sample (not per CV fold), shared by all tokens of the grid (tokens pooled over positions); the CV selects the dimension. Streaming covariance over all tokens.
3. All features z-scored (`StandardScaler` inside the CV pipeline, on the fold's training part; whole sample on refit), PCA outputs included. Feature vector = [token PCA features | E]; E and E-limit arms have no PCA.
4. Tie-break "smaller model": smaller PCA dim, then smaller C (linear) / smaller hidden then larger alpha (MLP).
5. Train windows: age 0 (NOW = the window start t, the tick frame), which is what `prevalence_stopbal.py` labels and what step 1 reads on eval.
6. HEAD-UNDERUSES: **fixed by `../AMENDMENT_2_S0P.md`** to the PAIRED rule: paired dAUC (V40+E probe - T) >= 0.037 AND its CI lower bound > 0, at BOTH BAL-NULL arms, linear OR MLP (`hu_models` names which). The probe's own CI lower bound vs T's AUROC point (this pipeline's first reading) is reported beside (`probe_lo`, `t_point`) and decides nothing.
7. Controls: label shuffle = window-level permutation of the train labels; episode-shuffled = each clip's label sequence replaced by another clip's (cycled); both refit with the REAL linear fit's selected hyperparameters (not reselected), on arms E and V40+E, scored on eval, CI must cover 0.5. Leak = the GT label appended as a feature on both sides of arm E (>= 0.999).
8. The "E equals the dumps' first-tick inputs max |diff| 0" control can only cover E's speed entry (the dumps' `v0` is the one stored field); the other blocks are covered by the second-route control. Stated, not hidden.
9. Eval-side T1-kin dumps use `--nav deadreckon`; train-side nav is the join's logged nav, as the trainer feeds it. They agree at tick 0 (second-route nav diff 0.0).

## Step 4 smoke (CODE CHECK, NOT A RESULT; clip-grouped half/half of the eval arrays: 1,191 "train" windows / 79 pos, 1,174 "eval" / 52 pos, 0 shared clips)
All 10 fits, the bootstrap (B 2,000) and the read ran end to end. Controls read as designed (label-shuffle and episode-shuffled CIs cover 0.5 on both arms; leak AUROC 1.0), outcome logic returned FLAT. Total wall ~9 min at 2 threads. The numbers carry no meaning (eval data on both sides, 1.2k windows). The report JSON was not kept (scratch removed); rerun `python s0p_smoke.py --eval <npz> --work <dir>` to regenerate.

## Untested
* The train extractor against box 2's real train cache (format MEASURED equal, extractor MEASURED on the eval cache in the same format); `pool8x20` / `mean` presence there; `v8f_t1_eval.py` on box 2's repo; the pool-count assert on the real join; box-2 runtime.
* The fit at full size (memory, wall time): extrapolated only.
