# FEASIBILITY_WMPOOL (2026-11-10, read-only measurements; CPU, nothing launched)

## 1. Shape of the adapter output per frame (MEASURED, code)
* `F:/TanitAD-scratch/egowm/tree/stack/tanitad/models/v8f_mini.py:322-326` `V8FMiniConfig`: `d = 256`, `d_feat = 1024`, `n_tok = 40`. Adapter = `Linear(LayerNorm(1024), 256) + pos[40,256]` (lines 961-963). Grid `pool4x10` = 4 x 10 = 40 tokens (zcache.py header: `GRID = "pool4x10"`, `D = 256, 40 tokens`).
* So per frame the adapter output is [40, 256]; the WM reads only `adapt.mean(2)` -> [256] (`z1` line 890-894, `z1_from_mean` 895-903, `forward_mean` 924-929).
* The ladder rig trains on `--wm-zcache` (`data/v8f_wm_windows.py` `_zmean`, lines 76-86; `item` line 125-131): per-clip `<clip>.npz` with `zmean` f32 [T,256], `ok` bool [T], `t_s` f32 [T]. Built by `theory/ladder/zcache.py` (`zmean_of`, mode `build`), one cache per frozen M2b adapter (s0, s1).

## 2. What is cached today (MEASURED)
* `ssh tanitad-box2 type D:\tanitad\zcache\MANIFEST.json`: 4,572 train + 141 eval clips per seed, D 256, T 96..105, T_mean 100.945 (hist 101: 8,636 files; 100: 686; 104: 84; 96: 18; 105: 2). Adapter sha256 s0 9c54ed20..., s1 c5cf95c5....
* Actual size, M2b_s0/train: 4,572 files, 478,276,158 bytes (0.478 GB) (`Get-ChildItem ... | Measure-Object Length -Sum`). Matches 4,572 x 100.945 x 256 x 4 B = 472.6 MB + headers.
* Training use: `summary.json` of every EGOWM / ladder run: n_clips 4,512 (60 excluded), n_windows 40,358, 19,164 steps, batch 64, F = 91-frame windows. `C:/Users/Admin/t2v8f_pulls/egowm_*/summary.json`.
* Build time of the mean cache: `theory/ladder/HANDOFF_ZCACHE.md`: 0.26 s/clip cold (first 450 clips), both seeds in one pass, ETA ~25 min; `ZCACHE_FINISH.log` ends 11:15:28 VERIFIED OK (launched ~10:55, so ~20 min).

## 3. Grid cache size (COMPUTED from the above; ESTIMATED until built)
frames per seed = 4,572 x 100.945 = 461,521 (4,512 trained clips: 455,465).
| cache | tokens | dtype | per seed | both seeds (s0, s1) |
|---|---|---|---|---|
| today, mean pool | 1 | f32 | 0.48 GB measured | 0.96 GB |
| full grid | 40 | fp16 | 9.45 GB | 18.9 GB |
| full grid | 40 | fp32 | 18.9 GB | 37.8 GB |
| 2 x 10 (pool rows only) | 20 | fp16 | 4.73 GB | 9.45 GB |
| 2 x 5 (2 x 2 avg pool) | 10 | fp16 | 2.36 GB | 4.73 GB |
| 1 x 5 | 5 | fp16 | 1.18 GB | 2.36 GB |
Ratio to today's mean cache: 40 tokens fp16 = 19.8x (fp32 39.6x); 10 tokens fp16 = 4.9x.
Per-step read at batch 64 x 91 frames: full grid fp16 = 64 x 91 x 40 x 256 x 2 B = 119 MB/step; 2 x 5 = 29.8 MB/step. Seed-2 replicate needs the s0 adapter's cache (argvs_egowm.txt: E-KIN_rep_s2 uses `M2b_s0` init ckpt and zcache), so no third cache.

## 4. Where it fits (MEASURED today unless marked)
| place | free | RAM | verdict |
|---|---|---|---|
| box 2 D: | 11.23 GB (`Get-PSDrive D`, 2026-11-10; the "about 10 GB" INHERITED figure held) | 34.2 GB total, 17.4 GB free | one seed of full-grid fp16 (9.45 GB) fits only if nothing else is on D:; two seeds (18.9 GB) do NOT. 2 x 5 fp16 (4.73 GB both) fits. |
| box 2 C: | 11.84 GB | | same as D: |
| dev C: | 12 GB (`df`; the 27 GB MEASURED 22:44 figure is stale) | | the raw feature cache is on box 2 only; building here is not possible |
| dev F: | 22 GB | | only a stage for a built cache, not a build site |
| dev E: | 7.4 GB | | no |
Box-2 disks (`Get-PhysicalDisk`): one NVMe SSD 1 TB and one Seagate Backup+ USB HDD 1 TB. Which letter is which is UNVERIFIED; the code comment says clip files sit on a USB HDD on box 2.
I/O: full grid needs 119 MB/step; at the measured ~0.46 s/step (ladder pilot) that is ~260 MB/s, beyond a USB HDD and only held by page cache (one seed's 9.45 GB fits in 17.4 GB free RAM, UNVERIFIED under a concurrent job). 2 x 5 needs ~65 MB/s and 4.7 GB total: fits page cache.

## 5. Cheapest route if the full grid does not fit
1. Pre-pool to 2 x 5 (average 2 x 2 patches) at cache-build time, fp16: 2.36 GB per seed, same builder (`zcache.py zmean_of`, one extra pool), ~20 min to build. Loses fine spatial detail, keeps left/right and near/far halves. This is the route recommended.
2. fp16 full grid for ONE seed at a time (build s0, train s0 and rep, delete, build s1, train s1): 9.45 GB, tight on D:; rule 5 of the auto-delete CLAUDE.md applies.
3. On-the-fly adapter forward from `cache_full` features: rejected by the ladder card log (pilot 3.29 s/step, 157.6 h for nine runs; MEASURED `ladder/` log of 2026-11-08 15:47).
4. Store only the frames the levels use (L1 reads 11 of 91 at stride 1 plus K=3 history); not designed here, UNVERIFIED saving.

## 6. Run time per arm (MEASURED seconds, ESTIMATED total)
`summary.json` seconds, `C:/Users/Admin/t2v8f_pulls/`: E-KIN s0 8,832; s1 9,156; E-NOW s0 9,161; s1 9,057; R35H s0 8,636 (2.4-2.5 h of steps; the EGOWM card's ~3.3 h includes ~43 min start-up). The pool arm adds a pool on [64, 91, n_tok, 256] per step: negligible compute at 10 tokens, small at 40, but the loader cost is the risk (section 4). ESTIMATE 3.3 h per arm at 2 x 5; unknown at 40 tokens. Three arms (s0, s1, rep s2) = ~10 h at 2 x 5, plus cache build ~0.5 h.
