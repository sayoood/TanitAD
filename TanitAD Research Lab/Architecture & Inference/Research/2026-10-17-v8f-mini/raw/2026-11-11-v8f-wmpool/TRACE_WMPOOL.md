# TRACE_WMPOOL (section 2 of IMPL_CHOICES_WMPOOL.md, filled in AFTER the patch; IMPL_CHOICES_WMPOOL.md itself is unchanged since its sha256 was taken)
Patched tree `F:/TanitAD-scratch/wmpool/tree` = `git apply wmpool.diff` on commit `1faa1a068f9f46dcb9d230c6b05efc20dce2fc1c` (sha256 of the diff in `wmpool.diff.sha256`). The file:line of every hop is also MEASURED by `controls_wmpool.py` (control 2, `hops`).

| hop | where |
|---|---|
| argparse `--wm-pool` | `stack/scripts/v8f_mini_train.py:318` |
| gate entry (every flag needs one) | `v8f_mini_train.py:123` (`FLAG_CHECKS["wm_pool"]`) |
| namespace -> config (the one line; control 2 mutates it) | `v8f_mini_train.py:418` `wm_pool=a.wm_pool` |
| refusal before any build (static half) | `v8f_mini_train.py` `pool_cache_problems`, called first in `main` |
| config field + validation | `stack/tanitad/models/v8f_mini.py:201`, the `wm_pool` clause of the `V8FMiniConfig` validation |
| model attribute: the query | `v8f_mini.py:703` `self.pool_q = nn.Parameter(torch.zeros(cfg.d))` (att25 only) |
| data: dataset gets the kind | `v8f_mini_train.py:3108` `V8FMiniWMWindows(..., zcache=a.wm_zcache, pool=a.wm_pool)`; record `info["wm_windows"]["pool"]`, `["zcell_shape"]` of the first real item |
| data: cells loaded / served / collated | `stack/tanitad/data/v8f_wm_windows.py` `_zmean` (`zcell` f16), `item` (132), `collate_wm` (149) |
| batch -> loss | `v8f_mini.py:1251` `wm_only_loss`: `self.wm.forward_cells(batch["zcell"].float().detach(), ...)` |
| forward | `forward_cells` -> `z1_from_cells` (815) -> `pool_weights` / the pool line (785, 786) -> `_z1_post` (`proj`, `enc1`) -> `stack.loss` |
| live-token spelling (full path / beta-style callers) | `WMAux.z1(adapt)` -> `cells_of_adapt` -> `z1_from_cells` (equal to the cached spelling, control 1: 0.0) |
| G-DVB read-back | `bad_wm_pool` (`declared_vs_built`), `WMAux.built_pool`, `wm_pool_readback` in `selfcheck.json levers` |

## Known limits (UNVERIFIED)
* `main()` end to end on real data, `--selfcheck-only`, and the full G-DVB path against a real cell cache were NOT run (the raw feature cache, the clip files and the v9 label files are on box 2). `bad_wm_pool`, `pool_cache_problems`, `wm_only_live_check`, the dataset/collate and `main()`'s refusal were run on built models, stub stores and a fixture manifest.
* The `stack/tests` suite was not exported or run here. The patch adds no test file. `launch_gate.py` was not touched; the gate's own G-DVB entry for `wm_pool` is its owner's.
* Which image direction rows and columns are in the 4 x 10 grid is UNVERIFIED (the feature extractor is not in the tree); the cell map is the programme's own 2 x 5 reader convention.
* Not bit-exact at zero query (measured gap in `CONTROLS_WMPOOL.md`).
