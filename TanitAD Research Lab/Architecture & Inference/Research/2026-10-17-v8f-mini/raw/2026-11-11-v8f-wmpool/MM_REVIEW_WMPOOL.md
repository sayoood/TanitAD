# MM_REVIEW_WMPOOL: Master Mind review of the WMPOOL code half (2026-11-11 ~02:45, before any WMPOOL cache build or run)

Card `PREREG_V8F_WMPOOL.md` (frozen 2026-11-11 01:18:11, sha256 `01e4dd80...`); code by a Sonnet agent (`IMPL_CHOICES_WMPOOL.md`, sha256 `639f2b5d...`, written before the code; `wmpool.diff`, sha256 `fb208bc7...`).

## 1. Controls rerun by the Master Mind
`controls_wmpool.py --tree F:/TanitAD-scratch/wmpool/tree --base <fresh export of 1faa1a06>` (CPU, 2 threads, 15.6 s): `all_pass` true. Compared field by field with the agent's `CONTROLS_WMPOOL.json`, the one difference is the output directory named inside a refusal message (control 5). `wmpool.diff` applies to the fresh export of `1faa1a06` (`git apply --check`), and the agent's tree reverse-applies it.

## 2. Base commit
`1faa1a068f9f46dcb9d230c6b05efc20dce2fc1c`: R35H s0's banked `gate_token.json` names it (agent, re-read in `WMPOOL_BASE_COMMIT.txt`), and EGOWM's patch sits on the same commit. The patch carries no `--wm-speed`, as the card's arm table requires (everything except the pool as R35H).

## 3. Cell layout: VERIFIED from the cache builder (the agent left it UNVERIFIED)
* `pull_encode_pooled.py` `pool_field` (repo: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py`, about line 255): the DINOv3-L/16 field is a row-major 16 x 40 token grid; `pool4x10` is `reshape(T, gh, 16/gh, gw, 40/gw, D).mean((2, 4))`, cells row-major.
* `raw/2026-10-25-viz/attn_extract.py` (docstring, "measured from the cache builder"): the frame is 256 x 640 (cylindrical); token `r*10 + c` is the pixel rectangle y in [64r, 64r + 64), x in [64c, 64c + 64).
* So `r` is the image row (top to bottom) and `c` the image column (left to right). `cells_of_adapt` (`adapt.reshape(B, F, 2, 2, 5, 2, d).mean((3, 5))`) makes cell `R*5 + C` the 128 x 128 px block y in [128R, 128R + 128), x in [128C, 128C + 128): the top or bottom half of the frame, in five sectors from left to right.
* The index map was checked by hand: 40 = (2, 2, 5, 2) gives token `20R + 10r' + 2C + c'` = row `2R + r'`, column `2C + c'`.

## 4. Design as built (read in the diff)
One zero-initialised query `pool_q` [256]; weights `softmax(cells . q / sqrt(256))` over the 10 cells, the cells themselves as keys and values (no key or value projection); then the existing `proj` and the same `enc1` tail. 256 new parameters. At q = 0 the weights are 0.1 and z1 equals `proj(token mean)` within 3e-7 (fp32 cells; fp16-stored cells 3.7e-5), not bit-exact, stated as the card allows. A P-ATT model refuses `z1_from_mean`, so an unchanged reader fails loudly rather than silently reading the mean pool.

## 5. Open before any WMPOOL run (none is a code change to this patch)
1. The `launch_gate.py` G-DVB entry for `wm_pool` (its owner; this review does not edit it), and the suite (`stack/tests`, G-SUITE) on the patched tree: not run here.
2. Box 2: the cell cache (`zcells.py build`, `manifest`) and control 3 on 8 real clips (`zcells.py check3`).
3. The read side (`READSIDE_WMPOOL.md`, about 35 lines in `lr_lib` / `lp_lib`).
4. A commit carrying the patch for the box-2 tree ship. `stack/tanitad` is held by another session (RefCV9) until it reports K2M1N20 launched, so the patch is banked here as a diff, as EGOWM's was.
5. The box-2 slot after STOPBAL is the PI's ordering call.

## 6. Box-2 package (`box2/`, Sonnet agent; reviewed 2026-11-11 ~03:25, before any WMPOOL run)
* `PACKAGE_MANIFEST.sha256`: 27 files, `sha256sum -c` clean. Unit tests rerun by the Master Mind: `UNIT ok=379 bad=0`, as the agent's `tests/UNIT_RESULT.log`. The stub dry run (153 ok, 0 failed, 13 min) was not rerun.
* Chain: wait for STOPBAL's end (DONE, or a STOP that is not a busy / GPU stop, rc=12), then ship (tree of `abc01481`), cells (`zcells.py build`, `manifest`, `check3` = card control 3), pilot (200 steps of P-ATT_s0, gates all three argvs, 72 h rule), P-ATT_s0, P-ATT_s1, P-ATT_rep_s2. GPU idle line `WMPOOL_GPU_IDLE_MIB`, default 3,500 MiB.
* Package constants fixed before any data (the card states no tolerance): check3 recomputed-cell identity 1e-5; stored fp16 cells vs the cached token mean 5e-3 max abs per element (`wmpool_check.py:49`); the measured values are recorded either way.
* **Named risk, box-2 D: space (ESTIMATED):** D: read 10 GB free at the OBD pilot (2026-11-11 01:24, chain log). OBD and STOPBAL leave 14 checkpoints of 47-74 MB each on box 2 (banked sizes). The cells stage asks for 10 GB (cache 2 x 2.36 GB + the 5 GB a run needs), and after the cache each run still asks for 5 GB. If D: is short when WMPOOL's turn comes, the cells stage refuses before building anything, and a run that finds less than 5 GB refuses before training; the measured D: free then decides between building one seed's cells at a time and removing box-2 copies of runs whose banks are verified.
* Item 5.4 is done: the patch commit `abc01481` (tree = `1faa1a06` + `wmpool.diff`, checked by a fresh apply) is on branch `agent/v8f-wmpool-20261111`, verified by `ls-remote`.
* Cost note: the package agent used ~373k tokens against a ~150k brief; a package of this size is split into two agents next time.

## 7. Read side (2026-11-11 ~07:15)
Item 5.3 is done: `READSIDE_WMPOOL.md` section "Implemented" (lr_lib / run_ladder_read / lp_lib, controls `CONTROLS_READSIDE.json` pass with three RED arms, a mean model bit-identical to the unedited readers, `test_ladder_read.py` 13 passed). Still open: 5.1 (G-DVB entry and suite, the gate's owner) and 5.5 (the PI's slot). The OBD pilot was relaunched on box 2 at 07:04 after a Windows Update restart (01:29) killed it; the WMPOOL chain waits for OBD and STOPBAL.

## 8. Box-2 disk thresholds set from measurements (2026-11-11 ~07:35, before any WMPOOL stage ran)
The section 6 risk, measured instead of met: box 2's D: read 10.36 GB free at 07:30 with OBD running; OBD and STOPBAL leave about 14 x 47 MB, so ~9.7 GB at WMPOOL's turn (ESTIMATED). The package asked 10 GB for the cells stage (9.72 on the box side) and 5 GB per run: the cells stage would have stopped, and after the cells (4.72 GB) no run could start. Both values were stated constants (a checkpoint ESTIMATED at <= 1.5 GB). MEASURED on box 2: an EGOWM run's `ckpt_last.pt` is 46,851,930 B, `D:\tanitad\runs\v8f_egowm` 226 MB for five runs; the token-mean cache `D:\tanitad\zcache\M2b_s0\train` 478,276,158 B (fp32), so fp16 cells are ~2.39 GB per seed, as the package's 2.36 estimate. New values (`wmpool_check.py`, `stages/wmpool_common.sh`): a run asks 1 GB; the cells stage asks cache + 1 GB = 5.72 GB (6 on the dev-box side). Tests changed with them (the pinned 9.72 / 10 and the 9 / 9.5 GB refusal cases become 5.72 / 6 and 5 / 5.5 GB; one check added that both files say 1 GB per run). Rerun by the Master Mind: `UNIT ok=380 bad=0`, `DRYRUN ok=153 failed=0`; `PACKAGE_MANIFEST.sha256` regenerated over the same 27 files (`sha256sum -c` clean). Expected D: after the cells ~4.9 GB and after the three runs ~4.8 GB, under the 5 GB sweep line while the runs train; the cells (re-creatable from the zcache and the M2b checkpoints) are deleted by the Master Mind once the three banks and the read are verified.
