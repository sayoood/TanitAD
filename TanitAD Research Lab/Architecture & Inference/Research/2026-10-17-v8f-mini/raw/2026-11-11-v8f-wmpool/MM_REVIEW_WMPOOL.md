# MM_REVIEW_WMPOOL: Master Mind review of the WMPOOL code half (2026-11-11 ~03:20, before any WMPOOL cache build or run)

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
