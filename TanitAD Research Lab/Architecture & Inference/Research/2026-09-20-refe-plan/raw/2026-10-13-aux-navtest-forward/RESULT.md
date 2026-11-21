# aux_navtest_forward -- the auxiliary head's outputs on navtest frames (tool built; trunk path VERIFIED on 32 of 1,058 C3 tokens; the full pass is NOT run)

Date 2026-10-13 (Berlin). Tier: T0-like (a head on teacher-forced frames; never driving performance). Evidence class: MEASURED, artifacts below.

## Headline

1. **The ctx path is the planner's, bit for bit, on 32 tokens.** Scorer logits from the tool's forward == `raw/2026-10-11-c3-selection-headroom/table.npz["logits"]`: **32/32 tokens bit-identical, 14,336 values, max |d| 0.0**, at c3_run's grouping (one `planner.infer` per token, batch 1, rule `navsim_v1_lane`, u400 md5 `8e382eeb...`). `identity.json`. This is 32 of 1,058, one log; the other 1,026 are unverified.
2. **The identity check bites.** Swapping camera slots 1 and 2 before the trunk, on the real path: 0/4 tokens identical, 1,792 values differ, max |d| 1.717, exit 2 (`identity.json`, `DELIBERATE_REGRESSION...`).
3. **The full 1,058-token pass was not run** (the brief's bar: projected <= 15 min). Projection 53 min (warm, 3.0 s/token) to 5.0 h (cold, 10.8 s/token + 54 s per log); c3_run's own 2026-10-11 dump ran 2.15 s/token = 38 min. All three are over the bar. `HANDOFF.md` has the exact chunked command (`--max-trunk-tokens`, resumable, each run holds the GPU lock ~10-20 min).
4. **No edit to `scorer_finetune.py`, `aux_drivable_head.py`, `aux_s1_readout.py` or any PREREG file is needed.** (`scorer_finetune.py` already carries the `--dump-scores` head dump from `PATCH_head_pred_dump.diff`.)
5. **A never-trained head already reads P3 pooled AUC 0.82 on the 32-token slice** (scorer 0.80 on the same 32; one log, so the CI is degenerate and the number is not a rate). See section 5; this is for the PREREG owner.

## 1. The S1 trunk is frozen, so one trunk pass serves every arm (quoted, `refe/scorer_finetune.py`)

- `:583-588` `for p in model.parameters(): p.requires_grad_(False)` then `for m in SCORER: ... p.requires_grad_(True)`; `SCORER = ("score_q_mlp", "score_dec", "score_head")` (`:71`). `backbone`, `registers`, `reg_compress`, `scene_proj`, `pos3d_mlp`, `dec`, `traj_head`, `ego_enc` are not in it. `visual_ctx = scene_proj(visual)` (`model.py`), so it depends on frozen modules only.
- Optimisers: `optA` over `SCORER` params of `model` (`:1281`, `:1283`), `optB` over the `armB` scorer copy (+ H2's feed projection) (`:1282`, `:1284`), the head's own `optH` over `aux.head.parameters()` (`:613`). No trunk parameter is in any of them.
- The preflight gate G1 requires `trainable_names(model) == the scorer's names` (`:1097-1099`).
- The head reads `args[1]` of `model.score_dec[0]` (hook `:603`, `cap["sctx"]`); this tool installs the same hook (`CtxTap`) and a test pins it to the second `scene_proj` call (= `visual_ctx`).
- Same checkpoint load as the planner (`ckpt_io.config_for_checkpoint` + `load_for_inference`), same image preprocessing (`train.py:808` and `planner._image_for`: `cv2.resize`, BGR->RGB, /255).

## 2. What it runs

`eval/aux_navtest_forward.py` drives `refe_navtest_seam`'s own loop (`NS.build_scenarios_for_log` -> `REFePlanner.initialize` -> `_image_for` -> `set_driving_command` -> `infer`), the invocation `c3_run.py` used. Nothing is re-implemented: ctx, scorer logits and head outputs come from ONE `infer` call. Heads: `--heads ARM=aux_drv.pt` (`aux_drv/1` format, built by the trainer's `AUX.build_head` from the file's own config, loaded **strictly**; `tap scene`, wrong format, N arms refused). Output `head_pred_<ARM>.npz` = the readout's INPUT CONTRACT (`log`, `token`, `logit` f32 [N,40,32], `sdf_m` f32 [N,40,32] metres = the head's own `6 tanh(.)`, `step` = 0) plus `meta_json`. Rows are in `table.npz` token order (`tokens.json` lists the same set in another order). `--ctx-cache` stores fp32 ctx (7.9 MB/token) + the scorer logits, so a head that exists later needs no trunk. GPU lock: `wait_idle.py` (fail closed) + O_EXCL `{pid, owner, t}` + memory.used <= 3,500 MiB after 4 s; released in `finally`. Lock held for ANY GPU use, a cache-only head pass included.

## 3. Smoke (32 C3 tokens, one log `2021.09.16.19.27.01_veh-45_01749_03230`, fresh head seed 0)

| | |
|---|---|
| identity | 32/32 bit-identical, max \|d\| 0.0 (`identity.json`) |
| head | `H2-s0`, **freshly initialised, never trained**, depth 2, tap visual, 2,440,450 params, sha256 `af19121c...` (`fixture_fresh_head_seed0/aux_drv.pt`, dev box only) |
| GPU memory | peak allocated 2.41 GB, reserved 3.02 GB (nvidia-smi 4,873 MiB incl. ~1.7 GB desktop), RTX 4060 |
| cold cost (first touch of the log) | 12.5 s/token total over 32: scenario init+route 3.2, image 2.3, trunk `infer` 3.2, heads + trainer-path 0.4, cache write 1.7; plus 54 s one-off scenario build |
| warm cost (3 tokens, same log) | 3.5 s/token incl. first-call CUDA warm-up: init+route 0.63, image 0.29, trunk 1.85, cache 0.19; per run: `wait_idle` ~47 s, model load 7 s, scenario build ~1 s/log |

Controls (final code): a chunked run (`--max-trunk-tokens 3` twice, so 3 cached + 3 fresh) and a head-only pass over a complete 6-token cache are **bit-equal** (logit and sdf_m) to the same rows of the 32-token inline run, and identity is 6/6. The 32-token cache pass (earlier cache format) was bit-equal on all 32. Both locks released (checked). The 32-token run itself used the code before the per-token-logits cache edit; the trunk path did not change.

## 4. Precision caveat (UNVERIFIED for a trained head)

The planner path is fp32 (no autocast in `planner.py` / `model.py`); the S1 trainer runs the trunk under bf16 autocast, so a trained head saw bf16 ctx. MEASURED on the 32 tokens: ctx rel L2 (planner vs trainer path) 1.05 % / 1.27 % / 1.46 % (min / mean / max), max abs 0.117 on a mean abs of 0.140. With the fresh head the outputs moved by at most 0.0099 (logit) and 0.070 m (sdf), 0.06 % of cells flipped mask sign (`smoke.json`). A fresh head says nothing about a trained head's sensitivity. `--head-ctx trainer` reproduces the trainer's precision (second trunk pass, no cache) if the owners want that read.

## 5. Plumbing runs of the readout (numbers MEAN NOTHING: a random head, 32 tokens, one log)

- Exact brief command with the full C3 table: `ZZAUXS1_PERCEPTION_OK`; P1/P2 computed on the 32 frames (all 32 labelled in the navtest store); `p3` = `NOT_COMPUTED` ("1026 of 1058 C3 tokens are not in this arm's head predictions") -- the readout refuses a partial P3 rather than reading it. `readout_plumbing_full_c3table.json`.
- Same command on a 32-token slice of the C3 table (`readout_plumbing_slice32.json`): P3 pooled AUC head **0.8245**, scorer 0.8005, difference +0.024 (CI [0.024, 0.024]: a single log, so the log-cluster bootstrap has nothing to resample), decision `UNCOMMITTED_BAND`; K2 fires (IoU 0.20). **Reading:** the min predicted SDF along a footprint depends on how far and how wide a proposal reaches, and a never-trained head already has a position-dependent SDF field (learnable BEV queries + Fourier position features), so it separates DAC 0 from 1 above chance with no image at all. On 32 tokens it is within 0.03 of the scorer. If that survives at 1,058 tokens, the P3 bar `max(0.90, scorer + 0.05)` is partly a position-prior bar, and a never-trained head (or the label-shuffled arm) should be read on the same tokens as a floor. Not established here (n = 32, one log).
- The labels checked: store sha256 `985a3bc7...` (navtest, 12,146 keys).

## 6. Tests (`eval/test_aux_navtest_forward.py`, `ZZTEST_PASS 21`, CPU, ~4 s warm / 140 s cold)

11 checks + 2 real-store + 8 RED mutations (each snippet occurs once; each named check goes RED): (a) cached ctx == inline ctx (exact) and the tap IS the `visual_ctx` call; (b) zeroed output layer -> logit exactly 0.0 and sdf exactly 0.0, with the un-zeroed control reading non-zero; (c) seed 0 != seed 1, same seed bit-equal, fixture == `AuxDrivable(cfg).head` tensor for tensor (and `meta` keys == `aux_meta()`); (d) the npz round-trips `aux_s1_readout.load_head_pred` exactly and, on the real pinned stores, passes `--labels-set navtest` while an npz carrying a TRAIN key is refused (rc 3); sdf is metres (saturated channel reads 6.0); identity is bitwise (one-ulp perturbation fails); camera-slot swap fails the identity; strict loading refuses a missing key, an extra key, a wrong format, tap scene, a depth mismatch; the lock protocol (live pid never taken, dead+old removed, dead+fresh respected, memory > 3,500 refused, idle waiter fails closed). Mutations: `strict=True->False`, `array_equal->allclose`, swap no-op, tap reads args[0], `step` != 0, no `O_EXCL`, idle waiter ignored, sdf/6.

## Manifest

| artifact | where |
|---|---|
| `eval/aux_navtest_forward.py` | repo |
| `eval/test_aux_navtest_forward.py` | repo |
| `raw/2026-10-13-aux-navtest-forward/{RESULT.md, HANDOFF.md, smoke.json, identity.json, smoke_run.log, launch.py, readout_plumbing_full_c3table.json, readout_plumbing_slice32.json}` | repo |
| fresh seed-0 head `fixture_fresh_head_seed0/aux_drv.pt` (9.8 MB, regenerable: `aux_navtest_forward.make_fresh_head_file(path, seed=0)`), `smoke32/` (`head_pred_H2-s0.npz`, `run.json`), `smoke32_ctx/` (32 x 7.9 MB), `chunk6*`, `cachepass/`, `swap4/`, logs | **dev box only**: `D:/Projects/TanitAD-artifacts/aux_navtest_forward/` |
