# TARGET_DEFS: what was built for the level-1 predictor levers EGOV and FUTTOK (2026-11-14)

Base: `C:/Users/Admin/v8proof_c/motion/tree_patch` (the MOTION result: residual / roll / diag levers on top of CELLPRED). CPU only, nothing trained, no git, no pod.
Paths relative to `stack/`; line numbers are those of `patched/` (the tree_patch files). Tests: `tests/test_v8f_target_levers.py`.

## Lever A: EGOV, `--wm-l1-act {acc_yaw,speed}` (default `acc_yaw` = today, bit for bit)

* **Field used for the third channel: `wm_speed`**, a new per-frame item / batch key, [F] per item and [B, F] per batch, float32, unit **m/s, unscaled**. It is built in
  `tanitad/data/v8f_mini_data.py:589` (`item`) as `np.nan_to_num(d["v"][idc])`, where `d["v"]` is the cache's logged ego speed array of the clip (`ClipStore.get`, lines 500 and 505:
  `"v": o["v"].numpy()`, the same `v` that `ra1_pooled.py:128` stores as the third raw action channel); `idc` is the index of the F cached frames the item already uses for `feats` and `act`.
  Collated as float32 at `v8f_mini_data.py:639`. It is logged ego motion: never a label of another agent, never a model output. It is an extra key: no earlier key changes.
  The key is present in every batch (items do not know the config); the model reads it only when `wm_l1_act == "speed"` (`v8f_mini.py:1285`).
* **Scale**: `WM_SPEED_SCALE = 10.0` (`v8f_mini.py:166`). The third channel = `speed / 10.0`, at the START frame of each transition (the action of transition j -> j+1 is built from frame j: `WMAux.forward`
  slices `[:, :-1]` after `scale_actions`, exactly as for the first two channels).
* **ONE spelling**: `WMAux.scale_actions(act, speed=None)` (`v8f_mini.py:864`). Off: returns exactly today's [..., 2] tensor (a `speed` passed anyway is not read). On: [..., 3]; `speed=None` raises
  `ValueError("... built with --wm-l1-act speed ... no `speed` was given ...")`; a wrong shape raises. `WMAux.forward(adapt, act, speed=None, feats=None)` passes it through.
* **Read-tool helper**: `tanitad.models.v8f_mini.wm_frame_speed(batch_or_item)` (`v8f_mini.py:750`) returns `batch["wm_speed"]`; `V8FMini.loss` calls the same function (line 1286). Missing key = `KeyError` with a message.
* **Other callers**: `v8f_beta.py` (lines 229, 255) calls `wm.scale_actions(...)` without speed: unchanged when the lever is off (file byte-identical, pinned), and it raises the clear error above on a speed-built model.
  `HJEPAStack.rollout` takes already-scaled level-1 actions, so a read tool builds `wm.scale_actions(act, wm_frame_speed(batch))` and passes it: width 3 for a speed model.
* **Build**: `WMAux.__init__` builds level 1 with `action_dim=3 if wm_l1_act == "speed" else 2` (`v8f_mini.py:777`). Levels 2 and 3 are unchanged (they read the 32-wide embedded level-1 actions).
* **Gate read**: `built_levers()["wm_l1_act"]` = `"speed"` iff the BUILT `stack.act_embed[0].in_features == 3` (`v8f_mini.py:1386`); `bad_wm_state` refuses declared != built.

## Lever B: FUTTOK, `--wm-l1-futtok-k K --w-wm-futtok W` (defaults 0 and 0.0 = off, bit for bit)

* **Config** `wm_l1_futtok_k: int = 0`, `w_wm_futtok: float = 0.0` (`v8f_mini.py:228-229`). Valid: `(K == 0 and W == 0.0)` or `(2 <= K <= hmax - 1 and W > 0, finite)`; on needs `wm_state == "grid"`
  (`ValueError("... needs wm_state grid ...")`); needs the WM auxiliary. Same checks in the stack (`LevelConfig`, level 1 only) and in the gate (`bad_wm_state`).
* **Head**: `WMAux.tok_head = nn.Linear(wm_cell_dim, d_feat)` (`v8f_mini.py:792`), built only when the lever is on (`tok_head = None` otherwise: no key in the state dict). One map for every cell.
  `d_feat` = `cfg.d_feat` (the width of the cached token, 1024 on the pod; the adapter's input width).
* **Term** (`tanitad/models/hjepa.py`, `_futtok_loss`, line 986): for starts `s in range(0, T - K, 2)` (the starts of `roll_pairs`), the rolled-out states of steps 1..K from the one true start state `z[:, s]`
  (the shared `_roll_embedded`, predictor context rule of the roll loss) are reshaped to `[B, K, n_tok, cell_dim]` (cell-major, the inverse of `WMAux.z1`, `_cells`, line 980), passed through `tok_head`,
  and compared by squared error with `F.layer_norm(feats[:, s + k].float(), (d_feat,))` (no affine parameters, built in `WMAux.forward`, `.detach()`ed). Mean over features, cells, steps, starts and batch
  (mean of per-start means; every start has the same element count). The start state keeps its graph (gradient reaches `proj` and the adapter through it).
* **Entry into the loss**: `level_loss = level_loss + c.futtok_weight * futt` (`hjepa.py:893`), next to the roll line; it is inside `T["wm"]`, so the trainer weights it by `w_wm` exactly as it does the roll term (the card says "mirror it": done). The latent one-step term is unchanged.
* **Shared rollouts**: `HJEPAStack.loss` keeps a per-call cache `rcache` keyed by K; ROLL and FUTTOK with the same K call `roll_pairs` once, different K once each (test `test_roll_and_futtok_with_the_same_k_share_one_rollout...`).
  `_roll_loss` gained an optional `rolled=` argument; its value is unchanged (test).
* **Inputs of the stack loss**: `loss(..., tok_head=None, tok_tgt=None)`: both or neither; given while `futtok_k == 0`, or missing while on, raises.
* **Logging**: `T["wm_futtok"]` (= `futtok/L1`, unweighted, detached) whenever K > 0 (`v8f_mini.py:1292`); G-LIVE requires it (`v8f_mini_train.py:1241` and `:2610`). With `--wm-l1-diag on`: `wm_futtok_copy` = the same head on the START state `z[:, s]`
  against the same future targets, `no_grad` (`hjepa.py:895-897`, `_futtok_copy`). The diag flag leaves every weight equal (t8).
* **Gate read**: `built_levers()` reports `wm_l1_futtok_k` and `w_wm_futtok` from the stack's loss config, and `0` / `0.0` whenever the head is not built (`v8f_mini.py:1387-1388`).
* **trainable_groups**: unchanged. `tok_head` is a child of `WMAux`, which is the whole `"WM"` group, so the gate's per-group gradient check sees it.
* **Checkpoints**: new `TARGET_CFG_DEFAULTS` (`v8f_mini.py:168`), `wm_l1_act / wm_l1_futtok_k / w_wm_futtok`. They are written into a checkpoint cfg only when they differ from the default; `INIT_CKPT_MATCH` has all three
  (absent = default); earlier checkpoints load unchanged. (A separate dict, not an extension of `MOTION_CFG_DEFAULTS`: the MOTION test asserts that dict's exact key set, and that test file stays byte-identical.)

## Decisions the card did not fix

1. **`wm_speed` is always collated** (an extra key in every batch), not only when the lever is on, because items do not know the config. No logged value, loss or weight of an earlier arm changes (t1).
   It is not listed in `EGO_PERTURB_RECORD`: that constant is pinned and tested. Under the DAgger / ego-perturb stages (not used by arms PE/PF/PEF as specified) `wm_speed` stays the logged true speed, like `act`.
2. **RNG stream of PE**: building level 1 with `action_dim=3` makes `act_embed[0]` 32x3 instead of 32x2, which shifts the global random stream for every module built after it INSIDE the WM stack (and the WM
   projection). The main model (everything built before `WMAux`) is identical to P's at init (test `test_egov_builds_the_rest_of_the_model_unchanged_in_init` checks the shapes that differ: the action embedding, the IDM output
   layer and the planning-prior buffers of level 1). So PE and P are not the same init for the WM; a same-seed pair differs by a re-draw of the WM, as any change of a layer width does.
3. **IDM at level 1 (wm `b`)**: the level-1 IDM head's output is sized by `action_dim`, so with EGOV on it also predicts the speed channel. The arms run wm `b` (gamma 100); this makes the IDM term ask for the speed
   as well. Not avoidable without changing the stack's IDM; named here. With wm `a` (gamma 0) it is not trained.
4. **Planning priors** (`prior_*_1`, width 3 with EGOV) keep their defaults (N(0,1), no clipping); no planner of these arms uses them.
5. **FUTTOK needs the cached token grid in the batch** (`feats`); an `--encoder lora` batch carries none, and FUTTOK then raises a clear error (the card's arms use the cache).
6. **Reduction**: mean over features, then the plain mean over cells, steps, starts and batch, regardless of `pred_reduction` (the card says "mean over features, cells, steps, starts and batch").
7. **Target precision**: `feats` (float16 from the collate) is cast to float32 before the layer norm.
8. **The tolerance in t4** is 1e-6 (the MOTION bar), read 5.1e-7. It is not 0.0 because the loss embeds the actions of all 15 transitions at once and slices, while `HJEPAStack.rollout` embeds the K-step slice alone:
   the same embedding layer on two matmul shapes differs by 1.3e-7 (`checks/t4_note.json`); the rollout core itself gives 0.0 on identical embedded inputs.

## Not built as written

Nothing differs from the card, apart from the notes above. Not run: real data, a pod-size model through the trainer, `v8f_t1_eval.py` and the read tools (e1, e2 of the card belong to those tools).
