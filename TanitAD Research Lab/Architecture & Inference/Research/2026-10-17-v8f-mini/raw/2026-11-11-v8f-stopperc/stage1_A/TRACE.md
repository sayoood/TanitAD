# TRACE: arm A (stop-location auxiliary), hop by hop, on the base 715ef8fc

Every file:line below is on the BASE (`715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf`, MEASURED by reading the export), unless marked PATCHED (a symbol this package adds; the diff holds its place).
Evidence class: MEASURED (read from source) unless stated.

## Verdict: every hop exists except ONE, and that one is built here, with a named data dependency

| hop | exists on the base? | where |
|---|---|---|
| stop location label `stop_x_m` / `stop_y_m` in a training batch | YES | below |
| NOW frame's adapted 4x10 tokens | YES | below |
| cylindrical camera model | YES (a tested module, not in the trainer) | below |
| camera intrinsics | YES, a corpus constant: the frame is the fixed 256x640 cylindrical `f_ref` 305.5775 | below |
| **`sensor_extrinsics` in the trainer or the batch** | **NO. It never reaches `v8f_mini_train.py`, `v8f_mini_data.py` or `v8f_mini.py`** (grep for `extrinsic` in those three files: 0 hits) | below |
| lead head build and guard | YES | below |
| total loss | YES | below |

## 1. How `stop_x_m` / `stop_y_m` reach a training batch

1. Label source. `stack/scripts/build_v9_labels.py:937-942` sets `col("stop_x_m")[i], col("stop_y_m")[i]` only on the branch `cls_l == 2` (a stop before any acceleration within the horizon), as the ego's position at rest expressed in the NOW frame by `to_now_frame(sx, sy, x0, y0, psi0)` (rig frame, +x forward, +y left). Every other row holds NaN (the column is created NaN-filled). So **NaN means "no stop within the horizon"**: a window whose ego is already stationary (class HOLD or CREEP) is NaN too.
2. They are the cells 4 and 5 of the 10 `LON_CONSTRAINTS` (`stack/tanitad/data/v8f_mini_data.py:60-61`), inside the 26 `CONS_FIELDS` (`:63`). The model's twin of that name list is `CONS_NAMES` (`stack/tanitad/models/v8f_mini.py:49`) and `CI` (`:53`): `CI["stop_x_m"]`, `CI["stop_y_m"]`.
3. The join. `build_label_join` (`v8f_mini_data.py:149`) writes `J["cons"][sl, q] = where(inside, R[f][rows], NaN)` for every `q, f in enumerate(CONS_FIELDS)` (`:219-220`), so a row outside the v9 release is NaN as well; `J["has"][sl] = inside` (`:197`).
4. The window item. `V8FMiniWindows.item` (`:571`) builds `rows = [row(ci, t), row(ci, t - a_t), row(ci, t - a_s)]` (`:584`) and packs `"lab": _lab_pack(J, rows[0])` (`:595`): **`lab` is the NOW row**; `lab_t` and `lab_s` are the older tick rows. `_lab_pack` (`:518-526`) carries `"has"` and `"cons": J["cons"][i]` (26 floats, NaN kept).
5. The collate. `collate` (`:644-655`) stacks `lab["cons"]` to `[B, 26]` float32 and `lab["has"]` to `[B]` bool.

**A missing stop gives NaN, not a mask.** The batch carries no mask of its own for it: `has` says "a v9 row exists", NaN says "no stop". PATCHED `stop_aux_labels` (`stack/tanitad/data/v8f_stopaux.py`) turns that into: `stop_has` = `lab.has` AND the clip has a camera row; `stop_pos` = `stop_has` AND finite AND the point projects into the frame; a `has` window with NaN is a NEGATIVE of the presence BCE (presence 0), and has no cell, so it carries no per-token term.

The label is read from `lab` (NOW), not `lab_t`: the head reads the NOW frame's tokens and the rest point is expressed in the NOW frame, so the two share one frame.

## 2. The NOW frame's adapted 4x10 tokens

1. The scene adapter: `self.adapter = LayerNorm(d_feat) + Linear(d_feat, d)` and `self.pos = Parameter([n_tok, d])` (`v8f_mini.py:700-701`). `_encode` (`:857-858`): `adapt = adapter(b["feats"].float()) + pos[None, None]` = **`[B, F, n_tok, d]`**, F = `hmax` history frames.
2. NOW is the LAST frame: `last_now = full((B,), Fn - 1)` (`:875`), `Fn = b["feats"].shape[1]`. The item's `feats` are `d["feats"][idc]` with `idx = arange(t - hmax + 1, t + 1)` (`v8f_mini_data.py:578`, `:587`), so index `Fn - 1` is the window's cache step t, the same row as `lab`.
3. The head reads `_at(adapt, last_now)` = `[B, 40, d]` (PATCHED `V8FMini._stop`). The lead head reads the same tensor (`_lead`, `:786`).

**How a token maps to image columns and rows.** The cache is `pool4x10`: the 16x40 patch grid (16-px patches of the 256x640 frame) average-pooled into 4x4-patch cells, **row-major, token = r * 10 + c** (builder `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py:13`, `pool_field` `:254-260`: `reshape(T, gh, GH//gh, gw, GW//gw, D).mean((2, 4))`). A cell is therefore a **64 x 64 px** block: column cell c = floor(col / 64) in 0..9, row cell r = floor(row / 64) in 0..3. Rows 0-1 are above image row 128 (sky and far scene), row 2 is image rows 128-191, row 3 is rows 192-255.

## 3. The cylindrical camera model and its intrinsics

1. The frame object: `PHYSICALAI_WIDE120_256x640 = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391, projection="cylindrical")` (`stack/tanitad/data/calib.py:1247-1248`). On a cylinder the column is LINEAR in azimuth: HFOV = 2 (W/2) / f_ref = 2.0944 rad = 120.0 deg. The pinhole 2 atan((W/2)/f) = 92.6 deg is the TRAPS.md error; this package never writes it.
2. The projection: `rig_projection.project_cam_to_frame` (`stack/tanitad/data/rig_projection.py:272`): `phi = atan2(x, z)`, `col = (W-1)/2 + f * phi`, `row = (H-1)/2 + f * y / hypot(x, z)`; `valid` = ahead and inside the frame. PATCHED `stop_cell` calls it unchanged.
3. Intrinsics source: a corpus constant, not a per-clip value. `v8f_mini_data.py:873` (`FRAME_HW = (256, 640)`) names the cache builder's cylindrical frame. This package takes the frame from `calib.PHYSICALAI_WIDE120_256x640`. (INHERITED from the card, section 2: the camera is the front-wide 120 deg one; the frame's own name and hfov agree.)

## 4. Where `sensor_extrinsics` enters: it does not, on the base

* `grep -c extrinsic` is **0** on `v8f_mini_train.py`, `v8f_mini_data.py` and `v8f_mini.py` (MEASURED). The batch has no camera pose. This is a real gap: the card asks for "the row from `sensor_extrinsics`", and the mount height is **per clip, not a constant** (`rig_projection.py:97-130`: 554 distinct heights over 2,400 clips, 1.2131-1.6672 m, forward offset 1.697-2.163 m, pitch -2.44..+3.94 deg).
* What exists: a per-clip table in the repo, `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv5-training-readiness/raw/rig_extrinsics_train2400.json` = `{clip_id: {qx, qy, qz, qw, x, y, z}}` for the **2,400 parity clips** (built by `stack/scripts/build_rig_extrinsics_table.py` from the dataset's `calibration/sensor_extrinsics` parquets), and `.../2026-09-06-refcv4b-landing/raw/extrinsics141.json` for eval-141.
* ⚠️ **NAMED DEPENDENCY, UNVERIFIED:** v8F-mini trains on **4,512 clips** (4,572 minus the 60 excluded; `stopbal_check.py` DATA). Whether the 2,400-clip table covers them is **not established here** (no data on this machine). The package therefore (a) takes the table as `--stop-aux-extr` (a JSON of the same format), (b) records its coverage of the TRAIN clips, and (c) **REFUSES the launch below 99 % coverage** (G-DVB, `bad_stop_aux`). If coverage is short, the table must be extended with `build_rig_extrinsics_table.py` for the missing clips before arm A can run. That is a data job, not a code gap.
* PATCHED `v8f_stopaux.py`: `clip_cam_table` aligns the table to the join's `J["clip"]` order; `stop_cell` builds the rotation from the quaternion with the same formula as `physicalai.FrontWideExtrinsics.rotation_cam_to_vehicle` (`physicalai.py:245`, parity-tested to 1e-12) and projects `p_rig = (stop_x_m, stop_y_m, 0)` (rig z = 0 IS the road plane, MEASURED over 87,481 cuboids, `rig_projection.py` docstring).

## 5. The lead head: how it is built and guarded (the template for the stop head)

* Config: `lead_head`, `w_lead`, `aux_cond` (`v8f_mini.py:229-231`); validated in `__post_init__` (`:261-266`).
* Built LAST so the global RNG stream of every module above is the off model's: `if cfg.lead_head == "on": self.lead_head = _mlp(d, d, 2)` (`:732-734`).
* Read: `_lead` (`:786-796`) takes `mean` of the NOW frame's adapted tokens, MLP, returns `(logit, gap_hat, lead_in)`; the loss path calls it at `:880-881`.
* Guard: `LEAD_FORBIDDEN_NAMES` (`:157`), `lead_head_source_problems` (`:1296`, a static scan of `_lead`'s identifiers), `_vision_only_run` (`:1306`: a key-guard that allows only `frame_ok`, a bit-identity check when every non-vision key is replaced, and a "moves with the features" check), `assert_lead_head_vision_only` (`:1344`). The trainer calls it from `lead_head_probe`.
* Trainer: flag registry `FLAG_CHECKS` (`v8f_mini_train.py:68`), `bad_lead` (`:881`), `make_lead_labels` / `lead_stage` / `lead_head_probe` (`:1441`, `:1457`, `:1466`), label call BEFORE the ego stages at `:2706` against `b = ego.apply(b, step)` at `:2712`.
* Difference for arm A: the lead head's outputs are read by O (the `lead_in` token); **the stop head has no consumer** (Q4: a pure auxiliary). So its gradient reaches the adapter only, the head is computed in the TRAINING forward only, and `forward_infer` never touches it (test: perturbing the head's weights leaves `forward_infer` bit-identical).

## 6. Where the total loss is summed

`V8FMini.loss`, `v8f_mini.py:1056-1066`: `total = w_ckpt T["ckpt"] + w_tac (...) + w_traj (...)`, then one line per lever (`w_acc` `:1059`, `w_level` `:1060`, `w_lead` `:1061`, each `total = total + w * T[...] if cfg.<lever> ... else total`), then the penalties and the WM term, then `T["total"] = total` (`:1066`). PATCHED: ONE new line in the same form, `total = total + cfg.w_stop_aux * T["stop_aux"] if cfg.stop_aux == "on" else total`, placed with the other lever lines. Mutation (c): deleting that single line turns the unit check (b) red AND the trainer's own G-LIVE red (its dead-parameter check names all eight head tensors).

## 7. Other hops the lever needed (all PATCHED, all read back)

argv `--stop-aux` / `--w-stop-aux` / `--stop-aux-extr` -> `make_config` (`:319`) -> `V8FMiniConfig` -> the BUILT model -> `declared_vs_built` -> `bad_stop_aux` -> `make_stop_aux` (camera arrays, coverage, sha256) -> `stop_aux_stage(b, SA)` in the probe batch and in the loop BEFORE `ego.apply` -> `b["stop_cell"], b["stop_pos"], b["stop_has"]` attached AFTER the stages -> `loss` -> `train_log.jsonl` (`stop_cell`, `stop_pres`, `stop_aux` every log row) -> `selfcheck.json` (`stop_aux_probe`, `stop_aux_extrinsics`) -> checkpoint (`cfg` keeps the base's key set when off).
`--init-ckpt`: `INIT_CKPT_MATCH` (`:662`) does not name the new fields, so an M2b checkpoint loads; `load_init_ckpt` (`:669`) allows exactly the keys under `NEW_LEVER_KEY_PREFIXES` to be fresh, which PATCHED now includes `stop_head.` and `stop_pres_head.`.
