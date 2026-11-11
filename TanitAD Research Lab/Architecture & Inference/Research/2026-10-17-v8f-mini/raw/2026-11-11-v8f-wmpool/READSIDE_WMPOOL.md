# READSIDE_WMPOOL: what must build z1 from cells for a P-ATT arm (listed, NOT implemented; each change is a few lines but needs the cell frames of the eval clips)
A P-ATT model RAISES on `wm.z1_from_mean` (by design: a token mean handed to it would silently run the mean pool), so an unchanged reader fails loudly rather than reading a wrong z1. The readers pass `fr["zmean"][ci]` [T, 256] (frames npz from `ladder_read/lr_lib.py build_eval_frames`).
1. `theory/ladder_read/lr_lib.py`
   * `build_eval_frames` (l. 180-196): also store `zcell` [sum T, 10, 256] = `zcells.zcell_of(model, d["feats"])` (same adapter + pos, `adapter_sha` already stored), via `np.savez(..., zcell=...)`.
   * `load_eval_frames` (l. 199-204): split `zcell` like `zmean` when the key exists.
   * `encode_clip` (l. 232-239): `z1 = wm.z1_from_cells(x[None])` when `getattr(wm, "pool_q", None) is not None` (x = the clip's cells [T, 10, 256]), else `wm.z1_from_mean`. Callers at l. 352, 606, 641, 682 pass `fr["cell" if pattn else "zmean"][ci]`.
2. `theory/ladder_plantest/lp_lib.py` `clip_z1` (l. 276-281): the same dispatch, `wm.z1_from_cells(torch.as_tensor(cell)[None])[0]`. `rollout_h` (l. 50) works on latents, not on tokens, so it is unchanged. Goal latents come from `clip_z1`, so they follow.
3. `theory/egowm/read/lr_speed.py` (IA-0 reader wrapper, `_encode_clip(model, zmean, act, contrast)` keyed by `id(fr['zmean'][ci])`): the speed registry is EGOWM's (`E-*` arms only); P-ATT has no speed channel, so the IA-0 read for P-ATT calls `lr_lib` directly and needs only item 1. `egowm/read/controls_read.py` plants `zmean` and is not used for P-ATT.
4. The frames file for P-ATT readers must be built with the SAME frozen adapter as the training cell cache; `adapter_sha` is the key (the manifest's `adapter_state_sha256` has the same formula).
5. Attention-weight entropy per level (the card's "beside, no bar" item): `wm.pool_weights(cells)` [B, F, 10] gives the weights; entropy = `-(w*log w).sum(-1)`, uniform = ln 10 = 2.3026.
Size: ~35 lines across items 1-2, so under the 60-line bar, but not written because no cell frames exist yet and the readers are another stream's files.
