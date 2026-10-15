# P2: what `scene_ctx` is computed from (admissibility, reported not decided)

SPEC 6 (`SPEC_AUX_DRIVABLE_HEAD.md`): the head's input carries **no ego, goal, navigation command or map**. All line numbers are at commit `8e106f9`, `refe/model.py` unless named.

## Finding (for the PI)

**`scene_ctx` is computed from the four camera images and the per-log camera calibration only. It consumes NO ego state, goal, navigation command or map.** Admissible under SPEC 6 as far as the source goes.
One thing to name for completeness: the per-camera 3D position embedding takes the camera rig's 16 calibration numbers per camera (`calib`, read per log from the nuPlan DB); that is a sensor property, not ego state, and
**`visual_ctx` (the S1 tap) carries exactly the same term** (`visual + _pos3d`), so it does not distinguish V from S.

## The hops (source, then the runtime proof)

| hop | file:line | what happens |
|---|---|---|
| entry | `planner.py:474` | `self.model(img, ego_vec, goal, calib=calib)`: ego and goal ARE arguments of the model, which is why the trace below matters |
| images | `model.py:744-746` | `flat = img.reshape(B*N, ...)`; `tok = self.backbone(flat)` (the frozen trunk, ViT, image input only: `VitS16.forward(img)`, `model.py:409`); `visual = tok.reshape(B, N*P, width)` |
| calibration | `model.py:747`, `_pos3d` `675-697` | `visual = visual + self._pos3d(img, N, calib)`: a 3D frustum embedding from the camera rig (`calib` [B, n_cam, 16]); no ego |
| compression | `model.py:757` | `scene = self.reg_compress(self.registers.expand(B,-1,-1), visual)`: learned register queries cross-attend the visual tokens (`RegisterCompress`, `model.py:451`); 16 registers x 4 cameras = **64 tokens** |
| **scene_ctx** | `model.py:758` | **`scene_ctx = self.scene_proj(scene)`** [B, 64, 256] |
| visual_ctx (S1's tap) | `model.py:759` | `visual_ctx = self.scene_proj(visual)` [B, 7680, 256]: the SAME linear on the uncompressed tokens |
| where ego / goal enter | `model.py:762-772` | `ego_tok = self.ego_enc(cat([ego, goal_feat]))` and `q = self.queries + ego_tok`: they enter only the trajectory-decoder QUERY, **after** `scene_ctx` is already computed (line 758 < 762) |
| consumers | `model.py:774`, `784-786` | the trajectory decoder reads `scene_ctx` (`blk(q, scene_ctx)`); the scoring decoder reads `visual_ctx.detach()` (`sctx`) |

So both taps are functions of (images, calibration, frozen/trained trunk weights); the scene tap is a learned 120-fold compression (7,680 -> 64 tokens) of the visual tap's input.

## Runtime proof (the same forward that produced `c3_full_ctx`, on real tokens)

`extract_scene_ctx.py` arms three hooks on the planner's ONE `planner.infer` call per token and records, per token, in `extract_log.jsonl`:
* `tap_is_scene_dec_ctx`: `AUX.tap_is(model.dec[0]'s ctx argument, scene_proj output #1)` is True, i.e. the saved tensor IS the tensor the trajectory decoder reads (same storage and shape [1, 64, 256]).
* `tap_is_visual_score_ctx`: `AUX.tap_is(model.score_dec[0]'s ctx argument, scene_proj output #2)` is True, i.e. output #2 is the S1 tap, and `scene_is_not_visual` is True.
* The 8-token smoke (`aux_navtest_forward/c3_full_scene_ctx/extract_log.jsonl`, all 8 lines): both identities True, `n_scene_proj_calls` 2, `scene_shape` [1, 64, 256].

The Master Mind decides whether the scene tap's compression is acceptable as an S1b lever; this note only states what it reads.
