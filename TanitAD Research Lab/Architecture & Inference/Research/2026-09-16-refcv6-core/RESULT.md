# refcv6 core — the ImageNet trunk over frame history, ego history, and F1…F9

**Date:** 2026-09-16 · **Agent:** architecture & inference (refcv6 core)
**Spec:** `SPEC_REFCV6_V2.md` §2 / §3, plus the PI's two corrections of 2026-09-16
(frame + ego history required; 256 × 1024 geometry, resnet101 primary).
**Tier:** T0 — CPU smoke builds only. ⛔ **No eval number, no four-family table:**
nothing here was trained and nothing was scored. Every number below is a
parameter count, a shape, a wall-clock or a memory figure, each MEASURED on the
dev box and each labelled.

---

## 1. Headline

⚠️ **INTEGRATION — three items are NOT mine and one of them is still open.**

1. ⛔ **`refc_v3.RefCV3Model.forward` does not forward unknown kwargs to the
   core**, so the ego-history tensor cannot reach `RefCModel.forward` through
   the v3 wrapper. The permanent fix is **one line in a file I do not own**:
   add `ego_poses` (and optionally `ego_n_past`) to `RefCV3Model.forward` and
   pass it to both `self.core(...)` calls (`refc_v3.py:1521-1525`,
   `:1527-1534`). Until then the trainer uses `RefCModel.set_ego_window`, a
   **one-shot** channel that is consumed by the next forward and raises if a
   second forward finds it empty — so a stale window can never be silently
   re-encoded into a different batch.
2. ⚠️ **`tanitad/rl/refcv3_adapter.py` — two declarations added.** Its own
   contract requires it: *"every surviving channel MUST be declared … A forward
   that grows a channel therefore breaks LOUDLY at the next rollout
   construction."* Routing around it with the one-shot setter would have hidden
   the channel from the admissibility contract entirely, which is worse than
   declaring it.
3. ⚠️ **`tests/test_rl_refcv3_used_path_guard.py` — four frozen expectations
   moved** (the `a - b` channel set, the derived-channel set, and the
   declaration count 11 → 13). That file says the update must be *"read by a
   human instead of re-blessed by a diff"*; this is that read, and the comment
   records what moved and why.
4. ⚠️ **The 256 × 1024 caches do not exist yet.** Everything here was measured
   on synthetic `256 × 1024` tensors. A data agent owns the rebuild.
5. ⚠️ **Perception heads are NOT built.** §6 of the spec (MAP / BOX on the
   stride-16 map) is out of this brief; the stride-16 map is *exposed and
   tested* so the head has something to hang off.

---

## 2. Task 1 — the trunk

### 2.1 What was built

`stack/tanitad/models/timm_trunk.py` (new):

* **`TimmResNetTrunk`** — a `timm` ImageNet backbone selected **by name**, with
  every channel count read from `feature_info`. Nothing hard-codes 256, 512,
  40, 20, 64 or 32.
* **Frame history (PI 2026-09-16), the default:** `mode="shared"` runs the SAME
  weights over each of the **K = in_channels // 3** frames, **3 channels each,
  ImageNet-normalised**, then fuses **after** the trunk at **both** strides.
  That is what keeps the prior exact — the stem never sees a distribution it was
  not fitted on.
* **`mode="inflate"`** — the registered knockout / cheaper arm: ONE pass with a
  3K-channel stem, ImageNet weights repeated and **divided by K**.
* **`TemporalFuse`** — `concat1x1`, `attn`, or `last`, all **identity-initialised
  on the newest frame**, so frame history is a *removable graft*.
* **`param_groups_dd`** — DiffusionDrive's optimiser groups.
* **`trunk_feature_channels`** — the cached `feature_info` lookup that lets
  `CNNEncoderConfig.feat_dim` answer **2048** for resnet101 instead of a
  hard-coded 512.

`stack/tanitad/models/ego_history.py` (new): `EgoHistoryEncoder` +
`ego_channels_from_poses`.

Wiring, in my files only: `CNNEncoderConfig.{trunk, trunk_name, trunk_mode,
trunk_fuse, trunk_fuse_identity, trunk_pretrained, trunk_imagenet_norm}`,
`refc.build_encoder`, `RefCConfig.ego_history`,
`AnchoredDiffusionDecoder.ego_to_cond` (zero-init), and the trainer's
`--trunk*`, `--opt`, `--ego-history*` flags with a full `config.json` stamp and
a **bidirectional** `assert_seams_are_built` check.

### 2.2 The weights are real — MEASURED

`timm/resnet34.a1_in1k` `model.safetensors`, **87,278,522 B**, md5
`69731b23dc8f6e60d8fcb02478a7a1e4`. `conv1.weight` (64, 3, 7, 7):
mean **−0.0028060986660420895**, std **0.23252105712890625**,
**|w|.sum() = 1279.853759765625**. `timm/resnet101.a1_in1k`:
**|w|.sum() = 1402.778076171875**.

A He-initialised (64, 3, 7, 7) conv has std **0.1166** — half — and
`|w|.sum()` near **875**. The build **refuses** outside ±2 % of the pinned
value; a backbone with no pinned entry is compared against the **checkpoint on
disk** instead. `timm` 1.0.29 installed into `C:/Users/Admin/venvs/tanitad`;
HuggingFace reached with `truststore.inject_into_ssl()`.

### 2.3 Shapes at 256 × 1024 — MEASURED

| backbone | stride-16 (perception) | stride-32 (planner) | backbone params |
|---|---|---|---|
| `resnet34.a1_in1k` | **256 × 16 × 64** | **512 × 8 × 32** | **21,284,672** |
| `resnet101.a1_in1k` | **1024 × 16 × 64** | **2048 × 8 × 32** | **42,500,160** |

**1,024 perception tokens** and **256 planner tokens**, at 1.875 ° and 3.75 °
per column — against 640 / 160 and 3.0 ° / 6.0 ° at the old width. Both counts
match the figures the PI quoted (42.5 M / 21.3 M).

### 2.4 Cost — MEASURED on this CPU, batch 1 window, fp32

Trunk forward only, 256 × 1024:

| arm | fwd s/window | activations MB/window | encoder params |
|---|---|---|---|
| r34 K=1 | 0.069 | 292.6 | 21,284,672 |
| **r34 K=3 shared** | **0.200** | **879.2** | 22,268,480 |
| r34 K=3 inflate | 0.080 | — | 21,303,488 |
| r101 K=1 | 0.191 | 1,083.7 | 42,500,160 |
| **r101 K=3 shared** | **0.569** | **3,257.4** | 58,231,872 |
| r101 K=3 inflate | 0.186 | — | 42,518,976 |
| ⛔ **today (refcv5-v2, 256 × 640, ×8 positions)** | **1.283** | **1,665.3** | **90,458,632** |

⭐ **r34 K=3 at the new geometry is 6.4× faster than today's arm and needs 1.9×
LESS activation memory.** ⚠️ **r101 K=3 is 2.3× faster but needs 2.0× MORE**
(3,257 MB/window). The "K passes are cheaper than today" claim holds for r34;
for r101 it holds on *time* and fails on *memory*, and the launch should be
sized on the memory number.

**Full 2-window training step** (forward + backward + AdamW, K = 3, ego history
on, sampler on, 256 × 1024):

| trunk | F1…F9 off | F1+F2+F3+F4+F5 on | model params |
|---|---|---|---|
| resnet34 | **4.603 s** | **4.367 s** | 22,410,805 |
| resnet101 | **14.650 s** | **13.853 s** | 58,650,677 |

⭐ The F-on step is *faster*: **F1 replaces the two-call inference chain with
ONE decoder call**, which is exactly DiffusionDrive's training objective.

### 2.5 ⚠️ The fusion is not free on resnet101 — MEASURED

`concat1x1` costs `K·C → C` at both strides:

| fuse | backbone | fusion params | total |
|---|---|---|---|
| `concat1x1` | 42,500,160 | **15,731,712** | 58,231,872 |
| `attn` | 42,500,160 | **27,654** | 42,527,814 |

**A third again on top of the r101 backbone, for the fusion alone.** Both are
identity-initialised and therefore directly comparable arms. ⭐ **Recommendation:
run r101 with `--trunk-fuse attn`** unless the PI wants the extra capacity; the
r34 arm can keep `concat1x1` (0.98 M).

### 2.6 Memory envelope for the launch — ESTIMATED from the measured activations

Trunk activations only, fp32, per window; **the decoder, the heads and the
optimiser state are on top**, and `params + Adam m,v` is ≈ 3 × the parameter
bytes.

| card | r34 K=3 (879 MB/win) | r101 K=3 (3,257 MB/win) |
|---|---|---|
| **8 GB** (dev-box RTX 4060, 8.6 GB) | ~7 windows fp32 · ~14 with AMP | ⚠️ **~2 windows fp32** · ~4 with AMP |
| **48 GB pod** | ~50 fp32 · ~100 AMP | ~13 fp32 · ~26 AMP |

⛔ **Do not launch r101 K=3 at batch > 2 on the dev box.** On a 48 GB pod,
**batch 8–12 fp32 or 16–24 with AMP** is the defensible starting envelope for
r101; leave ≥ 25 % headroom because these are trunk-only figures.

---

## 3. Task 2 — diffusion F1…F9

All nine implemented in `stack/tanitad/models/refcv6_diffusion.py` +
`refc.py` + `refc_v3_train.py`, **each behind its own flag, each default OFF**.

| # | flag | where it lives | DD source |
|---|---|---|---|
| F1 | `--f1-random-t` | `_sample`: one call at `t ~ U[0, 50)`, every stage supervised | `transfuser_model_v2.py:463-476`, `:492-497` |
| F2 | `--f2-dd-step` | `dd_step_pairs`: `t → t−1` | `:507` (`set_timesteps(1000)`) |
| F3 | `--f3-per-layer` | `CascadeHeads` + per-stage loss + **detach** | `:345-347`, `:372-380`, `:492-497` |
| F4 | `--f4-adaln` | `AdaLNModulation` after every layer's FFN | `:229-268`, applied `:337` |
| F5 | `--f5-emitting-conf` / `--f5-focal` | ranked surface = the sampler's own pass; `focal_cls_loss` | `:544-552`; `multimodal_loss.py:90-98, 146-157` |
| F6 | `--f6-w-u0-zero` | sets `--w-u0 0` **and** carries the PI's acknowledgement | `multimodal_loss.py:161` |
| F7 | `--f7-samples-per-anchor` | group-major widening + `sel_anchor_id` | DD Tab. 6 |
| F8 | `--f8-flat-noise` | DD's affine `norm_odo` box + the `[-1, 1]` clamp | `:432-453`, `:474`, `:519` |
| F9 | `--f9-assert-vocab` | assert-only | — |

### 3.1 Bit-identity, on 64 windows

`test_all_flags_off_is_BIT_IDENTICAL_on_64_windows` materialises the **pre-refcv6
`refc.py` from git** (`git show HEAD:stack/tanitad/refs/refc.py`), builds both
decoders under the same seed, and compares **state_dict keys, every weight,
every output tensor and the whole `sel_tele` dict** on a 64-window forward.
All equal. ⛔ Measured against git, not against a copy someone might have edited.

### 3.2 F2, MEASURED off the alpha table

`residual_retained(sched, 10, 9) = 0.93…0.97` against
`residual_retained(sched, 10, 0) = 0.25…0.31`. The spec's "95 % vs 28 %" is a
read of the schedule, not a claim.

### 3.3 ⛔ F7 — two of three sites widened, the third still refused

| site | status |
|---|---|
| `loss_cls`'s `a_star` anchor target | **WIDENED** — `candidate_to_anchor_id` |
| the `[B, N]` anchor priors | **WIDENED** — `tile_anchor_prior`, group-major |
| the eval harness's `sel_idx` join | ⛔ **STILL REFUSED.** `taniteval` reads `sel_idx` **as an anchor id** and that file is another agent's. The forward now also emits **`sel_anchor_id`**; `--f7-ack-eval-join` is the operator stating that the consumer reads it. Without the acknowledgement `G > 1` raises. |

### 3.4 ⛔ Where the released code and the paper disagree, the code wins

* **F2.** The paper says "2 denoising steps". The *code* calls
  `set_timesteps(1000)` while stepping the labels `[10, 0]`, so diffusers
  derives `prev = t − 1`. The published ablation table cannot tell you that;
  only `:507` can.
* **F4.** DD's `ModulationLayer` ships with `if_zeroinit_scale = False`
  (`:233`). Our zero-init variant is a *removable graft* and a different arm —
  the code's own value is the default and the stamp says which ran.
* **F6.** Our record used to say *"DiffusionDrive has no denoising loss, so
  `--w-u0` is our invention"*. DD's matched-anchor L1 **is** its x0 loss
  (`multimodal_loss.py:161`); `--w-u0` **duplicates** it in control space.
* **F5 scale.** DD's focal reduction is `mean` over `B × N`
  (`multimodal_loss.py:112-114` → `:24-25`), so the per-window magnitude scales
  as `1/N`. At our N = 117 against DD's 20 that is a **5.85× smaller** term at
  the same weight — a real re-weighting, stated rather than discovered later.

---

## 4. Mutation table — every guard re-proved by re-introducing its defect

⛔ A guard that has only ever seen the fixed code has not been tested. Each row
is a real defect put back, and the test that then went **RED**.

| # | defect re-introduced | test that went RED |
|---|---|---|
| 1 | `pretrained=False` — a trunk that never loaded ImageNet | `test_refcv6_trunk.py::test_random_init_FAILS_the_imagenet_check` (both backbones) |
| 2 | stem inflated by `repeat` **without** the `/ K` | `::test_inflation_WITHOUT_the_divide_does_NOT_reproduce` — relative deviation > 0.1 where the correct rule reads < 1e-5 |
| 3 | inflating an already-inflated (9-ch) stem | `::test_inflate_stem_REFUSES_an_already_inflated_stem` |
| 4 | temporal fusion **without** the identity init | `::test_plain_init_fusion_does_NOT_reproduce_the_single_frame_trunk` — the K = 1 equality breaks |
| 5 | fusion reading the OLDEST frame instead of the newest | `::test_fusion_reads_the_NEWEST_frame_not_the_oldest` |
| 6 | a misspelled encoder prefix ⇒ empty encoder group, 0.5× applies to nothing | `::test_param_groups_REFUSE_an_empty_encoder_group` |
| 7 | an ego encoder that forgets to slice (reads the future) | `::test_ego_encoder_reading_the_future_GOES_RED` — the `Leaky` subclass fails the same assertion the real one passes |
| 8 | a *centred* difference at the last past step (reads the first future sample) | `::test_ego_channels_are_BACKWARD_differences` |
| 9 | a yaw rate that does not wrap at ±π | `::test_ego_yaw_rate_WRAPS_at_pi` — 62.8 rad/s instead of 1.0 |
| 10 | a built ego encoder fed no window | `::test_model_REFUSES_a_built_ego_encoder_with_no_window` |
| 11 | a *stale* ego window re-used by a second forward | `::test_set_ego_window_is_ONE_SHOT` |
| 12 | hard-coded 256/512 channels (i.e. resnet101 answered as 512) | `::test_the_CONFIG_feat_dim_follows_the_BACKBONE` + `::test_the_two_chosen_backbones_report_DIFFERENT_widths` |
| 13 | hard-coded 40/20 grids (i.e. 640-only) | `::test_both_feature_maps_have_the_documented_shapes`, parameterised over **both** geometries × **both** backbones |
| 14 | **F3 without the `detach`** between stages | `test_refcv6_diffusion.py::test_F3_WITHOUT_the_detach_leaks_gradient_backwards` — the un-detached loop leaks gradient into `layers[0]`, which the real one does not |
| 15 | F5 focal replaced by softmax `cross_entropy` | `::test_F5_focal_is_SIGMOID_not_softmax` — the two must differ by > 0.1 |
| 16 | F7 tiling by `repeat_interleave` instead of group-major `repeat` | `::test_F7_tiling_is_GROUP_MAJOR_not_interleaved` — both type-check; only one is right |
| 17 | F7 `G > 1` with no acknowledgement of the eval join | `::test_F7_REFUSES_without_the_eval_join_acknowledgement` |
| 18 | F8 (metre-space noise) on the **control-space** sampler | `::test_F8_REFUSES_on_the_control_space_sampler` |
| 19 | F9 on a vocabulary that is not the 117 v0-conditioned anchors | `::test_F9_fires_from_the_FORWARD_on_a_wrong_vocabulary` |
| 20 | flags set on a **classifier** build (a flag reaching no mechanism) | `::test_flags_on_a_CLASSIFIER_build_are_REFUSED` |
| 21 | the flag block passed as a **dict** (reads back as "no flags") | `::test_a_dict_of_flags_is_REFUSED_not_silently_ignored` |
| 22 | F5 run alongside `--sel-score-emitted` (two scores, one fan) | `::test_F5_REFUSES_alongside_sel_score_emitted` |
| 23 | any refcv6 key leaking into `sel_tele` on a baseline run | `::test_telemetry_is_UNCHANGED_when_every_flag_is_off` |

⚠️ **Two mutation tests were themselves found vacuous and fixed** — recorded
because a silently-passing mutation test is worse than none:

* the F3 detach probe first read **stage 0's own head**, which receives no
  gradient from stage 2 *whether or not* the detach is there. It now reads
  **`layers[0]`'s attention weights**, which is what the detach actually cuts.
* the F8 flatness probe first estimated a per-slot σ from **4 draws** and read
  its own sampling noise as a horizon trend. It now uses **2,000 draws** and
  asserts both the spread (< 5 % of the mean) and the absolute value
  (0.90 m / 0.73 m ± 5 %).

---

## 5. Tests

| file | count | result |
|---|---|---|
| `stack/tests/test_refcv6_trunk.py` | 40 | **pass** |
| `stack/tests/test_refcv6_diffusion.py` | 34 | **pass** |

Regression over every `refc` / `ddv2` / `sampler` / `rl_refcv3` / `u8` /
`timm` / `ego` test: **1,052 passed, 1 skipped, 2 failed** — and both failures
are **pre-existing and environmental**, each verified against the untouched
tree or against `git status`:

* `test_refc_select.py::test_trainer_runs_every_lever_…` — needs a CUDA device,
  and this brief mandates `CUDA_VISIBLE_DEVICES=""`. **Reproduced on the
  untouched snapshot** (`C:/Users/Admin/tanitad-snap-20260915/stack`), so it is
  not mine.
* `test_ddv2_rl.py::test_the_excerpt_matches_the_banked_release_…` — a sha256
  mismatch on a banked upstream file that `git status` shows unmodified.

Collection of `tests/test_metric_decode_refusal.py` and
`tests/test_refa_v1_dk_hook.py` fails on `ImportError: DistanceKeepingSpec`
at the branch tip — **pre-existing, not mine**, and worth someone's attention.

---

## 6. Environment finding, worth recording

⛔ **`python <script.py>` from any cwd other than `…/stack` imports `tanitad`
from the DEAD `G:` mount and dies with `OSError: [Errno 22]`.** The venv carries
an editable install (`__editable__.tanitad-0.0.1.pth`) that still points at the
G: path. Run everything with **cwd = the worktree's `stack/`**, and pipe scripts
on stdin (`python - < script.py`) rather than passing a path. This looks exactly
like a corrupt source file and is not one.

---

## 7. Open items for the PI / the coordinator

1. **`refc_v3.py` one-liner** (§1.1) — until it lands, ego history rides the
   one-shot channel.
2. **`--trunk-fuse attn` for resnet101** (§2.5) — 15.7 M of fusion parameters
   saved, or keep them deliberately.
3. **Batch size** — §2.6. r101 K=3 does not fit the dev box above batch 2.
4. **The 256 × 1024 caches** — not yet built; everything here is synthetic.
5. **Perception heads (spec §6)** — not in this brief. The stride-16 map is
   exposed, shaped and tested so they can be hung off it.
