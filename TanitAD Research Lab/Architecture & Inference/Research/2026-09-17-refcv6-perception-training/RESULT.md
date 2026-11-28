# refcv6 §2/§6 — THE PERCEPTION BRANCH IS NOW TRAINED, NOT MERELY BUILT

**Stream:** Architecture & Inference · **Date:** 2026-09-17 · **Base:** `agent/arch-inf-20260803` tip
`c16b7f1` · **Worktree:** `C:/Users/Admin/tanitad-wt-perctrain` (control tree
`C:/Users/Admin/tanitad-wt-perctrain-tip`, same commit) · **Compute:** dev box, **CPU-only**
(`CUDA_VISIBLE_DEVICES=""`; the RTX 4060 was held by another package).

**The PI's instruction (2026-09-16), verbatim:** *"Let plan to train jointly the resnet-trunk, the bev
map (based on the sam2 maps as gt) and a head for 3d bounding boxes extracted from the resnet trunk …
I would also recommand to give additionally access to the truink for the diffusion planner"*.

**Headline:** the joint objective now runs end to end on the real corpus, and the two new gradient
paths are MEASURED to reach the ResNet trunk — **map alone 12,511.0**, **3-D boxes alone
1,818,181.9** `grad_abs_sum`, with the trunk falling to **exactly 0.0** when the seam is detached.
At the 0.0 default the checkpoint is **bitwise identical to the tip's on all 193 tensors /
131,977 elements**. ⛔ **Three defects were found on the way, and one of them silently swapped the
trunk and both control arms of every run on the PI's own 256×1024 geometry.**

---

## 0. The gap was BIGGER than the brief stated — verified before building

The brief said the heads were built and not wired into the trainer. **They were also never
instantiated in any model.** MEASURED on `c16b7f1`, two independent probes:

| probe | result |
|---|---|
| `grep -rn "BEVMapBranch\|BEVEncoder(\|MapHead(\|Box3DSlotDecoder\|Box3DMemory\|BEVLift("` over `stack/` | **1 hit**, and it is a *comment* in `refc.py:2232` |
| import graph (`from tanitad.models.bev_encoder / bev_lift / box3d_head`) | importers are **`stack/tests/` only** — **zero production modules** |

⇒ The island had no bridge at either end. `_w_map` / `_w_box3d` were READ by the conflict
detector's weight table (`refc_v3_train.py:4432-4433`) and **set nowhere**; `compute_losses_v3`
assembled only `loss_cls / loss_lat / loss_lon / loss_traj`.

⭐ **What DID exist, and is why no model-file edit was needed:** `RefCModel.forward` already returns
`fmap_s16` (`refc.py:4215`) — the **stride-16 map of the window's LAST OBSERVED frame**, graph
attached — on both the hierarchy and the flat path, and `RefCV3Model.forward` passes the core's dict
through. That is the seam. **`refc.py` and `refc_v3.py` are untouched by this work.**

---

## 1. What was wired

### New module — `stack/tanitad/models/refcv6_perception_branch.py` (the assembly that was missing)

| piece | what it does |
|---|---|
| `PerceptionBranchConfig` | the two weights + widths. ⛔ **Refuses to exist with both weights 0.0** — a branch at zero weight would put parameters in the optimiser and the checkpoint and train on nothing |
| `LiftGeometryBank` | `episode_id → (grid, valid)` for `BEVLift`, built once per clip from the per-clip extrinsics table; **refuses** an episode it does not cover |
| `PerceptionBranch` | `BEVLift → BEVMapBranch` (map) and `Box3DMemory → Box3DSlotDecoder` (boxes), one forward, one graph |
| `map_loss_row` / `box3d_loss_row` | thin wrappers so the arithmetic stays in the tested modules; each emits its **per-head counts** |
| `grad_reach_report` / `grad_abs_sum` | the instrument that PROVES reach per head rather than asserting it |

⛔ **No geometry literal anywhere**: `d_image` from `encoder.s16_dim` (timm `feature_info`),
`image_hw` from `encoder.s16_shape`, the frame from `trunk_shapes.frame_for_width`, the BEV grid
asserted by `assert_label_grid_unmoved`. The module is registered in
`tests/test_refcv6_geometry_agnostic.py::OWNED` so the existing, mutation-proven tokeniser scans it.

### Trainer — `stack/scripts/refc_v3_train.py`

* **CLI**: `--w-map` / `--w-box3d` (default **0.0**), `--map-gt-root`, `--map-min-coverage`,
  `--map-lru`, `--join3d`, and `--trunk-pretrained` / `--no-trunk-pretrained`.
* **Targets into the batch**: `V3Dataset.enable_map_gt()` (`MapGTStore` + a coverage **refusal**)
  emits `map_frac [9,120,64]`, `map_seen [120,64]`, `map_label`, `map_raw_frame`, `map_ep`;
  `enable_join3d()` widens the existing `_agent_item` block with `agent_cz` / `agent_h` /
  `agent_zh_mask` via `zh_for_frame` + `zh_targets`.
* **`compute_losses_v3`**: `loss_map` (`map_soft_ce`, 9-class soft CE, **seen cells only**) and
  `loss_box3d` (`box3d_set_loss`, Hungarian, metres), each with per-head counts in every log row
  (`n_map_cells`, `map_n_labelled`, `box3d_n_z`, `box3d_n_matched`, …).
* **Branch construction** happens **before `build_optimizer`** — deliberately: `Adam(
  model.parameters())` enumerates once, and a branch attached after it would be a head with
  gradients and **no optimiser state**, which would read as *"the perception heads do not help"*.
* **Per-head gradient reach in `metrics.jsonl`** (`ga_trunk`, `ga_lift`, `ga_bev_encoder`,
  `ga_map_head`, `ga_box_memory`, `ga_box_decoder` + `_n`), read **after `backward`, before the
  global clip**, exactly where `_grad_probe_row` is.
* **`config.json`** gains `refcv6_perception` (null when both weights are 0), carrying the branch's
  param breakdown, the trunk's channel/HW, the frame, **the coverage census**, and
  `"lidar_as_training_target": false`.

### Reader — `stack/scripts/train_p8_occupancy.py`

`JoinFileReader(..., with_track_ids=True)` + `lookup_track_ids()`, mirroring the existing
`_cls_by_clip`. ⛔ **Needed because the 3-D join is keyed BY TRACK**: without ids a loader could only
align `cz`/`h` to the 2-D rows **by position**, which is right only while no filter ever reorders
either file. Default `False`; paid for only when `--join3d` is passed.

### Refusals — `_pin_refcv6_perception`, all before `config.json` is written

| combination | refused because |
|---|---|
| `--w-map`/`--w-box3d` > 0 without `--trunk timm` | `fmap_s16` is `None`; the legacy trunk exposes stride 32 only |
| `--w-map` > 0 without `--map-gt-root` | the BEV branch would be built and never supervised |
| `--w-map` > 0 without `--agent-rig-camera extrinsics` | one road plane for a corpus with **554 distinct mount heights in 2,400 clips** |
| `--w-box3d` > 0 without `--agent-join` | nothing to Hungarian-match against |
| `--join3d` with `--w-box3d 0` | heights loaded, emitted, multiplied by zero, named in `config.json` |
| **`--map-gt-root` / `--join3d` with BOTH weights 0** | ⭐ the **reverse** refusal: a run whose record names the SAM3 corpus and whose trunk never saw one cell of it |
| `--w-box3d` > 0 **without** `--join3d` | ⚠️ **NOT refused** — that is the legal 2-D rung whose total is exactly the 2-D set loss. It **prints** the distinction instead |

Both flags are registered in `REFC_WEIGHT_GATES`, which `tests/test_v6_effective_weights.py`
enforces as **exhaustive over the parser** — that test failed on my first pass and is why the rows
exist. The live run's own table prints `--w-map … graph=yes TRAINS` and `--w-box3d … graph=yes
TRAINS`.

---

## 2. BIT-IDENTITY AT ZERO WEIGHT — proved on the artifacts

Tip trainer vs patched trainer, **same seed**, `--smoke --steps 3 --device cpu --synth-episodes 4`,
both weights at their 0.0 default. `raw/bit_identity.json`, `code/bitid.sh`, `code/bitid_check.py`.

| artifact | result |
|---|---|
| **`ckpt.pt`** | **BITWISE-IDENTICAL** — 193/193 tensors, **131,977 elements**, `n_tensors_bitwise_different: 0`, and identical key sets |
| **mutation control** | the comparison **detects a one-bit flip in 193/193 tensors** (byte-view XOR) ⇒ it is not vacuous |
| **`metrics.jsonl`** | **BYTE-IDENTICAL** — sha256 `1e4b12755ccd65c9…` on tip, patched **and** the same-code control, 1,559 bytes, **24 fields per row**, in **two independent executions** (`raw/bit_identity_run1.json`, `raw/bit_identity_run2.json`) |
| **the discriminating control** | the **same trainer run twice** (`tip` vs `tip2`) is compared in the same breath, so "identical" is measured against a run that could have differed |
| **perception keys** | `map*` / `box3d*` / `ga_*` / `n_map*` in the patched log: **none** — and the same-breath control confirms the key scan found the log at all (`has_loss_key: true`) |
| **`config.json`** | gains `refcv6_perception: null`; `argv` differs (different worktree path); `seams.agent_knobs` and `effective_weights` gain the two **zero-valued** weights — those two stamps are **derived from the parser** by design (`agent_knob_dests`, *"DERIVED, NEVER LISTED"*), so a new flag necessarily appears there |

⛔ **Honest statement of the claim:** *the model is bit-identical and the metrics log is
byte-identical; the run record is not, by one null key and two zero-valued entries in two
deliberately-derived stamps.*

⚠️ **And one caveat I will not hide, because it is the difference between a structural guarantee and
a lucky rounding.** `elapsed_s` is **wall-clock at 0.1 s resolution**. An earlier execution of this
same script, on a busier machine, had tip, `tip2` **and** patched all differing in exactly that one
field — i.e. the same-code control behaved identically to the patched one, which is the finding that
matters. ⇒ **Byte-identity of `metrics.jsonl` is not guaranteed by this patch and must not be quoted
as a structural property.** What IS structural, and is what the two banked runs assert, is the
**checkpoint's bitwise identity** and the identity of **every training field**.

⚠️ **The control earned its keep immediately:** the first patched run **crashed** because
`_clip_table_for_caches` ran unconditionally and `--synth-episodes` supplies no `--v2-cache`. The
whole block is now gated on `--map-gt-root or --join3d`, so a run that asks for no perception does
not even read a manifest.

---

## 3. GRADIENT REACH — per head, isolated, with the detached seam as the control

⛔ **Why the trainer's own `ga_trunk` is NOT sufficient.** In a joint run the planner loss also
reaches the trunk, so a non-zero `ga_trunk` is consistent with a perception head that reaches
nothing — precisely the shape of evidence that let `tac_goal_tok_head` sit at `grad_abs_sum`
**exactly 0 for all 40,284 steps**. `code/grad_isolate.py` backwards **one term at a time** on a real
batch (4 real clips, 684 windows, 256×1024 cache, resnet101 s16). `raw/grad_reach_isolated.json`.

| arm | trunk | lift | bev_encoder | map_head | box_memory | box_decoder |
|---|---:|---:|---:|---:|---:|---:|
| `planner_only` | 16,316.2 | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| `map_only` | **12,511.0** | 8,319.7 | 9,176.2 | 94.1 | 0.0 | 0.0 |
| `box3d_only` | **1,818,181.9** | 98,585.5 | 103,814.0 | 0.0 | 494,637.8 | 1,086,817.3 |
| `both` | 1,830,000 | 93,698.5 | 102,003.7 | 94.1 | 494,637.8 | 1,086,817.3 |
| `map_only` **SEAM DETACHED** | **0.0** | 8,319.7 | 9,176.2 | 94.1 | 0.0 | 0.0 |
| `box3d_only` **SEAM DETACHED** | **0.0** | 98,585.5 | 103,814.0 | 0.0 | 494,637.8 | 1,086,817.3 |

Four things this table settles, and none of them is settled by the joint run alone:

1. **The planner touches ZERO branch parameters** — the isolation is real, not assumed.
2. **Each head alone reaches the trunk.** The PI's *"train jointly"* is a measurement here.
3. ⭐ **Detaching `fmap_s16` drives the trunk to EXACTLY 0.0 while each head keeps its own
   gradient** ⇒ the live number is *that head's* gradient, not somebody else's. A probe that read the
   same value either way would be measuring nothing.
4. **`heads_with_zero_grad: []`** in the joint arm — no member of the `tac_goal_tok_head` class.

⚠️ Note `trunk_n`: the planner's backward touches **58,231,872** trunk parameters, the perception
heads **30,682,176**. That is correct and informative — perception reads stride 16, so the stage-4
layers beyond it are reached only by the planner's stride-32 path.

The same instrument is wired into the trainer, so **every live run logs `ga_*` per head** and a
future regression is one grep away.

---

## 4. THE FRAME AXIS — the check the whole wiring rests on, done three ways

⛔ If the stacked-row → raw-frame conversion is off by one, every window is labelled with a map
0.1 s away (**0.83 m at 30 km/h**) and **nothing in the loss, the counts or the gradients would say
so**. Re-running my own expression would measure determinism, not correctness.
`code/frame_axis.py`, `raw/frame_axis.json`.

| probe | independent of my code because | result |
|---|---|---|
| **The join's own two fields.** The 3-D join emits `frame` (RAW v2ep) **and** `frame_idx` (post-trim); their difference is an **analytic target** = `n_stack - 1` | written by a builder that has never seen this trainer | `frame - frame_idx == 2` on **26,394 / 26,394 lines**, **one distinct value**, at `n_stack 3` ✅ |
| **Pose alignment** (`ClipMapGT.check_pose_alignment`): move each RAW v2ep pose by `v·(t_img − t_query)` and require it to land on the GT's own `T_world_rig` | a physical residual, not an index comparison | **9/10 clips ALIGNED**, 1 **INCONCLUSIVE** (an ego that barely moves cannot identify the axis — reported, not passed) |
| **Deliberate misalignment**: feed the poses rolled by one frame | the check must be able to fail | **CAUGHT on 10/10** ✅ |

⇒ The trainer reads the map at `t + w - 1` (the window's NOW **as a stacked-row index**), and
`MapGTStore.raw_frames` adds `n_stack - 1` itself. `n_stack` is **read from the v2 manifest**, never
a literal, and caches that disagree on it are refused.

### 4b. ⛔ THE SAME TRAP ON THE 3-D JOIN — swept, not derived

The coordinator flagged this seam mid-task, independently of my own probes: **`JoinFileReader` (the
2-D agent seam the trainer already had) keys on `frame_idx`, the EPISODE index; `AgentJoin3D` keys on
`frame`, the RAW v2ep index.** Join on the wrong one and every `cz`/`h` belongs to a moment ~0.2 s
from the frame the trunk saw — **finite, non-zero, entirely plausible, and nothing raises.**

I measured it rather than taking anyone's derivation. ⭐ **The reference is analytic, not fitted:**
the 3-D join is a *byte-preserving annotation* of the 2-D join (its own `meta.json`: *"every line,
key, field, agent and byte of the 2-D join is carried; cz and h are APPENDED"*), so at the correct
offset the two readers resolve **the same rows** and `cx/cy/yaw/l/w` must be **exactly equal**.
`code/offset_sweep.py`, `raw/join3d_offset_sweep.json`, 8 real clips:

| offset | agents compared | exactly equal | mean \|Δx\| |
|---:|---:|---:|---:|
| −3 | 553 | 0.0 % | 16.4607 m |
| −2 | 780 | 0.0 % | 19.5593 m |
| −1 | 807 | 0.0 % | 17.3723 m |
| **+0** ← the naive choice | 498 | **0.0 %** | **3.2401 m** |
| +1 | 1,110 | 0.0 % | 9.7181 m |
| **+2 = `n_stack − 1`** | **2,955** | **100.0 %** | **0.0000 m** |
| +3 | 564 | 0.2 % | 4.8776 m |

⇒ **+2 is the unique exact match, with infinite separation** (0.0000 m against a 3.2401 m best
elsewhere), and it is what `_agent_item` uses (`zh_for_frame(cid, f + n_stack - 1, …)`).
**`AGREES_WITH_TRAINER: true`.** The naive +0 would have mis-keyed every cuboid by **3.24 m of mean
agent displacement**.

⚠️ ⇒ `n_stack` is load-bearing on this path too, and a 0 would apply an offset of **−1**. The
`--join3d` branch now **refuses** a manifest that reports no `n_stack`, and the run stamps
`join3d_stats.frame_key = "RAW v2ep (= episode index + n_stack - 1)"` so the record says which
index space was used. Pinned in `tests/test_refcv6_perception_training.py`.

⭐ **Independent agreement, worth recording:** the end-to-end forward agent measured the same offset
against the parquet's own `center_z` and reported a clean minimum at RAW + 0 with **4.40×
separation**. Two different references (its `center_z` residual, my exact-row equality), two
different methods, same answer.

---

## 5. ⛔ THREE DEFECTS FOUND ON THE WAY — one of them was swapping the trunk of every 1024 run

### D-REFCV6-IMAGEHW-DROPS-TRUNK-FIELDS — the serious one

`_pin_trainer_cfg`'s `--image-hw` branch reconstructed `CNNEncoderConfig` from a **hand-listed
subset of its fields**, carrying `trunk`, `trunk_pretrained` and `trunk_imagenet_norm` and
**silently dropping four**. MEASURED on both trees through each tree's own `_pin_trainer_cfg`
(`code/imagehw_trunk_drop.py`, `raw/imagehw_trunk_drop.json`):

| flag | field | without `--image-hw` | **with `--image-hw 256 1024`** |
|---|---|---|---|
| `--trunk-name resnet34.a1_in1k` | `trunk_name` | resnet34.a1_in1k | **resnet101.a1_in1k** |
| `--trunk-mode inflate` | `trunk_mode` | inflate | **shared** |
| `--trunk-fuse last` | `trunk_fuse` | last | **concat1x1** |
| `--trunk-fuse-plain-init` | `trunk_fuse_identity` | False | **True** |

⛔ **The last two are CONTROL ARMS** — `last` is the single-frame control and the plain init is the
deliberate regression of the identity graft — so a run that asked for either got the **primary arm**
while `config.json`'s `seams` block recorded the control. And **`--image-hw 256 1024` is the PI's own
2026-09-16 geometry**, so every refcv6 run on the 1024 cache took this path.

**How it surfaced:** my first live run passed `--trunk-name resnet34.a1_in1k` and
`config.json.refcv6_perception.fmap_s16_channels` read **1024** — resnet101's stride-16 width, not
resnet34's 256. The branch's own provenance stamp caught it, because it records the channel count it
was **built from** rather than the one that was **asked for**.

⭐ **The mechanism was already named in this file, three lines above the bug**: *"the trunk pin goes
FIRST, because the `--image-hw` rebuild below reconstructs `CNNEncoderConfig` field by field and
would otherwise silently drop it"*. The fix had been applied to **one** field. ⇒ Replaced by
`dataclasses.replace`, which carries every field by construction and **cannot fall behind a new
one**. `tests/test_refcv6_perception_training.py` pins it **against a literal dict** and ships a
**mutation arm** that reinstates the historical field list and asserts it drops exactly those four.

**Evidence class:** MEASURED (ours) · both trees · `raw/imagehw_trunk_drop.json`
(`probe_detects_the_defect_on_the_tip: true`, `patched_is_clean: true`).

### `trunk_pretrained` had no CLI flag — the ImageNet knockout arm was unreachable from argv

`CNNEncoderConfig.trunk_pretrained` is a real field (`refc.py:348`, *"False is the knockout arm"*),
consumed at `refc.py:1439` and stamped at `refc_v3_train.py:3325` — and **no flag reached it**, so
*"the ImageNet prior helps"* was unfalsifiable: the comparison arm could not be built. The probe
records the tip exiting **2 (`unrecognized arguments: --no-trunk-pretrained`)** where the patched
tree reads `False`. `--trunk-pretrained` / `--no-trunk-pretrained` now exist; `None` (no flag) leaves
the dataclass field alone, so every banked arm is untouched.

### A false refusal that would have pushed operators into a different experiment

`--agents off` refused **any** `--agent-rig-camera`, on the correct grounds that a camera with no
consumer is a dead flag. With `--w-map > 0` the camera is **not** dead — the BEV lift reads the same
per-clip table — so a legitimate **map-only** arm was unlaunchable, and the way past the guard was to
switch the detector on and quietly run something else. Narrowed, with both directions pinned.

### ⚠️ And one defect of my own, caught by its own output

`enable_map_gt` first read a `states` key that `require_map_coverage` does not emit (it emits
`n_ok` / `n_no_file` / `n_frame_out_of_range` / `n_inconclusive` at the top level). It printed
**"0/23,772 windows covered"** beside **`frac_ok 0.9712`** — the total right and **every column
zero**. That is the exact diagnostic shape that function's own docstring warns about, reproduced one
layer up in its caller. Fixed; the live run now prints the full state breakdown and a `verdict`.

---

## 6. THE LIVE RUN — real corpus, real labels, CPU-only

`code/live.sh`. Artifacts: 256×1024 cache (139 clips) · SAM3 maps (135 of 139) · the 3-D agent join
(905,512 agent-frames) · per-clip extrinsics (141 clips).

**Census the run printed** (its own log, `raw/live_run/`):

| | |
|---|---|
| agent join | **139/139 episodes** (stable-63bit), **22,663/23,772 windows labelled (95.3 %)**, 776,801 prefilter boxes |
| SAM3 map coverage | **23,088 / 23,772 windows ok (`frac_ok` 0.9712)**, floor 0.90 → **PASS**; `no_file` **684**, `frame_out_of_range` **0**, `inconclusive` **0**; layout `flat_sha12` |
| 3-D cuboid join | 26,394 lines / **905,512 agents** / 139 clips |
| ⭐ arithmetic cross-check | `no_file` **684 = 4 clips x 171 windows**, and the map store resolves **135 of 139** clips ⇒ the missing-map count is exactly the missing-file count. An independent check that the columns are right, not merely that the total is |
| rig camera | **139/139 episodes** covered from a 141-clip bank, z 1.213–1.662 m |
| effective weights | `--w-map 1 … graph=yes TRAINS` · `--w-box3d 1 … graph=yes TRAINS` |

**A log row, in full** (step 1, `raw/live_run/metrics.jsonl`, resnet34 @ 256x1024, batch 2):

```
map 2.03011  n_map_cells 12119  map_n_windows 2  map_n_labelled 2
box3d 45.86354  box3d_presence 0.92481  box3d_cls 2.41769  box3d_centre 31.01495
box3d_size 4.91084  box3d_yaw 1.07066  box3d_rates 6.47004  box3d_occ 0.64650
box3d_z 1.75317  box3d_h 0.21315
box3d_n_matched 56  box3d_n_z 56  box3d_n_h 56  box3d_n_dropped 0
ga_trunk 772328.0 (n 22,268,480)   ga_lift 19531.2 (n 131,328)
ga_bev_encoder 93136.3             ga_map_head 75.8
ga_box_memory 142521.6             ga_box_decoder 1140681.0
```

⭐ **`fmap_s16_channels: 256` and `seams.trunk_name: resnet34.a1_in1k` in this run's own
`config.json`** — the `--image-hw` fix of section 5, confirmed on the artifact rather than in the
diff. The *first* live run, with the same argv on the tip's plumbing, recorded **1024** (resnet101).

⭐ **`box3d_n_z 62` is the number that matters**: the 3-D heights are not merely loaded, they reach
the loss on 62 of 62 matched agents. With no `--join3d` the mask is all-False and those two terms
report `n = 0` rather than training on zeros — the module's own documented 2-D rung.

⚠️ **Cost, so nobody mistakes this for a training run:** ~**300 s/step** at batch 2 on CPU
(resnet101 @ 256×1024). It is a wiring proof, not an arm.

---

## 7. What I could NOT do, and precisely why

| not done | why | what would unblock it |
|---|---|---|
| **Any GPU step, any convergence claim** | ⛔ the RTX 4060 was held by another package for the whole session; CPU-only was mandated by the brief | a free GPU. Nothing here is a capability claim and none of the four metric families is reported — this is a **wiring** result |
| **The 1 ms frame-time assertion** (`collate_map_targets(frame_t_us=…)`) | ⛔ **STRUCTURAL: the v2ep payload carries no `t_cam_us`.** MEASURED on a real clip — its keys are `actions, clip_id, codec, episode_id, frame, image_h, image_size, image_w, jpeg_buf, jpeg_len, n_stack, poses, projection_mode, quality`. There is no timestamp to pass | either add `t_cam_us` to the v2 cache builder (`scripts/v2_compressed.py`), or keep the **pose-alignment** check, which is arguably stronger: it compares physical positions, not clock labels, and is what §4 uses |
| **A 2-seed replicate of anything** | not applicable — no effect is claimed. Per `H-ESTIM-SEED-1`, a separated CI from one seed is necessary and not sufficient, so I have claimed **no lever effect at all** | — |
| **`--w-box3d` without `--agents head`** — exercised? | the live run used `--agents head --w-agent 1.0`, so the core's own 2-D slot decoder and the refcv6 3-D head both exist. They are **different heads**: `--w-agent` supervises the core's planner-facing agent seam, `--w-box3d` the refcv6 branch's own decoder off the stride-16 map | a PI decision on whether the two should be **one** head; they currently duplicate ~3.6 M parameters. Named here rather than resolved |
| **Eval-split perception census on a second cache** | the box has one 256×1024 cache; the eval path is wired and gated (`enable_map_gt`/`enable_join3d` run their **own** census on the eval split, never quoting the train number) but is **UNEXERCISED** | a second cache directory |
| **Map coverage on the TRAIN parity corpus** | the SAM3 maps on this box are the 135-clip **eval** set. The 0.9712 figure is that corpus's, not the 2,376-episode parity corpus's | the SAM3 production run on Thor (4,719 clips) landing |

### ⚠️ One thing I deliberately did NOT do

I did **not** touch `refc.py` or `refc_v3.py`. The `fmap_s16` seam they publish was sufficient, a
second stream is validating the forward pass in those files, and keeping the branch on the trainer
side is what makes the zero-weight path bit-identical **by construction** rather than by assertion.

---

## 8. Tests

`stack/tests/test_refcv6_perception_training.py` — **24 tests**, every guard proven by **mutation**:

* the zero-weight default, and `PerceptionBranchConfig` refusing to exist at it;
* **every** dead combination refusing **with its own named reason**, plus ⭐ the discriminating arm
  that the **live** combination and the **default** do NOT refuse (without it, every refusal test
  passes on a bare `raise SystemExit`);
* both weights in `REFC_WEIGHT_GATES`, with their **gates** asserted, not just their presence;
* gradient reach per head **with the detached-seam control**, plus the cross-check that the map loss
  leaves the box decoder at exactly 0 and vice versa;
* `--image-hw` carrying every trunk field **against a literal dict**, with the historical hand-listed
  rebuild as the **regression arm** that must report exactly the four dropped fields;
* the stacked-row → raw-frame arithmetic **against literals** (`raw_frames([0,1,7], 3) == [2,3,9]`),
  and the 3-D join's `f + n_stack - 1` lookup pinned in the source with the sweep as its evidence;
* the module registered with the existing geometry scanner rather than a second, weaker copy of it.

Plus one registration in an EXISTING guard: `tests/test_refc_v3_agent_provenance.py`'s knob probe
now carries the refcv6 perception seam (see §9 — that guard caught my change, correctly).

**Local set:** `test_refcv6_perception_training.py`, `test_refcv6_geometry_agnostic.py`,
`test_refcv6_perception.py`, `test_refcv6_perception_realdata.py`, `test_v6_effective_weights.py`,
`test_tac_goal_trainer_flag.py`, `test_refcv6_no_session_paths.py`,
`test_refc_v3_agent_provenance.py` — **189 passed, 2 skipped**. Full-suite result in §9.

---

## 9. Full suite — and what it caught

`raw/pytest_full.txt`, `raw/suite_delta.json`.

| run | result |
|---|---|
| patched tree, full suite | **8,127 passed**, 71 failed, 47 skipped, 4 errors (29 m 22 s) |
| ⭐ **tip control** — the SAME 22 failing FILES on unmodified `c16b7f1` | **70 failed, 4 errors** |
| **failure-set diff** | **only-on-patched: 1** · **only-on-tip: 0** |

⛔ **A raw "71 failed" is not evidence about a patch**, so the failure NODE IDS were diffed against
the tip's. **Exactly one failure was mine:**
`test_refc_v3_agent_provenance.py::test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE` — an
exhaustive probe that sets **every** `--agent*`/`--w-*`/`--bev-aux*`/`--wp-index*` knob to a
distinctive value and requires it to reach `config.json` **by value**. My refusals made every probe
value for `--w-map` inadmissible, so it reported the knob unstampable. ⭐ **The guard was right and
my code was incomplete**: the fix is to register the perception seam in the probe's enabling context,
exactly as WP-D and WP-B are registered there with comments explaining why. ⚠️ It needed a **real**
per-clip extrinsics file, because `_pin_trainer_cfg` reads it — a placeholder path would have made
the probe fail for a reason unrelated to the knob. Now **39/39 in that file**.

⚠️ **The other 74 failures and 4 errors are PRE-EXISTING on `c16b7f1`** — v6 (`test_loss_determinism`,
`test_v6_*`), tooling (`test_pod_git_drift`, `test_mktree_commit`, `test_secret_scan`,
`test_runbook_commands`), and two **collection** errors (`test_metric_decode_refusal.py`,
`test_refa_v1_dk_hook.py` — `ImportError` on symbols their modules no longer export) that reproduce
identically on the unmodified tree. **I did not investigate or touch them**; they are named here so
the next reader does not attribute them to this change.

**The perception + provenance + weight-gate set: 189 passed, 2 skipped.**

---

## 10. Deliverable manifest

| artifact | where it lives | only place? |
|---|---|---|
| `stack/tanitad/models/refcv6_perception_branch.py` | `worktree:tanitad-wt-perctrain` → **staged in the repo** | no |
| `stack/scripts/refc_v3_train.py` (CLI, dataset, losses, refusals, stamps, `--image-hw` fix) | same | no |
| `stack/scripts/train_p8_occupancy.py` (`with_track_ids`) | same | no |
| `stack/tests/test_refcv6_perception_training.py` | same | no |
| `stack/tests/test_refcv6_geometry_agnostic.py` (`OWNED` += the new module) | same | no |
| `stack/tests/test_refc_v3_agent_provenance.py` (the knob probe's seam registration) | same | no |
| `…/2026-09-17-refcv6-perception-training/RESULT.md` + `code/` + `raw/` | same | no |
| the two scratch run directories (`bitid/`, `live/`) | scratchpad only | **yes — deliberately**: multi-hundred-MB checkpoints of 3-step smoke runs. Every NUMBER quoted from them is banked in `raw/` |

⛔ **Nothing is committed and nothing is pushed.** Staged by explicit path only, **25 paths**, no
staged deletions, every blob verified `index == worktree` with a 40-character shape assertion on both
sides (an empty-string pair compares equal and prints MATCH — `CLAUDE.md`).

⚠️ **WHERE the index is, precisely.** The work is staged in **`worktree:tanitad-wt-perctrain`'s own
index** (`.git/worktrees/tanitad-wt-perctrain/index`), not in `C:/Users/Admin/tanitad-push`'s shared
one. That is deliberate: that tree tracks 11,862 files with ~9 on disk and its shared index carries
**1,297 staged deletions that are NOT phantoms** (coordinator, this session) — a pathspec-free commit
from there would land them.

⚠️ **The branch moved under me.** This work is built on `c16b7f1`; the coordinator landed `1371727`
(an ego-history dead-product fix, `refc.py` +21 lines) during the session. **I did not rebase** —
`refc.py` is untouched by this change, so the two are orthogonal, but whoever integrates should
rebase rather than merge blind.

## 11. Escalations for the Master Mind

1. ⛔ **`D-REFCV6-IMAGEHW-DROPS-TRUNK-FIELDS` needs a `RETRACTION_LOG.md` entry and a sweep of any
   banked arm launched with `--image-hw` + a trunk flag.** Its `config.json` `seams` block records
   what was **asked for**, not what was **built**, so the record and the run disagree. Root-cause
   class: *a rebuild that re-lists fields by hand* — the same family as the stale-index and
   derived-constant traps, with the object being a dataclass.
2. **`--w-agent` and `--w-box3d` are two slot decoders.** ~3.6 M duplicated parameters, two
   Hungarian matchers over the same targets. Whether the refcv6 3-D head should **replace** the
   core's 2-D agent seam is a design decision, not a wiring one.
3. **The map path needs `t_cam_us` in the v2 cache** to run the 1 ms assertion; until then the
   pose-alignment check is the frame-axis guarantee, and it should be run as a **preflight** on any
   real launch.
4. **`--no-trunk-pretrained` now exists** ⇒ the ImageNet-vs-random-init knockout is runnable. It was
   not before.
