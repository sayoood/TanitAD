# refcv6 — the perception branch: BEV + MAP head, 3-D boxes, and the planner's two couplings

**Date:** 2026-09-16, **rebased 2026-09-17** (Europe/Berlin) · **Branch:** `agent/arch-inf-20260803` @ **`8c7d215`**
**Spec:** `SPEC_REFCV6_V2.md` §1, §6, §7 (PI-approved 2026-09-16)
**Worktree:** `C:/Users/Admin/tanitad-wt-percep-v6` · CPU only (`CUDA_VISIBLE_DEVICES=""`)

## 0. ⭐ The PI's geometry ruling of 2026-09-16 (after the spec) — what ran where

**256 × 1024 cylindrical, same 120° field; trunk becomes resnet101 with
resnet34 as a comparison arm.** Nothing in this branch hard-codes either.

| | 256×640 (today's cache) | **256×1024 (PI)** |
|---|---|---|
| stride-16 map | 16 × 40 = **640** tokens | 16 × 64 = **1,024** tokens |
| per column | 3.000° | **1.875°** |
| stride-32 map | 8 × 20 | 8 × 32 |
| `f_ref` | 305.5774907364391 | **488.92398517830253** |
| field | 120.0000° | 120.0000° |

| trunk | stride-16 ch | stride-32 ch |
|---|---|---|
| `resnet34` | 256 | 512 |
| `resnet101` | **1024** | 2048 |

Channel counts are read from `timm.feature_info` at runtime
(`stack/tanitad/models/trunk_shapes.py`), never written as a literal, and
`tests/test_refcv6_geometry_agnostic.py` **tokenises** every module this branch
owns and fails on a bare 640 / 1024 / 1504 / 40 / 20 / 160 in executable code
(one exemption, written down with its reason).

⛔ **The SAM3 label grid does NOT move**: 120 × 64 @ 0.5 m, metric.
`assert_label_grid_unmoved` states that in code, with its own mutation.

### Which checks ran at which geometry

| check | geometry | data |
|---|---|---|
| lift geometry, BEV lift, map branch, box head — **all four** trunk × geometry combinations | **640 AND 1024** | **synthetic tensors** |
| lift built from a real clip's extrinsics at 1024 | 1024 | real extrinsics, synthetic features — `feat_hw (16, 64)`, **6,814 valid cells, identical to 640** (the field is unchanged; only the sampling is denser) |
| everything in §4 below — the 135-clip join, coverage, alignment, cuboids, the **orientation guard** | **640 only** | real frames + real SAM3 maps |

⚠️ The 256×1024 rebuild of the 139 eval clips
(`D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl/`) **did not exist on
this box** when these checks ran. The real-data checks therefore stay at 640, on
purpose, so the two geometries can be compared later; the 1024 readiness is
proven on synthetic tensors at the exact shapes.

⚠️ The PI wrote `f_ref 488.92`; the exact 1024/640 scaling is
**488.92398517830253**, which holds the field at exactly 120.0000° (the rounded
value gives 120.0010°, 0.008 px at the frame edge). The exact value is used and
the difference is pinned by a test rather than silently adopted.

## Headline — and the integration risks, first

1. ⛔ **The eval agent join has NO 3-D labels.** MEASURED on
   `b1eval_agents.jsonl.xz`: every agent is
   `{cx, cy, yaw, l, w, occ, track_id, cls}` — **no `z`, no `h`**. The cuboids
   themselves do carry both (`center_z`, `size_z` in every
   `obstacle.offline` parquet), so the 3-D head is trainable, but **a 3-D join
   has to be produced before any refcv6 box arm reports a z or h number.**
   Until then `zh_mask` is all-False and the z/h terms report `n = 0` — they are
   masked, never zero-filled. `tanitad/data/agent_cuboid_gt.py` is the reader
   that closes the gap; a Data-Engineering join pass is what makes it a corpus.
2. ⚠️ **A naming collision in `refc.py`.** Line 2931 calls **WP-B**
   *"DiffusionDrive coupling (1)"*; the spec calls the **BEV sampler** coupling
   (1) and WP-B coupling (2). A reader who trusts the in-file comment will
   conclude the BEV seam already exists. It did not.
3. ⭐ **Coupling (2) needed no code at all.** `agent_pos` is already threaded
   `refc.py:3390 → forward → _decode/_decode_ctrl → _agent_index →
   CrossAttnLayer._agent_bias`, with WP-B's output layer zero-init. WP-B is
   *"never run"* because no **arm** switched it on, not because a wire is
   missing. The patch therefore adds only coupling (1).
4. ⛔ **The patch changes an ordering invariant `refc.py` documents in prose.**
   `refc.py:2932-2941` states `attach_wp_index` is the last statement of
   `RefCModel.__init__` and that this *is* WP-B's removability proof. The patch
   makes the two attachments the last **two** statements, in a fixed order, and
   proves the stronger property: `integration/APPLY_NOTE.md` §"the one ordering
   rule". The applier must re-run `test_wp_index.py::test_shared_params_bit_identical`.
5. ⛔ **A session path blocked the landing, and it is now a guard.**
   `integration/verify_patch.py` embedded this session's scratchpad, whose
   directory name carries a UUID-shaped session id — and **nothing can tell a
   session UUID from a clip UUID by looking at it**. The script now derives the
   repo root from its own location and takes `--repo` / `--patch` / `--tmp`;
   `stack/tests/test_refcv6_no_session_paths.py` scans every file this branch
   owns for that class on every run, and is mutation-proven against the exact
   string that blocked it. ⚠️ Writing that guard exposed two bugs in the guard
   itself: it **matched its own fixtures** (every pattern is now assembled from
   fragments, so the file contains no whole pattern), and it looked only ONE
   line back for an `os.environ.get`, so the repo's own three-line env-var
   default read as a bare absolute path.
6. ⚠️ **`person` cuboids are not ground-standing on a small sample.** At 8 clips
   the bottom-face median read **0.608 m** and the ground-standing assertion went
   RED; over the full 145-file eval corpus (945,091 cuboids) it is **0.010 m**
   and the pooled median is **-0.004 m**. The 8-clip number was not a data
   defect, it was a sample too small to carry the claim — so the sample size is
   now part of the test rather than an accident of a default.

Everything below is **MEASURED (ours)** unless the row says otherwise.

---

## 1. What was built

| artifact | what it is | params |
|---|---|---|
| `stack/tanitad/models/bev_encoder.py` | BEV encoder (stride-1 only) + 9-class MAP head + soft CE on seen cells + per-class metrics | 786,153 |
| `stack/tanitad/models/box3d_head.py` | the **existing seam widened to 3-D**: `Box3DSlotDecoder`, `Box3DMemory`, `box3d_set_loss`, `box3d_ap` | 3,635,991 @640·r34 |
| `stack/tanitad/models/trunk_shapes.py` | ⭐ the PI's 2026-09-16 geometry, in ONE place: frames, strides, and channel counts read from `timm.feature_info` | — |
| `stack/tanitad/models/refc_bev_coupling.py` | DiffusionDrive coupling (1): BEV at the candidate's own waypoints, zero-init gate | 92,681 / layer |
| `stack/tanitad/data/agent_cuboid_gt.py` | recovers `center_z` / `size_z` — the two columns the join dropped | — |
| `stack/tanitad/data/perception_targets.py` | the loader hooks: store, **coverage refusal**, alignment assertion, collation | — |
| `stack/tanitad/data/lift_orientation.py` | the orientation guard's instrument (probe + sign test) | — |
| `stack/tanitad/data/semantic_map_gt.py` | ⚠️ MODIFIED (additive): `open_clip` split into `gt_path` + new `open_path`, so a non-canonical layout reaches the **same** validation | — |
| `integration/refc_wiring.patch` (+ `APPLY_NOTE.md`, `mkpatch.py`, `verify_patch.py`) | the wiring, **not applied** — `refc.py` was never touched | — |

### Parameter counts

⚠️ The block below is the **256×640 / resnet34** combination — the one the
real-data checks ran at. The branch builds at all four; only the pieces that
touch the image change, and the BEV side does not move at all:

| combination | `BEVLift` | `Box3DMemory` | `Box3DSlotDecoder` |
|---|---|---|---|
| 640 · resnet34 (16×40, 256 ch) | 131,328 | 91,136 | 3,544,855 |
| 640 · resnet101 (16×40, 1024 ch) | 524,544 | 287,744 | 3,544,855 |
| **1024 · resnet34** (16×64, 256 ch) | 131,328 | 91,136 | 3,643,159 |
| **1024 · resnet101** (16×64, 1024 ch) | 524,544 | 287,744 | 3,643,159 |

`BEVEncoder` (785,280), `MapHead` (873) and `BEVWaypointSampler` (92,681/layer)
are **identical in all four** — they live on the metric BEV grid, which does not
move. The decoder grows with the token count (its positional table is per token)
and stays inside `PARAM_BAND` (2–4 M) in every combination.


```
BEVLift (existing, 4960c66)     d_in=256 x4 heights -> 128              131,328
  BEVEncoder                    128 -> 96, dil (1,2,4,8), RF +-15.5 m   785,280
  MapHead                       96 -> 9 logits/cell (one 1x1)               873
  Box3DMemory                   1,120 tokens (640 image + 480 BEV)       91,136
  Box3DSlotDecoder              100 slots, slot width 21 -> 23         3,544,855
    (2-D AgentSlotDecoder, for the delta)                              3,544,341
  BEVWaypointSampler            ddv2-faithful, per decoder layer          92,681
                                tanitad-extension (offsets on)            96,793
perception branch total                                                4,553,472
3-D widening cost over the 2-D seam                                          514
coupling (1) at 6 decoder layers                                         556,086
```

`Box3DSlotDecoder` is inside `agent_slots.PARAM_BAND` (2–4 M) and enforces it at
construction, the unchanged `AgentSlotDecoder` rule.

## 2. Design decisions the spec left open, and what decided them

| decision | choice | why |
|---|---|---|
| where the MAP head predicts | the **label's own 120×64 cells**, stride-1 convs only | no resampling between prediction and label; a half-cell (0.25 m) shift cannot hide inside a mIoU. `BEVEncoder._assert_stride_one` refuses any pool / transpose-conv / stride-2 / wrong-padding layer |
| BEV encoder receptive field | dilations (1, 2, 4, 8) → **±31 cells = ±15.5 m** | four lane widths; the context a cell needs to know it is drivable |
| MAP head depth | **one 1×1 conv** | the ``BEVOccupancyHead`` `PARAM_BAND` precedent: a deep head measures itself, not what the lifted trunk carries |
| 9 classes or 8 | **9, `not seen` kept** | on a SEEN cell the not-seen fraction is < 0.5 but usually non-zero, and it is a real predictable state. The spec says "9-class" |
| 3-D field layout | `cz`, `h` **APPENDED** to `SLOT_FIELDS` | every 2-D slice keeps its offset; asserted at import, and the 2-D decode is bit-identical on the shared columns |
| Hungarian cost | **unchanged, 2-D** (`MATCH_INCLUDES_Z = False`) | switching 3-D on must not silently change WHICH slot matches WHICH agent, or an ablation compares two matchings rather than two heads |
| AP metric | centre-distance, all-point interpolation | 3-D IoU at 30 m is dominated by size error and hides a metre of range error behind a large box |
| learned waypoint offsets | **OFF by default, flagged when on** | the released DD samples AT the waypoints with no offsets (`ddv2_src/blocks.py:80-108`); `provenance()` returns `ddv2-faithful` / `tanitad-extension` |
| guard verdict | **exact binomial sign test**, not a win-rate threshold | per-clip AUC on 3 frames is noisy; a "must win 90 %" rule would have called the TRUE lift a FAIL at p = 4e-07 |

## 3. The mutation table

⛔ Every guard is proven by **re-introducing the defect** and showing the check
goes RED. `stack/tests/test_refcv6_perception.py` (33) +
`stack/tests/test_refcv6_perception_realdata.py` (14) +
`stack/tests/test_refcv6_geometry_agnostic.py` (20) +
`stack/tests/test_refcv6_no_session_paths.py` (4). **148 pass in 63 s** at the
DEFAULT invocation -- no env var, the orientation guard scoring all 135 clips --
including the 77 pre-existing tests of the modules this branch touches.

| # | guard | the mutation re-introduced | verdict |
|---|---|---|---|
| M1 | BEV grid is never resampled | append a `MaxPool2d` / `ConvTranspose2d` / stride-2 conv / `padding=0` conv to the encoder | **RED** (4 variants) |
| M2 | map loss is on SEEN cells only | drop the mask and supervise all 7,680 cells (with unseen cells labelled `not seen`, which LOWERS the loss) | **RED** — loss and `n_cells` both move |
| M3 | the target is a distribution | zero the `not seen` channel — every value still a legal fraction, so a range check cannot see it | **RED**; with `check_sum=False` it passes, which is the point |
| M4 | 2-D decode contract | interleave `cz`/`h` in the middle of `SLOT_FIELDS` instead of appending | **RED** at import |
| M5 | a missing height is MASKED | zero-fill `cz`/`h` and mark them valid | **RED** — `n_z > 0` and the total moves |
| M6 | `grid_sample` component order | emit `(g_row, g_col)` — DD's own `[..., [1, 0]]` swap, omitted | **RED**: cell (17, 40) reads 1,128 instead of 40 |
| M7 | BEV lateral sign | negate the lateral grid component | **RED**: a waypoint 8 m LEFT reads the cell 8 m RIGHT |
| M8 | coupling (1) is removable | set the zero gate to 1e-6 | **RED** — proves the seam is gated, not dead |
| M9 | coverage floor | 20 clips with no map added to the window list | **RED** — `MapCoverageTooLow` raised |
| M10 | a window's LABEL FRAME must exist | windows whose raw frame is past the clip's end | **RED** — 4 counted `frame_out_of_range`, verdict FAIL |
| M11 | frame alignment | pass RAW frames as windows (forget `+ n_stack - 1`) | **RED** — `TimeMisalignment` |
| M12 | clip identity from CONTENT | open clip A's real GT file as clip B | **RED** — `ClipIdentityMismatch` |
| M13 | a 2-D obstacle table | drop `center_z`/`size_z`; set `reference_frame = world`; set `size_z = 0` | **RED** ×3 — reported, never imputed |
| M14 | **the orientation guard itself** | hand the guard a MIRRORED lift as the "true" geometry | **RED** — verdict FAIL, 31/115 wins |
| M15 | the lift's lateral sign, per clip | mirror the column index | **RED** on all 135 clips |
| M16 | the guard must refuse leakage | same clips in the fit and eval sets | **RED** — refused |
| M17 | no head pins a geometry | feed a 16×64 map to a head built for 16×40, and the reverse | **RED** — "geometry mismatch, not a resize" |
| M18 | no head pins a trunk | feed 1024-channel (resnet101) features to a head built for 256 (resnet34) | **RED** — "the channel count is a PARAMETER" |
| M19 | the label grid is metric | scale the label grid with the image (64 → 102) | **RED** |
| M22 | no session id / scratchpad / dead mount in a repo artifact | inject the exact blocking string into a copy of a real owned file | **RED** on all four classes (uuid, scratchpad, dead mount, home-abs) |
| M23 | the path guard must not match itself | spell a forbidden pattern in the guard's own source | **RED** — which is why every pattern is assembled from fragments |
| M21 | the guard must be POWERED, not merely correct | run the guard at the old default n (20 held-out clips) | **RED** — 14/20, p = 0.058; power is 0.24, so it now SKIPS with that number instead of asserting |
| M20 | the literal cannot come back | a token scan of every owned module, then the same scan on a file with `(16, 40)`, `640`, `1024` re-introduced | **RED** on all three; exemptions must be written down with a reason |

⭐ **Three of my own guards were too weak, and all three are recorded in the
test that replaced them** — a guard that cannot fail is the defect it was meant
to catch, wearing the other hat:

1. **M2 (the `seen` mask), twice.** *"the two losses differ by more than 1e-3"*
   on random logits — they differ by **6.6e-4**, so it would have passed a build
   with no mask on most seeds. Its replacement, *"the unmasked loss prefers a
   degenerate predictor"*, was simply **false** (MEASURED 4.9795 degenerate vs
   4.4464 good — the ranking never inverts). What holds exactly, with no
   threshold and no seed: **with the mask an unseen cell receives exactly zero
   gradient; without it, non-zero.**
2. **M2 was also ORDER-DEPENDENT**: it read the global RNG, so adding the
   geometry suite changed its verdict. It now uses an explicit generator.
3. **M20's first form was a line grep** and flagged the PI's own geometry prose
   inside a docstring. It now **tokenises**, so strings and comments cannot be
   mistaken for code.
4. **The orientation guard itself was UNDERPOWERED on its default run**
   (caught by the coordinator, not by me): power **0.24** at the n it ran at, so
   it reported a correct lift as RED. It now scores every scorable clip, and the
   threshold is computed from a power calculation rather than chosen. Details in
   §4.

⭐ **M10 found a real defect in my own code.** Caching a sentinel `n_frames = -1`
for a failed clip made every window after the first report
`frame_out_of_range` instead of `no_file`: the total was right and **every
column was wrong**. Fixed in `perception_targets.require_map_coverage`; the
comment there carries the story.

## 4. Real-data checks

Fixtures on this box: `D:/Projects/TanitAD-artifacts/sam3-maps-eval` (135
`*.sam3mapgt.npz`, named by **sha12**), `C:/Users/Admin/refcv5cmp/data/eval`
(141 `*.v2ep.pt`), `…/hf-corpus-aug-20260915/stage/calibration/sensor_extrinsics.parquet`
(4,719 clips), `…/labels/obstacle_offline_b1eval` (**145** parquets + a
`pull_manifest.json`).

| check | result |
|---|---|
| clips joining **map + calibration + pixels** | **135 / 135** |
| every map opens as its own clip (identity from `meta.source.clip_sha12`) | **135 / 135**, all layout `flat_sha12` |
| coverage on 40 clips × real windows | `PASS`, `frac_ok = 1.0000`, `n_inconclusive = 0` |
| alignment `t_img_us` vs the window's frame | worst \|dt\| = **0.0 µs** over 12 windows/clip |
| seen share of a real frame | 0.917 (clip `0dfe63214bc2`, frame 2) |
| `cuboids_at_time` cols 0-5 vs `agents_at_time` | **BIT-IDENTICAL**, 939 rows over 6 clips × 7 times |
| cuboid bottom face, whole eval corpus | **945,091** cuboids / 145 files; pooled median **-0.004 m**, automobile **-0.008**, person **0.010**, heavy_truck **-0.049**, `protruding_object` **+0.861** (airborne, named) |
| end-to-end: real geometry → `BEVLift` → `BEVEncoder` → `MapHead` → soft CE | runs; `loss.backward()` reaches the trunk stand-in (`grad.abs().sum() > 0`) |

### The orientation guard

⛔ *"the lift must put SAM3 drivable cells under the image's road pixels on real
frames, and a left-right MIRRORED index must go RED"*.

**The instrument.** A first attempt scored image intensity at the lifted
positions against a hand-designed road cue: it separated drivable from
non-drivable at **AUC 0.707**, but the MIRROR still scored **0.606** and won on
**39 %** of frames, because the ROW of a lifted cell survives a left-right
mirror and carries most of that separation. An instrument a mirror passes 39 %
of the time is a C9/C14 instrument. It was discarded.

What is reported instead runs **the MAP head's own operation in miniature**: a
logistic probe on **pure-image** features (the frame average-pooled to the
trunk's stride-16 grid, sampled through the lift's own `grid` at all 4 heights →
12 dims), fitted on TRAIN clips, scored on **HELD-OUT** clips. The cell's own
(x, y) is deliberately not a feature — with it the probe could learn the BEV
prior and score through a mirror.

| | AUC mean | AUC median |
|---|---|---|
| **TRUE** lift | **0.7164** | 0.7136 |
| **MIRRORED** index | 0.6544 | 0.6684 |

* 135 clips, 3 frames each, **1,622,484** scored cells; 20 fit / **115 held out**.
* TRUE beats MIRROR on **84 / 115** clips, **p = 4.01e-07** (exact sign test),
  mean margin **+0.0620** → **PASS**.
* **M14, the guard's own mutation:** hand it the mirrored lift as "true" →
  **31 / 115**, p = 1.00 → **FAIL**. The guard can fail.
* Stable across fit size: 60 fit / 75 held out gives 54/75 (0.720), margin +0.058.

#### ⛔ The guard was UNDERPOWERED on its own default run, and that is now fixed

The coordinator reproduced a **FAIL** from the default invocation:
``AUC 0.6770 true vs 0.6056 mirror, wins 14/20, p = 5.77e-02 → FAIL``. The
direction was right and the margin (+0.0714) was *larger* than the headline —
the test simply had no power. The first version sliced
``clips[:max(N_ORIENT, 12)]``, scoring ~20 held-out clips.

⛔ **At n = 20 the sign test's power is 0.24** (0.34 even at the best observed
win rate). A correct lift was expected to be reported RED about three runs in
four. A guard that goes red by default cannot be told apart from the defect it
exists to catch — the `R-2026-09-08-wpa-mirror` failure mode wearing the
opposite mask. It was a defect in the instrument, not in the lift.

**The fix, chosen rather than tuned.** The test now scores **every scorable clip
in the map store** by default — no sampling, no subset. The threshold is
*computed*, not picked: `lift_orientation.min_n_for_power(0.95, 0.70, 0.01)`
returns **96**, the smallest held-out n at which the test rejects with
probability ≥ 0.95 when the true win rate is **0.70** — the LOWEST of the three
rates this instrument actually measured (0.7304 / 0.7200 / 1.0000), because a
power calculation done at the rate you hope for is not a power calculation.
`MIN_CLIPS_FOR_POWER` is pinned against that function by a test, so the constant
and the calculation cannot drift.

| held-out n | power at win rate 0.70 | what the test does |
|---|---|---|
| 20 (the old default) | **0.24** | ⛔ would assert — and did, wrongly |
| 50 | 0.68 | skips |
| 95 | 0.9491 | skips |
| **96** = `MIN_CLIPS_FOR_POWER` | **0.9546** | asserts |
| **115** (today's default) | **0.9772** | asserts |

Below 96 it **SKIPS** with a message naming the n it needs and the power it
actually has — a skip is honest, a red guard is not. `$TANITAD_ORIENT_CLIPS`
still caps the run for a fast local check; that is a deliberate
under-powering, so it skips. MEASURED at the cap:

    SKIPPED: UNDERPOWERED, not failed: 10 held-out clips (30 scorable - 20 spent
    on the fit) is below MIN_CLIPS_FOR_POWER = 96 ... At n = 10 the power is only
    0.03, so neither a PASS nor a FAIL here would mean anything.

⭐ **The mutation direction needs no power argument and is asserted at ANY n:**
under a mirrored lift the win rate is ~0.27, so a one-sided test cannot reject
however few clips there are. The guard is conservative in exactly the direction
that matters, and `test_MUT_the_guard_itself_fails_when_the_lift_is_mirrored`
still passes at the capped n.

**At the new default** (`pytest tests/test_refcv6_perception_realdata.py`, no env
var): 135 clips joined, 135 scorable, 20 fit, **115 held out, power 0.977**,
`PASS` — and the whole file is **14 passed in 43.8 s**.

⚠️ **The mirror does not fall to 0.5 and must not be expected to** — a left-right
mirror preserves the row, so the vertical structure of a road scene survives it.
The guard is the **paired** comparison, not an absolute threshold on either arm.

**The parameter-free half**, independent of image content: a rig point at +y
(LEFT) must project left of the frame centre. Checked on all **135** clips'
own extrinsics — **135/135 pass**, and the mirrored column index fails.

## 5. The integration patch — rebased onto `8c7d215`, measured not argued

`integration/refc_wiring.patch` (347 lines, 22 hunks), built by editing a
**copy** of `refc.py` and diffing. `git apply --check` at `8c7d215`: **clean**.

⚠️ **The first cut was against `9a782fa` and did not apply.** `8c7d215`
("refcv6 core: ImageNet timm trunk with frame + ego history at 256x1024, and all
of F1-F9") took refc.py from 3,576 → 4,035 lines. Three anchors moved, and one
of the three was a live trap:

| the landed change | consequence for the patch |
|---|---|
| `RefCModel.forward` grew `ego_poses` / `ego_n_past` | signature anchor moved — re-cut |
| the decoder call grew `ego_hist=ego_vec` | threading anchor moved — re-cut |
| ⛔ **F3/F4 split `_decode_ctrl` into TWO layer loops** | patching only the fast path would have left coupling (1) **silently dead in every F3/F4 arm** — the arm refcv6 exists for |
| the timm factory + `CNNEncoderConfig.s16_dim` | `d_bev` became a REQUIRED argument; a **new edit 11** exposes the stride-16 map |

### ⭐ New in the rebase: the stride-16 perception seam

`TimmResNetTrunk.forward_features` returns `(s16, s32, pooled)` and `forward`
**discards `s16`**. refcv6's perception branch reads stride 16, so without this
seam a caller must run the backbone a second time for a map the trunk already
built. Edit 11 emits it as `out["fmap_s16"]`.

⚠️ An earlier cut wired only the last-frame branch and called the hierarchy
branch *"reported rather than half-wired"* — but `refc_smoke_config()` and every
hierarchy arm set `hierarchy = True`, so the seam would have been **dead on the
path actually taken**. Both branches are now wired. MEASURED:

| trunk | `hierarchy=False` | `hierarchy=True` |
|---|---|---|
| `resnet34.a1_in1k` | `(2, 256, 16, 64)` | `(2, 256, 16, 64)` |
| `resnet101.a1_in1k` | `(2, 1024, 16, 64)` | `(2, 1024, 16, 64)` |

— the PI's 16×64 geometry, channels from `feature_info` via `s16_dim`.

### ⭐ Correction C5 applied

Three `refc.py` comments called WP-B *"DiffusionDrive coupling (1)"*; the spec
numbers it **(2)**. All three are corrected by the patch (no behaviour change),
so the code and the programme record agree.

### The five checks, against the NEW baseline

`python integration/verify_patch.py` — copies `refc.py` into a temp tree,
applies the patch there, imports both modules in one process. ⛔ No machine- or
session-specific path; `--repo` / `--patch` / `--tmp` override.

| check | result |
|---|---|
| A. coupling off: parameter set + values vs the UNPATCHED module | 136 tensors, **0 differ** |
| B. forward on **64 fixed windows**, coupling off | **64/64 BIT-IDENTICAL** |
| C. coupling on | +3,466 params / 2 layers; `provenance = ddv2-faithful`; `n_points = 4 = n_steps`; shared params still bit-identical |
| D. gates at init | `[0.0, 0.0]`; gate 0 → exact identity, gate 1 → output changes |
| E. WP-B heads across BEV on/off (`cross_agent=True`) | 8 tensors, **0 differ** |

### Both sampler loops reach the coupling

Gate 0 vs gate 1 through `_decode_ctrl`, all four F3/F4 settings:

| `f3_per_layer` | `f4_adaln` | loop | max\|Δ\| conf |
|---|---|---|---|
| 0 | 0 | FAST | 1.0406e-01 |
| 1 | 0 | CASCADE | 4.4307e-01 |
| 0 | 1 | CASCADE | 1.0763e-01 |
| 1 | 1 | CASCADE | 3.8681e-01 |

⚠️ `max|Δ| du` is **0.0000e+00** in every row, and that is correct rather than a
null result: `control_head` is zero-init, so the first pass predicts exactly the
current state whatever the queries say. The coupling shows in **conf**; printing
only `du` would read as "live, delta 0" — true, and wrong for the reader.

### The repo's own suites, against the APPLIED patch

⚠️ **The control matters here.** Run in a throwaway copy of `stack/`, eight
tests fail — and **they fail identically in an UNPATCHED copy**, because the
copy has no sibling `taniteval/` or data directories. The honest comparison is
patched-copy vs unpatched-copy, same invocation:

| | failed | passed | skipped |
|---|---|---|---|
| UNPATCHED copy | 8 | 327 | 5 |
| **PATCHED copy** | **8** | **327** | **5** |

**Zero failures attributable to the patch**; every captured failure name is
common to both. Narrower, cleaner run: `test_wp_index.py` (incl.
`test_shared_params_bit_identical`) + `test_refcv6_diffusion.py` +
`test_refcv6_trunk.py` against the applied patch = **108 passed, 1 skipped**.
`test_wp_index.py` alone: **35 passed**, patched and unpatched.

⚠️ Had I reported the first number without the control, this would have read as
"the patch breaks 8 tests". It breaks none. (Project rule: verify before
alarming — three false alarms in one session once came from surface reads.)

## 6. What is owed

1. ⛔ **A 3-D agent join.** Data Engineering: re-run the join carrying
   `center_z` / `size_z` (or emit a side-car keyed by `(clip, frame, track_id)` —
   `agent_cuboid_gt.zh_by_track` is the lookup). Until then **no refcv6 arm may
   report a z or h number**; the counts will say `n_z = 0`.
2. **SAM3 maps for the TRAIN clips.** Only the 135 EVAL maps are on this box;
   the spec expects the full corpus ~22 Sep. `require_map_coverage` refuses a run
   under 0.90 — that refusal is the gate.
3. **The gradient-conflict detector** (spec §6: +1 / 0 / −1 analytic controls,
   30× mutation) is **not** in this branch.
4. **The trunk.** `timm resnet34.a1_in1k` (spec §2) is another agent's scope;
   everything here takes the stride-16 map as an argument and asserts 16×40.
5. Re-run `test_wp_index.py::test_shared_params_bit_identical` after applying.

## 7. Evidence classes

| claim | class |
|---|---|
| all parameter counts, mutation verdicts, patch checks A–E | **MEASURED (ours)** — `pytest` + `integration/verify_patch.py`, CPU, this box |
| orientation AUCs, win counts, p, cell counts | **MEASURED (ours)** — 135 clips, artifacts named in §4 |
| cuboid bottom-face statistics | **MEASURED (ours)** — 945,091 cuboids, 145 parquets |
| "the eval join has no z/h" | **MEASURED (ours)** — read from `b1eval_agents.jsonl.xz` |
| DiffusionDrive has no learned offsets | **RELEASED CODE** — `…/ddv2_src/blocks.py:80-108` |
| oracle AP 0.3341 (8×20) vs 0.4713 (16×40) | **QUOTED** — `SPEC_REFCV6_V2.md` §2, not re-measured here |
| "87,481 cuboids, ground-standing" | **QUOTED** — spec §6; the independent 945,091-cuboid measurement above agrees |
