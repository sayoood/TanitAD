# PREREG G-BOX-OVERFIT: can the refcv7 box head learn to detect on 16 real TRAIN frames?

**Registered:** 2026-09-27, about 02:20 Berlin, by the box-head audit (`…/2026-09-26-box-head-audit/`).

**Status: REGISTERED BEFORE ANY RUN.**
- No G-BOX-OVERFIT harness exists yet, and no refcv7 box number exists.
- The frame set was chosen from LABELS ONLY: GT cuboids plus the clip's own camera geometry (the VIS-1 z-buffer).
- No model output of any kind was read by the selector. Its only input, `vis_train_boxes.json`, carries no model field.

**Owners.**
- The refcv7 box-head builder implements the harness.
- The launch-gate agent enforces it.
- This audit wrote the protocol and the expected values.

**Changing any literal below after a run is a goalpost move.** It needs a dated amendment with the reason, and the run before it stays on the record.

**Binding source:** proposed as a `SPEC_REFCV7` amendment. The Master Mind registers it before any refcv7 box number exists.

---

## 1. What it proves, and what it does not

It proves that the refcv7 box path, **as built and configured for the launch**, can fit real 3-D boxes on real TRAIN frames. It checks four things at once:
- the wiring: image and BEV tokens reach the decoder;
- the loss: the presence, class and box terms all pull;
- the matcher;
- the decision rule: the declared presence gate turns a fitted head into a detector.

It is a **capacity test on 16 frames, so memorisation is the point.** It makes NO generalisation claim, and it is NOT a visibility test. The visibility rule has its own unit test and red arm (RESULT §2.4).

**Why it is needed.** MEASURED on 256 TRAIN windows of 64 TRAIN clips (`raw/analysis/analyze_train.json`), the refcv6 step-38,000 head reads on its own training data:
- AP@2 m BEV 0.230;
- precision 0.169 and recall 0.603 at its gate 0.5;
- best F1 0.326.

On eval it is no worse: AP 0.210 on 139 clips. The head **underfits**; it does not overfit. A head that cannot pass this test would repeat refcv6.

## 2. The frame set (FROZEN)

| item | value |
|---|---|
| file | `raw/visibility/gbo_frameset.json`, md5 **`b291404c36f83c3e397e3b90367e8e7b`** |
| selector | `code/gbo_select.py`, md5 **`f9f93f92b60a979b581b8f78be2ec6db`** (+ `code/vis_rules.py`, `code/vis_zbuf.py`) |
| selector input | `raw/visibility/vis_train_boxes.json` (its md5 is stamped inside the frame set) |
| split | TRAIN: the bha_dump train set, 256 windows of 64 train clips of `refcv6-b1-416x1024-train` |

**The rule** (literal, in the selector's docstring):
1. Label every row VIS-1.
2. A window is eligible when it has 3 to 15 POSITIVES.
3. Keep one eligible window per clip: the one with the most positives, ties to the smallest t.
4. Cover the classes. Taking clips in sha12 order, add clips until at least 3 frames each carry a POSITIVE `person`, `rider`, large vehicle (heavy_truck, bus, trailer or other_vehicle) and `automobile`. If a group has fewer than 3 available, take all of it.
5. Fill to 16 frames in sha12 order.

**The result:** 34 eligible clips, 16 frames, **113 POSITIVES** and 28 IGNORE rows.
- automobile 70, person 31, rider 8;
- bus 1, heavy_truck 1, trailer 1, stroller 1.

| clip sha12 | window t | POS | IGN | DROP | positives by class |
|---|---:|---:|---:|---:|---|
| `001e7d520dd0` | 149 | 5 | 1 | 0 | automobile 4, rider 1 |
| `03df23057f5a` | 107 | 11 | 2 | 8 | automobile 1, person 6, rider 4 |
| `086812263073` | 106 | 11 | 3 | 0 | automobile 11 |
| `0ddc926b7153` | 106 | 4 | 0 | 0 | automobile 2, person 1, trailer 1 |
| `1545b7c427e3` | 64 | 7 | 1 | 3 | automobile 6, rider 1 |
| `18af72139c3a` | 21 | 7 | 3 | 2 | automobile 7 |
| `205221007d48` | 149 | 4 | 0 | 0 | automobile 1, person 3 |
| `24354c935a5a` | 149 | 4 | 0 | 0 | automobile 3, person 1 |
| `2759fc498d64` | 21 | 12 | 10 | 31 | automobile 5, person 6, rider 1 |
| `2b162e47b23d` | 21 | 3 | 0 | 0 | automobile 3 |
| `384cb23868d0` | 149 | 13 | 1 | 0 | automobile 13 |
| `3d00847b9693` | 106 | 4 | 0 | 0 | automobile 4 |
| `412a12db2e6e` | 64 | 3 | 0 | 1 | automobile 3 |
| `50dd3fb9eb64` | 64 | 3 | 2 | 1 | automobile 2, person 1 |
| `545adb07993f` | 64 | 12 | 1 | 2 | automobile 2, person 9, stroller 1 |
| `df596ceb51e4` | 149 | 10 | 4 | 1 | automobile 3, bus 1, heavy_truck 1, person 4, rider 1 |

**Index convention.**
- t is the window start, and NOW = t + W − 1 = t + 7.
- The GT block is the trainer's own `V3Dataset._agent_item(ep, t + 7)`.
- The image is raw frame `t + 7 + n_stack − 1`, the newest of the stack (`v2_dataset.py:35-38`).

**VIS-1 per row.** A row must first pass the trainer's `visible_target_filter` (120° field ∩ decode box). Then:

| label | condition |
|---|---|
| POSITIVE | vis_frac ≥ 0.30 AND n_vis ≥ 100 px |
| IGNORE | 0.05 ≤ vis_frac < 0.30, OR vis_frac ≥ 0.30 with n_vis < 100 px |
| DROPPED | vis_frac < 0.05 |

- **vis_frac** is the visible AND in-image share of the cuboid's full silhouette. It comes from an exact per-pixel ray/cuboid z-buffer in the clip's own 416×1024 cylindrical camera: per-clip extrinsics, and the f-theta observed mask.
- **Occluders** are every GT row of the frame that has a 3-D label (`code/vis_zbuf.py::zbuffer`).
- **The harness MUST recompute these labels and match the file exactly.** That is control C3.

**Where it runs.** Canonical is Thor, where all 16 clips' v2ep payloads and the joins live. The dev box needs the 16 `.v2ep.pt` copied over the LAN (md5 recorded), and the GPU gate applies.

## 3. What trains, and how (LITERAL)

| item | value |
|---|---|
| model | the refcv7 LAUNCH box path: `Box3DMemory` + `Box3DSlotDecoder` as built by the launch `PerceptionBranchConfig`, reading the launch trunk's stride-16 map and the launch BEV features |
| trainable, GATING arm | exactly the launch's trainable set for the box loss: box decoder, box memory, and every module the launch lets the box loss reach (the BEV branch and the trunk, as launched) |
| trainable, INFORMATIVE arm `frozen_trunk` | box decoder + box memory only. Trunk and BEV branch are frozen at the launch init. Reported, not gated. |
| loss | EXACTLY the refcv7 launch box loss: the launch presence objective, VIS-1 POSITIVES as targets, the IGNORE mask (§4), the launch class weights (stamped by digest), z and h terms. A dry-run weights file is REFUSED. |
| optimiser | AdamW, lr **2e-4** constant for every trainable tensor, betas (0.9, 0.999), eps 1e-8, weight decay **0**, no warm-up, no clipping |
| batch / steps | **batch 4** frames per step by the harness's seeded `randperm` over the 16; **N = 2,000 steps** (500 passes) |
| seed / numerics | seed 0 (init, frame draws); fp32 in the head unless the launch runs another dtype there, in which case it is stamped |
| logging | every 100 steps: every loss term, AP@2 m, P/R at the declared gate, Σconfident / Σpositives, presence ECE, s/step, peak memory |

## 4. Scoring (the SAME rule the refcv7 eval uses)

- **Targets:** the 113 VIS-1 POSITIVES.
- **IGNORE rows:** 28. A confident detection that the greedy 2 m matcher pairs with an IGNORE row is removed from the PR count (KITTI DontCare). In the LOSS, an unmatched slot within 2.0 m (BEV) of an IGNORE row gets presence weight 0.
- **Matcher:** `box3d_head.box3d_match_rows(dist_thresh_m=2.0, use_z=False, score="presence")` + `ap_from_rows`, pooled over the 16 frames. Hungarian pairs come from `agent_slots.match_slots` on the POSITIVES.
- **Decision rule:** the launch config's DECLARED presence gate, on the probability the declared objective makes calibrated. The harness also reports the refcv6 rule (raw sigmoid ≥ 0.5) as information.

## 5. PASS (literal, committed now)

The GATING arm PASSES iff ALL of these hold at step 2,000 on the 16 frames, under §4:

1. **AP@2 m BEV ≥ 0.90.**
2. **Precision ≥ 0.90 AND recall ≥ 0.90** at the declared gate.
3. **Count calibration:** |Σ confident slots − 113| / 113 ≤ 0.10. Confident means at or above the gate, anywhere, IGNORE-matched slots excluded.
4. **Localisation, on Hungarian pairs:**
   - median centre error ≤ 0.30 m;
   - median |Δl| + |Δw| ≤ 0.30 m;
   - median |Δz| ≤ 0.15 m.
5. **Class:** argmax accuracy on greedy true positives ≥ 0.90.
6. **Liveness:** the loss is finite at every step, and the presence term at step 2,000 is ≤ 0.25 × its step-0 value.

**Why these numbers.** A 2-4 M-parameter decoder memorising 113 boxes in 16 frames should be near-perfect, so each bar sits well below perfection and far above refcv6.

| | refcv6, 38k, on TRAIN windows (MEASURED) | bar |
|---|---:|---:|
| AP | 0.230 | ≥ 0.90 |
| P / R at its gate | 0.17 / 0.60 | both ≥ 0.90 |
| Hungarian median centre error | 1.09 m | ≤ 0.30 m |

## 6. Arms that MUST FAIL, and controls that must read KNOWN values

| arm | the one change | required outcome |
|---|---|---|
| **R1 `memory_zeros`** | the memory entering the box decoder is `zeros_like` (image and BEV tokens). The loss and optimiser are unchanged. | **FAIL criterion 1** (AP < 0.90). With the same input on every frame, only frame-independent boxes are learnable. If it passes, the harness scores something other than vision, and the gate FAILS. |
| **R2 `presence_w0`** | the presence term's weight = 0 | **FAIL criteria 2 and 3.** The presence logit never leaves its prior, so recall at the declared gate is ≈ 0. |
| R3 `targets_filter_only` (INFORMATIVE) | refcv6's targets (visible_target_filter only: hidden rows are positives) | reported, not gated. Expected to fit too, because 16 frames can be memorised with hidden targets. This is why G-BOX-OVERFIT is not the visibility test. |

| control | must read |
|---|---|
| **C1** GT written into the slot format (`gt_as_slot_pred`) through the harness scorer | AP 1.0, P = R = 1.0, Σ confident = 113, exactly |
| **C2** constant presence | AUROC 0.5 exactly (tie-aware) |
| **C3** VIS-1 recount of the 16 frames | 113 POSITIVE and 28 IGNORE, per-frame counts equal to `gbo_frameset.json`, and the frame-set md5 equal to §2 |
| **C4** step 0 (the launch init) | reported. At a prior-initialised presence, recall at the gate is expected ≈ 0. |

## 7. What a result means

- **PASS.** The refcv7 box path CAN detect when allowed to memorise. The eval bars and the G-LIVE count check then decide whether it generalises.
- **FAIL.** The launch box path cannot even memorise 16 frames: wiring, loss or decision rule. **Do not launch** until it passes.
- **R1 or R2 not failing.** The harness is broken, and the result is VOID.
