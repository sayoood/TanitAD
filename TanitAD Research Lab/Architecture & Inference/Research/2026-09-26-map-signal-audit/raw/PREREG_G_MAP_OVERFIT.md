# PREREG G-MAP-OVERFIT — the 10 cm map head must be able to learn EVERY class

**Registered:** 2026-09-26, ~22:50 Berlin, by the map-signal audit (`…/2026-09-26-map-signal-audit/`).
**Status:** REGISTERED BEFORE ANY RUN. No G-MAP-OVERFIT harness exists yet and no NEW-2 number exists.
The only inputs used to write it are SAM3 LABELS (the frame set, selected label-only on Thor) and the
analytic ceilings in `raw/analytic_class_signal.json`. No model output of any kind was read.

**Owners:** the NEW-2 builder implements the harness; the launch-gate agent enforces it; this audit
wrote the protocol and the expected values. Changing any literal below after a run is a goalpost move:
it needs a dated amendment with the reason, and the run before it stays on the record.

**Binding source:** `Project Steering/SPEC_REFCV7.md` §8 (Amendment A3), item 3.

---

## 1. What it proves, and what it does not

It proves the 10 cm head **as built and configured for the launch** can fit every one of the 8 SAM3
classes on real TRAIN frames. It checks three things at once: the wiring (the stride-8 map reaches the
head), the loss (the class weights and the mask give every class a signal), and the resolution (the
lift plus decoder can express 10 cm structure).

It is a **capacity test on 16 frames, so memorisation is the point.** It makes NO generalisation claim.
The refcv7 eval bars (BAR-M7-1..4) are separate.

## 2. The frame set (FROZEN)

- **File:** `raw/gmo_frameset.json`, md5 **`4eafa03c2b6a6e6d6336be1d78acb91d`**.
- **Selector:** `code/gmo_select_train_frames.py`, md5 **`08a921597f222106da4c062a09754869`**.
  - It ran read-only on Thor over labels only.
  - v1 picked three identical frames of a stationary clip. It is kept as
    `raw/gmo_frameset_v1_SUPERSEDED_duplicate_frames.json`, and v2 requires distinct label maps.
- **Split:** TRAIN (`/home/nvidia/data/refcv6_train_clips.txt`). All four clips are in the refcv6 train cache.

| clip sha12 | chosen for | raw v2ep label frames r | window t (= r − 9) |
|---|---|---|---|
| `10497f0d664b` | hatched | 10, 110, 130, 150 | 1, 101, 121, 141 |
| `05c575ed45be` | arrow / text | 50, 85, 105, 130 | 41, 76, 96, 121 |
| `104d79ae052d` | crosswalk | 10, 90, 115, 135 | 1, 81, 106, 126 |
| `0dbd6c9cd776` | non-drivable edge | 15, 35, 60, 90 | 6, 26, 51, 81 |

**How t follows from r.** `t = r − (W − 1) − (n_stack − 1)`, with W = 8 and n_stack = 3. Sources:
`refc_v3_train.py:3185` and `semantic_map_gt.raw_frame_index`.

**Time guard.** The harness MUST assert, per frame, that the label's `t_img_us` equals the window's
current-frame timestamp within 1 ms (`ClipMapGT.check_times`). This is the training guard, applied unchanged.

**Label census** (fine_codes ≠ 255, rig x 0–20 m, APPROXIMATE camera cone), over all 16 frames:

| class | cells |
|---|---:|
| seen-no-class | 50,123 |
| drivable | 351,454 |
| lane / road line | 9,003 |
| crosswalk | 44,528 |
| arrow / text | 3,475 |
| non-drivable edge | 7,820 |
| hatched | 18,220 |
| sidewalk / verge | 190,081 |

The harness recounts these with the EXACT mask of §4 and prints them.

**Where it runs.** Canonical is Thor, where all four clips' v2ep and GT live. On the dev box:
- only `05c575ed45be` is in the local A6 train sub-cache;
- the other three clips' `.v2ep.pt` files and all four GT files must first be copied over the LAN, with
  their md5s recorded;
- the dev-box GPU gate applies.

## 3. What trains, and how (LITERAL)

This table is conformed to the builder's harness, `…/2026-09-26-refcv7-map-hires/code/fix/stack/scripts/map_hires_overfit.py` (`run_arm`), as read at 22:50. The machine-readable spec is `raw/gmo_spec.json`.

| item | value |
|---|---|
| model | the refcv7 LAUNCH build: the NEW-2 branch (`MapHiresBranch`, launch `MapHiresConfig`) and the launch trunk (`resnet101.a1_in1k`, frozen BN, `equalize_bottom_rows` 43, the stride-8 tap on) |
| trainable (GATING arm) | the NEW-2 branch AND the trunk's stem..stride-8 stage (`_s8_params`), exactly as `run_arm` does for `healthy`. This is the launch path, where the trunk trains jointly. |
| trainable (INFORMATIVE arm `frozen_trunk`) | the branch only; the trunk frozen at init. It is reported, not gated, and it is the stricter capacity read of the head. |
| loss | EXACTLY the launch 10 cm loss (`hires_map_ce`): hard-label CE on `fine_codes`, `ignore_index` 255, the lift-valid narrowing, the FROZEN launch class-weight vector. A dry-run weights file is REFUSED (`load_class_weights` rule). The sha256 goes in the record. |
| optimiser | AdamW, lr **1e-3** constant for every trainable tensor, betas (0.9, 0.999), eps 1e-8, weight decay **0**, no warm-up, no clipping |
| batch / steps | **batch 4** frames per step, drawn by the harness's seeded `randperm`; **N = 1,000 steps** (≈ 250 passes over the 16 frames) |
| seed / numerics | seed 0 (branch init, frame draws); fp32 unless the launch map path runs another dtype, which is then stamped |
| logging | per-class IoU under BOTH rules (§5), per-class CE, every 100 steps; s/step; peak memory |

## 4. The scored cells (the SAME mask as the loss)

The scored cells are the intersection of three conditions:
- `fine_codes ≠ 255`;
- the map-only lift's valid mask (the camera reaches the cell at THIS instant): any height valid, with
  the trunk's bottom-row `observed` mask, nearest-upsampled to 10 cm;
- rig x in **[0, 20) m** (rows 0–199).

Two further rules:
- **IoU is POOLED over the 16 frames:** Σ intersection / Σ union per class, never a mean of per-frame IoUs.
- **The 20–40 m and 40–60 m bands are REPORTED, NOT GATED.** MEASURED from real geometry
  (`raw/lift_resolution.json`), the road plane in those bands falls in only 3 and 2 distinct stride-8
  feature rows, and 20 and 7 image rows.

## 5. The decision rule (DECLARED, then gated on)

The harness reports per-class IoU under BOTH rules below, and GATES on the one the launch config
declares for inference and eval:
- **raw:** `argmax_c z_c`;
- **prior-corrected:** `argmax_c (z_c − log w_c)`, where w is the loss's class-weight vector. This
  equals raw when w is uniform, and control C3 checks that.

⚠️ **Why both.** Under a weighted CE the softmax learns `q_c ∝ w_c · P(c | x)`, so the raw argmax is a
rare-class detector:
- with median-frequency weights, "lane" beats "drivable" at a 5.0 % posterior under MF-present (the
  builder's definition), or 4.6 % under MF-global;
- "arrow/text" beats it at 0.65 % (MF-present), or 0.19 % (MF-global). These are the eval-kit proxy
  weights of `raw/analytic_class_signal.json`; the launch weights come from TRAIN.

ANALYTIC ceiling, 0–20 m, a calibrated posterior blurred by σ = 0.10 m:

| class | raw argmax, MF-global weights | raw argmax, MF-present weights (the builder's definition) | prior-corrected argmax |
|---|---:|---:|---:|
| lane | 0.51 | 0.52 | 0.91 |
| crosswalk | 0.56 | 0.60 | 0.95 |
| arrow | 0.38 | 0.43 | 0.89 |
| edge | 0.21 | 0.21 | 0.73 |
| hatched | 0.48 | 0.54 | 0.92 |
| drivable | 0.91 | 0.91 | 0.99 |

Source: `raw/analytic_class_signal.json` → `decision_rule_ceilings_0_20m`.

## 6. PASS (literal, committed now)

The MAIN arm PASSES iff ALL of these hold at step 1,000, under the declared rule of §5:

1. **Presence:** every one of the 8 classes has **≥ 1,000 scored cells**. If any class is short after
   the exact mask, the result is **INCONCLUSIVE ⇒ FAIL**: the set was chosen to contain all 8.
2. **Big classes:** seen-no-class, drivable and sidewalk/verge each reach IoU **≥ 0.85**.
3. **Thin classes:** lane/road line, crosswalk, arrow/text, non-drivable edge and hatched each reach
   IoU **≥ 0.50**.
4. **Every class learned:** each class's mean per-cell CE over its own scored cells at step 1,000 is
   **≤ 0.5 ×** its value at step 0.
5. **Liveness:** the loss is finite at every step.

**Why these numbers.** A head that localises to σ ≈ 0.10 m would pass every class by a wide margin
under the calibrated rule (§5 table). The analytic ceiling for non-drivable edge falls **below 0.50**
once σ grows to ~0.2 m, because it collapses to **0.04** at σ = 0.20 m. Such a head is not a 10 cm
head for edges, and it SHOULD fail.

## 7. Arms that MUST FAIL, and controls that must read KNOWN values

| arm | the one change | expected | required outcome |
|---|---|---|---|
| **R1 `lane_w0`** | the lane/road-line class weight = 0 in the loss; all else identical | lane IoU ≤ 0.05: its logit is never pulled up (`gno` = 0) | **FAIL on lane** |
| **R2 `s8_zeros`** | the tensor entering the map-only lift is `zeros_like(fmap_s8)`. This is the F3-whitelist class: a declared key that never arrives. The trunk still trains, but receives no map gradient. | every thin class IoU ≤ 0.20: with no image information only a constant/border map is learnable | **FAIL on ALL FIVE thin classes**. Any thin class passing means the harness scores something other than vision, and the gate FAILS. |
| R3 refcv6 head (informative; gating if built) | refcv6's 0.5 m branch design on the same frames (stride-16 lift, soft CE on `cart_frac`), argmax nearest-upsampled to 10 cm | non-drivable edge IoU ≤ 0.10: the 0.5 m target keeps 0.9 % of edge area | **FAIL** |
| `s8_detached` (informative, NOT gated) | `fmap_s8.detach()`: the trunk's stride-8 stage gets no gradient | the branch still sees ImageNet stride-8 features, so it may PASS. Reported as the frozen-trunk read. | none |
| `frozen_trunk` (informative, NOT gated) | the branch only trains | reported | none |
| **C1 constant** | "drivable everywhere" | IoU_drivable = n_drivable / n_scored exactly (to 1e-9); every other class 0.0 | reproduced exactly, and FAILS §6 |
| **C2 GT as logits** | logits = 20 · one-hot(label) | IoU = 1.0 for all 8 exactly | reproduced exactly |
| **C3 rule identity** | w = ones | the prior-corrected and raw argmax are bit-identical | reproduced exactly |

⚠️ **Why `s8_detached` cannot be a must-fail arm here.** Detaching the trunk does not remove image
information. The branch can still memorise 16 frames from ImageNet features. A gate that REQUIRES it
to fail would fail a healthy design, since `map_hires_overfit.py`'s rule is that "a regression arm that
passes ⇒ FAIL". The detached-seam control belongs to G-LIVE (`grad_abs_sum` of the stride-8 stage
must be > 0 attached and exactly 0 detached), and the builder's unit test already pins it.

## 8. The G-MAP-OVERFIT verdict

**PASS** requires all of:
- MAIN PASS (§6);
- R1 FAIL and R2 FAIL;
- C1, C2 and C3 reproduce their known values;
- if R3 was run, R3 FAIL.

Any other combination is **FAIL**. The PASS record is written by the harness, never by hand, and binds:
- the launch commit and the argv sha256;
- the class-weight vector sha256;
- the frame-set md5 `4eafa03c…`;
- the trunk-init fingerprint;
- the per-class IoU (both rules, all three bands), the per-class n, and the R/C arms' values.

A token is valid only for that exact tuple.

## 9a. Harness conformance: required changes to `map_hires_overfit.py` (routed to the Master Mind)

These are the gaps between the builder's harness as read at 22:50 and this protocol. The spec file
`raw/gmo_spec.json` is REFUSED by the current `load_spec` until items 1–2 exist. That refusal is
deliberate.

1. **Add arm `s8_zeros`:** `s8 = torch.zeros_like(s8)` before the branch. The trunk params stay in the optimiser.
2. **`must_fail_all`:** an arm listed there fails only if EVERY named class fails its bar, which R2
   requires. The existing ≥ 1-class rule stays for `lane_w0`.
3. **Presence floor:** a class with fewer than `thresholds.min_cells` (1,000) scored cells in the
   gated band makes the verdict `INCONCLUSIVE` ⇒ FAIL. It is never "absent, not gated". The set was
   selected to contain all 8.
4. **Decision rules:** the IoU is computed under the raw argmax AND `argmax(z − log w)`, and the
   verdict uses `spec.thresholds.decision_rule`. This is currently raw only: `per_class_signal`
   (`map_head_hires.py:506`) and `map_hires_metrics.py:93-97`.
5. **Controls C1–C3** run inside the gate run, on the same scored cells.
6. **Time guard:** each label's `t_img_us` is checked against the v2ep frame timestamp (1 ms), using
   `ClipMapGT.check_times`.
7. **Criterion §6.4:** the per-class CE over each class's own cells is recorded at step 0 and at the
   final step.
8. **Weight decay 0 and batch 4** come from the spec. They are the file's `spec.get` defaults, so they
   need no code change, but the record must show the values that ran.

## 9. What a FAIL means (so nobody argues after the fact)

- **MAIN fails on a thin class while R2 fails and C1–C3 pass:** the head cannot express that class at
  10 cm from these features. The design, not the gate, changes. Levers, in order:
  1. the decision rule, if it failed only under the raw rule;
  2. the lift (Z = 0 road plane, 0.1 m lift near range, stride 4);
  3. the decoder;
  4. the weights.
- **R1 or R2 passes:** the harness does not measure what it claims. The gate is broken, not the model.
