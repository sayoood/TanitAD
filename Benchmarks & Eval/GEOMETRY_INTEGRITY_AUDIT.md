# Flagship Camera-Geometry Integrity Audit

**Question (Sayed):** Is camera intrinsics/extrinsics **mishandling** corrupting the
TanitAD flagship training stream? Eliminate basic errors; exclude by evidence.

**Verdict — PARTIALLY YES.** One failure mode is **CONFIRMED** with a hard number
from the dataset's own calibration: the PhysicalAI front-wide focal is
canonicalized with the **wrong lens model** (a rectilinear pinhole applied to an
f-theta **fisheye**), so PhysicalAI canonical frames are **1.62× more zoomed** than
comma2k19 / than the shared `f_eff=266` "fingerprint" the pipeline claims. Because
PhysicalAI is **60 % of the flagship mix** and the model has **no per-corpus geometry
conditioning**, identical ego motion maps to ~1.6× different pixel motion across
corpora — the exact "corpus-dependent pixel motion, poison for action-conditioned
dynamics" that D-016 was written to prevent. All *other* candidate failure modes
(temporal desync, action-units, `_ego` handedness, gross camera pitch, per-corpus
representation collapse) are **EXCLUDED** with measured numbers. Extrinsic camera
**height** (1.43 m vs comma ≈1.2 m) is a smaller, secondary unnormalized inconsistency.

---

## Provenance (every number below is measured, not guessed)

| Item | Value |
|---|---|
| Checkpoint audited | `/workspace/ckpt27k_flagship.pt` — flagship `p0-sB01-realmix`, **step 27000** |
| Flagship mix | comma2k19 **0.40** / PhysicalAI-AV **0.60** (`realmix 0.40/0.60`, PROJECT_STATE 2026-07-12) |
| Val caches | comma2k19 90 ep, PhysicalAI 100 ep (train 410/400) |
| Model geometry conditioning | **NONE** — `grep -rniE 'domain\|corpus\|source'` in `tanitad/models/`,`tanitad/train/` = ∅; `WorldModel.encode(frames)` takes frames only |
| New scripts | `stack/scripts/geom_sanity.py`, `stack/scripts/pai_calib_probe.py` (checks 1-3,5); check 4 reuses `driving_diagnostic.py` per corpus |
| Pod artefacts | `/workspace/experiments/{geom_sanity.json, diag_comma_only.json, diag_pai_only.json, pai_calib.json}` |

---

## Verdict table

| # | Failure mode | Verdict | Decisive measurement |
|---|---|---|---|
| 1a | frame contract (shape/dtype/range) | **EXCLUDED** | both corpora **[T,9,256,256] uint8, range 0–255**, identical (D-015 9-ch stack) |
| 1b | pose↔frame temporal desync | **EXCLUDED** | comma vis-motion↔speed peaks at **lag 0** (curve flat 0.130–0.133); PhysicalAI flow field **expands forward** (divergence **+0.0037**, radial **+0.115**/m about center) consistent with poses; `accel↔Δspeed` **1.00** (PAI)/0.80 (comma); `steer↔yaw-rate` **0.72/0.68**; duplicate-frame frac **0.0** |
| 1c | action scale/units | **EXCLUDED** | both **steer[rad] / accel[m·s⁻²]**; the gap is *regime* not units — \|steer\|p99 comma **0.16** vs PAI **0.57** rad; speed p50 **25.1** vs **6.4** m/s (highway vs urban); wheelbase 2.9 vs real **2.85** (1.5 %) |
| 1d | `_ego` convention (handedness) | **EXCLUDED** | forward=+x both (mean ego-x **+22.2 / +6.2**); sign(Δyaw)==sign(ego-y) **0.78 / 0.98**; same CCW-left convention, no flip |
| **2** | **focal canonicalization (intrinsics)** | **CONFIRMED** | real f-theta focal **925.9 px** vs nominal pinhole **554.3 px** (**1.67×**); PhysicalAI canonical `f_eff` **431 px** vs **266** assumed = **1.62× zoom**; crop retains real **33.1°**, not the believed **51.4°** |
| 3 | extrinsic pitch / horizon | **PARTIAL** | pitch **−0.49°**, principal-point offset **+3 px** → horizon ≈ image center (comma empirical FOE **row 120**): **pitch EXCLUDED**. But camera **height 1.43 m** (PAI) vs ≈1.2 m (comma) is an unnormalized extrinsic → secondary scale inconsistency |
| 4 | per-corpus representation ceiling | **EXCLUDED** | PhysicalAI oracle ADE **0.81–1.31** ≤ comma **2.01–3.11** (speed-confounded, *not* worse); no collapse. Minor: I1 oracle-fit R² PAI **0.863** vs comma **0.952** |

---

## Check 5 — the decisive intrinsics evidence (real calibration exists and was never ingested)

The HF dataset **`nvidia/PhysicalAI-Autonomous-Vehicles`** ships a `calibration/`
feature that the pipeline never uses:
`camera_intrinsics`, `sensor_extrinsics`, `vehicle_dimensions` (gated; fetched one
chunk = 100 clips via `pai_calib_probe.py`). What it says about
`camera_front_wide_120fov` (native 1920×1080, mean over 100 clips):

- **Intrinsics are an f-theta polynomial** `r(θ)=925.9·θ − 4.14·θ² − 16.09·θ³ …`
  (px vs radians). This is a **fisheye**, paraxial focal **`fw_poly_1 = 925.9 px`**
  (σ 3.4), real HFOV **121.4°**, principal point (959.4, 543.0) — essentially centered.
- **The pipeline assumes a rectilinear pinhole**: `calib.py` /
  `physicalai._decode_mp4` set `f = W/(2·tan(120°/2)) = 554.3 px`. A 120° *rectilinear*
  lens and a 120° *fisheye* put the same field on the sensor with **different focals**
  → **error ratio 925.9 / 554.3 = 1.67×** (the nominal focal is ~40 % too small).

**Propagated consequence (why it corrupts dynamics):** the pipeline crops a central
`533 px` square (radius 266 px) *believing* it keeps 51.4° of field. Under the **real**
fisheye that radius subtends only **16.5°** half-angle → **33.1° retained**, and the
canonical 256-px frame has a **true `f_eff ≈ 431 px`**, versus the **266 px** it is
labelled with (and versus comma2k19, which really is ~266). So PhysicalAI frames are
**1.62× more zoomed than the model assumes**. The I7 task-identity fingerprint
(`f_eff_px: 266.0`, asserted identical for both corpora in `CORPUS_META`) is **false
for PhysicalAI**. Identical forward Δd (or identical ego yaw) therefore produces ~1.6×
different pixel motion between the two corpora — and with **no domain conditioning in
the model**, the shared dynamics head must average two inconsistent action→pixel maps.

**Extrinsics (same feature):** optical-axis pitch **−0.49°** (≈horizontal, so the
horizon sits at image center — matches comma's empirical FOE row 120, ruling out a
large pitch offset), camera **height 1.43 m**, wheelbase **2.85 m** (we hard-code 2.9 —
0.7–1.5 % off, negligible). Comma's EON sits ≈1.2 m (device mount; not re-measured
here) — the **height difference is a residual unnormalized extrinsic** that scales
ground motion on top of the focal error (D-016 explicitly defers height/pitch
normalization).

---

## Checks 1-4 — how the other modes were excluded

**1a/1b/1c/1d (basic sanity, `geom_sanity.py` on the real caches).** Frame contract
identical (9-ch uint8 256²). **No temporal desync:** comma's visual-motion↔speed
cross-correlation peaks at lag 0; PhysicalAI's optical-flow field *expands* (positive
divergence/radial flow) in lock-step with the pose displacement direction, and
`accel↔Δspeed`/`steer↔yaw-rate` correlations are strong at lag 0 (the weak raw
vis↔speed correlation for PhysicalAI is urban dynamic-object decoupling, not
misalignment — a flow overlay shows forward radial expansion with independent-vehicle
motion in the central-bottom). **Units consistent** (rad, m/s²); the large steering /
low speed of PhysicalAI is the genuine urban-vs-highway regime, not a scale bug.
**`_ego` handedness identical** across corpora (forward=+x, left=+y CCW).

**2/3 empirical ground-flow.** comma gives a clean focus-of-expansion horizon at
**row 120** (≈center), validating the method. PhysicalAI's per-row ground flow is
**contaminated by dense urban traffic** (intersections, oncoming vehicles fill the
central-lower rows), so it does **not** yield a clean per-corpus focal readout on this
val set — which is why the **calibration-file evidence above is the decisive intrinsics
proof**, not the flow. Both corpora move forward (no reversal).

**4 per-corpus oracle ceiling (`driving_diagnostic.py` per corpus).** If the geometry
defect had *destroyed* PhysicalAI's metric groundability, its in-distribution oracle
ceiling would be markedly worse than comma's. It is **not**: PhysicalAI oracle ADE₀₋₂ₛ
**0.81–1.31 m** ≤ comma **2.01–3.11 m** (PhysicalAI even lower, confounded by its lower
urban speeds → smaller metric displacements). Held-out both large (comma 7.5, PAI 4.8)
— the D1 route-generalization gap, corpus-agnostic. Conclusion: the damage is to
**cross-corpus dynamics consistency**, not to a frozen per-corpus readout ceiling.
(Minor caveat: PhysicalAI I1 oracle-fit R² 0.863 < comma 0.952 — the PhysicalAI latent
is marginally less linearly-metric, consistent with but not proof of geometry noise.)

---

## Is the flagship corrupted? — YES, PARTIALLY, and specifically by intrinsics

The flagship stream **is** corrupted, by exactly one mechanism: the PhysicalAI focal
is canonicalized with a pinhole model when the lens is an f-theta fisheye, so 60 % of
training carries a **1.62× geometry scale error** relative to comma2k19, fed
indistinguishably into a single shared dynamics model. This is not a "basic error"
(shapes/units/alignment/handedness are all clean) — it is a **lens-model** error in
`calib.py`. It is a **partial** corruption: it does not desync frames, does not flip
signs, and does not collapse the per-corpus representation; and the extrinsic pitch is
fine (only height is residual).

### Single highest-impact fix
**Ingest the real per-clip f-theta intrinsics** (`calibration/camera_intrinsics`,
`fw_poly`) and re-canonicalize PhysicalAI to comma's **true retained half-angle
±25.65°** — i.e. crop **radius ≈ 412 px** on the native 1920 frame (the current 266 px
is 1.62× too tight), or properly undistort the fisheye to a rectilinear `f_eff = 266`
image. This makes the shared `f_eff=266` fingerprint *true* and equalizes the
action→pixel geometry across corpora. Secondary: apply the D-016 R1 height/pitch
homography (pitch ≈ 0, so mainly a 1.43 m→reference height scale). `pai_calib_probe.py`
already emits the exact `fix_crop_radius_px_to_match_comma = 412`.

### Pre-registered normalize-and-retrain test (causal confirmation — cheap, runnable on pod1 now)
1. **One lever:** rebuild **only** the PhysicalAI cache with the corrected f-theta crop
   (comma-matched 25.65° half-angle). Comma cache unchanged.
2. **Two arms** fine-tuned from `ckpt27k` for **K=4000** steps (rollout_k=4, D-027):
   **A** = current PhysicalAI cache, **B** = corrected cache.
3. **Pre-registered metrics (fixed before running):** (i) PhysicalAI held-out open-loop
   ADE@1s/2s; (ii) comma held-out ADE — **guardrail, must not regress**; (iii) D2
   calibration gate; (iv) cross-corpus gap `|ADE_comma − ADE_physicalai|`.
4. **Pre-registered prediction:** correcting the focal shrinks the PhysicalAI↔comma
   pixel-geometry mismatch → arm B improves PhysicalAI open-loop ADE (and/or shrinks its
   oracle→held-out gap) and narrows the cross-corpus gap, with comma non-regressed.
   **Null** (no change) ⇒ the urban gap is model capability / data regime, not geometry.
5. **Cost:** cache rebuild ≈ decode 500 clips (~40 min, one-time, CPU); 4k-step
   fine-tune on the A6000 ≈ a few hours at micro 32. Fits alongside the current ~14 GB
   B1 finetune (≈34 GB free) or immediately after. **Cheap enough to run now.**

---

## Reproduce

```bash
# checks 1-3 (real caches, no model):
python stack/scripts/geom_sanity.py \
  --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
  --episodes 90 --out /workspace/experiments/geom_sanity.json
# check 5 (real calibration; dev box uses Keys.txt, pod needs HF_TOKEN):
python stack/scripts/pai_calib_probe.py --out /workspace/experiments/pai_calib.json
# check 4 (per-corpus oracle ceiling):
python stack/scripts/driving_diagnostic.py --ckpt /workspace/ckpt27k_flagship.pt \
  --cache-dirs /workspace/data/comma2k19/_epcache  --out .../diag_comma_only.json --episodes 40
python stack/scripts/driving_diagnostic.py --ckpt /workspace/ckpt27k_flagship.pt \
  --cache-dirs /workspace/data/physicalai/_epcache --out .../diag_pai_only.json --episodes 40
```
