# REVIEW 5 — the ONE→FOUR camera extension of REFe

*Independent review, 2026-09-20 ~23:40–00:35 Europe/Berlin. Reviewer: Architecture & Inference.*
*No code was changed. Every number below carries its evidence class and its artifact.*

Evidence classes used: **MEASURED** (ours, with the path) · **PUBLISHED** (cited) · **ANALYTIC**
(arithmetic from source, no run) · **DERIVED** (arithmetic over our own measurements) ·
**INHERITED** (another agent/doc, re-verified or not — stated) · **UNVERIFIED**.

⚠️ **`refe/model.py` and `refe/build_targets.py` CHANGED UNDER THIS REVIEW at 23:48:22**
(`ego_dim` 8→9). Everything below was re-verified against the post-edit files
(`model.py` md5 `c8633b5a441d034314f350fad1d859e7`, 650 lines; `build_targets.py` md5
`3549d3889511fb8ee6ca87c25747e954`). A new file `refe/diag_guard_audit.py` also appeared
mid-review and is audited in §6.

---

## 1 · Headline

⭐ **The geometry and the camera choice are RIGHT. The integration is NOT — five of the six
consumers of the 4-camera config are dead, including the validator that is supposed to prove it,
and the corpus it would train on does not exist. Nothing can train on four cameras today, and the
first thing that will happen if someone tries is an `AttributeError` in the target builder.**

What is correct, and was attacked hardest:

* **The four cameras are exactly the paper's.** PUBLISHED, p. 10 §3.2: *"the four camera views
  CAM_F0, CAM_B0, CAM_L0, and CAM_R0"*, repeated in Table A12 (p. 30, row *Sensors*). The choice is
  conformant and needs no defence. ⛔ But the **reason given in the code is false** — see D12.
* **The frustum is geometrically exact.** An independent NumPy reconstruction of eight patch rays
  across all four cameras matches `_frustum` to **8.806 × 10⁻⁸** (float32 noise). Quaternion
  convention `w,x,y,z` and the camera→ego direction `R·p + t` both **CONFIRMED** two ways.
* **The token order is right**, tested through the *real* forward pass with a live negative control.

⛔ **What blocks training on four cameras, in the order it will bite:**

| # | blocker | class |
|---|---|---|
| **D1** | `build_targets.py:140` raises `AttributeError: 'dict' object has no attribute 'size'` on **every log where all four cameras are present** — i.e. the success path. The refusal path works. **Zero 4-camera tuples can be built.** | MEASURED |
| **D2** | **0 of 3** non-front channels exist on disk. 31,500 JPEGs banked, **100 % `CAM_F0`**. Even with D1 fixed, coverage is **0 %**. | MEASURED |
| **D3** | `validate_model.py` feeds a single-camera tensor into the 4-camera model and **dies at section 4**. Sections 4–6 — including **both** constructed-regression arms this change is credited with adding, and the peak-memory assertion — **have never executed** under `n_cameras=4`. | MEASURED |
| **D4** | The 23:48 `ego_dim` 8→9 edit broke the **trainer** (all 4,146 banked rows are 8-D) and the **planner** (still builds 8 elements). The package's own headline control `train.py --overfit --synthetic` now dies with `mat1 and mat2 shapes cannot be multiplied (2x44 and 45x256)`. | MEASURED |

### ⛔ ESCALATION — the A40 request must be re-priced before it is submitted

**The compute basis in `MODULE_SIZING_STUDY.md` §5.1 is a ONE-CAMERA number now carrying a
FOUR-CAMERA model, and the cross-check that made it quotable was a coincidence.**

`MODULE_SIZING_STUDY.md:712` and `POD_HANDOFF.md:15` quote **0.780 s/sample** and derive
**608.5 A40-h**, then present agreement with the paper's own **608 GPU-hours** (16 × H20 × 38 h,
Table A12) as *"a cross-check that makes the basis quotable … a number the basis was not fitted
to."* That agreement compared **our one-camera cost** against **their four-camera cost**.

MEASURED tonight (`scratchpad/cost2.txt`): one camera, batch 2, warm = **0.803 s/sample** — the
0.780 figure reproduces to +2.9 %, so the *value* is sound even though REVIEW_4 correctly found its
*citation* empty. Four cameras is **×4.00 the work** (ANALYTIC, §4). ⇒ the basis becomes
**≈3.21 s/sample**, and the full reproduction **≈2,506 A40-h ≈ 104 days on one card**, not 608 h /
25 days. **The 608 ≈ 608.5 agreement breaks by 4.1× and must be retracted, not re-derived.**
This is a PI provisioning decision, not an implementation detail.

*(Same class as DE-C152: when you correct the artifact, re-run the cost; never port the old timing
onto the new format.)*

---

## 2 · Confirmation table — each change, verified from source

| # | claimed change | verdict | evidence |
|---|---|---|---|
| 1 | `n_cameras` 1→4, `cameras = (F0, L0, R0, B0)` | ✅ **CONFIRMED, and conformant** | `model.py:79-80`. Paper p. 10 §3.2 + Table A12 name exactly these four (PUBLISHED). ⚠️ order differs from the paper's listing (`F0,B0,L0,R0` vs ours `F0,L0,R0,B0`) — harmless, since `build_targets.CAMERAS` uses the *same* order and the model never assumes one |
| 2 | per-camera extrinsics from the decoded calibration | ✅ **CONFIRMED** | `model.py:120-121` vs `data/nuplan_cam_calib.json`. All four `t`/`q` match byte-for-byte; every `q` has ‖q‖ = 1.0000, det(R) = +1.000000, orthogonality error ≤ 8.9e-16 (`scratchpad/cam_geom.py`) |
| 3 | quaternion convention is **w,x,y,z** | ✅ **CONFIRMED twice** | (a) PUBLISHED: `nuplan-devkit/nuplan/database/common/templates.py:~84` — *"Coordinate system orientation as quaternion: w, x, y, z"*. (b) MEASURED: under w,x,y,z, CAM_F0's optical axis maps to ego **(1.000, −0.001, −0.020)** and CAM_B0's to **(−1.000, 0.000, 0.003)** — forward and backward exactly. Any other reading scrambles this |
| 4 | camera→ego lift `R @ p + t` (not the inverse) | ✅ **CONFIRMED** | `model.py:517` `cam_pts @ R.T + t` ≡ `R·p + t` for row vectors. MEASURED centre-patch far points: F0 **(+61.66, −0.71, 0.44)**, B0 **(−60.50, +0.65, 1.76)**, L0 **(+36.39, +49.06, 0.69)**, R0 **(+35.50, −49.70, 1.76)** m in ego. The inverse transform would put all four in the same place |
| 5 | frustum follows the **input's** camera count, not the config's | ✅ **CONFIRMED** | `model.py:493`, `_pos3d` at `:527`. Cache key includes `n_cam` (`:494`) |
| 6 | intrinsics rescaled 1920×1080 → 960×512 | ✅ **CONFIRMED, and the anisotropy is handled right** | `sx = 0.500000`, `sy = 0.474074` applied to fx/fy and cx/cy **independently** (`model.py:498-500`) — correct for `cv2.resize(im,(960,512))`, which squashes the 1.778 native aspect to 1.875. MEASURED: real JPEGs are **1920×1080** (three files, SOF marker), so `cam_native_h = 1080` and `cy = 560 ≠ h/2` are both right |
| 7 | "all eight cameras share intrinsics" | ✅ **CONFIRMED** | MEASURED over 300 randomly sampled log DBs (`scratchpad/dbcal2.py`): **295/300** logs give fx=fy=1545.0, cx=960.0, cy=560.0 on **all eight** channels; the minority rig (5/300) does vary per channel. Using CAM_F0's K for all four is right for **98.3 %** of the corpus |
| 8 | forward folds the camera axis into the batch (`B*N`), trunk SHARED | ✅ **CONFIRMED** | `model.py:560-562`. MEASURED: 1→4 cameras adds **exactly +49,152** trainable = 48 × 1024, all in `registers` (16,384 → 65,536). Parameter count does **not** grow with camera count anywhere else |
| 9 | **token order matches between features and position encoding** | ✅ **CONFIRMED — the attack found nothing** | MEASURED through the **real** `REFe.forward` with a tagging trunk and a pre-hook on `reg_compress` (`scratchpad/order_check.py`): for B=2, N=4, every block `n*P:(n+1)*P` recovers exactly its own camera's tag, within-block std **0.00e+00**. Cross-check: `_frustum` block 0 far-x = **+61.66 m**, block 3 = **−60.50 m**. **Negative control is live** — reversing the input camera order reddens **4 of 4** blocks |
| 10 | camera-count mismatch RAISES unless `allow_camera_mismatch` | ✅ **CONFIRMED, and it fires — on our own tools** | `model.py:562-567`. MEASURED: it is what kills `validate_model.py` (D3) and `diag_architecture.py` (D6). The guard is correct; the callers are not |
| 11 | `build_targets.py` indexes four channels, pairs each independently, keeps the WORST residual, refuses a missing channel | ⚠️ **HALF-CONFIRMED** | Indexing (`:70-88`), independent pairing and `dt_ms = max(dts)` (`:143-155`), and the missing-channel refusal (`:129-133`) are all present and the **refusal path runs clean**. ⛔ **But the success path raises** — D1 |
| 12 | `train.py` loads a LIST of paths per tuple and stacks them | ⚠️ **CODE CORRECT, PATH DEAD** | `train.py:297-308` is right, and the synthetic arm builds `[N,3,H,W]` correctly (`:289`). ⛔ But `FrameStore.read` hardcodes `{log}_CAM_F0.zip` (`:66`) — D8 — and no 4-camera row exists to load |
| 13 | `validate_model.py` gained two constructed-regression arms | ⛔ **PRESENT BUT UNREACHABLE** | `validate_model.py:138-163`. Both arms are downstream of the crash at `:92`. **Neither has ever run under the shipped config** — D3 |

---

## 3 · New defects, ranked

### ⛔ D1 · BLOCKER · `build_targets.py` crashes on its own success path

`build_targets.py:127` builds `cam_ts` as a **dict**; `:140` then does `if cam_ts.size == 0:` and
`:198` does `int(cam_ts.size)`. Both are `numpy` idioms applied to a dict.

```
File ".../refe/build_targets.py", line 140, in build_one
    if cam_ts.size == 0:
AttributeError: 'dict' object has no attribute 'size'
```

MEASURED (`scratchpad/bt_repro.py`, stubbing only nuPlan and `camera_index`, re-run after the
23:48 edit): the **happy path raises on the first step**; the **refusal path (a channel missing)
returns cleanly**, `{'kept': 0, …, 'missing_channels': ['CAM_B0']}`.

⭐ **Why this is the worst shape of bug in this package's own taxonomy.** The branch that says *"I
refuse"* works; the branch that does the work is dead. Any smoke test run against a log with an
incomplete rig **passes**. This is the *check that shares the defect it checks for*, inverted.

**Minimal fix:** `cam_ts[CAMERAS[0]].size` at `:140`, and `sum(v.size for v in cam_ts.values())`
at `:198`. The `:140` branch is in fact dead anyway — `:129-133` already returned for every empty
channel — so deleting `:140-142` is equally correct and one line shorter.

### ⛔ D2 · BLOCKER · three of four channels have no images (§5 quantifies)

### ⛔ D3 · BLOCKER · the validator cannot reach any runtime arm

`validate_model.py:33` builds `cfg = REFeConfig()` (⇒ `n_cameras = 4`) and `:86` builds
`img = torch.randn(B, 3, cfg.img_h, cfg.img_w)` — **4-D, one camera**.

MEASURED (`scratchpad/validate_run.txt`, exit **1**): sections 1–3 pass, then

```
File ".../refe/model.py", line 562, in forward
ValueError: REFe is configured for 4 cameras (CAM_F0, CAM_L0, CAM_R0, CAM_B0) but received 1.
```

⇒ shape checks, the WTA routing check, the detach check, its deliberate-regression arm, **both new
constructed regressions**, and the peak-memory assertion all never run. The change's own headline
deliverable ("gained two constructed-regression arms") is **unexercised**.

**Minimal fix:** `img = torch.randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w)`. ⚠️ At ViT-L, B=2,
four cameras that arm needs **~19 GB** and will not run on this box — so the fix must also drop to
`B=1` locally or carry an explicit "A40 only" skip that says so.

### ⛔ D4 · BLOCKER · `ego_dim` 8→9 landed without its consumers (a **second** instance of the same failure this review is about)

Introduced at 23:48:22 (`git diff` vs index): `model.py:136` `ego_dim: int = 9`, and
`build_targets.py:183-187` now appends `vp.length, vp.width`.

* ✅ The nuPlan API is valid — MEASURED: `VehicleParameters` exposes `length = 5.176`,
  `width = 2.297`. *(My first probe used `hasattr` on the **class** and read `False`; that is the
  class-vs-instance anti-pattern and I am correcting my own probe, not reporting a false defect.)*
* ⛔ **All 4,146 banked target rows carry 8-D ego** (MEASURED: `refe_targets/…rank0` 1,091,
  `rank1` 1,091, `refe_targets_real/…` 982 + 982, **`{8: …}` in every file**). `train.py:330` neither
  slices nor pads `r["ego"]` — unlike `goal`, which *is* sliced at `:331`.
* ⛔ **`planner.py:177-181` still builds 8 elements**, ending in a literal `0.0`.

MEASURED — the package's own control, run tonight:
`train.py --overfit --synthetic --backbone vits16 --batch 2 --steps 1 --overfit-n 4 --cpu` →
exit 1, `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x44 and 45x256)`
(8 + 36 goal features vs 9 + 36). Note it got **past** the camera guard — the synthetic arm is
correctly 4-camera — so `ego_dim` alone kills it.

⚠️ **The two new slots carry almost no information.** nuPlan's ego is a Pacifica throughout, so
`length`/`width` are effectively **constants** across the corpus — this replaced one hardcoded
`0.0` with two hardcoded-by-the-vehicle numbers, at the cost of invalidating the whole banked
corpus and the planner. Worth a sentence of justification it does not currently have.

**Minimal fix:** pad-or-refuse at the loader (`train.py:330`) with a loud message naming the bank,
add the two fields to `planner.py:177`, and rebuild the bank. Or revert to 8 until the bank is
rebuilt anyway for four cameras.

⭐ **The blast radius is currently CONTAINED, and whoever owns this should know before they commit.**
MEASURED at 00:10: `refe/model.py` and `refe/build_targets.py` are **new files staged but never
committed** (`git cat-file -e HEAD:<path>` fails for both), and the **index still holds the
pre-23:48 content** — index blob `70ee8ef` carries `ego_dim: int = 8` at line 127 while the
worktree blob `a6da734` carries 9. ⇒ the breaking edit exists **only in the worktree**. Two
consequences: a pathspec-free `git commit` right now would land the *old* file and silently revert
it (the stale-index hazard in `CLAUDE.md`), and conversely there is a free window to fix D4's three
consumers **before** the `ego_dim` change is ever staged. *(I staged neither file; verified.)*

### ⛔ D5 · HIGH · `planner.py` is single-camera end to end

`planner.py:68` queries `WHERE channel='CAM_F0'`; `:259-263` `cv2.imread` one path and returns
`[1,3,H,W]`; `:184` calls the model. ⇒ the **NAVSIM-facing arm raises the camera guard on step 1**.
Its `max_consec_holds` refusal machinery — the part carefully built to fail loud — is downstream of
an exception it will never reach.

### ⛔ D6 · HIGH · `diag_architecture.py` is dead

MEASURED, exit 1, same `ValueError` at `model.py:562`. `:38` builds `torch.randn(B, 3, …)` while
`:65` already computes `exp_scene = cfg.n_cameras * cfg.n_registers` — half-migrated.

### ⚠️ D7 · HIGH · the frustum unprojects an **ideal pinhole** through **distorted** images

PUBLISHED (`nuplan-devkit/.../templates.py:93`): the camera table's `distortion` is *"the Caltech
model (k1, k2, p1, p2, k3)"*. MEASURED from a real DB: **[−0.3561, 0.1725, −0.0021, 0.0005,
−0.0523]**, identical on all eight channels. MEASURED by grep over the whole devkit:
**`undistort` appears nowhere** — the JPEGs on disk are *not* rectified. `_frustum` uses
`(u − cx)/fx` with no distortion term.

MEASURED consequences (`scratchpad/dist2.py`):

| image point | assumed ray angle | true ray angle | error | lateral error at 60 m |
|---|---|---|---|---|
| horizontal edge | 31.86° | 36.12° | **+4.27°** | **+6.51 m** |
| corner | 35.73° | 42.12° | **+6.39°** | **+11.09 m** |
| quarter-width | 17.26° | 17.85° | +0.59° | +0.68 m |

⭐ **Two things follow, and the second is the one that matters.** (a) The true HFOV is **72.25°**,
not the 63.71° the intrinsics alone give. (b) **The lift into the ego frame stops being metric
across cameras** — its whole purpose. MEASURED: a world point at ego **(25, 12, 0)** (bearing
25.6°, inside both F0 and L0) is reconstructed at **(25.00, 10.99, 0.09)** from F0 and
**(24.08, 12.64, 0.10)** from L0 — **1.02 m and 1.13 m wrong, and ~1.7 m apart from each other.**

⚠️ **Do not over-read this.** The map patch→3D-code is deterministic and injective per camera, and
`pos3d_mlp` is *learned on top of it*, so a smooth warp is largely absorbable **within** a camera.
The part that is not absorbable is the cross-camera disagreement, which is precisely what four
cameras were added to exploit. **UNVERIFIED** how much this costs in PDMS; the discriminating
experiment is a distortion-corrected `_frustum` as an ablation arm.

**Minimal fix:** invert the Caltech polynomial once per (gh, gw) inside `_frustum` (it is cached,
so the cost is one-off), or undistort the images at load. The first is ~8 lines and changes no
interface.

### ⚠️ D8 · MEDIUM · three `CAM_F0` hardcodes survive in the frame-resolution path

* `train.py:66` — `FrameStore.read` looks only in `{log_name}_CAM_F0.zip`, and on `KeyError`
  **`return None` at `:74` without falling through** to the loose-file path at `:75`. ⇒ if a
  `_CAM_F0.zip` container exists at the root, **every L0/R0/B0 frame fails even when its loose file
  is present**.
* `build_targets.py:103` — `index_images` indexes only `*_CAM_F0.zip` containers.
* `planner.py:68` — D5.

This matters operationally because `POD_HANDOFF.md:64-71` makes containers **mandatory** on D:
(exFAT, 1 MiB clusters, 5.0× inflation measured).

### ⚠️ D9 · MEDIUM · the A40 cost basis is invalidated — see §1 escalation and §4

### ⚠️ D10 · MEDIUM · the 4-camera seconds in the cost table are paging, not compute — see §4

### ⚠️ D11 · MEDIUM · `visual_ctx.detach()` is an undeclared departure that an instrument now defends

PUBLISHED, p. 8: *"Candidate trajectories are detached before entering the scoring branch."* The
paper detaches the **trajectories**. `model.py:598` does that (`traj.detach()`) — conformant. But
`model.py:600` **also** detaches the scoring decoder's context (`visual_ctx.detach()`), which the
paper never states. ⇒ the six-component PDM loss contributes **no gradient at all** to the backbone
LoRA or to `scene_proj`.

⛔ **And `validate_model.py:150-152` now asserts that as a required PASS**, with the failure text
*"the scorer leaks into the frozen trunk"* — language that presupposes the extra detach is the
paper's. REVIEW_3 flagged it honestly as *"MATCHES + an undocumented extra"*; it has since been
promoted from *an extra* to *an invariant*, with a constructed regression defending it.

⚠️ **Why it is not cosmetic here.** REFe exists to compare DINOv3 against DriveVFM. Throwing away
the score loss's gradient to the encoder changes how much the encoder is adapted, which is the
variable under study. **Minimal fix:** none required in code — but the flag must be *declared*, and
the validator's message must say "REFe's extra detach, not the paper's", or an ablation must settle
it.

### ⚠️ D12 · LOW · the stated reason for the camera choice is false

`model.py:78`: *"Cameras chosen for 360 coverage from nuPlan's eight."*

MEASURED (`scratchpad/cov2.py`, all C(8,4) = 70 subsets, ray-traced over a 0.05° azimuth grid):

| set | coverage @ 63.71° | coverage @ 72.25° (true) | max gap | rank |
|---|---|---|---|---|
| **ours F0/L0/R0/B0** | 66.01 % | **70.76 %** | **52.7°** | **22 of 70** |
| best incl. a forward cam — F0/L1/R1/B0 | 70.78 % | 78.10 % | 39.8° | 2 |
| all eight | 100 % | 100 % | — | — |

Ours leaves **two 53° blind sectors at ±91°…±144°** — the rear quarters, i.e. the lane-change
blind spots. The ranking is **22/70 at both FOV readings**, so it is not an artifact of D7.

⭐ **This is NOT a defect in the reproduction.** The paper names these four (p. 10, Table A12) and
**never claims 360° coverage** — the subagent's sweep found zero occurrences of "field of view",
"FOV", "surround", or any coverage claim in all 32 pages. The choice is right *because it is the
paper's*. The code should say that, not invent a geometric justification that its own calibration
refutes. *(And F0/L0/R0 do give a contiguous 174°/191° forward arc with no holes, which is a real
property worth stating — just not "360".)*

### ⚠️ D13 · LOW · a load-bearing comment quotes a superseded number

`model.py:~446`: *"`reg_compress` is 12,598,272 trainable at 1024"*. MEASURED: it is **6,303,744**
— the figure predates the `reg_mlp_ratio = 1` ruling recorded 40 lines above it. The sentence that
follows (*"it is why REFe's trainable count exceeds the paper's 18.58 M"*) is still true but by
half the stated margin.

### ⚠️ D14 · LOW · `raw/refe_vitl_devbox_memory.txt` is **0 bytes**

A truncated artifact that reads like a complete one, in a package whose memory numbers have now
been wrong twice.

### ⚠️ D15 · LOW · the guard's error message asks for something the CLI cannot do

`model.py:566` tells the operator to *"set `REFeConfig.n_cameras=N`"*, but `train.py` has **no
`--n-cameras`** flag (MEASURED: full `add_argument` list, `:514-546`). A legitimate single-camera
ablation therefore requires editing source.

### ⚠️ D16 · LOW · `build_targets.py:156` `rel = rels[0]` is dead

### ⚠️ D17 · LOW · the analytic `pos3d` is still locked to **one rig**

`model.py:120-121` bakes a single vehicle's extrinsics into the config. MEASURED across 300 logs:
**22 distinct extrinsics per channel**. The change's own justification (`model.py:417-420`) is that
a learned table *"CANNOT condition on the camera geometry that varies across our own corpus"* — but
a hardcoded config tuple cannot either. The fix for resolution-locking landed; the fix for
rig-locking did not. To realise the stated rationale, the extrinsics must arrive **per sample** from
the tuple. **Practically:** with 22 rigs whose poses differ by centimetres this is small — but the
*claim* should be scoped to "resolution-free", not "a function of the rig".

---

## 4 · The cost table — corrected

### 4.1 The parameters: confirmed, with one figure now stale

| quantity | brief's figure | MEASURED tonight | verdict |
|---|---|---|---|
| total | 322,071,110 | **322,071,366** | ⚠️ **+256 stale** — measured before the 23:48 `ego_dim` 8→9 edit (one extra input × 256 `ego_enc` units) |
| trainable | 18,991,682 | **18,991,938** | same +256 |
| trainable % | 5.90 % | **5.90 %** | ✅ |
| 1→4 cameras trainable delta | +49,152 = 48 × 1024 | **+49,152** (18,942,786 → 18,991,938) | ✅ **exact** |
| paper | 338.46 M / 18.58 M (5.49 %) | PUBLISHED, Table A12 p. 30 | ✅ ⚠️ that table is titled **DriveZero-*Scale***; say so when comparing |

### 4.2 The memory: confirmed. The seconds: refuted.

MEASURED tonight, warm, best of 3, synchronised (`scratchpad/cost2.txt` · `cost2.py`):

| cams | batch | images | peak GB | resident on this card? | sec | sec/image |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | **3.55** | YES | **0.764** | 0.764 |
| 1 | 2 | 2 | **5.84** | YES | **1.606** | 0.803 |

⇒ **marginal 2.29 GB per image; fixed 1.25 GB** (weights + AdamW + grads).
The brief's 1-camera memory rows (3.55 / 5.82 GB) **reproduce exactly**. Its 1-camera *seconds*
(1.72 / 1.83) do **not** — `diag_cameras.py` has **no warm-up and no pre-timer synchronise**, so its
first row carries CUDA's lazy init (1.72 vs 0.764 = **2.25× inflated**).

**The four-camera rows are host-memory-fallback thrash, not compute.** Three independent facts:

1. **Capacity.** MEASURED: the card is **8.59 GB total, 7.44 GB free**. A peak of **10.09 GB** (and
   18.89 GB) is arithmetically impossible resident. DERIVED from the marginals above, a resident
   4-image step needs **10.42 GB** — so **no 4-image configuration of any kind is resident on this
   box**, which is why I could not re-measure the 4-camera seconds here and stopped trying.
2. **Host RAM.** MEASURED (`Get-Process`, PID 69032, my own 4-camera probe): working set
   **10,078,068,736 B = 10.08 GB** of *system* memory, matching the reported GPU peak almost
   exactly. The bytes were in host RAM.
3. **Arithmetic.** ANALYTIC (`scratchpad/flops.py`): 1→4 cameras is **×3.999 forward FLOPs**. The
   frozen trunk is **99.19 %** of the 4-camera total and scales **exactly ×4**; the head grows ×3.92
   but is **0.81 %** of the work.

| | ratio 1→4 cameras at batch 1 |
|---|---|
| tokens | ×4.00 |
| **compute (ANALYTIC)** | **×4.00** |
| time, brief's table | ×15.25 (26.18 / 1.72) |
| time, against a **warm** denominator | **×34.3** (26.18 / 0.764) |

⇒ **the excess is the machine, not the model.** *(`POD_HANDOFF.md:120` already records "Batch 4 on
an 8 GB card silently spills to host RAM" — the package knows the mechanism; the table was
published anyway.)*

⚠️ **And the table is not banked.** No artifact in `raw/` contains `CAMERA_COST_MEASURED`, `10.09`,
`18.89`, `26.18` or `7680`. By this programme's own rule a number carries its artifact path or it is
not quotable. `diag_cameras.py` must write to `raw/` before its output is cited again.

### 4.3 Guidance for an A40 (48 GB)

DERIVED from the marginals above; **all of it must be re-measured in the first pod hour.**

* A 4-camera **sample** is 4 images ⇒ **9.16 GB/sample**, plus **1.25 GB** fixed.
* At ~45 GB usable: **(45 − 1.25) / 9.16 = 4.78 ⇒ batch 4 is safe; batch 5 (47.05 GB) is not.**
  The brief's "about batch 5" is optimistic by one.
* **To reach the paper's global 256 you need 64 accumulation steps from batch 4, not 43.**
* Two levers before accepting batch 4, neither implemented: **(a) bf16 autocast** — the paper trains
  FP32 (Table A12), so this is a declared departure, but it roughly halves activations ⇒ batch ~8–9;
  **(b) gradient checkpointing on the trunk blocks** — since only LoRA A/B are trainable, backward
  needs little more than each block's normed input, so this is the high-leverage cut and it costs
  ~30 % compute.
* **Wall clock: ≈2,506 A40-h ≈ 104 days on one card** for the full 337 K × 25 reproduction
  (DERIVED: 3.21 s/sample × 337,000 × 25 ÷ 3600, with the **UNVERIFIED** ~3× A40/4060 factor that
  `MODULE_SIZING_STUDY.md:881` already flags). ⇒ §1 escalation.

### 4.4 Batch vs accumulation — what accumulation does and does not restore

PUBLISHED, Table A12 p. 30: **"GPUs 16 × H20"**, **"Batch size 256"**. ⇒ their 256 is a **global,
data-parallel** batch of 16/GPU. **Gradient accumulation is never mentioned in the paper** (zero
occurrences across 32 pages) and **is not implemented in `train.py`** (MEASURED: no `accum` symbol;
`:462-465` does `zero_grad / backward / clip / step` every micro-batch).

**Restored exactly by accumulation** — because there is **no BatchNorm anywhere** (all `LayerNorm`)
and no stochastic layer:
* the **WTA winner assignment**, which is computed **per sample** (`model.py:614-615`,
  `pos.argmin(dim=1)`) and is therefore batch-size-independent by construction;
* the **union of proposals updated per optimiser step** — 43–64 micro-batches at one weight point
  touch the same winners a true batch would;
* the AdamW moments and the mean trajectory gradient.

⛔ **NOT restored, and each needs naming before anyone claims equivalence:**
1. **The scorer's normaliser is per-batch.** `train.py:434`:
   `l_score = (per*cand_m).sum() / cand_m.sum().clamp(min=1.0) * score_w`. A true batch of 256
   divides by the mask count over **all 256**; naive accumulation divides each micro-batch by **its
   own** count and then averages — re-weighting samples by how many covered neighbours they
   happened to share. Fix: accumulate numerator and denominator separately.
2. **`clip_grad_norm_` must move.** At `:464` it clips every micro-batch. Under accumulation it must
   clip **once, after** the last micro-batch, or the global-norm-1.0 semantics of Table A12 are not
   reproduced.
3. **The uncovered-batch branch changes meaning.** `:448-459`: at ~25 % frame coverage,
   P(no covered sample) is **0.75⁴ = 31.6 %** at batch 4 versus ≈0 at batch 256. Accumulation fixes
   this at the *step* level but only once (1) is fixed too.
4. **`--epochs` arithmetic.** `:382` `per_epoch = ceil(len(ds)/batch)` counts optimiser steps at the
   *micro*-batch size; with accumulation it over-counts by the accumulation factor and the cosine
   `T_max` spans the wrong horizon.
5. **Wall clock.** 64× the forward/backward per optimiser step — real, and already in §4.3.

⭐ **And the thing that is true today, which is worse than any of the above:** `train.py` does **not
accumulate at all**, so at batch 4 it performs **64 separate optimiser steps** where the paper
performs **one**. That is not "approximately batch 256"; it is a different optimiser trajectory, and
the package's own `refe_wta_batch_starvation.txt` already measures the symptom (1/2 distinct
winners at batch 2).

---

## 5 · The data-path gap, quantified

**MEASURED, `D:/Projects/TanitAD/data/nuplan-camera/`:**

| | |
|---|---|
| JPEGs on disk | **31,500** (26,060 under `scenarios/`, 5,440 under `probe/`) |
| distinct `CAM_*` directories | **1** — `CAM_F0` |
| frames for CAM_L0 / CAM_R0 / CAM_B0 | **0 / 0 / 0** |
| banked training tuples | **4,146** (`refe_targets` 1,091+1,091; `refe_targets_real` 982+982) |
| of those with a 4-camera `image` list | **0** — every row's `image` is a **`str`**, and no row has the `cameras` key `build_targets.py:186` now writes |
| of those with 9-D ego | **0** — every row is 8-D (D4) |

**Fraction of tuples that can be built today: 0 %, and it is 0 % for three independent reasons.**

1. `build_one` **raises** before writing a row (D1) — so the number is 0 even in principle.
2. With D1 fixed and `--images` given: `:162-165` requires **all four** basenames to resolve in
   `_IMAGE_INDEX`; three never will ⇒ `missing_file += 1` for **every** step ⇒ **0 kept**, reported
   honestly as `missing-file N` in the per-log line. *(The builder does **not** silently drop —
   credit where due: the counter is printed.)*
3. With `--images` omitted, rows *are* written with the DB's relative paths for all four channels —
   and then `train.py`'s `FrameStore` fails to resolve three of four and raises `FileNotFoundError`
   at `__getitem__`. **Loud, but still 0 % trainable.**

**What the four-camera corpus costs.** MEASURED by the package itself
(`code/fetch_front_camera.py:4-7`): CAM_F0 is **12.2 %** of the 48.63 GB mini archive; front-only
across mini+test+val is **223 GB of data**. Four channels of equal size ⇒ **≈890 GB of data**, and
`POD_HANDOFF.md:64-71` measured **5.0× exFAT inflation** for loose files on D: ⇒ **containers are
not optional**. ⚠️ `fetch_front_camera.py` filters on the literal channel `CAM_F0`; a four-channel
fetch is a parameter change plus a re-index, and its byte-run merging (one request per contiguous
run) should be re-measured for the other three channels rather than assumed to be 7 runs each.

⇒ **The 4-camera decision implies a ~4× data pull and a container-format change. That is a PI
provisioning item, not an implementation detail** — and it is the same item as the §1 escalation.

---

## 6 · Instruments audit — can each arm go red?

⭐ **`refe/diag_guard_audit.py` (which appeared mid-review) is the right instrument and it works.**
MEASURED tonight, exit 0, `GUARDS_CAN_FAIL`:

| arm | mutation | clean | mutated | verdict |
|---|---|---|---|---|
| rope | identity rotation substituted | True | False | **CATCHES** |
| normalisation | mean→0, std→1 | True | False | **CATCHES** |
| camera guard | `allow_camera_mismatch = True` | True | False | **CATCHES** |
| scorer detach | `detach_scorer=False` via the real forward | True | False | **CATCHES** |
| pos3d | every camera given CAM_F0's pose | True | False | **CATCHES** |
| LoRA | every adapter `scale = 0.0` | True | False | **CATCHES** |

**6 of 6 go red under their own defect.** The `pos3d` arm in particular is the right shape — it
collapses the extrinsics and requires the encoding to notice, which is exactly the property the
4-camera change is justified by.

⛔ **But the audit answers the wrong question, and so does every other instrument in the package.**

| instrument | 4-camera aware? | status tonight |
|---|---|---|
| `diag_guard_audit.py` | ✅ yes (`img = randn(B, cfg.n_cameras, …)`) | **GREEN, 6/6 live** |
| `validate_model.py` | ⛔ no (4-D tensor) | **DEAD at §4** (D3) |
| `diag_architecture.py` | ⛔ no | **DEAD** (D6) |
| `planner.py` | ⛔ no | **DEAD** (D5) |
| `train.py` (synthetic) | ✅ images yes | **DEAD on `ego_dim`** (D4) |
| `train.py` (real) | ⛔ bank is 1-camera, 8-D | **DEAD twice** |
| `build_targets.py` | ✅ intent | **CRASHES on success** (D1) |
| `diag_cameras.py` | ✅ yes | runs, but **publishes paging as compute** and **banks nothing** (D10, §4.2) |

⭐ **THE STRUCTURAL FINDING, and it is why this change shipped with five dead consumers.**
**Every instrument builds its inputs *from the config it is testing*** — `diag_guard_audit.py:52-54`
does `randn(B, cfg.n_cameras, …)` and `randn(B, cfg.ego_dim)`; `validate_model.py:86-88` and
`diag_architecture.py:38-40` do the same. An expectation written as an expression over the code
under test **cannot detect a config/consumer divergence**: change `n_cameras` or `ego_dim` and every
arm silently follows. That is the anti-pattern this programme already has four logged instances of,
and it is the exact mechanism by which `ego_dim` 8→9 broke the trainer and the planner **twelve
minutes before this review** while every guard stayed green.

⇒ **The missing instrument is a CONSUMER CONFORMANCE arm**, and it is cheap. For each real
consumer — the banked bank, the planner's ego vector, the builder's emitted row — assert its shape
against the config as a **literal-bearing** check that reads the *consumer's* output, not a tensor
the test just fabricated. Concretely, three assertions that would have caught D1, D4, D5 and D13:

1. `len(json.loads(first_row)["image"]) == cfg.n_cameras` **and** `len(row["ego"]) == cfg.ego_dim`,
   read from the bank the trainer will actually open;
2. one call to `build_one` against a **stubbed all-four-cameras** log (my
   `scratchpad/bt_repro.py` is ~45 lines and is the whole fix) — the refusal path being green is
   not evidence the build path is;
3. a `param_report` regression pinning `reg_compress` to the **literal** 6,303,744.

⚠️ And one existing arm should change its message rather than its logic: `validate_model.py:150-152`
must stop calling REFe's extra `visual_ctx.detach()` the paper's requirement (D11).

---

## 7 · Manifest

**Deliverable (staged, not committed, not pushed):**

| artifact | location |
|---|---|
| this review | `D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/REVIEW_5_CAMERAS.md` |

**Reproduction scripts and raw outputs — BANKED IN THE REPO**, staged alongside this review at
`…/2026-09-20-refe-plan/raw/2026-09-21-review5-cameras/` (16 files). They were authored in the
session scratchpad and copied in, because a review whose evidence paths point at a scratchpad that
is about to be deleted fails this programme's own "a number carries its artifact path" rule.
⭐ Items 2, 3 and 7 are **reusable guards** and belong in `refe/` once someone owns them —
`bt_repro.py` in particular is the ~45-line consumer-conformance arm §6 asks for.

| # | file | what it establishes |
|---|---|---|
| 1 | `cam_geom.py` | the eight cameras' yaw/pitch, quaternion norms, det(R), orthogonality |
| 2 | `frustum_check.py` | independent patch-ray reconstruction, max error **8.806e-08** |
| 3 | `order_check.py` | camera token order through the real forward + a live negative control |
| 4 | `cov.py`, `cov2.py` | all 70 four-camera subsets ranked at both FOV readings |
| 5 | `dist.py`, `dist2.py` | Caltech-distortion ray error, true HFOV 72.25°, cross-camera disagreement |
| 6 | `dbcal2.py` | per-channel intrinsics/extrinsics over 300 log DBs |
| 7 | `bt_repro.py` | the `cam_ts.size` crash, both paths |
| 8 | `flops.py`, `cost2.py`, `cost2.txt` | ×3.999 analytic scaling; the resident memory/time measurements |
| 9 | `validate_run.txt`, `diag_arch.txt`, `train_control.txt`, `guard_audit.txt` | the four instrument runs |

**Files read but NOT modified** (per the brief): all of `refe/*.py`, the four prior reviews,
`MODULE_SIZING_STUDY.md`, `POD_HANDOFF.md`, `data/nuplan_cam_calib.json`,
`nuplan-devkit/nuplan/database/common/templates.py`, and the nuPlan log DBs (read-only URI).

**Primary sources quoted:** `drivezero_report.pdf` pp. 8, 10, 30 (extracted page-by-page this
session); `nuplan-devkit` `templates.py`; `data/nuplan_cam_calib.json`; the nuPlan log DBs.

**Not verified, and named rather than guessed:**
* the PDMS cost of D7 (distortion) — needs a corrected-frustum ablation;
* the A40/4060 multiplier (~3×) that every GPU-hour in §4.3 rests on — `MODULE_SIZING_STUDY.md:881`
  already flags it as unmeasured, and the four-camera move makes re-measuring it the **first** pod
  job, not a later one;
* the 4-camera seconds — **not measurable on this box at any configuration** (§4.2), by capacity,
  not by failure.
