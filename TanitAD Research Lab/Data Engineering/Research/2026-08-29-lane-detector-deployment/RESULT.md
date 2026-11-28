# Lane-detector deployment for the reference arm — CLRerNet vs newer SOTA on a 120° cylindrical camera

**Package** `Data Engineering/Research/2026-08-29-lane-detector-deployment` · **Author**
Research Lab agent (daily run 003, 2026-08-29) · literature + geometry only, 0 GPU. Continues
`2026-08-28-corridor-ref-untimed-cot` (lane-detector reference = 4th arm for D-DATA-GTAC-b
cross-agreement). Waiting consumers: **68 turn_suppression records + 41 CoT CORRIDOR_OFFSET
clips** [INHERITED from the register/tasking]. ⚠️ Corpus count discrepancy flagged, not
resolved: tasking says 4,719 clips, the register's CoT rows say 4,729 — pin at run time.

## Findings

**F1 — CULane SOTA has plateaued at ~81.5 F1@50; CLRerNet remains at the frontier, and the
newer entrants buy deployability, not accuracy.** [PUBLISHED, banked] CLRerNet DLA-34 81.12
(paper table) / **81.55 with EMA checkpoint** (repo); Polar R-CNN DLA-34 **81.49** [lib
`2411.01499`, Tab. I] with **20 anchors, NMS-free** (vs CLRerNet's NMS path); GFSR 81.46
[PUBLISHED-SECONDARY, unbanked — not registry-bound]. Nothing in 2024–2026 beats the
CLRerNet-EMA checkpoint by more than noise on CULane.

**F2 — Checkpoints and licenses are clean for the default choice.** [PUBLISHED, repo fetch]
`hirotomusiker/CLRerNet` is **Apache-2.0**, ships `clrernet_culane_dla34.pth` (81.11) and
`clrernet_culane_dla34_ema.pth` (81.55), mmdetection-3.3 environment, Docker recommended,
plus `tools/speed_test.py`. ⚠️ CurveLanes weights (paper: 86.47) are NOT in the fetched
README's checkpoint list — UNVERIFIED, check before relying. The checkpoints are trained on
CULane (academic-research dataset terms) — sanctioned under the binding research-use rule.

**F3 — The projection gap, quantified for OUR rig (derivation from MEASURED rig params:
256×640 cylindrical, f_ref 305.577, az=±60°).** A straight ground-parallel lane line at
lateral offset c maps to **v = (f·h/c)·sin(u/f)** in a cylindrical image — vs the straight
line of a pinhole. Deviation from linear: **2.0 % at ±20°, 7.9 % at ±40°, 17.3 % at ±60°**
(sinθ/θ). The ego corridor (|offset| ≤ ~2 m, Z ≥ 4 m) lives within **~±25° ⇒ ≤3.1 %** —
near-pinhole where the 68+41 records live; the distortion is material only for adjacent-lane
inventory toward the frame edges. Bonus geometric fit: 640×256 upscales **exactly ×1.25 to
CLRerNet's native 800×320 input** (aspect identical).

**F4 — Transfer must be MEASURED, never assumed: published precedent spans mild degradation
to near-zero collapse.** [PUBLISHED, lib `2507.18653`] Cross-dataset shift degrades lane
detectors ("lanes originate from different image regions... violating the model's learned
assumptions"); under extreme semantic shift (road → airport taxiway) *"the F1-score drops to
near-zero."* Their fix — fine-tune selected components in a separate branch, keep the source
branch, route by contrastive distribution ID — is the fallback recipe if our measured
transfer fails. Our two waiting test sets ARE the cheap transfer measurement.

**F5 — Inference cost on the 4060 is hours at worst, minutes for the waiting sets.**
[ESTIMATED — no published FPS for CLRerNet DLA-34 exists (empty search); assumption: 50–150
FPS at 800×320 on the 4060, DLA-34 class; clips ~20 s @ 10 Hz] Full corpus 4,719–4,729 clips
≈ 0.94 M frames ⇒ **1.7–5.2 h**; at 2 Hz keyframes ≈ 189 k frames ⇒ **21–63 min**; the
68+41 waiting records at 2 Hz ⇒ **minutes**. First action of any run: `tools/speed_test.py`
to replace this band with a MEASURED number.

**F6 — In-house rectify assets: two primitives exist, neither is cylindrical→pinhole.**
[MEASURED, repo] `stack/tanitad/data/calib.py` carries `ftheta_undistort` (f-theta fisheye)
and the D016 `pinhole_rectify` (Brown-Conrady pinhole → canonical focal, rectify-to-canvas +
`observed_frac` mask — `…/2026-07-17-d016-r1-pinhole-rectify/INTAKE.md`). A
cylindrical→pinhole `grid_sample` is a small sibling (u=f·θ → tan mapping, ~30 lines by the
D016 pattern) — do NOT cite D016 as already covering cylindrical.

## What this changes for TanitAD (≤3)

1. **Deploy CLRerNet DLA-34-EMA (81.55, Apache-2.0) as-is as the reference lane arm, and run
   the 68 turn_suppression + 41 CORRIDOR_OFFSET sets FIRST** — geometry supports as-is use in
   the ego corridor (≤3.1 % deviation, exact input-aspect match), the run is minutes-scale,
   and it doubles as the transfer measurement F4 demands. Gate any wider rollout on the
   measured F1/agreement from these sets, not on CULane numbers.
2. **Handle the projection gap by scope, then by rectify:** ego-corridor questions as-is;
   before any full-frame lane inventory (adjacent lanes at ±40–60°, 8–17 % deviation), add
   the cylindrical→pinhole primitive to `calib.py` (D016 pattern, with `observed_frac`).
   Never quote a pinhole lane result at frame edges without it — the FOV-formula trap in
   lane-detector costume.
3. **Polar R-CNN (81.49, NMS-free, 20 anchors) is the alternate, not the default** — relevant
   only if the lane arm is ever promoted from offline reference to on-device (TensorRT
   export without NMS plugins). If measured transfer fails, the `2507.18653` branch-fine-tune
   recipe is the pre-identified fallback before any from-scratch training.

## Named empty searches

- **Lane detection on a cylindrical projection directly**: NOT FOUND (fisheye AVM/side-camera
  work exists; no cylindrical-front-camera primary).
- **A released lane-detector checkpoint trained on a ≥120° wide-FOV front camera**: NOT FOUND
  this pass.
- **A verified CLRerNet DLA-34 FPS figure** (paper or repo): NOT FOUND — the repo ships
  `tools/speed_test.py` instead; F5's band is ESTIMATED until measured.

## Banked primaries

NEW: 2411.01499 (Polar R-CNN) · 2507.18653 (lane transfer under shift). CITED-BY UPDATED:
2305.08366 (CLRerNet). Repo facts (license, checkpoints, mmdet version) from the live GitHub
fetch 2026-08-29 — repo state is a moving target; re-verify at deployment.
