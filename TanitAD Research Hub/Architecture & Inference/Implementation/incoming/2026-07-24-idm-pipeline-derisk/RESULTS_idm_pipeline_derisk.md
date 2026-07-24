# RESULT — YouTube-scale IDM pretraining, DE-RISK (pipeline + held-out pseudo-label quality)

**Landed 2026-07-24** on pod3 (A40), `gpu_lock idm-pipeline-derisk` (released). Sequel to
`../2026-07-24-branchb-transfer-eval/v1-encoder-char/` (v1 + multi-domain IDM head = the cheap
substrate). **DE-RISK ONLY — NOT the YouTube scale-up (that stays Sayed-gated).** **Evidence
class: MEASURED (ours + artifact: `results_idm_pipeline_derisk.json`, `sample_pseudolabels.json`;
comma-yaw control from `../2026-07-22-own-dynamics-encoder/results_camcond_multirig.json`).**
Converged head (epochs=50; in-domain speed R² 0.862 = the C5 validity check).

---

## READINESS VERDICT — PARTIAL / CONDITIONAL. Not de-risked at the 0.7 bar; not blocked either.

The pipeline **works end-to-end** and emits structurally-correct pseudo-labels carrying a
**substantial-but-sub-0.7** signal. **Held-out speed R² 0.62–0.66, longitudinal-trajectory R² ~0.60
— below the pre-registered 0.7.** The binding gap is **cross-CLASS (rectilinear / YouTube-like)
transfer, and it is a REPRESENTATION gap, not a scale one** (calibration does not rescue it) — the
exact gap camera-conditioning was meant to close and that **both Branch A and Branch B failed to
close**. → **Lean NO-GO on a large as-is YouTube scale-up; GO on one cheap decisive downstream test.**

---

## 1. The pipeline (wired + validated — the de-risk artifact)

`v1 frozen encoder → multi-domain IDM head → per-frame {speed, yaw_rate, long_accel, traj[H,2]}`.
Reusable `PseudoLabeler.label_clip(frames)` in `stack/scripts/run_idm_pipeline_derisk.py`.
**Wire-it proof (MEASURED, `sample_pseudolabels.json`):** a 300-frame held-out comma clip →
**276 per-frame pseudo-labels**; pseudo-speed tracks GT with a consistent offset (e.g. 30.6 vs
27.6 m/s). End-to-end path confirmed.

## 2. Pseudo-label QUALITY on a HELD-OUT DOMAIN (the YouTube proxy) — MEASURED, zero-shot

L2D absent on pod3 → proxies: **comma** (rectilinear, different vehicle = cross-CLASS, the closest
YouTube analog) and **rig-B val** (episode-disjoint, cross-rig same-class). No target labels used
to fit the labeler. `pearson_r²` = affine-calibrated ceiling (what a weak per-clip speed prior /
speedometer-OCR would recover).

| held-out domain | speed R² (zero-shot) | speed R² (calib ceiling) | yaw R² | traj ADE@2s | traj long/lat | traj-x (2s long) R² |
|---|---|---|---|---|---|---|
| **comma** (cross-CLASS) | **+0.625** | **0.632** ⚠ | +0.000 † | 7.06 m | 7.02 / 0.32 | +0.599 |
| **rig-B val** (cross-rig) | **+0.657** | **0.711** | +0.504 | 5.74 m | 5.70 / 0.32 | +0.608 |
| in-domain ref (rig-A) | +0.862 | 0.874 | +0.924 | 3.60 m | 3.58 / 0.16 | +0.669 |

⚠ **comma calib-ceiling 0.632 ≈ zero-shot 0.625** (calib slope 1.08) → the cross-class error is
**NOT scale/bias**; a weak speed prior would **not** rescue it. Contrast rig-B: 0.657→**0.711**
with calibration (bias −1.65) — same-class error IS partly scale, so a weak prior lifts it past 0.7.
† comma yaw is **unreadable in-domain too** (MEASURED control: comma-co-trained head → comma yaw R²
−0.00003) — a comma-label artifact, **not** a transfer failure; comma cannot test yaw pseudo-labels.

## 3. The specific gaps (pre-registered "name the gap")

1. **Cross-CLASS speed plateaus at R² 0.63 and is NOT scale-recoverable** (calib-ceiling ≈ zero-shot).
   This is the **representation** gap for out-of-fisheye geometry — the intrinsics-variance risk made
   concrete, and the exact target camera-conditioning missed (Branch A −2.1, Branch B FAIL).
2. **Trajectory pseudo-labels are longitudinal-dominated and moderately noisy** — ADE 5.7–7.1 m
   cross-domain (~2× the 3.6 m in-domain), long-R² ~0.60. Even **in-domain** long-R² is only 0.67, so
   the trajectory head has limited precision regardless; the labels are a coarse longitudinal signal.
3. **Lateral / turning dynamics are UNTESTED** — both proxies are highway-straight (lateral ADE 0.32 m).
   YouTube has turns; cross-class lateral + yaw pseudo-labels are **unmeasured**.
4. **Accel pseudo-labels are unusable everywhere** (R² ≈ 0 even in-domain) — long-accel is not
   decodable from this latent; drop it from the pseudo-label set.
5. **Yaw pseudo-labels: OK for fisheye rigs (rig-B 0.50), untested cross-class** (comma disqualified).

## 4. Go/no-go for the Sayed-gated scale-up

- **Lean NO-GO on a large as-is YouTube scale-up.** The cross-class pseudo-labels (R² ~0.63, traj
  ADE ~7 m, no readable yaw) are too noisy for high-quality WM-pretraining *supervision*, and the
  binding gap is the unsolved cross-class representation transfer — not a calibration knob.
- **GO on ONE cheap decisive test first (the honest discriminator).** Label R² is only a *proxy* for
  "do pseudo-labels help WM pretraining." R² ~0.63 is a substantial signal and pretraining tolerates
  noise, so this is **not obviously disqualifying**. The definitive cheap experiment (Sayed-gated,
  small): **pretrain a small WM on pseudo-labeled held-out-domain data, measure downstream benefit vs
  no-pretrain.** That answers "good enough?" directly, where the label-R² proxy cannot.
- **If proceeding regardless:** use pseudo-labels as a **weak longitudinal/speed auxiliary** (not
  primary supervision), add a **per-clip weak speed prior** (recovers same-class rigs to ~0.71; the
  design's speedometer-OCR path), drop accel, and treat cross-class yaw/lateral as **unvalidated**.

## Honest bounds / caveats

- **C5:** converged head (in-domain 0.86 confirms). All quality is zero-shot (the YouTube condition);
  `pearson_r²` is the affine-calibrated ceiling, explicitly labelled.
- **C6:** comma yaw disqualified via the existing in-domain control (not a v1 gap).
- **Proxy limits:** L2D (unknown-intrinsics, the ideal YouTube analog) is absent on pod3; comma is the
  best available cross-class proxy but is highway-straight, so turning/lateral transfer is untested.
- The label-quality bar is a **proxy**; the definitive test is downstream WM-pretraining benefit
  (out of scope here — Sayed-gated).

## Provenance / reproduce

```
# pod3 (A40), venv python, PYTHONPATH=/workspace/TanitAD/stack:.../scripts
python scripts/run_idm_pipeline_derisk.py --epochs 50
```
Runner (+ `PseudoLabeler`): `stack/scripts/run_idm_pipeline_derisk.py` (staged). Reuses the v1 CLEAN
latent cache (`pod3:/workspace/tmp/branchb_eval/lat_flagshipv1`); no re-encode except the wire-it
sample. Raw: `results_idm_pipeline_derisk.json`, `sample_pseudolabels.json`. Leave-one-domain-out:
comma held out (head rigA+rigB), rig-B-val held out (head rigA+comma), rigA in-domain reference.
