# E-DPSI — NO heading shortcut below 12°. A pre-registered NULL.

**MEASURED 2026-07-29, `v4fs_ckpt.pt` step 29,999 (`head` + `goal_head`), `val40cache`,
881 windows × 7 heading offsets, 40 episodes.** Artifacts: `edpsi.py`, `edpsi.json`,
`edpsi_dp*.npz` (pod3 `/workspace/`).
**Pre-registration:** `Project Steering/PREREG_deep_research_2026-07-29.md`.

## The question

PlanT 2.0 (Tübingen; **DS 92.4 ± 1.7** on Bench2Drive, above every sensor model listed) carries a
**positional shortcut**: §5.4, *"at rotation values around 10-15 degrees, the predicted speed abruptly
increases to signal regular driving."* It *"learns a shortcut by using its own rotation as a
predictive signal for when the road is clear."*

⭐ **Our version is strictly cleaner than theirs.** `pseudosim`'s `dyaw` axis is an **exact camera
rotation** — `H = K R K⁻¹`, *"exact for ARBITRARY scene depth (max|dH| = 0.000e+00, 30 conditions)"* —
applied to **bit-identical real footage**. PlanT rotates the ego in simulation, moving world pose
**and** object input together; our warp changes **only the observed heading**. A positive here would
be unambiguous.

## The result — `tspeed_5s` (m/s) vs heading offset

| dψ (deg) | −12 | −8 | −4 | **0** | +4 | +8 | +12 |
|---|---|---|---|---|---|---|---|
| **mean** | 11.500 | 12.143 | 12.779 | **13.633** | 13.376 | 12.460 | 11.920 |
| **median** | 9.813 | 10.125 | 10.500 | **11.000** | 11.500 | 10.563 | 10.250 |
| Δ vs 0 | −2.134 | −1.490 | −0.855 | — | −0.258 | −1.173 | −1.713 |

n = 881 windows at every point (the same windows, re-warped).

## ⭐ Verdict: NULL — smooth and monotone, not a step

The profile **peaks at dψ = 0 and falls smoothly and near-symmetrically in both directions**. There
is **no discontinuity anywhere in the envelope**. Total excursion is 2.13 m/s (15.6 % of the mean)
spread evenly over 12°, i.e. ≈0.18 m/s per degree with no jump.

**The pre-registered signature of the shortcut is a STEP, and there is no step.** Per the
registration this outcome *"strengthens the estimation-problem reading — consistent with our IDM
finding that monocular speed is scale-limited (R² +0.865, smaller-is-better 0.86 M > 2.90 M > 19.98 M)
— and closes the shortcut hypothesis cheaply."*

**The smooth decline is expected and benign.** Rotating the camera changes what is visible: more kerb
and sky, less road ahead. A *graded* reduction in predicted target speed is a sensible scene response.
A shortcut would look like the model reading its own rotation as a categorical "road is clear" cue.

## ⛔ WHAT THIS DOES NOT SAY — registered in advance, restated here

1. ⛔ **This is NOT "we are clean."** Our measurement-grade envelope is **|dψ| ≤ 12°**
   (`pseudosim` falsifies anything beyond; 0 % out-of-envelope is what makes ≤12° quotable). PlanT's
   onset is **10–15°**. **We cover the LOWER EDGE ONLY.** The correct statement is
   **"NO SHORTCUT BELOW 12°"** — the 12–15° band where PlanT's jump actually lives is
   **unmeasured and unmeasurable on this instrument.**
2. ⛔ **This does NOT contradict PlanT 2.0.** Its root cause is CARLA-specific — a scripted expert
   that only turns when the road is free, plus a success-only dataset. **We train on human-driven
   PhysicalAI-AV logs with no such invariant**, so the mechanism may simply be **absent by
   construction**. A null here is consistent with PlanT being entirely right about itself.
3. ⛔ **It does not close the 88.7 % longitudinal gap** — it removes one candidate explanation. The
   gap (oracle speed recovers 0.1899 of 0.2140, entirely longitudinal) still stands and still needs a
   mechanism. This result makes **augmentation** a *less* promising fix and leaves the
   estimation/scale reading as the leading one.
4. ⚠️ **No confidence intervals were computed.** Per-window and per-episode arrays are saved
   (`edpsi_dp*.npz`) so a paired episode-cluster bootstrap over adjacent dψ points can be run if a
   close call ever depends on it. As a **null read from a shape**, the absence of a step is visible
   without an interval; a *positive* would have required one.

## A secondary observation, flagged not claimed

The profile is **mildly asymmetric**: +4° loses only 0.258 m/s while −4° loses 0.855 m/s, and the
median actually *rises* at +4° (11.500 vs 11.000 at zero). **HYPOTHESIS only** — could be a
right-hand-traffic asymmetry in the corpus (turning the camera left points into oncoming lanes;
right points toward the kerb), or could be noise. Not measured, not load-bearing, and **not** to be
cited as a finding without its own test.

## Evidence class

| claim | class |
|---|---|
| the dψ table | **MEASURED (ours)** — 881 windows × 7 offsets, 40 eps, **no intervals** |
| "no step inside ±12°" | **MEASURED**, per the pre-registered decision rule |
| "no shortcut ABOVE 12°" | ⛔ **NOT MEASURED — outside the validated envelope** |
| the smooth decline is a benign scene response | **HYPOTHESIS** — plausible mechanism, not tested |
| the ±4° asymmetry | **HYPOTHESIS** — flagged, not claimed |
