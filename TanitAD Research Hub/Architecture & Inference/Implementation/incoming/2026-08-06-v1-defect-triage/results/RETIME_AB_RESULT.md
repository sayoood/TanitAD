# Does the retrain-free re-timing actually improve v1arch? — full-scale A/B

**MEASURED 2026-08-06** · `flagship-v1arch-v2bal-30k` @ step **29999** · **6,834 windows**
(6,794 consecutive pairs) / 40 PhysicalAI OOD-val q90 episodes · idle A40 · evidence class
**MEASURED (ours)** · `results/retime_ab_v1arch_tangent.json.xz`.

Limits are a **rule, not a tuning**: the human's own p99 on this corpus, computed in the same
pass — **accel 2.6890 m/s²**, **jerk 6.3686 m/s³**.

| | **BEFORE** | **AFTER** | human | change |
|---|---|---|---|---|
| ADE @2 s (m) | 0.3585 | **0.3205** | 0 | **−10.6 %** |
| speed bias (m/s) | +0.3794 | **+0.0328** | 0 | **−91 %** |
| speed MAE (m/s) | 0.7172 | **0.4803** | 0 | −33 % |
| along-track bias @2 s (m) | +0.7527 | **+0.0647** | 0 | **−91 %** |
| accel RMS (m/s²) | 3.8166 | **1.1919** | 0.9075 | 4.21× → **1.31×** human |
| accel bias (m/s²) | +0.6767 | +0.3031 | 0 | −55 % |
| **jerk RMS (m/s³)** | **52.1281** | **4.9535** | 1.7051 | 30.6× → **2.90×** human |
| entry transient (m/s²) | 1.9825 | **0.0064** | 0.5487 | **−99.7 %** |
| **curvature MAE (1/m)** | **0.006103** | **0.006922** | 0 | ⚠️ **+13.4 % — WORSE** |

### Temporal, the same A/B

| | BEFORE | AFTER | change |
|---|---|---|---|
| replan shift mean (m) | 0.0947 | 0.0833 | −12 % |
| replan shift max (m) | 1.0722 | 0.7819 | −27 % |
| **replan accel jump mean (m/s²)** | **1.1021** | **0.4336** | **−61 %** |
| replan accel jump p90 (m/s²) | 2.8553 | 0.8172 | −71 % |
| intra-plan jerk RMS (m/s³) | 52.2148 | 4.9538 | −90 % |

⭐ **The frame-to-frame control jump fell 61 %, and I predicted it would not.** The write-up
before this run said re-timing each frame *independently* "may not" reduce the inter-frame jump
"since nothing couples the frames". It does: bounding each frame's accel independently shrinks
the range any frame can occupy, so consecutive frames cannot disagree as violently. **A
prediction committed in advance and falsified by the measurement — recorded as such rather
than quietly dropped.**

---

## ⛔ The regression, and its exact mechanism

**Curvature MAE gets 13.4 % WORSE.** On 39 clips it had *improved* (0.0083 → 0.0078), which is
why the first write-up claimed *"curvature slightly better, not worse"*. **At 175× the sample
that claim is false and is withdrawn.**

MEASURED diagnosis:

| | off-curve distance |
|---|---|
| windows whose schedule **under-runs** the curve (82.1 %) | **4.44 × 10⁻¹⁶ m** — machine zero |
| windows whose schedule **over-runs** it (17.9 %, all low-speed, mean v0 4.77 m/s) | up to **0.2146 m** |

⇒ The geometry is preserved **exactly** wherever the curve extends. The regression comes
entirely from the 17.9 % of windows where the re-timed schedule runs *past* the end of the
arm's own curve — because the ego's true `v0` exceeds what the arm's first waypoint implied —
and the tail was extrapolated along the **final tangent**, i.e. straight, where the curve was
still bending.

**Fix implemented:** `_extend_by_arc` continues past the curve end with the curve's **final
curvature** instead of its final tangent (a straight path is the κ→0 limit of the same branch,
so it is untouched). ⚠️ **Its effect at full scale is NOT YET MEASURED** — 39 clips are not
sensitive enough to resolve a 13 % curvature change, which is precisely how the regression
escaped the first read. The re-run is queued.

## What this is, stated plainly

* It is a **projection applied after a frozen head**, not a fix to the model. `v1arch` still
  *plans* infeasibly; this corrects its output. The unicycle retrain remains the right answer
  and now has a measured baseline instead of a hypothesis.
* **`v0` is not privileged information.** The arm already trains and evaluates with
  `speed_input=True`, so ego speed at the window origin is an **existing input channel**.
  Post-processing with it adds nothing the model was not already given.
* ⚠️ **No CI.** Windows within an episode are strongly dependent; an i.i.d. interval would be
  badly optimistic. The decision-grade form is the episode-cluster bootstrap over the 40
  episodes — **a work item**. It cannot flip the sign on jerk (52.1 → 5.0) or entry (1.98 →
  0.006); it could matter for the 10.6 % ADE move.
* ⚠️ **Limits come from val GT percentiles, not train.** Weak dependence — MEASURED sensitivity
  over a 6.5× range of the accel limit moved ADE only 0.2675–0.2916 on the 39-clip set, all
  better than baseline — but it is a val-derived prior and is stated as one.
* The ADE gain is **smaller at scale** than on 39 clips (−10.6 % vs −19.0 %). The 39-clip
  figure was optimistic; quote this one.
