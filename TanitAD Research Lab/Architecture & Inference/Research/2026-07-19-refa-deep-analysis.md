# REF-A dyn-in — empirical deep analysis (full-val, overfitting curve, failure modes)

*2026-07-19. Eval pod `tanitad-eval` (A40), non-destructive, pod3 read-only (not
contacted — intermediate ckpts were already staged locally from the prior run).
Companion to the DONE literature comparison
(`2026-07-19-refa-dinowm-literature-comparison.md`); this is the EMPIRICAL side.
Arm under test: **refa-dynin** = frozen DINOv2-B/14 + dyn-in (ego `[v0, yr0]`,
action_dim 4) + 4-brain, ego_dropout 0.25. Reference: **flagship-30k (v1)** =
trained ViT-12, action_dim 3. Protocol = the gate rollout (window 8, stride 8,
K=20 @ 10 Hz, true-action rollout, SE(2) accumulate); metrics = 8-split
episode-level held-out CV mean±CI95 unless noted.*

## TL;DR

1. **Full-val holds.** On the FULL held-out physicalai val (40 episodes / 881
   windows — this IS the full set; there is no larger val on the pod), REF-A
   dyn-in 30k ADE@2s = **2.92 m** (heldout) / 3.05 m (full-set) vs flagship-30k
   **0.452 m** / 0.427 m. A/B: Δ +2.62 m [2.45, 2.80], flagship wins **95.9 %**
   of windows, significant. **REF-A also loses to constant-velocity** (CV ADE@2s
   0.825 m; `beats_cv = false` at every checkpoint). The 2.92-vs-0.452 headline
   is confirmed at full scale.
2. **No temporal overfitting — the FINAL ckpt is best.** ADE@2s falls
   monotonically 5k → 15k → 20k → 30k (3.76 → 3.69 → 3.02 → **2.92**). Unlike
   refa-ijepa (7k beat 15k), **no earlier dyn-in ckpt beats 30k**; it is still
   descending at 30k. Overfitting is real but in the *generalization-gap* sense
   (train fwd_ade ≈ 0.65 → held-out 2.92, a ~4.5× gap), not the
   earlier-ckpt-wins sense.
3. **The failure is ~94 % LONGITUDINAL (speed), not lateral (path).** At 2 s,
   long-RMSE 6.21 m vs lat-RMSE 1.54 m; `long_frac_of_sqerr = 0.942`. The
   **speed-decoupled path geometry is nearly fine (0.27 m cross-track** vs
   flagship 0.10 m) — REF-A steers roughly right but **cannot get the speed
   profile**: +1.53 m along-track overshoot, +0.77 m/s speed bias. It
   **over-predicts speed when slow (+3.45 m overshoot, +1.72 m/s at 0–8 m/s) and
   under-predicts at top speed (−1.16 m, −0.58 m/s)** — regression toward a mean
   speed off frozen features, despite receiving v0 as an input channel.

**Verdict:** 2.92 m is a real, full-val, best-of-lineage frozen-encoder result
that does not beat even constant-velocity. The bottleneck is a **longitudinal /
speed-scale** failure, not path geometry. More training keeps helping (no
early-stop win) but approaches a frozen-generic-encoder ceiling well above
flagship. Consistent with the literature verdict: frozen generic DINOv2 +
supervised displacement regression is expected to fail high-speed AV planning.

---

## Task 1 — Full-val REF-A vs v1

The eval pod's held-out val is exactly **40 episodes → 881 windows**. `--episodes
40` already consumes all of them; the "40-ep subset" in the brief *is* the full
val (there is no larger corpus staged). So the numbers below are the full-scale
result, not a subsample.

**ADE / FDE by horizon (heldout mean, m):**

| horizon | REF-A dyn-in 30k | flagship-30k (v1) | CV baseline |
|---|---|---|---|
| ADE@0.5s | 1.268 | 0.076 | 0.129 |
| ADE@1s   | 1.820 | 0.158 | 0.297 |
| ADE@1.5s | 2.365 | 0.288 | 0.530 |
| **ADE@2s** | **2.920 ± 0.394** | **0.452 ± 0.031** | 0.825 |
| FDE@2s   | 4.583 | 0.944 | 1.708 |
| miss@2m  | 0.725 | 0.060 | 0.313 |
| full-set ADE@2s | 3.047 | 0.427 | 0.838 |
| beats CV? | **no** | yes | — |

**A/B (per-window, 881):** ade_a (REF-A) 3.047 vs ade_b (flagship) 0.427;
flagship win-rate **95.9 %**; Δ +2.62 m, CI95 [2.447, 2.798], significant.

**By speed (model ADE@2s | CV ADE@2s):** low 3.21 | 0.93 · med 3.16 | 0.93 ·
high 2.78 | 0.65. **By curvature:** straight 2.91 | 0.44 · gentle 3.41 | 1.36 ·
sharp 3.38 | 2.38. REF-A trails CV in *every* stratum; flagship beats CV in every
stratum. The gap to flagship is largest on straights / low-curvature (flagship
≈ 0.30 m there, near-perfect).

## Task 2 — Overfitting curve (5k / 15k / 20k / 30k)

Intermediate ckpts were already staged locally (`/root/models/refa-dynin-{5k,15k,
20k,30k}/ckpt.pt`, md5-distinct; snap ≡ 5k). Each was cloned from the canonical
`refa-dynin-30k` registry entry (field parity guaranteed), evaluated with the
identical protocol; frozen DINOv2 features are disk-cached so the four runs
reused one feature pass.

| step | ADE@2s heldout | ADE@2s full | FDE@2s | miss@2m | vs CV (0.825) |
|---|---|---|---|---|---|
| 5 000  | 3.755 ± 0.463 | 3.831 | 6.903 | 0.847 | worse |
| 15 000 | 3.694 ± 0.189 | 3.782 | 6.462 | 0.824 | worse |
| 20 000 | 3.016 ± 0.291 | 3.114 | 4.844 | 0.770 | worse |
| **29 999** | **2.920 ± 0.394** | **3.047** | **4.583** | **0.725** | worse — **BEST** |

**Best refa-dynin ckpt = 30k (the final), ADE@2s 2.92 m.** The curve is
monotone; **no earlier checkpoint beats it** — the refa-ijepa early-stop pattern
does not reproduce for the dyn-in arm. It is still improving at 30k (not
plateaued), so a longer run would likely shave a little more, but every ckpt is
still worse than constant-velocity and far from flagship.

**What training actually moves (per-speed ADE@2s across the curve):**

| step | low speed | med speed | high speed | straight | sharp-curve |
|---|---|---|---|---|---|
| 5k  | 2.64 | 2.94 | **5.90** | 3.96 | 3.68 |
| 15k | 2.88 | 3.44 | 5.03 | 3.88 | 3.66 |
| 20k | 3.28 | 3.13 | 2.93 | 3.03 | 3.42 |
| 30k | 3.21 | 3.16 | **2.78** | 2.91 | 3.38 |

Training **fixes high speed (5.90 → 2.78) and straights (3.96 → 2.91) but
*worsens* low speed (2.64 → 3.21)**. The aggregate improves because high-speed
windows carry more absolute error. This is the training-time signature of the
speed-regression-to-the-mean failure below: as it learns to push speed up (fixing
high-speed undershoot) it starts over-shooting at low speed.

## Task 3 — Failure modes (longitudinal vs lateral, overshoot)

Frenet along/cross decomposition (`taniteval.pathspeed`), all 881 windows, 2 s:

| quantity | REF-A dyn-in 30k | flagship-30k |
|---|---|---|
| long-RMSE (along-track) | **6.21 m** | 1.04 m |
| lat-RMSE (cross-track) | 1.54 m | 0.36 m |
| `long_frac_of_sqerr` (share longitudinal) | **0.942** | 0.893 |
| speed bias | **+0.77 m/s** | +0.19 m/s |
| along-track progress bias (**overshoot**) | **+1.53 m** | +0.38 m |
| path-geometry cross-track RMSE (speed-decoupled) | **0.27 m** | 0.10 m |

**The path shape is nearly right; the speed profile is broken.** 94 % of the 2 s
squared error is along-track. Once the speed profile is factored out (fixed-arc
resampling), REF-A's cross-track path error is only 0.27 m — ~0.17 m worse than
flagship, i.e. it steers acceptably. The +1.53 m overshoot ≈ the "+1.6 m
longitudinal overshoot" flagged earlier (long-bias at 2 s = +1.36 m; along-track
progress bias = +1.53 m).

**Overshoot is speed-dependent — the model regresses toward a mean speed:**

| stratum (n) | overshoot @2s | speed bias | long-RMSE | dominant |
|---|---|---|---|---|
| 0–8 m/s (334) | **+3.45 m** | **+1.72 m/s** | 6.79 | longitudinal |
| 8–16 m/s (303) | +0.23 m | +0.11 m/s | 6.51 | longitudinal |
| 16–24 m/s (112) | +0.89 m | +0.45 m/s | 4.88 | longitudinal |
| 24+ m/s (132) | +0.23 m | +0.12 m/s | 4.83 | longitudinal |
| fast top-10 % (89) | **−1.16 m** | **−0.58 m/s** | 5.20 | longitudinal (99 %) |

REF-A **over-predicts speed when slow and under-predicts at the very top** — the
classic frozen-feature memorization signature: it cannot read the true speed
scale from frozen DINOv2 tokens, so it collapses toward the dataset-mean speed
profile. Note it *receives* v0 as an input channel (dyn-in) yet still mis-scales
the 2 s speed integral — the frozen encoder + supervised displacement head does
not propagate v0 into a correct speed profile. Flagship (trained encoder) keeps
speed bias small in every stratum.

**Worst strata.** By ADE@2s the failure is pervasive (2.2–2.8 m everywhere), not
concentrated: med-speed 2.78, slow 2.66, sharp-curve 2.62. By overshoot the worst
is **low speed (+3.45 m)**. On **sharp curves** the lateral component finally
grows (path-geometry 0.88 m, `long_frac` drops to 0.68) — the only regime where
path geometry, not just speed, meaningfully degrades. Compounding ratio
(de@2s / de@0.5s) = 3.62 for REF-A vs 12.6 for flagship: flagship starts far
tighter and grows faster; REF-A is broadly wrong from the first 0.5 s (already
1.27 m at 0.5 s).

---

## Verdict

- REF-A dyn-in 30k is a **real, full-val, best-of-lineage** frozen-encoder result
  at ADE@2s **2.92 m**, that does **not beat constant-velocity (0.825 m)** and is
  ~6.5× worse than flagship v1 (0.452 m). The A/B is significant and holds across
  every speed/curvature stratum.
- **Best checkpoint = the final 30k**; no early-stop win. Overfitting here is the
  ~4.5× train→held-out *generalization gap*, not an intermediate-ckpt optimum.
- The bottleneck is **longitudinal / speed-scale**, not path geometry: steering
  is ~fine (0.27 m decoupled cross-track), but the model regresses to a mean
  speed (over-shoots slow +3.4 m, under-shoots fast −1.2 m) even with v0 fed in.
- This matches the literature verdict: frozen generic DINOv2 + supervised
  displacement regression is expected to fail high-speed AV planning. The v3
  levers that target this specific failure: (i) feature-space rollout + MPC/CEM
  (drop the memorizing regression head), (ii) an own-pretrained-on-driving frozen
  encoder, (iii) explicit speed-profile supervision / longitudinal loss up-weight.
  For the paper, REF-A stands as evidence the **from-scratch encoder is necessary
  for our recipe** — the frozen-generic ceiling is a longitudinal-speed ceiling.

## Provenance / artifacts

- Eval pod `tanitad-eval`, A40, GPU idle before/after. Pod3 **not contacted**
  (intermediate ckpts pre-staged locally, md5-verified distinct).
- Driver: `/root/taniteval/refa_overfit_driver.py` (clones the registry entry per
  ckpt — registry.py unmodified). Extraction: `/root/taniteval/pathspeed_extract.py`.
- Results: `/root/taniteval/results/overfit_curve.json`,
  `overfit_refa-dynin-{5k,15k,20k,30k}.json`,
  `windows_overfit_refa-dynin-*.pt`, `overfit_driver.log`. Pre-existing full-val +
  failure-mode inputs: `refa-dynin-30k.json`, `flagship-30k.json`,
  `ab_refa-dynin-30k_vs_flagship-30k.json`, `pathspeed_{refa-dynin-30k,flagship-30k}.json`.
- Protocol: `taniteval.rollout` (window 8, stride 8, K=20 @10 Hz),
  `taniteval.bench.run` (8-split episode CV, val_frac 0.2), `taniteval.pathspeed`
  (Frenet along/cross + fixed-arc path geometry). Open-loop / weak claim strength.
- Not committed to git (per brief).
