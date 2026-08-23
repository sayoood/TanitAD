# RESULT — Branch B held-out-rig transfer eval (the own-encoder go/no-go, decisive)

**Landed 2026-07-24** on pod3 (A40), under `gpu_lock branchb-transfer` (released).
Encoder under test: `dynenc-branchB` **step 40000** (from-scratch, GAIA-2 **all-block**
camera-conditioned, multi-rig video-SSL), `pod3:/workspace/experiments/dynenc-branchB/ckpt.pt`
md5 `a0d7e7c19e8105cde04e743f6ed6ee26` (weights identical to the step-40000 save
`14ec0235…`; final save re-inits the optimizer only). Paired frozen control: flagship-v1
(`b5f07d9e…`, step 29999). **Evidence class: MEASURED (ours + artifact:
`results_branchb_transfer_e50_CONVERGED.json`, `…_e10_UNDERFIT.json`).**

Gate (frozen, `PRE_REGISTRATION.md`): **cross speed R²>0.9 AND yaw R²>0.9 AND ADE@2s<1.5×
in-domain**. Pre-registered outcomes: **BEATS the −2.1 ablation** (→ conditioning works,
YouTube-scale thesis validated) · **≈ the ablation** (→ conditioning insufficient; report
the residual).

---

## Verdict — FAIL, decisively. Outcome = "≈ ablation", and stronger: a REGRESSION vs the plain encoder.

Branch B does **not** recover cross-rig ego-motion transfer. Its best cross-rig speed R² is
**−0.667** (gate is +0.9) — failed at every regime and every head-fit level. On the clean,
episode-disjoint held-out set it is **worse** than the plain flagship-v1 encoder it was meant
to improve on. The pre-registered "YouTube-scale IDM-pretraining thesis is VALIDATED" branch is
**not** reached.

---

## The numbers (MEASURED, converged head — epochs=50)

Cross-rig = held-out **rig-B** speed R² (the pre-registered headline). Paired dR2 = Branch B −
flagship-v1 on **identical windows + identical head-fit** (the C6-clean, regime-robust contrast).

| experiment | rig-B set | **Branch B** cross | **flagship-v1** cross | **paired dR2** [95% CI] frac+ | Branch B in-dom (rigA) | PASS |
|---|---|---|---|---|---|---|
| rig_train | train-cache (⚠ leaked) | **−2.662** | −2.948 | +0.286 [−0.27, +0.99] .83 | +0.039 | ❌ |
| multirig_train | train-cache (⚠ leaked) | **−1.703** | **+0.382** | −2.085 [−2.90, −1.52] **.00** | −0.603 | ❌ |
| **rig_val** (clean) | **val-cache (disjoint)** | **−1.923** | −1.169 | −0.755 [−1.34, −0.11] .01 | −2.171† | ❌ |
| **multirig_val** (clean) | **val-cache (disjoint)** | **−0.667** | **+0.657** | −1.325 [−2.30, −0.80] **.00** | −0.589† | ❌ |

†val in-domain also carries a train→val episode-shift (head fit on *train* rig-A); the clean
in-domain ceiling is the **train_train** row (flagship +0.862 / +0.910 — see harness check).

**Branch B's OWN 40k-trained head, in-sample on rig-B** (regime-free — no fresh-head fit):
speed R² **0.156** (train-cache rig-B) / **0.242** (val-cache rig-B). yaw R² ≈ 0. The deployed
model reads rig-B speed at R²≈0.2 *even where the head trained on rig-B*.

### Three findings, ranked by robustness

1. **Cross-rig transfer FAILS the gate by a wide margin, at every regime.** Best Branch B
   cross-rig speed R² = −0.667; gate +0.9. No arm passes; yaw R² is negative on every cross set.

2. **Branch B is a WEAKER dynamics substrate than the plain flagship-v1 encoder — even
   in-domain.** With a *converged* fresh head (same fit that gives flagship-v1 in-domain rig-A
   speed R² **+0.862 / +0.910**), Branch B's own in-domain rig-A reads **+0.039 / −0.603**.
   Corroborated independently by Branch B's own 40k head (in-sample rig-B **0.24**). Two
   independent heads read Branch B's latent weakly → it is the representation, not the head.

3. **Paired, same-regime: Branch B ≤ flagship-v1 cross-rig.** dR2 CI excludes 0 (Branch B worse)
   on multirig_train (−2.085), rig_val (−0.755), multirig_val (−1.325). The *only* arm where
   Branch B edges ahead — rig_train, +0.286 — has a CI spanning 0 **and is the leaked arm**
   (Branch B trained SSL+supervised on those exact rig-B clips; the edge vanishes on disjoint
   clips). Flagship-v1 frozen **does** transfer to rig-B with a converged multi-domain head
   (+0.382 / +0.657); Branch B does not.

---

## Why this is the honest read (not an artifact) — controls + caveats

- **Harness validity (MEASURED).** At the converged head, flagship-v1 frozen **in-domain**
  (train rig-A held-out) speed R² = **+0.862 / +0.910**, reproducing the known frozen-flagship
  quality (registry frozen in-dist ~0.93). The probe works; Branch B's low numbers are its own.
- **Episode-leakage controlled (the reason for the val set).** Branch B trained (SSL **and**
  supervised IDM — `training_step` runs `idm_loss` on rig-B every batch) on **all** rig-B
  *train-cache* clips. So the `*_train` cross sets are **best-case, not zero-shot** — flagged ⚠.
  The `*_val` sets use `physicalai-val-f1b378f295ae` (Branch B trained only on
  `…-train-e438721ae894`), **episode-disjoint**. The apparent rig_train edge (+0.286) is
  leakage: it collapses to a loss on disjoint clips.
- **"Held-out rig" = seen GEOMETRY, disjoint EPISODES.** rig-B's cy≈753 geometry *was* in
  Branch B's multi-rig SSL (GAIA-2's "conditioning ⊗ multi-rig, both required" — by design).
  The val test asks whether that recipe yields rig-invariance. It does not — so the stricter
  "never-seen rig" (YouTube) question is moot; the model fails the easier seen-geometry test.
- **C5 — head-fit convergence is a large lever; single numbers off an under-converged head are
  unreliable.** e10→e50 moved cross-rig R² by 1–3.5 pts (e.g. flagship multirig_train
  −3.014→+0.382; Branch B multirig_val −4.555→−0.667). Both JSONs are staged. The verdict is
  anchored on the **paired same-regime contrast** + the **regime-free own-head number**, NOT on
  the external −1.61/−2.06 point estimates (different regime — light-FT, not frozen+fresh-head).
- **⚠ Residual confound on finding #2 (in-domain weakness):** Branch B trained with
  `geom_augment` (±12 px vertical shift + matched cam) on every window; eval feeds **clean**
  frames. This train/eval mismatch may depress Branch B's clean-frame readout. But (a) the
  deployment target *is* clean heterogeneous video, and (b) the cross-rig head-to-head (finding
  #3) holds regardless. **Cheapest follow-up:** re-encode with matched augmentation to isolate.
  This does not touch findings #1/#3.
- **Side-observation (flag, do not over-claim):** flagship-v1 frozen + a converged multi-domain
  head transfers to rig-B at +0.38/+0.66 — materially better than the prior `-1.61` multirig
  *light-FT* baseline. Suggests the frozen flagship latent is more rig-robust than earlier
  baselines implied, and/or those baselines were head-fit-limited. Does **not** relitigate the
  prior MEASURED artifacts; flagged as a C5 head-fit-sensitivity note for the encoder line.

---

## Decision (pre-registered): the own-encoder / camera-conditioning path is NOT validated

- ❌ **Branch A (warm-start + suffix conditioning)** — already refuted (`RESULTS_camcond.md`, −2.1).
- ❌ **Branch B (from-scratch, all-block conditioning, multi-rig)** — refuted here. Explicit
  GAIA-2 camera conditioning, at 40k steps / 2466 clips, did **not** engineer rig-invariance;
  the from-scratch multi-task SSL latent is not even competitively speed-decodable in-domain.
- **What it implies.** The cross-rig problem is **not** closed by explicit conditioning at this
  scale. Branch B's deficit is upstream of rig-invariance — representation *quality*. Before any
  further scale (Plücker/PRoPE escalation, YouTube pretraining): the cheapest discriminating
  question is no longer "does conditioning help?" but **"can the from-scratch SSL recipe produce
  a latent as speed-decodable as the flagship WM encoder at all?"** — and the paired data already
  says the frozen **flagship-v1** encoder is the stronger cross-rig substrate (+0.66 val), so a
  flagship-warm-started, longer-trained, augmentation-matched variant is the more promising lever
  than more from-scratch conditioning. Pre-register before spending the GPU-days.

---

## Provenance / reproduce

```
# on pod3 (A40), venv python, PYTHONPATH=/workspace/TanitAD/stack:.../scripts
python scripts/run_branchb_transfer.py --epochs 50 \
  --branchb-ckpt /workspace/experiments/dynenc-branchB/ckpt.pt \
  --flagship-ckpt /workspace/tmp/idm/ckpt.pt \
  --out /workspace/tmp/branchb_eval/results_branchb_transfer_e50.json
```
Runner: `stack/scripts/run_branchb_transfer.py` (staged) — reuses the camcond harness downstream
probe (`idm_head.IDMHead` / `build_windows` / `evaluate`), swaps in the Branch B
`CameraConditionedEncoder` fed true per-clip cam params (`train_dynamics_encoder.clip_cam_raw`
convention Branch B trained on), plus the frozen flagship-v1 paired control on identical windows.
Raw: `results_branchb_transfer_e50_CONVERGED.json` (decision-grade), `…_e10_UNDERFIT.json`
(the head-fit-sensitivity lesson). Clips: train rigA 100 / rigB 120 / comma 80; val rigA 26 /
rigB 54 (episode-disjoint). Bootstrap: episode-cluster over rig-B eval clips, 2000×.
