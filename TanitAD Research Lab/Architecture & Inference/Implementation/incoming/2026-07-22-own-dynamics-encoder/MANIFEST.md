# Deliverable manifest — OUR rig-robust dynamics-estimation encoder (2026-07-22)

**Stream:** design + build-prepare our own rig-robust dynamics-estimation encoder (Sayed directive).
**Operating rules:** STAGE, NEVER PUSH. Everything below is `git add`-ed into the working tree; nothing
committed, nothing pushed, no branch switch. `pytest -q` in `stack/` green (778 passed, 2 skipped).

## Escalation (integration) — read first

0. **⚑ GO/NO-GO LANDED 2026-07-23 — FAIL ⇒ Branch B.** The pre-registered camera-conditioning ablation
   ran on pod3 (MEASURED, `RESULTS_camcond.md`). Conditioning ON vs OFF (warm-start v1, capacity-matched):
   cross-rig speed R² **−2.34→−2.25** (rig) / **−2.18→−2.06** (multirig) — a **consistent but marginal**
   lift (+0.09 / +0.12), both failing the 0.9 gate. Mechanism **not refuted** (ON>OFF every time), but the
   cheap warm-start suffix-conditioning shortcut (**Branch A**) is → **Branch B** (from-scratch, all-block
   conditioning, multi-rig) is the go, Plücker/PRoPE escalation on the table. GPU lock released.
1. **The recipe is now fixed by measurement.** The landed multi-rig verdict (`results_multirig.json`)
   **REFUTED data-diversity** (held-out rig-B light-FT speed R² −1.61 vs −1.65 single-domain). The
   collapse is REPRESENTATIONAL ⇒ the **camera-conditioned encoder is the recipe, not a fallback**, and
   the "cheap multi-domain-cotrain suffices" branch is retired. `IDM_VIDEO_PRETRAIN_DESIGN.md` §3's
   "multi-domain co-train, not frozen" should be updated to "**multi-domain co-train + EXPLICIT GAIA-2
   camera conditioning**, not frozen and not diversity-alone".
2. **The decisive next experiment is pre-registered and cheap** (`PRE_REGISTRATION.md`): the GAIA-2
   camera-conditioning **ablation** (ON vs OFF, held-out rig, ~hours on pod3). It should run **before** any
   multi-GPU-day launch. Both outcomes committed.
3. **New reusable code lives in `stack/`** (guarded by `pytest`), not stranded in this incoming dir —
   single source of truth, no drift. The incoming dir holds the design + plans + smoke evidence.

## Artifacts

| artifact | where it lives | only copy? | notes |
|---|---|---|---|
| Design doc (GAIA-2 conditioning centerpiece, cited) | `repo: …/incoming/2026-07-22-own-dynamics-encoder/DESIGN.md` | yes → staged | the synthesis of the WAM research (whose own synthesis died on a session limit) + our MEASURED evidence |
| Pre-registered launch plan (config + compute + both branches) | `repo: …/2026-07-22-own-dynamics-encoder/LAUNCH_PLAN.md` | yes → staged | Branch A (warm-start+cond, ~1.5–2.5 GPU-days) / Branch B (from-scratch video-SSL, ~4–8 GPU-days) |
| Pre-registration — camera-conditioning ablation (go/no-go) | `repo: …/2026-07-22-own-dynamics-encoder/PRE_REGISTRATION.md` | yes → staged | both outcomes committed; the cheapest cut beyond the multi-rig cotrain |
| This manifest | `repo: …/2026-07-22-own-dynamics-encoder/MANIFEST.md` | yes → staged | |
| Smoke report (MEASURED pipeline proof) | `repo: …/2026-07-22-own-dynamics-encoder/smoke_report.json` | yes → staged | grad-norm 338, 4 domains mixed, cam-cond live 2.7e-2, IDM loss 2.76→0.98, deployable 97.4M |
| **Ablation RESULT note** (go/no-go verdict) | `repo: …/2026-07-22-own-dynamics-encoder/RESULTS_camcond.md` | yes → staged | FAIL → Branch B; ON−OFF Δ +0.09/+0.12; full interpretation |
| **Ablation raw JSON** (rig + multirig) | `repo: …/2026-07-22-own-dynamics-encoder/results_camcond_rig.json`, `results_camcond_multirig.json` | yes → staged | MEASURED; pulled from pod3; ckpt md5-verified |
| **Ablation runner** (pod-side) | `pod3:/workspace/TanitAD/stack/scripts/run_camcond_ablation.py` + `repo:` (scratchpad copy) | ⚠️ pod + scratchpad only | GAIA-2 suffix conditioning on the re-gate light-FT infra; ON/OFF arms. SHOULD be staged into `stack/scripts/` if the line continues — see below |
| **Model module** — GAIA-2 camera-conditioned encoder + combined objective | `repo: stack/tanitad/models/dynamics_encoder.py` | yes → staged | `CameraConditionedEncoder`, `CameraEncoding`, `MaskedLatentPredictor`, `DynamicsEncoderModel`, `DynEncConfig` |
| **Trainer** — multi-domain dataloader + geometry-randomisation + `--smoke` | `repo: stack/scripts/train_dynamics_encoder.py` | yes → staged | `MultiDomainWindowDataset`, `geom_augment`, `build_domains_from_caches`, `maybe_warm_start` |
| **CPU smoke test** (pytest-guarded) | `repo: stack/tests/test_dynamics_encoder.py` | yes → staged | 5 tests: pipeline finite/differentiable/fits/mixes-domains; zero-init cond == identity; launch sub-300M + state_dim 2048; geom-aug consistency; dataset balancing |

## Reused (unchanged) assets it composes

`stack/tanitad/models/encoder.py` (ViTEncoder) · `readout.py` (SpatialGridReadout) · `predictor.py`
(OperativePredictor) · `sigreg.py` (LeJEPA SIGReg) · `metric_dynamics.py` (MetricInverseDynamics) ·
`stack/scripts/idm_head.py` (IDMHead). Data: `tanitad/data/{physicalai,comma2k19,l2d}.py`.

## MEASURED inputs consumed (unchanged, already in tree)

`…/incoming/2026-07-22-idm-proof/results.json` · `results_regate.json` · **`results_multirig.json`** (the
landed verdict) · `MODEL_REGISTRY.md §1.2/§2`. Research: `tasks/wgmi9zg09.output` (11 verified claims;
session-scratch, not repo — its content is synthesized into `DESIGN.md` §1/§7 with per-claim citations).

## How to reproduce the smoke

```
cd stack && PYTHONPATH=$PWD python scripts/train_dynamics_encoder.py --smoke
# or the pytest guard:
cd stack && python -m pytest -q tests/test_dynamics_encoder.py
```

## NOT done (by design — needs the ablation verdict + Sayed's go)

No multi-GPU-day training was launched (out of scope this task). Branch A/B launch is gated on the §1
camera-conditioning ablation (`PRE_REGISTRATION.md`) and Sayed's go (`LAUNCH_PLAN.md` §5). pod3 was
available for a GPU smoke but the CPU smoke + the launch-config instantiation test already validate the
pipeline and the sub-300M envelope, so no pod GPU load was added.
