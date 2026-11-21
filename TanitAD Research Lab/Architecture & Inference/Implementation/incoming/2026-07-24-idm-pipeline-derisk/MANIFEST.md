# Deliverable manifest — YouTube-scale IDM pretraining DE-RISK (2026-07-24)

**Stream:** de-risk (NOT scale-up) the YouTube-scale IDM-pretraining path — the sequel to the
own-encoder pivot (`../2026-07-24-branchb-transfer-eval/v1-encoder-char/`). Build + validate the
end-to-end pseudo-labeling pipeline and measure pseudo-label quality on a held-out domain.
**Operating rules:** STAGE, NEVER PUSH. `git add`-ed into the working tree; nothing committed/pushed.

## Verdict (one line)

**PARTIAL / CONDITIONAL — not de-risked at the 0.7 bar; not blocked.** The pipeline works end-to-end;
held-out pseudo-labels carry a substantial-but-sub-0.7 signal (speed R² 0.62–0.66, traj-long R² ~0.60,
ADE 5.7–7.1 m). Binding gap: **cross-CLASS (rectilinear/YouTube) speed transfer plateaus at R² 0.63 and
is a REPRESENTATION gap, not scale** (calibration doesn't rescue it) — the exact gap camera-conditioning
(Branch A/B) failed to close. → **Lean NO-GO on a large as-is scale-up; GO on one cheap downstream test.**
Full analysis: `RESULTS_idm_pipeline_derisk.md`.

## Escalation (integration) — read first

0. **The pseudo-label quality is a PROXY.** The definitive cheap discriminator for the Sayed go/no-go is
   a small **downstream WM-pretraining ablation** (pretrain on pseudo-labeled held-out data → measure
   benefit vs no-pretrain). Recommend this as the next gated step, not a blind scale-up.
1. **The cross-class gap is the SAME unsolved problem** the own-encoder line hit: neither Branch A
   (warm-start conditioning, −2.1) nor Branch B (from-scratch, FAIL) closed rig/intrinsics-variance.
   A YouTube scale-up inherits it. Do not spend YouTube-scale GPU-days expecting conditioning to be solved.
2. **New reusable code** (the pipeline + `PseudoLabeler`) is in `stack/scripts/run_idm_pipeline_derisk.py`
   (single source of truth), not stranded in this incoming dir.

## Artifacts

| artifact | where it lives | notes |
|---|---|---|
| Result note (verdict + quality + gaps + go/no-go) | `repo: …/incoming/2026-07-24-idm-pipeline-derisk/RESULTS_idm_pipeline_derisk.md` | the analysis |
| Pipeline + de-risk runner (`PseudoLabeler`) | `repo: stack/scripts/run_idm_pipeline_derisk.py` **+** `…/incoming/…/` copy | v1 enc → multi-domain head → per-frame pseudo-labels; leave-one-domain-out quality |
| Raw quality JSON | `repo: …/incoming/…/results_idm_pipeline_derisk.json` | per-channel R²/pearson-r²/MAE/bias/calib + traj long/lat; also pod3:/workspace/tmp/idm_derisk |
| Wire-it proof (sample pseudo-labels) | `repo: …/incoming/…/sample_pseudolabels.json` | 300-frame comma clip → 276 per-frame labels; pseudo vs GT speed |
| This manifest | `repo: …/incoming/…/MANIFEST.md` | |

## Pods (durable, not the only copy)

- v1 CLEAN latent cache reused: `pod3:/workspace/tmp/branchb_eval/lat_flagshipv1`.
- de-risk outputs: `pod3:/workspace/tmp/idm_derisk/`. Log: `pod3:/tmp/idm_derisk.log`.
- flagship-v1 ckpt (the substrate): `pod3:/workspace/tmp/idm/ckpt.pt` (md5 b5f07d9e…).

## Scope / NOT done

- **NOT the YouTube scale-up** (Sayed-gated, by design). No YouTube data touched; no WM pretraining run.
- **L2D absent on pod3** → the ideal unknown-intrinsics YouTube proxy was unavailable; comma is the
  best available cross-class proxy but is highway-straight (turning/lateral transfer untested).
- pod2 (from-scratch coupling test) / pod1 (better-planner) / eval — untouched throughout.
