# Deliverable manifest — YouTube-IDM downstream-benefit ablation (2026-07-24)

**Stream:** the Sayed-greenlit definitive test — do pseudo-labels actually help a WM, and what
fraction of the real-label-pretrain benefit do they capture? Settles the go/no-go the label-R²~0.63
proxy could not. **STAGE, NEVER PUSH.**

## Verdict (one line)

**GO — the YouTube-IDM scale-up is JUSTIFIED.** Pseudo-labels capture **~96% of the real-label-
pretraining benefit** (speed fraction: comma 0.965, rig-B 0.96; traj fraction 0.98 / 0.92),
CI-separated from the random-init floor on all 8 seeds across both domains. The label-R²~0.63 proxy
UNDERSTATED the labels — pretraining tolerates label noise. Full analysis: `RESULTS_idm_downstream_ablation.md`.

## Escalation (integration)

0. **The de-risk's "lean NO-GO" is OVERTURNED** by this definitive downstream test — the proxy was
   too conservative. Program rule 5 (settle with the experiment, not deference to a proxy). Update the
   IDM/YouTube line status to **GO (Sayed-gated scale-up)**.
1. **Design constraints the evidence implies for the scale-up** (in RESULTS §Recommendation): weak
   per-clip speed prior; speed+longitudinal-traj primary (drop accel, caveat yaw/lateral); validate
   downstream on the parity corpus. The cross-class *representation* gap (Branch A/B failed to close)
   caps absolute novel-rig quality but does NOT block pretraining value.
2. **New reusable code:** `stack/scripts/run_idm_downstream_ablation.py` (single source of truth).

## Artifacts

| artifact | where | notes |
|---|---|---|
| Result note (GO verdict + fractions + bounds + recommendation) | `repo: …/incoming/2026-07-24-idm-downstream-ablation/RESULTS_idm_downstream_ablation.md` | the analysis |
| Ablation runner | `repo: stack/scripts/run_idm_downstream_ablation.py` **+** `…/incoming/…/` copy | 3-arm paired pretraining-benefit ablation on cached v1 latents |
| Raw JSON (per-seed floor/pseudo/ceiling + benefits + fractions) | `repo: …/incoming/…/results_idm_downstream_ablation.json` | also pod3:/workspace/tmp/idm_ablation |
| This manifest | `repo: …/incoming/…/MANIFEST.md` | |

## Pods / scope

- Reused v1 CLEAN latent cache `pod3:/workspace/tmp/branchb_eval/lat_flagshipv1` (no re-encode).
  Outputs `pod3:/workspace/tmp/idm_ablation/`; log `pod3:/tmp/idm_ablation.log`.
- **NOT the YouTube scale-up** (Sayed-gated). Small proxy (dynamics readout on frozen v1, small data,
  speed+traj). pod2 (from-scratch) / pod1 (better-planner) / eval (RefcCL) — untouched.
