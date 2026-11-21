# Deliverable manifest — YouTube-IDM ON-TARGET parity validation (2026-07-24)

**Stream:** larger-scale, on-the-actual-target (parity) confirmation of the downstream-benefit GO —
does pseudo-label pretraining help a WM on the PARITY corpus the WM is fine-tuned on? **STAGE, NEVER
PUSH.** ⚠ SIDE de-risk: own IDM split of parity-domain data; no WM arm; no canonical re-selection;
licensing-free (parity, not YouTube).

## Verdict (one line)

**GO, DECISION-GRADE.** On parity val (both rigs, readable yaw, 300 pretrain clips / 26k windows,
4 seeds), pseudo-label pretraining captures **109% of the speed ceiling, 107% trajectory, 71% yaw**,
CI-separated from the random-init floor on every seed/metric (pseudo speed 0.751±.044 vs floor
−0.439±.224; absolute pseudo quality speed 0.75 / yaw 0.69). The comma/rig-B proxies (~96%)
UNDERSTATED it. Full analysis: `RESULTS_idm_parity_validation.md`.

## Escalation (integration)

0. **The YouTube-IDM GO is now decision-grade on our actual fine-tune target.** Update the IDM/YouTube
   line status accordingly. The pretraining-benefit **mechanism** is confirmed on-parity, at scale, with yaw.
1. **The one residual is unchanged and orthogonal:** the cross-class **absolute-quality** gap on
   truly-novel rigs (v1's representation, unsolved by Branch A/B) caps novel-rig quality but does NOT
   block the pretraining value. Design the scale-up around it (weak speed prior; keep real yaw where
   available — pseudo yaw captures ~71%; speed+long-traj primary; drop accel).
2. **New reusable code:** `stack/scripts/run_idm_parity_validation.py`.

## Artifacts

| artifact | where | notes |
|---|---|---|
| Result note (decision-grade GO + fractions + bounds) | `repo: …/incoming/2026-07-24-idm-parity-validation/RESULTS_idm_parity_validation.md` | analysis |
| Runner | `repo: stack/scripts/run_idm_parity_validation.py` **+** `…/incoming/…/` copy | 3-arm paired ablation, on-parity, both rigs, +yaw; encodes the extra parity chunk, else cached |
| Raw JSON (per-seed arms + fractions + firewall note) | `repo: …/incoming/…/results_idm_parity_validation.json` | also pod3:/workspace/tmp/idm_parity |
| This manifest | `repo: …/incoming/…/MANIFEST.md` | |

## Pods / scope

- Reused cached v1 latents (`pod3:/workspace/tmp/branchb_eval/lat_flagshipv1`); encoded +200 parity
  clips → `pod3:/workspace/tmp/idm_parity/lat`. Outputs `pod3:/workspace/tmp/idm_parity/`; log
  `pod3:/tmp/idm_parity.log`. **NOT the YouTube scale-up** (Sayed-gated). pod2 / pod1 / eval untouched.
