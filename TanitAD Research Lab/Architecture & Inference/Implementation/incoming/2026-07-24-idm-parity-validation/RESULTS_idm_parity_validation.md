# RESULT — YouTube-IDM pretraining benefit, ON-TARGET PARITY validation (decision-grade GO)

**Landed 2026-07-24** on pod3 (A40), `gpu_lock idm-parity-validation` (released). Larger-scale,
on-the-actual-target confirmation of the downstream-benefit GO
(`../2026-07-24-idm-downstream-ablation/`): does pseudo-label pretraining help a WM on the **PARITY
corpus** the WM is actually fine-tuned on? **Evidence class: MEASURED (ours + artifact:
`results_idm_parity_validation.json`).**

**⚠ PARITY FIREWALL:** SIDE de-risk. Reads parity-domain **episodes** as data with its **own IDM
split**; creates/affects **no WM arm** and does **not re-select** the canonical WM episode
selection. Licensing-free (parity, not YouTube — the YouTube ingest stays Sayed-gated).

---

## VERDICT — GO, DECISION-GRADE (on our actual target). The proxies UNDERSTATED it.

On the parity corpus, at ~6× the proxy scale, with **readable yaw**, pseudo-label pretraining
**captures the full pretraining value** — matching or exceeding real-label pretraining on speed and
trajectory, 71% on yaw — CI-separated from the random-init floor on **every seed and every metric**.
This upgrades the YouTube-IDM GO from proxy-grade to **decision-grade on our fine-tune target.**

---

## The numbers (MEASURED — parity val, both rigs, 4 seeds; pretrain = 300 parity clips / 26,384 windows held out from the labeler; downstream = pai-val, 15 finetune / 65 test)

| metric | FLOOR (random) | PSEUDO | CEILING (real) | **fraction of ceiling** | pseudo>floor all seeds |
|---|---|---|---|---|---|
| **speed R²** | −0.439 ±.224 | **0.751 ±.044** | 0.651 ±.060 | **109%** | ✅ |
| **traj ADE@2s** | 12.61 m | **4.81 m** | 5.34 m | **107%** | ✅ |
| **yaw R²** | 0.551 ±.070 | **0.691 ±.026** | 0.749 ±.023 | **71%** | ✅ |

Per-seed speed (floor / pseudo / ceiling): (−0.75/0.73/0.72) (−0.26/0.69/0.64) (−0.54/0.81/0.69)
(−0.19/0.77/0.56). **Pseudo ≥ ceiling on speed in all 4 seeds.** ceiling>floor all seeds/metrics
(the pretraining signal is real and measurable). Pseudo CI (0.751 ±.044) is nowhere near floor
(−0.439 ±.224) — clean separation.

### Three reads

1. **Pseudo-label pretraining fully captures the pretraining value on parity** — 109% speed, 107%
   trajectory, 71% yaw. The comma/rig-B proxies (~96%) **understated** it: on-target, higher-quality
   parity pseudo-labels make pretraining as good as real labels.
2. **Pseudo ≥ real on speed+traj (all seeds).** `HYPOTHESIS`: the multi-domain labeler emits
   **smoothed, well-generalizing** targets that regularize the pretraining toward the val
   distribution better than the exact pretrain-corpus real labels (which can slightly overfit the
   pretrain split). Reported as a measured fact + a mechanism hypothesis — not a claim that "pseudo
   beats real" in general.
3. **Yaw is the one channel where real labels help more (71%).** Consistent with yaw being the
   harder, rig-sensitive channel; pseudo yaw pretraining still captures a substantial fraction and
   the absolute (0.691) is strong.

## Honest bounds / caveats

- **This confirms the pretraining MECHANISM on-target; it does not erase the cross-class ABSOLUTE
  gap.** Parity pseudo-labels are high quality (the labeler knows parity). On truly-novel rigs
  (YouTube), pseudo-quality is lower (~0.63, `../2026-07-24-idm-pipeline-derisk`) and the absolute
  ceiling is still bounded by v1's cross-class transfer. So: **pretraining value = fully de-risked;
  absolute novel-rig quality = still capped by v1's representation** (the Branch A/B unsolved gap).
- **Still a small proxy** — the "WM" is a dynamics readout on frozen v1 latents (not the full
  action-conditioned WM predictor); small downstream data. The **full YouTube-scale pretrain remains
  the Sayed-gated commitment**; this de-risks the direction on the real target, cheaply.
- **C5:** all fits converged (pseudo/ceiling absolute R² 0.65–0.75 = healthy); the floor's negativity
  on speed is the genuine 15-clip-from-scratch regime, not underfit of the pretrained arms.

## Recommendation for Sayed (unchanged direction, now decision-grade)

**GO for the Sayed-gated YouTube-IDM scale-up.** The pretraining-benefit mechanism is now confirmed
**on our actual parity target** at scale, with yaw. Design constraints from the evidence:
weak per-clip speed prior; speed + longitudinal trajectory as primary pseudo-signal (they capture
≥100% of ceiling); accept that **yaw pseudo-pretraining captures ~71%** (real yaw labels add value —
keep real yaw where available); drop accel; validate downstream on parity (done). The **cross-class
absolute-quality gap on novel rigs** (v1's representation, unsolved by Branch A/B) is the one residual
to design around at YouTube scale — it caps novel-rig absolute quality but does **not** block the
pretraining value, now proven on-target.

## Provenance / reproduce

```
# pod3 (A40), venv python, PYTHONPATH=/workspace/TanitAD/stack:.../scripts
python scripts/run_idm_parity_validation.py --seeds 4 --pt-epochs 25 --ft-epochs 60
```
Runner: `stack/scripts/run_idm_parity_validation.py` (staged). Labeler = v1 + IDMHead{rigA[:60]+
rigB[:60]+comma[:40]}; pretrain D = rigA[60:200]+rigB[60:220] (parity, held out from the labeler);
downstream = physicalai-val (both rigs). Reuses cached v1 latents; encoded the extra parity chunk
(200 clips) into `pod3:/workspace/tmp/idm_parity/lat`. Raw: `results_idm_parity_validation.json`.
