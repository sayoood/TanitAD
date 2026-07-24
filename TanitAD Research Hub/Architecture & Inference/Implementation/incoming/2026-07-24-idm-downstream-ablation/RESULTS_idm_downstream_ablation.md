# RESULT — YouTube-IDM downstream-benefit ablation (the go/no-go the proxy could not decide)

**Landed 2026-07-24** on pod3 (A40), `gpu_lock idm-downstream-ablation` (released). Sayed-greenlit
definitive test: do pseudo-labels ACTUALLY help a (small) world-model, and what **fraction of the
real-label-pretrain benefit** do they capture? — settling what the label-R²~0.63 proxy
(`../2026-07-24-idm-pipeline-derisk/`) could not. **Evidence class: MEASURED (ours + artifact:
`results_idm_downstream_ablation.json`).**

---

## VERDICT — GO. The YouTube-IDM scale-up is JUSTIFIED.

**Pseudo-labels capture ~96% of the real-label-pretraining benefit** for a downstream WM, and beat
the random-init floor on **every seed in both domains** with clean separation. The label-R²~0.63
proxy (which read "lean NO-GO") **understated** the labels' value — because for *pretraining*, labels
need to convey **structure**, not be perfect. The definitive downstream test **overturns the proxy**
(program rule 5: settle with the experiment, not deference to a proxy).

| domain | metric | FLOOR (random) | PSEUDO | CEILING (real) | **fraction of ceiling** | pseudo>floor all seeds |
|---|---|---|---|---|---|---|
| **comma** (cross-CLASS, 5 seeds) | speed R² | −0.771 ±.062 | **0.447 ±.024** | 0.491 ±.012 | **0.965** | ✅ |
| | traj ADE | 17.49 m | 7.72 m | 7.56 m | **0.984** | ✅ |
| **rig-B** (same-class, 3 seeds) | speed R² | −2.235 ±.161 | **0.199 ±.087** | 0.296 ±.018 | **0.960** | ✅ |
| | traj ADE | 17.52 m | 8.82 m | 8.03 m | **0.917** | ✅ |

Pre-registered GO rule (pseudo beats floor all seeds **AND** fraction ≥ 0.5) — **met with a large
margin** (0.92–0.98 ≫ 0.5); pseudo mean ± std is nowhere near the floor (CI-separated) in both domains.

---

## The experiment (cheap, entirely on cached v1 latents)

Classic pretraining-benefit ablation in the **low-real-label regime** (the YouTube-relevant regime:
plentiful pseudo-labeled data + scarce real labels), on proxy domains where we ALSO have real labels.

- **substrate:** v1 frozen encoder. **model:** a small dynamics WM readout (IDMHead temporal trunk
  over v1 latent windows → speed + 2 s ego-trajectory). **pipeline:** v1 + IDMHead{rigA+rigB} →
  pseudo-label the proxy domain (held-out from the labeler) — the validated pipeline.
- **3 arms, PAIRED per seed, identical downstream finetune** on N real clips (comma 10 / rig-B 12):
  **FLOOR** (random init → finetune) · **PSEUDO** (pretrain 50 clips on *pseudo* labels → finetune) ·
  **CEILING** (pretrain 50 clips on *real* labels → finetune). Eval on held-out real-labeled test
  (comma 20 / rig-B 42 clips). Metric: downstream speed R² + traj ADE.
- **Key number:** `fraction_of_ceiling = (PSEUDO − FLOOR)/(CEILING − FLOOR)`, per seed.

**What it means:** with lots of pseudo-labeled data and only a handful of real labels, pseudo-label
pretraining lands **~96% of the way** to what real-label pretraining achieves — and both are
worlds above training on the handful of real labels alone (floor R² is negative: the trunk
underfits 10–12 clips from scratch). So R²~0.63 pseudo-labels are a **near-real-quality pretraining
signal**.

## Why the proxy was wrong (and this isn't)

The de-risk's label-R²~0.63 (< 0.7) read the labels as *marginal supervision*. But pretraining is
**tolerant of label noise** — it needs the labels to carry the right structure, which R²~0.63 does.
The downstream test measures the thing that actually matters (does the WM end up better?), and it
says **yes, ~96% as much as real labels**. Consistency across **2 domains × 2 metrics × 8 seeds**
(fraction 0.92–0.98) is strong evidence this is not a fluke.

## Honest bounds / caveats (do not over-claim)

- **This settles the PROXY, not the full run.** The "WM" here is a small dynamics readout on frozen
  v1 latents (not the full action-conditioned WM predictor), small data, speed+trajectory focused
  (comma yaw unreadable — C6, `../2026-07-22-own-dynamics-encoder`). The **full YouTube-scale WM
  pretrain remains the Sayed-gated commitment** — this de-risks the *direction*, cheaply.
- **The pseudo↔real gap is small but real (~4%).** Pseudo is slightly below ceiling (comma 0.447 vs
  0.491; rig-B 0.199 vs 0.296); pseudo-labels are *near*-real for pretraining, not identical.
- **Absolute downstream quality is still bounded by v1's cross-class transfer** (de-risk: cross-class
  ~0.63, not scale-recoverable). Pseudo-labels pretrain ~as well as real labels, but the **ceiling on
  truly-novel rigs is capped by v1's representation** — the scale-up should still pair with a weak
  per-clip speed prior and expect the unsolved **yaw/lateral cross-class gap**.
- **C5:** all fits converged (in-domain checks upstream); the floor's negativity is the genuine
  low-data-from-scratch regime, not underfit of the pretrained arms.

## Recommendation for Sayed (GO, with the design constraints the evidence implies)

**Proceed with the Sayed-gated YouTube-IDM scale-up.** Pipeline: **v1 frozen encoder + multi-domain
IDM head → pseudo-label YouTube → pretrain the WM → fine-tune on the parity corpus.** Design it to:
(a) use a **weak per-clip speed prior** (speedometer-OCR / speed-distribution — recovers same-class
scale to ~0.71); (b) treat **speed + longitudinal trajectory** as the primary pseudo-signal, **drop
accel** (unreadable), and **caveat yaw/lateral** (cross-class untested); (c) validate downstream on
the parity corpus (the real target). The cross-class *representation* gap (that Branch A/B failed to
close) still caps absolute novel-rig quality — but it does **not** block the pretraining value, which
is what this test proves.

## Provenance / reproduce

```
# pod3 (A40), venv python, PYTHONPATH=/workspace/TanitAD/stack:.../scripts
python scripts/run_idm_downstream_ablation.py --seeds 5 --pt-epochs 40 --ft-epochs 60
```
Runner: `stack/scripts/run_idm_downstream_ablation.py` (staged). Reuses the v1 CLEAN latent cache
(`pod3:/workspace/tmp/branchb_eval/lat_flagshipv1`); no re-encode. Raw:
`results_idm_downstream_ablation.json` (per-seed floor/pseudo/ceiling + benefits + fractions).
comma: pretrain 50 / finetune 10 / test 20; rig-B: pretrain 50 (train) / finetune 12 / test 42 (val).
