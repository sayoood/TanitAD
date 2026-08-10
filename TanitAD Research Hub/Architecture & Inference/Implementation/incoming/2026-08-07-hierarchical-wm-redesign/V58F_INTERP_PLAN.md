# v5.8f × WM-interpretability — integrated execution plan (PI directive 2026-08-10)

**The directive:** implement interpretability proposals 1–3 (P8 BEV occupancy readout, P9
probe-gradient saliency, latent-geometry views) IN ADDITION to the remaining v5.8f elements.
This plan binds both into one schedule with owners, gates and pod slots. Status ticks are
appended here as items land.

## Workstreams

| id | what | needs | cost | gate / deliverable |
|---|---|---|---|---|
| **I1a** | obstacle.offline → episode JOIN (clip_id+frame → agent cuboids jsonl) | pod4 HF cache + val-corpus clip ids | 0-GPU script + ~1 h pod4 CPU | join jsonl for the 40 val episodes (n agents/frame stats reported) |
| **I1b** | P8 occupancy readout TRAIN (encoded latents only; harness committed `04e000b`) | I1a + free GPU | 2 GPU-h | decoder IoU on encoded z (reference row) |
| **I1c** | P8 predicted-latent eval + reel | I1b | 1 GPU-h + CPU render | **gates**: IoU(ẑ,k=10) ≥ 0.8×IoU(z); occluded < 2× visible. Reel: camera \| decoded-BEV(ẑ) \| GT-BEV |
| **I2** | P9 saliency overlays (lead-gap / occupancy-cell / curvature probes → input-frame gradients) | I1b harness | 0.5 GPU-h | sanity gates pre-stated in WM_PHYSICS_PROOF (saliency on lead vehicle / road geometry, else "shortcut" flag) |
| **I3** | latent-geometry views: PCA/2-D of ẑ coloured by speed/curvature/lead-gap/road-class + counterfactual latent surgery (+gap ⇒ braking relaxes) | I1b dumps | 0-GPU | organisation figure + surgery sign-check table |
| **V1** | W4b gates (feat/kin) → selector decision | running | — | G1/G2 per prereg |
| **V2** | E4.4 gate recovery (--eval-only, fix `0f6367e`) | after W4b | 15 min | e44_gate.json |
| **V3** | v5.8f assembled T0 eval + frozen-argmax control | after V2 | 1.5 GPU-h | v58f_eval JSONs + windows for four-families/selgap rescore |
| **V4** | four-families + selgap bootstrap on V3 windows; registry **§1.14** row; HF release | V3, 0-GPU-ish | 1 h | the v5.8f row (T0-stamped) + HF `Sayood/tanitad-flagship-v5f-w120` `/v58f/` |
| **V5** | E1.4: T1 rows for v5.8f (+v1arch comparability) via `t1_eval` | V4; byte-close gate on pod4 dumps first | 2 GPU-h | the PRIMARY-tier v5.8f numbers |
| **V6** | E5.1/5.2 goal-conditioned decode + **E5.3 S-curve restoration (T1)** | E4.4 ckpt (have) | 1 d dev + 5 GPU-h | the pillar's make-or-break, prereg'd in V18 |
| **V7** | W5 (6 s horizon) + W3 (stage-A probes → also feeds P3/P6) | free pod | 5 GPU-h | E-H1 gate; controllability R² |
| **V8** | release video v2: add the **decoded-BEV pane** to the plan-fan video (camera \| plan fan \| decoded-BEV(ẑ) \| GT-BEV) | I1c + V4 | CPU render | the v5.8f showcase reel |

## Schedule (from 2026-08-10 ~16:00Z)

```
pod5 GPU: W4b-feat ─ W4b-kin ─ E4.4-eval ─ v58f evals ─┬─ I1b P8 train ─ I1c ─ I2 ─ V7(W3/W5) ─ V5(T1) ─ V6 arms
                                              (~21:30Z) │
pod4:     Alpamayo tail ─ aug_pack + card ─ I1a join ───┘   (join relays pod4→pod5, jsonl, jupyter route)
dev/0-GPU: V4 registry+HF ─ I3 views ─ V6 dev (E5.1 implementation) ─ V8 render script
```

Interpretability is NOT an appendix: **V4's registry row carries the P-battery column** (P7 ✅
already; P8/P9 verdicts appended as they land), and **V8 puts the decoded world-belief on
screen next to the plan** — the PI's "predicting the right relevant part" made visible in the
same artifact that shows the driving.

## Standing rules that bind every line

Pre-registered gates before runs; four families + tier stamps on every eval row; episode-cluster
bootstrap for any registry claim; agents stage-never-push; every artifact banked in-repo with
provenance; a failed gate is a result (recorded, root-caused, next lever named — never
narrated around).

- [x] V1 feat arm training (started 15:30Z) · [ ] V1 gates · [ ] V2 · [ ] V3 · [ ] V4 · [ ] V5 · [ ] V6 · [ ] V7 · [ ] V8
- [ ] I1a · [ ] I1b · [ ] I1c · [ ] I2 · [ ] I3   (P7 ✅ 2026-08-10, ρ=0.49)
