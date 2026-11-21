---
license: other
license_name: physicalai-av-derived-research-only
tags:
- tanitad
- autonomous-driving
- anchored-diffusion
- trajectory-prediction
- planner
- reference-arm
extra_gated_prompt: >-
  These weights are trained on NVIDIA PhysicalAI-AV data (TanitAD research
  program). Access is granted per request for research/evaluation use only;
  you agree not to redistribute.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
---

# TanitAD — REF-C-XL (anchored diffusion, step 29,999 / 30,000 FINAL)

Reference arm C of the TanitAD program: a **DiffusionDrive-style anchored-diffusion direct planner** —
the *budget-matched non-world-model control* for the hierarchical 4-brain flagship. This is the **XL**
rung (251.9 M params) of the three-size REF-C ladder (small 54.7 M · base 104.2 M · XL 251.9 M).

**Registry key:** `refc-diffusion-xl-30k` · **TanitEval arm key:** `refc-xl-30k`

**Source of truth for every number below:** `Project Steering/MODEL_REGISTRY.md` §4.1 and the raw eval
JSON `taniteval/results/refc-xl-30k.json`. Evidence class: **MEASURED** unless marked otherwise.

---

## Architecture

Read from this run's own `config.json` (shipped in this repo).

- **Encoder** — torchvision-free ResNet-L, `in_channels 9` (3 RGB frames at 100 ms spacing,
  channel-stacked), `image_size 256`, `base_width 124`, blocks `(3, 8, 20, 6)`.
- **Anchor vocabulary** — **256** trajectory anchors built by **furthest-point sampling** from a
  4,096-window pool, `seed 0` (FPS, *not* k-means: the corpus is ~74 % straight and k-means collapses
  onto the straight mode).
- **Decoder** — anchor queries cross-attend the conv feature map: `d 512`, 8 heads, 6 layers,
  `ff_mult 4`, `aux_hidden 512`, **2 truncated-denoise steps**, `noise_std 0.1`. Emits per-anchor
  confidence + offset.
- **Measurement encoder** — `hidden 128`, `d_out 128`, `ego_dropout 0.5` (anti-shortcut).
- **LAW** latent-world-model auxiliary — `hidden 2048`.
- **Strategic-context graft** — `hidden 768`, `d_ctx 96`; `hierarchy true`, `graft_maneuver true`
  (H19 maneuver-to-anchor prior, live from step 0).
- **Imagination graft (H15 belief field) — ON, XL-only**: `d 512`, depth 6, 8 heads, `ff_mult 4`,
  `head_hidden 1024`.
- Horizons `(5, 10, 15, 20)` steps @ 10 Hz = 0.5 / 1 / 1.5 / 2 s · `path_dists (2, 5, 10, 20)` m ·
  `speed_bins 4`, `speed_max 30.0` · `graft_target_latent false`, `grounded_selector false`,
  `refc1 false`.

### Parameters — measured at instantiation (`config.json` `param_breakdown`)

| module | params |
|---|---|
| encoder | 199,496,532 |
| decoder | 22,702,345 |
| imagination (H15) | 20,986,339 |
| strategic | 4,133,472 |
| law | 4,082,656 |
| aux | 513,960 |
| measurement | 17,280 |
| **total** | **251,932,584** |

`n_params_trainable = 251,932,584` (all trainable — no frozen encoder). Anchors are **buffers, not
parameters** (~0.048 M of buffers total).

---

## Training

| item | value |
|---|---|
| Corpus | **NVIDIA PhysicalAI-AV** front-wide, **2,376 episodes / 406,099 windows** (train split) |
| Strict-parity build key | **`physicalai-train-e438721ae894`** |
| Corrupt-clip skip-hash | **`f09e44db`** (24 corrupt front-wide clips excluded) |
| Steps | **29,999 / 30,000** (`metrics.json` `final.step = 29999`, `steps = 30000`) |
| Optimizer | **Adam** (DiffusionDrive/TCP convention, *not* AdamW), lr **1e-4**, warmup 2000, **cosine** |
| Loss weights | traj 1.0 · cls 1.0 · law 0.5 · route 0.1 · man 0.1 · speed_cls 0.2 |
| Batch / workers | 20 / 6 · seed 0 · device cuda |
| Route labels | **v1** (`route_target(nav_cmd)`) — see caveat 5 |
| Hardware | `tanitad-pod3`; finished 2026-07-20 09:19 UTC |

Exact command (read from live `ps` on the training pod):

```bash
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 scripts/refc_train.py \
  --data-root /workspace/pai_epcache \
  --out /workspace/experiments/refc-diffusion-xl-30k \
  --steps 30000 --mode diffusion --config xl \
  --anchors /workspace/experiments/refc_anchors_full.pt \
  --batch 20 --workers 6
```

Code: `stack/tanitad/refs/refc.py` + `stack/scripts/refc_train.py` (commits `6025769` redesign,
`7e9c402` sizing; 15 `tests/test_refc.py` pass). Anchors built by `stack/scripts/build_refc_anchors.py`.

Final training metrics (step 29,999, `metrics.json`): loss 1.86033 · traj 0.17534 · cls 1.50755 ·
law 0.02031 · route 0.91015 · man 0.76259 · anchor_acc 0.50 · man_acc 0.75.

---

## Evaluation

**TanitEval `taniteval.refc_eval`**, open-loop, on the **clean held-out split
`physicalai-val-0c5f7dac3b11`** — **40 episodes → 881 windows**, episode-disjoint from train.
Protocol: window 8, stride 8, K = 20 steps @ 10 Hz, waypoints [5, 10, 15, 20], metric-BEV ego frame,
**nav = follow**, decode = anchored diffusion with **2 truncated-denoise steps** over 256 anchors,
argmax-confidence select. `ckpt_step` recorded in the result JSON = **29999**.

### Headline — decision-grade (n = 881 windows / 40 episodes)

| metric | value | estimator |
|---|---|---|
| **ADE@2s (full-set)** | **0.4714** · CI95 **[0.3896, 0.5556]** (±0.0830) | **episode-cluster bootstrap**, B = 2000 over the 40 val episodes (`taniteval/ci.py`) |
| FDE@2s (full-set) | **1.0061** · [0.8301, 1.1875] | same |
| miss@2m (full-set) | **0.1419** · [0.0943, 0.1918] | same |
| TMS-openloop (full-set) | 0.2135 | point estimate |
| ADE@0.5 / 1 / 1.5 s (full-set) | 0.0681 / 0.1592 / 0.2932 | point estimates |

**Trivial floor on the same split:** constant-velocity ADE@2s **0.8377** (full-set) / 0.8248 (heldout).
REF-C-XL beats CV overall and **in every stratum**.

> **Estimator caveat — never quote an interval without its estimator.** An earlier published row read
> *0.458 ± 0.057*. That `±` is **`overlapping_holdout_se`** (historically mislabelled "8-split
> episode-disjoint jackknife"); it is **neither a jackknife nor a valid SE** and is measured
> **1.28–2.06× too narrow** across 10 arms (**1.45× for this arm**). It is retained here only for
> continuity with older publications. **Use the episode-cluster bootstrap row.**
> Source: `Project Steering/CI_RECOMPUTE_2026-07-20.json`.

### Strata (full-set, step 29,999)

| stratum | REF-C-XL ADE@2s | CV baseline | n |
|---|---|---|---|
| speed **high** | **0.3243** | 0.6468 | 294 |
| speed med | 0.4989 | 0.9345 | 293 |
| speed low | 0.5912 | 0.9322 | 294 |
| curv straight | 0.3865 | 0.4393 | 634 |
| curv gentle | 0.6751 | 1.3566 | 125 |
| curv sharp | 0.7040 | 2.3764 | 122 |

The high-speed stratum is where the hierarchical flagship is weakest (0.5513) — a direct-head diffusion
arm beats the world-model stack there by ≈ 41 %.

### Efficiency — batch 1, one A40, identical precision flags

| | fp32 | tf32 | amp16 |
|---|---|---|---|
| plan tick p50 | **44.06 ms** | 27.78 ms | 21.00 ms |
| plan tick p99 (fp32) | 44.44 ms | — | — |
| GFLOPs / peak alloc (fp32) | 702.2 / 1178.4 MB | — | — |

Encoder is **88.7 %** of the tick. Meets a 10 Hz budget in all three precisions. (REF-C-base is
1.32–2.02× faster at statistically indistinguishable accuracy — see caveat 2.)

---

## Honest caveats — all recorded in the registry; read them before quoting this model

1. **Flagship v1 vs REF-C-XL is a TIE, not an ordering.** Under the **paired** episode-cluster bootstrap,
   Δ(REF-C − flagship v1) = **+0.0443 m, CI95 [−0.0544, +0.1465]**, P(Δ>0) = 0.809 → **NOT separated**.
   A prior claim that REF-C finished "0.006 m behind the flagship" was **RETRACTED**: 0.006 m was a
   difference of *split means*; the full-set gap is 0.0443 m (8× larger) and still not separated.
   Per-window correlation is only 0.207 here, so pairing buys little power — the tie is real, not a
   weak test.

2. **REF-C-base (104.2 M) ties this model on everything that ships.** Paired Δ(base − XL): ADE@2s
   **+0.0013 [−0.0281, +0.0316]**, FDE **−0.0030 [−0.0619, +0.0584]**, miss **+0.0000
   [−0.0261, +0.0272]** — all **NOT separated**, per-window correlation 0.789 (the test is not weak).
   A **2.42× parameter cut** and a **2.20× encoder cut** cost nothing measurable on this corpus.

3. **Selection flaw — REF-C ranks with the UN-refined anchor's score.** All 256 anchors are denoised, but
   selection uses the t=0 classifier score over the *original* anchors; the denoise passes' own
   confidences are discarded. Geometry is refined, ranking is not. Corpus figures (n = 881): selected
   **0.4714** · **oracle-in-fan 0.1640** · gap **0.3075 m** · `frac_sel_2x_worse` **0.454**.

4. **The oracle gap is ~92 % IRREDUCIBLE — it is not available headroom.** Settled across 47 trained
   arms: a learned re-scorer recovers at most **8.4 %** of it on its own training data; a hand-written
   cost re-rank recovers **0.0 %**; scoring the *refined* confidences is **2.9× WORSE** than baseline
   (that head is unsupervised at denoise timesteps). Do **not** add a target-speed term to the selection
   score (REFUTED: a GT-perfect speed matcher scores worse than baseline). Selection is not the
   productive lever on this architecture.

5. **Route-label confound vs the base and small rungs.** XL trained with **v1** route labels
   (`route_target(nav_cmd)` — circular *and* straight-by-default; `labels_v2` was never set in
   `refc_train.py`). REF-C-base and REF-C-small trained with **v2.1**. **XL-vs-base therefore conflates
   scale and labels.** Calibration (flagship v1.5, the only end-to-end measurement of the label change):
   ADE **+0.025 m, not CI-separated**, but **oracle −0.058 m**. Do not present a clean scaling
   conclusion. The clean resolution is one control run (XL-with-v2.1 or base-with-v1) — **not yet run**.

6. **The fan lever is anchor-vocabulary WIDTH, not encoder scale.** base's 128 anchors are a bit-exact
   prefix of XL's 256, so the fans compare over an identical vocabulary. Oracle-in-fan over the first K:
   base ≤ XL at **every** matched K (4→128); XL's entire oracle advantage arrives with anchors 129–256.
   Anchors are buffers (~0.048 M), so widening the vocabulary is nearly free, while widening the encoder
   demonstrably bought nothing here. *(Caveat: a prefix restriction structurally penalises XL, whose
   winner-takes-all training spread modes across 256 slots.)*

7. **Closed-loop numbers are ENV-CONFOUNDED — do NOT read them as a model result.** On the AlpaSim NuRec
   suite (n = 12) this arm scores at-fault collision 33.3 %, off-road 25.0 %, pass 5/12, mean score
   0.246. But REF-C's **open-loop ADE on those same reconstructions is 1.52 m — 3.21× its real-footage
   0.4714**, i.e. the input is ~3× off the training distribution. Those rates measure model ×
   reconstruction fidelity, not the model. n = 12 also means one scene = 8.3 pp.

8. **Reproducibility gap (declared, not hidden):** the TanitEval harness that produced every number above
   lives on `tanitad-eval:/root/taniteval` and is **not committed to the TanitAD repo**. The checkpoint
   survives; the exact evaluator currently does not travel with it.

9. This is a **research reference arm, not a deployable driving system.** Open-loop ADE does not predict
   closed-loop behaviour (measured elsewhere in this program: 0.45 m open-loop → 1.69 m closed-loop for
   the flagship). Claim strength on all metrics above: **open-loop / weak**.

---

## Files

| file | contents |
|---|---|
| `ckpt.pt` | full training checkpoint — `{model: state_dict (673 tensors), opt: optimizer state, step: 29999}`, 3,024,021,445 B, **md5 `966d4eff1ea5ddf86efba01b8344e198`** |
| `config.json` | the run's own config: architecture, args, optimizer, loss weights, `param_breakdown` |
| `metrics.json` | final training + val metrics at step 29,999 |

Verify after download: `md5sum ckpt.pt` must print `966d4eff1ea5ddf86efba01b8344e198`.

## Intended use / license

Internal research and evaluation only. Derived from **NVIDIA PhysicalAI-AV** (gated corpus) — not
redistributable. Not a deployable driving system.
