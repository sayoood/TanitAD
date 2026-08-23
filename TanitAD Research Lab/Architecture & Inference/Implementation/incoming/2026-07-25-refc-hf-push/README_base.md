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

# TanitAD — REF-C-base / medium (anchored diffusion, step 29,999 / 30,000 FINAL)

Reference arm C of the TanitAD program: a **DiffusionDrive-style anchored-diffusion direct planner** —
the *budget-matched non-world-model control* for the hierarchical 4-brain flagship. This is the **base
(medium)** rung, **104.2 M params**, of the three-size REF-C ladder (small 54.7 M · base 104.2 M ·
XL 251.9 M).

**This is the rung the program actually recommends.** It ties the 2.42×-larger XL on every shipping
metric while running 1.32–2.02× faster.

**Registry key:** `refc-diffusion-base-v21-30k` · **TanitEval arm key:** `refc-base-30k`

**Source of truth for every number below:** `Project Steering/MODEL_REGISTRY.md` §4.3 and the raw eval
JSON `taniteval/results/refc-base-30k.json`. Evidence class: **MEASURED** unless marked otherwise.

---

## Architecture

Read from this run's own `config.json` (shipped in this repo).

- **Encoder** — torchvision-free ResNet-M, `in_channels 9` (3 RGB frames at 100 ms spacing,
  channel-stacked), `image_size 256`, `base_width 88`, blocks `(3, 6, 16, 6)`.
- **Anchor vocabulary** — **128** trajectory anchors by **furthest-point sampling** from a 4,096-window
  pool, `seed 0`. Verified a **strict, bit-exact prefix of XL's 256** (`refc_anchors_base128.pt`, same
  script / source / pool cap / seed; `max|A − B[:128]| = 0` at load).
- **Decoder** — `d 384`, 8 heads, 4 layers, `ff_mult 4`, `aux_hidden 384`, **2 truncated-denoise steps**,
  `noise_std 0.1`.
- **Measurement encoder** — `hidden 128`, `d_out 128`, `ego_dropout 0.5`.
- **LAW** latent-world-model auxiliary — `hidden 2048`.
- **Strategic-context graft** — `hidden 512`, `d_ctx 64`; `hierarchy true`, `graft_maneuver true`.
- **Imagination graft (H15) — OFF** by preset design (XL-only). Contributes **0** params here.
- Horizons `(5, 10, 15, 20)` steps @ 10 Hz = 0.5 / 1 / 1.5 / 2 s · `path_dists (2, 5, 10, 20)` m ·
  `speed_bins 4`, `speed_max 30.0` · `graft_target_latent false`, `grounded_selector false`,
  `refc1 false`.

### Parameters — measured at instantiation (`config.json` `param_breakdown`)

| module | params |
|---|---|
| encoder | 90,458,632 |
| decoder | 8,634,505 |
| law | 2,902,720 |
| strategic | 1,903,680 |
| aux | 274,760 |
| measurement | 17,280 |
| imagination | 0 (graft off) |
| **total** | **104,191,577** |

`n_params_trainable = 104,191,577`. The code docstring's "~110 M" was **5.6 % high** — the measured
number above is the one to quote. Anchors are buffers, not parameters.

---

## Training

| item | value |
|---|---|
| Corpus | **NVIDIA PhysicalAI-AV** front-wide, **2,376 episodes / 406,099 windows** (train split) — booked in this run's `config.json` `data` block |
| Strict-parity build key | **`physicalai-train-e438721ae894`** |
| Corrupt-clip skip-hash | **`f09e44db`** (24 corrupt front-wide clips excluded) |
| Steps | **29,999 / 30,000** (`metrics.json` `final.step = 29999`, `steps = 30000`) |
| Optimizer | **Adam** (DiffusionDrive/TCP convention, *not* AdamW), lr **1e-4**, warmup 2000, **cosine** |
| Loss weights | traj 1.0 · cls 1.0 · law 0.5 · route 0.1 · man 0.1 · speed_cls 0.2 |
| Batch / workers | 20 / 6 · seed 0 · device cuda |
| Route labels | **v2.1** (`refb_labels.route_from_future_v21`, `use_net_dyaw=False`, `ROUTE_UNKNOWN=3` **masked** out of the 0.1-weight CE, never clamped) |
| Hardware | `tanitad-pod3`; finished 2026-07-21 04:44 UTC. Evaluated on `tanitad-eval` 2026-07-21 05:18–05:19 UTC under the `refc-base-eval` GPU lock |

Exact command:

```bash
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 scripts/refc_train.py \
  --data-root /workspace/pai_epcache \
  --out /workspace/experiments/refc-diffusion-base-v21-30k \
  --steps 30000 --mode diffusion --config base \
  --anchors /workspace/experiments/refc_anchors_base128.pt \
  --labels v21 \
  --batch 20 --workers 6
```

**Parity with XL:** same corpus and parity key, 30 k steps, same optimizer/schedule, same loss weights,
`--mode diffusion`, `--batch 20 --workers 6`.
**Deliberate differences vs XL:** `--config base` (2.42× smaller) · 128 anchors (prefix of XL's 256) ·
H15 imagination OFF · **route labels v2.1** (see caveat 4).

**v2.1 label coverage** (4,000-window sample, recorded in `config.json`): left 0.121 · straight 0.5645 ·
right 0.115 · **UNKNOWN 0.1995 (masked out)** → **80.05 % judgeable**, against v1's straight-by-default
target. Unknown reasons: gray_zone 0.1103 · no_arc 0.0892 · road_following 0.5645 · tight_transient 0.236.

Code: `stack/scripts/refc_train.py` gained `--labels {v1,v21}` (**default `v1` = XL-reproducible**),
`RouteV21Dataset`, a fail-loud masked route CE, and 5 k/15 k/20 k/30 k milestone archiving (a gate series
XL lacks). 15/15 `tests/test_refc.py` pass.

Final training metrics (step 29,999, `metrics.json`): loss 1.37374 · traj 0.18426 · cls 1.09855 ·
law 0.01716 · route 0.28667 · man 0.5368 · anchor_acc 0.55 · man_acc 0.90 · route_acc 0.85.

---

## Evaluation

**TanitEval `taniteval.refc_eval`**, open-loop, on the **clean held-out split
`physicalai-val-0c5f7dac3b11`** — **40 episodes → 881 windows**, episode-disjoint from train.
Protocol **identical to XL**: window 8, stride 8, K = 20 steps @ 10 Hz, waypoints [5, 10, 15, 20],
metric-BEV ego frame, **nav = follow**, anchored-diffusion decode with **2 truncated-denoise steps**
over 128 anchors, argmax-confidence select. `ckpt_step` in the result JSON = **29999**.

Parity with XL's eval proven three ways: the same 881 episode ids, bit-identical ground truth, and a
**bit-identical CV baseline in every stratum**.

### Headline — decision-grade (n = 881 windows / 40 episodes)

| metric | **REF-C-base** (104.2 M) | REF-C-XL (251.9 M) | paired Δ (base − XL) |
|---|---|---|---|
| **ADE@2s (full-set)** | **0.4728** · CI95 [0.3835, 0.5699] | 0.4714 · [0.3896, 0.5556] | **+0.0013 [−0.0281, +0.0316] — NOT separated** |
| FDE@2s (full-set) | **1.0031** · [0.8148, 1.2087] | 1.0061 · [0.8301, 1.1875] | **−0.0030 [−0.0619, +0.0584] — NOT separated** |
| miss@2m (full-set) | **0.1419** · [0.0874, 0.2000] | 0.1419 · [0.0943, 0.1918] | **+0.0000 [−0.0261, +0.0272] — NOT separated** |
| TMS-openloop (full-set) | 0.1957 | 0.2135 | — |
| ADE@0.5 / 1 / 1.5 s | 0.0708 / 0.1620 / 0.2960 | 0.0681 / 0.1592 / 0.2932 | — |

**Estimator:** `taniteval/ci.py` **episode-cluster bootstrap**, B = 2000 over the 40 val episodes; the
**paired** form for the deltas. **Per-window ADE correlation 0.789**, so the pairing is doing real work —
the non-separation is not a weak test.

**Trivial floor on the same split:** constant-velocity ADE@2s **0.8377** (full-set) / 0.8248 (heldout).
REF-C-base beats CV overall and **in every stratum**.

> **Verdict: REF-C-base and REF-C-XL are statistically indistinguishable on everything that ships.**
> All three paired intervals straddle zero and the point deltas are ≤ 0.003 m — a **2.42× parameter cut**
> and a **2.20× encoder cut** (90,458,632 vs 199,496,532) cost **nothing measurable** on this corpus.

> **Estimator caveat — never quote an interval without its estimator.** A legacy row for this arm reads
> *0.4523 ± 0.0497*. That `±` is **`overlapping_holdout_se`** (historically mislabelled "8-split
> episode-disjoint jackknife"); it is **neither a jackknife nor a valid SE**, measured **1.28–2.06× too
> narrow** across 10 arms. It is retained only for continuity with older publications. **Use the
> episode-cluster bootstrap row.** Source: `Project Steering/CI_RECOMPUTE_2026-07-20.json`.

### Strata (full-set, step 29,999)

| stratum | base ADE@2s | XL ADE@2s | CV | n |
|---|---|---|---|---|
| speed **high** | 0.3510 | **0.3243** | 0.6468 | 294 |
| speed med | **0.4483** | 0.4989 | 0.9345 | 293 |
| speed low | 0.6189 | **0.5912** | 0.9322 | 294 |
| curv straight | 0.3866 | 0.3865 | 0.4393 | 634 |
| curv gentle | 0.6778 | 0.6751 | 1.3566 | 125 |
| curv sharp | 0.7105 | 0.7040 | 2.3764 | 122 |

The two arms **trade strata** (base wins med by 0.051; XL wins high by 0.027 and low by 0.028) — no
stratum-level ordering survives as a scale story.

### Fan quality — the read the decision actually needs

Raw: `taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json`.

| | base (128 anchors) | XL (256) | XL restricted to its first **128** |
|---|---|---|---|
| **oracle-in-fan** | **0.1914** [0.1654, 0.2184] | **0.1640** [0.1414, 0.1902] | 0.2624 [0.2262, 0.3011] |
| sel_gap (selected − oracle) | 0.2813 | 0.3075 | 0.2091 |
| `frac_sel_2x_worse` | 0.4109 | 0.4540 | 0.3190 |

*Paired:* base − XL(256) **+0.0275 [+0.0142, +0.0405] SEPARATED** (XL better) · base − XL(128)
**−0.0710 [−0.0965, −0.0502] SEPARATED** (base better).

Oracle-in-fan over the first K anchors (base ≤ XL at **every** matched K; XL's entire oracle advantage
arrives with anchors 129–256):

| K | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| **base** | 3.193 | **1.686** | **0.813** | **0.527** | **0.283** | **0.191** | — |
| **XL** | 3.535 | 2.274 | 1.226 | 0.806 | 0.437 | 0.262 | **0.164** |

> **The fan lever is anchor-vocabulary WIDTH, not encoder scale.** Anchors are buffers (~0.048 M total),
> and the decoder is only ~1.7 ms of base's 21.8 ms tick (encoder 90.7 %) — so widening the vocabulary is
> nearly free while widening the encoder demonstrably bought nothing here.

### Efficiency — batch 1, one A40, identical precision flags

| | base | XL | ratio |
|---|---|---|---|
| plan tick p50 fp32 / tf32 / amp16 | **21.78 / 15.81 / 15.88 ms** | 44.06 / 27.78 / 21.00 ms | **1.32–2.02× faster** |
| p99 fp32 | **22.33 ms** | 44.44 ms | meets 10 Hz in all 3 precisions (both arms) |
| GFLOPs / peak alloc | **292.5 / 556.7 MB** | 702.2 / 1178.4 MB | 0.42× / 0.47× |
| encoder share of the tick | 90.7 % | 88.7 % | — |

---

## Honest caveats — all recorded in the registry; read them before quoting this model

1. **Selection flaw — REF-C ranks with the UN-refined anchor's score.** All anchors are denoised, but
   selection uses the t=0 classifier score over the *original* anchors; the denoise passes' own
   confidences are discarded. Geometry is refined, ranking is not (base sel_gap **0.2813**,
   `frac_sel_2x_worse` **0.4109**).

2. **The oracle gap is ~92 % IRREDUCIBLE — it is not available headroom.** Settled across 47 trained
   arms: a learned re-scorer recovers at most **8.4 %** of it on its own training data; a hand-written
   cost re-rank recovers **0.0 %**; scoring the *refined* confidences is **2.9× WORSE** than baseline
   (that head is unsupervised at denoise timesteps). A target-speed term in the selection score is
   **REFUTED** (a GT-perfect speed matcher scores worse than baseline). Selection is not the productive
   lever on this architecture.

3. **base ≈ XL is a non-separation, not a proof of equality.** All three intervals straddle zero; that is
   evidence of no *measurable* difference on this corpus at n = 881 / 40 episodes, not of identity.

4. **CONFOUND — scale, anchor count and labels move together.** base trained on route labels **v2.1**,
   XL on **v1**. The matched-K control above removes the anchor-count confound; **nothing removes the
   label one.** Calibration from flagship v1.5, the only place the label change was measured end-to-end:
   ADE **+0.025 m (not CI-separated)** but **oracle −0.058 m** — i.e. v2.1 labels *improved the proposal
   set* by more than either oracle delta measured here, and base is the arm that had them. **Do not
   present a clean scaling conclusion.** What IS separable: on ADE/FDE/miss the arms tie. What is NOT
   separable: the sign and size of the encoder-scale effect on oracle-in-fan. The clean resolution
   remains one control run (XL-with-v2.1 or base-with-v1) — **not yet run**.
   The clean scale test in this ladder is **small-vs-base** (both v2.1): there small is
   SEPARATED-worse by ~0.053 m on selected ADE@2s, but **better per-anchor** on a matched 64-anchor
   vocabulary — the knee is anchor count, not encoder size.

5. **Closed-loop numbers are ENV-CONFOUNDED — do NOT read them as a model result.** On the AlpaSim NuRec
   suite (n = 12) this arm scores at-fault collision 33.3 %, off-road 16.7 %, pass 6/12, mean score
   0.345, dist-to-GT 1.642 m. But REF-C's **open-loop ADE on those same reconstructions is 1.52 m —
   3.21× its real-footage 0.4728**, i.e. the input is ~3× off the training distribution. Those rates
   measure model × reconstruction fidelity, not the model. The base ≥ XL *ordering* survives (both eat
   the same OOD); the *levels* do not. n = 12 means one scene = 8.3 pp.

6. **Reproducibility gap (declared, not hidden):** the TanitEval harness that produced every number above
   lives on `tanitad-eval:/root/taniteval` and is **not committed to the TanitAD repo**. The checkpoint
   survives; the exact evaluator currently does not travel with it.

7. This is a **research reference arm, not a deployable driving system.** Open-loop ADE does not predict
   closed-loop behaviour (measured elsewhere in this program: 0.45 m open-loop → 1.69 m closed-loop for
   the flagship). Claim strength on all metrics above: **open-loop / weak**.

---

## Files

| file | contents |
|---|---|
| `ckpt.pt` | full training checkpoint — `{model: state_dict (487 tensors), opt: optimizer state, step: 29999}`, 1,250,838,325 B, **md5 `8f10d6f934f4199e11ddc7352e074939`** |
| `config.json` | the run's own config: architecture, args, optimizer, loss weights, v2.1 label stats, data block, `param_breakdown` |
| `metrics.json` | final training + val metrics at step 29,999 |

Verify after download: `md5sum ckpt.pt` must print `8f10d6f934f4199e11ddc7352e074939`.

## Intended use / license

Internal research and evaluation only. Derived from **NVIDIA PhysicalAI-AV** (gated corpus) — not
redistributable. Not a deployable driving system.
