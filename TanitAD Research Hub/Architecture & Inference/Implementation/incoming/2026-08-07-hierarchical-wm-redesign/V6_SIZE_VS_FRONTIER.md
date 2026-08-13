# v6 component sizing vs the frontier — and how to reach the 250–350 M band

**PI, 2026-08-13:** *"research and review the size of the different components of v6, compare
it with frontier best known models in field of autonomous driving and predictive world
models. Ideally it is within our budget of 250 to 350 Million parameters."*

**Bottom line: v6 is 87.89 M — roughly 3× below the bottom of the requested band.** It can be
brought to **288.6 M** without touching the sub-300 M guard, and the measured evidence says
the extra capacity should go to the **operative predictor and the hierarchy**, *not* to the
encoder. Recommendation and the config are in §4.

Evidence classes are marked throughout: **MEASURED (ours)** = counted at instantiation in
this repo; **PUBLISHED (cited)** = from the linked source. Nothing here is INHERITED.

---

## 1. Where v6's parameters are today — MEASURED

| component | params | share | architecture |
|---|---:|---:|---|
| encoder (ViT) | 15.33 M | 17.4 % | 384 wide × 8 deep, 6 heads, patch 16, 9ch 256×640 → 640 tokens |
| readout | 0.05 M | 0.1 % | 4×4 grid × 128 → **state_dim 2048** (the geometry firewall) |
| **predictor_op** | **60.29 M** | **68.6 %** | 768 × 6, 12 heads — the only attention-over-tokens in v6 |
| layer_tac | 5.77 M | 6.6 % | `FTac` residual MLP, d 512 @ 2 Hz, 3 blocks + goal/LAT/LON heads |
| layer_str | 4.15 M | 4.7 % | `FTac` residual MLP, d 256 @ 0.5 Hz + goal/action heads |
| planner | 0.66 M | 0.7 % | 8 candidates × 60-step unicycle rollout |
| aux | 1.65 M | 1.9 % | masked-cell (O3) head |
| vocabularies | 0.012 M | <0.1 % | 5 shared tables (identity-shared, emitter ↔ consumer) |
| **TOTAL** | **87.89 M** | | budget 300 M · headroom 212 M |

**The hierarchy — the programme's entire thesis — is 10.6 M, 12 % of the model.** Two thirds
of v6 is one operative-band transformer.

---

## 2. The frontier, by class — PUBLISHED

The field **bifurcates**, and the comparison is only meaningful within a class. v6 is a
*latent, non-generative* world model with a planner: it never renders a pixel, so it does not
pay for a video tokenizer or a diffusion decoder.

### Class A — generative video world models (pixel-space)
| model | params | note |
|---|---:|---|
| [Wayve GAIA-1](https://wayve.ai/thinking/scaling-gaia-1/) | **9 B** total (6.5 B world model) | ~23 000 A100-h |
| [Wayve GAIA-2](https://wayve.ai/thinking/gaia-2/) | **~7.5 B** ⚠️ inferred — GAIA-3 is stated as "double GAIA-2" | video tokenizer + latent diffusion |
| [Wayve GAIA-3](https://wayve.ai/press/wayve-launches-gaia3/) | **15 B** | 5× GAIA-2 compute, ~10× data |
| [NVIDIA Cosmos](https://blogs.nvidia.com/blog/cosmos-world-foundation-models/) | **4 B – 14 B** | AR 4B/12B, diffusion 7B/14B |
| Epona | 2.5 B | via the [driving WM survey](https://github.com/HaoranZhuExplorer/World-Models-Autonomous-Driving-Survey) |
| [Orbis v1](https://arxiv.org/abs/2507.13162) | **469 M** | flow-matching |
| [Orbis 2](https://www.automotiveworld.com/news/university-of-freiburg-and-natix-unveil-orbis-2-model/) | ~½ of Cosmos-v2.5 | ⭐ see §3 |

### Class B — VLA / reasoning driving models
| model | params | note |
|---|---:|---|
| [NVIDIA Alpamayo 1](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) | **10 B** | chain-of-thought VLA on a Cosmos-Reason backbone |

### Class C — latent / JEPA predictive world models — **our family**
| model | params | note |
|---|---:|---|
| [V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | **1.2 B** | ViT-g encoder **1 B** (width 1408, depth 40, 22 heads) + ViT-s predictor **22 M** |
| V-JEPA 2 (smaller arm) | ~300 M | ViT-L encoder |
| [DINO-WM](https://dino-wm.github.io/) | **~19 M predictor** | ViT depth 6, 16 heads, MLP 2048, over FROZEN DINOv2 ViT-S/16 (384-d) |

### Class D — end-to-end planners with real driving scores (nuPlan / NAVSIM)
| model | params | note |
|---|---:|---|
| [UniAD](https://www.emergentmind.com/topics/unified-autonomous-driving-uniad) | **>100 M** (a 0.5 B variant exists) | CVPR'23 best paper |
| [DiffusionDrive / SparseDrive](https://arxiv.org/html/2411.15139v1) | **21.8 M** backbone (ResNet-34) | 88.1 PDMS on NAVSIM navtest |
| [Hydra-MDP++ / GoalFlow](https://arxiv.org/html/2503.12820) | **96.9 M** backbone (V2-99) | 91.0 drive score |

---

## 3. What the comparison actually says

**⭐ The single most informative datapoint is Orbis 2** — the first *hierarchical* driving
world model, and therefore our nearest published analogue. It factorises exactly the way v6
does: *"a high-level predictor that forecasts coarse scene structure over extended temporal
horizons, and a low-level generator that produces detailed predictions conditioned on the
high-level output."* It compresses **DINOv2 features into a lower-dimensional latent** as the
high-level target — the same move as our readout firewall. And it *"outperformed NVIDIA
Cosmos-v2.5 while using roughly half the parameters and one-third of the data"*, trained in
**under 3 000 H100-hours** against ~23 000 A100-hours for GAIA-1.
⇒ **Hierarchy is a parameter- and compute-EFFICIENCY lever in the published record, not a
capacity lever.** That is direct external support for v6's thesis, and it is also a warning
against reading "small" as "under-powered".

**Three readings that follow:**

1. **v6's 88 M is not anomalous for its class — it is anomalous for its AMBITION.** Class C
   spans 19 M (DINO-WM) to 1.2 B (V-JEPA 2), and Class D — the models with actual driving
   scores — sits at **22–100 M**. Against Class D, v6 is already *comparable or larger*.
   Against Class A it is 50–170× smaller, but Class A pays almost all of that for pixel
   generation we deliberately do not do.
2. **The generative giants are not the right target.** GAIA-3 at 15 B and Cosmos at 14 B are
   *simulators and evaluators*; Wayve positions GAIA-3 for "simulation to evaluation". Our
   product is a planner. Matching their parameter count would buy a decoder we would throw
   away.
3. ⚠️ **But v6 IS under-provisioned where its own thesis lives.** V-JEPA 2 spends 1 B of 1.2 B
   on the encoder and only 22 M on the predictor — the opposite split from ours. DINO-WM
   freezes a pretrained encoder entirely and spends everything on the predictor. v6 does
   neither cleanly: it trains a *small* encoder (15 M) AND puts 69 % into one operative
   predictor, leaving the hierarchy at 12 %. **No frontier system in Class C allocates the
   way we do.**

---

## 4. Reaching 250–350 M — MEASURED configurations

All counted at instantiation today:

| config | TOTAL | enc | pred_op | tac | str | in band |
|---|---:|---:|---:|---:|---:|:--:|
| current default | 87.89 M | 15.3 | 60.3 | 5.8 | 4.2 | — |
| A: pred 1024×10 only | 188.35 M | 15.3 | 160.7 | 5.8 | 4.2 | — |
| B: enc 768×12 + pred 768×10 | 193.01 M | 87.3 | 93.4 | 5.8 | 4.2 | — |
| **C: enc 768×12 + pred 1024×10** | **260.39 M** | 87.3 | 160.7 | 5.8 | 4.2 | ✅ |
| **D: enc 512×12 + pred 1024×12 + tac 768 / str 512 + 6 blocks** | **288.61 M** | 39.3 | 190.1 | **29.3** | **27.4** | ✅ |
| **E: enc 768×12 + pred 1024×12 + tac 768 / str 512 + 6 blocks** | **336.62 M** | 87.3 | 190.1 | 29.3 | 27.4 | ✅ |
| F: pred 1280×14 + tac 1024 / str 768 + 8 blocks | 517.41 M | 15.3 | 338.4 | 81.7 | 79.6 | ✗ over |

### Recommendation: **config D — 288.6 M**

1. **It is the only in-band config that grows the HIERARCHY.** C reaches 260 M by making the
   operative predictor bigger and leaves tac+str at 10 M (**3.8 %** of the model) — that is
   the current imbalance, scaled up. D lifts the hierarchy to **56.7 M (19.6 %)**, which is
   the first allocation that matches what the architecture claims to be.
2. **It respects the one piece of evidence we actually have.** E-ENC MEASURED at step 500
   that the *encoder* at 768×12 was **worse** on 7 of 8 objectives and **1.50× slower** than
   384×8. D therefore routes growth to the predictor and hierarchy and takes the encoder only
   to a middle 512×12 (39.3 M) rather than back to v5's 87.3 M. E is D + the wide encoder and
   is the arm E-ENC already argued against.
3. **It fits the existing guard.** `param_budget` defaults to **300 M** and
   `build_stack_from_args` refuses above it *before any GPU time*. D at 288.6 M is the
   **largest configuration that needs no change to the invariant**. Anything above (E at
   336.6 M) requires the PI to raise `--param-budget` — a deliberate act, which is the right
   friction for changing a stated invariant.

⚠️ **What I will not claim.** Growing to 288 M is **not** a fix for our measured defects.
T1 found hold-action beating the closed loop **22×** and ~**99 %** of the gap longitudinal —
those are *conditioning and selection* failures, and E-ENC just measured that more capacity
in the visual trunk did not help at step 500. The honest case for D is **headroom for the
6 s / 60-step contract and for a hierarchy that is currently too small to carry its own
thesis**, not a performance prediction. Anyone quoting D as "bigger therefore better" is
repeating the mistake E-ENC was run to prevent.

### Cost, MEASURED-derived
Current S-W runs at **7.19 s/step** at 87.9 M. Config C at 768×12 measured **10.76 s/step**
(1.50×). D is smaller in the encoder but larger in the predictor; **ESTIMATED** ~1.6–2.0×
current → **13–17 h per 5 000 steps**, ~**80–100 h** for a full 30 k. That is a real
step-change in spend and belongs in the same decision as the band itself.

---

## 5. Open, and not guessed

- **GAIA-2's exact count is inferred**, not published in the sources reachable from here
  (arxiv, HuggingFace and openaccess are egress-blocked in this environment). It is marked ⚠️
  above and must not be quoted as MEASURED or PUBLISHED-exact.
- **Orbis 2's absolute parameter count** is stated only relative to Cosmos-v2.5 in the press
  release; the paper itself was unreachable. Its *architecture* claim is what the argument
  above rests on, and that is quoted directly.
- **Class D counts are backbone-only** in several sources (ResNet-34 21.8 M, V2-99 96.9 M);
  full-system totals are larger and were not consistently reported.
