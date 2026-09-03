<title>RESULT - SOTA pass: encoder quality - frozen vs fine-tuned vs shaped (theme 2 of 4)</title>

# RESULT - E-ARCH-SOTA-ENC-1: the sign of "freeze vs fine-tune" depends on the trunk; frozen DINO patch tokens suffice for prediction but nobody shows them action-sensitive; frozen-compatible shaping exists at 0 trunk compute

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-encoder/` · Research Lab overnight pass 2026-09-03 · literature only · 0 GPU · no pod/Thor contact
**Spec** `SPEC.md` (staged before any primary was opened; H-SOTA-E1…E4 with both outcomes committed)
**Evidence classes** PUBLISHED-PRIMARY (banked PDF read in full, `lib:<key>`), MEASURED (registry / register), INHERITED (own summaries, not re-verified), UNVERIFIED (flagged). Nothing PUBLISHED-SECONDARY enters this file.
**Register rows proposed** `PROPOSED_HYPOTHESES.md` → LAB-ENC-1 … LAB-ENC-3.

## 0. Verdict in five lines

1. **Frozen-vs-fine-tuned ablations exist and point both ways, and the sign is decided by the trunk's strength** (H-SOTA-E1 → A): FROST-Drive `2601.03460` — a frozen 14 B VLM encoder scores RFS 8.17 (ADE@3 s 1.04) and the SAME encoder fully fine-tuned 8.13 (ADE 1.47), while a frozen ImageNet ViT 7.39 loses to its fine-tuned self 7.79. The Observer Effect `2602.12218` gives the mechanism: full fine-tuning erases the DYNAMIC variables (ρ 0.94 → −0.03 on a kinematic invariant; frozen linear probe 0.91 vs full fine-tune 0.05 vs last-layer 0.65), keeps the static ones.
2. **Frozen DINOv2/v3 PATCH tokens suffice for planning-quality prediction** (H-SOTA-E2 → A, conditional): DINO-WM `2411.04983` Table 2 — patch tokens 0.90 vs CLS 0.44 / R3M 0.42 / ResNet 0.20 on Push-T; What-Drives-Success `2512.24497` — frozen DINOv2/v3 beat frozen V-JEPA/V-JEPA 2 at equal ViT-L, DINOv3 wins only on photorealistic data; DINO-world `2507.19468` — a frozen DINOv2 + 1.1 B predictor beats copy-last on dense forecasting where a jointly trained V-JEPA encoder–predictor collapses (VSPW mid-term 47.0 vs 42.1 vs 7.7). refav1's frozen-DINOv3 design is field-supported IF its predictor consumes patch tokens (refav1 has no registry row — UNVERIFIED).
3. **No frozen-trunk world model reports a NUMERIC action-sensitivity metric** (DINO-WM, V-JEPA 2-AC, DINO-world, Utonia-WM, What-Drives-Success — the last has a qualitative counterfactual figure only), and the inverse-dynamics multiplier on DINO-family features is small (`2606.07687`: Web-DINO −0.01 → 0.16 vs V-JEPA 2 0.40 → 0.85). Prediction: refav1 will reproduce our action-insensitivity; it is a predictor/target problem AND a feature problem.
4. **Frozen trunks lose fine metric content and the field compensates with proprioception**: What-Drives-Success states that precise metric quantities are only implicitly encoded and quantised by the patch grid; Utonia-WM (frozen 137 M PTv3) scores 46.6 on Push-T against 83.6 for a trained tiny encoder — "the price of a representation that cannot adapt to the control task" (`2608.29434`). Our E-DEC-64 (freeze: nrmse +14.6 %) agrees in direction — with the caveat that our freeze arms froze a DISTILLED TINY trunk, not a large pretrained one (equivalence UNVERIFIED).
5. **Frozen-compatible shaping exists with numbers** (H-SOTA-E4 → A): a 196×384 → 196×32 bisimulation adapter on frozen DINOv2 patch tokens, trained jointly with the transition model, raises PointMaze success under lighting + colour + geometry shift from 0.48 (DINO-WM) to 0.78, encoder-agnostic (iBOT 0.72); training the encoder end-to-end without a pretrained trunk gives 0.26 vs 0.86 (`2602.18639`). GS-2 narrows: an adapter-IDM/LDAD arm on refav1's banked fp8 features costs 0 trunk compute (LAB-ENC-1).

## 1. Findings

### F1. Frozen vs fine-tuned vs shaped — the published rows (H-SOTA-E1 → A)

| primary | setting | frozen | fine-tuned / shaped | metric | reading |
|---|---|---|---|---|---|
| FROST-Drive `2601.03460` | end-to-end driving policy on a frozen VLM vision encoder + trained adapter/decoder | VLM 14 B **8.17** (ADE@3 s **1.04**); VLM 38 B 8.17; VLM 78 B **8.24**; ImageNet ViT 7.39 | VLM 14 B fine-tuned **8.13** (ADE **1.47**); ImageNet ViT fine-tuned **7.79** | Rater Feedback Score (↑), ADE (↓) | fine-tuning HURTS a strong trunk and HELPS a weak one; embedding width matters (256 → 7.68 vs full 8.17) |
| Observer Effect `2602.12218` | physics world model; probes on frozen features vs invasive adaptation | PhyIP (linear, frozen) ρ **0.91** | full fine-tune (IBP) ρ **0.05**; last-layer fine-tune 0.65; on a kinematic invariant ρ 0.94 → **−0.03** after fine-tuning | correlation with physical invariants; MAPE | invasive adaptation destroys the geometry that carried Speed / Radius (deep blocks B5–B10 shift most, CKA < 0.2) while Mass survives; adaptation-based evaluation "hallucinates competence" (MAPE 140 % → 18 % by learning the probe task) |
| Latent-WAM `2603.24581` | end-to-end driving WM on a DINOv2 backbone with geometric distillation | **no frozen row** (absence) | Base full fine-tune **89.3**; Small 86.3; Small-LoRA 84.7; Base-LoRA 68.5 (Table 5); distillation INTO the trunk 89.3 vs concatenating frozen geometric features 88.0 vs none 88.3 (Table 4) | NAVSIM v2 EPDMS | shaping the trunk beats attaching frozen features; partial adaptation (LoRA) is worst |
| Point-cloud JEPA WMs `2608.29434` | frozen 137 M PTv3 + random orthogonal projection + trained predictor (Utonia-WM) vs trained tiny PointViT (Point-LeWM / Point-Delta-JEPA) | Utonia-WM 94.0 / 71.2 / **46.6** / 71.2 (Two-Room / Reacher / Push-T / Cube); Vox-WM (frozen encoder ablated, parameter-free geometry) 81.4 / 54.4 / 35.2 / 63.4 | Point-LeWM 87.0 / 80.4 / **83.6** / 66.0; Point-Delta-JEPA 100.0 / 77.6 / 70.8 / 83.4 (Table 7 top row, 10 seeds) | planning success | the frozen trunk wins navigation and loses the task that needs adaptation |
| DINO-world `2507.19468` | encoder ALWAYS frozen; the ablation is at the predictor | predictor from scratch 46.9 / 87.1 / 59.4 | pretrained predictor + action blocks only 49.4 / 91.1 / 61.6; pretrained predictor fine-tuned **59.4 / 93.8 / 68.7** (Table 4) | planning success (3 tasks) | on a frozen trunk the capacity that matters is the predictor's pre-training |
| LeWM `2603.19312` Fig. 6 | trained 5 M ViT-tiny encoder vs frozen-DINOv2 DINO-WM | DINO-WM (bars) ≈ 100 / 79 / 74 / 86 | LeWM ≈ 87 / 86 / 96 / 74 | planning success (Two-Room / Reacher / Push-T / Cube) | split by task; numbers read from the bar chart pre-compaction — **INHERITED, not text-verifiable** |
| Depth-Reg JEPA `2607.16314` | trained 18 M encoder ± depth shaping vs frozen DINOv2 | frozen DINOv2 VO-probe MSE 0.0029; rollout goal-similarity h = 10 **0.951** | LeWM-trained 0.0022 → depth-shaped **0.0015**; rollout 0.903 → 0.890 | probe MSE (↓), similarity (↑) | a shaped tiny encoder beats the frozen giant on the metric probe and loses on rollout similarity (which rewards slow latents — authors' caution) |

### F2. What a frozen DINO trunk needs to work: PATCH tokens and a trained predictor (H-SOTA-E2 → A)

- DINO-WM `2411.04983` Table 2 (Push-T success; Rope chamfer ↓): patch tokens **0.90 / 0.41** vs CLS 0.44 / 0.84, R3M 0.42 / 1.13, ResNet 0.20 / 1.08 — the pooled global feature loses what planning needs.
- What-Drives-Success `2512.24497` Table 7: all encoders frozen, raw patch tokens, no aggregation or entry/exit projection; DINOv2/v3 > V-JEPA/V-JEPA 2 at ViT-L (video encoders need frame duplication); DINOv3 > DINOv2 only on photorealistic DROID / Robocasa; proprioception helps consistently because precise metric quantities are "only implicitly encoded and subject to quantization by the patch architecture"; scaling encoder + predictor helps only on real data (DROID); final recipe: DINOv2 (sim) / DINOv3 (photoreal) frozen, ViT-L predictor depth 12, AdaLN, 2-step rollout loss (Table 2: Ours 83.9 / 78.8 / 70.2 / 58.2 / 41.6 / 25.4 / 30.7 / 48.2 vs DINO-WM 81.6 / 64.1 / 66.0 / 44.8 / 35.1 / 19.1 / 21.7 / 39.4 vs V-JEPA 2-AC — / — / — / — / — / 16.2 / 33.1 / 42.9).
- DINO-world `2507.19468` Table 1 (VSPW mIoU | Cityscapes mIoU | KITTI depth, columns current / short / mid): Copy-Last 52.8 / 47.9 / 42.1 | 68.6 / 53.2 / 39.7 | 2.963 / 3.778 / 4.745; **DINO-world (frozen DINOv2 ViT-B + 1.1 B predictor, 60 M videos)** 52.8 / **51.6 / 47.0** | 68.6 / **64.7 / 55.1** | 2.963 / **3.214 / 4.268**; V-JEPA ViT-L (jointly trained encoder–predictor) 29.1 / 8.2 / 7.7 | 48.8 / 15.5 / 14.0 | 3.502 / 7.217 / 7.491. Table 3 encoder ablation (mid-term CS / VSPW): SD3.5-VAE 13.0 / 1.5, SigLIP2 50.5 / 41.0, DINOv2 53.2 / 46.8. The predictor beats persistence on a physical target with a frozen trunk — the one published L3 clearance at scale (theme 4).
- V-JEPA 2-AC `2506.09985` §3.1: frozen ViT-g, 16×16×1408 feature maps per frame, 300 M predictor; the attentive probe (4 layers) is the standard readout on frozen features for understanding tasks.
- Latent-WAM `2603.24581`: 16 learnable query tokens per view (attention pooling), attention concentrates on lane markings — the readout family for fine driving geometry is token-level attention, not grid pooling (theme 4, E-DEC-25).

### F3. Fine-tuning corrupts what the trunk carries — mechanism and instrument (H-SOTA-E3 → A)

- Observer Effect `2602.12218`: invasive adaptation moves the deep blocks (B5–B10, δ(l) > 0.10, CKA < 0.2), erases time-varying invariants (Speed, Radius) and preserves static ones (Mass); correlation on a kinematic invariant 0.94 → −0.03; the SN-3D invasive probe reports MAPE 18 % where the frozen read says 140 % — competence hallucinated by the probe learning the task. Instrument rule: judge a trunk by LINEAR probes on FROZEN features with a raw-input baseline, time-dependent probes for dynamic variables, OOD splits.
- FROST-Drive corroborates at the driving level (14 B: 8.17 → 8.13, ADE 1.04 → 1.47).
- Prediction for our trained tiny trunk (rdw8p30k): the L2 pattern — `n_agents` +0.3881 up, `lead_range_m` −0.1611 and `d_ego` −0.0399 down (§13.0d, MEASURED) — is "static kept, dynamic/metric erased"; E-DEC-63 (the encoder discards pixel content that predicts its own latent's future) is the same family. Our ridge panel on frozen features complies with the instrument rule.

### F4. Frozen-compatible shaping (H-SOTA-E4 → A)

| route | primary | what is trained on top of the frozen trunk | number |
|---|---|---|---|
| bisimulation adapter | `2602.18639` | 196×384 → 196×32 per-patch MLP + dynamics-similarity loss + PCA-VICReg, trained jointly with the transition model | PointMaze success under LCG shift **0.48 → 0.78** (DINO-WM → ours); under LC: end-to-end without a pretrained trunk 0.26 vs DINOv2 0.86; encoder-agnostic: iBOT 0.72, SimDINOv2 0.40 (Table rows); latent 12× smaller |
| inverse-dynamics objective on frozen features | `2606.07687` | small head; k-step ID (peak k = 4) | action-relevant R²: V-JEPA 2 0.40 → 0.85, **Web-DINO −0.01 → 0.16**, SigLIP2 0.05 → 0.17 — a multiplier on temporal-predictive structure, small on image-DINO features |
| random orthogonal projection, no shaping | `2608.29434` Utonia-WM | projection + predictor | 94.0 / 71.2 / 46.6 / 71.2 |
| planning-time IDM on frozen WM latents | `2607.02403` ACID | IDM only | Le-WM Reacher 76 → 88 (theme 3) |

GS-2's "LDAD inapplicable as published" stands for the trunk and narrows for the adapter: the Δz needed by LDAD can be built from adapter outputs on frozen DINOv3 tokens. The prior for a DINO trunk is WEAK (0.16), which is exactly why the test is discriminating rather than a bet.

### F5. Trained-from-scratch tiny encoders vs frozen giants — where the field's split comes from

LeWM (5 M ViT-tiny), Delta-JEPA (ViT-Tiny), SMWM, Point-LeWM all train the encoder and reach 74–100 % on toy planning; their argument against freezing is speed (LeWM: ∼200× fewer tokens, ∼50× faster planning than DINO-WM) and adaptability (Push-T: Point-LeWM 83.6 vs Utonia-WM 46.6). On REAL data the What-Drives-Success finding is the relevant one: larger frozen encoders and deeper predictors improve DROID consistently while simulation saturates. Our v7-tiny (19 M, INHERITED) on real driving video is on the real-data side of that split.

### F6. What the field's numbers PREDICT for ours — agreement / disagreement

| our MEASURED fact | field | verdict |
|---|---|---|
| freeze → nrmse +14.6 %, degenerate (E-DEC-64) | frozen trunks lack fine metric content (WDS quantisation; Utonia Push-T 46.6) | **AGREE in direction**; caveat: our frozen trunk is a distilled tiny one — equivalence to the field's case UNVERIFIED |
| encoder discards pixel-predictive content (E-DEC-63); L2 target-specific: agents ↑, range / ego ↓ (§13.0d) | Observer Effect: fine-tuning keeps static, erases dynamic | **AGREE** |
| DINOv3 beats V-JEPA 2 9/9 on static decodability (E-DEC-68) | frozen DINOv2/v3 > V-JEPA/2 (WDS); joint V-JEPA fails forecasting (DINO-world 7.7); DINOv3 on photoreal | **AGREE** |
| frozen distilled trunk beats DINOv3 on `n_agents` (+0.3881 vs +0.2754) | shaped tiny encoders beat frozen DINOv2 on a VO probe (0.0015 vs 0.0029) | **AGREE** — target-specific wins are expected |
| no arm action-sensitive, frozen or trained (§13.0c) | no frozen-trunk WM reports a sensitivity metric; ID multiplier on DINO small (0.16); action-conditioning paradox 0.26 → −0.32 | **AGREE**; refav1 predicted insensitive |
| "LDAD has nothing to shape on a frozen trunk" (GS-2) | adapter routes with numbers (`2602.18639`, `2606.07687`) | **PARTIAL DISAGREE** — narrowed to "nothing to shape IN the trunk" |
| REF-A frozen-encoder ceiling 2.14 m, speed R² 0.61 (INHERITED) | FROST-Drive frozen 14 B VLM ADE@3 s 1.04 beats fine-tuned 1.47 | **DISAGREE in kind** — the field's frozen trunks are 14–78 B VLM / ViT-g / ViT-L; REF-A's ceiling is not evidence against freezing a strong trunk |

## 2. Hypothesis outcomes

| id | outcome | evidence | plan change |
|---|---|---|---|
| H-SOTA-E1 | **A** (sign trunk-dependent) | F1 | the v7f trunk decision gets a published prior in BOTH directions: freeze a strong pretrained trunk (refav1); train a weak/tiny one (v7) — E-DEC-64 is consistent with the field, not the only datapoint |
| H-SOTA-E2 | **A, conditional on patch-token readout** | F2 | refav1 baseline role stands; its readout shape must be verified before any number is quoted (no registry row) |
| H-SOTA-E3 | **A** | F3 | a partial / late-unfreeze arm is field-supported for v7f but LoRA-style partial adaptation is the WORST row in Latent-WAM — prefer "frozen + adapter" or "full + distillation" over LoRA |
| H-SOTA-E4 | **A** | F4 | LAB-ENC-1 priced at 0 trunk compute |

## 3. Cheapest discriminating experiment (Q-E5), both outcomes → `PROPOSED_HYPOTHESES.md`

**LAB-ENC-1 (0 trunk compute; probes + a small adapter on banked features):** on refav1's banked fp8 DINOv3 features and on V-JEPA 2 features from the E-DEC-68 rig (INHERITED that both exist), train a 2-layer per-token adapter with the `2606.07687` k-step inverse-dynamics objective (k = 4) on realised ego motion (a, κ; v excluded), then read the linear decodability of (a, κ) from the adapter's Δz with the constant-only control and the raw-pixel floor, split by clip. Outcome A (DINOv3 gain ≥ the published Web-DINO gain of +0.17 R², CI excluding 0): the frozen trunk CAN be shaped through an adapter — GS-2 is narrowed in the register and an adapter-LDAD arm on refav1 is registered. Outcome B (no gain on DINOv3 while V-JEPA 2 gains): DINOv3 is in the Web-DINO regime — refav1's role in the paper is the static-decodability baseline, v7 stays trainable, and the compromise arm is "frozen DINOv3 + trained adapter + trained predictor" only if theme 3's LAB-ACT-1 finds a masked channel.

## 4. Evidence-class ledger

| claim | class | source |
|---|---|---|
| all numbers in F1–F5 | PUBLISHED-PRIMARY | banked PDFs, keys / tables as stated |
| LeWM Fig. 6 bar values | INHERITED (own pre-compaction read of a figure) | `2603.19312` Fig. 6 — not text-verifiable |
| §13.0 / §13.0c / §13.0d numbers; E-DEC-63/64/68 | MEASURED | `MODEL_REGISTRY.md`, `GOALS_AND_CLAIMS.md` (line refs in SPEC §1) |
| refav1 = frozen DINOv3, fp8 features, readout shape | INHERITED / UNVERIFIED | readiness report §D; no registry row |
| our freeze arms froze a distilled TINY trunk | INHERITED | SPEC §1 (`postrain30k_freeze`, `splitp30k`) — the trunk's identity is not re-read here |
| V-JEPA 2 features exist on the E-DEC-68 rig | INHERITED | E-DEC-68 as cited in the readiness report |
| REF-A ceiling 2.14 m / R² 0.61 | INHERITED | `MODEL_REGISTRY.md` §2 not re-read |

## 5. Named empty searches (two probes each)

1. **No primary reports a frozen-vs-fine-tuned row for a LATENT WORLD MODEL's encoder on an action-sensitivity metric.** FROST-Drive is a policy (RFS/ADE); Observer Effect measures decodability; Latent-WAM has no frozen row. Probe 1: full reads (§7). Probe 2: searches #3, #7, #13 (`raw/search_log.md`) surfaced no such table.
2. **No frozen-DINO world model reports a numeric action-sensitivity metric** (DINO-WM, V-JEPA 2-AC, DINO-world, Utonia-WM, What-Drives-Success). Probe 1: full reads. Probe 2: WDS Fig. 2 is the only counterfactual comparison and it is qualitative.
3. **Drive-JEPA (`2601.22032`) frozen-or-trained question**: NOT answered — the paper was not banked in this pass (search #11 confirmed existence only). Open.

## 6. Relay corrections and caveats

- "Utonia-WM" is the frozen-PTv3 world model of `2608.29434`; "Vox-WM" is its parameter-free geometric control (frozen encoder ablated), not a separate method.
- The Latent-WAM LoRA rows (84.7 / 68.5) are Small-LoRA / Base-LoRA; the 89.3 is the Base full fine-tune with geometric distillation.

## 7. Papers banked and read for this theme

| key | short | read | quoted from |
|---|---|---|---|
| `2601.03460` | FROST-Drive | full | Table 3 rows (7.39, 7.79, 8.13 / 1.47, 8.17 / 1.04, 8.17, 8.24; width 256 → 7.68) |
| `2602.12218` | Observer Effect | full | PhyIP / IBP / LL-FT rows, block-shift analysis, MAPE 140 → 18, ρ 0.94 → −0.03 |
| `2411.04983` | DINO-WM | full | Table 2 (patch vs CLS / R3M / ResNet) |
| `2512.24497` | What Drives Success | full (chunks) | Fig. 4, Table 2, Table 7, §5.3, conclusion |
| `2507.19468` | DINO-world | full | Tables 1, 3, 4 |
| `2602.18639` | invariant / bisimulation representations | full | shift table, encoder table |
| `2608.29434` | point-cloud JEPA WMs | full | Table 2 / Table 7 top row, §4, App. D |
| `2603.24581` | Latent-WAM | full | Tables 4, 5, readout description |
| `2606.26217` | Fast-LeWM | full | probe Table 3 (trained encoders) — not quoted numerically |
| `2607.16314` | Depth-Reg JEPA | full | Table 1 |
| `2506.09985` | V-JEPA 2 | §3.1 | frozen ViT-g, attentive probe |
| `2606.07687`, `2607.02403`, `2603.19312`, `2608.24044` | cross-theme | as in their themes | Table 1 / Table 1 / Fig. 6 (INHERITED) / protocol |

## 8. Integration escalations

1. **The v7f trunk decision has a published prior in both directions and E-DEC-64 is not the only datapoint**: freeze a STRONG pretrained trunk (refav1, FROST-Drive-style) and shape it through an adapter; train a tiny trunk fully (v7) — and avoid LoRA-style partial adaptation (worst row in Latent-WAM).
2. **Verify refav1's readout consumes DINOv3 PATCH tokens** before its first number is quoted; a pooled/CLS readout would put it in DINO-WM's 0.44 regime.
3. **Run LAB-ENC-1 (0 trunk compute) before any refav1 shaping arm** — it decides whether DINOv3 features have the temporal-predictive structure an inverse-dynamics adapter can multiply.
4. **Judge trunks only by linear probes on frozen features** (Observer Effect); never by a fine-tuned readout — our ridge panel already complies; keep it that way in the v7f prereg.
