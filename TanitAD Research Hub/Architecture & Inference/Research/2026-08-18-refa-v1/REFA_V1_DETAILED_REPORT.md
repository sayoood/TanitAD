# REF-A v1 — DETAILED DESIGN REPORT

Companion to `REFA_V1_DESIGN.md` (the decision record). This document is the **engineering
detail**: every component with its measured size, the tensor path end to end, the compute and
storage bill, the DINOv3 licence position, and the two things that are **deliberately not in v1**.

All figures `MEASURED` on 2026-08-18 unless marked otherwise.

---

## 1. COMPONENT SIZES — measured, not estimated

Launch config: `RefAV1Config(strategic_cfg=StrategicPolicyConfig(), tactical_cfg=TacticalPolicyConfig())`.

| component | params | share | what it is |
|---|---:|---:|---|
| **OPERATIVE predictor** | **80,043,008** | 45.99 % | 6 causal blocks, d=1024, over 640 patch tokens |
| ├─ `.blocks` (6×) | 75,577,344 | 43.42 % | the world model proper |
| ├─ `.mix` (2d→d) | 2,098,176 | 1.21 % | action⊕field fusion (DINO-WM's scheme) |
| ├─ `.act` (action MLP) | 1,052,672 | 0.60 % | (a, κ) → d |
| ├─ `.head` | 1,051,648 | 0.60 % | residual delta |
| └─ `.intent` proj | 263,168 | 0.15 % | ⭐ the hierarchy's closing link |
| **TACTICAL field predictor** | **54,850,560** | 31.52 % | 4 blocks over 64 query tokens, Δ0.6 s |
| └─ + `tac_pool` + queries | 4,263,936 | 2.45 % | cross-attention 640 → 64 |
| **TACTICAL policy brain** | **21,684,493** | 12.46 % | manoeuvre + 2 s goal + intent token |
| **STRATEGIC policy brain** | **7,990,275** | 4.59 % | route logits + ctx token |
| **WideAdapter** | **2,762,752** | 1.59 % | 640×1024 → 640×1024, no compression |
| ├─ `.proj` | 2,101,248 | 1.21 % | per-token MLP (shared across tokens) |
| ├─ `.pos` | 655,360 | 0.38 % | learned spatial embedding, 640×1024 |
| └─ `.tmix` | 4,096 | 0.00 % | depthwise temporal mix |
| **STRATEGIC subspace predictor** | **1,911,040** | 1.10 % | own predictor, 256-d strategy space |
| **proposal head** (auxiliary) | **537,108** | 0.31 % | seeds the planner; never in the loss |
| `FeatureStandardizer` | 0 | — | frozen buffers, not parameters |
| **TRAINABLE TOTAL** | **174,043,172** | 100 % | **sub-300M ✅** |
| frozen encoder | **0 in graph** | — | DINOv3 ViT-L/16 (~300 M) lives on the disk side |

Without the brains (`RefAV1Config()` bare — the change-#7 ablation arm): **143,842,068**.

**Where the capacity actually went.** 77.5 % of trainable parameters are the two *field predictors*
— i.e. **the world model**, not heads. In REF-A the comparable share sat behind a compact-state
adapter feeding a supervised head. This is the parameter-level expression of the redesign.

## 2. THE TENSOR PATH, END TO END

```
stage 1 (offline, no gradients, separate job)
  frames 256×640 @120° ──DINOv3 ViT-L/16, patch 16──▶ [T, 640, 1024] fp16  (CLS discarded)

stage 2 (this trainer)
  feats            [B, 4, 640, 1024]      observed window W=4
    │ FeatureStandardizer (fit once; refit raises)
    │ WideAdapter  (+pos, +depthwise temporal mix)
    ▼
  field            [B, 4, 640, 1024]
    ├─ last        [B, 640, 1024]                     ← the operative state
    ├─ pooled_win  [B, 4, 1024]  ──▶ StrategicPolicy ──ctx──▶ TacticalPolicy ──intent──┐
    │                                                                                  │
    ├─ OPERATIVE  rollout  Δ0.2 s × 30 = 6.0 s   [B, 30, 640, 1024]  ◀─── intent ──────┤
    ├─ TACTICAL   pool 640→64, rollout Δ0.6 s × 10 = 6.0 s  [B, 10, 64, 1024] ◀────────┤
    └─ STRATEGIC  subspace 1024→256, rollout Δ1.5 s × 4 = 6.0 s  [B, 4, 256]           │
                                                                                       │
  loss = 1.0·MSE(op, tgt) + 0.5·MSE(tac, tgt_q) + 0.25·MSE(str, tgt_s)   ← NO trajectory label

deployment (no gradients)
  search on the TACTICAL field (64 tokens) with iCEM over (a, κ)
      candidates ⊕ {cv, hold_v0, decel_1.5, proposal}   ← the floor
  ▶ re-score winner + baselines on the OPERATIVE field (640 tokens)
  ▶ execute first action, re-encode, replan (receding horizon)
```

## 3. COMPUTE AND STORAGE BILL

### 3.1 Planning (measured, RTX 4060, real geometry)

| quantity | value |
|---|---:|
| one 10-step rollout, 640×1024 field | **160 ms / candidate** |
| scaling at n = 16 / 32 / 64 | 2553 / 5129 / 10519 ms (**linear — saturated at n=16**) |
| DINO-WM config (300×30) on the **operative** field | **325 s / MPC tick** |
| DINO-WM config (300×30) on the **tactical** field ⭐ | **21.5 s / MPC tick** (~3.5 s on A40-class) |
| peak CUDA, coarse-to-fine, chunk 64 | **0.95 GB** |
| naive rollout store (300 × 10 fields) — **avoided** | 3.93 GB |

### 3.2 ⛔ THE STAGE-1 CACHE IS THE REAL LOGISTICS BILL, AND IT IS NEW

| cache | size |
|---|---:|
| one latent field, fp16 | 1.31 MB |
| one episode (T ≈ 200) | **262 MB** |
| **2,376-episode corpus, fp16** | ⛔ **~0.62 TB** |
| same at fp8/int8 | ~0.31 TB |
| *REF-A's old 256-token / 768-d cache, for scale* | ~0.19 TB |

⇒ **v1's cache is ~3.3× REF-A's.** That is the direct price of change #2 (640 vs 256 tokens) and
change #1 (1024 vs 768 d), and it is a **fleet decision, not a model decision**. Three options,
with their trade:

| option | storage | training cost | note |
|---|---:|---|---|
| **A. fp16 cache** (as designed) | 0.62 TB | fastest steps | needs the disk; MooseFS quotas have bitten twice |
| **B. fp8/int8 cache** | 0.31 TB | ~same | needs a quantisation error check before it is admissible |
| **C. encode on the fly** | **0** | encoder forward every step, frozen/`no_grad` | removes the cache entirely; costs GPU time per step and puts the encoder back in the loop (though still not in the graph) |

**Recommendation: B, with A as fallback** — and measure the quantisation error against the fp16
reference on one episode before committing the corpus. Option C is the only one that survives a
disk shortage, and it should be benchmarked before the cache job starts, not after.

## 4. DINOv3 — THE LICENCE POSITION, AND WHY THE ARM IS NOT HOSTAGE TO IT

**Probed directly (HF API, 2026-08-18):**

| repo | gated | licence |
|---|---|---|
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | ⛔ **`manual`** | `other` (custom DINOv3 licence) |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | ⛔ **`manual`** | `other` |
| `facebook/dinov2-base` | `False` | **`apache-2.0`** |

`manual` = a **human at Meta approves each request**; it cannot be auto-accepted, and it cannot be
accepted by an agent on the PI's behalf — it is a legal agreement.

**Licence terms that actually bind us** (from `facebookresearch/dinov3/LICENSE.md`):

* ✅ Commercial use permitted, royalty-free, worldwide, non-exclusive.
* ✅ **No copyleft** — *"you are and will be the owner of such derivative works and modifications"*.
  Our trained checkpoints are ours.
* ⚠️ **Redistribution obligation** — *"you shall provide a copy of this Agreement with any such DINO
  Materials"*. ⇒ every HF push of a REF-A v1 checkpoint must ship the DINOv3 licence.
* ⚠️ **Publication obligation** — *"you must acknowledge the use of DINO Materials in your
  publication"*. ⇒ a line in `TANITAD_PAPER.md`.
* ⚠️ Trade-control compliance; prohibited fields include military/warfare, nuclear, espionage,
  weapons. **Autonomous driving is not restricted.**
* ⚠️ Indemnity clause runs against us; warranty disclaimer is absolute.

### ⭐ THE FALLBACK THAT MAKES THIS A PREFERENCE, NOT A BLOCKER

**DINOv2-L/14 at 224×560 yields 16×40 = 640 tokens at d = 1024 — the identical interface.**

| | DINOv3 ViT-L/16 | DINOv2-L/14 (fallback) |
|---|---|---|
| input | 256×640 | 224×560 |
| patch | 16 | 14 |
| grid | 16×40 | 16×40 |
| **tokens** | **640** | **640** |
| **d_enc** | **1024** | **1024** |
| HFOV | 120° | 120° |
| licence | custom, gated `manual` | **Apache-2.0, ungated** |

⇒ Every one of the nine v1 changes **except #1** is preserved on the fallback. The arm can start on
DINOv2-L/14 with zero licence exposure and swap encoders later — the cache contract is the same
shape, so only the cache is rebuilt, not the model.

**Recommendation:** request DINOv3 access now (free, and it is the better encoder), but **build the
stage-1 cache against whichever encoder is available on the day**, defaulting to DINOv2-L/14. Do not
let a manual approval queue sit on the critical path of a 5 k-step arm.

## 5. ⛔ WHAT IS DELIBERATELY *NOT* IN v1

### 5.1 The DUAL ENCODER is **NOT implemented** — and it does not belong here

The literature review recommends a dual encoder (frozen anchor ‖ trainable branch; measured
35.03 → 55.55 → 78.46 with tokenizer + co-training) as the best *fine-tuning* form. **v1 does not
have it, and adding it would break v1's premise**, for a concrete reason:

> v1 trains from **cached feature tensors on disk**. No encoder exists in the graph — that is REF-A
> stability item 2, it is what makes the arm cheap, and it is pinned by a test asserting zero
> encoder parameters. A trainable branch requires a **live encoder over images**, which changes the
> data path, the memory profile, and the cost class entirely.

⇒ The dual encoder is the natural body of **`E-XENC-1F`** (the registered trainable-pretrained-encoder
arm), and the literature says it should be built there as *dual*, not as full fine-tuning. Sketch,
for when that arm is funded:

```
images ─┬─ φ_frozen (DINOv3, no_grad)  ─┐
        └─ φ_train  (DINOv3 init, LoRA/full) ─┴─ concat [B,N,2048] ─▶ WideAdapter(d_enc=2048)
```
`RefAV1Config` already refuses `d_state < d_enc`, so the concatenated 2048-d interface would force
`d_state ≥ 2048` — the no-bottleneck rule composes correctly with the dual encoder without a change.

### 5.2 Not in v1 either
* **Diffusion decoder** — REF-C's territory; v1's behaviour comes from search, not sampling.
* **Lane topology / map** — `PREPARE_LANE_CHANGE` remains blocked on 2 of 4 inputs.
* **T1 evaluation harness changes** — v1 reuses the existing one; a driving claim needs T1.

## 6. RISK REGISTER

| risk | severity | mitigation in place |
|---|---|---|
| stage-1 cache is 0.62 TB | **high** | §3.2 options A/B/C, decide before the job starts |
| DINOv3 manual approval delay | medium | DINOv2-L/14 fallback, identical interface (§4) |
| the cost model is miscalibrated (C101 repeat) | **high** | `cost_fidelity` gate G1 blocks quoting any planner number |
| coarse tactical search misranks | medium | fine re-score + `coarse_fine_agree` reported per window (G4) |
| nine simultaneous changes → unattributable win | **high** | declared confound; five one-flag ablations; E-RECON-2 kept as the attribution arm |
| adapter collapse | medium | per-dim std logged every interval; gate G5 |
