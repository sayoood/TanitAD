# REF-C origins & successors — DiffusionDrive, its measured successor line, and what transfers to refcv3

**Date:** 2026-08-29 · **Stream:** Research Lab special topic (Architecture & Inference)
**PI ask, verbatim intent:** *"extract again the performance of the original implementation (reference paper) of refc which was based on resnet and diffusion planner; … outlook works mentioned by the authors, … successor works improving the end2end approach based on diffusion planner, or any better alternatives than resnet etc."*
**Protocol discipline:** every score row carries its protocol per `NAVSIM_PROTOCOL.md` §6 — **P1** = NAVSIM v1 `navtest` PDMS · **P4** = "EPDMS on navtest", single-stage (the protocol the NavSim authors *discourage in print*; ~35 pts above navhard's scale) · P1/P2/P3/P4 are **mutually incomparable — never merge**.
**Evidence classes:** `PUBLISHED-PRIMARY (banked)` = read from a Library-banked PDF whose sha256 I re-verified by content this session · `PUBLISHED-PRIMARY (bank-pending)` = read from a primary downloaded this session, kb_add queued (G: outage) · `INHERITED` = another repo doc / another paper's comparison table, not re-verified against that method's own paper · `MEASURED (ours)` = registry/design-doc artifacts.

---

## 0. Findings first

1. **The reference paper is DiffusionDrive, arXiv 2411.15139 (CVPR 2025), and this is settled by the repo, not assumption**: `stack/tanitad/refs/refc.py:1-8` — REF-C *was* TCP-C (arXiv 2206.08129, two-branch GRU) and the current REF-C "REPLACES the GRU trajectory + control branches with an ANCHORED TRUNCATED-DIFFUSION trajectory decoder in the DiffusionDrive spirit (arXiv 2411.15139)". Library entry 2411.15139 (banked 2026-08-21, sha `6ad4f8a37949`, re-verified today): *"THE paper REF-C is derived from (D-030)."*
2. **Original headline (P1, banked primary, Tab. 1/2):** **88.1 PDMS** on navtest — ResNet-34, camera+LiDAR, 20 k-means anchors, **2 denoising steps**, 60 M params, 45 FPS (RTX 4090). Sub-scores NC 98.2 / DAC 96.2 / TTC 94.7 / Comf 100 / EP 82.2.
3. **The authors published no future-work/limitations section** (v3 PDF read 100 %, 15/15 pages — conclusion is summary-only). Their *own successor* supplies the limitation statement instead (§2.3).
4. **The direct successor is DiffusionDriveV2 (arXiv 2512.07745, Dec 2025, same senior authors): 91.2 PDMS (P1) at the same ResNet-34** — +3.1 over the original, EP +5.3 — via RL post-training (intra-anchor GRPO + inter-anchor truncated advantage + multiplicative exploration noise) from a DiffusionDrive cold start, 2 denoise steps kept. It **beats GoalFlow and Hydra-MDP running V2-99 (96.9 M backbone) with a 21.8 M backbone**.
5. **Backbone ladders are real but second-order vs the policy/selection mechanism** at this frontier: DriveSuprim same-paper ladder R34→V2-99→ViT-L = 89.9→92.1→93.5 (P1); but *at fixed R34* the method spread is larger: GoalFlow 85.7 < DD 88.1 < WoTE/DIVER 88.3 < DriveSuprim 89.9 < **DDv2 91.2**. Strongest published non-ResNet datapoint: Drive-JEPA (V-JEPA ViT-L, 307 M) **93.7 PDMS** (P1).
6. **The successors independently converged on our own measured story about selection**: DDv2's stated motivation is that IL trains only the positive mode, so the raw fan is full of unconstrained colliding candidates and the system "forces reliance on a downstream selector, which is often less robust than the generator … prone to failure … particularly in out-of-distribution scenarios" — quantified: DD's raw-fan PDMS falls 93.5→75.3 from top-1 to top-10; DDv2 holds 84.4. This is the same defect family our SEL-1 winner's-curse refusal measured on our own fan.

---

## 1. Identity of the reference paper (repo-derived)

| repo source | statement |
|---|---|
| `stack/tanitad/refs/refc.py` docstring (lines 1–8) | TCP-C (arXiv **2206.08129**) → anchored truncated-diffusion decoder "in the DiffusionDrive spirit (arXiv **2411.15139**)" |
| `TanitAD Research Lab/Library/LIBRARY.md` entry 2411.15139 | *"THE paper REF-C is derived from (D-030 …). Was UNBANKED until 2026-08-21 despite being REF-C's origin."* Banked, sha `6ad4f8a37949`, cited by `REFCV3_REVIEW.md` |
| `REFC_V3_DESIGN.md` + `REFCV3_REVIEW.md` §3 | v3 keeps the paper's two load-bearing mechanisms (truncated diffusion @2 steps, anchored priors) and replaces the two our measurements refuted (5-way softmax, learned re-scorer) |

`REFCV3_REVIEW.md` §7 states its DiffusionDrive comparison was "from the abstract, figures and headline numbers, not a full read" — this document is the full read.

## 2. The original, from the banked primary (2411.15139v3, CVPR 2025; Liao, Chen, … , X. Wang — HUST + Horizon Robotics)

### 2.1 Architecture facts
- **Truncated diffusion policy:** diffusion schedule truncated to **50/1000** around **20 k-means anchors** (an *anchored Gaussian* prior, not VADv2's 8192-anchor vocabulary — a 400× reduction); inference = **2 DDIM steps** from the anchored prior; Ninfer is free at inference (their "inference flexibility").
- **Cascade diffusion decoder** (×2 layers, params shared across denoise steps): deformable spatial cross-attn on **BEV features** → agent cross-attn → FFN → timestep modulation → MLP → per-anchor confidence + coordinate offset; **top-1 confidence is the emitted trajectory** (a learned, candidate-conditioned selector).
- **Backbone/inputs (NAVSIM):** ResNet-34 (Transfuser-aligned, ImageNet init), 3 front cameras concat 1024×256 + rasterized LiDAR BEV; **ego status is consumed** via the Transfuser recipe (NavSim agents receive driving command + velocity + acceleration; `NAVSIM_PROTOCOL.md` §7 caveat 3: no published fully ego-free NavSim number exists). Aux perception: 3D det + BEV semseg. 100 epochs navtrain, AdamW 6e-4, batch 512, 8×4090.
- **Cost:** 60 M total params; plan module 7.6 ms (2×3.8 ms); **45 FPS** end-to-end on a 4090.

### 2.2 Performance (all PUBLISHED-PRIMARY, banked, sha-verified)
| protocol | result | context rows (same table) |
|---|---|---|
| **P1** navtest PDMS (Tab. 1) | **88.1** (NC 98.2/DAC 96.2/TTC 94.7/Comf 100/EP 82.2) | Transfuser 84.0 · UniAD 83.4 · PARA-Drive 84.0 · DRAMA 85.5 · VADv2-V8192 80.9 · Hydra-MDP-V8192 83.0 · Hydra-MDP-V8192-W-EP 86.5 (post-processing + EP-fitting; DD +1.6 without either) |
| Roadmap (Tab. 2) | Transfuser 84.0 → +vanilla 20-step diffusion policy (TransfuserDP) 84.6 @7 FPS, diversity D=11 % → +truncation (TransfuserTD) 85.7 @27 FPS, D=70 % → +cascade decoder = **88.1 @45 FPS, D=74 %**, 60 M vs 101 M | the truncation is worth +1.1 PDMS *and* 10× fewer steps; the decoder +2.4 PDMS and −39 % params |
| Ablations (Tab. 4/5/6) | steps 1/2/3 → 87.9/**88.1**/88.1 (⚠️ Tab. 4's step-3 EP "92.2" is a paper typo) · cascade stages 1/2/4 → 87.4/**88.1**/88.2 · Ninfer 10/20/40 → 84.9/**88.1**/88.2 | everything saturates at the shipped config — the recipe is at its own knee |
| nuScenes open-loop (Tab. 7, ST-P3 metrics, on SparseDrive, R50, 18 anchors) | L2 avg **0.57 m**, collision 0.08 %, 8.2 FPS | vs VAD 0.72/0.22 (1.8× faster) · SparseDrive 0.61/0.08 |
| Suppl. Tab. 8 — **prior comparison** | anchored prior **88.1** vs kinematic-extrapolation prior: 81.3 (swap at inference) / 84.7 (trained on it) | ⭐ direct published support for anchors-over-extrapolation — relevant to our kinematic-comparator prereg rider |
| Suppl. Tab. 9 — anchor-source generalization | CARLA Longest6 DS **64.27 ± 2.43** (NAVSIM-clustered anchors) vs Transfuser 47.30 ± 5.72 | the anchor vocabulary transfers across corpora |

### 2.3 The authors' outlook — what exists and what does not
**MEASURED-ABSENCE (full read):** the paper has **no limitations / future-work section**; the conclusion only summarizes. The closest authorial forward statements: (a) *inference flexibility* — Ninfer adjustable to compute at deployment; (b) truncated diffusion "introduc[es] concepts that have not yet been explored in the robotics field" (§2); (c) the anchored-prior and anchor-generalization supplementary analyses (§2.2 above).
**The de-facto outlook is stated by the same group in DiffusionDriveV2 (bank-pending primary), as the original's named defects:** mode-level supervision gap — IL "is simplified in practice to optimizing only the parameters of the single positive mode … neglects to impose any explicit constraints on trajectories sampled from the negative modes, which constitute the vast majority" → "high-quality trajectories alongside a multitude of unconstrained, low-quality, and often colliding ones"; and **selector over-reliance** — "a downstream selector … often less robust than the generator due to much fewer parameters … prone to failure … particularly in out-of-distribution scenarios."

## 3. Successors and alternatives (per-row protocol; camera/LiDAR and ego usage per the NavSim agent contract unless noted)

### 3.a Direct successors of the diffusion-planner e2e line
| work | mechanism added on top of DD | P1 PDMS | P4 EPDMS | class |
|---|---|---|---|---|
| **DiffusionDriveV2** (2512.07745, R34 21.8 M backbone, C&L) | RL post-train from DD cold start: scale-adaptive **multiplicative** exploration noise (beats additive 90.1 vs 89.7 in their ablation) + **intra-anchor GRPO** (advantage within each anchor's group; +0.9) + **inter-anchor truncated advantage** (negatives→0, collision→−1; +0.6) + IL regularizer; DriveSuprim-style two-stage selector w/ margin-rank loss; 2 steps kept; 10 RL epochs, 8×L20 | **91.2** (EP 87.5) | **85.5** | PUBLISHED-PRIMARY (bank-pending) |
| DIVER (2507.04049) | "reinforced diffusion" — the other RL-on-diffusion successor (Liao co-authors) | 88.3 | — | INHERITED (DDv2 Tab. 1 "official same-backbone scores"); bank queued |
| ARTEMIS (2504.19580) | autoregressive planning + mixture-of-experts (paradigm alternative, R34) | 87.0 | 83.1 | INHERITED (DDv2 Tab. 1/2); bank queued |
| TrajHF (2503.10434, banked) | RLHF fine-tuning of a generative trajectory planner (preference alignment) | ~94 rows exist **only with a PDMS-aligned selector** — metric-informed selection; not comparable to selector-free rows | — | PUBLISHED-PRIMARY (banked); number NOT quotable as deployable |
| Raw-fan quality (DDv2 Tab. 3, 20 trajs, pre-selector) | DD: diversity 42.3, PDMS@1/5/10 = 93.5/84.3/**75.3** → DDv2: 30.3, 94.9/91.1/**84.4** (TransfuserTD: 0.1, flat 85.7) | | | the floor-raising is the successor's core measured claim |

### 3.b Backbone alternatives to ResNet (the encoder ladder)
| ladder (same paper, same method) | R34 | V2-99 (96.9 M) | ViT-L | class |
|---|---|---|---|---|
| DriveSuprim (2506.06659v3 Tab. 2, P1) | 89.9 | 92.1 | **93.5** | PUBLISHED-PRIMARY (banked, sha-verified) |
| DriveSuprim (P4, Tab. 3) | 83.1 | 86.0 | 87.1 | same |
| Hydra-MDP -A/-B/-C (2406.06978v4 Tab. 2, P1) | (83.0 base) | 90.3 / 91.0 | 89.9 (A) | PUBLISHED-PRIMARY (banked) via `NAVSIM_PROTOCOL.md` §6.1 |
| Drive-JEPA (2601.22032, banked): **V-JEPA ViT-L 307 M**, distilled multimodal-trajectory planner | — | — | **93.7 P1 / 87.8 P4** (abstract; ⚠️ the library note says 93.3 — that is a table-row value; reconcile at next library pass) | PUBLISHED-PRIMARY (banked, sha-verified) |
| ⚠️ note on the `refc.py:100-102` docstring claim "Hydra-MDP went 86.6→91.0 purely by swapping ResNet-34→V2-99" | 86.6 is **Hydra-MDP++** R34 *camera-only* (2503.12820v1 Tab. 1); 91.0 is **Hydra-MDP-C** V2-99 (2406.06978v4 Tab. 2) — two papers, two configs; the clean same-paper ladder is DriveSuprim's +2.2/+3.6 | | | correction, minor |

### 3.c Planner-head alternatives (does anything beat truncated diffusion at matched backbone?)
| head paradigm | representative (banked) | P1 @R34 | note |
|---|---|---|---|
| **truncated diffusion + RL** | DiffusionDriveV2 | **91.2** | the current best at R34; generation-side fix |
| vocabulary scoring + coarse-to-fine selection + rotation aug + self-distillation | DriveSuprim (2506.06659) | **89.9** | beats plain DD **without diffusion** — a selection-side fix; its motivation (hard negatives, directional imbalance) is our SEL family's problem statement |
| world-model candidate evaluation (BEV WM scores futures) | WoTE (2504.01941) | 88.3 | +0.2 over DD; the literature's imagination-based-selection counterpoint to our C101 CEM refusal — gain exists but is small |
| flow matching + goal-point vocabulary | GoalFlow (2503.05689v6) | **85.7** @R34 (90.3 with V2-99; **92.1 only with GT-endpoint goal — privileged, never quote as deployable**) | flow matching alone does **not** beat truncated diffusion at matched backbone; the *goal* carries the headline |
| vocabulary + expert distillation (camera-only) | Hydra-MDP++ (2503.12820) | 86.6 | EPDMS sub-metric origin (TLC/LK) |
| flow-matching world-action model | SimWAM (2608.07468, banked) | steps 1→68.9 / 5→90.1 / 10→90.3 / 20→90.2 (INHERITED from `REFCV3_REVIEW.md` §5; my re-download sha-mismatched the banked copy — arXiv re-render or version bump; verify against the banked bytes before quoting further) | step-count behavior differs from truncated diffusion (1-step collapses there; DD's 1-step holds 87.9) |

## 4. What this changes for TanitAD (≤4 recommendations)

Our refcv3 delta vs the original (from `REFC_V3_DESIGN.md`/`refc_v3.py`, MEASURED): kept truncated-diffusion@2 + anchored priors (FPS not k-means, 128 anchors); replaced the learned selector with the candidate-independent `GoalDistanceScorer` behind a σ≤0.8 m admission gate; added the strategic→tactical→operative goal cascade (+2.05 M params on a 60.9 M core); factored lat×lon decision; 6 s horizon; vision-only goal heads (E11).

1. **RL post-training on the fan is the highest-evidence successor lever and it grafts onto refcv3 without touching the registered dominance pair.** DDv2's +3.1 PDMS / raw-floor 75.3→84.4 comes from a 10-epoch RL post-train of a *frozen-recipe* DD — i.e., a *phase*, not an architecture change. For us: after the v3-H/v3-F read, a v3.2 prereg candidate = intra-anchor GRPO on the 128-anchor fan with a reward from our own instruments (reach-clamp violation, collision/lead TTC from the val40 lead block, progress) — the collision→−1 / negatives→0 truncation is directly portable. ⚠️ Their reward is NavSim's privileged simulator; ours must be declared and leak-checked (reward reads labels at TRAIN only — same direction as "labels may use ego").
2. **The literature has converged on our selection finding — and names the defect our GoalDistanceScorer avoids; the cheap novel experiment is ours to run.** DDv2 (generation-side) and DriveSuprim (selection-side hardening) both respond to selector fragility; nobody has published a *candidate-independent* selection reference. On our banked REF-C-XL fans, goal-distance-vs-coarse-to-fine is a 0-GPU comparison (fans + windows already banked) and either outcome is a claimable result. Note our SEL-1 refusal was measured on OUR fan/estimator — DriveSuprim's BCE+rank coarse-to-fine at R34 89.9 says a *hardened* learned selector can work in-distribution at scale; the honest frame is "two mechanisms, one measured comparison missing".
3. **Encoder scale (D-008 tension): the ladder is real (+2.2 V2-99, +3.6 ViT-L on DriveSuprim) but the R34-class frontier moved +3.1 by policy quality alone — spend the next GPU-day on the policy/selection mechanism, not the trunk.** This corroborates our measured "small fan ≈ base fan" sizing and weakens "scale the encoder first" as the default reading of D-008. If/when the D-008 rung is paid, the published frontier direction is a pretrained ViT/V-JEPA-class trunk (Drive-JEPA 93.7 P1 at 307 M) — which intersects our frozen-encoder/FROST-Drive review rather than the ResNet ladder.
4. **Goal-conditioned *generation* is the one goal lever the successors measured that refcv3 does not have.** GoalFlow's headline lives on its goal (deployable 90.3 w/ V2-99; GT-goal 92.1 = the ceiling-vs-goal-quality argument — exactly our σ-admission curve, in the literature's costume), while its flow-matching head *loses* to truncated diffusion at matched backbone (85.7 vs 88.1). Ours conditions *selection* on ĝ_tac (E9) and the decoder only via the tactical latent (E7); feeding the predicted goal into the *decoder condition* (anchored-prior modulation, GoalFlow-style) is untested-in-ours, is compatible with E12 (goal stays predicted, vision-only), and is the natural v3.1+ lever if the goal head clears admission. Keep flow matching off the roadmap on current evidence.

**Untested in ours vs successors' measured gains (explicit):** RL post-train (their +3.1) — nothing measured ours; goal→generation conditioning (their +4.6 on GoalFlow's own goal-ablation family, deployable-form) — ours selection-only; coarse-to-fine learned selection (DriveSuprim's margin over DD +1.8) — REFUSED in our SEL-1 form, not re-measured in theirs; 2-step choice — corroborated (their 1/2/3 flat), no action.

## 5. Provenance & banking

**Banked + sha-re-verified by content this session (local copy = banked bytes):** 2411.15139 (`6ad4f8a37949`) · 2506.06659 (`33cf828e4a5c`) · 2503.05689 (`4206cc1b3aa4`) · 2503.12820 (`e70fea2dddbf`) · 2406.06978 (`8927b0ffbfb0`) · 2504.01941 (`04569f633b3c`) · 2503.10434 (`eb18c94e3eaf`) · 2601.22032 (`d88c053a67ac`).
**Bank queued (G: outage — commands in the session report):** 2512.07745 (DiffusionDriveV2, local sha `076ce47e0b0a`) · 2507.04049 (DIVER) · 2504.19580 (ARTEMIS) · 2206.08129 (TCP — REF-C's pre-diffusion base, cited here, not yet in the Library index).
**Sha-mismatch, not re-verified:** 2608.07468 (SimWAM) — both current arXiv and v1 differ from the banked `03ce230505c4` (arXiv re-render or version bump); its numbers above stay INHERITED until read from the banked bytes.
**Repo docs consumed (via Drive API during the mount outage):** `REFC_V3_DESIGN.md` · `REFCV3_REVIEW.md` · `NAVSIM_PROTOCOL.md` (§6 protocol families, §7 ego-shortcut) · `LIBRARY.md` (2026-08-29 state).
