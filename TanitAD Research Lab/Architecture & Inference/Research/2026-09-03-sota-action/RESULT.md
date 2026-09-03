<title>RESULT - SOTA pass: prediction / action sensitivity (theme 3 of 4)</title>

# RESULT - E-ARCH-SOTA-ACT-1: action-conditioning collapse is named, measured and partially solved in the 2026 literature; the cheapest arm for us costs 0 GPU

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-action/` · Research Lab overnight pass 2026-09-03 · literature only · 0 GPU · no pod/Thor contact
**Spec** `SPEC.md` (staged before any primary was opened; hypotheses H-SOTA-A1…A4 with both outcomes committed)
**Evidence classes used** PUBLISHED-PRIMARY (a banked PDF read in full, `lib:<key>`), MEASURED (registry / register line), INHERITED (our own summaries, not re-verified here). Nothing PUBLISHED-SECONDARY enters this file.
**Register rows proposed** `PROPOSED_HYPOTHESES.md` → LAB-ACT-1 … LAB-ACT-4 (the Master Mind applies them; this package does not edit `GOALS_AND_CLAIMS.md`).

## 0. Verdict in five lines

1. Our failure has a published name — **"Context Collapse"** (ActSWM, lib `2607.26712`): rollouts that predict well and ignore the action. The field's diagnostics on LeWM-family models read a zero-action gap of **−0.005 … 0.002** (ActSWM Table 5) and an action-separation of **0.002** (Dueling WM, lib `2608.06706`) — our h=1 ratio of **0.00408–0.00595** (MM-E10) is that regime, not an outlier.
2. Every one of our interventions was a PREDICTION lever (k, O14, EMA, τ-ramp); the field shows prediction levers leave the gap at ≈0 (ActSWM Table 5: cosine 0.239 → 0.972 with H 3→32 and multi-step training, gap stays ≤ 0.002). **Only an explicit sensitivity term moves it** (gap 0.002 → 0.760).
3. The published fixes **partition** as pre-registered (H-SOTA-A2 supported): encoder-shaping (LDAD, SMWM, EB-JEPA's IDM) needs a trainable trunk; ACID, AD-JEPA anchoring and ActSWM's hinge act on the predictor/planner and survive a frozen trunk. GS-2 is confirmed from the Delta-JEPA PDF, not from a figure.
4. **The cheapest discriminating experiment is 0 GPU**: AD-JEPA's action-mean centering works POST-HOC on frozen models (raw action probe −0.002…+0.052 → centred 0.19–0.52; trained-in vs post-hoc parity on DMC). Reading it on the 32 banked census arms tells us whether the FiLM channel is dead or masked before any arm is trained (LAB-ACT-1).
5. Checkpoint selection by validation loss picks action-collapsed checkpoints **17/36** times (Dueling WM) — our nrmse-selected T1 arms are exposed to the same error (LAB-ACT-3, 0 GPU).

## 1. Findings (findings first; numbers are PUBLISHED-PRIMARY unless stamped otherwise)

### F1. The field measures action sensitivity directly, before and after its fixes (H-SOTA-A1 → outcome A)

| primary | metric (definition) | number BEFORE fix | number AFTER fix | where |
|---|---|---|---|---|
| ActSWM `2607.26712` | cosine of K-step rollout to truth under recorded actions vs under zero actions; **gap** = GT − Zero at step 31 | LeWM H=3: 0.239 / 0.244 / **−0.005**; LeWM H=32: 0.816 / 0.815 / **0.001**; H=3 256 px + rollout: 0.810 / 0.810 / **0.000**; H=32 + rollout: 0.972 / 0.970 / **0.002** | + joint (frozen random) action readout: 0.698 / 0.106 / **0.592**; ActSWM (hinge + readout): 0.923 / 0.163 / **0.760** | Table 5 |
| Delta-JEPA `2606.31232` | displacement of the prediction from its zero-action counterpart, 512 histories | LeWM: means concentrated at the origin | Delta-JEPA: spread | Fig. 6 |
| Delta-JEPA | linear decodability of Δx from Δz (MSE / r), Two-Room | PLDM 0.355 / 0.813; Sub-JEPA 0.601 / 0.674; LeWM 0.444 / 0.765 | Delta-JEPA **0.016 / 0.992** (MLP 0.005 / 0.997) | Table 5 |
| ATM `2606.09028` | action-consistency transfer matrix D_TT / D_TP / D_PT / D_PP (2-layer MLP action probes over true and predicted transitions) | ρ(−L_pred, success) **0.4983** vs ρ(−D_TT, success) **0.8130**, S(2) 0.8239 (Cube) | AITS training: LeWM 87 / 96 / 74 → 92 / 100 / 84 (Table 1); Cube D_TT 36.65 → 15.87 (Table 6) | Tables 2, 1, 6; ranking 82.84 % of pairs, 98.81 % at > 5 % margin (Table 3); 3–5 s per read vs minutes–hours of CEM |
| Dueling WM `2608.06706` | action separation (distance between predictions under different actions, normalised) | collapses **1.28 → 0.002** for every variant at ≥ 30 distractors while validation loss keeps improving; val-loss checkpoint selection picks collapsed checkpoints **17/36** (Freeway) | post-hoc centring on frozen RePo/TIA: raw action-delta probe −0.002 … +0.052 → **0.19 – 0.52** | §5 tables |
| PhyLatent `2608.05720` | three collapse modes on LeWM/Cube: invariance 15.60 %, identifiability 6.71 %, counterfactual (r_cf = d_pred/d_true < 0.5) 8.41 % | as left | PhyLatent 7.53 / 0.95 / 4.62; success 70.00 → 78.10 | Tables 1, 5 |
| UWM-JEPA `2605.25313` | ‖H1‖/‖H0‖ (action-dependent vs action-independent prediction energy); indicator under correct / no / shuffled / wrong action | teacher-forced target: **≈ 0.03** | counterfactual simulator targets: **1.00 ± 0.15**; indicator 0.770 / 0.733 / 0.685 / 0.639 vs LSTM-CF 0.530 flat (Table 2) | §4, Tables 1–3 |

Consequence for GS-8: `actdiv` should adopt the **zero-action-anchored displacement** (ActSWM / Delta-JEPA Fig. 6 / AD-JEPA form) so our read is comparable to the published effect sizes above. The raw scene-vs-action spread ratio (MM-E10) has no published counterpart and — per our own 2026-09-02 scene-matched re-analysis (INHERITED) — floats with the scene denominator.

### F2. The fixes partition exactly as pre-registered (H-SOTA-A2 → outcome A)

**(i) Encoder-shaping — needs a trainable trunk by construction**
- **Delta-JEPA / LDAD** `2606.31232`: the auxiliary decodes `a_t` from `Δz_t = z_{t+1} − z_t`, where BOTH terms are ENCODER outputs (§3); encoder = ViT-Tiny **trained from scratch**; `L = L_pred + λ L_action`, λ = 10 in the main tables, sweep {0…1000}: **λ = 0 nearly collapses, λ = 50 best** (Fig. 3). Success rates (Two-Room / Reacher / Push-T / Cube; Table 1): PLDM 93.73 / 64.33 / 76.13 / 57.27; LeWM 74.93 / 79.87 / 84.53 / 64.13; Sub-JEPA 90.60 / 81.00 / 63.73 / 62.67; **Delta-JEPA 100 / 81.33 / 89.07 / 79.27**. Decoding from `Δz` beats decoding from `concat[z_t, z_{t+1}]` on all four envs (Table 2: 95.93 → 100, 80.27 → 81.33, 76.47 → 89.07, 78.60 → 79.27). Decode target matters (Reacher, Table 3): raw action 81.33, Δjoint 80.47, Δfinger 64.93, both 76.40 — the target closest to the commanded quantity wins, the realised end-effector displacement loses 16 points. → GS-2 "inapplicable on a frozen trunk as published" is CONFIRMED from the PDF (the Ledger's reading was from the figure — now verified by content).
- **SMWM** `2606.20104`: `L = L_fwd + λ L_inv`, inverse MLP on `[z_t; z_{t+1}]` as the SOLE anti-collapse term; λ is environment-specific (Two-Room 0.1, Reacher 5, Push-T 30, Cube 1; λ = 0.1 collapses Reacher). Success (SMWM / SIGReg / random / forward-only; Fig. 5): 99 / 94 / 30 / 37; 66 / 67 / 14 / 11; 83 / 87 / 3 / 2; 84 / 59 / 43 / 44. Latent effective dimension tracks the controllable DOF; distractors are filtered; the forward model becomes ≈ `z + ρ(a)`. Stated limits: a single-frame encoder cannot capture velocities; action-correlated distractors are captured.
- **EB-JEPA** `2602.03604`: Two-Rooms 97 ± 2 % (MPPI); ablation Table 4: variance α = 0 → 47 ± 3, covariance β = 0 → 46 ± 3, temporal δ = 0 → 61 ± 2, **IDM ω = 0 → 1 ± 1 (collapse)**; IDM = MLP(z_t, z_{t+1}) (concat form); K = 8 rollout, Pareto k = 4.
- **PhyLatent** `2608.05720`: needs task-specific physical targets from a simulator (PSG); not transferable to our corpus as published.

**(ii) Through the predictor / at planning time — frozen-trunk compatible**
- **ACID** `2607.02403`: planning-cost augmentation, WM untouched: `c = c_g + w_a · c_a`, `c_a = (1/H) Σ ‖a_t − G_φ(ẑ_t, ẑ_{t+1})‖²`, adaptive `w_a = λ σ_g / σ_a`; the IDM `G_φ` is a flow-matching prefix–suffix transformer (4 layers, width 192) trained 200 K steps **on the frozen WM's latents**, 1 Euler step at test time. Table 1: Le-WM Cube 70 → 74, Reacher 76 → 88, Push-T 96 → 100; PLDM 58 → 68, 76 → 90, 72 → 76; Table 2: DINO-WM Rope chamfer 1.38 → 0.56, Granular 0.49 → 0.30; NWM ATE −2.3 %. λ robust over 0.005–0.1; +8.9–39 % planning latency (net ≈ 0.7× compute at equal quality). Limitation stated by the authors: the IDM needs consecutive observations to identify the action. Frozen-trunk compatible by construction.
- **ActSWM** `2607.26712`: hinge `max(0, cos(ẑ^gt, ẑ^0) − (1 − m))` between the recorded-action and the zero-action K-step rollouts (λ 0.5, m 0.3, K 12) + a **frozen random** action readout on `[z_t, z_{t+1}]` applied to encoded AND predicted transitions (λ 1.0) + SIGReg 0.09; H = 32, 128 px, 100 k steps. Effect: gap 0.002 → 0.760 (Table 5); Minecraft tasks 19/20 → 20/20, 10/20 → 19/20, 11/20 → 17/20 (Table 8). Trained end-to-end with SIGReg — **its behaviour with a frozen encoder is UNVERIFIED (not shown to fail either)**.
- **ATM's AITS-P** `2606.09028` (inverse loss on the PREDICTED transition) is the warning for this family: it creates predicted-domain-specific action codes (large L_ATM-sym) — an inverse loss on predicted transitions can be gamed unless the readout is frozen (ActSWM) or the comparison is anchored (AD-JEPA).

**(iii) Target construction — predictor/target side**
- **UWM-JEPA** `2605.25313`: teacher-forced targets give ‖H1‖/‖H0‖ ≈ 0.03; counterfactual SIMULATOR-rollout targets give 1.00 ± 0.15; blind ΔR² 0.023 / 0.069 / 0.091 (k = 1 / 3 / 5) vs LSTM 0.413 / 0.678 / 0.013 (Table 3). Toy environments; requires a simulator — not available on our corpus. Its diagnosis, however, is ours: our O5 target is teacher-forced (the target encoder sees the realised future; INHERITED from `2026-08-31-command-conditioning-and-l3`).
- **LatentAlign / DriveFuture** `2605.09701`: BEV latent + ego-status token; predictor `f_ψ(Q_f; [Z_t ∥ E_τ])`; action-source mixture `{p_gt, p_kin, p_∅}` with a learned null token and a constant-acceleration surrogate; target annealed from the grounded `sg(φ_enc(I_{t+T}))` to self-prediction with `α(e) = 1 − σ(β(e − e0))`. NAVSIM navhard EPDMS without scorer (Table 4): 30.9 → 32.1 (MSE) → 32.0 → **34.6** (full); sensitivity (Table 5): t_f 0.5 / 1.0 / 1.5 s → 30.2 / 31.2 / 34.6; q_s 4 / 16 / 64 → 28.2 / 34.6 / 33.7; e_o 0.75 / 0.83 / 0.95 → 29.2 / 34.6 / 28.9. **No action-sensitivity metric is reported**, and ego status is used at inference (not admissible for our vision-only edge tier).

**(iv) Anchored / advantage-style action channel — frozen-trunk compatible, post-hoc form exists**
- **Dueling WM / AD-JEPA** `2608.06706`: `ẑ' = B(z) + [Δ(z,a) − Δ̄(z)]` with `Δ̄(z)` the action-mean (K = 16 Monte-Carlo draws for continuous actions); exact common-mode cancellation (Props. 1–2). Training loss = cosine prediction vs EMA target (τ 0.996) + InfoNCE over K counterfactual actions (0.1) + offset compactness (1e-4) + VICReg (0.1). Post-hoc centring on the authors' own standard predictor reads 0.23 / 0.66 / 0.70 / 0.59 vs trained-in 0.31 / 0.54 / 0.71 / 0.66; DMC parity 0.533 / 0.328 / 0.186 (trained-in) vs 0.528 / 0.323 / 0.192 (post-hoc). Boundary: action-correlated distractors (d30ac) → chance. DMC-scale control null was pre-registered by the authors.

**Falsifier check.** No banked primary shows a predictor-side consistency loss FAILING with a frozen encoder ⇒ the partition stands; (ii)/(iv) are the arms drawable without re-opening the trunk decision.

### F3. A realised-motion action channel: what survives (H-SOTA-A3 → A for the encoder-side route, B for predictor-side fixes)

- `2606.07687` (What makes video WM latents action-relevant): LIBERO actions are **end-effector deltas** (realised motion — our r 0.9988 case, E-DEC-57 INHERITED). An inverse-dynamics objective on FROZEN features multiplies the action-relevant R² (probe MLP D→256→128→7 on mean-pooled features, task-OOD, 3 seeds; Table 1): V-JEPA 2 ViT-L **0.40 → 0.85**, V-JEPA 2.1 ViT-B 0.44 → 0.82, VideoMAE 0.46 → 0.75, **Web-DINO ViT-L −0.01 → 0.16**, SigLIP2 0.05 → 0.17, LAPA 0.41 → 0.51, DIFF 0.43 → 0.57, SDXL-VAE −0.55 → −0.41, Cosmos-1 −0.36 → −0.29, Dreamer 4 −0.04 → −0.04. ID is a MULTIPLIER on temporal-predictive structure, not λ-sensitive (Web-DINO stays 0.07–0.17 across λ). Ablation of the objective on DIFF: L1 inverse 0.629 > L2 0.578 > single-frame 0.523 > forward dynamics 0.408 > temporal contrastive 0.073; k-step ID peaks at k = 4. **Action-conditioning paradox** (Table 6): feeding the action INTO the trunk drops R² 0.26 → −0.32 — the published form of our MM-E17/E18 (the action is destroyed inside the conditioning pathway, and that deafness is an optimum).
- V-JEPA 2-AC `2506.09985` §3.1: actions = change in end-effector state between adjacent frames (realised motion), frozen ViT-g encoder, teacher-forcing L1 + a 2-step rollout loss; it plans (What-Drives-Success `2512.24497` Table 2, re-trained VJ2AC: Rc-R 16.2, Rc-Pl 33.1, DROID 42.9) — but **no action-sensitivity metric is reported anywhere**; the counterfactual comparison in `2512.24497` Fig. 2 is qualitative.
- No banked primary raises a sensitivity METRIC with a PREDICTOR-side fix on a realised-motion channel. ⇒ P2(b) (a command channel) keeps its rank as the fallback, and GS-2's tautology test runs first — with one construction-level point in favour of (iv): AD-JEPA's centring subtracts `Δ̄(z)`, the scene-predictable part of the action's effect; on a realised-motion channel the residual `Δ(z,a) − Δ̄(z)` is exactly the motion NOT predictable from the scene, i.e. the quantity the anchored `actdiv` (GS-8) is built to read. If the action were 100 % scene-predictable the residual would be 0 — the correct answer, not a metric failure.

### F4. Sensitivity over training (H-SOTA-A4 → partial)

- Dueling WM: separation collapses DURING training while validation loss improves; val-loss checkpoint selection picks collapsed checkpoints 17/36 — sensitivity is non-monotone in the loss. Delta-JEPA's λ sweep (λ = 0 near-collapse, 50 best) is a training-condition dependence, not a trace. ActSWM: context length and multi-step training improve prediction and leave the gap at ≈0. No primary reports late emergence under a schedule ⇒ MM-E13 ("acquired from ~0, 20–50× too slowly") is corroborated in DIRECTION (loss is not the signal) but has **no published schedule lever**; the actionable item is checkpoint selection by the sensitivity read (LAB-ACT-3).

### F5. Where our numbers sit (Q-A5)

Our MM-E10 ratio (0.00408–0.00595) is a different construction from every published metric (scene-vs-action spread, not zero-action anchored), so no field number is quoted AGAINST it. Regime-wise it is theirs: ActSWM LeWM gaps −0.005 … 0.002, AD-JEPA 0.002, Delta-JEPA LeWM displacements at the origin — models that at the same time plan at 74–96 % on toy environments (Delta-JEPA Table 1, LeWM column) — "collapsed by the metric, functional on the task", which is our T1 pattern (`copy_detector` CLEAN, echo 0.0000, and LOSES_TO_HOLDV0 on all three arms; registry §13.10 T1 block, MEASURED). Published effect sizes available as PRIORS for an anchored read: ActSWM gap 0.002 → 0.592 (readout only) → 0.760 (hinge + readout); AD-JEPA post-hoc 0 → 0.19–0.52.

### F6. Cost of each fix on our rigs

| fix | family | extra compute per step | extra modules / labels | frozen trunk? | realised-motion safe? |
|---|---|---|---|---|---|
| AD-JEPA post-hoc centring | (iv) | 0 training; K = 16 forward passes of the action head at read time | none | yes | yes (residual is the non-scene-predictable part) |
| AD-JEPA trained-in offset head | (iv) | K = 16 MLP evaluations | offset MLP, InfoNCE over K actions | yes | yes |
| ActSWM hinge + frozen readout | (ii) | + 1 zero-action K-step rollout (≈ +1× predictor cost) | frozen random readout | UNVERIFIED | hinge is action-agnostic; readout on `[z_t, z_{t+1}]` is endpoint-shaped (GS-1) |
| ACID | (ii) | 0 at training; IDM training 200 K steps (small transformer) on banked latents; +8.9–39 % planning latency | IDM | yes | needs consecutive observations (we have them at T1) |
| LDAD | (i) | + 1 MLP on Δz | none | **no** | decode target must be the commanded quantity (Table 3) |
| SMWM / EB-JEPA IDM | (i) | + 1 MLP on concat | none | **no** | concat form = endpoint-shaped (GS-1) |
| UWM-JEPA counterfactual targets | (iii) | simulator rollouts | simulator | n/a | n/a for us |
| LatentAlign anneal | (iii) | 0 | schedule only | yes (acts on the target) | no sensitivity metric published |

### F7. The endpoint-concatenation shape is the field's default and only Delta-JEPA tests it

EB-JEPA, SMWM, ATM, ACID and ActSWM's readout all consume `[z_t, z_{t+1}]`; Delta-JEPA Table 2 is the only ablation and it favours `Δz` by +4.07 / +1.07 / +12.60 / +0.67 points. GS-1's leak audit is therefore field-supported and TanitAD-novel in scope.

## 2. Hypothesis outcomes (SPEC §3)

| id | outcome | evidence | plan change (SPEC §4) |
|---|---|---|---|
| H-SOTA-A1 | **A** | F1: five metric formulations with before/after numbers | GS-8 adopts the zero-action-anchored displacement; priors: 0.002 → 0.59–0.76 (ActSWM), 0 → 0.19–0.52 (AD-JEPA) |
| H-SOTA-A2 | **A** | F2: partition holds; no falsifier found | first arm from (ii)/(iv); (i) recorded inapplicable for refav1; the freeze decision is the P2 decision only for the encoder-shaping family |
| H-SOTA-A3 | **A for the encoder-side ID route, B for predictor-side fixes** | F3 | P2(b) keeps its rank as fallback; GS-2 tautology test first |
| H-SOTA-A4 | **partial** | F4 | LAB-ACT-3 (checkpoint selection); no schedule lever registered |

## 3. What the field's numbers PREDICT for ours — agreement / disagreement

| our MEASURED fact (source) | field's number (lib) | verdict |
|---|---|---|
| ratio 0.004–0.006; no intervention raised it; two lowered it (MM-E10/E11, MM-E19-K8) | prediction levers leave the gap ≤ 0.002 while cosine rises 0.239 → 0.972 (`2607.26712` Table 5) | **AGREE** — all our arms were prediction levers |
| action signal healthy at the embedding, destroyed in FiLM (56× / 128×), gain converged (MM-E17/E18) | action-conditioning paradox 0.26 → −0.32 (`2606.07687` Table 6); non-anchored channel collapses to 0.002 with val loss improving (`2608.06706`) | **AGREE** — the deaf optimum is published |
| sensitivity acquired from ~0, too slowly (MM-E13) | collapse during training; val-loss selection picks collapsed ckpts 17/36 (`2608.06706`) | **PARTIAL** — non-monotone with loss; no "too slow" trace |
| `copy_detector` CLEAN yet LOSES_TO_HOLDV0 (§13.10 T1) | ρ(−L_pred, success) 0.50 vs ρ(−D_TT) 0.81 (`2606.09028` Table 2) | **AGREE** — prediction loss is a poor proxy for action use |
| O5 target teacher-forced (INHERITED) | teacher-forced target ⇒ ‖H1‖/‖H0‖ ≈ 0.03 (`2605.25313`) | **AGREE** on mechanism; their fix needs a simulator |
| LDAD inapplicable on a frozen trunk (GS-2, INHERITED from the Ledger) | Δz from two ENCODER outputs, ViT-Tiny from scratch (`2606.31232` §3) | **CONFIRMED by content** |
| h≥2 heads never trained (MM-E14) | ActSWM K = 12, EB-JEPA K = 8, ACID H-step cost — every fix acts on MULTI-step rollouts | **DISAGREE with our design** — a one-horizon predictor cannot host (ii) as published; the hinge needs K ≥ 2 |

## 4. Cheapest discriminating experiment (Q-A5), both outcomes committed → `PROPOSED_HYPOTHESES.md`

**LAB-ACT-1 (0 GPU-h training, banked checkpoints):** AD-JEPA post-hoc action-mean centring on all 32 census arms — read the anchored displacement `‖Δ(z,a) − Δ̄(z)‖` (K = 16 corpus-drawn actions) against the raw displacement, with the shuffled-action (ROLL) and constant-only controls and the episode-cluster bootstrap. Outcome A (anchored ≥ 3× raw on ≥ 1 arm, CI excluding 1): the pathway is alive and masked by the common mode — MM-E17/E18 is re-read, and the trained-in AD-JEPA offset head is the first v7-tiny arm. Outcome B (anchored ≈ raw on all arms): the pathway is dead — ActSWM's hinge (LAB-ACT-2, one variable, v7-tiny) is the next arm and the K ≥ 2 rollout is its precondition. Sibling work: the Master Mind's GS-8/GS-9 agent was already running an anchored `actdiv` during this pass (INHERITED, output not read); LAB-ACT-1 adds the K-sample centring and the parity read that makes the number comparable to `2608.06706`.

## 5. Evidence-class ledger

| claim | class | source |
|---|---|---|
| all numbers in F1–F7 tables | PUBLISHED-PRIMARY | banked PDFs, keys as stated; sections/tables as stated |
| MM-E10 ratio, MM-E11/E12/E13/E14/E17/E18, §13.0c census, §13.10 T1 block | MEASURED | `GOALS_AND_CLAIMS.md` rows; `MODEL_REGISTRY.md` §13.0c, §13.10 (line refs in SPEC §1) |
| our action channel = realised motion r 0.9988 | INHERITED | E-DEC-57 via readiness report (row not re-opened) |
| O5 target teacher-forced | INHERITED | `2026-08-31-command-conditioning-and-l3` |
| MM-E19 ratio decomposition (denominator floats) | INHERITED | `2026-09-02-scene-matched-action-criterion/RESULT.md` |
| GS-8 anchored actdiv running in parallel | INHERITED | background-agent notification in this session; output file empty at read time |
| ActSWM frozen-encoder behaviour | UNVERIFIED | not tested in `2607.26712` |
| DriveFuture encoder trainable or frozen | UNVERIFIED | not extracted in this read |

## 6. Named empty searches (absence claims; two independent probes each)

1. **No banked primary raises an action-sensitivity metric with a predictor-side fix on a realised-motion action channel.** Probe 1: full reads of the 12 theme-3 primaries (§7). Probe 2: the pass's web searches #4, #8, #9, #12 (`raw/search_log.md`) returned no such paper; V-JEPA 2-AC / DROID work reports planning only.
2. **No banked primary reports the sensitivity metric as a function of training step with a schedule that makes it emerge earlier.** Probe 1: full reads. Probe 2: search #4/#8 phrasing "curriculum"/"dropout" surfaced action-dropout/CFG work for generative driving models only (not banked; secondary).
3. **No banked primary shows a predictor-side consistency loss failing under a frozen encoder** (falsifier of H-SOTA-A2). Probe 1: full reads. Probe 2: grep of all 36 extracted texts for "frozen" co-occurring with "inverse" — hits are ACID (frozen WM latents, works), `2606.07687` (frozen features, works), `2602.18639` (frozen DINOv2 + adapter, works).

## 7. Papers banked and read for this theme (all `sota-2026-09-03-action` unless noted)

| key | short | read | quoted from |
|---|---|---|---|
| `2606.31232` | Delta-JEPA | full | §3, Fig. 3, Fig. 6, Tables 1, 2, 3, 5 |
| `2607.26712` | ActSWM | full | §3 loss, Tables 5, 8 |
| `2608.06706` | Dueling World Models / AD-JEPA | full (two chunks) | Props. 1–2, §5 tables (post-hoc, DMC, distractor sweep, checkpoint selection) |
| `2606.09028` | ATM | full | Tables 1, 2, 3, 6 |
| `2607.02403` | ACID | full | §3, Tables 1, 2, latency table |
| `2606.20104` | SMWM | full (two chunks) | §3, Fig. 5, limitations |
| `2602.03604` | EB-JEPA | full | Table 4 |
| `2605.25313` | UWM-JEPA | full (two chunks) | §4, Tables 1, 2, 3 |
| `2608.05720` | PhyLatent | full | Tables 1, 4, 5, 12 |
| `2605.09701` | DriveFuture / LatentAlign (backlog P-1) | full (two chunks) | §3 method, Tables 4, 5 |
| `2606.07687` | action-relevant latents | full (two chunks) | Table 1, Table 6, per-layer and ablation tables |
| `2606.15032` | How should WMs be evaluated (position) | full (two chunks) | L0–L7 ladder, objective mismatch, interventional do(a), pipeline entanglement |
| `2506.09985` | V-JEPA 2 (tag: encoder) | §3 (AC) only | §3.1 eqs. 2–4 |
| `2512.24497` | What Drives Success (tag: encoder) | full (chunks) | Table 2, Fig. 2 |
| `2607.09185` | Causally Debiased Latent Action Model | **banked, NOT read, NOT cited** (pass ran out of time before extraction; `cited_by` already carries this package's path from the banking step — treat as a false citation record until re-banked with a note) | — |

## 8. Relay corrections and caveats

- The search subagent's summary attributed lib `2607.05238` to "MoP-JEPA"; the banked PDF is **Branch-JEPA** (drift theme). Recorded in `raw/search_log.md`.
- Delta-JEPA's decode-target result (Table 3) reads AGAINST a realised end-effector target (64.93 vs 81.33) — the Ledger's decode-target rule "(a, κ) excluding v" is consistent with it but was derived independently; both are now on the same page.
- The AD-JEPA "post-hoc on frozen models" numbers are on RePo/TIA (visual RL) and DMC — not driving; effect sizes are priors for direction and order of magnitude only.

## 9. Integration escalations (for the Master Mind's headline)

1. **Adopt the anchored (zero-action / action-mean) displacement as the GS-8 `actdiv` formulation** before any arm is judged — otherwise no published effect size applies.
2. **Run LAB-ACT-1 (0 GPU) before the O11 re-run**: it decides between "dead channel" and "masked channel", and the two branches lead to different first arms (AD-JEPA offset head vs ActSWM hinge).
3. **Re-select the T1 candidates by the sensitivity read** (LAB-ACT-3) — the field's 17/36 error rate for val-loss selection applies to our nrmse-selected arms.
4. The one-horizon predictor (MM-E14) is a design constraint for every predictor-side fix: a K ≥ 2 rollout is the precondition, and the 2026-09-01 curriculum recommendation is the way to get it.
