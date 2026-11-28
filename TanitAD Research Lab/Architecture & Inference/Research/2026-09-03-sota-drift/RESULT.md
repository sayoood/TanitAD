<title>RESULT - SOTA pass: latent drift over the rollout (theme 1 of 4)</title>

# RESULT - E-ARCH-SOTA-DRIFT-1: the field's "drift" is not our drift; one published lever has a measured effect and is admissible for us; drift is a diagnostic, not a gate

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-drift/` · Research Lab overnight pass 2026-09-03 · literature only · 0 GPU · no pod/Thor contact
**Spec** `SPEC.md` (staged before any primary was opened; H-SOTA-D1…D4 with both outcomes committed)
**Evidence classes** PUBLISHED-PRIMARY (banked PDF read in full, `lib:<key>`), MEASURED (registry / register line), INHERITED (own summaries, not re-verified). Nothing PUBLISHED-SECONDARY enters this file.
**Register rows proposed** `PROPOSED_HYPOTHESES.md` → LAB-DRIFT-1 … LAB-DRIFT-3.

## 0. Verdict in five lines

1. **Our "drift" (E-DEC-59: the `r` of a linear probe predicting `Δz` from `z_t`) is not the field's drift.** Every banked primary that measures drift measures a rollout-vs-encoded-future divergence — JEPA-x normalises it by persistence, `drift(h) = E‖ẑ_{t+h} − z_{t+h}‖² / E‖z_t − z_{t+h}‖²`. No primary defines it as increment predictability (two probes, §6). **No field number may be quoted against our 0.669** until L-17 (the divergence instrument) exists and L-18 (the rename) is done.
2. **One lever has a measured, large effect on a divergence metric and is admissible for us**: JEPA-x's cross-prediction against a privileged physical state, training-only — drift 0.361 → 0.104 (multi-task; single-task 0.503 → 0.221 Two-Room), control 53.6 → 78.2 %. Its control arms are our register in miniature: a plain state-REGRESSION head raises decodability (R² 0.978 → 0.991) and leaves drift at 0.373 — the published form of E-DEC-67 (O14 absorbed pixels, drift unchanged).
3. **EMA/teacher targets are nowhere an anti-drift lever** (H-SOTA-D3 → B): used for stability by V-JEPA 2, Latent-WAM, Dueling WM, Branch-JEPA, explicitly unnecessary in LeWM; no primary ablates EMA on a drift quantity. Our reading (EMA moves prediction +24.5 % cos, drift +3.6 %) agrees with the field's usage; backlog row 4 closes.
4. **The field's multi-step rollout loss is universally truncated and short**: V-JEPA 2-AC differentiates through ONE recurrent step (T = 2); What-Drives-Success finds K = 2 optimal in simulation, K = 6 on DROID, and proves the accuracy–robustness trade-off (Remark 1: the rollout loss lowers the effective Lipschitz constant along the rollout at the cost of one-step accuracy). Our E-DEC-66 (k = 4 INERT on the trainable line) and MM-E19 (k 8 → 60 lowered the ratio) sit inside that curve.
5. **Drift is not the capability-limiting quantity** (H-SOTA-D4 → A): the best proxy for planning success in What-Drives-Success is the ONE-step embedding loss (Tables 13–16); ActSWM raises the 31-step rollout cosine from 0.239 to 0.972 with context and multi-step training and the action gap stays ≤ 0.002. P3 is demoted below P1/P2 in the v7f prereg; drift stays a diagnostic.

## 1. Findings

### F1. What the field calls drift (H-SOTA-D2 → outcome A)

| primary | quantity | protocol |
|---|---|---|
| JEPA-x `2608.24044` | `drift(h) = E‖ẑ_{t+h} − z_{t+h}‖² / E‖z_t − z_{t+h}‖²`, persistence-normalised, mean over h = 1…20 | "forecastability": freeze the encoder, fit a FRESH predictor (3-layer MLP, 512, context H = 3, residual, 30 k steps, disjoint episodes), 20-step autoregressive rollout on recorded actions |
| Fast-LeWM `2606.26217` | open-loop latent error vs the encoded future latents; growth SLOPE over the horizon | trained predictor, recorded actions |
| What-Drives-Success `2512.24497` §G | embedding-space L1/L2 error at unroll steps H = 1…3; proprioceptive decoding error along the unroll (state decoder trained on the frozen encoder) | every 300 iterations on the validation split |
| GRWM `2510.26782` | frame-wise error along decoded rollouts against an oracle with the true state | trained dynamics on a reconstruction-trained autoencoder latent vs a geometrically regularised one |
| Deep supervision `2504.03861` | "distribution drift" = growth of the prediction loss after the teacher-forcing cut-off | LSTM / MDN on an 8-d latent |
| Depth-Reg JEPA `2607.16314` | rollout goal-similarity at h = 10 / 20, in-domain and OOD | trained predictor |
| ActSWM `2607.26712` | cosine of the K-step rollout to the truth at step 31 | trained predictor |

None is `r(Δz | z_t)`. Second probe (grep over the 36 extracted texts): 20 files mention "drift"; "persistence" appears in 5 (`2510.26782`, `2606.31672`, `2607.27017`, `2608.06706`, `2608.24044`); "predictab*" next to "increment / delta / Δz / displacement": **0 hits**. The falsifier in SPEC §6 (a primary defining drift as increment predictability) was not met.

### F2. Ablations with a MEASURED effect on a divergence quantity (H-SOTA-D1 → outcome A)

| primary | lever | divergence metric before → after | side reads | cost / applicability |
|---|---|---|---|---|
| JEPA-x `2608.24044` | cross-prediction: the visual latent predicts a privileged physical state and vice versa, TRAINING-ONLY (inference is vision-only) | multi-task drift VISUAL 0.361 → **0.104**; single-task (Table 1) Two-Room 0.503 → 0.221, Block 0.507 → 0.269, Push-T 0.399 → 0.258, Reacher 0.313 → 0.233; Table 3: CROSS-ONLY 0.101 / control 73.0, SHARE-ONLY 0.309 / 56.7, ALIGN-ONLY 0.158 / 72.6 | control 53.6 → 78.2 %; **REGRESS (state-regression head) 0.373, DISTILL 0.516, SHUFFLE 0.540**; state regression raises decodability R² 0.978 → 0.991 but not drift; action-conditioned next-state / increment heads 0.386 / 0.359 (control 54.4 / 51.3) — no gain; rank-matched projection gap persists (0.121 vs 0.453 at k = 57) | one extra head + a state stream at training; **admissible for us: ego kinematics are training-time labels under the labels-may-use-ego rule (INHERITED), inference stays vision-only** |
| What-Drives-Success `2512.24497` | multi-step rollout loss (TBPTT: gradient only through the last prediction; scheduled-sampling family, §C) | Fig. 3b: 1 → 2 steps improves success in simulation, more steps decline; DROID optimum K = 6 | Proposition 1 / Remark 1: the K-step loss reduces the effective Lipschitz constant Λ_K along the rollout at the price of one-step accuracy (accuracy–robustness trade-off); Tables 13–16: unroll metrics at H > 1 correlate better with success only on Metaworld | 0 extra modules; our O5 already is this loss |
| V-JEPA 2-AC `2506.09985` §3.1 | teacher-forcing L1 + rollout loss, **T = 2, "only differentiate the predictor through one recurrent step"** | not reported as a number | frozen ViT-g encoder; 300 M predictor | the field's standard truncation |
| Fast-LeWM `2606.26217` | replaces recursion: all future latents predicted in parallel from `z_t` and action PREFIXES | open-loop latent error lower AND slower-growing (slope) on four tasks | success 85.8 → 90.5 (92.0 with self-consistency); dynamics time 31.4 → 8.0 s; Table 4: Long-Action LeWM 76 / 70 / 80 / 58 vs terminal-only prefix 96 / 80 / 90 / 72 vs full prefix 98 / 88 / 96 / 80 — dense prefix supervision matters | a new predictor interface (not a flag) |
| Flow-JEPA `2608.29029` | conditional flow matching over the whole h = 5 future trajectory, no recursion, SIGReg kept | success only: clean mean 86 → 92 (100 / 90 / 96 / 82), noisy 67 → 86 | 8 Euler steps; 5.0 ms per flow step vs 2.1 ms LeWM | no divergence metric |
| Depth-Reg JEPA `2607.16314` | `1 − cos(f(RGB), sg f(depth))` (0.1) + training-only RepLinear over-parameterisation | rollout goal-similarity in-domain h = 10 0.903 → 0.890 (frozen DINOv2 0.951); OOD h = 10 0.716 → 0.745, h = 20 0.458 → 0.537 | VO probe MSE 0.0022 → 0.0015 (DINOv2 0.0029); surprise separation 0.067 → 0.120 / OOD 0.035 → 0.099; 18 M, 31 K real frames; **single run**; authors caution: rollout similarity rewards smooth / slow latents | depth target needed |
| Sub-JEPA `2605.09241` | SIGReg per frozen row-orthonormal subspace (K = 32; Push-T 16) | Fig. 5 rollout to t = 95: LeWM drifts, Sub-JEPA holds — **qualitative, no number** | success 84.33 / 82.67 / 84.67 / 67.33 → 95.00 / 84.00 / 89.00 / 76.33; Push-T collapses at K = 32 (28.00); random-frozen projections 53.00 / 68.00 / 13.33 / 61.00 vs trainable-ortho 61.67 / 82.67 / 57.00 / 70.33 | one regulariser change |
| Deep supervision `2504.03861` | a linear-probe loss on the latent (λ up to 64) | reduces "distribution drift" (loss growth after the cut-off); claimed equivalent to ~2× model size | LSTM / MDN, 8-d latent — a different regime; **conflicts on the surface with JEPA-x REGRESS** (recorded, not resolved) | n/a |
| GRWM `2510.26782` | temporal-contrastive geometric regulariser on the latent | with the TRUE state the rollout error stays near zero (oracle); a standard reconstruction-trained autoencoder latent accumulates error fast; the regulariser closes most of the gap; probe MSE 0.082 → 0.031 | "the representation geometry, not the dynamics model, is the bottleneck" | encoder-side |
| ActSWM `2607.26712` | context H 3 → 32 and multi-step training | step-31 cosine 0.239 → 0.972 — the largest anti-divergence effect in the set | action gap unchanged (≤ 0.002) | already available to us (context, K) |
| Latent-WAM `2603.24581` Table 6 | prediction stride | EPDMS 88.4 (0 → 8) / 89.3 (−3 → 0 → 4 → 8) / 89.1 (stride 2) | planning, not drift | — |

### F3. EMA / teacher targets (H-SOTA-D3 → outcome B)

EMA target encoders appear in V-JEPA 2 (pre-training), Latent-WAM (target encoder), Dueling WM (τ 0.996), Branch-JEPA (τ 0.996) — every time as a stability / collapse device. LeWM `2603.19312` trains with no EMA and no stop-gradient (SIGReg replaces both), and `2607.27017` states the same in its related work. Second probe: six extracted texts contain both "EMA" and "drift" (`2603.20327`, `2605.25313`, `2606.21775`, `2607.05238`, `2607.27017`, `2608.06706`); none has an EMA-on/off row on a drift quantity. ⇒ our EMA finding (drift 0.6952 vs 0.6709, cos_ctr 0.7524 vs 0.6043, nrmse 0.7466 vs 0.8288; §13.10, MEASURED) AGREES with the field: EMA buys prediction, not drift, and there is no published prior for an EMA drift effect. Backlog row 4's remaining question closes.

### F4. Does drift limit planning? (H-SOTA-D4 → outcome A)

- What-Drives-Success §5.2: "even with models which are able to faithfully unroll a large number of actions, success at the planning task is not an immediate consequence"; §G.3 / Tables 13–16: the metric most correlated with success is the ONE-step visual-embedding loss (mean −ρ: Metaworld 0.47, Push-T 0.86, Wall 0.81, Maze 0.57 for L1 H = 1); unroll metrics at H > 1 win only on Metaworld.
- ActSWM: a rollout that is nearly drift-free (cosine 0.972) but action-blind fails Minecraft tasks (10/20) until the sensitivity term is added (19/20).
- How-should-WMs-be-evaluated `2606.15032`: logged-future prediction (their L1) is only weakly tied to control (objective mismatch).
- JEPA-x: drift and control co-move under cross-prediction, but REGRESS moves decodability without moving either — the three quantities are separable.
- Ours (MEASURED): `e4_ctrl_shuf` reproduces −20.1 % drift with a SHUFFLED constraint (§13.10) — our statistic is trivially reducible; the field's guards are the persistence normalisation (JEPA-x) and the "slow latents" caution (Depth-Reg).

### F5. What the field's numbers PREDICT for ours — agreement / disagreement

| our MEASURED fact | field | verdict |
|---|---|---|
| one-variable freeze: drift 0.3905 but nrmse +14.6 % (E-DEC-64) | JEPA-x: forecastability differs by encoder; What-Drives-Success: frozen trunks encode metric quantities only implicitly (proprioception needed); Utonia-WM Push-T 46.6 (`2608.29434`) | **AGREE** that freezing costs fine prediction; the drift number itself is NOT comparable |
| `o5_k` 4 INERT on the trainable line (E-DEC-66); k 8 → 60 lowered the ratio (MM-E19) | K = 2 optimal in sim, K = 6 on DROID, decline beyond (`2512.24497` Fig. 3b, Remark 1) | **AGREE** — the K lever is small and non-monotone; no primary uses K > 6 |
| O14 future-observation auxiliary absorbed pixels, drift unchanged (E-DEC-67) | REGRESS head: R² 0.978 → 0.991, drift 0.373 ≈ 0.361 (`2608.24044`) | **AGREE** (exception on record: `2504.03861`, different regime) |
| EMA: prediction +24.5 % cos, drift +3.6 % (§13.10) | EMA never ablated on drift; used for stability | **AGREE** |
| k = 60 full chain diverged (INHERITED) | T = 2 through one recurrent step (`2506.09985`); TBPTT last-step (`2512.24497`) | **AGREE** (banked 2026-09-01) |
| drift trivially reducible (`e4_ctrl_shuf`) | persistence normalisation; slow-latent caution | **AGREE** — import the normalisation (L-17) |
| never run: a training-only cross-prediction to ego kinematics | JEPA-x 0.361 → 0.104 with control 53.6 → 78.2 | **NEW LEVER, untested on our line** → LAB-DRIFT-2 |

## 2. Hypothesis outcomes (SPEC §3)

| id | outcome | evidence | plan change |
|---|---|---|---|
| H-SOTA-D1 | **A** | F2 (JEPA-x, WDS, Fast-LeWM, Depth-Reg, GRWM, ActSWM) | JEPA-x cross-prediction becomes the v7-tiny arm candidate with 0.361 → 0.104 as its prior, pre-registered against the L-17 instrument (LAB-DRIFT-2) |
| H-SOTA-D2 | **A** | F1 | L-17 and L-18 move up; no cross-paper comparison until then (LAB-DRIFT-1) |
| H-SOTA-D3 | **B** | F3 | our EMA reading agrees with the field; backlog row 4 closes |
| H-SOTA-D4 | **A** | F4 | P3 demoted below P1/P2 in the v7f prereg; the freeze question stays closed (E-DEC-64) |

## 3. Cheapest discriminating experiment (Q-D4), both outcomes committed → `PROPOSED_HYPOTHESES.md`

**LAB-DRIFT-1 (0 GPU-h training, banked checkpoints):** implement the JEPA-x persistence-normalised divergence `D(h)` on `postrain30k`, `postrain30k_freeze`, `splitp30k`, `emao14_30k`, `k4_30k` — two reads: (a) the trained predictor's own autoregressive rollout on recorded actions, h = 1…8 (h > 1 needs the rollout, MM-E14 caveat noted), (b) JEPA-x forecastability of each ENCODER (fresh 3-layer MLP predictor, 30 k steps, disjoint episodes). Controls: persistence = 1.0 by construction, constant-only 0, shuffled-action, and the E-DEC-59 statistic on the same windows. Outcome A: `D(h)` orders the arms like nrmse (freeze worst) → E-DEC-59 was the outlier; L-18 renames it and P3 is re-based on `D(h)`. Outcome B: `D(h)` orders the arms like E-DEC-59 (freeze best) while nrmse says the opposite → the JEPA-x dissociation holds on our line; P3 is a diagnostic and leaves the gate.

**LAB-DRIFT-2 (one v7-tiny arm):** cross-prediction head to ego kinematics (a, κ; v per the Ledger's decode-target rule), training-only, one variable on the `postrain30k` recipe, with the REGRESS-style control head (expected null, per JEPA-x). Prior: 0.361 → 0.104; falsifier: no `D(h)` change with the L-17 instrument.

## 4. Evidence-class ledger

| claim | class | source |
|---|---|---|
| all numbers in F1–F4 | PUBLISHED-PRIMARY | banked PDFs, keys and tables as stated |
| E-DEC-59 definition; §13.9 / §13.10 drift, nrmse, cos_ctr, `e4_ctrl_shuf`, τ-ramp; E-DEC-60/61/64/66/67 | MEASURED | `GOALS_AND_CLAIMS.md`, `MODEL_REGISTRY.md` (line refs in SPEC §1) |
| drift flat across k 4 → 60; k = 60 full chain diverged | INHERITED | `2026-09-02-drift-metric-axis-audit`, `PREREG_MM_E19_K60_HORIZON.md` |
| labels-may-use-ego rule (ego kinematics admissible as training targets) | INHERITED | programme rule as cited in the readiness report; not re-read here — **the Master Mind should confirm before LAB-DRIFT-2 is registered** |
| Utonia-WM Push-T 46.6 | PUBLISHED-PRIMARY (encoder theme) | `2608.29434` Table 2 |

## 5. Named empty searches (two probes each)

1. **No primary defines drift as the predictability of the latent increment from the current latent.** Probe 1: full reads (§7). Probe 2: grep "predictab*" × {increment, delta, Δz, displacement} over 36 texts — 0 hits; "persistence" in 5 texts, all rollout-divergence usages.
2. **No primary ablates EMA / teacher targets on a drift or divergence quantity.** Probe 1: full reads. Probe 2: 6 texts with both "EMA" and "drift"; none has the row.
3. **No primary back-propagates a rollout loss through more than one recurrent step at K > 6** (re-confirmed; first banked 2026-09-01): V-JEPA 2-AC T = 2, WDS K ≤ 6, EB-JEPA K = 8 (Pareto at 4), ActSWM K = 12 hinge (not a prediction loss).

## 6. Relay corrections

- lib `2607.05238` is **Branch-JEPA**, not "MoP-JEPA" (search-summary error; library note corrected in this pass). It is tangential to drift (multimodal successors on Argoverse 2; Table 1 energy score 2.0766 vs 2.2201 specialisation vs 2.3827 MDN; Table 2 K = 1 2.8426 vs output-only branching 2.1142).
- Flow-JEPA reports planning success only; it was banked on the expectation of a drift metric — none exists in the PDF.

## 7. Papers banked and read for this theme

| key | short | read | quoted from |
|---|---|---|---|
| `2608.24044` | JEPA-x | full (two chunks) | §3 protocol, Tables 1, 3, control arms, rank-matched analysis |
| `2607.16314` | Depth-Reg JEPA | full | Table 1, caveats |
| `2607.05238` | Branch-JEPA | full (two chunks) | Tables 1, 2 |
| `2608.29029` | Flow-JEPA | full | Tables 1, 2, timing |
| `2605.09241` | Sub-JEPA | full | Tables 1, 2, 3, Fig. 5 |
| `2512.24497` | What Drives Success | full (chunks) | Fig. 3b, §C, Prop. 1, Remark 1, §5.2, §G.3, Tables 13–16 |
| `2506.09985` | V-JEPA 2 | §3.1 only | eqs. 2–4 |
| `2606.26217` | Fast-LeWM | full | Table 4, timing, slope |
| `2603.19312` | LeWM | full (two chunks) | §3 loss (no EMA / SG) |
| `2504.03861` | Deep supervision with linear probes | full | drift definition, λ sweep |
| `2510.26782` | GRWM / cloning deterministic worlds | full | oracle analysis, probe MSE |
| `2607.26712`, `2603.24581`, `2606.15032`, `2608.29434` | cross-theme | as in their themes | Table 5; Table 6; L-ladder; Table 2 |

## 8. Integration escalations

1. **Rename E-DEC-59 and add the persistence-normalised divergence (L-17/L-18) BEFORE the v7f prereg quotes any drift number** — every field comparison is otherwise invalid.
2. **P3 leaves the gate**; it stays a diagnostic read next to nrmse and the anchored action read.
3. **Register LAB-DRIFT-2 (training-only cross-prediction to ego kinematics)** after the Master Mind confirms the labels-may-use-ego rule — it is the only published anti-drift lever with a measured effect that we have never run, and JEPA-x's REGRESS control already predicts why O14 was null.
