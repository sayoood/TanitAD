<title>Frontier Scan 2026-09-18 — flow is the load-bearing term, displacement carries the action, and TRT 11 ends weak typing</title>

# Frontier Scan · 2026-09-18 (LAB-RUN-015)

`22/22 tracks scanned · Band D ×3 claims (Tesla) + 1 register correction · FULL TEXT ×3 (Delta-JEPA, FlowR2A, 2607.07196, all via local pypdf) + 2 HTML primaries (DriveZero, LGS) · 10 empties named · Library 507 → 513`

---

## 0 · Findings first — the six that change something

| # | finding | class | moves |
|---|---|---|---|
| **1** | ⭐⭐⭐ **A THIRD independent line says the predictor's transition should be GENERATIVE, not a deterministic regression, and this one isolates it.** LGS `2602.11229`: removing flow matching from the latent transition inflates 5/10-step error by **+145 % / +155 %**, about **5×** the damage of removing input noise. Its input noising gives a **contraction bound**: rollout error is bounded when `L_T(1−k) < 1`. The price is **≤ 16 %** at 1 step, and LGS wins **0/16** systems at 1 step but **16/16** at 20. After WA-JEPA (+1.0 EPDMS, 09-17) and FlowR2A's generative decoder (today), that is three lines in three disciplines. | PUBLISHED lib `2602.11229` (HTML, via summariser) | **LR14-5** (flow predictor) ⇒ strengthens the case; **ASK-1 / W-17-3** (long-rollout instability) ⇒ a zero-architecture lever (input noising). ⚠️ 1-step metrics will read it as a regression: **stamp T0 vs T1** |
| **2** | ⭐⭐⭐ **Delta-JEPA (full text) + our own probe: the action is readable from the latent DISPLACEMENT, and on our frozen trunk it already is, longitudinally.** Displacement-decoding beats endpoint-concat by up to **+12.60 pp** (3 seeds). Our `E-AI-LDAD-0` measured **R² 0.33** for LON acceleration from frozen `Δz` vs ≈ 0 pixel floor and +0.28 over the scene alone. | PUBLISHED + MEASURED (`Architecture & Inference/Research/2026-09-18-ldad-displacement-probe`) | **P-2** gets a third, cheapest candidate lever (predictor-side LDAD, AI18-1) |
| **3** | ⭐⭐⭐ **DriveZero `2609.06055` takes navhard two-stage to 57.1 EPDMS (Scale) / 51.5 (base) — above the top of our stamp table (56.6).** Its recipe is **frozen multi-VFM consolidation** (DINOv3 + SigLIP2 + SAM + Depth-Anything-V2, then LoRA): **+0.53 PDMS** over DINOv3-only at ViT-S. It adds a **5.70 M-param privileged PPO teacher** distilled into the vision student. RL-only supervision **93.61 < human 93.92**; **+ goal augmentation 94.41**. ⛔ Consumes **ego kinematics + nav command at inference** ⇒ a **benchmark-arm** number only (row 11). | PUBLISHED lib `2609.06055` (HTML, via summariser); navhard = **two-stage** | **AI13-2** (freeze), **row 11** (two-arm design), refav1 RL line, **G3** |
| **4** | ⭐⭐ **A privileged-observation RL teacher is a LABEL FACTORY, and our binding rules already allow it.** DriveZero's teacher sees *"up to 96 participants"* and *"256 tokens of local vector elements"*, all privileged; the student sees cameras. Under the PI's 2026-08-03 ruling (*labels may use anything; inference is vision-only*), that is **admissible** — the teacher is label derivation. ⚠️ The student must drop ego status at inference to stay inside our rule, and DriveZero does **not** test that arm. | analysis | new route for the RL line that does not need RL *on* the student |
| **5** | ⭐⭐ **TensorRT 11 REMOVES weak typing and the per-precision builder flags** (`kFP8`, `kFP4`, `kINT8`, …); strongly typed networks are now required. ⛔ **But TRT 11.3.0 states "NVIDIA JetPack is not supported"** ⇒ Thor stays on the TRT 10.x line (JetPack 7.2.1 → TRT 10.16.2, 09-17). The silent-FP32-fallback hazard behind **D-B1-GATE** is therefore **still live on Thor**. The mechanism it exploits is **gone upstream**. | PUBLISHED-RELEASE-NOTE (NVIDIA docs 11.0.0 / 11.3.0, read 2026-09-18) | **Backlog row 1**: write the B1 recipe **strongly typed + explicit Q/DQ now**. It is the only form that survives the TRT 11 migration, and the gate is still needed until JetPack gets TRT 11 |
| **6** | ⭐⭐ **Open-loop metrics are blind to vision dependence: an independent, controlled measurement.** `2605.31041`: removing the images entirely moves open-loop L2 only **+7.1 %**, but closed-loop NeuroNCAP **−14.6 %** (downsampling 75 %: **−27.2 %**). Single model (Impromptu-VLA 3B), by the authors' own concession. | PUBLISHED lib `2605.31041` (HTML) | `EVAL_DOCTRINE.md` (T0/T1 vs T2), confirms our 97.9 %-vs-0.0 % action echo from outside |

---

## 1 · ⭐⭐ Band D — Tesla doctrine, adjudicated (§7.1, seven steps each)

**Documents.** (a) Tesla AI release notes, **FSD v14 Lite, 2026.20.5.1, 2026-06-29** (official account; read via `notateslaapp.com` verbatim reproduction ⇒ `PUBLISHED-RELEASE-NOTE`, relayed page). (b) Ashok Elluswamy (VP AI) talks, ICCV-25 WDFM-AD and ScaledML 2026-01-29, read via coverage only ⇒ **`RELAYED`** (E-T182-primary: no transcript primary found). Competitive context: Tesla scaling unsupervised robotaxi in Austin, and HW3 owners owed an upgrade path.

| id | claim | experiment? | confirming (independent) | contradicting (actively sought) | verdict | binds on us? | flips it |
|---|---|---|---|---|---|---|---|
| **T-18-1** | *"Distilled the intelligence from HW4 V14 into HW3 … unlocks the improvements … including Reinforcement Learning (RL)"* (HW3 has ~15 % of AI4's memory bandwidth, RELAYED) | ⛔ **NO** — no metric, no A/B, no before/after number in the notes | Waymo teacher→student (W-concession, different org) · **DriveZero** privileged teacher → student, measured (+0.80 PDMS with goal aug) | ⭐ **TAKD `1902.03393`**: distillation **degrades when the capacity gap is large**; an intermediate *teacher assistant* is needed | **SUPPORTED** as a practice; ⚠️ **UNSUPPORTED-AS-STATED** that *RL gains* transfer (no number) | ⭐ **YES, favourably** — train-large / deploy-small is our sub-300 M thesis, now stated by **two** opponents | A published HW3-v14-Lite vs HW3-v13 safety/intervention comparison showing no gain |
| **T-18-1c** ⭐ | **CONCESSION** (same notes): *"This feature does not make your vehicle autonomous"* | admission | — | — | **SUPPORTED** | ⭐ **YES** — the distilled student ships **supervised**. Distillation reached *supervised* quality, and the claim that it reaches *unsupervised* quality is **not made** | an unsupervised deployment on a distilled student |
| **T-18-2** | *"loss on open-loop predictions might not correlate to great performance in the real-world"* ⇒ closed-loop neural world simulator for evaluation | ⛔ **NO** in the talk (RELAYED) | `2605.00066` (open-loop metrics only partly track closed-loop) · **`2605.31041`** (+7.1 % vs −14.6 %) · our own 97.9 % vs 0.0 % | `2506.04218` pseudo-simulation reaches **R² 0.8** vs closed loop from **non-reactive** data — i.e. *some* offline metrics DO correlate | **SUPPORTED** — for open-loop *loss*; contested for offline evaluation in general | ⭐⭐ **YES — CONFIRMS-US.** It is `EVAL_DOCTRINE.md` in an opponent's mouth. ⚠️ And it is a **concession about Tesla's own training objective** (imitation loss) | an open-loop loss shown to rank closed-loop outcomes at ρ ≥ 0.8 across architectures |
| **T-18-3** | End-to-end is *"the only scalable solution"*; the modular approach cannot *"codify human values"* | ⛔ **NO** — anecdotes (puddle vs oncoming lane; geese) | Wayve, Tesla deployments (M-1 row) | **Waymo W-13** (hybrid, structured intermediates) · **Mobileye M-1** · **DriveZero** itself **decomposes** into perception + action models (today) | **CONTESTED** — same state as M-1; no side has a matched ablation | ⚠️ **PARTIALLY** — our hierarchy is neither pure E2E nor modular; the claim does not meet our design. Row 18 (H1b) is the experiment | a matched-params E2E vs hybrid ablation on a shared benchmark |

### Guidelines

* **S-7 (strategic)** — *"Train large, deploy small" is now the stated doctrine of Waymo **and** Tesla. Our claim is not that distillation works; it is that a **sub-300 M model trained as a world model** needs no giant teacher.* **Falsifier:** a controlled result where our student distilled from a large teacher beats our directly-trained model at matched deploy params. ⇒ that ablation is worth one v7-tiny pair.
* **T-8 (tactical)** — *if we distil (DriveZero-style privileged teacher, or AlpaSim-trained teacher), budget a **teacher-assistant** step when teacher/student params exceed ~10×* (TAKD). **Falsifier:** a direct large→small run matching the TA route at our scale.
* **T-9 (tactical)** — *quote T-18-2 in the paper's evaluation section as opponent corroboration of the tier doctrine, **labelled RELAYED*** until a transcript primary is banked.

### ⭐ Opponent STRENGTHS recorded (required)

9. **Tesla ships a distilled student onto hardware with ~15 % of the teacher platform's memory bandwidth**, at fleet scale. That is deployment competence we have not demonstrated on Thor.
10. **Tesla runs a closed-loop neural world simulator for evaluation *and* RL at fleet-data scale.** We have AlpaSim unprovisioned and T1 as our primary offline tier.
11. **DriveZero** (academic, not an opponent) **proves a frozen-VFM + privileged-teacher recipe reaches the navhard top** — the bar G3 now has to clear.

### ⛔ Register bookkeeping correction (found today)

The 09-17 debt table lists **D-2 and D-3 as "STANDS, not reached"**. But `OPPONENT_CLAIMS_REGISTER.md:74-75` recorded both as **DISCHARGED** (W-11…W-15, *Demonstrably Safe AI* read in full). The 09-17 table re-used the ids for the same documents, and today's primary fetch confirms the post's claims match W-13 exactly. ⇒ **D-2 and D-3 are CLOSED**; D-3's residual (*"a dedicated Foundation Model read is not owed unless a claim depends on it"*) stands. Recorded as a correction entry, not a rewrite.

---

## 2 · DEEP items — five dimensions

### A1 / A3 / B5 · DriveZero `2609.06055` ⭐⭐⭐
| dim | |
|---|---|
| RELEVANCE | AI13-2 (freeze), refav1 RL line, G3, row 11 |
| CONSEQUENCE | The navhard bar for G3 moves to **57.1** (two-stage, ego-consuming). Our stamp table must add it with **both** stamps (split + ego-arm), per V-5 |
| COMBINATION | Frozen-VFM consolidation (+0.53, ViT-S) is small next to FROST-Drive's width lever (+0.49 RFS, 09-17), and both point at the **readout/width** rather than the encoder (LR14-4). The privileged teacher gives our RL line a route that never does RL on the vision student |
| CHANCES / RISKS | Upside: a published recipe from frozen parts. Risks: **no CIs, no seeds** in what we read; VFM ablation at ViT-S only; ego at inference; latency unreported (E-DZ-latency) |
| EXPERIMENT | **FS18-1 (0 GPU):** add DriveZero to the navhard stamp table as `57.1 · two-stage · ego+nav @ inference · benchmark-arm only`, then re-state G3's gap. **FS18-2 (v7-tiny):** a privileged-teacher (obstacle.offline + ego future) distillation arm vs the direct arm at matched student params. **Committed:** the teacher arm must beat direct by more than the replicate floor, or the route is closed for us |

### A2 · Delta-JEPA `2606.31232` (FULL TEXT) → see `Architecture & Inference/Research/2026-09-18-ldad-displacement-probe/RESULT.md` §2.

### A4 · VLA visual grounding `2605.31041` ⭐⭐
| dim | |
|---|---|
| RELEVANCE | EVAL_DOCTRINE tiers; vision-only rule; the action-echo history |
| CONSEQUENCE | An independent **controlled** measurement that open-loop scores can barely see whether the model uses its eyes (image removal +7.1 % L2). That is the same pathology as our open-loop echo |
| COMBINATION | ⭐ Gives us a **cheap audit to run on our own arms**: an image-removal / image-shuffle intervention at T1. If our T1 moves as little as their open-loop score, T1 has the same blindness |
| CHANCES / RISKS | Upside: a vision-dependence audit we can run offline. Risk: one 3B model, nuScenes open loop; NeuroNCAP is its own sim |
| EXPERIMENT | **FS18-3 (0 GPU on banked dumps, or 4060):** at T1, replace the trunk tokens with (a) the clip's mean token, (b) a temporally shuffled token. **Committed:** T1 ADE degrades **< 10 %** ⇒ T1 is vision-blind like their open-loop metric, so no T1 result may be read as a vision claim; **≥ 30 %** ⇒ T1 does see vision |

### A5 · Admissibility ladder `2607.07196` (FULL TEXT) → see `Benchmarks & Evals/Research/2026-09-18-ladder-l1-row14/RESULT.md`.

### B7 · FlowR2A `2606.24231` (FULL TEXT) ⭐⭐ — rotation debt, missed twice, **discharged**
| dim | |
|---|---|
| RELEVANCE | LR14-5 (flow predictor), the selection/vocabulary line, D-TAC (tactical head), FS9-5 |
| CONSEQUENCE | Learning `p(a \| r)` from dense (trajectory, reward) pairs: **92.8 PDMS navtest v1 / 88.9 EPDMS navtest v2** (⛔ **navtest, one-stage** — not our navhard table). Per-timestep reward conditioning lifts TTC **88.8 → 94.9**; latency **91.3 ms** on an H20 (denoising 70.2 ms; K = 10 gives 53.2 ms at −0.6 PDMS) |
| COMBINATION | ⭐⭐ **Its reward-noise ablation is our nav-echo defect in another costume:** without noise *"the decoder treats rewards as trajectory identifiers rather than quality indicators"*, and PDMS collapses at high target reward (84.8 at σ = 0 → 89.4 at σ = 0.05). A conditioning signal that **identifies** the target is a leak, whether it is a route or a reward. ⇒ Any reward/goal-conditioned head we build needs a **noise-on-condition** arm by default |
| CHANCES / RISKS | Upside: a generative decoder that absorbs hard constraints (safety saturates at 1 proposal). ⛔ Risk: it **requires a per-candidate reward oracle** (NAVSIM's rule simulator). PhysicalAI has **no map**, so DAC/DDC/lane terms cannot be computed. We could label **collision/TTC only** (obstacle.offline), a partial reward |
| EXPERIMENT | **FS18-4 (0 GPU):** price the partial-reward route. Over the vocabulary on the parity corpus, compute per-candidate collision + TTC from `obstacle.offline`. **Committed:** if ≥ 30 % of windows have a non-trivial reward spread (not all-safe), a TTC-only FlowR2A arm is worth a v7-tiny slot; below that, the reward is too sparse on our corpus and the route is closed |

### B11 · Latent Generative Solver `2602.11229` ⭐⭐⭐ (see finding 1)
| dim | |
|---|---|
| RELEVANCE | LR14-5, ASK-1 (k=60 BPTT divergence), W-17-3 (Waymo concedes long rollouts unsolved), our 6 s horizon |
| CONSEQUENCE | Two separable levers with measured, isolated effects: **generative transition (large)** and **input noising (moderate, with a stability bound)** |
| COMBINATION | ⭐ Input noising is **zero-architecture**: `x̃ = (1−k)x + kz` on the context latent at train time. It is also exactly the recipe of diffusion-forcing/scheduled-sampling families. It attacks our drift where it lives |
| CHANCES / RISKS | Upside: cheap and theory-backed. Risks: PDEs, not driving; 2-D; *"isotropic Gaussian noising may underperform at multimodal bifurcation points"* — **which is what a junction is** |
| EXPERIMENT | **FS18-5 (2 v7-tiny arms + replicate):** k ∈ {0, 0.1} on the O5 predictor's context input. **Committed:** success iff T1 h=6 s ADE improves beyond the replicate floor **while** h=1 degrades ≤ 16 % (LGS's price). T1 flat ⇒ noise is not our lever. Controls per `H-ESTIM-SEED-1` |

### C2 · TensorRT 11 (see finding 5) — row 1 consequence stated; release notes are `PUBLISHED-RELEASE-NOTE`, read at `docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/{11.0.0,11.3.0}.html` on 2026-09-18 (not bankable as PDF; recorded here with retrieval date).

---

## 3 · Band C sweep (short)

C1: Wayve × Uber supervised London rides (15 Mustangs, 2026-09); no new doctrine. C3: NHTSA AV-Framework interim-guidance comments **extended to 2026-09-30**. C4: navhard leaderboard stub lists DriveZero-Scale (2026-09-05), SimWAM `2608.07468` (2026-08-07); ⭐ DriveVLA-M0 `2608.10413` reports **47.0 EPDMS navhard** — below DriveZero and inside our table's range. MM standing question: **no new posted-limit supplier found** (E-MMQ); today's `E-DE-SIGN-3` gives the in-corpus side a sourced count (**43 valued, not 31**).

## 4 · ⛔ Coverage honesty

**22/22 tracks carry a named query.** DEEP: **D** (Tesla ×3 + correction), **A1/A3** (DriveZero), **A2** (Delta-JEPA, full text), **A4** (`2605.31041`, the pre-committed A4 deep), **A5** (ladder, full text), **B5** (DriveZero's DriveRL), **B7** (FlowR2A, full text), **B11** (LGS), **C2** (TRT 11). **Band-B deep: 3 dedicated (B7, B11, B5)**, clearing the ≥ 3 bar without the transfer-number crutch 09-17 had to declare.
⚠️ **Where depth is thinner than it looks:** DriveZero and LGS were read through the fetch summariser from HTML (numbers quoted are the summariser's extraction of the primary; the PDFs are banked, not locally re-read). Tesla T-18-2/-3 are **RELAYED**. B4/B6/B10/B12 broke their empty streaks but are **scan-only**.

## 5 · Next rotation — pre-committed

1. **B12 DEEP** (DeepSight `2605.10564`), now that the empty streak is broken, for the 6 s vs 30 s horizon question.
2. **B10 DEEP** (DriveVLA-M0 `2608.10413`, failure-aware memory via LoRA test-time training) — touches I-3.
3. **Re-read DriveZero + LGS from the banked PDFs locally** (pypdf) to replace summariser extraction for any number that enters the registry.
4. **Elluswamy primary** (ICCV-25 WDFM-AD recording/transcript) to lift T-18-2/-3 off RELAYED.
5. **B4 DEEP** (Muon-in-ViT `2605.24770`) only if an efficiency row is live.
