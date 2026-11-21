<title>LEDGER A1 — world models and world-action models</title>

# LEDGER A1 — World models + world-action models

⛔ **APPEND-ONLY.** Never rewrite an entry. Corrections are added as dated correction entries below
the original, `RETRACTION_LOG` discipline. This file is the accumulating state of the art in this
track **with our position in it** — the daily `RESULT.md` is a snapshot, this is the memory.

---

## Our position in this track (restated each entry; edit only by appending a new "our position" block)

**2026-08-31.** Sub-300M hierarchical 4-brain latent world model. Objective family: layered
prediction against per-layer stop-grad/EMA targets (`train_v6_staged.py:48`), i.e. **teacher-forced
in the UWM-JEPA sense**. Measured defect: **h=1 action/scene ratio 0.00408 / 0.00416 / 0.00595**
across three 30k arms (MM-E10) — action-deafness **5–7× deeper** than the 0.03 published for the
same failure mode. No arm has ever been evaluated at **T1** (action-closed loop).

---

## Entry 2026-08-31-01 — the leaders kept the realised-motion action channel

**Sources:** lib `2605.09701` (DriveFuture, full text) · lib `2602.06521` (DriveWorld-VLA, full text).
**Evidence class:** PUBLISHED (both banked, both read in full).

Both current NAVSIM leaders condition on **realised ego motion**, the same channel we have been
treating as our defect:

- DriveFuture: *"a normalized differential representation (Δx, Δy, sin θ, cos θ), a linear
  projection, and a temporal positional embedding"*.
- DriveWorld-VLA: *"Historical ego actions are serialized into natural language prompts and
  concatenated with textual instructions"*, with a teacher-forced target — *"we reuse the Stage 1
  encoding pipeline to obtain the corresponding ground truth (GT) BEV latent representation"*.

⇒ **Neither bought its way out with a better command.** Both spent the design budget on the
conditioning schedule and the target, which is where our own mechanism evidence points.

**Two mitigations they run and we do not:**

| mechanism | source | what it does |
|---|---|---|
| **LatentAlign** | DriveFuture | *"a sigmoid-based annealing schedule that smoothly transitions the planning condition from the grounded future latent to the self-predicted forecast over the course of training"*, `α(e) = 1 − σ(β(e−e₀))` |
| **action-source mixture** | DriveFuture | training trajectory drawn from `{p_gt, p_kin, p_∅}` — ground truth · constant-acceleration kinematic extrapolation · **a learned NULL token** |

**Numbers (their protocol, their measurement):**

| item | value | note |
|---|---|---|
| DriveFuture NAVSIM-v1 navtest PDMS | 90.7 | |
| DriveFuture NAVSIM-v2 navtest EPDMS | 89.9 | |
| DriveFuture NAVSIM-v2 navhard EPDMS | **55.5** | ⛔ **see correction-watch CW-1 below** |
| DriveFuture ablation, GT-grounding on/off | 32.1 → **34.6** EPDMS | navhard *ablation setting* |
| DriveFuture schedule sensitivity `e₀` | 0.75 / 0.83 / 0.95 → 29.2 / **34.6** / 28.9 | ±0.12 costs ~5.5 EPDMS |
| DriveWorld-VLA NAVSIMv1 PDMS | 91.3 | |
| DriveWorld-VLA NAVSIMv2 EPDMS | 86.8 | |
| DriveWorld-VLA progressive vs non-progressive training | 83.6 → **91.3** PDMS | **+7.7 from training strategy alone** |
| DriveWorld-VLA task-only vs task+feature supervision | 87.9 → **91.3** PDMS | +3.4 |
| DriveWorld-VLA training cost | 8× H20, ~120 h (NAVSIMv1) | |

⚠️ **CW-1 — OPEN CORRECTION-WATCH.** DriveFuture's headline navhard **55.5** cannot be reconciled
with its own navhard ablation rows (**30.9–34.6**) from the text. **No DriveFuture navhard *level*
may enter a comparability table until resolved.** The ablation *deltas* are usable (same table,
same setting). Resolve by reading the leaderboard submission config.

⚠️ **NAVSIM comparability caveat inherited from `2026-08-31-frontier-action-provenance-sweep`:** the
scoring basis moved on 2025-04-28 and 2025-09-29. An EPDMS number is comparable only to another on
the same side of both dates. None of the above has been checked against those dates yet.

**What it changes for us:** raises the priority of an objective/schedule arm over a
command-channel arm on the V7 gate. See `Daily/2026-08-31/RESULT.md` F1/F2 for the five-dimension
analysis and the two pre-registered experiments.

---

## Entry 2026-08-31-02 — scale does not move the trajectory

**Source:** HF model cards `nvidia/Alpamayo-1.5-10B`, `nvidia/Alpamayo2-Super`.
**Evidence class:** PUBLISHED (vendor model cards, fetched 2026-08-31).

| model | params | LingoQA | AlpaSim closed-loop | open-loop minADE₆ @6.4 s |
|---|---:|---:|---:|---:|
| Alpamayo-1.5 | ≈10.5 B | 74.2 | 1.37 ± 0.10 | 0.916 m |
| Alpamayo2-Super | ≈34.3 B | 79.2 | 1.50 ± 0.13 (913 scen.) | 0.911 m (1434 samples) |

**3.3× the parameters moves open-loop trajectory by 0.5 %; the stated closed-loop intervals
OVERLAP; the language score moves most (+5.0).**

⛔ **`minADE₆` is BEST-OF-6, not ADE — never place it in a column with our `fwd_ade`.**

**What it changes for us:** the strongest external support the sub-300M thesis has received, and
independent corroboration of the `EVAL_DOCTRINE` T0/T1 split — an open-loop metric that cannot
separate a 3.3× parameter gap is a diagnostic, not a capability measure.

`Next in this track: resolve CW-1; check whether Alpamayo's eval split intersects our parity split e438721ae894.`


---

## Entry 2026-09-01-01 - the Waymo World Model is a SIMULATOR, and it concedes long-horizon instability

**Source:** `waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/`
**Evidence class:** PUBLISHED-BLOG (primary fetched 2026-09-01). **Discharges register debt D-3.**

The highest-value unread document known to the programme is read. It is **built upon Genie 3**
("Google DeepMind's most advanced general-purpose world model") and **adapted via "specialized
post-training"**. Its role is **simulation**, one of "three key pillars"; it is **not** claimed to
drive the car.

> **THE CONCESSION:** *"longer scenes...is harder to do because the longer the simulation, the
> tougher it is to compute and maintain stable quality."*

**ZERO ablations and ZERO metrics** appear in the document; its only number (200 M autonomous miles)
is an exposure figure unrelated to the world model.

**What it changes for us:** their WM and ours are **different products** - theirs renders futures for
training/testing, ours predicts futures inside the driving loop. Their announcement therefore does
not refute our thesis. ⭐ And their concession is **our measured problem** (drift 0.4531 -> 0.36-0.40
under EMA, MM-E1): **with Genie 3 and Google-scale compute, long-horizon rollout stability is still
unsolved.** Independently corroborated - see the B13/A5 line and `LEDGER_C1_releases.md`.
Same shape as the B13 reframing: **a property of the field, not our bug.**

`Next in this track: score our rollouts on a segment-based (not start-vs-end) drift metric - see A5.`

## Entry 2026-09-01-02 - a latent-WM taxonomy that names the open-loop/closed-loop mismatch

**Source:** `arXiv 2603.09086` (Zeng, Dong; 2026-03-10). **Evidence class:** PUBLISHED lib 2603.09086
**abstract-only - DECLARED, may not decide a GPU-day.**

Organises latent world models by target/form (latent worlds, latent actions, latent generators),
representation (continuous / discrete tokens / hybrid) and structural priors (geometry, topology,
semantics). Proposes "a closed-loop metric suite and a resource-aware deliberation cost, designed to
reduce the open-loop / closed-loop mismatch".

**What it changes for us:** a third external line converging on `EVAL_DOCTRINE`'s T0/T1 split, and a
ready-made vocabulary for describing our own latent design. **No comparative numbers in the abstract.**

`Next in this track: full-text read to extract the closed-loop metric suite definition.`

---

## 2026-09-09-01 - Latent-WAM: a 104M world-action model, and a parameter-count unit trap

`lib 2603.24581. FULL TEXT results section. Class PUBLISHED. Retrieved 2026-09-09.`

Two modules: a **Spatial-Aware Compressive World Encoder** that *"distills geometric knowledge from a
foundation model and compresses multi-view images into compact scene tokens via learnable queries"*,
and a **Dynamic Latent World Model**, *"a causal Transformer to autoregressively predict future world
status conditioned on historical visual and motion representations."*

**The parameter statement, verbatim, and the trap in it:** *"The model contains **104M parameters at
inference time**. During training, an additional EMA encoder is introduced for self-supervised learning,
**bringing the total to 191M, of which only 104M are trainable**."*

**Guideline T-5 derived: every parameter count in a comparability table carries INFERENCE-TIME or
TRAINING-TIME**, exactly as every metric carries its eval tier. Backlog row 32's "beat ~40 M" premise
rests on a DrivoR count that is (a) unverified (L-15) and (b) probably inference-time.

**Relevant to backlog row 4 / MM-E1:** they run *"a frozen SCWE updated via Exponential Moving Average"*
as a self-supervised target while the primary backbone stays trainable - the same teacher-target shape
our EMA question concerns, at 104M, in driving.

**Not quotable yet:** the 89.3 EPDMS is under D-10's bar (see `LEDGER_A5_benchmarks.md` 2026-09-09-01)
and may not enter any comparability table until its split is stamped.

## 2026-09-10-01 — Drive-HWM: the hierarchy edge, priced and unmatched

`FULL TEXT` · arXiv **2609.03572v1** (retrieved 2026-09-10) · Fan, Zhang, Wu, Wang, Jin, Zhao, Zhu, Yan · submitted 2026-09-03.
Hierarchical slow–fast WM. **Slow:** multi-step future representations, Dynamic-Aware Latents via optical-flow prediction, runs once per N=8 ticks. **Fast:** lightweight multimodal backbone + autoregressive expert, next frame + immediate action, every tick.

**Table IV (ablation), verbatim:** Fast only NC 99.3 / DAC 97.4 / **PDMS 93.0** · Slow only 98.2 / 97.1 / **90.2** · Drive-HWM (K=8) 99.6 / 99.0 / **93.8**.
**Table III (latency), verbatim:** DriveVLA-W0 Tf 117.8, Tavg 117.8, PDMS 93.0 · Fast Model Only Tf 81.6, Tavg 81.6, PDMS 93.0 · Drive-HWM N=8, Ts 25.6, Tf 81.6, **Tpeak 107.2**, Tavg 84.8, PDMS 93.8.
**NAVSIM v2:** EPDMS **86.4**, *"best or tied-best performance on seven of the ten metrics"* — ⭐ stamped **navtest** population (LEDGER_A5 2026-09-10-01), **not** navhard.

⛔ **The paper states NO parameter matching** between "Fast only" / "Slow only" and the full model. The +0.8 PDMS therefore confounds *hierarchy* with *capacity*. **This is an existence result, not an ablation** — the Mobileye M-1/M-2 shape GS-5 named. ⇒ **backlog row 18 (H1b) is NOT answered; it is sharpened and re-specified as params-matched.**

⚠️ **V-5 unit refusal:** `N = K = 8` with **no seconds-per-timestep given**. Their K=8 is NOT corroboration of our K=8 (16 s) rung in backlog P-17. Do not pair them.

⭐ Two readings of +0.8 PDMS, both to be carried: absolute **+0.86 % relative**; against remaining headroom **0.8 / 7.0 = 11.4 %**.

**Conceded limitation, verbatim:** *"the current model does not explicitly capture multimodal futures or predictive uncertainty, which may limit its performance in ambiguous and rare driving scenarios."* ⭐ Our REF-C fan and the B7 line address exactly this — a differentiator named by the opponent.

→ `Architecture & Inference/Research/2026-09-10-hierarchy-edge-external-datapoint/RESULT.md`


## 2026-09-13-01 — HWM: the capacity-controlled hierarchy result, on a FROZEN encoder

`FULL TEXT` · arXiv **2604.03208v2** (16 Jun 2026; banked 2026-08-27, read 2026-09-13) · Zhang, Terver, Zholus, Chitnis, Sutaria, Assran, Balestriero, Bar, Bardes, LeCun†, Ballas† (FAIR/NYU).
Two latent WMs at different temporal scales in ONE shared latent space; the high-level model's first predicted latent is the low-level MPC's subgoal (latent matching); learned macro-action encoder (dim 4).

**Table 16 (App. D.2), values:** Push-T flat 44M 55 % / 17 % · flat **98M 35 % / 15 %** · hierarchy **94M 78 % / 61 %** (H=10/15). Diverse Maze flat 54k 85/63 · flat **178k 82/59** · hierarchy **182k 95/83**. *"increasing the capacity of single-level models does not improve performance—in fact, it often degrades it"*.
**Table 2:** Push-T d=25/50/75 — DINO-WM 84/55/17, hierarchy 89/78/61. Frozen DINOv2 encoder at both levels (25M low / 75M high predictor).

⭐ **Our position:** the capacity control row 18 / H1b lacked now exists — **outside driving**. With Drive-HWM (2026-09-10-01: driving, NOT matched) the field holds one controlled non-driving positive and one uncontrolled driving positive; **the controlled driving cell is empty and is ours to fill.**
⛔ **Not settled:** Table 16 has no interval and no seed count; matching is ±4 % (flat larger on Push-T); the high-level model also gets stride-10 waypoint data and a smaller search space (capacity is controlled, data is not); the 98M flat model degrading hints at an untuned baseline.
⭐ **The first lever this month that works on a frozen encoder.**

→ `Architecture & Inference/Research/2026-09-13-hwm-capacity-matched-hierarchy/RESULT.md`

**Scan, same pass:** SV-WAM `2609.03602` (surround-view world-action model, future-video supervision for actions — unread) · `2609.03225` (long-horizon interaction-aware WM + GRPO, multi-style driving — unread).


---

## 2026-09-17 (LAB-RUN-014) — ⭐⭐⭐ the first controlled ablation on whether TEST-TIME imagination is needed at all

**Fast-WAM `2603.16666` — FULL TEXT.** *"do WAMs need to imagine future observations at test time, or
do they benefit primarily from learning to model them during training?"*

Three variants share training and differ only at inference, plus a direct control that removes the
video objective:

| variant | RoboTwin avg | LIBERO avg | latency |
|---|---|---|---|
| Fast-WAM (no test-time future) | **91.8** | **97.6** | **190 ms** |
| Fast-WAM-Joint (shared attention during denoising) | 90.6 | 98.5 | — |
| Fast-WAM-IDM (generate video, then act) | 91.3 | 98.0 | 810 ms |
| ⛔ Fast-WAM **w/o video co-training** (the control) | **83.8** | **93.5** | — |

Also: LingBot-VA 92.2, Motus 87.8 (pretrained baselines). Real-world towel-folding: the no-co-training
arm drops to **10 %**. 6 B total (5 B video DiT from Wan2.2 + 1 B action expert), action horizon 32,
10 denoising steps, CFG 1.0. LIBERO 2,000 trials / 40 tasks; RoboTwin 100 trials per task.

⭐ **Our position.** This splits the programme's thesis into two claims that must now be evidenced
separately: *(a) a world model is the right thing to TRAIN* — **supported**, and it is the paper's
sharpest number; *(b) rolling it at INFERENCE is what makes the planner good* — **no measurable
benefit found**. Our planner rolls at inference. ⭐ It combines with our own measurements rather than
contradicting them: at an h=1 action/scene ratio of **0.004** a rollout is close to a constant
unrolled, and dropping such a rollout should cost nothing — which is what they measure.

⛔ **Not settled.** No driving experiment exists in the paper (LIBERO / RoboTwin / towel-folding are
tabletop manipulation); **no std and no seed count** for a 91.8-vs-91.3 comparison; the authors
*"omit the outer auto-regressive loop"*, i.e. exactly the long-horizon regime driving needs; and they
concede *"pretrained pi-0.5 remains the strongest method"* on the real-world task. **Transfer to
driving is a HYPOTHESIS**, and `E-AI-NOIMAG-1` (0 GPU, banked dumps) is registered to test it on us.

⚠️ **Correction watch:** this ledger's 09-13 entry credits the hierarchy result (HWM Table 16) to
capacity control. Nothing today touches that. What today touches is the **inference-time rollout**,
which is a different axis and must not be merged with it.

→ `Frontier Scan/Daily/2026-09-17/RESULT.md` FS17-1 · `Architecture & Inference/Research/2026-09-17-frozen-trunk-external-evidence/RESULT.md`

**Scan, same pass:** PWM `2510.19654` (unify world modelling + planning) · GraphWorld `2606.16274` ·
UniDrive-WM `2601.04453` · DriveWAM `2605.28544` · CoWorld-VLA `2605.10426` · DAWN `2605.11550` ·
`2608.24882` (latent action as intention) · `2605.06222` (when to trust imagination) — all unread.

## 2026-09-18-01 — DriveZero: frozen VFMs + a privileged RL teacher reach navhard 57.1

`HTML primary via summariser` · arXiv **2609.06055** (2026-09-05) · banked. navhard **two-stage 57.1 EPDMS** (Scale) / 51.5 (base); navtest 95.3 / 94.8 PDMS; HUGSIM zero-shot 46.6 HD-Score. Frozen DINOv3+SigLIP2+SAM+DA-V2 (+LoRA): **+0.53 PDMS** over DINOv3-only (ViT-S). 5.70 M-param privileged PPO teacher (DriveRL) distilled into the vision student; RL-only 93.61 < human 93.92; + goal augmentation 94.41. ⛔ ego kinematics + nav at inference ⇒ benchmark-arm only.
**Our position:** above our stamp table's top (56.6). The privileged teacher is **admissible for us as a label factory** (labels may use anything). Experiments FS18-1 (stamp), FS18-2 (teacher-distillation arm). Also scanned: IDOL `2605.31476` (banked abstract-only — inverse dynamics on adjacent predicted BEV latents).

## 2026-09-19-01 — The TARGET beats the pixels, and a backdoored WM checkpoint passes clean checks

`FULL TEXT (local pypdf)` DeepSight **2605.10564** · `HTML via summariser` World Tokens **2608.09730** · `abstract-only` **2609.15781** · all banked. Detail: `Frontier Scan/Daily/2026-09-19/RESULT.md` §0 rows 2–3, §2.
* **Two independent controlled ablations:** DeepSight T3 (Dev-10, one seed) VAE-codebook target **27.75 / 14.66 DS** (1 / 5 frames) vs DINOv3-feature target **74.79 / 86.57**; World Tokens T2 (LIBERO-Long) RGB anchor **91.5 < no-WM 92.8 < full 97.0**.
* **Backdoor:** a poisoned latent-WM checkpoint steers the victim's own MPC/actor optimisation to the attacker's action on **100 %** of triggered steps, with **≥ ~75 %** clean success, so it passes pre-deployment diagnostics.
**Our position:** CONFIRMS the latent/semantic-target choice (T-12) and adjudicates N-19-2 (NVIDIA, pixel coupling) **UNSUPPORTED-AS-STATED**. New hygiene item: pin the sha256 of every upstream checkpoint (DINOv3, SAM3, Cosmos, Alpamayo) in run records. Experiment FS19-3.

### 2026-09-20-01 — Drive-HWM `2609.03572` (full text, local pypdf) ⭐⭐⭐

**The hierarchy edge has a published matched number for the first time, and it is small.** Table IV,
NAVSIM v1 PDMS: `Fast only` (flat) **93.0** · `Slow only` **90.2** · hierarchical **K=4 93.0 · K=8 93.8 · K=12 93.2**.
⇒ +0.8 at one horizon, **+0.0 at K=4**, +0.2 at K=12; one run per row, **no CIs**.
Authors' reading of `Slow only`: *"long-horizon predictions alone cannot promptly adapt to newly observed changes."*
Table VII (slow→fast conditioning): **FiLM 93.8** > AdaLN 93.3 > GCA 93.1 > cross-attn 93.0 > concat 92.5
⇒ published support for the FiLM conditioning the v7 predictor already uses.
**Class:** PUBLISHED, numbers re-read locally ⇒ registry-admissible.
**Moves:** backlog row 18 (H1b) — the thesis gains an external datapoint and loses expected effect size.
**Source:** Daily/2026-09-20/RESULT.md
