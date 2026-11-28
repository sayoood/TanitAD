# Latent action models and modern action-world-model research — what TanitAD should adopt

**Date:** 2026-07-27 · **Agent:** `latent-action-research` · **Hosts touched:** NONE (CPU + web only;
pod1/pod2/pod3/eval pod untouched, per brief).
**Reads with:** `…/Implementation/incoming/2026-07-26-idm-v2/IDM_V2_RESULTS.md` (all "ours" numbers),
`…/Implementation/incoming/2026-07-27-v4-instrument/V4_INSTRUMENT.md` (the `v0` shortcut),
`CITATIONS.md` in this directory (per-row citation depth).

**Headline for the sibling `idm-v3` stream — three things, in this order:**

1. **The full Genie/LAPA-style latent-action stack is NOT worth adopting as a replacement for our IDM.**
   Two of my four pre-registered refutation criteria fired (§6). A clean "stick with a supervised IDM"
   is the verdict on the *stack*.
2. **Three separable mechanisms ARE worth adopting**, and after reading the sibling's staged v3 work I
   have re-ranked them (§7):
   - ⭐ **A1 — the single most valuable hand-off, and it is nearly free.** The sibling's v3 endpoint E5
     already classifies `long_accel` over **21 uniform bins with a hard target**. **Farebrother et al.
     measure that exactly this family (Two-Hot) UNDERPERFORMS MSE**, and that the active ingredient is
     **Gaussian label smoothing (HL-Gauss, σ/ς = 0.75)**, not the categorical output. Adding smoothing
     and non-uniform (symexp/foveated) bins is a few lines on an arm already running.
   - ⭐ **A3b — the top NEW proposal.** Distil a **feedforward geometry teacher** for ego motion instead
     of regressing it with a geometry-blind head (LFG: **81.4 PDMS at 10 % labels vs 85.2 at 100 %**).
   - 🔴 **A3 is WITHDRAWN.** I had recommended camera-height scale grounding; **the sibling has already
     measured it and it fails** — shuffled heights work as well as real ones, and the oracle scale
     correlates with height at the **opposite sign**. I am reporting that instead of handing over a
     dead design.
3. **The answer to the PI's crux question is NO, with a precise replacement claim** (§2). A latent
   action does not *sidestep* metric-scale ambiguity. It **separates** the scale-free part of ego
   motion (which monocular video determines) from the one scalar it cannot determine, and relocates
   that scalar into the grounding head. That is worth something — but it is a different claim, and I
   nearly imported the stronger one from a fabricated quotation (§8).

⚠️ **Read §8 before quoting anything in this report.** An automated PDF-extraction hop **invented** a
verbatim quotation that would have settled the PI's crux question by citation. I caught it by
re-reading the same paper at HTML depth. Two independent fabrications from that one path.

---

## 1. Pre-registration — what would make me say "do not adopt"

*Fixed before the full-text literature pass. At the time of writing only our own artifacts and the
four paper **abstracts** had been read, and those abstracts contain none of the facts these criteria
turn on. Not revised afterwards; §6 scores them as written.*

| # | Refutation criterion — if TRUE, the latent-action direction is NOT for us |
|---|---|
| **R1** | The published gains rest on action spaces **structurally unlike continuous vehicle control** — i.e. small discrete/categorical action sets — **and** no paper demonstrates a latent action recovering a *continuous, metrically-scaled, near-zero-mean* quantity of the kind `long_accel` is. |
| **R2** | The **grounding sets are large** relative to our CAN-labeled holdings, i.e. the "big unlabeled + small labeled" asymmetry we actually have is not the asymmetry the method exploits. |
| **R3** | The latent action is **structurally unable to carry metric scale** *and* we have no independent geometric route to supply it — in which case latent actions buy nothing on `speed` either, and direct regression + geometry wins outright. |
| **R4** | The published methods' benefit derives from **suppressing** ego/camera motion (the inverse of our problem) and **no paper demonstrates the extraction direction on driving** — in which case the evidence base is about a different problem and may not transfer. |

**Committed in advance:** if R1 or R4 fires, the maximum I may recommend is a **CPU/sub-GPU-day probe**,
never a training commitment. If R2 fires, I recommend nothing. If R3 fires fully, I write
"stick with direct regression" and stop.

---

## 2. ⭐ THE CRUX: does a latent action sidestep metric-scale ambiguity?

**Verdict: NO. Tier — DECISION-GRADE for the "no", PROVISIONAL for the replacement claim.**

### 2.1 The geometric fact

`PUBLISHED (multi-view geometry; operationalised in StableCamH, arXiv:2312.04530, full text)`
From a single moving camera, inter-frame image motion is determined by rotation **R** (fully
observable, and **scale-free** — a rotation induces the same flow field regardless of scene scale)
and by translation **t** only through the ratio **t/Z** (translation over depth). Metric |t| is not
recoverable without an external scale reference. StableCamH exists precisely because of this: it
recovers metric scale not from the images alone but from an **invariance** (§7 A3).

### 2.2 Our own measurements have exactly this shape — and I want to flag that as a *hypothesis*

`MEASURED (ours — IDM_V2_RESULTS.md §3.2, §3.3, §5)`

| our channel | physical type | our number | consistent with? |
|---|---|---|---|
| `yaw_rate` (PhysicalAI) | rotation → **scale-free** | R² **0.9035** | recoverable from monocular video ✓ |
| `speed` | translation → **scale-ambiguous** | R² 0.8651 pooled; after removing the clip-level term the **oracle** reaches **0.942**; the clip-level part is a *shrinkage toward the training prior*, **gain 0.830** | an unobservable per-clip scale factor, answered MMSE-optimally ✓ |
| `long_accel` | d²(translation)/dt² → **scale-ambiguous AND differentiated** | R² **−0.240** pooled (−0.298 PhysicalAI, −0.254 comma) | doubly cursed ✓ |
| `steer` | ≈ rotation/translation ratio | R² **0.742** (A0) | scale-free ✓ |

**The ordering is monotone in "how much metric scale the quantity requires".** That is a striking fit.

⚠️ **I am deliberately NOT calling this a finding.** This program logged a retraction where two numbers
moving together by 43.5 %/43.6 % looked like a mechanism and the counterfactual inverted the sign. This
is four channels ordering correctly under a story. There is at least one competing explanation I cannot
exclude: the four channels also order by **label quality** (yaw's PhysicalAI label is a quaternion;
`long_accel`'s label correlates only r = 0.434 with its own dv/dt). Label quality alone would produce
the same ordering.

🟢 **And the sibling's staged v3 work has already sharpened this in a way I did not expect.**
`MEASURED (…/incoming/2026-07-27-idm-v3/PRE_REGISTRATION_IDMV3.md §3, §4 E4)` — their derivation splits
the two channels *exactly* along the scale axis: **`v = (f·h)·Φ(image motion)`** but **`ω = (du/dt)/f`**
— yaw depends on focal alone, speed on `f·h`. Since our pipeline already canonicalises `f_eff ≈ 266 px`
on every corpus, **"the yaw channel is already geometry-matched and the speed channel is not."** That is
an independent, physics-derived version of §2.1's rotation-vs-translation split, and it makes the
scale-ambiguity reading of our channel ordering **substantially more credible than a coincidence.**

⚠️ **But their E4 also shows the scale factor is not sourced by camera height** (§7 A3): the oracle
per-clip scale is real and large (MAE **2.960 → 1.607**) yet correlates with camera height at
**r = −0.466 — the opposite of the ground-plane sign**, and **shuffled heights work as well as real
ones.** ⇒ **The decomposition survives; one candidate source for the scalar is dead.** Tier: the
decomposition is now **CONFIRMED-ish** (two independent derivations + the E3 discriminator pending);
the *source* of the scalar is **OPEN**.

### 2.3 So what does a latent action actually buy?

A latent action trained to compress frame-to-frame change is, by construction, a **projective**
quantity. It can carry {yaw, translation direction, t/Z} at full fidelity. It cannot carry m/s.

**⇒ Precise replacement claim (PROVISIONAL):** the latent action does not remove the scale problem; it
**isolates** it. What was a metric regression spread across four coupled channels becomes
(scale-free latent action) × (one scalar per clip). Two published routes then supply that scalar:

- **A small labeled set.** All full-text: **Genie** — "as few as **200** expert samples to adapt", and
  the LAM-based policy "achieves the same score as the oracle" that had ground-truth actions.
  **LAPO** — "a decoder trained on less than **256** labeled transitions matches the performance of a
  policy trained from scratch for 4M steps". **LAPA** — **150 trajectories per task** (450 total for 3
  real-world tasks); 100 trajectories on SIMPLER. **V-JEPA 2-AC** — **<62 h / 23k trajectories**.
  ⇒ **R2 does NOT fire.** We hold far more CAN-labeled data than any of these.
- **A geometric prior.** StableCamH, §7 A3.

### 2.4 The inversion that reframes the whole reading — and it cuts both ways

`PUBLISHED (UniVLA arXiv:2505.06111 full text; MVP-LAM arXiv:2602.03668 full text; "Why Latent Actions
Fail" arXiv:2605.20223 abstract verbatim)` — **CONFIRMED, three independent sources.**

The manipulation latent-action literature's central failure mode is that **latent actions absorb
camera/ego motion**:

- **UniVLA**: naive reconstruction objectives "capture task-irrelevant dynamics, such as movements of
  non-ego agents or unpredictable camera shifts." Their Table III quantifies the cost: task-centric
  latent actions **88.7 %** average success on LIBERO vs task-irrelevant **56.5 %**; on LIBERO-Long,
  **79.4 % vs 0.2 %**.
- **MVP-LAM**: "Viewpoint changes introduce camera movements and perspective shifts, entangling visual
  transitions with the agent's action."
- **"Why Latent Actions Fail"** (abstract, verbatim): "minimizing the standard reconstruction objective
  produces latent actions that encode exogenous information from future observation".

**For TanitAD the sign is inverted: ego motion IS the action.** The quantity the manipulation
literature spends its entire method budget suppressing is the quantity we want to extract. That is a
genuinely favourable structural argument for the mechanism.

**But it is also why the numbers do not transfer.** UniVLA's 88.7-vs-56.5 gap *is* the gain from
removing ego motion. Reading it as evidence that latent actions work for us would be reading a
control's improvement as the treatment's. **⇒ R4 FIRES.** The published evidence base for
latent actions is largely evidence about a different problem.

**The one paper on our side of the inversion is VPT — and VPT is not a latent-action model at all**
(§4). Our YouTube pipeline is already VPT. That is the correct reference class.

---

## 3. Thread 1 — latent action models, mechanism by mechanism

All rows `PUBLISHED`, full-text depth unless marked.

| | **Genie** (2402.15391) | **LAPO** (2312.10812) | **LAPA** (2410.11758) | **UniVLA** (2505.06111) |
|---|---|---|---|---|
| latent action space | VQ, **\|A\| = 8**, embed dim **32** | 128-d continuous → VQ; "2 codebooks × 4 discrete latents, each 64 embeddings of 16 dims" ⚠️ | **8⁴** default | **\|C\| = 16**, **N = 4** tokens (vs OpenVLA's 256⁷) |
| LAM sees the future frame? | **Yes** — encoder takes x₁:ₜ **and** xₜ₊₁ | **Yes** — IDM reads o_{t−k..t+1} | **Yes** — x_t and x_{t+H} | yes |
| what stops future-leakage | the VQ bottleneck | **the VQ bottleneck**: "the IDM learns to encode only the difference between o_{t+1} and o_t, rather than full information about o_{t+1}" | NSVQ + codebook replacement | codebook capacity + language conditioning |
| discarded at inference | "apart from the VQ codebook, the entire LAM is discarded at inference time" | latent policy kept, IDM discarded | — | — |
| grounding-set size | **200** expert samples | **< 256** labeled transitions | **150** traj/task | 20–80 traj/task; decoder **10.8 M** params |
| params | LAM **300 M**; tokenizer 200 M; dynamics 10.1 B | — | — | decoder 10.8 M + 12.6 M LoRA |
| unlabeled scale | **30,000 h** / 6.8 M clips, filtered from 55 M | ~8 M Procgen frames | 970 k (Open-X) + 220 k (SSv2) traj | — |

⚠️ The LAPO codebook row is **internally inconsistent in my extraction** (it reports both "quantized
into 8 discrete latents with 16-dimensional embeddings" and "2 codebooks with 4 discrete latents per
codebook, each with 64 embeddings of 16 dimensions"). Treat the exact shape as **UNVERIFIED**; the
load-bearing facts (VQ is the bottleneck; <256 labeled transitions) are stable across both readings.

**The R1 test.** Every action space above is **small and discrete**: 8, ~8, 8⁴, 16⁴. **No paper in
this thread demonstrates a latent action recovering a continuous, metrically-scaled scalar.** LAPO's
only continuous-control evidence is an appendix (§A.3, DMC) which my extraction describes as
providing "minimal analysis of performance differences" — i.e. thin, and I will not lean on it.
**⇒ R1 FIRES.**

**Contradicting evidence found, reported fairly:**
- LAPO's own limitations: "Actions that have a delayed effect in observations will be predicted to take
  place with the same delay, i.e. the latent policy actually models the visible effects of an action,
  not the action itself." **For driving this is severe** — throttle→acceleration→visible motion is
  exactly a delayed-effect chain, and `long_accel` is its most delayed member.
- LAPO: "Significant stochasticity can make it difficult for the IDM to compress the useful bits of
  information among the noise, degrading the quality of the latent representation." Ours is
  in-the-wild video with independent traffic.
- LAPA **underperforms** OpenVLA on pick-and-place (**50 % vs 66.67 %**) and collapses on
  cross-environment transfer (**33.6 % / 29.6 %** vs ActionVLA's 64.8 % / 54.0 %). Cross-domain transfer
  is precisely what we would be asking of a YouTube→PhysicalAI latent action.
- Genie's limitations: 16-frame memory, ~1 FPS, hallucinated futures.

---

## 4. Thread 3 — VPT: our actual reference class, and what the recipe really rests on

`PUBLISHED (arXiv:2206.11795, full text via ar5iv)` — **CONFIRMED.**

- **Non-causality confirmed verbatim:** "the IDM can be non-causal, meaning its prediction for a_t can
  be a function of both past and future events." 128-frame context; the first 3D conv has **temporal
  kernel width 5** (t−2 … t+2). ✅ Our clip-context token exploits the same licence, and IDM v2 measured
  it as the single change that carries most of the gain (+0.106 speed R² off B0).
- **What the recipe rests on — the parts we may be under-weighting:**
  - **IDM labeled data: 1,962 h** of contractor data. IDM accuracy: **90.6 %** keypress, **R² 0.97**
    mouse movement.
  - **"The IDM is two orders of magnitude more data efficient than a BC model trained on the same
    data."** This is the load-bearing justification for the whole pipeline and it is *ours to inherit*.
  - **Filtering is a first-class component**, not a detail: 270,000 h collected → **~70,000 h** after an
    SVM "clean video" classifier trained on **8,800 human-labeled frames**.
  - **Action discretisation is FACTORISED PER AXIS**: 13 independent binary key actions + **separate**
    11-bin camera-X and 11-bin camera-Y with *foveated* (non-uniform) binning — finer near zero.
  - IDMs trained on ≥100 h give usable pseudo-labels; gains **plateau after 100 h**.

**Two direct implications for us.** (a) Our IDM's `speed` R² 0.865 / `yaw` 0.90 is in the usable band by
VPT's own standard — the pipeline is not broken, one channel is. (b) **Foveated binning finer near
zero** is exactly the right prior for `long_accel`, whose mass is concentrated near zero. That is a
free design detail we are not using.

---

## 5. Thread 2 — ⭐ discrete vs continuous heads, and how to keep the axes separable

### 5.1 The case for classification on `long_accel`

`PUBLISHED (Farebrother et al., "Stop Regressing", arXiv:2403.03950, full text)` — **CONFIRMED.**

**HL-Gauss:** place m evenly-spaced bins over [v_min, v_max]; model the target as
`Y ~ N(μ = target, σ²)`; project that Gaussian onto the histogram; train with cross-entropy.
Tune **σ/ς** (std ÷ bin width), **default 0.75**, spreading mass over ~6 neighbouring bins.
Results: **~30 %** IQM improvement on Atari MoE scaling; **45 %** over C51 on 40-game multi-game
offline at ResNet-101; **40 %** on Wordle (125 M); **70 %** of the gap to Stockfish on chess (270 M);
**67 %** higher peak on robotic manipulation.

`PUBLISHED (DreamerV3, arXiv:2301.04104, full text)` — the mechanism, verbatim: the categorical loss
"only depends on the probabilities assigned to the bins but not on the continuous values associated
with the bin locations, **decoupling the size of the gradients from the size of the targets**."
DreamerV3 pairs this with **symlog** (`sign(x)·ln(|x|+1)`) and **41 symexp-spaced bins** — designed
exactly for wide-magnitude-range, heavy-tailed, signed targets. **That is `long_accel`'s profile.**

### 5.2 ⚠️ The contradicting evidence — and it kills the mechanism I would naively have invoked

Farebrother's **own ablations** test three explanations and **two of them fail**:

| explanation | their test | outcome |
|---|---|---|
| robust to **noisy targets** | add synthetic reward noise | HL-Gauss degrades slower ✓ … **but Figure 15: HL-Gauss only outperforms MSE under DETERMINISTIC dynamics; the advantage disappears under high environment stochasticity** ✗ |
| categorical **representation** helps | MSE loss with softmax-parameterised outputs | "we do not observe any gains" ✗ — **the cross-entropy loss is essential, not the parameterisation** |
| **ordinal structure + label smoothing** | sweep σ/ς ∈ {0.25…2.0} × bins ∈ {21…201} | optimal σ/ς is **independent of bin count** ✓ — survives |

**⇒ The single most natural reason to expect this to help us — "our `long_accel` target is noisy, so a
robust classification loss will help" — is the one their ablation falsifies.** I must not claim it. The
surviving mechanism is ordinal structure + gradient/target decoupling. Also: **Two-Hot underperforms
MSE in online RL** — so "just use two-hot" is not the safe default; HL-Gauss's smoothing is the active
ingredient.

### 5.3 ⚠️ And a ceiling no head can beat

`MEASURED (ours — IDM_V2_RESULTS.md §3.3)` On PhysicalAI the CAN `long_accel` label correlates
**r = 0.434** with the vehicle's own dv/dt ⇒ **a perfect kinematic estimator caps at R² 0.188.**
**No head architecture beats a label ceiling.** So A1's honest target is **comma2k19**, where the label
*is* dv/dt and the frozen-latent linear probe currently reads **−0.095**. On PhysicalAI the right
action remains IDM v2's pre-committed one: **drop the channel**.

### 5.4 How successful systems keep the axes separable — the answer to our 0/881 failure

Our measured failure: a **5-way manoeuvre softmax that MIXES lateral and longitudinal**, explaining
0 of 881 accelerate predictions `(MEASURED — ours, program record)`.

| system | action encoding | depth |
|---|---|---|
| **VPT** | 13 **independent** binary keys + **separate** 11-bin camera-X and 11-bin camera-Y, foveated | full text |
| **RT-1** | each of **11 dimensions** independently into **256 uniform bins** (7 arm + 3 base + **1 separate mode variable**) | ⚠️ summary |
| **UniVLA** | 4 tokens over \|C\|=16 instead of a **256⁷ joint** space | full text |
| **LAPA** | discretises the continuous delta-EE action space per dimension | full text |

**⇒ The rule, and it is unanimous: discretise PER AXIS, and put "mode" on its own axis.**
RT-1 is the sharpest illustration — the arm/base/terminate **mode** is a *separate* discrete variable,
not fused into the motion classes. **Our failure was not discretisation. It was JOINT discretisation
over a product of two physically independent axes with wildly unequal marginals** — the rare
longitudinal classes get absorbed by the dominant lateral ones. Every system above avoids exactly this.

---

## 6. Scoring the pre-registration

| # | fires? | evidence |
|---|---|---|
| **R1** | **✅ FIRES** | All demonstrated latent-action wins use small discrete spaces (\|A\|=8, 8⁴, 16⁴). No paper found demonstrates recovery of a continuous metrically-scaled scalar. LAPO's continuous-control evidence is an appendix with "minimal analysis". |
| **R2** | ❌ does not fire | Grounding sets are **tiny**: 200 / <256 / 150-per-task / 23k trajectories. Our asymmetry is exactly the exploited one. |
| **R3** | ⚠️ **half-fires** | The structural clause holds (§2.1). The "no geometric route" clause **fails** — StableCamH is a concrete, buildable route that beats GT-camera-height baselines. |
| **R4** | **✅ FIRES — for latent actions specifically** | UniVLA/MVP-LAM/"Why LA Fail" all derive their gain from **suppressing** ego motion, and **no paper found demonstrates a LATENT ACTION extracting ego motion on driving**. ⚠️ **Important nuance found late:** the *extraction* direction **is** demonstrated on driving — by **LFG** (arXiv:2602.22091), which distils a geometry teacher into pose/point-map/segmentation pseudo-labels from unposed YouTube video and reaches **81.4 PDMS at 10 % downstream labels vs 85.2 at 100 %**. But LFG uses **teachers, not a latent-action bottleneck**. So the asymmetry we hold is *proven exploitable on driving* — **just not by this mechanism.** That is what moves A3b to rank 3 and keeps A4 at rank 4. |

**⇒ Verdict, honoring the pre-commitment: R1 and R4 fired, so the maximum admissible recommendation for
the latent-action stack is a sub-GPU-day probe (A4) — NOT a training commitment, NOT an architecture
change.** Meanwhile R2's failure to fire and R3's half-failure mean the *component* mechanisms
(classification heads, geometric grounding) are admissible on their own evidence — and they rank above
the latent action itself.

---

## 7. ⭐ THE RANKED ADOPTION PROPOSAL

Ranked by (expected value) ÷ (cost), weighted toward *refutable today, on assets we already hold*.

### A1 — HL-Gauss ordinal-classification head for `long_accel` (and later `speed`) · **RANK 1**

> 🟢 **THE SIBLING IS ALREADY RUNNING THE BASE VERSION OF THIS — and reached §5.4's conclusion
> independently.** `idm-v3` endpoint **E5 (`Dacc`)** is a **21-bin softmax-expectation decode** of
> `long_accel`, bar **R² > 0 on both corpora**, and its pre-registration already states: *"The bins are
> over the longitudinal axis alone. Our 5-way manoeuvre softmax mixed lateral and longitudinal and
> produced '0 of 881 accelerate'; the failure mode is discretisation done wrong, so the axes stay
> separable here."* **I am not proposing this — they own it.**
>
> **What I can add are two specific, cheap upgrades to an arm already in flight, and one warning:**
>
> | # | upgrade | why, with the citation |
> |---|---|---|
> | **1** | **Replace the hard/two-hot target with HL-Gauss** — a Gaussian centred on the target, integrated over bins, **σ/ς = 0.75** (std ÷ bin width, ~6 neighbouring bins) | ⚠️ **Farebrother et al. measure that Two-Hot UNDERPERFORMS MSE in online RL.** The categorical *parameterisation* is not the active ingredient — softmax+MSE gives **no gain** in their ablation. **The Gaussian label smoothing is.** A 21-bin hard-target softmax is in the family they measured as *not* beating regression. This is the single highest-value detail in this report for the arm they are running. |
> | **2** | **Non-uniform bins, finer near zero** — symexp spacing (DreamerV3's 41 symexp bins) or VPT's **foveated** camera binning | `long_accel`'s mass is concentrated at zero with heavy tails. **With 21 uniform bins over a heavy-tailed near-zero-mean target, most mass lands in 1–2 bins** and the head can satisfy cross-entropy while still being a mean-predictor — reproducing the pathology in a new coordinate system. |
> | **3** | **Bin count is a cheap free parameter** | Farebrother sweep bins ∈ {21, 51, 101, 201} and find the **optimal σ/ς is independent of bin count** ⇒ tune σ/ς once, then raise m for free resolution. 21 is at the bottom of their swept range. |
>
> **Their bar (R² > 0) and their ceiling reasoning (r = 0.434 ⇒ cap 0.188 on PhysicalAI) are both
> correct and I endorse them** — they set the bar better than IDM v2's unreachable 0.30.

**Original write-up (retained — the mechanism, and it is what the upgrades attach to):**

- **Mechanism.** Replace the scalar regression output with m ordinal bins; convert the target via a
  Gaussian integrated over bins (HL-Gauss, σ/ς = 0.75); cross-entropy; read out the distribution mean.
  Use **symexp/foveated bin spacing finer near zero** (DreamerV3's 41 symexp bins; VPT's foveated
  camera bins) — `long_accel`'s mass is concentrated at zero.
- **Citation.** Farebrother et al. 2403.03950 (full text); Hafner et al. 2301.04104 (full text);
  Baker et al. 2206.11795 (full text, foveated binning).
- **Delta vs ours.** `…/2026-07-26-idm-v2/idm2_v2.py:66` — `self.scalar_head = nn.Linear(d_model,
  n_scalars)` → `nn.Linear(d_model, n_scalars * m)`, plus HL-Gauss target construction and a CE loss
  term. **~40 lines.** Nothing else in the pipeline changes.
- **Why it should work here.** Our −0.240 R² is *worse than predicting the mean* — the textbook
  signature of regression collapsing to the mean on a heavy-tailed near-zero-mean target. CE decouples
  gradient size from target size.
- ⚠️ **Do NOT claim the noise-robustness mechanism** (§5.2 — their own ablation falsifies it).
  ⚠️ **Ceiling:** PhysicalAI caps at R² 0.188 by label (§5.3). Target comma2k19.
- **Cheapest refutation — 0 GPU-days, 1–2 CPU-hours, runnable TODAY.** Re-fit the **already-encoded
  frozen latents** (`tanitad-eval:/root/idm2/lat/`, regenerable in 102 s by `idm2_encode.py`) with
  (a) the existing MSE head and (b) an HL-Gauss head, m = 101, `long_accel` only, comma2k19 subset,
  the same 68/36 episode split, scored with the same paired episode-cluster bootstrap.
  **PRE-REGISTERED, both outcomes committed: if HL-Gauss `long_accel` R² on comma does not clear +0.05
  with a paired CI excluding the MSE arm, classification is refuted for this channel and I say so —
  and IDM v2's "drop `long_accel`" recommendation stands unmodified.**
- **Cost.** ≈0. **This is the highest-value cheapest test in the whole report.**

### A2 — Factorise the manoeuvre head into independent per-axis categoricals · **RANK 2**

- **Mechanism.** Replace the 5-way mixed softmax with two independent ordinal heads —
  `lat ∈ {left, straight, right}` and `lon ∈ {decel, hold, accel}` — each HL-Gauss-smoothable since
  both are ordinal. Mode, if any, gets its own axis.
- **Citation.** VPT (full text, factorised per-axis + foveated); UniVLA (full text, 16⁴ over 256⁷);
  RT-1 (⚠️ summary depth, per-dimension 256 bins + separate mode variable).
- **Delta vs ours.** One head becomes two; the loss becomes a sum of two CEs.
- **Cheapest refutation — 0 GPU, MINUTES, and it must be run FIRST.** Compute the **accelerate class
  marginal in the LABELS** at the window length used. If the labels carry no accelerate mass, the head
  change is a **no-op** and the fix is the label/window, not the architecture — and I say so. Only if
  labels carry accelerate mass that predictions do not is the head the culprit.
- **Cost.** ≈0 for the check; small for the change. **Ordering matters: the label check is a
  potential refutation of the entire item.**

### A3 — Metric-scale grounding from camera height · **DOWNGRADED — the ground-plane half is ALREADY REFUTED on our corpus**

> 🔴 **REVISED AFTER READING THE SIBLING'S STAGED WORK.** I wrote this item recommending StableCamH's
> camera-height route. The `idm-v3` sibling has **already measured the closed form on our data and it
> fails.** I am reporting that rather than handing them a dead design.
>
> `MEASURED (sibling — …/incoming/2026-07-27-idm-v3/PRE_REGISTRATION_IDMV3.md §4 E4, from
> idm3_geomtest.py, n = 40 PhysicalAI clips held out from A0)`
>
> | test | result |
> |---|---|
> | apply `v̂ · h/h̄` — exactly what the ground-plane physics prescribes | MAE **2.960 → 3.236**, ΔMAE CI **[+0.051, +0.551] — significantly WORSE** |
> | apply the **opposite** sign `v̂ · h̄/h` | 2.960 → 2.826, not separated |
> | **SHUFFLED heights** (negative control) | 2.960 → 2.862, not separated — **as good as the real thing** |
> | oracle per-clip scale (the headroom) | 2.960 → **1.607**, CI [−1.869, −0.881] |
> | oracle scale factor `k` vs camera height | **r = −0.466**, partial r given v_mean **−0.352 — the OPPOSITE of the ground-plane sign** |
>
> **Three independent ways of saying the same thing: on our corpus, camera height is not the missing
> scalar.** The shuffled-height control passing is decisive — the height carries no information here.
>
> **What survives, and it matters:**
> 1. **The per-clip scale factor is REAL and LARGE.** The oracle moves MAE 2.960 → 1.607 (and R²
>    0.865 → 0.942 in IDM v2). **§2.3's decomposition is intact — the prize exists.** What has been
>    refuted is one *source* for the second factor, not the decomposition.
> 2. **"There is no constant" is now MEASURED, independently, twice.** The sibling pulled PhysicalAI's
>    own gated calibration: camera height is **per-clip, 1.2450–1.6066 m**, 37 distinct values in 40
>    clips, and **all three circulating constants (1.5 / 1.43 / 1.22) are wrong**. StableCamH's design
>    principle — *optimise the height, never assume it* — is corroborated; its *application* to speed
>    scaling is refuted. ⇒ **`IDM_V2_RESULTS.md` §5 item 5's "reconcile the three cam_h values"
>    prerequisite is already CLOSED by the sibling.** My escalation on that point is withdrawn.
> 3. **StableCamH's OTHER half is untested and is a different mechanism.** Its absolute scale comes not
>    from the height alone but from a **learned vehicle-size prior** (a network trained on ~1.45 M
>    vehicle images, comparing predicted vs. estimated object heights). **That route never touches
>    camera height and is therefore NOT refuted by E4.** Known-object-size is a genuinely independent
>    scale source. `PROVISIONAL — untested on our corpus, and I am not claiming it will work.`
>
> **Revised recommendation:** do **not** spend further effort on height-based scaling. If the scale
> question is re-opened after A3b, the surviving candidates are (a) a learned size prior, (b) grounding
> from a small CAN-labeled set (§2.3's first route). **Cost: 0 — this item now consumes no budget.**

### A3 *(original, superseded — retained for provenance)* — camera-height INVARIANCE

> 🔴 **DO NOT EXECUTE the "cheapest refutation" below — the sibling already ran its closed form and it
> FAILED (E4, above).** This block is kept only so the reasoning that led to a refuted recommendation
> stays auditable. The one line still live is the **learned-size-prior** clause, which does not use
> camera height.

- **Mechanism.** StableCamH: compute surface normals from predicted depth; on road pixels the negated
  inner product of normal and reprojected 3D point **is** the camera height; take the per-frame median;
  then exploit **"the camera height does not change in the sequence"** as an optimisation constraint,
  with a learned vehicle-size prior supplying absolute scale. **The camera height is NOT measured a
  priori — it is jointly optimised.**
- **Citation.** arXiv:2312.04530 (full text). KITTI **AbsRel 0.108 / RMSE 4.740**, *beating* VADepth's
  0.120 which uses **ground-truth** camera height, and Monodepth2's 0.968 without median-scaling.
- **⭐ Why this is a real integration and not a survey item.** `IDM_V2_RESULTS.md` §5 item 5 states that
  reconciling our **three mutually inconsistent `cam_h` values (1.5 / 1.43 / 1.22)** is a *prerequisite*
  for metric grounding. **StableCamH says the prerequisite is the wrong object.** You do not need the
  value; you need the invariance — and per-sequence optimisation *outperforms* using the GT constant.
  This is consistent with our own §5.2 measurement that a **per-corpus constant `cam_h` calibration is a
  NO-OP**: a constant is the wrong parameterisation, a per-sequence optimised height is the right one.
  It also handles our **two-rig** exposure (rig A cy≈543 / rig B cy≈755) better than any constant could.
- **Delta vs ours.** A new per-clip scalar estimated at ingest, multiplying the metric decode.
- **Cheapest refutation — 0 GPU, CPU-hours.** On the 104 already-encoded episodes, estimate one scale
  factor per clip by the road-plane/height invariance and apply it to A0's speed predictions.
  **PRE-REGISTERED: we already know the ORACLE clip-level correction moves speed R² 0.865 → 0.942. If
  the geometrically-estimated scale recovers < 30 % of that oracle gain (i.e. lands below R² 0.888),
  geometric grounding is refuted as the speed lever and I say so.** This *also* discriminates the §2.2
  hypothesis from the competing label-quality explanation.
- ⚠️ **Contradicting/limits.** StableCamH has **no ablation on sensitivity to camera-height error** and
  **no evaluation on hills, slopes or pitch variation**; it needs off-the-shelf road + instance
  segmentation. Our corpus is not guaranteed flat-road.
- **Cost.** ~0 GPU for the probe; a small module at ingest if it passes.

### A3b — ⭐ Use a feedforward GEOMETRY TEACHER for ego-motion pseudo-labels, not a geometry-blind IDM head · **RANK 3 — the top NEW proposal**

- **Mechanism.** `PUBLISHED (LFG, arXiv:2602.22091, abstract + authors' project page)` LFG pretrains on
  **unposed, unlabeled YouTube driving video** by distilling **multi-modal teachers** — a **π3**
  backbone for depth/point maps and **SegFormer** for semantics — into a model that jointly predicts
  **point maps, camera poses, semantic segmentation and motion masks**, "without poses, labels, or
  LiDAR". **Camera pose — i.e. ego motion — is a directly supervised target produced by a geometry
  model, not regressed by a geometry-blind head.**
- **The numbers that matter to us.** NAVSIM single-camera: **PDMS 85.2** with 100 % labeled downstream
  data, and **PDMS 81.4 with only 10 %** — **95.5 % of full-label performance at one tenth the labels.**
  ⇒ **This is the "large unlabeled + small labeled" asymmetry demonstrated ON DRIVING**, which is
  exactly what R4 said was missing (§6). It is the single most transferable result in the report.
- **⭐ Where it sits in §2.3's decomposition — and be precise about what it does NOT solve.**
  π3/VGGT-family feedforward geometry models produce structure and pose that are **scale-free**
  (accurate up to one global scale per sequence). So A3b attacks the **first** factor —
  *scale-free ego motion at high fidelity, from a component made of geometry* — and it does **not**
  supply the metre. **I originally paired it with A3's camera height for the second factor; the sibling
  has since refuted that pairing** (§7 A3). ⇒ **Honest status: A3b upgrades the factor we CAN determine
  from video, and leaves the per-clip scalar OPEN.** That is still worth doing — the oracle headroom
  (MAE 2.960 → 1.607) is the combined prize, and a better scale-free estimate is a prerequisite for
  ever collecting it — but **A3b alone will not close the speed gap, and I am not claiming it will.**
  The surviving candidate sources for the metre are a **learned size prior** and **grounding from a
  small CAN-labeled set**; both untested.
- **It attacks failure #3 directly.** Our IDM is geometry-blind (verified: no intrinsics/extrinsics/
  focal/FOV anywhere in it). A geometry teacher is *made of* geometry. This replaces the missing
  inductive bias rather than hoping a transformer head rediscovers it from 68 clips.
- **Delta vs ours.** The IDM's `speed`/`yaw_rate` targets stop being "regress a CAN scalar from frozen
  DINO features" and become "distil a pose teacher, then scale it". `steer` and `long_accel` are
  unaffected.
- **Cheapest refutation — 0 GPU, CPU-hours, on episodes we already hold.** Run an off-the-shelf
  feedforward pose model over a sample of the 104 encoded episodes' *source frames*; take the predicted
  inter-frame translation magnitude **up to an unknown per-clip scale**; regress our CAN `speed` on it
  with **one free scale per clip**. **PRE-REGISTERED, both outcomes committed: if the per-clip-scaled
  geometry-teacher speed does not beat A0's R² 0.8651 and approach the 0.942 oracle, the geometry-teacher
  route is refuted and the IDM head stays as the estimator.** This is also the cleanest possible test of
  §2.2's hypothesis, because a geometry teacher has *no* access to CAN labels and therefore cannot be
  shrinking toward a training prior.
- ⚠️ **Limits.** LFG's **hours of video and its metric-scale mechanism are NOT disclosed** in the
  abstract or project page (§10) — I am citing the *architecture pattern* and the *10 %-label result*,
  **not** a scale mechanism. Whether π3-family pose is metric or scale-free on our corpus is
  **UNVERIFIED** and is the first thing the experiment measures.
- **Cost.** ~0 GPU for the probe (inference only).

### A4 — Latent-action bottleneck as an AUXILIARY probe in frozen-latent space · **RANK 4**

- **Mechanism.** LAPO/Genie in *our* latent space, not pixels: `IDM(z_t, z_{t+1}) → VQ → a_latent`,
  `FDM(z_t, a_latent) → ẑ_{t+1}`; then a linear grounding probe `a_latent → (speed, yaw_rate,
  long_accel)`. The VQ **is** the anti-leakage mechanism (LAPO, verbatim: the IDM "learns to encode only
  the difference between o_{t+1} and o_t").
- **Why it is the right shape for our #2 failure.** `MEASURED (ours —
  …/2026-07-27-v4-instrument/V4_INSTRUMENT.md)`: zeroing `v0` degrades the imagined decode **×93.73 (v1)
  / ×39.43 (v4)** while the perceived decode is **bit-exactly unchanged**; deleting **both** commanded
  action channels but keeping `v0` costs only **×1.32 / ×1.07** — a **71× / 37× separation**. Our
  imagined trajectory is `v0` integrated forward. A latent action is by construction a representation
  of the *change*, and is by construction **not handed in at inference**. That is the correct shape for
  the disease.
- **Why RANK 4 and not RANK 1.** R1 and R4 both fired (§6). Adopting this first would be adopting a
  **mechanism** — the C3 root-cause class this program has already been burned by (IDM v2 §4.2's own
  accel error budget predicted +0.657 and measured −8.81).
- **Cheapest refutation — ≈0.3 GPU-day, on already-encoded latents, NO encoder training, NO pixels.**
  Train the IDM/FDM pair with \|A\| ∈ {8, 64}; linear-probe the latent action.
  **PRE-REGISTERED, both outcomes committed:**
  1. *Bottleneck integrity:* if the probe's `yaw_rate` R² is **below** the existing direct linear probe
     on frozen latents, the bottleneck is destroying information and the direction is refuted for us
     **regardless of anything it does for scale**.
  2. *The crux test:* measure the latent action's `speed` R² **before** and **after** a per-clip scale
     is supplied (oracle scale is available — §5 of IDM v2). **If supplying scale does not move it, the
     "scale lives in the grounding head" claim is FALSE for our data and §2.3 is retracted.** This is
     the direct experimental test of the PI's crux question and it costs well under a GPU-day.
- **Cost.** ~0.3 GPU-day. **Do not schedule before A1–A3 have reported.**

### A5 — Steal two details from V-JEPA 2-AC (it *is* our architecture) · **RANK 5**

`PUBLISHED (arXiv:2506.09985, full text)` — encoder **frozen** ("we freeze the video encoder");
predictor **~300 M, 24 layers, 16 heads, 1024 hidden**; **block-causal** attention where each patch
attends to *action + end-effector state + patch features* from the same and previous timesteps;
action = **7-d delta end-effector**; trained on **<62 h / 23k Droid trajectories**; loss =
**L1 teacher-forcing + L1 rollout (T = 2)**; inference = **CEM, 800 samples, 10 refinements, 16 s/action,
image goals, receding horizon**. Results: pick-place cup **80 %** vs Octo 15 %; grasp cup 65 % vs 15 %.

**Two things to steal:** (a) the **T=2 L1 rollout loss alongside teacher forcing** — a cheap, direct
counter to imagination/perception divergence; (b) **L1, not L2, in latent space.**

⚠️ **The limitation that is our exact exposure:** the model "must implicitly infer the action coordinate
axis from the monocular RGB camera input", and the authors "manually tried different camera positions
before settling on one that worked well." **We have two rigs plus a different-camera corpus.**
I attempted to retrieve their §11.4 quantitative camera-sensitivity numbers **and could not** —
**UNVERIFIED**, flagged rather than guessed. Note also they report **no data-scaling ablation** for the
AC head, so "62 h is enough" is a single point, not a curve.

**Cheapest refutation.** Add the T=2 L1 rollout term to the existing predictor; measure closed-loop
divergence at matched steps. Small.

### A6 — Driving world models: the "zero the action for unlabeled video" trick is a live competitor to our whole IDM · **RANK 6, but strategically important**

`PUBLISHED (Vista arXiv:2405.17398, full text)` Vista trains on **OpenDV-YouTube (2000 h, no action
labels)** *collaboratively* with nuScenes **"with the action conditions for OpenDV-YouTube set to
zero."** **No pseudo-labeling. No latent action.** Action modalities (angle+speed / trajectory /
command / goal point) are injected as Fourier embeddings via cross-attention. FID **6.9**, FVD **89.4**.

`PUBLISHED (DriveVA arXiv:2604.04198, 2026 preprint, full text)` Explicitly **not** a latent-action
model: supervised, requires paired video+action, jointly decodes video and a **3-D (x, y, yaw)** action
in **metric** trajectory space via flow matching on a Wan2.2-TI2V-5B backbone. NAVSIM **90.9 PDMS**;
nuScenes zero-shot **0.84 m L2 / 0.06 %** collision; Bench2Drive **1.33 m / 1.79 %**.

`PUBLISHED (GenAD/OpenDV-2K arXiv:2403.09630)` ~2000 h of YouTube driving video; two stages (image-domain
transfer → video-prediction pretraining), then adapted into an action-conditioned predictor or planner.
⚠️ **SUMMARY DEPTH only** — HTML 404'd on both versions and the PDF exceeded the fetch size limit.
**Nothing load-bearing rests on this row.**

**⇒ The strategically uncomfortable reading, and I am reporting it because it cuts against the brief's
framing:** the field's current best driving results are being obtained **without** latent actions —
either by *zeroing* the action on unlabeled video (Vista) or by *requiring* paired labels (DriveVA).
**That is a direct competitor to our IDM investment, and it is cheaper.**
- **Cheapest refutation.** Train the predictor with the YouTube portion's action **set to zero**
  (Vista's trick) vs. with IDM pseudo-labels, everything else identical.
  **If zeroing matches pseudo-labels, the IDM is not paying for itself** — decision-grade either way,
  and it is the kind of result that only appears if someone runs the null arm.

### A7 — Flow matching vs discrete for the action head · **RANK 7 — no change recommended**

⚠️ **SUMMARY DEPTH** (HF blog + pi.website via search, not paper full text). π0 uses flow matching
(~100 ms/chunk on a 4090, 50 Hz); π0-FAST uses DCT+BPE discrete tokens — **5× faster training / 3×
fewer steps to converge**, but **~750 ms** inference per chunk.

**For us the literature does not say to replace our DiffusionDrive-style decoder.** It says the
*training-time* target representation (discrete/CE) and the *inference-time* head (flow) are
**separable** decisions. Our `long_accel` problem is a *learning* failure, not an inference-speed
failure ⇒ the fix is **A1**, not a flow-matching change. **No experiment proposed.**

### The ranked list at a glance

| rank | item | cost | status after reading the sibling's staged v3 work |
|---|---|---|---|
| **A1** | **HL-Gauss smoothing + non-uniform bins on the `long_accel` classification head** | **~0** | 🟢 **base arm ALREADY IN FLIGHT (v3 E5, 21 uniform bins).** My contribution narrows to **3 upgrades**, and #1 is load-bearing: **their hard-target 21-bin softmax is in the family Farebrother measured as NOT beating MSE.** ⭐ **Highest-value hand-off in this report.** |
| **A2** | Per-axis factorised manoeuvre heads | ~0 | 🟢 **principle already adopted** by v3 E5. Contribution = the citation base (VPT/RT-1/UniVLA) + the **label-marginal check** that could refute the item outright. |
| **A3b** | **Geometry-teacher ego-motion pseudo-labels** (LFG pattern) | **0 GPU probe, inference only** | ⭐ **NEW — not in v3, and the top new proposal.** Attacks failure #3 with a component *made of* geometry. |
| **A3** | Camera-height scale grounding | **0** | 🔴 **REFUTED on our corpus by the sibling** (E4: shuffled heights as good as real; oracle-`k` correlation has the **opposite sign**). Withdrawn. Only the *learned-size-prior* half survives, untested. |
| **A4** | Latent-action VQ probe in frozen-latent space | ~0.3 GPU-day | gated behind A1/A3b |
| **A5** | V-JEPA-2-AC's T=2 L1 rollout loss + L1 in latent space | small | new |
| **A6** | Vista's "zero the action" NULL ARM | small | new — tests *whether our IDM pays for itself* |
| **A7** | Flow matching | — | **no change recommended** |

**⭐ The report's main new proposal is A3b, and A3's refutation is what sharpens it.** §2.3 said metric
ego motion factors into (scale-free structure) × (one per-clip scalar). The sibling has now shown the
**scalar is real and large** (oracle MAE 2.960 → 1.607) but **is NOT sourced by camera height**. So the
decomposition stands and the second factor is currently **unsourced** — which makes getting the *first*
factor right, with an actual geometry model instead of a geometry-blind head, the live move.

**And the pattern across every driving system in this reading is unanimous:** exploiting large unlabeled
video is done with **teachers or explicit action conditioning**, never with a latent action. LFG distils
a geometry teacher (**81.4 PDMS at 10 % labels vs 85.2 at 100 %**); Vista *zeroes* the action on 2000 h
of YouTube; VPT trains a **supervised** IDM on 1,962 h of labels; DriveVA requires paired labels.
**The latent-action bottleneck is the one mechanism here with no driving demonstration at all** — which
is why it is A4, gated behind a probe, and not the headline.

---

## 8. ⚠️ A methodological finding the program should log — an extraction hop FABRICATED a quotation

While researching this I fetched **"Why Latent Actions Fail, and How to Prevent It"** (arXiv:2605.20223)
twice: once as **PDF** (summarised by an automated extraction hop) and once as **HTML full text**.

| the PDF hop returned | the HTML/abstract pass returned |
|---|---|
| a named **"Mode 3: Metric/Scale Information Loss"** | **NOT FOUND** — the abstract names **two** insights, neither about scale |
| *"The paper establishes that 'latent actions cannot inherently recover metric/scale information' without auxiliary supervision"* — presented as a quotation | **"NO explicit statement about metric or scale information in latent actions themselves."** |
| a three-clause **"when NOT to use latent actions"** recommendation | **"NOT FOUND — the paper does not recommend against using latent actions."** |

**The fabricated claim was precisely the one that would have been most load-bearing in this report** —
it would have "settled" the PI's crux question by citation instead of by reasoning, and it would have
been wrong in a way no reader could have caught without re-fetching.

The same path did it again on **MVP-LAM** (arXiv:2602.03668): the PDF hop reported "codebook size 512,
latent dim 32", "~15–20 % alignment improvement", "~100–500 annotated action sequences". The HTML pass
returns **NOT FOUND** for codebook size, latent dim and labeled-data size, and reports an entirely
different metric — **mutual information I(Z;A)**, MVP-LAM ≈ **1.1 bits** vs UniVLA ≈ **0.5 bits**.

**Two independent fabrications from one extraction path, in one session.** Root-cause class: **C4
(inherited without re-verification)**, with a new and worse twist — *the intermediary did not merely
propagate an error, it invented a verbatim quotation and a section heading.*

**Recommended standing rule:** a claim reaching us through a PDF-summarisation hop is
**SUMMARY DEPTH and inadmissible as a quotation**, and any load-bearing row must be re-read at HTML or
abstract depth before it may be cited. `CITATIONS.md` marks depth per row for exactly this reason.
Suggest an entry in `Project Steering/RETRACTION_LOG.md`.

---

## 9. Contradicting evidence — consolidated, so it cannot be skipped

1. **The manipulation latent-action gains are gains from SUPPRESSING ego motion** (UniVLA 88.7 vs 56.5;
   LIBERO-Long 79.4 vs 0.2). Reading them as support for extracting ego motion inverts treatment and
   control. **This is the strongest argument against the brief's framing and I am not softening it.**
2. **Farebrother's own ablation falsifies the noise-robustness story** for classification heads
   (advantage vanishes under stochastic dynamics), and **Two-Hot underperforms MSE** in online RL.
3. **LAPO's delayed-effect limitation** is structurally bad for driving: latent policies "model the
   visible effects of an action, not the action itself" — and throttle→accel→visible motion is the most
   delayed chain we have, i.e. exactly our worst channel.
4. **LAPA fails at cross-environment transfer** (33.6 %/29.6 % vs 64.8 %/54.0 %) — which is precisely
   what a YouTube→PhysicalAI latent action would be asked to do.
5. **`long_accel` has a LABEL ceiling of R² 0.188 on PhysicalAI** (ours). No head fixes that.
6. **The best current driving results use no latent actions at all.** Vista *zeroes* the action on
   2000 h of unlabeled YouTube; DriveVA requires paired labels; LFG distils **teachers**; VPT trains a
   **supervised** IDM on 1,962 h of labels. **Four independent driving/embodied systems exploit exactly
   our asymmetry, and not one of them does it with a latent-action bottleneck.** This is the single
   most decision-relevant pattern in the report, and it is the reason A4 is ranked fourth and gated.
   A6's null arm may additionally show our IDM is not paying for itself.
7. **StableCamH has no camera-height-error ablation and no hill/pitch evaluation**, and needs road +
   instance segmentation. **And more decisively: its ground-plane mechanism is REFUTED on our corpus by
   the sibling's E4** (shuffled heights as good as real; oracle-scale/height correlation at the opposite
   sign). ⭐ **This is the report's own recommendation being killed by our own measurement — a published
   result with strong KITTI numbers that does not survive contact with our data.** It is the clearest
   illustration in this document of why every item carries a refutation experiment.
8. **V-JEPA 2-AC is camera-pose fragile by the authors' own admission** and has **no data-scaling
   ablation** for the AC head — "62 h suffices" is one point, not a curve.
9. **§2.2's channel ordering has a competing explanation (label quality) that I cannot currently
   exclude.** A3's experiment separates them; nothing staged today does.

---

## 10. What I could NOT verify — flagged, not guessed

| gap | status |
|---|---|
| V-JEPA 2-AC §11.4 camera-pose sensitivity numbers | **UNVERIFIED** — fetched twice, appendix not returned |
| GenAD/OpenDV-2K action-conditioned finetuning data volume & nuScenes L2/collision | **UNVERIFIED** — HTML 404 ×2, PDF over size limit; **SUMMARY DEPTH only** |
| LAPO exact codebook shape (two inconsistent readings) | **UNVERIFIED**; load-bearing facts stable across both |
| LFG (arXiv:2602.22091) — **hours** of YouTube video, and its **metric-scale mechanism** | **PARTIALLY RESOLVED.** Abstract + authors' project page give: pseudo-labels = point maps, camera poses, semantic segmentation, motion masks; teachers = **π3** (depth/point maps) + **SegFormer** (semantics); NAVSIM single-camera **PDMS 85.2 @ 100 % labels, 81.4 @ 10 %** (NC 98.2 / DAC 93.7 / TTC 94.4). ⚠️ **Still UNVERIFIED: the hours of video, and how (or whether) metric scale is recovered — "mechanism not disclosed" on the project page; the PDF exceeded the fetch size limit.** A3b is written so that this gap is the *first thing the experiment measures* rather than an assumption. |
| RT-1 per-dimension binning | **SUMMARY DEPTH** (search result, not paper) |
| π0 / π0-FAST comparison | **SUMMARY DEPTH** (blog/vendor page, not paper) |
| Whether `long_accel` mass exists in our manoeuvre labels at the deployed window | **NOT CHECKED — this is A2's first step and it may refute A2 outright** |

---

## 11. Deliverable manifest

All in `TanitAD Research Hub/Architecture & Inference/Research/2026-07-27-latent-action-models/`
(**repo working tree, STAGED via `git add`, NOT committed, NOT pushed**).

| artifact | what | where it lives | only one place? |
|---|---|---|---|
| `LATENT_ACTION_RESEARCH.md` | this file — pre-registration, six threads, ranked proposal, contradicting evidence | `repo:` (staged) | no — repo only, but staged |
| `CITATIONS.md` | every source with **citation depth marked per row** | `repo:` (staged) | no — repo only, but staged |

**Nothing was produced on any pod.** No pod, worktree or branch was touched. No file outside this
directory was modified.

**Coordination with the `idm-v3` sibling.** I read their **staged** `…/incoming/2026-07-27-idm-v3/`
(pre-registration + `idm3_geomtest.py` results) from the git index and revised this report against it.
**I touched none of their files, and no pod.** Their E5 and E2/E4 supersede parts of my first draft;
that is recorded in place (§7 A1, §7 A3) rather than quietly edited away.

**ESCALATION — integration, not a README request:**

1. ⭐ **A1 upgrade #1 is time-critical and nearly free.** The sibling's E5 arm is a **hard-target
   21-bin softmax** — the family Farebrother et al. measure as **not beating MSE**. **Adding HL-Gauss
   smoothing (σ/ς = 0.75) and non-uniform bins costs a few lines and could be the difference between
   E5 passing and failing its R² > 0 bar.** This must reach them **mid-iteration**, not after.
2. **A3b (geometry-teacher ego-motion pseudo-labels) is a NEW proposal not represented in v3's
   endpoints.** It needs an owner and a decision on whether it enters v3 or waits for v4.
3. 🔴 **My A3 recommendation is WITHDRAWN in light of their E4** — logged in §7 A3 and §9 item 7. **My
   earlier escalation about `IDM_V2_RESULTS.md` §5 item 5 is also withdrawn: the sibling has already
   closed it** (camera height is per-clip 1.2450–1.6066 m; all three circulating constants are wrong).
4. **§8 needs a `Project Steering/RETRACTION_LOG.md` entry** — class **C4** with a new
   *fabricating-intermediary* variant — plus, ideally, a standing rule that PDF-summarisation-hop
   content is **inadmissible as a quotation**. This is a program-wide evidence-discipline issue, not a
   finding about latent actions, and it is the item with the widest blast radius in this report.
