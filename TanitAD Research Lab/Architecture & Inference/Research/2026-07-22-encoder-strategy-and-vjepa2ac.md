# Encoder strategy for TanitAD + closing the V-JEPA2-AC literature gap

**Author:** research agent (Sayed request, 2026-07-22). **Type:** research + synthesis, no pods/GPU.
**Closes:** `IDM_VIDEO_PRETRAIN_DESIGN.md` §7 open question #1 (V-JEPA 2 / V-JEPA2-AC — "zero verified claims"
last pass). **Reads:** AGENT_OPERATING_STANDARD, RETRACTION_LOG, IDM design §7, MODEL_REGISTRY §1–2, the
IDM proof + **landed re-gate** JSON.

**Evidence-class legend (CLAUDE.md operating standard):** `PUBLISHED` (external, URL-cited) ·
`MEASURED` (ours, artifact-cited) · `INHERITED` (another doc/agent, not re-verified) · `HYPOTHESIS`
(extrapolation, not measured).

---

## ⚑ UPDATE 2026-07-24 — the own-encoder bet was BUILT and MEASURED (C.2 #2 / C.4 fork now partly resolved)

This note (2026-07-22) recommended, as the strategic data-lever bet, **"our-own pretrained video encoder
with rig-robustness *engineered in*** (multi-rig pretraining + explicit camera conditioning)." That bet was
then **launched and measured**, and the decisive instantiation **FAILED** (`MEASURED`,
`…/incoming/2026-07-24-branchb-transfer-eval/results_branchb_transfer_e50_CONVERGED.json`;
`MODEL_REGISTRY.md §10`):

- **`dynenc-branchB`** — from-scratch, GAIA-2 **all-block** camera-conditioned, multi-rig (2 466 clips),
  40 k steps — held-out-rig transfer **REFUTED**: best cross-rig speed R² **−0.667** (gate +0.9), cross yaw
  R² negative everywhere; paired vs the plain frozen **flagship-v1** encoder the dR2 CI excludes 0
  **Branch-B-worse on 3 of 4 arms**; even in-domain it is a **weaker** speed-decodable substrate (its own
  40 k head reads rig-B speed R² **0.24**). This follows **Branch A** (warm-start suffix conditioning, −2.1)
  already refuted (`RESULTS_camcond.md`). **⇒ explicit camera-conditioning at 40 k / 2 466-clip scale does
  NOT engineer rig-invariance.**
- The C.4 **"multi-rig co-train fixes it"** fork was **also** refuted upstream (the multi-rig light-FT
  cotrain landed rig-B speed R² **−1.61** — representational, not data-diversity;
  `results_branchb…CONVERGED.json` meta `multirig_lightft_rigB_speed_r2`, and the own-encoder `DESIGN.md`
  §0). So **both** cheap fixes this note weighed (data-diversity, and conditioning bolted on) are now
  `MEASURED`-dead.
- **⭐ The one positive:** the plain frozen **flagship-v1 trained encoder** is the **stronger cross-rig
  substrate** (speed R² **+0.657** on multirig_val) — strategy **#1** (trained-from-scratch on-domain)
  extends further cross-rig than **#2**. **But it is NOT uniformly rig-robust** (**−1.169** on the
  single-domain rig_val arm), so the cross-rig problem is **narrowed, not solved.**
- **Recommended pivot (`HYPOTHESIS`, NOT proven — a Sayed-gated NEW training arm, not auto-launch):** a
  **flagship-warm-started, longer-trained, augmentation-matched** encoder variant, in place of more
  from-scratch camera-conditioning. See C.2 / C.4 (annotated inline below) and `MODEL_REGISTRY.md §10`.

*Everything below this banner is the original 2026-07-22 analysis; inline `[UPDATE 2026-07-24]` marks show
where the measurement changed the reading.*

---

## TL;DR — the one-paragraph verdict

For our metric-precision driving tasks the encoder question is **settled in-distribution and freshly
settled cross-domain**, and they point opposite ways about *frozen*: (1) **in-distribution, an encoder
TRAINED on our domain wins by 4.7–6.5×** over a frozen external image-SSL encoder (flagship v1 **0.452 m**
vs REF-A frozen-DINOv2 **2.13–2.92 m**, strict parity — `MEASURED`); (2) **cross-rig / cross-domain, even
our own trained encoder's latent is rig-specific** — frozen it collapses (speed R² **0.930 → −2.465**
cross-rig) and, as of the **landed re-gate (2026-07-22)**, this is **not** an intrinsics/front-end artefact
(f_eff matched across rigs) and is **not** rescued by light fine-tuning (cross-domain speed R²
**0.406 → 0.411**, inert) — `MEASURED`. The closest published precedent, **V-JEPA 2-AC (Meta, 2025)**,
independently confirms both halves: it freezes a **1M+ hour** video-pretrained encoder and trains only a
small action head on **< 62 h** of robot data — *and openly reports the same failure mode we measured*,
camera-position sensitivity, requiring the authors to "manually [try] different camera positions before
settling on one that worked well" (`PUBLISHED`, arXiv:2506.09985). **Recommendation:** flagship WM v4/v5 →
**trained-from-scratch** (evidence-backed); IDM/YouTube labeler → **do NOT ship frozen or light-FT** (the
re-gate is a NO-GO), pursue **multi-domain encoder training + a speed-prior scale head**; the frozen-encoder
line is **MEASURED-dead** for metric precision and should be retired to diagnostics only. The single
cheapest experiment that resolves the biggest open question is a **multi-rig co-train** of the existing IDM
head (below).

---

# PART A — V-JEPA 2 / V-JEPA2-AC (all `PUBLISHED`, cited)

Sources: arXiv:2506.09985 (paper + HTML v1) · Meta AI blog (ai.meta.com/blog/v-jepa-2-world-model-benchmarks) ·
corroborating reviews (emergentmind topic pages). The design doc flagged this as THE closest precedent for
option (a) — "an action head on a (mostly) frozen video encoder trained with limited action data."

### A.1 The facts, verified against the primary paper

| question | answer (`PUBLISHED`, arXiv:2506.09985) |
|---|---|
| Encoder **frozen or fine-tuned** when the AC head trains? | **FROZEN.** Verbatim: *"we freeze the video encoder and train a new action-conditioned predictor, V-JEPA 2-AC, using less than 62 hours of unlabeled robot videos."* |
| Encoder pretraining scale | **> 1,000,000 hours** of internet video + images, action-free JEPA (masked latent prediction). |
| Action data for the AC head | **< 62 hours**, unlabeled, from the open-source **Droid** dataset (Franka Panda arm). |
| AC objective | Latent-space (representation) prediction, **dual loss**: teacher-forcing single-step L1 + autoregressive **rollout** (T=2). `L = L_teacher-forcing + L_rollout`. No pixel reconstruction. |
| AC head architecture | ~**300M** transformer, **24 layers, 16 heads, 1024 hidden**, **block-causal** attention. |
| Action space | **7-D end-effector** (Δ3-D position, Δ3-D orientation, 1-D gripper) — robot manipulation. |
| "Zero-shot" transfer claimed | Deployed on **Franka arms in two different labs**, "without collecting any data from the robots in these environments"; 65–80% success on grasp / reach-with-object / pick-and-place. |

### A.2 Is the transfer cross-domain / cross-embodiment? — precisely characterized

**It is same-embodiment, new-lab / new-object zero-shot — NOT cross-morphology.** The AC head is trained on
Droid (Franka Panda) and deployed on Franka arms elsewhere. That is impressive environment/object
generalization, but the **embodiment (robot type, action coordinate frame) is held fixed**. This matters
for us: it is *not* evidence that a frozen encoder transfers across **camera geometries / rigs**, which is
exactly our failure axis.

### A.3 ⭐ The load-bearing finding — V-JEPA2-AC reports OUR failure mode

The paper's own **limitation** section is the published analogue of our cross-rig collapse (`PUBLISHED`,
arXiv:2506.09985 §limitations, verified in HTML v1):

> *"Since the V-JEPA 2-AC model is trained to predict representations of the next video frame given an
> end-effector Cartesian control action, without any explicit camera calibration, it must therefore
> implicitly infer the action coordinate axis from the monocular RGB camera input."*

and, operationally:

> *"we manually tried different camera positions before settling on one that worked well."*

**Interpretation.** A frozen encoder + action head has **no explicit geometry**, so it must *implicitly*
bind the action frame to the camera pose. Change the camera pose and that binding breaks — the authors
worked around it by **hand-tuning the camera position to match**. This is the same mechanism as our
rig-A→rig-B collapse: a pure camera-rig change breaks the frozen latent's action-coordinate inference. The
key escalation for us: **this persists even with 1M+ hours of video-SSL pretraining.** Scale of the SSL
pretraining does **not** buy camera/rig invariance for free.

### A.4 Does the recipe transfer to driving / monocular ego-motion?

**Partially — the *paradigm* transfers, the *robustness* does not, and the action space does not.** (mixed
`PUBLISHED` + `HYPOTHESIS`)

- **Against direct transfer** (`PUBLISHED`): (i) the action space is 7-D robot end-effector, not vehicle
  ego-motion; (ii) the model is explicitly **camera-pose-sensitive** and required hand-tuning — a
  non-starter for driving, where rigs vary continuously (our corpus alone has two rigs, and YouTube is
  unbounded rig diversity); (iii) "zero-shot" is same-embodiment.
- **For the paradigm** (`HYPOTHESIS`): the *structure* — massive action-free video pretraining → a small
  action-conditioned head on limited real-action data — is exactly the VPT-for-driving move
  `IDM_VIDEO_PRETRAIN_DESIGN.md` proposes. V-JEPA2-AC is the strongest published existence proof that
  "< 62 h of action data on a frozen video encoder" can drive control **when camera geometry is
  controlled**. Our program's contribution would be making it *rig-robust* (the exact gap the re-gate now
  isolates).

**Net for the IDM head design:** V-JEPA2-AC **validates the head shape** (small, latent-predictive,
teacher-forcing + rollout — which matches our design doc §1/§2) and simultaneously **red-flags the frozen
substrate for the multi-rig setting**. Do not read V-JEPA2-AC's zero-shot success as license to freeze; read
its limitation section as confirmation of why our re-gate failed.

### A.5 Cross-check — DINO-WM and driving world-model encoders

- **DINO-WM** (ICML'25, openreview `D5RNACOZEI`): frozen **DINOv2** patch features + a causal-ViT latent
  predictor, zero-shot planning — but in **manipulation / maze / navigation**, not driving (`PUBLISHED`).
  It is the canonical "frozen encoder world model works" citation, and it works in domains where the camera
  is fixed or the task is object-centric. It is the same bet REF-A made and lost on driving metric
  precision.
- **Driving world models** (Vista 2405.17398, GenAD 2403.09630): these **train / fine-tune their own latent
  spaces** (Vista adapts a Stable-Video-Diffusion latent to driving); they do **not** freeze an external
  image-SSL encoder for metric ego-motion (`PUBLISHED`). A recent line (e.g. "Back to the Features: DINO as
  a Foundation for Video World Models", arXiv:2507.19468) revisits frozen-DINO video WMs but is not a
  driving metric-precision result.
- **Evidence gap (informative):** no clean **published head-to-head of frozen vs trained/fine-tuned encoder
  for a *driving* world model** surfaced. **Our REF-A-vs-flagship strict-parity comparison is a rare direct
  measurement of exactly that axis** — the field mostly asserts one side. We should treat that as an asset,
  not a hole.

---

# PART B — OUR measured encoder-transfer evidence (all `MEASURED`, artifact-cited)

Two independent encoders (external DINOv2; our trained ViT), two independent tasks (WM forward prediction;
IDM inverse dynamics), one consistent story. Every number below is read from the raw JSON / registry, not
from prose.

### B.1 The evidence table

| # | evidence | number | artifact |
|---|---|---|---|
| 1 | **flagship v1 — trained-from-scratch ViT encoder** (WM, in-dist) | ADE@2s **0.4522 m** heldout / **0.4271** full-set; encoder speed-probe R² **0.861** | `MODEL_REGISTRY.md` §1.2; raw `taniteval/results/flagship-30k.json` |
| 2 | **REF-A — frozen DINOv2-B/14** (WM, in-dist, strict parity vs #1) | ADE@2s **2.1322 m** (canonical 4B, §2.1) → **2.9196 m** (dyn-in final answer, §2.3); paired vs flagship **Δ +2.62 m [2.45, 2.80]**, flagship wins 95.9% | `MODEL_REGISTRY.md` §2.1/§2.3; raw `results/refa-dinov2-4b`, `results/refa-dynin-30k.json` |
| 3 | REF-A frozen encoder is **speed-blind / integrator** | pre-fix ablation: `vision_use` **3.4%**, imagination 1.5% → "earns ~96% of accuracy by integrating v0"; failure 94.2% longitudinal; train fwd-ADE **0.65** → heldout **2.92** (**4.5× gap**) | `MODEL_REGISTRY.md` §2.3 |
| 4 | flagship v1 no-speed **causal control** (only `speed_input` differs from #1) | ADE@2s **2.9176 m** — architecture identical, so 2.918 vs 0.452 is a clean lever | `MODEL_REGISTRY.md` §1.1; `results/flagship-nospeed.json` |
| 5 | **IDM proof — frozen OUR-trained v1 encoder** (inverse dynamics) | in-dist speed R² **0.930**, yaw 0.924, ADE@2s 2.73 → **cross-domain (comma2k19) 0.657** → **cross-rig (rig-B) −2.465** (ADE 17.47, ratio 4.0×). **PASS: false** | `…/incoming/2026-07-22-idm-proof/results.json` |
| 6 | **IDM RE-GATE — the landed NO-GO** (2026-07-22) | see B.2 — neither fix reaches the gate | `…/incoming/2026-07-22-idm-proof/results_regate.json` + `regate.log` |
| 7 | flagship v1 trained encoder **also degrades OOD** | comma2k19 0.849 vs floor 0.372 (win 17.5%); cosmos 0.583 vs 0.358 (win 29.4%) | `MODEL_REGISTRY.md` §1.2 |
| 8 | Leaderboard structure | **every trained-encoder arm sits above CV; the frozen-DINOv2 arms sit at the bottom** (slots 9 & 11; slot 10 is the no-speed *trained* control) | `MODEL_REGISTRY.md` §6 leaderboard |

**Precision note (matters for the reading):** rows 5–6 freeze **our own trained flagship-v1 encoder**, not
DINOv2. So the IDM's *in-distribution* speed R² 0.930 is **good even frozen** — freezing is not the in-dist
problem once the encoder was trained on-domain. The problem is **purely cross-rig / cross-domain**. This is
sharper than "frozen bad, trained good": it is **"external-pretrained-frozen fails even in-dist (REF-A);
our-trained encoder succeeds in-dist but its latent is rig-specific, and neither light-FT nor front-end
canonicalization fixes cross-rig (IDM re-gate)."**

### B.2 The re-gate verdict (load-bearing — freshest data point)

The FAIL-branch re-gate tested the two natural rescues. Both fail (`MEASURED`,
`results_regate.json` / `regate.log`):

| fix | what it did | result | verdict |
|---|---|---|---|
| **#1 f-theta canonicalization front-end** | intended to null out intrinsics variance | **NO-OP — already applied in baseline.** Measured f_eff: rig-A **266.13** / rig-B **266.1** / comma **266.5**, all = F_REF **266.0**, per-clip principal point | ⇒ **the rig-B collapse is NOT intrinsics-driven.** A pure camera-rig change breaks the frozen latent even with identical principal-point canonicalization |
| **#2 light-FT** (unfreeze last 4 ViT blocks, lr 5e-5, 1000 steps) | let the top of the encoder adapt | comma: cross speed R² **0.406 → 0.411** (inert); rig: **−3.21 → −1.65**, yaw 0 → 0.242, ADE −29% (partial, still far from 0.9 gate) | ⇒ **light-FT does NOT recover cross-domain transfer** |

**Committed reading (verbatim from `results_regate.json`):** *"the frozen-encoder supervised-IDM line needs
a rethink; do NOT proceed to YouTube on this recipe. Next: encoder RETRAIN on undistorted multi-domain data
+ speed-prior scale head."*

This is the crucial triangulation with V-JEPA2-AC (A.3): both say the frozen video encoder's latent is
**camera/rig-bound**, and our re-gate adds that the two obvious cheap fixes (front-end canonicalization,
light-FT) **do not lift it**.

---

# PART C — Synthesis & recommendation

### C.1 The two-layer finding

- **Layer 1 — representation source (in-distribution): TRAINED ≫ frozen-external. `MEASURED`, parity-controlled.**
  flagship 0.452 vs REF-A 2.13–2.92 (4.7–6.5×); the no-speed control (2.918) proves the flagship's win is
  the trained encoder + speed channel, not confounds. A frozen *external image-SSL* encoder (DINOv2) hits a
  capability ceiling on driving metric precision.
- **Layer 2 — cross-domain / cross-rig transfer: the latent is RIG-SPECIFIC, and cheap fixes don't help.
  `MEASURED` + `PUBLISHED`.** Frozen our-trained encoder collapses cross-rig (0.930 → −2.465); not an
  intrinsics artefact (f_eff matched); not rescued by light-FT (0.406 → 0.411). Even the trained flagship
  degrades OOD on comma2k19 (win 17.5%). V-JEPA2-AC reports the identical camera-pose sensitivity at 1M+ h
  pretraining scale.

### C.2 Ranking the four encoder strategies (driving, metric precision)

| rank | strategy | in-dist | cross-domain/rig | evidence class | verdict |
|---|---|---|---|---|---|
| **1** | **Trained-from-scratch on target domain** (flagship recipe) | **best (0.452)** `MEASURED` | best available, but still an OOD gap `MEASURED` | `MEASURED` for in-dist; multi-domain fix is `HYPOTHESIS` | **Adopt for the flagship WM.** |
| **2** | **Our-own pretrained video encoder** (V-JEPA2-style SSL on our + YouTube video) then small AC/WM head | unproven → **weak** at 40k (`dynenc-branchB` own-head rig-B speed R² **0.24**) `MEASURED` | **only path that also unlocks YouTube-scale data** — but SSL scale alone does NOT buy rig-invariance (A.3), **and explicit conditioning did not either (`MEASURED`, below)** | was `HYPOTHESIS`+`PUBLISHED` → now `MEASURED`-negative for the camera-conditioned from-scratch instantiation | **[UPDATE 2026-07-24] REFUTED at 40k/2466-clip scale.** `dynenc-branchB` (from-scratch, GAIA-2 all-block conditioning) cross-rig speed R² **−0.667**, **worse than plain frozen flagship-v1 (+0.657)** paired on 3/4 arms (`MODEL_REGISTRY §10`). Rig-robustness was **not** achieved by engineering it in. Surviving lever = a **flagship-warm-started** variant (`HYPOTHESIS`, Sayed-gated), not more from-scratch conditioning. |
| **3** | **Light-FT of a frozen pretrained encoder** | marginal | **does not rescue cross-domain (0.406→0.411)** `MEASURED` | `MEASURED` (weak) | Weak lever; not a substrate on its own. |
| **4** | **Frozen pretrained encoder** (DINOv2 / frozen-trained-v1) | ceiling (REF-A 2.13–2.92) `MEASURED` | **collapses (−2.465)** `MEASURED` | `MEASURED`-dead | **Retire to diagnostics/probes only.** |

### C.3 Recommendation by program line

**(a) Flagship WM (v4 / v5): trained-from-scratch encoder. `MEASURED` / evidence-backed.**
Settled by strict-parity measurement (rows 1–4, 8). Do **not** adopt a frozen external encoder for the
flagship — that is REF-A's closed H4 ceiling. The residual open item is **cross-domain robustness** (OOD win
17.5% on comma2k19), which is a **training-data-diversity** problem, not a frozen/trained-axis problem →
address by widening the encoder's training distribution (multi-rig / multi-source), not by freezing.

**(b) IDM / YouTube labeler: do NOT ship frozen or light-FT. `MEASURED` NO-GO.**
The re-gate is a hard stop on the current recipe. The evidence-backed next step is the re-gate's committed
direction: **encoder trained/co-trained on multi-domain, multi-rig data + a speed-prior metric-scale head.**
The open fork — (i) multi-domain-cotrain OUR encoder vs (ii) invest in a V-JEPA2-style video-SSL encoder —
should be decided by the cheapest experiment (C.4), *not* by adopting the frozen recipe V-JEPA2-AC used
(its own camera-sensitivity limitation says that would inherit our exact failure).

**[UPDATE 2026-07-24 — the committed direction was executed and it did not clear the gate.]** Fork leg (ii)
was built as `dynenc-branchB` (from-scratch, multi-domain/multi-rig, GAIA-2 camera-conditioned + supervised
IDM + metric grounding — the re-gate's committed recipe) and **REFUTED** on held-out-rig transfer (cross-rig
speed R² **−0.667**; `MODEL_REGISTRY §10`); leg (i)'s data-diversity premise was independently refuted
(multi-rig cotrain −1.61). So the "trained/co-trained multi-domain encoder + speed-prior head" step is
**`MEASURED`-negative as instantiated**. The NO-GO on shipping frozen/light-FT **stands and hardens**; the
revised next step is the **flagship-warm-started, longer, augmentation-matched** variant (`HYPOTHESIS`,
Sayed-gated), pre-registered before any spend — not more from-scratch camera-conditioning.

**(c) Any frozen-encoder line: retire for metric precision. `MEASURED`-dead.**
Keep frozen encoders only for cheap probes/diagnostics (e.g. the speed-probe R² instrument, quick feature
sanity checks). Never as a deployment or labeling substrate for a task needing continuous metric accuracy
across rigs. REF-A remains valuable **as the reference that closed H4**, not as a path to revive.

### C.4 ⭐ The single cheapest experiment (resolves the biggest open uncertainty)

**Biggest open question:** is the cross-rig collapse a **DATA-DIVERSITY** problem (fixable cheaply by
training the encoder on multiple rigs) or a **REPRESENTATION** problem (requires a fundamentally different,
expensive video-SSL encoder)? This is the fork that decides whether option (2) — the V-JEPA2-scale bet — is
even necessary. Everything else about the YouTube line hangs on it.

**Experiment — multi-rig co-train, on existing assets, pre-registered (both outcomes committed):**
Reuse the *identical* IDM head + re-gate harness (`results_regate.json` infra already runs 70-ep arms in
~20 min each). Change **one** thing: instead of training on rig-A only, **co-train the light-FT arm (last 4
ViT blocks + head) on rig-A + rig-B jointly** (and, as arm 2, PhysicalAI + comma2k19 jointly), holding out
the **same** cross-rig / cross-domain val, and re-run the same §5 gate (cross speed R² > 0.9 AND yaw > 0.9
AND ADE@2s < 1.5× in-dist).

- **PASS (cross-rig speed R² > 0.9):** the collapse is **data-diversity** → co-training OUR encoder on
  multi-rig data solves it → **no V-JEPA2-scale SSL pretraining needed** → proceed to multi-domain IDM →
  YouTube. The cheapest possible win, and it retires option (2) as unnecessary.
- **FAIL:** data diversity alone is insufficient → the representation itself is the bottleneck → this is
  the **pre-registered justification** for the expensive V-JEPA2-style video-SSL encoder (and/or a VO /
  known-height metric-scale auxiliary), with the go decision already committed before any spend.

**Why this is the right cut:** it isolates *diversity* from *representation* on data we already have (rig-A,
rig-B, comma2k19, CAN labels), touches **no** WM parity key, spends **~hours not GPU-days**, and is the
minimal, decisive refinement of the re-gate's own committed "retrain on multi-domain data" direction —
co-train first (cheap), retrain-from-scratch only if co-train fails. Secondary lever to fold in: a
**speed-prior scale-calibration head** (the re-gate's second committed component), testable in the same run
since CAN `v0` is already in every clip.

> **[UPDATE 2026-07-24 — this fork has been RESOLVED by measurement; the answer was neither branch as
> framed.]** The C.4 cut asked "is the cross-rig collapse **data-diversity** or **representation**?" Both
> were then run:
> - **Data-diversity leg → REFUTED.** The multi-rig light-FT cotrain landed held-out rig-B speed R²
>   **−1.61** (no recovery vs −1.65 single-domain) — the collapse is **representational** (`MEASURED`;
>   own-encoder `DESIGN.md` §0, `results_branchb…CONVERGED.json` meta).
> - **Representation leg → the expensive video-SSL encoder was BUILT and it also FAILED.** `dynenc-branchB`
>   (from-scratch, GAIA-2 all-block camera-conditioned, multi-rig, 40 k) — the pre-registered
>   representation fix — **REFUTED**: cross-rig speed R² **−0.667**, weaker than plain frozen flagship-v1
>   (+0.657) paired on 3/4 arms (`MODEL_REGISTRY §10`;
>   `…/incoming/2026-07-24-branchb-transfer-eval/`). So **more explicit camera-conditioning is not the
>   answer at this scale** either.
> - **The pivot (`HYPOTHESIS`, NOT proven — Sayed-gated NEW arm, not auto-launch):** the paired data shows
>   the plain **frozen flagship-v1 trained encoder** is the *stronger* cross-rig substrate (+0.657
>   multirig_val) — so the cheapest next discriminating cut is **not** "condition harder" but **"train a
>   flagship-warm-started, longer, augmentation-matched encoder"** and re-run the same gate. ⚠️ flagship-v1
>   is **not uniformly rig-robust** (−1.169 on rig_val), so this narrows the problem, it does not close it;
>   pre-register both outcomes and get Sayed's go before any GPU-days.

---

## Evidence-class ledger (what is proven vs extrapolated)

| claim | class |
|---|---|
| Trained-from-scratch encoder beats frozen-DINOv2 in-dist by 4.7–6.5× (parity) | `MEASURED` (registry §1.2/§2, raw JSON) |
| Frozen our-trained encoder collapses cross-rig (0.930 → −2.465) | `MEASURED` (`results.json`) |
| Cross-rig collapse is NOT intrinsics (f_eff matched) and NOT rescued by light-FT | `MEASURED` (`results_regate.json`) |
| V-JEPA2-AC freezes the encoder; < 62 h robot data; 1M+ h pretrain; camera-pose sensitive | `PUBLISHED` (arXiv:2506.09985 + HTML v1 + Meta blog) |
| V-JEPA2-AC "zero-shot" is same-embodiment (Franka), new-lab — not cross-morphology | `PUBLISHED` (arXiv:2506.09985) |
| Multi-rig **co-training** recovers cross-rig transfer | ~~`HYPOTHESIS`~~ → **`MEASURED`-REFUTED 2026-07-24** — multi-rig light-FT cotrain → held-out rig-B speed R² **−1.61** (no recovery; representational, not data-diversity). `DESIGN.md` §0; `results_branchb…CONVERGED.json` meta |
| A V-JEPA2-style SSL encoder would be rig-robust for driving | ~~`HYPOTHESIS`~~ → **`MEASURED`-NEGATIVE for the built instantiation 2026-07-24** — `dynenc-branchB` (from-scratch, GAIA-2 all-block camera-conditioned, 40k) cross-rig speed R² **−0.667**, FAILS the gate; `MODEL_REGISTRY §10` |
| `dynenc-branchB` is a **weaker** cross-rig (and in-domain) dynamics substrate than the plain frozen flagship-v1 encoder (paired dR2 CI excludes 0, Branch-B-worse, on 3/4 arms) | **`MEASURED`** (`…/2026-07-24-branchb-transfer-eval/results_branchb_transfer_e50_CONVERGED.json`; `MODEL_REGISTRY §10`) |
| Frozen **flagship-v1** trained encoder is the stronger cross-rig substrate (+0.657 multirig_val) **but not uniformly** rig-robust (−1.169 rig_val) | **`MEASURED`** (same artifact) |
| A **flagship-warm-started, longer-trained, augmentation-matched** encoder recovers cross-rig transfer | `HYPOTHESIS` — the recommended pivot; a **Sayed-gated new arm, not launched**, both outcomes to be pre-registered |
| No clean published frozen-vs-trained head-to-head exists for a *driving* WM | `PUBLISHED` (absence across the surveyed driving-WM literature; our parity comparison fills it) |

## Sources (PUBLISHED)

- V-JEPA 2 / V-JEPA2-AC: arXiv:2506.09985 — https://arxiv.org/abs/2506.09985 · https://arxiv.org/html/2506.09985v1 · Meta blog https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/
- DINO-WM: https://openreview.net/forum?id=D5RNACOZEI (ICML'25)
- DINO video WM follow-up: arXiv:2507.19468
- Driving WMs (own/fine-tuned latents): Vista arXiv:2405.17398 · GenAD arXiv:2403.09630
- Our design doc: `IDM_VIDEO_PRETRAIN_DESIGN.md` (VPT 2206.11795, DriveWAM 2605.28544, Seer 2412.15109 in its ledger)

## Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| This note | `repo: TanitAD Research Hub/Architecture & Inference/Research/2026-07-22-encoder-strategy-and-vjepa2ac.md` (staged) | yes — stage it |
| MEASURED inputs (unchanged) | `repo: …/incoming/2026-07-22-idm-proof/results.json`, `results_regate.json`, `regate.log` | already in tree |
| Registry facts (unchanged) | `repo: Project Steering/MODEL_REGISTRY.md` §1.1/§1.2/§2 | in tree |

**Escalation (integration):** (1) `IDM_VIDEO_PRETRAIN_DESIGN.md` §7 open-question #1 is now **closed** — its
head shape (small, latent-predictive, teacher-forcing+rollout) is *validated* by V-JEPA2-AC, but its
**"start frozen" substrate (§3) is contradicted** by the re-gate; §3 should be updated to "multi-domain
co-train, not frozen." (2) The C.4 multi-rig co-train is the recommended next IDM experiment and should be
pre-registered before any YouTube spend.
