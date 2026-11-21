> Fleet-review commissioned, Sayed-directed 2026-07-17.
> Status: **plan** (P8 — no runs yet; nothing below is measured beyond the cited prior evidence).
> Next actions: **Exp1 `refa-distaux-30k`** (4060 pre-gate: distilled-adapter probe R² < 0.7 ⇒ do NOT spend the 30k) → **Exp2 `refa-lora4-30k`** on pod3-idle.

# REF-A (frozen-encoder) improvement plan — evidence-grounded, ranked, with falsifiers

**Sources read:** `TanitAD Research Hub/HYPOTHESIS_LEDGER.md` (H4 row: *open*, "arm B at D1–D3"; changelog: W31 head-to-head voided, H4 evidence RESET), `Architecture & Inference/Research/2026-07-17-blind-rollout-uncertainty-dissipation-and-readout-orthogonality.md`, `Architecture & Inference/Research/ENCODER_MULTICAM_OPTIMIZATION.md`, `PROJECT_STATE.md` (W32 header).

## 0. What the evidence actually says (the frame everything hangs on)

Five measurements triangulate the same diagnosis:

1. **REF-A plateaued at 2.14 m ade@2s @30k** with the full 4-brain + v0-input, on the identical 2376-ep corpus where the flagship (trained encoder) reaches **0.628 @19k** and CV sits at **0.825**. Plateau ≠ overfit — the corpus demonstrably supports 0.628, so for REF-A the binding constraint is the frozen features / their extraction, **not data** (this answers most of (g) below).
2. **REF-A is a dynamics integrator**: real ≈ mean-replaced ADE; vision_use 3.4 %, imagination 1.5 %. The arm earns ~96 % of its score by integrating v0 through learned dynamics priors.
3. **Frozen-DINO speed R² 0.61 vs trained 0.81+** (pre-v0-fix): the ego-motion-relevant signal is either absent from frozen DINOv2-B/14 features or present-but-unextracted by the temporal adapter. **This ambiguity is the single most valuable thing to resolve — every intervention below bets on one side of it.**
4. **E2 (07-17 note): readout capacity is over-provisioned** (active rank ~23–26 ≪ 2048, transition rank ~43). Adding raw capacity is not an evidence-backed move; extraction/adaptation is.
5. **Corpus/metric ceiling caveat (Bench-Eval 07-17)**: comma is **73.9 % straight**, aggregate open-loop L2 is a weak capability test, and the honest kinematic floor is 0.056 m@1s. On mostly-straight highway, vision has little *aggregate* open-loop marginal value — so every intervention below must be judged **per-stratum (skill_score vs best-of-3 floor), especially curvature strata**, or a real win will be invisible.

**H4 status implication:** the original H4 question — "are off-the-shelf frozen SSL features sufficient?" — is already effectively answered **no at this scale** by (1)+(2)+(3) jointly. The archived REF-A 30k ckpt *is* the frozen control, forever. The scientifically live question is now **H4′: what is the minimal adaptation (adapter supervision, distillation, or trainable-parameter fraction) that closes the frozen→trained gap?** That reframing is what makes intervention (a) admissible rather than heresy.

---

## 1. Ranked interventions

### Rank 1 — (b) Distill the trained flagship encoder's latents into the frozen-adapter stack
**Why first:** it is the only intervention that *disambiguates measurement (3) directly*, is decisive in **both** outcomes, and fully preserves frozen purity (only the adapter — which was always trainable — learns). Teacher ckpt distribution already solved: the flagship ckpt is on HF (`Sayood/flagship-4b-phase0`, pushed 07-16), so pod3 can pull it without touching pod2 (pod2 is RAM-fragile; an eval already OOM-killed the flagship once — never load work onto it).
- **Design:** one-time cache of paired latents (frozen-DINO tokens, flagship-encoder tokens) over the 2376 eps → train the adapter with latent-regression (cosine+L2) alongside/before the WM loss. Two-stage: **Stage 1** = cached-pair regression + speed/yaw probe on the distilled adapter output (no WM run). **Stage 2** = full 30k WM run with distill as aux loss, only if Stage 1 passes.
- **Cost:** Stage 1: caching pass on pod3/A40 (hours), regression + probes on **RTX 4060** (1–2 days, cached tensors). Stage 2: one pod3 30k run (~same as the previous REF-A run; cached-feature training preserved).
- **Falsifier (crisp, two-sided):** distilled-adapter speed/yaw probe R² **≥ 0.85** ⇒ the information *is* in frozen features; the adapter was the bottleneck ⇒ (d)/(f) become primary and H4′ looks winnable cheaply. Probe R² **stuck < 0.7** despite converged distill loss ⇒ the information is *absent* from frozen DINOv2 features ⇒ the frozen ceiling is real at the feature level ⇒ **H4 closes decisively-negative** for DINOv2-B and the program pivots to (a)/(e)-swap. Either read is decision-grade; no other intervention buys that.
- **Resource:** 4060 (Stage 1) → pod3-idle (Stage 2).

### Rank 2 — (f) Aux ego-motion heads ON the adapter (yaw-rate / accel / curvature R² targets)
**Why second:** cheapest positive-precedent lever — mirrors the refbpatch aux-yaw success on REF-B, and matches the original bottleneck-diagnosis prescription (aux ego-motion supervision). Labels are free (comma CAN). One nuance the evidence forces: **post-v0-fix, speed is an input — an aux *speed* head is now low-value.** The diagnosis said rotation/shape was already good (0.31 m); what vision must supply *given v0* is **forward geometry**: curvature preview, lead-vehicle state. Target aux heads at **yaw-rate@+Δt (preview, not current), longitudinal accel, path curvature** — quantities the integrator cannot get from v0 alone.
- **Cost:** loss-head code + rerun; rides in the same pod3 run as (b) Stage 2 (they compose — dense distill signal + sparse metric signal).
- **Falsifier:** adapter-latent yaw/curvature R² rises AND **curvature-stratum** skill_score improves. If R² rises but no stratum improves, the bottleneck is downstream of ego-motion decoding (dynamics-integrator saturation confirmed) — stop stacking adapter supervision.
- **Resource:** pod3 (same run as b); probes on Colab T4 / 4060.

### Rank 3 — (a) Unfreeze last-k blocks / LoRA-adapt — as the reframed H4′ arm
**Science cost, quantified honestly:** it does *not* destroy H4 — H4's frozen-purity control already exists (archived 30k ckpt, 2.14 m, final). What it costs is the ability to claim "zero encoder training" for this *new* arm; what it buys is the **adaptation-efficiency frontier**, which is both the better science question now and the better product question (LoRA is cheap to train, merge-able for TRT export — consistent with the ENCODER_MULTICAM production constraint; a distill-to-deploy path is already "production-standard" in that dossier).
- **Design:** LoRA r=16 on QKV of the **last k blocks**, k ∈ {2, 4, 8} (k=4 midpoint first). Key cost trick: cache activations at block 12−k so only the last k blocks run online — step cost ~1.4–1.8× the frozen arm, *not* flagship-cost. 5k-step probe per k; only the winner earns 30k.
- **Falsifier:** best LoRA arm @30k fails to beat frozen 2.14 m by > 2× eval noise ⇒ feature adaptation is not the binding constraint (points hard at adapter/objective). Best arm still ≥ CV 0.825 ⇒ pretrained-init + light adaptation is not competitive with training the encoder ⇒ kills the cheap-adaptation product route; flagship recipe wins H4′.
- **Watch-item:** LoRA arms re-open the data question — monitor train/val gap; if it balloons at 2376 eps, (g) becomes binding *for this arm* (that is the honest 320-ep-overfit lesson transposed).
- **Resource:** pod3-idle (after or interleaved with Rank-1 Stage 2).

### Rank 4 — (c) v0-dropout — companion knob, never a standalone arm
Dropout cannot create information the features don't extractably contain (speed R² 0.61 bounds it); as a standalone it most likely degrades the v0 path via train-test mismatch. But as a **p≈0.15 companion** inside any retrain arm it cheaply pressures the network toward feature use and gives a free diagnostic.
- **Cost:** config flag. **Falsifier:** vision_use rises > 10 % at flat ade ⇒ useful reliance shift; ade regresses with vision_use still < 5 % ⇒ features genuinely have no marginal signal (corroborates a negative Rank-1 read). **Resource:** rides along, any pod.

### Rank 5 — (d) Adapter capacity: temporal-only → spatio-temporal cross-attention over the 256 tokens
DINO-WM lineage support is real, but two pieces of local evidence gate it: E2 says capacity is over-provisioned generally, and the dynamics-integrator ablation can't distinguish "adapter ignores spatial tokens" from "spatial tokens carry nothing marginal". **Run only after Rank-1 Stage 1 says the info exists** (R² ≥ 0.85 branch) — then it's the natural architecture to exploit it.
- **Cost:** new module + one pod3 run; 5k matched probe first. **Falsifier:** 5k delta vs temporal-only ≤ noise on fwd-ade and vision_use ⇒ spatial pooling was not the loss; drop it. **Resource:** pod3.

### Rank 6 — (e) Two-encoder fusion (DINOv2 semantics + I-JEPA dynamics)
Parked. The supporting evidence is the weakest kind we have: the I-JEPA win (3.194 vs 3.796 @15k, best 2.816@7k) was measured at **320 eps where both arms overfit hard** — an overfit-regime ranking with data binding, not a feature-quality verdict. Fusion doubles cache + adds **5.5× encode cost** (a direct CNCE/H5 wound; the encoder is already 60 %+ of the tick per the multicam dossier). If Rank-1 returns "DINO features lack the dynamics info," the cheaper clean test is an **I-JEPA-H swap at full 2376** (single-encoder, matched recipe) — fusion only if the swap wins *and* semantics measurably regress.
- **Falsifier (if ever run):** fusion beats best single encoder at **matched compute budget** (CNCE-normalized, not matched steps). **Resource:** pod3, later.

### Rank 7 — (g) Data scale (2376 → owned lake / ZOD) — not binding for REF-A now, binding later
Verdict from the evidence: for the *frozen* arm, more data cannot add information to frozen features, and the plateau-not-overfit signature plus the flagship's 0.628 on identical data show the corpus is not REF-A's constraint. The 320-ep overfit evidence tells you data is binding **below ~2k eps** and will become binding again the moment (a) adds trainable parameters. So: keep ZOD/owned-lake on the Data-Eng track at its own pace (with the **per-clip cy rig handling** — two rigs cy~543/755 — as a hard hygiene precondition for any new corpus build, and D-016 pad-crop/undistort as the geometry gate), and do **not** delay any architecture experiment waiting for it.
- **Falsifier that flips the priority:** a LoRA arm shows train/val divergence at 2376 that the frozen arm never showed ⇒ escalate the lake.

---

## 2. Cross-cutting protocol (applies to every arm)

- **Eval per-stratum**, skill_score vs the honest best-of-3 floor (0.056 m@1s), curvature strata speed-gated (v ≥ 2 m/s) — an aggregate-only read on 73.9 %-straight comma will hide real vision wins.
- **Eval off pod2 always** (A40 eval pod or pod3 between runs); pod2's flagship must never share memory with an eval.
- 5k probes gate every 30k spend; report vision_use/imagination panel + adapter probe R² alongside ade in every result.
- Teacher/ckpt movement via the HF repos (single-pod-attach volumes make HF the sanctioned cross-pod path).

## 3. Next-run recommendation: two experiments for the pod3-idle window

**Experiment 1 — `refa-distaux-30k` (Ranks 1+2+4-knob combined):** frozen DINOv2-B + adapter, + flagship-latent distillation loss (teacher = newest flagship ckpt from HF), + aux yaw-preview/accel/curvature heads, + v0-dropout p=0.15. **Pre-gate on the 4060 first (1–2 days, cached pairs): if the distilled-adapter speed/yaw probe R² < 0.7, do NOT spend the 30k** — publish the "frozen-feature ceiling is real" close of H4 instead and go straight to Experiment 2. **Expected numbers if the gate passes:** probe R² 0.61 → 0.85–0.95; fwd-ade 2.14 → **1.2–1.5 m** @30k (adapter-bottleneck hypothesis); headline scenario ≤ 0.83 (beats CV) — possible but not expected; curvature-stratum skill_score is the honest win metric either way.

**Experiment 2 — `refa-lora4-30k` (Rank 3):** LoRA r=16 on the last 4 blocks (block-8 activation caching), otherwise the identical recipe, 5k probe gate (must beat the frozen arm's own 5k trajectory outside noise). **Expected:** lands between Exp 1 and the flagship — **0.9–1.4 m** @30k; ≤ 0.825 with < 1 % trained encoder params would be the strong H4′ result ("pretrained-init + light adaptation is competitive"). If it plateaus ≥ 1.8 m, adaptation isn't the constraint and attention shifts to the objective/adapter side with (d).

Sequencing: Exp 1's 4060 pre-gate starts immediately (costs pod3 nothing); pod3 runs Exp 1 then Exp 2 (single pod, sequential). Both fit inside the existing pre-approved REF-A spend framing and neither touches pod2 or the flagship@30k verdict due ~Jul-19–23, which remains the program's top gate regardless.

---

**Key files (absolute paths):**
- `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub\HYPOTHESIS_LEDGER.md` — H4 row (line 13, *open*; changelog "H4 evidence RESET / stays OPEN")
- `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub\Architecture & Inference\Research\2026-07-17-blind-rollout-uncertainty-dissipation-and-readout-orthogonality.md` — E2 over-provisioned/non-orthogonal readout (grounds the anti-capacity stance for (d))
- `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub\Architecture & Inference\Research\ENCODER_MULTICAM_OPTIMIZATION.md` — distillation "production-standard", encoder = 60 %+ of tick (grounds the CNCE cost argument against (e))
- `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\PROJECT_STATE.md` — W32 header: clean encoder isolation, REF-A 2.14 final, flagship-gated top risk
- `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack\experiments\reset-speed4b\` — archived reset code (refa_train_plus.py 4-brain port) that Exp 1/2 would extend
