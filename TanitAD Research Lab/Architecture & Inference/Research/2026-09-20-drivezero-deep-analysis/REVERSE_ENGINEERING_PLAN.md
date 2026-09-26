<title>Reverse-engineering DriveZero with TanitAD's resources — five stages, ordered by evidence-per-GPU-hour, with the expensive half deliberately last</title>

# Reverse-engineering DriveZero on our fleet

`Research Lab · 2026-09-20 (LAB-RUN-017) · companion to RESULT.md · ⛔ a PLAN, nothing here has run`

> **The organising decision.** DriveZero's headline needs 96 GPUs and a batched simulator. Its **ablation** says the gain comes from **goal augmentation**, which needs neither. So this plan is ordered by **evidence per GPU-hour**, not by the paper's own structure: the mechanism that carries the result is **Stage 1** and costs ~nothing; the RL teacher that carries the *title* is **Stage 4** and may never be worth building.

---

## 0 · Honest resource ledger

| resource | state | consequence for this plan |
|---|---|---|
| **Jetson Thor** (aarch64, sm_110) | 🟢 only non-pod compute; `tanitad-edge` / `tanitad-train` venvs never mixed | edge inference, gsplat rendering, four-family evals. **Not** a training fleet |
| **Dev box RTX 4060** (8.6 GB) | 🟢 usable | v7-tiny ladder arms, CPU/GPU probes. **1 GPU** |
| **pod2** | ⚪ evacuated, idle | re-provisionable; historically A6000/A40 class, **single-digit GPUs** |
| **nuPlan maps** | 🟢 **local** at `data/nuplan-maps/` (v1.0 + v1.1 zip) | maps yes — **logs/sensor data NO** |
| **nuPlan DB / NAVSIM sensors** | ⛔ **absent**; ~450 GB, PI-gated | blocks any direct DriveRL reproduction |
| **Dev-box uplink** | ⛔ **1.2 MB/s measured** — a multi-GB pull takes days | ⇒ **never download the corpus here**; build where the compute is |
| **AlpaSim + NuRec + gsplat** | 🟡 renders on Thor (492 FPS); ⛔ **`alpasim_runtime` unfinished ⇒ no collision / offroad / scene score** | we have a renderer, **not** a scored simulator. This is the real blocker for any RL work |
| **PhysicalAI-AV corpus** | 🟢 2,376 eps parity / 4,719 clips, strict-parity build key | our corpus. **Numerically incomparable** to nuPlan — no DriveZero number transfers |
| **SAM3 map corpus** | 🟢 4,719-clip run on Thor | our closest analogue to their vector map |
| **DriveRL release** | 🟡 inference code + checkpoints public; ⛔ **no license located** | ⛔ **PI/legal gate before any use** |

⛔ **The two hard walls:** (1) we have **~2 GPUs**, they used **96**; (2) we have **no scored closed-loop simulator**, which is the substrate PPO needs. Stage 4 is gated on wall (2), not on wall (1).

---

## Stage 1 — Goal augmentation without an RL teacher ⭐⭐⭐

**Cost: ~0 GPU for the data work + 2 v7-tiny arms + 1 replicate. Blocked on nothing.**

This is the whole point of the exercise. Table 7 says the teacher alone **loses** to imitation (93.61 vs 93.92) and the gain is **+0.80 from augmentation**. So test the augmentation first, on our corpus, with our labels.

**What to build**
1. **A goal-conditioned target generator.** Adopt DriveRL's **two-point, permutation-invariant, speed-scaled goal anchors** (near + far, look-ahead scaling with `v0`) as our nav representation — it is cheap, and it is identical at train and deploy by construction, which our current nav channel is not.
2. **Counterfactual targets.** For each window, perturb the route intent and synthesise a goal-consistent trajectory with our **existing kinematic model** (no RL, no simulator). Start with the perturbations that are unambiguous: lateral lane-offset within the drivable corridor, and target-speed scaling within the Kamm/comfort envelope.
3. **WTA multi-proposal head.** Replace the single-trajectory head with **M proposals + winner-takes-all L1**, plus a **detached** scoring branch. This is what lets queries specialise into modes instead of regressing to the mean — and our mean-regression is exactly what `hold-action` exploits.

**Pre-registered arm — `E-AI-GOALAUG-1` (AI20-1).** Arms: `{no-aug baseline · goal-aug · goal-aug + WTA}` + an **A0b replicate** and `ha0_ext` (the hold-action echo control) **in every arm**.
**Committed:** the augmented arm beats `ha0_ext` beyond the replicate floor on T1 h=6 s ADE **and** drops nav-echo recovery R² from **0.9702** below **0.80** ⇒ augmentation is the anti-echo lever; fails ⇒ the mechanism needs the teacher and the cheap route closes **with a number**.
⛔ **Controls that must read known values:** `ha0_ext` must reproduce its banked score; the no-aug baseline must reproduce the current v7-tiny arm within the replicate floor. If either drifts, the run is `VOID-HARNESS`.

⭐ **Why this is first:** it is the only item here that attacks **P-1**, the programme's actual gate problem, and it is affordable this week.

---

## Stage 2 — Cheap architecture transfers, no new data ⭐⭐

**Cost: 1–3 v7-tiny arms each. All independently testable.**

| # | transfer | why | committed read |
|---|---|---|---|
| **2a** | **Rate-level action space** — predict **jerk + steering-angle rate** through a kinematic bicycle model instead of `(steer, accel)` | smoothness becomes a *structural* prior instead of a loss term; speaks directly to our comfort/jerk family | comfort metrics improve with **no** ADE regression beyond the replicate floor |
| **2b** | **Beta action head** with shape params in (1, ∞) via `1 + softplus` | unimodal, bounded, no extreme-command mass; the published fix (CaRL) for exactly the pathology a Gaussian head has at bounds | calibration improves; no ADE regression |
| **2c** | **Per-term decomposed critic** — one value channel per reward/score term summing to the total | ⭐ composes with **LR15-2**: a *progress* value head is one channel of this; makes the value head interpretable per family, which our four-families rule wants anyway | each channel's prediction correlates with its own term (r ≥ 0.5) — a control that can fail |
| **2d** | **Multiplicative soft-score objective** `Π_k q_k` instead of a weighted sum | forces **joint** satisfaction; a weighted sum lets a model buy progress with comfort, which is how our arms drift | no family regresses while the aggregate improves |
| **2e** | **16 register tokens per camera** (DrivoR, adopted by DriveZero) | already on our backlog as **DR-4**; two independent papers now use it | ⇒ **merge this plan's 2e into DR-4; do not run it twice** |

⚠️ **2a and 2b change the action parameterisation**, which is a **regime boundary** in the sense of `MODEL_REGISTRY §0.1.1` — new arms only, never a paired test across the line.

---

## Stage 3 — DriveVFM-lite: agglomerative distillation at our scale ⭐⭐

**Cost: 1 backbone pretraining run. The honest scoping question is whether it is worth it at all.**

⛔ **Do not attempt 1.64 B samples.** Their own Table A13 is the argument against copying it: **DriveVFM ViT-S (94.41) ≈ DINOv3 ViT-L (94.55)**, and the DriveVFM-over-DINOv3 margin **shrinks from +0.53 at ViT-S to +0.28 at ViT-L**. We already run a DINOv3-class frozen trunk.

**So run the cheap decision first, not the backbone.**

- **3a (0 GPU → 1 GPU-day): does the gap exist on OUR data at all?** Their Table 7 says **DINOv2 = DINOv3 = 93.88**. Reproduce *that* comparison on our probe suite before spending anything on a fifth backbone. **Committed:** if DINOv2 and DINOv3 also tie on our four-family probes, then backbone identity is not our lever and Stage 3 closes.
- **3b (only if 3a says the backbone matters):** distil **two** teachers, not four — **SAM (patch)** and **DINOv3 (summary+patch)**. Their ablation ranks the additions **SAM +0.41 > depth +0.31**, and we already hold a **SAM3 corpus on Thor**, so the boundary-sensitive teacher is free for us in a way it was not for them.
- **3c:** if 3b runs, adopt **PHI-S** standardisation and **QK-Clip** verbatim — both are cheap, both are the parts that make multi-teacher distillation *stable*, and skipping them is how this kind of run diverges.

⭐ **Our unfair advantage here:** they had to buy SAM features from a frozen SAM; we have a **SAM3 map corpus already computed** over our whole 4,719-clip corpus. Distilling from stored features is far cheaper than running the teacher.

---

## Stage 4 — The RL teacher ⛔ gated, and possibly never

**Cost as published: 2,016 GPU-hours on 96 GPUs. We have ~2 GPUs and no scored simulator.**

⛔ **The blocker is not GPUs, it is the scorer.** PPO needs a reward, and a reward needs collision / drivable-area / progress signals in the loop. `alpasim_runtime` is unfinished, so our closed loop today **renders but does not score** (`C1`, `D-12`). **No RL work should start before that is closed** — this is the same class of error as running a gate before characterising it.

**The staged, falsifiable route, if the PI wants it:**

| step | what | cost | gate |
|---|---|---|---|
| **4a** | Finish the **`alpasim_runtime` wire contract** so a rollout returns collision / offroad / progress / TTC. Bounded work (`cargo` present) | engineering | ⇒ unblocks **D-12** and **C1** regardless of whether RL ever runs. **Do this even if Stage 4 is cancelled** |
| **4b** | **Measure our simulator's throughput** in sim-hours per GPU-hour, against their **~1,430**. ⭐ **This single number decides Stage 4.** | hours | ≥ ~100 sim-h/GPU-h ⇒ a *small* teacher is conceivable; ≪ that ⇒ **Stage 4 is closed on arithmetic**, and we say so |
| **4c** | Only then: a **5.7 M** structured-input policy (their size is a gift — it is tiny) on **our** scenes, PPO + Beta + per-term critic, self-play off | 1–2 GPUs × days | beats a hold-action and an IDM control in closed loop |
| **4d** | Distil 4c into the camera student under **augmented goals** — i.e. Stage 1's machinery, now with a real teacher | — | beats Stage 1's no-teacher augmentation beyond the replicate floor |

⭐ **Note the ordering prize:** Stage 1 is the **control arm for Stage 4d**. If Stage 1 already recovers most of the +0.80, Stage 4 has to justify itself against that, not against nothing.

⚠️ **Scene-consistent actor insertion** (ego within **5 m / 0.35 rad** of the logged pose, no collision on release) is a small, high-value detail to copy if 4c ever runs — it is the fix for the "ego left the log so the insert is nonsense" failure any log-seeded simulator hits.

---

## Stage 5 — External comparability (G3)

| # | item | note |
|---|---|---|
| **5a** | **Do not chase navtest.** It is saturated (human 94.8, the whole camera field 91.7–94.6) | a 0.5-point move there is not evidence |
| **5b** | **navhard two-stage EPDMS is the right target** — it is where the 6.6-point spread lives, and our stamp table already tracks it (57.1 DriveZero, 54.6 DrivoR-Scale, 50.1 GigaPixel, 48.1 ZTRS) | ⛔ still gated on the **PI benchmark-portfolio approval** (backlog row 3) and the ~450 GB NAVSIM pull, which **must not** happen over the dev box's 1.2 MB/s link |
| **5c** | ⭐ **Publish what they do not: latency.** Our **60.3 ms p50 / 63.1 ms p95 on Jetson Thor against a 100 ms budget**, on real weights, end-to-end, is a number no paper in this family reports | restate the efficiency wedge as **params × measured edge latency**, not params alone |
| **5d** | **HUGSIM** is the true-closed-loop benchmark of this family and is reconstruction-based (KITTI-360, nuScenes, PandaSet, Waymo) — the same technology as our NuRec/gsplat stack | ⇒ the most natural external closed-loop target *for our existing renderer*; worth pricing after 4a |

---

## Sequenced recommendation

| order | item | cost | why here |
|---|---|---|---|
| **1** | ⭐⭐⭐ **`E-AI-GOALAUG-1`** (Stage 1) | 2 arms + replicate | attacks **P-1**; the mechanism the ablation says carries the result; blocked on nothing |
| **2** | **3a — DINOv2 vs DINOv3 on our probes** | ~0 GPU | decides Stage 3 before Stage 3 costs anything |
| **3** | **2c + 2d** (per-term critic, multiplicative objective) | 1 arm | composes with **LR15-2**; both are loss-shape changes, not new data |
| **4** | **4a — finish `alpasim_runtime` scoring** | engineering | unblocks **D-12**/**C1** on its own merits, RL or no RL |
| **5** | **4b — measure sim throughput** | hours | the arithmetic that decides whether Stage 4 exists |
| **6** | **2a + 2b** (rate-level action, Beta head) | 2 arms | regime boundary — new arms only |
| **7** | Stage 3b/3c, Stage 5b, Stage 4c/4d | large | each gated on the step above |

## Escalations carried by this plan

1. ⛔ **PI/legal: the `XiaomiAutoL3/DriveZero` repo carries no located license.** Inference code and checkpoints are published, but absence of a license is **not** permission. Nothing from it may be used until this is resolved.
2. ⛔ **PI: NAVSIM/nuPlan data (~450 GB)** remains the gate on Stage 5b — and per the measured 1.2 MB/s dev-box uplink it must be pulled **where the compute is**, never here.
3. ⭐ **Master Mind: merge Stage 2e into `DR-4`** — the register-token item is already on the backlog and should not be proposed twice.
4. ⭐ **Master Mind: I-1's expected payoff should fall.** The published quality gain from value-guided search on a strong policy is **+0.56 CLS (0.6 % relative) at N=64, and noise below N=32**.
