# IDM-for-YouTube — design & training concept (v0, 2026-07-22)

**Author:** orchestrator (11-h autonomous run, Sayed request). **Status:** DESIGN, staged, not implemented.
**Grounded in:** the 108-agent deep-research pass (`tasks/wrsrfoqc1.output`, 25 sources → 21 confirmed /
4 refuted claims) + our own assets. **Evidence class of every external number below: PUBLISHED (cited).**

---

## 0. Thesis

Real action-labeled driving data is scarce (CAN bus / our parity corpus = thousands of hours at most);
**action-free driving video on YouTube is effectively unbounded.** If we can train an **inverse-dynamics
model (IDM)** that recovers ego **action** (steer, longitudinal accel / target-speed, yaw-rate) and ego
**metric trajectory** from raw monocular video, we can **pseudo-label YouTube at scale** and use it to
**pretrain our action-conditioned world model** — the same move that took VPT from 1,962 labeled hours to
~70,000 pseudo-labeled hours [VPT, arXiv:2206.11795], applied to driving for the first time.

This is the scalable-data lever the program has been missing: not a better loss on the 2,376-episode
parity corpus, but **10²–10³× more pretraining video** at near-zero labeling cost.

---

## 1. The verdict (research) — supervised predictive non-causal IDM head

Ranked **(a) > (b) > (c)** with high confidence on the a>b core:

| option | what | verdict |
|---|---|---|
| **(a) supervised predictive IDM head** on our **trained** encoder, CAN-labeled, applied to YouTube as a **non-causal** pseudo-labeler | **CHOSEN** | proven paradigm (VPT); our two premises (trained encoder + real CAN labels) make it the natural fit |
| (b) Genie/LAPO/LAPA latent-action model, grounded with a small labeled set | rejected as primary | engineered for the **no-label** regime we are NOT in; its failure modes bite our exact need (continuous metric precision) |
| (c) self-supervised VO + metric-calibration head | auxiliary only | essentially **unassessed** by the surviving corpus (evidence gap); use for metric scale, not as the backbone |

**Why (a).** The IDM formulation `a_t = f(o_t, o_{t+1})` [BCO, arXiv:1805.01954] is a *lower-complexity,
less-stochastic* target than a behavior-cloning policy, so a **small labeled corpus suffices** to train it
[IDM-vs-BC, arXiv:2602.02762]. Two design commitments from the evidence:

1. **NON-CAUSAL labeler.** VPT's key trick: the labeling IDM predicts `a_t` from **past *and* future**
   frames (3D temporal conv + unmasked attention), which is far more accurate than a causal policy — and
   we only need it for *offline labeling*, where the whole clip is available. Our YouTube labeler must be
   **bidirectional over the clip window**, not a causal deployment policy. [VPT, 2206.11795]
2. **Continuous regression, NOT a discrete codebook.** Genie's latent action space is deliberately tiny
   (`|A|=8`) [Genie, arXiv:2402.15391]; that quantization is a mismatch for continuous steer/accel/yaw, and
   LAPA underperforms real-action grounding on fine-grained motion with ~150 trajectories
   [LAPA, arXiv:2410.11758]. We output real-valued steer / yaw-rate / longitudinal accel / target-speed /
   metric ego-trajectory directly.

**Head architecture — the Seer/DriveWAM "forecast-then-decode" auxiliary.** A *predictive* IDM (forecast
the future latent, then decode the action from it) beats a bare regression head and scales with params
(Seer CALVIN ABC-D: 3.31 → 3.41 → 3.64 → 3.98 Avg-Len; 65M→316M monotone) [Seer, ICLR'25, arXiv:2412.15109].
DriveWAM is the **in-domain AV precedent**: "generate future video latents, then decode ego actions
conditioned on the generated future latent — inverse-dynamics action generation" [DriveWAM, arXiv:2605.28544].

> **Our synthesis (asset connection #1):** the forecast-latent path is *exactly what our WM transition
> predictor already does.* The IDM's auxiliary can **reuse the flagship predictor**: encode the window →
> (optionally roll the WM predictor forward to an imagined future latent) → decode `a_t`. The IDM and the
> world model share the trunk and the predictor; the IDM is a **readout head**, not a separate network.
> This unifies the two and is why (a) on *our* trained encoder — not a frozen DINO (REF-A's ceiling) —
> is the right substrate.

---

## 2. Architecture (concrete)

```
YouTube clip  ──►  intrinsics front-end  ──►  our trained ViT encoder (shared with WM)  ──►  IDM head  ──►  a_t, traj
   raw frames        (f-theta canon)            per-frame latents  z_{t-k..t+k}          non-causal
```

- **Input window:** `2k+1` frames centred on `t` (non-causal; e.g. k=4 → 9 frames, mirroring our WM's
  window=8). Encoder is the **flagship v1 trained ViT** (9-ch, 256px), *shared* with the world model.
- **IDM head:** a small temporal transformer (bidirectional attention over the window) → per-step readout.
  Outputs at each `t`: `steer`, `yaw_rate`, `long_accel`, `target_speed` (continuous), and the 2 s
  **metric ego-trajectory** (the same waypoint target space our planners use, `refb_labels.waypoint_targets`).
  ≈ a few M params (a readout, not a backbone).
- **Auxiliary (Seer/DriveWAM):** a second head that decodes `a_t` from the WM predictor's **imagined**
  future latent `ẑ_{t+Δ}` rather than the observed `z_{t+Δ}`. Trained jointly; at YouTube-labeling time we
  use the *observed* (non-causal) path for accuracy and keep the forecast path as the transfer objective.

---

## 3. Training recipe

- **Data:** comma2k19 + PhysicalAI-AV, **CAN steering + derived kinematics** (v0, yaw-rate, ego trajectory)
  as ground-truth actions. Both are already in our lake / parity cache.
- **Encoder:** ⚠️ **UPDATED 2026-07-22 — do NOT start frozen.** The IDM re-gate MEASURED that a *frozen*
  (and *light-FT*) our-trained encoder does **not** transfer across a rig/domain shift — cross-domain speed
  R² 0.406→0.411 inert, cross-rig −2.465 (`…/incoming/2026-07-22-idm-proof/results_regate.json`), and the
  intrinsics front-end is a no-op (f_eff already matched). V-JEPA2-AC reports the same camera-pose
  sensitivity at 1M+ h SSL (arXiv:2506.09985 §limitations). So the substrate is **multi-domain co-train,
  not frozen** (`Research/2026-07-22-encoder-strategy-and-vjepa2ac.md` §C). The C.4 multi-rig co-train
  (`results_multirig.json`) is the pre-registered fork-decider: PASS ⇒ multi-domain-cotrain OUR encoder +
  speed-prior scale head suffices; FAIL ⇒ the expensive video-SSL encoder is justified.
  **UPDATE 2026-07-22 — the fork-decider LANDED FAIL** (`results_multirig.json`: held-out rig-B light-FT
  speed R² **−1.61** vs −1.65 single-domain; data-diversity **REFUTED**, collapse is REPRESENTATIONAL). ⇒
  the substrate is now **multi-domain co-train + EXPLICIT GAIA-2 camera-parameter conditioning** (per-block
  intrinsics/extrinsics/distortion embeddings, arXiv:2503.20523 verified 3-0), NOT frozen and NOT
  diversity-alone. Full design + smoke-validated scaffolding + pre-registered camera-conditioning ablation:
  `Implementation/incoming/2026-07-22-own-dynamics-encoder/` (`DESIGN.md`, `LAUNCH_PLAN.md`,
  `PRE_REGISTRATION.md`).
- **Losses:** Huber on continuous actions + trajectory ADE + the forecast-latent auxiliary (decode action
  from `ẑ`). Optional small maneuver-class CE (reuse the v3 factorised LAT×LON vocabulary).
- **Parity firewall (asset connection #2):** the IDM is a **SIDE model**. It does **not** touch the WM
  train corpus or its parity key (`e438721ae894` / skip-hash `f09e44db`). YouTube-pseudo-labeled video is a
  **pretraining prefix** for the WM, strictly separated from the parity fine-tune; nothing here re-selects
  parity episodes. Licence tier for YouTube governed by `YOUTUBE_DASHCAM_STRATEGY.md` (likely `nc`/`refuse`
  under YouTube ToS — resolve before any re-hosting; the IDM *labels* inherit the strictest input tier per
  the TanitDataSet augmentation rule).

---

## 4. The three failure modes — and our assets against them

The research documents two robustly and flags two as **evidence gaps** (least de-risked). Our program has a
direct asset against each:

| failure mode | research status | **our mitigation** |
|---|---|---|
| **Domain gap** (IDM-labeling assumes unlabeled data in-distribution with labels; latent bottleneck collapses under moving-camera/action-correlated distractors) [2602.02762, 2502.00379] | **dominant, well-documented** | the pre-registered **cross-rig experiment** (§5) measures it directly and cheaply, on data we already have |
| **Intrinsics variance** across heterogeneous YouTube sources | **evidence gap — zero claims** | ⭐ **our f-theta canonicalization** (`ftheta_crop_resize(center="principal")`, just validated end-to-end in AlpaSim Option A, `f_eff≈266` self-check). YouTube frames, once per-source intrinsics are estimated (horizon/vanishing-point or a learned intrinsics head), canonicalize to **our F_REF=266 pinhole** — the exact distribution the encoder trained on. The normalization front-end the literature says is missing, **we already built.** |
| **Monocular metric-scale ambiguity** | **evidence gap — zero claims** | resolve scale with a **speed prior**: our actions include `v0`; many dashcams show a speedometer (OCR) or we assume a per-clip speed distribution; predict *scale-normalized* trajectory + a speed head, recover metric scale from the prior. Optionally borrow (c)'s VO as a scale auxiliary. |

> **Asset connection #3 — the two PhysicalAI rigs as a built-in intrinsics testbed.** PhysicalAI-AV
> contains **two camera rigs** (front-wide cy≈543 rig-A / cy≈755 rig-B — MEASURED, our data note). Train the
> IDM on rig-A, test on rig-B *within the same corpus* — an even cheaper intrinsics-variance probe than
> comma2k19, with identical scenes/labels, isolating pure intrinsics shift.

---

## 5. Cheapest discriminating experiment (PRE-REGISTERED — both outcomes committed)

Per program rule 5 (settle conflicts with a pre-registered experiment, both outcomes committed in advance):

- **Setup:** train the supervised predictive-IDM head on **PhysicalAI-AV CAN labels only**, frozen encoder,
  **zero comma2k19 in training**. Score metric ego-action/trajectory recovery on **held-out comma2k19**
  (different vehicle/camera/intrinsics, already CAN-labeled → ground truth is free). Uses ONLY existing
  assets: no YouTube, no new labels, a small head (doesn't touch WM parity).
- **Metrics (mirror VPT's held-out R²):** speed R², yaw-rate R², steering R², trajectory ADE@2s vs CAN.
- **Committed decision:**
  - **PASS** if cross-rig **speed R² > 0.9 AND yaw R² > 0.9 AND traj ADE@2s < [bound TBD from the in-rig
    number ×1.5]** → the domain/intrinsics gap is tolerable → **scale the supervised IDM to YouTube**.
  - **FAIL** otherwise → the domain-gap/intrinsics mode dominates → add the **f-theta canon front-end +
    speed-prior scale head** (our assets, §4) and/or a VO auxiliary, re-test, *before* any YouTube spend.
- **Optional 2nd arm (a-vs-b at matched cost):** train a LAPO/LAPA latent-action head grounded on the same
  small labeled subset; compare metric recovery at **150 vs full** trajectories. Committed: if latent
  grounding matches the supervised head at low budgets, (b) is viable; else (a) is confirmed.
- **Even cheaper pre-probe:** the **rig-A→rig-B** within-PhysicalAI split (§4) — run first, hours not days.

---

## 6. Integration with the world model

1. Train IDM on CAN corpora (§3). 2. Pass §5. 3. Canonicalize + IDM-label a YouTube slice. 4. **Pretrain**
the flagship trunk + action-conditioned predictor on pseudo-labeled YouTube. 5. **Fine-tune** on the parity
corpus (sacred, unchanged). This is the GenAD/Vista pattern (pretrain on ~2000h / ~1740h action-free
YouTube, then ground) — except we bridge to action by **IDM recovery**, which no published driving WM does
(GenAD/Vista use forward-conditioning / null-action co-training) [GenAD 2403.09630; Vista 2405.17398].
**Our proposal occupies genuinely open territory** (the research's own words) — VPT (games) and DriveWAM
(labeled AV policy) are the nearest anchors; the novel claim is *supervised IDM pseudo-labeling of
uncalibrated YouTube driving video to metric actions, improving a downstream action-conditioned WM.*

---

## 7. Open questions to close (from the research's evidence gaps)

1. ✅ **CLOSED 2026-07-22** (`Research/2026-07-22-encoder-strategy-and-vjepa2ac.md` §A). **V-JEPA 2 / V-JEPA2-AC**
   (arXiv:2506.09985) freezes a 1M+ h video-pretrained encoder + trains a ~300M AC head on <62 h robot data.
   It **validates our head shape** (small, latent-predictive, teacher-forcing + rollout) BUT its
   "zero-shot" is same-embodiment/new-lab, and it **openly reports the same camera-pose sensitivity we
   measured** ("manually tried different camera positions") — published evidence that SSL scale does NOT buy
   rig-invariance for free. Net: validates the head, red-flags the frozen substrate (see §3 update).
2. **Monocular metric-scale recovery** for uncalibrated dashcam video — known-height vs speed-prior vs
   learned-scale — is named but unquantified; decide the scale mechanism and its residual error.
3. **Min CAN budget** to ground driving-metric precision is unknown (LAPA's 150 traj was for grasping, not
   driving kinematics) — the §5 experiment should also sweep the label budget.
4. **Refuted claims to NOT repeat** (transparency): VPT's "~2 orders of magnitude more label-efficient than
   BC" (refuted 0-3 — use only the modest lower-complexity/less-stochastic argument); BCO's "random-explore
   then apply" two-phase framing (refuted 0-3); "egocentric latent actions cleanly capture camera motion"
   (refuted 1-2 — a point *against* (b) for ego-motion).

---

## 8. Immediate next step

Launch the **§5 cheapest experiment** (rig-A→rig-B pre-probe, then PhysicalAI→comma2k19) — a small head on
the existing encoder + CAN caches, no YouTube, no parity impact, both outcomes committed. It de-risks the
single dominant failure mode (domain/intrinsics transfer) before any YouTube engineering, and its number is
the go/no-go for the whole line.

**Sources (primary):** VPT 2206.11795 · BCO 1805.01954 · IDM-vs-BC/LAPO+ 2602.02762 · Seer 2412.15109 ·
DriveWAM 2605.28544 · Genie 2402.15391 · LAPA 2410.11758 · LAPO (schmidtdominik/LAPO) · distractor-collapse
2502.00379 · CLAM 2505.04999 · GenAD 2403.09630 · Vista 2405.17398. Full claim ledger + votes:
`tasks/wrsrfoqc1.output`.
