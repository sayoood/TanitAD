# Launch plan — OUR rig-robust dynamics-estimation encoder

> 🟥 **OUTCOME 2026-07-24 — Branch B (§4) ran to 40k and FAILED the held-out-rig transfer gate (`MEASURED`).**
> Best cross-rig speed R² **−0.667** (gate +0.9), weaker than plain frozen flagship-v1 (+0.657) paired on
> 3/4 arms → explicit camera-conditioning did **not** engineer rig-robustness at 40k/2466-clip scale.
> Full: `../../2026-07-24-branchb-transfer-eval/RESULTS_branchB.md`; `Project Steering/MODEL_REGISTRY.md §10`.
> Both Branch A (§3, refuted) and Branch B (§4, refuted) are now spent; the surviving lever is a
> **flagship-warm-started** encoder variant (HYPOTHESIS, **Sayed-gated new arm, not auto-launch**).

**Status:** PRE-REGISTERED, NOT launched. A multi-GPU-day run needs (1) the §1 camera-conditioning
ablation verdict and (2) Sayed's go. This stream built everything up to that point.
**Evidence class:** compute numbers are `ESTIMATED` (anchored to a `MEASURED` flagship datum); decision
rules are `HYPOTHESIS` (pre-committed). **Nothing below a decision rule may be edited after its first
result is read.**

---

## 0. Why the branch structure changed (the multi-rig verdict landed)

The earlier plan forked on *"is the collapse data-diversity (cheap) or representational (expensive)?"*.
That is now **answered — REPRESENTATIONAL** (`MEASURED`, `results_multirig.json`: held-out rig-B light-FT
speed R² **−1.61** vs −1.65 single-domain; data-diversity **REFUTED**). So the **camera-conditioned
encoder is the recipe, not a fallback**, and the fork is now about **how much it costs to install the
mechanism**: cheaply on the warm-started trained encoder (Branch A), or from-scratch jointly with the SSL
objective (Branch B). The decisive, cheap experiment (§1) chooses.

---

## 1. GO/NO-GO — the camera-conditioning ablation (cheapest discriminating experiment)

> **⚑ VERDICT LANDED 2026-07-23 — FAIL ⇒ Branch B is the go.** (MEASURED, `RESULTS_camcond.md` /
> `results_camcond_{rig,multirig}.json`.) Conditioning ON vs OFF, warm-started from the md5-verified v1:
> cross-rig speed R² **−2.344→−2.253** (rig, Δ+0.091) and **−2.176→−2.057** (multirig, Δ+0.119) — a
> consistent but **marginal** lift, both failing the 0.9 gate (ADE ratio ~3.7 vs 1.5). The mechanism is
> **not refuted** (ON>OFF every time, speed *and* yaw), but the cheap warm-start suffix-conditioning
> shortcut (**Branch A**) is — a +0.1 nudge on a −2.2 collapse. **Proceed to Branch B** (§4): from-scratch,
> all-block conditioning, multi-rig — the full GAIA-2 regime the probe deliberately lacked.

Pre-registered in full in `PRE_REGISTRATION.md`. One-paragraph form:

> **Add GAIA-2 per-block camera conditioning to the IDM re-gate harness and re-run the SAME gate.** Change
> exactly ONE thing vs the multi-rig cotrain: the encoder gains the `CameraConditionedEncoder` conditioning
> (intrinsics/extrinsics/distortion, per-block), fed the per-domain camera params. Arms: **conditioning
> ON vs OFF**, co-trained on {rig-A + comma}, evaluated on **held-out rig-B**. Gate (unchanged):
> **cross speed R² > 0.9 AND yaw R² > 0.9 AND ADE@2s < 1.5× in-domain**. ~hours on pod3, reuses the
> 70-ep infra (~20 min/arm), touches no parity key.

- **PASS or material recovery** (cross-rig speed R² lifts decisively from the −1.61 floor toward/above
  0.9): the mechanism works → **Branch A** (warm-start + conditioning at modest scale) is the recipe;
  Branch B only if YouTube-scale data later demands a from-scratch SSL encoder.
- **FAIL / marginal** (conditioning bolted onto a light-FT PhysicalAI-only encoder does not lift cross-rig
  transfer): the conditioning must be **learned from scratch jointly with the SSL objective** → **Branch
  B** is justified, and this is its pre-registered go. Escalate the §3-fallback geometry-as-input options
  (Plücker/PRoPE) inside Branch B.

The ablation is **committed both ways before any number is read**. It is the single cheapest cut that
decides the GPU-day spend, exactly as the multi-rig cotrain was for the diversity question.

---

## 2. The launch config (frozen)

`DynEncConfig()` defaults (`stack/tanitad/models/dynamics_encoder.py`), MEASURED param budget:

```
backbone      : 9-ch 256px, patch16 -> 16x16 tokens, d_model 768, depth 12, heads 12
readout       : spatial grid 4x4 x d_readout 128 -> state_dim 2048   (flagship parity)
camera cond.  : GAIA-2 intrinsics/extrinsics/distortion embeds (cam_hidden 128),
                summed, per-block zero-init injection (depth 12)
window        : 9 (k=4 non-causal IDM) ; forward-pred history 8 ; mask_ratio 0.4
objective wts : w_idm 1.0 · w_fwd 1.0 · w_mask 1.0 · w_sigreg 0.1 · w_ground 1.0
optimizer     : AdamW lr 3e-4 wd 0.05, warmup, grad-clip 1.0, batch 32 (per-GPU)

param budget (MEASURED, instantiated):
  ViT backbone 87.02M · GAIA-2 cond 7.39M · readout 0.10M · IDM head 2.90M
  => DEPLOYABLE 97.41M  (sub-300M envelope; ~200M headroom)
  + predictor 3.49M + masked-pred 2.63M + invdyn 2.37M  => TOTAL(train) 105.90M
```

Loss-weight rebalance is a launch-time sweep at a mid checkpoint (the flagship discipline); the smoke used
equal weights and it already trains, but the SSL/forward terms should be tuned so they neither swamp nor
starve the supervised IDM (the smoke showed SIGReg at ~5–14 raw scale → its λ=0.1 keeps it ~0.5–1.4 of the
total, which is the LeJEPA operating point).

---

## 3. Branch A — warm-start + GAIA-2 conditioning + multi-domain light co-train (if §1 PASSES)

**Recipe.** Warm-start the ViT backbone + readout from **flagship-v1** (`maybe_warm_start`; zero-init
conditioning ⇒ identical forward at step 0), then co-train the **full objective** on the multi-domain mix
(PhysicalAI rig-A + rig-B + comma2k19 + L2D) with geometry domain-randomisation, letting the whole encoder
(incl. conditioning) adapt. The IDM head + metric grounding are supervised on CAN; masked-latent +
forward-pred are the SSL scaffolding.

**Command (pod3 or pod1, NON-training pod, under `gpu_lock.sh acquire dyn-encoder`):**
```
PYTHONPATH=/workspace/TanitAD/stack python scripts/train_dynamics_encoder.py \
  --pai-cache /workspace/pai_epcache/physicalai-train-e438721ae894 \
  --pai-rig-table /workspace/tmp/idm/rig_table.json \
  --comma-cache /workspace/data/comma2k19-val-61c46fca8f7f \
  --l2d-cache /workspace/data/l2d-slice \
  --warm-start /workspace/tmp/idm/ckpt.pt \
  --steps 15000 --batch 32 --max-dv 12 --out experiments/dyn-encoder-A
```

**Compute** (`ESTIMATED`, anchored to the `MEASURED` flagship 10.888 s/step, 90.7 h / 30k on one A6000):
warm-started ⇒ ~**10–15k steps** ⇒ ~**1.5–2.5 GPU-days** on one A40/A6000. Plus the §1 ablation probe
(~hours). This is the modest-scale recipe.

**Success read** (post-run, on the held-out rig + comma): the same gate as §1 on the *fully trained*
encoder, plus the in-dist IDM staying near the frozen-encoder in-dist ceiling (speed R² ~0.93).

---

## 4. Branch B — from-scratch camera-conditioned video-SSL encoder (PRIMARY scale target; if §1 FAILS or for YouTube)

> 🟥 **RAN & REFUTED 2026-07-24.** Launched 2026-07-23 (`BRANCHB_LAUNCH.md`), trained to 40k, and **FAILED**
> the held-out-rig transfer gate: cross-rig speed R² **−0.667**, weaker than plain frozen flagship-v1
> (`MODEL_REGISTRY §10`; `../../2026-07-24-branchb-transfer-eval/`). The Plücker/PRoPE escalation below was
> **not** reached; any further encoder-line spend must be **re-pre-registered** against this evidence.

**Recipe.** Train the encoder **from scratch** (no warm-start) with the masked-latent SSL as the dominant
objective + forward-pred + SIGReg + camera conditioning learned jointly, on the multi-domain mix; add the
supervised IDM + metric grounding as the action/metric anchors. For the true video-ViT, swap the 3-frame-
stack backbone for a **tubelet + temporal-attention** stack (still sub-300M — the 200M headroom covers a
d768 tubelet ViT or a widen to d1024). This is the encoder that later ingests **YouTube-scale** IDM-labeled
video (the data lever) once the in-house gate passes.

**Compute** (`ESTIMATED`): from-scratch ⇒ ~**30–60k steps**, heavier per step with temporal attention ⇒
~**4–8 GPU-days** on one GPU, or **~1–2 days on 4 GPUs**. The **YouTube pretraining phase** (after the
in-house encoder passes) is a **separate, larger** commitment (~**10s–100s GPU-days** depending on hours
ingested) and gets its **own** pre-registration + Sayed go — not part of this launch.

**Escalation inside Branch B** (if global conditioning still misses the gate): add the §3-fallback
**geometry-as-input** signals — per-pixel **Plücker ray-maps** concatenated to the 9-ch input (2510.02268)
and/or **PRoPE** projective relative positional encoding (2507.10496) — pre-registered as the next lever,
measured before any further scale.

---

## 5. What must be true before ANY launch (checklist)

1. **§1 ablation verdict is in** and points to Branch A or B (do not launch a multi-GPU-day run on a
   HYPOTHESIS the ablation can settle in hours). ✅ built; ⏳ not yet run.
2. **Sayed's explicit go** for the chosen branch's GPU-day spend.
3. **pod3/pod1 free** and NOT training (pod2 = v4.1; eval = AlpaSim); `gpu_lock.sh acquire dyn-encoder`
   with the job PID.
4. **Data caches present:** PhysicalAI rig table (`run_idm_proof.py --stage rig`), comma cache, L2D slice.
   L2D adapter emits the episode contract with **estimated+flagged** intrinsics (`l2d.py`).
5. **Parity firewall check:** confirm the trainer reads rig/corpus splits only, never re-selects
   `e438721ae894`. (`build_domains_from_caches` uses `select_episodes` by rig, never the parity key.)
6. **`pytest -q` green in `stack/`.** ✅ 778 passed, 2 skipped with this stream's additions.

---

## 6. Risks & mitigations (named before launch — C6/C3 discipline)

| risk | mitigation |
|---|---|
| Conditioning learns to **ignore** the camera input (the FiLM/inject stays ~0) | geometry domain-randomisation makes ignoring it *costly* (the perturbation is only explainable via the fed params); the §1 ablation measures |Δz| response directly (smoke: 2.7e-2, live). |
| SSL terms **collapse** the latent | SIGReg (LeJEPA, validated λ=0.1) + supervised IDM/grounding anchors with real targets; smoke shows all five terms co-train without collapse. |
| Metric scale rides a **rig-specific** visual cue | supervised odometry grounding (Cosmos-3-validated) + speed-prior head decouple scale from any single rig's pixels. |
| L2D/YouTube **unknown intrinsics** feed garbage | the known/unknown **mask** makes "unknown" in-distribution; never assert a false f_eff as truth. |
| Per-step compute over-estimate (C1/C5) | the flagship anchor is `MEASURED` (10.888 s/step); the encoder is similar-scale; ranges are wide and honest, and a real step-time is logged in the first 100 steps before extrapolating. |
