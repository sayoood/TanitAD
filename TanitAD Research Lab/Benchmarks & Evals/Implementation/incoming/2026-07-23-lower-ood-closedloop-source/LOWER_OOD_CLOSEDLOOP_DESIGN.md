# A lower-OOD closed-loop eval source — design, prototype, and Gate-1 recommendation

**Date:** 2026-07-23 (Berlin) · **Host:** `tanitad-pod` (pod1, RTX A6000) · **Status:** prototype MEASURED, design + recommendation complete.

**Author's note on evidence class.** Every number below is tagged `MEASURED` (ours + artifact),
`PUBLISHED`/`INHERITED` (registry/other-agent, not re-verified here), or `HYPOTHESIS`/`ESTIMATED`. A
claim that would decide a GPU-day is MEASURED or it is flagged as not-yet-decidable.

---

## 0. The blocker this attacks

Every closed-loop number in the program is confounded by **NuRec / AlpaSim reconstruction OOD**. The
control that proved it (Sayed's idea, 2026-07-22, `RETRACTION_LOG` 07-22 C6):

| quantity | value | source |
|---|---|---|
| REF-C base open-loop ADE@2s on **real** PhysicalAI val | **0.4728** | INHERITED (taniteval) |
| REF-C base open-loop ADE@2s on **NuRec reconstructions** (force-GT, 4 scenes, 288 preds) | **1.5157** | MEASURED (`REFC_openloop_diagnostic.json`) |
| **reconstruction-OOD ratio** | **3.21×** | MEASURED |

Because the model is fed observations ~3.2× off its training distribution, **every within-sim
closed-loop score conflates model quality with reconstruction fidelity**. This blocks a clean Gate-1
read and makes all closed-loop numbers "relative only." Breaking it is the point of this work.

**The structural insight.** Closed-loop needs an observation at the ego's *actual* pose, which under a
deviating planner is off the recorded path. There are only two ways to get one:
1. **Synthesize a novel view** at the deviated pose → this is what NuRec does, and the synthesis *is*
   the OOD (appearance artifacts, floaters, geometry error across the whole frame).
2. **Show the real recorded frame** and accept that it matches only while the ego stays near the
   recorded pose → **zero reconstruction OOD**, traded for a **pose-mismatch** that is 0 on-path and
   grows with deviation.

This document quantifies option 2's pose-mismatch, because if the usable deviation envelope is wide
enough, real-footage log-replay is a *strictly cleaner* closed-loop source than any renderer.

---

## 1. The three candidate sources — design, feasibility, expected OOD

### (a) Real-footage log-replay + kinematic ego model  ⭐ recommended
Drive on the **real recorded frames**; integrate the ego forward with a kinematic bicycle model from
the planner's action. The observation is always a real frame (**reconstruction OOD ≡ 0**). Two
deviation axes behave completely differently:

- **Longitudinal (speed / spacing) — served OOD-free by arc-length re-indexing.** The recorded footage
  is a 1-D manifold of *real* observations sampled along the path at spacing `v·Δt` (`MEASURED`:
  median highway speed ≈ 12–13 m/s × 0.1 s ⇒ **1.2–1.3 m/frame**; ≤ ~2.5 m at the 25 m/s tail). If the
  planner ends up at arc-length `s` at sim-time `t`, show the **real** frame whose recorded arc-length
  is nearest `s` (optionally linearly interpolate the two bracketing frames). The shown frame is a
  genuine observation from within **≤ ½ frame-spacing (~0.6 m)** of the ego's along-track position —
  exactly the forward jitter the model sees every step in training. **Longitudinal closed-loop therefore
  carries ~0 added OOD.** This matters enormously: the flagship's residual is **89 % longitudinal**
  (`MODEL_REGISTRY §1.2`), so the source covers the model's #1 weakness with zero reconstruction OOD.
- **Lateral / heading — bounded by a deviation envelope.** The car never strafes, so a lateral/heading
  offset has no real frame to re-index to; the recorded frame becomes progressively wrong. **§2 measures
  exactly how wrong, as a function of the offset.**

**Feasibility: HIGH — nothing to build but the loop.** No renderer, no NGC image, no Vulkan/OptiX. Needs
only the model + the real val frames (both already on pod1) + a bicycle integrator + an arc-length
re-index. The prototype in §2 is this harness in its open-loop (force-GT) limit.

**Expected OOD: ~0 longitudinally, small and bounded laterally (MEASURED in §2).**

### (b) A lighter / tuned renderer with lower OOD than NuRec
Keep synthesizing, but reduce the synthesis gap: native **1080×1920** instead of the 480×854 we ran
(`RETRACTION_LOG` 07-23 C5 flags resolution as a live residual), post-hoc **appearance matching**
(color/sharpness transfer to close the sim-real gap), or a denser per-scene 3DGS/NeRF.

**Feasibility: LOW–MEDIUM, and mostly out of our hands.** NuRec ships only as the NGC `nre-ga` image;
we do **not** train it and cannot change its reconstruction quality — only resolution and
post-processing are ours. The render service lives on the eval pod (off-limits this run) and is heavy
(`LOOP_STATE`: renderer :6011 + physics + controller topology).

**Expected OOD: > 0 and unverified.** Even in the best case a synthesized frame keeps *some* appearance
gap; there is no reason to expect it below option (a)'s ~0. Resolution/appearance tuning **might** pull
1.52 down, but by an unmeasured amount. `HYPOTHESIS`, and expensive to test.

### (c) Hybrid — real frames near GT, reconstruction only in the tail
Use real-footage re-index while `|deviation| ≤ envelope`; fall back to the renderer only when the ego
leaves the envelope. Bounds reconstruction OOD to the **rare large-deviation tail**.

**Feasibility: MEDIUM, but it inherits (b)'s renderer dependency** (you still need NuRec for the tail)
**plus** a switching criterion and world-frame registration between the two observation sources.

**Expected OOD: ~0 in-envelope, NuRec-level only in the tail.** Given §2's wide measured envelope, the
tail branch would fire *rarely* for a well-behaved planner — so the hybrid is "(a) with a renderer
safety net," valuable once a renderer is cheap, but strictly heavier than (a).

---

## 2. Prototype (option a) — MEASURED on pod1

**What was run (`MEASURED`).** The deployed flagship v1 (`/root/models/flagship-30k/ckpt.pt`, step
29999, speed-input, `grounding` present) on the **clean** val split
`/root/valdata/physicalai-val-0c5f7dac3b11` (**12 episodes → 265 windows**; the canonical 40-ep set
lives on the eval pod). The rollout is **byte-for-byte `scripts/eval_grounded_rollout_4b.py`** at Δ=0
(encode window → operative predictor 20-step rollout under TRUE actions → per-step Δpose via
`grounding.step['op']` → SE(2) → ADE@{0.5,1,1.5,2}s). The **only** change per condition is an
**observation-only** warp of the input frames simulating the ego being offset from the frame's capture
pose (the fed speed `v0` and the GT trajectory stay the true ego). Harness: `lowood_probe.py`; raw:
`lowood_envelope.json`; geometry selfcheck passes (Δ=0 warp identity to 1.5e-5; a +0.5 m lateral shift
moves a near-field ground marker left by the geometrically-exact 27 px). Intrinsics: canonical pinhole
`f_eff=266`, 256², centered principal point (`build_pai_cache` asserts `|f_eff−266|<8`), nominal camera
height 1.5 m, level. Lateral = flat-road ground-plane homography; yaw = exact rotation homography;
pixshift = calibration-free column roll (cross-check).

### 2.1 Headline — reconstruction OOD is eliminated

| source (open-loop ADE@2s, force-GT, SAME protocol) | ADE | ratio to real |
|---|---:|---:|
| **Real-footage log-replay (this prototype, Δ=0)** | **0.4045** | **1.00×** |
| NuRec / AlpaSim reconstruction (REF-C control) | 1.5157 | 3.21× |

The log-replay harness reproduces the model's real-footage level (registry full-set 0.4271 on 40 eps;
0.4045 on this 12-ep subset — representative). **Reconstruction OOD = 0 by construction, confirmed.**

### 2.2 The deviation envelope — MEASURED

| Δ_lat (m) | ADE@2s | vs base | | Δψ (deg) | ADE@2s | vs base |
|---:|---:|---:|---|---:|---:|---:|
| 0.0 | 0.4045 | — | | 0 | 0.4045 | — |
| 0.25 | 0.3977 | −1.7 % | | 1 | 0.4040 | −0.1 % |
| 0.5 | 0.4014 | −0.8 % | | 2 | 0.4080 | +0.9 % |
| 0.75 | 0.4020 | −0.6 % | | 3 | 0.4211 | +4.1 % |
| 1.0 | 0.4071 | +0.6 % | | 5 | 0.4303 | +6.4 % |
| 1.5 | 0.4219 | +4.3 % | | 8 | 0.4438 | +9.7 % |
| 2.0 | 0.4299 | +6.3 % | | 12 | 0.4596 | +13.6 % |
| 3.0 | 0.4703 | +16.3 % | | | | |

**Reading it.** Across the **entire** plausible closed-loop deviation range — **±3 m lateral** (nearly a
full lane) and **±12° heading** — the real-frame source's open-loop ADE stays **≤ 0.47 = 1.16× its own
baseline**. It **never approaches** NuRec's 1.52 (3.21×). The pose-mismatch OOD of even a large
real-frame offset is **~14× smaller** than NuRec's reconstruction OOD (peak +16 % vs +221 %). The
"usable envelope before frame-mismatch dominates" is therefore **not the binding constraint** in any
realistic deviation range; at a strict +20 % operating tolerance the envelope is roughly
**Δ_lat ≲ 3 m and Δψ ≲ 12°**. (pixshift cross-check trends identically — `lowood_envelope.json`.)

**Why real-frame offset is so much gentler than NuRec.** A lateral homography of a real frame keeps
real textures/appearance and only geometrically shears the ground plane — which the encoder is robust
to — whereas NuRec corrupts appearance (blur, floaters, color, geometry) across the **whole** image,
which throws the encoder latent globally. **Appearance fidelity, not geometric pose-exactness, is what
keeps the encoder in-distribution** — and real footage has appearance fidelity for free.

---

## 3. Honest limits (state them or don't state the claim)

1. **Protocol leverage (the load-bearing caveat).** The open-loop force-GT operative rollout is
   conditioned on the **true future actions**, so the observation's leverage on ADE is *partial*. What
   §2 measures is precisely the **observation-OOD of the source as seen by the same metric that
   produced both 0.47 and 1.52** — i.e. the confound we set out to remove, measured apples-to-apples.
   It does **not** yet measure how a closed-loop *planner* (choosing actions from the warped
   observation) reacts to residual pose-mismatch; that leverage is larger and must be validated in the
   actual closed loop before a Gate-1 number is quoted. Vision *does* have real leverage on this metric
   (yaw 12°/lat 3 m move it +14–16 %; NuRec moves it +221 %) — the contrast is the valid finding.
2. **The lateral homography is ground-plane-only** → it under-models 3D-structure parallax, so the
   measured lateral envelope is an **optimistic upper bound**; a true novel view would be somewhat
   worse (still far below NuRec). Yaw is an **exact** rotation homography (depth-independent) — trust it
   most. Calibration (h, pitch) barely matters here: the envelope is so flat that a ±50 % error in
   camera height cannot lift it near 1.52.
3. **n = 12 episodes / 265 windows, single seed, one model.** Enough to settle the direction (the
   effect sizes are large and monotone); a Gate-1 commit should re-run on the full 40-ep clean val and
   add REF-C (its diffusion planner may weight vision differently).
4. **Longitudinal-free is a geometric argument** (frame spacing + re-index), not a model measurement —
   but it needs none: the shown frame is a *real* observation within ≤0.6 m of the ego's along-track
   position.

---

## 4. Recommendation for the Gate-1 clean-eval source

**Build Gate-1 on option (a): real-footage log-replay + kinematic ego, with arc-length re-index.**
The evidence:

- **Reconstruction OOD = 0** (MEASURED 0.4045 vs NuRec 1.5157) — the confound is removed, not reduced.
- **Longitudinal deviation carries ~0 added OOD** (geometry) and **covers the flagship's dominant
  failure mode** (89 % longitudinal). A longitudinal-first Gate-1 (target-speed / spacing / stop-line
  behaviour on real frames) is *both* the cleanest read available *and* aimed straight at the weakness.
- **Lateral / heading deviation stays ≤ 1.16× real out to ±3 m / ±12°** (MEASURED) — wide enough that a
  reasonable planner spends nearly all its time in a near-zero-OOD regime.

**Deviation / coverage limits to state honestly in the Gate-1 card:**
- Valid while the ego stays within the envelope (operating point ≈ **Δ_lat ≤ 3 m, Δψ ≤ 12°** at +20 %
  tolerance; the geometry-only homography makes this an optimistic bound — verify with the closed-loop
  planner in the loop before quoting a hard limit).
- Agent-interaction failure modes (crossing traffic at intersections; `GATE1_ROLLOUTS_NOTE`:
  5/7 intersection at-fault collisions) are **not** exercised by static log-replay — the recorded other
  agents don't react to the ego. Real-footage log-replay scores **route-following / lane-keeping /
  longitudinal control**, not reactive collision avoidance. Use AlpaSim (with its OOD caveat) or the
  hybrid (c) for reactive-agent scenarios.

**Sequencing.** Ship (a) now for the longitudinal + lane-keeping Gate-1 read (the confound-free core).
Add **(c) hybrid** later — real frames in-envelope, renderer only in the rare large-deviation tail —
once a renderer is cheaply available, to extend coverage to large lateral excursions and reactive
agents without paying NuRec's OOD except in the tail. Skip (b) as a standalone: it keeps the synthesis
gap it was meant to shrink.

**One-line verdict.** *A real recorded frame, even shown to an ego offset by 3 m or 12°, is ~14× closer
to the model's training distribution than a NuRec reconstruction of the same scene — so the clean
Gate-1 source is the real footage itself, driven longitudinally on-rails and laterally within a
measured envelope.*

---

## 5. Deliverable manifest

| artifact | where it lives | what it is |
|---|---|---|
| `LOWER_OOD_CLOSEDLOOP_DESIGN.md` | repo (staged), this dir | design comparison (a/b/c) + prototype results + Gate-1 recommendation |
| `lowood_probe.py` | repo (staged), this dir · pod `tanitad-pod:/workspace/lowood_probe.py` | the OOD-characterization harness (self-contained; Δ=0 == `eval_grounded_rollout_4b.py`) |
| `lowood_envelope.json` | repo (staged), this dir · pod `/workspace/lowood_envelope.json` | ⭐ MEASURED sweep: baseline + lat/yaw/pixshift conditions + selfcheck |
| `lowood_run.log` | repo (staged), this dir · pod `/workspace/lowood_run.log` | run stdout (per-condition ADE, `LOWOOD_DONE`) |

**Inputs (pre-existing, pod-side):** flagship v1 ckpt `tanitad-pod:/root/models/flagship-30k/ckpt.pt`
(step 29999); clean val split `tanitad-pod:/root/valdata/physicalai-val-0c5f7dac3b11` (12 eps).
**Pod left clean:** `gpu_lock` released, GPU 0 %/2 MiB; no deletions; harness + outputs left in
`/workspace` (small, regenerable). **Reproduce:**
`gpu_lock.sh acquire lowood; PYTHONPATH=/workspace/TanitAD/stack python3 /workspace/lowood_probe.py --out /workspace/lowood_envelope.json; gpu_lock.sh release lowood`.

**Not done / next (for whoever takes Gate-1):** (1) re-run on the full 40-ep clean val + add REF-C;
(2) close the loop — bicycle integrator + arc-length re-index + a *planner choosing actions* from the
warped observation, to measure the planner's residual pose-mismatch sensitivity (§3.1); (3) build the
(c) hybrid switch once a renderer is cheaply available.

**Staging note:** per the Agent Operating Standard these files are `git add`-ed into the working tree
and **not committed / not pushed**; the index may contain other agents' concurrent work — commit with
an explicit pathspec.
