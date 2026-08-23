# PRE-REGISTRATION — does GAIA-2 explicit camera-conditioning recover cross-rig transfer?

**Written 2026-07-22, BEFORE any camera-conditioning result was seen.** This is the committed gate for the
cheapest discriminating experiment on OUR dynamics-encoder line — the single cut that decides the GPU-day
spend (`LAUNCH_PLAN.md` §1). Program rule 5: both outcomes committed in advance.

**Author:** dynamics-encoder stream. **Evidence class of every threshold below: HYPOTHESIS (pre-committed
decision rule).** Every measured number lands in `results_camcond.json` as `MEASURED (ours + artifact)`.

---

## 0. The claim under test

> Adding **GAIA-2-style explicit camera-parameter conditioning** (separate intrinsics/extrinsics/
> distortion embeddings, summed, injected per transformer block) to our trained driving encoder recovers
> ego-motion recovery **across a held-out camera rig** — where multi-domain co-training alone did NOT
> (`results_multirig.json`: held-out rig-B light-FT speed R² **−1.61**).

This is the exact next question after the multi-rig verdict. The multi-rig cotrain changed the *data*
(add a domain) and failed. This changes the *architecture* (add explicit geometry) — the one lever GAIA-2
credits its rig-generalization to (arXiv:2503.20523, verified 3-0) and the one neither our cotrain nor
V-JEPA2-AC had.

## 1. Fixed apparatus (frozen before running)

- **Encoder:** `CameraConditionedEncoder` (`stack/tanitad/models/dynamics_encoder.py`), ViT + readout
  **warm-started from flagship-v1** (`flagship4b-speedjerk-30k`, `ckpt.pt`, MODEL_REGISTRY §1.2), the same
  ckpt the re-gate/multi-rig used (md5 in the results JSON). Conditioning modules zero-init ⇒ step-0
  forward identical to the re-gate baseline.
- **Head + grounding:** the existing non-causal `IDMHead` (state_dim 2048) + `MetricInverseDynamics`
  odometry grounding — identical to the multi-rig harness so the ONLY new variable is the conditioning.
- **Training:** co-train on **{rig-A + comma2k19}** with per-domain camera params + geometry domain-
  randomisation (`geom_augment`); the encoder suffix + conditioning + head train (the multi-rig light-FT
  regime: last 4 ViT blocks + norm + readout + conditioning + head), enc-suffix lr 5e-5, head lr 3e-4.
- **Ground truth:** CAN-derived speed/yaw/steer/accel + 2 s ego trajectory (the frozen §-contract from
  `PRE_REGISTRATION.md` of the IDM proof — unchanged).
- **Parity firewall:** SIDE model; rig/corpus splits only; never touches `e438721ae894` / `f09e44db`.

## 2. The two arms (both run; both reported) — capacity-matched

- **ON (camera-conditioning):** the conditioning modules receive the **true per-domain camera params**
  (intrinsics/extrinsics/distortion + known mask).
- **OFF (control):** the **identical model** (same params, same capacity), but the camera vector is
  replaced by a **constant all-unknown vector** (mask 0) for every clip — so the conditioning path carries
  **no geometry information** while the parameter count / capacity is byte-identical to ON. This isolates
  *camera information* from *extra capacity* (C6: the ON−OFF delta cannot be a capacity artefact).

Direct comparison point: the multi-rig cotrain **light-FT** arm (no conditioning) = held-out rig-B speed
R² **−1.61** — the OFF arm should reproduce it (a sanity check that the harness is unchanged).

## 3. Splits (frozen)

- **Train:** rig-A (60 clips) + comma2k19 (40 clips), window-balanced — identical to `run_idm_ft.py`
  `--experiment multirig`.
- **Cross (the go/no-go):** held-out **rig-B** (never in training).
- **In-domain ceiling:** held-out **rig-A** (same corpus, episode-disjoint).
- Symmetric confirmation arm (secondary): train {rig-A + rig-B}, held-out **comma2k19**.

## 4. COMMITTED DECISION RULE (frozen before any number)

Gate unchanged from the baseline / re-gate / multi-rig:

> **PASS** iff held-out cross **speed R² > 0.9 AND yaw R² > 0.9 AND traj ADE@2s < 1.5× in-domain held-out
> ADE@2s**, for the **ON** arm on the **held-out rig-B** contrast.

- **PASS (ON reaches the gate):** explicit conditioning solves cross-rig on the *warm-started* encoder ⇒
  **`LAUNCH_PLAN` Branch A** (modest-scale warm-start + conditioning) is the recipe; re-confirm on full
  data, then proceed toward the multi-domain IDM.
- **MATERIAL RECOVERY (ON lifts cross-rig speed R² decisively from the −1.61 OFF floor toward ≥ ~0.5–0.9
  but misses the gate):** the mechanism is real but under-powered when bolted onto a warm-started
  PhysicalAI-only encoder ⇒ **Branch B** (from-scratch camera-conditioned video-SSL) is justified, with
  the conditioning learned jointly; report the magnitude, do not force a pass.
- **FAIL (ON ≈ OFF, no material lift):** explicit conditioning as a warm-start add-on is insufficient ⇒
  **Branch B** justified AND escalate the geometry-as-input fallbacks (Plücker ray-maps / PRoPE) inside
  it; report plainly, do not force a pass.

The **headline number** is ON − OFF cross-rig speed R² (and vs the −1.61 multi-rig floor). The direction is
reported regardless.

## 5. Pre-registered confounds (C6 — named before reading the contrast)

1. **Capacity vs information** — handled by the capacity-matched OFF arm (§2): both arms have the
   conditioning params; only the fed geometry differs.
2. **Geometry-randomisation as a separate lever** — the aug is ON for BOTH arms, so its effect is not
   attributed to conditioning. (A 3rd arm — conditioning ON, aug OFF — is optional to isolate the aug; not
   in the primary gate.)
3. **steer R² cross-corpus mismatch** (comma STEER_RATIO 15.3 vs PhysicalAI kinematic) — steer stays
   SECONDARY, not in the gate; speed + yaw (SI-comparable) decide (unchanged from the IDM proof).
4. **R² is distribution-relative** — report MAE alongside R² so a narrow-distribution penalty cannot
   masquerade as a domain failure.
5. **rig-B not scene-identical to rig-A** — measures "rig-A/comma-trained transfer to rig-B", not "only
   geometry changed"; noted, not overclaimed.

## 6. How it is run (existing scaffolding)

Reuses the multi-rig splits (`run_idm_ft.py` selection) with `CameraConditionedEncoder` in place of the
plain light-FT encoder; ON/OFF differ only by the camera vector fed. ~hours on pod3 (~20 min/arm on the
70-ep infra), `gpu_lock.sh acquire dyn-encoder`. Results → `results_camcond.json`; verdict reported
plainly regardless of direction. **Nothing above this line may be edited after the first result.**

---

## RESULT — landed 2026-07-23 (MEASURED; full analysis in `RESULTS_camcond.md`)

Ran on pod3, warm-started from the md5-verified flagship-v1 (`b5f07d9e…`, step 29999). Raw JSON:
`results_camcond_rig.json`, `results_camcond_multirig.json`.

| experiment | cross-rig speed R² OFF | ON | Δ(ON−OFF) | ON PASS? |
|---|---|---|---|---|
| rig (rig-A→rig-B) | −2.344 | −2.253 | **+0.091** | ❌ |
| multirig ({A+comma}→B) | −2.176 | −2.057 | **+0.119** | ❌ |

**Verdict: FAIL (both).** Per §4: ON ≈ OFF (marginal, consistent +0.09/+0.12; cross yaw also up
−0.20→−0.07 in multirig — the mechanism is **not refuted**, it nudges the right way every time), but
cross-rig speed R² stays ~−2.1 (gate 0.9) and ADE ratio ~3.7 (gate 1.5); even ON does not beat the plain
re-gate light-FT (−1.61/−1.65). ⇒ the CHEAP warm-start suffix-conditioning shortcut (**Branch A**) is
**refuted**; **Branch B** (from-scratch, all-block conditioning, multi-rig — the full GAIA-2 regime) is
the pre-registered primary path, with Plücker/PRoPE geometry-as-input escalation on the table. GPU lock
released; the run completed normally (momentarily mis-flagged as crashed — C2, see RESULTS).

---

> 🟥 **BRANCH B FOLLOW-ON — landed 2026-07-24 (`MEASURED`).** Branch B (from-scratch, all-block conditioning,
> multi-rig, 40k) was judged against this same frozen gate and **FAILED**: best held-out-rig cross-rig speed
> R² **−0.667** (gate +0.9), yaw R² negative everywhere, and it is a **weaker** substrate than the plain
> frozen flagship-v1 encoder (+0.657) paired on 3/4 arms (the one favouring arm is episode-leaked, CI spans
> 0). Full: `../../2026-07-24-branchb-transfer-eval/RESULTS_branchB.md`; `Project Steering/MODEL_REGISTRY.md
> §10`. **Both the cheap (Branch A) and expensive (Branch B) camera-conditioning routes are now refuted;**
> the pivot is a flagship-warm-started, longer-trained, augmentation-matched encoder — a **Sayed-gated new
> arm, pre-registered before spend, not auto-launch.**
