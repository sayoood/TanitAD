# P0 — data-efficient closed-loop robustness for REF-C's anchored-diffusion planner

**Stream D2** (`a1f26c92`, pod-free research+design). **Date:** 2026-07-23 (Berlin).
**Scope:** the ONE quadrant the program's prior closed-loop sweeps left open — a covariate-shift fix that
is **renderer-free, non-self-referential, AND data-efficient** (does not need 100+ real junction scenes).

**Evidence discipline (CLAUDE.md).** Every external claim is `PUBLISHED (cited; ✓=abstract/quote
re-verified this session, ○=from record)`; every inference to our stack is `HYPOTHESIS`; every internal
fact is `MEASURED (artifact)` or `INHERITED (doc, not re-measured)`. Read `RETRACTION_LOG.md` first — this
doc touches **C3** (mechanism-as-fact: I flag every "because"), **C4** (INHERITED never decides a GPU-day),
**C6** (the confound in every closed-loop number is reconstruction-OOD, named throughout).

This does **not** re-survey the RL / IL / diffusion landscape — three angle docs already did
(`…/closed-loop-wm-research/angle{1,2,3}*.md`, ~40 sources). It **builds on them** and adds the
recovery-augmentation lineage they missed.

---

## 0. TL;DR

1. **The measured failure is covariate shift in the PLAN, not the plan's on-road representability.** Gate-0b
   proved every *planned* trajectory is on-road yet the ego still departs; G1prep measured the plan itself
   drifting `plan_xte 0.57 → 12.98 m` on-policy. `MEASURED (gate0b_gradient_results.json; LOOP_STATE
   G1prep)`. So the lever must reshape what the planner outputs **from off-path states**, and REF-C senses
   "off-path" **only through pixels** (`MEASURED`, source-read `refc.py`: inputs are frames + scalar `v0`;
   no ego-history). ⇒ the covariate shift is a **visual** one, and the fix must arrive through the frame.
2. **Every closed-loop-training family the program has costed is blocked on ONE of three axes:**
   renderer-OOD (CAT-K/RoaD need a sim → NuRec's 3.2×), self-reference (DAgger on our WM → HURT, MEASURED),
   or **data scarcity** (Gate-1 real-junction FT → memorised at n≈15, MEASURED held-out Δ≈0). The open
   quadrant is a fix that needs **none of a renderer, a WM rollout, or scarce scenes**.
3. **That fix exists and predates the modern literature: synthetic viewpoint-perturbation recovery
   augmentation** (PilotNet 2016 ✓; ChauffeurNet 2018 ✓). Warp a *real* frame by an analytic homography to
   simulate an off-path pose, and supervise the return-to-path trajectory. It is **data-efficient** (every
   window → a recovery example, not 15 scenes), **renderer-free** (a homography, not NuRec), and
   **non-self-referential** (a real frame, not a WM hallucination). Our unique addition: **bound the warp to
   the P1-MEASURED low-OOD envelope**, so the augmentation is provably in-distribution for the frozen
   encoder, and use the **same warp operator as the closed-loop instrument** → train==test by construction.
4. **Do NOT add ego-history to "fix" it** (the tempting architecture move): more observational context is
   the canonical way IL gets *worse* closed-loop (causal confusion / copycat, de Haan 2019 ✓). REF-C's
   pixels-only design is protective; keep it.

---

## 1. What the failure actually is (MEASURED, ours — the thing the lever must move)

Three internal results converge and are re-verified against artifacts, not prose:

- **Free inference-time plan-shaping is exhausted.** Gate-0 (selection) and Gate-0b (per-denoise
  gradient nudge, cost-guided diffusion proper) both leave junction off-road **identical**
  (`ΔOFFROAD +0.000`, every one of 15 scenes unchanged), *even though the nudge provably escapes the anchor
  vocabulary and makes every plan on-road* (`base_off≈0` at junctions). `MEASURED
  (gate0b_gradient_results.json)`. Rung-3 WM-MPPI/CEM ties the single-step re-plan (Δ+0.005..+0.011, n.s.;
  off-road proxy separated *worse*). `MEASURED (wm_mpc_result.json)`.
- **The residual is the EXECUTED/on-policy plan, not representability.** "The planned trajectory is on-road
  at junctions — and the closed-loop junction off-road rate is STILL 0.73." `MEASURED (gate0b NOTE §MECHANISM)`.
  G1prep localised it precisely: the ego tracks its plan tightly (0.49 m) while **the plan degrades
  on-policy, `plan_xte 0.57 → 12.98 m` — textbook covariate shift in the planner.** `MEASURED (LOOP_STATE
  G1prep)`.
- **REF-C reads pose from pixels only.** `MEASURED (source-read refc.py::RefCModel.forward)`: inputs are the
  8-frame window + a scalar `v0` (with 0.5 ego-dropout); there is no past-trajectory / past-action channel.
  ⇒ when the ego drifts, the ONLY thing that tells REF-C it has drifted is the (warped) image. `HYPOTHESIS`
  (strong, source-grounded): the fix has to be learned through the visual channel, which is exactly what a
  viewpoint-perturbation augmentation supplies.

**So the target is:** teach the anchored-diffusion decoder to emit a *return-to-path* trajectory when the
frame shows an off-path viewpoint — without a renderer, without a WM rollout, without scarce junction data.

---

## 2. Why each costed closed-loop-training family is blocked (the open quadrant, drawn)

| family | key papers | what blocks it FOR US | evidence |
|---|---|---|---|
| **Inference-time floor** (selection / guided diffusion / WM-MPC) | Diffusion-Planner `2501.15564` ○; TD-MPC2 `2310.16828` ○ | **RULED OUT** for junction off-road — shapes the *plan*, not the *executed* covariate-shifted plan | `MEASURED` Gate-0/0b/rung-3 |
| **Closed-loop-aware IL via sim rollouts** | CAT-K `2412.05334` ○; RoaD `2512.01993` ○ | need a sim to roll out → **NuRec 3.2× OOD**; RoaD's own sim2sim degrades 0.75→0.58 | `INHERITED` (angle3); `MEASURED` OOD |
| **DAgger on our WM** | Ross 2011 ○ | on-policy states are the WM's **own off-manifold latents** → over-react → **HURT** (closed ADE +0.266) | `MEASURED (dagger_result.json)` |
| **Gate-1 real-junction recovery FT** | (our proto) | works mechanically (offroad 11→7) but **memorises at n≈15** (held-out recovery-L1 5.06→5.06, Δ≈0) | `MEASURED (gate1_clean_loo.json)` |
| **Analytic-grad closed-loop-imitation through the WM** | Nachkov `2409.07965` ○; SHAC `2204.07137` ○; DreamerV3 `2301.04104` ○ | high-ceiling but **WM-exploitation** off-highway + a v4-scale *training* project (heavy) | angle1 verdict `INHERITED` |
| ⭐ **Viewpoint-perturbation recovery augmentation** | **PilotNet `1604.07316` ✓; ChauffeurNet `1812.03079` ✓** | **NONE of the three axes** — analytic homography of a real frame, every window | this doc |

The last row is the open quadrant: **renderer-free ∧ non-self-referential ∧ data-efficient.** It is the
only family that attacks the executed-plan covariate shift at the *training* level without paying a
renderer, a WM rollout, or 100+ scenes.

---

## 3. The recovery-augmentation lineage (primary sources, verified this session)

### 3.1 PilotNet — viewpoint-transform recovery for LANE-KEEPING (the direct precedent) `PUBLISHED ✓`
Bojarski et al., NVIDIA, *End to End Learning for Self-Driving Cars*, 2016, arXiv:1604.07316. Re-verified
this session: *"augmented data by adding artificial **shifts and rotations** to teach the network how to
**recover from a poor position or orientation** … additional shifts and all rotations were simulated
through **viewpoint transformation of the image** from the nearest camera … the network must learn how to
recover from any mistakes, or the car will slowly **drift off the road**."* Perturbation magnitude ~ a
zero-mean normal with std ≈ 2× the human-driver std.
- **This is our lever's exact skeleton** for lane-keeping: synthesize off-center/rotated viewpoints by an
  image homography, label the corrective control. `HYPOTHESIS` our upgrades over 2016: (i) a modern
  **anchored-diffusion trajectory** decoder instead of a steering-scalar CNN; (ii) the perturbation is
  **bounded to a MEASURED OOD envelope** (P1), not a heuristic normal — so we can *prove* the warped frame
  stays in-distribution for the frozen encoder; (iii) the warp is the **same operator the closed-loop
  instrument applies on-policy** → train==test.

### 3.2 ChauffeurNet — "synthesize the worst" trajectory perturbation `PUBLISHED ✓`
Bansal, Krizhevsky, Ogale (Waymo), 2018, arXiv:1812.03079 (RSS 2019). Re-verified: *"synthesized data in
the form of **perturbations to the expert's driving**, which creates interesting situations such as
collisions and/or going off the road … augment the imitation loss with additional losses that penalize
undesirable events."* The perturbations are the recovery signal, **no simulator required**.
- `HYPOTHESIS` for us: ChauffeurNet perturbs the *agent pose in a mid-level BEV* (it has a map render);
  REF-C is camera-input, so our pose perturbation must be realised as the **image homography** (3.1), and
  our "penalize going off-road" analog is the corridor-departure metric the instrument computes. The
  **recovery target** — the expert future re-expressed in the perturbed ego frame — is validated in our
  `perturb.py` (`identity_target_maxerr 0.0`) and is the same construction the DAgger stream's
  `dagger_probe.py` validated (max|err| 0.0).

### 3.3 Causal confusion — why NOT to "fix it" by adding history `PUBLISHED ✓`
de Haan, Jayaraman, Levine, NeurIPS 2019, arXiv:1905.11979. Re-verified: *"access to more information can
yield worse performance"* — a BC policy latches onto nuisance correlates (the past-ego-motion "copycat"
shortcut) and fails under distribution shift.
- `HYPOTHESIS`: this is a direct warning against the obvious architecture move (condition REF-C on
  ego-history / past waypoints to "know it drifted"). That would hand the planner the copycat shortcut and
  is the classic way lane-keeping IL gets *worse* closed-loop. **REF-C's pixels-only design is protective;
  the recovery augmentation supplies the drift signal through the image, not through a history channel.**

### 3.4 Diffusion-policy closed-loop robustness levers (design menu) `PUBLISHED`
- **Diffusion Policy** (Chi et al., RSS 2023, arXiv:2303.04137 ✓): receding-horizon control + **action-
  chunking** (predict a horizon, execute a subset, re-plan) + a time-series (history-conditioned)
  transformer. The action-horizon-<-prediction-horizon design buys temporal consistency and closed-loop
  robustness. `HYPOTHESIS`: REF-C already outputs a multi-step trajectory and the instrument re-plans every
  tick (action horizon = 1); **lengthening the executed chunk before re-planning is a zero-training
  deployment knob** worth an ablation — but it is a *smoothing* lever, orthogonal to (and weaker than)
  fixing the covariate shift at the source, and the rung-3 verdict already found re-plan≈chunked-MPC a tie.
- **DiffusionDrive** (Liao et al., CVPR 2025, arXiv:2411.15139 ○): the architecture REF-C is built in the
  spirit of (truncated anchored diffusion). Nothing to add; it is our substrate.
- **Consistency Policy** (Prasad et al., RSS 2024, arXiv:2405.07503 ○): distil to 1–3 denoise steps for
  fast closed-loop. `HYPOTHESIS`: a deploy-latency lever, not a robustness lever; out of scope for D2.

### 3.5 The recovery TARGET should be feasible — the CAT-K discipline, for free `PUBLISHED ○ → MEASURED`
CAT-K (Zhang/Karkus, CVPR 2025, arXiv:2412.05334) keeps closed-loop supervision **anchored to the expert
manifold** so recovery targets stay feasible (don't point sharply backward). Gate-1 MEASURED that on real
NuRec recovery labels, **49 % (328/675)** were catastrophic/off-manifold and drove the high-deviation
side-effect (`gate1_clean_loo.json`). **Our envelope bound IS the CAT-K filter, applied at generation
rather than post-hoc:** because we only ever perturb within the P1 low-OOD envelope (≤1.75 m / ≤5°), the
recovery target is a *bounded* return-to-path by construction — it can never be a catastrophic backward
cut. This is why the augmentation should not reproduce the Gate-1 high-deviation failure — and the
`λ_dev` trust-region (Gate-1's second fix, MEASURED −80 % held-out plan-shift) is retained as a belt.

---

## 4. Data-efficiency, classified (the mission's explicit ask)

| lever | needs 100+ junction scenes? | needs a renderer? | self-referential? | verdict for D2 |
|---|---|---|---|---|
| Real-junction recovery FT (Gate-1) | **YES** (memorises at 15) | yes (NuRec) | no | blocked |
| DAgger on WM | no | no | **YES** (HURT) | blocked |
| RoaD/CAT-K sim rollouts | one-off ok | **YES** (3.2× OOD) | no | blocked (OOD) |
| Analytic-grad thru WM | no | no | **partly** (WM-exploit) | heavy (v4-scale) |
| ⭐ **In-envelope viewpoint recovery aug** | **NO** — every window is a recovery example | **NO** — homography | **NO** — real frame | **the lever** |
| Action-chunk / receding-horizon knob | no | no | no | cheap ablation, weak (rung-3 tie) |
| Ensemble/uncertainty selection | no | no | no | a floor (ruled-out family) |

`HYPOTHESIS` The augmentation multiplies data by geometry: the 40-ep clean val alone has ~881 windows;
each yields a family of recovery examples under the envelope of perturbations, so the effective recovery-
supervision set is **~1–2 orders larger than Gate-1's 675 labels from 15 scenes**, drawn from *all*
geometry (not just the 13–22 junction episodes). That is the precise mechanism by which it can generalize
where the scene-collected FT memorised — and it is the pre-registered question in `PRE_REGISTRATION.md`.

---

## 5. The honest boundary (do not overclaim)

- **Lane-keeping, not off-road/collision.** The instrument (and therefore this whole stream) measures
  on-policy corridor departure at low OOD; it is **structurally unable** to measure map off-road or
  reactive-agent collision (`INHERITED`, gate1-clean §2, MEASURED code-read). A real safety rate still needs
  a lower-OOD renderer. The lever is validated here as a **closed-loop lane-keeping / covariate-shift**
  mechanism, exactly what corridor-departure scores.
- **Ground-plane homography is exact in yaw, optimistic in lateral** (`INHERITED`, P1 §2). We therefore
  weight the exact axis (yaw) and cap lateral at the corridor half-width; the P1 envelope says both stay
  ≤1.16× baseline observation-OOD, ~17× below NuRec.
- **A perturbation is not a rollout.** The augmentation covers *single-step* off-path recovery, not the full
  compounding trajectory a faithful sim would visit. `HYPOTHESIS`: this is sufficient for the corridor /
  lane-keeping failure (which is dominated by per-tick heading/lateral correction, MEASURED longitudinal
  drift in LowOOD) but is NOT a substitute for AlpaSim on the reactive-agent tail.

---

## 6. Sources (✓ = re-verified this session)

- Bojarski et al. *End to End Learning for Self-Driving Cars* (PilotNet), 2016, arXiv:1604.07316 ✓ (shift+
  rotation viewpoint-transform recovery augmentation for lane-keeping).
- Bansal, Krizhevsky, Ogale. *ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the
  Worst*, 2018, arXiv:1812.03079 ✓ (RSS 2019) — trajectory perturbation, no simulator.
- de Haan, Jayaraman, Levine. *Causal Confusion in Imitation Learning*, NeurIPS 2019, arXiv:1905.11979 ✓.
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023, arXiv:2303.04137 ✓.
- Liao et al. *DiffusionDrive*, CVPR 2025, arXiv:2411.15139 ○. Prasad et al. *Consistency Policy*, RSS 2024,
  arXiv:2405.07503 ○.
- Zhang, Karkus et al. *CAT-K: Closed-Loop SFT of Tokenized Traffic Models*, CVPR 2025, arXiv:2412.05334 ○.
  *RoaD*, arXiv:2512.01993 ○. (both surveyed in `…/closed-loop-wm-research/angle3_alternatives_verdict.md`.)
- Ross, Gordon, Bagnell. *DAgger*, AISTATS 2011 ○. Nachkov et al. arXiv:2409.07965 ○; SHAC arXiv:2204.07137
  ○; DreamerV3 arXiv:2301.04104 ○ (all in `angle1_model_based_rl.md`).

**Internal (MEASURED/INHERITED, artifact-cited inline):** `gate0b_gradient_results.json`,
`wm_mpc_result.json`, `dagger_result.json`, `gate1_clean_loo.json`, `P1_DECISION_GRADE_FINDINGS.md`,
`lowood_lanekeep.py`, `refc.py`, `refc_train.py`. All web claims sourced via WebFetch/WebSearch 2026-07-23.
