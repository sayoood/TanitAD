# How E2E / AI driving models solve LONGITUDINAL planning & control — a cited survey for TanitAD v3

*2026-07-19. Research agent, for Sayed. Scope: how SOTA end-to-end and model-based AD systems set speed, react to lead vehicles, comply with speed limits, and toggle longitudinal behaviour modes. Primary sources: arXiv (2020–2026), openpilot/comma docs, nuPlan/NAVSIM. Each claim is tagged **[established]** (multiple independent sources or a load-bearing primary paper) or **[open/emerging]** (single source, contested, or research-stage). This directly informs the v3 direction from the DINO-WM comparison (`2026-07-19-refa-dinowm-literature-comparison.md`): frozen encoder → feature-rollout world model → CEM/diffusion/MPC planner, **no supervised head**.*

---

## TL;DR / Verdict

**Sayed's hypothesis is substantially SUPPORTED, but the mechanism needs reframing.**

1. **The "regress toward mean speed" failure is real and understood.** It is *expectation/mode averaging*: L2-regressing a single trajectory over a multimodal action distribution collapses to the conditional mean → over-predict when the true speed is low, under-predict when it is high. This is exactly why the field abandoned single-trajectory regression for **anchor/vocabulary scoring and diffusion**. **[established]** (MultiPath; VADv2; DiffusionDrive.)

2. **Explicit desired/target speed IS used by the best systems — but as a control set-point / planning cost, not as a supervised regression target.** IDM's `v0`, openpilot's set speed, and PDM's enumerated target speeds (20–100 % of the map limit) are all *objectives the planner optimizes against*, not labels a head predicts. **[established]** (IDM; openpilot; PDM/nuPlan.)

3. **CAVEAT — the ego-status shortcut.** Naively feeding ego velocity (or a target) as an *input to a regression head* lets the network cheat by extrapolation on open-loop benchmarks (and causes the classic "inertia problem" — the car won't start from a stop). Target speed must enter as a cost the planner evaluates **on predicted consequences**, not as a leaked input. **[established]** ("Is Ego Status All You Need?"; Codevilla behaviour-cloning limitations.)

4. **You probably do NOT need discrete longitudinal modes.** IDM produces the free-cruise↔car-following toggle *emergently* from just two competing cost terms (a target-speed term and a gap/headway term, combined multiplicatively) with no mode switch. openpilot does the same by taking the binding constraint (min) over a cruise pseudo-obstacle and the lead. **[established]**

5. **For v3 (DINO-WM-style WM + CEM/diffusion/MPC): encode longitudinal as a planning COST over imagined rollouts** = a target-speed term (IDM free-flow) + a gap/TTC safety barrier (IDM interaction / openpilot obstacle) + an accel/jerk comfort term. The planner samples action (accel/speed) sequences, rolls them out in the frozen-feature world model, decodes speed/gap/TTC from the predicted latent, scores, and picks the best. This kills mean-regression (no single-trajectory regression) *and* dodges the ego-status shortcut (target speed is a cost on predicted outcomes). Use a **sampling/gradient-free** optimizer (CEM or Diffusion-ES) because the safety terms are non-differentiable. **[synthesis]**

---

## A. Longitudinal representation in E2E AD planners

**The classic E2E line represents longitudinal purely IMPLICITLY.** UniAD, VAD/VADv2, GenAD and kin regress or score future ego waypoints `(x,y)` at fixed *time* intervals; speed = spacing between consecutive waypoints. There is no explicit "target speed" scalar and no separate longitudinal objective — desired speed is whatever the imitation loss bakes in. UniAD's planner is "a simple attention-based planner [that] predict[s] future waypoints of the ego-vehicle" with no speed set-point. **[established]** (UniAD, arXiv 2212.10156; GenAD, arXiv 2402.11502.)

**Why single-trajectory regression fails on speed (the core of Sayed's observation).** Human driving is multimodal (brake vs coast vs accelerate all plausible). A model that regresses one trajectory under L2 minimizes error by predicting the *mean* of the modes → it systematically over-predicts slow situations and under-predicts fast ones. This "mode averaging / mode collapse" is documented as the central failure the field moved to fix. **[established]** (MultiPath note; VADv2; DiffusionDrive §mode-collapse; "Mode Collapse Happens," arXiv 2506.23164.)

**The frontier fix = discretize the action space into a trajectory VOCABULARY and score/classify, not regress.** Each vocabulary anchor is a *complete spatio-temporal trajectory*, so it already encodes a full speed profile; the network outputs a distribution over anchors instead of averaging them.
- **VADv2** (arXiv 2402.13243): 4096-anchor "planning vocabulary," tokenized, outputs a **probabilistic** action distribution supervised by demonstrations — explicitly to "cope with the uncertainty problem" that deterministic regression cannot. **[established]**
- **Hydra-MDP / Hydra-MDP++** (arXiv 2503.12820): vocabulary = **k-means over 700k nuPlan trajectories**, each 40 timesteps (x,y,heading) at 10 Hz over a 4 s horizon. Longitudinal quality comes from **simulation-metric distillation** (below), not a speed input. 91.0 DS on NAVSIM by scaling the encoder. **[established]**
- **DiffusionDrive** (arXiv 2411.15139): anchored **truncated diffusion** — denoises from a fixed multi-mode anchor set to the action distribution, explicitly to avoid mode collapse; 88.1 PDMS, 45 FPS. **[established]**
- **TCP** (arXiv 2206.08129): two branches — a trajectory branch *and* a direct control (throttle/brake/steer) branch — fused situation-adaptively; the control branch predicts longitudinal actuation directly. **[established]**

**nuPlan / NAVSIM planners DO use explicit target speed — via the map.** The nuPlan-winning **PDM-Closed** enumerates IDM policies at **{20 %, 40 %, 60 %, 80 %, 100 %} of the posted speed limit** × 3 lateral centerline offsets = 15 proposals, simulates each, and picks the best-scoring. Here "desired speed" is *derived from the HD-map speed limit* and *enumerated as candidate targets*, then selected by a cost — the cleanest existence proof that explicit target speed helps. **[established]** (Dauner et al., "Parting with Misconceptions," arXiv 2306.07962.) **PLUTO** adds a "longitudinal-lateral aware model architecture [for] flexible and diverse driving behaviors" and is the first learned planner to beat PDM in closed-loop. **[established]** (arXiv 2404.14327.)

**Conditional Imitation Learning (the ancestor) already fed current speed + a high-level command.** CIL (Codevilla 2018, arXiv 1710.02410) inputs image + **current speed** + a branch-selecting command (left/right/straight/follow). But this is *navigation* conditioning, not a *target-speed* set-point — and the speed input caused the well-known **"inertia problem"** (model correlates low speed with staying stopped → refuses to start). **[established]** (Codevilla 2019 behaviour-cloning limitations; command augmentation, arXiv 1909.09721.)

**VLA / world-model frontier:** NVIDIA **Alpamayo-R1** (arXiv 2511.00088) autoregressively emits chain-of-thought + discrete **trajectory tokens**, then a flow-matching action decoder turns them into kinematically feasible waypoints — longitudinal lives in the tokens, conditioned by reasoning; no explicit speed scalar. Wayve **GAIA-1** (arXiv 2309.17080) is an action-conditioned generative world model (actions include speed/steering); **LINGO-2** conditions driving behaviour (incl. speed) via **language**. **[established for existence; open on longitudinal specifics.]**

**Answer to A:** In the mainstream E2E line, longitudinal is **implicit** in trajectory regression — and that is *precisely* the source of mean-speed regression. The systems that handle speed well either (i) score a **vocabulary of speed-profile anchors**, (ii) distill **simulation metrics** (Hydra), or (iii) enumerate **explicit target speeds from the map** (PDM). Explicit "set-speed as a scalar NN input" is rare in research E2E but universal in production ACC/openpilot — as a *cost/set-point*, not a regressed label.

---

## B. Lead-vehicle following / ACC

### The classical grounding — IDM **[established]** (Treiber et al. 2000; 25-yr survey arXiv 2506.05909)

Acceleration:
```
a = a_max · [ 1 − (v/v0)^δ − (s*/s)^2 ]
s* = s0 + max( v·T + v·Δv / (2·√(a_max·b)), 0 )
```
- **Free-flow term** `1 − (v/v0)^δ`: dominates when no leader constrains you → drives `v` toward the **desired speed `v0`**.
- **Interaction/following term** `(s*/s)^2`: dominates when the actual gap `s` drops below the desired dynamic gap `s*` → decelerates to hold **time headway `T`**.
- The **toggle between free-cruise and car-following is emergent and continuous** — the multiplicative structure just takes whichever term binds; no mode switch. `Δv` (closing speed) adds anticipatory braking.
- Typical params: `v0 ≈ 30 m/s`, `T ≈ 1.6 s`, `a_max ≈ 0.73 m/s²`, `b ≈ 1.67 m/s²`, `s0 ≈ 2 m`, `δ = 4`.
- IDM is also the **default reactive background-traffic model in nuPlan** and the longitudinal core of PDM proposals. **[established]**

**This is the single most important prior for TanitAD v3:** IDM shows you get correct free-flow ↔ following behaviour from *exactly two cost terms* (target speed + gap/headway), no explicit mode variable.

### openpilot / comma longitudinal MPC **[established]** (comma blog 2021, 0.9.x/0.10; long_mpc.py)

- The classical (Chill-mode) longitudinal MPC "takes the fused (neural network + radar) estimates of lead cars **and the desired set speed**, feeds it into an MPC solver, and computes a good acceleration profile." Lead estimates from Supercombo = **relative speed, absolute speed, distance**, fused with radar.
- **State** `(x_ego, v_ego, a_ego, j_ego)`; quadratic running cost tracks a reference profile plus **obstacle/danger-zone** costs; weights include `X_EGO_OBSTACLE_COST = 3`, plus accel-change and jerk penalties (velocity-weighted: `a_ego·(0.1·v_ego+1)`, `j_ego·(0.1·v_ego+1)`) for comfort. Cost terms = **collision, follow-distance, acceleration, jerk**.
- **ACC = min over constraints.** The cruise set speed and each lead are encoded as separate "obstacle" sources; the MPC naturally obeys the *binding* one → effective behaviour ≈ `min(set_speed, lead-constrained speed)`. **[established]**
- **Desired follow distance is headway-based**, personality-tunable: **Standard `T_FOLLOW ≈ 1.45 s`, Relaxed ≈ 1.75 s, Aggressive closer (~1.25 s, exact value not confirmed in sources — flag)**, plus a standstill `STOP_DISTANCE`. **[established for standard/relaxed; open for aggressive value]**
- **Learned longitudinal exists but is fenced.** openpilot's **Experimental mode** replaces the ACC MPC with the driving model's own end-to-end longitudinal (a "model MPC" that tracks the *neural* reference trajectory and **does not consume explicit lead info**); **Chill mode keeps the classical lead policy** as the reliable fallback. comma states it "plans to replace" Chill's longitudinal with the E2E policy — i.e. even the leading production shop does not yet fully trust learned longitudinal. **[established]** (comma 0.10 release; Medium model_mpc writeup.)

### How LEARNED E2E handles car-following + stop-and-go, and its failure modes

- **Gap-keeping is hard** for regression models because the correct behaviour is multimodal and safety-critical; mean-averaging yields sloppy gaps. Vocabulary/diffusion planners mitigate this but do not encode a *guaranteed* headway the way IDM/MPC do. **[established]**
- **Phantom braking** — false-positive perception or over-aggressive braking calibration triggers unwarranted decel; a top real-world complaint (NHTSA >400k complaints across programs), worst in compressed-headway conditions (stop-and-go, on-ramps). **[established]** (NHTSA/industry reporting.)
- **Inertia problem** — with ego speed as input, learned policies over-correlate low speed → stopped, hesitating to launch. **[established]** (Codevilla 2019.)
- **Stop-and-go / creep** — smooth low-speed following and standstill→launch remain a known weak spot for pure learned longitudinal; this is why production stacks retain a classical lead policy. **[established]**

---

## C. Speed-limit / sign → target speed **[established, three routes]**

1. **HD-map speed limit** (dominant in nuPlan-style planners): `v0` = posted limit read from the map graph; PDM enumerates fractions of it. Reliable where maps exist. **[established]** (PDM/nuPlan, arXiv 2306.07962.)
2. **On-board Traffic-Sign Recognition / Intelligent Speed Assist** (production ADAS): CNN/SSD sign detectors (e.g. Mobileye ISA) feed a target-speed set-point; handles explicit *and* implicit limits (school/work zones). **[established]** (Mobileye ISA; SSD-based TSR, IEEE.)
3. **VLM sign-reading** (emerging): GPT-4V reads "60" and explains it; modular pipelines (TLS-Assist, arXiv 2511.14391) detect signs → structured text → injected into an LLM/VLM planner; work-zone VLMs handle signage missing from maps (arXiv 2606.08860). Powerful for long-tail/temporary signage but latency- and reliability-limited. **[open/emerging]**

**For v3:** treat the speed limit as *one input to `v_target`* — `v_target = min(route/user desired speed, map limit, sign-read limit)` — feeding the planner's target-speed cost term. Sign-reading can be a later VLM add-on; start with map/route.

---

## D. Longitudinal mode-switching — explicit vs implicit

- **IDM & openpilot: implicit.** Free-cruise↔follow emerges from competing cost terms / binding-constraint `min`; no discrete state. **[established]**
- **Classical AV stacks (DARPA-era Junior/Boss and successors): explicit** finite-state longitudinal behaviours (FOLLOW / STOP / YIELD / CREEP). Robust and interpretable but hand-engineered. **[established, historical]**
- **Modern E2E: mostly implicit**, occasionally an explicit "meta-action" label (accelerate / keep / decelerate / stop) in LLM/VLM-hybrid planners. **[open/emerging]**
- **Does explicit structure demonstrably help?** The strongest evidence is *indirect*: PDM's explicit enumerated target speeds beat learned planners in closed-loop, and Hydra's explicit metric teachers (below) improve compliance. But the **ego-status caveat** shows explicit *inputs* can backfire (shortcut/inertia). Net: **explicit target-speed as a planning objective helps; explicit discrete modes are not required if the cost carries a target-speed term and a gap term** (IDM proves this). **[established, with nuance]**

---

## E. How PLANNERS (CEM / MPC / diffusion) encode the speed target + safety in the COST/REWARD

This is the load-bearing section for v3.

**openpilot MPC** — quadratic tracking cost on `(x,v,a,j)` toward a reference + obstacle/danger-zone barriers. Cruise = a pseudo-obstacle placed so the ego settles at the set speed; lead = an obstacle at `desired_dist = STOP_DISTANCE + T_FOLLOW·v`. The planner minimizes cost → ACC behaviour. **Target speed and safety are literally two cost terms.** **[established]**

**IDM as a 1-step controller/cost** — the free-flow term *is* the target-speed cost; the interaction term *is* the gap-safety cost. Analytic, differentiable, cheap. **[established]**

**Model-based RL / world-model planning:**
- **Think2Drive** (arXiv 2402.16720): DreamerV3 latent world model learns transition + **reward** + termination; the policy maximizes *predicted* reward (progress/speed + safety). First to solve CARLA-v2. The **reward model carries the speed/progress objective**. **[established]**
- **MILE** (model-based imitation, Dreamer-style): imagines futures, plans without HD maps; lifts CARLA driving score 46→61. **[established]**
- **DINO-WM** (arXiv 2411.04983) — **the closest analog to v3**: frozen DINOv2 patch features, a ViT transition model predicts future *patch embeddings*, and planning is **MPC + Cross-Entropy Method** at test time with **no policy head**. Its cost is *distance in feature space to a goal-image embedding*. **Key gap for driving:** a goal-image cost does NOT encode a target speed or a gap constraint — for longitudinal you must *replace/augment* the goal-reaching cost with an explicit speed + safety cost decoded from the predicted latent. **[established + synthesis]**

**Diffusion planners:**
- **Diffuser** (arXiv 2205.09991): generates whole trajectories; steers toward high reward via **reward-gradient / classifier guidance** during denoising while staying on the data manifold. **[established]**
- **Decision Diffuser**: return-conditioned generation via **classifier-free guidance** (condition on desired return — a scalar objective, analogous to a target). **[established]**
- **Diffusion-ES** (arXiv 2402.06559) — **the best fit for a non-differentiable driving cost**: sample trajectories from a diffusion prior, score with a **black-box reward** (can be non-differentiable *or language-shaped*), keep elites, mutate via truncated noise-then-denoise (evolutionary search). **SOTA on nuPlan closed-loop**, beats reward-gradient guidance and reactive policies. Lets you drop a progress-to-target-speed + collision/TTC + comfort reward straight in with zero retraining. **[established]**
- **DiffusionDrive**: learned, anchored truncated diffusion (fast, but the objective is imitation, not an explicit cost). **[established]**

**NAVSIM PDMS / nuPlan closed-loop score = the de facto longitudinal reward.** A weighted combination of **No-Collision, Drivable-Area, Time-to-Collision, Comfort, and Ego-Progress**. **Ego-Progress** (progress along route relative to a PDM reference) is effectively a *"achieve the target speed the reference planner would"* term; TTC + collision are the safety terms. Hydra-MDP++ distills exactly these as **per-metric learned cost values** and selects the trajectory with lowest weighted cost `−Σ_m k_m·log S_m`. **[established]** — this is a working template for a driving cost: **progress/target-speed term + collision + TTC + comfort**, no explicit target-speed *input* needed.

---

## F. Verdict + concrete v3 recommendations

### Is "explicit target velocity as input + longitudinal mode structure" the fix?

**Partly — reframed:**
- ✅ **The diagnosis is right.** Mean-speed regression = multimodal averaging under single-trajectory L2. **[established]**
- ✅ **A target speed helps** — but as a **cost/set-point the planner optimizes**, exactly as IDM's `v0`, openpilot's set speed, and PDM's enumerated targets. **[established]**
- ⚠️ **NOT as a raw NN input to a regression head** — that invites the ego-status shortcut and the inertia problem. **[established]**
- ⚠️ **Explicit discrete modes are optional.** IDM gets the cruise↔follow toggle from two cost terms with no mode variable; add modes only for interpretability, not necessity. **[established]**

### SOTA best practice for longitudinal representation
Stop regressing one trajectory. Either (a) **score a vocabulary of speed-profile anchors** (VADv2/Hydra), (b) **diffuse** trajectories and select by cost (DiffusionDrive/Diffusion-ES), or (c) **plan with an explicit cost** that contains a **target-speed/progress term + a gap/TTC safety term** (IDM / openpilot MPC / PDM / NAVSIM PDMS). All three beat mean-regression; (c) is the natural fit for a world-model planner.

### How TanitAD v3 (frozen DINO WM → CEM/diffusion/MPC, no head) should encode longitudinal

**Encode the longitudinal objective as a planning COST over imagined rollouts** — the world model predicts action-consequences (per the DINO-WM recipe already chosen for v3), and the planner optimizes candidate **action (accel or Δspeed) sequences** against:

1. **Target-speed term (IDM free-flow):** `w_v · Σ_t (v̂_t − v_target)²`, where `v̂_t` is speed decoded from the predicted latent (a light probe / the ego-motion readout you already have) and `v_target = min(route/user desired, map limit, sign-read limit)`. *This is the direct antidote to mean-regression: the planner is told the target instead of averaging.*
2. **Gap / safety term (IDM interaction + openpilot obstacle):** penalize violating `s* = s0 + T·v̂ + v̂·Δv/(2√(a·b))` to the nearest lead, plus a **collision / TTC danger-zone barrier**. Free-flow↔following emerges as the `min` of terms (1) and (2) — no mode switch.
3. **Comfort term:** `w_a·â² + w_j·ĵ²` (velocity-weighted like openpilot), and **up-weight longitudinal error** — matches your validated v2.1 longitudinal-lever finding.
4. **Optimizer:** because (2) is non-differentiable, use **CEM** (DINO-WM's choice) or **Diffusion-ES** (samples a diffusion prior, scores the black-box cost, evolutionary-mutates) — Diffusion-ES is SOTA on nuPlan closed-loop for exactly this "black-box driving reward" setting and keeps proposals on the human-data manifold.
5. **Keep current speed `v0` as an action/state conditioning channel** (you already validated speed-as-3rd-action-channel: R² 0.61→0.965) — but keep **desired speed as a COST term, never as the regressed output.** This preserves the win while avoiding the shortcut.

**Why this fixes REF-A's ~99%-longitudinal failure:** REF-A regressed displacements from static frozen features (no motion) with a learned head → it averaged speed. Moving the target speed into the *cost* and letting the *world model* supply predicted `v̂/gap/TTC` means (i) no single-trajectory averaging, (ii) target speed is an explicit optimization objective, (iii) the free-flow↔follow toggle is emergent, and (iv) safety is a hard barrier, not a hope. The remaining risk is that **frozen generic DINO features carry too little motion signal to decode `v̂` accurately** — which is the *encoder* problem flagged in the DINO-WM comparison (own-pretrained / motion-aware encoder), orthogonal to the cost design here.

### Established vs open — summary
- **[established]:** mean-regression mechanism; vocabulary/diffusion as the fix; IDM two-term toggle; openpilot ACC = min(cruise, lead) with headway `T_FOLLOW`; PDM target-speeds-from-map; NAVSIM PDMS progress+safety reward; ego-status shortcut; DINO-WM = MPC+CEM goal-cost; Diffusion-ES black-box reward SOTA on nuPlan.
- **[open/emerging]:** VLM sign→target-speed reliability; explicit discrete longitudinal modes helping learned planners (weak/indirect evidence); learned end-to-end longitudinal fully replacing classical ACC (comma still fences it); exact aggressive-personality `T_FOLLOW` value.

---

## Sources

- UniAD — Planning-oriented Autonomous Driving: https://arxiv.org/pdf/2212.10156
- VADv2 — End-to-End Vectorized AD via Probabilistic Planning: https://arxiv.org/abs/2402.13243
- GenAD — Generative End-to-End Autonomous Driving: https://arxiv.org/abs/2402.11502
- Hydra-MDP++ — Expert-Guided Hydra-Distillation: https://arxiv.org/html/2503.12820v1
- DiffusionDrive — Truncated Diffusion for E2E AD: https://arxiv.org/abs/2411.15139
- TCP — Learning Control from Trajectory Planning: https://ar5iv.labs.arxiv.org/html/2206.08129
- AlignDrive — Aligned Lateral-Longitudinal Planning: https://arxiv.org/html/2601.01762
- "Mode Collapse Happens" (joint trajectory prediction): https://arxiv.org/pdf/2506.23164
- Conditional Imitation Learning (Codevilla 2018): https://arxiv.org/abs/1710.02410
- CIL command augmentation / behaviour-cloning limits: https://arxiv.org/abs/1909.09721
- "Is Ego Status All You Need for Open-Loop E2E AD?": https://arxiv.org/abs/2312.03031
- IDM — 25 Years of the Intelligent Driver Model (survey): https://arxiv.org/html/2506.05909v1
- IDM — Wikipedia (equation/params): https://en.wikipedia.org/wiki/Intelligent_driver_model
- openpilot in 2021 (longitudinal MPC + lead + set speed): https://blog.comma.ai/openpilot-in-2021/
- openpilot 0.10 (learned world-model longitudinal, Experimental vs Chill): https://blog.comma.ai/010release/
- openpilot long_mpc.py (cost weights): https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py
- openpilot model_mpc writeup (Hao Zhou / Medium): https://howardchow92.medium.com/the-model-mpc-designed-for-end-to-end-longitudinal-self-driving-at-openpilot-comma-ai-11c0fe363e67
- PDM / "Parting with Misconceptions about Learning-based Motion Planning": https://arxiv.org/pdf/2306.07962
- PLUTO — Pushing the Limit of IL-based Planning: https://arxiv.org/html/2404.14327v1
- Alpamayo-R1 — Reasoning + Action Prediction (NVIDIA): https://arxiv.org/pdf/2511.00088
- Wayve GAIA-1 — Generative World Model: https://arxiv.org/pdf/2309.17080
- Wayve LINGO-2 — Driving with Language: https://wayve.ai/thinking/lingo-2-driving-with-language/
- Think2Drive — MBRL with latent world model (CARLA-v2): https://arxiv.org/abs/2402.16720
- MILE — Model-based Imitation Learning (via world-model survey): https://arxiv.org/pdf/2403.02622
- DINO-WM — World Models on Pre-trained Visual Features (MPC+CEM): https://arxiv.org/abs/2411.04983
- Diffuser — Planning with Diffusion: https://ar5iv.labs.arxiv.org/html/2205.09991
- Diffusion-ES — Gradient-free Planning with Diffusion (nuPlan SOTA): https://arxiv.org/abs/2402.06559
- TLS-Assist — LLM AD with traffic-light/sign recognition: https://arxiv.org/html/2511.14391
- Work-zone VLM speed regulation: https://arxiv.org/html/2606.08860v1
- Mobileye Intelligent Speed Assist (ISA): https://www.mobileye.com/blog/intelligent-speed-assist-isa-computer-vision-adas-solution/
