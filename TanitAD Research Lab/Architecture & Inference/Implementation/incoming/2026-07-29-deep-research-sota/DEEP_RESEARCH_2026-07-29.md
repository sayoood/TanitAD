# Deep research 2026-07-29 — SOTA sweep (Jun 2025 → Jul 2026) and what transfers

**Requested by the PI 2026-07-29.** Two adversarially-verified surveys: (1) autonomous driving,
(2) robotics / foundation models / LLMs / world models / world-action models / JEPA.

| | agents | tokens | tool calls | survived | refuted |
|---|---|---|---|---|---|
| Report 1 — AD | 106 | 6,415,995 | 1,111 | **4 external + 3 internal** | most |
| Report 2 — robotics/WM/JEPA | 109 | 6,495,538 | 1,096 | **7** | **18** |

Harness: fan-out search → URL-dedup → claim extraction → **3-vote adversarial verification**
(2/3 refutes kills) → synthesis. Raw outputs preserved beside this file as
`raw_report1_autonomous_driving.json` / `raw_report2_robotics_wm_jepa.json`.

⛔ **READ THE COVERAGE GAPS FIRST (§0).** Both surveys failed to find admissible evidence across
most of the requested surface. Silence here is *absence of verified evidence*, never a null result.

---

## §0 COVERAGE GAPS — what these reports may NOT be cited on

**Report 1 (AD).** Five of eight requested areas produced **zero** admissible claims — data
engineering, edge inference, planners, low-data training, self-supervised representations. Every
candidate was voted down **0-3**, including several that would have been load-bearing:

| refuted candidate | vote | why it hurts |
|---|---|---|
| anchored-vocabulary scaling (1024→16384 anchors, EPDMS 85.02→87.35) | 0-3 | bears on our anchor-count knee |
| **factorised path × velocity vocabulary** | **0-3** | maps 1:1 onto our diagnosed "5-way maneuver softmax mixes LAT+LON" |
| SparseDriveV2 beats diffusion | 0-3 | decoder choice |
| World4Drive self-supervised latent WM (all 3 claims) | 0-3 | closest external analogue to our design |
| dense-navigation lever (84.0 → 85.5 PDMS) | 0-3 | goal conditioning |
| Alpamayo edge latency (13.33 s → 4.10 s on DGX Spark) | 0-3 | the entire edge-inference question |

⭐ **Re-verifying the factorised lat/lon vocabulary is the single most valuable follow-up survey
task** — it is the one refuted claim that speaks directly to our largest measured lever.

**Report 2 (robotics/FM/WM/JEPA).** Four of six areas produced **zero** surviving claims: VLAs /
world-action models (π-0/π-0.5, GR00T N1.x, OpenVLA, latent action models, action chunking,
flow-matching heads), robotics (cross-embodiment, sim-to-real, hierarchical policies, failure
recovery), LLMs (reasoning, distillation, structured latent reasoning), foundation models
(small-model scaling laws, data quality vs quantity, MoE at small scale). **Genie 3, DreamerV4 and
Cosmos appear in no surviving claim.**

⇒ **Neither report may be cited on encoder scale, anchor count, target-speed heads, or the no-map
strategic brain.** Our anchor-count-over-encoder-scale knee (54.7 M / 104.2 M / 251.9 M) and our
smaller-IDM-is-better ladder (0.86 M > 2.90 M > 19.98 M) received **neither support nor challenge**.

---

## §1 ⭐ THE HEADLINE: our own imagination number is CONFOUNDED

**Evidence class: PUBLISHED (SkyJEPA, verified 3-0) applied to MEASURED-ours.**

On 2026-07-29 we reported a pure-imagination local exponent rising **1.03 → 1.63 → 1.87 → 1.91**
across 0.4–2.0 s (ADE 0.0406 / 0.0829 / 0.1607 / 0.2753 / 0.4215; global OLS 1.456) and called it
*"decay accelerates"*. **That reading is not supported by the measurement.**

SkyJEPA (arXiv 2606.23444, NYU ARPL + Brown; Rao/Zhang/Balestriero/LeCun/Loianno) defines the
missing instrument:

```
CR_k = e_k,rollout / e_k,teacher-forced       # compounding ratio
ER_k = E[e_k − e_{k−1}]                       # per-step growth
```

ADE-vs-horizon **cannot separate**:
- **(a)** predicting 2 s ahead is intrinsically harder than 0.4 s ahead → acceleration is *task
  difficulty*, no architecture change justified;
- **(b)** the rollout compounds its own error → compounding is real, and the fix is rollout-recovery
  *training*, not a bigger horizon.

SkyJEPA trains rollout on T=20 steps (1.0 s @ 20 Hz) yet measures to k=60 (3.0 s, **3× the trained
horizon**), reporting **CR ≈ 1.4** vs **≈ 2.4** for an autoregressive baseline. Fig. 6 caption
confirms the confound-separation reading: *"values above 1 indicate error accumulation caused by
recursion"*.

⚠️ **Caveats that must travel.** The 1.4/2.4 figures are prose readings of a figure carrying the
authors' own "approximately" hedge. **CR is still RISING at k=60, not asymptoting** — the word
"bounded" is an editorialisation. Domain is quadrotor state over a low-dimensional physics vector
with a physics-structured prober, **not** a high-dimensional visual world model: **the metric design
transfers, the magnitudes do not.**

### EXPERIMENT E-CR (RANK 1 overall) — pre-registered
Add a teacher-forced arm to `taniteval/taniteval/imagination.py`; re-score the **same** 40 val
episodes both ways; report **CR_k and ER_k at k=4/8/16/20** with the **episode-cluster bootstrap**
(`taniteval/ci.py`), paired form.

- **Outcome (a)** CR_k flat near 1 ⇒ the accelerating-decay narrative is **FALSIFIED**; the
  acceleration is task difficulty; **no architecture change is justified**.
- **Outcome (b)** CR_k rising super-linearly ⇒ compounding is real ⇒ **§3 rollout-recovery training**
  is indicated, **not** a bigger horizon.

**Cost: ESTIMATED 0–6 GPU-hours** (zero if per-step teacher-forced latents are already dumped).

---

## §2 The 2 s cap is cheaper to lift than we assumed — our code already reserves the path

**Evidence class: MEASURED-ours (code) + PUBLISHED (HorizonDrive, verified 2-1).**

We described the cap as architectural — an `index_select` over a learned horizon embedding. True but
incomplete. `stack/tanitad/models/predictor.py` already carries:

1. a fixed-size causal window (`PredictorConfig.window=6`, learned `self.pos` asserted equal),
2. a **k=1 residual head** (horizons default `(1,2,4)`),
3. an unused projection annotated verbatim
   `self.out_proj = nn.Linear(state_dim, d)  # reserved: feed predictions back`.

A rolling-window rollout is therefore **drop oldest → append the k=1 prediction → re-run**: no
retrain, no new parameters, cost flat in horizon. HorizonDrive (arXiv 2605.11596) is the published
form — per-step compute and memory bounded by a T+K_S context *"regardless of total horizon"*,
sustained 1.8 s per 10-frame chunk on one RTX 5090 at 256×512 (5.8 s at 384×768, the unflattering
operating point reported alongside — not cherry-picked).

### EXPERIMENT E-ROLL (RANK 2, inference only)
Recursively apply the existing k=1 head past 20 steps on the deployed arm; measure ADE **and CR_k**
out to **4 s and 6 s** (2× and 3× the trained horizon, matching SkyJEPA's 3× protocol).

⚠️ **PRE-REGISTERED: we EXPECT divergence to worsen past 2 s**, because the k=1 head was trained
under teacher forcing. **A negative result is informative and is the trigger for §3 — not a
refutation of the approach.** Cost: **ESTIMATED 2–4 GPU-hours.**

---

## §3 Nobody has solved long latent rollout. The field has three workarounds.

**Evidence class: PUBLISHED, three independent primaries (V-JEPA 2 2-1, MoP-JEPA 2-1, HorizonDrive 3-0).**

⭐ **V-JEPA 2 states it against its own interest**: *"autoregressive prediction suffers from error
accumulation: the accuracy of the representation-space predictions decreases with longer
autoregressive rollouts, thereby making it more difficult to reliably plan over long horizons"*.
Its **deployed mitigation is a SHORTER horizon** — CEM with **planning horizon = 1 for ALL tasks**
(Table 3, same for the Cosmos comparator), executing only the first action before re-planning, each
action constrained to an L1 ball of radius 0.075 (~13 cm). **At 4 fps that horizon is ~0.25 s.**

The three workarounds:
1. **per-step replanning** (V-JEPA 2);
2. **sub-goal / graph search over REAL dataset states** (MoP-JEPA) — retains 0.05–0.15 success to
   goal distance 14 where Dense and MDN baselines hit zero by 6–8, **but its x-axis is true goal
   distance, not rollout step, and it never feeds a predicted latent back into itself**;
3. **training-time recovery from self-corrupted histories** (HorizonDrive).

⇒ **Our 2 s cap is NOT obviously behind the state of the art, and pure-imagination decay is the
field's open problem, not a TanitAD defect.**

⚠️ Qualification: V-JEPA 2 *does* attack error accumulation at training time (a rollout loss at T=2
alongside teacher forcing), so "their mitigation is a shorter horizon" is true of deployment, not of
the whole method.

---

## §4 CONTRADICTION AGAINST US — driving-specific horizon SOTA is ~20 s, 10× our cap

**Evidence class: PUBLISHED-preprint (verified 3-0, MEDIUM confidence — single source, no replication).**

HorizonDrive reaches **~20 s of genuine autoregressive rollout** (student rolls N=20 AR steps at
K_S=10 on a T=11 context = 211 frames ≈ 21 s @ 10 FPS), improving on the strongest long-horizon
streaming baselines by **FID −52 %, FVD −37 %, ARE −21 %, DTW −9 %**.

**Two deflations, both the authors' own:**
- The headline *"minute-scale"* is demonstrated **ONLY qualitatively** — Appendix G, self-collected
  dataset, **no metric attached**. Defensible reading: *"quantitatively evaluated to ~20 s, degrading
  more slowly than baselines"*, **never "stable to 60 s"**.
- Error still **GROWS** within the 20 s (their Appendix D).

**Why it does not falsify our numbers:** HorizonDrive rolls out in a **pixel-reconstructive VAE
latent** scored with **video-generation metrics** (FID/FVD/VBench) plus trajectory metrics; we roll
out in a **predictive latent with no decoder** and score ADE in metres. Not like-for-like.

⇒ **What it DOES falsify is any comfort that our 2 s cap matches SOTA on horizon for driving.**
**The MECHANISM transfers, not the number: scheduled rollout recovery — train on
prediction-corrupted histories.**

---

## §5 An online-replanning executor MASKS transition-model error (0 GPU, protocol change)

**Evidence class: PUBLISHED-but-unrefereed (MoP-JEPA, verified 2-1). Authorial concession.**

Verbatim: *"With 10 % false edges, replanning raises success from 0.40 to 1.00. Execution can
therefore repair a low-precision graph online, whereas realroute measures model fidelity."*
**A model whose transition set is 10 % wrong scores PERFECT under an online-replanning executor.**

**Both directions bite us:**
1. Any CEM/MPC/receding-horizon executor on the 4-brain — precisely the V-JEPA 2 recipe and our
   obvious next step — will **HIDE** the rising local exponent rather than fix it.
2. Our open-loop 0.45 m → closed-loop 1.69 m gap is **not a pure fidelity signal either**, because
   the executor and its 2 s replanning cadence are part of the measurement. *(Consistent with
   `f818a49`, which already retracted that verdict as confounded.)*

### ACTION (RANK 4, 0 GPU)
Report open-loop fidelity (CR_k) and closed-loop success as **two separate decision inputs** in
`taniteval/taniteval/closedloop.py`. **PRE-REGISTER: a closed-loop improvement with flat CR_k is an
EXECUTOR gain, not a world-model gain, and MAY NOT be quoted as world-model progress.**

---

## §6 Latent action-conditioned prediction is validated on real hardware, and the AC head is data-cheap

**Evidence class: PUBLISHED (Meta FAIR tech report, verified 3-0 against primary; weights MIT-licensed).**

V-JEPA 2-AC post-trains an action-conditioned **latent** predictor on **under 62 hours** of DROID
video — no reward, no task-specific training, no data from the deployment environments — then does
zero-shot image-goal pick-and-place on Franka arms in **two unseen labs**: Lab 1 cup **80 %** / box
**80 %**, Lab 2 cup **80 %** / box **50 %** (n=10 each). The weak 50 % cell reproduces ⇒ not
cherry-picked. Inputs: raw video + 7-D end-effector state only.

⇒ **The strongest external support for our latent-space design intent over pixel reconstruction**,
and evidence that an action-conditioned transition head can be fitted on a **very small
action-labelled slice** — relevant to adding a target-speed or intent channel without a corpus rebuild.

⚠️ **Bounds:** sub-goals are **human-provided** (two sub-goal images + final goal, auto-switch after
4 timesteps), which is why pick-and-place (80 %) scores **above** grasp (65–70 %) — they are not
nested tasks under one protocol. Planning horizon 1, 800 CEM samples × 10 refinements, **16 SECONDS
PER ACTION** ⇒ **not evidence for long-horizon latency or for real-time control**.

⛔ **This finding MUST NOT be used to launder a frozen-encoder recommendation** — see §7.

---

## §7 Frozen vs trained encoders — our measurement stands UNCHALLENGED and UNCORROBORATED

**Evidence class: this is a statement about the evidence set.**

**All seven candidate claims failed verification, in BOTH directions:**

| candidate | direction | vote |
|---|---|---|
| V-JEPA 2-AC's frozen 1B ViT-g contradicts our ceiling | against us | **1-2** |
| Web-DINO/SigLIP-2 action-R² ceiling (−0.01/+0.05 → 0.16–0.17 after full IDM fine-tune) | **for us** | **0-3** |
| PSNR-vs-action-R² orthogonality | **for us** | 1-2 |
| frozen-encoder collapse under blur/noise | **for us** | **0-3** |
| FF-JEPA: 18 M task-trained beats frozen DINOv2, 91.80 % vs 61.0 % long-horizon | **for us** | **0-3** |
| (2 further FF-JEPA claims) | for us | 0-3 |

⭐ **Absence of a verified contradiction is NOT corroboration.** Our own MEASURED result — frozen
DINOv2/I-JEPA at **2.92 m** fwd ADE and speed-blind; from-scratch + speed action channel
**3.73 → 0.83 m**, speed R² **0.61 → 0.965** — remains the strongest evidence this programme
possesses in *either* direction. Under our own taxonomy the frozen-encoder position rests **entirely
on MEASURED-ours**; any sentence implying external corroboration would be an **INHERITED claim
deciding a GPU-day**, which CLAUDE.md rule 1 forbids.

⭐ **THE ONE GENUINELY UNEXPLORED AXIS.** V-JEPA 2-AC's frozen encoder is **VIDEO/motion-pretrained
on 1 M+ hours**, unlike DINOv2's and I-JEPA's **IMAGE** pretraining. **Nobody has run
frozen-video-encoder vs from-scratch on a metric driving task.** Cheapest discriminator: freeze a
V-JEPA 2 ViT-L, train only our predictor head, same corpus and steps.

⚠️ Time-sensitivity: arXiv 2606.07687 and 2606.09311 are ~2 months old and were **unverifiable
within the search budget rather than shown false** — UNVERIFIED LEADS worth a second probe, **not
refuted science**.

---

## §8 Spectral constraint on the latent transition — right idea, wrong domain, sequence it last

**Evidence class: PUBLISHED-preprint, 7 days old (verified 3-0 for what the paper SAYS; confidence LOW).**

Koopman Dreamer (arXiv 2607.19719) constrains the deterministic latent transition to Koopman-style
2D rotation-scaling blocks with bounded spectral radius: over a 64-step action-conditioned rollout
from a 32-step context, **−23.2 %** decoded observation MSE (8/9 tasks) and −89.4 % latent MSE (9/9)
vs DreamerV3.

⛔ **Relay ONLY the 23.2 % observation-space figure.** The authors concede latent coordinates are not
comparable across models; Koopman's state is 2048-d vs DreamerV3's size12m preset, MSE is raw and
un-normalised in each model's own latent space, and **a spectrally contractive latent has bounded
norm BY CONSTRUCTION** — an unknown fraction of the 89.4 % may be a units artifact.

⛔ **Wrong domain:** all DMC results are low-dimensional **proprioceptive** state; second domain is
LiDAR vectors; the encoder symlog-transforms continuous vector inputs; **there are NO pixel/CNN/ViT
observations anywhere.** Seed count never stated.

⚠️ **The stated trade-off is dangerous for us:** lowering the radius attenuates error but **erases
persistent task information** — exactly the speed/scale magnitude our speed-channel fix bought
(speed R² 0.965).

**SEQUENCING: do NOT run before §1.** If CR_k is flat there is no compounding to attenuate and this
is unmotivated. If CR_k rises, it becomes a legitimate **second** lever after rollout-recovery
training, judged on **CR_k and speed R² jointly** (the mechanism predicts a trade).

---

## §9 AD-side findings (Report 1)

### 9.1 ⭐ TOP AD RECOMMENDATION — a STALE-ABSENCE BUG IN OUR OWN CODE (0 GPU)

**Evidence class: MEASURED-ours.**

`taniteval/taniteval/pseudosim.py` computes the right PDM shape — `(∏ gate_m) × (Σ w·s / Σ w)` — but
**the gate product is EMPTY**, and the stack correctly refuses to call the result a Driving Score.
Four places say so, e.g. `control.py:1460`: *"With no collision gate and no drivable-area term,
NOTHING computed here is a PDMS number. It scores PROGRESS, RECOVERY and CONTROL."*
**Credit where due: the stack refuses to emit rather than emitting a constant.**

The stated blocker: `obstacle.offline` cuboids cannot be joined because
`episode_id = int.from_bytes(clip_id[:4])` **collides** — 242 clip_index rows onto 40 val episode_ids.

⛔ **THAT BLOCKER IS STALE** (CLAUDE.md rule 2 — absence at ONE location is not absence). Three later
agents independently proved the join by replaying the cache's own recipe
(`discover_r0_clips → sorted → split_clips(val_frac=0.2, seed=0)`), reproducing **600/600 val,
2376/2376 train, 40/40 eval-pod** episode_ids with committed `join_proof.json` artifacts
(`2026-07-26-h2-classifier/PRE_REGISTRATION.md`; `2026-07-27-idm-v3/IDM_V3.md:78`). A real collision
exists (2376 → 2342 distinct ids, 34 collisions) but **resolves by construction** once candidates are
restricted to the parity clip list. `pseudosim.py:690` itself concedes `obstacle.offline`
(97.4438 % corpus coverage, `reference_frame='rig'` on 100 %) *"WOULD supply them"*.

**Root-cause class: a blocker written down once, never re-probed, while the fix landed in a sibling
directory.** Residual blocker is a **file download**, not a science problem.

**ACTION (RANK 1 on the AD side):** (i) re-run the join proof against pseudosim's 40-episode val
cache and publish its own `join_proof.json` (~1 engineer-hour, 0 GPU); (ii) download the matching
`obstacle.offline` chunks; (iii) wire the NC gate **with the `filter_m` semantics of §9.3**.

### 9.2 A SOTA planner's speed output is a POSITIONAL SHORTCUT — and we can test ourselves today

**Evidence class: PUBLISHED (PlanT 2.0, Tübingen; verified 3-0 against raw PDF text + un-primed re-probe).**

PlanT 2.0 scores **DS 92.4 ± 1.7** on Bench2Drive — above Think2Drive (91.85) and every sensor model
listed — **while carrying this**: §5.4 *"Positional shortcuts"*, verbatim *"In all four scenarios, at
rotation values around 10-15 degrees, the predicted speed abruptly increases to signal regular
driving."* Mechanism: the expert only turns when the road is free and the dataset contains only
successes, so the model *"learns a shortcut by using its own rotation as a predictive signal for when
the road is clear"* and *"once a decision is made to go around the obstacle, it ignores the scene and
does not reconsider"*. Boxed Insight 4: *"Shortcuts introduced by rigid expert demonstrations can be
exploited and lead to unrealistic behavior."*

⚠️ **Four qualifications that must travel:** (a) PlanT 2.0 is a **PRIVILEGED** planner (GT perception)
— "benchmark-SOTA" is true but is **not** a sensorimotor end-to-end result; (b) root cause is
CARLA-specific (scripted expert + success-only data), and **we train on HUMAN-DRIVEN logs with no
"only turn when road is free" invariant, so a NULL on our data would NOT contradict this paper**;
(c) the paper reports partial mitigation (positional augmentation helps but *"the model continues to
be influenced by high rotation values"*); (d) the paper **actively contradicts** any
"transfers to any model with a speed channel" clause — it chose an object-centric planner *because*
perturbing sensor inputs in a controlled fashion is hard. **For us that qualification does not
apply** — see below. *(Its data-composition ablation was voted 0-3: do not quote its Town13 numbers.)*

⭐ **WHY OUR VERSION IS STRICTLY CLEANER THAN PLANT'S.** `taniteval/taniteval/pseudosim.py` already
ships a heading-perturbation axis that is an **exact** controlled ego-rotation on **real frames** —
`H = K R K⁻¹`, *"a pure camera rotation is exact for ARBITRARY scene depth (max|dH| = 0.000e+00,
30 conditions)"* — exposed as `GridSpec.dyaw_deg`, with a MEASURED **0 %-out-of-envelope span of
|dψ| ≤ 12°**. PlanT rotates the ego in simulation, moving world pose **and** object input together;
our warp changes **ONLY the observed heading** — the scene is bit-identical real footage under an
exact homography. **If our target-speed head moves with dψ, it is unambiguously keyed to heading
rather than scene content. There is no confound to argue about.** The flagship exposes explicit
`vt_band` / `vt_speed` channels (`stack/tanitad/models/flagship_v15.py:413,519,577`).

### EXPERIMENT E-DPSI (RANK 2 on the AD side) — pre-registered
Sweep `GridSpec.dyaw_deg` across the validated envelope on the 40-episode val cache with `dlon`
fixed; plot **(i)** produced `vt_speed`, **(ii)** `tspeed_5s`, **(iii)** `long_abs` against dψ.
**Look for a STEP, not a slope.** Cost **~0.3 GPU-day** on existing checkpoints — no retrain, no
renderer, no AlpaSim. Free rider: run the same sweep on the CL-SFT arm.

- **ADVANCES:** directly interrogates our strongest measured result — oracle speed recovers **0.1899
  of the 0.2140** oracle gap (**88.7 %**), entirely longitudinal. If target speed is heading-sensitive,
  **part of that longitudinal deficit is a SHORTCUT, not a capacity or scale limit, and the fix is
  augmentation rather than a new channel.**
- **FALSIFIES:** if flat in dψ, it strengthens the estimation-problem reading (consistent with our
  IDM scale-limitation result) and closes the shortcut hypothesis cheaply.

⛔ **THE SHARP LIMIT, PRE-REGISTERED.** PlanT's onset is 10–15°; our measurement-grade envelope tops
out at **|dψ| ≤ 12°** (`pseudosim` falsifies anything beyond). We cover the **LOWER edge only**.
**A null inside ±12° is NOT "we are clean" — it is "no shortcut below 12°", and must be written up
that way.**

### 9.3 EPDMS — with the correction a naive reading gets wrong

**Evidence class: PUBLISHED + PRIMARY, code-backed (devkit fetched twice, identical).**

```
EPDMS = (∏_{m ∈ {NC, DAC, DDC, TLC}} filter_m) × (Σ_{m ∈ {TTC,EP,HC,LK,EC}} w_m·filter_m / Σ w_m)
weights: Ego Progress 5, TTC 5, Lane Keeping 2, History Comfort 2, Extended Comfort 2
gates:   No at-fault Collision {0,½,1}, Drivable Area {0,1}, Driving Direction {0,½,1}, Traffic Light {0,1}
```

⭐ **THE CORRECTION:** `filter_m(agent, human) = 1.0 if m(human) = 0 else m(agent)` — **an agent's
zero does NOT zero the score on scenes the human reference also fails.** Implementing from the plain
wording yields **PDMS-v1 semantics and over-penalises arms.** Independent arithmetic check:
third-party 2026 papers report an EPDMS denominator of **16** (= 5+5+2+2+2) vs PDMS-v1's **12**
(= 5+5+2) — consistent only with these weights. Our `taniteval/registry.py:272` already carries
PDM-Closed (NAVSIM v2) EPDMS 51.3 as an external baseline.

⚠️ **Two further defects:** full EPDMS also needs the two-stage pseudo-closed-loop protocol, reactive
background traffic, traffic-light state, and a **HUMAN-REFERENCE rollout** for the filter ⇒ **a lane
graph is NECESSARY, NOT SUFFICIENT.** And LK *"is disabled on intersections where the centerline
annotations often don't match the actual lane markings"*.

### 9.4 The open-loop repudiation is REAL but SCOPED — it does not reach our headline

**Evidence class: PUBLISHED (Bench2Drive, NeurIPS 2024 D&B; against-interest; quote verbatim-verified twice).**

README news 2025/02/05: *"authors' should stop reporting results on nuScenes open-loop planning and
reviewers should not ask for nuScenes open-loop planning results"*; plus *"L2 error is not a
meaningful indictor at all"*. Still live 2026-07-29 with newer entries above it ⇒ actively maintained.

**Three reasons it does not land on us, all from the source:**
1. **SCOPE** — the endorsed carla_garage doc states the criticism *"does not extend to other nuScenes
   benchmarks for perception and prediction tasks"*. Our headline is action-conditioned world-model
   fidelity, **not** a planning score.
2. **THE NAMED PATHOLOGY DOES NOT APPLY** — the mechanism is **AD-MLP**, a model that *"solely
   extrapolates past movement"* topping the board. **We publish the CV floor (0.8377 / 0.6917) and
   beat it CI-separated (0.4271 [0.3675,0.4871] / 0.4108 [0.3956,0.4273]) — we PASS the exact
   diagnostic.** Other cited flaws (ego-status leakage, tiny val) are nuScenes-specific.
3. **DATE** — 2025/02/05 predates the window. In-window anchor: arXiv 2605.00066 (Bosch, Apr 2026),
   L2 vs Bench2Drive DS **Spearman ρ = −0.36, p = 0.43**; NAVSIM PDMS **ρ = 0.90, p = 0.002**.

⚠️ **ATTRIBUTION NOTE:** *"L2 error is not a meaningful indictor at all"* does **not** appear in
carla_garage despite being typeset as a quotation from it — it is Bench2Drive's own stronger wording.

⛔ **INTERNAL FLAG — the claim's own corroboration was unsound:** it leaned on our 0.45 → 1.69
open-to-closed divergence, which commit `f818a49` **already retracted** as confounded, and which
measures the PLANNER, not the validity of the fidelity metric. **Do not re-cite it as confirmation.**

**ACTION (0 GPU):** rename the headline to `wm_fidelity_ade_2s` in `MODEL_REGISTRY.md` and every
report, and **co-locate the CV floor with it** so it cannot be misread downstream.

### 9.5 Fail2Drive — the failure is REPRESENTATIONAL, not sensory

**Evidence class: PUBLISHED-preprint (Tübingen, 2026-04-09; verified 2-1).**

TransFuser++ — the only camera+LiDAR arm and the **strongest** generalizer (67.5 HM shifted vs 80.8
in-distribution) — **fails to stop for obstacles clearly present in its own LiDAR returns**, and
collapses **33.15 → 0.00 HM** in ConstructionPermutations when **only the familiar warning sign is
removed**. Abstract verbatim: *"ignoring objects clearly visible in the LiDAR and failing to learn
the fundamental concepts of free and occupied space."* Fig. 7: returns present, detector emits no
box, policy does not react.

⚠️ **Three defects that must travel:** (1) "0.00 HM" is partly a **HARMONIC-MEAN FLOOR ARTIFACT** —
HM = 0 whenever SR = 0 regardless of DS, and per-scenario DS/SR are not published separately: **quote
it as SR → 0, never as a magnitude**; (2) per-scenario **n is UNPUBLISHED** (~5 implied by "4 out of
5 cases"), **no CIs anywhere** ⇒ by our own interval rules this is a **DIRECTION, not a decision-grade
effect size**; (3) the appended *"adding a sensor modality should not fix it"* is the claim-writer's
**PREDICTION, not a paper result** — the study is observational on off-the-shelf checkpoints with
n=1 camera+LiDAR arm. **Strip that sentence when citing.**

*(Note: the verifier's first probe wrongly reported the LiDAR check absent and nearly refuted on it —
a textbook rule-2 single-location absence error, caught by a second probe.)*

**EXPERIMENT (RANK 3 on the AD side, shares §9.1's unblock):** a **LINEAR** probe on **frozen** v1
latents predicting free-vs-occupied structure from `obstacle.offline` (87,481 boxes, 10 dynamic
classes, 97.44 % coverage, `reference_frame='rig'` 100 %). Linear-only, so it measures what the latent
**already** encodes. **<0.5 GPU-day**, depends on §9.1.

---

## §10 RANKED PROGRAMME (expected gain / cost) — nothing here justifies a retrain

| rank | action | cost | depends | §|
|---|---|---|---|---|
| **1** | **CR_k teacher-forced arm** — disambiguates our own headline | **0–6 GPU-h** | — | §1 |
| **2** | **`obstacle.offline` join + chunk download** → real NC gate | ~2–3 eng-days, **<1 GPU-day** | — | §9.1 |
| **3** | **dψ speed-shortcut sweep** on existing checkpoints | **~0.3 GPU-day** | — | §9.2 |
| **4** | **Reporting protocol**: separate open-loop fidelity from closed-loop success; rename headline; co-locate CV floor | **0 GPU** | — | §5, §9.4 |
| **5** | **Recursive k=1 rollout past 2 s** (expect divergence — that is the point) | **2–4 GPU-h** | — | §2 |
| **6** | Linear free/occupied probe on frozen latents | <0.5 GPU-day | R2 | §9.5 |
| **7** | Re-score ~6 archived arms under the gated composite | <1 GPU-day | R2 | §9.3 |
| **8** | Frozen **VIDEO**-pretrained encoder (V-JEPA 2 ViT-L) vs from-scratch | 1 training run | — | §7 |
| **9** | Rollout-recovery training (scheduled corruption) | training run | **R1 outcome (b)** | §3, §4 |
| **10** | Spectral/Koopman reparameterisation | fine-tune | **R1 outcome (b)**, after R9 | §8 |
| **11** | DAC/LK/TLC on AlpaSim/NuRec | weeks, renderer-bound | **do NOT start before R1–R7** | §9.3 |

**Total GPU for R1–R7: under ~2.5 GPU-days, all on EXISTING checkpoints.**

⭐ **NEITHER SURVEY JUSTIFIES A RETRAIN, A NEW ARCHITECTURE, OR A QUANTISATION TARGET.** Every claim
that would have is listed refuted in §0.

**GUARDRAIL ON R7, pre-registered now:** if gated ranking **agrees** with ADE ranking, ADE-decided
gates are vindicated for this corpus and we say so. If it **disagrees**, name in advance which
decisions re-open — and note 2605.00066 makes disagreement the **EXPECTED** outcome, so a flip must
not be reported as a surprise or as an indictment of v1.

---

## §11 Leads worth a second probe — HYPOTHESIS class, NOT admissible for decisions

Both split **1-2** in verification ⇒ admissible for **experiment design only**:

1. **Sub-JEPA** (arXiv 2605.09241, public code): impose the isotropic Gaussian prior inside multiple
   random **LOW-DIMENSIONAL SUBSPACES** rather than the full ambient embedding. A cheap regulariser
   swap against our existing `SigRegConfig` (`stack/tanitad/config.py`, `n_slices=512`, weight 0.1,
   `free_dims` exemption). Low-dimensional control only in the source.
2. **Delta-JEPA** (arXiv 2606.31232): an inverse-dynamics decoder should take the latent
   **DISPLACEMENT** `z_{t+1} − z_t` rather than concatenated endpoints (**+12.60** on Push-T). A
   one-line change to our IDM — and it asserts that decoder **INPUT FORM, not capacity**, is the
   lever, which bears directly on our smaller-is-better IDM ladder.

Plus the §0 re-verification target: **the factorised path × velocity action vocabulary.**

---

## Provenance

- `raw_report1_autonomous_driving.json` — 106 agents, 6,415,995 tokens, 1,111 tool calls, 2,046,956 ms
- `raw_report2_robotics_wm_jepa.json` — 109 agents, 6,495,538 tokens, 1,096 tool calls, 2,015,144 ms
- Both include full `findings` / `refuted` / `unverified` / `sources` arrays with per-claim votes.
