# RESULT — E-REFC-EGO-1: the REF-C ancestors fed ego speed straight into the planner, and the field has already measured what that costs

*Architecture & Inference FlyWheel · Research Lab literature stream · 2026-09-03. **0 GPU.**
Pre-registration in `SPEC.md`. Every paper cited here is **banked** — library key given
inline, sha256 in `library.json`. Every PDF was opened and its sha re-checked against the
registry entry before any number was read; `raw/PRIMARY_EXTRACTS.md` names the sections.*

**Tier stamp.** Every number in §2–§7 is **PUBLISHED (primary, read from a banked PDF)**
about *other people's models on other people's benchmarks*. **No number here is a TanitAD
result and none is T0/T1.** The eval tier of each source is named where it matters —
nuScenes L2 = open-loop displacement; NAVSIM PDMS = non-reactive simulation; nuPlan CLS
and CARLA DS = closed-loop — because **the finding of this WP is that these tiers
DISAGREE, and the disagreement IS the echo.**

---

## 0. The verdict in ten lines

1. **"The original REF-C paper" is TWO papers, and only one of them carries the ego.** Our
   `refc.py` docstring names both: REF-C **was** TCP-C (**lib `2206.08129`**, NeurIPS 2022);
   the current decoder is DiffusionDrive-style (**lib `2411.15139`**, CVPR 2025). The
   measurement encoder that eats `v0` today is the **TCP** half. *(MEASURED at our source,
   `stack/tanitad/refs/refc.py:1-13`; corroborated `MODEL_REGISTRY.md:2167-2174`.)*
2. **TCP feeds the ego speed into the PLANNER path, unguarded.** `m = [g, v]` → a 2×FC-128
   MLP measurement encoder → `j_m`; `j_traj = avgpool(F) ⊕ j_m` → GRU → waypoints. **No
   dropout, no mask, no presence bit, and no head is denied it.** *(lib `2206.08129` §3.2 +
   Table 5.)*
3. **⚠️ OUR DOCSTRING LAUNDERS AN UNVALIDATED HYPER-PARAMETER AS INHERITED PRACTICE.**
   `refc.py:10-11` says the TCP-C stack is kept verbatim including *"the measurement encoder
   with per-sample ego-dropout"*. The **encoder** is TCP's (FC-128 ×2 — bit-for-bit our
   `MeasurementConfig(hidden=128, d_out=128)`). The **ego-dropout is not**: the string
   "dropout" occurs **zero times** in TCP *(two probes: a full-text grep for
   `dropout|drop out|mask| drop `; and TCP's own layer table, which lists the measurement
   encoder with no regulariser)*. `ego_dropout = 0.5` is a TanitAD invention that **nobody,
   upstream or here, has ever swept.**
4. **DiffusionDrive's decoder has NO ego port at all** — its condition is scene context
   (deformable spatial cross-attn + agent/map cross-attn + timestep modulation). Where ego
   status exists it comes from the **host backbone** (Transfuser on NAVSIM, SparseDrive on
   nuScenes), and DiffusionDrive **never documents it** *(two probes: zero hits for
   "velocity"/"acceleration"/"driving command" in the body; "ego status" appears only in the
   bibliography; its Table 3 "Ego Query" column is ✓ in all six rows, i.e. never ablated)*.
5. **⭐ THE CENTREPIECE — a blind ego-only agent ties the best camera planners on nuScenes.**
   Ego-MLP (velocity, acceleration, yaw angle, command; **no perception**) scores nuScenes
   **L2 avg 0.35 m** = BEV-Planner++ (0.35), better than VAD-with-ego (0.37) and
   UniAD-with-ego (0.46). PARA-Drive's independent, standardised protocol reproduces it to
   four decimals: **PARA-Drive without ego 0.5574 vs AD-MLP blind 0.5568.**
   *(lib `2312.03031` Tab. 1; lib `paradrive-cvpr2024` Tab. 6.)*
6. **The fix is a metric that can see the road, plus a split that removes the trivial
   scenes.** The same two blind agents that tie on L2 are **10.1× worse on off-road rate**
   (1.21 vs 0.12), **3.0× worse off-lane** (2.45 vs 0.83), and **6.7× worse on collisions in
   the non-straight subset** (0.94 vs 0.14). *(lib `paradrive-cvpr2024` Tab. 6.)*
7. **The benchmark, not the model, decides whether the echo is visible.** NAVSIM filtered
   out scenes a constant-velocity agent already solves, and the CV agent fell from **PDMS 79
   → 22**; the ego-only MLP then sat **18.4 PDMS behind** the sensor agents (65.6 vs 84.0).
   On NAVSIM **v2/navhard** the gap widens to **4.0×** (Ego MLP **EPDMS 14.1** vs best
   **56.6**) — *even though v2 gives the blind agent MORE ego (velocity **and motion
   history**)*. *(lib `2406.15349` §3.1 + Tab. 1; lib `2506.04218` §3 + Tab. 2.)*
8. **The echo is detectable on a trained checkpoint for zero training compute, as an
   ASYMMETRY.** VAD-Base **with** ego: blanking every camera moves L2 **0.37 → 0.46
   (×1.24)** while detection NDS → **0.0**; scaling velocity ×0.5 moves it **0.37 → 3.19
   (×8.6)**. The **same model without** ego: blanking the cameras moves L2 **1.25 → 4.33
   (×3.46)**. *(lib `2312.03031` Tab. 2 + App. Tab. 3.)*
9. **⭐ WHERE the ego enters is the whole argument.** Injected into the **scene
   encoder/backbone** it is measurably harmful — CARLA-TransFuser's own Table 10:
   **DS 56.68 → 45.35** when the velocity is summed into the positional embedding at all
   four backbone stages, *"a sharp drop in DS, which cannot be recovered"*. Injected into the
   **planner head only**, PARA-Drive calls the same information *"marginal"* on the val set
   and *useful* on the hard subset, and NAVSIM's TransFuser gains **1.5–2.6 PDMS** from it.
   *(lib `2205.15997` Tab. 10; lib `paradrive-cvpr2024` §5; lib `2406.15349` Tab. 2.)*
10. **The best-evidenced mechanism is per-scalar ego dropout with a presence bit — and it
    COSTS OPEN-LOOP SCORE when it is working.** PlanTF: **OLS 88.55 → 87.07 (−1.48)** while
    **NR-CLS 83.19 → 86.48 (+3.29)** and **R-CLS 74.79 → 80.59 (+5.80)**; rate sweep
    0/0.25/0.50/0.75 → NR-CLS 77.28/81.70/83.71/**86.48**. DRAMA reproduces it on NAVSIM
    PDMS: **0.835 → 0.848** at state-rate 0.5. *(lib `2309.10443` Tab. II + VIII; lib
    `2408.03601` Tab. 2.)* ⛔ **Any refcv4 gate scored on ADE will read the working guard as
    a regression.**

---

## 1. Q1 — which paper is "the original REF-C"? (two, and our docs do NOT disagree)

`stack/tanitad/refs/refc.py:1-13`, verbatim from our own source (MEASURED):

> *"REF-C model: Anchored-Diffusion-C — a DiffusionDrive-style trajectory head. REF-C was
> TCP-C (a two-branch GRU trajectory/control stack, arXiv 2206.08129). This revision
> REPLACES the GRU trajectory + control branches with an ANCHORED TRUNCATED-DIFFUSION
> trajectory decoder in the DiffusionDrive spirit (arXiv 2411.15139) … The rest of the
> TCP-C stack is KEPT verbatim: the … encoder, the measurement encoder with per-sample
> ego-dropout, the LAW latent-world-model aux, the strategic-ctx hierarchy graft, and the
> REF-C.1 target-speed class head."*

`MODEL_REGISTRY.md:2167-2174` agrees (*"REF-C — Anchored-Diffusion-C (DiffusionDrive-style)
… Replaces the old TCP-C GRU"*). **No doc conflict.** But the ambiguity in *"the original
refc paper"* is exactly where the ego question lives:

| | **TCP** (lib `2206.08129`) | **DiffusionDrive** (lib `2411.15139`) |
|---|---|---|
| what we kept | encoder, **measurement encoder**, LAW aux, hierarchy graft, speed head | **the anchored decoder** (anchors + per-anchor conf + per-anchor offset + truncated denoise) |
| where our `v0` goes | **here** | not here |
| feeds ego? | **YES, into the planner** | **no ego port in the decoder** |

**⇒ For the ego question the ancestor that matters is TCP.**

---

## 2. Q2–Q5 — TCP (lib `2206.08129`), the ego-carrying ancestor

**Inputs at inference** (§3.1 "Problem formulation", verbatim):

> *"Given the state x comprised of the sensor signal i, **the speed of the vehicle v**, and
> the high level navigation information g including a discrete navigation command and the
> coordinates of navigation target provided by the global planner…"*

| ego input | present? | note |
|---|---|---|
| **speed `v`** | ✅ | the only kinematic channel, current step |
| longitudinal acceleration | ❌ | absent |
| yaw rate | ❌ | absent |
| ego history | ❌ | *"we only have sensor inputs at the current time step"* (§3.2.2) |
| nav command (discrete) | ✅ | part of `g` |
| nav target coordinates | ✅ | from the global planner — **a supplied route**, which our E12 refuses |

**Where it enters** (§3.2, verbatim):

> *"the navigation information g is concatenated with the current speed v to form the
> measurement input m, then an MLP based measurement encoder takes m as its input and
> outputs the measurement feature `j_m`."*

then (§3.2.1):

> *"the image feature map F is average pooled and concatenated with the measurement feature
> `j_m` to form `j_traj`. … we feed `j_traj` into a GRU to auto-regressively obtain future
> waypoints."*

Layer table (Table 5): **Measurement Encoder = FC 128, ReLU, ×2** — *exactly* our
`MeasurementConfig(hidden=128, d_out=128)`. The inheritance is literal.

### Q4 — is any ego input withheld from the planner path? **No — the opposite.**

`j_m` is **shared by both branches** (trajectory AND multi-step control). TCP has no notion
of a vision-pure goal path.
⇒ **Our edge E11** (`refc_v3.py:213` and `REFC_V3_DESIGN.md:174` — no ego state into
`{z_tac, g_str, ĝ_tac}`) **has no ancestor precedent. It is stricter than TCP, not
inherited from it** — worth saying plainly, because E11 has been argued as if it were a
lineage constraint. §5 shows it is stricter than *every* planner in the comparator set.

### Q5 — what does TCP say about ego shortcut learning?

**It names the failure and defends against it in the wrong place.** §1, verbatim:

> *"Simple PID controllers may perform worse in situations such as taking a big turn or
> **starting at the red light due to the inertial problem** of end-to-end models [29]."*

Reference [29] is Codevilla et al., *Exploring the Limitations of Behavior Cloning*
(lib `1904.08980`) — the origin of the inertia problem, which attributes it to *"the
spurious correlation between input velocity and output acceleration"*. **TCP attributes it
to the controller and fixes it with multi-step control prediction** — not to the speed
input, and with no input-side guard.

⇒ ⛔ **`ego_dropout = 0.5` (`refc.py:429`) is NOT inherited from TCP.** Documentation fix
required, in the same root-cause class as the registry's *"REF-C ranks with the un-refined
anchor's score"* correction: a true statement about our code, phrased so a reader infers a
false provenance.

*(The one TCP mechanism worth carrying forward is a **grounding aux**: a speed-prediction
head that predicts the current speed **from the image feature**, loss weight 0.001 —
*"to help the agent better estimate its current state"*. We inherited it and have never
ablated it. See §6.5.)*

---

## 3. Q2–Q5 — DiffusionDrive (lib `2411.15139`), the decoder ancestor

**The decoder has no ego port** (§3.4, verbatim):

> *"Given the set of sampled noisy trajectories … we begin by applying deformable spatial
> cross-attention to interact with Bird's Eye View (BEV) or Perspective View (PV) features
> based on the trajectory coordinates. Subsequently, cross-attention is performed between
> the trajectory features and the agent/map queries derived from the perception module,
> followed by a feed-forward network (FFN). To encode the diffusion timestep information,
> we utilize a Timestep Modulation layer, which is followed by a MLP that predicts the
> confidence score and the offset relative to the initial noisy trajectory coordinates."*

Ego status is a property of the host (§4.2, App. A): on NAVSIM it takes *"the same
perception modules and ResNet-34 backbone as Transfuser"* and *"the training and inference
recipe directly follows Transfuser"* — and **NAVSIM's Transfuser consumes ego status**.

⚠️ **Scale note.** DiffusionDrive uses **20 k-means anchors, 8 waypoints / 4 s**, 60 M
params, 2 denoise steps. We run **128 FPS anchors × 8 slots** — 6.4× the vocabulary, a
different sampling rule (FPS, deliberately, on a ~74 % straight corpus). **Their anchor
ablations do not transfer unexamined**; quoting them as ours is the `df`-scope class.

### ⭐ The DiffusionDrive result that kills one candidate method

App. B, **Table 8 "Comparison on driving priors"** (NAVSIM navtest):

| train prior | infer prior | NC | DAC | TTC | Comf. | EP | **PDMS** |
|---|---|---|---|---|---|---|---|
| Anchored Gaussian (k-means anchors) | Anchored Gaussian | 98.2 | 96.2 | 94.7 | 100 | 82.2 | **88.1** |
| Anchored Gaussian | **Extrapolated traj. from current status** | 96.3 | 91.7 | 90.4 | 100 | 76.8 | **81.3** |
| **Extrapolated traj. (single anchor)** | Extrapolated traj. | 97.3 | 94.0 | 92.6 | 100 | 79.6 | **84.7** |

Their reading, verbatim: the extrapolated prior *"fails to cover the potential action space
and can not effectively handle challenging scenarios (e.g., obstacle avoidance and
turning)"*, *"consistent with comparisons to ego-status-based planners in Tab. 1 of
NAVSIM"*.

⇒ **A kinematic-extrapolation prior costs 3.4 PDMS even when trained for**, 6.8 when
swapped in at inference. §7.4 turns this into a design rule.

---

## 4. Q6 — THE CENTREPIECE: the ego-status baseline

### 4.1 The original claim — AD-MLP (lib `2305.10430`)

An MLP on **ego state only** (past 4-frame pose, velocity, acceleration, one-hot command —
21 channels; no camera, no LiDAR) reaches nuScenes L2 **0.20/0.26/0.41 m (avg 0.29)** and
collision avg **0.19 %**, beating UniAD (1.03 m / 0.31 %).
⚠️ **It has a known leak** and must never be quoted without it: AD-MLP feeds the **past ego
trajectory ground truth**. Two independent groups re-implemented it for that reason —
BEV-Planner (as Ego-MLP) and PARA-Drive, whose footnote states the released checkpoint *"is
trained with GT data leakage"*. **For our purposes the admissible floor arm is the
re-implementation, not AD-MLP's published row.**

### 4.2 The definitive statement — BEV-Planner (lib `2312.03031`, CVPR 2024)

**What they fed.** Ego-MLP = **velocity, acceleration, yaw angle, driving command**, the
history-GT leak removed. Ego status enters existing models at **two separate sites**,
ablated independently:
- **in the BEV encoder** — BEVFormer's `can_bus` path (*"projecting the ego status onto the
  hidden features and incorporating it into the BEV query"*). ⚠️ **Both UniAD and VAD do
  this by default and neither paper discusses it** (App. C: *"a nuance not addressed by
  current end-to-end autonomous driving approaches"*). Confirmed at source in the official
  code: `UniAD/modules/transformer.py:44` `use_can_bus=True`, `:154-155`
  `bev_queries = bev_queries + can_bus_mlp(can_bus) * use_can_bus`; the same construction at
  `VAD/modules/transformer.py:41,71`.
- **in the planner** — *"directly concatenating the ego query with a vector containing ego
  status"*; in VAD's code `VAD_head.py:418`,
  `ego_fut_dec_in_dim = embed_dims*2 + len(self.ego_lcf_feat_idx)`.

**Table 1 (nuScenes val, open-loop):**

| ID | method | ego in BEV | ego in Planner | **L2 avg (m)↓** | **Coll. avg (%)↓** | **CCR avg (%)↓** |
|---|---|---|---|---|---|---|
| 1 | UniAD | ✗ | ✗ | 1.03 | 0.77 | **1.93** |
| 2 | UniAD *(official ckpt)* | ✓ | ✗ | 0.66 | 0.62 | 1.72 |
| 3 | UniAD | ✓ | ✓ | 0.46 | 0.37 | **1.59** |
| 4 | VAD-Base | ✗ | ✗ | 1.25 | 1.09 | 3.82 |
| 5 | VAD-Base | ✓ | ✗ | 0.72 | 0.54 | 2.72 |
| 6 | VAD-Base *(official ckpt)* | ✓ | ✓ | **0.37** | 0.33 | 2.47 |
| 7 | **GoStraight** *(continue at current velocity)* | — | ✓ | **0.83** | 1.08 | **8.62** |
| 8 | **Ego-MLP** *(no perception at all)* | — | ✓ | **0.35** | 0.37 | **2.93** |
| 10 | BEV-Planner *(4-frame temporal, no ego)* | ✗ | ✗ | 0.55 | 0.59 | 4.26 |
| 12 | BEV-Planner++ | ✓ | ✓ | **0.35** | 0.34 | 3.16 |

Three readings: (1) **a blind agent is state of the art on L2**; (2) **one ego vector is
worth more than the entire perception stack on this metric** (VAD 1.25 → 0.72 → 0.37 as ego
is added first to the BEV then to the planner); (3) ⭐ **the road-adherence metric REVERSES
the ranking** — Ego-MLP's CCR 2.93 is worse than UniAD-with-ego (1.59) *and* worse than
UniAD-**without** ego (1.93); GoStraight is worst at 8.62.

**Table 2 + App. Table 3 — the perturbation asymmetry (our best echo instrument):**

| model | perturbation | L2 avg (m) | ×baseline | Det. NDS | Map mAP |
|---|---|---|---|---|---|
| VAD-Base **w/ ego** | — | 0.37 | 1.00× | 45.5 | 47.0 |
| VAD-Base **w/ ego** | **blank all images** | **0.46** | **1.24×** | **0.0** | **0.0** |
| VAD-Base **w/ ego** | snow / fog / glare / rain | 0.45 / 0.45 / 0.44 / 0.45 | 1.19–1.22× | 36.1 / 34.3 / 41.7 / 29.1 | — |
| VAD-Base **w/ ego** | **v × 0.0** | **6.16** | **16.6×** | 45.5 | 47.0 |
| VAD-Base **w/ ego** | v × 0.5 | 3.19 | **8.6×** | 45.5 | 47.0 |
| VAD-Base **w/ ego** | v × 1.5 | 3.20 | 8.6× | 45.5 | 47.0 |
| VAD-Base **w/ ego** | v = 100 m/s | 208 | 562× | 45.5 | 47.0 |
| VAD-Base **w/o ego** | — | 1.25 | 1.00× | 45.1 | 53.7 |
| VAD-Base **w/o ego** | **blank all images** | **4.33** | **3.46×** | **0.0** | **0.0** |

⇒ **The diagnostic is the ratio of ratios.** With ego, destroying the entire perception
system costs **1.24×** while halving one scalar costs **8.6×**. Without ego, destroying
perception costs **3.46×**. *That gap is the echo, and it needs no training compute.*

**Two more signatures worth copying:**
- **Convergence** (Fig. 5): *"Introducing ego status in the BEV-Planner++ enables the model
  to converge very rapidly"* — the classic shortcut fingerprint.
- **Attribution** (Fig. 6): with ego status *"the activation range of the feature map
  predominantly encompasses the immediate vicinity around the ego vehicle, frequently
  manifesting behind the vehicle itself"*, versus ahead of it without. Their conclusion:
  *"the BEV-Planner++ method has almost not learned any effective information."*

⚠️ **The authors are explicitly NOT arguing for removing ego status** (App. C, verbatim):
> *"our position is not opposed to the use of ego status; rather, we argue that within the
> context of current datasets and evaluation metrics, the integration of ego status can
> significantly impact, and even determine, the planning results."*

**⇒ This is the published form of the PI's 2026-09-03 reinterpretation.** The rule is not
"no ego"; it is "prove the plan is not *determined* by the ego."

### 4.3 The independent replication that also supplies the FIX — PARA-Drive (lib `paradrive-cvpr2024`)

*Weng, Ivanovic, Y. Wang, Y. Wang, Pavone — NVIDIA/USC/Stanford, CVPR 2024, pp. 15449–15458.*
⚠️ **CVF-only; there is no arXiv mirror and no released code.** ⭐ **Now banked as a
PRIMARY** (`--local`, sha256 recorded), so it is admissible; it was PUBLISHED-SECONDARY at
the start of this WP.

**Ego inputs (§4, verbatim):** *"This BEV feature map, in conjunction with the data from the
ego vehicle (e.g., **high-level commands, CAN bus, history trajectories**), forms the
exclusive input to the planning head."* CAN bus is defined in §5 as *"velocity,
acceleration, angular velocity, etc."* — i.e. **almost exactly refcv4's intended triple.**
**Where:** ⭐ **the planner head only — never the BEV encoder.** No ego dropout *(two probes:
`dropout` 0 hits, `drop out` 0 hits over the full text; no code exists to check)*.

**Table 6 (nuScenes val, PARA-Drive's own standardised protocol):**

| split | method | ego? | Coll. Ave_all (%) | **L2 Ave_all (m)** | **Offroad** | **Offlane** |
|---|---|---|---|---|---|---|
| val | UniAD | No | 0.40 | 0.8317 | 0.91 | 1.74 |
| val | VAD | No | 0.30 | 0.7830 | 1.03 | 1.93 |
| val | **PARA-Drive** | **No** | 0.17 | **0.5574** | **0.12** | **0.83** |
| val | **AD-MLP (blind, re-implemented)** | **Yes** | 0.20 | **0.5568** | **1.21** | **2.45** |
| val | PARA-Drive+ | Yes | 0.13 | 0.4939 | 0.11 | 0.78 |
| **targeted** | **PARA-Drive** | No | **0.14** | 0.9082 | – | – |
| **targeted** | **AD-MLP (blind)** | Yes | **0.94** | 0.9360 | – | – |

**The reading (arithmetic mine, on PUBLISHED values):** a full camera stack (0.5574) and a
**blind extrapolator with no images at all** (0.5568) are **identical to three decimals on
L2**. What separates them: **Offroad 10.1×**, **Offlane 3.0×**, and on the hard subset
**collision 6.7×** (0.94 vs 0.14) — with **collision@3 s 3.62 vs 0.72, 5.0×**.

**Their conclusion, verbatim:** *"the improvements brought by the CAN bus and history
trajectories become **marginal** in the val set … We find that, in the targeted scenarios as
well as the map compliance error rates, AD-MLP has significantly worse performance than
PARA-Drive. This suggests that the **open-loop evaluation scheme is still very
informative**…"*

⭐⭐ **Two instruments we can copy directly, and both are cheap:**
- **The "targeted" split**, verbatim: *"we **exclude frames with a command of 'keep
  forward'**, which results in a total of **686 challenging key-frames** on the nuScenes val
  set."* This is an **open-loop-compatible, label-only version of NAVSIM's constant-velocity
  filter** — and we already have manoeuvre labels.
- **The command-only floor (Table 4)**: give the planner *only* the high-level command — no
  BEV, no ego, no perception. Col Ave_all **5.88**, L2 Ave_all **4.66**, versus 0.13 / 0.53
  with BEV. **That is the true no-information value**, and the distance from 4.66 down to
  the blind agent's 0.5568 *is* the size of the ego shortcut. **This is precisely the
  `CLAUDE.md` "constant-only control that must read the no-information value" requirement,
  published.**

### 4.4 The counter-case, and how it was engineered — NAVSIM v1 (lib `2406.15349`) and v2 (lib `2506.04218`)

**v1 ego status** (§3, verbatim): *"the vehicle's **current speed, acceleration, and
navigation goal**, jointly termed the ego status … we provide the navigation goal as a
one-hot vector with three categories: left, straight, or right"* — **current timestep only,
no history**, and the goal is **derived from the lane graph, not the human trajectory,
explicitly "to prevent label leakage"**.

**Table 1 (navtest, PDMS):** Constant Velocity **20.6** · **Ego Status MLP 65.6** · UniAD
83.4 · LTF 83.8 · TransFuser **84.0** · PARA-Drive 84.0 · Human 94.8.
**Table 3 (leaderboard 1.1, 3 seeds):** Ego Status MLP **66.4 ± 0.9** · LTF 83.5 ± 0.6 ·
TransFuser 83.9 ± 0.4 · Hydra-MDP 91.3.
⇒ **an 18.4-PDMS gap** where nuScenes showed none: *"a clear gap between agents relying
solely on the ego status and those considering sensor data, **in contrast to results on
nuScenes**."*

⭐⭐ **WHY the gap exists — the single most actionable item in this WP.** NAVSIM curated the
corpus by the **triviality of the kinematic extrapolation** (§3.1, verbatim):

> *"the baseline of maintaining a constant velocity and heading achieves a PDMS of 79 % on
> the OpenScene dataset, where human-level performance corresponds to 91 %. … We remove
> highly simplistic scenes by detecting if the **constant velocity agent exceeds a PDMS of
> 0.8**. Similarly, we remove scenes in which the human trajectory results in a PDMS of less
> than 0.8. … the score of the constant velocity agent **dropping to 22 %**, whereas the
> human expert achieves a score of 95 %."*

**Constant-velocity PDMS 79 → 22 by data curation alone** — a larger effect than any
architectural fix in this document. **The echo was never primarily a model defect; it was a
corpus property, and the fix was a filter.**

**TransFuser's own ego ablation (Table 2):** B1 *goal only* (drop velocity **and**
acceleration) → **−1.5 to −2.6 PDMS**; B2 *goal + velocity only* (drop acceleration) →
**−1.0 to −2.1**. Seed spread A1–A3 **± 0.56**. Verdict, verbatim: *"while TransFuser
benefits from the ego status, it is not purely relying on the kinematic state for
planning."* ⭐ **That sentence, backed by a small ablation delta against a large blind-vs-
sensor gap, is the published template for the claim refcv4 must be able to make.**

**NAVSIM v2 = "Pseudo-Simulation for Autonomous Driving" (lib `2506.04218`, CoRL 2025).**
Two-stage evaluation, Stage-2 synthetic views rendered by 3D Gaussian Splatting, scored by
**EPDMS**, on **navhard** (450 Stage-1 + 5,462 Stage-2 observations). Their own property
table claims pseudo-simulation exposes **causal confusion** where open-loop does not.
**Table 2 (navhard leaderboard, snapshot 03/2026, EPDMS):**

| CV | **Ego MLP** | LTF | LTFv6 | NavFormer | RAP | ZTRS | GuideFlow | SimScale | DrivoR | PDM-C |
|---|---|---|---|---|---|---|---|---|---|---|
| 11.4 | **14.1** | 25.1 | 31.9 | 34.1 | 39.6 | 48.1 | 51.5 | 53.2 | 54.5 | **56.6** |

⇒ v1 separated blind from best by **1.28×**; navhard separates by **4.0×** — **and v2 gives
the blind agent MORE ego** (*"ego status features such as the velocity **and motion
history**"*). ⚠️ **A figure of "Ego Status EPDMS 64.0 on navhard" is in circulation and is
wrong** — the primary says 14.1; the authors explicitly *"discourage the use of self-reported
and unofficial 'NAVSIM v2' benchmark splits, such as reporting the EPDMS on the NAVSIM v1
navtest dataset without conducting two-stage pseudo-simulation."*

---

## 5. Q2–Q4 — the comparator table (every row read from the primary)

| model | lib key | ego inputs at inference | where they enter | withheld from the planner? |
|---|---|---|---|---|
| **TCP** | `2206.08129` | **speed `v`** + nav cmd + nav target coords | `m = [g, v]` → **MLP measurement encoder** (FC128×2) → `j_m`, concatenated to pooled image features for **BOTH** branches | **No** |
| **TransFuser (CARLA)** | `2205.15997` | **NONE.** *"we do not use velocity as an input to our models"* | — (the velocity variant sums a linear projection of `v` into the **positional embedding at all 4 backbone stages**) | n/a — ⭐ and see §5.1 |
| **TransFuser (NAVSIM)** | `2406.15349` | **speed, acceleration, nav goal** (3-way one-hot, from the lane graph) | ego-status MLP fused with BEV/image features before the waypoint head | No |
| **UniAD** | `2212.10156` + code | ego status via BEVFormer `can_bus` in the **BEV encoder**; planner variant is third-party | ⚠️ **the paper never mentions it**; `transformer.py:44,154-155` has it ON by default | not by design — **by silence** |
| **VAD** | `2303.12077` + code | *"the current status of the ego vehicle `s_ego` **(optional)**"* + command | `f_ego = [Q'_ego, Q''_ego, s_ego]` → **MLP planning head** (`VAD_head.py:418`), **plus** `can_bus` in BEVFormer | ⚠️ **paper says it omits ego status *"to avoid shortcut learning in the open-loop planning"* — but the released official checkpoint uses it in the planner, and that is the checkpoint scoring L2 0.37.** A real paper-vs-code contradiction, on the headline number |
| **VADv2** | `2402.13243` | **ego state `E_state`** + navigation `E_navi`, MLP-embedded | **added** at the scoring head: `p(a) = σ(MLP(φ(E(a), E_scene) + E_navi + E_state))` over a 4096–8192 vocabulary | No |
| **Hydra-MDP** | `2406.06978` | **ego status `E`** | ⭐ `V'_k = Transformer(MLP(V_k)) + E` — the ego embedding is **ADDED to the anchor queries** *before* the environment cross-attention | No |
| **PARA-Drive** | `paradrive-cvpr2024` | **CAN bus (velocity, acceleration, angular velocity)** + command + history traj | ⭐ **the planner head ONLY — never the BEV encoder** | No (but the trunk is protected) |
| **DiffusionDrive** | `2411.15139` | none in the decoder; inherits the host's | — | n/a — **undocumented** |

**The pattern:** of the eight whose primaries we hold, **seven feed ego status and NONE
withholds it from the planner by design.**
⇒ **Our E11 is unprecedented in this literature, in the strict direction.** Its nearest
relative is AdaptiveAD (lib `2511.13079`), which removes ego status from the **BEV encoder**
and runs a separate ego branch fused late — i.e. the literature's structural line is drawn
at the **scene encoder**, not at the planner.

### 5.1 ⭐ The one CLOSED-LOOP measurement, and it is the sharpest thing in the WP

CARLA-TransFuser **Table 10** (Longest6, mean over 3 evaluations):

| velocity input? | creeping? | **DS ↑** | RC ↑ | IS ↑ |
|---|---|---|---|---|
| ✗ | ✗ | 46.35 | 78.28 | 0.63 |
| ✗ | ✓ | **56.68** | 92.28 | 0.62 |
| ✓ | ✗ | 37.34 | 64.27 | 0.65 |
| ✓ | ✓ | **45.35** | 86.22 | 0.52 |

Verbatim: *"Including the velocity input leads to a **sharp drop in DS, which cannot be
recovered** through the creeping behavior."* **−11.33 DS with creeping, −9.01 without.**

⚠️ **Read the ATTACH POINT before generalising this** — it is the whole lesson. Their
velocity injection is *"combined with the learnable positional embedding through
element-wise summation and fed into the transformer at **all 4 stages of the backbone**"*.
**That is a scene-encoder injection**, exactly the site AdaptiveAD root-causes and exactly
the site PARA-Drive protects. **This result does NOT say "ego input is bad"; it says "ego
input into the trunk is bad", and it is the only closed-loop number in the literature that
says it.** *(Same evidence class as the `df`/`step_s` scope traps: a true measurement that
becomes false the moment it is quoted outside its attach point.)*

⇒ **refcv4's placement is already the protected one** — the measurement encoder feeds the
decoder condition; the encoder trunk never sees `v0`. **Keep it that way, and say so in the
design doc with this citation**, because it is the difference between the +2 PDMS result and
the −11 DS result.

---

## 6. Q7 — anti-echo METHODS, ranked by evidence, with attach points

### 6.1 ⭐ Per-channel ego dropout with a PRESENCE BIT — best evidence, ~zero cost

**Mechanism (PlanTF's State Dropout Encoder, lib `2309.10443` §IV-A, Fig. 3):** each ego
scalar gets **its own linear embedding** + positional encoding; a **learnable query
cross-attends** the token set; **during training each embedded state token EXCEPT position
and heading is dropped with probability `p`.** Velocity, acceleration and steering are
droppable; pose is exempt.

**Table II (nuPlan Test14-random / Test14-hard):**

| model | SDE | OLS | NR-CLS | R-CLS | (hard) OLS | NR-CLS | R-CLS |
|---|---|---|---|---|---|---|---|
| state3 *(pose only — no v, no a, no steer)* | — | 81.13 | **85.99** | 79.38 | 71.43 | 68.44 | 63.14 |
| state5 | ✗ | 87.71 | 81.76 | 74.51 | 84.54 | 68.67 | 54.91 |
| state5 | ✓ | 88.80 **(+1.09)** | 86.73 **(+4.97)** | 75.75 (+1.24) | 84.29 (−0.25) | 71.28 (+2.61) | 61.88 **(+6.97)** |
| state6 | ✗ | 88.55 | 83.19 | 74.79 | 85.89 | 67.57 | 58.99 |
| state6 | ✓ | 87.07 **(−1.48)** | 86.48 **(+3.29)** | 80.59 **(+5.80)** | 83.32 **(−2.57)** | 72.68 **(+5.11)** | 61.70 (+2.71) |

**Table VIII, the rate sweep (state6+SDE):**

| dropout | OLS | NR-CLS | R-CLS |
|---|---|---|---|
| none | 88.33 | 77.28 | 74.10 |
| 0.25 | **89.11** | 81.70 | 78.44 |
| 0.50 | **89.12** | 83.71 | 77.52 |
| **0.75** | 87.07 | **86.48** | **80.59** |

Their deployed config: *"a state attention dropout encoder with a **dropout rate of 0.75**"*,
plus state perturbation at probability 0.5. Their own reading of the base problem:
*"models incorporating historical motion data exhibit superior off-policy evaluation
performance (OLS), [but] manifest significantly poorer performance in closed-loop metrics …
attributed to the well-established 'copycat' problem or learning shortcuts."*

**Corroboration on our metric family** — DRAMA (lib `2408.03601` Tab. 2), NAVSIM PDMS:

| method | state dropout | fusion dropout | PDMS |
|---|---|---|---|
| Transfuser baseline | 0 | 0 | 0.835 |
| +FSD | 0 | 0.1 | 0.842 |
| +FSD | **0.5** | 0 | **0.844** |
| +FSD | **0.5** | **0.1** | **0.848** |

⭐ **DRAMA's differentiated policy is the transferable design**: a **high** rate on the ego
state (0.5) and a **low** one on the perception features (0.1) — *"A relatively low dropout
rate is assigned for the fusion feature to preserve its integrity."* **PLUTO** (lib
`2404.14327`) reuses PlanTF's SDE unchanged for **+2.60 nuPlan score from the SDE alone**.
Three independent adoptions, three benchmarks, one mechanism.

**The presence bit** — GRU-D (lib `1606.01865`) is the primary for why a zeroed channel
needs a companion indicator, and its argument is the PI's verbatim: mean/forward imputation
*"cannot distinguish whether missing values are imputed or truly observed"*. Fix: a **mask
`m_t ∈ {0,1}^D`** plus **time-since-last-observation `δ_t`**, and a learned decay. Ablation
(MIMIC-III mortality AUC): **mask only 0.8367**, interval only 0.8266, both + decay
**0.8527**, vs GRU-mean 0.8192.

⛔ **STATUS IN OUR CODE — half-built, and the other half is already pre-registered by a
sibling stream.** `refc.py` computes `keep` on every forward and passes it to the decoder as
`ego_keep`, but withholds it from the measurement encoder because
`ego_valid_channel = False` (`refc.py:585`, gate at `:2060`). The
`2026-09-03-ego-zero-collision` WP MEASURED the consequence: **52.227 % of training samples
present a zero speed and only 4.263 % of those are a genuine standstill (22.5 : 1), against
100 % genuine at eval — a 23.5× shift in the token's meaning.** That WP owns the fix
(`H-ARCH-EGOZERO-1`). **This stream's contribution is the external mandate and the RATE**:
nobody upstream validated 0.5, and the two papers that swept it landed on **0.5 (NAVSIM
PDMS) and 0.75 (nuPlan closed-loop)** — with 0.75 *costing open-loop score*.

**Attach point for refcv4:** replace the single concatenated `[v, nav, keep, known]` vector
with **one token per scalar** (`v`, `a_long`, `yaw_rate`, `nav`), each with its own presence
bit, aggregated by a learnable query. **~10⁴ params, zero inference-time cost.**
⚠️ **PlanTF exempts pose from dropout; we have no pose channel, so all three of our planned
inputs fall in the droppable class and none in the exempt class.**

### 6.2 ⭐ Keep the ego OUT of the scene encoder — the structural fix

**AdaptiveAD (lib `2511.13079`)** names the root cause as *premature fusion of ego status in
the upstream BEV encoder* and splits the network into a **scene branch with ego status
deliberately omitted** (multi-task supervised) and an **ego branch** on the planning loss,
fused late by a scene-aware gate. nuScenes L2 avg **0.47 vs VAD 0.61**, collision **0.12 %
vs 0.28 %**, 3.0 FPS vs 3.4. Its **perturbation panel is the anti-echo evidence** (L2 avg,
AdaptiveAD vs VAD): clean 0.47/0.61 · v×0 **4.08/5.54** · ×0.5 2.41/3.05 · ×1.5 2.74/3.22 ·
**100 m/s 5.06/14.93**; on NAVSIM PDMS **86.4/81.2** clean and **61.4/51.5** at v×0.
⚠️ **The dual branch alone made L2 WORSE (0.57 → 0.62)** and only recovered with a
distillation term added — this fix is not free.

Corroborated in three places: CARLA-TransFuser Table 10 (§5.1, trunk injection, **−11.33
DS**); PARA-Drive (planner-head-only injection, *"marginal"* and safe); and BEV-Planner
App. C (UniAD/VAD's undeclared `can_bus`).

⇒ **We are already ahead**: our encoder never sees `v0`, and E11 keeps the goal path
ego-free on top. **Recommendation: keep it, cite §5.1 for why, and re-scope E11 from an
absolute prohibition to a stated boundary** — the literature's line is at the scene encoder,
not at the planner, and the PI's reinterpretation moves us to the same place.

### 6.3 ⭐ Counterfactual / copycat mitigation — the mechanism AND the detector

The copycat/inertia lineage is the closest published analogue of ego-echo: *"copying my
previous action"* and *"extrapolating my current velocity"* are one failure with two input
names. Its root is **Codevilla et al.** (lib `1904.08980`), *"the spurious correlation
between input velocity and output acceleration"*.

| paper | lib key | mechanism | headline |
|---|---|---|---|
| Causal Confusion in IL | `1905.11979` | graph-parameterised policy `π_G(X) = f_φ([X ⊙ G, G])` + **targeted intervention** to select `G` | GTA-V driving: **with** action history 0.834 val perplexity / 2.94 collisions; **without** 0.989 / **1.30** — *better validation, worse driving.* ⚠️ **plain dropout gave minimal improvement** on their confounded Atari suite |
| Fighting Copycat Agents | `2010.14876` | adversary predicts `a_{t−1}` from the embedding **conditioned on `a_t`** (so only the *non-shared* information is stripped) + a VIB. α = 2, λ = 1e-3 | PO-Hopper 293±83 → **1086±262**; ablation w/o the conditioning → 322±74, i.e. **the conditioning IS the method** |
| Keyframe-Focused | `2106.06452` | reweight the BC loss by **action-predictability error** `APE_t = (ψ*(a_{t−1},…) − a_t)²` — upweight frames the history cannot predict | CARLA NoCrash-Dense 33.00±4.19 → **43.44±0.79**; **zero architectural cost** |
| **Residual action prediction** | `2207.09705` | a **memory branch** trained to predict the **residual `Δa_t = a_t − a_{t−1}`**, entering the policy trunk through a **stop-gradient** | CARLA NoCrash Train-Dense **34.1±7.5 → 52.0±2.3**. ⭐ target ablation: absolute `a_t` **41.3**, `a_{t−1}` **47.0**, **residual 52.0**; **without stop-gradient 45.0** |
| OREO | `2110.14118` | drop feature-map units **grouped by shared VQ-VAE code** (object-structured dropout), p = 0.5 | CARLA Navigation BC 16.9±7.6 → **35.7±10.2**; plain Dropout 30.4, DropBlock 21.7 |
| CausalVAD | `2603.18561` | backdoor adjustment over a prototype dictionary of scene contexts; subtractive intervention on perception logits and queries | +4.7 M params / +6 ms; nuScenes L2 **0.54 vs 0.74**, collision **0.11 % vs 0.44 %**; NAVSIM PDMS **87.2 vs 80.5**; **turning split 0.69 vs 1.07** |

⚠️ **Two cautions.** (a) The residual in `2207.09705` sits on the **auxiliary target of a
memory branch**, not on the plan's output parameterisation — see §7.4. (b) `1905.11979`
measured that **plain dropout barely helps** on its benchmarks, which is in real tension
with §6.1's driving evidence. **Both are primary. That tension is a reason to run the arm,
not to pick a side in prose.**

### 6.4 Adversarial / gradient-reversal removal of ego from the scene representation

**DANN (lib `1505.07818`)** is the primary: GRL is identity forward, `−I` backward, between
the trunk and a nuisance classifier. ⭐ **Its schedule is the part people get wrong**:
`λ_p = 2/(1 + exp(−γ·p)) − 1`, γ = 10, `p` = training progress 0→1 — **ramp λ from zero;
never start at full strength.**

⛔ **NOT FOUND: a driving-specific adversarial head that strips ego *kinematics* from a
visual feature.** Two differently-worded probes returned nothing; the nearest instance
(`2010.14876`) strips the **previous action**, not speed/accel/yaw. **A genuine gap, and
publishable if it works** — but a research bet, and it partly fights our own design (we
*want* the ego in the measurement encoder; we want it out of the goal path, which E11
already achieves structurally and for free).

### 6.5 Auxiliary scene-grounding losses — ⚠️ the literature is genuinely conflicted

| for | against |
|---|---|
| **PRIX (lib `2507.17596`)**, raw-pixel planner, NAVSIM PDMS ladder: planning loss only **70.4** → +det-box **82.3** → +semantic seg **85.7** → +det-cls **86.9** → full **87.8**. **+17.4 PDMS from grounding alone.** | **SSR (lib `2409.18341`)** removes *all* supervised perception and beats UniAD/VAD (L2 avg **0.75** vs 1.03 / 1.22) at 10.9× the speed; its Table 4 shows **adding map/obstacle supervision WORSENS L2 to 0.81–0.86**. |
| **NAVSIM (lib `2406.15349`)** Table 2 config E1: removing the aux tasks drops PDMS — BEV segmentation is load-bearing. | **BEV-Planner+Map (lib `2312.03031` Tab. 3)**: adding map perception moved L2 **0.55 → 0.96** and collision **0.59 → 0.89**, *while CCR improved **4.26 → 2.60***. |
| **PARA-Drive**: removing online mapping does not move L2/collision but **map-compliance error rises by a large margin**; and NAVSIM v2's own NavFormer baseline adopts *"object tracking and map segmentation decoders [to] provide auxiliary perception supervision. Following PARA-Drive."* | |
| **TCP itself (lib `2206.08129`)** carries a **speed head predicting the current speed FROM THE IMAGE** (λ 0.001) — a grounding aux we inherited and have never ablated. | |

⭐ **The conflict resolves once you stratify.** `2312.03031` Tables 4–5 break the *same*
BEV-Planner+Map comparison down by command: on **straight** commands the map aux *hurts*
(L2-ST 0.48 → 0.97, Collision-ST 0.40 → 0.91); on **turns** it *helps decisively*
(**Collision-LR 2.25 → 0.78, a 2.9× reduction**). Turns are **13 %** of the eval set, so the
aggregate reads as a regression. **This is our four-metric-families rule and our
stratification discipline, published — and it is the same shape as PARA-Drive's L2-versus-
Offroad split.**

### 6.6 Data-side: filter or stratify by the triviality of the kinematic extrapolation

Not a model method at all, and on the evidence the **highest-leverage** one:

- **NAVSIM's CV filter** moved the constant-velocity agent from **PDMS 79 → 22**
  (lib `2406.15349` §3.1), and the blind-vs-sensor gap from ~0 to 18.4 PDMS.
- **PARA-Drive's "targeted" split** does the same thing with **labels only and zero
  simulation**: drop `keep forward` frames → **686 keyframes**, on which the blind agent's
  collision rate goes **6.7× worse** than the camera model's.
- Corpus properties measured: nuScenes is **73.9 % straight driving**, **13 % turning
  scenes**, **87 % of eval samples straight** (lib `2312.03031`); NAVSIM v1 ~**75 %** trivial
  (lib `2406.15349`).

⇒ **Before adding any mechanism, measure how much of OUR corpus a constant-velocity /
constant-yaw-rate agent already solves.** If the answer is "most of it", no anti-echo method
is measurable on our current val split and the first work item is a split, not a loss.

---

## 7. Q8 — evaluation-side ECHO DETECTION (what actually catches it)

Eight instruments, all published, ordered by cost to us. ⛔ **Each is a `CLAUDE.md`-compliant
probe only with the panel controls**: a constant-only control that must read the
no-information value (PARA-Drive's command-only floor **is** that control, published), a
raw-input floor, and printed `n` and `d`.

| # | instrument | primary | what it reads | published reference value | our cost |
|---|---|---|---|---|---|
| **D1** | **Blank-image sensitivity** | `2312.03031` Tab. 2 + App. Tab. 3 | Δmetric when the image is destroyed | echoing **×1.24**; healthy **×3.46** | **0 GPU-train** |
| **D2** | **Ego-perturbation response** (v × {0, 0.5, 1.5}, 100 m/s) | `2312.03031`; `2511.13079`; `2603.18561` | Δmetric when one ego scalar is scaled | echoing **×8.6 at v×0.5**, ×16.6 at v=0 | 0 GPU-train |
| **D3** | ⭐ **Endpoint-gradient magnitude `‖∂(x_T, y_T)/∂ s_0‖`** | `2309.10443` Fig. 2(c)(d) | how strongly the plan endpoint depends on the initial kinematic state | qualitative: the SDE model is *"less sensitive to variations in kinematic states"* | **one backward pass per window** |
| **D4** | ⭐ **Plan-predictability probe** | `2010.14876` §5 | fit a probe predicting the plan from `(v, a, yaw)` ALONE; compare its error on OUR model vs on the HUMAN expert. **More self-predictable than the human ⇒ echoing** | copycat is **~10× more self-predictable** (Ant 0.66e-2 vs expert 6.91e-2); after the fix 0.66e-2 → 2.20e-2 | ridge on banked dumps, 0 GPU |
| **D5** | **The ego-only floor arm** | `2305.10430`; `2312.03031` ID-8; `paradrive-cvpr2024`; `2406.15349` | train an MLP on ego state only. **If the full model does not beat it, the full model is an echo** | nuScenes L2 **0.35** (ties SOTA); NAVSIM **65.6** (−18.4); navhard EPDMS **14.1** (−42.5) | one tiny training run |
| **D5b** | ⭐ **The command-only floor** — the constant-only control | `paradrive-cvpr2024` Tab. 4 | planner gets ONLY the discrete command; must read the no-information value | Col Ave_all **5.88**, L2 Ave_all **4.66** (vs 0.13 / 0.53 with BEV) | one tiny run |
| **D6** | **A metric that can see off-road** | `2312.03031` §4.2 (CCR); `paradrive-cvpr2024` (Offroad/Offlane) | plan vs road boundary / lane assignment | reverses the ranking: Ego-MLP CCR **2.93** vs UniAD 1.59; Offroad **1.21 vs 0.12** | ⛔ **needs a map — PhysicalAI has none** (§9.2) |
| **D7** | **Manoeuvre-stratified everything** | `paradrive-cvpr2024` ("targeted", 686 frames); `2312.03031` Tabs. 4–5 | the same metrics on the non-straight subset | blind-vs-camera collision **0.94 vs 0.14 (6.7×)** on targeted, versus indistinguishable on val | **0 GPU — we have the labels** |
| **D8** | **Open-loop / closed-loop dissociation** | `2306.07962`; `2309.10443`; `2506.04218` | the same arms on both tiers; look for the INVERSE relation | PDM-Open (centerline+ego) **OLS 86 (SOTA) / CLS-R 51**; IDM **OLS 38 / CLS-R 77**; PlanTF SDE **−1.48 OLS / +5.80 R-CLS** | needs our T1 harness |

⚠️ **D6 has its own blind spot, from the primary:** *"When the input velocity is zero, the
model produces almost stationary trajectories, resulting in excellent performance in the
CCR."* A stationary plan scores perfectly on a road-departure metric. **No single metric
detects the echo; the RATIO between two of them does.**

⚠️ **The aggregation trap, measured** (`2312.03031` App. Tab. 4): with cameras blanked and no
ego status, aggregate CCR moved **3.82 → 3.81** — apparently nothing — while **CCR-ST went
9.13 → 17.6 and CCR-LR went 3.05 → 1.69**, because the model retreated to driving straight.
**The aggregate hid a total behavioural collapse.** Stratify by manoeuvre (D7) or the
instrument is blind. This is the four-metric-families rule with an independent proof.

⭐ **And the sharpest statement that ADE cannot do this job**, CADET (lib `2606.14438`):
suppressing the spurious agents it identifies *"moves the official open-loop metrics by
**under 0.1 m**, which shows that **displacement error does not register a change in causal
reliance**."* Its **CRI** — *the plan must respond in the RIGHT DIRECTION to a causal
change* — is the transferable idea, and it is stronger than a magnitude-only sensitivity
test. ⚠️ **CADET does not probe ego speed/accel/yaw**; its PCR is per-*agent-query*. Adopt
CRI's form, not its numbers.
⚠️ **CADET also independently rediscovers our own probe rule**: *"the natural stability probe
is degenerate: it is satisfied by construction whenever it selects its targets with the same
score the repair masks by."* Same family as the `CLAUDE.md` rule that a probe tuning on the
data it scores manufactures a result.

### 7.4 ⛔ The one candidate the evidence tells us NOT to adopt as designed

**"Predict the residual over a constant-velocity / constant-curvature extrapolation rather
than the absolute path."**

- **NOT FOUND as a published driving method** (two differently-worded searches; no driving
  planner regresses residuals over an explicit kinematic prior with an ablation).
- **The nearest measurement points the other way, in our own architecture class**:
  DiffusionDrive Table 8 — learned anchors **88.1**, trained-on-extrapolated **84.7**,
  extrapolated-at-inference **81.3** (§3).
- **The residual that DID work is a different object**: `2207.09705` puts the residual on
  the **auxiliary target of a memory branch** behind a stop-gradient, and its own ablation
  has the residual target beating the absolute target **52.0 vs 41.3**.

⇒ **Use the constant-velocity / constant-yaw-rate extrapolation as a REFERENCE, not a
parameterisation.** Two admissible uses: (a) a **CV-excess metric** — report every family as
an improvement over the CV agent on the same windows, which is NAVSIM's own calibration
practice; (b) as **one anchor among 128**, never as the vocabulary. And **never publish a
headline number without the CV agent's score on the same windows beside it.**

---

## 8. Q9 — THE VERDICT for refcv4 (128 anchors × 8 slots, monocular 256×640 cylindrical, consuming `v`, `a_long`, `yaw_rate` at t0)

⭐ **Read this first — our intended topology is the published one, and it is the SAFE
placement.** VADv2, Hydra-MDP and PARA-Drive all combine plan/anchor embeddings with ego
status **at or just before the cross-attention to scene features**, which is exactly what
refcv4 will do through the measurement encoder → decoder condition. The measured harm
(TransFuser Table 10, **−11.33 DS**) is at a *different* site — inside the backbone — and our
trunk has never seen `v0`. **The design is not the risk. The corpus and the metric are.**

### 8.1 Ranked methods

| rank | method | why it ranks here | cost | evidence |
|---|---|---|---|---|
| **1** | **Per-scalar ego dropout + a presence bit per channel** (§6.1) | three independent adoptions (PlanTF, PLUTO, DRAMA), a published **rate sweep**, one of them on **NAVSIM PDMS**; and **half of it is already computed in our code** (`keep`) and merely not fed | ~10⁴ params, **0 inference cost** | ⭐⭐⭐ |
| **2** | **The detection panel D1+D2+D3+D4+D5b+D7 as a standing instrument** (§7) | ⛔ **without it no method in this table can be shown to have worked**, and five of the six cost no training compute. This is the *precondition*, not a competitor | ~0 GPU | ⭐⭐⭐ |
| **3** | **CV-triviality census + a manoeuvre-stratified hard split** (§6.6) | NAVSIM moved the CV agent **79 → 22 PDMS by filtering alone** — larger than any architectural fix here; PARA-Drive achieved the same separation with **labels only**. It also tells us whether our val split can *show* an echo at all | 0 GPU | ⭐⭐⭐ |
| 4 | **Keep the trunk ego-free; re-scope E11 from prohibition to boundary** (§6.2, §5.1) | the literature's structural line is at the scene encoder, and **we already comply**; the only closed-loop harm measurement is at that site | 0 | ⭐⭐⭐ (already held) |
| 5 | **Scene-grounding aux** (§6.5) — TCP's image→speed head, and an agent-detection head off `obstacle.offline` | +17.4 PDMS in PRIX; but SSR and BEV-Planner+Map are direct counter-evidence and the benefit **concentrates in the ~13 % of turning windows** | 1 head + labels | ⭐⭐ conflicted |
| 6 | **Copycat-style residual auxiliary branch with stop-gradient** (§6.3) | the best mechanism ablation in the set (52.0 / 47.0 / 41.3; 45.0 without stop-grad) but transferred from action-history to ego-kinematics — **an analogy, not a replication** | a small branch | ⭐⭐ analogical |
| 7 | **Adversarial / GRL stripping of ego from the visual feature** (§6.4) | ⛔ **no driving precedent found**; high risk; fights a design we want | a head + a λ ramp | ⭐ research bet |
| **✗** | **Residual-over-CV as the output parameterisation** (§7.4) | **contrary evidence in our own architecture class (−3.4 to −6.8 PDMS)** | — | ⛔ **refused as designed** |

### 8.2 The three cheapest discriminating experiments

**⭐ E-ECHO-2 — the CV-triviality + manoeuvre census of our own val split. 0 GPU, CPU only. RUN THIS FIRST; it gates the other two.**
- **What:** score all 881 canonical val windows (and a train sample) with a
  **constant-velocity + constant-yaw-rate** agent seeded from the measured `(v, yaw_rate)`
  at t0. Report the distribution of its per-window error, the **fraction of windows it
  already solves** at each of our thresholds, and the same stratified by our existing
  manoeuvre labels. Then construct a **`hard` split** two ways — NAVSIM's (drop windows the
  CV agent solves) and PARA-Drive's (drop `keep-forward`/straight-command windows) — and
  report their overlap.
- **The read:** NAVSIM's threshold in our units. **If ≳80 % of windows are CV-trivial, every
  number the programme has published on this split is dominated by the trivial subset**, and
  the first deliverable is a filtered split, not a loss term.
- **Discriminates:** "our corpus can show an echo" vs "it cannot" — which decides whether
  E-ECHO-3 is even interpretable. **Cheapest of the three; also produces the CV-excess
  reference every future table needs (§7.4).**

**⭐ E-ECHO-1 — the perturbation-response panel on the LIVE refcv3 checkpoint. 0 GPU-training.**
- **What:** on the canonical 40-episode / 881-window val, score the existing arm under
  (a) unmodified, (b) **blank image**, (c) **v × {0.0, 0.5, 1.5}**, (d) **v = 100 m/s**,
  (e) yaw/accel perturbations once those channels exist. Report **all four metric families**
  per condition and **stratified by manoeuvre (D7)**, plus **D3**
  (`‖∂(x_T,y_T)/∂s_0‖`, one backward pass).
- **The read:** the ratio `Δ(blank image) / Δ(v × 0.5)`. Published anchors: an echoing model
  reads **1.24 / 8.6 ≈ 0.14**; the same architecture without ego reads **3.46**.
- **Controls (mandatory, and all published):** a **command-only / constant-only** arm that
  must read the no-information value (PARA-Drive Tab. 4 is the precedent), and a **CV floor**
  that must be *maximally* sensitive to `v` and *completely* insensitive to the image. **If
  refcv3's ratio sits at the CV floor's, the verdict is decided in one run.**
- ⚠️ **Do not run this on `tanitad-refcv3` — it is training.**

**⭐ E-ECHO-3 — the 4-arm ego-input panel on the v7-tiny ladder. 4 tiny runs.**
- **Arms** (one variable each, both outcomes pre-registered):
  - **A — no ego at all.** The `state3` analogue and the honest upper bound on "vision only".
  - **B — ego, `p = 0`, no presence bit.** *(today's shipped config with dropout off — the
    **deliberate-regression arm**: it should look BEST on ADE and WORST on D1–D4.)*
  - **C — ego, per-channel dropout `p ∈ {0.5, 0.75}`, presence bit ON.** The candidate.
  - **D — ego-only floor, no image** (D5), with **D5b** (command-only) as the constant control.
    **C must beat D, or refcv4's vision has added nothing.**
- **Scored on:** the four metric families **and** the E-ECHO-1 ratio, **manoeuvre-stratified**
  — ⛔ **explicitly NOT on ADE as the gate.** PlanTF measured **OLS −1.48 while R-CLS +5.80**
  for exactly this mechanism, and CADET measured that displacement moves **< 0.1 m** for a
  real change in causal reliance. **A displacement gate will reject the working guard.**
- **Both outcomes committed:** if **C ≈ B** on the echo instruments, the guard is inert on
  our corpus and the binding constraint is E-ECHO-2's corpus finding; if **C is worse than B
  on ADE and better on the echo ratio**, that is the published signature and the gate must
  be re-specified before any flagship compute is spent.

---

## 9. Corrections, gaps, and things that did not check out

1. ⚠️ **`refc.py`'s docstring implies TCP gave us the ego-dropout. It did not** (two probes,
   §2). **Documentation fix required** — it launders an unswept hyper-parameter as inherited
   practice, the same root-cause class as the registry's un-refined-anchor correction.
2. ⛔ **D6 (road-adherence) is NOT available to us.** PhysicalAI-AV has **no map, no lane
   graph, no road-boundary annotation** — the card says verbatim *"we do not include open
   maps data"*, settled at five independent probes and pinned by
   `stack/tests/test_physicalai_feature_readset.py`. Any CCR/Offroad/Offlane metric must come
   from AlpaSim/NuRec (`map.xodr` exists there) or an external corpus. **This is a real
   instrument gap on the exact axis the literature says matters most** — and it is precisely
   why D1/D2/D7, which need no map, are the recommended first instruments.
3. ⚠️ **`2012.05329` is NOT "object-aware regularization"** — it is *"Know Your Limits:
   Uncertainty Estimation with ReLU Classifiers Fails at Reliable OOD Detection"*. The
   object-aware regularization paper is **`2110.14118` (OREO)**. Correcting an ID from the
   brief so it does not propagate.
4. ⚠️ **A figure of "Ego Status EPDMS 64.0 on NAVSIM v2 navhard" is in circulation and is
   wrong.** The primary (lib `2506.04218` Tab. 2) says **14.1**; the authors explicitly
   discourage EPDMS reported on the v1 navtest split without two-stage pseudo-simulation.
5. ⚠️ **NAVSIM never uses the word "misleading"** about nuScenes (0 hits). Quote instead:
   *"most frames have a trivial solution of extrapolating the historical driving behavior"*;
   *"leaking ground-truth information into inputs"*; *"about 75 % … trivial straight
   driving"*; *"displacement metrics are not correlated to closed-loop driving"*.
6. **ModDrop's per-channel drop rate — NOT FOUND** (arXiv abs gives the mechanism, TPAMI is
   paywalled). The paper is banked; the rate is not quotable.
7. **PARA-Drive per-component ego ablation (velocity vs acceleration vs yaw separately) —
   NOT FOUND**; its Table 6 bundles CAN bus + history into one Yes/No column. **NAVSIM
   Table 2 B1/B2 is the only per-component split located anywhere.**
8. **VAD's exact `ego_lcf_feat` components — NOT RESOLVED at three probes.** It is written
   by `vad_nuscenes_converter.py:532`. Worth one more read, since it is the closest published
   analogue of our `(v, a_long, yaw_rate)` triple. **Open work item.**
9. ⚠️ **`2605.00066` measured the ADE↔closed-loop correlation at ρ = −0.36, p = 0.43,
   n = 7** — **not significant.** Quote as *"no meaningful correlation at n = 7"*, never as
   *"ADE anti-correlates with closed-loop"*. Flagged because the temptation to over-quote it
   in exactly this argument is high.
10. **`1905.11979` measured that plain dropout barely helps** on its confounded benchmarks,
    in tension with §6.1's driving evidence. **Both primary. Run the arm; do not adjudicate
    in prose.**
11. **DiffusionDrive uses 20 anchors; we use 128.** Its anchor numbers are its own and must
    not be transferred without re-measurement (the `df`-scope class).
12. **Two papers surfaced but NOT read, and are NOT cited here:** `2605.13646`
    (Causality-Aware E2E via Ego-Centric Joint Scene Modeling — downloaded, 0 hits for
    "perturb"/"shortcut", so it is not an ego-shortcut measurement paper) and `2606.31106`
    (*What Probing Reveals about Autonomous Driving*).

---

## 10. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `SPEC.md` | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refc-ego-inputs-and-anti-echo/SPEC.md` | no — staged |
| `RESULT.md` (this file) | `repo: …/2026-09-03-refc-ego-inputs-and-anti-echo/RESULT.md` | no — staged |
| `COMMS.md` | `repo: …/2026-09-03-refc-ego-inputs-and-anti-echo/COMMS.md` | no — staged |
| `raw/PRIMARY_EXTRACTS.md` (verbatim tables + provenance) | `repo: …/2026-09-03-refc-ego-inputs-and-anti-echo/raw/PRIMARY_EXTRACTS.md` | no — staged |
| `raw/stage_verified.py` — **a tool this WP had to write** (see below) | `repo: …/2026-09-03-refc-ego-inputs-and-anti-echo/raw/stage_verified.py` | no — staged |
| **15 newly banked primary PDFs** | `repo: TanitAD Research Lab/Library/papers/` + `library.json` + regenerated `LIBRARY.md` | no — staged |
| KB findings | `repo: …/Architecture & Inference/Research/KNOWLEDGE_BASE.md` (appended) | no — staged |

**Newly banked by this WP** (all `--tag refc-ego`, all `--cited-by` this RESULT.md):
`2309.10443` PlanTF · `2408.03601` DRAMA · `2606.14438` CADET · `2511.13079` AdaptiveAD ·
`1606.01865` GRU-D · `2404.14327` PLUTO · `2409.18341` SSR · `1505.07818` DANN ·
`2110.14118` OREO · `2106.06452` Keyframe-Focused · `2603.18561` CausalVAD ·
`2407.06546` Causality of E2E AD · `1501.00102` ModDrop · **`paradrive-cvpr2024` PARA-Drive
(local bank of the CVF PDF — this WP converted it from PUBLISHED-SECONDARY to PRIMARY)** ·
(+ `2010.14876` re-tagged).

⛔ **A TOOL THIS WP HAD TO WRITE, AND WHY IT IS NOT OPTIONAL.** Staging this deliverable
uncovered that **`git add` on this mount stages NUL-filled blocks with exit 0, reproducibly**
— `library.json` went into the index with **147,456 NUL bytes** while Python read the same
file with zero NULs and full JSON validity, and `git hash-object` on a fresh read returned
the *identical corrupted sha*. **Three of six text files and one of fourteen PDFs were hit.**
⚠️ **The corruption preserves the byte count**, so a size check and a marker grep both
*passed* on a blob that was already damaged. `raw/stage_verified.py` compares a full
**sha256 of `git cat-file blob :<path>`** against the worktree bytes and repairs through
`git hash-object -w --stdin` + `git update-index --cacheinfo`, which never lets git read the
file. **Every path staged by this WP was verified this way; the full detail is in
`COMMS.md` §5.2.**

**Already banked, re-read primary for this WP:** `2206.08129` TCP · `2411.15139`
DiffusionDrive · `2312.03031` BEV-Planner · `2305.10430` AD-MLP · `2406.15349` NAVSIM ·
`2506.04218` NAVSIM v2 / Pseudo-Simulation · `2205.15997` TransFuser · `2212.10156` UniAD ·
`2303.12077` VAD · `2402.13243` VADv2 · `2406.06978` Hydra-MDP · `2306.07962` Parting with
Misconceptions · `1905.11979` Causal Confusion · `1904.08980` Codevilla · `2306.07957`
Hidden Biases · `2507.17596` PRIX · `2207.09705` Residual Action Prediction · `2605.00066`
open/closed-loop correlation.
