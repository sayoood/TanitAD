# raw/ — PRIMARY EXTRACTS: provenance for every number in `RESULT.md`

*This is the quotable layer. Each block names the **library key**, the **exact section or
table**, and the extraction method. Every PDF listed was opened from
`TanitAD Research Lab/Library/papers/` and its **sha256 re-checked against `library.json`
before the text was read** — presence proved transfer, the sha proved bytes.*

**Extraction method (identical for all):** `PyMuPDF (fitz)` full-document `get_text()`,
no OCR. Table values below are transcribed from that text; PDF table extraction linearises
columns, so **every table transcribed here was cross-read against the surrounding prose**,
and any row where the prose and the linearisation disagreed is flagged.

**Re-derivation command (0 GPU, any box):**

```bash
python - <<'PY'
import json, hashlib, fitz
d = json.load(open('TanitAD Research Lab/Library/library.json', encoding='utf-8'))
v = d['entries']['2312.03031']                      # any key below
b = open(v['path'], 'rb').read()
assert hashlib.sha256(b).hexdigest() == v['sha256'] # bytes, not presence
print("\n".join(p.get_text() for p in fitz.open(stream=b, filetype='pdf')))
PY
```

---

## 1. `2206.08129` — TCP (NeurIPS 2022) — sha verified

| claim in RESULT.md | location | verbatim / value |
|---|---|---|
| ego inputs = speed + nav cmd + nav target coords | §3.1 "Problem formulation" | *"Given the state x comprised of the sensor signal i, the speed of the vehicle v, and the high level navigation information g including a discrete navigation command and the coordinates of navigation target provided by the global planner…"* |
| where the speed enters | §3.2 "Overview" | *"the navigation information g is concatenated with the current speed v to form the measurement input m, then an MLP based measurement encoder takes m as its input and outputs the measurement feature j_m"* |
| it reaches the TRAJECTORY branch | §3.2.1 | *"the image feature map F is average pooled and concatenated with the measurement feature j_m to form j_traj. … we feed j_traj into a GRU"* |
| no ego history | §3.2.2 | *"it is difficult to predict future control actions since we only have sensor inputs at the current time step"* |
| measurement encoder = FC 128 ReLU ×2 | Table 5 "Detailed network structure" | `Measurement Encoder | FC | 128 | ReLU | ×2` |
| image→speed auxiliary head, λ = 0.001 | §"Auxiliary heads" + §"hyper-parameters" | *"we add a speed prediction head to predict current speed s from the image feature"*; *"0.001 for speed and value regression"* |
| the inertia problem is named, blamed on the controller | §1 | *"Simple PID controllers may perform worse in situations such as taking a big turn or starting at the red light due to the inertial problem of end-to-end models [29]"* |
| ⛔ **NO dropout anywhere** — 2 probes | (a) full-text `grep -i "dropout\|drop out\|mask\| drop "` → **0 hits**; (b) Table 5 lists the measurement encoder with no regulariser | — |

## 2. `2411.15139` — DiffusionDrive (CVPR 2025) — sha verified

| claim | location | verbatim / value |
|---|---|---|
| decoder composition, no ego port | §3.4 "Architecture" | *"deformable spatial cross-attention to interact with BEV or PV features … cross-attention … between the trajectory features and the agent/map queries … a Timestep Modulation layer, which is followed by a MLP that predicts the confidence score and the offset"* |
| ego status comes from the host | §4.2 + App. A | *"We adopt the same perception modules and ResNet-34 backbone as Transfuser"*; *"The training and inference recipe directly follows Transfuser"*; nuScenes: *"We follow the SparseDrive baseline"* |
| 20 anchors, 8 waypoints / 4 s, 2 denoise steps, 60 M params | §4.2 + Tab. 2 | *"truncated diffusion policy with 20 clustered anchors"*; *"8-waypoint trajectory over 4 seconds"* |
| ⛔ ego status never documented — 2 probes | (a) `grep -ai "ego status\|ego state\|velocity\|acceleration\|driving command"` over the body → only 1 hit, in the **bibliography** (the BEV-Planner citation); (b) Tab. 3's "Ego Query" column is ✓ in **all six rows**, i.e. never ablated | — |
| **Table 8 (App. B) — driving priors, NAVSIM navtest** | App. B | Anchored/Anchored **NC 98.2 · DAC 96.2 · TTC 94.7 · Comf 100 · EP 82.2 · PDMS 88.1**; Anchored/Extrapolated **96.3 · 91.7 · 90.4 · 100 · 76.8 · 81.3**; Extrapolated/Extrapolated **97.3 · 94.0 · 92.6 · 100 · 79.6 · 84.7** |
| their reading of Table 8 | App. B | *"the superiority of the proposed anchored Gaussian distribution over extrapolated prior, which fails to cover the potential action space and can not effectively handle challenging scenarios (e.g., obstacle avoidance and turning)"* |

## 3. `2312.03031` — BEV-Planner / "Is Ego Status All You Need?" (CVPR 2024) — sha verified

*(sha256 `c289b890…e7899`, 9,245,010 B — independently re-checked against a second copy
fetched by a different route in this session; byte-identical.)*

**Table 1 (nuScenes val) — transcribed rows used in RESULT.md §4.2.** Format
`L2 1s/2s/3s/Avg · Coll 1s/2s/3s/Avg · CCR 1s/2s/3s/Avg`:

```
ID-1  UniAD        ✗/✗   0.59 1.01 1.48 1.03 | 0.16 0.51 1.64 0.77 | 0.35 1.46 3.99 1.93
ID-2  UniAD        ✓/✗   0.35 0.63 0.99 0.66 | 0.16 0.43 1.27 0.62 | 0.21 1.32 3.63 1.72
ID-3  UniAD        ✓/✓   0.20 0.42 0.75 0.46 | 0.02 0.25 0.84 0.37 | 0.20 1.33 3.24 1.59
ID-4  VAD-Base     ✗/✗   0.69 1.22 1.83 1.25 | 0.06 0.68 2.52 1.09 | 1.02 3.44 7.00 3.82
ID-5  VAD-Base     ✓/✗   0.41 0.70 1.06 0.72 | 0.04 0.43 1.15 0.54 | 0.60 2.38 5.18 2.72
ID-6  VAD-Base     ✓/✓   0.17 0.34 0.60 0.37 | 0.04 0.27 0.67 0.33 | 0.21 2.13 5.06 2.47
ID-7  GoStraight    —/✓  0.38 0.79 1.33 0.83 | 0.15 0.60 2.50 1.08 | 2.07 8.09 15.7 8.62
ID-8  Ego-MLP       —/✓  0.15 0.32 0.59 0.35 | 0.00 0.27 0.85 0.37 | 0.27 2.52 6.60 2.93
ID-9  BEV-Planner* ✗/✗   0.27 0.54 0.90 0.57 | 0.04 0.35 1.80 0.73 | 0.63 3.38 7.93 3.98
ID-10 BEV-Planner  ✗/✗   0.30 0.52 0.83 0.55 | 0.10 0.37 1.30 0.59 | 0.78 3.79 8.22 4.26
ID-11 BEV-Planner+ ✓/✗   0.28 0.42 0.68 0.46 | 0.04 0.37 1.07 0.49 | 0.70 3.77 8.15 4.21
ID-12 BEV-Planner++✓/✓   0.16 0.32 0.57 0.35 | 0.00 0.29 0.73 0.34 | 0.35 2.62 6.51 3.16
```

Caption, verbatim (the Ego-MLP input spec): *"Without the perception module, Ego-MLP (ID-8),
utilizing solely **ego velocity, acceleration, yaw angle, and driving command**, achieves
performance on par with current state-of-the-art models on previous L2 distance and
collision rate metrics."*

**Table 2 (VAD-Base robustness) — L2 avg / Coll avg / CCR avg / Det NDS / Map mAP:**

```
VAD-Base*(no ego in planner) —        0.72 | 0.54 | 2.72 | 46.0 | 47.5
VAD-Base (ego in planner)    —        0.37 | 0.33 | 2.47 | 45.5 | 47.0
VAD-Base  Snow                        0.45 | 0.32 | 2.82 | 36.1 | 29.4
VAD-Base  Fog                         0.45 | 0.30 | 2.78 | 34.3 | 29.4
VAD-Base  Glare                       0.44 | 0.26 | 2.63 | 41.7 | 38.3
VAD-Base  Rain                        0.45 | 0.29 | 2.89 | 29.1 | 13.0
VAD-Base  Blank                       0.46 | 0.54 | 3.71 |  0.0 |  0.0
VAD-Base  v x 0.0                     6.16 | 7.98 | 1.23 | 45.5 | 47.0
VAD-Base  v x 0.5                     3.19 | 1.71 | 2.83 | 45.5 | 47.0
VAD-Base  v x 1.5                     3.20 | 3.08 | 7.01 | 45.5 | 47.0
VAD-Base  v = 100 m/s                  208 | 9.38†| 27.0†| 45.5 | 47.0
```
† the paper's own dagger: *"The collision rate is not precise as the ego car may have
departed from the local BEV area."*

**App. Table 3 (dropping cameras, the control that makes D1 a metric):**
```
VAD-Base  —      ego ✓   L2 avg 0.37 | Coll 0.33 | Int. 2.47 | NDS 45.5 | mAP 47.0
VAD-Base  Blank  ego ✓   L2 avg 0.46 | Coll 0.54 | Int. 3.71 | NDS  0.0 | mAP  0.0
VAD-Base  —      ego ✗   L2 avg 1.25 | Coll 1.09 | Int. 3.82 | NDS 45.1 | mAP 53.7
VAD-Base  Blank  ego ✗   L2 avg 4.33 | Coll 7.63 | Int. 3.81 | NDS  0.0 | mAP  0.0
```
⚠️ **Transcription note.** The `ego ✗ / no corruption` row linearises with a column shift in
the raw text (`0.69 1.22 1.83 0.06 0.68 2.52 0.84 0.37 …`). Its **L2 avg is taken as 1.25
from Table 1 ID-4**, which carries the identical 1s/2s/3s triple `0.69 / 1.22 / 1.83`. This
is the only value in the WP recovered by cross-table agreement rather than direct read, and
it is flagged here because §0 line 8 depends on it.

**App. Table 4 (the aggregation trap):** ego ✗, no corruption — CCR avg **3.82**, CCR-ST
**9.13**, CCR-LR **3.05**; ego ✗, blank — CCR avg **3.81**, CCR-ST **17.6**, CCR-LR **1.69**.

**Table 3 (map aux):** BEV-Planner 0.55 / 0.59 / 4.26 → BEV-Planner+Map **0.96 / 0.89 /
2.60**. **Tables 4–5 (stratified):** L2-ST 0.48 → 0.97, L2-LR 0.81 → 0.89; Collision-ST
0.40 → 0.91, **Collision-LR 2.25 → 0.78**.

**Corpus properties:** *"73.9 % of the nuScenes data involve scenarios of driving
straightforwardly"*; *"the relatively small proportion (13 %) of turning scenes"*;
*"straight-driving scenarios (87 % of all evaluation samples)"*.

**App. C (ego in the BEV encoder):** *"For BEVFormer, it involves projecting the ego status
onto the hidden features and incorporating it into the BEV query … the incorporation of ego
status within the perception module is often overlooked"*; the perception cost is nil —
BEVFormer with ego **mAP 41.6 / NDS 51.7**, without **41.3 / 51.5**. Reproduction switch:
*"we set the `use_can_bus` flag to False"*.

**Fig. 5 / Fig. 6 readings:** *"Introducing ego status in the BEV-Planner++ enables the model
to converge very rapidly"*; *"in BEV-Planner++, the activation range of the feature map
predominantly encompasses the immediate vicinity around the ego vehicle, frequently
manifesting behind the vehicle itself … the BEV-Planner++ method has almost not learned any
effective information."*

**Their stated position (App. C):** *"our position is not opposed to the use of ego status;
rather, we argue that within the context of current datasets and evaluation metrics, the
integration of ego status can significantly impact, and even determine, the planning
results."*

## 4. `paradrive-cvpr2024` — PARA-Drive (CVPR 2024, pp. 15449–15458) — LOCAL BANK

*Source: `https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.pdf`
· 1,093,903 B · sha256 `a484b25a148de08da9ae111b0ed66a119ce1acfd3d7a0f69d7cbc6a072c50658` · 10 pp.
⚠️ **CVF returns HTTP 403 to a plain fetch; a browser User-Agent is required.** No arXiv
mirror; the GitHub repo holds only the project page. Banked with `kb_add.py --local`, so it
is PRIMARY.*

| claim | location | verbatim / value |
|---|---|---|
| ego inputs | §4 | *"This BEV feature map, in conjunction with the data from the ego vehicle (e.g., **high-level commands, CAN bus, history trajectories**), forms the exclusive input to the planning head."* |
| CAN bus contents | §5 "Using Ego Vehicle's States" | *"CAN bus (**velocity, acceleration, angular velocity**, etc.)"* |
| planner-head-only injection | §4 + Fig. 5 | ego information joins the **plan queries** at the cross-attention with BEV; **no ego in the BEV encoder** |
| their conclusion | §5 | *"the improvements brought by the CAN bus and history trajectories become **marginal** in the val set"*; *"in the targeted scenarios as well as the map compliance error rates, AD-MLP has significantly worse performance than PARA-Drive. This suggests that the open-loop evaluation scheme is still very informative"* |
| **the "targeted" split** | §5 (Evaluation) | *"For targeted scenario evaluation, we **exclude frames with a command of 'keep forward'**, which results in a total of **686 challenging key-frames** on the nuScenes val set."* |
| AD-MLP has a leak | footnote 5 | the released checkpoint *"is trained with GT data leakage"* — re-implemented by the authors |
| ⛔ **no ego dropout** — 2 probes | `dropout` → **0 hits**; `drop out` → **0 hits** over the full text. **No code exists to check.** | — |

**Table 6 (val + targeted), the four columns used:**
```
val       UniAD       ego No   Coll_all 0.40 | L2_all 0.8317 | Offroad 0.91 | Offlane 1.74
val       VAD         ego No   Coll_all 0.30 | L2_all 0.7830 | Offroad 1.03 | Offlane 1.93
val       PARA-Drive  ego No   Coll_all 0.17 | L2_all 0.5574 | Offroad 0.12 | Offlane 0.83
val       AD-MLP      ego Yes  Coll_all 0.20 | L2_all 0.5568 | Offroad 1.21 | Offlane 2.45
val       PARA-Drive+ ego Yes  Coll_all 0.13 | L2_all 0.4939 | Offroad 0.11 | Offlane 0.78
targeted  PARA-Drive  ego No   Coll_all 0.14 | L2_all 0.9082 | Coll@3s 0.72
targeted  AD-MLP      ego Yes  Coll_all 0.94 | L2_all 0.9360 | Coll@3s 3.62
```
**Table 4 (command-only floor):** planner receives only the high-level command — Coll_all
**5.88**, L2_all **4.66**; with BEV **0.13 / 0.53**.

⚠️ **All ratios in RESULT.md §4.3 (10.1× / 3.0× / 6.7× / 5.0×) are MY arithmetic on these
PUBLISHED values, not quoted from the paper.**

## 5. `2406.15349` — NAVSIM (NeurIPS 2024 D&B) — sha verified

| claim | location | verbatim / value |
|---|---|---|
| ego status definition | §3 "Task description" | *"the vehicle's **current speed, acceleration, and navigation goal**, jointly termed the ego status … we provide the navigation goal as a one-hot vector with three categories: left, straight, or right"* |
| goal is leak-free by construction | §2 | *"We derive a navigation goal from the lane graph instead of the human trajectory to prevent label leakage"* |
| nuScenes triviality | §2 | *"about 75 % of the scenarios in nuScenes involve trivial straight driving"*; *"most frames have a trivial solution of extrapolating the ego-motion"* |
| **the CV filter** | §3.1 | *"the baseline of maintaining a constant velocity and heading achieves a PDMS of 79 % on the OpenScene dataset, where human-level performance corresponds to 91 %. … We remove highly simplistic scenes by detecting if the constant velocity agent exceeds a PDMS of 0.8. Similarly, we remove scenes in which the human trajectory results in a PDMS of less than 0.8. … the score of the constant velocity agent **dropping to 22 %**, whereas the human expert achieves a score of 95 %."* Splits: navtrain **103k**, navtest **12k** |
| **Table 1 (navtest, PDMS)** | §4 | Constant Velocity **20.6** (NC 68.0 DAC 57.8 TTC 50.0 Comf 100 EP 19.4) · Ego Status MLP **65.6** (93.0 / 77.3 / 83.6 / 100 / 62.8) · LTF 83.8 · TransFuser **84.0** · UniAD 83.4 · PARA-Drive 84.0 · Human 94.8 |
| Ego Status MLP inputs | §4 "Methods" | *"an MLP for trajectory prediction given only the ego velocity, acceleration and navigation goal"* |
| **Table 2 (TransFuser ablation)** | §4 | A1/A2/A3 seeds 83.3 / 84.0 / 84.4 (**σ ± 0.56**); **B1 "Goal only" 81.8**; B2 "Goal and velocity only" 82.3. Verdict: *"Discarding velocity and acceleration (B1) lowers PDMS by 1.5 − 2.6, whereas only removing the acceleration (B2) lowers the score by 1.0 − 2.1. We conclude that while TransFuser benefits from the ego status, it is not purely relying on the kinematic state for planning."* |
| aux tasks are load-bearing | §4 (config E1) | *"We check the impact of the auxiliary tasks by excluding them, where performance drops without BEV Segmentation (E1)."* |
| **Table 3 (leaderboard 1.1)** | §4 | TransFuser 83.9 ± 0.4 · LTF 83.5 ± 0.6 · **Ego Status MLP 66.4 ± 0.9** · Hydra-MDP 91.3 · CV 20.6 |
| ⚠️ the word "misleading" | full-text grep | **0 hits.** Use the four statements above instead. |

## 6. `2506.04218` — NAVSIM v2 / "Pseudo-Simulation for Autonomous Driving" (CoRL 2025) — sha verified

| claim | location | verbatim / value |
|---|---|---|
| it IS NAVSIM v2 | §1 | *"To enable standardized benchmarking, we release NAVSIM v2"* |
| v2 inputs (history now in scope) | §3 | *"The inputs include multi-view camera images and **ego status features such as the velocity and motion history**."* |
| navhard size | §4 | *"450 Stage 1 and 5462 Stage 2 observations"* |
| open-loop misses causal confusion | §2 | *"NAVSIM v1 remains limited to open-loop evaluation … and does not account for compounding errors or causal confusion"* |
| **Table 2 (navhard leaderboard, 03/2026, EPDMS)** | §4.2 | CV **11.4** · **Ego MLP 14.1** · LTF 25.1 · LTFv6 31.9 · NavFormer 34.1 · RAP 39.6 · ZTRS 48.1 · GuideFlow 51.5 · SimScale 53.2 · DrivoR 54.5 · PDM-C **56.6** |
| ⛔ against unofficial splits | §4.2 | *"we discourage the use of self-reported and unofficial 'NAVSIM v2' benchmark splits, such as reporting the EPDMS on the NAVSIM v1 navtest dataset without conducting two-stage pseudo-simulation."* |
| NavFormer copies PARA-Drive's aux recipe | §4.2 | *"Following PARA-Drive, object tracking and map segmentation decoders provide auxiliary perception supervision. Finally, NavFormer uses a Hydra-MDP decoder head."* |

## 7. `2205.15997` — TransFuser (PAMI) — sha verified

| claim | location | verbatim / value |
|---|---|---|
| default takes NO velocity | §4.11 "Inertia Problem" | *"though we do not use velocity as an input to our models, we observe that creeping in the controller increases the RC significantly"* |
| the inertia problem's cause | §4.11 | *"typically attributed to the **spurious correlation that exists between input velocity and output acceleration** in an IL dataset"* (cites [38] = Codevilla, lib `1904.08980`) |
| **the injection site** | §4.11 | *"we provide the current velocity as input by projecting the scalar value into the same dimensions as the transformer positional embedding using a linear layer. This velocity embedding is combined with the learnable positional embedding through element-wise summation and **fed into the transformer at all 4 stages of the backbone**."* |
| **Table 10 (Longest6, mean of 3 evals)** | §4.11 | velocity ✗ / creep ✗ → **DS 46.35** RC 78.28 IS 0.63 · velocity ✗ / creep ✓ → **DS 56.68** RC 92.28 IS 0.62 · velocity ✓ / creep ✗ → **DS 37.34** RC 64.27 IS 0.65 · velocity ✓ / creep ✓ → **DS 45.35** RC 86.22 IS 0.52 |
| the verdict | §4.11 | *"Including the velocity input leads to a **sharp drop in DS, which cannot be recovered** through the creeping behavior."* |

## 8. `2309.10443` — PlanTF (ICRA 2024) — sha verified

| claim | location | verbatim / value |
|---|---|---|
| the SDE mechanism | §IV-A + Fig. 3 | *"Each state variable undergoes individual embedding through a linear layer before being combined with positional encoding. A learnable query aggregates state embeddings through a cross-attention module. During training, **each embedded state (except position and heading) token will be dropped with a certain probability.**"* |
| history is the shortcut | §IV-A | *"models incorporating historical motion data exhibit superior off-policy evaluation performance (OLS), [but] manifest significantly poorer performance in closed-loop metrics … attributed to the well-established 'copycat' problem or learning shortcuts"* |
| **the D3 instrument** | §IV-A + Fig. 2(c)(d) | *"the magnitude of the gradient of the endpoint's position (X_T, Y_T) w.r.t. the initial kinematic states s_0. The results demonstrate that the model employing SDE is **less sensitive to variations in kinematic states**"* |
| **Table II** | §IV-A | state3 — 81.13/85.99/79.38, hard 71.43/68.44/63.14 · state5 ✗ 87.71/81.76/74.51, hard 84.54/68.67/54.91 · state5 ✓ 88.80/86.73/75.75, hard 84.29/71.28/61.88 · state6 ✗ 88.55/83.19/74.79, hard 85.89/67.57/58.99 · **state6 ✓ 87.07/86.48/80.59, hard 83.32/72.68/61.70** |
| **Table VIII (rate sweep, state6+SDE)** | App. | none 88.33/77.28/74.10 · 0.25 89.11/81.70/78.44 · 0.50 89.12/83.71/77.52 · **0.75 87.07/86.48/80.59** |
| deployed rate | App. (impl. details) | *"a state attention dropout encoder with a **dropout rate of 0.75**. During training, we apply state perturbation with a probability of 0.5."* |

## 9. `2408.03601` — DRAMA (2024) — sha verified

| claim | location | verbatim / value |
|---|---|---|
| the FSD mechanism | §3.3 | *"we implement feature state dropout for image feature fusion from two modalities and the ego status … the features to be encoded are added with a learnable positional embedding, followed by the differentiated dropout to mask some features"* |
| the differentiated policy | §3.3 | *"a differentiated dropout policy … applies distinct dropout rates to the fusion and ego status features. **A relatively low dropout rate is assigned for the fusion feature to preserve its integrity.**"* |
| **Table 2** | §4 | Transfuser 0/0 → **0.835** · +FSD 0/0.1 → **0.842** · +FSD **0.5**/0 → **0.844** · +FSD **0.5/0.1** → **0.848** |

## 10. Smaller extracts (verified, used once each)

| lib key | claim | verbatim / value |
|---|---|---|
| `2303.12077` VAD | ego is optional and in the planner head | *"the planning head takes the updated ego queries (Q'_ego, Q''_ego) and **the current status of the ego vehicle s_ego (optional)** as ego features f_ego, as well as the driving command c"* |
| `2303.12077` VAD | the paper-vs-code contradiction | *"in the main results, VAD **omits ego status features to avoid shortcut learning** in the open-loop planning [50], but the results of VAD using ego status features are still preserved in Tab. 1 for reference"* — while the **released official checkpoint** is the one `2312.03031` ID-6 scores at L2 0.37 |
| `2402.13243` VADv2 | ego in the conditioning | *"navigation information and ego state are also encoded into embeddings (E_navi, E_state) with an MLP"*; `o = (E_scene, E_navi, E_state)` |
| `2406.06978` Hydra-MDP | ⭐ ego ADDED to the anchor queries | `V'_k = Transformer(Q, K, V = Mlp(V_k)) + E` — *"added to the ego status E"* — **before** the environment cross-attention `V''_k = Transformer(Q = V'_k, K, V = F_env)` |
| `1606.01865` GRU-D | the "confident lie" argument | mean/forward imputation *"cannot distinguish whether missing values are imputed or truly observed"*; ablation (MIMIC-III AUC): masking only **0.8367**, interval only 0.8266, both + decay **0.8527**, GRU-mean 0.8192 |
| `2207.09705` Residual Action | the target ablation | memory objective `a_t` **41.3 ± 1.9** · `a_{t−1}` **47.0 ± 5.1** · **residual Δa_t 52.0 ± 2.3** · without stop-gradient **45.0 ± 5.3** (CARLA NoCrash Train-Dense) |
| `2010.14876` Copycat | the D4 detector | expert vs BC-OH action-predictability MSE: Ant **6.91e-2 vs 0.66e-2**, Walker2d 2.47e-2 vs 0.46e-2; after the fix Ant 0.66e-2 → **2.20e-2** |
| `1505.07818` DANN | the λ ramp | `λ_p = 2/(1 + exp(−γ·p)) − 1`, **γ = 10**, `p` = training progress 0→1 |
| `2507.17596` PRIX | the aux ladder | NAVSIM PDMS: planning loss only **70.4** → +det-box **82.3** → +semantic seg **85.7** → +det-cls **86.9** → full **87.8** |
| `2409.18341` SSR | the counter-evidence | nuScenes L2 avg **0.75** vs UniAD 1.03 / VAD-Base 1.22 with **no perception supervision at all**; its Table 4: adding map/obstacle supervision **worsens L2 to 0.81–0.86** |
| `2606.14438` CADET | ⭐ ADE cannot see this | suppressing the identified spurious agents *"moves the official open-loop metrics by **under 0.1 m**, which shows that displacement error does not register a change in causal reliance"*; **CRI** = the plan must respond in the **right direction** to a causal change |
| `2605.00066` | ⚠️ under-powered | ADE vs Bench2Drive Driving Score: **ρ = −0.36, p = 0.43, n = 7** — **not significant** |
| `1905.11979` Causal Confusion | dropout is not always enough | *"dropout baseline gives minimal improvement"* on the confounded Atari suite; GTA-V driving with history **0.834 perplexity / 2.94 collisions** vs without **0.989 / 1.30** |

---

## 11. Sources read but NOT admissible / NOT read

| item | status | why |
|---|---|---|
| VAD `ego_lcf_feat` component list | **UNRESOLVED** at three probes | written by `vad_nuscenes_converter.py:532`; the array construction was not read. **Open work item** — it is the closest published analogue of our `(v, a_long, yaw_rate)` triple |
| PARA-Drive per-component ego ablation | **NOT FOUND** | Table 6 bundles CAN bus + history into one Yes/No column |
| ModDrop per-channel drop rate | **NOT FOUND** | arXiv abs gives the mechanism; TPAMI is paywalled |
| "Ego Status EPDMS 64.0 on navhard" | ⛔ **REFUTED** | the primary (`2506.04218` Tab. 2) says **14.1**; the 64.0 figure is a v1-navtest computation the authors explicitly discourage |
| `2012.05329` as "object-aware regularization" | ⛔ **WRONG ID** | it is *"Know Your Limits: …OOD Detection"*. OREO is `2110.14118` |
| `2605.13646`, `2606.31106` | **NOT READ** | not cited anywhere in `RESULT.md` |
| a driving GRL head stripping ego kinematics | **NOT FOUND**, 2 phrasings | see `RESULT.md` §6.4 — a genuine gap |
| a driving planner regressing residuals over a kinematic prior, with an ablation | **NOT FOUND**, 2 phrasings | see `RESULT.md` §7.4 — and the nearest evidence contradicts it |
