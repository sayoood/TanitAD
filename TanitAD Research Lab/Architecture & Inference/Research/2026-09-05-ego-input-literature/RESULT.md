# RESULT — E-EGO-LIT-1: how successful planners handle ego-state input, ego dropout and speed-conditioned vocabularies — and what REF-C's origin paper (DiffusionDrive v1) actually does in code

*Architecture & Inference FlyWheel · Research Lab literature stream · 2026-09-05. **0 GPU.**
Research Lab agent; no sub-agents spawned (the predecessor's four died unbanked).*

`status: DONE 2026-09-05 04:40 (dev-box local clock) — 13 code bases read at pinned commits, 21 primaries
re-read from banked PDFs, GTRS 2506.06664 metadata completed + cited, cited-by recorded on 20
primaries; KB findings appended; register rows H-EGO-LIT-1..4 + D-EGOLIT-WITHHELD1 inserted
(beside the Master Mind's D-REFCV4B-EGODROP2 / H-EGODROP-PRED, which this WP's §3.6 panel is the
discriminating experiment for); kb_add --verify result in §5.`

**PI question (verbatim):** *"do research how it is done in typical similar successful works and
the original paper of refc."*

**Tier stamp.** Every number in §1–§2 is **PUBLISHED (primary: read from a banked PDF, or from
source code at a named commit — `repo@sha:path:line`)** about other people's models on other
people's benchmarks; none is a TanitAD result. TanitAD numbers appear only in §3 and carry their
artifact path and tier (`D-REFCV4B-EGODROP1`, `H-ECHO-8`, `EGODROP_9500.json`; all **T1
self-action open loop / T0** early-training diagnostics, never driving performance).

**Builds on (cited, not repeated):**
- `…/2026-09-05-refc-vs-diffusiondrive-audit/` (commit `7c67b3d`): DD's decoder as re-implemented
  by REF-C — status vector = command 4 + velocity 2 + acceleration 2 through the ego planning
  query; k-means anchors in metres, 20 on NAVSIM.
- `…/2026-09-03-refc-ego-inputs-and-anti-echo/RESULT.md` (E-REFC-EGO-1): TCP has no ego dropout
  (`H-ECHO-7`); PlanTF SDE numbers; DRAMA 0.835 → 0.848; PLUTO +2.60; BEV-Planner / PARA-Drive
  echo asymmetry; CARLA-TransFuser Tab. 10 (velocity in the backbone: DS −11.33).
- `…/2026-09-04-refcv4b-turn-coverage/` (`D-REFCV4B-EGODROP1`): the withheld-row bank rolled at a
  fixed 10 m/s has a 7.59 m oracle-in-vocabulary ceiling against 1.11 m at the measured v0.

Provenance of every code line and paper: `raw/sources.md` (repos, full SHAs, per-file sha256) and
`raw/code_manifest.json`.

---

## 0. The verdict in ten lines

1. **No successful planner rolls its candidate GEOMETRY from the measured speed.** Every
   anchor/vocabulary planner we read (DiffusionDrive, Hydra-MDP/++/GTRS, DriveSuprim, WoTE,
   VADv2, SimpleVSF) uses a **fixed bank in metres** (k-means of expert trajectories) and lets
   the ego status in as a **feature** — a token attended by the plan query (DD, TransFuser), a
   vector **added** to every candidate's feature (Hydra family: `dist_status = tr_out +
   status_encoding`), or **concatenated** per anchor (WoTE). refcv4b's per-window kinematic bank is
   **unprecedented in this set**, and so is the "withheld-row bank" problem it creates: with a
   speed-blind bank there is no bank to mis-roll.
2. **The only published designs that DO build candidates from the current speed are rule-based
   or self-predicted.** PDM-Closed's 15 IDM proposals (5 target speeds × 3 lateral offsets)
   integrate from the measured velocity — and PDM is the *teacher* Hydra-MDP distils from.
   SparseDrive (DiffusionDrive's nuScenes host) refuses the measured velocity in its ego anchor
   *"to avoid ego status leakage"* and initialises it from **its own predicted velocity of the
   previous frame** — the published precedent for option (c).
3. **DiffusionDrive v1 has NO ego dropout, NO ego masking, NO status augmentation** — two code
   probes at `9b52ed0`, zero hits. Its ego status is the TransFuser 8-vector, encoded once, and
   reaches the diffusion head **only through the `ego_query` cross-attention**: the
   `status_encoding` argument threaded into every decoder layer is **never read** inside the
   layer (a dead parameter). Its 20 anchors are a `requires_grad=False` k-means table in metres.
4. **`ego_dropout` exists in three of the nineteen works, and all three are regression
   planners without a bank**: PlanTF and PLUTO (identical `StateAttentionEncoder`, p = 0.75,
   per scalar, **pose never dropped**, the dropped scalar **absent from the attention key set**,
   not zeroed) and DRAMA (state 0.5 / fusion 0.1, learnable positional embedding as the
   presence signal). Everyone else — TransFuser, TF++, Hydra family, DriveSuprim, WoTE, iPad,
   UniAD, VAD, GenAD, SparseDrive, PARA-Drive — feeds the ego state **unmasked** and guards
   against echo elsewhere (site, benchmark, augmentation, prediction).
5. **Where the field puts the guard instead:** (i) the **site** — planner head / scorer, never
   the visual trunk (PARA-Drive; CARLA-TransFuser's −11.33 DS is a trunk injection); (ii) the
   **benchmark** — NAVSIM's CV filter, PARA-Drive's "targeted" split; (iii) **pose
   perturbation** — PlanTF's state perturbation (p 0.5), DriveSuprim's 3-view rotation
   ensemble with self-distillation; (iv) **decoupled longitudinal output** — TF++ classifies
   the target speed into 8 bins and consumes the measured speed only in the controller.
6. **UniAD, VAD and GenAD feed ego kinematics ONLY through BEVFormer's `can_bus` MLP into the
   BEV queries** (`use_can_bus=True` in all three shipped configs); their planning heads get
   the command and — in VAD/GenAD's *public* configs — **no** `ego_lcf_feat`
   (`ego_lcf_feat_idx=None` in both commits that ever touched the config). The 0.37 m VAD
   checkpoint BEV-Planner perturbed *"uses ego status in its planner"* and came from the
   authors; the public model-zoo VAD-Base is the 0.72 m non-ego model. **This sharpens a
   sibling sentence** (§4.1).
7. **VAD's `ego_lcf_feat` is now resolved** (sibling open item #8): a 9-vector
   [v_x, v_y, a_x, a_y, yaw-rate, length, width, **v₀ (CAN longitudinal speed)**, **κ = 2·steer /
   2.588**] — the closest published analogue of our `(v0, a_long, yaw_rate, curvature)` block,
   and its curvature comes from the **steering-angle feedback**, not from the trajectory.
8. **Speed enters the CANDIDATE-LEVEL decision late in every successful vocabulary planner**
   — after the scene cross-attention (Hydra/DriveSuprim add it to `tr_out`; VADv2 adds
   `E_state` at the scoring head) — which is exactly the placement that makes a per-anchor
   longitudinal choice *scene-first, speed-second*. Our v0-conditioned bank does the opposite:
   speed decides the geometry before the scene is consulted.
9. **On our own measurements the fixed 10 m/s withheld bank is the outlier**, and the model's
   own vision-derived 2 s speed already prices better: withheld-row oracle-in-vocabulary
   **7.61 m (fixed) → 2.00 m (rolled at the model's own predicted speed) → 1.12 m (leak
   bound)**, paired −5.61 m [−7.26, −4.20] (`EGODROP_9500.json`; step 9,500; MAE of that
   speed vs GT **1.84 m/s**, 4.91 at step 5,000). The withheld-row classifier is visibly
   compensating for the mis-rolled bank: its modal anchor is the **+2.92 m/s² accelerating**
   control (20.4 %) instead of straight-ahead (60.1 % on kept rows), and straight-ahead is
   chosen on **0 %** of withheld rows above 15 m/s.
10. **Recommendation (§3):** keep `ego_dropout 0.5` + X15 (the one lever that made an arm read
    the scene, `H-ECHO-8`); **do not** bucket the measured speed into a withheld bank (a coarse
    leak of the very channel the draw withheld); run the **5-arm tiny-rig panel** pre-registered
    in §3.6 — fixed 10 m/s (control) · **detached, vision-only predicted-speed bank** (the
    SparseDrive pattern, training-only) · random-marginal-speed bank (blindness control) ·
    `ego_dropout 0.25` · **speed-blind fixed vocabulary with the ego block as a feature** (the
    field's design) — scored on the `H-ECHO-8` separation instrument and the four families on
    kept AND withheld rows, never on ADE alone. Both outcomes are written down.

---

## 1. Q1 — the nineteen works, read at source (ego state at inference · where it enters · dropout/masking · vocabulary & speed conditioning · evidence)

Legend: **cmd** = discrete driving command; **v/a** = 2-D velocity / acceleration in the ego
frame; **code** = read at the pinned commit in `raw/sources.md`; **paper** = read from the banked
PDF. "Dropout" means *ego-state* dropout/masking; ordinary attention dropout is not counted.

| # | work | ego state at inference | where it enters | ego dropout / masking (train) | vocabulary; conditioned on speed? | evidence given |
|---|---|---|---|---|---|---|
| 1 | **DiffusionDrive v1** — `2411.15139`; `hustvl/DiffusionDrive@9b52ed0` | **cmd 4 + v 2 + a 2** from `ego_statuses[-1]` (current only) — `transfuser_features.py:45-49` | `Linear(8, 256)` (`transfuser_model_v2.py:41`) → **65th key/value token** beside the 8×8 BEV tokens (`:112`) → the TransFuser decoder's trajectory query attends it (`:126-129`) → that query is the **`ego_query`** the diffusion layers cross-attend (`:331`). ⚠️ `status_encoding` is passed into every diffusion layer (`:322, :370, :376`) and **never used** inside `CustomTransformerDecoderLayer.forward` (`:316-343`) | **NONE.** `tf_dropout = 0.0` (`transfuser_config.py:76`); the only `nn.Dropout(0.1)` sits on attention residuals (`:278-279, :325, :331`); no `rand`/`mask` touches the status (probe 1: `rand\|mask\|Dropout` in the model → only the diffusion noise `:465, :469, :517`); no augmentation in agent / callback / loss (probe 2: `status\|ego_vel\|augment` → 0 hits) | **20 k-means anchors in metres**, `nn.Parameter(…, requires_grad=False)` (`:407-412`), `kmeans_navsim_traj_20.npy` (`transfuser_config.py:19`). **Not speed-conditioned.** Noise added in normalised coordinates; train `t ~ U[0,50)` (`:465`), test `trunc 8`, DDIM `[10, 0]` (`:505-518`) | Tab. 3 "Ego Query" ✓ in all six rows — never ablated (sibling §3). App. Tab. 8: k-means anchors **88.1** vs *extrapolated-from-current-status* prior **81.3** (inference) / **84.7** (trained) PDMS |
| 2 | **TransFuser (CARLA)** — `2205.15997` | **none** (*"we do not use velocity as an input"*) | — | — | none (GRU waypoints) | Tab. 10: velocity summed into the positional embedding at **all 4 backbone stages** → DS **56.68 → 45.35** (sibling §5.1) |
| 3 | **TransFuser (NAVSIM)** — `2406.15349`; devkit `@0a380a9` | **cmd 4 + v 2 + a 2**, current only (`transfuser_features.py:45-49`) | `Linear(8, d)` (`transfuser_model.py:39`) → **one extra keyval token** (`:106-112`, `Embedding(8²+1)` at `:34`) | **none** | none (regression head) | Tab. 2: drop v+a (goal only) **−1.5 to −2.6 PDMS**; drop a only **−1.0 to −2.1**; seed spread ±0.56 (sibling §4.4) |
| 4 | **TransFuser++** — `2306.07957`; `autonomousvision/carla_garage@72f39a6` (`leaderboard_2`) | **speed (1 scalar, speedometer)** + discrete command (6) — `config.py:465 use_velocity = 1`, `:741 use_discrete_command = True`; `sensor_agent.py:496-498` | `BatchNorm1d(1)` → 2-layer MLP → **one "velocity token"** with a learnable positional embedding, concatenated to the BEV tokens of the transformer decoder (`model.py:219-226, :315-327`; paper App. B.1). ⭐ **Longitudinal output is a classification over 8 target-speed bins** `[0, 4, 8, 10, 13.9, 16, 17.8, 20]` m/s with two-hot labels (`config.py:430, :527`); the measured speed is consumed again only by the PID controller (`model.py:493-497`); inference scales targets by **0.8** (`:789`) | **none** (no masking of `ego_vel` anywhere in `model.py`) | none — path waypoints + speed classes (the "disentangled" output) | no velocity ablation in the paper; a **TP shortcut** ablation is its analogue of ego echo (attention pooling + augmentation fixes it). The paper flags *"causal confusion … for inputs such as velocity [15]"* as known (p. 415 of the extract) |
| 5 | **PlanTF** — `2309.10443`; `jchengai/planTF@f540608` | **current state only**: `[x, y, heading, v_x, a_x, steering]` (`state_channel 6` of the 7-vector built at `nuplan_feature_builder.py:192-203`; yaw-rate is the unused 7th; steering/yaw-rate derived from the heading difference, `:459-475`); **no ego history** (`use_ego_history: false`) | **`StateAttentionEncoder`** (`agent_encoder.py:97-139`): one `Linear(1, dim)` per scalar (`:103`) + learnable positional embedding, a learnable query cross-attends the scalar tokens; the result **replaces the ego agent token** (`:88-90`) | ⭐ **`state_dropout: 0.75`** (`config/model/planTF.yaml`): in training, the first **3 tokens (x, y, heading) are always visible**, each remaining kinematic token is dropped **independently** with p = 0.75 via `key_padding_mask` (`:120-137`) — the scalar is **absent from the key set**, not zeroed; eval mask = `None`. Plus **state perturbation** augmentation (uniform noise on x, y, yaw, v, a, steer, steer-rate; p 0.5; `state_perturbation.py:38-47`) | none (6-mode regression) | Tab. II: state6 SDE **OLS 88.55 → 87.07, NR-CLS 83.19 → 86.48, R-CLS 74.79 → 80.59**; Tab. VIII rate sweep 0/0.25/0.5/0.75 → NR-CLS 77.28/81.70/83.71/**86.48** (sibling §6.1) |
| 6 | **PLUTO** — `2404.14327`; `jchengai/pluto@b9964b6` | same 6-vector (`pluto_feature_builder.py:361-380`; steering/yaw-rate from the simulator state in closed loop) | `agent_encoder.py` **byte-identical to PlanTF's** (`diff -q`), `state_dropout: 0.75` (`config/model/pluto_model.yaml`) | same SDE, same rate | none (12 lateral-longitudinal queries on reference lines) | Tab. III: **M0 87.04 → M1 (+SDE) 89.64 (+2.60)**; Collisions 95.92 → 97.37, TTC 91.43 → 95.14, Speed 91.01 → 96.91 |
| 7 | **DRAMA** — `2408.03601` (no public code) | **v, a, cmd** (§4.2) | status encoding concatenated with the fused LiDAR-BEV feature as the Mamba-Transformer decoder's K/V (§2, Fig. 1) | ⭐ **Feature State Dropout**: fusion + status features + **learnable positional embedding**, then a *differentiated* dropout — **state 0.5, fusion 0.1** (§3.3, Fig. 5) | none (8-waypoint regression) | Tab. 2 PDMS: 0.835 → 0.842 (fusion 0.1) → 0.844 (state 0.5) → **0.848** (both) |
| 8 | **Hydra-MDP / Hydra-MDP++** — `2406.06978`, `2503.12820`; public code of the family `NVlabs/GTRS@92a740d` (`gtrs_dense/`) | **cmd 4 + v 2 + a 2**, current (`hydra_features.py:73-85`; `num_ego_status = 1`, `use_hist_ego_status = False`, `hydra_config.py:41, :62`) | `Linear(8·n, d)` (`hydra_model.py:48`) **added to every vocabulary feature after the scene cross-attention**: `dist_status = tr_out + status_encoding.unsqueeze(1)` (`:255-256`) — the paper's `V'_k = Transformer(MLP(V_k)) + E` | **none on the ego.** The code has a **vocabulary** dropout (`vocab_dropout`, default `False`, `hydra_config.py:39`): at train, a random **half** of the vocabulary is scored (`hydra_model.py:232-240`) | **fixed k-means vocabulary** of 700 K nuPlan trajectories, 40 × (x, y, heading), k = 4096 / 8192 / 16384 (`self.vocab = nn.Parameter(np.load(vocab_path))`, `:147-149`). **Not speed-conditioned.** Sub-scores distilled from PDM simulation of the whole vocabulary | Hydra-MDP: V8192 > V4096 across methods (§4); no ego ablation in either paper |
| 9 | **UniAD** — `2212.10156`; `OpenDriveLab/UniAD@609ee08` (tag `v2.0`) | **can_bus (18-d: pose Δ, quaternion, a 3, ω 3, v 3, angles)** into the BEV encoder; **command (3)** into the planner | `bev_queries = bev_queries + can_bus_mlp(can_bus)` (`modules/transformer.py:152-155`, `use_can_bus=True`, `configs/stage2_e2e/base_e2e.py:156`); planner: `plan_query = cat(sdc_traj_query, sdc_track_query, navi_embed[command])` (`planning_head.py:166-168`) — **no ego kinematics in the planning head** | **none** (first test frame zeroes the can_bus pose deltas for BEVFormer alignment, `uniad_e2e.py:274-279` — a coordinate convention, not a guard) | none | BEV-Planner Tab. 1: UniAD w/o ego 1.03 → official (ego in BEV) 0.66 → ego in BEV+planner 0.46 m (sibling §4.2); the paper never mentions ego status (0 hits) |
| 10 | **VAD** — `2303.12077`; `hustvl/VAD@1688c4b` | **can_bus into the BEV encoder** (`VAD/modules/transformer.py:153-156`, `use_can_bus=True`, `VAD_base_e2e.py:170`); **command** selects the output mode (`VAD.py:422-423`); planner-side `ego_lcf_feat` **wired but OFF**: `ego_lcf_feat_idx=None, ego_his_encoder=None` (`VAD_base_e2e.py:78-79`; identical in **both** commits that ever touched the file, `0a20dc7` 2023-08-01 and `156a744` 2023-08-29) | `ego_fut_decoder` in-dim = `2D + len(ego_lcf_feat_idx)` when set (`VAD_head.py:418-424`), four concatenation branches (`:764-789`) | **none** | none (3 command-conditioned modes) | paper Tab. 1: VAD-Base **0.72** (ego off, the public model-zoo checkpoint, README) vs **0.37** (†, ego on, *"deactivated … to avoid shortcut learning"*). ⭐ `ego_lcf_feat` = **[v_x, v_y, a_x(=can_bus[7]), a_y, ω, length, width, v₀ (CAN), κ = 2·steer/2.588]** (`vad_nuscenes_converter.py:462-513`) |
| 11 | **VADv2** — `2402.13243` (paper) | **ego state `E_state`** + navigation | **added at the scoring head**: `p(a) = σ(MLP(φ(E(a), E_scene) + E_navi + E_state))` (sibling §5) | none | **4096–8192 fixed vocabulary** (k-means / furthest-point on expert trajectories); **not speed-conditioned** | NAVSIM Tab. 1: 83.0 PDMS |
| 12 | **PARA-Drive** — `paradrive-cvpr2024` (no code) | CAN bus (v, a, angular velocity) + command + history | **planner head only, never the BEV encoder** | none (two probes, sibling §4.3) | none | Tab. 6: without ego **0.5574** vs blind AD-MLP **0.5568** on L2, but Offroad 0.12 vs 1.21, targeted-split collision 0.14 vs 0.94 |
| 13 | **GenAD** — `2402.11502`; `wzzheng/GenAD@b16667f` | **can_bus into the BEV encoder only** (`GenAD_config.py:171 use_can_bus=True`); planner: `ego_lcf_feat_idx=None, ego_his_encoder=None` (`GenAD_config.py:78-79`) → the ego token is a **learnable embedding** (`GenAD_head.py:692-695`); command selects the mode | latent generative decoder (VAE + GRU) over the instance tokens | **none** | none | the paper has **no** ego-status statement at all (0 hits) — ego handling is code-only |
| 14 | **SparseDrive** — `2405.19620` (paper; DD's nuScenes host) | ego anchor box `(x, y, z, size, yaw, v)`: pose/size set from knowledge; ⭐ **velocity NOT from measurement** — *"directly initialized from ground truth velocity leads to ego status leakage [27]"* | auxiliary head **predicts current ego status** (v, a, ω, steering) from the ego instance feature; *"at each frame, we use the predicted velocity from last frame as the initialization of ego anchor velocity"* (§3.3) | none (the guard is the predicted-not-measured velocity) | 6 planning modes × 3 commands; k-means anchors only for detection/map | Tab. 4 ID-3 (random ego feature + all anchor params 0): L2 avg **0.61 → 0.63 m**, collision **0.08 → 0.11 %** — a bundled ablation, not velocity alone |
| 15 | **WoTE** — `2504.01941`; `liyingyanUCAS/WoTE@298957c` | **cmd 4 + v 2 + a 2** (`WoTE_features.py:36-40`) | `Linear(8, d)` (`WoTE_model.py:40, :58`) → **concatenated with every anchor's feature** (`:404-411`, `_concatenate_ego_and_traj_features`) → offset head, latent BEV world model per anchor (`:316`), imitation + simulation rewards, argmax (`:652-654`) | **none** | **256 k-means anchors** `nn.Parameter(np.load(cluster_file), requires_grad=False)` (`:69-72`; `configs/default.py num_traj_anchor 256`), PDM sub-scores pre-computed per anchor. **Not speed-conditioned.** Unseen N = 1024 anchors at test **+1.3 PDMS** (§5) | Tab. 1 (NAVSIM v1): Ego Status MLP 65.6 · WoTE 88.3 |
| 16 | **iPad** — `2505.15111`; `Kguo-cs/iPad@78f990a` (`navsim/agents/pad/`) | **pose 3 + v 2 + a 2 + cmd 4 = 11**, last status only (`pad_features.py:46-57`, `pad_model.py:36`) | `Linear(11, d)` (`pad_model.py:20`) → **added to the 64 × 8 learnable proposal queries** (`:48-50`), 4 shared refinement passes (`navsim_config.py ref_num 4`), scorer argmax (`:62-78`) | **none** (`tf_dropout 0`; the only zeroing is Bench2Drive-specific: `if self.b2d: ego_status[:, 1:3] = 0`, `:41-42`) | **no fixed vocabulary** — 64 learnable proposals refined iteratively; ego enters the query init | paper: *"ego status, including … current velocity, acceleration, and future commands, is encoded into the ego feature E using a linear layer"*; no ego ablation |
| 17 | **DriveSuprim** — `2506.06659`; `William-Yao-2000/DriveSuprim@80fe792` | **cmd 4 + v 2 + a 2**, current (`drivesuprim_features.py:94-106`, `num_ego_status 1`) | Hydra pattern: `dist_status = tr_out + status_encoding.unsqueeze(1)` (`drivesuprim_model.py:207`); coarse-to-fine (8192 → top-256 → 3-layer refinement, `RefinementConfig`) | **none on the ego.** The guard is **rotation augmentation**: 3 rotated views of the stitched camera + correspondingly rotated trajectories (`drivesuprim_features.py:186-197, :228-240`, `EgoPerturbConfig`) with **self-distillation** (teacher on the clean view) | **8192 fixed vocabulary** (`vocab_size 8192`, `self.vocab = nn.Parameter(np.load(vocab_path))`, `:123-124`). **Not speed-conditioned** | Tab. 2 (navtest PDMS): Hydra-MDP 86.5 → DriveSuprim **89.9** (R34); Tab. 3 (**EPDMS on navtest, one-stage**): Ego Status MLP **64.0**, TransFuser 76.7, HydraMDP++ 81.4, DriveSuprim 83.1 — ⚠️ see §4.2 on which split this is |
| 18 | **SimpleVSF** — `2510.17191` (no code) | GTRS scorers' status token + a **VLM prompt carrying speed, acceleration and the command** (§2.2) | GTRS-Dense scorers over a super-dense vocabulary + diffusion-policy anchors (`GTRS dp_model.py:326, :387` also append the status token to the DP condition); VLM scores fused by a fixed-weight log-sum | **none** mentioned | fixed super-dense vocabulary + DP anchors; not speed-conditioned | ICCV-2025 NAVSIM v2 challenge winner; no ego ablation |
| 19 | **PDM-Closed** — `2306.07962` (rule-based; NAVSIM's teacher) | measured **position, velocity, acceleration** + route centreline | ⭐ **the proposal set IS speed-conditioned**: IDM rolled at **5 target speeds {20, 40, 60, 80, 100} % of the speed limit × 3 lateral offsets {−1, 0, +1} m = 15 proposals**, 4 s, simulated with the bicycle model and scored (§4 "Proposals") | n/a | 15 kinematic proposals per frame | Tab. 1: PDM-Open with **ego history** *"only yield[s] little improvement and lead[s] to a drop in CLS"*; IDM a = 1.0 → 0.1 raises OLS, lowers CLS |
| — | **Ego-Status MLP** (the floor arm) — devkit `ego_status_mlp_agent.py:25-31, :71-80` | **v 2 + a 2 + cmd 4** | 3-layer MLP → 8 poses | — | — | NAVSIM 65.6 PDMS; navhard two-stage EPDMS **14.1** (`2506.04218`); navtest-EPDMS one-stage 64.0 (DriveSuprim Tab. 3) |

### 1.1 The three patterns behind the table

**P1 — Speed is a FEATURE, never the GEOMETRY (7 of 7 vocabulary planners).** DD, Hydra-MDP/++,
GTRS, DriveSuprim, WoTE, VADv2 and SimpleVSF all score/refine a **fixed bank in metres** and inject
the ego status *late* — after the candidate has attended the scene (Hydra/DriveSuprim add it to
`tr_out`; VADv2 adds it at the scoring MLP; WoTE concatenates it per anchor before the offset
head; DD's diffusion layers see it only through `ego_query`). The candidate-level longitudinal
decision is therefore *scene-first, speed-second*. ⇒ **None of them has a "withheld-row bank"
problem, because none of them has a speed-rolled bank.** The refcv4b design — a per-window
kinematic vocabulary — is unprecedented in this set (the prior audit's row #14 already noted DD
has nothing like it).

**P2 — When the ego IS dropped, it is dropped per scalar, as ABSENCE, with pose exempt, in a
regression planner (PlanTF, PLUTO, DRAMA).** PlanTF's `key_padding_mask` removes the scalar from
the attention's key set; the query still produces an embedding from whatever is visible. There is
no zero-collision by construction — the closest thing to our X15 bit is the *masked key*, and
PlanTF never withholds `(x, y, heading)`. All three deployed rates are **≥ 0.5** (0.75 / 0.75 /
0.5), each drawn **independently per scalar** (PlanTF: P(all three kinematic tokens withheld) =
0.42, P(all present) = 0.016), and all three papers report the guard **costing open-loop score
while raising closed-loop/simulation score**.

**P3 — Where a candidate set MUST be built from the current state, the field uses either a
rule-based rollout with the measured state (PDM-Closed, the teacher) or the model's OWN PREDICTED
state (SparseDrive's ego anchor).** SparseDrive's phrasing is the PI's rule in the authors' own
words: initialising from the measured velocity *"leads to ego status leakage"*, so the ego anchor
carries the network's previous-frame estimate — detached from the current measurement by
construction. This is the published precedent for option (c) in §3, with two caveats: it is a
*temporal* self-prediction (last frame), and its ablation (Tab. 4 ID-3) bundles the feature
initialisation with the anchor parameters, so **the velocity part is not separately measured**.

### 1.2 What is NOT in the literature (two-probe absences)

- **No planner rolls a per-window kinematic anchor bank from the measured speed** (19 works, code
  where public; the only per-frame kinematic proposal set is PDM-Closed's rule-based IDM roll).
- **No planner rolls a bank at a FIXED reference speed for withheld rows** — the concept does not
  arise without a speed-rolled bank.
- **No planner conditions the vocabulary on a speed BUCKET.** Hydra's only vocabulary
  manipulation is a random halving (`vocab_dropout`, off by default).
- **No anchor/vocabulary planner uses ego dropout at all.** The intersection "bank + ego
  dropout" is empty; the intersection "speed-rolled bank + ego dropout" is ours alone.
- **No planner ablates velocity vs acceleration vs yaw-rate separately** (still true; NAVSIM's
  Tab. 2 B1/B2 remains the only per-component split, sibling §9.7).

---

## 2. Q2 — DiffusionDrive v1, specifically (code at `hustvl/DiffusionDrive@9b52ed0`, paper `2411.15139`)

### 2.1 Anchor construction
- `transfuser_config.py:19`: `plan_anchor_path = ".../kmeans_navsim_traj_20.npy"` — **20 × 8 × 2
  (x, y) in metres, ego frame**, k-means over NAVSIM expert trajectories (paper §3.3: 20 on NAVSIM,
  18 on nuScenes).
- `transfuser_model_v2.py:407-412`: `self.plan_anchor = nn.Parameter(torch.tensor(plan_anchor,
  dtype=float32), requires_grad=False)  # 20,8,2` — **frozen, not learned, not conditioned on
  anything.** Train: `plan_anchor.unsqueeze(0).repeat(bs, …)` → `norm_odo` (`:463-464`), one
  noised copy per example at `t ~ U[0, 50)` (`:465-469`); test: same repeat, noise at
  `trunc_timesteps = 8`, two DDIM steps `[10, 0]` (`:505-518`). The bank is identical for a
  standing car and a car at 30 m/s; speed only ever changes which anchor wins and how far the
  offset head moves it.
- The paper's App. Tab. 8 is the one place DD *tests* a speed-conditioned prior — the
  **extrapolated-from-current-status single anchor — and it loses 3.4 PDMS trained-for, 6.8
  swapped in** (sibling §3); their reading: it *"fails to cover the potential action space"*.

### 2.2 Ego features and where they enter (three hops, one of them dead)
1. **Vector** (`transfuser_features.py:45-49`): `status_feature = cat(driving_command[4],
   ego_velocity[2], ego_acceleration[2])` from **`ego_statuses[-1]` — the current frame only**;
   no history, no pose, no yaw-rate, no steering.
2. **Encoding + site** (`transfuser_model_v2.py:41, :110-112`): `Linear(8, 256)`; the encoded
   status is appended as the **65th token** of `keyval` (64 BEV tokens + 1) with its own
   positional embedding, so the TransFuser decoder's **trajectory query and 30 agent queries all
   attend it** (`:126-129`). The trajectory query becomes the diffusion head's **`ego_query`**
   (`:132`).
3. **Inside the diffusion decoder** (`CustomTransformerDecoderLayer.forward`, `:316-343`): the
   layer receives `status_encoding` as an argument (`:322`) and **never reads it** — the body is
   grid-sample BEV attention (`:324`) → agent cross-attention (`:325`) → **`cross_ego_attention
   (traj_feature, ego_query, ego_query)`** (`:331`) → FFN (`:335`) → **time modulation with
   `global_cond=None`** (`:337`) → heads (`:340`). ⇒ **The ego status reaches the diffusion head
   ONLY through `ego_query`** — a scene-conditioned query that *has attended* the status token —
   not as a direct condition and not through the timestep modulation. (The 2026-09-05 audit's
   row #21 stands; this is the finer-grained statement of it, and the dead argument is new.)

### 2.3 Ego dropout / masking / augmentation — NONE, at two probes
- Probe 1 (`transfuser_model_v2.py`, `grep -n "rand\|mask\|Dropout\|dropout"`): `tf_dropout`
  passed to attention layers (`:72, :290, :296`) with `tf_dropout = 0.0`
  (`transfuser_config.py:76`); `nn.Dropout(0.1)` on two attention residuals (`:278-279, :325,
  :331`); `torch.randint` / `torch.randn` only for the diffusion timestep and noise (`:465, :469,
  :517`). Nothing touches `status_feature` / `status_encoding`.
- Probe 2 (`transfuser_agent.py`, `transfuser_callback.py`, `transfuser_loss.py`,
  `grep -in "status\|ego_vel\|augment"`): **0 hits.**
- The paper: "dropout" appears **0 times** in the ego context (the sibling's TCP finding
  `H-ECHO-7` therefore extends to the DD half of REF-C's ancestry: **neither ancestor has an ego
  dropout; `ego_dropout = 0.5` is ours.**)

### 2.4 How the truncated diffusion is conditioned on them
Per layer, in this order: candidate geometry (grid-sampled BEV at the noisy waypoints) → detected
agents → **ego query** → FFN → **AdaLN time modulation** (`ModulationLayer`, `global_cond=None`,
`global_img=None`) → `(offset, score)` heads with `poses_reg[..., :2] += noisy_traj_points`. The
timestep never mixes with the ego status; the ego status never mixes with the noise schedule. The
selection reads the **last layer's** score over the fan it emits (`forward_test`, `:549-552`).

### 2.5 The nuScenes host (paper §4.2: *"we follow the SparseDrive baseline"*)
DD's nuScenes numbers inherit **SparseDrive's** ego handling, which is the leak-aware one (§1
row 14): the ego anchor's velocity is the network's **own previous-frame prediction**, an auxiliary
head decodes (v, a, ω, steering), and the measured velocity is refused *"to avoid ego status
leakage"*. So REF-C's origin paper has, across its two hosts, **both** field patterns — the
status-token feature on NAVSIM and the self-predicted state on nuScenes — and **no dropout on
either.**

---

## 3. Q3 — the withheld-row bank for refcv5, priced (MEASURED inputs are ours; tier stamps inline)

### 3.1 The measured situation (refcv4b, `EGODROP_9500.json`, step 9,500, ckpt md5 `ab4cd79e…`, 4,823 windows / 141 episodes, paired episode-cluster bootstrap n_boot 2000 seed 0; **T0/T1 early-training diagnostic, not driving performance**)

| quantity (withheld rows unless stated) | value |
|---|---|
| training regime | `ego_dropout 0.5`, `ego_valid_channel` (X15) on, `anchor_v0_cond` on, `anchor_ref_speed 10.0`, controls `alat`, 117 anchors, `refc1_speed_head` off |
| withheld bank ≡ the bank rolled at 10 m/s | `withheld_bank_equals_ref_speed_roll_max_abs_m = 0.0` (bit control) |
| oracle-in-vocabulary, **fixed 10 m/s** | **7.594 m** (numpy re-roll, = PART6's 7.5941); trainer's `a_star` masked ADE **7.610 m** |
| oracle-in-vocabulary, **rolled at the model's own 2 s speed** (`g_tac`, vision-derived on withheld rows) | **2.002 m**; paired vs fixed **−5.608 [−7.262, −4.195]**, separated |
| oracle-in-vocabulary, rolled at the **true v0** (the LEAK bound — refused as a design) | **1.125 m**; paired pred − true **+0.877 [+0.741, +1.032]** |
| MAE of that predicted 2 s speed vs GT | **1.836 m/s** at step 9,500 (**4.91** at step 5,000; kept-regime `g_tac` 0.880) |
| by v0 band — fixed / predicted / true (m) | [0,5): 6.92 / 1.20 / 0.89 · [5,10): 2.40 / 1.71 / 1.24 · [10,15): 2.59 / 1.93 / 1.10 · [15,20): 7.87 / 2.90 / 1.25 · [20,25): 19.53 / 2.92 / 1.25 · 25+: 37.56 / 3.42 / 1.09 |
| what the withheld classifier does with the mis-rolled bank | modal anchor **112 = (a_lon +2.92 m/s², a_lat 0)** at 20.4 % vs kept modal **67 = straight-ahead** at 60.1 %; straight-ahead chosen on **0.0 %** of withheld rows in every band above 15 m/s (vs 75.7–98.3 % kept); selection entropy 2.50 vs 1.72 nats; agreement with its own oracle 0.379 vs 0.471 (paired −0.091 [−0.153, −0.028]) |
| goal graft changes the winner | withheld **8.67 %** vs kept 1.87 % of windows |
| speed MAE of the open-loop plan (os_w, 2 s families) | 1.859 m/s, bias −0.760 (under-speed), along-track final bias −1.54 m |

*(The same JSON is the evidence of the Master Mind's `D-REFCV4B-EGODROP2` — the selected-anchor
offset is **6.18 m** on withheld vs **0.64 m** on kept rows, ×10, 6.03 m of it along-track — and
of its hypothesis `H-EGODROP-PRED`, that rolling the withheld bank at the model's own predicted
2 s speed *"removes most of the burden without any ego input"*. §3.6 below is the discriminating
experiment for that hypothesis; the rows agree on every number.)*

**Reading.** The fixed bank does not merely cap the ceiling; it **teaches the withheld regime a
different policy** — accelerate hard from a 10 m/s fan whenever the true speed is higher, brake
when it is lower. That is what the trainer's `a_star` supervises on ~50 % of rows
(`refc_v3_train.py:546-553`, target measured against `out["anchor_bank"]`, the bank actually
decoded). The E9 goal graft already overrides the classifier on 4.6× more withheld than kept
windows — the model is compensating through the *other* path.

### 3.2 Why the PI's constraints still bind (restated so the options are priced against them)
- **Vision-only at inference (2026-08-03)** and **measured v0 at t0 is admissible (2026-09-02)**:
  at eval every row is "kept", so the withheld bank is a **training-regime object only** — its job
  is to shape what the scene pathway learns, not to be a deployable input.
- **Anti-echo evidence (`H-ECHO-8`)**: `ego_dropout 0.0 → 0.5` is the single lever that turned an
  `ECHOING` arm into the only `READS_BOTH` arm, at the cost PlanTF predicted (worst 6 s
  displacement). Any option that weakens the withholding must be scored on that instrument, not
  on ADE.
- **The design's own leak rule** (`refc.py:1329-1333`): *"rolling the bank from [the pre-dropout
  speed] on a withheld row would put the withheld channel into the candidate GEOMETRY"* — the
  bank is part of the input surface the classifier reads (`traj_proj(x_est)`), so any function of
  the measured v0 on a withheld row is a leak of that channel.
- **Goal / situation information-disjointness (CLAUDE.md, 2026-08-03)**: an input must not carry
  the output of the situation classifier; generalised — *ask whether an input at inference
  contains something the thing being measured also produces*.

### 3.3 The options, priced

| option | what it does | ceiling (withheld rows) | leak of measured v0 into a withheld row? | echo / feedback risk | precedent | verdict |
|---|---|---|---|---|---|---|
| **(a) fixed 10 m/s** (today) | one reference roll for all withheld rows | **7.59 m** (1.11 at v0) | none | none — but it trains a *wrong policy* on ~50 % of rows (§3.1) and biases the withheld regime toward accelerate anchors | **none** (no published planner rolls a bank at a fixed speed; the concept exists only with a speed-rolled bank) | control arm only |
| **(b) speed-bucketed reference** (roll at the centre of the row's v0 bucket) | k buckets → k banks | between 1.11 and 7.59 (≈ the within-bucket mismatch) | ⛔ **YES** — log₂ k bits of the withheld channel enter the geometry; the classifier can read the bucket off the anchor spacing, which is exactly the leak `refc.py:1329` refuses and the X15 bit was built to remove | re-opens the echo through the bank (the same mechanism `H-ECHO-8` shut) | none (Hydra's only vocab manipulation is random halving) | ⛔ **refuse as designed**; admissible only if the bucket is NOT a function of the row — see (b′) |
| **(b′) random reference speed** drawn from the training marginal of v0 per withheld row | a bank that is right *on average* but uninformative | ≈ the marginal's mean absolute deviation — no better than (a) on average, unbiased in speed | none | none | none | **control arm** — separates "10 m/s is a biased choice" from "any speed-blind bank is bad" |
| **(c) roll at the model's own predicted speed** (`g_tac` 2 s speed, **detached**, **training-only**, kept rows unchanged, eval unchanged) | the withheld bank fits the row to ~2 m/s | **2.00 m** now (MAE 1.84 m/s); tracks the head's learning curve (4.91 m/s at step 5,000 ⇒ needs a warm-up) | **none**: on withheld rows `ego_e = ego_inj(zeros ⊕ keep=0)` (`refc_v3.py:868-879`) and `tactical_speed_input=False` (`:447`), so `g_tac` is a function of the image, nav and constants only — the measured v0 cannot reach the bank | (i) **attribution**: the withheld-row longitudinal decision is made twice on one path (`g_tac` → bank geometry → classifier), so a longitudinal gain cannot be assigned to the classifier vs the goal head; (ii) **non-stationary target**: `a_star` on withheld rows becomes a function of the model → early training rolls a bad bank (mitigate: warm-up on (a) until the withheld-row speed MAE < threshold, clamp v̂ to [0, 35] m/s); (iii) **no gradient loop** if detached — the same property SparseDrive gets from using the *previous frame's* prediction; (iv) at inference nothing changes, so the disjointness rule (an inference rule) is not touched, but the *training* coupling must be declared | ⭐ **SparseDrive** §3.3 (ego anchor velocity = own previous-frame prediction; *"to avoid ego status leakage"*); DD-on-nuScenes inherits it | **candidate**, run as an arm |
| **(d) lower `ego_dropout`** (0.5 → 0.25) | halves the mis-rolled fraction | unchanged per withheld row | none | weakens the only lever that produced `READS_BOTH`; PlanTF's sweep is monotone (NR-CLS 86.48 → 83.71 → 81.70 at 0.75 → 0.5 → 0.25) and DRAMA landed on 0.5; 0.25 untested on our rig | rate sweeps: PlanTF Tab. VIII, DRAMA Tab. 2 | **arm**, expected to lose on the echo instrument |
| **(e) the field's design: speed-blind fixed vocabulary in metres + the ego block as a FEATURE** (D-REFCV4-VOCAB1's 128 slot-normalised k-means, or the 13 × 9 controls rolled once at a fixed reference for ALL rows) | no per-row bank at all; speed enters the classifier/offset late, as in Hydra/DD/WoTE | the fixed bank's own ceiling on ALL rows (to be measured; DD lives with 20 anchors + offsets) | none (there is no withheld bank) | none new; the classifier's longitudinal choice becomes scene-first, speed-second (P1) | **7 of 7 vocabulary planners** | **arm** — the null hypothesis the v0-conditioned bank has to beat |

### 3.4 What the successful works do that we do not (the direct answer to 3e)
1. **They never let the speed decide the geometry.** The bank is fixed; speed is a late feature
   (P1). We roll the bank from v0 and then have to hide v0 from half the rows.
2. **They never zero a scalar; they remove it** — masked key + pose exempt (P2). Our X15 bit is the
   nearest equivalent and is now on; the *value* channel still reads 0.0 on withheld rows, which
   on our corpus collides with a genuine standstill 4.45 % of the time (sibling
   `2026-09-03-ego-zero-collision`).
3. **They predict the ego state as an auxiliary target and feed the prediction** (SparseDrive; TCP's
   image→speed head, λ 0.001, inherited and never ablated by us; TF++ classifies the target
   speed). We have `refc1_speed_head` **off** in refcv4b.
4. **They guard with pose/view perturbation** (PlanTF state perturbation p 0.5; DriveSuprim
   3-view rotation ensemble + self-distillation). We have none.
5. **They put the guard at the site or in the split**, not only in the input (PARA-Drive; NAVSIM's
   CV filter). Our E-ECHO-2 census is still the open item that decides whether our split can
   show an echo at all.

### 3.5 Recommendation
- **Keep** `ego_dropout 0.5` and X15 for refcv5 (the measured lever), and **refuse (b)**.
- **Change the withheld bank**, and decide *how* by the panel below rather than by argument. The
  two live candidates are **(c)** — the SparseDrive pattern, which our own step-9,500 numbers
  already price at a 3.8× lower ceiling than today with zero measured-v0 leak — and **(e)** — the
  field's design, which removes the problem instead of patching it and is the only option every
  successful vocabulary planner shares.
- Whatever wins, **state the attribution caveat of (c) in the design doc** if (c) is adopted: the
  withheld-row longitudinal decision is downstream of `g_tac`, so the longitudinal family on
  withheld rows scores the goal head and the classifier jointly.
- ⛔ **Do not score the panel on ADE.** Three primaries (PlanTF, PLUTO, DRAMA) and our own
  `H-ECHO-8` say a working guard *loses* displacement while *gaining* scene-reading.

### 3.6 Pre-registration — the cheapest discriminating experiment on the v7-tiny rig (0 pod GPU; the same rig, step budget and instrument as `H-ECHO-8`, `…/2026-09-03-refc-v4-design/RESULT.md` §6.3)

**Arms (one variable each; everything else refcv4b's shipped config):**
- **A0** — fixed 10 m/s withheld bank (today; the control).
- **A1** — withheld bank rolled at the **detached, vision-only `g_tac` 2 s speed**, clamped to
  [0, 35] m/s, with warm-up on A0's bank for the first N steps (N = the step at which the
  withheld-row speed MAE first drops below 2.5 m/s on A0's own log; if never, N = ⅓ of the
  budget). Kept rows and eval untouched.
- **A2** — withheld bank rolled at a **random speed drawn from the training marginal of v0**
  (blindness control: unbiased but uninformative).
- **A3** — `ego_dropout 0.25`, A0's bank.
- **A4** — **speed-blind fixed vocabulary** (D-REFCV4-VOCAB1's 128 k-means, or the 13 × 9 controls
  rolled once at 10 m/s for ALL rows) with the ego block entering only as a feature.

**Scored on (all four are mandatory; ADE is one row of the families, never the gate):**
1. the **`H-ECHO-8` separation instrument** (`tanitad.eval.echo_gate`: scene-degradation delta
   vs ego-degradation delta, each with its paired CI; verdict `ECHOING` / `READS_BOTH` /
   `SCENE_ONLY`);
2. the **four metric families on kept AND withheld rows** (the `EGODROP_*.json` protocol);
3. the withheld-row **oracle-in-vocabulary** and trainer-`a_star` ceilings, by v0 band;
4. **D3** (`‖∂(x_T, y_T)/∂ s_0‖`, one backward pass) and **D4** (plan predictability from `(v, a,
   yaw)` alone) from the sibling's panel.

**Outcomes committed in advance:**
- **A1 stays `READS_BOTH` and beats A0 on the withheld-row families/ceiling** ⇒ adopt (c) for
  refcv5 with the attribution caveat recorded in the design doc and the registry.
- **A1 reads `ECHOING`** (scene delta no longer separated) ⇒ the predicted-speed bank re-opens the
  echo through the tactical head ⇒ **refuse (c)**; the bank problem is then solved by (e) or
  lived with under (a).
- **A4 ≥ A0 on the echo instrument AND on the families** ⇒ the v0-conditioned bank is not
  load-bearing; adopt the field's design and retire the per-window roll (the simplest outcome).
- **A3 beats A0 on the families but loses separation** ⇒ the PlanTF signature; the rate is the
  wrong knob — keep 0.5.
- **A2 ≈ A0** ⇒ the harm is "any speed-blind bank", not "10 m/s" ⇒ (b′)-style fixes are dead.
- **A2 ≫ A0** ⇒ 10 m/s is a biased choice; report the marginal's mean and re-run A0 there
  before drawing any other conclusion.

Cost: five tiny-rig runs at the `H-ECHO-8` budget; **no pod GPU**; the instrument, dump protocol
and CI code already exist. The one code change is a `ref_speed` source switch in
`roll_bank` (`refc.py:1318-1368`) plus a `--withheld-bank {fixed,pred,random,none}` flag on the
trainer — and the config **must stamp it** (`refcv3`'s `config.json` stamps neither ego knob;
sibling `2026-09-03-ego-zero-collision`).

---

## 4. Corrections and nuances to sibling claims (each with the primary)

1. **"VAD's released official checkpoint uses ego status in the planner" (sibling §5, row VAD) —
   sharpened, not retracted.** `hustvl/VAD@1688c4b` ships `ego_lcf_feat_idx=None,
   ego_his_encoder=None` in `VAD_base_e2e.py:78-79` — and so did **both** commits that ever
   touched that file (`0a20dc7` 2023-08-01, `156a744` 2023-08-29); the README model zoo lists
   VAD-Base at **0.72 m** avg L2 (the ego-off row of the paper's Tab. 1). The **0.37 m** model is
   the paper's † row; BEV-Planner's Tab. 2 caption says it used *"the official VAD-Base checkpoint
   that uses ego status in its planner module"* and its acknowledgements thank the VAD authors
   for *"the model weights of VAD"*. ⇒ The contradiction is between the **paper's headline
   choice** (ego off, *"to avoid shortcut learning"*) and a **† checkpoint the authors provided**,
   not between the paper and the **public** code, whose config never enabled the planner-side
   ego. `MODEL_REGISTRY` and the KB line should say "the † checkpoint BEV-Planner obtained from
   the authors", not "the released checkpoint".
2. **"Ego Status EPDMS 64.0 on navhard is wrong; the primary says 14.1" (sibling §9.4) — both
   numbers are real, on different splits.** DriveSuprim Tab. 3 ("Evaluation on NAVSIM v2") reports
   Ego Status MLP **EPDMS 64.0**, TransFuser 76.7, HydraMDP++ 81.4 — the **navtest one-stage
   EPDMS** family that Hydra-MDP++ introduced (its Tabs. 2/4 are titled *"Performance on the
   Navtest Benchmark with extended metrics"*). `2506.04218`'s **14.1** is **navhard two-stage**.
   The sibling's warning stands (the NAVSIM v2 authors discourage exactly this reporting), but
   the 64.0 is not a fabrication — it is a *different benchmark wearing the same name*, the
   `df`/`step_s` scope class.
3. **`H-ECHO-7` extends to the DD half** — DiffusionDrive v1 has no ego dropout either, at two code
   probes (§2.3). *"Nobody upstream"* is now true of both ancestors at source, not only of TCP's
   paper text.
4. **Sibling open item #8 (VAD's `ego_lcf_feat`) — RESOLVED** (§1 row 10): 9 channels, with
   `v0` from the CAN `pose.vel[0]` and `κ = 2·steer/2.588` from the steering-angle feedback (sign
   flipped for Singapore); when the CAN messages are missing the converter falls back to
   `v0 = ‖Δ(his[-1] + fut[0])‖`, `κ = 0` (`vad_nuscenes_converter.py:502-506`).
5. **A vocabulary dropout exists in the Hydra family and is OFF** (`vocab_dropout=False`): when
   on it scores a random half of the vocabulary per step (`hydra_model.py:232-240`). It is not
   an ego mechanism, but it is the only published "dropout on the bank", and it drops *candidates*,
   never *the speed*.

---

## 5. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `RESULT.md` (this file) | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-ego-input-literature/RESULT.md` | no — committed via `mm_commit.py` |
| `raw/sources.md` (repos, full SHAs, per-file sha256, papers, TanitAD inputs) | `repo: …/2026-09-05-ego-input-literature/raw/sources.md` | no |
| `raw/code_manifest.json` (124 fetched files with sha256) | `repo: …/2026-09-05-ego-input-literature/raw/code_manifest.json` | no |
| third-party source pulls (13 repos, 124 files) | `devbox: C:/Users/Admin/ego_lit/code/<owner>_<repo>@<sha7>/` | ⚠️ **one place** (deliberately not committed — third-party code; re-fetchable from the SHAs in the manifest) |
| paper text extracts | `devbox: C:/Users/Admin/ego_lit/papers/<key>.txt` | one place (regenerable from the banked PDFs with `pdftotext -layout`) |
| **newly banked primary**: GTRS `2506.06664` | `repo: TanitAD Research Lab/Library/papers/2506.06664_*.pdf` + `library.json` + regenerated `LIBRARY.md` | no |
| `--cited-by` recorded on 20 already-banked primaries (tag `ego-input`) | `repo: TanitAD Research Lab/Library/library.json` | no |
| KB findings | `repo: …/Architecture & Inference/Research/KNOWLEDGE_BASE.md` (appended) | no |
| register rows `H-EGO-LIT-1`, `H-EGO-LIT-2`, `H-EGO-LIT-3`, `D-EGOLIT-WITHHELD1`, `H-EGO-LIT-4` (pre-registration) | `repo: Project Steering/GOALS_AND_CLAIMS.md` (inserted after `H-ECHO-9`, same table; `D-EGOLIT-WITHHELD1` was first written as `D-REFCV4B-EGODROP2` and renamed the moment the Master Mind's row of that id appeared at the tail of the file) | no |
| `kb_add.py --verify` | run at the end of the WP; outcome recorded in the final report to the caller (the mount flaps; a verify that could not read a file is reported as UNREADABLE, never as a mismatch) | — |
| scripts used (`fetch_code.py`, `fetch_code2.py`, `bank_cites.py`, `gen_sources.py`) | `devbox: C:/Users/Admin/ego_lit/` | one place (tooling, not results) |

**Escalations (integration, not "please merge in a README"):**
- The pre-registration in §3.6 needs an owner and a trainer flag (`--withheld-bank`); it is 0 pod
  GPU and reuses the `H-ECHO-8` rig.
- KB line and registry phrasing for VAD (§4.1) and the EPDMS-64.0 note (§4.2) should be
  propagated by whoever next touches `MODEL_REGISTRY.md`'s comparator rows.
