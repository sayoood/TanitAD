# refcv6 — the grounded design: every DiffusionDrive piece, trained with our hierarchy

**Master Mind, 2026-09-15 (Europe/Berlin).** Status: **DESIGN for PI approval.** Nothing launched, no model code changed. Every number carries its evidence class. Companion: `Project Steering/Reports/2026-09-15-1445-refcv6-refav1-v7-status.md`.

**Why this document exists.** Your directives, verbatim:

* **2026-09-11:** *"My goal is to implement all pieces of diffuision drive and combine them with all our features like multuhierarchacy etc."* (`PI_DECISION_QUEUE.md:21-25`)
* **2026-09-13:** *"I prefer to finalize the sam 3 map generation and then use those as gt for our new bev head…"*
* **2026-09-15:** SAM3 maps for the complete corpus, pushed with every other augmentation (production running, all clips by ~22 Sep).

The 2026-09-10 refcv6 (`REFCV6_ARCHITECTURE_REVIEW.md`, `PREREG_REFCV6.md`) carried only DiffusionDrive's **planner-side** pieces; the claim that it carried all of them was retracted on 09-13. **This design adds the perception grounding and makes refcv6 what you asked for.** The 09-10 pre-registration stays as written, marked superseded.

---

## 0. In one page

**Thesis.** The REF-C line fails in two measured places — **placement/speed** (the hold-action control beats every trained arm on speed, along-track and acceleration) and **grounding** (the trunk exposes almost no metric scene structure). DiffusionDrive's missing pieces target exactly these, and for the first time we have the supervision for them at corpus scale: **SAM3 semantic maps (4,719 clips), obstacle tracks (4,585), LiDAR BEV ground truth (315, as an independent reference), v8.1 max-speed and nav labels.**

**The model (≈ 110–112 M, ESTIMATED; counted at build):**

```
3-frame front camera 256×640 ─► REF-C ResNet trunk (90.5 M, EXISTS) ─► stride-16 [16×40] + stride-32 [8×20] features
                                        │
                  BEV lift (NEW): multi-height bilinear sampling of the stride-16 map into the 120×64 @ 0.5 m rig grid
                  (precomputed per-clip geometry from calibration; zero learned geometry parameters)
                                        │
                        BEV encoder (NEW, small conv) ─► BEV features 120×64 ─► BEV tokens 30×16
                        ├─ MAP head (NEW) ── supervised by SAM3 maps: 9-channel soft cell fractions
                        └─ AGENT head (EXISTS: DETR slots + Hungarian loss) ── memory switched to BEV tokens, supervised by obstacle tracks
                                        │
   nav + max-speed + v0 ─► CONDITION (strengthened: FiLM into every anchor query, NEW)
                                        │
   117 v0-conditioned anchors ─► truncated diffusion decoder (EXISTS, control-space DDIM)
        each layer: image-token CA (EXISTS) · agent CA indexed by waypoints (EXISTS, WP-B) · BEV spatial CA at the anchor's waypoints (NEW)
        ─► matched-anchor L1 + focal scoring (EXISTS; no denoising term, as in DiffusionDrive)
   hierarchy heads: strategic route (771 params, 30 s label) · tactical 22-token goal (weight 0.05) (EXIST)
```

**Training:** warm-start every arm from refcv5-v2's final checkpoint, 12,000 further steps, identical data for every arm (v8.1 labels, agent join, SAM3 maps).

**Arms:** control `G0` · full system `GF` · replicate `GFb` · then knockouts that each remove **one** piece from `GF` (map information, agents, max-speed, our denoising term).

**The bar adds what refcv6 never had:** beat hold-action **on the longitudinal family**, **use the nav command**, and show the map head sees the scene — on top of the executable 09-10 bar (ADE vs both controls, lateral and strategic non-regression, replicate floors).

**Compute:** Thor is producing the maps until ~22 Sep; Thor-only fits G0 + GF + GFb + 2–3 knockouts by ~29 Sep, then the evaluation panel and (if you want it) closed loop before 5 Oct. A pod doubles the arms.

**Decisions for you (§10):** approve this design; pod or Thor-only; warm start; closed loop for 5 Oct.

---

## 1. What the evidence requires — the design drivers

| # | driver | evidence (MEASURED unless marked) | what it forces in the design |
|---|---|---|---|
| D1 | **Longitudinal is the failure.** `ha` beats refcv5-v2 and refcv4b on speed MAE (0.2540 vs 0.2919/0.2900), along-track (0.2348 vs 0.2655/0.2544), accel (0.3166 vs 0.3596/0.4346) | T1, 4,823 windows / 141 episodes, `…/2026-09-07-refcv5-v2-comparison/raw/` | a **longitudinal bar** (G-LON), and the levers aimed at it: agents (closing rate), max-speed input, no denoising term |
| D2 | **Lateral is the win to keep.** refcv5-v2 curvature 0.512× the straight-line floor (refcv4b 1.198×) | same panel | lateral non-regression clauses stay binding |
| D3 | **The trunk is not grounded.** A BEV transformer on the frozen refcv5-v2 trunk: test AP 0.4140 vs zero-information 0.3801; stride-16 tokens, azimuth prior and an unfrozen last stage do not move it | `E-BEVHEAD-FROZEN-1`, `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md:245-296,435-453` | grounding must be **trained with the planner**, not read out |
| D4 | **Data is the lever that moved grounding.** +181 clips → AP 0.4352 [0.3976, 0.4737], clears B1/B2 on two seeds | same, §6.2b | supervise with **SAM3 maps on all clips** (15× the LiDAR train clips) |
| D5 | **Auxiliary tasks compete for the trunk.** DD's agent head failed at 17 M (deranged join reproduced the cost); WP-D's BEV aux head cost lateral accuracy at 108 M (heading 3.3×, curvature 6.0×, yaw-rate 3.5× the floor) | `D-P1-AGENTCOND-1`; `…/2026-09-07-wpd-bev-aux/PANEL_RESULT.md` | the **gradient-conflict detector** of `E-BEVLIDAR-CAP-1` in every arm; a **zero-information (shuffled-map) knockout** |
| D6 | **refcv5-v2 does not use its nav command.** `os − os_navzero` −0.0059 [−0.0125, +0.0005] vs refcv4b −0.0961 [−0.1102, −0.0813] | raw `paired_os_minus_navzero`, T1 − T1 | nav + max-speed **FiLM into every anchor query**; a **NAV bar** |
| D7 | **Selection leaves 0.0775 m on the table.** `oracle_sel − os` −0.0775 [−0.0937, −0.0635] | raw `paired_oraclesel_minus_os`, T0 − T1 | map- and agent-aware **scoring features** (scorer reads BEV CA output) |
| D8 | **18,472 parameters never trained** (offset_head, tac_goal_tok_head, scorer.goal_point) | checkpoint optimizer state, `…/2026-09-11-untrained-param-gate/GATE_RESULT.md` | `--w-tac-goal 0.05`; `scorer.goal_point` supervised or removed; the post-training gate is a launch precondition |
| D9 | **One-seed panels lie on this rig.** A zero-lever replicate read "separated" on 5 of 9 metrics | WP-D `D0b`; `H-ESTIM-SEED-1` | a replicate arm from the start; every effect stated against the same-panel floor |
| D10 | **The SAM3 GT is non-causal and imperfect.** Near-field road in front of a queued car is "seen, no class"; night quality lower; 9 channels as soft fractions | `…/2026-09-13-sam3-only-road-map/RESULT.md` §21.4 | label-only (never an inference input); loss only on seen cells; soft targets; LiDAR GT as an independent reference |

---

## 2. The architecture

### 2.1 Trunk — EXISTS

`refc.ResNetEncoder`: 9 input channels (3-frame stack, current frame last), base width 88, blocks (3, 6, 16, 6), **90,458,632 params**; stride-32 map [704, 8, 20], stride-16 map [352, 16, 40] (MEASURED, `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md:220-223`). Input 256×640 cylindrical, f_ref 305.577, per-clip cy crop. ⛔ Vision only at inference.

### 2.2 BEV lift — NEW (≈ 0.1–0.7 M params)

* **Grid:** the SAM3 GT's Cartesian grid, **120×64 cells at 0.5 m** (x 0–60 m ahead, y ±16 m, rig frame, +y LEFT). The label and the prediction live on the same cells — no resampling between them.
* **Method:** for each BEV cell and each of **4 heights** above the road plane (0, 0.5, 1.5, 2.5 m), project the 3-D point through the clip's extrinsics into the **cylindrical** image model (column linear in azimuth, `az = (u − W/2)/f_ref`) and bilinearly sample the stride-16 feature map (352-d); concatenate over heights → 1×1 conv → 128-d. Cells outside the 120° field get a learned "unobserved" embedding.
* **Why this and not lift-splat or cross-attention:** the geometry is exact and parameter-free (the planner cannot overfit it), multi-height sampling covers raised objects without a depth distribution, and `E-BEVLIDAR-CAP-1`'s sizing showed the cross-attention route costs 12–59× the aux head that already competed with the planner (MEASURED, `PREREG_BEV_CAPACITY_COMPETITION.md:51-68`). Multi-height bilinear sampling is the "Simple-BEV" family (PUBLISHED; ⚠️ primary to be banked in the Library before this design is quoted outside the repo).
* **Geometry is precomputed per clip** (two rigs, per-clip cy and pitch) at dataset load and passed as a sampling grid; it never depends on labels.
* ⛔ **Orientation guard (gate C8):** the lift must place SAM3's drivable cells under the image's road pixels and must go RED on a left-right mirrored index (the `R-2026-09-08-wpa-mirror` class).

### 2.3 BEV encoder and map head — NEW (≈ 1–2 M)

* BEV encoder: 3 residual conv blocks at 120×64 (128-d), then a 4× downsample to **30×16 BEV tokens** (480 tokens, 256-d) for attention.
* **Map head:** upsample to 120×64 → 9 logits per cell. **Loss:** soft cross-entropy against SAM3's 9 cell fractions, **only on cells with ≥ 50 % seen**, weight `w_map` (set by the smoke's loss-budget check, §6, not guessed). Training GT = `semantic_maps/gt/` only; the flagged tier is excluded.
* The map head is **label-only**: its logits are an auxiliary output; the planner reads BEV **features**, never the map labels.

### 2.4 Agent head — EXISTS, memory switched to BEV

`AgentSlotDecoder` (DETR slots + Hungarian-matched box/class loss, `agent_slots.py:256-523`) is defined over a spatial `memory` that is currently the image tokens. **Change:** its memory becomes the **480 BEV tokens** (with their metric positional encoding), as in DiffusionDrive. Supervision: obstacle join md5 `1c985e6d…` (96.83 % of train clips), `--w-agent 1.0` (pre-registered by the P1 gate). ⚠️ The 17 M failure was capacity competition; at 108 M with BEV memory it is a **hypothesis**, which the `GF−agents` knockout tests.

### 2.5 Condition — STRENGTHENED

Today nav (a 4-way one-hot) enters once, folded into the pooled condition (`refc.py:3208-3211`), and refcv5-v2's sampler/selection path ignores it (D6). **Change:** nav command, **max-speed** (`--max-speed-input`, v8.1 `speed_max_input`, units m/s declared) and v0 are embedded and applied as **FiLM on every anchor query at every decoder layer and on the scorer**. ⚠️ Both nav and the max-speed training value are derived from the ego's future (`provenance: ego-future`, `oracle: true`) — stamped in every run record, as the v8.1 card requires.

### 2.6 Planner — EXISTS + one NEW coupling

* Truncated anchor diffusion over 117 v0-conditioned `alat` anchors, control-space DDIM, P14 emitted-fan ranking, P12 `a_star` fix, P5 refinement — **inherited unchanged**.
* Cross-attention per decoder layer:
  1. image tokens (EXISTS);
  2. **agent slots indexed by waypoint** — WP-B's `refc_wp_index.py` (+2,208 params, EXISTS, never run; requires agents on);
  3. ⭐ **BEV spatial cross-attention at the anchor's waypoints — NEW.** For each anchor query, sample the 120×64 BEV feature map bilinearly at its K = 10 waypoint positions plus P = 4 learned offsets each (deformable sampling, DiffusionDrive's trajectory-indexed spatial attention); attention-weight the 40 samples into the query. ≈ 0.3 M params.
* **Scoring** reads the same waypoint-sampled BEV and agent features (D7).
* **Losses (DiffusionDrive's):** matched-anchor L1 + focal scoring; **no denoising MSE** (`--w-u0 0`; DD-v1 has none, `D-DDV1-NO-DENOISING-LOSS`), behind the existing `--ack-ddim-no-u0` record flag.

### 2.7 Hierarchy — EXISTS, now actually trained

Strategic route head (771 params; 30 s label as supervision, nav kept as conditioning — PI 09-05) · tactical 22-token goal head with **`--w-tac-goal 0.05`** (measured recommendation, item 10) · `scorer.goal_point` either given its FDE loss or removed (a dead head may not ship, D8).

### 2.8 Parameter budget

108,257,502 (refcv5-v2, MEASURED) + lift ≈ 0.1–0.7 M + BEV encoder/map head ≈ 1–2 M + BEV spatial CA ≈ 0.3 M + FiLM ≈ 0.2 M ⇒ **≈ 110–112 M (ESTIMATED)**, far under the 300 M ceiling. Counted with `sum(p.numel())` at build and printed into `config.json`.

---

## 3. DiffusionDrive, piece by piece — what refcv6 now carries

| DiffusionDrive piece | refcv6 (09-10) | refcv6 grounded (this design) |
|---|---|---|
| truncated diffusion from anchors | ✅ | ✅ inherited |
| cascade decoder with CA into perception | image tokens only | ✅ image + **BEV spatial CA at waypoints** + agent CA |
| BEV perception branch | ⛔ absent | ✅ **BEV lift + encoder** |
| map segmentation supervision | ⛔ no map existed | ✅ **SAM3 maps, all clips** |
| learned agent detector (Hungarian) | built, off | ✅ **on, over BEV memory** |
| trajectory-indexed agent attention | built, never run | ✅ on |
| matched-anchor L1 + focal scoring, no denoising loss | u0 0.5 (our invention) | ✅ **u0 0** (faithful); `GF+u0` tests ours |
| ego status input | ✅ v0 at t0 | ✅ + max-speed, nav as FiLM |
| **V2:** RL with ≥GT truncation | pilot built, T0 only | ✅ **stage 2** on the best SFT arm, if compute allows before 5 Oct, else October |
| **V2:** multi-candidate selector | library built, no caller | ✅ scorer with BEV/agent features; replay selector as stretch |
| **V2:** map-based metrics (drivable-area compliance) | ⛔ "PhysicalAI has no map" | ⭐ **now possible at label time** from SAM3 drivable cells — as an RL reward term and a scorer target, never an inference input |
| **V2:** per-step chain REINFORCE, PDM simulator reward, 800-candidate fan | ⛔ | ⛔ still not implementable (no `log π` chain in our decoder; no simulator; no stochastic generator of that size) |

⇒ **Everything DiffusionDrive does that our data can supervise is in; the three remaining pieces are recorded with the reason, as before.**

---

## 4. Data

| component | use | coverage / identity |
|---|---|---|
| camera v2ep cache (256×640 cylindrical) | input | 4,572 train / 147 eval clips |
| v8.1 labels | planner, tactical, strategic, nav, max-speed | train md5 `b45377a1…`, eval `eefc38d1…` |
| **SAM3 maps** `semantic_maps/gt/` | **map head target** | all clips by ~22 Sep; ⛔ gate C7 refuses < 90 % of train windows |
| obstacle join | agent head target | md5 `1c985e6d…`, 96.83 % of train clips; ⛔ C4 refuses < 90 % |
| calibration | BEV lift geometry | `calibration/` on HF, 4,719 clips |
| **LiDAR BEV GT** | ⛔ **evaluation reference only** (134 eval clips) | never a training target in these arms — it stays independent of what the map head learned |

⚠️ **The map labels are aligned by construction:** `semantic_maps` frames are the v2ep episode frames (`t_query = linspace(t_cam[0], t_cam[-1], int(span_s·10))`), so a training window's frame index is the label's `axis0` index. ⛔ Gate C7 asserts it on real batches (`t_img_us` of label == frame timestamp of the window) before any arm trains.

---

## 5. Arms, ordered by what 20 days can buy

**Common:** warm start from refcv5-v2 `ckpt.pt` (md5 `9405ec73…`, currently on the dev box only — copied and md5-verified to the training box), **12,000 further steps**, v8.1 labels, agent join and map loader on in **every** arm (loss weights select the pieces), seed 0 unless stated. New parameters initialised fresh; loading is strict with an explicit allow-list of new keys.

| priority | arm | differs from its pair by | pairs against | question |
|---|---|---|---|---|
| 1 | **`G0`** | — (planner only: map/agent/BEV-CA weights 0, FiLM off, u0 0.5 as refcv5-v2) | — | what 12 k more steps alone do |
| 1 | **`GF`** | the full grounded system (§2) | `G0` | ⭐ does grounded refcv6 clear the bar? |
| 1 | **`GFb`** | seed 1 | `GF` | the training-seed floor every effect is read against |
| 2 | `GF-mapshuf` | SAM3 targets **shuffled across the batch** (same parameters, same gradient path, zero information) | `GF` | does the **map information** pay, or only the task? |
| 3 | `GF−agents` | agent head + agent CA off | `GF` | does agent grounding buy the **longitudinal** family? |
| 4 | `GF−speed` | max-speed input off | `GF` | the longitudinal input's share |
| 5 | `GF+u0` | `--w-u0 0.5` restored | `GF` | our denoising invention (09-10 arm D, inverted onto the full system) |

⛔ **`GF − G0` is the system effect and is deliberately NOT attributed to any single piece.** Attribution comes only from the knockouts, each of which moves exactly one piece (checked on parsed namespaces with the existing `check_one_variable.py`, extended with the new flags and the same mutation proof).

**Cost, Thor (ESTIMATED):** 12,000 × ~5 s/step (4.09 s MEASURED for the refcv6 base + ~20 % for the BEV branch, re-measured in the gate smoke) ≈ **17 h per arm**. Priority 1 = 51 h; all seven = ~119 h.

---

## 6. The bar — committed before any arm trains

Tier **T1**, paired episode-cluster bootstrap, four families never pooled, `n` windows and episodes in every table; floors from `GFb − GF` in the same panel. The **09-10 executable bar stays** (`verdict_refcv6.py`, with its stale "refcv4b" literals replaced by the re-read values before scoring):

| clause | family | `GF` must |
|---|---|---|
| P1 / P2 | ADE | beat `ha0_ext` **and** `ha`, separated, ≥ 0.10 relative margin |
| P3 | ADE | effect ≥ 3× the replicate floor |
| L1–L4 | lateral | no separable regression vs `G0` on heading, yaw-rate, cross-track, masked curvature (reported beside the straight-line floor) |
| S1 | strategic | route accuracy not separably worse than 0.7791, n > 0 |
| T1 | tactical | reported with n; `--w-tac-goal` head trained (post-training gate) |
| ⭐ **G-LON (new)** | longitudinal | **speed MAE and along-track separated BETTER than `ha`** — the do-nothing bar no arm has cleared |
| ⭐ **NAV (new)** | conditioning | `os − os_navzero` separated below zero (the nav command is used) |
| ⭐ **MAP (new)** | perception (no tier) | map head on the 147 eval clips beats the per-cell train prior on drivable and lane-line IoU, separated (clip bootstrap); **and** against the independent LiDAR GT on 134 eval clips, cells predicted drivable are LiDAR-occupied no more often than SAM3's own GT cells are |

⛔ **SUCCESS = all clauses.** Clearing ADE while missing G-LON is a **FAIL as written** — that is the trade refcv5-v2 made.

**Knockout reading:** a knockout "matters" only if its delta vs `GF` is separated **and** ≥ 3× the `GFb` floor, family by family.

**Failure twins, each naming its next lever:**

| if | then |
|---|---|
| `GF` misses G-LON, `GF−agents` ≈ `GF` | agents are not the longitudinal lever ⇒ **RL stage with the ≥GT bar** (targets the along-track axis; reward adds map drivable-area compliance and agent collision from labels) |
| `GF-mapshuf` ≈ `GF` | the map task, not its information, drives any change ⇒ shrink the map head / stage it after the planner converges (`E-BEVLIDAR-CAP-1` twin G1) |
| lateral regresses (L1–L4) with the conflict detector negative early | capacity competition confirmed at 108 M ⇒ **gradient projection on the trunk** or a detached-map variant, pre-registered separately |
| MAP fails | grounding did not form ⇒ check the lift (C8), then data weight; ⛔ planner results of that arm are not credited to grounding |
| NAV fails | the FiLM path is not reaching the sampler ⇒ nav-dropout-free ablation and an inference trace, before any retrain |
| everything inside the floor | **UNDERPOWERED, not refuted** ⇒ seeds, not windows |

**Early detector in every arm with an auxiliary loss** (`E-BEVLIDAR-CAP-1` §4): `cos(g_traj, g_aux)` on the shared trunk parameters every step, with its three analytic controls (+1 self, 0 detached, −1 negated) and the 30×-loss mutation that must fire. Logged, reported, **not** a stop rule.

---

## 7. The gate before any GPU-hour

The 09-10 pre-launch gate (C1–C6, JSON verdict, never the exit code) plus:

| check | refuses when |
|---|---|
| **C7** map coverage + alignment | < 90 % of train windows have a map frame, or any sampled window's label `t_img_us` ≠ its frame timestamp |
| **C8** lift orientation | the lift does not put SAM3 drivable cells under road pixels on 20 sampled frames, or a mirrored index passes |
| **C9** removability | with every new weight 0 and the new modules bypassed, the warm-started planner output is **bit-identical** to refcv5-v2's on 64 fixed windows |
| **C10** loss budget | in a 400-step smoke, `w_map` / `w_agent` put the auxiliary gradient norm on the trunk above 1× the trajectory gradient norm (the measured `E-DEC-18` failure mode) |
| **C11** untrained parameters | the post-training gate (`check_untrained_params.py`) finds a built head with no optimizer state |
| **C12** throughput | measured s/step on the launching box — the cost table is re-stated before launch |

---

## 8. Plan to 5 October

| dates | Thor | dev box (RTX 4060) |
|---|---|---|
| **15–19 Sep** | SAM3 production | implement §2 (data loader, lift, map head, BEV spatial CA, FiLM, warm start), unit tests, mutation tests; **tiny-rig smoke** (REF-C `--size tiny`, 17 M) on the clips that already have maps: learning signal, conflict detector, removability |
| **20–22 Sep** | SAM3 production | pre-registration committed (this bar, frozen); gate code; copy the warm-start checkpoint to Thor |
| **22 Sep eve** | gate C1–C12 (20/400-step smokes) | — |
| **23–25 Sep** | `G0`, `GF`, `GFb` (priority 1, ~51 h) | T1 panels as each arm lands (~25 min per arm) |
| **25–29 Sep** | knockouts in priority order (~17 h each) | panels, map metrics vs SAM3 and LiDAR |
| **29 Sep–2 Oct** | ⭐ RL stage on the best arm **or** closed loop (your call, §10) | videos, four-family panel |
| **2–5 Oct** | freeze | evaluation write-up |

⚠️ **With a pod from ~17 Sep**, planner-only arms (`G0`, `GF−map`-type) can start before the maps finish and the knockouts double up; the RL stage and closed loop then both fit.

---

## 9. Risks, stated plainly

* **Capacity competition** (D5) is the most likely failure; the conflict detector and the shuffled-map knockout are there to see it.
* **SAM3 label noise** — near-field "seen, no class" in queues, weaker night maps, crosswalk over-segmentation. Soft targets on seen cells limit the damage; the LiDAR reference catches a head that learns the labels' artefacts.
* **Warm start** inherits refcv5-v2's nav blindness and its longitudinal habits; `G0` carries the same start, so every comparison is fair, but a from-scratch run could differ — recorded as untested.
* **The oracle inputs** (nav and max-speed training values are ego-future derived) keep T1 numbers optimistic by construction; stamped in every record.
* **Time.** Implementation is ~5 working days of careful code in a 5,640-line trainer; any gate failure eats Thor days. Priority 1 arms are the minimum that answers "does grounded refcv6 work".
* **One camera.** BEV beyond the 120° field is unobserved by design; the grid's side cells near the ego are filled only by motion in the labels, never at inference.

---

## 10. Decisions for you

| # | decision | options | default if silent |
|---|---|---|---|
| 1 | **Approve the grounded design as refcv6** (supersedes the 09-10 V0/D/A/B/C ladder) | approve · amend | implementation starts on the dev box; **nothing launches** |
| 2 | **Compute** | Thor only (from ~22 Sep) · add one A100/A40-class pod for ~10 days | Thor only |
| 3 | **Warm start** from refcv5-v2 (12 k steps) vs from scratch (40 k, ~47 h/arm on Thor) | warm · scratch | warm start |
| 4 | **29 Sep–2 Oct slot** | RL stage (DiffusionDrive V2) · closed loop on Thor (AlpaSim, restored from the 09-14 archive; 2 renderable scenes) | RL stage; open-loop T1 panel + videos for 5 Oct |
| 5 | `w-tac-goal` | ratify 0.05 · other | 0.05 |
| 6 | SAM3 flagged tier in training | exclude · include | exclude |

---

## 11. What this design does NOT claim, and what it owes

* ⛔ No result of any kind. Every capability statement above is a **hypothesis** with its bar.
* ⛔ The ≈ 110–112 M parameter count and the ~17 h/arm cost are **ESTIMATED** until the build prints and the gate measures them.
* ⚠️ Owed before quoting outside the repo: bank the Simple-BEV and DiffusionDrive V1/V2 primaries through `tools/kb_add.py`.
* ⚠️ Owed before scoring: `verdict_refcv6.py`'s stale literals replaced by the re-read values (status report §5, item 3).
* ⚠️ The 09-10 `PREREG_REFCV6.md` is **not rewritten**; when you approve this design, the new pre-registration `PREREG_REFCV6_GROUNDED.md` is committed with §6 frozen verbatim, before any arm trains.
