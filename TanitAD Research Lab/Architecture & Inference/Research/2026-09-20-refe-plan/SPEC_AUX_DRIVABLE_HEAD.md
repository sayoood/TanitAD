# Auxiliary drivable-area head for REFe (design spec, P7.1)

> Written 2026-10-11. **Design only.** No training, no pod, no GPU. `refe/train.py`, `refe/model.py`, the launch-gate code and
> `eval/P6_GO_SIGNAL.md` are **not edited**. The pre-registration is `eval/PREREG_AUX_DRIVABLE_DRAFT.md` (DRAFT).
> Evidence classes: **MEASURED** (ours, artifact named) · **PUBLISHED** (banked primary, cited by Library key) ·
> **INHERITED** (another doc, not re-verified here) · **ESTIMATED** (arithmetic) · **UNVERIFIED**.
> P = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Prototypes and raw output:
> `raw/2026-10-11-aux-drivable/`.

## 0 · Headline

**Recommendation: (b), a separate arm. P6 stays paper-faithful and head-less.** The head is built and gated in parallel at
0 GPU, then tested at a scale that gives attribution (§5). Putting it inside P6 (a) costs ~1–3 h of the 169 h
(ESTIMATED), but it turns a 7-day, one-seed run into a 7-lever bundle that cannot say which lever did what.

| item | value | class |
|---|---|---|
| the head | 2,440,450 params (+12.9 % of the 18,991,426 trainable); 32.2 GFLOP fwd / 96.7 GFLOP fwd+bwd per sample = **0.53 % / 0.79 %** of the trunk's fwd / fwd+2×fwd | MEASURED (`head_proto_selftest.out`) + analytic attention term |
| step-time cost on the A40 | **+0.8 % to +1.6 %** (FLOP share; ×2 for lower MFU of small attention) = +1.3 to +2.7 h of 169 h | ESTIMATED; no GPU benchmark was run (the dev-box GPU is shared and contended, §3.4) |
| label build, navtrain (103,288 tokens) | **1.1–2.0 CPU-hours** single process (median 38.5 ms, mean 67.8 ms per warm sample); ~8–15 min on 8 workers | MEASURED on 20 samples / 4 maps, extrapolated |
| label storage | 20,480 B/frame raw int8 → **2.115 GB**; zlib 3,247 B/frame → **0.335 GB** | MEASURED per frame, ×103,288 |
| label fidelity vs NAVSIM's exact polygon verdict | corners **99.03 %** (nearest cell), **99.12 %** (bilinear SDF); whole-proposal verdict **97.7 % / 98.1 %**; every disagreeing corner within **0.22 m** of the true boundary | MEASURED, 42,640 corners / 260 verdicts, 20 samples |
| expected effect on u400 | rescuing 30 % / 50 % / 100 % of the DAC-zero tokens is worth **+1.0 / +1.6 / +3.3 PDMS** (upper bound: rescued tokens score like the rest) | ESTIMATED arithmetic, §1 |
| published seed floor | NAVSIM TransFuser, 3 training seeds: **std 0.56 PDMS** (other model; ours is UNMEASURED) | PUBLISHED, `2406.15349` Table 2 text |

⚠️ **The head alone cannot close the 6.30-point gap.** Its ceiling is +3.3 (the DAC-zero part of the 4.15 zero tail); the
realistic range (+1.0 to +1.6) is **1.8–2.9× the published seed std**. A single-seed run cannot certify a lever of that size.

### Decisions needed (Master Mind / PI)

| id | decision | recommendation |
|---|---|---|
| **D-AUX-1** | (a) head inside P6, or (b) separate arm | **(b)**; (a) only if the PI accepts losing attribution for one shot at 94.55 (§5) |
| **D-AUX-2** | start the label store build now (CPU only, dev box, ~1–2 h, 0.34 GB) | **yes**: it blocks every variant, costs no GPU |
| **D-AUX-3** | a pod slot for the graft arms (~40 A40-h for head + control, ESTIMATED from P2's ~20 h/arm) | decide after S1 reads (§5) |
| **D-AUX-4** | register the proposal-level field loss (§8) as its own arm | **yes**: it has the largest controlled published effect (+2.4 PDMS) and needs no extra labels |

### Conflicts found (reported, not fixed)

1. **`MODEL_REGISTRY.md` has no §14.2 and no SFT-4 / u400 row.** §14 ends at 14.1 and `grep -c "88\.25"` on the registry = 0.
   Every SFT-4 number below is from `eval/RESULT_SFT4.md` and `eval/P6_GO_SIGNAL.md`, whose gap decomposition I re-read in
   its raw output (`raw/2026-10-11-ep-gap-decomposition/ep_lowtail.out` lines 1–2: 4.15 / 1.31 / 0.86; DAC = 0 on 3.42 %).
   **ESCALATE:** the registry needs the u400 row before any report quotes it.
2. **`eval/REFE_LAUNCH_GATE.md` does not exist yet.** Gate entries (§7) are named after `SPEC_P6_PAPER_FAITHFUL.md` §5.
3. **`SPEC_P6` §6 says "G-DVB already guards its one rule: the aux output never feeds the planner."** That is a P6 rule.
   The PI's 2026-10-11 ruling admits a vision-predicted field at inference, so the feed variant (§2.4) needs that entry to
   become a *declared value checked in both directions*, not a prohibition.
4. **REVIEW_7 P7.1's "+4 if DAC fail drops from 6.3 % to ≈1.5 %" is stale for u400.** u400's DAC-zero share is 3.42 %, so
   the ceiling is +3.3.
5. `D-REFE-SLOWPICK-SFT4-1` is a *proposed* row in `Research/2026-10-11-refe-data-and-rl/RL_POSTTRAIN_REVIEW.md`
   (MEASURED there, `raw/pick_length_telemetry.json`); it is not yet in `GOALS_AND_CLAIMS.md`. INHERITED here.

## 1 · Why this head (the evidence it rests on)

| fact | value | class |
|---|---|---|
| human 94.55 − u400 88.25 | **6.30 PDMS** = zeroed tokens (NC×DAC = 0) **4.15** + TTC 1.31 + EP 0.86 + comfort −0.01 | MEASURED (`ep_lowtail.out`) |
| DAC = 0 / NC = 0 / TTC = 0 on u400 | **3.42 %** / 0.96 % / 4.37 % of 12,146 tokens | MEASURED (same file) |
| u400 sub-scores vs the paper's student | NC 98.94 / TTC 95.63 / **DAC 96.58** / EP 81.20, against 98.80 / 95.74 / **99.25** / 92.72 | MEASURED / PUBLISHED (Table A13) |
| tracker-made share of student DAC failures | **188 of 643 = 29.2 %** repairable by plan compensation; the other 71 % are real path errors | INHERITED (`eval/RESULT_P4_TEACHER_NAVTEST.md` F2; raw `raw/2026-10-05-teacher-navtest/f2_result.json`) |
| dirty fans | on 15.2 % of tokens < half the proposals are NAVSIM-safe; they hold 72 % of all zero picks | INHERITED (REVIEW_7 §0 item 6) |
| the missing information is visual | a cross-fitted re-ranker with NAVSIM labels but **no images** recovers only +1.12; within-set DAC AUC 0.75 → 0.81 | INHERITED (REVIEW_7 §0 item 5) |
| the pick is still short | 79.7 % of 10,070 comparable tokens | INHERITED (`D-REFE-SLOWPICK-SFT4-1`, proposed) |

**Arithmetic for the ceiling (ESTIMATED).** The DAC-zero tokens are 415 of the 529 zeroed tokens (3.42 % × 12,146), i.e.
415/529 × 4.15 = **3.26 PDMS** of the zero tail. A head that rescues a fraction x of them, with the rescued tokens scoring
like the rest, gains ≈ 3.26·x: x = 0.3 → +0.98, 0.5 → +1.63, 1.0 → +3.26. (NC-only zeros and the TTC/EP terms are not
touched by a drivable-area signal.)

### Published precedent (all PDFs are in `TanitAD Research Lab/Library/papers/`; sha256 in `library.json`)

| source (library key) | setting | controlled effect | class / caveat |
|---|---|---|---|
| NAVSIM Table 2 (`2406.15349`) | TransFuser, camera + LiDAR, BEV-semantic aux removed (E1) | **81.6 vs 83.3 / 84.0 / 84.4** (3 seeds, std 0.56): about **−2.3 PDMS**, DAC 90.5 vs 91.3–93.0 | PUBLISHED; E1 is one run; BEV semantics are in the LiDAR frame |
| ReflectDrive-2 Table 6 (`2605.04647`) | DAC **field loss** on the waypoint distribution, GT BEV drivable indicator → distance field | **84.8 → 87.2 (+2.4), DAC 93.9 → 97.0** | PUBLISHED; one run; an **output-side** loss, not a feature head (§8) |
| Hydra-MDP++ Table 3 (`2503.12820`) | ResNet34, perception tasks as aux supervision (P) | **86.6 → 86.1 (−0.5), DAC 96.0 → 95.6**; text: "auxiliary training of perception worsens the planning performance" | PUBLISHED; one run; the **only negative** |
| TransFuser Table 7 (`2205.15997`) | CARLA Longest6, 4 aux tasks | none 44.29 DS; all losses 49.49–56.68 (worst–best seed); removing HD map alone 50.96, **inside the seed range** | PUBLISHED; closed loop; the paper itself says single-task removals are not separable from seed noise |
| WoTE Table 3 (`2504.01941`) | BEV future states (semantic-map supervision) feeding a reward model | 83.2 → 85.6 (+2.4), DAC 92.9 → 94.7 | PUBLISHED; confounded: the future-state predictor, not a map head alone |
| DriveZero §2.2 (`2609.06055`) | the paper we reproduce | **deliberately uses no annotated aux heads** ("replaces manually annotated auxiliary tasks with frozen foundation models") | PUBLISHED; so this head is a **declared departure** from the paper recipe |
| DiffusionDrive (`2411.15139`) | follows TransFuser: 3D detection + BEV semantic aux, feeds BEV features to the decoder | no ablation of the aux | PUBLISHED |

Banking: all seven PDFs were already in the Library; `tools/kb_add.py` recorded this spec and the pre-registration as `cited_by` and the tag
`aux-drivable-head` (`library.json`, `LIBRARY.md` regenerated, both staged), and each PDF's sha256 was re-checked by content: 7 of 7 match.
A full `kb_add.py --verify` of all 614 files (4.2 GB) timed out at 170 s and was **not** run to completion.

⇒ The literature is split (−2.3 vs −0.5 sign, +2.4 on a different mechanism). It supports a pre-registered test, not an
assumption of gain. I read the banked PDFs' text for this table; tables were re-extracted in raw order and cross-checked
against each paper's prose (the NAVSIM and ReflectDrive-2 table layouts scramble under `pdftotext -layout`).

## 2 · The head

### 2.1 Where it attaches (read from `refe/model.py`)

REFe's forward (`model.py:692-770`): trunk → `visual` [B, 4×1920, 1024] + 3D position → `reg_compress` → `scene` [B, 64, 1024]
→ `scene_proj` → `scene_ctx` [B, 64, 256] (read by the 4-layer proposal decoder) and `visual_ctx = scene_proj(visual)`
[B, 7,680, 256] (read by the scoring decoder, and **un-detached in P6**: the detach is `model.py:759`, `scene_ctx` / `visual_ctx` are formed at `:739-740`).

**Primary tap: `visual_ctx`.** Reasons: (i) it is the tensor the scorer cross-attends, and the scorer is where the cross-fit
evidence says visual information is missing; (ii) with P6's un-detached context the aux gradient and the PDM gradient meet
in the same tensor; (iii) it reaches `scene_proj`, `pos3d_mlp` and the LoRA but **not** `reg_compress` or `traj_head`
(proved by the exact-0.0 controls in `head_proto.py`). **Registered variant: `scene` tap** (K/V = the 64 scene tokens),
which forces the compression to keep the layout the proposal decoder needs. It is a separate arm, never mixed in.

### 2.2 Architecture (`raw/2026-10-11-aux-drivable/head_proto.py`, reference code)

- 40 × 32 learnable BEV queries (2 m cells over x ∈ [−8, 72), y ∈ [−32, 32)) + a Fourier encoding of the cell centre.
- 2 × `CrossBlock(256, 8 heads, ctx 256)` (the class `dec` and `score_dec` already use): self-attention over the queries, cross-attention to `visual_ctx`, MLP.
- Output per cell: a mask logit and a signed distance `6 m · tanh(·)`.
- **Inputs are image tokens only.** No ego, goal, navigation command or map. This is what makes it admissible (§6).
- MEASURED (selftest): 2,440,450 params; output (1, 40, 32); FLOPs as in §0.

**Why a signed distance field and not a mask (MEASURED, `resolution_ceiling.out`, 20 samples, 260 proposal verdicts).** A
perfect head at cell size c gives NAVSIM's exact verdict at the footprint corners this often:

| head output | corner agreement | whole-proposal verdict agreement |
|---|---|---|
| SDF, 0.5 m cells (the label itself) | 99.12 % | 98.08 % |
| SDF, 1 m cells, bilinear | 99.31 % | 98.08 % |
| **SDF, 2 m cells, bilinear (chosen)** | **99.26 %** | **98.08 %** |
| SDF, 4 m cells | 98.57 % | 96.92 % |
| mask fraction, 2 m cells, nearest | 96.05 % | **90.00 %** |

A 2 m SDF loses nothing against the native grid; a 2 m mask loses 8 verdict points. The field is smooth (1-Lipschitz), so
a coarse grid plus interpolation holds the boundary to sub-cell accuracy. 4× fewer queries than a 1 m grid.

### 2.3 Loss and weight

`L_aux = BCE(logit, mask fraction of the 4×4 fine cells) + 1.0 · Huber(ŝ/6, s̄/6, β = 0.1)`, mean over cells, over frames with a
label (a frame without one contributes exactly 0). Total loss `l_traj + l_score + w_aux·L_aux` (`train.py:1588` today sums the
first two). **Initial `w_aux = 0.1`**, equal to `score_w`. The weight is set once, before launch, by a **gradient-norm balance**
(G-LIVE reports ‖∇_LoRA L_aux‖ / ‖∇_LoRA (l_traj + l_score)‖ over 8 real micro-batches; accept 0.05–0.5, otherwise move `w_aux`
by ×2 steps), then frozen. It is never tuned on PDMS.

### 2.4 Does the scorer consume the prediction? Two registered variants

- **H1 (representation only, `--aux-drv-feed off`, the default).** The head only shapes the shared tokens. The planner's
  forward and eval path do not contain it. Bit-identical outputs with the head weights removed are a gate (G-EVAL).
- **H2 (feed, `--aux-drv-feed on`).** For each proposal k the scorer receives three scalars, from the **detached** predicted
  SDF ŝ sampled bilinearly at the proposal's footprint corners over the 0.1 s grid: min ŝ, mean ŝ at the centre, share of
  corner-steps with ŝ < 0. They are concatenated to `score_q_mlp`'s input (60 → 63). Detached so the head is trained by map
  labels only and the scorer cannot teach it to game the score. **This is a vision-derived feature at inference**: admissible
  under the 2026-10-11 ruling; no map is read (§6). H2 is only worth building if the perception proof (PREREG §Reads, P1) passes.

## 3 · Labels

### 3.1 Design (MEASURED: `proto_label.py`, `proto_label.out`, `proto_label_results.json`)

- **Frame:** the rear-axle frame at t0 (x forward, y left), the frame of every REFe pose and of NAVSIM's `rear_axle`.
- **Polygons:** exactly `refe/navsim_dac.py:drivable_polygons`: ROADBLOCK + INTERSECTION + CARPARK_AREA around the ego **centre**,
  radius 100 m. That module's agreement with NAVSIM's own check is **100.00 % on 25,600 proposals** (INHERITED, MODEL_REGISTRY §14.1
  departure 6). A footprint corner is inside iff it is in *any* polygon, so a point-in-union raster is the exact label.
- **Raster:** x ∈ [−8, 72), y ∈ [−32, 32), **0.5 m**, 160 × 128 = 20,480 cells. **Stored: the signed distance, int8, 0.1 m unit,
  clipped ±12.7 m**, computed on a 13.2 m-padded grid so cells near the edge see boundaries outside the window. The mask is
  `sign(SDF)` (|SDF| ≥ 0.25 m, so never 0). The head's 2 m targets are 4 × 4 block statistics of this, built in the loader.
- **Extent (MEASURED, `extent_stats.out`, 4,993 rank-0 rows / 54 logs):** the teacher's 4 s endpoint x reaches 49.65 m at p99.9
  (max 53.8 m); |y| ≤ 19.0 m. ⚠️ That shard is low-speed (v0 max 13.1 m/s), so the forward extent for fast tokens is
  UNVERIFIED; +72 m covers a 4 s mean speed of 18 m/s. A footprint corner outside the extent is "unknown": H2 maps it to SDF = +6 m
  with an out-of-extent flag, H1's loss simply has no cell there.
- **No augmentation to invert:** `train.py` has no flip/crop/jitter (grep = 0), so one label per frame is valid for every goal
  twin. Key = `(log_name, token)`; twins share it.
- **Label–image timing skew:** cameras pair to the lidar token within ≤ 60 ms (INHERITED, `navtrain_scenarios.py`), at most ~0.8 m
  at 13 m/s. Below the 2 m cell, above the 0.5 m native grid.

### 3.2 Two build paths, identical output (MEASURED)

| path | needs | result |
|---|---|---|
| A: `NuPlanScenario` (what `onpolicy_label.py` does) | the navtrain log DB + maps | median 61.7 ms, mean 229 ms per sample just to build the scenario |
| **B: bank row `origin_world` + the map** | the map, and the log's map name | **A and B masks differ in 0 of 20,480 × 20 cells; `origin_world` equals NAVSIM's `rear_axle` to 0.0 m and 0.0 rad on all 20** |

⇒ **Build with B.** It needs only the 4 nuPlan maps (`nuplan-maps-v1.0`: Boston, Pittsburgh, Las Vegas, Singapore) and a
log → map-name table, not the log DBs. Path A is kept for **navtest** frames (the head's evaluation labels, 12,146 tokens,
~9–25 min).

### 3.3 Cost (MEASURED on 20 samples in 4 locations; the dev box was running other agents' jobs, so these are upper-ish)

| stage | median | note |
|---|---|---|
| map query (`get_proximal_map_objects`) | 5.8 ms | **cold: 2.2 / 3.1 / 7.2 / 4.8 s the first time per location**, once per worker |
| rasterise 212 × 180 padded points (shapely `contains_xy`) | 29.4 ms | p90 58.9 ms |
| signed distance (`scipy` EDT) | 3.2 ms | |
| **warm total** | **38.5 ms median, 67.8 ms mean, 159 ms p90** (n = 16) | |
| reading a log's map name from its DB | 54.8 ms per **log** | 1,192 logs ≈ 65 s |

**103,288 tokens × 67.8 ms = 117 min single process (46 min at the median, 274 min at p90); on 8 workers ≈ 8–15 min plus
~17 s cold loads each.** Storage in §0.

### 3.4 What was NOT measured

- **No GPU benchmark of the head.** The dev-box GPU was at 61 % utilisation with 5.5 of 8.2 GB held by another agent's job
  (`nvidia-smi`, 2026-10-11 19:05) and a lock protocol governs it (`raw/2026-10-11-c3-selection-headroom/HANDOFF.md`), so a timing
  taken now would measure the neighbour. The cost in §0 is the FLOP share. Activation memory at micro-batch 4 is ESTIMATED at
  ~0.13 GB + 0.04 GB parameters and Adam state, against the MEASURED 21.78 GB peak of the final run; G-LIVE-MEM measures it.
- 20 samples is small. The fidelity numbers carry that n; the full store build must re-run the same check on ≥ 64 samples (G-LIVE).

## 4 · The wiring a trainer author needs (today's anchors; nothing edited)

| change | anchor | note |
|---|---|---|
| config fields `aux_drv`, `aux_drv_tap`, `aux_drv_depth`, `aux_drv_feed` | `model.py:57` `REFeConfig` | declared dataclass fields, or the G-HYG guard (`model.py:187`) refuses them |
| build the head **after** the trunk, from its own `torch.Generator` | `REFe.__init__`, `model.py:469` | the head's init must not consume the trunk's RNG stream, or the "detached ⇒ bit-identical trunk" control cannot hold |
| return the head output | `forward`, `model.py:692-770`; `train.py:1491,1494` | e.g. `traj, score, aux` |
| dataset returns `(label_int8, valid)` | `TargetBank.__getitem__` return tuple `train.py:860-863`; loop unpack `:1481` | the label store is a memmap keyed by `(log_name, token)` |
| loss | `train.py:1588` | `loss = (l_traj + l_score + a.w_aux_drv * l_aux) / a.accum` |
| flags `--aux-drv`, `--w-aux-drv`, `--aux-drv-tap`, `--aux-drv-feed`, `--aux-drv-labels`, `--aux-drv-shuffle-seed` | `build_parser` `:1720`, pattern `:1776` | **add every one to `IDENTITY_ARGS` `:945`**, so a resume with a different head is refused |
| the head's loss and gradient telemetry | `metrics.jsonl` | `aux_bce`, `aux_huber`, per-group grad-norm ratio |

⛔ **`--w-aux-drv` must default to 0.0** (so a head-less arm cannot silently carry one) **and the arm's argv must name it**:
a flag whose default disables it, absent from the argv, is the "wired, tested, unreachable" class (CLAUDE.md, 2026-09-10/11).

## 5 · How it enters P6

P6 is one A40 for 6.8–8.2 days with no margin (`SPEC_P6` §0.2), one seed, and already differs from its comparator C1 in six
levers (from-scratch vs warm start, online PDM labels, un-detached context, command nav, unwrapped yaw, full bank / 25 epochs).

| | **(a) inside P6, declared deviation** | **(b) separate arm** |
|---|---|---|
| compute | +1.3 to +2.7 h of 169 h (ESTIMATED) | **a second full P6 does not fit on one A40**: 13.6–16.4 days sequential, or a second A40 for ~7 days (PI spend). Feasible forms: **(b-i)** dev-box pilot S1; **(b-ii)** graft arms S2 at P2 scale, ~20 A40-h per arm (ESTIMATED from `P6_GO_SIGNAL.md` S1), so ~40 A40-h for head + control |
| attribution | **none.** P6 − C1 is a 7-lever bundle with the head inside; "GOAL MET" would not say whether the paper recipe or the head did it, and a miss would not say whether the head hurt | **clean.** Same init, same recipe, same tokens, differing by the head only; paired |
| seed variance | one seed. The head's expected +1.0 to +1.6 is 1.8–2.9× the published std 0.56, so a single-seed delta cannot be a lever claim (H-ESTIM-SEED-1: separated is necessary, not sufficient) | S1 is cheap enough for **≥ 3 seeds per arm**; S2 for 2 |
| risk | if the head hurts (Hydra-MDP++ −0.5), it costs the whole 7 days and cannot be isolated | bounded to an arm's cost |
| what is claimed | P6 stops being "= paper" in the SPEC's own table | P6's identity is kept; the head is judged on its own |
| weakness | – | a graft on a converged trunk tests a **weaker** version of the representation lever (small LR). The strong version (head from step 0) exists only in (a) or a second full run |

### Recommended ladder (each rung leaves the next one; RULE ZERO)

| rung | what | spend | decides |
|---|---|---|---|
| **S0** | build + gate the label store (§3) and the head code; run the head's G-LIVE on the dev box | CPU ~1–2 h, no GPU, no pod | readiness; D-AUX-2 |
| **S1** | SFT-4 recipe (frozen trunk, scorer fine-tune on the dev box) with the head and the H2 feed, ≥ 3 seeds, vs the banked u400 and a no-head replicate | dev box GPU, no pod | **perception proof** (can frozen u400 features be read into a drivable map; does the feature separate DAC-failing proposals) and a first read of H2. ⚠️ UNVERIFIED: the SFT-4 trainer's interface at the point where `sctx` is formed (`refe/scorer_finetune.py`) was not read here |
| **S2** | graft head (H1 and H2) on the P2 output or the P6 `model_final.pt`, vs a matched no-head control from the same init | ~40 A40-h | the representation lever and H2 with attribution |
| **S3** | include in the next full run as a declared deviation | 7 days | only with S1/S2 evidence |

**If C3 (`P6_GO_SIGNAL.md` §3) reads GENERATION-BOUND**, its committed consequence is that P6 as designed is aimed at the
wrong component and "the map-supervised drivable-area head" goes first: S1/S2 then run **before** P6, and §8's field loss is
the sharper generation-side lever. If C3 reads SCORER-BOUND, P6 proceeds unchanged and this ladder is P7.1 after it, as
`PREREG_P6.md` already says. (C3 was running at 18:56 on 2026-10-11 and had no result at the time of writing.)

**The case for (a), stated fairly:** the PI wants excellent results (RULE ZERO), the zero tail is the largest term, and a 7-day
run is the only place the head acts on the trunk from step 0. If the PI prefers one shot at 94.55 over attribution, (a) is
available: label store ready (S0), +1–3 h, a new launch-gate PASS token (the argv changes), and `PREREG_P6` amended **before
launch** to name the head as a declared deviation and to drop "GOAL MET ⇒ paper recipe works" as an inference.

## 6 · Admissibility check (explicit; binding rulings)

1. **PI 2026-10-11 (this brief):** map-derived drivable area is allowed as a TRAINING label and reward, never as an inference
   input; a head that predicts it from cameras may be used at inference.
   - Inputs of the head at inference: **the four camera images** (through the trunk) and constant BEV coordinates. No map, no map-frame
     ego pose, no route, no lane graph. `DrivableAuxHead.forward(ctx)` takes one tensor.
   - Test to add: `grep -c "map_api\|nuplan\|origin_world" refe/model.py refe/planner.py` must stay 0 (the planner never imports the label path).
2. **PI 2026-09-28 "no HD-map geometry for planning"** (memory `no-hd-map-geometry-for-planning.md`; A8 lane fallback NOT DEPLOYED): not reopened.
   The map reaches only the loss.
3. **Labels may use ego; inference is vision-only (PI 2026-08-03):** the label uses the ego's map pose (privileged) offline only.
4. **Goal ⟂ situation path (PI 2026-08-03):** the head takes no goal, ego or navigation input, so nothing it outputs can have been
   computed from the goal; the goal still enters only through `ego_tok` on the proposal queries (`model.py:746-747`). The trunk is shared
   (as is every aux head's), so the claim to defend is: *the head adds no goal/ego path*, which the exact-0.0 control on `traj_head`
   and `reg_compress` shows for the gradient direction and the module signature shows for the forward direction.
5. **Evaluation labels:** navtest map labels are used only to *evaluate* the head and by the NAVSIM scorer, never to train or select.
   G-LIVE asserts `label_store_keys ∩ navtest_tokens = ∅`. (Log-disjointness of navtrain and navtest is NAVSIM's split definition.)
6. **Tier:** NAVSIM v1 PDMS is an open-loop pseudo-simulation. Nothing here is a closed-loop claim.
7. **DriveZero comparison:** the paper deliberately avoids annotated aux heads (§1). Anything built on this head is "paper recipe +
   a declared deviation", never "reproduction".

## 7 · Gate entries the head needs (LISTED, not implemented; `eval/REFE_LAUNCH_GATE.md` does not exist yet)

Each entry has a non-zero check, an **exact-0.0 control**, and a **mutation arm that must turn it RED**.

| gate | entry | control / mutation |
|---|---|---|
| **G-HYG** | `REFeConfig` and the argparse namespace refuse `aux_drv*` typos | set `cfg.aux_drv_feeed` → raises; remove the guard → the gate FAILS |
| **G-DVB** | `aux_drv` declared = `model.aux_drv_head is not None`; `w_aux_drv` declared = the weight multiplied into the built loss; `tap` declared = the tensor the head's K/V **is** (`data_ptr` equal to `visual_ctx` / `scene_ctx`); `cell`, `x0`, `y0` equal the label manifest's; label manifest sha256 and frame count (103,288) recorded; **`aux_drv_feed` declared = the scorer's input width (60 vs 63) AND there is no autograd path head → `score` when off** (`torch.autograd.grad(score.sum(), head.parameters(), allow_unused=True)` all `None`) | wire the feed with the flag off → FAIL; `--w-aux-drv` default non-zero or absent from the argv of an arm that meant it → FAIL; tap the wrong tensor → FAIL |
| **G-DVB** (shuffle arm) | `--aux-drv-shuffle-seed` set ⇒ the key map is a **derangement**: 0 fixed points, identical multiset of labels | a map with one fixed point → FAIL |
| **G-LIVE** (2 optimiser steps, real data) | `aux_bce` and `aux_huber` present and finite; **head params non-zero gradient**; aux-only backward gives **non-zero gradient on LoRA, `scene_proj`, `pos3d_mlp`** and **exactly 0.0 on `traj_head` and `reg_compress`** (tap = visual); `w_aux_drv = 0` ⇒ every gradient exactly 0.0; initial `aux_bce` ≈ ln 2 ± 0.15; the gradient-norm ratio of §2.3 is reported and in band | `ctx.detach()` mutation → trunk groups read exactly 0.0 (this is `head_proto.py`'s arm, MEASURED RED/GREEN) |
| **G-LIVE** (label alignment) | on ≥ 64 real frames: the raster verdict agrees with `navsim_dac.violations` on ≥ 95 % of shifted rank-0 targets (13 shifts, §3 protocol), and ≥ 90 % of the unshifted rank-0 targets lie inside the label (the teacher leaves the area on 4.97 % [1.30, 10.31], REVIEW_7) | flip the y axis, or offset the label by one frame → agreement collapses, gate must FAIL |
| **G-LIVE** (no leakage) | label store keys ∩ navtest tokens = ∅; label store coverage of training scenes ≥ 99.9 % (a missing label contributes 0 silently, so coverage is *reported and bounded*) | – |
| **G-LIVE-MEM** | peak with the head ≤ the gate's bound; the head adds ≤ 1 GB at micro-batch 4 | – (a measurement) |
| **G-EVAL** | `planner.py` builds the identical model **including** the head from `ckpt_last.pt` (strict load); with `aux_drv_feed off`, trajectories and scores are **bit-identical with the head weights removed** | wire the feed with the flag off → outputs differ → FAIL |
| **G-CKPT** | head parameters and Adam state round-trip; 2 resumed steps bit-identical | drop the head's optimiser state → FAIL |
| **G-CLOCK** | the label for a micro-batch is looked up by the same `(log_name, token)` as its images | shift the lookup by one sample → G-LIVE alignment FAILS |
| **detached control (exact)** | with the head detached (`ctx.detach()`, feed off), **the trunk after N steps is bit-identical to the head-less run from the same seed**; requires the head's own `Generator` (§4) | head drawing from the global RNG → FAIL |
| **post-training gate** | every head parameter has Adam state after training (the `probe_zerograd3.py` pattern: registered vs with-state counts equal) | – |
| **G-SUITE** | `head_proto.py` selftest and the label-fidelity check run in the clean-tree suite | – |

## 8 · Companion lever B: a proposal-level field loss (NOT part of this head; its own arm)

The largest controlled published effect on drivable-area compliance is **output-side**: ReflectDrive-2's field loss, DAC
93.9 → 97.0, PDMS +2.4 (§1). It uses the same labels and needs no head. REFe's WTA loss trains only the best-matching proposal
per sample, so the other 63 proposals are free to leave the road, and "dirty fans" hold 72 % of the zero picks (INHERITED).

- `L_field = mean over proposals k, 0.1 s steps t, footprint corners c of relu(m − SDF_gt(corner_{k,t,c}))`, with the fine int8 SDF
  sampled bilinearly (`grid_sample`) at corners built differentiably from the proposal's poses (linear interpolation 5 Hz → 10 Hz),
  margin m ≈ 0.25 m, on **all 64 proposals**. A hinge: a clean fan gives exactly 0.0 (the control).
- Training-only; **no inference dependency**; no admissibility question beyond §6 items 1–3.
- Limits: it acts on the raw plan, so it does not touch the 29 % of failures made by the tracker (F2); it may shorten plans (EP
  risk, pre-registered as an adverse-component check).
- Registered as a separate arm because bundling it with the head confounds the one question this spec asks. **D-AUX-4.**

## 9 · Risks and what would change my mind

| risk | evidence | mitigation / what would refute |
|---|---|---|
| the trunk already encodes the road (DINOv3 ViT-L is a strong dense feature extractor), so the head adds little | UNVERIFIED for u400; S1's perception proof answers it | head-only IoU high from step 0 and no PDMS gain ⇒ representation lever refuted, H2/§8 remain |
| harm: aux tasks worsened planning in Hydra-MDP++ | PUBLISHED −0.5, one run | harm kill (PREREG K4); label-shuffled control separates "extra capacity" from "map information" |
| the effect is below the detection limit | MDE ≈ 0.75 PDMS (ESTIMATED from SFT-4's paired CI half-width 0.745, `RESULT_SFT4.md` row "primary") against an expected +1.0–1.6 | the **mechanism endpoints** (DAC, DAC-zero share) carry more power; replicates at S1/S2 |
| timing skew / unseen forward extent | §3.1 | G-LIVE alignment; extent UNVERIFIED for fast tokens |
| the GENERATION-BOUND branch of C3 | pending | §5 |

## 10 · Deliverable manifest

See the hand-back report; all files are staged by exact path, none committed.
