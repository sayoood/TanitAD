# REF-C ARCHITECTURE REVIEW — from source, with the encoder question attached

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Agent:** refc-architecture-review
**Why now:** the PI's framing — *"REF-C is the only model where we proved that it is driving on
vision's own capabilities"* — makes REF-C's encoder the one encoder attached to demonstrated
vision-grounded driving, and nobody had (a) reviewed the architecture from source in one place or
(b) run the C104 readout ladder on it (JOB 2, `REFC_ENCODER_LADDER.md`, sibling file).
**Evidence classes:** `MEASURED (ours + artifact path)` · `PUBLISHED` · `INHERITED (not
re-verified)` · `ESTIMATED` · `HYPOTHESIS`. Sources are `stack/tanitad/refs/refc.py` (2,215
lines), `refc_select.py` (397), `refc_tactical.py` (379), `stack/scripts/refc_train.py`,
`Project Steering/MODEL_REGISTRY.md` §4 (:2135), and raw eval JSON only.
⚠️ **Line-citation frame:** `refc.py` lines are this session's working tree, which carries a
sibling agent's staged +31-line seam edit inside `RefCModel.forward` (insertions ~:1933-1999);
citations ≥ :1933 sit 30 lines lower in HEAD until that lands. Every such citation also names its
SYMBOL (the refc_tactical.py discipline), so it survives either resolution.

---

## 0. THE ONE-PARAGRAPH ANSWER

REF-C (`Anchored-Diffusion-C`) is a **single-frame-decode, three-size, end-to-end-from-scratch**
DiffusionDrive-style planner: a **torchvision-free ResNet-34-style trunk** (9-channel 3-frame
stack, 256×256, ~51.4° effective HFOV) feeds an **8×8 conv map** that a fixed **FPS anchor
vocabulary** (64/128/256 anchors over a 4-waypoint, 2 s ego-frame trajectory space) cross-attends
through 3/4/6 MHA layers; the decoder emits per-anchor confidence + offset, optionally refines
geometry with **2 truncated-denoise steps**, and **argmaxes a selection score** that today is the
t=0 classifier score (the S1 defect, repaired-but-default-off in D-SEL). Aux heads: 5-way
maneuver (H19-grafted into anchor priors), 3-way route, LAW latent-world-model, all reading the
**globally mean-pooled** encoder feature. The encoder is **trained, never frozen, never
pretrained** — one Adam optimizer over the whole model. At inference every published number runs
**vision + v0 only** (`nav_cmd=None` → constant `follow`), which is what makes the PI's framing
defensible; the strategic/tactical→operative conditioning that exists today is **internal**
(ctx→condition, maneuver→anchor-prior), and the **information-disjointness ruling is satisfied
in the current wiring because the situation classifier's output simply does not exist in
REF-C's graph** — declared in code by `RefCModel.goal_provenance()` (refc.py:1767) and now
checkable interventionally by `tanitad/eval/goal_provenance.py` (C120 instrument).

---

## 1. THE VISION BACKBONE — MEASURED from source

| fact | value | source |
|---|---|---|
| architecture | ResNet-34-style BasicBlock trunk, torchvision-free: 7×7/2 stem + maxpool/2, four stages at widths (w, 2w, 4w, 8w), strides (1,2,2,2) → total stride 32 | `refc.py:920-981` (`BasicBlock`, `ResNetEncoder`) |
| input | **9 channels = D-015 3-frame RGB stack** (latest = `[-3:]`, 100 ms apart), **256×256** | `refc.py:248-249`; stack def `tanitad/data/physicalai.py:19` |
| geometry | `ftheta_crop` episode build: fisheye front-wide-120fov **cropped about the per-clip principal point** and resized; keeps the sensor's radial warp; **f_eff = F_REF = 266 px** at 256 px → **~51.4° effective HFOV** (2·atan(128/266)) | `physicalai.py:105-160`; `tanitad/data/calib.py:38` (`F_REF = 266.0`) |
| output | conv map `[B, F, 8, 8]` (grid = 256/32) + globally mean-pooled `[B, F]`; **F = base_width×8**: 512 (small) / **704 (base)** / **992 (XL)** | `refc.py:255-259, 976-981` |
| per-size trunk | small: w=64, blocks (3,6,16,6) → **47,862,976** · base: w=88, blocks (3,6,16,6) → **90,458,632** · XL: w=124, blocks (3,8,20,6) → **199,496,532** | MEASURED, §2 below |
| **pretrained?** | ⛔ **NO — from scratch.** No pretrained init exists anywhere: `refc_train.py`'s only `load_state_dict` is its own resume (`:1058`); no torchvision/hub/warm-start path (grep over the trainer returns none); the docstring states the torchvision-free trunk is deliberate | `refc_train.py:1058`; `refc.py:1-13` |
| **frozen?** | ⛔ **NO — fully trained.** ONE optimizer over ALL parameters: `torch.optim.Adam(model.parameters(), lr=1e-4)` — "REF-C trains end-to-end from scratch under ONE optimizer; there is no trunk/planner boundary" | `refc_train.py:925, 689`; `refc_select.py:104-107` |
| window handling | with `hierarchy` (default True) the encoder runs over all **W=8** window frames; the **decoder cross-attends the LAST frame's map only**; the other 7 pooled vectors feed only the StrategicCtx GRU | `refc.py:1958-1968` |

⭐ Two structural facts that matter for every readout claim: (i) REF-C is
**single-instant at the decode surface** — `forward` cross-attends the last frame's `fmap` only
(refc.py:1960-1966), so relative motion is available only through the 3-frame channel stack and
the ctx GRU; (ii) the aux heads (`route_head`, `maneuver_head`, `goal_head`) read the **globally
mean-pooled** vector — a 64:1 spatial pool — which is exactly the surface JOB 2 probes.

## 2. PARAMETER BUDGETS — MEASURED by building all three presets (this session)

`RefCModel(cfg)` instantiated on CPU for each preset; `param_breakdown` (refc.py:2167) output
verbatim. **All three totals reproduce `MODEL_REGISTRY.md` §4 exactly** (registry:2149-2152), and
the XL per-module split matches the registry row digit-for-digit.

| module | small (`refc_small_config`) | base (`refc_config`) | XL (`refc_xl_config`) |
|---|---|---|---|
| encoder | 47,862,976 | 90,458,632 | 199,496,532 |
| measurement | 17,280 | 17,280 | 17,280 |
| strategic (ctx GRU) | 1,608,768 | 1,903,680 | 4,133,472 |
| decoder | 2,950,729 | 8,634,505 | 22,702,345 |
| imagination (H15) | 0 | 0 | 20,986,339 |
| aux (maneuver+route) | 134,152 | 274,760 | 513,960 |
| law | 2,116,096 | 2,902,720 | 4,082,656 |
| **total** | **54,690,001** | **104,191,577** | **251,932,584** |

Encoder share: 87.5 % / 86.8 % / 79.2 % — the budget deliberately lands in the trunk (the
registry's Hydra-MDP ResNet-34→V2-99 rationale, refc.py:99-110). Decoder config per size:
d=256×3L×4h / d=384×4L×8h / d=512×6L×8h; anchors 64/128/256 over pool 2048/4096/4096
(refc.py:648-689). All three arms **trained to 30 k and evaluated** (registry §4.1/4.2/4.3):
ADE@2s full-set 0.5261 (small) / 0.4728 (base) / 0.4714 (XL), tier: `taniteval.driving/tier0`.

## 3. THE ACTION / TRAJECTORY REPRESENTATION

* **Trajectory space:** 4 time-indexed ego-frame waypoints at steps **(5, 10, 15, 20)** of the
  10 Hz grid = **0.5/1/1.5/2.0 s** — a `[4, 2]` (x fwd, y left, metres) object (refc.py:287-294).
  Under the gated `refc1` (OFF in all trained arms — registry §4.1 `refc1 false`) the same slots
  become fixed-distance path checkpoints at (2, 5, 10, 20) m + a target-speed class head.
* **Anchors:** a **fixed vocabulary buffer** `[N, 4, 2]` (refc.py:1069), built by
  **furthest-point sampling** over a 4,096-trajectory synthetic pool (`synth_anchor_pool`,
  refc.py:159; FPS not k-means because the corpus is ~74 % straight and k-means collapses,
  refc.py:15-18). Production anchors from `stack/scripts/build_refc_anchors.py`
  (`refc_anchors_full.pt`), installed via `load_anchors` (refc.py:1180).
* **Decode:** anchor trajectories → queries (`traj_proj`), cross-attend the 64 conv-map tokens
  (`feat_proj`), FiLM-style condition (measurement + grafts) → per-anchor confidence
  (`conf_head`) + per-anchor `[4, 2]` offset (`offset_head`) (refc.py:1188-1199).
  `traj = anchors + offset` (refc.py:1395).
* **Diffusion:** `steps=0` is the pure classifier; the trained/published mode is
  **`steps=2` truncated denoising** (`diffusion_steps: 2`, refc.py:309; dumps carry `steps: 2`):
  each pass re-queries with the current estimate + timestep embedding and adds the new offset;
  noise (σ=0.1 m) only in training — eval is deterministic (refc.py:1397-1416).
* **What the model emits:** `traj [B,4,2]` = the fan entry at `sel_idx` (argmax of the selection
  score, refc.py:1485-1489), plus the full fan `anchor_traj [B,N,4,2]`, `anchor_logits`,
  `refined_logits`, `sel_score`, maneuver/route logits, `law_pred` (refc.py:2117-2126).
  ⛔ **No controls are emitted** — REF-C outputs a trajectory, not steer/throttle; the old TCP-C
  GRU control branch was deleted in the redesign (refc.py:1-13).

## 4. WHAT EXISTS TODAY IN `refc_select.py` AND `refc_tactical.py`

**`refc_select.py` — the D-SEL selection surface (all levers DEFAULT OFF; an all-off build is
byte-identical to pre-D-SEL REF-C, pinned by tests).** The module header documents the five
measured selection defects as one defect (the ranking, not the proposals) — including
**S1** (the refined fan is ranked by the UNREFINED t=0 score — the denoise passes' own
confidences are discarded; refc.py:1391-1400 keeps them now, gated), **S1b** (the emitted fan
`x_in + off` is scored by NO head — one extra conf-only pass fixes it, refc.py:1418-1435),
**S2/S2b** (72.08 % of the XL fan is not physically flyable; the bounded-acceleration band is
argmax-only, and the pre-decode anchor prefilter is output-exact because `CrossAttnLayer` has no
candidate-axis self-attention — 3.46-3.70× decode saving, refc.py:1325-1364), **S3**
(`consequence_scores` through `law_head` — the only `cond_imagination` REF-C can express, +1
parameter, refc_select.py:347), **S4** (`apply_seam_clamp` — norm cap on grafts, with
sustained-saturation fail-loud counters, refc_select.py:271-345), **S5** (`route_to_anchor`,
zero-init) and **S6** (the predicted geometric goal, §5). What does NOT transfer from the
flagship and why is argued from source in the same header (`imagine_probes` has no candidate
axis and REF-C has no rollable predictor — `law_head` maps `[pooled, traj] → pooled_{t+0.5s}`,
`LAW_AHEAD = 5`, refc_train.py:114, so it cannot be iterated). Total D-SEL capacity with every
flag on: **+385 parameters** (param_breakdown docstring, refc.py:2171-2177).

**`refc_tactical.py` — the factorised tactical vocabulary (D-TAC1, gated
`factored_maneuver`, DEFAULT OFF — the trained 30 k arms all run the 5-way head).** The header
derives, in algebra from source, why the single 5-way softmax (3 lateral + 2 longitudinal
classes in one simplex, minted by the label's priority collapse turn>brake>accel>lane_keep)
structurally suppresses longitudinal decisions — and the MEASURED confusion (base-30k, n=1364):
`accelerate` recall **0.0000** (0 predicted / 146 true), `brake_stop` **0.0256**, while the turns
are emitted at nearly their true rate. The module provides the exact `COLLAPSE_TABLE`,
`derive_man5_logprobs` (so the 5-way surface survives as a derivation), `invert_man5` (external
5-way priors are factorised, not dropped), `logit_adjust` (F3 prior-corrected decode — measured
NOT separated on macro-F1, stays default-off), and the kinematic label factorisation
(`window_factored_labels{,_v2}`). Status: **implemented, tested, un-trained-through** — no 30 k
arm with `factored_maneuver=True` exists (registry §4 lists only the three shipped arms; the
pre-registered D-TAC1 arms are a PI-gated GPU spend).

## 5. WHAT CONDITIONS ON WHAT — and the disjointness ruling

The conditioning graph at inference, from `RefCModel.forward` (refc.py:1931-2166) and
`AnchoredDiffusionDecoder.forward` (refc.py:1275-1540):

```
frames ──ResNet──> fmap(last) ──feat_proj──> KV (64 tokens)          [vision]
   └──(all 8 pooled)──> StrategicCtx GRU ──ctx_to_cond (0-init)──┐
v0, nav_cmd ──measurement MLP ──cond_proj──────────────────────────┼──> CONDITION
external target_latent (optional) ──FiLM (0-init) ─────────────────┘    (warps every
                                                                         candidate)
pooled ──maneuver_head──> man_logits ──H19 maneuver_to_anchor──> anchor-prior term
                                                                  (SELECTS among
external maneuver_logits (optional) — replaces man_logits in H19   candidates)
t=0 classifier score (+priors) ──────────────> sel_score ──argmax──> emitted traj
```

* **Strategic→operative today:** the ctx GRU token enters the decoder **condition** (zero-init
  seam, refc.py:1082-1087) — it can *warp* every candidate but never *choose* among them; the
  route head is aux-only (its readout reaches selection only under the off-by-default S5).
* **Tactical→operative today:** **H19 is LIVE from step 0** — the model's own 5-way maneuver
  logits reweight the anchor confidence priors through a learned `maneuver_to_anchor` matrix
  (refc.py:1104-1117, applied at :1374-1381). This is the one place where a "tactical decision"
  influences which trajectory is emitted in the shipped arms.
* **The seams an external brain could use:** `maneuver_logits`, `target_latent`, `lan` are
  injectable `forward` arguments (refc.py:1931-1935). In every published number they are None
  (or model-internal), and **`nav_cmd=None` → constant `follow` index 0**
  (refc.py:2004-2006; the LanConfig docstring states it verbatim: "every published REF-C number
  decodes with `nav_cmd=None`", refc.py:343-347) — the source-level basis of the PI's
  "vision's own capabilities" framing: at eval the only non-vision input is v0.
* **The disjointness ruling (Sayed 2026-08-03):** **satisfiable AND currently satisfied.**
  `RefCModel.goal_provenance()` (refc.py:1767-1836) answers the mandated check in code: the S6
  goal is a function of `pooled` alone; the situation classifier is a separate model
  (sitclf `head_img`) that REF-C neither imports, loads, nor receives as a batch field — *"a
  shared trunk cannot launder a signal that is absent from the graph"*. The declaration is
  emitted into every run `config.json`. ⚠️ The ruling's risk surface is the three injectable
  seams above: an integrator COULD pipe classifier output through `maneuver_logits` or
  `target_latent`. That is exactly what the **interventional probe**
  `stack/tanitad/eval/goal_provenance.py` (C120; landed `19f02067`) now measures — it is
  model-agnostic (`module_runner(model, batch, nodes, …)` + `dependency_matrix`, forward
  interventions, not backward gradients — a `detach()`ed wire leaks signal with zero gradient
  and the gradient probe certifies it clean), so it **adapts to RefCModel directly**: roles
  {frames, v0, nav_cmd, maneuver_logits, target_latent, lan} × outputs {sel_score, traj,
  goal_bearing}. Recommended gate before any external-tactical-brain wiring lands.

## 6. THE PI FRAMING, CHECKED AGAINST PRIMARY SOURCES

| claim | status |
|---|---|
| flagship-v1's route head is an oracle-nav echo (bijection, 1.0000) | MEASURED elsewhere (registry §1; four_families.py:1090-1101 documents route_acc_nav ≈ copying and vision-only route_acc_follow = the majority-straight rate) |
| REF-C evaluates with `nav_cmd=None` | **MEASURED from source** — refc.py:343-347, :2004-2006; the banked latent dump carries `nav_mode: "follow_constant"` |
| REF-C's driving numbers' tier | ⛔ **the C122 name collision is live here**: `taniteval/results/driving_refc-*.json` are `taniteval.driving/tier0` — the METRIC-SUITE tier-0 (open-loop windows, CPU, from banked dumps; driving.py:1-60) — **not EVAL_DOCTRINE T1**. C123 (HEAD) made the same correction for REF-A's numbers the same day. |
| "won the STRATEGIC family (−0.5000) on 77 T1 scenes" | **INHERITED (orchestrator brief) — artifact NOT LOCATED in this clone, and one banked finding cuts against the "T1" label.** Probed six shapes: registry grep ("77 scenes"/"strategic"), hub `*.md` grep (−0.5000), a python scan of every hub JSON for refc+strategic+tier, `taniteval/results` listing (no `t1_*`), `git log --all --grep` (T1, strategic), and the sibling review's own text. The mechanism it would come from exists (`taniteval/tools/t1_eval.py` + `four_families.strategic` → `strategic_optionset.strategic_family`, map-derived option sets vs best-constant, paired bootstrap), and the 2026-08-11 T1 runs lived pod-side. ⛔ **But the banked sibling review (`51cbc15f`, `…/Research/2026-08-18-refc-improvement-review/REFC_IMPROVEMENT_REVIEW.md`) states: "Today REF-C cannot be measured at the primary tier AT ALL — the arm the programme's one vision-grounded strategic win rides on has no closed-loop measurement path."** If both are true, the −0.5000 strategic result is not a T1-of-REF-C number as the phrase suggests (plausibly the option-set family scored on another harness's scenes, or a tier mislabel — the C122 collision's third appearance in two days). ⇒ do not quote it with a tier until the raw JSON is banked; **flagged to the orchestrator as an integration item.** |

## 7. DELIVERABLE MANIFEST (this package)

| artifact | where |
|---|---|
| this review | `TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-refc-architecture-review/REFC_ARCHITECTURE_REVIEW.md` (repo, staged) |
| JOB 2 ladder result + doc | `REFC_ENCODER_LADDER.md` + `raw/rl_main.json` (same dir) |
| target builder + gates | `code/rl1_targets.py`, `raw/rl_targets.npz`, `raw/rl_gates.json` |
| fit driver | `code/rl2_fit.py` (imports er10/pc6/ll1/taniteval.ci — no re-implementation) |
| param measurement | §2 table (reproduces registry §4 exactly; command in the section) |

*Nothing here trains, touches Thor, selects episodes, or edits parity artifacts.*
