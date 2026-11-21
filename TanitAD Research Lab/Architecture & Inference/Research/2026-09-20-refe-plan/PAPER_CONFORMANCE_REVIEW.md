<title>REFe vs DriveZero (arXiv 2609.06055) — a source-only conformance review</title>

# PAPER CONFORMANCE REVIEW — REFe against the DriveZero specification

**Reviewer:** independent review agent · **Date:** 2026-09-20
**Method:** every row below was read from SOURCE and, where a number is quoted, MEASURED by running
the code or parsing the banked artifact. Our own prose was used only to establish whether a
divergence is *documented* — never as evidence that the implementation does something.

⛔ **READ STAMP — THE PACKAGE WAS BEING EDITED WHILE THIS REVIEW WAS WRITTEN.**
`refe/model.py` and `refe/train.py` were modified at **19:24** and **19:25** on 2026-09-20, mid-review,
in response to an interim finding. Every statement about those two files is against:

| file | md5 (first 32) | mtime |
|---|---|---|
| `refe/model.py` | `d398faee941344b68f4752b5a2a9d4db` | 2026-09-20 19:24:12 +0200 |
| `refe/train.py` | `f7d7285339aa8613ba31e8067a4565e7` | 2026-09-20 19:25:20 +0200 |
| `refe/planner.py` | (unmodified) | 2026-09-20 15:47:36 +0200 |
| `refe/build_scorer_targets.py` | (unmodified) | 2026-09-20 19:05:34 +0200 |

⚠️ **AND THE SCORER TARGET BANK IS A LIVE, GROWING ARTIFACT.** Over four reads between ~19:05 and
19:23 it returned **3,957 → 4,317 → 4,713 → 4,857 rows** while its file mtimes stayed pinned at
18:49/18:51 (a Windows writer holding the handle open does not flush mtime). **Every bank number in
this review carries the stamp `19:23:11`.** Re-measure before quoting any of them.

---

## 1 · Headline verdict

REFe's **architecture skeleton is faithful and its backbone discipline is exactly right** — 64
proposals, a 256-d planning representation, one ego token added to M queries, a separate detached
scoring decoder over six PDM components, a genuinely frozen trunk with rank-32 LoRA on **Q and V
only** and *nothing else* trainable (MEASURED: 96 trainable backbone tensors, all `attn.q.A/B` and
`attn.v.A/B`, 3,145,728 params, **zero** outside them), and teacher-rollout supervision under
**non-reactive log-replay background agents**, which is precisely the paper's *"all background actors
following the driving log"*. The two PI-authorised departures (one front camera, DINOv3-only) are
implemented cleanly and documented. But **three structural claims of §2.3 are not implemented at
all**, and one data defect breaks the paper's own consistency requirement on half the training bank:
(a) the DrivoR **registers do not compress anything** — 16 free parameters are *concatenated* to the
full visual-token set, so the decoder's context is **1,936 tokens, not 16**, and the information
bottleneck the paper describes is absent; (b) the trajectory decoder and the scoring decoder receive
**the identical tensor** (MEASURED: same storage pointer), where the paper routes one to scene tokens
and the other to visual tokens; and (c) — **the highest-priority finding, and the reason this review
should block any scorer or selection number** — the paper's *"the same augmented route is used to
evaluate student proposals, keeping the navigation command, teacher trajectory, and proposal scores
mutually consistent"* is **BROKEN**: `build_scorer_targets.py` has no rank concept at all, the scorer
bank was built **entirely from the rank-0 run**, `train.py` loads **both** ranks and joins them on a
rank-free key, and **540 of the 1,080 covered training tuples — exactly half — carry a rank-1 goal
and a rank-1 teacher trajectory against rank-0 proposal scores**, on frames whose two ranks' goals
differ by a median of **3.80 m** (max 28.84 m) and whose teacher targets differ by a median of
**2.418 m** (max 16.81 m). Nothing in any loss curve can see this. Separately, two of the six PDM
components are mis-mapped (**PROGRESS has no counterpart anywhere in the released reward set**, and
our `center_line` slot measures lateral deviation while the paper's DRIVING-DIRECTION COMPLIANCE
signal lives in a channel we never read), one consumed component (**comfort**) is constant across
candidates in **0/540 frames** and therefore cannot rank proposals at all, and inference selection is
an unweighted **sum of six raw logits** rather than the benchmark scoring rule. The three training-
hyperparameter divergences I was asked to confirm (`dec_depth`, lr, scheduler) were **real and are
now fixed in the working tree** by the package author during this review; I have verified the fixes
and report them as MATCHES, with one caveat left UNVERIFIED (that the cosine LR actually reaches
zero, not merely that a scheduler is attached).

**On the scoring-decoder depth question put to me:** on the authoritative text I hold, the paper
specifies *"a 4-layer proposal decoder"* and then only *"a separate scoring decoder"* — **no depth is
given for the scoring decoder**. The new `score_dec_depth: int | None` field (`refe/model.py:93`) with
its comment saying so is the correct reading of that text. I have not seen the full paper, so any
statement outside the two supplied excerpts is UNVERIFIED.

---

## 2 · Conformance table

Verdicts: **MATCHES** · **DIFFERS-DELIBERATELY** (with where it is stated) · **DIFFERS-UNINTENDED**
(a defect) · **NOT-IMPLEMENTED** · **UNVERIFIED**. Paths are relative to the package root
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`.

### 2.1 · §3.2 Training dataset

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| training split | NAVSIM `navtrain`, 100 K interactive scenarios | **nuPlan mini**, 10 scenarios over 7 drive logs, **2,182 tuples** (MEASURED `D:/Projects/TanitAD/data/refe_targets/targets_rank{0,1}.jsonl`, 1,091 + 1,091). No NAVSIM data exists on this box. | plan targets navtrain `REFE_PLAN.md:20`; staging stated `REFE_PLAN.md:99`; absence stated `REFE_PLAN.md:205` | **DIFFERS-DELIBERATELY** — a stated staging step, not a silent divergence |
| supervision source | teacher rollouts, **not** logged human trajectories | teacher rollouts: the target is the ego track of the **simulated** rollout, from the released teacher checkpoint | `refe/build_targets.py:104` (`SimulationLog.load_data`), `:109-110` (ego states), `code/run_mini_teacher.sh:54-55` (`driverl_teacher.yaml`, `checkpoint_2400.pt`) | **MATCHES** |
| "no human demonstrations" | none | none — no logged-expert path is read anywhere | `refe/build_targets.py` (whole file: no expert/GT trajectory read) | **MATCHES** |
| SimScale OOD scaling, 237 K, `DriveZero-Scale` | 237 K OOD scenes | excluded by plan | `REFE_PLAN.md:20` "**navtrain (100 K)**, no SimScale" | **DIFFERS-DELIBERATELY** — the *exclusion* is stated; the *consequence* is not (see §4) |

### 2.2 · Planner / training spec

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| trajectory proposals | 64 | 64 (MEASURED forward: `traj (2, 64, 20, 3)`) | `refe/model.py:80` | **MATCHES** |
| planning representation | 256-dimensional | 256, and it is genuinely end-to-end: the scene projection, the ego token, the M queries, both decoders' `dim`, and both heads' input all carry 256 (MEASURED: `queries (1,64,256)`, `ctx (B,·,256)`, `traj_head 256→60`, `score_head 256→6`) | `refe/model.py:85`, `:249`, `:251-253`, `:254`, `:256-257`, `:258-260`, `:266` | **MATCHES** |
| proposal decoder depth | 4 layers | **4** (MEASURED: 4 `dec` blocks executed) — was **3** at the start of this review and fixed mid-review on PI instruction | `refe/model.py:86-88` | **MATCHES** *(was `DIFFERS-UNINTENDED`; CONFIRMED as real, now repaired — see D-FIXED-1)* |
| scoring decoder depth | **not stated in the supplied text** | mirrors `dec_depth` unless set, with the silence documented | `refe/model.py:89-93`, `:263` | **MATCHES** (paper is silent; the choice is labelled as a choice) |
| register tokens per camera | 16, following DrivoR | 16 (with `n_cameras=1`: `registers (1,16,1024)`) | `refe/model.py:79`, `:244` | **MATCHES** *(count only — see the mechanism row in §2.3)* |
| backbone frozen | DriveVFM frozen | frozen: patch-embed, pos table, cls/reg tokens, K, output proj, both LayerNorms, MLP, both LayerScale gammas, final norm — all `requires_grad_(False)` | `refe/model.py:144`, `:172-176`, `:192-193`, `:195`, `:200-203`, `:207-208` | **MATCHES** |
| trainable adapters | rank-32 **Q/V** LoRA only | rank 32; `q` and `v` are `LoRALinear`, `k` is a plain frozen `nn.Linear`, the MLP is untouched. **MEASURED: 96 trainable backbone tensors — 24×`attn.q.A`, 24×`attn.q.B`, 24×`attn.v.A`, 24×`attn.v.B` — 3,145,728 params, and ZERO trainable backbone tensors outside `attn.q`/`attn.v`.** | `refe/model.py:78`, `:140-142`, `:144` | **MATCHES** |
| trainable parameter budget | 338.46 M full / 18.58 M trainable / 5.49 % *(their ViT-L, 4 cameras)* | **MEASURED 319,031,874 total / 13,986,370 trainable / 4.384 %** (ViT-L, 1 camera, `dec_depth=4`) | measured by instantiating `refe/model.py` | **MATCHES in kind** — not the same configuration, correctly reported as such at `refe/model.py:17-21` |
| epochs | 25 | `--epochs N` derives `steps = N × ceil(len(bank)/batch)` **before** the optimiser is built; default 0 falls back to `--steps 300` | `refe/train.py:308-312`, `:455-457` | **MATCHES** *(was `NOT-IMPLEMENTED`; fixed mid-review — D-FIXED-3)* |
| batch size | 256 | default **8**; no committed value — deferred to a pod measurement | `refe/train.py:448`; `POD_HANDOFF.md:115-116` | **DIFFERS-UNINTENDED** in effect, though honestly deferred — see D5 |
| optimiser | AdamW | AdamW | `refe/train.py:310` | **MATCHES** |
| initial learning rate | 2 × 10⁻⁴ | **2e-4** (was 3e-4) | `refe/train.py:449-450` | **MATCHES** *(fixed mid-review — D-FIXED-2)* |
| LR schedule | decayed to zero via cosine annealing | `CosineAnnealingLR(T_max=steps, eta_min=0.0)`, stepped **per optimizer step** (verified: `sched.step()` at `:395` immediately after `opt.step()` at `:393`) | `refe/train.py:325-327`, `:393-395` | **MATCHES** — ⚠️ that the LR *reaches* 0.0 over a real run is **UNVERIFIED** (a scheduler being attached is not the same claim) |
| weight decay | not stated | 1e-4, now exposed as `--weight-decay` | `refe/train.py:310`, `:451` | paper silent — not a conformance item |
| LoRA α/r scaling | not stated | `scale = 1/rank` (α = 1) | `refe/model.py:120` | paper silent — noted for reproducibility |

### 2.3 · §2.3 Architecture

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| camera-only, open-loop on logged frames, no rendering | yes | yes — the model's only perceptual input is one JPEG | `refe/model.py:268` | **MATCHES** |
| inputs: images + ego kinematics + **navigation command derived from the goal point** | a derived command | the **raw goal points** (2 points = 4 floats), taken straight from what the teacher consumed; no command is derived anywhere | `refe/model.py:268`, `:274`; `refe/build_targets.py:155-156` | **DIFFERS-UNINTENDED** *(undocumented; arguably stronger than the paper — see D8)* |
| visual tokens enriched with **3D position embeddings** | 3D PE | a free learned per-patch table, randomly initialised and trained. **No camera intrinsics or extrinsics are read anywhere in `refe/`** (grep over all of `refe/*.py`: 0 hits for intrinsic/extrinsic/frustum/ray) | `refe/model.py:247-248`, `:271` | **DIFFERS-UNINTENDED** *(undocumented — see D7)* |
| learnable registers **COMPRESS** visual tokens into a small set of **scene tokens** | compression to 16/camera | **concatenation**: `tok = cat([visual, registers])`. **MEASURED: the decoder context is `visual + 16`, and no decoder ever receives a 16-token context.** At the production config that is **1,920 + 16 = 1,936 tokens**. | `refe/model.py:272-273` | **DIFFERS-UNINTENDED** — **D2** |
| ego kinematics + command → **one ego token ADDED to M trajectory queries** | one token, added | exactly that | `refe/model.py:274-275` | **MATCHES** |
| trajectory decoder cross-attends to the **SCENE tokens** | scene tokens | the visual+register union | `refe/model.py:273`, `:277` | **DIFFERS-UNINTENDED** — **D2** |
| scoring decoder encodes **each candidate trajectory** as a score query | trajectory → score query | `s = q.detach()` — the decoder's **latent query**, not an encoding of the output poses | `refe/model.py:280` | **DIFFERS-UNINTENDED** *(minor; see D9)* |
| scoring decoder **attends to the VISUAL tokens** | visual tokens | the **same tensor** the trajectory decoder gets. **MEASURED: identical storage pointer for `dec[i]`'s and `score_dec[i]`'s context.** | `refe/model.py:282` | **DIFFERS-UNINTENDED** — **D2** |
| predicts the six PDM components | 6 | 6 (MEASURED `score (2, 64, 6)`) | `refe/model.py:83`, `:266` | **MATCHES** |
| trajectory generation and scoring trained with **separate objectives** | separate | separate terms, summed | `refe/train.py:361`, `:379` | **MATCHES** |

### 2.4 · Teacher rollouts as supervision

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| frozen teacher + structured state + goal point | released teacher | released `driverl_teacher.yaml` + `checkpoint_2400.pt`, TTS disabled | `code/run_mini_teacher.sh:54-55`, `:75` | **MATCHES** |
| **all background actors following the driving log** | log replay (non-reactive) | **non-reactive log replay.** `PROTO=nr` → task `val14_nr` → `closed_loop_nonreactive_agents_driverl` → `override /observation: box_observation` → `TracksObservation` (log replay). Both banked runs are `m-nr-n` (rank 0) and `m-nr-r1` (rank 1). | `code/run_mini_teacher.sh:18`, `:63`, `:86-87`; `DriveRL/scripts/run_driverl_nuplan_eval.sh:188-190`; `nuplan-devkit/…/experiments/simulation/closed_loop_nonreactive_agents_driverl.yaml:5` | **MATCHES** — ⚠️ but **nowhere stated in our own package** (see D11) |
| target representation | ego-relative (x, y, yaw) | ego-relative (x, y, yaw), 20 samples × stride 2 × 0.1 s = 4.0 s | `refe/build_targets.py:142-146` | **MATCHES** |
| `L_traj = min_m dist(τ̂_m, τ_T)`, only the closest proposal | winner-takes-all | winner-takes-all over M | `refe/model.py:286-289` | **MATCHES** |
| `dist` = **L1 distance averaged over POINTS** | over points | `.abs().mean(dim=(2,3))` — averaged over points **AND over the 3 channels**, so a yaw error in **radians** is summed into a metre budget at equal weight | `refe/model.py:288` | **DIFFERS-UNINTENDED** — **D6** |

### 2.5 · Proposal scoring

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| component 1 — **collision avoidance** | NC | `collision.NuPlanCollision.info`, binary, ego-polygon overlap | `refe/train.py:168`; `DriveRL/…/collision/nuplan_collision.py:42-51` | **MATCHES** |
| component 2 — **drivable-area compliance** | DAC | `off_road.OffRoad.info` | `refe/train.py:167`; `DriveRL/…/off_road.py:238-249` | **MATCHES** *(with a latent category bug — see D4b)* |
| component 3 — **PROGRESS** | ego progress along the route | `goal_reaching.GoalReaching.info` — a **terminal first-reach indicator** (`info = first_time_reached.int()`) against a point re-derived ≥60 m ahead. **NO PROGRESS CALCULATOR EXISTS in the release**: `ROUTE_PROGRESS_DELTA`/`ROUTE_PROGRESS_REWARD` are declared and **never written** (verified: `grep -rn ROUTE_PROGRESS src/` hits only `base_reward_calculator.py:53,76,161,179,237`). | `refe/train.py:171`; `DriveRL/…/goal_reaching.py:171`; `DriveRL/…/goal_position_utils.py:569-572` | **DIFFERS-UNINTENDED** — **D4a**, an unsound mapping |
| component 4 — **time to collision** | TTC | `ttc.NuPlanTTC.ttc_reward`, continuous in [0,1] | `refe/train.py:169`; `DriveRL/…/nuplan_ttc.py:182`, `:32-35` | **MATCHES** |
| component 5 — **comfort** | C | `comfort.Comfort.reward` ∈ {0.5, 1.0}. **MEASURED at 19:23: varies across candidates in 0 of 540 frames** (and `Comfort.info` likewise 0/540) — it cannot rank proposals. | `refe/train.py:170`; `DriveRL/…/comfort.py:224-227`, `driverl_teacher.yaml:72` | **DIFFERS-UNINTENDED** — **D3a**: present but inert as a ranking signal |
| component 6 — **driving-direction compliance** | DDC | `center_line.CenterLine.reward` = **lateral deviation**, normalised to [0.75, 1.0]. Heading is explicitly **not** used (`DriveRL/…/center_line.py:73-74`, both angle features commented out as `# unused`). The real DDC detector exists (`center_line.py:713-762`) but routes into `OffRoad.info ∈ {3,4}` and `CrossLane.reward` — channels we never read. **MEASURED: `OffRoad.info` in our bank takes only {0.0: 3,556, 1.0: 1,301} — categories 3/4 never occur, so there is currently ZERO driving-direction supervision.** | `refe/train.py:172`; `DriveRL/…/center_line.py:277-280`, `:229-231`, `:202-215` | **DIFFERS-UNINTENDED** — **D4b**, an unsound mapping |
| `L_score = BCE(σ(ŝ), s)` | sigmoid + BCE | `binary_cross_entropy_with_logits` (sigmoid applied internally) | `refe/train.py:361` | **MATCHES** |
| candidates **detached** before the scoring branch | detach candidates | `s = q.detach()` **and additionally** `ctx.detach()` — an extra stop-gradient the paper does not describe | `refe/model.py:280`, `:282` | **MATCHES + an undocumented extra** — see D9 |
| aggregation by the **benchmark scoring rule** | PDM aggregation | **not implemented** — no aggregation function exists | `refe/planner.py:155` | **NOT-IMPLEMENTED** — **D5** |
| inference: select the **highest predicted aggregate score** | argmax of the PDM aggregate | `int(score[0].sum(-1).argmax())` — an **unweighted sum of six raw logits**, no sigmoid, no PDM rule | `refe/planner.py:150-155` | **DIFFERS-UNINTENDED** — **D5** |
| `L = λ_traj·L_traj + λ_score·L_score` | two weights | `λ_score = --score-w` (default 0.1); `λ_traj` implicit 1.0, not exposed | `refe/train.py:352`, `:379`, `:459` | **MATCHES** (the paper gives no values) |

### 2.6 · Goal augmentation

| spec item | paper value | our value | file:line | VERDICT |
|---|---|---|---|---|
| teacher queried with alternative driving intents at the same scene state | supported by DriveRL | **reconstructed by us** — the released code has **no API for an alternative goal or route** (its stochastic goal sampler `sample_valid_positions` is dead code with no caller). Our seam is the first roadblock's lane rank, with their continuity rule carried forward unchanged. | `code/augment_routes.py:46-70`; `code/route_lane_rank_patch.py`; documented `STAGE1_GOAL_AUGMENTATION.md:110-114` | **DIFFERS-DELIBERATELY** — stated, and with a rank-0-reproduces-theirs control (`code/augment_routes.py:90-101`) |
| route intent + nav command augmented **before** generating supervision | before | before — the rollout runs under the patched route | `code/run_stage1_rollouts.sh:3-4` | **MATCHES** |
| **the same augmented route evaluates student proposals; nav command, teacher trajectory and proposal scores MUTUALLY CONSISTENT** | 3-way consistency | **BROKEN.** `build_scorer_targets.py` has **no `--rank` argument and writes no `rank` field**; `ScorerBank` keys on `(log_name, token, step)` with no rank; `TargetBank` loads **every** `targets_rank*.jsonl`. **MEASURED (19:23): the scorer bank's own stats record `"run": "C:/dzo/m-nr-n"` = rank 0; all 540 scorer frames are matched by BOTH a rank-0 and a rank-1 tuple; 1,080 of 2,182 training tuples are covered and exactly 540 of them are rank-1 tuples served rank-0 targets.** | `refe/build_scorer_targets.py:147-159`, `:241-252`; `refe/train.py:149`, `:203`, `:252` | **DIFFERS-UNINTENDED** — **D1, the top defect** |
| goal-point count | `num_goal_positions = len(goal_count_probs) = 2` | 2 | `refe/model.py:91`; `DriveRL/src/driverl/env/config.py:21`, `:25-27` | **MATCHES** |

---

## 3 · Defects, ranked by whether they would change a published number

### D1 — the goal-augmentation three-way consistency is broken on half the bank ⛔ TOP PRIORITY

**Would change every scorer number and every selection number.** Invisible to every loss curve.

**Evidence, all MEASURED at read stamp 2026-09-20T19:23:11:**

| fact | value | artifact |
|---|---|---|
| trajectory bank | 1,091 rank-0 + 1,091 rank-1 = **2,182** tuples | `D:/Projects/TanitAD/data/refe_targets/targets_rank{0,1}.jsonl` |
| rank-0 provenance | `"run": "C:/dzo/m-nr-n"` | `…/refe_targets/targets_rank0_stats.json` |
| rank-1 provenance | `"run": "C:/dzo/m-nr-r1"` | `…/refe_targets/targets_rank1_stats.json` |
| **scorer bank provenance** | **`"run": "C:/dzo/m-nr-n"` — rank 0 only** | `D:/Projects/TanitAD/data/refe_scorer_targets_full/scorer_targets_stats.json` |
| scorer rows / frames | 4,857 rows over **540** frames | `…/refe_scorer_targets_full/scorer_targets*.jsonl` |
| `(log, token, step)` overlap between ranks | **1,091 / 1,091 — total** | measured |
| scorer frames matched by rank-0 / rank-1 / **both** | 540 / 540 / **540** | measured |
| covered training tuples | **1,080 of 2,182 (49.5 %)**, of which **540 are rank-1 tuples joined to rank-0 targets** | measured |
| do the two ranks' **goals** differ? | **1,064 of 1,091 differ**; max-abs-diff median **3.80 m**, max **28.84 m** | measured |
| do the two ranks' **teacher targets** differ? | median **2.418 m**, max **16.81 m**; **310/440 differ by >1 m, 83/440 by >5 m** | measured |
| is the scorer's own `teacher` candidate rank-0? | **yes — 0.000 m from the rank-0 target, 0.074–0.092 m from rank-1** | measured |

**Mechanism.** `refe/build_targets.py:172` takes `--rank` and stamps it into every row
(`:159`). `refe/build_scorer_targets.py` has **no equivalent**: its argparse (`:147-159`) carries no
rank and its emitted row (`:241-252`) carries no `rank` field. `ScorerBank` therefore keys on
`(log_name, token, step)` (`refe/train.py:149`) and `TargetBank` looks up with the same three fields
(`refe/train.py:252`), while loading **every** rank file it can find (`refe/train.py:203`). The
candidate set itself is built around the rank-0 teacher's realised path
(`refe/build_scorer_targets.py:196`, `:201`), so a rank-1 tuple is supervised with proposal scores
computed for a **different driving intent**.

⚠️ `code/driverl_mini_bank.yaml:22-23` already records that *"the augmented rank-k rollouts are paired
to rank-0 by (token, step)"* — the very property that makes this join silent.

⚠️ The arithmetic identity the package uses to validate the scorer join (`REFE_MODEL.md:287`,
*"every candidate reads n = 727 against 727 covered samples"*) is **structurally blind to this**: a
rank collapse preserves exactly one row per candidate per covered sample.

**Minimal fix (do not apply — described only):**
1. `refe/build_scorer_targets.py`: add `ap.add_argument("--rank", type=int, default=0)` beside `:150`,
   write `"rank": a.rank` into the row dict at `:241-252`, and suffix the output filename
   (`scorer_targets_rank{rank}{suffix}.jsonl`) so two ranks cannot overwrite each other.
2. `refe/train.py:149`: key on `(r["log_name"], r.get("token",""), int(r.get("rank", 0)), int(r["step"]))`;
   `:252`: pass `r.get("rank", 0)` through; `ScorerBank.get` signature likewise.
3. Build a rank-1 scorer bank from `C:/dzo/m-nr-r1`.
4. **Until (3) exists**, point `--targets` at a directory containing only `targets_rank0.jsonl`.
   That halves the bank and is the honest option; training on both ranks today trains half the
   scorer against the wrong intent.
5. Add a guard that **refuses** a `(log, token, step)` key served to two different ranks — this defect
   is exactly the shape of the `(log_name, step)` collision already fixed at `refe/train.py:139-148`,
   one field further out.

### D2 — the registers do not compress, and both decoders read the same tensor

**Would change every published number** (it changes the architecture's information bottleneck and its
compute).

Paper §2.3: *"Following DrivoR, learnable registers then compress these tokens into a small set of
scene tokens"*; the trajectory decoder cross-attends to the **scene tokens**, the scoring decoder
**attends to the visual tokens**.

Ours (`refe/model.py:271-282`):
```
tok = backbone(img) + pos3d              # [B, P, width]
tok = cat([tok, registers.expand(B,…)])  # [B, P+16, width]   <- CONCATENATION, not compression
ctx = scene_proj(tok)                    # [B, P+16, 256]
for blk in self.dec:      q = blk(q, ctx)
for blk in self.score_dec: s = blk(s, ctx.detach())
```
**MEASURED** (forward hooks on a reduced-resolution build of the same code path): every `dec[i]` and
every `score_dec[i]` receives a context of `visual + 16` tokens with the **identical storage
pointer**; **no decoder ever receives a 16-token context**. At the production geometry (512×960,
patch 16) that is **1,936 tokens instead of 16 — a 121× larger cross-attention context** than the
paper describes, and it makes the 16 registers free bias parameters rather than a compression basis.

Our docs describe what the code does (`REFE_MODEL.md:78`: the task registers are *"added after the
trunk"*) but **never reconcile it with the paper's "compress"**, and say nothing at all about which
tensor each decoder attends to.

**Minimal fix:** give the registers one cross-attention pass over the visual tokens —
registers as queries, `tok` as K/V — producing `scene = [B, n_cameras*16, width]`; feed
`scene_proj(scene)` to `self.dec` and `scene_proj(tok).detach()` to `self.score_dec`. That restores
both the bottleneck and the paper's asymmetry in one change.

### D3 — a consumed component cannot rank proposals, and the guard that should catch it measures the wrong axis

**Would change every scorer number.**

**(a) Comfort is constant within every frame.** MEASURED at 19:23 over 4,857 rows / 540 frames:
`comfort.Comfort.reward` takes **>1 distinct value across candidates in 0 of 540 frames** (and so
does `comfort.Comfort.info`). Its only two values bank-wide are `{1.0: 4,434, 0.5: 279}`. Source:
`DriveRL/…/comfort.py:224-227` gates to `{comfort_weight, 1.0}` with `comfort_weight: 0.5`
(`driverl_teacher.yaml:72`). A per-frame constant target trains a bias and contributes nothing to
ranking.

**(b) The inertness guard is on the wrong axis.** `ScorerBank.variance_report`
(`refe/train.py:175-189`) pools every row in the bank and flags a component only if
`sd < 1e-9`. Comfort's **pooled** sd is 0.1111, so it passes — while being constant in every frame.
`REFE_MODEL.md:242` quotes that guard as proof that *"All six components discriminate"*. It measures
across-frame variance; the question is within-frame variance across candidates.

**(c) The trainer consumes the flat leaf for centre-line, and the file already knows it.**
`refe/score_proposals.py:186-188` records the MEASURED finding that `CenterLine.reward` is saturated
(*"0.7500 for both"*) while `CenterLine.info` discriminates. `refe/train.py:172` consumes
`.reward` anyway. MEASURED: `.reward` varies across candidates in **399/540 frames (73.9 %)**;
`.info` varies in **540/540 (100 %)**. Source confirms it: `CenterLine.reward = 1 − (1−0.75)·t`
(`DriveRL/…/center_line.py:298-308`), floored at 0.75, while `info` is the raw `min_dist` in metres
(`center_line.py:280`).

**Minimal fix:** (i) in `ScorerBank.components` (`refe/train.py:166-173`) replace
`clip(g("center_line.CenterLine.reward", 1.0))` with a normalised form of
`center_line.CenterLine.info` (e.g. `1 − clip(min_dist / d_max)`); (ii) rewrite `variance_report` to
compute variance **across candidates within a frame** and name any component constant in >95 % of
frames; (iii) keep comfort in the vector but report it as a per-scene bias, not a ranking signal.

### D4 — two of the six components are mis-mapped

**Would change every scorer number and any claim that we reproduce their six-component score.**

**(a) PROGRESS has no counterpart, and our proxy rewards overspeed.** The released reward set
contains **no progress calculator** — `ROUTE_PROGRESS_DELTA`/`ROUTE_PROGRESS_REWARD` are declared at
`DriveRL/…/base_reward_calculator.py:53,76` and never written by anything (verified by grep over
`src/`). `goal_reaching.GoalReaching.info` is `first_time_reached.int()`
(`DriveRL/…/goal_reaching.py:171`) — a terminal indicator that the ego's last motion segment came
within `goal_reaching_threshold` (3.0 m) of a goal point placed
`max(speed, 5.0) × 12.0 s` ahead (`DriveRL/…/goal_position_utils.py:569-572`), i.e. ≥60 m out. Our
offline rig scores a **4.0 s** candidate against that, so the component fires for whichever candidate
travels furthest: `refe/README_DIAGNOSTICS.md:21` records `lon x1.5` at 45/74, `teacher` 13/74,
`stopped` 0/73. **That is the opposite of a bounded route-relative progress ratio — it rewards
overspeed.** MEASURED: it varies in 261/540 frames.

**(b) Driving-direction compliance is absent, and our `center_line` slot is lateral deviation.**
The DDC detector exists (`DriveRL/…/center_line.py:713-762`, thresholds 2.0 m / 6.0 m of backward
progress over 1.0 s) but writes into `OffRoad.info ∈ {3, 4}` (`center_line.py:202-210`) and
`CrossLane.reward` (`:229-231`) — neither of which we read. Our `center_line` slot is pure lateral
deviation with heading explicitly unused (`center_line.py:73-74`). **MEASURED: `off_road.OffRoad.info`
in our bank takes only `{0.0: 3,556, 1.0: 1,301}` — categories 2, 3 and 4 never occur, so we have
zero DDC supervision today.**

**(c) A latent category bug.** `refe/train.py:167` computes `1.0 − clip(OffRoad.info)`, but
`OffRoad.info` is a **category** (0 none / 1 off-road / 2 solid-line crossing / 3 wrong-way / 4 severe
wrong-way — `DriveRL/…/off_road.py:220-222`). Clipping to [0,1] collapses 2, 3 and 4 into the same
value as 1. Harmless on today's bank (only {0,1} present) and active the moment a richer scene
appears.

**Minimal fix:** (i) rename the `COMPONENTS` tuple honestly (`refe/train.py:117`,
`refe/score_proposals.py:45`) so nobody reads `goal_reaching` as EP; (ii) derive DDC from
`off_road.OffRoad.info in {3,4}` or take `off_road.CrossLane.reward`, instead of clipping; (iii) for
PROGRESS, compute it — `DriveRL/…/center_line.py:663-711 calculate_baseline_progress` already returns
arc-length progress and is callable from our rig — and emit it as its own leaf.

### D5 — inference selection is not the benchmark scoring rule

**Directly decides which trajectory is executed ⇒ changes every closed-loop number.**

`refe/planner.py:155`:
```python
return int(score[0].sum(-1).argmax())
```
An **unweighted sum of the six raw logits**. No sigmoid, and no PDM aggregation — the paper requires
*"aggregated according to the benchmark scoring rule"* and *"the candidate with the highest predicted
aggregate score"*. A sum of logits is the log-odds **product**, which is neither the PDM
multiplicative-times-weighted-average rule nor a monotone transform of it.

**Minimal fix:** `p = score[0].sigmoid()`, then
`agg = p_nc * p_dac * p_ddc * (w·[p_ep, p_ttc, p_comfort] / Σw)` with the NAVSIM weights, and
`argmax(agg)`. Keep `--select first|mean` as controls.

### D6 — the WTA distance mixes radians into a metre mean

**Would change which proposal receives gradient, i.e. every trained number — but the measured
magnitude on today's bank is small.**

`refe/model.py:288`: `d = (traj - target.unsqueeze(1)).abs().mean(dim=(2, 3))` averages over points
**and** over the 3 channels, so 1 radian costs exactly what 1 metre costs. The paper says *"the L1
distance averaged over points"*.

**MEASURED, constructed counterexample:** against a zero target, a proposal with **0.00 m position
error and 0.60 rad (34.4°) yaw error** scores **0.2000** while one with **0.30 m position error and
zero yaw error** scores **0.1000** — ours picks the second; a points-only distance scores the first
**0.0000** and picks it. **The winner flips.**

**MEASURED, practical magnitude on the real bank:** per-point mean `|x| 13.566 m`, `|y| 0.847 m`,
`|yaw| 0.0846 rad` ⇒ the yaw channel carries **0.6 %** of the pooled mean cost. So the typical effect
is small; the effect on **near-tied** proposals is **UNVERIFIED** (it needs a trained checkpoint,
which does not exist).
⚠️ Note that mean-vs-sum over the coordinate axis is a constant factor and **cannot** change the
winner — only the yaw channel can.

**Minimal fix:** `d = (traj[..., :2] - target[..., :2].unsqueeze(1)).abs().sum(-1).mean(-1)`, and give
yaw its own loss term with its own weight.

### D7 — the frozen trunk is not pretrained DINOv3: RoPE is absent, replaced by a random frozen table

**Would change every published number, and it sits inside — but is not covered by — the authorised
"DINOv3 only" departure.** The PI authorised DINOv3 *instead of DriveVFM*, not a DINOv3 without its
position encoding.

`refe/model.py:195-196` creates `self.pos` as a `trunc_normal_` table with `requires_grad=False`, and
`Attention.forward` (`refe/model.py:142-148`) applies **no rotation**.

**MEASURED from the checkpoint itself** (`D:/Projects/TanitAD/data/backbones/dinov3-vitl16/model.safetensors`):
318 tensors, 303,079,424 params, and **zero keys matching `pos`, `rope`, `rot` or `freq`** — DINOv3
computes rotary position encoding on the fly and stores none. Arithmetic cross-check: our frozen
backbone totals **305,045,504**; `305,045,504 − 303,079,424 = 1,966,080 = 1920 × 1024`, exactly the
`pos` table ⇒ **`pos` is the only trunk tensor with no checkpoint counterpart**, and every other
tensor loads.

So the trunk (i) never receives the positional signal it was pretrained with, and (ii) additionally
receives a random frozen vector per patch that it has never seen. `refe/load_dinov3.py:39-41`
documents the *omission* (*"nothing to copy"*) but not the *consequence*, and the loader's
no-partial-load control passes precisely because `pos` is on its allow-list.

**Minimal fix:** implement DINOv3's axial RoPE inside `Attention.forward` (rotate `q` and `k`) and
delete `self.pos` entirely. If that is deferred, the interim option is to make `self.pos` **trainable**
and say so — a frozen random table is the worst of the three options.

### D8 — the navigation command is not derived; the goal point is fed raw (undocumented)

**Would not change a number relative to the paper's intent — it is arguably a stronger input — but it
is an undocumented divergence.**

Paper: inputs are images, ego kinematics and *"a navigation command … derived from the goal point
given to the teacher"*. Ours feeds the **goal points themselves** (`refe/model.py:268`, `:274`), taken
from `refe/build_targets.py:155-156`. No command is derived anywhere. The programme's own binding
ruling holds that a predicted geometric goal point is the admissible and stronger signal, so this is
defensible — but `REFE_PLAN.md:92` is the only line in the package that even mentions a navigation
command, and the built bank has none.

**Minimal fix:** state the substitution and its rationale in `REFE_MODEL.md`; or derive the command
(a bearing bucket from the goal point) and feed both, which is also what would make an
ablation against the paper possible.

### D9 — the scoring branch detaches the scene context as well as the candidates

**Would change the scorer's achievable quality; small but real.**

Paper: *"Candidate trajectories are detached before entering the scoring branch."* Ours detaches the
candidates (`refe/model.py:280`) **and** the scene context (`:282`). Consequence, MEASURED: the score
loss can reach only `score_dec` + `score_head` = **4,215,302 of 13,986,370 trainable parameters
(30.1 %)**; it cannot shape the LoRA, `pos3d`, `registers`, `scene_proj` or the queries.

**Minimal fix:** pass `ctx` undetached to `score_dec`, keeping `q.detach()`.

### D10 — the closed-loop planner can never resolve a log name and would silently drive a stationary car

**Would make the first closed-loop REFe number a stationary-planner number.** Latent today (no
closed-loop run exists).

`refe/planner.py:158-160` resolves the log name as
`getattr(getattr(self._init, "mission_goal", None), "log_name", None) or getattr(self._init,
"log_name", None) or self._log_hint`.

**VERIFIED against the installed devkit:** `PlannerInitialization` has fields
`['route_roadblock_ids', 'mission_goal', 'map_api']` — no `log_name`; `mission_goal` is a `StateSE2`
with fields `['x', 'y', 'heading']` — no `log_name`. And `_log_hint` is a class attribute defaulting
to `None` (`refe/planner.py:170`) that **nothing in the package ever assigns** (grep: 2 hits, both in
`planner.py`). All three branches therefore yield `None`, `_image_for` returns `None`, and
`compute_planner_trajectory` returns `self._hold(ego)` on **every** step (`:130-135`).

The planner's own `report()` would expose it (`steps_without_a_frame`), so this is recoverable — but
it is a refusal that looks like a run.

**Minimal fix:** take the log name from the scenario. Set `requires_scenario = True`
(`refe/planner.py:95`) and read `scenario.log_name` in `initialize`, or have the runner set
`_log_hint` explicitly per scenario; and make `compute_planner_trajectory` **raise** after N
consecutive unresolved frames rather than holding indefinitely.

### D11 — reporting discipline: moving artifacts quoted without stamps, and a protocol never written down

**Would not change a number, but it is why our own documents disagree with each other.**

* The scorer bank grew **3,957 → 4,317 → 4,713 → 4,857 rows** across four reads in one session while
  its mtimes stayed pinned at 18:49/18:51. Our docs quote **664 rows** (`POD_HANDOFF.md:102`,
  `REFE_MODEL.md:245`) and **2,096 rows / 233 frames** (`REFE_MODEL.md:264-265`) with no stamp.
  Today's disk at 19:23: **4,857 rows / 540 frames**.
* The trajectory bank is quoted as **1,746 tuples** (`REFE_PLAN.md:131`, `POD_HANDOFF.md:54`,
  `REFE_MODEL.md:14`, `:132`) and as **1,964 tuples** (`REFE_MODEL.md:307`, `RESULT.md:226`).
  **MEASURED on disk today: 2,182.** All three doc figures are stale; neither cites the other as
  superseded.
* The **parameter counts are now stale in both steering docs**: `REFE_PLAN.md:146-147` and
  `REFE_MODEL.md:35-38` carry **316.9 M / 11.88 M**, which was the `dec_depth = 3` figure.
  **MEASURED at `dec_depth = 4`: 319,031,874 total / 13,986,370 trainable / 4.384 %.**
* Scorer coverage is quoted at **14.6 %** (`RESULT.md:226`), **22.7 %** (`REFE_MODEL.md:265`) and
  **25 %** (`REFE_MODEL.md:299`). **MEASURED today: 49.5 %.**
* The **background-actor protocol of the target bank is nowhere stated** in the package, although it
  is the one thing the paper is explicit about and the package's own rule at `RESULT.md:104` says a
  result is *"a statement about a PROTOCOL, not about a policy"*. It is **non-reactive log replay**
  (chain verified in §2.4) and should be written into `REFE_MODEL.md` beside the bank.
* `refe/score_proposals.py:28-29` and `:319-320` still assert the scorer is *"NOT yet wired into
  training"* and that *"proposal selection at inference stays arbitrary"*. The first half is now
  false — `refe/train.py:288-300` consumes the bank. `POD_HANDOFF.md:97` and `REFE_MODEL.md:15,142,152`
  carry the same stale claim against `REFE_PLAN.md:109`'s *"targets consumed by training ✅"*.

### Fixed during this review — CONFIRMED as real defects, repaired by the package author

I was asked to confirm or refute two discrepancies. **Both were real.** They were fixed in the
working tree at 19:24/19:25 while this review was in progress; I did not apply them and I have
verified them from source.

| id | defect as found | current state | verdict |
|---|---|---|---|
| **D-FIXED-1** | `dec_depth = 3` against the paper's 4-layer proposal decoder; nothing in our docs justified the 3 (`REFE_PLAN.md:155` commits to *"4-layer proposal decoder"*, and agent-level grep over all six docs found **no acknowledgement of a deviation**) | `refe/model.py:86-88` → **4**, with the divergence labelled. Separately, `score_dec_depth` (`:89-93`) now records that the paper gives **no** scoring-decoder depth — correct on the text I hold. | **CONFIRMED, fixed** |
| **D-FIXED-2** | `--lr` default 3e-4 against the paper's 2e-4 | `refe/train.py:449-450` → **2e-4** | **CONFIRMED, fixed** |
| **D-FIXED-3** | no LR scheduler at all; and the trainer counted only steps, so *"25 epochs"* could not be expressed | `CosineAnnealingLR(T_max=steps, eta_min=0.0)` at `:325-327`, stepped per optimizer step at `:393-395`; `--epochs` derives steps **before** the optimiser is built at `:308-312` | **CONFIRMED, fixed** — ⚠️ that the LR **reaches** 0.0 over a real run is **UNVERIFIED** |

⚠️ **Still open from that family: batch size.** `refe/train.py:448` defaults to **8** against the
paper's **256**, and no document commits to a value (`POD_HANDOFF.md:115-116` defers it to a pod
measurement). That is honest, but it is load-bearing for a reason the package itself states
(`POD_HANDOFF.md:110-113`): winner-takes-all touches at most `batch` of the 64 proposals per step, so
at batch 8 only 12.5 % of the proposal set can be updated in a step. **A REFe number produced at
batch 8 is not comparable to their table**, and the arm should carry the realised batch in its stamp.

---

## 4 · What the scaled-training plan is missing relative to the paper

**The plan names the right target and is honest that it is not there.** `REFE_PLAN.md:20` commits the
training data to *"navtrain (100 K), no SimScale"*; `REFE_PLAN.md:99` states the built scorer work is
*"on nuPlan rather than navtrain"*; `REFE_PLAN.md:205` states *"No NAVSIM/OpenScene data exists on the
dev box … the backlog-row-3 pull is still PI-gated."* **This is a stated, deliberate staging step, not
an unacknowledged divergence.** I want that on the record because the gap is four orders of magnitude.

| dimension | paper | staged today | gap |
|---|---|---|---|
| corpus | NAVSIM `navtrain`, **100 K** scenarios | nuPlan mini, **10 scenarios / 7 drive logs** (MEASURED) | **10⁴** |
| training samples | 100 K (+237 K SimScale = 337 K) | **2,182** tuples (MEASURED) | **~154× to navtrain, ~154×** |
| scorer supervision | the student's own 64 proposals, scored **online** | a fixed 9-candidate perturbation basis, precomputed; **49.5 %** frame coverage | stated at `REFE_MODEL.md:231-235`, `RESULT.md:239-242` |
| batch | 256 | 8 (default); pod value deferred | stated at `POD_HANDOFF.md:115-116` |
| epochs | 25 | expressible since 19:25; arms committed at 50/25/25/25 (`POD_HANDOFF.md:36-39`) | closed |
| compute | 16 × H20 · 38 h = **608 GPU-h** | one A40; arm D costed at **~608 h = 25 days on one card** (`POD_HANDOFF.md:39`) | provisioning decision |

**What is genuinely missing, in priority order:**

1. ⛔ **The navtrain pull has no landing place and it cannot happen here.** MEASURED by the package
   author today from the Hugging Face metadata API with nothing downloaded: **449.0 GB over 64 files**
   (`navtrain_current` 304.5 GB / 32 files, `navtrain_history` 144.5 GB / 32 files), banked at
   `raw/navtrain_size_measured.txt` — confirming `REFE_PLAN.md:214`'s ~450 GB estimate to 0.2 %. At
   this box's measured 1.6 MB/s uplink that is **77 h**; on the pod it is ~1.2 h. **The PI has ruled
   that training uses an A40 pod and the data must live there.** ⇒ the pull is a pod-side first
   action, and nothing about it should be attempted from the dev box. This is on
   `Project Steering/PI_DECISION_QUEUE.md`.
2. ⛔ **No offline navtrain target builder exists.** `REFE_PLAN.md:92-94` specifies it (NAVSIM `Scene`
   → teacher `ScenarioData` → rollouts) but the built builder (`refe/build_targets.py`) reads a
   **nuPlan `SimulationLog`** (`:103-104`) and a **nuPlan camera DB** (`:65-75`). Neither exists in
   NAVSIM's format. This is the single largest unbuilt item between today and a comparable number,
   and it is not costed anywhere.
3. ⚠️ **The scorer bank does not scale by the same path as the trajectory bank.** MEASURED cost in
   the bank's own stats: 2,490.9 s for 249 frames = **10.0 s/frame** at `--stride 2` with 9
   candidates. At navtrain's 100 K frames that is **278 single-core CPU-hours per rank**, and the
   paper needs it **per augmented route**. Nothing in `POD_HANDOFF.md`'s arm table prices this — the
   arm costs there are GPU-hours for the student.
4. ⚠️ **SimScale is excluded but its cost is never stated.** `REFE_PLAN.md:20` drops it in one cell.
   No document says what dropping 237 K OOD scenes costs, or that **`DriveZero-Scale` is therefore
   out of reach as a comparison row** — so a future reader could compare a navtrain-only REFe against
   a `-Scale` number. The OOD-scaling question itself appears nowhere in the package (verified by an
   exhaustive grep pass over all six documents: `simscale`, `drivezero-scale`, `OOD`,
   `out of distribution`, `distribution shift`, `237`). **This should be a one-paragraph position,
   not an invention — I am not supplying one.**
5. ⚠️ **The evaluation half is scoped down and stated** (`REFE_PLAN.md:21`: navtest PDMS + navhard
   EPDMS, HUGSIM optional/later). That is fine. But D5 and D10 mean that **no closed-loop number can
   be produced today at all** — selection is not the benchmark rule and the planner cannot resolve a
   frame. Both are prerequisites for Stage 5, and neither is on the pod-handoff critical path.
6. ⚠️ **The anchor is stale in four places.** `REFE_PLAN.md:140-141` correctly re-anchors to DINOv3
   ViT-L = 94.55 (Table A13) after the ViT-L decision, but `REFE_PLAN.md:23`, `:39`, `:156` and
   `RESULT.md:6` still carry the ViT-S **93.88** anchor. A REFe result compared against 93.88 would
   be compared against the wrong row.

---

## 5 · Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| **This review** | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/PAPER_CONFORMANCE_REVIEW.md` — staged (blob-verified) | no (staged in the index) |
| rank-join measurement script | `scratchpad:rankjoin.py` | **YES — scratchpad only** |
| goal-divergence measurement script | `scratchpad:goaldiff.py` | **YES — scratchpad only** |
| parameter/LoRA-placement probe | `scratchpad:param_probe.py` | **YES — scratchpad only** |
| decoder-context + WTA-metric probe | `scratchpad:shape_probe.py` | **YES — scratchpad only** |
| bank scale / coverage / component-variance probe | `scratchpad:bankscale.py`, `scratchpad:perframe.py`, `scratchpad:comfort.py` | **YES — scratchpad only** |
| final stamped measurement | `scratchpad:final_measure.py` | **YES — scratchpad only** |
| DINOv3 checkpoint key census | `scratchpad:ckpt_keys.py` | **YES — scratchpad only** |

⚠️ **The seven probe scripts exist only in the session scratchpad and are deliberately not staged.**
Each is a few dozen lines of throwaway measurement against artifacts that are still being written;
banking them would add code to a package under active edit and would bank a snapshot of a moving
bank. **Every number they produced is reproduced inline in §2 and §3 with its artifact path and its
read stamp**, so nothing is stranded that a reader needs. If the PI wants them kept, the right home
is `refe/diag_*.py` beside the existing diagnostics — say the word and they can be re-derived in one
pass.

**No code was changed by this review.** The edits to `refe/model.py` and `refe/train.py` observed
mid-review were made by the package author on PI instruction, not by me.

---

## 6 · Escalation — what needs a decision, not a merge

1. ⛔ **D1 blocks every scorer and selection number.** Until the scorer bank carries a rank, training
   on both ranks teaches half the scorer against the wrong driving intent. The cheap interim is to
   train on rank 0 only; the real fix is a rank-1 scorer bank. **This is a go/no-go on any REFe
   scorer result, not a code-quality note.**
2. ⛔ **D2 is an architecture decision, not a bug fix.** Restoring the register compression changes
   the model's compute profile and its parameter count, and it should happen **before** the pod arm
   runs — otherwise arm B measures an architecture that is not DriveZero's.
3. ⚠️ **D5 + D10 together mean no closed-loop REFe number is producible today.** Neither is on the
   pod-handoff critical path and both should be.
4. ⚠️ **The stale parameter counts (316.9 M / 11.88 M) in `REFE_PLAN.md:146-147` and
   `REFE_MODEL.md:35-38` are now wrong** and one of them is the basis of the sub-300M-thesis
   statement the PI signed off on. Measured today: **319.03 M / 13.99 M / 4.38 %**.
