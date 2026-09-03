---
license: other
license_name: physicalai-av-derived-research-only
tags:
- tanitad
- autonomous-driving
- trajectory-prediction
- anchor-based-planner
- one-shot-planner
- reference-arm
extra_gated_prompt: >-
  These weights are trained on NVIDIA PhysicalAI-AV derived data (TanitAD
  research program). Access is granted per request for research/evaluation use
  only; you agree not to redistribute.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
---

# TanitAD — REF-C v3 (goal-mediated hierarchy on the supervised anchor arm)

**REF-C v3 is a supervised ONE-SHOT anchor-trajectory model.** Given the observed frames, a
navigation command and one measured ego scalar, it emits the **entire 6-second trajectory in a
single forward pass** — selected out of a fixed vocabulary of **128 anchors x 8 time slots**.

> **There is no action input, no rollout, and no per-step decode.** It is not autoregressive, it
> does not imagine, and there is no loop to close inside the model.

**This is a REFERENCE ARM of the TanitAD research programme, not the flagship and not a
deployable driving system.** Its job is to be the budget-matched *supervised* control against
which the hierarchical world-model flagship is measured.

**Registry key:** `refcv3-b1-v72-30k` · **Arm:** `hier` · **Size preset:** `base`
**Registry section:** `Project Steering/MODEL_REGISTRY.md` §4.5

**Source of truth for every number below:** this run's own `config.json` (shipped in this repo),
the checkpoint itself, and `Project Steering/MODEL_REGISTRY.md`. Evidence class is stamped on
every claim: **MEASURED** (ours, artifact named) · **INHERITED** (another agent/doc, not
re-verified here) · **NOT MEASURED**.

---

## 1. What it is, precisely

```python
# stack/tanitad/refs/refc_v3.py:480
def forward(self, frames, nav_cmd=None, v0=None, steps=0, lan=None, nav_known=None) -> dict:
```

There is **no action argument in the signature** — MEASURED, `refc_v3.py:480`. A closed-loop
action rollout cannot be ported onto this model because there is nothing to feed back.

| property | value | evidence |
|---|---|---|
| Output | one trajectory, whole horizon, one forward pass | MEASURED — `refc_v3.py:480`, `:518`–`:527` |
| Anchor vocabulary | **128 anchors x 8 slots x (x, y)** | MEASURED — checkpoint buffer `core.decoder.anchors` has shape `(128, 8, 2)`; `refc_v3.py:196`, `:197` |
| Horizon slots | `(5, 10, 15, 20, 30, 40, 50, 60)` steps @ 10 Hz = **0.5 / 1.0 / 1.5 / 2.0 / 3.0 / 4.0 / 5.0 / 6.0 s** | MEASURED — `refc_v3.py:106` (`V3_HORIZONS`), `config.json` `horizons` |
| Selection | `sel_score_v3 = apply_seam_clamp(sel_score, goal_gate * goal_distance_score)`, masked by `reach_keep`, then `argmax` | MEASURED — `refc_v3.py:518`–`:527` |
| Goal taus | `(20, 40, 60)` steps = **2 / 4 / 6 s** geometric goals `(x, y, heading, speed)` | MEASURED — `config.json` `goal_tau_steps` |
| Autoregressive? | **no** | MEASURED — no action argument, no per-step readout |

### Inputs at inference — exactly three

1. **Vision** — `frames`, 3 RGB frames at 100 ms spacing, channel-stacked to 9 channels,
   **256 x 640 cylindrical** geometry.
2. **A navigation command** — the v7.2 `nav_command` token (`follow` / `left` / `right`),
   mapped position-pinned onto the legacy `NAV_COMMANDS` one-hot. **See caveat C3: on this
   corpus that token is ORACLE-derived.**
3. **One measured ego scalar** — `v0`, the **speed at the last OBSERVED frame**
   (`v0 = pose_last[:, 3]`, MEASURED at `refc_v3_train.py:445`). This is admissible under the
   PI ruling of 2026-09-02 ("velocity at cycle time is a legal initial state").

**It consumes no future ego data and no ground-truth future.** The `lan` corridor argument is a
**training label only** and is never read at inference (edge E12). No ego state reaches any goal
head (edge E11, refused by design and pinned by test).

> ### `ego_dropout` is a TRAINING-ONLY mechanism
> `ego_dropout = 0.5` (`refc.py:429`) applies a per-sample Bernoulli zeroing of `v0`. It is
> guarded on `self.training`:
> ```python
> # stack/tanitad/refs/refc.py:2033
> if self.training and self.cfg.ego_dropout > 0:
>     keep = keep * (torch.rand(b, 1, device=v.device) >= self.cfg.ego_dropout).to(v.dtype)
>     v = v * keep
> ```
> **At inference `v0` is always present** — the dropout is an anti-shortcut regulariser during
> training, not an inference-time behaviour. MEASURED, `refc.py:2033`.

### The hierarchy this version adds

REF-C v3 composes an **unmodified** `RefCModel` core with a goal cascade:

* **strategic** — the core's existing `StrategicCtx` GRU; a 771-param head reads a *predicted*
  geometric route goal `g_str` off it;
* **tactical** — `PhiTac`, a causal-TCN window pool, conditioned on `g_str` through a
  **zero-init FiLM** so the cascade starts bit-inert;
* **tactical outputs** — factored **lat(3) / lon(3)** heads (the 5-way manoeuvre softmax
  provably destroys the longitudinal decision, which is why it is factored here), geometric
  goals at 2 / 4 / 6 s, and a tactical latent;
* **conditioning** — fed into REF-C's two existing external-tactical-brain ports
  (`maneuver_logits` anchor prior, `target_latent` zero-init FiLM on the decoder condition);
* **selection** — distance to the *predicted* tactical goal at the 2 s slot, through a
  `GoalDistanceScorer` behind a zero-init `goal_gate`.

Goals travel **downward detached**: selection can never train the goal head toward the fan.

**Goal / situation-classifier disjointness (binding PI ruling, 2026-08-03).** The run records its
own provenance audit in `config.json`:
`contains_situation_classifier_output: false`, `situation_classifier_in_graph: false`,
`supplied_or_predicted: "predicted"`, `inference_inputs: ["pooled (mean-pooled conv features,
last frame)"]`. The goal path shares the **encoder** with the route/manoeuvre heads (declared as
a common ancestor); attributability rests on the zero-init gates. MEASURED — `config.json`
`goal_provenance` / `provenance_roles`.

---

## 2. Parameters — MEASURED two independent ways

| module | params |
|---|---|
| `core` (REF-C base: encoder + decoder + LAW + strategic + aux + measurement) | 104,879,522 |
| `phi_tac` (causal-TCN tactical pool) | 1,757,440 |
| `tac_latent_proj` | 262,656 |
| `gstr_cond` (strategic-to-tactical FiLM) | 66,816 |
| `nav_inject` | 50,176 |
| `tac_heads` (factored lat/lon + goals) | 14,364 |
| `scorer` (goal-distance) | 1,156 |
| `str_goal_head` | 771 |
| **total** | **107,032,901** |

* **Probe A** — `config.json` `param_breakdown.total` = **107,032,901**.
* **Probe B** — loading `ckpt_30000.pt` and summing the `model` state_dict: **544 tensors,
  107,082,365 elements**, of which **201 buffer tensors = 49,464 elements** (BatchNorm
  `running_mean` / `running_var` / `num_batches_tracked`, the `(128, 8, 2)` anchor table, and the
  `lat_log_prior` / `lon_log_prior` vectors). **107,082,365 − 49,464 = 107,032,901.** Exact
  agreement.

⚠️ **107,082,365 is the all-tensor count, not the parameter count.** Quote **107,032,901**.
Anchors are buffers, not parameters.

---

## 3. Training

| item | value | evidence |
|---|---|---|
| Trainer | `stack/scripts/refc_v3_train.py`, `--arm hier --size base` | MEASURED — live `ps` on the training pod |
| Corpus | **B1** `physicalai-b1-w120-256x640cyl` — 120°-FOV front-wide, **256 x 640 cylindrical**, PNG-coded `*.v2ep.pt` episode cache | INHERITED — `TanitAD Research Lab/…/2026-09-01-refcv3-training-readiness/READINESS.md` §2 |
| Clips used | **4,572 train** / **141 eval** | MEASURED — `config.json` `nav_from_v7_stats`, `v2_parity.clips_present` |
| Labels | `s2_labels_v7.2_train.jsonl.gz`, md5 **`0ff902130ce76886b8a925eceed9e3a5`**, 4,572 records, schema `s2-geom-v7`; eval labels md5 **`aa12c948f062181c3297265b51526ec5`** | MEASURED — `config.json` `v7_labels` |
| Nav-command split (train) | follow 2,897 · left 811 · right 864 · missing 0 | MEASURED — `config.json` |
| Steps | **40,284** planned | MEASURED — `--steps 40284` |
| Optimizer | lr **1e-4**, warmup **2,000** | MEASURED — `config.json` `argv` |
| Batch / workers / seed | 20 / 6 / 0 · `--u8-batches` (uint8 in-flight batches) · `--v2-lru 24` | MEASURED — `config.json` `argv` |
| Geometry | `--image-hw 256 640` (encoder built at corpus geometry; params unchanged — fully convolutional) | MEASURED |
| Hardware | one A40, pod `tanitad-refcv3` | MEASURED |

### The train/eval split is leak-guarded, and the arithmetic closes

`/root/data/train` and `/root/data/eval` are **symlink views** of one built B1 epcache, split by
v7.2 `clip_id` by `stack/scripts/refcv3_make_split.py`. That script exists because of a **measured
leak** (2026-09-02): the v7.2 release splits 4,572 train / **147** eval clips with zero
intersection, but the raw B1 corpus holds **4,713 = 4,572 + 141 of those eval clips** (the other 6
are the val40 clips the parity gate drops). Training on all of B1 and evaluating on the v7.2 eval
split would therefore have trained on **141 of the 147 eval clips** — and *"the leak would not
announce itself, because the eval would simply look good."*

The split script **refuses to run** if the two label sets intersect. The run's own counts —
**4,572 train + 141 eval = 4,713** — reconcile exactly with the B1 corpus size, which is the
independent check that the split actually happened. MEASURED — `refcv3_make_split.py` docstring
and guard, `config.json` `nav_from_v7_stats`.

Exact command (read from live `ps` on the training pod):

```
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py --arm hier --size base --v2-cache /root/data/train --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz --eval-cache /root/data/eval --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz --eval-every 500 --eval-batches 8 --image-hw 256 640 --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 --lr 1e-4 --warmup 2000 --seed 0 --log-every 50 --save-every 500 --nav-from-v7 --u8-batches --out /workspace/experiments/refcv3-b1-v72-30k
```

### In-training monitor at step 30,000 — NOT A RESULT

The trainer runs a held-out **T0 loss-surface monitor** every 500 steps over a fixed 160-window
set. Its own source says, verbatim, that it *"is NOT the four-metric-family result and must never
be quoted as one"*. Recorded here only as run health:

`eval_loss 7.94893 · eval_traj 0.93026 · eval_anchor_acc 0.56875 · eval_slot_valid_frac 0.91953 ·
eval_goal2s_err_m 1.91019 · goal_gate 0.15595`
(MEASURED — `metrics.jsonl`, step 30000.)

The last figure is the **Caveat-B instrument**: `goal_gate` is zero-initialised and must *learn*
to open, or the goal-selection edge contributed nothing. At step 30,000 it reads **0.15595**, i.e.
non-zero — the edge is live. It says nothing about whether the edge *helped*; that requires the
eval in §4.

---

## 4. Evaluation

### 4.1 ⛔ TIER — OPEN LOOP ONLY. NO CLOSED-LOOP CLAIM IS MADE.

**Binding PI ruling, 2026-09-02:** a model consuming **its own planner's output** is **still open
loop**, because the emitted trajectory does not affect the incoming ego data. **True closed loop
requires the trajectory to actually drive the vehicle** — in simulation (AlpaSim) or in a real
test vehicle.

> **This card makes NO closed-loop claim for REF-C v3, and none may be inferred from it.**
> Open-loop trajectory error is not a driving-performance metric. Elsewhere in this programme,
> an arm at 0.45 m open loop measured 1.69 m closed loop, and open-loop lateral skill was
> shown to be an **action echo** (S-curve reproduction 97.9 % open loop, ~5 % closed loop).

### 4.2 The four binding metric families

Every TanitAD evaluation must report **four families, never pooled into one score**, each with
its `n` and a **paired episode-cluster bootstrap** CI (`taniteval/ci.py`) — ADE alone is an
incomplete result:

| family | must report | status for this checkpoint |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy, distance-keeping (headway / time-gap / TTC) | **NOT MEASURED** |
| **LATERAL** | heading error, curvature error, yaw-rate error, cross-track | **NOT MEASURED** |
| **TACTICAL** | manoeuvre-decision quality, tactical goal-setting (selected vs executed, class confusion, anchor selection) | **NOT MEASURED** |
| **STRATEGIC** | strategic decision + goal/route setting quality | **NOT MEASURED** |

**State of the evidence, stated rather than implied:** the evaluation *instrument* exists —
`taniteval/tools/refcv3_arm.py` writes a `t1_eval`-compatible dump for the arms
`os` / `os_navshuf` / `ha` / `ha0`, reports all four families with `n` and episode-cluster
bootstrap CIs, and carries two degeneracy gates (18/18 tests pass). **No evaluation run against
these weights has been completed.** When it lands, the raw artifact is
`taniteval/results/refcv3-30k-openloop-*.json` and this section is replaced by its contents.

⛔ **Until then there is no accuracy number for this model, and none should be quoted from any
other source.** In particular the training monitor in §3 is not a substitute.

**Also open:** the *tier* of this arm's `os` (one-shot) reading is an **unresolved ruling** in the
programme's register. The instrument stamps it `T1` with `status: UNRULED` on every block and
frames every headline as a **margin over the `ha0` floor**, so the numbers will be correctly
labelled whichever way the ruling goes.

---

## 5. Honest caveats — read these before quoting anything

**C1 — ⛔ NOT ON THE CANONICAL PARITY CORPUS. This model is NOT directly comparable to
`tanitad-refc-base` or `tanitad-refc-xl`.**
Those arms trained on `physicalai-train-e438721ae894` (2,376 episodes, skip-hash `f09e44db`) at
256 x 256. This one trained on the **B1 v7.2 corpus** at **256 x 640 cylindrical**, and the run
records `require_parity: false`, `v2_parity.parity: false`, `v2_parity.checked: false`,
`corpus_key: null` (MEASURED, `config.json`). The B1 corpus is **not registered** in
`parity_manifest.json`. Any REF-C v3 vs REF-C base/XL comparison is confounded by corpus,
geometry, horizon and labels simultaneously.

The trainer says so itself, on line 2 of its own log:

> `[parity] ⚠ NON-PARITY v2 corpus for v3 v2-cache: ['/root/data/train'] reference no registered`
> `parity key. Results off it are NOT cross-arm comparable with the parity arms.`

**C2 — the 6 s horizon is new and its statistics are not inherited.** v1/v2 REF-C planned to
2.0 s over 4 slots; v3 plans to 6.0 s over 8. Reachability and anchor-coverage numbers measured
at the 2 s band **must not** be quoted for the 6 s band. `slot_valid_frac ~0.92` at step 30,000
means roughly 8 % of the far slots are masked out on a given batch (MEASURED, `metrics.jsonl`).

**C3 — ⚠️ the navigation command is ORACLE-derived on this corpus.** `config.json` records
`nav_cmd_derivation: "v7.2 nav_command token (oracle, provenance ego-future;
allow_oracle_nav=True)"`. On PhysicalAI-derived data the only available route supplier is the
ego's own future path, so a *supplied* nav command is **optimistic by construction**. A
deployment would need a predicted goal instead. MEASURED, `config.json`.

**C4 — the goal head and the route/manoeuvre heads share the encoder.** They are
information-disjoint at the *signal* level (the situation-classifier output is not in the graph
and is not a label source here — MEASURED, `config.json` `goal_provenance`), but they do share a
trunk. Attributability rests on the zero-init gates, which is an argument, not a measurement.

**C5 — one seed, one arm.** Seed 0, `--arm hier`, no replicate. The registered ablation delta is
`{core.graft_target_latent: True→False, hier: True→False}`; the flat control is **not** in this
repo.

**C6 — REF-C's inherited selection flaw.** In the REF-C line, all anchors are refined but ranking
uses the pre-refinement score; across 47 trained arms a learned re-scorer recovered at most 8.4 %
of the oracle gap and a hand-written cost re-rank recovered 0.0 %. v3's goal-distance scorer is a
*different* mechanism (candidate-independent, so no winner's curse) but it has **not** been shown
to beat the flat baseline on this corpus. INHERITED — `MODEL_REGISTRY.md` §4.1.

**C7 — ⚠️ THE RUN IS A RESUMED COMPOSITE AND ITS RECIPE CHANGED MID-RUN.** The trainer resumes
with a strict `load_state_dict` (`refc_v3_train.py:1093`), and the supervisor log records
relaunches at steps 2,000 / 4,500 / 6,000 / 10,500 / 17,000 / 17,500 / 18,500. Two of those
changed the flags:

* **`--nav-from-v7` was added at step 17,000** (supervisor v2, 2026-09-02T19:25Z). Steps 0–17,000
  therefore did **not** take the nav command from the v7.2 token.
* **`--u8-batches` was added at step 18,500** (supervisor v3, 2026-09-02T22:06Z) — a memory fix
  (uint8 in-flight batches) after repeated cgroup OOM kills at eval boundaries.

MEASURED — `supervisor.log`. **This is not a clean single-recipe run**, and it should not be
described as one.

**C8 — ⚠️ six buffer values may carry a held-out-label EMA from the run's early segments.**
`compute_losses_v3` used to call `core.update_tactical_prior()` unconditionally, so the
in-training eval's **held-out** label marginals EMA'd into `core.lat_log_prior` and
`core.lon_log_prior` (3 + 3 values, which shape the manoeuvre decode through `logit_adjust`).
The guard `if model.training:` (`refc_v3_train.py:501`) is **confirmed live in the running
trainer** (verified by reading the same lines in the repo and on the pod), but it landed on
2026-09-02 — i.e. **after** the early segments of this run, and the buffers carry across resumes.
Affected: **6 of 107,032,901 values.** Stated because it is real, not because it is large.

**C9 — research artifact.** This is a reference arm in an active research programme. It is not
validated for, and must not be used for, operation of a vehicle.

---

## 6. Files

| file | contents |
|---|---|
| `ckpt.pt` | the final training checkpoint — `{model: state_dict, opt: optimizer state, step}` |
| `ckpt_30000.pt` | the step-30,000 milestone — `{model: state_dict (544 tensors), step: 30000}`, **428,519,790 B**, md5 **`00da81c6efcd91e7b618a1fbddb3b78f`** |
| `config.json` | the run's own config: `argv`, `param_breakdown`, `horizons`, `goal_tau_steps`, `goal_provenance`, `provenance_roles`, `v2_parity`, `v7_labels`, nav-derivation stats |
| `metrics.jsonl` | per-step training rows and the in-training T0 monitor rows |

Verify after download: `md5sum ckpt_30000.pt` must print
`00da81c6efcd91e7b618a1fbddb3b78f` (MEASURED on the pod and again after transfer — both agree).

Load it with:

```python
import torch
ck = torch.load("ckpt_30000.pt", map_location="cpu", weights_only=True)
print(ck["step"], len(ck["model"]))       # 30000 544
```

The architecture is `tanitad.refs.refc_v3.RefCV3Model` built from
`tanitad.refs.refc_v3.refc_v3_config()` with `image_hw = (256, 640)`.

---

## 7. Intended use, limitations and licence

**Intended use.** Research and evaluation only: reproducing the TanitAD reference-arm ladder,
studying goal-mediated hierarchies in supervised trajectory models, and benchmarking against
world-model arms on the same corpus.

**Out of scope.** Vehicle operation of any kind; any safety-relevant decision; any deployment
where a wrong trajectory has physical consequences. This model has **no measured driving
performance** (see §4), has never driven closed loop, and was trained on a single research corpus
with an oracle navigation input.

**Known failure modes to expect.** Degenerate anchor selection (a fixed anchor chosen on every
window) is the characteristic failure of this architecture and is **not** caught by the standard
trivial-profile gate — check the selection profile explicitly. The 6 s slots are the least
supervised and are partially masked during training.

**Data and licence.** Derived from **NVIDIA PhysicalAI-AV** (gated corpus) via the TanitAD B1
w120 / 256 x 640 cylindrical rebuild. Licensed `physicalai-av-derived-research-only`: research
and evaluation use, **no redistribution**. The upstream corpus contains **no map, lane graph,
junction annotation, traffic-light feature or route/goal signal**, and its ego track carries no
GNSS coordinates — a limitation inherited by this model.

**Programme context.** TanitAD is a sub-300M hierarchical 4-brain latent world-model programme
for autonomous driving. REF-C v3 is a *reference arm* inside it. PI: Sayed Bouzouraa.

---

*Card generated by the TanitAD publication agent. Every number is stamped with its evidence
class; anything not sourced to `config.json`, the checkpoint, `metrics.jsonl` or
`Project Steering/MODEL_REGISTRY.md` is marked NOT MEASURED and is absent rather than estimated.*
