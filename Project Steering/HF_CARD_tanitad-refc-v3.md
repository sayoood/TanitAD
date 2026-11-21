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

**Run status: ✅ COMPLETE at step 40,284.** The run's own `summary.json` reads
`{"done": true, "final_step": 40284, "target": 40284}`, and the shipped checkpoint's own `step`
field reads **40284** — two independent probes. This repo ships **two evaluated checkpoints**, the
**final step-40,284** weights and the earlier **step-30,000** weights, each with its own raw eval
JSON under `eval/`. §4 reports the final ones and keeps the step-30,000 column beside them so the
two can never be confused.

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
* **Probe B** — loading the weights and summing the `model` state_dict: **544 tensors,
  107,082,365 elements**, of which **201 buffer tensors = 49,464 elements** (BatchNorm
  `running_mean` / `running_var` / `num_batches_tracked`, the `(128, 8, 2)` anchor table, and the
  `lat_log_prior` / `lon_log_prior` vectors). **107,082,365 − 49,464 = 107,032,901.** Exact
  agreement. Verified on **both** shipped checkpoints — `ckpt_40284.pt` and `ckpt_30000.pt` carry
  the identical 544-tensor inventory, so the architecture did not change across the run.

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
| Steps | ✅ **40,284 — COMPLETE, and the final weights are evaluated.** §4 reports step 40,284 with step 30,000 beside it | MEASURED — `summary.json` `{"done": true, "final_step": 40284}`, and `torch.load(...)["step"]` = `40284` on the shipped file |
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

### In-training monitor — NOT A RESULT

The trainer runs a held-out **T0 loss-surface monitor** every 500 steps over a fixed 160-window
set. Its own source says, verbatim, that it *"is NOT the four-metric-family result and must never
be quoted as one"*. Recorded here only as run health (MEASURED — `metrics.jsonl`, shipped):

| monitor row | step 30,000 | **step 40,284 (final)** |
|---|---|---|
| `eval_loss` | 7.94893 | **7.94446** |
| `eval_traj` | 0.93026 | **0.92696** |
| `eval_anchor_acc` | 0.56875 | **0.62500** |
| `eval_slot_valid_frac` | 0.91953 | **0.91953** |
| `eval_goal2s_err_m` | 1.91019 | **1.97867** |
| `eval_goal_gate` | 0.15595 | **0.17444** |
| `eval_goal_score_absmean` | — | **4.07210** |

⛔ **These are 8 batches / 160 windows of a loss surface, not driving quality.** Note that they do
not all move the same way — `eval_goal2s_err_m` got *worse* over the final 10,284 steps while the
loss fell. Only §4 is a result.

`goal_gate` is the **Caveat-B instrument**: it is zero-initialised and must *learn* to open, or the
goal-selection edge contributed nothing. It reads **0.17444** at the final step (up from 0.15595 at
30,000) — the edge is live. ⛔ **Report the gate with the score scale it multiplies, or the reading
is not falsifiable**: the independent eval over all 4,823 windows measures `gate_mean` **0.17444**
against `score_absmean` **4.0686**, i.e. the gate is genuinely open and acting on a non-trivial
score. None of this says the edge *helped* — that is the `os` − `ha` row in §4.2, and it is a loss.

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

**Every arm below is OPEN LOOP, without exception** — the raw JSON stamps
`loop_class.applies_to: "every arm in this suite, without exception"`. The `T1` letter is retained
because the criteria registry keys on it; the doctrine prose attached to that letter is superseded
by the ruling above.

⛔ **NON-PARITY.** This run did **not** assert corpus parity: `config.json` records
`v2_parity.parity false`, `checked false`, `corpus_key null`, and the trainer prints its own
non-parity warning on line 2 of `train.log`. ⇒ **REF-C v3 is NOT cross-arm comparable with
`tanitad-refc-base` or `tanitad-refc-xl`.** The only admissible cross-model statistic here is each
arm's **margin over the shared `ha0` floor**, which is identically defined everywhere — never a
level against another model's level.

### 4.2 The four binding metric families — MEASURED on the FINAL `ckpt_40284.pt`

**Scope.** Step **40,284**, verified two ways: the checkpoint's own `step` field reads `40284`, and
the run's `summary.json` reads `{"done": true, "final_step": 40284, "target": 40284}`.
⭐ **The evaluator loaded the shipped file itself** — the raw JSON records
`manifest.model.ckpt: "…/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt"` with `step: 40284`, and that file
is shipped here under the same name, md5 `fc304b62686ddb9e685d14bdab482404`, verified identical on
the training pod and again after transfer. `ckpt_40284.pt` carries **bitwise identical** weights
(§6), so §4.2 describes both shipped files and nothing else.
**Protocol:** `taniteval/tools/openloop_suite.py` over the dump from `taniteval/tools/refcv3_arm.py`,
arm `os` (one forward pass). **n = 4,823 windows over 141 episodes** of the v7.2 EVAL split (labels
md5 `aa12c948f062181c3297265b51526ec5`), grid `2s`, dt 0.5 s, 4 horizon steps.
**Estimator: episode-cluster bootstrap, B = 2,000, seed 0, the episode as the cluster unit**; the
**paired** form for every delta. Raw, shipped in this repo: `eval/refcv3-40284-openloop.json` and
`eval/refcv3-40284-openloop.ARM.json`. The criteria checker (registry 2.5.0) reads
**0 violations / 0 work items** on this artifact.

---

> ## ⛔ THE HEADLINE: A TRIVIAL HOLD-ACTION CONTROL STILL BEATS THIS MODEL
>
> `ha` — simply holding the (a, steer) that closes at t0 — scores **ADE 0.2996 m**. The model
> scores **0.4419 m**. Paired on the same 4,823 windows:
> **`os` − `ha` = +0.1423 m [+0.1187, +0.1658], separated ⇒ THE TRIVIAL CONTROL WON.**
>
> Training to the final step did **not** overturn that verdict. It narrowed the gap from
> **+0.1803** [+0.1563, +0.2047] at step 30,000 to **+0.1423** — **21 % of the deficit closed in
> 10,284 further steps.** Extrapolating that rate is not supported by anything measured here.
>
> ⭐ **And the deficit is not in selection, and not in routing.** Decomposing the model's own error:
> a **perfect anchor chooser** over this model's own fan would buy **0.0751 m**
> (`oracle_sel` − `os`, `[−0.0884, −0.0618]`, T0 minus T1 — a ceiling, not an arm), and the
> **oracle nav command** is worth **0.0239 m** (`os` − `os_navzero`, `[−0.0428, −0.0089]`).
>
> **0.0751 + 0.0239 = 0.099 m — still short of the 0.1423 m by which `ha` wins.**
>
> ⇒ **A perfect selector WITH its oracle route would still not reach hold-action.** The deficit
> therefore lies in the **ANCHOR FAN itself** — the trajectories on offer — not in the scorer that
> ranks them nor in the nav that conditions them. **This is the most informative single line on
> this card**, and it says where the next experiment belongs.

---

#### The arms, and the honest ranking

Each arm's own level, episode-cluster bootstrap over 141 episodes (n = 4,823 windows):

| arm | what it is | ADE (m) | FDE (m) |
|---|---|---|---|
| ⭐ **`ha`** | **hold-action control** — the (a, steer) closing at t0, held | **0.2996** [0.2755, 0.3278] | 0.6588 [0.6044, 0.7192] |
| `oracle_sel` *(T0)* | the GT-nearest anchor's refinement — **a ceiling, not a driveable arm** | 0.3668 [0.3437, 0.3914] | 0.7770 [0.7261, 0.8297] |
| **`os`** | **the model**, its own `sel_score_v3` choice, **with the oracle nav** | **0.4419** [0.4098, 0.4743] | 0.9288 [0.8611, 0.9947] |
| `os_navshuf` | as `os`, nav permuted across windows | 0.4563 [0.4240, 0.4884] | 0.9582 [0.8887, 1.0246] |
| **`os_navzero`** | as `os`, **nav withheld — this is what deployment looks like** | **0.4659** [0.4310, 0.5010] | 0.9655 [0.8935, 1.0362] |
| `ha0` | **constant velocity at the measured v0** — the trivial floor | 0.6723 [0.6007, 0.7469] | 1.4029 [1.2484, 1.5646] |

#### Paired margins — the admissible comparisons

⛔ A paired interval that excludes zero **while favouring the control** means the trivial baseline
**WON**. It is rendered as LOST, never as a tie.

| contrast | ADE Δ (m) | CI95 | verdict |
|---|---|---|---|
| **`os` − `ha0`** | **−0.2304** | [−0.2881, −0.1781] | ✅ **WON** |
| `os_navzero` − `ha0` *(deployment)* | **−0.2064** | [−0.2673, −0.1479] | ✅ **WON** |
| ⛔ **`os` − `ha`** | **+0.1423** | [+0.1187, +0.1658] | ⛔ **LOST — the control won** |
| `ha` − `ha0` | −0.3727 | [−0.4346, −0.3161] | (the control clears the floor by 1.6× the model's margin) |
| `os` − `os_navzero` *(what the oracle nav is worth)* | −0.0239 | [−0.0428, −0.0089] | ✅ separated |
| `os` − `os_navshuf` | −0.0144 | [−0.0220, −0.0069] | ✅ separated |
| `oracle_sel` − `os` *(selection headroom, T0−T1)* | −0.0751 | [−0.0884, −0.0618] | ✅ separated |

FDE tells the same story: `os` − `ha0` **−0.4741** [−0.5997, −0.3600] and
`os_navzero` − `ha0` **−0.4373** [−0.5667, −0.3141], both separated wins over the floor.

> ⚠️ **The oracle nav is worth 2.39 cm of ADE and will not exist at deployment** (caveat C3 — on
> PhysicalAI-derived data the only route supplier is the ego's own future path). `os_navzero` is
> the honest deployment row, and it still clears the constant-velocity floor.

#### LONGITUDINAL — one row LOST to the floor

*(all n = 4,823 windows / 141 episodes unless stated)*

speed MAE **0.4516 m/s** [0.4197, 0.4828] · speed bias **+0.0327** [−0.0106, 0.0744] ·
speed RMSE 0.7384 [0.6836, 0.7917] · along-track MAE **0.4030 m** [0.3722, 0.4332] ·
along bias +0.0372 [−0.0047, 0.0799] · **accel MAE 0.6806 m/s²** [0.6464, 0.7156] ·
target-speed accuracy within 0.5 / 1.0 / 2.0 m/s = **0.7135 / 0.8784 / 0.9713** (over 19,292 horizon
*steps*; the bands are proposed reporting tolerances, **not** a gate) · ego-progress ratio
**1.0031** [0.9886, 1.0144], median 1.0007, n = 4,614 (209 windows excluded below 0.5 m progress).

**Paired against the `ha0` floor:** speed MAE **−0.0364** [−0.0636, −0.0079] ✅ WON ·
along-track MAE **−0.0674** [−0.0923, −0.0415] ✅ WON ·
⛔ **accel MAE +0.2020 [+0.1713, +0.2342] — LOST.**
⇒ **family verdict: LOST** (2/3 won, 1/3 lost — the trivial control won where the arm lost).

⭐ **This is the one place the final checkpoint changed a verdict.** At step 30,000 the model added
**nothing measurable over constant velocity longitudinally** (speed MAE +0.0112 [−0.0169, +0.0395],
not separated) and the instrument stamped `_longitudinal_claim_admissible: false`. At step 40,284 it
**beats hold-v0**, separated, and the flag is now **`true`**: the anti-echo block records
*"the arm beats hold-v0 on speed_mae_mps by 0.0364 m/s [0.0079, 0.0636] … the longitudinal head
carries something its v0 input does not"* (arm 0.4516 vs hold-v0 0.4880). ⚠️ **But the deployment
row does not inherit this**: `os_navzero` − `ha0` speed MAE is **−0.0134 [−0.0437, +0.0179], TIED**.
And acceleration is worse than constant velocity in both conditions.

**Distance-keeping** (n = 1,252 windows / 67 episodes): mean min headway **28.06 m**
[24.02, 32.32] · mean min time-gap **3.97 s** [3.21, 4.86] (n = 1,160 / 66 eps) · mean min TTC
**24.99 s** [23.56, 26.32].
⚠️ **788 of 1,252 windows never close on the lead** and are censored at the 30 s cap; only
**n_closing = 464** are informative. **Quote `n_closing` beside the TTC mean, never the mean alone.**
⚠️ The min is taken over only **2 instants (1.0 s, 2.0 s)** — coarser than the block's own grid, so
a lead closest *between* them is not seen.

#### LATERAL — one row LOST to the floor, and the levels and the paired delta DISAGREE IN SIGN

heading MAE **1.3109°** [0.8087, 2.2671] · yaw-rate MAE **1.7940 °/s** [1.5761, 2.0309] ·
curvature MAE **0.008815 m⁻¹** [0.0063, 0.0121] (bias −0.0002) · cross-track MAE **0.1084 m**
[0.0958, 0.1222] · cross-track at the final step **0.2337 m** [0.2050, 0.2648].
(18,115 heading steps; 13,538 curvature/yaw-rate steps; **1,177 steps excluded** below the 0.25 m
minimum arc length.)

**Paired against the `ha0` floor:** cross-track **−0.2048** [−0.2686, −0.1500] ✅ WON ·
heading **−1.3741°** [−1.7569, −1.0418] ✅ WON (n = 4,548; 275 non-finite dropped) ·
⛔ **yaw-rate +0.1700 rad/s [+0.1006, +0.2534] — LOST.**
⇒ **family verdict: LOST** (2/3 won, 1/3 lost).

> ⛔⛔ **READ THE YAW-RATE ROW WITH ITS SCOPE, OR IT SAYS THE OPPOSITE OF WHAT IT SAYS.** The two
> yaw-rate numbers on this card are computed over **different populations and disagree in sign**,
> and both are MEASURED:
> * the **level** is over the **13,538 steps that pass the 0.25 m minimum-arc filter** — and there
>   `os` **1.7940 °/s** is *better* than `ha0` **2.3714 °/s**;
> * the **paired delta** is over **all 4,823 windows with no arc-length filter** — and there `os`
>   is *worse* by 0.1700 rad/s (≈ 9.7 °/s), far above either arm's filtered level.
>
> The family verdict above uses the **paired, unfiltered** figure, because that is the contrast the
> criteria registry keys on. **HYPOTHESIS (not measured here):** the sign flip is carried by the
> excluded near-stationary steps, where yaw rate = dθ/dt is numerically unstable because the vehicle
> barely moves while heading jitters. ⚠️ Note also that yaw-rate is the **one lateral metric the
> instrument does NOT mark dt-invariant** (`dt_invariant: ['heading_mae_deg', 'curvature_*',
> 'cross_*']`), so it is the most fragile row in this family. **Do not quote either yaw-rate number
> without saying which population it is over.**

#### TACTICAL — MIXED

Trajectory-derived, labelled by the programme's own
`tanitad.refs.refc_tactical.factor_from_kinematics`, n = 4,823.
⛔ **This is NOT "selected vs executed".** Both label streams are **executed** manoeuvres — the
arm's own and the human's. Scoring a *declared* decision against the driven path needs a tactical
head and stays unavailable on a trajectory dump.

**Lateral decision — accuracy 0.9540 [0.9396, 0.9664], Cohen's κ 0.8113 [0.7541, 0.8578]:**

| class | n true | n pred | recall | precision |
|---|---|---|---|---|
| lane_keep | 4,176 | 4,157 | 0.9715 | 0.9759 |
| turn_left | 251 | 239 | 0.8048 | 0.8452 |
| turn_right | 396 | 427 | 0.8636 | 0.8009 |

**Longitudinal decision — accuracy 0.7477 [0.7170, 0.7790], κ 0.3078 [0.2484, 0.3659]:**

| class | n true | n pred | recall | precision |
|---|---|---|---|---|
| **brake_stop** | 631 | 470 | **0.3835** | 0.5149 |
| steady | 3,654 | 3,897 | 0.8700 | 0.8158 |
| **accelerate** | 538 | 456 | **0.3439** | 0.4057 |

> ⚠️ **The model still misses 62 % of braking decisions and 66 % of accelerations.** This is the
> programme's known longitudinal defect, still present at the final step — though both improved
> over step 30,000 (brake recall 0.3106 → 0.3835; the 30k over-prediction of `accelerate`, 1,014
> predicted against 538 true, is gone: it now **under**-predicts at 456). No class is never-predicted,
> so this is a weak decision rather than a dead head.

**Paired against the `ha0` floor:** lateral manoeuvre agreement **+0.0881** [+0.0580, +0.1233]
✅ WON · longitudinal manoeuvre agreement **−0.0100** [−0.0356, +0.0162] **TIED**.
⇒ **family verdict: MIXED** (1/2 won, 1/2 tied).

**Tactical goal-setting** (n = 4,823): goal-point error **0.9288 m** [0.8611, 0.9947] *(this is the
FDE at the tactical horizon, reported for continuity — not offered as a new metric)* · goal bearing
MAE **1.5851°**, bias +0.3469° (n = 4,614) · goal range ratio **1.0087** · longitudinal goal bias
+0.0706 m [−0.0187, +0.1603] · lateral goal bias −0.0040 m [−0.0330, +0.0245].
**Anchor-selection quality: UNAVAILABLE**, verbatim — *"this dump's arm commits to ONE path per
window, so there is no candidate fan and no selection to score. Closing it needs a fan+selector
surface … A WORK ITEM, not a pass."*

#### STRATEGIC — ⛔ **UNAVAILABLE, n = 0**

The four-family block reports this family as **UNAVAILABLE**. Verbatim from the raw JSON
(`arms.os.four_families.strategic`):

> **`status`:** `"UNAVAILABLE"` · **`n`:** `0`
>
> **`reason`:** *"strategic decisions not present in the scored pass (missing `['route_pred',
> 'route_gt']`). A world-model FIDELITY pass does not traverse the hierarchy — run_one prints this
> explicitly. Producing this family needs a hierarchy-traversing eval, which is a WORK ITEM."*
>
> **`how_to_populate`:** *"supply `optionset` (map-derived option sets from
> `stack/experiments/nurec-gsplat/strategic_gt.py`, consumed by `taniteval.strategic_optionset`). A
> route label read off the ego's own future yaw is NOT a substitute: it cannot tell whether the map
> admitted a choice."*

⛔ **The route-head probe below is NOT a substitute for this family and must never be reported as
one.** It is a **sidecar** (`_strategic_source: "refcv3 route head (sidecar)"`), and substituting it
is a logged programme retraction (RETRACTION_LOG #16).

⇒ The block therefore reports **`_complete: false`**, `_families_unavailable: ['strategic']`.
Three families measured, one declared missing with its reason and its `n`.
**It is not a complete result and this card does not present it as one.**
*(`_rule_satisfied: true` records only that the missing family was declared properly, per clause 5
of the binding four-family rule — it is not a pass.)*

#### The route head, reported AS A PROBE

Route accuracy **0.7667 [0.7097, 0.8224]** against a no-information (majority-class) rate of
**0.6742**; κ 0.4604; n = 3,622 windows / 128 episodes (**1,201 windows excluded** for having no
route label).

⛔ **It reads 0.7667 identically under true nav, shuffled nav AND zero nav**, and the paired
`true − shuffled` accuracy is **exactly +0.0000 [0.0000, 0.0000]**. The head is **nav-insensitive by
construction**, so:

* it is **not** an echo of its nav input — but that statement is **vacuous**, since the head does
  not read the nav at all. It cannot echo what it never sees.
* the shuffled/zero controls, which are the *only* evidence of route skill in this design, have
  **no power here** — they cannot distinguish anything.

⚠️ The programme's own echo rule exists because flagship v1's route head scored **1.0000** on a
bijection of the nav it was fed. This head fails that trap in the opposite direction: it is immune
because it is deaf.

#### Degeneracy and echo guards — the reason these numbers are readable at all

* **Selection profile** — the gate the generic trivial-profile check is **blind to** for this
  architecture: **50 of 128 anchors** actually selected, modal anchor 57 at **14.82 %**, entropy
  2.8425 of a possible 4.8520 nats (**ratio 0.5858**), agrees with the oracle selection on
  **56.52 %** of windows ⇒ **`degenerate: false`**. The model is genuinely choosing, not collapsing.
* **Trivial profile:** `os` `trivial_frac` **0.0000**; the only degenerate arm is **`ha0` itself**,
  by construction, as the floor should be.
* **Constant-only control (`const0`) — the check that must read a KNOWN value, and does.** Paired
  against itself it reads exactly `delta 0.0, [0.0, 0.0]` (tolerance: **none — bit-exact**), and its
  analytic values reproduce the closed-form answers: ADE **14.248286** against an expected
  14.248286 (|Δ| 1.07e-07, float32 resolution) and speed MAE **11.395674** against an expected
  11.395674 (|Δ| 0.0). ⭐ **If this block ever fails, the harness is wrong, not the model.**
* **Anti-echo:** hold-v0 **BEATS_HOLDV0** (see LONGITUDINAL) · copy-detector **CLEAN**
  (echo index 0.0193 against GT 0.1719).
* **`goal_gate` 0.17444** against `score_absmean` **4.0686** — the zero-init E9 gate did open, on a
  non-trivial score scale.
* **`law_diagnostic`: REFUSED** — inputs missing (a T0 world-model diagnostic with no place in a T1
  row). A declared work item, not a pass.
* **`gaps`: `n_gaps: 0`** — no KPI was silently dropped.

#### What changed from step 30,000 to step 40,284

| | step 30,000 | **step 40,284** | |
|---|---|---|---|
| `os` ADE | 0.4798 | **0.4419** | improved |
| **`os` − `ha`** | **+0.1803** | **+0.1423** | ⛔ still LOST; 21 % of the deficit closed |
| `os` − `ha0` | −0.1924 | **−0.2304** | improved |
| `os_navzero` − `ha0` | −0.1689 | **−0.2064** | improved |
| longitudinal claim admissible | **false** | **true** | ⭐ the one verdict that flipped |
| lateral decision κ | 0.7753 | **0.8113** | improved |
| brake_stop recall | 0.3106 | **0.3835** | improved, still misses 62 % |
| STRATEGIC | UNAVAILABLE, n = 0 | **UNAVAILABLE, n = 0** | unchanged — a work item |
| selection entropy ratio | 0.5782 | **0.5858** | unchanged in kind |

⚠️ **Both columns are the same instrument on the same 4,823 windows**, so the comparison is
paired-valid in kind; the per-cell deltas above are **differences of separately-estimated values**
and only the `os` − `ha` row is quoted with a paired interval at both steps.

### 4.3 How to evaluate this model

The adapter is `taniteval/tools/refcv3_arm.py`; its definition of the arm is
`taniteval/tools/REFCV3_ARM.md` §2. Arms: `os` (one forward pass — the deployed path,
`out["traj"]` ranked by `sel_score_v3`, **never** the training-time `a_star`), `os_navshuf`
(nav permuted — breaks the pairing, keeps the marginal), `os_navzero` (**nav withheld — this is
what deployment looks like**, since the nav here is an oracle), and `ha0` (constant velocity at
the measured `v0`) which is the **shared floor** every arm must be scored against.

```bash
OMP_NUM_THREADS=6 python taniteval/tools/refcv3_arm.py \
  --ckpt ckpt_40284.pt \
  --config config.json \
  --episodes <B1 eval v2 cache> --labels <s2_labels_v7.2_eval.jsonl.gz> \
  --nav-source v72 --grid 2s --action-units steer --with-navzero \
  --dump-dir <dump> --out refcv3-40284-openloop.ARM.json
```

⚠️ **On checkpoint filenames — the earlier note on this card was correct and is now
superseded.** `refc_v3_train.py:103` sets `MILESTONES = (5000, 15000, 20000, 30000)`, so **the
trainer never writes a `ckpt_40284*.pt`**: only a milestone step gets a `ckpt_<step>.pt`, and the
final step writes **`ckpt.pt`** (model + optimizer + step). `ckpt_40284_FINAL.pt` was created
**afterwards, by the evaluation stream**, as an immutable copy of that `ckpt.pt` — md5
`fc304b62686ddb9e685d14bdab482404`, identical at both ends. **Both names therefore refer to the same
bytes**; this repo ships it under the explicit step-stamped name, and **no file here is called bare
`ckpt.pt`, on purpose** (see §6).

⛔ **`OMP_NUM_THREADS` is not optional for a multi-arm panel.** torch spawns ~113 threads per
process; concurrent arms without it sit at 0–6 % GPU making no progress and look exactly like a
hang.

**Open ruling.** The *tier* of the `os` (one-shot) arm is unresolved in the programme's register.
The instrument stamps it `T1` with `status: UNRULED` on every block and frames every headline as a
**margin over the shared `ha0` floor**, so the numbers above are correctly labelled whichever way
the ruling lands — but a cross-model comparison must go through that margin, never through `os`
levels against another model's closed-loop levels.

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
repo. ⚠️ **With no replicate, none of §4's margins can be separated from seed variance**, and the
step-30,000 → step-40,284 movement is one trajectory, not a learning curve.

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

MEASURED — `supervisor.log` (shipped in this repo). **This is not a clean single-recipe run**,
and it should not be described as one. ⭐ **However, the final segment is clean:** the supervisor log
records **no relaunch after step 18,500**, so the last **21,784 steps ran uninterrupted under one
recipe** (`--nav-from-v7 --u8-batches`) to the final step 40,284.

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

**Which file do you want?** For inference or evaluation, `ckpt_40284.pt`. To resume training,
`ckpt_40284_FINAL.pt`. **⛔ No file in this repo is named bare `ckpt.pt`** — every checkpoint carries
its step in its filename, on purpose, so the final and the earlier weights can never be confused.

| file | contents | step | size (B) | md5 | evaluated? |
|---|---|---|---|---|---|
| ⭐ **`ckpt_40284.pt`** | **the FINAL weights §4.2 measures** — `{model: state_dict (544 tensors), step}` | **40,284** | 428,518,255 | `b1ed7075ff730d0993d2eaa3c86f6b56` | ✅ **yes — §4.2** |
| **`ckpt_40284_FINAL.pt`** | the same final weights **plus optimizer state**, for resuming — `{model, opt, step}` | **40,284** | 1,284,991,701 | `fc304b62686ddb9e685d14bdab482404` | ✅ its `model` is **bitwise identical** to `ckpt_40284.pt` (verified: 544/544 tensors equal) |
| `ckpt_30000.pt` | the **earlier** step-30,000 weights — `{model: state_dict (544 tensors), step}` | 30,000 | 428,519,790 | `00da81c6efcd91e7b618a1fbddb3b78f` | ✅ yes — the step-30,000 column in §4.2 |
| `eval/refcv3-40284-openloop.json` | **raw four-family suite output for step 40,284** — every number in §4.2 | 40,284 | — | `5cfe3258c18871218bd85d691904eb20` | — |
| `eval/refcv3-40284-openloop.ARM.json` | raw per-arm output for step 40,284 — the arm **levels** and paired deltas | 40,284 | — | `0de8e8a4ece162332bc3a387fd5d679c` | — |
| `eval/refcv3-30k-openloop-20260903-2004.json` | raw per-arm output for the **earlier** step 30,000 | 30,000 | — | — | — |
| `config.json` | the run's own config: `argv`, `param_breakdown`, `horizons`, `goal_tau_steps`, `goal_provenance`, `provenance_roles`, `v2_parity`, `v7_labels`, nav-derivation stats | whole run | 4,191 | `8d8e10084c84cb2e81084a91facd3345` | — |
| `metrics.jsonl` | per-step training rows and the in-training T0 monitor rows, **through step 40,284** | whole run | 455,804 | `c50041e883c6c51274aa4ecfa308c59e` | — |
| `summary.json` | the run's own done-marker: `{"done": true, "final_step": 40284, "target": 40284}` | — | 53 | `bb8bdc7aad73b80c30898cf2941b6d65` | — |
| `supervisor.log` | the relaunch history behind caveat C7 | — | — | — | — |

Verify after download — every md5 above was MEASURED on the training pod **and again after
transfer**, and both agree:

```bash
md5sum ckpt_40284.pt        # b1ed7075ff730d0993d2eaa3c86f6b56
md5sum ckpt_40284_FINAL.pt  # fc304b62686ddb9e685d14bdab482404
```

Load it with:

```python
import torch
ck = torch.load("ckpt_40284.pt", map_location="cpu", weights_only=True)
print(ck["step"], len(ck["model"]))       # 40284 544
```

The architecture is `tanitad.refs.refc_v3.RefCV3Model` built from
`tanitad.refs.refc_v3.refc_v3_config()` with `image_hw = (256, 640)`.

---

## 7. Intended use, limitations and licence

**Intended use.** Research and evaluation only: reproducing the TanitAD reference-arm ladder,
studying goal-mediated hierarchies in supervised trajectory models, and benchmarking against
world-model arms on the same corpus.

**Out of scope.** Vehicle operation of any kind; any safety-relevant decision; any deployment
where a wrong trajectory has physical consequences. What §4 measures is **open-loop trajectory
accuracy**, which is not driving performance: this model has **never driven closed loop**; a
**trivial hold-action control still predicts trajectories better than it does** on this corpus
(+0.1423 m, separated); its **acceleration** is worse than constant velocity (+0.2020 m/s²,
separated); it misses **62 % of braking decisions** and 66 % of accelerations; its **STRATEGIC
family is unmeasured (n = 0)**; and it was trained on a single research corpus, on **one seed with
no replicate**, with an **oracle** navigation input that will not exist at deployment.

**Known failure modes to expect.** Degenerate anchor selection (one anchor chosen on every window)
is the characteristic failure of this architecture, and it is **not** caught by the standard
trivial-profile gate — that gate reads `trivial_frac 0.0000` on a randomly-initialised RefCV3 that
is selecting a single anchor on every window. **Check the selection profile explicitly.** For the
shipped step-40,284 weights it was checked and is clean (50/128 anchors used, modal anchor 14.82 %,
entropy ratio 0.5858, `degenerate: false`); the step-30,000 weights are likewise clean (51/128,
ratio 0.5782). Beyond that: the longitudinal decision head is weak in both directions, the 6 s
slots are the least supervised and are partially masked during training
(`slot_valid_frac ≈ 0.92`), and the **yaw-rate** metric flips sign between its filtered and
unfiltered populations (§4.2) — never quote it without its scope.

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
