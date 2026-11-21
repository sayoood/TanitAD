# PRE-REGISTRATION — `os_navpred`: the planner driven by its OWN predicted route

**Written BEFORE any GPU cycle was spent on this arm.** Skill: `TanitAD_ValidateAIDesign`.
**Author:** nav-replacement agent, 2026-09-06. **Branch:** `agent/arch-inf-20260803`.

---

## 0. WHY — the lever this tests, and the exact number it attacks

The refcv4b landing read (`…/2026-09-06-refcv4b-landing/LANDING_RESULT.md`, commit `b01cc14`)
established, on n = 4,823 windows / 141 held-out episodes:

| fact | value |
|---|---|
| refcv4b ties hold-action | `os − ha` **−0.0021** [−0.0178, +0.0154], not separated |
| refcv4b ties the echo control | `os − ha0_ext` **+0.0101** [−0.0050, +0.0273], not separated |
| ⛔ **strip the nav and it LOSES** | `os_navzero − ha0_ext` **+0.1054** [+0.0874, +0.1241], **separated WORSE** |
| what the supplied route is worth | `os − os_navzero` **−0.0953** [−0.1096, −0.0804], separated |

⇒ **the margin that carries refcv4b level with the echo control is supplied by an input the model
does not compute.** This arm asks whether the model can supply it itself.

⚠️ **THE ADMISSIBILITY FRAME, STATED SO IT IS NOT OVERCLAIMED.** `D-NAV-IS-GT` (BINDING PI ruling,
2026-09-04) holds that the v7.2 nav command **is** ground truth and a first-class input — *"a ROUTE
signal of the kind a map router supplies at deployment"* — and names **`os` the deployment arm**,
with `os_navzero` a **robustness ablation**. The same ruling records the caveat this arm measures:
*"ours is noiseless and perfectly timed because it is derived from the ego's future path (PhysicalAI
ships no map) — a CLEAN version of the deployment signal."*
⇒ **This SPEC does NOT claim `os` is inadmissible.** It measures **how much of the supplied route's
value the model can produce from vision alone** — i.e. the residual when no external router exists,
which is the deployment condition on a corpus with no map. That is a *comparability* question, and
the ruling names it as one.

## 1. THE ARM

`os_navpred` — identical to `os` in every respect except **where the nav token comes from**:

```
route_z  = argmax  model(frames, nav_cmd=None, v0, ego).route_logits      # deployment-legal: no nav in
nav_pred = _ROUTE_TO_NAV[route_z]                                          # refb_labels.py:483-484, IMPORTED
os_navpred = model(frames, nav_cmd=nav_pred, v0, ego).traj                 # the deployed sel_score_v3 pick
```

⛔ **The predicted route is taken from the `nav_cmd=None` forward, not from the `nav_true` row**, so
**no oracle nav enters the arm at any point**. This is admissible under both binding PI rulings:
it is a **predicted goal** (2026-08-03: *"a goal input is admissible … including a predicted
geometric goal point"*) and it carries **no situation-classifier output** (the route head is E13's
strategic aux head reading the observed window; the situation classifier is a different instrument
and is not in this path).

⚠️ **A per-window token where the oracle is per-CLIP.** `nav_cmd` is one token per episode
(`ds._nav_by_sid[episode_id]`); the route head predicts **per window**, so `os_navpred`'s token can
change within a clip. That is what a deployed router-free system would do, and it is a **stated
asymmetry**, not a defect.

## 2. THE ONE VARIABLE, AND WHAT IS HELD

```yaml
hypothesis: H-NAVPRED-1
one_variable: the SOURCE of the nav token fed to the forward (oracle v7.2 token -> the model's own route argmax)
held_constant: [ckpt md5 99b573e8277d94a5e3bfbf630cb4d751, config.json md5 58ad809aaf241729c1695bec2ab30d8e,
                anchors.pt md5 297f6f1db52f6a56094846b0d7f71ed9,
                labels md5 aa12c948f062181c3297265b51526ec5, the 141 eval clip ids,
                grid 2s (dt 0.5, K 4), window_stride 5, action-units steer,
                lead-block md5 33a48e15a52eb9dbd69ecd6026fd5023,
                estimator (paired episode-cluster bootstrap, n_boot 2000, seed 0),
                all six reference arms rolled in the SAME dump]
controls: [model_free_known_value, os_reproduction, navpred_marginal_regression,
           route_head_non_circularity, coverage_identity, n_and_grid]
splits: {fit: "none - no parameter is fitted", val: "none", test: "the 141 held-out episodes, scored, never tuned on"}
```

⛔ **No hyper-parameter is selected on the scored split.** Nothing is fitted at all: the arm is a
forward pass with a different input.

## 3. COMMITTED OUTCOMES — both written before the roll

Let `A(x)` = ADE over the 2 s grid, and every margin be the **paired episode-cluster bootstrap**
(`taniteval/ci.py`, B = 2,000, seed 0) on the same 4,823 windows.

Reference values (same surface, banked): `os` **0.2975** · `ha` **0.2996** · `ha0_ext` **0.2874** ·
`ha0` **0.6723** · `os_navshuf` **0.3013** · `os_navzero` **0.3928**.

Define the **recovery fraction**
`R = (A(os_navzero) − A(os_navpred)) / (A(os_navzero) − A(os))`, denominator = 0.0953 m.

| outcome | criterion, committed in advance | verdict |
|---|---|---|
| ⭐ **SUCCESS** | `os_navpred − os_navzero` **separated NEGATIVE** (CI upper < 0) **AND** `os_navpred − ha0_ext` **NOT separated worse** (CI lower ≤ 0) | **H-NAVPRED-1 SUPPORTED** — the oracle dependency is largely removable **now**, on the existing checkpoint, and refcv4b's tie with the trivial controls is not borrowed |
| ⚠️ **PARTIAL** | `os_navpred − os_navzero` separated NEGATIVE **but** `os_navpred − ha0_ext` still separated WORSE | **PARTIALLY SUPPORTED** — report `R` and the residual gap; route prediction is a refcv5 *improvement* target, not a refcv5 *blocker* |
| ⛔ **FAILURE** | `os_navpred − os_navzero` **not separated**, or positive | **H-NAVPRED-1 REFUTED** — the model's own predicted route carries little/none of the supplied route's value; the 0.1054 m gap stands and **route prediction becomes a refcv5 REQUIREMENT** (a geometric goal point, per the +4.7 vs +0.2 PDMS literature lever), not a refcv4b fix |
| ⛔ **ADDITIONAL, and it gates the causal reading** | `os_navpred − os_navshuf` must be **separated NEGATIVE** for any claim that the *prediction* rather than the *marginal* is doing the work | if not separated, the arm is reported as **indistinguishable from a distribution-matched random token** |

⭐ **Which variance the interval answers, stated in advance.** `os`, `os_navpred`, `os_navzero` and
`os_navshuf` come from **one checkpoint, one dump, the same windows**, and refcv4b's diffusion
decoder is **deterministic at inference** (`refc.py:1599` — `torch.zeros_like` when not training,
recorded in `D-REFCV4B-SEED-SCOPE`). ⇒ neither the **training-run** variance (`H-ESTIM-SEED-1`) nor
the **inference** variance enters a *within-model input intervention*: the only remaining question
is *"would another draw of EPISODES say this?"*, and that is exactly what the episode-cluster
bootstrap answers. ⛔ This scoping licence covers **only** the within-model nav margins. Any
statement comparing refcv4b to **another trained model** still inherits `H-ESTIM-SEED-1`.

## 4. CONTROLS THAT MUST READ KNOWN VALUES — hard gates, checked before any family is read

| # | control | must read | if it fails |
|---|---|---|---|
| C1 | **model-free arms** `ha`, `ha0` on the new surface | **0.2996** / **0.6723**, the banked pod values | ⛔ the Thor surface is not the pod surface — **VOID**, nothing is comparable |
| C2 | **grid** | exactly **4,823** windows / **141** episodes, `window_stride 5`, grid `2s` | ⛔ VOID |
| C3 | **`os` reproduction** | \|ΔADE\| < **0.001 m** vs 0.2975 (it moves from row 0 of a 2-row batch to row 0 of a 3-row batch; the documented cross-call GEMM floor is **5.96e-07 m**) | ⛔ investigate before reading anything |
| C4 | **route-head non-circularity** | `route_pred` under the `nav_predicted` row equals `route_pred` under `nav_zero` on **≥ 4,822 / 4,823** windows (the banked identity rate) | ⛔ feeding the model its own prediction moved the head ⇒ the construction is circular and the arm is **VOID** |
| C5 | **coverage identity** | on the windows where `nav_pred == nav_cmd`, `os_navpred`'s path equals `os`'s to **< 1e-4 m** | ⛔ the token is not reaching the forward |
| C6 | **the arm is not degenerate** | `nav_pred != nav_cmd` on **1,917 / 4,823** windows (**predicted from the banked dump before the roll**; see §5) | ⛔ if it reads 0, the intervention did not take |
| C7 | **the route→nav map is IMPORTED, not re-derived** | `refb_labels._ROUTE_TO_NAV.__module__`-level assert: the mapping used is the one in `stack/scripts/refb_labels.py:483-484` | ⛔ a re-implemented control drifts from the thing it controls |

⭐ **The deliberate-regression arm is already in the panel and is `os_navshuf`** — a nav token with
the pairing destroyed and the marginal preserved. If `os_navpred` cannot beat it, the *prediction*
adds nothing over a plausible random token, and a SUCCESS verdict is refused regardless of §3.

## 5. THE ZERO-GPU PREDICTION THIS SPEC COMMITS TO (falsifiable before the roll)

Computed from the **banked** dump `refcv4b_t1_dump/decisions/*.npz` (`route_pred_nav_zero`,
`nav_cmd`), applying `_ROUTE_TO_NAV`, with **no forward pass**:

| predicted quantity | committed value |
|---|---|
| `nav_pred` distribution over 4,823 windows (follow / left / right) | **3,465 / 614 / 744** |
| windows where `nav_pred != nav_cmd` | **1,917** (39.75 %) |
| windows where `nav_pred == nav_cmd` | **2,906** (60.25 %) |
| `nav_cmd` distribution (follow / left / right) | 3,080 / 445 / 1,298 |

⇒ the model **under-calls right turns 744 vs 1,298** and over-calls `follow` — the direction the
route head's per-class recalls already implied (`route_right` 0.411, `route_left` 0.357).
The roll must reproduce all four rows exactly, or C6 has failed.

## 6. FOUR METRIC FAMILIES — binding, and none may be dropped

LONGITUDINAL (speed MAE + bias + target-speed accuracy + along-track + distance-keeping),
LATERAL (heading, yaw-rate, **curvature MAE with the `ha0` straight-line floor beside it**,
cross-track), TACTICAL (lat/lon decision accuracy + κ + per-class recall, goal/anchor selection),
STRATEGIC (route accuracy + κ + per-class recall + echo index). Per family, never pooled.
⛔ **`oracle_sel`, `anchor_acc` and `sel_agrees_oracle` are NOT quoted** — `D-REFCV4B-ASTAR-GEOMETRY`
(the `a_star` geometry defect) is unrepaired, so `--with-oracle-sel` is **OFF** for this roll.
⛔ Curvature is read against **`ha0` 0.006802 and `ha0_ext` 0.003712**, never alone.

## 7. EXECUTION

Compute: **Thor** (`tanitad-thor-wifi`), `OMP_NUM_THREADS=6`. ⛔ The A40 is left alone for refcv5.
All inputs md5-verified at both ends before the launch (§2 `held_constant`).
Artifacts: `raw/refcv4b_navpred.json`, `raw/navpred_dump/`, `raw/PREDICTION_ZERO_GPU.json`,
`raw/paired_navpred.json`.
