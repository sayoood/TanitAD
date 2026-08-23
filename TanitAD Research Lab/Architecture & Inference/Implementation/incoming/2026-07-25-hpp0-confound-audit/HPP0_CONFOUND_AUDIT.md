# HPP-0 — the confound audit. What is actually broken, measured.

**Date:** 2026-07-25 (Europe/Berlin) · **Agent:** hpp0-confound-audit · **Compute:** dev box only,
**zero GPU, zero pod SSH, read-only on every checkpoint and corpus.**
**Spec:** `Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/01_EXECUTION_PLAN.md`
Part A (the Hierarchy Proof Program, PC1–PC4, HP-1…HP-6).

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + file:line or JSON path) ·
`PUBLISHED` · `INHERITED` (another doc/agent, NOT re-verified here) · `ESTIMATED` · `HYPOTHESIS`.
Every "X is broken" claim below cites code or a raw artifact. Nothing here is a model-quality claim.

**Framing (binding, PI 2026-07-25).** The hierarchy thesis is not on trial. This document measures
*what would have to be true for a hierarchy effect to be visible*, and finds that **none of the four
pre-conditions holds today**. Every one of the four is a fixable instrument/label/wiring defect. Two
of the four fixes already exist in the repo and have simply never been carried to a scored checkpoint.

---

## 0. Verdict

| PC | Pre-condition | Met today? | The single sentence that decides it |
|---|---|:--:|---|
| **PC1** | The route input must actually work | 🟥 **NOT MET** | The route *target* is a pure function of the route *input* (`refb_labels.route_target = _NAV_TO_ROUTE[nav_cmd]`), so the deployed v1's route CE reaches **exactly 0.0** by step ~14.5 k and stays there — and with the command withheld the head answers `straight` on **240/240** valid windows. |
| **PC2** | The hierarchy must be in the loop for the scored number | 🟥 **NOT MET — and worse than reported** | `metric_dynamics.rollout_decode` calls `predictor(win_s, win_a)` with **no `intent`, no `ctx`, no `nav`** — *and* it is fed the **expert's true future actions** (`rollout.py:127-129`, `bench.py:292-294`). The 0.4271/0.4522 headline is a dynamics decode of a known control sequence; it cannot express a decision at *any* level. |
| **PC3** | The instrument must be able to see it | 🟥 **NOT MET** | Every standing eval path tops out at **K=20 steps = 2.0 s** (`WP_STEPS` max 20 in `rollout.py`, `bench.py`, `closedloop.py`, `hierarchy.py`), and `driving.py` tier-0 **explicitly refuses** intersection/roundabout/merge and lane-centre metrics. `corridor_departure_rate` exists in **zero** files under `taniteval/taniteval/`. |
| **PC4** | The corpus must contain decisions | 🟥 **NOT MET** | On the canonical val (40 eps / 881 windows) only **240 windows (27.2 %) carry a judgeable route label at all**, exactly **6 per episode** — i.e. ≈ **40 route situations, ~13 of them turns**, against PC4's requirement of n ≥ 40 *decision* episode-clusters. |

**One-line summary.** The hierarchy has never been given a route to follow, has never been in the
scored loop, has never been measured at its own timescale, and has never been shown a corpus with
choices in it. Four independent instrument failures — **not** four pieces of evidence against H1.

---

## 1. PC1 — the route/nav input, traced end to end

### 1.1 How the command is produced — and why it cannot teach anything

`MEASURED` — `stack/scripts/refb_labels.py:138-175`:

```
nav_command(poses, t)  ->  net heading change of the EGO'S OWN FUTURE poses over
                           min(NAV_HORIZON_STEPS=250, available) steps, thresholded
                           at NAV_TURN_RAD = pi/4  ->  (NAV_LEFT | NAV_RIGHT | NAV_FOLLOW, valid)

route_target(nav_cmd)  ->  return _NAV_TO_ROUTE[nav_cmd]          # refb_labels.py:172-175
```

Two defects, both structural:

1. **The "route command" is not a route.** It is a *shape descriptor of the ego's realized future
   trajectory*. There is no map, no lane graph, no waypoint sequence, no alternative branch. Nothing
   in the input tells the model *which of several available roads* to take, because the corpus has no
   representation of "several available roads" (§4).
2. **The supervision is circular.** The route-head target is a **deterministic lookup of the input
   command**. A network with a 4-entry `nn.Embedding` on its FiLM condition
   (`fourbrain.py:58,77`) can reach zero loss by copying the embedding to the logits. There is no
   gradient anywhere that rewards inferring the route from the scene.

### 1.2 Coverage collapse — the command is *wrong*, not merely uninformative, on ~73 % of windows

`MEASURED` — `refb_labels.py:71-73,161-169`: `NAV_HORIZON_STEPS = 250` (25 s), `NAV_MIN_STEPS = 150`
(15 s). PhysicalAI clips are ~199 frames (~20 s), so `h = T-1-t >= 150` holds only for
`t_last <= T-151 ≈ 48`. With `window=8`, `K_MAX=20`, `stride=8`, an episode yields 22 windows of which
**exactly 6** clear the guard.

- On the other ~73 %, `nav_command` returns **`(NAV_FOLLOW, False)`** — and `NAV_FOLLOW` **is fed to
  the model as a real input** (`flagship_losses.py:239-245`; the `valid` flag only masks the CE).
  The strategic level is therefore told **"follow"** on three quarters of windows *including the ones
  where the vehicle is mid-turn*.
- Corroborated in the trainer logs (`MEASURED`, `taniteval/results/trainlogs/*_train_log.jsonl`,
  key `nav_valid_frac`, batch 16): v1 **0.125–0.3125**, v2 **0.125–0.5**, v3enc **0.0625–0.375**,
  nospeed **0.25**. Matches the retraction-log figure of 0.21–0.25 (`RETRACTION_LOG.md:33`).

### 1.3 What training does with it

| Arm | `v2_labels` | `v2_nav_dropout` | `v2_route_from_vision` (LEVER A) | Steps reached | Hierarchy panel exists? |
|---|:--:|:--:|:--:|---:|:--:|
| **flagship v1 `flagship4b-speedjerk-30k`** (deployed, 0.4271) | ❌ | ❌ | ❌ | 29 999 | ✅ |
| `nospeed-phase0` (ablation control) | ❌ | ❌ | ❌ | 22 950 | ✅ (via v1 panel lineage) |
| flagship v2 | ✅ | 0.5 | ✅ | **7 700 (abandoned)** | ❌ |
| flagship v3enc | ✅ | 0.5 | ✅ | **10 800 (abandoned)** | ❌ |

`MEASURED` — `taniteval/results/trainlogs/{v1-speedjerk,v2,v3enc,nospeed-phase0}_config.json`
(`cfg.v2_labels`, `cfg.v2_route_from_vision`, `cfg.v2_nav_dropout`).

**The echo, measured at the loss:** `v1-speedjerk_train_log.jsonl`

| step | `route` (CE) | `route_acc` | `nav_valid_frac` |
|---:|---:|---:|---:|
| 0 | 1.0166 | 0.6667 | 0.1875 |
| 100 | 0.5522 | 1.0 | 0.1875 |
| 14 550 | **0.0** | 1.0 | 0.125 |
| 29 999 | **0.0** | 1.0 | 0.1875 |

`nospeed-phase0` @22 950: `route` **2.98e-08**, `route_acc` 1.0. An **exactly-zero** cross-entropy is
the signature of a deterministic input→target map, not of learning.

⚠️ **`route_acc` in the trainer log is not admissible** (class C1 + tiny n): it is computed over the
`nav_valid` subset of a **batch of 16**, i.e. **1–5 rows**. `route_acc = 1.0` at n=1 is a coin flip
that landed. Only the `taniteval.hierarchy` panel numbers below are quotable.

**LEVER A already exists and has never finished a run.** `flagship_losses.py:329-342` adds a second
strategic pass with nav forced to `follow(0)` and a class-weighted CE against the true route — exactly
the PC1 fix. `MEASURED` on the two arms that ran it: the aux loss does **not** collapse (last-20-row
mean `route_vis` — v2 **0.763**, v3enc **0.608**, expA-nodrop **0.761**), i.e. route-from-vision is a
genuinely hard, non-degenerate objective. Both arms died before any evaluation
(7 700 / 10 800 steps; `ARCHITECTURE_WIRING_COMPARISON.md:2.8` — "flagship v3enc: no checkpoint has
been evaluated at all"). *(Note: `route == route_vis` on ~50 % of logged rows is not a bug — per-row
nav-dropout plus a 67 %-follow label distribution makes both passes identical whenever every valid row
is effectively `follow`.)*

### 1.4 What the eval harness actually feeds — per arm. **Three different failures, none of them "works".**

| Eval path | What is fed as route/nav at inference | Class of failure | Evidence |
|---|---|---|---|
| **flagship v1 leaderboard** (`taniteval/rollout.py::collect` → `driving.tier0`) | **nothing** — `rollout_decode` has no nav/ctx/intent parameter | route input **absent from the scored path** | `metric_dynamics.py:220-244`; `rollout.py:127-136` |
| **`taniteval.hierarchy` panel** | `follow = zeros(b)` for the deploy read; true `nav` only for the by-construction control | constant command | `hierarchy.py:275,282` |
| **`taniteval.planning` panel** | `follow = zeros(b)` | constant command | `planning.py:154-157` |
| **REF-C `refc_eval.py`** | `model(fw, nav_cmd=None, …)` → `nav_cmd = zeros` inside | **`nav_cmd=None` — the 07-21 C6 confound, still live** | `refc_eval.py:78`; `refc.py:786-788` |
| **REF-C `refc_rerank.py`, `plan_fan.py`** | `nav_cmd=None` | same | `refc_rerank.py:262`, `plan_fan.py:549` |
| **v1.5 `eval_flagship_v15.py`** | `route = labels["route"][e][ch]`, `route_graded`, `vt_band`, `vt_speed` — **GT labels minted from the ego's own future** | **route ORACLE at eval** | `eval_flagship_v15.py:92-103,193-195` |
| **v1.6 `eval_flagship_v16.py`** (headline 0.4375) | identical GT-label feed | **route ORACLE at eval** | `eval_flagship_v16.py:135-143,219-221` |
| **v4 `eval_flagship_v4.py` MODE B** | `_goal_inputs(head.cfg, b, v0)` reads `route`/`route_graded` off a `FlagshipV4Dataset` batch, which mints them **per window from the full-episode future poses** | **route ORACLE at eval** | `eval_flagship_v4.py:127-143,322`; `train_flagship_v4.py:167-173`; `flagship_v4_data.py:12-14,67` |

**This is the finding that reframes PC1.** The program does not have one route-input bug, it has
**three mutually exclusive ones**, and no arm has ever been scored with a *produced, non-oracle,
non-constant* route:

- **Echo** (v1 / REF-A / REF-B): input present, target is a function of the input, seam untestable.
- **Withheld** (REF-C): input never exercised → a marginal-mode decoder is compared to a hierarchy.
- **Oracle** (v1.5 / v1.6 / v4): input exercised but derived from the future → the number is not
  deployable and is not comparable to either of the above.

`V4_FLAGSHIP_DESIGN.md:558-560` states the rule that the oracle path violates: *"No leaderboard number
may come from a GT-derived plan or a GT-derived goal."* `V4_FLAGSHIP_DESIGN.md:806-807` registers the
fix as a **precondition that has not landed**: *"the v4 evaluator must feed the produced goal.
`refc_eval.py` / `plan_fan.py`'s `nav_cmd=None` constant must not be inherited."* Same item as
`§15 P7(c)`. It is still open in both directions (§7 below).

### 1.5 `route_skill` today — MEASURED

Two definitions exist and they agree:

- `taniteval/planning.py:201` — `route_skill_vs_chance = route_acc_follow − majority_route_base_rate`.
- `taniteval/hierarchy.py:423-424` — `vision_route_beats_majority = route_acc_follow > majority + 0.03`,
  the gate emitter `nonav_route_beats_majority` (`gate_emitters.py:176-213`).

| Arm / artifact | `route_acc_nav` | `route_acc_follow` | `majority_straight_rate` | **route_skill** | follow prediction histogram | `n_valid` |
|---|---:|---:|---:|---:|---|---:|
| flagship v1 30k (`hierarchy_flagship-30k.json`) | **1.0000** | 0.6708 | 0.6708 | **0.0000** | L 0 / **S 240** / R 0 | 240 |
| flagship v1 30k (earlier 72-window run) | **1.0000** | 0.7083 | 0.7083 | **0.0000** | L 0 / **S 72** / R 0 | 72 |
| flagship v4.2b step-4000 (`hierarchy_flagship-v4.2b-dryrun.json`) | **1.0000** | 0.6708 | 0.6708 | **0.0000** | L 0 / **S 240** / R 0 | 240 |

`MEASURED` — `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-{30k,v4.2b-dryrun}.json` and
`…/2026-07-23-v4-gate-emitters/artifacts/hierarchy_flagship-30k_v1.json`, key
`seam_nav_to_strategic`.

**`route_acc_nav = 1.0000` is the proof of the echo, not a success.** It is perfect accuracy on a
target that is a lookup of the input. `route_acc_follow` equals `majority_straight_rate` to **four
decimal places** while the head emits `straight` on every single valid window. This is *exactly* the
signature the brief describes, and it reproduces on two checkpoints, two window counts and two
architectures.

`nonav_route_beats_majority = 0 → FAIL`, and it is a **KILL** secondary of the v4 30k gate
(`V4_FLAGSHIP_DESIGN.md:1305,1376`). `INHERITED` from the v4 gate dry-run: as written, the 30k gate
fails on this metric alone.

⚠️ **Estimator caveat, and it matters for the PC1 exit criterion.** The whole `hierarchy.py` panel
uses `_jack` = `overlapping_holdout_se` (`hierarchy.py:154-181`), which the panel itself labels
*"DEPRECATED, not a jackknife"* and which CLAUDE.md records as **1.28–2.06× too narrow**. PC1's gate
says *"CI-separated from the majority-class baseline"* — **that CI does not exist yet in an admissible
estimator.** Porting `hierarchy.py` to `ci.episode_cluster_bootstrap` is a prerequisite of PC1's own
pass criterion (already listed as `V4_FLAGSHIP_DESIGN §15 P7(a)`, "do this first — it changes what the
other panels mean").

### 1.6 Is there any path by which the strategic level infers route from vision?

**Architecturally yes; in supervision, no; in any scored checkpoint, no.** Probed at four locations
(CLAUDE.md rule: absence at one location is not absence):

1. `fourbrain.py:41-86` — `StrategicPolicy` has a `route_head` on the transformer state, so the
   *capacity* exists; the `nav_emb` FiLM is what makes it copyable.
2. `hierarchy.py:112-121` `_zero_nav` — the panel can and does run the head with the nav embedding
   zeroed (`route_acc_zeronav` 0.2167–0.2361, i.e. **below chance 0.333**: with the command removed
   the head is worse than guessing).
3. `flagship_losses.py:329-342` — LEVER A is the deterministic vision-only training signal. Present
   in code, enabled only in two abandoned arms (§1.3).
4. `refb_labels.py:459-745` — **v2 / v2.1 / v3 labelers break the circularity by construction**:
   `route_target_v2` is *"the curvature-relative target, NOT a function of any fed nav command
   (breaks the v1 circularity where the input command and the target were the same derivation)"*
   (`refb_labels.py:468-474`), and v2.1 refuses to emit `straight` when it cannot judge
   (`ROUTE_UNKNOWN`, `refb_labels.py:511-516`). Coverage rises from 27 % to **80.4 %**
   (`labels_train_v4_provenance.json` → `coverage.route_valid = 0.8043`). **No completed, evaluated
   arm has ever trained on them.**

Fifth probe, v4 specifically: `stack/tanitad/models/strategic_goal.py` — the only strategic module in
the v4 lineage is a `GoalScalarHead`, a 2-layer MLP emitting **four continuous scalars**
(ttm · curv@3 s · curv@5 s · tspeed@5 s). **There is no route classifier in v4 at all**; the strategic
planner ① is unbuilt (P6). `eval_flagship_v4.py:640-650` records this in the gate output verbatim.

### ✅ PC1 — what must change (exact, file-level)

| # | Change | Where |
|---|---|---|
| 1 | Train the route head against a target that is **not** a function of the fed command — i.e. `refb_labels.route_target_v21` / `route_from_future_v3`, never `route_target(nav_cmd)` | `flagship_losses.py:318-327`; trainer flag `--labels-v2` (already exists) |
| 2 | Turn LEVER A on and **carry it to a scored checkpoint** — the always-on nav-forced-to-follow aux CE | `cfg.v2_route_from_vision` (`train_flagship4b.py:251`), weight `--route-vis-weight` (default 0.3) |
| 3 | Replace the 25 s / 15 s fixed-horizon labeler with the **adaptive-arc v2.1** so coverage goes 27 % → 80 % and unjudgeable ≠ straight | `refb_labels.route_from_future_v21`; already written, unused by every finished arm |
| 4 | **Kill `nav_cmd=None`** in the REF-C eval trio | `refc_eval.py:78`, `refc_rerank.py:262`, `plan_fan.py:549` |
| 5 | **Kill the GT-route feed** in the v15/v16/v4 evaluators, or report every number twice (oracle-goal **and** produced-goal) with the estimator named | `eval_flagship_v15.py:92-103`, `eval_flagship_v16.py:135-143`, `eval_flagship_v4.py:322` |
| 6 | Port the hierarchy panel's CI to `ci.episode_cluster_bootstrap` so `route_skill > 0` can be *CI-separated* as PC1 requires | `hierarchy.py:154-181` (`_jack`) |

---

## 2. PC2 — is the hierarchy in the loop for the scored number?

### 2.1 The scored path, verbatim from source

`MEASURED` — `stack/tanitad/models/metric_dynamics.py:220-244`:

```python
def rollout_decode(predictor, states, actions, future_actions, step_readout, k):
    win_s, win_a = states, actions
    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]          # <-- NO intent=, NO ctx, NO nav
        dposes.append(step_readout(win_s[:, -1], z_hat))
        if j < k - 1:
            a_next = future_actions[:, j]           # <-- the EXPERT'S TRUE next action
            ...
    return accumulate_se2(step_dpose), step_dpose
```

Callers (`MEASURED`):

- `taniteval/rollout.py:126-136` — `fa = ep.actions[t+window : t+window+fwd_k]`, then
  `rollout_decode(model.predictor, states, aw, fa, step_readout, fwd_k)`.
- `taniteval/bench.py:291-298` — identical.
- `stack/scripts/eval_grounded_rollout_4b.py:13-15` — its own docstring: *"roll the OPERATIVE
  predictor `fwd_k` steps under the **TRUE action sequence** (intent-free…)"*.

`driving.tier0` — the leaderboard block — consumes exactly this window dump
(`driving.py:526-540`, `from_windows`).

### 2.2 Two bypasses, and the second is the bigger one

**(a) The hierarchy bypass** (already documented, re-verified here): zero of the three seams
participate. `ARCHITECTURE_WIRING_COMPARISON.md:207-216` calls this *"the most important row in the
document."* Confirmed independently from the function signature above.

**(b) The decision bypass — and this one is not in the review.** The scored rollout is fed the
**expert's actual future `[steer, accel]` sequence at every step**. The model is not choosing a
trajectory; it is *decoding a trajectory it has been told*. The registry says so in its own words —
`MODEL_REGISTRY.md:1419-1420`: *"operative rollout with **true** actions (**the WM ceiling**) 0.452
open-loop / 0.424 closed-loop"*, sitting next to the P2 planner's **0.893** when the actions must
actually be *chosen*.

**Consequence for the HPP:** the headline number cannot express a strategic decision, a tactical
decision, **or an operative decision.** A hierarchy-vs-flat ADE comparison on this surface is not
merely underpowered — it is measuring a quantity in which no policy of any shape can differ, except
through the fidelity of its dynamics decode. This is the strongest single argument for PC2, and it is
`MEASURED` from three files.

### 2.3 Per-arm scored surface

| Arm | Scored surface | Strategic in loop? | Tactical in loop? | Operative *chooses*? |
|---|---|:--:|:--:|:--:|
| **flagship v1 (0.4271)** | `rollout_decode`, true future actions | ❌ | ❌ | ❌ (actions given) |
| **flagship v2 / v3enc** | same | ❌ | ❌ | ❌ |
| **REF-A** | same | ❌ | ❌ | ❌ |
| **REF-B v1/v2 (0.5921)** | tactical head's direct per-horizon waypoints (`refb_eval.py`) | ⚠️ upstream of the head, never ablated | ✅ | n/a |
| **REF-C base/XL (0.4728 / 0.4714)** | `decoder(fmap, m, ctx, maneuver_logits, steps)` → 256 anchors → argmax | ⚠️ GRU `ctx` enters the confidence logits — with `nav_cmd = follow` constant | ⚠️ 5 maneuver logits reweight anchors | ✅ |
| **v1.5 / v1.6 (0.4375)** | anchored fan on the v15 head, selected on `refined_logits` | ⚠️ conditioned on a **GT** route token | ✅ | ✅ |
| **flagship v4 MODE B** | `head(st, v0, **_goal_inputs(...))` | ⚠️ **GT** route/vt tokens; no strategic module exists | ✅ | ✅ |
| **P2 CEM planner (0.893)** | CEM over 20×2 future actions, rolled through the frozen v1 WM | ❌ | ❌ | ✅ **yes — the only flagship-lineage number where actions are chosen** |

`MEASURED` (code) + `INHERITED` (the ADE values, from `Benchmarks & Eval/LEADERBOARD.md` §1 and
`MODEL_REGISTRY.md` §1.4b/§6).

### ✅ PC2 — what must change

1. **Assert it, don't inspect it.** Add a hard assertion in the eval harness that the scored
   forward pass traversed strategic → tactical → operative (e.g. a forward-hook counter on
   `strategic_policy`, `tactical_policy` and the predictor's `intent_proj`, asserted non-zero;
   fail loud if the arm claims a hierarchy and the counter is 0). Natural home:
   `taniteval/rollout.py::collect` and `taniteval/runner.py`.
2. **Score a path where actions are chosen.** Any HPP-4 comparison must run on
   `planner_p2`-style (or closed-loop) surfaces, never on the true-action `rollout_decode`. The
   true-action rollout stays as what it honestly is: a **WM-fidelity diagnostic**, and it should be
   renamed as such wherever it is quoted as "driving".
3. **Report the intent-free rollout and the intent-threaded rollout side by side**, as
   `hierarchy.py:489-499` already does under `diagnostic_intent_in_grounded_rollout_OUT_OF_REGIME` —
   but with the readout re-calibrated for the intent regime so the comparison is on-manifold.

---

## 3. PC3 — can the instrument see a strategic effect?

### 3.1 Horizons — every standing path is 2.0 s

| Path | Horizon | Evidence |
|---|---|---|
| `taniteval/rollout.py` → `driving.tier0` (the leaderboard) | `K_MAX = max(WP_STEPS) = 20` steps = **2.0 s** | `rollout.py:57` |
| `taniteval/bench.py` (diagnostic panel) | **2.0 s** | `bench.py` `K_MAX` |
| `taniteval/closedloop.py` (the closed-loop block) | `K_MAX = 20`, `HORIZONS_S = {5:"0.5s",10:"1s",15:"1.5s",20:"2s"}` | `closedloop.py:96-98` |
| `taniteval/hierarchy.py` (the seam panel) | `K_MAX = 20`, `GOAL_H = K_MAX` = **2.0 s** | `hierarchy.py:66-67` |
| `taniteval/planning.py` | 2 s waypoints | `planning.py` `WP_STEPS` |
| v4 head horizons | `(1,2,3,4)` × 0.5 s = **2.0 s** | `train_flagship_v4.py:359` |
| **E1a horizon sweep (one-off, `incoming/` bundle)** | **K=185 = 18.5 s** | `2026-07-25-closedloop-horizon-and-shift/e1a_horizon.py` |

`hierarchy.py:550-552` states the mismatch in its own comment: *"route is a 15-25 s heading;
maneuver/trajectory are 2 s — some cross-timescale disagreement is CORRECT."* **The panel that judges
the strategic seam therefore scores a 15–25 s quantity through a 2 s window.**

The E1a result is the proof that this blindness is decisive, not theoretical
(`MEASURED`, `e1a_horizon_heldout44_K185.json`, paired common-start, 43 identical windows,
`episode_cluster_bootstrap` B=2000):

| stratum | K=20 (2.0 s) CDR@1.75 m | K=185 (18.5 s) CDR | peak XTE 2 s → 18.5 s |
|---|---:|---:|---|
| overall | **0.0035** | **0.5877** | 0.35 m → **38.94 m** |
| junction | 0.0395 (n=124) | **0.8414** (n=6) | 1.23 m → **46.25 m** |

A 168× change in the failure rate between the standing horizon and the event's horizon.

### 3.2 Metrics that could express route compliance — inventory

**What exists in the standing harness (`taniteval/taniteval/`):** ADE/FDE per horizon, miss@2m,
along/cross decomposition, speed MAE, heading error, curvature-sign agreement, comfort/jerk bounds,
curvature and speed strata, plus the seam ablations and agreement/kappa in `hierarchy.py`.

**What does not exist there — probed twice each:**

| Missing capability | Probe 1 | Probe 2 | Status |
|---|---|---|---|
| `corridor_departure_rate` (the horizon-capable primary the review wants) | `grep -rn corridor_departure --include=*.py .` | `grep -rn "departure\|corridor" taniteval/taniteval/*.py` | **Zero hits inside the package.** Exists only in 5 one-off scripts under `…/Implementation/incoming/…` (`eval_corridor_split.py`, `tolerance_rescore.py`, `powered_departure_eval.py`, `e1a_horizon.py`, `e1b_eval.py`) |
| junction / multi-option stratification | `driving.py` strata are `curv_buckets`, `speed_strata`, `kinematic_strata` | `driving.py:581` explicitly **refuses** `intersection_roundabout_merge_capability`: *"events are 5-20 s, horizon is 2 s"* | **refused by design at 2 s** |
| route-compliance / goal-reached | no such key in `driving.py`, `closedloop.py`, `bench.py` | `hierarchy.py` reports route *accuracy*, never route *following* | **absent** |
| lane-relative position | `driving.py` refuses `lane_centre_deviation`: *"no lane geometry exists"* | `LANE_HALF_M` is an assumed constant | **absent** |
| counterfactual route-swap (HP-3) | no callsite passes two different `nav_cmd` values to the same window | `hierarchy.py` compares `nav` vs `follow` vs `zeroed` — the closest existing machinery, but on route *accuracy*, not on trajectory divergence | **absent; `hierarchy.py:277-283` is the hook to build it on** |

**Landed during this audit (working tree, uncommitted):** T3-11 — `rollout.collect` now persists
`pred_dense`/`gt_dense` `[N, 20, 2]` at 10 Hz (`rollout.py:108-152`), and `bench.collect_full` matches
(`bench.py:282,307`). This unblocks per-tick lateral error, which every route-compliance metric needs.
`git status --short` shows `M taniteval/taniteval/{rollout,bench,driving,closedloop}.py` — a sibling
agent's work in flight; note the interaction with HPP-2 rather than duplicating it.

### 3.3 The estimator problem

`hierarchy.py:154-181` (`_jack`) is `overlapping_holdout_se` — deprecated, 1.28–2.06× too narrow.
`driving.py:491` has an `assert_no_deprecated_estimator` guard; **the hierarchy block is not behind
it.** Every seam number in §5 below therefore carries an interval that is not decision-grade.

### ✅ PC3 — what must change

1. `corridor_departure_rate @ K=max` promoted into `taniteval/` as a first-class metric with
   `episode_cluster_bootstrap` intervals (Wave-2 T1-1 already owns this; HPP-2 depends on it).
2. A **junction / multi-option stratum** in `driving.py`, using the E1a definition
   (`|net heading change over the first 2 s| ≥ 10°`, `e1a_horizon.py:434`) as the *starting* point and
   the v2.1/v3 curvature-relative route label as the *better* one.
3. **New metrics that ADE cannot express**, all now computable from `pred_dense`:
   `route_compliance_rate` (did the executed path take the commanded branch),
   `counterfactual_route_divergence` (HP-3: ‖traj(nav=L) − traj(nav=R)‖ on identical observations),
   `route_reacquisition_rate` after a lateral perturbation (HP-6, reuses the E2a machinery),
   `time_to_maneuver_error` (the v4 `ttm` scalar already has labels at 20.5 % coverage).
4. Port `hierarchy.py::_jack` → `ci.episode_cluster_bootstrap` and put the block behind
   `assert_no_deprecated_estimator`.

---

## 4. PC4 — how many route DECISIONS are in the corpora?

### 4.1 The canonical val set (`physicalai-val-0c5f7dac3b11`, 40 eps, 881 windows) — this is where every headline number lives

`MEASURED` — `hierarchy_flagship-30k.json`, keys `seam_nav_to_strategic` and
`consistency.distributions`:

| quantity | value | fraction of all 881 windows |
|---|---:|---:|
| windows with a **judgeable route label** (`nav_valid`) | **240** | **27.2 %** |
| of those, route = straight | 161 (0.6708 × 240) | 18.3 % |
| of those, route = **left or right** | **79** (54 L + 25 R) | **9.0 %** |
| windows whose 2 s GT net heading ≥ 0.15 rad (`gt_dir` turn) | 191 (117 L + 74 R) | 21.7 % |
| windows where the model's own trajectory turns (`trajectory_dir`) | 197 | 22.4 % |

**The power reading, and it is worse than the percentages suggest.** `240 / 40 episodes = exactly 6`
— the six windows per episode that clear the 15 s-lookahead guard (§1.2), at `t_last ∈ {7,15,23,31,39,47}`,
i.e. **the first 0.7–4.7 s of each clip, all sharing essentially the same 25 s lookahead**. They are
near-duplicates of **one** route situation per episode. `MEASURED` arithmetic (240 = 6 × 40) +
`ESTIMATED` label homogeneity within an episode ⇒ the val set contains ≈ **40 route situations, of
which ≈ 13 involve a turn** (79 ÷ 6). PC4 asks for **n ≥ 40 episode-clusters *per stratum***. The
decision stratum today is **n ≈ 13**.

### 4.2 Train corpora — the v2 balanced corpus is a real improvement, and it is not enough

`MEASURED` — `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-25-v2-corpus-qa/v2_corpus_qa.json`
(`P2_distribution`; 9 000 clips / 49.74 h scanned, integrity 9 000/9 000 loadable):

| quantity | v1 parity `physicalai-train-e438721ae894` (2 376 eps, 13.13 h) | v2 balanced `physicalai-v2bal-4b7eeeac222d` (9 000 clips, 49.74 h) |
|---|---:|---:|
| clips containing a junction | **37.8 %** | **61.4 %** |
| clips containing a turn | 42.6 % | **76.5 %** |
| clips with net heading > 45° | 25.0 % | 40.3 % |
| clips with net heading > 90° | 10.4 % | 13.9 % |
| **step-weighted** turn_left + turn_right | **14.2 %** | **28.0 %** |
| step-weighted lane_keep | 59.6 % | 44.9 % |
| clips with **no turn label at all** | — | 2 119 / 9 000 (23.5 %) |

**A 2× improvement in turn density, achieved by kinematic selection.** But `IMP-8`'s own caveat holds
and is the PC4 blocker: *"kinematic selection cannot buy semantic scenarios."* A "junction" here is a
**heading-change signature**, not a topology with alternatives. Nothing in either corpus encodes *the
branch not taken*, which is precisely what a strategic level is for.

### 4.3 Label coverage under v3 (the v4 lineage) — the one place where the picture is genuinely better

`MEASURED` — `…/2026-07-22-v4-labels/labels_train_v4_provenance.json` (2 376 eps / 406 099 windows):

| label | coverage |
|---|---:|
| `route_valid` (v2.1/v3 curvature-relative) | **80.43 %** (val 79.35 %) |
| `route_token` (v3, 9-token vocab) | 80.70 % |
| `lat_target` / `lon_target` | 100 % |
| `dist_target` (distance-to-next-maneuver) | 62.92 % |
| `strat_scalars.ttm` (time-to-maneuver) | **20.48 %** |
| `lat_active_rate` (a lateral maneuver is happening) | 28.55 % |

**3× the route coverage of the v1 labeler, and the circularity is gone.** But `mintability` records the
hard ceiling: **4 of the 9 ROUTE tokens are never minted** — `straight`, `exit_left`, `exit_right`,
`merge` — because each *"asserts a junction exists = a MAP fact"*. Likewise `follow_lead` /
`close_gap` / `open_gap` are never emitted (`lead_state` is a `None` stub), and `TACPOINT` names stay
`unknown`. **Every route label the program can mint today is a description of what the ego did, never
an instruction about what it should do.**

### ✅ PC4 — what must change

1. Build `route_eval_v1` on the **v2.1/v3 adaptive-arc labels** (80 % coverage), not the v1
   fixed-25 s labeler (27 %). This alone multiplies the usable decision windows ~3×.
2. Stratify by **route token × distance-to-maneuver band** (`dist_target`, already minted at 62.9 %)
   so "approaching a decision" is separable from "executing a turn" and from cruise.
3. Draw the eval set from the **v2 balanced corpus** (junction 61.4 %, turns 28.0 % step-weighted),
   parity-safe because v2 selection re-derives labels on the same episodes and never re-selects
   (`corpus_key_match: true`, `skip_hash f09e44db` unchanged).
4. For HP-3 (counterfactual route swap), pair windows by **identical observation, different feasible
   command**. Note honestly: with kinematic-only labels this can only be built where a *branch* is
   observable. The two candidate sources are (a) `obstacle.offline` 3D tracks — real, on **96.90 %**
   of the corpus, currently **2 of 36 features ingested** — which give *other agents' divergent
   paths through the same junction* as a proxy for available branches; and (b) an external
   map-bearing benchmark (NAVSIM / Bench2Drive / nuPlan). **Without one of these, HP-3 and HP-4
   cannot be built on our data at all** — this is the single most important scoping fact in PC4.

---

## 5. Supporting evidence FOR the hierarchy — verified against artifacts

Each row re-checked against its raw artifact, with the caveat that makes it honest.

| # | Claim | Verified value | Artifact | Verdict on the evidence |
|---|---|---|---|---|
| **E1** | **H18 — grounding dominance, and it GREW** | grounded operative rollout **0.4271 m** vs ungrounded tactical head **3.3839 m**; Δ **+2.6979** (ci95 0.3418, n=881, `separated: true`). At 19 k the same pair was 0.615 vs 3.43 (Δ 2.82 on a worse operative). Ledger's "Δ2.70 m" ✅ reproduces exactly. | `hierarchy_flagship-30k.json` → `h18_grounded_vs_ungrounded` | ✅ **Strong and directly hierarchy-supporting**: the *grounded consequence readout* beats the *direct supervised head* by 7.9×. ⚠️ interval is `overlapping_holdout_se`; the effect is far too large to be an estimator artifact, but the number needs the bootstrap before it is quotable in the proof. |
| **E2** | **H26 — ctx→tactical is load-bearing** | maneuver-acc Δ(real − mean-ctx) **+0.0439** (ci95 0.0310, n=881, `separated: true`, ≥ MIN_ACC 0.02); goal-latent-cos Δ **+0.0084** (ci95 0.0037, separated); wp-ADE Δ +0.0336 (ci95 0.1904, **NOT** separated) | same file → `seam_ctx_to_tactical` | ✅ **Real, and it appeared with training** — 0/3 seams at 19 k → 1/3 at 30 k, on the *on-distribution* mean-ctx control (the strict test). ⚠️ 1 of 3 tactical metrics separates; deprecated estimator. |
| **E3** | **H19 — discrete tactical vocabulary → anchor prior** | REF-C-base **0.4728** [0.3835, 0.5699] at **104.2 M** ties flagship v1 **0.4271** [0.3675, 0.4871] at 263.4 M; the maneuver→anchor graft (`graft_maneuver true`) is live in REF-C and is v4's proposal mechanism | `LEADERBOARD.md` §1; `MODEL_REGISTRY.md:1023,1583` | ⚠️ **Weakest of the four.** The graft has **never been ablated** — `ARCHITECTURE_WIRING_COMPARISON.md:2.8`: REF-C seam measurements *"none — and none obtainable without new code."* "Discrete vocabulary helps" is currently `INHERITED` from an ADE tie, not from a controlled contrast. **HPP should either ablate it cheaply or stop citing it as hierarchy evidence.** |
| **E4** | **IMP-2 — planning over the WM beats supervised heads** | G1 open-loop **0.893 ± 0.114** vs tactical head **3.150**, Δ **+2.257 ± 0.329**, CI-separated, 880 windows / 40 eps; G4 closed-loop **1.038 ± 0.202** vs **1.685 ± 0.098**, divergence >5 m **8.7 %** vs 22.2 % | `MODEL_REGISTRY.md:1413-1417` (`planner_p2_flagship-30k.json`) | ✅ **The strongest structural argument for PC2** — it is *why* the hierarchy must be evaluated through planning. ⚠️ **Honest caveat, found here:** the P2 cost tracks a `v_target` minted as *"the 85th percentile of **future** speed over the next 10–20 s"* (`planner_p2.py:105-112`), i.e. a **future-derived** goal. That is the same input-asymmetry class as the 07-24 C6 retraction. The head baseline gets no such target. The *direction* is safe (planning > head readout); the **magnitude +2.257 is not a deployable margin** and should not be quoted as one in the proof. |
| **E5** | **Layers cohere** | maneuver↔trajectory agreement 0.872, **kappa 0.612** at 19 k; the 30 k panel reports the same block with `n_turn_active = 248` | ledger 07-18; `hierarchy_flagship-30k.json` → `consistency` | ✅ Supports "the levels are mutually consistent" (thesis part B), independent of whether they *drive* each other. |
| **E6** | **The strategic head is not merely idle — removing its input makes it worse than chance** | `route_acc_zeronav` **0.2167** vs `chance_1_of_3` 0.3333; Δ(nav − zeronav) **+0.7734** (ci95 0.0476, separated) | `hierarchy_flagship-30k.json` | ✅ A useful positive control: the strategic pathway **is** wired and does carry signal end-to-end. What it carries today is the command; the pipe works, the water is wrong. |

**Not evidence, and it should stop being cited as such:** `route_acc_nav = 1.0` and
`seam_nav_to_strategic.load_bearing = true`. The panel itself annotates the latter as *"load-bearing
**BY CONSTRUCTION**"* (`hierarchy.py:426-431`). Any summary that reports "nav→strategic is
load-bearing" without that qualifier is reporting the echo as a success.

---

## 6. Prioritized work list for HPP-1 … HPP-3

Ordered so that a killed agent still yields value; every item is file-level.

### HPP-1 — make the route input real (the PC1 gate: `route_skill > 0`, CI-separated; `nonav_route_beats_majority` PASS)

| P | Item | Files | Cost | Note |
|:--:|---|---|:--:|---|
| **1** | **Kill the label circularity.** Train the route head on `route_target_v21` / `route_from_future_v3` — never `route_target(nav_cmd)`. Coverage 27 % → **80.4 %**, and `unjudgeable ≠ straight`. | `stack/tanitad/train/flagship_losses.py:318-327`; `stack/scripts/refb_labels.py:705-745` (exists); trainer flag `--labels-v2` (exists) | S (config) | The single highest-leverage change. It is the reason `route` CE can reach 0.0 today. |
| **2** | **Enable LEVER A and carry it to a scored checkpoint.** `v2_route_from_vision` + `--route-vis-weight 0.3`. | `train_flagship4b.py:251,594`; `flagship_losses.py:329-342` | S (config) + train | Code exists and works; only two abandoned arms ever ran it. |
| **3** | **Port `hierarchy.py::_jack` → `ci.episode_cluster_bootstrap`** and put the block behind `assert_no_deprecated_estimator`. | `taniteval/taniteval/hierarchy.py:154-181`; `taniteval/taniteval/ci.py` | S | **Blocking**: PC1's exit criterion is "CI-separated", and that CI does not exist in an admissible estimator. Same as `V4_FLAGSHIP_DESIGN §15 P7(a)`. |
| **4** | **Remove `nav_cmd=None`** from the REF-C eval trio; feed the produced/commanded route. | `taniteval/taniteval/refc_eval.py:78`, `refc_rerank.py:262`, `plan_fan.py:549` | S | The exact confound of the 07-21 C6 retraction, still live in three files. |
| **5** | **Disclose or remove the eval-time route ORACLE** in the v15/v16/v4 evaluators — report produced-goal **and** oracle-goal, never oracle alone. | `stack/scripts/eval_flagship_v15.py:92-103`, `eval_flagship_v16.py:135-143`, `eval_flagship_v4.py:322` + `train_flagship_v4.py:167-173` | S/M | Violates `V4_FLAGSHIP_DESIGN §5.3 rule 3` today. Until fixed, v1.6's 0.4375 and every v4 MODE-B number are **goal-oracle numbers**. |
| **6** | Add a `route_skill` regression test: assert `route_acc_follow > majority + 0.03` on a synthetic corpus where route is inferable from the scene — so the echo cannot silently return. | `stack/tests/test_vision_levers.py` (exists, extend) | S | |

### HPP-2 — the decision-rich eval set + the metrics ADE cannot express (PC3 + PC4)

| P | Item | Files | Cost | Note |
|:--:|---|---|:--:|---|
| **1** | **`corridor_departure_rate @ K=max` into `taniteval/`** with `episode_cluster_bootstrap`. Lift from `e1a_horizon.py:305-320` — it is already written and validated. | new `taniteval/taniteval/corridor.py` + wire into `driving.py` | S/M | Converges with Wave-2 T1-1. Today it lives only in 5 `incoming/` one-offs. |
| **2** | **Junction / multi-option stratum** in `driving.py`, keyed on the v2.1/v3 route label + `dist_target` band, with the E1a `≥10°` definition as the fallback. | `taniteval/taniteval/driving.py:362-419` | M | Unlocks HP-2. |
| **3** | **`route_eval_v1`** — build from the **v2 balanced corpus** (junction 61.4 %, turns 28.0 %) with v3 labels; target ≥ 40 episode-clusters *per stratum*. | new, `taniteval/` + a label pass | M | Today's decision stratum on canonical val is **n ≈ 13**. |
| **4** | **Counterfactual route-swap harness (HP-3)** — same encoded window, `nav ∈ {left, follow, right}`, measure trajectory divergence + correctness. The hook already exists: `hierarchy.py:277-283` runs three strategic passes on one encode. | `taniteval/taniteval/hierarchy.py` → new `strategic_probes.py` | M | A flat marginal model cannot pass this by construction — **the cheapest discriminating experiment in the whole battery, and it needs zero new training.** Runnable the moment HPP-1 lands. |
| **5** | **Long-horizon rollout in the standing harness** — parametrise `fwd_k` past 20 in `rollout.collect` / `closedloop.collect`; the E1a evidence says K=185 is where strategic value lives. Honest bound: episode `T ≈ 199` ⇒ **K ≤ 190 is a hard ceiling** on this corpus. | `taniteval/taniteval/{rollout,closedloop}.py` | M | Coordinate with the sibling agent currently editing these files. |
| **6** | **Price `obstacle.offline` for branch extraction** (other agents' divergent paths through the same junction ⇒ observable alternatives). Currently 2 of 36 features ingested. | `stack/scripts/lake_ingest.py`; `DATA_STRATEGY_FOR_HIERARCHY.md §5` | M | **This or an external map-bearing benchmark is the only route to HP-3/HP-4 with real branches.** Decide explicitly; do not let it become the third 10-day orphan. |

### HPP-3 — pre-registration + the PC2 assertion

| P | Item | Files | Cost |
|:--:|---|---|:--:|
| **1** | **PC2 assertion in the harness**: forward-hook counters on `strategic_policy` / `tactical_policy` / `predictor.intent_proj`; any arm declaring a hierarchy whose scored pass leaves a counter at 0 **fails loud**. | `taniteval/taniteval/rollout.py::collect`, `runner.py` | S |
| **2** | **Retire the true-action rollout as a "driving" number.** Rename it `wm_fidelity_ade_2s` wherever quoted; HPP-4 scores only surfaces where actions are chosen. | `LEADERBOARD.md`, `MODEL_REGISTRY.md`, `driving.py` `SPEC` string | S |
| **3** | `HPP_PRE_REGISTRATION.md` — HP-1…HP-6, both outcomes committed in advance, estimator named (paired `episode_cluster_bootstrap`, B=2000), matched params ±5 %, matched steps, ≥4 seeds where a curve is claimed. | new, this bundle's sibling | S/M |
| **4** | Register the **negative** explicitly: if HP-1…HP-6 run under PC1–PC4 and show no separated advantage, that is a real result. Both outcomes, in advance. | same | — |

---

## 7. Where a doc claims something the code contradicts

| # | Doc claim | Code | Severity |
|---|---|---|---|
| **1** | `V4_FLAGSHIP_DESIGN.md:558-560` — *"No leaderboard number may come from a GT-derived plan or a GT-derived goal."* | `eval_flagship_v16.py:135-143` and `eval_flagship_v15.py:92-103` feed `route`, `route_graded`, `vt_band`, `vt_speed` **straight from the GT label file**; `eval_flagship_v4.py:322` + `train_flagship_v4.py:167-173` feed route tokens minted per-window from the episode's **future** poses. No produced-goal path and no goal-dropped control exists in any of the three evaluators. | 🟥 **High.** v1.6's 0.4375 and every v4 MODE-B number are oracle-goal numbers, and neither the registry nor the leaderboard says so. |
| **2** | `V4_FLAGSHIP_DESIGN.md:806-807` — *"**Precondition (code):** the v4 evaluator must feed the produced goal. `refc_eval.py` / `plan_fan.py`'s `nav_cmd=None` constant must not be inherited."* | `refc_eval.py:78`, `refc_rerank.py:262`, `plan_fan.py:549` all still pass `nav_cmd=None`. Listed as `§15 P7(c)`; not landed. | 🟠 Medium-high — this is the 07-21 C6 confound, still shipping. |
| **3** | `V4_FLAGSHIP_DESIGN.md:735-737` — v4 is *"the first arm in the program with a **working route input**… evaluated with the produced goal rather than a constant."* | v4 has **no route classifier at all**: `strategic_goal.GoalScalarHead` emits 4 continuous scalars; the strategic planner ① is unbuilt (P6). `eval_flagship_v4.py:640-650` says so in the gate output. And what v4 *is* evaluated with is the **GT** route token, not a produced one. | 🟠 Medium — the design's own claim is ahead of its implementation; the gate metric that would catch it (`nonav_route_beats_majority`) is emittable and reads **0/FAIL**. |
| **4** | `hierarchy.py` verdict string prints *"nav→strat **LOAD-BEARING**"* whenever `delta_nav_vs_follow` separates (`hierarchy.py:650-654,425`). | It separates **by construction** — the target is a lookup of the input. The `_note` field says exactly this, but the verdict line, the `load_bearing` boolean and every downstream summary do not. | 🟠 Medium — this is the surface through which "the hierarchy works" has been repeated. Same shape as the 07-25 C4 header-propagation retraction: **the correction lives in the body, the headline still says the wrong thing.** |
| **5** | `driving.py:551-555` — *"The dense 20-step 10 Hz path **IS** persisted since 2026-07-25"*; `01_EXECUTION_PLAN.md:126` (T3-11) — *"Land the 1-line dense-path persistence (`taniteval/rollout.py:94`) — unmerged since 07-09."* | Both were true at different moments **today**: `rollout.py:108-152` now emits `pred_dense`/`gt_dense`, as uncommitted working-tree changes (`git status`: `M taniteval/taniteval/{rollout,bench,driving,closedloop}.py`). | 🟢 Low — resolved in flight by a sibling agent; recorded so HPP-2 does not re-implement it. |
| **6** | `hierarchy.py:571-572` labels its own CI *"DEPRECATED, not a jackknife"*, yet `driving.py:491`'s `assert_no_deprecated_estimator` guard does not cover the hierarchy block. | Every seam number in §5 carries a `1.28–2.06×`-too-narrow interval. | 🟠 Medium — blocks PC1's "CI-separated" exit criterion. |

---

## 8. Deliverable manifest

| Artifact | Where it lives |
|---|---|
| **This audit** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-25-hpp0-confound-audit/HPP0_CONFOUND_AUDIT.md` (repo working tree, **not** staged — orchestrator stages) |
| Primary code evidence read (unmodified) | `stack/tanitad/models/metric_dynamics.py`, `stack/tanitad/models/fourbrain.py`, `stack/tanitad/models/flagship_v15.py`, `stack/tanitad/models/strategic_goal.py`, `stack/tanitad/refs/refc.py`, `stack/tanitad/train/flagship_losses.py`, `stack/scripts/{refb_labels,train_flagship4b,train_flagship_v4,v4_labels,flagship_v4_data,gate_emitters,eval_flagship_v4,eval_flagship_v15,eval_flagship_v16,eval_grounded_rollout_4b}.py`, `taniteval/taniteval/{rollout,bench,driving,closedloop,hierarchy,planning,refc_eval,refc_rerank,plan_fan,planner_p2}.py` |
| Primary raw artifacts read | `…/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-{30k,v4.2b-dryrun}.json`; `…/2026-07-23-v4-gate-emitters/artifacts/hierarchy_flagship-30k_v1.json`; `…/2026-07-25-v2-corpus-qa/v2_corpus_qa.json`; `…/2026-07-22-v4-labels/labels_{train,val}_v4_provenance.json`; `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44_K185.json`; `taniteval/results/trainlogs/{v1-speedjerk,v2,v3enc,nospeed-phase0,expA-nodrop}_{train_log.jsonl,config.json}` |
| Nothing staged, nothing committed, nothing pushed | ✅ per brief |
| No pod touched, no GPU used, no training or eval run | ✅ per brief |

**Escalations (do not let these live only in this file):**

1. **PC1 fix #1 and #2 are config-only and already in `train_flagship4b.py`.** The next flagship
   launch can carry them at zero engineering cost. This needs a *launch decision*, not a work package.
2. **The eval-time route oracle (§7 #1) touches three shipped headline numbers** (v1.5, v1.6, v4
   MODE B). It should be surfaced to the registry owner today, because the fix is a disclosure, not a
   re-run.
3. **HP-3 is runnable with zero training the moment HPP-1 lands** — `hierarchy.py:277-283` already
   runs three strategic passes on one encode. It is the cheapest discriminating experiment in the
   battery and should be scheduled ahead of HPP-4.
4. **HP-3/HP-4 with *real* branches need either `obstacle.offline` ingest or an external map-bearing
   benchmark.** Neither exists today. This is a PI-level scoping decision, not an agent-autonomous one.
