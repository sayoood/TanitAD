# DESIGN — refcv8 WP-B: the tactical layer CONDITIONS fan generation and selection; the route-checkpoint input; X1–X4; warm start

*WP-B (Architecture & Inference), 2026-10-04. Requirement: PLAN_REFCV8 §0 R8-3, R8-4, §0.2 X1–X4. Status: stage 1 of 4
(DESIGN) — for the Master Mind's go. Nothing here is trained; nothing here is a lever result.*

**Stamps.**
* **Code read:** tip `50efa52` extracted to `C:/Users/Admin/r8_wpb` (`git archive`); every `file:line` below is the
  TIP. The tip differs from the launch tree `fec3a0d` in `refc.py` / `refc_v3.py` / `refc_v3_train.py` ONLY by the R4
  restart commit `0fcb1c1` (ceiling reaches the emitted plan, `e9_rank`, the ten-module bypass freeze) and `b60cba6`
  (opt-in inference fixes F1/F2/F4); `refcv6_tactical.py`, `refcv6_selection.py`, `refc_select.py`, `refc_tactical.py`,
  `refcv6_max_speed.py`, `v7_labels.py` are blob-identical (40-char blobs both sides).
* **Run record:** `D:/refcv7_eval_kit/ckpt/config.json` (refcv7-r101-s0 argv; `param_breakdown` total **100,468,987**).
* **Evidence classes:** `MEASURED (D0)` = this package (`raw/d0_dose_response.json`, `raw/d0b_prior_follow.json`,
  `raw/d0c_heading_posthoc.json`; banked fans md5-checked); `MEASURED (route)` = `…/2026-10-04-refcv7-route-following/RESULT.md`
  / `RESULT_A5.md`; `MEASURED (D1/D2/D4)` = the refcv8 data audit; `PUBLISHED` = a banked primary, key + page;
  `HYPOTHESIS` = a design claim awaiting its rung. **Tier of every planner number: OPEN-LOOP**, logged frames of the
  held-out eval139, one TRAINING seed; the route package's sampler-seed-1 replicate is carried where it exists.

---

## 0. The design in ten lines (each with the evidence that chose it)

1. **The lever is the CONSTRAINT, not the class.** Re-selecting refcv7's own fan with a simulated tactical
   posterior: a perfect lateral CLASS buys only −0.103 m all-window (= B1), and only clears the route bar at
   **≥ 0.95 accuracy**; a 6-s PROGRESS constraint of ≤ 10 % lognormal error buys **−0.484 m** (σ 0.10), a terminal
   HEADING constraint of ≤ 15° error lifts heading-within-15° from 0.514 to **0.70** (MEASURED: D0 registered;
   the heading row is D0c, POST-HOC, §2.4). ⇒ the
   tactical layer must emit **actions WITH constraints** (distance, time, target speed, Δyaw), and the planner must
   consume the constraints.
2. **Per-CANDIDATE hypothesis tags, not one window vector.** Every candidate carries ONE tactical hypothesis
   h = (lat, lon) and that hypothesis' predicted constraint c_h; the generator is conditioned on (h, c_h) through a
   zero-init per-candidate modulation. The tags are one-hot at train AND test, so there is no soft-vector train/test
   mismatch, and the fan stays multimodal across tactical modes (GoalFlow's no-goal arm "averages all generated
   trajectories", 85.6 vs 88.5–90.3 PDMS with goals — PUBLISHED 2503.05689 Tab. 2 p7; not a clean goal on/off
   ablation, since M0 also averages — which is the point here). R1's H5 (one 16-d action
   vector added to `cond`) is the window-level foil, already registered (SPEC_R1).
3. **"The most probable actions influence planning" twice:** (a) ALLOCATION — M = 32 extra candidates are drawn for the
   top-k = 4 hypotheses ∝ p(h) (min 2 each), each started from an anchor of that mode; (b) SELECTION — a factorised
   score s_k = s_conf,k + β·log p(h_k) + Σ γ_j·sat_j(k) + … with zero-init β, γ.
4. **Selection is TRAINED on the emitted fan (X1)**: a listwise soft-target CE (cost = speed-profile + heading, R1-H4's
   form, the Hydra/TNT soft imitation target — PUBLISHED 2406.06978 Eq 8–10 p3; 2008.08294 Eq 5–6) **plus per-candidate
   BCE sub-score heads** (direction-correct, progress-within-10 %, heading-within-15°, constraint-satisfied), combined
   Hydra-style (Eq 11). Hydra measured per-sub-score heads > one distilled overall score (83.0 vs 80.2 PDMS, p4).
5. **Route checkpoint (R8-3)** enters at three places — tactical condition, operative condition, and a param-free
   LATERAL-ONLY bearing term in selection — each behind a zero-init gate, with checkpoint dropout 0.3 and four echo
   controls. Its known risk is the target-point shortcut ("Steering directly towards a TP is a shortcut", PUBLISHED
   2306.07957 p1), whose published remedies are attention (not pooled) conditioning and shift/rotation augmentation.
6. **X2** residual-prior LATERAL dropout (p 0.3) and a prior-free sampler group, both as arms with a regression arm —
   the literature gives no isolated gain for past dropout and a MuJoCo counter-example (§3.5).
7. **X3** the past-only speed input (N2/N3, PI §8-2 pending) through the EXISTING 4-way one-hot slots (shape-preserving),
   the "unknown" row trained at p 0.45, the ceiling on the emitted plan (already on the tip, `refc_v3.py:1021-1041`).
8. **X4** the tactical decoder's OWN parameters are fed only by the tactical loss (planner feeds are detached), so under
   Adam their update is (to first order) independent of the 0.1 budget (Adam: "magnitudes of parameter updates are
   invariant to rescaling of the gradient", PUBLISHED 1412.6980 p1). The 0.29 % starves the SHARED scene producers, not
   the head ⇒ the budget is set from the measured gradient share on the last shared layer (GradNorm's object, PUBLISHED
   1711.02257 p2), and the module-state defect is removed from `v7_labels.py:430-436`.
9. **Warm start**: every new path is zero-init / gated / flag-excluded, new modules are separate keys (no existing
   tensor changes shape), every refcv8 random draw uses a DEDICATED generator, so step 0 reproduces refcv7-50,400's
   emitted plan bit-for-bit (adaLN-Zero "initializes the full DiT block as the identity function", PUBLISHED 2212.09748
   §3.2 p5; ControlNet zero convolution, 2302.05543 §3.1 p4).
10. **Bars proposed to SPEC_REFCV8 from D0** (§2.4): lateral 3-way tactical accuracy ≥ 0.95 if lateral conditioning of
    selection is to clear on its own; 6-s progress constraint lognormal σ ≤ 0.10 (median relative error ≤ 6.7 %);
    terminal-heading constraint ≤ 10–15° (from the POST-HOC D0c — proposed for registration, not a result); and — because refcv7's own progress predictor HURTS when conditioned on
    (+0.213 m, §2.3) — the real head's conditioned arm, not the σ, is the deciding bar.

---

## 1. As built: how refcv7 makes and picks its 117 candidates (tip `50efa52`)

### 1.1 The bank (generation's starting point)
* **Vocabulary:** 117 anchors, each ONE constant control pair (a_lon m/s², a_lat m/s²; `control_units: alat`,
  `horizon_s 6.0`, `dt 0.1`, `kappa_cap 0.12`, `alat_v_floor 4.0`, declared in the artifact; config.json `anchors`).
  `anchors` buffer `refc.py:1737`, `anchor_controls` `:1753`.
* **Residual on a causal prior P** (`--residual-prior ha0_ext_pose`): a0 = (v[t0] − v[t0−1])/dt, κ0 = ω0 / max(v0, 2)
  clamped ±0.3 (`kinematic_prior.py:53-59, 340-343`); the bank is `roll_plan(anchor seq, a0, κ0, v)` — Δ = 0 rolls to P
  exactly (`kinematic_prior.py:433-463`; `refc.py:2328-2345`). Rows whose ego was withheld (ego dropout 0.5) get the zero
  prior at the reference speed (`kinematic_prior.withhold`, `:346-363`; `refc.py:2285-2326`).
  ⇒ **every candidate is "the current yaw/accel continued + a residual"** — the mechanism behind §1.6's prior bias.

### 1.2 The condition — ONE vector per window, shared by all 117 candidates
`cond = cond_proj(m) [+ ego_to_cond(ego_hist)] [+ tgt_film(cond, tac_latent_proj(z_tac.detach()))]`
(`refc.py:3165-3177`), with `m = measurement([v0/10, nav one-hot(4)])` (`refc.py:4557-4616`) and z_tac = PhiTac(pooled
sequence) + the E13 nav embedding (`refc_v3.py:1575-1621`; target latent `:1800-1801`). Each decoder layer applies it as
`q + mlp(film(norm_f(q), cond.unsqueeze(1)))` (`refc.py:1613`) — **broadcast over candidates**; F4 AdaLN modulates by
the TIME embedding only (`refc.py:2625-2626`). **No candidate-specific condition exists anywhere in the decoder.**

### 1.3 The sampler (DiffusionDrive-style truncated diffusion, control space)
x0 = anchor control sequence / control_norm (4, 3); one ε at t = 8 (eval) or t ~ U[0, 50) (F1, train, one pass);
ladder 2 steps with DD step semantics (F2); each pass `_decode_ctrl`: `traj_proj(x_path) + time_mlp(t)` → 4 cross-attn
layers (image fmap; agent tokens via a zero-init gate; BEV sampled at the candidate's own waypoints) → AdaLN → per-layer
cascade heads (conf, du); x0̂ = x_n + du; fan = roll(denorm(x0̂)) on P (`refc.py:2577-2644, 2693-2915`, call `:3392-3403`).
MEASURED (route): the fan holds a direction-correct candidate on **100 %** of GT-turn windows (oracle-117 ADE 1.19 m).

### 1.4 The ranked score
* Classifier pass conf (`:3257-3260`) — focal CE vs `a_star` = the BANK anchor nearest GT (`refc_v3_train.py:4605-4618`).
* `--f5-emitting-conf`: the ranked base is the SAMPLER's last-pass conf + the same priors (`refc.py:3402-3403, 3463`).
* Priors on that surface: `tac8_lat_to_anchor` / `tac8_lon_to_anchor` — zero-init `Linear(8 → 117)` over the tactical
  decoder's DETACHED lat/lon log-probs (`refc.py:1834-1844, 3282-3294`; `refcv6_tactical.py:661-692`).
* Rank terms: the valid-behaviour gate (zero-init 22 → 117; the 5 situation columns structurally dead,
  `refc_v3.py:1366-1373`, `refc.py:3544-3549`) and `navc_gate` × the nav-compliance predicate (`refc.py:3559-3565`).
* Filters: reach band (`:3580-3593`), speed ceiling (`:3606-3613`, exported as `ceil_keep` `:3698-3704`).

### 1.5 E9 and the emitted plan
`blended = seam_clamp(sel_score, goal_gate × GoalDistanceScorer(fan @ 2 s, z_tac, ĝ = g_tac 2 s detached))`, argmax over
reach ∧ ceiling (`refc_v3.py:2355-2405`, `e9_rank` `:1021-1041`). Selection is trained by ONE single-winner CE: target =
argmin of the emitted fan's ADE over the reach survivors, `SEL_V3_WEIGHT` 1.0 (`refc_v3.py:2821-2833`,
`refc_v3_train.py:4806-4814`). The sampler's last-pass conf — the F5 ranked base — gets gradient only through this CE
(the cascade loss excludes the last stage, `refc_v3_train.py:4996-5029`).

### 1.6 Where the tactical decoder's outputs enter TODAY
| output of `TacticalBehaviourDecoder` (`refcv6_tactical.py:412-582`, cond = [nav(4), vmax one-hot(4), v0, a0] `:611-654`) | enters | form |
|---|---|---|
| lat / lon 8-way posterior | the ranked score | detached log-probs → zero-init 8 → 117 Linear (an anchor-ID prior) |
| 22-token validity (17 admissible) | the ranked score | detached probs → zero-init 22 → 117 gate |
| max-speed one-hot | the argmax | ceiling filter (+ E9 on the tip) |
| **anything** | **fan GENERATION** | **nothing.** The only tactical signal in generation is PhiTac's z_tac (a pooled-image head, not the scene decoder) through the target-latent FiLM |

**MEASURED state of that chain (D0 §4 / route):** the tactical decoder's lat posterior, collapsed to 3-way, is
**0.7375** accurate on the 800 classified EVAL windows vs the majority-class control **0.7388** (no skill; macro-F1
0.586; side-correct on GT turns **0.402**); lon 5-way 0.490 vs majority 0.457 (macro-F1 0.376). The pick agrees with
the tactical argmax on **0.449** of GT-turn windows while agreeing with GT on **0.841** — the planner is NOT consistent
with its tactical layer, and is right more often than it. Pick correctness on GT turns is **0.967** (59/61) when the
residual prior's side is right and **0.674** (31/46) when it is wrong (seed 1: 0.967 / 0.696); 14 of the 17 wrong turn
picks take the prior's side (D0b, reproducing route §1.5).

---

## 2. Evidence that sizes the design — D0 (registered `SPEC_D0_DOSE_RESPONSE.md`, sha256 `7c7e4f52…`, 11:11:58Z)

**What it is:** a label-side simulation on refcv7's banked fans (eval 1,112 windows / 139 episodes; λ, β fitted on the
TRAIN capture only; 5 noise draws; paired episode-cluster bootstrap B 2000; sampler-seed-1 replicate). Not a lever.
**Controls (all PASS):** E9 argmax reproduces `sel_idx` 0/1,112 mismatches on all three captures; the instrument
reproduces A4's bounds B1 −0.1041 / −0.5658 and B3 −0.6061 / −0.8917 (registered −0.104 / −0.566, −0.606 / −0.892);
C-U (uninformative posterior) reads **exactly 0.000**; C-SH (deranged progress constraint) **+0.013 [−0.022, +0.049]**.

### 2.1 Lateral class only (S-LAT, ΔADE vs the shipped pick, seed 0; seed 1 in brackets where it matters)
| q (3-way accuracy) | all | GT-turn | GT-straight | turn dir-correct | head ≤ 15° (turn) |
|---|---|---|---|---|---|
| 0.74 = refcv7's head (real, H-LAT) | +0.002 [−0.009, +0.012] | −0.001 | +0.002 | 0.841 | 0.514 |
| 0.80 | −0.036 [−0.065, −0.012] | −0.091 [−0.222, +0.017] | −0.019 | 0.884 | 0.518 |
| 0.90 | −0.041 | −0.102 [−0.249, +0.020] | −0.023 | 0.894 | 0.520 |
| **0.95 (R1: q\*)** | −0.056 [−0.095, −0.020] | **−0.149 [−0.308, −0.025]** (s1 −0.127 [−0.275, −0.010]) | −0.023 | **0.908** | 0.520 |
| 1.00 (= B1) | −0.103 | −0.566 | −0.004 | 1.000 | 0.561 |

### 2.2 Longitudinal constraint (S-LON: 6-s progress P̂ = P_GT·e^{σε}) and the trivial predictor
| σ (median rel. err.) | all | GT-turn | GT-straight | turn dir | head15 |
|---|---|---|---|---|---|
| 0 | −0.773 [−0.904, −0.644] | −1.189 | −0.651 | 0.888 | 0.570 |
| **0.10 (6.7 %) — R2: σ\*** | **−0.484 [−0.615, −0.370]** (s1 −0.438) | −0.696 | −0.417 | 0.862 | 0.529 |
| 0.20 (13.4 %) | −0.290 [−0.392, −0.204] | −0.383 | −0.247 | 0.864 | 0.516 |
| 0.50 (32.7 %) | −0.118 [−0.203, −0.051] | −0.088 | −0.103 | 0.845 | 0.516 |
| C-CV: P̂ = 6 s × v0 (non-oracle) | −0.008 [−0.059, +0.041] | −0.011 | +0.001 | 0.832 | 0.505 |

R4: constant-velocity progress captures **1.1 %** of the oracle constraint's gain ⇒ the speed constraint is NOT
reachable from v0 alone; σ\* is an absolute bar.

### 2.3 The correlated-error penalty (the reason a σ is necessary, not sufficient)
refcv7's OWN 6-s progress predictor — the E8 goal head's 6-s point, median relative chord error **0.176** (≈ σ 0.26,
whose simulated row is ≈ −0.24 m) — **HURTS when selection is conditioned on it: +0.213 m [+0.121, +0.313]** all-window
(seed 1 +0.239). Its errors are not the independent noise the simulation assumes: they are aligned with the pick's own
bias (both read the same trunk). The penalty (~0.45 m) exceeds the whole simulated gain at that error. ⇒ **a
constraint head is judged by its own conditioned-selection arm on held-out windows**, never by its σ alone.

### 2.4 Heading constraint (D0c, POST-HOC — written after D0's numbers; a design input, not a bar)
Terminal heading θ̂ = θ_GT + σψ·ε, TRAIN-fitted λ: σψ 10° → turn dir 0.961, **head15 0.787** (s1 0.753), turn ΔADE
−0.498; σψ 15° → 0.951 / **0.701** (s1 0.671); σψ 20° → 0.938 / 0.654. With the σ 0.10 progress term as well:
all-window −0.586, turn −1.183, head15 0.761 at σψ 10°. ⇒ **R8-4 (iv)'s route bars (dir ≥ 0.95, head15 ≥ 0.70) are
reachable by re-selection on refcv7's own fan only with a ≤ 10–15° terminal-heading constraint** — lateral CLASS
conditioning (q 0.95 → head15 0.520) cannot reach them.

### 2.5 What the design takes from D0
* The tactical layer's job is not only "which action" but "**which action, starting when, ending at what heading, at
  what speed / stopping where**" — exactly the PI's "actions and their constraints like distance and time".
* The constraint must come from information the pick does not already have: the scene decoder (agents + BEV), the
  announced nav with its distance/time, the route checkpoint — and the arm that consumes it must be measured, not
  inferred (§2.3).

---

## 3. The refcv8 design

### 3.0 Information flow (inference) and admissibility

```
frames ──► trunk ──► {image tokens, agent slots, BEV tokens}
                          │ (scene only)
nav(announced, +d,t) ─┐   ▼
route checkpoint RC ──┼─► TACTICAL DECODER (38 queries + constraint heads; FiLM cond [nav, nav args, RC, vmax, v0, a0])
vmax (N2/N3|map) ─────┤        │  outputs: p(lat), p(lon), constraints c_h, 22 goal validities
v0, a0, past ego ─────┘        │  ── DETACHED planner feeds: p(lat), p(lon), c_h, 17 admissible goals ──┐
                               │  ── TARGETS ONLY: TL tokens, YIELD, tac_SIT (never a feed) ⛔          │
                               ▼                                                                         ▼
operative cond (per window): [v0, nav] + ego hist + z_tac FiLM + RC (zero-init)        per-candidate tags φ_k = (h_k, c_{h_k}, log p(h_k))
                               └──────────────────────────► SAMPLER (117 base + 32 allocated candidates) ◄─┘
                                                                     ▼
                     SELECTION: s_conf,k + β log p(h_k) + Σγ_j sat_j(k, c) + ρ·RC-bearing_k + tac8 + behaviour + navc(announced) [+E9]
                                                                     ▼  reach ∧ ceiling ──► emitted plan
```

* **Admissible at inference:** vision; v0 and past ego (PI 2026-09-02); the supplied nav (announced, per frame, with
  distance/time), the route checkpoint and the speed limit (PI, R8-3 / R1). On PhysicalAI the nav and the RC are
  ego-future-derived ⇒ optimistic by construction, stamped on every result.
* **The tactical posteriors conditioning the planner ARE the hierarchy the PI asked for** (R8-4). What enters: lat/lon
  posteriors, the predicted constraints, the 17 admissible goal validities — all DETACHED (the planning loss must not
  reshape the tactical head into "whatever correlates with the trajectory", `refcv6_tactical.py:674-683`).
* ⛔ **No situation-classifier output enters any goal input or the RC path.** TL tokens / YIELD / tac_SIT stay targets
  only; they are not in φ_k, not in the allocation, not in selection (the existing structural-dead-column mechanism +
  mutation proof `refcv6_selection.assert_situation_columns_dead` is extended to the new feeds). **The RC is computed
  from the supplied route only and NOTHING flows into it** — pinned by an interventional test (perturb every tactical
  output, assert the RC feature tensor and every RC-derived term are bit-identical). ⚠ Declared, as before: the
  tactical decoder is ONE DETR whose query self-attention lets the lat/lon queries read the TL query's state; that is a
  shared ancestor (as the shared trunk is), not a feed — the same declaration refcv6/7 carried.

### 3.1 (a) Conditioning fan GENERATION on the tactical hypotheses

**Hypothesis set H.** h = (lat, lon) with lat ∈ {LANE_KEEP, TURN_L, TURN_R} (+ LANE_CHANGE_L/R only if WP-A's SAM3 rule
passes its referee) and lon ∈ the v9 longitudinal classes (D4: CRUISE, ACCELERATE, BRAKE_TO, ADAPT-braking, CREEP, HOLD,
FOLLOW; STOP as v9 defines it). Mapped onto the EXISTING 8-wide lat/lon queries with the dead classes masked
(`effective_mask`) — no head changes shape. p(h) = p(lat)·p(lon) (factorised; a joint table is a later option).

**Constraints c_h** — NEW per-query regression heads on the tactical decoder's action queries (the query IS the class,
`refcv6_tactical.py:448-455`, so "the constraint if this action" is that query's readout):
lat query → (t_start, Δψ_end, κ_max) ; lon query → (v_target, t_reach) | (d_stop, t_stop) for STOP/BRAKE-to-stop |
(d_lead, time gap) for FOLLOW. Normalised (t/8 s, d/60 m, v/30 m/s, Δψ/π, κ·20) with a validity bit. Supervised ONLY on
the GT-active class of the window from the v9 constraint fields (masked smooth-L1; log for distances/times). Teacher
forcing for "the constraint of a non-GT class" is impossible by construction and not attempted.

**Tags.** Every candidate k carries one hypothesis h_k:
* **base fan (117):** h_k := `label_path(bank_k)` — the anchor's OWN mode at this window's v0, by the SAME geometric
  rule as the v9 labels applied to the candidate's [0, 6] s path (`refcv8_conditioning.label_path`, pure, analytic
  tests). So a TURN_L anchor is told "you are TURN_L; the tactical layer predicts TURN_L starts at 3.1 s and ends at
  +85°"; a braking anchor is told "STOP at 23 m".
* **allocated (M = 32, §3.1.2):** h_k := the hypothesis being allocated.
* **forced (eval interventions, R8-4 ii):** h_k := the forced hypothesis, c := the forced constraint.

**Per-candidate feature** φ_k = [onehot lat(3) | onehot lon(8, masked) | log p(lat_k), log p(lon_k) | c_lat(h_k) (3 + 1
valid) | c_lon(h_k) (2 + 1 valid)] ≈ 21 dims, built from DETACHED tactical outputs.

**Where φ_k enters — a zero-init per-candidate modulation after every decoder layer** (both the classifier pass and
every sampler pass): `q ← q ⊙ (1 + γ_i(φ_k)) + β_i(φ_k)`, `(γ_i, β_i) = Linear_i(φ_k)` ZERO-INIT (4 layers × (21·768 +
768) ≈ 68 k params). It is applied AFTER the layer (the F4 AdaLN position, `refc.py:2625-2626`), NOT by widening `cond`
to [B, N, d]: `x·(1+0)+0` is exact in IEEE arithmetic, whereas a [B, N, d] `cond` would route the existing FiLM Linear
through a different GEMM shape and is not guaranteed bit-identical. This is DiT's adaLN-Zero placement argument
(PUBLISHED 2212.09748 §3.2 p5) applied to a per-candidate condition.

**Why per-candidate and not one vector (R1-H5's form).** One conditioning vector per window conditions ALL candidates on
the same (soft, at inference) action mixture: with p(TURN_L) = p(LANE_KEEP) = 0.5 every candidate is denoised toward
the average hypothesis — the mode-averaging the anchored fan exists to avoid (DD Tab. 2: TransfuserDP mode diversity
11 % vs DD 74 %, PUBLISHED 2411.15139 p6; GoalFlow's no-goal arm 85.6 PDMS, 2503.05689 Tab. 2). And training feeds a
GT ONE-HOT while inference feeds a SOFT posterior — a conditioning distribution the generator never saw. Per-candidate
one-hot tags have neither problem; only the constraint VALUES change between train (GT, scheduled) and test
(predicted). CIL's branched vs command-input contrast points the same way (88 % / 64 % vs 78 % / 52 % success,
PUBLISHED 1710.02410 Tab. 1 p6 — single run, no CI, 2017 CARLA: suggestive only). R1-H5 vs WP-B-T2 is the measured
comparison (§5).

#### 3.1.1 Training signal for the conditioning
* **Matched L1 (exists):** the candidate nearest GT (`a_star`) is supervised to GT; its tag is GT-mode almost always,
  and its constraint input is the **GT constraint with probability 1 − ε(step)**, the predicted one otherwise —
  scheduled sampling with a LINEAR decay of the teacher-forcing ratio 1.0 → 0.25 over the run, drawn PER WINDOW (the
  primary's per-sequence coin was "much worse" than per-token, PUBLISHED 1506.03099 fn 2 p3 — our unit of decision is the
  window, and the arm records which), with "Always Sampling" collapse as the known failure (BLEU-4 11.2 vs 30.6, Tab. 1
  p6). TNT feeds the GT target in training the same way (PUBLISHED 2008.08294 p5).
* **Constraint-satisfaction loss L_sat (NEW, every tagged candidate, param-free targets):** differentiable hinges on the
  candidate's own path: TURN_x — heading change by 6 s ≥ min(Δψ̂, 30°) on the tagged side once t_start ≤ 5 s, ≤ 10°
  before t_start; LANE_KEEP — |Δψ| < 30° or κ_max < 1/40 m; STOP-at-d — |P_6 − d̂| ≤ max(2 m, 0.1 d̂) and terminal speed
  ≤ 0.5 m/s; v_target — |v(t̂_reach) − v̂| ≤ 1 m/s. Weight w_sat (start 0.1 of the trajectory weight, budgeted under X4).
  This is what gives NON-GT candidates a training signal, i.e. what makes "force TURN_L ⇒ the fan turns left" true
  rather than hoped. ⚠ HYPOTHESIS that it is needed: the regression arm (L_sat off) is pre-registered (§5).
* **Condition dropout** p_uncond 0.15 on φ_k (CFG: 0.1 and 0.2 performed alike, 0.5 worse — PUBLISHED 2207.12598 §4.2
  p8, image domain only), so the generator also runs untagged (windows with no usable tactical output, NavSim fallback).
  Guidance at inference is NOT adopted (it doubles the sampler cost); it is an optional arm.

#### 3.1.2 Allocation ∝ probability
* Per window: rank joint hypotheses by p(h); take the top-k = 4; n_h = max(2, round(M · p_h / Σ_topk p)), adjusted to
  sum to M = 32. Each allocated candidate starts from an anchor whose tag == h (nearest in control space to the
  constraint-implied prototype; repeats allowed with fresh noise — the F7 group-major infrastructure `refc.py:3193-3202`
  and `candidate_to_anchor_id`), conditioned on (h, c_h). A hypothesis with no anchor of its mode at this v0 (e.g. TURN
  at 30 m/s) is skipped and counted.
* TRAIN: the GT hypothesis is forced into the top-k with ≥ 4 candidates (teacher forcing); the rest ∝ predicted p.
* Cost: 117 → 149 candidates on the sampler passes (+27 % decoder compute, ESTIMATED from candidate count).
* **Warm-start rule:** allocated candidates are TRAINED from step 0 (matched L1 over the allocated GT-hypothesis subset +
  L_sat + the selection CE sees them) but are EXCLUDED from the EMITTED argmax until step S_emit (a flag; the identity
  test runs with it off). ⚠ The value of allocation is unmeasured (the 117-fan already holds a correct-direction
  candidate on 100 % of turns); its claim is within-mode DENSITY — finer speed profiles and turn onsets inside the
  probable mode, the B3/B4 headroom (B4 heading ∧ speed −0.674 all, route A4). Its regression arm allocates to RANDOM
  hypotheses.

### 3.2 (b) Conditioning SELECTION, and the selector trained on the emitted fan (X1)

**Factorised score** (all new terms zero-init ⇒ step 0 = refcv7):
`s_k = s_conf,k + tac8 + β_lat·log p(lat_k) + β_lon·log p(lon_k) + Σ_j γ_j·sat_j(k) + ρ·rc_bearing_k + behaviour + η·navc_ann,k`, E9 blended after.
* `log p(h_k)` — the hierarchy made literal: a candidate is weighted by its OWN hypothesis' posterior (the tac8 8 → 117
  Linear stays as the anchor-ID prior it is; it could only ever learn per-anchor-ID biases, not per-candidate mode
  agreement, and it learned little: the tactical head carries no skill, §1.6).
* `sat_j(k)` — param-free constraint scores computed from the candidate path and the PREDICTED constraint of its
  hypothesis: progress −|log((P_6,k + 1)/(P̂_6(h_k) + 1))| (P̂ from v̂_target, t̂_reach or d̂_stop by a constant-
  acceleration profile); terminal heading −|wrap(θ_k − Δψ̂(h_k))|/π when t̂_start ≤ 5 s; turn onset. These are exactly
  D0's S-LON / S-HEAD with the real head in place of the simulated one — so §2's σ bars and §2.3's penalty apply to
  them directly.
* `navc_ann` — the shipped nav-compliance predicate fed the ANNOUNCED time-localised nav (A5/A6 T3 form; D4 L2) instead
  of the per-clip token; `navc_gate` (0.163 at 50,400) keeps its value and η starts at 0.

**Training the selector (X1)** — the single-winner CE (`selection_ce`) is replaced by:
1. **Listwise soft-target CE** over the reach survivors (plus allocated candidates): p\*_k ∝ exp(−cost_k / T), cost_k =
   mean_j |s_k(j) − s_GT(j)| / 1 m (cumulative arc length at each slot = the speed profile) + |wrap(θ_k − θ_GT)| / 15°
   (terminal heading), T = 1 — **identical to R1-H4** so R1's result transfers. It is Hydra's imitation target
   y_i = softmax(−‖T̂ − T_i‖²) (PUBLISHED 2406.06978 Eq 8 p3) and TNT's ψ = softmax(−D/α) (2008.08294 Eq 5–6) with a
   speed-profile + heading distance.
2. **Per-candidate BCE sub-score heads** on the emitting pass's per-candidate query q_k (concatenated with φ_k and 8
   geometry features): dir-correct, progress-within-10 %, heading-within-15°, constraint-satisfied — the four A4
   bounds turned into learned critics. Combined Hydra-style, `+ Σ_m w_m · log σ(ŝ_m,k)`, w_m zero-init (Hydra Eq 11: its grid search put the
   rule-based weights an order of magnitude or more above the imitation weight, p3). PUBLISHED: imitation-only 80.9 → per-sub-score 83.0 →
   weighted 85.7 → +EP 86.5 PDMS; one distilled overall score 80.2 (2406.06978 Tab. 1 p4).
* ⚠ No primary measures listwise vs per-candidate BCE for trajectory SELECTION (the literature check found none);
  refcv7's own evidence is that 16-feature linear / boosted re-scorers on the FROZEN terms fail (route §2: the
  information is not in those terms), so the selector must read the scene (q_k, BEV at its waypoints) and its gradient
  must reach the decoder — which is what (1)–(2) do and A2/A3 could not.
* E9 stays (W3 "E9 off" −0.071 did NOT replicate, route §2); it is re-trained under the listwise target.

### 3.3 (c) The route checkpoint (R8-3)

**Input:** rc = (x/L, y/L, d_arc/L, valid) in the NOW vehicle frame, L = WP-A's lookahead (RC-A: fixed arc length
L ∈ {30, 50, 80} m on a SMOOTHED path; RC-B: the end of the next announced manoeuvre) — WP-A's echo study chooses.
Training dropout p 0.3 (valid = 0, values 0, the bit re-applied in the model — the `nav_args` X15 rule,
`refc_v3.py:1609-1618`).

**Three entry points, each zero-init:**
1. tactical decoder condition — a SEPARATE zero-init `_QueryFiLM`-style module over [nav args (3), rc (4)] added to the
   existing per-layer FiLM output (no existing tensor changes shape): the tactical layer learns "the road turns left
   in 40 m" → TURN_L, t_start, Δψ.
2. operative condition `rc_to_cond` Linear(4 → 384), zero-init (the `lan_to_cond` / `ego_to_cond` seam, `refc.py:3169-3175`).
3. selection: `ρ · cos(φ_k(L) − φ_rc)` — the bearing of the candidate at arc length min(L, its length) vs the RC
   bearing, param-free, LATERAL ONLY (the `_lan_anchor_prior` form, `refc.py:2917-2941`; GOAL_INPUT's rule that a
   supplied route must never carry along-track = speed), ρ zero-init.

**The echo risk, and what is done about it.** Hidden Biases: TP conditioning is a shortcut — out of distribution the
waypoints "extrapolate towards the nearest TP"; a TP beats a discrete command on route completion (RC 84 vs 56) while
reliance on it is "strong" (PUBLISHED 2306.07957 p1, Tab. 1 p3, p4, p8). Its remedies: attention pooling instead of
global pooling (RC 84 → 93) and shift/rotation augmentation (Tab. 2 p5). CIL's goal-vector conditioning "veers off the
road attempting to shortcut to the goal" (24 % / 30 % success, PUBLISHED 1710.02410 p6).
* Our planner's candidates cross-attend the image and read the BEV at their OWN waypoints (spatial, not pooled); the RC
  reaches SELECTION only through a geometric, lateral-only term.
* **Yaw augmentation is exact for the yaw component on our camera:** the corpus is CYLINDRICAL (column linear in
  azimuth, CLAUDE.md optics trap), so a yaw of θ is a horizontal shift of f·θ pixels; rotate GT, ego history, nav args
  and RC by −θ. θ ~ U[−π/6, π/6] is DriveSuprim's range (PUBLISHED 2506.06659 p4–5: +0.3 EPDMS, Tab. 5; "only 18 % of GT
  trajectories involve turns exceeding 30°", p2). ⚠ It touches the trunk's input distribution and the perception
  targets (map raster, boxes must rotate too) — a v7-tiny lever with its own regression arm, NOT a default.
* **Mandatory controls on every RC result (PLAN R8-3):** E-1 the trivial planner that drives to the RC at constant v0
  (floor the model must beat); E-2 RC shuffled across windows (must lose the gain); E-3 speed leak — OOF R² of the 6-s
  speed profile from v0 vs v0 + RC (≈ equal required); E-4 lateral leak — the RC's offset from a heavily smoothed route
  at the same arc length; plus RC-OFF (valid = 0) on the same windows. ⚠ D0c shows WHY this matters: a ≤ 15° terminal-
  heading cue alone reaches the route bars — an ego-future RC can supply exactly that cue, so a route-bar pass with the
  RC on is quoted ONLY beside its RC-off and shuffled rows.
* NavSim deploy analogue: the scene's route centreline at the same arc length (EvalFlyWheel bridge, WP-A §1).

### 3.4 (d) X2 — the residual prior
**Evidence:** pick correctness on GT turns 0.967 when the prior's side is right vs 0.674 when wrong (D0b); 14/17 wrong
turn picks follow it. The mechanism is the copycat shape: a candidate "near Δ = 0" continues the current yaw.
* **X2a — lateral prior dropout:** per training row with p 0.3, κ0 → 0 (a0 and v0 kept), so on dropped rows the bank is
  the absolute lateral vocabulary at v0 and "the smallest lateral residual" stops being a free winner. ChauffeurNet
  drops past motion on 50 % of examples (PUBLISHED 1812.03079 §4.2 p8) but reports NO isolated measurement of it, and
  Fighting Copycat measured a history-dropout baseline WORSE than plain BC in 5 of 6 MuJoCo tasks (2010.14876 Tab. 2 p7).
  ⇒ X2a is a HYPOTHESIS with a regression arm (p 0, 0.3, 0.5), never a default by citation.
* **X2b — a prior-free sampler group:** G = 2 groups (`sampler_groups`, F7 infrastructure): group 0 rolled on P, group 1
  with κ0 = 0, at train and inference — a structural guarantee that every lateral mode has candidates whose Δ is not
  measured from the current yaw. Cost: 2× the sampler candidates (or 117 split 59/58 with the anchor subset).
* **Measure:** pick correct on GT-turn windows where the prior's side ≠ GT (refcv7 0.674, n 46, seed 1 0.696) — target:
  the prior-right rate's neighbourhood; wrong turn picks that follow the prior (14/17); straight windows unchanged.

### 3.5 (d) X3 — the speed input
* PI §8-2 decides N2 vs N3. Both map onto the EXISTING 4-way one-hot slots {30, 50, 100, 120} km/h (N2's 70/80 → 100,
  130 → 120; N3 urban/rural/motorway → 50/100/120), so `MaxSpeedOneHotEncoder` and every trained weight keep their
  shapes; only the VALUE source changes (past-only, leak 3.5 % / 1.9 % vs refcv7's 23.8 % — MEASURED D4).
* The all-zero "unknown" row is trained with p 0.45 (the NavSim navtest share, INHERITED D4) on the tactical condition
  AND the ceiling (unknown ⇒ +∞, `refc_v3.py:1893-1894`).
* The ceiling reaches the emitted plan on the tip (`e9_rank`, `refc_v3.py:1021-1041`). Required beside every speed
  result: input-REMOVED and input-SHUFFLED rows, and obedience (plan > fed ceiling, refcv7 110 / 2,059 reel windows).

### 3.6 (d) X4 — loss budget and label-state isolation
* **The 0.29 % is mis-read if it is read as "the tactical head is starved".** The tactical decoder's own parameters
  receive gradient ONLY from the tactical loss (every planner feed is detached, `refcv6_tactical.py:686-692`), and Adam's
  update magnitude is invariant to rescaling the gradient (PUBLISHED 1412.6980 p1) ⇒ w_tac barely changes how fast the
  HEAD learns (weight decay and ε aside — derived, not measured). What 0.29 % does starve is the SHARED producers the head
  reads — the BEV tokens (the PI let the tactical loss reach the trunk, `refc_v3.py:1960-1961`) and the agent tokens —
  against box3d 40.3 % and agent 37.4 % of the loss value (D4).
* **Rule:** the budget is set from the MEASURED gradient share on the last shared layers (GradNorm's object, "the last
  shared layer of weights", PUBLISHED 1711.02257 p2): a `grad_share` instrument (every 500 steps, one batch,
  `autograd.grad` per family on {`feat_proj`, `agent_embed`, the planner BEV pool, the trunk's last stage}) logs
  ‖∇L_f‖ per family; the v7-tiny rung chooses the FIXED w_tac that puts the tactical family at a pre-registered share
  (proposal: ≥ 5 % on the BEV/agent producers); GradNorm (adaptive) is a secondary arm only. Inverse-√frequency class
  weights on the lat/lon CE (D4).
* **Label-state isolation (D1 F2):** `v7_labels.load_v7_labels` writes `_MEASURED_GEOMETRY_TOKENS` /
  `_MEASURED_COT_TOKENS` as module globals (`v7_labels.py:430-436`); the trainer loads TRAIN, then EVAL, then forks the
  workers — so the train workers read the EVAL policy and `LANE_CHANGE_L` was a negative on 100 % of in-band windows
  (173,409 vs the census' 12,145, D1). Fix: the negative policy becomes a frozen field of the returned `LabelManifest`,
  passed explicitly into `tactical_goal_targets`; no module state. Test: load train then eval in one process, build the
  train dataset, round-trip it through pickle (= a worker), assert the train policy is in force; **mutation** —
  re-introduce the global read ⇒ RED. ⚠ `v7_labels.py` is a DATA module (not on my ownership list): the fix ships in
  `code/fix/` for the Master Mind to route; the v9 reader (WP-A) is required to hold no module state at all.

### 3.7 (e) Warm start — step 0 reproduces refcv7-50,400
| new piece | init / gate | why step 0 is unchanged |
|---|---|---|
| per-candidate modulation (4 layers × Linear(21 → 768)) | zero | `q·(1+0)+0 = q` exactly |
| tactical constraint heads | default init | they feed only φ_k (zero-modulated) and sat terms (zero gates) |
| tactical condition for nav args + RC (separate module) | zero | additive to an existing FiLM output |
| `rc_to_cond` | zero | additive to `cond` |
| β_lat, β_lon, γ_j, ρ, η, sub-score weights w_m | zero scalars | additive score terms ×0 |
| sub-score heads, listwise loss | — | affect training, not the step-0 forward |
| allocated candidates | excluded from the emitted argmax until `S_emit` | the emitted pick is over the 117 |
| X2a / X2b, RC dropout, condition dropout, scheduled sampling, allocation noise | training only; **dedicated `torch.Generator`** | the refcv7 RNG stream (the sampler's ε) is untouched |

* **State dict:** no existing tensor changes shape (the 8-wide heads keep 8 outputs with masks; the 4-way speed one-hot
  keeps 4 slots; the tactical FiLM is extended by a separate module, never by widening `_QueryFiLM.proj`). Load = strict
  on every refcv7 key + an explicit DECLARED list of new keys (the declared-vs-built registry gets one entry per new
  module, and G-DVB / G-LIVE see every new tensor take a gradient within the smoke).
* **Construction order:** every new module is built LAST (the `refcv7_wta` discipline, `refc_v3.py:1429-1463`), so an
  OFF build is byte-identical to refcv7's init stream.
* **The identity test (stage 4, CPU, smoke-size model):** build refcv7-config M7 and refcv8-config M8 (every new flag
  ON), copy M7's state dict into M8 (strict on old keys, declared new keys), same inputs, same seed, eval mode ⇒
  `torch.equal` on `traj`, `sel_idx`, `anchor_traj`, `sel_score`, `sel_score_v3`; train mode with the training-only
  draws at p = 0 ⇒ the refcv7 loss terms equal. A deliberate-regression arm initialises ONE new gate to 1e-3 and must go
  RED.
* Optimiser: fresh AdamW moments for new tensors; the refcv7 moments are reset (warmup 1k steps) — a decision for the
  launch SPEC, stated so it is not made silently.

### 3.8 Size
New parameters ≈ 0.2 M (modulation 68 k, sub-score heads ≈ 0.1 M, constraint heads, tactical extra FiLM, scalars) on
refcv7's 100,468,987 ⇒ ≈ 100.7 M, far under the 300 M ceiling. Compute: +27 % sampler candidates with allocation
(+100 % for X2b if it doubles instead of splits).

### 3.9 As implemented (stage 4, `code/fix/`, base tip `49a0655`) — every deviation from §3.1–3.8

The text above is the stage-1 design. The code differs in the points below. Where they disagree, the code and the registered
`SPEC_WPB.md` (sha256 `c952d4b4…`) win, and this section says why.

1. **Hypothesis vocabulary in frozen v7 ids. v9 has not landed.** lat3 = (LANE_KEEP, TURN_L, TURN_R) → v7 (0, 6, 7). lon6 =
   (FOLLOW, CRUISE, BRAKE_TO, CREEP, HOLD, ACCELERATE) → v7 (0, 1, 3, 4, 5, 7). That gives 18 joint hypotheses. The masked
   v7 classes are lat {1–5} (NUDGE / LANE_CHANGE) and lon {2, 6}. The posteriors are masked BEFORE the softmax, detached,
   and floored at log 1e-4.
2. **φ_k has 17 dims, not ≈ 21:** [lat3 one-hot 3 | lon6 one-hot 6 | log p lat | log p lon | c (4) | c-valid | allocated-bit].
   The 4 constraint numbers are lat (ψ_term/π, t_onset/6 s at a 10° onset) and lon (log1p(P₆)/log1p(120 m), v(6 s)/30 m/s).
   They are regressed by `cons_lat` / `cons_lon` [B, 8, 2] on the action queries. Their targets come from the window's own
   future (`constraint_targets`).
   * **NOT built:** κ_max, d_stop / t_stop, d_lead / time gap, v_target / t_reach. Each needs the v9 constraint fields, and
     D0 sized only progress (σ) and heading.
3. **Base-fan candidates get tag + log p, but NO constraint (`base_constraints` False).** A base candidate's tag is usually
   not the GT hypothesis, so its constraint input would always be a predicted constraint of a non-GT class. That is exactly
   the case §3.1 calls unsupervisable. Constraints therefore enter only the allocated and forced candidates. SPEC_WPB's T1
   row registers "tag + log p, no constraint". `--r8-base-constraints` exists as an arm, not as a default.
4. **The tagger is the R1 / v9 turn bar applied to the candidate's [0, 6] s plan** (`tag_paths`):
   * lateral: 30° heading excursion, paths under 5 m are LANE_KEEP;
   * longitudinal: speed literals 0.5 / 2.0 / 1.5 m/s.

   ⚠ **Window mismatch:** v9 labels are banded [NOW+2, NOW+8] s, but the plan, the tags and the constraint targets live on
   [0, 6] s. Before v9 labels supervise the heads, the label↔tag agreement on the overlap [2, 6] s must be measured (an
   item for the §7 v7-tiny registration). Consistency and controllability score only plan-observable hypotheses (§4).
5. **Allocation.**
   * Counts: largest remainder, top-4, ≥ 2 each.
   * Source anchors are chosen by tag-match level: 2 = lat + lon, 1 = lat, 0 = any.
   * x0 is gathered by source anchor.
   * Noise: base ε comes from the global stream; extra candidates' ε comes from the dedicated r8 generator. The refcv7 RNG
     stream is therefore untouched (pinned).
   * ⚠ Because sources are selected by tag, an allocated TURN candidate's own direction share is 1.0 BY CONSTRUCTION. The
     controllability reading is therefore the PICK under forced classes and the forced-condition passes, never that share
     alone.
6. **Selection.**
   * The factorised terms (β_lat, β_lon, γ_prog, γ_head, ρ) are zero-init scalars, as designed.
   * Listwise soft-target CE REPLACES the single-winner selection CE when `w_listwise > 0` (the SEL CE is multiplied by 0),
     as in R1-H4.
   * The η·navc term is not duplicated: refcv7's `navc_gate` already carries it.
7. **The extended fan and floating point:**
   * `traj`, `sel_idx` and the base fan are bit-identical.
   * Base-candidate SCORES move by ≤ 1e-5 (MEASURED 9.5e-7 on the CPU rig). The cause is a GEMM over 149 rows instead of
     117, and SPEC I-W states this.
   * E9's clamp scale is computed over the base candidates only.
   * The DECODER seam clamp normalises over the whole row, so `attach_refcv8` REFUSES `seam_clamp > 0`.
8. **Route checkpoint (RC) as built:**
   * tactical `cond_extra` = [nav args (6, incl. known) | RC (4)], through per-layer zero-init `_QueryFiLM` modules;
   * operative `r8_rc_to_cond` = Linear(4 → d), zero-init and gated;
   * selection bearing ρ, lateral only;
   * training dropouts: RC ≥ 0.3 (the pin refuses lower values); nav args 0.5, with the token kept and the
     "unknown distance/time" flag set.

   **Information flow.** The RC is built from the route input only. `assert_no_situation_feed` and
   `test_the_route_checkpoint_path_carries_nothing_from_the_tactical_layer` pin that no situation-classifier or tactical
   output enters it: the RC tensor reaching `rc_to_cond` is bit-identical under a forced tactical posterior. The tactical
   layer's OWN posteriors conditioning the planner is the hierarchy the PI asked for, and it is a separate path.
9. **Label-state isolation (trainer side).** A `V7PolicyScope` snapshot is taken right after each label load. Each dataset
   carries its own snapshot, pickles with it, and applies it around `tactical_class_ids` / `tactical_goal_targets`. The
   module-level fix belongs to WP-A. `test_refcv8_label_scope.py` pins this on the real v8 blobs: it reproduces the
   LANE_CHANGE_L flip, and a late-snapshot mutation goes RED.
10. **The v9 release is wired** (WP-A `INTEGRATION.md`; MM binding 2026-10-04, items 1–6). The flags are
    `--r8-v9-labels` / `--r8-v9-labels-eval` (md5-pinned by `--r8-v9-md5` / `--r8-v9-eval-md5`), `--r8-v9-lat-variant`
    and `--r8-nav-from-v9`.
    * **Join.** Keyed by (sid, k = t + w − 1 + raw offset). Every window is joined at launch through the reader's own
      refusing `row_for_now` (clock tolerance 1e-6 s). MEASURED on eval139 through the trainer's path: 23,772 / 23,772
      windows, clock residual 0.0 s (`raw/v9_join_eval139.json`).
    * **Targets.** lat / lon (frozen v7 ids, partial labels) and the 22 goals REPLACE v7.2's. The goal pos_weight and
      class mask are re-derived from the v9 TRAIN windows. Partial labels enter as `−log Σ_allowed p` inside the SAME
      tactical weights (`v9_partial_correction`).
    * **Inputs.** nav args (log-scaled distances, Δψ/90, side, known) and RC-A50 (x/50, y/50, ψ/90, valid).
    * **Training treatment.** RC dropout ≥ 0.3. Registered RC noise of σ 2.0 m along-track and 0.75 m lateral in the
      route-tangent frame; smaller values are refused, because the clean point leaks (E2′). Nav-args dropout 0.5 with
      the token kept.
    * **Switches.** `--r8-no-rc` turns the RC path off, since the RC ruling awaits the PI. `model._r8_legal_row` gives
      the NavSim-legal row: known 0, RC invalid, token kept. `navsim_legal_nav_cmd` maps left/forward/right/unknown to
      1/0/2/0.
    * **Contract census at load (refused on any violation).** It checks: no exact LANE_CHANGE action; reversing rows
      carry no action label and no geometry goal; no absence class or absence goal on a band shorter than 8 s (a
      TURN_x negative that is entailed by the other side's positive is exempt); exact masks equal their one bit;
      SPEED_BAND is never supervised. Both real releases read 0 on all seven counts.
    * **Corpus.** `--refcv8 --agent-join` REFUSES without `--join-defect-masks`, which is the 693-box ego mask. WP-C's
      list equals WP-A's manifest mask frame for frame at k = frame_idx + 2: 693 / 693 frames, 18 / 18 clips
      (`raw/mask_crosscheck.json`). The 8-clip drop list is NOT applied (PI decision 8 is open).
    * **v7 policy.** With WP-A's fixed `v7_labels` the policy travels on the labels, so no scope is installed.
      `V7PolicyScope` remains only as the fallback for an unfixed module.
    * **Not consumed yet:** v9's constraint vectors (`lat_c`, `lon_c`, `speed_goal`, on the [2, 8] s band). The constraint
      heads still regress the plan-derived [0, 6] s quantities that the generator conditions on. Supervising the v9 set
      is an item for the §7 v7-tiny SPEC.
11. **WP-C I1–I3 are wired** (opt-in; `INTEGRATION_WPC.md`): `--det-nms`, `--det-zh-trust` (the `model._det_zh_range`
    carrier, also set in `refcv7_loader`) and `--join-defect-masks` (TRAIN reader only). The G-DVB registry holds 223 + 3 +
    30 = 256 entries: 24 from refcv8 stage 4, plus 6 from the v9 wiring.
12. **The five `--w-r8-*` weights default to 0.0 and are gated.** They are registered in `REFC_WEIGHT_GATES` (the
    effective-weight audit's exhaustiveness contract) and in `launch_gate.LIVE_WEIGHT_RULES` (G-LIVE's). A non-zero
    default would be refused as NO_GRAPH on every run that does not pass `--refcv8`, because the audit does not
    consult explicitness. A refcv8 launch therefore STATES its weights, at the SPEC values:
    * `--w-r8-cons 0.05` always;
    * `--w-r8-alloc-l1 1.0` with allocation;
    * `--w-r8-sat` / `--w-r8-listwise` / `--w-r8-subscore` per arm.

    `_pin_refcv8` refuses the converse cases: `--refcv8` with `--w-r8-cons 0`, and allocation with `--w-r8-alloc-l1 0`.
    The harness path (`wpb_arms.py`) builds `R8Config` directly and is unaffected.
13. **Not built yet (named):**
    * yaw augmentation (§3.3, a v7-tiny lever);
    * the X4 `grad_share` instrument;
    * classifier-free guidance (an optional arm).

---

## 4. The R8-4 measures (stage 2 — what will be built in `taniteval`, no model change needed)

| measure | definition | control that must read a known value | mutation that must go RED |
|---|---|---|---|
| (i) tactical accuracy | lat/lon accuracy + macro-F1 per class with n on dense v9 labels (and on the old anchor band); goal AP per token vs its prevalence (a constant scorer's AP = prevalence); constraint MAE (t_start, Δψ, d_stop, v_target, t_reach) vs the TRAIN-median constant and vs a v0-kinematic prior | majority-class accuracy; constant-predictor AP = prevalence exactly | labels shifted +6 s ⇒ accuracy falls; sign-flipped lat ⇒ L/R recall swap |
| (ii) controllability | force h ∈ {TURN_L, TURN_R, LANE_KEEP, STOP-at-d} on the same scene; read the share of the hypothesis-conditioned candidates and of the pick whose OWN path class equals h; STOP: \|stop distance − d\| ≤ max(2 m, 0.1 d); bar ≥ 0.95 per forced class, both sampler seeds | a model that ignores the condition (forced == unforced fan) reads the base rate; analytic paths (R 15 m arcs, straight, 1.5 m/s² stop at d) read 1.0 / 0.0 | mirrored y (L ↔ R) ⇒ TURN_L reads 0 |
| (ii′) shuffled-condition arm | the same read when the condition comes from a deranged window | must sit at the base rate (lose controllability) | — |
| (iii) consistency | share of candidates whose path class equals their tag; pick vs its tag; pick vs tactical argmax; on all / GT-turn windows; refcv7 baseline (tag := tactical argmax) 0.371 / 0.352 fan, 0.751 / 0.449 pick (D0) | a fan built from its own labels reads 1.0 | tags permuted ⇒ falls to the class base rate |
| (iv) route following | turn dir-correct pick ≥ 0.95, heading-15 ≥ 0.70 (refcv7 0.841 / 0.514) — the route package's `route_metrics` | B1 / B3 reproduction (as D0's C1) | — |

⛔ The path-class rule used by the METRIC is written independently of the model's tagger (the cross-check must not share
the code it checks) and the two are held against each other on analytic tracks; the plan horizon is 6 s while the v9
band is [2, 8] s, so consistency/controllability are scored only for hypotheses observable inside the plan (t_start ≤
5 s), with n reported for the rest.

## 5. Validation plan (stage 3 will draft it as a SPEC; bars committed before any number)

* **On the R1 harness** (R1's cache and `r1_heads.py` are reused — no second cache; the decoder and sampler are
  trainable there): T1 = R1-H4 + per-candidate tags with constraints (no allocation); T1s = tags from a deranged window
  (regression); T2 = T1 + L_sat; T3 = T1 + allocation; T3r = allocation to random hypotheses (regression); T4 = factorised
  selection with sat terms; T4d = sat terms fed deranged constraints (regression; D0's C-SH twin); X1h = Hydra sub-score
  heads on top of H4; X2a / X2b with p = 0 replicate. Foil: R1-H5 (one window vector) on the same windows.
* **On the v7-tiny ladder** (trunk-dependent): X4 budget from `grad_share`; RC input with and without yaw augmentation;
  the label-state fix's census check; each with a same-flag replicate (H-ESTIM-SEED-1: on that rig a "separated" result
  needs the replicate floor).
* **Launch gate** for the warm-started run: the identity test, G-DVB / G-LIVE on every new tensor, and the R8-4 suite
  producing all four families for refcv7-50,400 as the baseline row (X8).

## 6. Open items and decisions owed

1. **v9 tensors** (WP-A `INTEGRATION.md` not yet written): constraint fields, the 3-way lat mapping, the lon class list,
   the RC variant and L. The design is written against D4's definitions and the WP-A brief; integration waits for them.
2. **PI §8-2** (N2 vs N3) — both fit the design.
3. **`v7_labels.py` ownership** for the X4 fix (data module) — the Master Mind routes it.
4. **Allocation and X2b cost** (+27 % / up to +100 % sampler compute) — measured on R1 before any adoption.
5. **Yaw augmentation** touches perception targets — v7-tiny only, with WP-C/WP-D.

## 7. Literature (all banked; claims verified on the primary by page)

| key | used for | verified claim |
|---|---|---|
| 2503.05689 GoalFlow | goal-conditioned generation; oracle-goal gap | goals 85.6 → 88.5 → 89.4 → 90.3 PDMS (Tab. 2 p7); GT-endpoint goal 92.1 vs 90.3 (Tab. 1); selector is hand-set (Eq 12), not learned; no shortcut analysis |
| 2411.15139 DiffusionDrive | anchored truncated diffusion; per-anchor BCE | 20 anchors, 2 steps (Eq 4–6); mode diversity 11 % vs 74 % (Tab. 2 p6); anchored prior 88.1 vs extrapolated prior 81.3 / one extrapolated anchor 84.7 (Tab. 8 p12) |
| 2406.06978 Hydra-MDP, 2503.12820 Hydra-MDP++ | soft imitation target; per-sub-score heads; log-sum aggregation | Eq 8–11 p3; 80.9 → 83.0 → 85.7 → 86.5; single distilled score 80.2 (Tab. 1 p4) |
| 2506.06659 DriveSuprim | rotation augmentation; turn imbalance | Θ = π/6 (p4–5); 18 % of GT trajectories turn > 30° (p2); Tab. 5 +0.3 EPDMS |
| 2306.07957 Hidden Biases | the target-point shortcut and its remedies | "Steering directly towards a TP is a shortcut" (p1); TP vs command RC 84 vs 56 (Tab. 1 p3); attention pooling RC 84 → 93 (Tab. 2 p5) |
| 2205.15997 TransFuser | how a goal point is fed | rasterised BEV channel + GRU input (§3.2, §3.4); no-goal-raster ablation "unlikely to be significant" (Tab. 9 p14) |
| 1710.02410 CIL | branched vs command-input; goal vector shortcut | 88/64 vs 78/52 %; goal vector 24/30 % "veers off the road" (Tab. 1 p6) — single run |
| 2312.03031 Ego status | the perturb-the-input echo test | velocity ×0 ⇒ 6.16 m L2 vs 0.37 m (Tab. 2 p5) |
| 1812.03079 ChauffeurNet, 2010.14876 Copycat | X2 | 50 % past dropout, no isolated measurement (§4.2 p8); history dropout below plain BC in 5/6 tasks (Tab. 2 p7) |
| 2207.12598 CFG | condition dropout | p_uncond 0.1 ≈ 0.2 < 0.5 (§4.2 p8) |
| 2212.09748 DiT, 2302.05543 ControlNet | zero-init identity at warm start | "initializes the full DiT block as the identity function" (§3.2 p5); zero conv ⇒ "yc = y" at step 1 (Eq 2–3 p4) |
| 1506.03099 Scheduled Sampling | constraint teacher forcing | linear / exponential / inverse-sigmoid schedules (p4); per-sequence coin "much worse" (fn 2 p3); Always Sampling 11.2 vs 30.6 BLEU-4 (Tab. 1 p6) |
| 2008.08294 TNT, 1910.05449 MultiPath, 2209.13508 MTR | target-conditioned generation; hard vs soft mode assignment | TNT teacher-forces the GT target and scores with softmax(−D/α) (p5–6); MultiPath / MTR hard-assign the closest anchor / intention (Eq 3 p4; §3.3 p6) |
| 1412.6980 Adam, 1711.02257 GradNorm | X4 | "magnitudes of parameter updates are invariant to rescaling of the gradient" (p1); balancing on "the last shared layer of weights" (p2) |
| 1909.06722 Plackett-Luce LTR | listwise loss background | ListNet = KL between distributions; ListMLE "rather unstable in many other cases" (p2) |

⚠ Not supported by any primary (do not quote): a measured gain of listwise over per-candidate BCE for trajectory
selection; an isolated gain of past-motion dropout; a TP-placement mitigation in Hidden Biases; a branched-vs-input
difference with intervals.

---

## Deliverable manifest (stage 1)

| artifact | where | note |
|---|---|---|
| `DESIGN.md` | this package | this file |
| `SPEC_D0_DOSE_RESPONSE.md` + `raw/SPEC_SHA256.txt` | this package | registered before any number (sha256 `7c7e4f52…`, 2026-10-04T11:11:58Z) |
| `code/d0_dose_response.py` | this package | refuses on a SPEC hash mismatch; md5-checks the banked fans |
| `code/d0b_prior_follow.py`, `code/d0c_heading_posthoc.py` | this package | descriptive / POST-HOC, labelled so |
| `raw/d0_dose_response.json`, `raw/d0b_prior_follow.json`, `raw/d0c_heading_posthoc.json`, `raw/d0_run.log` | this package | every D0 number above |
| banked papers (2212.09748, 2302.05543, 1506.03099, 1910.05449, 2209.13508, 2008.08294, 1909.06722, 2512.12302, 1412.6980) + `--cited-by` on 13 already-banked | `TanitAD Research Lab/Library/` (D: only; `library.json` / `LIBRARY.md` regenerated by `kb_add`) | `kb_add --verify` 605 entries / 0 orphans / 0 problems after the last add |
| inputs read (not modified) | `D:/refcv7_route_bin/2026-10-04/{eval_s0g,eval_s1,train_s0}.npz` | md5 `1c48a53e…`, `5f783c84…`, `53e06ec5…` |
