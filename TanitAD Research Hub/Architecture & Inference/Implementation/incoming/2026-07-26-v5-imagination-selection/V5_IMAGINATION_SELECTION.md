# V5 — imagination-scored selection over the frozen v4 fan

**Stream:** v5 synthesis (PI directive, 2026-07-26, after Bar A refuted the discriminative selector).
**Host:** `tanitad-eval` (`1e0bac0df88a`). **Corpus:** the canonical 40-episode / 881-window val
deployment. **Surface:** `produced` (deployable) unless a row says `oracle`.

> ⛔ **Everything above the line `--- MEASUREMENTS BEGIN ---` was written and staged BEFORE any
> number in this experiment existed.** BOOST_PROGRAM M2/M3.

---

## 0. PRE-REGISTRATION

### 0.1 The hypothesis, and why Bar A does not already answer it

Bar A established (MEASURED, `…/2026-07-26-bar-a-selector/raw/bar_a_produced.json`) that **re-scoring
the frozen v4 fan with a learned head over the existing latent features cannot reach v1**: fitting
*and* scoring on the **same** 6,844 windows — maximum overfit, zero generalization gap, 796 k free
parameters — bottoms out at **0.4907** (CE) / **0.5224** (regret) against v1's **0.4271**. Out of
fold both arms were *worse than doing nothing* (−4.2 % / −11.0 % of the waste recovered).

Four things Bar A did **not** establish:

1. **that the fan is bad** — it is good: `oracle_in_fan` = **0.2505** on the deployable surface,
   41.3 % better than v1's 0.4271;
2. **that a different selector INPUT fails** — only the existing feature set (`q_final`,
   `state_last`) was tested;
3. **that SIMULATIVE selection fails** — Bar A tested a *discriminative* scorer, which reads
   features and emits a rank;
4. **that HIERARCHICAL selection fails** — a flat 1-of-256 scorer was tested.

⇒ **H-V5: the information a discriminative head could not extract from the features is recoverable
by ROLLING EACH CANDIDATE FORWARD THROUGH THE WORLD MODEL AND LOOKING AT WHAT HAPPENS.**

### 0.2 The instrument — reused, not rebuilt

`tanitad/models/metric_dynamics.py::rollout_decode` advances its window by appending the model's
**own predicted latent** (`win_s = cat([win_s[:, 1:], z_hat])`); **no new frame is ever encoded** and
`k` is free. Verified in the pod's live tree (`/root/v4eval/stack`), which is byte-identical to repo
HEAD for this module modulo line endings (§1.2). It is a per-candidate imagination roll-out as it
stands.

### 0.3 Candidate → action sequence (the one new piece of mechanism)

The v4 fan is 256 **trajectories** (`[B, 256, 20, 2]`, dense 0.1 s ego waypoints), not action
sequences. `rollout_decode` needs actions. The map is the *exact inverse of the corpus's own action
definition* (`tanitad/data/physicalai.py:51,412` — `WHEELBASE = 2.9`,
`steer = arctan(WHEELBASE · curvature)`, `accel` = the dataset's accel), so it introduces no new
convention:

```
dt = 0.1 ;  p = [ (0,0) , w_c[0..19] ]                        # 21 points
d_j   = p_j - p_{j-1}                       v_j = ||d_j|| / dt        (v_0 := observed v0)
psi_j = atan2(d_j.y, d_j.x)  (held when ||d_j|| < 1e-3)      psi_0 = 0
omega_j = wrap(psi_j - psi_{j-1}) / dt      kappa_j = omega_j / max(v_j, 0.5)
steer_j = clip(arctan(2.9 * kappa_j), ±MAX_STEER_RAD)
accel_j = (v_j - v_{j-1}) / dt
3rd channel = v0 / SPEED_SCALE (=10.0), constant — the leakage-safe v1 contract
```

**Rollout convention (recorded so it is falsifiable):** transition 1 is driven by `win_a[:, -1]`, so
the candidate takes control of it by **overwriting the last window action with `a_c[0]`**;
`future_actions = a_c[1:]` then drives transitions 2..20. The candidate therefore controls the whole
2 s roll-out and nothing else changes.

### 0.4 The arms — frozen before measurement. NO TRAINING in the first pass.

| id | rule (argmin/argmax over the 256 frozen candidates) | uses WM? | per-candidate roll? |
|---|---|---|---|
| **A0** | `as_trained` — v4's own selector score. The thing to beat. | – | – |
| **A1** ⭐ | **`imag_consistency`** — argmin_c mean_j ‖τ̂_c[j] − w_c[j]‖. *"Does the world model agree that executing this plan produces this plan?"* **PRIMARY.** | ✅ | ✅ |
| **A2** | `imag_goal_speed` — argmin_c \|v̂_term(τ̂_c) − v_goal\|, `v_goal` = produced `vt_speed` clamped to the same reachable set `select()` uses. | ✅ | ✅ |
| **A3** | `imag_kinematic` — argmin_c mean plausibility (‖Δ²τ̂_c‖, i.e. imagined accel/jerk magnitude). | ✅ | ✅ |
| **A4** | `imag_combo` — weighted sum of A1–A3 + the normalised base score. Weights fitted **(i) in-sample** (comparable to Bar A's in-sample ceiling) and **(ii) 5-fold episode-disjoint out-of-fold** (deployable). | ✅ | ✅ |
| **C1** | **CONTROL, no WM** — A1 with τ̂ replaced by a CTRV/constant-velocity forward model. | ❌ | ✅ |
| **C2** | **CONTROL, one imagination** — argmin_c ‖w_c − τ̂_ref‖, τ̂_ref = the WM roll-out under the **observed** (zero-order-hold) action. No per-candidate rolling. | ✅ | ❌ |
| **R** | `random` — the deliberately failing input. | – | – |
| **O** | `oracle_in_fan` — upper bound. **Never a deployable number.** | – | – |

**C1 and C2 are not decoration.** They are what makes a positive result attributable: C1 asks whether
a WM is needed at all, C2 asks whether *per-candidate* rolling is needed as opposed to one reference
trajectory. A win that C1 or C2 matches is **not** an imagination result.

### 0.5 The scoring world model — the experiment's biggest confound, run as two arms

v4's own `wm_canary_ade_2s` is **1.1409** against a ≤ 0.55 bar; the v1 line is cited at ~0.452
(INHERITED — established here, §2). **Scoring with a bad simulator scores badly.** Two scorer arms:

- **W-v4** — v4's own world model (`/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt`);
- **W-v1** — **v1's world model scoring v4's fan** (`/root/models/flagship-30k/ckpt.pt`). This is the
  PI's direction (1) literally: v1's world model + the REF-C-derived anchored-diffusion fan.
  **Feasibility is to be ESTABLISHED, not assumed** (§4); if infeasible, the reason is a real finding
  about arm compatibility and is reported as such.

### 0.6 Outcomes — committed in advance

| verdict | condition |
|---|---|
| **CONFIRM** | the best pre-registered imagination rule's **in-sample** `ade_0_2s` **< 0.4907** ⇒ imagination-scoring does something discriminative re-scoring of this fan provably cannot. |
| **STRONG** | additionally its **out-of-fold** `ade_0_2s` **< 0.4271** ⇒ the synthesis is real and v5 has a spine. |
| **REFUTE** | fails to beat **0.4907** ⇒ the limitation is the **fan's own action-sequence information**, not the scorer. Say so plainly; do not re-scope. |
| **NOT ATTRIBUTABLE** | (modifier) any of the above where **C1 or C2 matches the winning arm within its paired CI** — reported as a *scoring-rule* result, not an imagination result. |

**Why the 0.4907 comparison is in-sample on both sides:** 0.4907 is itself Bar A's *in-sample*
ceiling. A0/A1/A2/A3/C1/C2 are **parameter-free**, so their single number is simultaneously in-sample
and out-of-fold. Only A4 has free weights, and it is reported both ways.

### 0.7 Estimator

**Paired episode-cluster bootstrap** (`taniteval/ci.py`, B = 2000, resampling unit = **episode
cluster**, 40 clusters / 881 windows), on identical windows. **`overlapping_holdout_se` is never
used, and the `legacy_*` blocks in any JSON are not quoted.** Every number is decomposed into
**along-track (LONGITUDINAL) / cross-track (LATERAL)**, because v4's 15k→30k regression was 100 %
longitudinal and an undecomposed number hides which axis a lever acts on.
⚠️ **A 40-episode "not separated" is UNPOWERED, not refuted** (half-widths shrink ×2.8–3.9 at
n = 600, `MODEL_REGISTRY.md §1.2a`). Power is stated wherever a null is reported.

### 0.8 Self-tests the harness must pass before it may adjudicate (M3)

Both directions, as Bar A did — a harness that has only seen good input has not been tested.

| # | test | pass condition |
|---|---|---|
| S1 | **cache fidelity** | as-trained `ade_0_2s` = **0.8563**, `oracle_in_fan` = **0.2505**, and the pick agrees with the real forward pass on **100 %** of windows |
| S2 | **committed-number reproduction** | the true-action roll-out reproduces `wm_canary_ade_2s` = **1.1409** on these windows |
| S3 | **inverse-map fidelity** | `traj_to_actions(GT waypoints)` recovers the dataset's own `future_actions` (per-channel MAE + correlation reported) |
| S4 | **derivation error budget** | rolling under `traj_to_actions(GT)` lands close to the true-action roll-out; the gap is the derivation's own error and is **reported, not hidden** |
| S5 | **failing input** | a uniform-random pick scores ≈ **15.36** (Bar A's number), and an anti-oracle (argmax fan error) is worse still |

Any S1/S2 failure **aborts before any arm is scored** and this file records `ABORTED`.

### 0.9 What I commit to reading into a REFUTE, in advance

If A1/A4 cannot beat 0.4907, then combined with Bar A the correct reading is: **the 256 fan
candidates do not differ from one another in a way that either their features OR their dynamical
consequences can rank.** `oracle_in_fan` = 0.2505 would then be a statement about the *marginal*
coverage of a 256-anchor vocabulary, not about a recoverable selection surplus — and the v5 lever
must move to the **fan itself** (proposal generation / candidate count / goal-conditioned anchors),
not to any scorer over it. I will write that, not a scoped-down version of the current claim.

### 0.10 Goal-oracle honesty

`route` / `route_graded` / `vt_band` are minted from the ego's own future poses — **three** channels
(`vt_speed` is overwritten with the observed `v0`). Oracle and produced are reported **as a pair**;
**oracle numbers are never deployed capability.**

### 0.11 🔒 Confidentiality

PhysicalAI-AV is gated-confidential. No clip UUIDs and no raw content appear in any artifact here;
episodes are referenced by their val index only.

---
--- MEASUREMENTS BEGIN ---
---

*(Everything below was written after the corresponding measurement.)*
