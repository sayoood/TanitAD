# PRE-REGISTRATION — does giving the tactical planner an ego/action input move the closed-loop surface?

**Frozen before any new number existed.** Nothing below is reinterpreted after the fact.
**Stream:** Architecture & Inference · **Branch:** `agent/benchmarks-eval-20260721` · **Repo HEAD at start:** `d07da62`
**Wall-clock date:** 2026-07-27 (Europe/Berlin). ⚠️ The directory is dated `2026-07-28` because the
brief named it so; the repo's narrative clock runs ~1 day ahead of wall-clock (known artefact).

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

---

## 0. The premise, VERIFIED IN CODE BEFORE ANY MEASUREMENT (this is priority 1 of the brief)

The brief states: *"`TacticalPolicy.forward(states, ctx)` HAS NO ACTION INPUT."* I was told not to
take that from the brief. I read the code. **It is partly false, and the true statement is more
useful.** All line numbers are at repo HEAD `d07da62`.

| # | claim | verdict | primary source |
|:--:|---|---|---|
| 1 | `TacticalPolicy.forward` has no ego/action parameter | ⛔ **FALSE** | `stack/tanitad/models/fourbrain.py:328` — the signature is `forward(self, states, ctx, ego=None)` |
| 2 | The parameter is inert unless a config lever is on | ✅ **TRUE** | `:326` `self.ego_emb = nn.Linear(2, d_cond) if ego_input else None`; `:330` `if self.ego_emb is not None and ego is not None:` |
| 3 | The lever is OFF by default | ✅ **TRUE** | `stack/tanitad/config.py:196` `v2_ego_to_planners: bool = False`; wired at `fourbrain.py:435,443` |
| 4 | It is OFF on every arm in the panel | ✅ **TRUE, MEASURED** | `arm_v1_tactical_oracle.json` / `arm_nospeed_tactical_oracle.json`: `ego_input_on_planners = False` |
| 5 | What the input actually carries | ⚠️ **`[v0/pose_scale, yr0]` — EGO PROPRIOCEPTION, NOT AN ACTION** | `stack/tanitad/train/flagship_losses.py:202-210` |
| 6 | The **strategic** brain has the identical gap | ✅ **TRUE** | `fourbrain.py:73-79` — same `ego=None`, same `ego_emb` gating |
| 7 | The **operative** predictor does NOT have the gap | ✅ **TRUE** | `flagship_losses.py:227-238` appends `v0/10.0` as the 3rd **action** channel; `run_hierarchy` calls `model.predictor(states, actions, …)` (`fourbrain.py:388`) |
| 8 | ⭐ **A SECOND, INDEPENDENT GAP THE BRIEF DID NOT NAME** | ✅ **TRUE** | **Every** eval/planner call site passes `(states, ctx)` positionally and **never** `ego=`: `taniteval/closedloop.py:245,317` · `planner_p2.py:279,340` · `planning.py:166` · `corpus_overlay.py:307` · `blindimag.py:101` · `probe_overlay.py:49` · `panel_run.py:137,165`. **An ego-trained checkpoint is evaluated ego-BLIND, silently.** |
| 9 | ⭐⭐ **REF-C's planner ALREADY CONSUMES `v0`** | ✅ **TRUE** | `stack/tanitad/refs/refc.py:760` `forward(frames, nav_cmd=None, v0=None, …)`; `:786-787` `v = … (v0/10.0)` — the same `SPEED_SCALE=10.0` contract. And REF-C is **separated BELOW `cv_holdv0` at all three scales** (−0.0203 … −0.0255, `PUBLISHED` panel §4). |

⇒ **The corrected premise:** the tactical policy has an ego input *in the class* and **no ego input
in any deployed arm**, for two independent reasons — a config lever that is off, and a harness that
drops the argument even when the lever is on. And **a planner family that already has the input
still loses to constant velocity**, which is the first thing that makes REFUTE a live outcome.

⚠️ **Honesty condition on (9), found before it could be quoted:** REF-C trains with
`ego_dropout = 0.5` (`refc.py:287`) — `v0` is Bernoulli-zeroed on **half** of all training steps, on
purpose, as a shortcut guard. So (a) a `v0 = 0` ablation is **in-distribution**, not OOD, which is
what makes Block A clean; and (b) REF-C was **trained not to lean on `v0`**, so a small Block-A
effect is partly by design and must not be read as "speed inputs do nothing in general".

---

## 1. Why a *trained* ego-on flagship arm is not producible in this session (stated, not hidden)

Only two flagship checkpoints ever had `v2_ego_to_planners = true`:
`flagship4b-v2-30k` (registry §1.3 — **ABANDONED at step 7,800**, ADE@2s **6.179 m**, on **pod2**)
and `flagship-v2corpus-30k` (registry §1.7 — **RUNNING on pod1**). ⛔ The brief forbids touching
pod1 and pod2, and the first is a broken arm whose ego/no-ego contrast would be measured at a
non-competent operating point. **Training a new one is out of budget.** So the design below reaches
the same question by two routes that need no new training, and I say plainly that neither is
"a flagship retrained with `--v2`".

---

## 2. Surface, estimator, gate — fixed before any arm is scored

* **Surface:** pseudo-simulation, `taniteval/taniteval/pseudosim.py`, **imported, never reimplemented**.
  40 val episodes (`physicalai-val-0c5f7dac3b11`, first 40 `ep_*.pt`) · stride 8 · 21 grid points
  (7 heading × 3 longitudinal, lateral refused in code) · **15,981 rows per arm** · 0 rollout steps.
* **Primary quantity:** `PSS_recovery_progress` (the panel composite). ⛔ **NOT `ade_0_2s`** — the
  ADE-optimal pick collides 4.7× more than the rule-optimal one, and published L2/ADE vs closed-loop
  Driving Score is ρ = −0.36, p = 0.43 while Ego Progress is ρ = 0.83 (`PUBLISHED`, weak tier).
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, **B = 2000**, unit = **val episode**,
  paired on **identical `(ep_i, anchor, dlat, dyaw, dlon)` rows — asserted by `panel_combine.py`, not
  assumed**. ⛔ **`overlapping_holdout_se` is refused** (it biases the point estimate, not just the interval).
* **Gate:** the **PANEL-WIDE** gate (`ego_progress` + `recovery`; `comfort` dropped because it is
  inadmissible for `cv_holdv0` and `v4_*`). The per-arm gate is reported only as a sensitivity.
  ⛔ Stated because the brief names it: under the per-arm gate `cv − refc_base` reads +0.1303 vs
  +0.0252, and `cv − v1` flips verdict. **I use the panel-wide gate and say so on every number.**
* **The bar that matters:** `cv_holdv0` = **0.5705 [0.5558, 0.5844]**. ⛔ Not "better than the
  previous version of itself".
* ⛔ **`0.4271` is NOT a bar here** — verified in code, `taniteval/rollout.py:170` sets
  `actions_source="expert_future"`; it is `wm_fidelity_ade_2s`.

**Reproduction gate (run and passed BEFORE the pre-registration was frozen):** `panel_combine.py`
re-run locally on the 10 committed `pw_*.npz` dumps with `CUDA_VISIBLE_DEVICES=""` reproduces the
published artifact — **all 10 composites and all 45 paired blocks byte-identical**. This licenses
every arithmetic-only arm below.

---

## 3. THE ARMS

### Block B (primary) — give v1's tactical plan the speed it never saw. **No GPU: pure arithmetic on the committed `pw_v1_tactical_follow.npz`.**

A plan is a curve plus a schedule. Factor them:

* **shape** `γ(s)` — the curve traced in the plane, parameterised by arc length.
* **schedule** `s_t` — how far along that curve the plan is at step *t*.

`v1`'s tactical head emits both, having never seen `v0`. A speed input's entire job is the
**schedule**. So the 2×2 factorial:

| | shape = **v1's** (it steers) | shape = **straight** (no steering) |
|---|---|---|
| **schedule = v1's own** | `v1_tactical_follow` **0.5471** *(published)* | `v1_lat_straight` |
| **schedule = `v0·t·dt`** | ⭐ **`v1_ego_v0`** — *the arm with the speed input* | `cv_holdv0` **0.5705** *(published)* |

⭐ **The construction validates itself in both directions, because two of its four corners are
already-published arms.** `γ_v1 ∘ s_v1` must reproduce `pw_v1_tactical_follow.npz` **bit-exactly**
and `γ_straight ∘ s_cv` must reproduce `pw_cv_holdv0.npz` **bit-exactly**. If either fails, the
transform is wrong and Block B is void — I stop and report that.

Additional arms:

| arm | definition | role |
|---|---|---|
| `v1_ego_identity` | `γ_v1 ∘ s_v1` | ✅ identity control — must be bit-exact vs `v1_tactical_follow` |
| ⭐ `v1_ego_v0` | `γ_v1 ∘ (v0·t·dt)` | **the arm with the speed input.** Zero fitted parameters ⇒ no out-of-fold problem, and it cannot be dismissed as under-trained |
| `v1_ego_oracle_lon` | `γ_v1 ∘ (true logged arc length)` | ⚠️ **ORACLE — the absolute CEILING of ANY longitudinal input**, including a perfect accel predictor. Upper bound only |
| `v1_lat_straight` | `γ_straight ∘ s_v1` | the other factorial cell — isolates v1's steering |
| ⛔ `v1_ego_half` | `γ_v1 ∘ (0.5·v0·t·dt)` | **DELIBERATE DEGRADATION. Must score WORSE.** |
| `cv_ego_identity` | `γ_straight ∘ s_cv` | ✅ identity control — must be bit-exact vs `cv_holdv0` |

Extrapolation: when `s_t` exceeds v1's own path length the curve is extended along its terminal
tangent. **The fraction of rows requiring extrapolation is reported, per arm.**

### Block A (corroboration) — ablate a *trained* speed input inside one checkpoint. GPU, `tanitad-eval`.

| arm | definition |
|---|---|
| `refc_xl_v0on` | REF-C-XL, real `v0` — must reproduce `refc_xl_produced` **0.5499** to 4 dp |
| `refc_xl_v0off` | identical checkpoint, `v0 = 0` everywhere the planner consumes it |
| `refc_base_v0on` / `refc_base_v0off` | the same contrast at base scale — replication |

---

## 4. THE RULES, and the value of each that returns a FAILING verdict

Primary control = **`v1_tactical_follow`** (the deploy path; the oracle-nav variant is 75 %
degenerate and is reported alongside, never substituted).

| verdict | fires when | ⛔ a value that makes this rule return FAIL |
|---|---|---|
| **CONFIRM** | `Δ(v1_ego_v0 − v1_tactical_follow)` **separated-positive** *(lo > 0)* **AND** `Δ(v1_ego_v0 − cv_holdv0)` **not separated-negative** *(hi ≥ 0)* | any `Δ` vs the control whose CI contains 0 (e.g. `+0.004 [−0.006, +0.013]`); **or** reaching the control but `Δ` vs CV = `−0.015 [−0.026, −0.005]` → falls through to PARTIAL |
| **PARTIAL** | separated-positive vs its control **but** separated-BELOW `cv_holdv0` | `Δ` vs CV with `lo > 0` → promotes to CONFIRM; `Δ` vs control with CI containing 0 → demotes to REFUTE |
| **REFUTE** | `Δ(v1_ego_v0 − v1_tactical_follow)` **not separated-positive** | `lo > 0` on that same contrast → REFUTE cannot fire |

**⭐ The instrument's ability to return each verdict is DEMONSTRATED on these exact rows, not
asserted:** the same estimator on the same 15,981 rows already returns **SEPARATED** for
`cv − refc_base` (+0.0252 [+0.0150, +0.0349]), **n.s.** for `cv − v1_oracle` (+0.0182 [−0.0019,
+0.0373]), and **SEPARATED at +0.1882** for the G1 sighted-vs-blind gate. Both outcomes are
reachable at this n.

**Block A rules.** `A-CONFIRM`: `Δ(v0on − v0off)` separated-positive at **both** scales ⇒ a trained
speed input measurably reaches a planner's output. `A-NULL`: not separated ⇒ even a planner *with*
the input does not materially use it. Precondition: `refc_xl_v0on` must reproduce **0.5499**; if it
does not, Block A is quarantined and reported as such.

---

## 5. ⛔ GUARDS — each earned by a specific failure

1. **The composite can go UP for a degradation.** A sibling slowed a planner down and the composite
   rose **+0.1698**, because a barely-moving plan is scored `recovery = NaN` **by construction**
   (`pseudosim.py:515` — `rc = NaN where xt_hold <= 0.10`). ⇒ **`v1_ego_half` must score worse. If it
   does not, Block B is VOID and I say so instead of reporting the other arms.**
2. **`recovery` `defined_fraction`** is reported for every arm. If it moves by **> 2 pp** vs the
   control, that arm's `recovery` (and hence its composite) is flagged as
   **not a like-for-like comparison** and is not quoted as a clean win.
3. **Both components are always reported separately.** A composite move that is entirely a
   `recovery`-exclusion artefact is disqualified.
4. **Lateral / longitudinal decomposition is mandatory** (the regression this addresses was 100 %
   longitudinal): `along_track_end_m` and `cross_track_end_m` per arm, plus the 2×2 factorial.
5. **Parity is untouched.** No episode is re-selected; every arm uses the identical 15,981 rows and
   row identity is asserted by `panel_combine.py`, which **refuses** a non-matching arm.
6. **No default that alters a running arm is changed.** `stack/tanitad/models/fourbrain.py`,
   `stack/scripts/train_flagship4b.py` and everything they import are **not modified**; new code
   lands in a new module. pod1 keeps training.
7. `OMP_NUM_THREADS=6` on every process (torch spawns ~113 threads/process; 7 concurrent arms once
   sat at GPU 0–6 % for 50 min looking exactly like a hang).

---

## 6. What I will write if the answer is REFUTE

> The missing ego input is **not** the mechanism behind the closed-loop null. Handing v1's tactical
> plan the speed it never saw — at zero fitted parameters, and even with an **oracle** longitudinal
> schedule — does not produce a separated improvement over its own control on the pseudo-simulation
> composite. The closed-loop null has another cause. **I do not re-scope this into a smaller claim,
> and I do not substitute "better than the previous version of itself" for "better than doing
> nothing".**
