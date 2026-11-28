# SPEC — R10 · the refav1 planner COST SURFACE, decomposed and attributed

**Pre-registered 2026-09-03 (Europe/Berlin), BEFORE the probe was run.**
Stream: Architecture & Inference FlyWheel · branch `agent/arch-inf-20260803`
Register row proposed by this work: `H-REFAV1-COST-FLAT`.
Tier: **T0** for the cost surface itself (it is a WM/cost diagnostic, teacher-anchored at
t0, no closed loop). Any *driving* claim needs the T1 adapter (`taniteval/tools/t1_eval.py`
/ `taniteval/tools/refav1_arm.py`) and is out of scope here.

---

## 0. What is already MEASURED and is NOT re-litigated

| row | the fact | where |
|---|---|---|
| `D-REFAV1-PAIRED-READ-VOID` | refav1 step-1,000's closed-loop plan is the constant-velocity straight line on **140/140** windows under **both** banked checkpoints; `baseline_won_frac` 0.75 | `Project Steering/GOALS_AND_CLAIMS.md` |
| `D-ACTDIV-ANCHORED-REFAV1` | the predictor is NOT lateral-deaf (2,298× fp32 / 11.8× EMA+bf16 over its permutation null, linear 1 : 2.500 : 5.005, antisymmetric cos −0.9998) — **but at matched ≈2σ the lateral channel moves the imagination 1/300 (incumbent) to 1/40 (clean epoch) as far as the longitudinal one** (‖m(κ)‖/‖m(a)‖ 0.00329 / 0.02514) | same register |
| `C-STEER-CURVATURE-INTERFACE` | `actions[:,0]` in the v2ep cache is a road-wheel **STEER ANGLE** (`stack/tanitad/data/physicalai.py:620`) and the predictor is correctly trained on it; the defect is the **planner→model boundary**, which hands TRUE curvature to a steer-trained predictor unconverted | same register + PI ruling |

**L (the encoding wheelbase) is no longer an assumption.** The sibling Data-Engineering
stream landed `stack/tanitad/models/kinematic.py` `STEER_WHEELBASE_M = 2.9` with the
provenance *"MEASURED 2026-09-03 by algebraic inversion against the producer's own input
(the raw egomotion `curvature` column) on 20/20 local eval clips: L_enc = 2.9000000
(spread 3e-8, pointwise IQR 1.2e-7)"*, and the note that the clips' **true** wheelbases
are {2.73, 3.135, 3.165, 3.216} — none of them 2.9 — so a real wheelbase here would
inject an error the encoding never had. **This SPEC therefore uses L = 2.9 as a MEASURED
encoding constant, and still reports the sensitivity over L ∈ {2.73, 2.9, 3.085, 3.216}**,
because that work is not yet staged in the repo at the time of writing (see §7).

---

## 1. The cost's EXACT decomposition — read from source before running

`RefAV1.plan._cost_chunk`, `stack/tanitad/refs/refa_v1.py:1795-1820`. Quoted verbatim by
line:

| # | line | term | weight | units of the squared quantity | sees κ … |
|---|---|---|---|---|---|
| T1 | `:1807-1813` | `1 − cosine_similarity(z_terminal, goal)` | **1.0** (implicit) | dimensionless, ∈ [0, 2] | **THROUGH THE MODEL** (κ enters at `:1803` `augment_actions` → `pred.rollout`) |
| T2 | `:1814-1815` | `0.02 · mean_k((a_{k+1}−a_k)/dt)²` — comfort/jerk | 0.02 | (m/s³)² | not at all (channel 0 only) |
| T3 | `:1816` | `0.05 · mean_k(κ_k²)` — curvature penalty | 0.05 | (1/m)² | **RAW** — the proposed curvature, never the model's |
| T4 | `:1817-1819` | `0.10 · (v_0 + Σa·dt − target_speed)²` | 0.10 | (m/s)² | not at all |

**⚠️ The brief's "four terms" is only nominally right — T4 IS DEAD CODE IN EVERY BANKED
READ.** `target_speed` is `None` on every call: the T1 adapter calls
`model.plan(feats, v0=…, nav_cmd=…, plan_cfg=…, goal_field=…)`
(`taniteval/tools/refav1_arm.py:698-699`) and the token `target_speed` does not appear
anywhere in that file (two probes: `grep -c` on the off-Drive mirror = 0;
`Select-String` on the G: repo copy = 0). **So the live cost is THREE terms, and the
tactical brain's longitudinal decision reaches the cost ONLY through T1.**

**Where the units break.** `PlanConfig.kappa_max = 0.2`, `_clip`
(`refa_v1_plan.py:161-164`), `GOAL_KAPPA_MAX/TURN` (`refa_v1.py:115-116`) and
`unicycle_paths` (`refa_v1_plan.py:291-301`) are all **GEOMETRY (1/m)**. `augment_actions`
(`refa_v1.py:1150-1193`) hands the same tensor to a predictor trained on the **COMMAND**
channel (road-wheel angle, rad). With `speed_channel=false` — which BOTH banked
checkpoints have (`ckpt_ep2/config.json` `"speed_channel": false`; `a_dim: 2`) —
`augment_actions` returns the controls **unchanged** (`:1179-1180`), so the crossing is a
bare pass-through. T3 charges the **proposed** κ; T1 pays back only the benefit of
`arctan(L·κ) ≈ κ/2.9` worth of turn.

**Two structural facts the decomposition forces, both checked before any run:**
1. On a **constant-control** grid, T2 ≡ 0 exactly (jerk is a first difference of a
   constant). The comfort term therefore cannot move the grid minimum. It is still
   reported, and it is exercised by the named non-constant candidates (§3).
2. T1 ≥ 0, T2 ≥ 0, T3 ≥ 0 ⇒ **total ≥ 0, with equality iff the terminal field is exactly
   the goal AND the controls are exactly zero.** A window whose imagined goal is the
   zero-control rollout therefore has its GLOBAL minimum at (a=0, κ=0) *by construction*,
   independent of lateral potency, of the ×2.9 boundary and of the κ² penalty.

---

## 2. The fourth factor this SPEC adds to the Master Mind's three

The brief's reading is (i) intrinsic lateral potency, (ii) the ×2.9 boundary,
(iii) the κ² penalty. Reading `_imagine_tactical_goal` (`refa_v1.py:1647-1684`) and
`canonical_controls` (`:128-164`) against the BANKED decision dumps adds a factor that
must be measured first because it can dominate all three:

**(0) THE IMAGINED GOAL IS ITSELF THE CONSTANT-VELOCITY ROLLOUT ON MOST WINDOWS.**
`plan()`'s default goal is the tactical predictor rolled under `canonical_controls(lat,
lon, v0)`. `LANE_KEEP` ⇒ κ ≡ 0; `CRUISE` ⇒ `v_t = v0` ⇒ a ≡ 0; `ADAPT_SPEED_FOR_CURVE`
⇒ `v_t = min(v0, 8)` ⇒ a ≡ 0 whenever `v0 ≤ 8`. When the canonical controls are all-zero
the goal rollout and the `cv` candidate's rollout are **the same forward pass**, so
T1 = 1 − cos(x, x) = 0 **exactly**.

Counted directly off the banked decision dumps (`C:\Users\Admin\refav1_eval_slice\
t1_dump{,_ep2}\decisions\ep*.npz`, key `goal_lat_cl`/`goal_lon_cl`, vocabulary
`v7.0`) — this is a MEASURED input to the SPEC, not a hypothesis:

| checkpoint | `goal_lat` | `goal_lon` | windows whose canonical controls are exactly zero |
|---|---|---|---|
| incumbent (`ckpt/ckpt.pt`) | `LANE_KEEP` **140/140** | CRUISE 98 / ADAPT 42 | **129 / 140** |
| clean epoch (`ckpt_ep2/ckpt.pt`) | `LANE_KEEP` 115 / **`TURN_R` 25** | CRUISE 98 / ADAPT 42 | **110 / 140** |

and the banked `plan_cost_cl` is `0.0` or ±1e-7 on both — i.e. the winner sits at the
exact floor of a non-negative cost.

⇒ **The discriminating population for factors (i)–(iii) is the windows whose goal is NOT
the zero-control rollout: n = 11 (incumbent, longitudinal-only) and n = 30 (clean epoch,
of which 25 carry a TURN).** This is pre-registered as the primary stratum, with the full
140 reported alongside. n = 25–30 is SMALL and the CI will be wide; that is a finding
about the available evidence, not a licence to quote the point estimate bare.

---

## 3. The instrument

`taniteval/tools/cost_surface_probe.py` (NEW, owned by this stream).

**Population.** The same 140 windows as both banked reads: 20 episodes
(`C:\Users\Admin\refav1_eval_slice\eps`), fp8 cache
(`…\fp8`), `--window-stride 10`, loader `RefAV1Windows` built exactly as
`taniteval/tools/refav1_arm.py:514-523`, model loaded exactly by
`taniteval/tools/refav1_arm.py:280-373` (`load_model`, strict, fitted standardizer
required) — imported BY FILE from the sibling so no loader logic is re-derived.

**Grid.** Constant-over-horizon candidates at the planner's own horizon `H = 10`
(`cfg.plan_steps`, 2.0 s at `op_dt = 0.2`):
`κ ∈ linspace(−0.2, +0.2, 21)` (`PlanConfig.kappa_max`), `a ∈ linspace(−4, +4, 9)`
(`PlanConfig.a_max`) ⇒ 189 grid candidates. Plus the NAMED candidates
`cv`/`hold_v0` (zeros), `decel_1.5`, `proposal` (the imitation head's plan) and
`goal_canonical` (`canonical_controls[:H]` — the seed `plan()` injects at
`refa_v1.py:1790-1793`), which are NOT constant and therefore exercise T2.

**Every term is recorded separately** — `c_goal`, `c_jerk`, `c_kappa`, `c_total` — for
every candidate, so every ablation below is post-hoc arithmetic on ONE set of rollouts
(no second forward pass, and no possibility of the arms disagreeing on the rollout).

**Two conventions, differing ONLY at the model boundary:**

* **A — as shipped.** `controls` handed to `augment_actions` unconverted.
* **B — repaired.** `steer = arctan(L · κ)` applied to channel 1 **immediately before**
  `augment_actions` and nowhere else. The search space, `_clip`, T3's κ² and any
  integrated path stay in curvature. This is exactly
  `kinematic.as_command(controls, "steer")` (the sibling stream's helper); the probe
  implements the same one-line map itself so the measurement does not depend on an
  unstaged file, and asserts the two agree to 1e-7 when the helper is importable.

**The attribution panel (4 arms × 2 checkpoints × 140 windows):**

| arm | conversion | T3 (0.05 κ²) | isolates |
|---|---|---|---|
| `A_full` | none (shipped) | on | the shipped surface |
| `A_nopen` | none | **off** | factor (iii) alone |
| `B_full` | `arctan(L·κ)` | on | factor (ii) alone |
| `B_nopen` | `arctan(L·κ)` | off | (ii) + (iii) together |

Reported per arm: `argmin (a*, κ*)`, `|κ*|`, `turn_frac` = fraction of windows with
`|κ*| ≥ 0.04` (= ½·`GOAL_KAPPA_TURN`), the T1 range across the κ axis at `a = a*`
(`ΔT1_kappa` — the *modelled benefit available* to a turn) and the T3 charge at the
goal's own κ (`0.05·κ²`). Factor shares are defined in §5.

---

## 4. CONTROLS THAT MUST READ KNOWN VALUES — pre-registered, panel is VOID if any fails

| id | control | the KNOWN value it must read | what it would catch |
|---|---|---|---|
| **C0** | **decomposition identity** | `c_goal + c_jerk + c_kappa == c_total`, max abs error ≤ 1e-6 | a term recorded from a different rollout than the total |
| **C1** | **harness gate — the shipped `_cost_chunk`** | the probe's re-scoring of `cv`/`hold_v0`/`decel_1.5`/`proposal` must equal `PlanResult.baseline_costs` from a REAL `model.plan()` call on the same window, **relative error ≤ 1e-6** | a re-implementation that is not the shipped cost |
| **C2** | **harness gate — the banked winner** | on ≥ 3 windows whose banked winner is known from `t1_dump\ep*.npz`, a real `model.plan()` re-run must reproduce the banked `cl` trajectory to **< 3.8e-6 m** max abs (the previous agent's gate) | a stale checkpoint, a drifted loader, a different window grid |
| **C3** | **ZERO-MODEL control** | a predictor wrapper that feeds the ZERO action for every candidate ⇒ `c_goal` must be **exactly constant** across the whole grid (variance == 0.0), so `total(κ) − total(0) == 0.05·κ²` **exactly** and `argmin` is at κ = 0 to machine precision | a surface whose κ-dependence is an artifact of anything but the model; **it also measures T3's contribution exactly, by construction** |
| **C4** | **CONSTANT-COST control** | all three weights set to 0 ⇒ `c_total ≡ 0` for every candidate and `argmin` is the first grid index (deterministic tie) | a probe that manufactures structure from its own bookkeeping |
| **C5** | **unit-invariance of channel 0** | A and B must give **identical** `c_jerk` and identical `a`-marginal at κ = 0 (max abs diff 0.0) | a conversion that leaks into the longitudinal channel |
| **C6** | **n and d printed** | every table carries its `n` (windows) and the grid size | an underpowered read presented as a null |

**A failed control does not get "explained". It voids the panel.**

---

## 5. The verdict this must produce, with BOTH outcomes committed in advance

Define, per window, `κ*_arm` = the argmin curvature of arm `arm`, and
`turn(arm) = 1[|κ*_arm| ≥ 0.04]`.

**Factor shares** are reported as the change in `turn_frac` produced by each single
repair, on the discriminating stratum and on all 140:

* share(iii) = `turn_frac(A_nopen) − turn_frac(A_full)` — the κ² penalty alone.
* share(ii) = `turn_frac(B_full) − turn_frac(A_full)` — the ×2.9 boundary alone.
* share(ii+iii) = `turn_frac(B_nopen) − turn_frac(A_full)` — both.
* the **residual** `1 − turn_frac(B_nopen)` is attributed to factor (i) (intrinsic lateral
  potency) **plus** factor (0) (a degenerate goal), and the two are separated by the
  stratum: on the zero-canonical-goal stratum the residual is factor (0) by construction
  (§1.2), on the TURN stratum it is factor (i).

Intervals: **paired episode-cluster bootstrap** over the 20 episodes
(`taniteval/taniteval/ci.py`), 10,000 resamples, paired on the window. ⛔ Never
`overlapping_holdout_se`.

**Pre-registered outcomes:**

| factor | outcome X (it matters) | outcome Y (it does not) |
|---|---|---|
| **(0) degenerate goal** | ≥ 50 % of windows have an exactly-zero canonical goal ⇒ the flat plan is EXACTLY OPTIMAL there and the repair target is the TACTICAL HEAD, not the cost | < 50 % ⇒ the goal is live and (i)–(iii) carry the phenomenon |
| **(ii) ×2.9 boundary** | `share(ii) > 0` with a CI excluding 0 on the discriminating stratum ⇒ the one-line `arctan(L·κ)` conversion is the cheapest repair | CI includes 0 ⇒ the conversion alone does not make the planner turn, and must not be sold as the fix |
| **(iii) κ² penalty** | `share(iii) > 0`, CI excluding 0 ⇒ re-weight or drop T3 | CI includes 0 ⇒ **T3 is NOT the binding constraint** and removing it is cosmetic |
| **(i) lateral potency** | residual after B_nopen on the TURN stratum > 0.5 ⇒ the cost cannot see a turn because the imagination barely moves; the repair is a training-side one (an action-conditioning loss), not a planner one | residual ≤ 0.5 ⇒ potency is sufficient once (ii)/(iii) are repaired |

**A "cheapest repair" claim is admissible only if** `turn_frac` rises on the stratum where
the HUMAN turns. The human's turn is read from the banked ground-truth path `g`
(`t1_dump\ep*.npz`) as `|Δheading| over the 2 s horizon`; the "human turns" stratum is
pre-registered as windows with GT heading change ≥ 5° over 2.0 s.

---

## 6. What this SPEC will NOT claim

* No driving claim. The surface is **T0**. `turn_frac` is a property of the cost, not of
  the car. A rise in `turn_frac` is a necessary, not sufficient, condition for better
  driving, and the T1 adapter is the only instrument that can close that.
* No re-ranking of the two checkpoints. They are reported side by side.
* No edit to `refa_v1.py`, `refa_v1_plan.py` or `refav1_arm.py` — all read-only to this
  stream. Any repair is delivered as a **PROPOSED diff** in RESULT.md.

## 7. Environment, and a concurrency hazard recorded up front

* Device: dev-box **RTX 4060 (8,188 MiB)**, checked free of python compute apps before and
  during. Thor and the A40 are NOT contacted (refav1's speed-channel epoch and refcv3 are
  training on them).
* The stack cannot run from the G: mount; the run uses the off-Drive mirror
  `C:\Users\Admin\tanitad-wt` with
  `PYTHONPATH=C:/Users/Admin/tanitad-wt/stack;…/colab;…/taniteval`, venv
  `C:\Users\Admin\venvs\tanitad\Scripts\python.exe`.
* ⚠️ **The mirror is AHEAD of the repo on three files this stream reads.** At 06:43–06:45
  today a sibling stream edited `stack/tanitad/models/kinematic.py` (new: the
  command↔geometry bridge, `STEER_WHEELBASE_M`), `stack/tanitad/refs/refa_v1.py` (new:
  `plan(..., model_action_units=…)`), `stack/tanitad/refs/refa_v1_plan.py` and
  `taniteval/tools/refav1_arm.py` in the mirror; **none of it is in the repo working tree**
  (md5s differ; repo `refa_v1.py` = `cdc25592…`, mirror = `c26ea573…`). The new
  `model_action_units` argument DEFAULTS to `"kappa"`, which the sibling documents as
  byte-identical to the legacy path, so the banked reads remain reproducible — but this
  probe must not silently measure a different file than it reports. **Mitigation, binding:
  the probe records the md5 of every source file it imports into its own raw JSON**, and
  RESULT.md quotes those md5s. This is escalated in the final report.
