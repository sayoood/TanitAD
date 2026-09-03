# RESULT — the refav1 kinematic contract is broken by a UNIT, and the unit is a STEERING ANGLE

**Package:** `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refav1-kinematic-contract-lateral/`
**SPEC:** `SPEC.md` in this directory — committed to the repo BEFORE the panel was run.
**Run:** 2026-09-03, dev box, **0 GPU, CPU only**. ⛔ Thor was never contacted; no checkpoint was re-read.
**Evidence class:** MEASURED — `raw/kin_contract.json`, `raw/kin_contract_probe.log`, `raw/per_window_hypotheses.csv` (140 rows), `raw/per_window_dump_arms.csv` (140 rows).
**Tool:** `taniteval/tools/refav1_kin_contract_probe.py` (new, staged).
**n:** 140 windows over 20 episodes; 72 curved (`max_k |y_gt| >= 0.3 m`) / 68 straight.
**Estimator:** full-set pooled means over windows; 95 % intervals = episode-cluster bootstrap (`taniteval/ci.py`, 2 000 resamples, 20 episode clusters); cross-hypothesis deltas = the **paired** bootstrap on the same windows. `overlapping_holdout_se` never used.

---

## ⛔ HEADLINE — THIS NEEDS A DECISION, NOT A MERGE

`stack/tanitad/data/physicalai.py:620-630` (`signals_at`) writes

```python
curv  = col("curvature")
steer = np.arctan(float(wheelbase) * curv)      # road-wheel angle proxy [rad]
actions = np.column_stack([steer, accel])
```

so **`v2ep actions[:, 0]` is a STEERING ANGLE `atan(L·kappa)`, not a curvature.** `refav1_loader.py:264`
feeds that channel to the model and to every unicycle integration **as `kappa`**, inflating every
curvature by **~L = 2.9x**. The panel below settles it in one variable.

**THE BLAST RADIUS IS NOT THE EVAL INSTRUMENT. THERE ARE THREE INCOMPATIBLE UNIT CONVENTIONS LIVE
AT ONCE, AND ONE OF THEM IS THE TRAINING INPUT OF THE RUN ON THOR RIGHT NOW:**

| where | what it assumes `kappa` is | consequence |
|---|---|---|
| `refav1_loader.py:264` -> `RefAV1` action input | steer, labelled kappa | the model's whole action distribution is 2.9x-scaled steer. **Thor is training on this now.** |
| `kinematic.rollout_unicycle:181` (`yaw += v*kappa*dt`), reached by `unicycle_paths` | true curvature [1/m] | **every trajectory built from a model action over-rotates by ~2.9x** — the `ol`/`ha` arms today, and every `cl` arm the moment the planner moves kappa off 0 |
| `refa_v1.canonical_controls:158-163` (`GOAL_KAPPA_TURN = 0.08` 1/m, R 12.5 m) | true curvature [1/m] | the tactical proposals are minted in curvature and fed into an action space the model learned as steer: a TURN proposal is **under-actuated ~2.9x** (0.08 rad of steer = kappa 0.0277, **R 36 m**, not 12.5 m). `PlanConfig.kappa_max = 0.2` likewise bounds *steer*, so the search cannot reach the curvature it believes it commands (true kappa <= 0.069, R 14.5 m). |

⇒ **This is a PI/Master-Mind decision, not a patch to land quietly.** The repair changes the training
input distribution of a live run. Nothing is applied in this package — the diff is PROPOSED in §6.

---

## 1. Reproduction (task 1) — the four amendment numbers reproduce EXACTLY, 0 GPU

`refav1_kin_contract_probe.py` stage A, on the banked dumps alone. Per-window rows:
`raw/per_window_dump_arms.csv`.

| arm | straight plans | const-speed plans | LAT curved (n 72) | LAT straight (n 68) | LON all (n 140) |
|---|---|---|---|---|---|
| `cl` | 140/140 | 140/140 | 0.4960 | 0.0350 | 0.3768 |
| `ha` | 3/140 | 1/140 | 0.7625 | 0.1203 | 0.3175 |
| **`ol`** | 1/140 | 1/140 | **0.7156** | **0.0685** | **0.2395** |
| `cl_navshuf` | 140/140 | 122/140 | 0.4960 | 0.0350 | 0.4839 |
| `cl_oraclegoal` | 140/140 | 96/140 | 0.4960 | 0.0350 | 0.5685 |
| the `y == 0` line | 140/140 | 140/140 | **0.4960** | **0.0350** | — |

Amendment: 0.716 / 0.496 / 0.069 / 0.035 and `ha` 0.763 / 0.120. **All six reproduce to 4 dp.**
The amendment's 0.24 m along-track figure is `ol`'s LON over all 140 windows: **0.2395**.

## 2. Harness checks — the panel is gated on them (SPEC §4)

| control | must read | read |
|---|---|---|
| my replay of the SHIPPED path vs the banked `ol` | < 1e-4 m | **3.81e-6 m** |
| my recomputed GT vs the banked `g` | < 1e-4 m | **0.0 m (exact)** |
| my numpy integrator vs `kinematic.rollout_unicycle` | < 1e-4 m, every window | verified per window |
| `ref_line` reproduces the dump's straight line | 0.4960 / 0.0350 | 0.4960 / 0.0350 |
| `ref_floor` on human-STRAIGHT windows | ~0 | **0.0096 m** |

The probe **refuses to print a panel** if the first two fail (`refav1_kin_contract_probe.py`, the
`harness check FAILED` exit). A replay that does not reproduce the arm it claims to replay is not a
replay, and every number under it would be about my re-implementation.

## 3. The one-variable panel (task 2)

The exact path replayed: `refav1_loader.py:258-264` `_kin_actions` (called on the REAL loader object)
-> `refav1_arm.py:606` -> `refav1_arm.py:383-397` `paths_from_controls` -> `refa_v1_plan.py:291-301`
`unicycle_paths` -> `kinematic.py:176-184` `rollout_unicycle`. Bar to beat = the straight line's
**0.4960 m**. Per-window rows: `raw/per_window_hypotheses.csv`.

| id | one variable | LAT curved | 95 % CI | paired vs shipped | LAT straight | LON all | verdict |
|---|---|---|---|---|---|---|---|
| `ref_ol` | none (shipped) | 0.7156 | [0.4804, 0.9405] | — | 0.0685 | 0.2395 | control |
| `ref_line` | a=0, k=0 | 0.4960 | [0.3439, 0.6572] | −0.2196 [−0.3331, −0.1115] **sep** | 0.0350 | 0.3768 | control (the bar) |
| `ref_floor` | true pose yaw + speed, no action channel | **0.0527** | [0.0373, 0.0698] | −0.6630 [−0.8784, −0.4394] **sep** | 0.0096 | 0.1189 | control (the FLOOR) |
| `ref_kapose` | kappa from the pose yaw, `a` as shipped | 0.0527 | [0.0373, 0.0698] | −0.6630 [−0.8784, −0.4394] **sep** | 0.0096 | 0.1189 | control (the contract's ceiling) |
| `ref_floor_mid` | the floor under MIDPOINT quadrature | 0.0325 | [0.0134, 0.0567] | −0.6831 [−0.8882, −0.4654] **sep** | 0.0107 | 0.1217 | control |
| **(a)** `h_a_sign` | kappa -> −kappa | 1.7052 | [1.1703, 2.2375] | +0.9896 [+0.6852, +1.3121] **sep** | 0.1306 | 0.2395 | REFUTED (much worse) |
| **(b)** `h_b1_shift_m1` | kappa = kap[2j−2] | 0.7258 | [0.4898, 0.9529] | +0.0102 [−0.0188, +0.0387] | 0.0839 | 0.2413 | REFUTED (not separated) |
| **(b)** `h_b2_shift_p1` | kappa = kap[2j+2] | 0.7166 | [0.4811, 0.9325] | +0.0009 [−0.0274, +0.0318] | 0.0570 | 0.2372 | REFUTED |
| **(b)** `h_b3_midframe` | kappa = kap[2j+1] | 0.7144 | [0.4808, 0.9335] | −0.0012 [−0.0161, +0.0149] | 0.0616 | 0.2384 | REFUTED |
| **(c)** `h_c0_half` | kappa/2 (per-frame vs per-second) | 0.1542 | [0.0969, 0.2121] | −0.5615 [−0.7269, −0.3833] **sep** | 0.0226 | 0.1375 | partial — **wrong factor** (§3.1) |
| **(c)** ⭐ `h_c1_wb` | **kappa = tan(kap[2j]) / 2.9** | **0.0600** | **[0.0427, 0.0800]** | **−0.6556 [−0.8701, −0.4349] sep** | **0.0141** | **0.1199** | ✅ **RESOLVES** |
| **(c)** `h_c2_wbfit` | kappa = tan(kap)/L̂_ep (ORACLE) | 0.0607 | [0.0424, 0.0814] | −0.6549 [−0.8664, −0.4338] **sep** | 0.0141 | 0.1194 | no gain over 2.9 (§5) |
| **(d)** `h_d1_yawfirst` | yaw before the position step | 1.0828 | [0.7360, 1.4114] | +0.3671 [+0.2498, +0.4761] **sep** | 0.0965 | 0.2972 | REFUTED (worse) |
| **(d)** `h_d2_midpoint` | position on the half-updated yaw | 0.9008 | [0.6075, 1.1787] | +0.1851 [+0.1261, +0.2408] **sep** | 0.0823 | 0.2665 | REFUTED (worse) |
| combination (labelled) | `h_c1_wb` + `h_d2_midpoint` | 0.0433 | [0.0220, 0.0673] | −0.6723 [−0.8759, −0.4566] **sep** | 0.0152 | 0.1224 | ✅ (its own floor is `ref_floor_mid` 0.0325) |

**(e) THE FRAME is not the cause.** `h_e1` (GT `y -> −y`): 1.7052 m — flipping the sign makes it
2.4x worse, so `y` is positive LEFT in both the GT and the integrator, as `kinematic.py:148-150`
says. `h_e2` (GT in the t0 **velocity** frame instead of the t0 **heading** frame): 0.7518 m LAT /
0.2419 m LON against the shipped 0.7156 / 0.2395 — a +0.036 m change, inside the committed
±0.04 band. The mean |heading − velocity-direction| angle at t0 is **0.236°**: the two frames are
the same frame at these speeds.

**(f) SLIP — the decisive test, and it says the miss IS kappa.** Yaw accumulated over the 10 steps,
integrated `sum kappa·v·dt` divided by the pose yaw change, median over the **75** windows that
actually turn (`|dyaw_pose| > 0.02 rad`):

| kappa read as | ratio integrated / pose | yaw RMSE vs the pose (n 140) |
|---|---|---|
| shipped | **x2.870** | **19.81°** |
| `h_c0_half` (kappa/2) | x1.435 | 4.77° |
| `h_c1_wb` (tan/2.9) | **x0.995** | **0.632°** |

The shipped channel over-rotates by **2.87x** — the wheelbase, measured off the trajectories and
never assumed. Under the repair the integrated yaw and the human's own pose yaw agree to
**0.6°**, so there is no residual slip term to attribute the miss to: **the miss was the unit.**

### 3.1 The committed pass/fail rule was TOO LOOSE — reported, not quietly tightened

SPEC §3 committed: RESOLVES iff curved LAT < 0.496 **and** LON <= 0.26 **and** straight LAT <= 0.069.
**`h_c0_half` also passes that rule** (0.1542 / 0.1375 / 0.0226) — and it is the WRONG repair. The
rule alone cannot discriminate a factor of 2 from a factor of 2.9. What discriminates them are the
two controls the SPEC also committed, and they discriminate cleanly:

* against the **FLOOR** (`ref_floor` 0.0527): `h_c1_wb` sits **0.0073 m above it**; `h_c0_half` sits
  **0.1015 m above it — 2.9x the floor**.
* against the **yaw ratio**: `h_c1_wb` x0.995; `h_c0_half` x1.435, i.e. still 43 % over-rotating.

⇒ The verdict is `h_c1_wb`, and it is the verdict because of the CONTROLS, not because of the
threshold. Recorded here because a pass rule that admits a wrong answer is itself a finding: **a
threshold without a floor is not a test.** (CLAUDE.md's probe rule, in a new costume.)

### 3.2 Prediction vs outcome (SPEC §3, scored)

| id | committed | measured | held? |
|---|---|---|---|
| `ref_ol` | 0.716 | 0.7156 | ✅ |
| `ref_line` | 0.496 | 0.4960 | ✅ |
| `ref_floor` | < 0.10 | 0.0527 | ✅ |
| `ref_kapose` | < 0.15 | 0.0527 | ✅ |
| `h_a_sign` | > 1.0 | 1.7052 | ✅ |
| `h_b1/b2/b3` | 0.70 ± 0.06 / 0.73 ± 0.06 / 0.71 ± 0.03 | 0.7258 / 0.7166 / 0.7144 | ✅ |
| `h_c0_half` | 0.30 – 0.45 | **0.1542** | ❌ **too pessimistic** — a 2x cut of a 2.87x gain leaves 1.43x, and at these curvatures that is only 0.154 m, not 0.3–0.45. My arithmetic was wrong, not the data. |
| `h_c1_wb` | 0.10 – 0.25 | **0.0600** | ✅ (better than committed) |
| `h_c2_wbfit` | <= `h_c1` | 0.0607 (≈ equal) | ~ |
| `h_d1/d2` | 0.70 ± 0.10 / 0.71 ± 0.06 | **1.0828 / 0.9008** | ❌ both WORSE than committed — with a 2.87x-inflated kappa a "better" quadrature integrates a wrong yaw rate more faithfully. The conclusion (order cannot fix a gain) stands, but the committed magnitudes were wrong. |
| `h_e1` | > 1.0 | 1.7052 | ✅ |
| `h_e2` | 0.716 ± 0.04 | 0.7518 | ✅ (+0.036) |
| `h_f` | shipped ~2.9x; repaired < 15 % | x2.870; **0.5 %** | ✅ |

Three committed magnitudes were wrong (`h_c0`, `h_d1`, `h_d2`); no committed VERDICT was wrong.

## 4. What this changes in the shipped arms

**`ha` was not a weak control — it was the SAME defect.** Recomputing the hold-action arm with the
repaired channel, same windows, same integrator (`raw/` reproduction script logic; 0 GPU):

| arm | LAT curved (72) | LAT straight (68) | LON all (140) |
|---|---|---|---|
| `ha` as shipped | 0.7625 | 0.1203 | 0.3175 |
| `ha` REPAIRED (`tan/2.9`) | **0.1659** | **0.0330** | **0.1984** |
| `ha0` (constant velocity) | 0.4960 | 0.0350 | 0.3768 |

⇒ The amendment's *"the hold-action arm holds an observed kappa that drifts 0.12 m even on straight
roads — a control weaker than a constant-velocity baseline"* is **true of the shipped instrument and
false of hold-action control**. Repaired, `ha` beats `ha0` on both channels — it is a *strong*
baseline, and the echo test gets harder, not easier. The `ha0` arm (§7) is still required: it is the
arm that is trivial *by construction* and cannot be broken by a channel defect at all.

## 5. The wheelbase, and what stays UNVERIFIED

Least-squares `tan(steer) ~ L·kappa_pose` per episode (`kappa_pose` from `poses[:,2]`, `v > 3 m/s`),
n 20 episodes — **an ORACLE fit on the scored slice, reported as a diagnostic, never as the repair**:

* r >= 0.999 on 8 episodes; those cluster at **2.834 – 2.879** (five) and **3.084 / 3.108** (two).
* Whole set: min 1.782 (r 0.61, a near-straight episode — the fit is unidentifiable there),
  max 3.108, median **2.855**. Corpus population (`.../2026-07-26-wheelbase-impact/wheelbase_population.json`):
  {2.730, 2.850, 3.135, 3.165, 3.216}.

⚠️ **UNVERIFIED — which build regime minted this corpus.** `physicalai.DEFAULT_WHEELBASE_MODE` is
`const2p9` (L = 2.9 for every clip), but **no tight fit lands on 2.9** — they land at ~2.85 and
~3.09, which looks like `per_clip_v1`. It could equally be a systematic offset between the two
sensor channels the fit compares (quaternion yaw vs the dataset's `curvature` column), so the two
readings are not separable here. **The panel cannot settle it and does not need to:** `h_c2_wbfit`
(per-episode L̂) reads **0.0607** against `h_c1_wb`'s **0.0600** — the per-clip refinement is worth
**< 1 cm** on this slice, below the slip floor. The dataset's own
`calibration/vehicle_dimensions` is not on this box (`$TANITAD_PAI_WHEELBASE` unset), so this is a
**WORK ITEM for the Data FlyWheel**, not a blocker on the repair.

## 6. THE FIX — **PROPOSED, NOT APPLIED** (Thor is training on this loader)

⛔ Nothing in this package touches `refav1_loader.py`. The diff below is the recommendation.

```diff
--- a/stack/tanitad/data/refav1_loader.py
+++ b/stack/tanitad/data/refav1_loader.py
@@ ACTIONS — (a, kappa), and the channel order is MEASURED, not assumed:
-  v2ep `actions[:, 0]` correlates r = 0.995 with pose-derived curvature and
-  `actions[:, 1]` only r = 0.47 with pose-derived accel (6 episodes,
-  2026-09-01) ⇒ the STORED order is (kappa, accel-like) — the REVERSE of
-  RefAV1Config's `a_dim: (a, kappa)`. This loader emits
+  v2ep `actions[:, 0]` correlates r = 0.995 with pose-derived curvature and
+  `actions[:, 1]` only r = 0.47 with pose-derived accel (6 episodes,
+  2026-09-01) ⇒ the STORED order is (kappa-like, accel-like) — the REVERSE of
+  RefAV1Config's `a_dim: (a, kappa)`.
+  ⛔ AND THE CHANNEL IS NOT A CURVATURE. `physicalai.signals_at` (physicalai.py:
+  620-630) writes `steer = atan(L * curvature)` — a ROAD-WHEEL ANGLE. The
+  correlation above is r ~ 1 and CANNOT see that, because a correlation is
+  scale-invariant; the gain is L ~ 2.9. MEASURED 2026-09-03 (140 windows, 20
+  eps): reading it as kappa over-rotates by x2.870 and misses the human's
+  lateral position by 0.716 m on curved windows, WORSE than a straight line's
+  0.496 m. Inverting it (kappa = tan(steer)/L) reads 0.060 m against a
+  0.053 m pose-yaw floor, and the along-track error HALVES (0.240 -> 0.120 m).
+  This loader emits
       a      = (v[2(j+1)] - v[2j]) / 0.2      # poses ch3, exact on the grid
-      kappa  = actions[2j, 0]                  # the measured true-kappa channel
+      kappa  = tan(actions[2j, 0]) / wheelbase # steer -> curvature (SEE ABOVE)
   so a silent channel swap cannot reach the model.
@@ class RefAV1Windows.__init__
                  lru: int = 32, seed: int = 0,
+                 wheelbase: float | None = None,   # None = physicalai.WHEELBASE
+                 steer_channel: bool = True,       # False = legacy (kappa-as-stored)
@@ def _kin_actions
     def _kin_actions(self, v: Tensor, kap: Tensor, j0: int, n: int) -> Tensor:
         idx = torch.arange(j0, j0 + n)
         f = idx * 2
         f_next = torch.clamp((idx + 1) * 2, max=v.shape[0] - 1)
         a = (v[f_next] - v[f]) / self.dt
-        return torch.stack([a, kap[f]], dim=-1)       # ⭐ (a, kappa) — swapped
+        k = (torch.tan(kap[f]) / self.wheelbase) if self.steer_channel else kap[f]
+        return torch.stack([a, k], dim=-1)            # ⭐ (a, kappa) — swapped
```

**`steer_channel` defaults True and the legacy path stays reachable** because every refav1
checkpoint that exists was trained on the un-inverted channel; a run resumed under the new default
is a DIFFERENT EXPERIMENT and must say so in its config. The flag is the honest way to make that
visible instead of silently re-scaling a live run's inputs.

**Two more sites move with it, and they are the reason this is a decision:**

1. `refa_v1.canonical_controls` (`refa_v1.py:128-164`) and `GOAL_KAPPA_*` are already in TRUE
   curvature — under the repair they become CORRECT for the first time. Under the legacy path they
   stay 2.9x under-actuated. **They cannot both be right.**
2. `PlanConfig.kappa_max = 0.2` (`refa_v1_plan.py:87`) is documented as 1/m and is only 1/m after
   the repair.

**Eval-only stopgap (adapter, no retrain):** `ol` and `ha` are pure kinematic replays of *recorded*
actions and can be repaired in `refav1_arm.py` alone, restoring the kinematic-contract control
immediately. **It does NOT repair `cl`**: the model's own planned actions live in steer units and
`unicycle_paths` integrates them as curvature, so every T1 trajectory over-rotates by ~2.9x the
moment the planner moves kappa off 0. At step 1,000 `cl` is kappa ≡ 0 on 140/140 windows, so the
defect is currently invisible on that arm — **it will appear as soon as the planner starts steering.**
I have NOT applied the stopgap either: shipping a repaired `ol` beside an unrepaired `cl` would put
two unit conventions in one table, which is how this started.

## 7. Task 3 — the `ha0` control arm and the trivial-profile instrument (SHIPPED, staged)

Added to `taniteval/tools/refav1_arm.py`:

* **`ha0`** — `a = 0, kappa = 0` at the measured `v0`: a constant-velocity STRAIGHT line, tier
  **T1**, in the default arm list beside `ha`, with its `arm_meaning`, the four family rows, and the
  paired **`cl − ha0`** block (`paired_cl_minus_ha0`). It consumes strictly less than `ha` — not even
  the last observed action — and it cannot be broken by a channel defect, which §4 shows `ha` can.
  `hold_v0_controls()` carries the rule; the manifest banks it as `hold_v0_rule`.
* **The trivial-profile instrument** (`trivial_profile` / `_print_trivial_profile`) — printed
  **BEFORE any family row** and banked first in the record (`rec['refav1']['trivial_profile']`):
  per arm the fraction of windows that are straight, constant-speed, both (the constant-velocity
  profile), and which other arms it is **bit-identical** to.

**It works. Re-analysing the banked step-1,000 dump (0 GPU, `--analyze-only`) prints, before any
metric:**

```
[trivial-profile] 140 windows — arm SHAPE before any family row
  cl             n= 140 straight=1.0000 const_speed=1.0000 CONSTANT-VELOCITY=1.0000  identical_to: cl_navshuf=122/140, cl_oraclegoal=96/140, ha=1/140, ol=1/140
  ha             n= 140 straight=0.0214 const_speed=0.0071 CONSTANT-VELOCITY=0.0071  ...
  ol             n= 140 straight=0.0071 const_speed=0.0071 CONSTANT-VELOCITY=0.0071  ...
  cl_navshuf     n= 140 straight=1.0000 const_speed=0.8714 CONSTANT-VELOCITY=0.8714  identical_to: cl=122/140, ...
  cl_oraclegoal  n= 140 straight=1.0000 const_speed=0.6857 CONSTANT-VELOCITY=0.6857  identical_to: cl=96/140, ...
  ⚠️ CONSTANT-VELOCITY on > 50 % of windows: ['cl', 'cl_navshuf', 'cl_oraclegoal'] — read their LATERAL rows as a control, never as planning skill
```

That is the void read, named in six lines, before a single family number is computed.
(Full record: `raw/reanalyze_step1000_with_instrument.json`, log `raw/reanalyze_step1000_with_instrument.log`.)

**Tests** — `stack/tests/test_refav1_kin_contract.py`, **26 passed**, CPU, random-init RefAV1,
3 synthetic windows:

| id | pins |
|---|---|
| A1 | `ha0` is straight AND constant-speed on **every** window, and its step length is exactly `v0·dt` |
| A2 | `ha` is NOT straight where the observed `kappa != 0`, and differs from `ha0` there |
| A3 | the instrument reads `straight_frac = const_speed_frac = trivial_frac = 1.0` for `ha0` and flags it degenerate |
| A4 | two arms that ARE equal are reported as equal (`identical_to` frac 1.0); one that differs is not listed |
| A5 | `ha0` reaches the record as **T1** with an `arm_meaning`, four family rows, and `paired_cl_minus_ha0` |
| A6 | no existing arm moved: `ol`, `ha`, `g`, `v0` are each REDERIVED from their own documented rule and matched bit for bit; the pre-existing tier map and paired blocks are unchanged |
| B1 | with a TRUE-curvature channel the contract HOLDS (< 0.15 m on the fixture's own 10 Hz→5 Hz floor, and < 0.4x the straight line) |
| B2 | **deliberate regression:** with the `atan(L·kappa)` channel the SAME replay is worse than a straight line (> 1.2x) and > 5x the true-channel replay — it must fail, or the probe could not have found the defect |
| B3 | `kappa = tan(steer)/L` restores it exactly (lands ON the true-channel value to 1e-5) and does not worsen the along-track channel |
| B4 | the defect is INVISIBLE to a correlation (r > 0.999) while the regression SLOPE reads ~L — why `refav1_loader.py:17-24`'s "r = 0.995" could not catch it |
| B5 | the probe's numpy integrator reproduces `rollout_unicycle` (< 1e-4 m) — it is a re-expression, not a second convention |

Full refav1 suite after the change: **69 passed** (`test_refav1_arm.py`, `test_refav1_kin_contract.py`,
`test_refav1_loader.py`, `test_refav1_loader_labels.py`, `test_refav1_lead_block.py`).

⚠️ **ONE INTEGRATION ITEM, and it is a one-liner outside my ownership:** `t1_eval.DEFAULT_TIERS`
(`t1_eval.py:145`) does not know `ha0`, so the **standalone** `t1_eval.py --analyze-only` CLI
REFUSES a refav1 dump containing it (`t1_eval.py:307` — and that guard is correct). `refav1_arm.py`'s
own analysis path is unaffected (`ha0` is in `ARM_TIERS`). Workaround in place: `--tiers ha0=T1`,
which is what `test_refav1_arm.py` now passes. **The durable fix is `"ha0": "T1"` in
`DEFAULT_TIERS`** — I did not make it because `t1_eval.py` is read-only for this task.

## 8. PROPOSED REGISTER ROWS (for `Project Steering/GOALS_AND_CLAIMS.md`)

```
C-REFAV1-KIN-CONTRACT-LAT — RESOLVED (2026-09-03, Benchmarks & Evals FlyWheel)
  Claim: the refav1 kinematic contract ("the recorded (a, kappa) must reproduce GT")
  fails LATERALLY on the banked step-1,000 slice — |y_ol - y_gt| 0.716 m on 72 curved
  windows vs a straight line's 0.496 m.
  RESOLUTION — the cause is a UNIT, in ONE variable. `physicalai.signals_at`
  (physicalai.py:620-630) stores `steer = atan(L*kappa)`, a ROAD-WHEEL ANGLE;
  `refav1_loader.py:264` feeds it as `kappa`, inflating every curvature by ~2.9x.
  MEASURED (140 windows / 20 eps, 0 GPU, taniteval/tools/refav1_kin_contract_probe.py):
    kappa = tan(steer)/2.9  ->  lateral 0.716 -> 0.0600 m [0.0427, 0.0800],
      paired -0.6556 [-0.8701, -0.4349] (separated), against a pose-yaw FLOOR of
      0.0527 m; along-track 0.2395 -> 0.1199 m (improves); straight windows
      0.0685 -> 0.0141 m. Integrated-vs-pose yaw ratio x2.870 -> x0.995,
      yaw RMSE 19.81 deg -> 0.63 deg (n 75 turning windows).
    REFUTED in the same panel: sign flip (1.7052), one-cache-step timing in either
      direction (0.7258 / 0.7166, intervals cross 0), mid-frame (0.7144),
      integration order (yaw-first 1.0828, midpoint 0.9008 — both worse),
      GT frame (heading vs velocity 0.236 deg apart; y-sign flip 1.7052),
      slip (no residual: the repaired yaw agrees to 0.6 deg).
    A naive /2 ("per-frame vs per-second") also clears the committed threshold but
      leaves x1.435 over-rotation and sits 2.9x above the floor — the FLOOR control,
      not the threshold, is what separates them.
  BLAST RADIUS (escalated, NOT patched): the same channel is (i) the live Thor run's
  TRAINING input, (ii) integrated as curvature by kinematic.rollout_unicycle for every
  arm's trajectory, and (iii) inconsistent with refa_v1.canonical_controls / GOAL_KAPPA_*
  / PlanConfig.kappa_max, which are in true 1/m — so the tactical proposals are ~2.9x
  under-actuated. THREE conventions live at once. Repair is PROPOSED (RESULT.md §6),
  not applied; it changes a live run's input distribution and needs a PI/Master-Mind call.
  CONSEQUENT RETRACTION: the amendment's "hold-action is a control weaker than
  constant velocity" is an artefact of this defect — repaired, `ha` reads 0.1659 /
  0.0330 / 0.1984 and BEATS `ha0` (0.4960 / 0.0350 / 0.3768).
  Package: TanitAD Research Lab/Benchmarks & Evals/Research/
           2026-09-03-refav1-kinematic-contract-lateral/ (SPEC pre-registered, RESULT, raw/)
  Status of the LATERAL suspension: the CAUSE is identified and the instrument is
  quantified; the suspension on refav1 LATERAL rows LIFTS only when the repair (or the
  eval-only stopgap) is applied and the dump re-rolled. Until then every refav1 LATERAL
  row is off by a known 2.9x over-rotation, in a known direction.

D-REFAV1-HA0-ARM — DECIDED (2026-09-03)
  `taniteval/tools/refav1_arm.py` gains ONE arm, `ha0` (tier T1): constant velocity,
  a = 0, kappa = 0, at the measured v0 — the STRONGEST TRIVIAL BASELINE and the bar the
  echo test must be run against. It joins the default arm list with its arm_meaning, the
  four family rows and the paired `cl - ha0` block; no existing arm's semantics change
  (pinned by test A6). Beside it, a TRIVIAL-PROFILE INSTRUMENT prints BEFORE any family
  row: per arm the straight / constant-speed / constant-velocity fractions and the
  bit-identical cross-arm counts. On the banked step-1,000 dump it reads cl
  CONSTANT-VELOCITY 1.0000 and cl identical to cl_navshuf on 122/140 and to
  cl_oraclegoal on 96/140 — i.e. it names the void read in six lines before any metric.
  Tests: stack/tests/test_refav1_kin_contract.py (26 passed); full refav1 suite 69 passed.
  OPEN INTEGRATION (one line, outside this task's ownership): add `"ha0": "T1"` to
  `t1_eval.DEFAULT_TIERS` (t1_eval.py:145) — until then the standalone t1_eval CLI needs
  `--tiers ha0=T1` on any refav1 dump.
```

## 9. Deviations and limits

* **20-clip local slice**, non-parity, of the 141-clip eval split; a step-1,000 checkpoint's dump.
  The KINEMATIC finding does not depend on the checkpoint at all (`ol`/`ha` never touch the model),
  but the `cl`-side statements do and inherit the amendment's exploratory-power caveat.
* `h_c2_wbfit`'s per-episode L̂ is fit on the scored windows. It is reported as an ORACLE CEILING and
  it changes nothing (0.0607 vs 0.0600); no repair uses it.
* The per-clip wheelbase and the build regime are **UNVERIFIED** here (§5) — a Data FlyWheel item.
* `ha` under the repair (§4) was computed by a one-off reproduction beside the probe, not by the
  probe's panel; it is a consequence of the same measured gain, not an independent result.
* TACTICAL and STRATEGIC families are **REFUSED with a reason** for this probe (a replay of the
  human's own recorded actions declares no manoeuvre and no route) — not silently omitted.
