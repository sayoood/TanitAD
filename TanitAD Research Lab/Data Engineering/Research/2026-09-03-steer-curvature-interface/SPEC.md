# SPEC — the steer↔curvature INTERFACE: one contract, and where it is crossed

**Package:** `TanitAD Research Lab/Data Engineering/Research/2026-09-03-steer-curvature-interface/`
**Owner:** Data Engineering FlyWheel · **PI:** Sayed · **Written:** 2026-09-03 ~07:10 Berlin, dev box,
**0 GPU, no pod contacted** (CPU arithmetic on local v2ep + local raw egomotion only).
**Supersedes the framing of** `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refav1-kinematic-contract-lateral/`
(its measurements stand; its proposed repair site does not).

---

## 0. The PI ruling this SPEC implements (verbatim, 2026-09-03 ~06:50 Berlin)

> *"why steering is wrong? i think if steering is correctly processed by the predictor, the model
> will learn the physical dependency between curvature and steering. I think whats important: if we
> are using the planner output for the predictor, that the curvature is translated to steering as
> input for the predictor as action"* — and, on the wheelbase: *"Find the true source first"*.

⇒ `actions[:, 0] = steer` is **NOT a defect**. It is the intended action representation. The
predictor is a *command* model; a road-wheel angle is what a driver commands. **No data change, no
cache rebuild, no Thor restart.**

## 1. THE CONTRACT (the thing this package is really delivering)

There are exactly **two** unit domains, and every module belongs to exactly one:

| domain | unit | who lives here |
|---|---|---|
| **COMMAND** | `steer` = road-wheel angle [rad] | the v2ep `actions[:,0]` channel · the loader · the predictor's action input · every trained refav1/v6/v7 checkpoint |
| **GEOMETRY** | `κ` = path curvature [1/m] | `rollout_unicycle` · `unicycle_paths` · `canonical_controls`' κ grid · `GOAL_KAPPA_*` · `PlanConfig.kappa_max` · every metre-valued trajectory |

The bridge is **one** pair of exact inverses with **one** constant `L_enc`:

```
κ  = tan(steer) / L_enc          (GEOMETRY ← COMMAND)
steer = arctan(L_enc · κ)        (COMMAND ← GEOMETRY)
```

**Every crossing must apply one of them. A crossing that applies neither is the defect.**
Today there is exactly one class of unconverted crossing — geometry code reading the command
channel as if it were κ — and it is what the Benchmarks package measured as a **×2.870
over-rotation** and a 0.7156 m curved-window lateral miss.

The crossings this SPEC will enumerate and then FIX (COMMITTED list — anything found beyond this
list is reported as a SPEC miss, not folded in silently):

| # | crossing | direction | file:line (to be pinned in RESULT) | today |
|---|---|---|---|---|
| X1 | eval open-loop / hold-action replay of RECORDED actions | COMMAND → GEOMETRY | `taniteval/tools/refav1_arm.py` (`ol`, `ha`) | **unconverted** |
| X2 | planner scores a candidate by integrating it | COMMAND → GEOMETRY | `refa_v1_plan.py` `unicycle_paths` / `kinematic.rollout_unicycle` | **unconverted** |
| X3 | planner hands a candidate to the MODEL as an action | GEOMETRY → COMMAND | `refa_v1_plan.py` (candidate → `a_t` fed to `RefAV1`) | **unconverted** |
| X4 | goal constants → candidate κ grid | stays GEOMETRY | `refa_v1.GOAL_KAPPA_*`, `canonical_controls` | correct, no change |
| X5 | loader → predictor training input | stays COMMAND | `refav1_loader.py:258-264` `_kin_actions` | correct, no change |

## 2. Question A — `L_enc`, settled by CONTENT, not by fitting

⚠️ **`L_enc` is an ENCODING CONSTANT, not a vehicle property.** `physicalai.signals_at` wrote
`steer = arctan(wheelbase · curvature)` with *whatever `wheelbase` the build passed*; the inverse
must use **that same number**, whether or not it is any real car's wheelbase. Fitting `L` to
trajectories answers a different (and noisier) question. Therefore:

**A1 — THE TRUE SOURCE (PI: "find the true source first").** Locate the vehicle parameter the
PhysicalAI release actually ships. Two probes for any absence claim (`grep` AND `Select-String`,
or `ls` AND a reader). Report the exact artifact path and schema.

**A2 — THE EXACT RECOVERY (the decisive measurement).** For each of the 20 local eval clips, read
the **raw egomotion `curvature` column** — the very input `signals_at` consumed — alongside the
stored v2ep `actions[:, 0]`, on the same 10 Hz query grid, and solve

```
L_enc(clip) = tan(actions[:,0]) / curvature            (pointwise, |curvature| above a floor)
```

This is an **algebraic inversion of the producer, not a fit**: if the corpus is `const2p9`, every
clip must return `2.9` to float precision and the pointwise spread must be ~machine-epsilon.

* **COMMITTED expectation:** `L_enc = 2.9000 ± 0.0005` on **20/20** clips, pointwise
  IQR < 1e-3.
* **COMMITTED falsifier:** if any clip returns a value from the shipped population
  {2.730, 2.850, 3.135, 3.165, 3.216} instead, the corpus is `per_clip_v1` and the single-constant
  inverse is WRONG. This is a real possible outcome and is why the probe is run.
* **CONTROL that must read a known value:** the recovered `L_enc` fed forward
  (`arctan(L_enc · curvature)`) must reproduce the stored channel to < 1e-6 rad. If it does not,
  the egomotion file I read is not the one that minted the cache and the whole probe is void.

**A3 — THE TRUE PER-CLIP WHEELBASE, for the same 20 clips**, from A1's artifact where reachable.
Report it *beside* `L_enc` and state plainly that they are different quantities.

**A4 — THE TRAJECTORY FIT, re-run and re-interpreted.** Per clip, least-squares
`tan(steer) = w · κ_pose` with `κ_pose = Δyaw_pose / (v·dt)` from `poses[:,2]` — the estimator the
previous agent used, which produced the "~2.85 / ~3.09 bimodality". Plus a second, lower-variance
estimator that integrates over a window instead of differentiating pointwise:

```
w_win = Σ tan(steer)·v·dt  /  Δyaw_pose      (per 2 s window, |Δyaw_pose| > 0.02 rad)
```

* **COMMITTED expectation, conditional on A2 returning 2.9:** the A4 spread is **estimator noise in
  `κ_pose`, not wheelbase variation**. Predictions: (i) the window estimator's spread is materially
  tighter than the pointwise one; (ii) the per-clip fits do **not** concentrate on the shipped
  population — in particular **2.730 covers 47 % of the corpus and must be essentially absent**
  from the fits if `const2p9` holds; (iii) the per-clip fitted `w` does **not** correlate with the
  clip's true wheelbase from A3.
* **COMMITTED cost read:** the metres-at-2 s penalty of using the single constant instead of the
  per-clip fitted value. The banked panel already says this is **< 1 cm** (`h_c1_wb` 0.0600 vs
  oracle `h_c2_wbfit` 0.0607, i.e. the oracle is *worse*); this SPEC re-derives it on the same
  windows and states whether a single constant is defensible.

## 3. Question B — the repair, at the crossings, flag-gated, default OFF

**Default OFF is not caution, it is correctness bookkeeping:** every banked refav1 number was
produced by the unconverted path, and a silently converted eval would make old and new numbers
incomparable while Thor is mid-run.

* **`B-OFF` (byte-identical):** with the flag OFF every arm, every metric and every test must
  reproduce today's value **bit for bit** (`torch.equal` on the trajectories, not `allclose`).
* **`B-ON` (the repaired numbers, by the NEW route):** with the flag ON, the eval replay must
  reproduce the banked repaired numbers from the Benchmarks package:
  **curved LAT 0.0600 m**, **LON(all 140) 0.1199 m**.
  **COMMITTED tolerance: ±0.002 m** (the two routes differ only in *where* the same algebra is
  applied, so agreement should be far tighter than that; the tolerance is for float ordering and
  window bookkeeping).
  ⛔ **This is a genuine cross-check, declared in advance:** the Benchmarks package converted the
  DATA (`κ = tan(steer)/L` at read time); this package converts at the INTEGRATOR. If the two do
  **not** land on the same number, one of the two framings is wrong and RESULT.md says which,
  loudly, instead of quoting the agreeing half.
* **`B-PIN` (the unit test the PI asked for):** on REAL windows, the converted κ must match the
  pose-derived κ — the physics ground truth — not merely some other code path.
* **`B-PLAN` (the X3 direction):** a candidate κ handed to the model must arrive as
  `arctan(L_enc·κ)`, and the round trip `κ → steer → κ` must be the identity to < 1e-6. Pin that a
  `GOAL_KAPPA_TURN = 0.08` (R = 12.5 m) candidate is *imagined* at R = 12.5 m under the flag and at
  R ≈ 36 m without it — the flat-cost-surface mechanism, made into a number.

## 4. Question C — blast radius, re-read under the contract

Every consumer of `actions[:, 0]` in the repo, classified by the contract of §1:

* **feeds the PREDICTOR** (stays COMMAND) ⇒ **NOT AFFECTED**
* **integrates GEOMETRY / emits metres or 1/m** ⇒ **AFFECTED**
* **converts already** ⇒ **ALREADY CORRECT** (and is evidence the contract is the house style)
* anything I cannot classify from the source ⇒ **UNVERIFIED**, with the file:line

Two probes (`grep` and `Select-String`) for every "absent" claim. A wrong *not affected* is worse
than a flagged unknown; there is no third option where a consumer goes unlisted.

## 5. Question D — are the trained checkpoints mis-trained?

**COMMITTED expected answer under the ruling:** *(i) correctly trained, provided inference feeds
steer.* The SPEC's job is the **discriminating test**, not the assertion:

* the **units test** (0 GPU): the training input path and the inference input path are the same
  function of the same stored tensor ⇒ pure reparametrisation ⇒ *no* checkpoint is mis-trained.
  A refutation would be an inference path that feeds κ where training fed steer.
* the **checkpoint test** (minutes on the 4060 when free, or CPU): feed a banked checkpoint a
  windowed action sequence in COMMAND units and in GEOMETRY units and compare the rollout's
  agreement with the recorded future. If COMMAND wins, the model learned steer (fine). If GEOMETRY
  wins, the model learned to treat the channel as κ and the ruling's premise fails.

## 6. Families

**LONGITUDINAL** and **LATERAL** are scored (both are geometry and both move under the repair).
**TACTICAL** and **STRATEGIC** are **REFUSED with a reason**: this package changes a unit interface
and replays recorded/candidate controls; it declares no manoeuvre and no route, so those two
families have no arm to score here. They are refused, not omitted — the refusal is the finding that
the interface repair is *upstream* of them and they must be re-read after it lands.

## 7. What this SPEC does NOT do

* It does **not** touch `physicalai.py`, `refav1_loader.py`'s action semantics, or any cache. The
  training input is ruled correct.
* It does **not** restart, pause or reconfigure the live Thor run, and it does not change any
  default.
* It does **not** re-read a checkpoint under a new unit (that is §5's *proposed* test, costed here,
  run only if the GPU is free and nothing else needs it).
* It cannot repair `cl` retroactively: banked closed-loop numbers were produced by the unconverted
  planner and stay quoted as such.
