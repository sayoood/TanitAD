# RESULT — the steer↔curvature INTERFACE: one contract, five crossings, one constant

**Package:** `TanitAD Research Lab/Data Engineering/Research/2026-09-03-steer-curvature-interface/`
**Owner:** Data Engineering FlyWheel · **PI:** Sayed · **2026-09-03, dev box, 0 GPU, no pod contacted.**
**SPEC:** `SPEC.md` in this directory, written and staged BEFORE any measurement below.

---

## ⛔ HEADLINE FOR THE MASTER MIND — three items, in order of urgency

1. **THE DATA IS NOT WRONG AND THOR MUST NOT BE RESTARTED.** `actions[:, 0]` is a road-wheel
   STEER angle by design; the predictor is a command model and the loader hands it through
   unchanged. **Recommendation: the live Thor run continues untouched.** Nothing in this package
   changes a default, a cache, or `refav1_loader.py`'s action semantics.
2. **INTEGRATION NEEDED — the repair is staged behind `--action-units`, default OFF, and someone
   has to decide when to turn it ON.** OFF is byte-identical to today; ON reproduces the repaired
   panel exactly. It is a *decision*, not a merge: every banked refav1 lateral number was produced
   OFF, and the two are not comparable.
3. **⭐ THESE FILES MUST LAND TOGETHER WITH THE ArchInf COST-SURFACE PACKAGE.**
   `taniteval/tools/cost_surface_probe.py` (staged `A` by another FlyWheel at ~07:20 today, while
   this package was being written) **imports `STEER_WHEELBASE_M` and `as_command` from
   `stack/tanitad/models/kinematic.py` — i.e. from this package's bridge — and will not import
   without it.** Its own `C7 CONVERSION AGREEMENT` control cross-checks its one-line map against
   `as_command`; I ran that control against the staged code and it reads **max |diff| = 0.000e+00
   at L ∈ {2.9, 2.73, 3.216}**. ⇒ the two packages are consistent, and neither is shippable alone.
4. **A SECOND ARM HAS THE SAME DEFECT AND IS NOT FIXED HERE: `taniteval/tools/refcv3_arm.py:477-507`**
   (`recorded_controls` → `integrate_select`) replays the recorded steer channel through the same
   integrator and calls it `kappa`. It picks the new keyword up automatically but nothing passes it.
   **Owner: whoever owns refcv3.** (Full list in §4.)

---

## 1. THE CONTRACT (the deliverable the PI asked for)

There are exactly **two** unit domains for action channel 1. Every module belongs to one:

| domain | unit | meaning | who lives here |
|---|---|---|---|
| **COMMAND** | `steer` [rad] | what a driver commands | v2ep `actions[:,0]` · the loader · the predictor's action input · every trained checkpoint |
| **GEOMETRY** | `κ` [1/m] | what the road makes the car do | `rollout_unicycle` · `unicycle_paths` · `GOAL_KAPPA_*` · `PlanConfig.kappa_max` · every metre-valued path |

One bridge, one constant, two exact inverses:

```
κ     = tan(steer) / L_enc          L_enc = 2.9   (MEASURED exactly, §2)
steer = arctan(L_enc · κ)
```

**Every crossing applies one of them. A crossing that applies neither is the defect.**

### 1.1 The crossings, with file:line and status

| # | crossing | direction | file:line | before | after |
|---|---|---|---|---|---|
| **X1** | eval replays a RECORDED action (`ol`, `ha`) | COMMAND→GEOMETRY | `taniteval/tools/refav1_arm.py:669` / `:672` | unconverted | **flag-gated, `--action-units steer`** |
| **X2** | a control sequence is integrated into metres | COMMAND→GEOMETRY (only when the input is a command) | `stack/tanitad/refs/refa_v1_plan.py:292-321` `unicycle_paths` → `stack/tanitad/models/kinematic.py:220` `rollout_unicycle` | one unit assumed for both callers | **`action_units=` kwarg; the unit travels with the CALL** |
| **X3** | a planner candidate is handed to the MODEL | GEOMETRY→COMMAND | `stack/tanitad/refs/refa_v1.py:1832` (inside `plan._cost_chunk`) | unconverted | **`plan(model_action_units="steer")`** |
| **X4** | goal tokens → candidate κ grid | stays GEOMETRY | `stack/tanitad/refs/refa_v1.py:118-119` `GOAL_KAPPA_MAX/TURN`, `:131-167` `canonical_controls`, `refa_v1_plan.py:88` `kappa_max` | correct | **unchanged** (pinned by test D4) |
| **X5** | loader → predictor training input | stays COMMAND | `stack/tanitad/data/refav1_loader.py:258-264` `_kin_actions` | correct | **unchanged — Thor is training on it** |

⛔ **Why the conversion is NOT inside `rollout_unicycle`.** A planner candidate (κ) and a recorded
action (steer) reach the *same* integrator in *different* units. Baking a conversion into the
integrator would repair X1 and simultaneously **break** X4/X2-for-candidates. The unit therefore
travels with the call (`action_units=`), and `rollout_unicycle` keeps its single meaning
`yaw_rate = v·κ` — pinned by `test_C2_the_integrator_itself_is_untouched`.

### 1.2 The house already agrees — two independent sites implement this contract today

* `taniteval/tools/t1_eval.py:775-781` `roll_closed`: computes `kappa = yaw_rate/max(v,0.3)` from
  geometry and then feeds the model `steer = atan(wheelbase*kappa)` — **X3, already correct**, on
  the v6/v7 T1 line. `t1_eval.py:825` `implied_controls` does the same.
* `stack/scripts/train_v6_staged.py:4692-4738` `_lift3`: `COND_INCUMBENT` is literally named
  `"steer_accel_v"` and passes channel 0 through to the predictor; `COND_EGO_STATE` converts
  `omega = v·tan(steer)/L` — **a correct COMMAND→GEOMETRY conversion**, with the note that the
  wheelbase cancels.
* `stack/scripts/stage_a_probes.py:168-177` already ships `kappa_of_steer` / `steer_of_kappa` and
  applies deltas in κ space through the encoding.

* `stack/tanitad/models/v6.py:5316-5339` converts κ → steer before the predictor, naming the
  channel `steer_road_rad`; `stack/scripts/train_stage_a.py:182-208` builds its counterfactuals as
  `steer_of_kappa(kappa_of_steer(steer) + Δκ)` — perturb in curvature, hand back a command.

⇒ The ruling is not a new convention; it is the convention **the rest of the programme already
uses at five independent sites**, and the refav1 line is the one place that never crossed.

---

## 2. QUESTION A — the wheelbase, settled by CONTENT (PI: *"find the true source first"*)

### 2.1 A1 — THE TRUE SOURCE: FOUND, PULLED, SCHEMA READ

The PhysicalAI release **does** ship a per-clip vehicle parameter. It is not a metadata field or a
dataset-card note; it is its own calibration product:

```
HF nvidia/PhysicalAI-Autonomous-Vehicles ::
    calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet
```

**Schema (MEASURED, read off the file):**
`clip_id: str · length · width · height · rear_axle_to_bbox_center · wheelbase · track_width`
(all float64, metres).

Pulled for the 19 chunks the 20 local eval clips live in: **1,784 clips, 20/20 of ours resolved.**
The stack already knows this path — `physicalai.wheelbase_for_clip` (`physicalai.py:411-443`)
resolves it, and `…/Data Engineering/Implementation/incoming/2026-07-26-wheelbase-impact/wb_pull_dims.py`
pulled it once for the whole 2,400-clip corpus on 2026-07-26. **The SPEC's and the Benchmarks
package's "not on this box / UNVERIFIED" is now closed.**

**The 20 eval clips' TRUE wheelbases** (`raw/true_wheelbase.json`, `raw/true_wheelbase_per_clip.csv`):

| value [m] | clips |
|---|---|
| 2.730 | 4 |
| 3.135 | 2 |
| 3.165 | 10 |
| 3.216 | 4 |

mean **3.085**. **NOT ONE of them is 2.9.** (Corpus-wide, from the 2026-07-26 pull:
2.730 / 2.850 / 3.135 / 3.165 / 3.216, clip-mean 2.9568.)

### 2.2 A2 — THE ENCODING CONSTANT, recovered EXACTLY (this is the decisive measurement)

⚠️ **`L_enc` is an ENCODING constant, not a vehicle property.** `signals_at` wrote
`steer = arctan(wheelbase · curvature)` with whatever the *build* passed. The inverse must undo
**that** number. So the question is not "what wheelbase did the car have" but "what number did the
producer multiply by", and that is answerable by algebra, not by fitting.

**Method (an inversion of the producer, not a fit):** reconstruct the builder's own query grid
(`t_query = linspace(t_frames[0], t_frames[-1], n_target)` from the clip's video-timestamp parquet
— `physicalai.py:707-712` / `v2_compressed.py:116-121`), interpolate the RAW egomotion
`curvature` column onto it exactly as `signals_at` does, and solve `L = tan(steer_stored)/curvature`
pointwise.

**Result (`raw/lenc_exact_per_clip.csv`, `raw/lenc_exact.json`), 20/20 clips:**

| quantity | value |
|---|---|
| `L_enc` per clip (OLS through origin) | **min 2.8999999827 · max 2.9000000313 · median 2.9000000** |
| pointwise `tan(steer)/curvature` IQR | **1.2 × 10⁻⁷** (median over clips) |
| forward map `arctan(2.9·curvature)` vs the STORED channel | **max 1.49 × 10⁻⁸ rad over all 20 clips** |

**Controls (SPEC §2, all must read a known value):**

| control | reads | max over 20 clips |
|---|---|---|
| C1 · reconstructed grid reproduces `poses[:,3]` (v) — the channel used to build it | ~0 | **1.70 × 10⁻⁶ m/s** |
| C2 · reproduces `poses[:,2]` (yaw) — **not used in the reconstruction** | ~0 | **5.93 × 10⁻⁸ rad** |
| C2 · reproduces `actions[:,1]` (ax) — **not used** | ~0 | **1.19 × 10⁻⁷ m/s²** |
| C2 · reproduces `poses[:,0]` (x) — **not used** | ~0 | **3.02 × 10⁻⁵ m** |
| grid length | `n_target == n_stored` | **20/20 (201 frames)** |

⇒ **`L_enc = 2.9, EXACTLY. Evidence class MEASURED.** The corpus is `const2p9`. The Benchmarks
package's UNVERIFIED item is CLOSED.

**Two independent probes for the code half of the same claim** (the CLAUDE.md absence rule):

| probe | on `stack/scripts/v2_compressed.py` — the builder that minted these `.v2ep.pt` | result |
|---|---|---|
| `grep -n "wheelbase\|WHEELBASE"` | | **exit 1, 0 hits** |
| `Select-String -Pattern 'wheelbase' -CaseSensitive:$false` | | **0 hits** |

A **third, independent** confirmation arrived from the whole-repo probe: the ARCHIVED copy of the
same builder, `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-07-24-v2-corpus-50h-balanced/v2_compressed.py`,
likewise has **0** wheelbase occurrences and likewise calls `signals_at(ego, t_query)` bare (`:89`).

`v2_compressed._resampled:124` calls `signals_at(ego, t_query)` with **no wheelbase argument at
all** — so the module cannot mint anything but `const2p9`, and `physicalai.DEFAULT_WHEELBASE_MODE`
is `"const2p9"` (`physicalai.py:82`), whose `label_params` contributes **no cache key**
(`physicalai.py:97-99`) precisely so the legacy corpus keeps its identity.

### 2.3 A3/A4 — the per-clip TRAJECTORY FIT, and why it is the wrong instrument

The previous agent's "bimodality at ~2.85 / ~3.09" is **not a wheelbase distribution**. Re-run here
on the same 20 clips (`raw/per_clip_fits.csv`):

| estimator | n | spread |
|---|---|---|
| per-clip pointwise LS `tan(steer) ~ w·κ_pose`, `v>3 m/s` | 15 fit / 5 too straight | **2.572 – 3.140**, median 2.865, sd 0.126 |
| per-window integral `Σ tan(steer)·v·dt / Δyaw_pose` (per-clip median) | 17 | **2.405 – 3.106**, sd 0.159 |
| the EXACT inversion (§2.2) | **20/20** | **2.9000000 ± 3 × 10⁻⁸** |

**The fit is ANTI-correlated with the truth.** `corr(per-clip fitted w, TRUE wheelbase)` =
**−0.60** (pointwise, n = 15) and **−0.47** (window estimator, n = 17). A `per_clip_v1` encoding
would demand ≈ **+1**. Two clips whose true wheelbase is the corpus's *smallest* (2.730) fit at
**3.077** and **3.140** — the wrong end of the range. And **2.730 is 47 % of the corpus but only
1/15 fits land within 0.05 of it.**

⇒ **The scatter is the κ_pose estimator's error (quaternion-yaw finite difference ÷ speed against
the dataset's own `curvature` solution), not a wheelbase.** The fit answers a different question
and answers it noisily.

### 2.4 The cost of using a single constant — and it is NEGATIVE (a constant is *better*)

Same 140 windows, same integrator, only the divisor changes (`raw/interface_per_window.csv`):

| divisor | LAT curved (72) | LAT straight (68) | LON all (140) |
|---|---|---|---|
| **`L_enc = 2.9` (the encoding constant)** | **0.0600** | **0.0141** | **0.1199** |
| the clip's TRUE wheelbase {2.73…3.216} | 0.0846 | 0.0147 | 0.1206 |
| the clip's own fitted `w` (ORACLE, fitted on this slice) | 0.0710 | 0.0125 | 0.1087 |
| pose-yaw floor (no action channel at all) | 0.0527 | 0.0096 | 0.1189 |

⇒ **A single constant is not merely defensible — it is the only correct choice, and it is exact.**
Using the real wheelbase costs **+0.025 m** of curved-window lateral error at 2 s (+41 % over the
constant, and it moves *away* from the 0.0527 m floor). Even an oracle per-clip fit is worse on
LAT. **The number to use is 2.9, and the reason is that 2.9 is what the encoder multiplied by.**

⚠️ **The one caveat, stated because it will matter later:** if a `per_clip_v1` cache is ever built
(`build_pai_cache.py --wheelbase-mode per_clip_v1`), its `L_enc` is per clip and this constant is
wrong for it. The bridge therefore takes `wheelbase` as a parameter and the manifest stamps
`L_enc_m`; a per-clip cache must carry its own value. Pinned by `test_F3`.

---

## 3. QUESTION B — the repair, at the crossings, flag-gated, default OFF

### 3.1 What was implemented

| file | change | risk to the live run |
|---|---|---|
| `stack/tanitad/models/kinematic.py` | **pure addition**: `STEER_WHEELBASE_M = 2.9`, `ACTION_UNITS`, `kappa_of_steer`, `steer_of_kappa`, `as_curvature`, `as_command`, `_check_units` — plus the contract in prose, with the measurement | none (nothing existing touched) |
| `stack/tanitad/refs/refa_v1_plan.py` | `unicycle_paths(..., *, action_units="kappa", wheelbase=STEER_WHEELBASE_M)` | none — default returns the input object unchanged |
| `stack/tanitad/refs/refa_v1.py` | `plan(..., model_action_units="kappa")`; conversion applied at `:1832` only | none — **a CALL-SITE kwarg, deliberately NOT a `RefAV1Config` field**, so no serialised config dict moves while a run is live; `plan()` is not on the training path |
| `taniteval/tools/refav1_arm.py` | `paths_from_controls(..., action_units=)`; `--action-units {kappa,steer}` default `kappa`; `ol`/`ha` and `plan()` both read it; manifest stamps `action_units` provenance | none — default is today's behaviour |
| `stack/tests/test_steer_curvature_interface.py` | **new**, 22 tests | — |

`as_curvature(c, "kappa")` **returns the same object** (`is c`), so the OFF path cannot even
acquire a float round-trip. Pinned by `test_B1`.

### 3.2 B-OFF — byte-identical

`test_B2` / `test_B3` assert `torch.equal` (not `allclose`) between the new default call and the
pre-change expression. `test_B4` reads the CLI source and pins `default="kappa"`. `test_D1` asserts
`plan()`'s default is bit-identical in controls, source and cost. **Full existing refav1 suite after
the change: 91 passed** (`test_refav1_arm.py`, `test_refav1_kin_contract.py`,
`test_refav1_loader.py`, `test_refav1_loader_labels.py`, `test_refav1_lead_block.py`,
`test_steer_curvature_interface.py`); plus **95 passed** across `test_refa_v1*.py` /
`test_refa_v1_plan_goal.py` / `test_refa_v1_proposal.py` / `test_refa_v1_speed_channel.py` and
**61 passed** across `test_kinematic_losses.py` / `test_kinematic_nan.py` /
`test_cond_parameterisation.py` / `test_ego_plan.py`. The whole-repo sweep then surfaced six more
suites that touch the integrator or the bridge and had not been run: **121 passed** across
`test_unicycle_action_space.py`, `test_stage_a.py`, `test_stage_a_train.py`,
`test_t1_v2_adapter.py`, `test_retime_path.py` — **and `test_cost_surface_probe.py`, another
FlyWheel's suite, which passes against this package's staged bridge.**

### 3.3 B-ON — the SPEC's declared cross-check, and it HOLDS

⛔ Declared in SPEC §3 before running: the Benchmarks package converted the **DATA**
(`κ = tan(steer)/L` at read time); this package converts at the **INTEGRATOR**. If the two disagree,
one framing is wrong. Measured through the shipped `paths_from_controls` on the banked step-1,000
dump, 140 windows, 72 curved:

| arm | mode | LAT curved | LAT straight | LON all | banked (data-side conversion) | agree? |
|---|---|---|---|---|---|---|
| `ol` | **OFF** (`kappa`) | **0.7156** | 0.0685 | **0.2395** | 0.7156 / 0.0685 / 0.2395 | ✅ **exact** |
| `ol` | **ON** (`steer`) | **0.0600** | 0.0141 | **0.1199** | 0.0600 / 0.0141 / 0.1199 | ✅ **exact** |
| `ha` | OFF | 0.7625 | 0.1203 | 0.3175 | 0.7625 / 0.1203 / 0.3175 | ✅ |
| `ha` | ON | 0.1659 | 0.0330 | 0.1984 | 0.1659 / 0.0330 / 0.1984 | ✅ |
| `ha0` | — | 0.4960 | 0.0350 | 0.3768 | 0.4960 / 0.0350 / 0.3768 | ✅ (unit-invariant) |

**Committed tolerance was ±0.002 m; the agreement is to the 4th decimal on every cell.**
⇒ **THE TWO FRAMINGS ARE THE SAME ALGEBRA APPLIED AT DIFFERENT SITES, AND THEY AGREE.** The PI's
"verify that it does and say so explicitly" is answered: it does. Neither framing is wrong; the
*site* is the only thing that differs, and the ruling picks the integrator/boundary site because it
leaves the training input — and Thor — untouched.

### 3.4 B-PIN — the physics pin on REAL windows

`test_F1`: over the 75 turning windows (`|Δyaw_pose| > 0.02 rad`), the ratio of integrated yaw to
the human's own pose yaw is

| reading | median ratio |
|---|---|
| unconverted (`kappa`) | **2.870** |
| converted (`tan(steer)/2.9`) | **0.995** |

asserted as `|median − 1| < 0.05` for ON **and** `median > 2.5` for OFF, so a pass cannot be vacuous.

### 3.5 B-PLAN — the planner→model crossing, as a number

`test_D4` pins that `GOAL_KAPPA_TURN = 0.08` stays curvature (R = 12.5 m) and becomes
`arctan(2.9 × 0.08) = 0.22797 rad` at the model boundary. `test_C4` pins the mechanism the PI
described: a channel value of 0.08 read as κ is **R = 12.5 m**; read as steer it is
**R = 2.9/tan(0.08) = 36.16 m** — **×2.893**. That is why the cost surface is flat in curvature
exactly where the tactical goals live. `test_D2` proves the flag reaches the model (baseline costs
move) while the zero-curvature baselines `cv`/`hold_v0` do **not** move (arctan(0) = 0).

### 3.6 The deliberate regression (the flag is not a universal improver)

`test_E2`: on a synthetic corpus whose channel **really is** a curvature, the ON path must be
clearly WORSE. Measured on a fixture generated on the 0.2 s scoring grid (so the 10 Hz→5 Hz
quadrature floor is removed and the difference is attributable to the unit alone): OFF replay
error **< 1e-3 m** (the contract holds exactly), ON **> 5×** worse. Without this, `test_E1` would
prove nothing about which unit the real corpus is in.

---

## 4. QUESTION C — BLAST RADIUS, enumerated under the contract

Rule applied: **feeds the PREDICTOR ⇒ NOT AFFECTED · integrates GEOMETRY or emits metres/1-per-m ⇒
AFFECTED · converts already ⇒ ALREADY CORRECT.** Two probes for every absence.

| # | consumer | file:line | reads channel 1 as | verdict |
|---|---|---|---|---|
| 1 | **refav1 TRAINING input** — `_kin_actions` → `refa_v1_train.py` → `RefAV1.forward` | `stack/tanitad/data/refav1_loader.py:258-264` → `stack/scripts/refa_v1_train.py:627,667` | COMMAND, straight to the predictor | ✅ **NOT AFFECTED** — correct under the ruling. **Thor keeps running.** |
| 2 | refav1 T0 eval forward (same tensor, same function) | `taniteval/tools/refav1_arm.py:648,680` | COMMAND | ✅ **NOT AFFECTED** |
| 3 | **refav1 `ol` / `ha` open-loop replay** | `taniteval/tools/refav1_arm.py:669,672` | integrated as GEOMETRY | ⛔ **AFFECTED — FIXED HERE (flag)**. 0.7156 → 0.0600 m curved LAT |
| 4 | **`unicycle_paths` / `rollout_unicycle`** | `stack/tanitad/refs/refa_v1_plan.py:292-321`, `stack/tanitad/models/kinematic.py:220-248` | GEOMETRY (by definition) | ⛔ **AFFECTED at the CALL, not the body — FIXED HERE (`action_units=`)**; integrator body unchanged |
| 5 | **refav1 planner → model** | `stack/tanitad/refs/refa_v1.py:1832` | GEOMETRY handed to a COMMAND model | ⛔ **AFFECTED — FIXED HERE (`model_action_units`)** |
| 6 | refav1 `cl` path integration (planner's own κ) | `taniteval/tools/refav1_arm.py:749` | GEOMETRY, correctly | ✅ **NOT AFFECTED** — a candidate is already curvature |
| 7 | `canonical_controls` + `GOAL_KAPPA_MAX/TURN` | `stack/tanitad/refs/refa_v1.py:118-119,131-167` | GEOMETRY | ✅ **NOT AFFECTED** — they are geometry and stay geometry (pinned, `test_D4`) |
| 8 | `PlanConfig.kappa_max` / `_clip` | `stack/tanitad/refs/refa_v1_plan.py:88,162-166` | GEOMETRY | ✅ **NOT AFFECTED** — and only now is "1/m" true of the whole path |
| 9 | **refcv3 `ol` / `ha` replay** | `taniteval/tools/refcv3_arm.py:477-488` `recorded_controls`, `:491-498` `hold_controls`, `:501-507` `integrate_select` | integrated as GEOMETRY, variable literally named `kap_ep` (`:766`), docstring says "the MEASURED true-kappa channel" | ⛔ **AFFECTED — NOT FIXED (out of my ownership).** Same defect, same ×2.9. **ESCALATED** |
| 10 | `refb_labels` maneuver / nav labels | `stack/scripts/refb_labels.py:112-152` | — reads POSES only | ✅ **NOT AFFECTED.** Probes: `grep -n "actions"` → 1 hit, and it is the docstring *"no actions required"* (`:116`); `Select-String` → same 1 hit |
| 11 | **v7.2 label pipeline** | `stack/scripts/s2_geom_emit_v7.py`, `stack/tanitad/data/v7_labels.py`, `stack/scripts/s2_labels.py` | — | ✅ **NOT AFFECTED.** Probes: `grep -rn "actions\["` → 0 in all three; `Select-String -Pattern 'actions\['` → 0, 0, 0 |
| 12 | **v7 line — `train_v6_staged._lift3`** | `stack/scripts/train_v6_staged.py:4692-4738`; `COND_INCUMBENT = "steer_accel_v"` (`:176`) | COMMAND passed to the predictor | ✅ **NOT AFFECTED** (incumbent) / ✅ **ALREADY CORRECT** (`COND_EGO_STATE` converts `ω = v·tan(steer)/L`, `:4727-4729`) |
| 13 | `flagship_v15.SPEED_SCALE` and neighbours | `stack/tanitad/models/flagship_v15.py:85` | a SPEED normaliser, channel 2 | ✅ **NOT AFFECTED** — unrelated to channel 1 |
| 14 | **`t1_eval.roll_closed` / `implied_controls`** | `taniteval/tools/t1_eval.py:775-781, 795-826` | converts κ→steer at the model boundary | ✅ **ALREADY CORRECT** — the contract, implemented independently |
| 15 | `stage_a_probes` counterfactuals | `stack/scripts/stage_a_probes.py:168-177, 186+` | converts; perturbs in κ space | ✅ **ALREADY CORRECT** (duplicates the constant — consolidation item, §7) |
| 16 | `w7_roll_rerank` | `stack/scripts/w7_roll_rerank.py:166,300,1038` | imports `steer_of_kappa` | ✅ **ALREADY CORRECT** |
| 17 | `geom_sanity.check_action_scale` | `stack/scripts/geom_sanity.py:152-160` | percentiles, named `steer` | ✅ **NOT AFFECTED** — a statistic, no geometry |
| 18 | `latency_cnce_baseline.episode_telemetry` | `stack/scripts/latency_cnce_baseline.py:100-101` | `steer_rate = d/dt(steer)` | ✅ **NOT AFFECTED** — a steer-rate comfort metric on a steer channel is correct |
| 19 | `cosmos_drive` build + stat | `stack/tanitad/data/cosmos_drive.py:115,318` | mints `atan(2.9·κ)` itself | ✅ **NOT AFFECTED** — same encoding, same constant |
| 20 | `metric_dynamics` (`gt_ego_waypoints`, `accumulate_se2`, `UnicycleStepReadout`) | `stack/tanitad/models/metric_dynamics.py:155-167, 510+, 668` | κ derived from POSES, never from the action channel | ✅ **NOT AFFECTED** |
| 21 | `four_families` "kappa" | `taniteval/taniteval/four_families.py:936-1071` | **Cohen's kappa** — a name collision, not a curvature | ✅ **NOT AFFECTED** (flagged so nobody "fixes" it) |
| 22 | `l2d` loader | `stack/tanitad/data/l2d.py:53,408` | mints `atan(2.72·yaw_rate/v)` — its own corpus, its own constant | ✅ **NOT AFFECTED** — but its `L_enc` is **2.72**, not 2.9; never cross-apply |
| 23 | `alpasim-gsplat/closedloop_drive` | `stack/experiments/alpasim-gsplat/closedloop_drive.py:316,455,519` | bicycle model with its **own** `WHEELBASE = 2.7`, from waypoints | ✅ **NOT AFFECTED** — self-consistent, different corpus |
| 24 | **banked refav1 evals quoting a LATERAL number** | `C:\Users\Admin\refav1_eval_slice\refav1_t1_step1000.json`, `…_ep2_step1000.json`; `…/Architecture & Inference/Research/2026-09-03-refav1-step1000-read/`; `…/Benchmarks & Evals/Research/2026-09-03-refav1-kinematic-contract-lateral/` | produced with `action_units = kappa` | ⛔ **AFFECTED (as READINGS, not as files).** Every `ol`/`ha` LATERAL row is a ×2.9-over-rotated replay. The `cl` rows are unaffected *at step 1,000 only* because `cl` is κ ≡ 0 on 140/140 windows — **the defect appears the moment the planner steers** |
| 25 | `refav1_kin_contract_probe.py` | `taniteval/tools/refav1_kin_contract_probe.py:274,295` | the previous agent's diagnostic; applies its own transform | ✅ **NOT AFFECTED** — a probe, not production |
| 26 | flagship / REF-B / REF-C **training** action inputs | via `physicalai` v2 caches → predictor | COMMAND | ✅ **NOT AFFECTED** — same reasoning as row 1 |
| 27 | `build_pai_cache.py --wheelbase-mode per_clip_v1` caches | `stack/scripts/build_pai_cache.py:49-50,135,147` | would encode a **per-clip** `L_enc` | ⚠️ **UNVERIFIED — none is known to exist.** If one is ever built, `STEER_WHEELBASE_M` is the wrong divisor for it and the manifest's `L_enc_m` must carry the per-clip value. Probes: no `wheelbase_mode` key found in any local cache manifest, and `label_params` returns `{}` for legacy so a legacy cache carries **no** marker either way — **absence of the key does NOT prove `const2p9`; only §2.2's inversion does, and it was run per clip** |
| 28 | `stack/scripts/rebuild_pai_rolling.py` | `:87-88,111,215` | same regime switch | ⚠️ **UNVERIFIED**, same as row 27 |

| 29 | **`taniteval/tools/cost_surface_probe.py`** (ArchInf FlyWheel, staged `A` ~07:20 today) | `:199-204` `to_steer`, `:254-282` `score(units=)`, `:579-584` the C7 control, `:999-1002` `--wheelbase` | scores the planner cost in BOTH conventions, converting at the model boundary | ✅ **ALREADY CORRECT — and it CONSUMES this package's bridge.** Imports `STEER_WHEELBASE_M` + `as_command` from `kinematic.py`; **cannot import without this package staged.** Its C7 control reads **0.000e+00** against `as_command` at L ∈ {2.9, 2.73, 3.216} (verified here) |

| 30 | `stack/tanitad/models/v6.py:5316-5339` | imports `steer_of_kappa`, converts κ → steer, docstring says *"channel 0 = steer_road_rad (`steer_of_kappa`, the corpus encoding)"* | GEOMETRY → COMMAND at the model boundary | ✅ **ALREADY CORRECT** |
| 31 | `stack/scripts/train_stage_a.py:182-208` | `steer_of_kappa(kappa_of_steer(steer) + dk)` — perturbs in κ space, returns to steer | round trip through the bridge | ✅ **ALREADY CORRECT** |
| 32 | `stack/tests/test_t1_v2_adapter.py:357-363` | pins `steer_of_kappa(yaw_rate / v)` | test | ✅ **ALREADY CORRECT** |
| 33 | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-07-18-zod-loader/zod.py:143,237,261-268` | ZOD's own corpus: stores `("steer_road_rad", "accel_mps2")`, derived ratio-free from OxTS, with a measured CAN:road steering ratio | COMMAND, its own encoding | ✅ **NOT AFFECTED** — same convention, **different `L_enc`**; never cross-apply 2.9 to it (same caveat as `l2d.py` 2.72 and alpasim 2.7) |
| 34 | `TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-09-03-refcv3-arm-UNVERIFIED/refcv3_arm.py` | an incoming COPY of row 9's file, 3 integrator call sites | integrated as GEOMETRY | ⛔ **AFFECTED — NOT FIXED.** The refcv3 defect exists in **two** places; E2 must fix both |

⚠️ **COVERAGE, CLOSED OUT.** Rows 1-28 were first swept against a fast off-Drive
mirror of `stack/`, `taniteval/`, `colab/`, `tools/`, and **row 29 did not exist when that sweep
ran** — it was created by another FlyWheel *during* this session and surfaced only when the slow
whole-repo `Select-String` probe returned. **A mirror is not the repo, and a sweep is a snapshot.**
⇒ rather than leave that as a caveat, the table was then RE-SWEPT across the **whole repository**,
enumerating files through **`git ls-files` + `git status`** rather than directory recursion — which
covers files created during this session, and inherently excludes `.git` and `.claude/worktrees`.
**2,434 tracked/pending `.py` files, 0 unreadable (no G: flap), three pattern families** —
channel-1 reads (**31** files), integrator calls (**16**), bridge symbols (**16**). Rows 30-34 are
what that added. ⚠️ Two runs of the same sweep **seconds apart returned 2,434 and 2,435**: another
FlyWheel is adding files to this tree right now, which is the timestamp caveat above stated as a
measurement rather than a worry. **It found NO new
AFFECTED consumer** — four more sites that already implement the contract, and one more COPY of the
known refcv3 defect. The earlier probe also surfaced
`.claude/worktrees/**` (≈ 25 stale worktree copies of `physicalai.py` and its tests — **not
production, not affected**) and seven Research Lab package scripts, each checked here and each
reading **0** occurrences of `actions[:,0]` / `rollout_unicycle` / `unicycle_paths`
(`sc_dump_poses.py`, `fov_labels.py`, `closed_loop_dump.py`, `score_v2_pool.py`,
`wb_tier1_label_delta.py`, `run_join.py`, `make_standin_cache.py`).

**Anything I could not classify is row 27/28 and is marked UNVERIFIED, not "not affected".**

---

## 5. QUESTION D — are the already-trained models mis-trained?

### 5.1 The mechanism — **(i) merely reparametrised, NOT mis-trained**

`steer = arctan(2.9·κ)` is a **smooth, strictly monotone bijection** on the whole physical range, so
the channel carries *exactly* the same information; only its coordinate differs. What makes a
reparametrisation harmless is not that property alone but that **training and inference apply the
SAME function to the SAME tensor**, and they demonstrably do:

* training: `stack/scripts/refa_v1_train.py:627` `actions = b["actions"]` → `:667`
  `model(feats, actions, …)`
* T0 eval: `taniteval/tools/refav1_arm.py:648` `act = b["actions"]` → `:680`
  `model(feats, act[:, :k_wm], …)`

Both `b["actions"]` come from the same `RefAV1Windows._kin_actions`
(`refav1_loader.py:258-264`). **Same function, same tensor, same units ⇒ no checkpoint is
mis-trained, and none needs retraining.** What the model learned is "this number, whatever it is
called, produces this optical flow" — a perfectly learnable and correct thing to learn.

**The refutation condition, stated so this is falsifiable:** an inference path that feeds κ where
training fed steer. Exactly one such path exists — the planner→model crossing, X3
(`refa_v1.py:1832`) — and it is fixed here. That is *not* a training defect; it is an inference
defect, and it makes the *planner* blind, not the *weights* wrong.

### 5.2 The discriminating test — SPECIFIED, RUN, and it returns NO SIGNAL (with the reason)

Test: feed the banked checkpoint the same window twice — `(a, steer)` vs `(a, tan(steer)/2.9)` —
and compare `loss_feat_op` against the recorded future features. If COMMAND wins, the model learned
steer; if GEOMETRY wins, the ruling's premise fails.

**Run on CPU, `ckpt/ckpt.pt` step 1,000, K_wm = 30 (`raw/units_checkpoint_test.csv`):**

| window | command | geometry | winner |
|---|---|---|---|
| 4/4 | 0.451900 (mean) | 0.451900 (mean) | COMMAND 4/4 — but **identical to 6 decimals** |

⛔ **The control says THIS instrument has no signal.** An action-ABLATION probe on both local
checkpoints:

| checkpoint | recorded | actions **zeroed** | κ ≡ 0.3 (R = 3.3 m!) | steer × 10 | max Δ |
|---|---|---|---|---|---|
| `ckpt` step 1000 | 0.44238099 | 0.44253373 | 0.44234648 | 0.44237840 | **1.53 × 10⁻⁴** |
| `ckpt_ep2` (also step 1000) | 0.41706672 | 0.41714469 | 0.41695386 | 0.41705853 | **1.13 × 10⁻⁴** |

**Deleting the entire action tensor moves the pooled 30-step feature loss by 0.03 % relative** —
far below the command-vs-geometry difference I am trying to read. ⇒ the units test is
**UNINFORMATIVE on the checkpoints available to me**, and I refuse to report its 4/4 as evidence.

⚠️ **THIS IS A STATEMENT ABOUT MY INSTRUMENT, NOT ABOUT THE MODEL — and the distinction matters,
because the opposite reading would contradict a MEASURED register row.** `H-REFAV1-LAT-INSENSITIVE`
was **REFUTED** the same day by the ArchInf FlyWheel's anchored-displacement probe
(`…/Architecture & Inference/Research/2026-09-03-anchored-actdiv-refav1/`,
`D-ACTDIV-ANCHORED-REFAV1`): at h = 1 the predictor's response to κ is **2,298×** (fp32 incumbent)
and **11.8×** (clean epoch) its own permutation null, perfectly linear in κ and antisymmetric. The
two readings are consistent and I must not write "the action channel is inert": a real,
sign-correct latent displacement — which that row measures as only **1/300th to 1/40th** the
longitudinal channel's at matched ≈ 2σ levels — is simply invisible inside a pooled 30-step MSE
against the true future, which is dominated by everything else in the scene. **What is measured
here is that `loss_feat_op` is the WRONG READOUT for a units test.** The right readout is the
anchored displacement `d(a) = ẑ(a) − ẑ(0)` that row already ships: run it once with the candidate
in COMMAND units and once in GEOMETRY units and compare which better predicts the realised
displacement. *(Class: `true but wrong for the reader` — a correct measurement that would have
implied a wrong next action, and the second time this session that a scale-blind or
aggregate-blind statistic nearly carried a verdict it could not support.)*

**Admissibility rule for a future run of this test (committed here):** a units test on a checkpoint
is admissible only when its readout's ACTION-ABLATION Δ exceeds the command-vs-geometry Δ by at
least 10×. `loss_feat_op` does not clear that bar at step 1,000 and probably never will;
`actdiv_anchored`'s `d(a)` does clear it by construction (2,298× / 11.8× its null).
**⇒ THE TEST TO RUN, and it is cheap:** re-run `taniteval/tools/actdiv_anchored.py` twice — once
feeding the candidate as `(a, κ)` and once as `(a, arctan(2.9κ))` — and compare each against the
realised displacement. **~2 min/window on CPU, ~40 s/window on the 4060; 20 windows ≈ 15 min CPU,
0 GPU required.** It is also, verbatim, item (3) of the Master Mind's reading of
`D-ACTDIV-ANCHORED-REFAV1` (*"the next probe is therefore the COST SURFACE ITSELF … in BOTH
conventions"*), so it closes two open items at once.

---

## 6. Prediction vs outcome (SPEC §2-§3, scored honestly)

| id | committed | measured | held? |
|---|---|---|---|
| A1 true source exists | find it or prove absence | **FOUND**: `calibration/vehicle_dimensions/*.parquet`, schema read, 20/20 resolved | ✅ |
| A2 `L_enc` | 2.9000 ± 0.0005, pointwise IQR < 1e-3 | **2.9000000 ± 3e-8, IQR 1.2e-7** | ✅ (far tighter) |
| A2 falsifier | a value from {2.73, 2.85, 3.135, 3.165, 3.216} | none | ✅ not triggered |
| A2 control C3 | forward map < 1e-6 rad | **1.49e-8** | ✅ |
| A4 (i) window estimator materially TIGHTER than pointwise | tighter | **REFUTED — sd 0.159 vs 0.126, i.e. WIDER.** My reasoning ("integrating cancels noise") ignored that the window estimator divides by a small Δyaw and is therefore *less* stable on gentle turns | ❌ |
| A4 (ii) 2.730 essentially absent from the fits | absent | **1/15 within 0.05** (vs 4/20 clips truly at 2.730) | ✅ |
| A4 (iii) fit does NOT correlate with the true wheelbase | ρ ≈ 0 | **ρ = −0.60** — not zero, but the *wrong sign*; `per_clip_v1` requires ≈ +1. Reported, not glossed: the conclusion is stronger than predicted, the prediction was still imprecise | ⚠️ partial |
| A4 cost of a constant | < 1 cm | **the constant is BETTER: true-wheelbase +0.025 m, oracle fit +0.011 m on LAT** | ✅ (stronger) |
| B-OFF byte-identical | `torch.equal` | 91 + 95 + 61 tests pass; `torch.equal` asserted | ✅ |
| B-ON reproduces 0.0600 / 0.1199 within ±0.002 | ±0.002 | **0.0600 / 0.1199 — exact to 4 dp** | ✅ |
| B-PIN yaw ratio ON ≈ 1 | — | **0.995** (OFF 2.870) | ✅ |
| D expected answer | (i) reparametrised, not mis-trained | mechanism confirmed at code level; **checkpoint test returns NO SIGNAL** because the action channel is inert at step 1,000 | ⚠️ **half-verified, and said so** |

Two committed expectations were wrong (A4-i outright, A4-iii in magnitude). **No committed VERDICT
was wrong.**

---

## 7. PROPOSED REGISTER ROWS — for `Project Steering/GOALS_AND_CLAIMS.md`

⚠️ **These two rows EXTEND `C-STEER-CURVATURE-INTERFACE`, which the Master Mind already applied
today from the Benchmarks package + the PI ruling. They do not replace it.** What they add is the
half that row leaves open: the wheelbase settled BY CONTENT, and the implementation with its
cross-check and blast radius. **I have NOT edited `GOALS_AND_CLAIMS.md` myself** — the register's
own convention is that a FlyWheel supplies PROPOSED text and the Master Mind applies it verbatim
(*"the three rows below are the FlyWheel's PROPOSED text, applied VERBATIM"*), and that file is
being actively edited today; a concurrent write from me would be a merge hazard, not a service.

```
C-KAPPA-UNIT | CLAIM | MEASURED | 2026-09-03 | Data Engineering FlyWheel
  (EXTENDS C-STEER-CURVATURE-INTERFACE — closes its open wheelbase question)
  The v2ep action channel `actions[:,0]` is a road-wheel STEER angle
  `arctan(L_enc * curvature)` with L_enc = 2.9 EXACTLY, and 2.9 is an ENCODING
  constant, not a vehicle property. Recovered by algebraic inversion of
  `physicalai.signals_at` against its own input (the raw egomotion `curvature`
  column) on 20/20 local eval clips: L_enc in [2.8999999827, 2.9000000313],
  pointwise IQR 1.2e-7; the forward map at exactly 2.9 reproduces the stored
  channel to 1.49e-8 rad. Controls: the reconstructed query grid reproduces
  yaw / ax / x — channels NOT used to build it — to 5.9e-8 rad / 1.2e-7 m/s^2 /
  3.0e-5 m. The clips' TRUE wheelbases (release
  `calibration/vehicle_dimensions/*.parquet`: clip_id, length, width, height,
  rear_axle_to_bbox_center, wheelbase, track_width) are {2.730 x4, 3.135 x2,
  3.165 x10, 3.216 x4}, mean 3.085 — NOT ONE is 2.9, and using the true value
  instead COSTS +0.025 m of curved-window lateral error.
  ⇒ SUPERSEDES the "per-clip wheelbase UNVERIFIED / bimodal ~2.85 and ~3.09"
  item in `…/2026-09-03-refav1-kinematic-contract-lateral/RESULT.md` §5: the
  bimodality is the kappa_pose ESTIMATOR's error, and it is ANTI-correlated
  with the true wheelbase (rho = -0.60, n = 15; per_clip_v1 would require ~ +1).
  Artifacts: raw/lenc_exact_per_clip.csv, raw/true_wheelbase.json,
  raw/per_clip_fits.csv.

D-KAPPA-REPAIR | DECISION | 2026-09-03 | PI ruling, implemented by Data Eng
  (EXTENDS C-STEER-CURVATURE-INTERFACE — the contract, now in code and tested)
  ONE CONTRACT, TWO UNIT DOMAINS. COMMAND (steer) = the v2ep channel, the
  loader, the predictor's input, every trained checkpoint. GEOMETRY (kappa) =
  rollout_unicycle, unicycle_paths, GOAL_KAPPA_*, PlanConfig.kappa_max, every
  metre-valued path. Crossings convert: kappa = tan(steer)/2.9 into geometry,
  steer = arctan(2.9*kappa) into the model.
  (a) THE DATA IS NOT REPAIRED and no cache is rebuilt. `actions[:,0]` stays a
      steering angle; `refav1_loader._kin_actions` is unchanged.
  (b) ⭐ THE LIVE THOR RUN IS **NOT** RESTARTED. Training and inference apply the
      SAME function to the SAME tensor (refa_v1_train.py:627,667 vs
      refav1_arm.py:650,679 -> refav1_loader.py:258-264), so every existing
      checkpoint is REPARAMETRISED, not mis-trained.
  (c) The repair ships FLAG-GATED, DEFAULT OFF: `unicycle_paths(action_units=)`,
      `RefAV1.plan(model_action_units=)`, `refav1_arm --action-units {kappa,steer}`,
      and the bridge in `kinematic.py`. OFF is byte-identical (torch.equal);
      ON reads 0.0600 m curved LAT / 0.1199 m LON against 0.7156 / 0.2395 OFF,
      and the yaw ratio 2.870 -> 0.995. 22 new tests + 91/95/61 existing pass.
  (d) CROSS-CHECK, declared before running: converting at the INTEGRATOR
      reproduces the Benchmarks package's DATA-side conversion to 4 decimals on
      every cell. The two framings are the same algebra; only the site differs.
  (e) OPEN: `taniteval/tools/refcv3_arm.py:477-507` has the same defect and is
      NOT fixed; `t1_eval.DEFAULT_TIERS` still lacks `ha0`; every banked refav1
      `ol`/`ha` LATERAL row is an unconverted replay and must be re-read or
      re-stamped before it is quoted again.
  (f) The empirical half of (b) is NOT yet measured. `loss_feat_op` is the wrong
      readout (action ablation moves it 0.03 %); the admissible instrument is
      `actdiv_anchored`'s d(a), run in BOTH conventions — which is also item (3)
      of D-ACTDIV-ANCHORED-REFAV1's own next-probe list. ~15 min CPU, 0 GPU.
```

---

## 8. ESCALATIONS — decisions and merges someone else must make

| # | item | owner | why it cannot wait |
|---|---|---|---|
| E1 | **When does `--action-units steer` become the default?** Every banked refav1 lateral number is OFF; flipping it silently makes old and new incomparable. Proposal: flip it at the next refav1 T1 read and re-stamp the banked ones as `action_units=kappa`. | Master Mind | the next T1 read will otherwise quote an unconverted `ol` again |
| E0 | **`taniteval/tools/cost_surface_probe.py` and this package are ONE landing.** That file imports `STEER_WHEELBASE_M` / `as_command` from `kinematic.py` and does not import without it. Land them together, or neither. | Master Mind | staging one without the other leaves a tracked file that cannot be imported |
| E2 | **`taniteval/tools/refcv3_arm.py:477-507` has the identical defect**, and so does its incoming copy at `TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-09-03-refcv3-arm-UNVERIFIED/refcv3_arm.py` — **BOTH** need it. Not fixed here (not my ownership); both pick up the new kwarg for free, someone must pass it and re-read refcv3's lateral rows. | refcv3 owner | refcv3 lateral numbers are currently ×2.9 over-rotated |
| E3 | `t1_eval.DEFAULT_TIERS` (`taniteval/tools/t1_eval.py:145`) still does not know `ha0` — carried over unresolved from the Benchmarks package. One line: `"ha0": "T1"`. | t1_eval owner | the standalone `--analyze-only` CLI refuses any dump containing `ha0` |
| E4 | `stack/scripts/stage_a_probes.py:145,168-177` duplicates `WHEELBASE` + `kappa_of_steer`/`steer_of_kappa`, and `w7_roll_rerank.py:166` imports them from there. Consolidating onto `kinematic.STEER_WHEELBASE_M` is correct but touches a script another line imports. | Master Mind | two copies of a constant is how this class of defect starts |
| E5 | **Re-run the §5.2 checkpoint units test** on the first checkpoint whose action-ablation Δ clears 10× the unit Δ. ~15 min CPU, 0 GPU. | whoever banks the next refav1 checkpoint | it is the only *empirical* half of "not mis-trained" |

---

## 9. Families

**LONGITUDINAL** and **LATERAL** are scored throughout (§3.3, §2.4) — both move under the repair.
**TACTICAL** and **STRATEGIC** are **REFUSED with a reason, not omitted**: this package changes a
unit interface and replays recorded/candidate controls; it declares no manoeuvre and no route, so
those families have no arm to score here. The refusal is itself the finding — the interface repair
is *upstream* of them, and every tactical/strategic read that depended on a planner trajectory must
be re-taken after E1 lands.

---

## 10. DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| `SPEC.md` (written before measuring) | `repo:TanitAD Research Lab/Data Engineering/Research/2026-09-03-steer-curvature-interface/SPEC.md` | no |
| `RESULT.md` (this file) | `repo:…/2026-09-03-steer-curvature-interface/RESULT.md` | no |
| `raw/lenc_exact_per_clip.csv`, `raw/lenc_exact.json` — the exact L_enc inversion, 20 clips + controls | `repo:…/raw/` | no |
| `raw/true_wheelbase.json`, `raw/true_wheelbase_per_clip.csv` — the TRUE source, schema + per-clip (UUID-free, ordinal-keyed) | `repo:…/raw/` | no |
| `raw/per_clip_fits.csv` — the trajectory fits + true wheelbase + L_enc, per clip | `repo:…/raw/` | no |
| `raw/interface_per_window.csv` — 140 windows × {OFF, ON, true-WB, oracle-L} × {ol, ha, ha0} | `repo:…/raw/` | no |
| `raw/interface_summary.json` — the §3.3 and §2.3 aggregates | `repo:…/raw/` | no |
| `raw/units_checkpoint_test.csv` — the §5.2 checkpoint units test | `repo:…/raw/` | no |
| `raw/probe_lenc.py`, `raw/probe_truewb.py`, `raw/probe_interface.py`, `raw/probe_units_ckpt.py` — every probe, reproducible | `repo:…/raw/` | no |
| **CODE — the bridge** | `repo:stack/tanitad/models/kinematic.py` | no |
| **CODE — `unicycle_paths(action_units=)`** | `repo:stack/tanitad/refs/refa_v1_plan.py` | no |
| **CODE — `plan(model_action_units=)`** | `repo:stack/tanitad/refs/refa_v1.py` | no |
| **CODE — `--action-units`, `paths_from_controls`, manifest provenance** | `repo:taniteval/tools/refav1_arm.py` | no |
| **TESTS — 22, all passing** | `repo:stack/tests/test_steer_curvature_interface.py` | no |
| UUID-bearing per-clip wheelbase join (GATED-CONFIDENTIAL) | `scratchpad:true_wheelbase_per_clip_UUID.csv` | **YES — deliberately NOT staged** (clip UUIDs never enter the repo; the ordinal-keyed table is staged instead) |
| HF `vehicle_dimensions` parquet cache (19 chunks) | `scratchpad:vd_cache/` | **YES — deliberately not staged** (re-derivable via `raw/probe_truewb.py`) |

**Nothing that took real effort lives only on a pod or only in a worktree.** No pod was contacted.
