# THE PSEUDO-SIMULATION ARM PANEL — the first admissible closed-loop ranking in TanitAD's history

**Date:** 2026-07-27 (Europe/Berlin) · **Stream:** Benchmarks & Eval · **Branch:** `agent/benchmarks-eval-20260721`
**Host:** `tanitad-pod2` (A40) only. ⛔ pod1 (v2corpus training), pod3, and the eval pod were **not touched**.
**Instrument:** `taniteval/taniteval/pseudosim.py` (md5 `86a11c46…`), imported and **not reimplemented**.
**Pre-registration:** `PRE_REGISTRATION.md` in this directory, written before any new arm was scored, frozen.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not re-verified)
· `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ⭐⭐ **ALL THREE PUBLISHED NUMBERS REPRODUCE EXACTLY ON A DIFFERENT HOST.** `v4_oracle` **0.5622 [0.5496, 0.5725]**, `v4_blind` **0.3749 [0.3076, 0.4368]**, `cv_holdv0` **0.5705 [0.5558, 0.5844]** — identical to 4 decimals *and* identical in `n_windows` (15 662 / 15 668 / 15 478) and `n_episodes`. The instrument gate reproduces at **+0.1882 [+0.1240, +0.2557] SEPARATED**; the v4-vs-CV fidelity gate reproduces at **−0.0034 [−0.0138, +0.0078] n.s.** and recovery **−0.0168 [−0.0332, −0.0008] SEPARATED**. | `MEASURED` **tier 2** |
| **2** | ⭐⭐ **THE PRIOR REPORT'S "v1 IS NOT A PLANNER" IS TRUE OF ONE SURFACE AND FALSE OF THE MODEL — v1 IS IN THE PANEL.** `strategic_policy → tactical_policy → waypoints` is **state-only**: no future actions, no expert control sequence. It loads and plans on `flagship4b-speedjerk-30k` (md5 `b5f07d9e…`, step 29999). Pre-registered outcome **V1-IN**. | `MEASURED` **tier 1** (it ran) |
| **3** | ⛔⛔ **CONSTANT VELOCITY IS AT THE TOP OF THE PANEL AND NO LEARNED ARM BEATS IT.** `cv_holdv0` **0.5705** is the highest score of any arm. It is **separated ABOVE REF-C at ALL THREE SCALES** — base **+0.0252 [+0.0150, +0.0349]**, small **+0.0255 [+0.0146, +0.0358]**, XL **+0.0203 [+0.0097, +0.0303]** — and above the no-speed control **+0.0235 [+0.0037, +0.0430]**; it is **not separated from** v4 (+0.0034) or v1 (+0.0182 / +0.0178). **The best case for the program is a tie with a zero-parameter baseline.** | `MEASURED` **tier 2** |
| **4** | ⛔⛔ **THE SPEED CHANNEL — the program's single largest measured effect, 2.918 m → 0.452 m ADE (6.5×) — IS INVISIBLE HERE.** `nospeed_tactical_oracle` (the ablation control) vs `v1_tactical_oracle`: **−0.0055 [−0.0130, +0.0011] n.s.** And the mechanism is not mysterious: `tactical_policy(states, ctx)` **takes no action channel at all**, so `--speed-input` can only reach the plan through the trained representation. **The 6.5× was measured on a surface the planner does not use.** | `MEASURED` **tier 2** + mechanism |
| **5** | ⭐⭐ **THE REF-C SCALE LADDER SPLITS: the base-vs-small NULL REPRODUCES, but XL SEPARATES ABOVE BOTH — a scale effect open-loop ADE could not see.** base − small = **+0.0002 [−0.0024, +0.0025] n.s.** (the tightest interval in the panel), matching the registry's ADE null. But **XL − base = +0.0050 [+0.0023, +0.0082] SEP** and **XL − small = +0.0055 [+0.0027, +0.0084] SEP**, where the registry's open-loop ADE contrast was **+0.0013 [−0.0281, +0.0316] n.s.** ⚠️ **Confounded** — XL trained on v1 route labels, base/small on v2.1 (registry §4.3), so this conflates scale, anchor count and labels. | `MEASURED` **tier 2**, ⚠️ confounded |
| **6** | ⭐ **THE PRE-REGISTERED DISCRIMINATIVE-POWER VERDICT IS `D-PARTIAL`, NOT `D-NULL`.** **12 of 28** pairwise contrasts among the non-blind, non-adversary arms separate. So a **partial order** is publishable and a **total ranking is not** — and I do not write one. The un-separated pairs are named in §5.1. | `MEASURED` **tier 2** |
| **7** | ⛔ **THE PUBLISHED "flagship-v4 30 k" IS `flagship-v4-fromscratch-30k`, NOT `flagship-v4-30k`.** The run.log of the 2026-07-27 report loads `/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json`. `MODEL_REGISTRY` §1.5.1 records `flagship-v4-30k` as **KILLED ~step 3 500** — and pod2's copy of it stops at **step 4 400**. **I started the panel on the wrong checkpoint and caught it from the run log**, not from the artifact, which names neither. | `MEASURED` **tier 1** |
| **8** | ⚠️ **THE "ORACLE nav" FED TO v1 AND THE NO-SPEED CONTROL IS 75 % DEGENERATE.** `refb_labels.nav_command` needs 15–25 s of future; PhysicalAI clips are ~20 s, so **6 000 of 7 964 indices** return `(NAV_FOLLOW, valid=False)`. The `v1_tactical_follow` control is run to bound it. **This is a weaker goal than v4's, and the v4-vs-v1 contrast conflates the planner with the goal interface.** | `MEASURED` **tier 1** |
| **9** | ⭐ **THE STANDING-STILL ADVERSARY IS REFUSED OUTRIGHT — the strongest possible form of the gate.** `stand_still` gets `recovery` **defined on 0 of 15 981 rows**, `ego_progress` identically 0, `comfort` saturated at 1.0, and `composite()` raises `VacuousMetric`: **no score at all**, not a high one. The pre-fix metric scored this shape **+0.597 ABOVE** a sighted arm. | `MEASURED` **tier 1** |
| **10** | ⚠️ **A PANEL-WIDE GATE HAD TO BE ADDED, AND IT MOVES EVERY CONCLUSION BY UP TO 5×.** The shipped per-arm gate admits `comfort` for REF-C/v1 (pass-rate 1.25 %) but drops it for v4 (all-zero) and CV (all-one) — **different weight sets per arm**. Under the per-arm gate `cv − refc_base` reads **+0.1303**; under the panel gate it is **+0.0252**. Both are reported. | `MEASURED` **tier 1** |
| **11** | ⚠️ **A PERFORMANCE TRAP THAT LOOKED LIKE A HANG.** torch spawned **113 threads per process** (one OMP thread per core × 7 arms) and the panel made **no measurable progress in 50 minutes** at GPU `sm` 0–6 %. With `OMP_NUM_THREADS=6` the identical `cv_holdv0` arm finished in **232 s** — *faster than the published 417 s*. Not a numerical change; a scheduling one. | `MEASURED` **tier 1** |

### 0.1 The verdict in one sentence

**The instrument ports exactly, passes all five of its own gates on this host, and produces the
program's first admissible closed-loop-class ranking — and that ranking puts a zero-parameter
constant-velocity baseline first, separates it above REF-C at every scale and above the no-speed
control, and leaves the two flagship arms merely tied with it.** ⛔ **On the only closed-loop surface
we own that is a MEASUREMENT rather than an EXTRAPOLATION, nothing this program has trained beats
holding v₀.**

### 0.2 The panel's tier, stated as required

**Tier 2**, and it may not be quoted without all four qualifiers: **oracle goal** (an upper bound for
the flagship arms, not deployable), **non-reactive log replay**, **no collision or TTC gate** (the val
cache has no cuboids — emitted as `None` with a reason, never a constant), and **`comfort` dropped by
the gate, not retuned**. The composite is `PSS_recovery_progress` and **is not a Driving Score**.
The envelope proof (§2, gates G3/G4) is **tier 1**: deterministic, model-free, CPU, seconds.

---

## 1. Pre-registration (verbatim pointer)

`PRE_REGISTRATION.md`, frozen before any new arm was scored, committed the five gates (G1–G5), the
**D-NULL / D-PARTIAL / D-RANK** discriminative-power criterion, both outcomes of the **V1-IN / V1-OUT**
question, and the tier. Nothing below reinterprets it after the fact. The two readings that could
have killed the panel — **G1 fail** (no arm score admissible) and **D-NULL** (no ranking publishable)
— are stated there with the exact sentences I would have written.

**What fired:** G1 ✅ · G2 ✅ · G3 ✅ · G4 ✅ · G5 ✅ · **V1-IN** ✅ · **D-PARTIAL**.

---

## 2. The instrument's own gates, on this host

### 2.1 G3 — the envelope assertion (tier 1, runs before any checkpoint is loaded)

MEASURED identically on **every one of the 10 arm processes**, printed before the model touch:

```
frac_steps_lat_over_3m                 = 0.0
frac_steps_yaw_over_12deg              = 0.0
frac_steps_any                         = 0.0
frac_windows_any_step_out_of_envelope  = 0.0
EXTRAPOLATION_VERDICT → 'MEASUREMENT — every step stayed inside the MEASURED envelope'
```

Against the same corpus and the **same** `ood.envelope_fractions` function, the sequential loop reads
**12.26 %** of windows outside at the standing gate horizon K=20 and **90.24 %** at K=185
(`PUBLISHED`, `…/2026-07-27-pseudo-simulation/artifacts/before_after_envelope.json`). This is why the
panel is a measurement and every previous closed-loop ranking was not.

### 2.2 G4 — validation in the FAILING direction, exercised on pod2

Every arm process runs the falsifier before it runs the arm:

| input | result | MEASURED |
|---|---|:--:|
| `GridSpec(dyaw_deg=(12.0,))` — the envelope edge | **accepted** | ✅ |
| `GridSpec(dyaw_deg=(12.0012,))` — `ENV_YAW_MAX × 1.0001` | **raises `EnvelopeViolation`** | ✅ |
| `GridSpec(dlat_m=(1.0,))` — the refused axis | **raises `LateralAxisRefused`** | ✅ |

`{'edge_value_accepted': True, 'just_outside_raises': True, 'G4_PASS': True}` appears in all 10 logs
and in every `arm_*.json` under `_meta.G4_falsifier_exercised`. **A grid 0.0012° too wide is refused.**

### 2.3 G1 — instrument sensitivity, the only clause that licenses the panel

> **`v4_oracle − v4_blind` PSS = +0.1882 [+0.1240, +0.2557] · ⭐ SEPARATED**
> paired episode-cluster bootstrap, B=2000, unit = val episode, identical rows.

Published on the eval pod: **+0.1882 [+0.1240, +0.2557]**. **Identical to 4 decimals.** The two arms
differ in exactly one thing — the image is zeroed — and the heading perturbation is visible only in
the image. **G1 fires, so §4 is admissible.** Had it not, §4 would have reported the failure and no
arm score (pre-registered R-b / G1-fail).

⚠️ **Inherited confound, restated because it bounds the claim:** `v4_blind` keeps the **oracle goal**,
so it is blind to the image but not to its own future. A doubly-blind control is the fix and was not
run.

### 2.4 G2 — port fidelity: three exact reproductions

| quantity | published (`tanitad-eval`) | **MEASURED here (`pod2`)** | identical? |
|---|---|---|:--:|
| `v4_oracle` PSS | 0.5622 [0.5496, 0.5725] · n 15 662 / 40 | **0.5622 [0.5496, 0.5725] · n 15 662 / 40** | ✅ |
| `v4_blind` PSS | 0.3749 [0.3076, 0.4368] · n 15 668 / 40 | **0.3749 [0.3076, 0.4368] · n 15 668 / 40** | ✅ |
| `cv_holdv0` PSS | 0.5705 [0.5558, 0.5844] · n 15 478 / 40 | **0.5705 [0.5558, 0.5844] · n 15 478 / 40** | ✅ |
| `v4_oracle − v4_blind` PSS | +0.1882 [+0.1240, +0.2557] SEP | **+0.1882 [+0.1240, +0.2557] SEP** | ✅ |
| `v4_oracle − cv_holdv0` PSS | −0.0034 [−0.0138, +0.0078] n.s. | **−0.0034 [−0.0138, +0.0078] n.s.** | ✅ |
| `v4_oracle − cv_holdv0` recovery | −0.0168 [−0.0332, −0.0008] SEP | **−0.0168 [−0.0332, −0.0008] SEP** | ✅ |

Different host, different GPU, different thread count, a re-pulled checkpoint — and the numbers are
bit-stable. **The pre-registered fidelity gate passes without qualification**, so nothing new below is
quarantined.

### 2.5 G5 — the standing-still adversary, run as a full arm

The metric was once gameable by not moving: with a `v0 × horizon` denominator it scored the **BLIND**
arm **+0.597 ABOVE** the sighted one. That fix was verified on a 2-episode smoke; here it is verified
on the **whole panel**, as an arm.

| component | `stand_still`, 40 episodes × 21 grid points | reading |
|---|---|---|
| `recovery` | **0 finite values of 15 981** (`defined_fraction = 0.000000`) | every row **excluded**, none scored 1.0 |
| `ego_progress` | min 0.0, max 0.0, mean 0.0 | ⛔ `range below range_min` |
| `comfort` | min 1.0, max 1.0 | ⛔ `SATURATED at the ceiling` — *a stopped car is perfectly comfortable* |
| **composite** | ⛔ **`VacuousMetric` — REFUSED TO EMIT** | **no score at all** |

⭐ And the number that shows why the naive metric was wrong: `stand_still` has the **smallest mean
cross-track error in the entire panel — 1.007 m**, against 3.45 m for CV and 3.53 m for v4. *Under the
pre-fix denominator that would have been the best recovery score on the board.*

---

## 3. What changed in the method, and why (both changes are disclosed, neither is optional)

### 3.1 ⭐ The PANEL GATE — a component enters the composite only if it is admissible for EVERY arm

**Stated as a rule before it was applied; forced by a MEASURED defect, not by a preference.**

The shipped discriminative-range gate is evaluated **per arm**. On this panel that produces
**different weight sets for different arms**:

| arm | `ego_progress` | `recovery` | `comfort` | per-arm verdict on comfort |
|---|:--:|:--:|:--:|---|
| `v4_oracle` | ✅ | ✅ | ⛔ | all rows 0.0 → `range below range_min` |
| `cv_holdv0` | ✅ | ✅ | ⛔ | all rows 1.0 → `SATURATED at the ceiling` |
| `v1_tactical_oracle` | ✅ | ✅ | **✅** | pass-rate 0.25 % → *admissible* |
| `refc_base_produced` | ✅ | ✅ | **✅** | pass-rate 1.25 % → *admissible* |

A paired delta between a 3-component composite and a 2-component composite **is not a model
comparison** — it mixes a metric change with a model change. So: **admissible for every candidate arm,
or dropped from every candidate arm.** `comfort` is dropped, which is exactly where the published run
left it — **this is not retuning a bound after seeing who fails; it is refusing to compare two
different objects.**

⚠️ **The rule is load-bearing, and the artifact publishes both sides.** Under the per-arm gate the
same contrasts read:

| contrast | **panel gate (used)** | per-arm gate (sensitivity) | factor |
|---|---|---|---:|
| `cv_holdv0 − refc_base` | **+0.0252 [+0.0150, +0.0349]** SEP | +0.1303 SEP | **5.2×** |
| `cv_holdv0 − refc_small` | **+0.0255 [+0.0146, +0.0358]** SEP | +0.1318 SEP | 5.2× |
| `cv_holdv0 − v1_tactical_oracle` | **+0.0182 [−0.0019, +0.0373] n.s.** | +0.1260 **SEP** | ⛔ **verdict flip** |
| `v1_tactical_oracle − v4_oracle` | **−0.0147 [−0.0274, −0.0028]** SEP | −0.1219 SEP | 8.3× |
| `v4_oracle − cv_holdv0` | **−0.0034** n.s. | −0.0034 n.s. | 1.0× (both drop comfort) |
| `refc_xl − refc_base` | **+0.0050 [+0.0023, +0.0082]** SEP | +0.0029 SEP | 1.7× — **same verdict both ways** |

Every `_PSS_under_per_arm_gate_SENSITIVITY` block is in the artifact. ⭐ Note the last row: the §5.5
REF-C-XL finding is the one conclusion that **survives both gate choices**, which is why it is stated
as a result (with its label confound) and the `cv − v1` verdict is stated as *unsupported*.

### 3.2 The adversary does not vote on the gate

`stand_still` is inadmissible on **every** component by construction. Left in the intersection it
would make the composite vacuous for every real arm — **the probe would delete the metric it exists to
test.** It is scored, reported, and excluded from the gate and from the ranking
(`PANEL_GATE.validation_probes_excluded_from_the_gate`).

### 3.3 The densification adapter cannot move either admissible component

The flagship tactical head and REF-C both emit 4 waypoints at 5/10/15/20 steps;
`taniteval.closedloop.densify_plan` interpolates them onto the 20-step 10 Hz grid. Both admissible
components read **only index −1**, and interpolation through the knots reproduces the 20-step knot
exactly. **Asserted numerically at run time in every arm process**
(`_meta.densify_endpoint_max_err = 0.0`).

---

## 4. THE PANEL

40 val episodes · stride 8 · 21 grid points (7 heading × 3 longitudinal, **lateral refused in code**)
· **15 981 planner calls per arm, 0 rollout steps** · `traffic_mode: log_replay_nonreactive`.
Estimator: `taniteval.ci.episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`, B = 2000,
unit = **val episode**, paired on **identical `(episode, anchor, dlat, dyaw, dlon)` rows — asserted,
not assumed** (`row_identity` in the artifact; any arm whose key sequence differs is **refused**, not
silently dropped).

<!-- PANEL_TABLES_START — generated by scripts/summarize_panel.py from artifacts/pseudosim_arm_panel.json; DO NOT hand-edit -->

### Panel gate

admitted: `['ego_progress', 'recovery']`  ·  dropped: `['comfort']`

* `comfort` — INADMISSIBLE for cv_holdv0, v4_blind, v4_oracle -> dropped from EVERY arm so the composite is the same object on both sides of every paired delta

### Arm scores

| arm | n evals | n eps | goal provenance | `ego_progress` | `recovery` | **PSS_recovery_progress** |
|---|---:|---:|---|---|---|---|
| `cv_holdv0` | 15981 | 40 | none | 0.9407 [0.9169, 0.9610] | 0.0776 [0.0604, 0.0959] | **0.5705 [0.5558, 0.5844]** |
| `v4_oracle` | 15981 | 40 | ORACLE | 0.9462 [0.9320, 0.9591] | 0.0629 [0.0449, 0.0816] | **0.5622 [0.5496, 0.5725]** |
| `refc_xl_produced` | 15981 | 40 | none | 0.9438 [0.9279, 0.9585] | 0.0259 [0.0150, 0.0378] | **0.5499 [0.5421, 0.5566]** |
| `v1_tactical_follow` | 15981 | 40 | CONSTANT `follow` | 0.9081 [0.8871, 0.9279] | 0.0785 [0.0598, 0.0985] | **0.5471 [0.5340, 0.5595]** |
| `v1_tactical_oracle` | 15981 | 40 | ORACLE nav command | 0.9047 [0.8834, 0.9247] | 0.0817 [0.0621, 0.1024] | **0.5467 [0.5338, 0.5591]** |
| `refc_small_produced` | 15981 | 40 | none | 0.9315 [0.9120, 0.9485] | 0.0296 [0.0184, 0.0421] | **0.5444 [0.5360, 0.5514]** |
| `refc_base_produced` | 15981 | 40 | none | 0.9317 [0.9112, 0.9493] | 0.0293 [0.0175, 0.0422] | **0.5439 [0.5345, 0.5519]** |
| `nospeed_tactical_oracle` | 15981 | 40 | ORACLE nav command | 0.8961 [0.8727, 0.9177] | 0.0800 [0.0610, 0.1004] | **0.5394 [0.5242, 0.5540]** |
| `v4_blind` | 15981 | 40 | ORACLE | 0.5999 [0.4857, 0.7050] | 0.1159 [0.0913, 0.1426] | **0.3749 [0.3076, 0.4368]** |
| `stand_still` | 15981 | 40 | none | 0.0000 [0.0000, 0.0000] | — | **—** |

### Diagnostics

| arm | recovery defined frac | mean along-track end (m) | mean cross-track end (m) | wallclock s | planner calls | rollout steps |
|---|---:|---:|---:|---:|---:|---:|
| `cv_holdv0` | 0.8203 | 24.944 | 3.448 | 232.2 | 15981 | 0 |
| `v4_oracle` | 0.8377 | 24.859 | 3.531 | 3137.8 | 15981 | 0 |
| `refc_xl_produced` | 0.8256 | 24.957 | 4.264 | 3470.3 | 15981 | 0 |
| `v1_tactical_follow` | 0.8423 | 26.356 | 3.604 | 1973.1 | 15981 | 0 |
| `v1_tactical_oracle` | 0.8419 | 26.041 | 3.567 | 2705.5 | 15981 | 0 |
| `refc_small_produced` | 0.8253 | 24.775 | 4.257 | 1901.3 | 15981 | 0 |
| `refc_base_produced` | 0.8250 | 24.758 | 4.257 | 2606.5 | 15981 | 0 |
| `nospeed_tactical_oracle` | 0.8451 | 26.060 | 3.563 | 2853.7 | 15981 | 0 |
| `v4_blind` | 0.5748 | 21.092 | 3.192 | 3304.5 | 15981 | 0 |
| `stand_still` | 0.0000 | 0.000 | 1.007 | 262.5 | 15981 | 0 |

### Paired vs `v4_oracle` (episode-cluster bootstrap, B=2000, identical rows)

| arm | Δ `ego_progress` | Δ `recovery` | **Δ PSS** |
|---|---|---|---|
| `cv_holdv0` − `v4_oracle` | **-0.0055** [-0.0205, +0.0081] n.s. | **+0.0168** [+0.0008, +0.0332] ⭐ SEP | **+0.0034** [-0.0078, +0.0138] n.s. |
| `refc_xl_produced` − `v4_oracle` | **-0.0024** [-0.0114, +0.0061] n.s. | **-0.0360** [-0.0481, -0.0263] ⭐ SEP | **-0.0169** [-0.0238, -0.0105] ⭐ SEP |
| `v1_tactical_follow` − `v4_oracle` | **-0.0381** [-0.0625, -0.0142] ⭐ SEP | **+0.0152** [+0.0016, +0.0300] ⭐ SEP | **-0.0140** [-0.0268, -0.0019] ⭐ SEP |
| `v1_tactical_oracle` − `v4_oracle` | **-0.0415** [-0.0660, -0.0172] ⭐ SEP | **+0.0181** [+0.0048, +0.0331] ⭐ SEP | **-0.0147** [-0.0274, -0.0028] ⭐ SEP |
| `refc_small_produced` − `v4_oracle` | **-0.0147** [-0.0266, -0.0029] ⭐ SEP | **-0.0322** [-0.0450, -0.0214] ⭐ SEP | **-0.0216** [-0.0284, -0.0149] ⭐ SEP |
| `refc_base_produced` − `v4_oracle` | **-0.0145** [-0.0269, -0.0027] ⭐ SEP | **-0.0319** [-0.0443, -0.0216] ⭐ SEP | **-0.0217** [-0.0293, -0.0141] ⭐ SEP |
| `nospeed_tactical_oracle` − `v4_oracle` | **-0.0502** [-0.0771, -0.0243] ⭐ SEP | **+0.0164** [+0.0037, +0.0304] ⭐ SEP | **-0.0201** [-0.0345, -0.0068] ⭐ SEP |
| `v4_blind` − `v4_oracle` | **-0.3464** [-0.4593, -0.2437] ⭐ SEP | **+0.0564** [+0.0314, +0.0811] ⭐ SEP | **-0.1882** [-0.2557, -0.1240] ⭐ SEP |
| `stand_still` − `v4_oracle` | **-0.9462** [-0.9591, -0.9320] ⭐ SEP | — | — |

### Paired vs `cv_holdv0` (episode-cluster bootstrap, B=2000, identical rows)

| arm | Δ `ego_progress` | Δ `recovery` | **Δ PSS** |
|---|---|---|---|
| `v4_oracle` − `cv_holdv0` | **+0.0055** [-0.0081, +0.0205] n.s. | **-0.0168** [-0.0332, -0.0008] ⭐ SEP | **-0.0034** [-0.0138, +0.0078] n.s. |
| `refc_xl_produced` − `cv_holdv0` | **+0.0031** [-0.0058, +0.0131] n.s. | **-0.0528** [-0.0711, -0.0352] ⭐ SEP | **-0.0203** [-0.0303, -0.0097] ⭐ SEP |
| `v1_tactical_follow` − `cv_holdv0` | **-0.0326** [-0.0633, -0.0013] ⭐ SEP | **-0.0017** [-0.0156, +0.0127] n.s. | **-0.0178** [-0.0368, +0.0022] n.s. |
| `v1_tactical_oracle` − `cv_holdv0` | **-0.0360** [-0.0666, -0.0052] ⭐ SEP | **+0.0017** [-0.0132, +0.0167] n.s. | **-0.0182** [-0.0373, +0.0019] n.s. |
| `refc_small_produced` − `cv_holdv0` | **-0.0092** [-0.0177, +0.0003] n.s. | **-0.0497** [-0.0672, -0.0323] ⭐ SEP | **-0.0255** [-0.0358, -0.0146] ⭐ SEP |
| `refc_base_produced` − `cv_holdv0` | **-0.0090** [-0.0160, -0.0007] ⭐ SEP | **-0.0490** [-0.0674, -0.0307] ⭐ SEP | **-0.0252** [-0.0349, -0.0150] ⭐ SEP |
| `nospeed_tactical_oracle` − `cv_holdv0` | **-0.0447** [-0.0753, -0.0129] ⭐ SEP | **-0.0004** [-0.0134, +0.0138] n.s. | **-0.0235** [-0.0430, -0.0037] ⭐ SEP |
| `v4_blind` − `cv_holdv0` | **-0.3409** [-0.4501, -0.2420] ⭐ SEP | **+0.0403** [+0.0211, +0.0589] ⭐ SEP | **-0.1939** [-0.2595, -0.1332] ⭐ SEP |
| `stand_still` − `cv_holdv0` | **-0.9407** [-0.9610, -0.9169] ⭐ SEP | — | — |

### Paired vs `v1_tactical_oracle` (episode-cluster bootstrap, B=2000, identical rows)

| arm | Δ `ego_progress` | Δ `recovery` | **Δ PSS** |
|---|---|---|---|
| `cv_holdv0` − `v1_tactical_oracle` | **+0.0360** [+0.0052, +0.0666] ⭐ SEP | **-0.0017** [-0.0167, +0.0132] n.s. | **+0.0182** [-0.0019, +0.0373] n.s. |
| `v4_oracle` − `v1_tactical_oracle` | **+0.0415** [+0.0172, +0.0660] ⭐ SEP | **-0.0181** [-0.0331, -0.0048] ⭐ SEP | **+0.0147** [+0.0028, +0.0274] ⭐ SEP |
| `refc_xl_produced` − `v1_tactical_oracle` | **+0.0391** [+0.0144, +0.0652] ⭐ SEP | **-0.0543** [-0.0722, -0.0393] ⭐ SEP | **-0.0019** [-0.0157, +0.0120] n.s. |
| `v1_tactical_follow` − `v1_tactical_oracle` | **+0.0034** [+0.0008, +0.0069] ⭐ SEP | **-0.0032** [-0.0056, -0.0013] ⭐ SEP | **+0.0004** [-0.0008, +0.0018] n.s. |
| `refc_small_produced` − `v1_tactical_oracle` | **+0.0268** [+0.0005, +0.0545] ⭐ SEP | **-0.0515** [-0.0684, -0.0367] ⭐ SEP | **-0.0072** [-0.0210, +0.0075] n.s. |
| `refc_base_produced` − `v1_tactical_oracle` | **+0.0270** [+0.0002, +0.0550] ⭐ SEP | **-0.0504** [-0.0676, -0.0352] ⭐ SEP | **-0.0067** [-0.0208, +0.0078] n.s. |
| `nospeed_tactical_oracle` − `v1_tactical_oracle` | **-0.0086** [-0.0209, +0.0027] n.s. | **-0.0019** [-0.0093, +0.0038] n.s. | **-0.0055** [-0.0130, +0.0011] n.s. |
| `v4_blind` − `v1_tactical_oracle` | **-0.3048** [-0.4229, -0.1949] ⭐ SEP | **+0.0349** [+0.0153, +0.0569] ⭐ SEP | **-0.1744** [-0.2440, -0.1080] ⭐ SEP |
| `stand_still` − `v1_tactical_oracle` | **-0.9047** [-0.9247, -0.8834] ⭐ SEP | — | — |

<!-- PANEL_TABLES_END -->

---

## 5. Reading — what the ranking is, and how confident I am

### 5.1 The pre-registered discriminative verdict: **D-PARTIAL**

Of the **28** pairwise paired-PSS contrasts among the non-blind, non-adversary arms, **12 separate**.
That is neither D-NULL (no ranking at all) nor D-RANK (a total order). Per the pre-registration I
publish **only the separated contrasts** and name the rest as unsupported. **I do not chain n.s.
contrasts into a leaderboard.**

**The partial order that IS supported** (Δ PSS, paired, identical rows):

```
 cv_holdv0 0.5705  ≈  v4_oracle 0.5622                       (Δ +0.0034 n.s.)
      │                    │
      │                    ├─ SEP ABOVE ─► refc_xl        −0.0169 [−0.0238, −0.0105]
      │                    ├─ SEP ABOVE ─► v1_follow      −0.0140 [−0.0268, −0.0019]
      │                    ├─ SEP ABOVE ─► v1_oracle      −0.0147 [−0.0274, −0.0028]
      │                    ├─ SEP ABOVE ─► refc_small     −0.0216 [−0.0284, −0.0149]
      │                    ├─ SEP ABOVE ─► refc_base      −0.0217 [−0.0293, −0.0141]
      │                    └─ SEP ABOVE ─► nospeed        −0.0201 [−0.0345, −0.0068]
      │
      ├─ SEP ABOVE ─► refc_xl    −0.0203 · refc_base −0.0252 · refc_small −0.0255
      ├─ SEP ABOVE ─► nospeed    −0.0235
      └─ NOT separated from ──► v4_oracle (+0.0034) · v1_oracle (+0.0182) · v1_follow (+0.0178)

 refc_xl 0.5499 ─ SEP ABOVE ─► refc_base (+0.0050) · refc_small (+0.0055)   ⚠️ label-confounded
 refc_base 0.5439 ≈ refc_small 0.5444                        (Δ +0.0002 n.s.)
 v1_oracle 0.5467 ≈ v1_follow 0.5471                         (Δ +0.0004 n.s.)

 EVERY scoring arm ── SEP ABOVE ──► v4_blind 0.3749          (+0.168 to +0.194)
 stand_still ─────── NO SCORE AT ALL (VacuousMetric)
```

**Unsupported orderings, stated explicitly** (all n.s., no ranking claimed): `cv` vs `v1` (either
goal) · `v1` vs REF-C at any scale · `v1` vs the no-speed control · the no-speed control vs REF-C at
any scale · REF-C-base vs REF-C-small · `v1_oracle` vs `v1_follow`.

### 5.2 ⛔ The finding that should decide something: **the floor wins**

`cv_holdv0` — zero parameters, no image, no goal, no training — has the **highest PSS in the panel**
and is **separated above every REF-C arm and above the no-speed control**. The two flagship arms are
**tied** with it, and one of them (`v4_oracle`) is tied only with an **ORACLE goal minted from its own
future**.

This is not a new claim; it is the program's standing open-loop finding — *no arm beats hold-v₀ at
cruising* — now **stated on a closed-loop-class surface that is a MEASUREMENT**, with a bounded
perturbation the arms have to react to, and with the ranking quantity that the field says predicts
closed-loop performance (`ego_progress` ρ = 0.83) rather than the one it says does not (L2/ADE
ρ = −0.36, p = 0.43).

⚠️ **The honest counterweight, stated at the same volume (inherited from the prior report and
confirmed here): constant velocity is a strong baseline at a 2 s horizon by construction.** Over 2 s
most logged driving *is* nearly constant-velocity, so `ego_progress` has little room. **The
instrument's *sensitivity* is demonstrated (G1, +0.1882); its *resolution between competent planners*
is demonstrated only partially (7/15).** The direct remedy is a longer scoring horizon, which costs
**nothing** envelope-wise on this protocol — the cap is the 20-waypoint head, not the protocol.

### 5.3 ⭐ v1 is in the panel — and its headline `ade_0_2s` 0.4271 describes a different object

Pre-registered outcome **V1-IN**. The state-only plan step
(`strategic_policy → tactical_policy → waypoints`) loads and runs on `flagship4b-speedjerk-30k`.

**And the consequence I committed to in advance:** v1's published **`ade_0_2s` 0.4271 (full-set)** is
produced by `taniteval/rollout.py`, which feeds `rollout_decode(…, fa, …)` — **the expert's true
future actions**. That module's own record says `actions_source="expert_future"`,
`pc2_pass = False by construction`, and `honest_metric_name = "wm_fidelity_ade_2s"`. **0.4271 is a
world-model fidelity number for a known control sequence.** The number in this panel is v1's
**planner**. They are different objects and a gap between them is **not a regression** — it is the
program having ranked its arms on a surface where the answer was supplied.

⚠️ **What this does NOT license.** It does not make 0.4271 wrong; it makes it *not a planning number*.
And it does not make the prior report's §9.8 dishonest — that claim is **exactly right about
`taniteval/rollout.py`**. It is wrong only as a statement about the model, which is the class of error
the operating standard's rule 2 exists to catch: *absence found at one location is not absence.*

### 5.4 ⛔ The speed channel does not reach the planner

`nospeed_tactical_oracle` is the causal control for the program's largest measured effect: identical
architecture and data, `--speed-input` the only difference, **2.918 m vs 0.452 m ADE — 6.5×**. On this
panel:

| contrast | PSS |
|---|---|
| `nospeed − v1_tactical_oracle` | **−0.0055 [−0.0130, +0.0011] n.s.** |
| `nospeed − refc_base` | +0.0015 [−0.0142, +0.0163] n.s. |
| `nospeed − v4_oracle` | −0.0201 [−0.0345, −0.0068] SEP |

**The 6.5× ablation is invisible.** And the mechanism is legible rather than mysterious:
`TacticalPolicy.forward(states, ctx)` has **no action input** — `--speed-input` is the *third action
channel of the operative predictor* (`v0 = poses[t,3] / 10.0`, `SPEED_SCALE = 10.0`). It can reach the
tactical plan only indirectly, through whatever the shared trunk learned. **The 6.5× was measured on
the operative rollout surface; the planner surface never consumes that channel.**
⚠️ **This is a mechanism, not a licence to ignore the speed fix** — it says the speed fix improved the
*world model*, and that the tactical head is where the plan is actually lost.

### 5.5 ⭐⭐ The REF-C scale ladder SPLITS — and the split is invisible to ADE

| contrast | **this panel (PSS)** | `MODEL_REGISTRY` §4.3 open-loop (ADE@2s, full-set) |
|---|---|---|
| base (104.2 M) − small (54.7 M) | **+0.0002 [−0.0024, +0.0025] n.s.** | *(not published as a pair)* |
| **XL (251.9 M) − base** | **+0.0050 [+0.0023, +0.0082] ⭐ SEP** | **+0.0013 [−0.0281, +0.0316] n.s.** |
| **XL − small** | **+0.0055 [+0.0027, +0.0084] ⭐ SEP** | *(indistinguishable, §4.2)* |

Two things at once, and they point in opposite directions:

1. **The bottom of the ladder is flat, exactly as published.** A **2.42× parameter cut and a 2.20×
   encoder cut** (base → small) cost **+0.0002** — the tightest interval in the whole panel, on
   15 981 identical rows. The registry's *"statistically indistinguishable on everything that ships"*
   survives onto a surface that is a MEASUREMENT.
2. ⭐ **The top of the ladder is NOT flat here, and it was flat on ADE.** XL separates above both
   smaller rungs by ~0.005 PSS with intervals ~6× tighter than the ADE contrast that called them
   tied. **This is a scale effect the program's open-loop primary could not resolve** — which is
   precisely the class of thing this instrument exists to find (`ego_progress` ρ = 0.83 vs L2/ADE
   ρ = −0.36, p = 0.43, `PUBLISHED`, weak tier).

⚠️ **And it is confounded, so it is not a scaling law.** `MODEL_REGISTRY` §4.3 records that **XL
trained on v1 route labels and base/small on v2.1**, so base-vs-XL conflates **scale, anchor count
(256/128/64) and labels**. XL also carries H15 imagination ON (preset design, XL-only). The clean read
is the flat one (base vs small, matched labels); **the XL gap is a flag for a follow-up, not a
result.** Note also that XL's advantage is entirely on `ego_progress` (0.9438 vs 0.9317/0.9315) —
its `recovery` is **the worst in the panel** (0.0259).

### 5.6 The component split — where each arm actually loses

`PSS` is the mean of `ego_progress` (w 5) and `recovery` (w 5). They do **not** rank the arms the same
way, and that is the most informative thing in the panel:

| component | best → worst |
|---|---|
| `ego_progress` | **v4_oracle 0.9462** · refc_xl 0.9438 · cv 0.9407 · refc_base 0.9317 · refc_small 0.9315 · v1_follow 0.9081 · **v1_oracle 0.9047** · nospeed 0.8961 |
| `recovery` | **v1_oracle 0.0817** · nospeed 0.0800 · v1_follow 0.0785 · **cv 0.0776** · v4_oracle 0.0629 · refc_small 0.0296 · refc_base 0.0293 · **refc_xl 0.0259** |

* **v4 buys progress and pays for it in recovery.** It has the panel's best `ego_progress` but is
  **separated BELOW CV on `recovery`** (−0.0168 [−0.0332, −0.0008]): displaced by up to 12° of heading
  it cancels **6.3 %** of its own induced drift where a planner that **never steers** cancels **7.8 %**.
* **v1 is the mirror image**: worst-but-one `ego_progress` (−0.0415 vs v4, SEP) but the panel's
  **best `recovery`**, statistically **tied with CV** (+0.0017 [−0.0132, +0.0167] n.s.) and
  **separated above v4** (+0.0181 [+0.0048, +0.0331]).
* **REF-C is last on `recovery` at every scale** (−0.050 vs v1, −0.032 vs v4, both SEP) while sitting
  at or above mid-pack on progress. **Bigger REF-C recovers slightly worse, not better.**
* ⭐ **The oracle nav buys v1 nothing**: `v1_follow − v1_oracle` = **+0.0004 [−0.0008, +0.0018] n.s.**
  — consistent with §0.8's finding that the labeler is 75 % degenerate on these clip lengths. **The
  route input is not what is limiting v1's plan.**

⚠️ **`recovery` is a marginal component and must be read as such** — its between-arm spread clears the
`RANGE_MIN = 0.05` bar narrowly, three quarters of `v4_oracle`'s rows sit at the floor, and ~17 % of
rows are excluded as undefined. It is evidence that **no arm recovers much**, not a well-conditioned
score. The effects above are separated; they are also small.

---

## 6. Everything that is wrong with this panel, stated by me

| # | limitation | status |
|:--:|---|---|
| 1 | **Goal provenance is not matched across families.** v4 gets a 3-field oracle goal (route / route_graded / vt_band, incl. a speed band); v1 and the no-speed control get a 4-way nav command; REF-C produces its own route from the image. **The v4-vs-v1 contrast conflates the planner head with the goal interface** and I do not claim otherwise. | disclosed |
| 2 | ⚠️ **The v1 / no-speed "oracle" nav is 75 % degenerate** — 6 000 of 7 964 indices lack the 15–25 s of future `refb_labels.nav_command` needs and return `(NAV_FOLLOW, valid=False)`. `v1_tactical_follow` is run as the bound. | disclosed, bounded |
| 3 | **Oracle-goal numbers are upper bounds, not deployable.** `--goal-mode produced` was not run for v4 (time). | not done, flagged |
| 4 | **No collision, no TTC.** The val cache has no cuboids and its `episode_id` collides 242 → 40. Emitted as `None` with a reason; **no constant substituted**. ⇒ `PSS_recovery_progress` **is not a Driving Score.** | inherited blocker |
| 5 | **`comfort` is dropped, not fixed.** At `dt = 0.1` an 8 mm third-difference trips the jerk bound. Retuning after seeing who fails is metric-shopping. A usable clause needs the **controller-tracked** trajectory. | deliberate |
| 6 | **Non-reactive traffic**, on every node. Here it is stronger than a config claim: the observations are **real recorded frames**, so other agents *cannot* react by construction. (⚠️ Note the AlpaSim `trafficsim` disclosure is a separate fact and `c662bd4` showed it **was** enabled for the tactical gate — do not merge the two claims.) | deliberate |
| 7 | **Lateral axis refused.** We publish a protocol **narrower than NAVSIM v2's** and say so. | inherited, correct |
| 8 | **2 s scoring horizon**, because `FlagshipV4Head` emits 20 waypoints. NAVSIM v2 scores over 4 s. The cap is the head, not the protocol — and lengthening it costs **nothing** envelope-wise. | flagged |
| 9 | **flagship-v2corpus is NOT in the panel.** It is training on pod1, which this brief forbids touching; no checkpoint is reachable without touching it. | not run, by rule |
| 10 | **REF-A / REF-B are not in the panel.** REF-A consumes frozen features, not raw frames, so the warp cannot be applied to its input in this harness. Stated, not attempted. | out of scope |
| 11 | **`v4_blind` keeps the oracle goal** — blind to the image, not to its own future. Inherited confound; a doubly-blind control is the fix. | disclosed |
| 12 | ⚠️ **I ran ~50 minutes of the panel on the wrong v4 checkpoint** (`flagship-v4-30k`, killed at step ~4 400) before reading the published run log closely enough to find `flagship-v4-fromscratch-30k`. Caught before any number was produced; recorded because the artifact I was reproducing **names neither path in its own report body**. | corrected, recorded |

---

## 7. Deliverable manifest

Repo dir: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-pseudosim-arm-panel/`
Everything `git add`-ed, **NOT committed, NOT pushed**. ⚠️ marks anything in only ONE place.

| artifact | where it lives | what it is |
|---|---|---|
| `PSEUDOSIM_ARM_PANEL.md` | repo (this dir) | this report |
| `PRE_REGISTRATION.md` | repo | the frozen pre-registration (G1–G5, D-NULL criterion, V1-IN/V1-OUT) |
| `scripts/panel_run.py` | repo · `pod2:/workspace/_pspanel/scripts/` | one-arm driver + the planner adapters (`flagship`, `refc`, `cv`, `still`, `blind`) |
| `scripts/panel_combine.py` | repo · `pod2:/workspace/_pspanel/scripts/` | **the no-GPU path**: row-identity assertion, panel gate, composites, paired panel, pre-registered verdicts |
| `scripts/summarize_panel.py` | repo · `pod2:/workspace/_pspanel/scripts/` | §4's tables, **generated from the artifact, not hand-typed** |
| `artifacts/pseudosim_arm_panel.json` | repo · `pod2:/workspace/_pspanel/out/` | the panel: every arm node, range gate, panel gate, all paired contrasts, all gate verdicts |
| `artifacts/arm_<name>.json` × 10 | repo · `pod2:…/out/` | per-arm node incl. ckpt md5 / step / goal provenance / envelope proof / G4 falsifier |
| `artifacts/pw_<name>.npz` × 10 | repo · `pod2:…/out/` | **per-window dumps** — `traj`, `ref_path`, `ref_yaw`, `v0`, grid keys, `eid`, 15 981 rows each. Every bar in this report recomputes from these with **no GPU, no checkpoint, no corpus**: `python3 panel_combine.py --in-dir artifacts --out /tmp/x.json` |
| `artifacts/panel_tables.md` | repo · `pod2:…/out/` | the generated §4 tables, verbatim as inserted |
| `artifacts/run_logs/*.log` × 12 | repo · `pod2:…/out/` | per-arm run logs incl. the live envelope proof, the G4 falsifier and the HF pulls |

**Nothing that took real effort exists in only one place.** The checkpoints themselves are unchanged
and were only read.

⭐ **The no-GPU recompute path is VERIFIED, not asserted.** Re-run from a directory containing only
the `pw_*.npz` + `arm_*.json` files with `CUDA_VISIBLE_DEVICES=""`:

```
CUDA_VISIBLE_DEVICES="" python3 scripts/panel_combine.py --in-dir <dir> --out repro.json
```

**Every composite CI and all 45 paired blocks come back byte-identical** to
`artifacts/pseudosim_arm_panel.json` (`composites identical: True`, `paired identical: True`).
This is the arithmetic-only path whose absence forced five closed-loop artifacts to be re-driven in
July — any reviewer can recompute or re-derive a variant metric without a checkpoint, a corpus, or a
GPU.

### 7.1 Checkpoint provenance (every arm, md5-verified at run time)

| arm | checkpoint | md5 | step |
|---|---|---|---:|
| `v4_oracle` / `v4_blind` | `pod2:/workspace/experiments/flagship-v4-fromscratch/ckpt.pt` | `8771c1d9d3da696dcde2a745d628f6a8` | 29 999 |
| — anchors | `pod2:/workspace/experiments/flagship_v4_anchors_dense.pt` | `322358fac5bbf0f22692c0f251746a66` | — |
| `v1_tactical_oracle` / `_follow` | `pod2:/workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt` | `b5f07d9e3dd2ca643949bc86832e6585` | 29 999 |
| `nospeed_tactical_oracle` | `pod2:/workspace/experiments/flagship4b-phase0-30k/ckpt.pt` | `74be81035699c362e2fd0e5197880506` | 22 000 |
| `refc_base_produced` | HF `Sayood/tanitad-refc-base` → `pod2:/workspace/models/refc-base-30k/ckpt.pt` | `8f10d6f934f4199e11ddc7352e074939` ✅ **matches `MODEL_REGISTRY` §4.3** | 29 999 |
| `refc_xl_produced` | HF `Sayood/tanitad-refc-xl` → `pod2:/workspace/models/refc-xl-30k/ckpt.pt` | `966d4eff1ea5ddf86efba01b8344e198` ✅ **matches §4.1** | 29 999 |
| `refc_small_produced` | `pod2:/workspace/experiments/refc-diffusion-small-v21-30k/ckpt.pt` | `6a18476d1d2f09175f1acfb178742063` | 29 999 |
| `cv_holdv0` / `stand_still` | analytic, no checkpoint | — | — |

Val substrate: `pod2:/root/valdata/physicalai-val-0c5f7dac3b11`, first 40 `ep_*.pt` — the canonical
40-episode deployment, the same selection the published run used. **No episode re-selection. Parity
untouched.** 🔒 No clip UUID or raw PhysicalAI content appears in any artifact.

### 7.2 What this unblocks

| stream | what unblocks |
|---|---|
| **Benchmarks & Eval / gates** | A **10-arm closed-loop-class panel that is a MEASUREMENT**, with the paired estimator the gate protocol requires, on identical rows, reproducible with no GPU from the `pw_*.npz` dumps. A gate that wants a closed-loop primary now has arms to calibrate a bar against. |
| **Architecture & Inference / hierarchy** | The `ego_progress`-vs-`recovery` split (§5.6) localises each arm's failure: v4 loses on recovery, v1 loses on progress, REF-C loses on recovery at every scale. This is a **per-brain** signal that open-loop ADE cannot contain. |
| **Architecture & Inference / v5** | §5.4: the speed channel — the program's largest measured lever — **does not reach the tactical head**, and mechanistically cannot. If v5 wants the plan to know the speed, the tactical policy needs an ego input (the `v2_ego_to_planners` lever, off in v1). |
| **Model registry** | §0.7: the "flagship-v4 30 k" closed-loop rows describe `flagship-v4-fromscratch-30k`, not `flagship-v4-30k`. Two rows, never one name. |

---

## 8. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **0** | ⛔⛔ **ON THE ONLY CLOSED-LOOP SURFACE WE OWN THAT IS A MEASUREMENT, A ZERO-PARAMETER CONSTANT-VELOCITY BASELINE IS FIRST.** `cv_holdv0` **0.5705** is separated above REF-C at **all three scales** (+0.0203 to +0.0255) and above the no-speed control (+0.0235), and is merely **tied** with both flagship arms — one of which (`v4_oracle`, 0.5622) needs an **ORACLE goal minted from its own future** to tie. Tier 2, 2 s horizon, and CV is a strong 2 s baseline by construction — **but this is the ranking, and it should decide what v5 optimises.** **Not an agent's call to act on.** | **Sayed / PI** |
| **1** | ⛔ **`MODEL_REGISTRY` NAMING: the published pseudo-simulation arm is `flagship-v4-fromscratch-30k`** (step 29 999, val ade@2s 0.5063, md5 `8771c1d9…`), while §1.5.5 still reads *"READY, not launched"* and §1.5.1's `flagship-v4-30k` is the **killed** run that pod2 still holds at step 4 400. **This is the `flagship4b-phase0-30k` inversion in a new costume** and it already cost this agent a wrong-checkpoint start. Both rows need updating and the artifact filename `pseudosim_v4_30k.json` is misleading. | **Model-registry agent** |
| **2** | ⭐ **v1's headline `ade_0_2s` 0.4271 is a WM-FIDELITY number, not a planning number** — `taniteval/rollout.py` feeds the expert's true future actions and its own `honest_metric_name` is `wm_fidelity_ade_2s`. Every cross-arm table that puts 0.4271 beside a REF-C direct-decode number is comparing a decode-with-the-answer against a plan. §6 of the registry is the place this must be stamped. | **Model-registry agent / PI** |
| **3** | ⭐ **THE PANEL GATE BELONGS IN `taniteval/pseudosim.py`, NOT IN A DRIVER.** The shipped per-arm `discriminative_range` silently produces **different weight sets per arm**, and a paired delta across them is not a model comparison — MEASURED here at up to **5.2× and one verdict flip**. `composite()` should take the panel's admissible set, or `emit()` should refuse when the by-arm admissibility is not uniform. **This will not happen by itself.** | `taniteval` maintainer |
| **4** | ⚠️ **`TacticalPolicy` has no ego input on v1**, so `--speed-input` cannot reach the plan (§5.4). The `v2_ego_to_planners` lever exists and is **off** on every arm in this panel. Whether v5 turns it on is an architecture decision with a measured motivation now. | **Architecture & Inference** |
| **4b** | ⭐ **REF-C-XL SEPARATES ABOVE base AND small ON THIS SURFACE WHILE ADE CALLED THEM TIED** (+0.0050 / +0.0055, separated under **both** gate variants). ⚠️ **Confounded by the v1-vs-v2.1 route labels and by H15-imagination-on** (registry §4.3 / §4.1). The **cheapest discriminating experiment** is a base-scale run with XL's v1 labels — one training run — and it would tell us whether the ladder's top is scale or labels. Registry §4.3's *"a 2.42× parameter cut costs nothing"* is now **surface-dependent** and should say so. | **Model-registry agent / D-030 owner** |
| **5** | ⚠️ **Thread-count trap, for `AGENT_OPERATING_STANDARD` traps preflight.** torch defaults to one OMP thread per core; **N concurrent eval processes on a 96-core pod spawn ~113 threads each and make no progress** (GPU `sm` 0–6 %, 50 min with nothing written). `OMP_NUM_THREADS=6` made the identical arm finish in **232 s**. This looks exactly like a hang and there is no log line that says so. | orchestrator |
| **6** | ⭐ **The 2 s scoring horizon is the panel's main resolution limit, and lifting it is FREE on this protocol** (no observation is re-synthesised after the grid point, so the out-of-envelope fraction stays 0 at any scoring horizon). It is capped by the 20-waypoint head. A longer-horizon head would get a longer-horizon MEASUREMENT out of this harness with no re-validation. | **PI / architecture** |
| **7** | ⭐ **The collision gate still needs the episode→clip join** (full `clip_id` in the episode cache). Until then `PSS` can never become a Driving Score. Unchanged from the prior report's escalation 1, and this panel is the second artifact blocked by it. | **Data Engineering** |

---

## 9. Self-refutations, and what was deliberately NOT done

| # | what | status |
|:--:|---|---|
| 1 | **I started the panel on the wrong v4 checkpoint** (`flagship-v4-30k`, killed at step ~4 400) and burned ~50 min. Caught by reading the published `run.log`, which is the only place the real path appears. | corrected, §0.7 |
| 2 | **My first panel would have compared different composites across arms** (comfort admitted for REF-C/v1, dropped for v4/CV). Caught on the 2-episode smoke, before any panel number existed. Fixed by the panel gate, with both sides published. | corrected, §3.1 |
| 3 | ⚠️ **I nearly let the adversary delete the metric**: `stand_still` is inadmissible on every component, so the naive intersection would have made the composite vacuous for every real arm. | corrected, §3.2 |
| 4 | ⚠️ **The "oracle" nav I gave v1 and the no-speed control is 75 % degenerate.** I did not discover this until after the runs started; it is bounded by the `follow` control rather than hidden. | disclosed, §6.2 |
| 5 | **I did not re-run v4 with `--goal-mode produced`**, so the flagship arms sit on an upper bound while REF-C sits on a *produced* goal. The direction of this bias **favours the flagship arms**, i.e. it makes §0.3 conservative. | not done, flagged |
| 6 | **I did not fix the `v4_blind` goal confound** (doubly-blind control). Inherited. | not done |
| 7 | **`comfort` is not retuned**, deliberately — retuning after seeing who fails is metric-shopping. | deliberate |
| 8 | **The lateral axis stays refused.** I did not re-derive the L-BAD verdict; I inherited it and exercised its enforcement (`LateralAxisRefused` raised in every arm process). | inherited, enforced |
| 9 | **No comparison to a published leaderboard.** Internal numbers on our own corpus. | out of scope |
