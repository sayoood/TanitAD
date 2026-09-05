# RESULT — A FEASIBILITY-AWARE DECODE: refcv3's driven path is measurably safer, and the cost is 1.2 mm

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-feasible-decode/RESULT.md`
Architecture & Inference FlyWheel · 2026-09-05 · **ZERO GPU, ZERO training, ZERO new parameters** ·
⛔ `tanitad-refcv3` (refcv4b live) and Thor never touched.
Executes `Project Steering/Decisions/2026-09-05-mm-decisions.md` **§M23.5**.
Register (same turn): `D-FEASDEC-1`, `D-FEASDEC-FRONTIER-1`, `D-FEASDEC-T1-1`, `D-PROG-RANK-FIX-1`,
`H-PROG-SAT-1`, `D-FEASDEC-STOPSTEP-1`.
Evidence classes: **MEASURED** (ours + artifact path) · **INHERITED** (another WP, not re-run).
Tiers: **T0** = readout on an emitted fan, never a driving claim · **T1** = self-action OPEN loop
(PI ruling 2026-09-02). ⛔ No number here is a closed-loop claim.

---

## 0. The answer to the question the PI asked, in one line

> **YES. refcv3's driven path is measurably safer than it was this morning: its envelope-violation
> rate falls `0.0865 → 0.0000` — a STRUCTURAL ZERO, not a small number — its yaw-rate error falls
> **80.4 %** (`0.2176 → 0.0427 rad/s`, separated) from *worse than a constant-velocity straight
> line* to *better than it*, and its friction load falls 9.6 %. The whole cost is
> `ade_0_2s` **+0.0012 m** — one point two millimetres — measured at T1 on 4,823 windows / 141
> episodes with a paired episode-cluster bootstrap, against a zero-lever floor that is
> **bit-identical on 4,823 of 4,823 windows**.**

For scale: the V2-faithful RL stage this replaces failed its committed exit at `ade_m`
**+0.0362 m separated** — **30× more ADE** — while making the fan *less* safe.

---

## 1. What was built, and why it is not the two things already refuted

`stack/tanitad/refs/feasible_decode.py` (+ `stack/tests/test_feasible_decode.py`, **15 pins**) —
a **control-space projection** wired into `AnchoredDiffusionDecoder` behind
`DecoderConfig.feasible_decode` (default **OFF**, and the disabled path returns the *same object*).

It inverts the exact finite-difference map `fan_safety.score_paths` uses, clamps the recovered
controls, and re-integrates:

1. `|accel| <= 4.0 m/s^2`, `|kappa| <= 0.2 1/m` — the scorer's own box;
2. `(accel, lat_acc)` projected **radially** onto the friction disc of radius `mu*g`;
3. re-integrated forward, **sequentially in k**, because `v_mid[k]` depends on `accel[0..k-1]`.

⭐ **Because the clamp is the exact inverse-then-forward of the scorer's own differences, an
envelope- or Kamm-violating path is UNREPRESENTABLE. That is an identity, not an estimate** — no
seed and no bootstrap can move a structural zero (`H-ECHO-4` precedent) — and the artifact asserts
it through `fan_safety.score_paths`, the **consumer**, rather than through the module's own
bookkeeping (RETRACTION #30).

⛔ **It is applied at EVERY pass that moves a waypoint** — `refc.py`'s classifier pass
(`x = bank + offset`) *and* each refinement iteration (`x = x_in + off`). A penalty on
`out["offset"]` reaches **one of three**, which is part of why the RL stage's authority was smaller
than its trainable-parameter count suggested.

| already refuted | why this is different |
|---|---|
| **a feasibility REWARD** (`feasibility x4` moves ρ(reward, envelope) by **0.001**) | additive and out-votable inside a weighted sum; this is a **constraint on the representable set** |
| **a post-train VETO** (closed **~2.7 %** of the gap, regressed T1 `ade_m` **+0.0362 separated**) | soft — it penalises a path the decode can still emit; this removes the ability |
| **the top-2 kinematic GATE** (a sibling stream, `sel_envelope` −31 %) | a selection rule closes **0.00 %** of the fan-level gap — a structural zero — and its ceiling on the driven path is `sel_env` ≈ 0.06. **They compose; neither replaces the other.** |

---

## 2. P2 — the frontier, and the scope error it refuses to repeat

**Object:** `kingate_bank_drawA_v1_selscore.npz`, md5 `9bda7715a6571186935243b242c2ab74`,
**400 windows × 128 candidates / 70 episodes**, the SAME file the sibling's λ-frontier ran on.
Artifact: `raw/mu_frontier.json` / `.log`.

| arm | fan `envelope` | fan `kamm_over` | fan `peak_g` | `off_reach` | oracle ADE | sel ADE |
|---|---|---|---|---|---|---|
| **emitted** | 0.8879 | 0.8408 | **4.1789 g** | 0.1094 | 0.2063 | 0.4486 |
| proj `mu=box` | **0.0000** | 0.6352 | 1.9404 | 0.2360 | 0.1967 | 0.4491 |
| proj `mu=1.00` | **0.0000** | 0.6352 | 0.7720 | 0.3338 | 0.1967 | 0.4491 |
| ⭐ **proj `mu=0.70`** | **0.0000** | **0.0000** | **0.5964 g** | 0.3405 | **0.1962** | 0.4491 |
| ⭐⭐ **proj `mu=0.70` +entry** | **0.0000** | **0.0000** | **0.5795 g** | **0.0000** | **0.1962** | 0.4491 |
| frozen anchor bank (λ=0) | 0.0156 | 0.1953 | 0.4808 | 0.7822 | 1.0985 | 3.9824 |

**P2-C1 PASS** — structural zero at `mu = 0.70`.
**P2-C3** — **96.87 %** of the `D1 = 3.6981 g` fan gap closed; the **8.69×** blow-up becomes **1.24×**.
**P2-C4 FAIL** — `off_reach` 0.1094 → 0.3405 (**+0.2311**, committed bar +0.05). ⛔ Reported as
written. ⭐ The **pre-registered second variant** `+entry` (SPEC §3.1, banked before any arm ran)
reads `off_reach` **0.0000** and drives `fan_infeasible` **0.8916 → 0.0000 exactly** over all
51,200 candidates — and *that identity is not a coincidence*: the entry clamp bounds the first
step's speed by `a_max*dt = 2.0` and the accel clamp adds `a_max*dt*mean(0,1,2,3) = 3.0`, summing
to exactly the reachability band's `A_MAX_REACH * horizon = 2.5 * 2.0 = 5.0 m/s`.

### ⛔ 2.1 The scope error this package refuses to make

The sibling's λ-frontier sweeps `bank + λ(fan − bank)` — an **isotropic shrink**. Its verdict is
about the SHRINK family and **cannot** be quoted against a PROJECTION, which removes only the
infeasible *component* of the control profile. Both were therefore computed **in one script, on one
npz, on the same windows**, and read at **matched `fan_peak_g`** — the only fair axis:

| at matched `fan_peak_g` | projection Δ oracle-ADE | shrink Δ oracle-ADE | ratio |
|---|---|---|---|
| 1.9404 g | **−0.0098 m** | +0.3035 m | **−0.032** |
| 0.7720 g | **−0.0098 m** | +0.6753 m | **−0.015** |
| **0.5964 g** | **−0.0103 m** | **+0.7680 m** | **−0.013** |

**P2-C2 PASS** (committed: < 0.5×; measured **−0.013**). ⭐⭐ **The projection does not merely cost
less than the shrink — it costs NEGATIVE: projecting the fan onto the feasible set moves it CLOSER
to the human** (oracle-in-fan `0.2063 → 0.1962 m`, **−4.9 %**). ⇒ **the infeasible component of
refcv3's 9.29 m displacement is not a trade against accuracy; it is waste.**

**Controls, all PASS:** `C-OFF` (disabled lever returns the same object) · `C-ROUNDTRIP` (an
already-feasible path is a fixed point, **0.00e+00 m**) · `C-KNOWN` (the frozen bank reproduces
`bank_vs_fan_feasibility.json`'s bank block — a different script, a different draw — `peak_g`
0.480782 vs 0.480782) · `C-LAMBDA1`. ⭐ And the **λ column reproduces the sibling's independently
written `displacement_frontier.py` row for row**, which is the object control `C-LAMBDA1` cannot
give (λ=1 is the emitted fan whatever the left operand is — that is exactly how #30 passed).

---

## 3. P3 — T1, all four families, against both floors

**Instrument:** `taniteval/tools/paired_openloop.py`, paired **episode-cluster bootstrap**,
`n_boot 2000`, `seed 0`, floor `ha0`. ⛔ `overlapping_holdout_se` never used.
**n = 4,823 shared windows / 141 episodes, 0 dropped.** `void: false`; all six gates pass; GT and
`v0` bit-identical across the two dumps (`0.000e+00`).
Artifacts: `raw/paired_proj07_vs_base.{json,md}`, `raw/derived_dump_report.json`.

### 3.1 The driven path's safety — a structural zero

| arm | `envelope` | `kamm_over` | `infeasible` | `peak_g` mean | `peak_g` p95 |
|---|---|---|---|---|---|
| **base `os`** | 0.0865 | 0.0021 | 0.0865 | 0.1415 | 0.3321 |
| `projoff` (zero-lever floor) | 0.0865 | 0.0021 | 0.0865 | 0.1415 | 0.3321 |
| ⭐ **`proj07`** | **0.0000** | **0.0000** | **0.0000** | **0.1279** | 0.3293 |
| the HUMAN's own logged path | 0.0068 | 0.0000 | 0.0068 | 0.0899 | 0.2674 |

⭐ The base arm violates the envelope **12.7×** as often as the human. After the projection it
violates **less often than the human does** — because it cannot.

### 3.2 The four families, per family, never pooled

| family | metric | base | proj07 | (B−f)−(A−f) | 95 % CI | separated |
|---|---|---|---|---|---|---|
| **ADE** | `ade_m` | 0.4419 | 0.4431 | **+0.0012** | [0.0002, 0.0024] | YES |
| | `fde_m` | 0.9288 | 0.9301 | +0.0014 | [−0.0017, 0.0046] | no |
| ⭐ **LONGITUDINAL** | `LON_speed_mae_mps` | 0.4516 | 0.4495 | **−0.0021** | [−0.0037, −0.0008] | **YES (better)** |
| | `LON_accel_mae_mps2` | 0.6806 | 0.6746 | **−0.0060** | [−0.0107, −0.0024] | **YES (better)** |
| | `LON_along_mae_m` | 0.4030 | 0.4038 | +0.0008 | [−0.0003, 0.0020] | no |
| ⭐⭐ **LATERAL** | `LAT_yaw_rate_mae_radps` | 0.2176 | **0.0427** | **−0.1749** | [−0.2563, −0.1088] | **YES (better)** |
| | `LAT_heading_mae_deg` | 1.3591 | 1.3630 | −0.0079 | [−0.0209, 0.0088] | no |
| | `LAT_cross_mae_m` | 0.1084 | 0.1093 | +0.0009 | [0.0004, 0.0015] | YES (worse) |
| **TACTICAL** | `TAC_traj_lat_correct` | 0.9540 | 0.9544 | +0.0004 | [−0.0017, 0.0023] | no |
| | `TAC_traj_lon_correct` | 0.7477 | 0.7479 | +0.0002 | [0.0000, 0.0006] | no |
| | `TAC_declared_lon_correct` | 0.5134 | 0.5134 | **0.0000** | [0, 0] | no (structural) |
| **STRATEGIC** | `STR_route_correct` | 0.7667 | 0.7667 | **0.0000** | [0, 0] | no (structural) |

**Family verdicts as the tool prints them: LONGITUDINAL → `proj07`. LATERAL → SPLIT. ADE →
`base`. TACTICAL / STRATEGIC → no separation.**

⭐⭐ **The yaw-rate row is the result.** Base's yaw-rate margin over the `ha0` floor is **+0.1700**
— *worse than a constant-velocity straight line*. `proj07`'s is **−0.0049** — better than it. An
**80.4 % reduction**, separated. That is the LATERAL family finally reading a real improvement
instead of cross-track alone, which is exactly why the four-family rule exists.

⚠️ **TACTICAL and STRATEGIC read exact 0.0000 by CONSTRUCTION, not by measurement.** The
projection changes waypoints; the declared tactical and route heads read logits it never touches.
That is an identity, and it is stated so nobody reads "no separation" as evidence of anything.
The two **trajectory-derived** tactical rows (`TAC_traj_*`) do move and do not separate.

⚠️ **The ADE cost is at the edge of detectability, and the two implementations of the same
estimator disagree on the flag.** `paired_openloop.py` reads +0.0012 **[0.0002, 0.0024], separated**;
this package's own direct read of the same delta on the same windows reads **[0.0000, 0.0024], not
separated**. Both are episode-cluster bootstraps; at 1.2 mm the interval touches zero and the
`separated` bit is implementation-sensitive. ⛔ **The binding instrument's answer is the one
reported: separated, +0.0012 m.**

### 3.3 The floors — and an honest statement of which one applies

The brief binds a **seed replicate** (`H-ESTIM-SEED-1`) and **`ctrl0` (lr = 0)**, both because
**AdamW normalises by the gradient's own scale**, so a control that zeroes a *loss* still moves
every tensor and separated 35 of 57 metrics.

⭐ **This lever takes ZERO gradient steps.** There is no optimiser, no update and no training seed:
the arms differ by a deterministic geometric function of the emitted path. The floors are therefore
answered by a **strictly stronger** control, and it is MEASURED:

* **F1 — the disabled lever.** `projoff` runs the identical derivation pipeline with
  `enabled=False`: **4,823 / 4,823 windows bit-identical, max |Δ| = 0.0 exactly**, and
  `paired_openloop` independently reports `base:os` `identical to proj off:os 4823/4823`.
  ⇒ **the rig's run-to-run noise floor is EXACTLY zero**, which no `ctrl0` can improve on.
* **F2 — determinism.** Re-run under a different `torch` seed: bit-identical, asserted per episode.
* ⚠️ **Scope, stated plainly: F1/F2 retire the TRAINING-variance question for THIS arm only.** Any
  future arm that **fine-tunes** the decoder under the projection re-acquires `H-ESTIM-SEED-1` in
  full, and nothing here discharges it in advance.

---

## 4. P1 — `progress` was FIXED on the ranking bar and FAILED the progress bar, twice, and then the
question dissolved

Artifacts: `raw/progress_rank_fix_240w.json` (240 w / 121 ep, md5 `ccecf0ade756…`),
`raw/progress_rank_fix_400w.json` (400 w / 70 ep — an independent draw that carries GT, so the C2
floor is the **human's own** distance and not a substitute), `raw/projected_fan_rank_400w.json`.
Estimator, `n_boot` and seed identical to the number being replaced (episode-cluster bootstrap,
4000, seed 11).

| term | ρ(·, `peak_g`) | ρ(`envelope`) | ρ(`ttc_below`) | ρ(`contact`) |
|---|---|---|---|---|
| `progress` (stock) | **+0.2900** | −0.0078 | **+0.4697** | **+0.3286** |
| ⭐ `progress_v2` = `min(r,1)·w(ex)` | **−0.5897** [−0.6707, −0.5002] | −0.5311 | −0.1820 | +0.1241 |
| `progress_v2` cap only | +0.1294 | −0.1015 | +0.3463 | +0.2938 |
| `progress_v3` = progress of the **projected** path | **+0.5042** | +0.2997 | +0.4924 | +0.2805 |
| composed DEFAULT | −0.4603 | −0.5367 | −0.3544 | −0.6326 |
| composed DEFAULT with `progress_v2` | **−0.6898** | −0.5370 | **−0.5396** | −0.6326 |

* **P1-C1 PASS** — ρ(`progress_v2`, `peak_g`) **−0.5897**, far below the committed **+0.05**, and it
  carries the composed reward from **−0.4603 to −0.6898**, recovering ~86 % of what *deleting*
  progress would buy while keeping a progress term.
* ⛔ **P1-C2 FAIL** — on 160 straight windows the argmax-`progress_v2` candidate covers **18.672 m**
  against argmax-`progress_v1`'s **26.216 m**, ratio **0.7123** below the committed **0.95**. It
  clears the driving floor (v0-hold 17.229 m; on the 400 w draw, the human's own **16.624 m**) but
  the committed ratio is the committed ratio. **VERDICT: FAIL.**
* ⛔ **P1b FAIL, on both draws** — `progress_v3` reads **+0.5042 / +0.5135**, *worse* than the stock
  term. ⭐ **Mechanism, which is why it is a finding and not a dead end:** `along` is NET `+x`, so a
  candidate that swings sideways is penalised by its own lateral excursion; the projection
  **straightens** exactly those candidates, so their `+x` **rises**. **The projection is the right
  fix for the DECODE and the wrong fix for this REWARD TERM, and those two sentences are not in
  tension.**
* ⭐⭐ **P1c — and this is where the question dissolves.** On the projected fan, ρ against `envelope`
  and `kamm_over` is **UNDEFINED on 400 / 400 windows**: there are no violating candidates left to
  rank. **P1c-C0 PASS.** ⛔ **P1c-C1 FAIL**: the *weaker* residual question — does the reward still
  prefer the harder-driving candidate inside a feasible fan — reads ρ(progress, `peak_g`)
  **+0.3232 [+0.2229, +0.4115]**, essentially unchanged. ⇒ **the safety-grade defect is removed by
  the decode; a comfort-grade preference survives and a reward change is still owed.** These are
  different stakes and are never reported as one number.
* ⭐ **P1c-C3's committed prediction held**: the projection does not touch lead geometry, and
  ρ vs `ttc_below` moved 0.4431 → 0.4123 (|Δ| 0.031). A control that read its predicted value.

### 4.1 ⛔ Two claims this package made and had to correct against its own controls

1. **SPEC §1 was WRONG as written.** It claimed *any per-window monotone renormalisation of
   `along` moves ρ by exactly zero*. **MEASURED: the UNCLAMPED ratio moves by 0.00e+00 — exact —
   but the CLAMPED component moves by 0.0234**, because the `[-1, 1.5]` clamp is **not monotone**:
   it creates TIES, and a Spearman moves when ties move. The claim is true of the ratio the
   component clamps and false of the component. Caught by a control the SPEC ran *deliberately to
   confirm a no-op*.
2. **`H-PROG-SAT-1` REFUTED.** Predicted: `feasibility x4` is a null lever because the term is
   saturated (median within-window IQR < 0.05). **MEASURED 0.2440 (240 w) / 0.2262 (400 w)** — the
   term discriminates fine; 26.3 % of candidates read < 0.01 but the bulk does not. The 270× is not
   a saturation artifact, and the hypothesis is reported refuted.

---

## 5. ⛔ THE DEFECT THIS PACKAGE ALMOST SHIPPED, AND THE CONTROL THAT CAUGHT IT

`D-FEASDEC-STOPSTEP-1`. The first build left **2 of 51,200** fan candidates violating `envelope`
under `clamp_entry` — both in `v0 = 0.000` windows with speeds `[2.00, 4.00, 2.00, 0.00]`, reading
**|kappa| = 0.3396** against a 0.2 cap: **a stationary vehicle performing a hard turn.**

* **Mechanism:** a path that decelerates to a stop emits a zero-length step, and `atan2(0, 0) == 0`,
  so the step *before* it appears to have turned through the whole previous heading.
* ⚠️ **My first hypothesis — "a scorer artifact, not a leak" — was REFUTED by its own discriminator.**
  `models.kinematic.unicycle_controls_from_path`, the programme's OTHER and *explicitly guarded*
  recovery, reads the **same 0.3396**. Both substitute a *direction* for a non-moving step instead
  of propagating the previous one. ⇒ a second **implementation** (not the same probe re-run) is
  what settled it, and it settled it **against the comfortable answer**.
* **Fix:** hold the heading through a stop and emit the halted step with a length of
  `|p| * 1e-3` — exact, because `atan2` is scale-invariant — and **freeze the yaw** there, because
  `yaw_rate = v * kappa` and a stationary vehicle does not turn.
* ⛔⛔ **And the fix's FIRST version was wrong in the same family, twice:**
  1. a **1e-6 m** step is exact in float64 and *noise in float32*: at a 4 m offset fp32's resolution
     is ~4.8e-7 m. The regression fixture still read `envelope = 1.0` when scored in fp32 while
     reading a clean 0 in float64. ⇒ **the contract is what the fp32 CONSUMER recovers, never what
     our float64 arithmetic believes.** Same shape as `MARGIN` below, one dtype down.
  2. a **1 mm ABSOLUTE** threshold then inflated *slow-but-moving* paths — a `ha0` hold at
     v0 = 0.0005 m/s takes well-conditioned 0.25 mm steps and was rewritten to 1 mm. **C-ROUNDTRIP
     went 0.00e+00 → 2.63e-03 m.** ⇒ "degenerate" is a statement about a step **relative to its own
     path**; a fixed constant that is right at one scale and wrong at another is the
     `df` / `free` / `step_s` family again — *introduced by the fix for another instance of it*.
* **After the fix: 0 / 51,200 on BOTH arms, exact** (`raw/stopped_step_residual.json`).

⭐ A sibling finding of the same class, kept because it is cheap and load-bearing: **`MARGIN`.**
`score_paths` flags with a strict `>`, and a projection clamping *exactly* to `a_max` saturates on
the boundary — an fp32 round-trip then lands above it about half the time. MEASURED **0.609375**
envelope on a 64-candidate fixture, which would have read as "the projection leaks". Clamping to `(1 - MARGIN)` of each limit removes it;
MARGIN is **1e-2** in the shipped module, sized so it also covers the ~4.8e-4 curvature error a
held stopped step can contribute.

---

## 6. What this does NOT establish

1. ⛔ **Not closed loop.** T0/T1 only (PI ruling 2026-09-02).
2. ⛔ **The measured arm projects the EMITTED fan; the selector's ranking is unchanged.** The
   in-decoder wiring (`DecoderConfig.feasible_decode`) is implemented and unit-pinned so the conf
   head sees the projected geometry at every pass, but **measuring THAT needs one GPU forward pass
   over the eval split (~1.5 h)**, and the dev-box 4060 is saturated by refav1 + the sibling gate
   panel + a veto suite. It is a NOT-YET-RUN, not a NOT-POSSIBLE, and it is the first successor.
3. ⛔ **Only the uniform 2 s prefix is projected.** Slots 4–7 are on a 1 s grid; the tail is rigidly
   **translated** so the path stays C0-continuous and the tail's own kinematics are unchanged.
   Velocity continuity across the seam is **not** enforced. Projecting the tail on its own grid is a
   named work item.
4. ⛔ **`off_reach` is the headline arm's cost** (+0.2311) and only the pre-registered `+entry`
   variant removes it. The two must be reported together; trading one failure mode for the other is
   the failure the successor doc named in advance.
5. **A reward change is still owed** for the comfort-grade residual (P1c-C1 FAIL). It is not a
   safety defect any more.

## 7. Successors, in priority order

1. **Run the in-decoder arm** (`feasible_decode=True` through `refcv3_arm.py`) when the 4060 frees:
   it is the only way to see whether a selector ranking *projected* geometry beats one ranking raw
   geometry. ~1.5 h, 0 pod-hours.
2. **Compose with the sibling's top-2 kinematic gate.** They act on different objects — the gate
   picks a better candidate (`sel_peak_g` → 0.106), the projection removes what remains (`sel_env`
   → 0.0000, which the gate's own ceiling of ≈0.06 cannot reach). Neither is a substitute.
3. **Project the 2–6 s tail on its own 1 s grid**, and re-read the 6 s families.
4. **A longitudinal PAIR for `progress`** — P1, P1b and P1c together say no reshaping of a single
   scalar clears both bars on this fan. Scoped, not started.
5. **`clamp_entry` as the default**, gated on a window-level read of what it costs where `v0` is
   small: it drives `fan_infeasible` to an exact zero and costs nothing measured here.
