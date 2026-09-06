# RESULT — the `W_KAPPA` frontier: **YES, an arm tracks the road better than a straight line and still turns left — and it is `W_KAPPA = 7`, which was already on disk**

`TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-wkappa-frontier/RESULT.md`
Architecture & Inference FlyWheel · 2026-09-06 · dev-box RTX 4060 only.
⛔ The A40 (refcv5, ETA ≈ 2026-09-08 07:33 UTC) and Thor were **never touched**.
Pre-registration: `SPEC.md` (commit `4a9ec31`) + `SPEC_AMENDMENT_A1.md`, **no criterion changed**.
Register (same turn): `D-WKAPPA-FRONTIER-1`, `D-WKAPPA-VARIANCE-SPLIT-1`,
`D-WKAPPA-ZEROCENSUS-2`, `H-WKAPPA-VOCAB-STRUCTURAL` (**refuted**), `D-WKAPPA-UNDEREXEC-1`.
Evidence classes: **MEASURED (ours + artifact path)** · **INHERITED** (banked, not re-derived).
Tiers: **T1** = self-action OPEN loop (PI ruling 2026-09-02) for `cl` / `ha` / `ha0` / `ha0_ext`;
**T0** for `ol`. ⛔ **No number here is a closed-loop claim.**

---

## 0. The one line the brief asked for

> **An arm exists.** `W_KAPPA = 7` tracks the road **3.3× the inference-seed floor better than a
> perfectly straight line** (curvature MAE **0.033522** vs the `ha0` floor **0.040083**), **still
> turns left** (`turn_left` recall **0.2727 = 3 of 11**, against a measured recall seed floor of
> **exactly 0.00000**), and does it with **zero friction-circle violations, non-vacuously**
> (`kamm_over` **0.0000** at `n = 27`, `max|kappa|` **0.0800 = 4.0×** the `M58` gate).
> ⛔ **The trade-off is therefore NOT fundamental to this cost geometry — but it is also not
> abolished:** the arm is on a *knee*, not a plateau, and the frontier point is established on the
> **inference-seed** axis and **not** on the **episode** axis at `n = 40 / 8`.

⭐ **And the sharpest fact in the package: this arm was already on disk.** `wk7` was rolled at
06:02 UTC today. What had never been done was **measuring its feasibility** — the banked
zero-violation census contains `cos_wk`, `wk15`, `wk151`, `combined`, `best` and `bestlad`, and
**none of `wk1` / `wk3` / `wk7`**. The prize cost **zero GPU** to find.

---

## 1. ⛔ THE RUNG SPACING, DERIVED FROM THE 0.00200 FLOOR AND STATED BEFORE LAUNCH

Binding inference-seed floors (`D-REFAV1-CG-SEEDFLOOR`, INHERITED): curvature **0.00200**,
ADE 0.1035, heading 1.1458, **`turn_left` recall exactly 0.00000**, recall resolution 1/11 = 0.0909.

### 1.1 ⛔ The scalar knee is NOT resolvable — three arms refused in advance

`wk7` 0.033522 → `wk15` 0.030982 over `ΔW = 8.11245` ⇒ slope **3.131e-4 / unit** ⇒ minimum
curvature-resolvable step `0.00200 / 3.131e-4` = **6.39 units**, against an interval only **8.11**
wide = **1.27 minimum steps**. ⇒ **no interior scalar rung is distinguishable from BOTH neighbours**;
a `wk9 / wk11 / wk13` ladder would have measured noise and printed a curve. **Not run, and this is
the reason, fixed in `SPEC.md` §2.1 before launch.**

### 1.2 ⭐ The axis that resolves: the goal-conditioned TURN weight

`--w-kappa-by-goal <lane_keep>,<turn>` picks the curvature weight per window from the **decoded
tactical lateral goal token**. Endpoints MEASURED: `gkappa` (`w_turn` 0) curvature **0.040167**;
`wk15` (`w_turn` 15.11245) **0.030982** ⇒ slope **6.078e-4 / unit**, **1.94× steeper** than the
scalar knee because the weight now moves on only the 22 of 40 turn-goal windows.
* curvature-resolvable step = **3.29 units**
* recall-resolvable step (≥ 1 window of 11, at 0.265 windows/unit) = **3.77 units**
* binding minimum = **3.77** ⇒ ⭐ **ADOPTED SPACING `Δw_turn` = 5.0375 (= 15.11245/3), 1.34× the
  minimum**, expected curvature step **0.00306 = 1.53× the floor**.
⇒ **RUNGS `w_turn ∈ {0, 5.03748, 10.07497, 15.11245}`**, `w_lane_keep = w_shift = 15.11245` pinned.

⚠️ The `w_turn = 15.11245` endpoint **is** the scalar `wk15` arm, and that is the tool's own
statement rather than an assumption: `refav1_arm.py` **raises** on an all-equal map — *"gives every
goal class the SAME weight, which is bit-identical to `--cost-weights` with that `W_KAPPA` … To run
the scalar control, OMIT this flag."*

---

## 2. ⭐⭐ BOTH CURVES, ON THE SAME RUNGS — and the friction circle beside the turn, per `I22`

`n = 40` windows / 8 episodes, ckpt 21,109, `ccos`, `W_JERK 0`, `W_VEND 64.2972`.
`kamm_over` at `v0 >= 2 m/s`, **`n = 27`**, a **COUNT** — quoted with its `n` and **never** with a
CI it does not have.

| `W_KAPPA` | **curv MAE** (vs floor 0.040083) | **`turn_left` recall** (n_true 11) | `kamm_over` | `max\|kappa\|` | vacuity (≥0.02) | `peak_g` max | ADE (m) |
|---|---|---|---|---|---|---|---|
| **0** `ccos_argmax` | 0.055369 ⛔ **worse than straight** | 0.3636 | ⛔ **0.2963** | 0.2000 | pass | ⛔ **3.262** | 1.3272 |
| **1** | 0.039406 (−0.00068, **under floor res.**) | **0.3636** | ⭐ **0.0000** | 0.0800 | ✅ 4.0× | 0.371 | 0.9388 |
| **3** | 0.038737 (−0.00135, **under floor res.**) | **0.3636** | ⭐ **0.0000** | 0.0800 | ✅ 4.0× | 0.332 | 0.9301 |
| ⭐⭐ **7** | ⭐ **0.033522 (−0.00656 = 3.3× floor)** | ⭐ **0.2727** | ⭐ **0.0000** | 0.0800 | ✅ 4.0× | 0.332 | 0.8935 |
| **15.11** `wk15` | **0.030982 (−0.00910 = 4.6× floor)** | ⛔ **0.0000** | 0.0000 | 0.0800 | ✅ 4.0× | 0.332 | 0.8934 |
| **151** `wk151` | 0.038019 | ⛔ 0.0000 | 0.0000 | 0.0166 | ⛔ **VACUOUS** | 0.158 | 0.9084 |
| by-goal `w_turn=0` `gkappa` | 0.040167 ⛔ **= the floor** | 0.3636 | ⛔ **0.0370** | 0.2000 | pass | ⛔ **2.176** | 0.9777 |
| **the human `g`** (CONTROL) | — | — | **0.0000** | 0.1701 | — | 0.373 | — |
| **FLOOR `ha0`** (perfectly straight) | **0.040083** | — | — | 0.0000 | — | 0.000 | 0.9251 |

⛔ **`I22` is honoured on every row**: no friction-circle zero appears in this package without
`turn_left` recall printed beside it. Three of the zeros above (`W_KAPPA` 15.11, 151, and every
`best` variant) are bought with recall **0.0000** and are labelled accordingly.

⭐ **`wk1` and `wk3` FAIL PASS clause 1 on the very floor this SPEC adopted** (−0.00068 / −0.00135,
both inside 0.00200). They turn more than `wk7` and they are zero and non-vacuous — but they are
**not** separated below the straight line, and they are reported as failing. That is what a
criterion that still bites looks like.

---

## 3. ⭐⭐⭐ WHY 7 — THREE SATURATIONS, MEASURED, AT THREE DIFFERENT PLACES

This is the part that makes `W_KAPPA = 7` an *interior optimum for a reason* rather than a bend in
a curve. All three are read off the banked dumps with **zero GPU**.

| what saturates | where | evidence |
|---|---|---|
| ⭐ **SAFETY** | **`W_KAPPA` = 1** | `kamm_over` already **0.0000** non-vacuously at 1; and the speed above which large curvature is never emitted is **6.378 m/s at `W_KAPPA` 1, 7, 15 AND `best` — identical** |
| ⭐ **LANE-KEEP straightening** | **≈ 7** | `LANE_KEEP` mean `k²`: **5.6e-5 → 1.8e-5 → 2.0e-6 → 0.0** at 1/3/7/15.11. By 7 there is nothing left to straighten |
| ⛔ **TURN EXECUTION** | **lost between 7 and 15.11** | `TURN_L` med\|k\|max holds at the commanded **0.08000** through 1/3/7 and falls to **0.02066** at 15.11 — while `TURN_R` holds at 0.08000 throughout |

⇒ **7 is the unique weight at which all three hold.** Every unit above it buys no safety (saturated
since 1), no straightening (saturated at 7), and costs the left turn.

### 3.1 ⛔ A hypothesis of mine, REFUTED by its own control

`max|kappa|` reads **exactly 0.0800** — `GOAL_KAPPA_TURN`, the `L1-0.08` vocabulary's single turn
level — on **seven** arms. I predicted the zero was therefore **structural**: `a_lat = v²κ`, so
`κ ≤ 0.08` is inside the `µg` circle below `sqrt(0.700·9.81/0.08)` = **9.265 m/s**.
⛔ **MEASURED: the panel reaches 15.853 m/s**, well above it — where 0.08 would read **2.05 g**.
**The cap alone does not guarantee the zero. HYPOTHESIS REFUTED.**

⭐ **And the refutation exposed the real mechanism.** The discriminator is the *speed at which*
large curvature is emitted:

| arm | max v where `\|κ\| ≥ 0.05` | max `a_lat` | in g |
|---|---|---|---|
| ⛔ `ccos_argmax` (`W_KAPPA` 0) | **15.087 m/s** | 31.96 | **3.258 g** |
| ⛔ `gkappa` (`w_turn` 0 on turns) | **10.579 m/s** | 21.28 | **2.169 g** |
| **the recorded human `g`** | 7.436 m/s | 3.199 | 0.326 g |
| `wk1` / `wk7` / `wk15` / `best` | **6.378 m/s** | 3.25 | 0.332 g |

⇒ ⭐⭐ **The curvature penalty's safety function is to stop the planner turning hard at speed.**
With it off, the circle is exceeded **4.7×**. With **any** `W_KAPPA ≥ 1`, large curvature is
confined below the human's own 7.436 m/s and the load lands at 0.332 g against her 0.326 g.

**Controls, same generation, both passed:** the human `g` reads `max|kappa|` **0.1797** — it
**must** exceed the vocabulary cap, being a real vehicle and not a token — and `ha0` reads
`max|kappa|` **exactly 0.0000**, being a perfectly straight line.

### 3.2 ⛔ `I22` sharpened — `wk15` does not REFUSE the turn, it UNDER-EXECUTES it

`frac k ≠ 0` stays **1.0000 on 9/9 `TURN_L`-goal windows** at `W_KAPPA` 15.11245. The plan turns on
every one of them, at **26 %** of what its own tactical brain commanded — below the labeller's
threshold. ⇒ **three failure modes must be distinguished and only two have instruments:**

1. **VACUOUS** — never moves (`cos_wk`); caught by the `M58` magnitude gate.
2. **DECLINES the class** — the `I22` case as written.
3. ⭐ **ACCEPTS the class and under-executes it** — invisible to the vacuity gate *and* to recall
   alone, visible only in realised curvature **conditioned on the decoded goal token**.

### 3.3 ⭐⭐ …and the `TURN_L` / `TURN_R` asymmetry is EXPLAINED — it moves the next lever off `W_KAPPA`

A **symmetric** `|κ|` penalty produced a **signed** outcome, which said the cause had to lie in the
term it trades against. **It does.** Reading the decisions sidecar's own cost columns per decoded
goal class (`raw/TURN_ASYMMETRY_EXPLAINED.md`, zero GPU):

| decoded goal | n | med `finecost_cv` | med `finecost_plan` | **med margin (cv − plan)** |
|---|---|---|---|---|
| `LANE_KEEP` | 18 | 0.9242 | — | **0.00000** |
| **`TURN_L`** | 9 | **1.7030** | 1.9964 | **−0.00812** |
| **`TURN_R`** | 13 | 0.7685 | 0.0526 | **+0.71638** |

⇒ ⭐⭐ **The penalty is symmetric; the BENEFIT is not.** The `ccos` goal term pays **+0.71638** for
turning right and **≈ 0** for turning left. On a `TURN_L` window the planned candidate is worth
essentially nothing over doing nothing, so **any** curvature price tips it — which is exactly why
the left turn dies first.

⭐ **The cleanest proof it is not the penalty's doing:** on `gkappa`, whose turn weight is **zero**,
the gap is still there and **larger** — `TURN_L` **−0.14789** vs `TURN_R` **+0.76118**. With the
curvature penalty switched off entirely on turn windows, the goal term still will not pay for a
left turn.

⇒ ⛔ **THE NEXT LEVER IS NOT `W_KAPPA` AT ALL.** The do-nothing candidate already costs **1.7030**
on a median `TURN_L` window against **0.7685** on a `TURN_R` one, and the planner's own choice
costs *more* than doing nothing there. **The goal field for a left turn is not giving the planner
anything to aim at**, and no setting of a curvature weight repairs that.

**Controls, all three read their known values:** `w_kappa_eff_cl` reads **7.00000** on every window
of the scalar `wk7` and on `gkappa` **exactly** the by-goal map (15.11245 / 0.0 / 0.0), which is
what proves `goal_lat_cl` is the field the cost actually keyed on; `LANE_KEEP`'s margin is
**exactly 0.00000** on all three arms, as it must be (a `LANE_KEEP` decode forces curvature exactly
0, so the plan *is* the constant-velocity candidate); and `wk15`'s `w_kappa_eff_cl` column is
**absent**, that arm predating the by-goal instrumentation — reported absent, not imputed.

---

## 4. ⛔⛔ THE TWO VARIANCES DISAGREE — and naming them is what decides the verdict

The brief binds: *name which variance each interval answers.* Here it is not bookkeeping — **it
flips the verdict on `wk7`.**

| question | instrument | `wk7` vs the straight floor |
|---|---|---|
| *would another **INFERENCE RUN** say this?* | the measured seed floor **0.00200** | margin **0.00656 = 3.3×** ⇒ ⭐ **CLEARS** |
| *would another draw of **EPISODES** say this?* | paired episode-cluster bootstrap, 8 clusters | **−0.0089 [−0.0242, +0.0016]** ⇒ ⛔ **NOT separated** |
| *would another **TRAINING RUN** say this?* | — | ⛔ **UNPRICED** (`I17`/V2); one checkpoint |

Only `wk15` — the arm that does **not** turn — separates on the episode axis
(**−0.0105 [−0.0259, −0.0009]**).

⛔ **The episode interval is reported INDICATIVE, NOT DECISION-GRADE, because a control failed.**
The recomputed `cl` mean reproduces the banked `curvature_mae_1pm` at `min_ds` 0.05 (**5/5 within
5e-4**), but the `ha0` side reads **0.042584 against the banked 0.040083** — a 0.0025 gap, **above
the 0.00200 floor** — and **no masking gate tried reproduces both arms**. It is quoted at all only
because a **sensitivity sweep over `min_ds` 0.02 / 0.05 / 0.10 returns the identical
separated / not-separated verdict on every arm**.

⇒ ⭐ **This names the next lever precisely, and it is the OPPOSITE of the tiny-rig guidance:** the
binding interval here is the **episode** one and it is wide at `n = 40 / 8`. **The cheapest next
experiment is MORE EPISODES, not more seeds.** (Seeds were still owed and were run — §5 — because
the *recall* claim and the *safety* claim are seed claims; the *curvature* claim is an episode one.)

⚠️ **A stated gap, not a silent omission:** per-class lateral recall carries **no interval** in the
records, and one could **not** be constructed from the dumps — the sidecar's `lat_label` is a
different label object from the four-family TACTICAL block's (it reads `n_true = 2` for
`lane_keep` against the record's 21, so the control failed and nothing was quoted).
`raw/recall_ci_NOT_CONSTRUCTIBLE.txt` records the attempt and the one-array fix.

---

## 5. Replication across inference seeds

⛔ **refav1's planner SAMPLES**, so a single-seed arm is not a result. `wk7` seeds 1 and 2 and the
goal-conditioned rungs were launched on the dev-box 4060 under the pre-registered protocol
(`SPEC.md` §2.3, ≥ 3 inference seeds); the as-launched argv audit is `raw/AS_LAUNCHED.txt` and each
arm's `[cost]` / `[cost-goal]` / `[grid]` header is in `out/<tag>.log`, differing from the endpoint
in **exactly one token**.

**STATUS: see `RESULT_SEEDS.md` in this package.** Until that file reports them, every row above is
a **single-inference-seed** number and is labelled so.

---

## 6. The four families, per family, never pooled

**LONGITUDINAL** — ⛔ **every rung LOSES to `ha` by ~0.44 m/s.** `cl − ha0_ext` speed MAE is
**separated WORSE** on all of them (`wk1` **+0.4384 [+0.2558, +0.6565]**, `wk3` **+0.4395**,
`wk7` **+0.4458 [+0.2584, +0.6792]**) while every `cl − ha0_ext` **ADE** is **NOT** separated.
⇒ **the rungs tie on position and lose on speed** — the split ADE hides.
⚠️ **Distance-keeping UNAVAILABLE, `n = 0`**, reason stated by the tool: *"no lead-agent track
supplied"*. Per clause 5, reported with its reason, not dropped.

**LATERAL** — the frontier metric; §2 and §3 above. `wk7` heading MAE 19.08° vs the `ha0` floor
20.14°; yaw-rate MAE 6.87 vs 7.09.

**TACTICAL** — `wk7` lat acc 0.7000, lat κ **0.4552** (vs the `ha0_ext` control's 0.6277), lon acc
0.4500, goal FDE 2.1700 m. Per-class lateral recall is the frontier's second axis (§2).

**STRATEGIC** — ⚠️ **UNAVAILABLE, `n = 0`**, reason stated: *"strategic decisions not present in
the scored pass (missing `['route_pred', 'route_true']`)"* — the refav1 arm tool emits no route
head. A stated gap with its reason and its `n`.

**The known-value control that licenses every comparison:** the four model-free arms are
**bit-identical across all nine records read** — `ha` 0.8888 / `ha0` 0.9251 / `ha0_ext` 0.8772 /
`ol` 0.8052, `ha0` curvature **0.040083**. One surface, not nine.

---

## 7. ⛔ The census correction, and what it costs an earlier claim

Auditing **25** p4 arms in **ONE generation** finds **16** configurations at `kamm_over` **0.0000**
(`v0 ≥ 2`, `n = 27`) — **12 non-vacuous**, **4 vacuous** — not six:

* **non-vacuous:** `wk1`, `wk3`, `wk7`, `wk15`, `best`, `best_seed1`, `combined` (s0 only),
  `seambase`, `seamon`, `lonseam`, `loncomb3`, `lonshift`
* **vacuous:** `cos_wk` 0.0000, `wk15_ladder` 0.0020, `bestlad` 0.0049, `wk151` 0.0166

⛔ **`wk1`, `wk3` and `wk7` had never had their feasibility measured at all.** The earlier count was
again **scoped to the arms one artifact happened to hold** — the same error its own root-cause note
(`D-REFAV1-CG-ZEROVIOL-SCOPE`) was written about. ⇒ **the friction-circle zero is bought at
`W_KAPPA = 1`**, and `best`'s standing as the arm that *earns* it does not survive the wider census.
⭐ What survives, and is re-verified here: `best`'s zero **is** replicated across two inference
seeds and **is** non-vacuous — but it is not rare, and it is bought with `turn_left` **0.0000**.

---

## 8. What this does NOT establish

1. ⛔ **Not closed loop.** T0/T1 only.
2. ⛔ **40 windows / 8 episodes, one checkpoint (21,109).** No corpus-scale or parity claim.
3. ⛔ **Inference-seed replicates are not training replicates.** `I17`/V2 unpriced for refav1 and
   named as a blocker, not discharged.
4. ⛔ **`kamm_over_rate` is a feasibility statistic, not a collision or off-road rate.** This
   programme has no reactive-agent simulator, so no safety-grade behavioural metric exists here.
5. ⛔ **The curvature claim is not established under episode variance** (§4). That is the single
   biggest hole and it has a named, cheap fix.
6. **Stack scope:** the `tanitad-ctg` clone — the same snapshot that produced every banked p4 arm,
   chosen for comparability. It differs from repo HEAD by an **additive, default-off**
   distance-keeping cost term (`dk_spec=None` ⇒ bit-identical), a live sibling's work.

## 9. Successors, in priority order

1. ⭐⭐ **More EPISODES, not more seeds**, for the curvature claim — §4 measures that the episode
   interval is the binding one. The `p4` slice is 8 episodes; the eval corpus has 40.
2. ⭐⭐ **REPAIR THE `ccos` GOAL TERM'S LEFT-TURN FIELD** — §3.3 measured that this, not `W_KAPPA`,
   is where the turn is lost: the goal term pays **+0.716** for a right turn and **≈ 0** for a left
   one, *with the curvature penalty switched off*. No curvature weight can repair a goal field that
   offers nothing to aim at.
   ⭐ **And the decomposition is EXACT, not inferred** (`raw/LEFT_TURN_GOAL_FIELD.md`): the
   constant-velocity candidate has `kappa ≡ 0`, so `w·kappa²` contributes **exactly zero** to
   `finecost_cv` — **checked, not argued**, because `med finecost_cv` is **bit-identical across
   `wk7`, `wk15` and `gkappa`** (0.9242 / 1.7030 / 0.7685 per class in all three). ⇒ the
   `TURN_L`/`TURN_R` ratio of **2.22×** sits entirely outside the curvature penalty.
   ⛔ `H-CCOS-TURNL-FIELD` — *the left-turn goal field points where no candidate reaches* — is
   registered as a **HYPOTHESIS, not a result**: `finecost` is the composed cost and carries
   `W_VEND = 64.2972`, so `finecost_plan` **1.9964** on `TURN_L` (against **0.0526** on `TURN_R`)
   is *consistent with* anti-alignment and is not proof of it. **The discriminating experiment is
   pre-registered with both outcomes** — emit the goal and `W_VEND` terms as separate sidecar
   columns and re-read the table.
3. **A second lateral vocabulary level.** `max|kappa|` is pinned at `GOAL_KAPPA_TURN = 0.08` on
   seven arms while the human reaches **0.1797**; the planner's only curvature-carrying candidate
   is the decoded token's canonical profile, so **no weight can cross that ceiling** — `L1-0.08` is
   a hard cap on how sharply this arm can ever turn.
4. **Emit the tactical block's per-window hit indicator** into the decisions sidecar (one array),
   which would make every future per-class recall claim interval-bearing (§4).
5. **The goal-conditioned rungs at `w_turn` 5.04 / 10.07** complete the second curve; `gkappa`
   (`w_turn` 0) is already measured to be worse than every scalar rung on every axis, so the
   by-goal form is **not** currently a better frontier than the scalar one.
