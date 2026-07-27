# THE TACTICAL BRAIN'S MISSING ACTION INPUT — verified in code, measured on the closed-loop surface, and **REFUTED as the mechanism**

**Wall-clock date:** 2026-07-27 (Europe/Berlin). ⚠️ The directory is dated `2026-07-28` because the
brief named it so; the repo's narrative clock runs ~1 day ahead of wall-clock (known artefact).
**Stream:** Architecture & Inference · **Branch:** `agent/benchmarks-eval-20260721` · **Repo HEAD at start:** `d07da62`
**Pre-registration:** `PRE_REGISTRATION.md` in this directory, frozen before any new number existed.
**Instrument:** `taniteval/taniteval/pseudosim.py` (md5 `86a11c46b6949bcfc4805bfae29fa27d`), **imported, never reimplemented.**
**Hosts:** dev box (Block B — no GPU at all) · `tanitad-eval` A40 (Block A). ⛔ pod1 (training) and pod2 (120° build) were **not touched**.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not re-verified)
· `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ⛔ **THE BRIEF'S PREMISE IS PARTLY FALSE, AND THE TRUE VERSION IS WORSE.** `TacticalPolicy.forward(states, ctx, ego=None)` **does** take an ego vector (`fourbrain.py:328`). It is inert for **two independent reasons**: the `v2_ego_to_planners` lever is off by default (`config.py:196`, and `ego_input_on_planners = False` on every panel arm), **and — not previously named anywhere — every one of the 8 eval/planner call sites drops the argument.** An ego-**trained** checkpoint is evaluated ego-**blind**, silently, with no error. | `MEASURED` **tier 1** |
| **2** | ⛔⛔ **PRE-REGISTERED VERDICT: `REFUTE`.** Handing v1's tactical plan the speed it never saw moves the pseudo-simulation composite **+0.0078 [−0.0110, +0.0260] — NOT SEPARATED** from its own no-input control. It also stays **separated BELOW `cv_holdv0`** (−0.0100 [−0.0170, −0.0033] SEP). **The missing action input is not the mechanism, and I do not re-scope.** | `MEASURED` **tier 2** |
| **3** | ⭐⭐ **AND THE REASON IS THE FINDING: THE MECHANISM IS REAL AND HUGE, AND THE COMPOSITE CANNOT SEE IT.** The same intervention cuts v1's along-track endpoint RMS from **8.799 m → 1.557 m (5.65×)** and its longitudinal share of squared error from **76.9 % → 8.9 %**. **A 5.65× longitudinal fix returns `n.s.` on `PSS_recovery_progress`.** | `MEASURED` **tier 1** (arithmetic) |
| **4** | ⭐⭐ **A PERFECT LONGITUDINAL INPUT *DOES* BEAT CONSTANT VELOCITY — separated.** `v1_ego_oracle_lon` (v1's own curve, walked to the **true** logged distance) scores **0.5946 [0.5868, 0.6033]** vs `cv_holdv0` **0.5705**: paired **+0.0228 [+0.0064, +0.0412] SEP**, and **+0.0407 [+0.0308, +0.0512] SEP** over its own control. ⚠️ **ORACLE — an upper bound, not deployable.** *To my knowledge the first thing in this program to clear CV on this surface, and it needs the future to do it.* | `MEASURED` **tier 2**, ⚠️ oracle |
| **5** | ⭐⭐⭐ **THE REALISABLE/ORACLE GAP LOCATES THE ACTUAL MISSING QUANTITY, AND IT IS NOT `v0`.** `v0·t` recovers **89.3 %** of the achievable along-track RMS reduction but only **19.2 %** of the composite's oracle move. **The missing input is the 2 s FUTURE DISPLACEMENT, which `v0` does not determine** — independently converging with `V5_PLAN` §8's E-GOAL result (*"the 88 % lives on how far the car will travel in 2 s"*). | `MEASURED` **tier 2** |
| **6** | ⛔ **v1's TACTICAL STEERING IS WORTH NOTHING, AND IS HARMFUL ONCE THE TIMING IS FIXED.** Replacing v1's entire curve with a straight line on its own schedule: **+0.0006 [−0.0065, +0.0072] n.s.** (the tightest contrast in this study). On the `v0·t` schedule the straight line is **separated BETTER**: **+0.0100 [+0.0033, +0.0170] SEP**. Cross-track: v1 **3.604 m** vs a straight line **3.448 m**. | `MEASURED` **tier 2** |
| **7** | ⛔ **THE NAMED GATE TRAP FIRED ON MY OWN PRIMARY RESULT.** Under the shipped **per-arm** gate the same contrast reads **+0.0836 SEPARATED** — a **verdict flip**. Cause, measured: re-timing at constant speed *smooths* the plan, so `comfort` jumps **0.0004 → 0.2882 (720×)** and enters the composite for both arms. **The per-arm gate would have manufactured a CONFIRM out of a smoothing artefact.** I use the panel-wide gate throughout and report both. | `MEASURED` **tier 1** |
| **8** | ⚠️ **`ego_progress` IS ONE-SIDED AND ALREADY SATURATED, WHICH IS WHY (3) HAPPENS.** It is `clamp(along/human, 0, 1)`: **56.0 %** of v1's rows are *already* at the ceiling, so there is no room to pay for an improvement; and a plan travelling **twice** as far scores `ego_progress` **+0.0785 SEP HIGHER** than the control. ⛔ **The component cannot punish over-travel at all.** | `MEASURED` **tier 1** |
| **9** | ⚠️ **THE METRIC REFUSES A SOLVED PROBLEM.** A straight plan walked to the true distance (`oracle_lon_straight`) has `ego_progress` spanning **0.9781–1.0000**, `observed_range = 0.0219 < RANGE_MIN = 0.05` ⇒ **`range below range_min` ⇒ INADMISSIBLE**. An arm that gets the axis right is refused by the gate that exists to keep the axis meaningful. | `MEASURED` **tier 1** |
| **10** | ✅ **THE DEGRADATION GUARD HELD — the composite did NOT rise for a degraded planner.** `v1_ego_half` **−0.2421 [−0.2565, −0.2285] SEP worse**. ⚠️ The over-travel probe is a **partial** pass: its composite falls only because `ego_progress` becomes inadmissible for it; on the admissible components it goes **UP**. Reported, not hidden. | `MEASURED` **tier 1** |
| **11** | ⭐ **A THIRD GAP, FOUND WHILE PINNING THE SECOND: `FiLM.to_scale_shift` IS ZERO-INITIALISED** (`predictor.py:25-26`). On a from-scratch policy the **entire `cond` path — `ctx` and any ego graft — has zero effect and zero gradient at init.** A v5 ego graft on a fresh brain must account for this; on a *trained* brain it does not apply. | `MEASURED` **tier 1** |

### 0.1 The verdict in one sentence

**The tactical brain really is speed-blind, the blindness really does cost 5.65× in along-track
accuracy, and fixing it with the ego speed really does not move the closed-loop composite — because
the composite is saturated on the axis the fix repairs, and because the quantity that *would* move it
is not the current speed but the 2 s future displacement, which `v0` does not determine.**
⛔ **Pre-registered outcome: `REFUTE`. Constant velocity is still not beaten by any realisable arm,
and I say so.**

### 0.2 Tier, stated as required

**Tier 2** for every composite number, inheriting all four of the panel's qualifiers verbatim
(**oracle goal** where used, **non-reactive log replay**, **no collision or TTC gate**, **`comfort`
dropped by the gate, not retuned**), plus two of my own: (a) Block B's arms are **plan transforms, not
trained planners** — they bound what an ego input could buy, they are not deployable controllers;
(b) the two `oracle` arms **read the ego's own future** and are upper bounds only.
The code reading (§1), the decomposition (§4) and the instrument findings (§5) are **tier 1**:
deterministic, model-free, CPU, and recomputable from the committed dumps with no GPU.

---

## 1. ⭐ PRIORITY 1 — THE VERIFIED CODE READING (the brief said not to take this from the brief)

All line numbers at repo HEAD `d07da62`. Every row was read in the file, not inferred from a docstring
— **class C21 was earned this week by treating a docstring as verification.**

### 1.1 What the tactical policy actually sees today

```python
# stack/tanitad/models/fourbrain.py:328-339   ← THE SIGNATURE
def forward(self, states: Tensor, ctx: Tensor,
            ego: Tensor | None = None) -> dict:
    if self.ego_emb is not None and ego is not None:      # ← BOTH must hold
        ctx = ctx + self.ego_emb(ego.to(ctx.dtype))
    ...
    x = self.in_proj(states) + self.pos[:, :w]            # states: VISION ONLY
    cond = ctx.unsqueeze(1).expand(-1, w, -1)             # cond: ctx (+ego)
```

| # | question the brief told me to answer | verified answer | source |
|:--:|---|---|---|
| 1 | Is there an action input? | ⛔ **The brief is wrong: there IS an `ego` parameter.** | `fourbrain.py:328` |
| 2 | Is `v0` genuinely absent, or merely unused? | ⭐ **BOTH, and that is the point.** `ego_emb` is `None` unless `v2_ego_to_planners` (`fourbrain.py:326,443`; `config.py:196` default `False`) — **absent**. AND the argument is never passed by any caller — **unused**. Each alone would suffice; both hold. | below |
| 3 | What does `ctx` already carry? | `ctx = StrategicPolicy(states, nav_cmd, ego)`. `states` are the ViT-readout of the **image window only**; `nav_cmd` is a 4-way route token. ⇒ **no speed, no yaw rate, no action anywhere on the path.** | `fourbrain.py:73-86` |
| 4 | Does the strategic level have the same gap? | ✅ **Identical**, same `ego=None`, same gating. | `fourbrain.py:73-79` |
| 5 | Does the **operative** level have the gap? | ⛔ **No.** `run_hierarchy` calls `model.predictor(states, actions, intent=…)`; under `--speed-input` the trainer appends `v0/10.0` as the **third action channel**. | `fourbrain.py:388`, `flagship_losses.py:227-238` |
| 6 | Is it an "action" input? | ⚠️ **No — `ego = [v0/pose_scale, yr0]` is PROPRIOCEPTION.** Different port *and different scale* from the operative action channel (`/10.0`). Swapping them decodes garbage (registry §1.2). | `flagship_losses.py:202-210` |

### 1.2 ⭐ THE SECOND GAP — not named in the brief, the panel, or `V5_PLAN`

**Every** call site invokes the policy positionally with two arguments. Even a checkpoint trained with
`--v2` — which *has* `ego_emb` weights — is evaluated **ego-blind**, silently, with no error and no log line:

| file | line | call |
|---|---|---|
| `taniteval/taniteval/closedloop.py` | 245, 317 | `model.tactical_policy(win_s, ctx)` |
| `taniteval/taniteval/planner_p2.py` | 279, 340 | `model.tactical_policy(states, ctx)` |
| `taniteval/taniteval/planning.py` | 166 | `model.tactical_policy(states, sf["ctx"])` |
| `taniteval/taniteval/corpus_overlay.py` | 307 | `model.tactical_policy(states, sf["ctx"])` |
| `taniteval/taniteval/blindimag.py` | 101 | `tactical_policy(win_s, …["ctx"])` |
| `taniteval/probe_overlay.py` | 49 | `model.tactical_policy(states, sf["ctx"])` |
| `…/2026-07-27-pseudosim-arm-panel/scripts/panel_run.py` | 137, 165 | `model.tactical_policy(st, ctx)` |
| `stack/tanitad/refs/refa.py` | 260 | `run_hierarchy(self, states, actions, nav_cmd)` — no `ego=` |

⇒ ⛔ **`flagship-v2corpus-30k`, now training on pod1 with `v2_ego_to_planners = true`, will be
evaluated ego-blind by every harness in the repo unless this is fixed.** That is escalation **E1**
and it is the single most actionable item in this report. Pinned by
`test_passing_ego_to_a_stock_policy_is_SILENTLY_IGNORED`.

### 1.3 ⭐⭐ The fact that made `REFUTE` a live outcome before I measured anything

`RefCModel.forward(frames, nav_cmd=None, v0=None, …)` — `refc.py:760` — and `:786-787`:

```python
v = torch.zeros(b, 1, …) if v0 is None else (v0.to(pooled.dtype) / 10.0).reshape(b, 1)
```

⇒ **REF-C's planner ALREADY consumes the ego speed, on the same `SPEED_SCALE = 10.0` contract** — and
REF-C is **separated BELOW `cv_holdv0` at all three scales** (−0.0203 … −0.0255, `PUBLISHED`). A
planner family that *has* the input already loses to doing nothing.
⚠️ **Honesty condition, found before it could be quoted:** REF-C trains with `ego_dropout = 0.5`
(`refc.py:287`), so it was deliberately trained *not* to lean on `v0`. This both makes a `v0 = 0`
ablation **in-distribution** (which is what licenses Block A) and **bounds** what Block A can conclude.

### 1.4 The third gap, found while pinning the second

`FiLM.to_scale_shift` is **zero-initialised** (`predictor.py:25-26`), so on a freshly built policy the
whole `cond` path — `ctx` *and* any ego graft — is numerically dead **and has no gradient**. Pinned by
`test_a_fresh_policy_ignores_ctx_entirely_because_film_is_zero_init`. On a *trained* policy the FiLM is
non-zero and the seam is live; this matters only for how a v5 graft is initialised and warmed up.

---

## 2. The design, and why it is this one

⛔ **A trained ego-on flagship arm was not producible and I say so plainly.** Only two flagship
checkpoints ever set `v2_ego_to_planners = true`: `flagship4b-v2-30k` (registry §1.3, **ABANDONED at
step 7,800**, ADE@2s **6.179 m**, and on **pod2**) and `flagship-v2corpus-30k` (§1.7, **RUNNING on
pod1**). The brief forbids both hosts, and the first is a broken arm whose ego/no-ego contrast would be
read at a non-competent operating point. Training a new one is out of budget.

So the question is reached two ways that need no training, and **neither is "a flagship retrained with
`--v2`"**:

**Block B (primary).** A plan is a **curve** `γ(s)` plus a **schedule** `s_t`. v1's tactical head emits
both, having never seen `v0`; an ego-speed input's entire job is the **schedule**. Re-timing v1's own
curve onto a `v0`-derived schedule therefore measures **exactly what a speed input could buy**, at
**zero fitted parameters** — so there is no out-of-fold question and it cannot be dismissed as
under-trained. Arithmetic on the panel's committed per-window dumps: **no GPU, no checkpoint, no corpus.**

⭐ **The construction validates itself in BOTH directions, because two of its four cells are
already-published arms.** The build script **refuses to write anything** unless both hold:

| identity | required | **MEASURED** |
|---|---|---|
| `γ_v1 ∘ s_v1` reproduces `pw_v1_tactical_follow` | < 1 mm | **7.629 × 10⁻⁶ m** ✅ |
| `γ_straight ∘ s_cv` reproduces `pw_cv_holdv0` | < 1 mm | **0.000 × 10⁰ m — exact** ✅ |

**Block A (corroboration).** Ablate REF-C's **trained** speed input inside one checkpoint: identical
weights, identical rows, `v0` real vs `v0 = 0`. In-distribution by `ego_dropout = 0.5`.

**Reproduction gate, run before the pre-registration was frozen.** `panel_combine.py` re-run locally
on the 10 committed dumps with `CUDA_VISIBLE_DEVICES=""`: **all 10 composites and all 45 paired blocks
byte-identical** to the published artifact. Every published arm below still reads exactly its
published value with my 6 arms added — so the added arms did not perturb the gate.

---

## 3. THE RESULT

40 val episodes · stride 8 · 21 grid points · **15,981 rows per arm, 0 rollout steps** · row identity
asserted by `panel_combine.py` (a non-matching arm is **refused**, not dropped). Estimator:
`taniteval.ci.paired_episode_cluster_bootstrap`, **B = 2000, unit = val episode**.
⛔ `overlapping_holdout_se` **refused**. **PANEL-WIDE gate**: `ego_progress` + `recovery`, `comfort`
dropped for every arm.

### 3.1 Arm scores

| arm | what it is | **PSS** | `ego_progress` | `recovery` | rec. def. frac |
|---|---|---|---|---|---:|
| ⚠️ `v1_ego_oracle_lon` | v1's curve, **true** logged distance — ORACLE | **0.5946 [0.5868, 0.6033]** | 0.9805 | 0.0741 | 0.8234 |
| **`cv_holdv0`** | ⛔ **the bar** | **0.5705 [0.5558, 0.5844]** | 0.9407 | 0.0776 | 0.8203 |
| `v4_oracle` | published | 0.5622 [0.5496, 0.5725] | 0.9462 | 0.0629 | 0.8377 |
| ⭐ **`v1_ego_v0`** | **v1's curve, `v0·t` — THE ARM WITH THE SPEED INPUT** | **0.5608 [0.5470, 0.5727]** | 0.9324 | 0.0653 | 0.8196 |
| `refc_xl_produced` | published (already has `v0`) | 0.5499 [0.5421, 0.5566] | 0.9438 | 0.0259 | 0.8256 |
| **`v1_tactical_follow`** | ⭐ **the no-input control** | **0.5471 [0.5340, 0.5595]** | 0.9081 | 0.0785 | 0.8423 |
| `v1_tactical_oracle` | published | 0.5467 [0.5338, 0.5591] | 0.9047 | 0.0817 | 0.8419 |
| `v1_lat_straight` | v1's schedule, **no steering at all** | 0.5460 [0.5288, 0.5608] | 0.9138 | 0.0747 | 0.8459 |
| `refc_small_produced` | published | 0.5444 [0.5360, 0.5514] | 0.9315 | 0.0296 | 0.8253 |
| `refc_base_produced` | published | 0.5439 [0.5345, 0.5519] | 0.9317 | 0.0293 | 0.8250 |
| `nospeed_tactical_oracle` | published | 0.5394 [0.5242, 0.5540] | 0.8961 | 0.0800 | 0.8451 |
| `v4_blind` | published | 0.3749 [0.3076, 0.4368] | 0.5999 | 0.1159 | 0.5748 |
| ⛔ `v1_ego_half` | **probe**: half speed | 0.3117 [0.3019, 0.3217] | 0.4864 | 0.0761 | 0.8116 |
| ⛔ `v1_ego_double` | **probe**: double speed | *(not comparable — §5.2)* | **0.9866** | 0.0706 | 0.8234 |
| ⚠️ `oracle_lon_straight` | **probe**: straight + true distance | *(refused — §5.3)* | 0.9916 | 0.0816 | — |
| ⛔ `stand_still` | published adversary | **`VacuousMetric` — no score** | 0.0000 | — | 0.0000 |

⚠️ **All ten published arms reproduce their published values exactly** (`cv_holdv0` 0.5705,
`v4_oracle` 0.5622, `refc_xl` 0.5499, `v1_tactical_follow` 0.5471, `nospeed` 0.5394, `v4_blind` 0.3749).
The four probes are **excluded from the panel gate**, following the panel's own rule for `stand_still`
(§3.2 there) — a probe whose purpose is to be refused must not delete the metric for every real arm.
⚠️ I discovered this the hard way: my first combine let `v1_ego_double` into the gate and it **dropped
`ego_progress` for all 16 arms**. Recorded in §7.

### 3.2 ⛔ The pre-registered contrasts

| contrast | **Δ PSS (panel gate — USED)** | per-arm gate (sensitivity) | verdict |
|---|---|---|---|
| ⭐ **`v1_ego_v0` − `v1_tactical_follow`** | **+0.0078 [−0.0110, +0.0260] n.s.** | +0.0836 **SEP** ⛔ **FLIP** | ⛔ **REFUTE** |
| **`v1_ego_v0` − `cv_holdv0`** | **−0.0100 [−0.0170, −0.0033] ⭐ SEP** | −0.0673 SEP | still below the bar |
| ⚠️ `v1_ego_oracle_lon` − `v1_tactical_follow` | **+0.0407 [+0.0308, +0.0512] ⭐ SEP** | +0.0928 SEP | oracle ceiling |
| ⚠️ **`v1_ego_oracle_lon` − `cv_holdv0`** | **+0.0228 [+0.0064, +0.0412] ⭐ SEP** | −0.0483 SEP | ⭐ **beats the bar, with the future** |
| ⚠️ `v1_ego_oracle_lon` − `v4_oracle` | +0.0264 [+0.0177, +0.0362] ⭐ SEP | −0.0411 SEP | |
| `v1_ego_v0` − `v1_tactical_oracle` | +0.0082 [−0.0104, +0.0261] n.s. | +0.0839 SEP | replication of the primary |
| `v1_ego_v0` − `nospeed_tactical_oracle` | +0.0136 [−0.0063, +0.0330] n.s. | +0.0879 SEP | |
| `v1_ego_v0` − `v4_oracle` | −0.0066 [−0.0165, +0.0029] n.s. | −0.0535 SEP | |
| `v1_ego_v0` − `refc_xl_produced` | +0.0103 [+0.0019, +0.0187] ⭐ SEP | +0.0862 SEP | |
| ⛔ `v1_ego_half` − `v1_tactical_follow` | **−0.2421 [−0.2565, −0.2285] ⭐ SEP** | −0.0436 SEP | ✅ guard holds |

**Reading the rules exactly as written in the pre-registration:**

* **CONFIRM** required `Δ(v1_ego_v0 − v1_tactical_follow)` separated-positive. It is **+0.0078
  [−0.0110, +0.0260]** — the interval contains 0. **CONFIRM does not fire.**
* **PARTIAL** required the same separated-positive precondition. **PARTIAL does not fire either.**
* ⛔ **`REFUTE` fires**, as the only rule whose condition is met.

⭐ **Every rule was able to fire, demonstrated on these exact rows, not asserted.** The same estimator
on the same 15,981 rows returns **SEPARATED** for `cv − refc_base` (+0.0252), for
`v1_ego_oracle_lon − control` (+0.0407) and for the G1 gate (+0.1882); it returns **n.s.** for
`cv − v1_oracle` (+0.0182). A `lo > 0` on the primary would have produced CONFIRM or PARTIAL; it did not.

---

## 4. ⭐ THE DECOMPOSITION — and it is where the real result lives

The regression this addresses was **100 % longitudinal**, so no composite delta is admissible without
its axes. Computed with `taniteval.pseudosim._cross_and_along` **imported** — the same function the
composite is built on. Sign: `along +` = the plan travelled **further** than the human.

| arm | along err mean | along **\|err\|** | along **RMS** | cross **\|err\|** | **longitudinal share of sq. err** |
|---|---:|---:|---:|---:|---:|
| `nospeed_tactical_oracle` | +0.747 | 6.611 | 9.146 | 3.563 | **78.9 %** |
| **`v1_tactical_follow`** | +1.041 | **6.246** | **8.799** | 3.604 | **76.9 %** |
| `v1_lat_straight` | +1.201 | 6.240 | 8.780 | 3.581 | 77.3 % |
| `v4_oracle` | −0.458 | 1.509 | 2.401 | 3.531 | 18.7 % |
| `refc_xl_produced` | −0.353 | 1.199 | 1.654 | 4.264 | 6.7 % |
| ⭐ **`v1_ego_v0`** | −0.529 | **1.154** | **1.557** | 3.547 | **8.9 %** |
| `cv_holdv0` | −0.366 | 1.067 | 1.465 | **3.448** | 8.5 % |
| ⚠️ `v1_ego_oracle_lon` | −0.275 | 0.407 | 0.686 | 3.550 | 1.8 % |
| ⚠️ `oracle_lon_straight` | −0.096 | **0.331** | **0.534** | 3.462 | 1.2 % |

### 4.1 ⭐⭐ The number that should decide something

**Giving the tactical plan the ego speed cuts its along-track endpoint RMS 8.799 m → 1.557 m — a
5.65× reduction — and collapses the longitudinal share of its squared error from 76.9 % to 8.9 %.
The closed-loop composite responds with +0.0078, n.s.**

⇒ **The mechanism the panel named is REAL and LARGE. The surface the panel ranked on cannot resolve it.**
Both halves have to be said at the same volume, and §5 is why.

### 4.2 The realisable/oracle gap locates the actual missing quantity

| quantity | control | `v1_ego_v0` (realisable) | `v1_ego_oracle_lon` (oracle) | **share of the gap `v0` closes** |
|---|---:|---:|---:|---:|
| along-track RMS (m) | 8.799 | 1.557 | 0.686 | ⭐ **89.3 %** |
| Δ PSS vs control | 0 | +0.0078 | +0.0407 | ⛔ **19.2 %** |

⭐⭐ **`v0` closes 89.3 % of the achievable longitudinal error but only 19.2 % of the composite's move.
Four fifths of the score lives in the last 10.7 % of longitudinal error** — a wildly non-linear
response that is explained exactly by §5.1.
⇒ **The input the tactical brain is missing is not the current speed; it is HOW FAR THE CAR WILL
TRAVEL IN THE NEXT 2 s.** `v0` is a good but insufficient estimator of it.
⭐ **This converges, from a completely different instrument, with `V5_PLAN` §8's E-GOAL result** —
*"oracle along-track recovers +83.7 % (separated); oracle cross-track +2.9 % (NOT separated) ⇒ the 88 %
lives on how far the car will travel in 2 s, not on where the road goes."* Two independent streams,
same axis, same conclusion.

### 4.3 ⛔ The lateral axis: v1's tactical steering is worth nothing

The 2×2 factorial (Δ PSS, paired, identical rows, panel gate):

|  | shape = **v1's** (it steers) | shape = **straight** | **shape effect** (straight − v1) |
|---|---|---|---|
| schedule = **v1's own** | `v1_tactical_follow` **0.5471** | `v1_lat_straight` **0.5460** | **+0.0006 [−0.0065, +0.0072] n.s.** |
| schedule = **`v0·t`** | `v1_ego_v0` **0.5608** | `cv_holdv0` **0.5705** | ⭐ **+0.0100 [+0.0033, +0.0170] SEP** |
| **schedule effect** (`v0·t` − own) | +0.0078 [−0.0110, +0.0260] n.s. | +0.0149 [−0.0041, +0.0327] n.s. | |

* ⛔ **Deleting v1's entire tactical curve and replacing it with a straight line costs +0.0006,
  n.s. — the tightest contrast in this study.** On its own schedule, v1's steering is indistinguishable
  from not steering.
* ⛔⛔ **Once the timing is correct, v1's steering is separated-HARMFUL** (+0.0100 SEP in favour of the
  straight line). Corroborated on the raw axis: cross-track **3.604 m** (v1) vs **3.448 m** (straight),
  and `oracle_lon_straight`'s along-track error (0.331 m) is *better* than `v1_ego_oracle_lon`'s
  (0.407 m) — **v1's curvature costs accuracy on both axes.**
* ⇒ **On this surface the flagship's tactical head contributes no lateral value over a straight line.**
  That is a stronger and more uncomfortable statement than the one this brief set out to test, and it
  is the tightest-interval result in the report.

---

## 5. ⚠️ WHY THE COMPOSITE COULD NOT SEE A 5.65× FIX — three measured instrument facts

### 5.1 `ego_progress` is one-sided and already saturated

`ego_progress = clamp(along_end / human_dist, 0, 1)` (`pseudosim.py:499-503`).

| arm | `ego_progress` mean | **frac. of rows already at the ceiling** |
|---|---:|---:|
| **`v1_tactical_follow`** | 0.9081 | ⛔ **55.91 %** |
| `cv_holdv0` | 0.9407 | 26.90 % |
| `v1_ego_v0` | 0.9324 | 24.49 % |
| `v1_ego_oracle_lon` | 0.9805 | 17.77 % |
| ⛔ `v1_ego_double` | **0.9866 — the highest in the panel** | 97.49 % |

⛔ **Two defects at once.** (a) **56 % of the control's rows are already scored 1.0**, so the metric
cannot pay for an improvement on them — an improvement can only be banked on the other 44 %.
(b) **The clamp is one-sided: over-travel is never punished.** A plan travelling **twice** as far as it
should (50.07 m against a 25.4 m ground truth — a ~90 km/h plan where the car did 45) scores
`ego_progress` **+0.0785 [+0.0565, +0.1006] SEP HIGHER** than the control, and higher than the
*perfectly correct* arm. **`ego_progress` cannot distinguish "exactly right" from "twice too far".**

⚠️ **Does this inflate my own `v1_ego_v0` result?** No — and I checked in the direction that could have
embarrassed me. `v1_ego_v0` travels **less** on average than its control (25.08 m vs 26.36 m mean
endpoint x), so its `ego_progress` gain comes from removing **under**-travel, not from harvesting the
one-sided clamp. The one-sidedness makes my headline **conservative**, not generous.

### 5.2 The over-travel probe's composite is not a model comparison

`v1_ego_double`'s `ego_progress` ceiling fraction is **97.49 % ≥ 0.95** ⇒ inadmissible for that arm
⇒ its own composite collapses to `recovery` alone (0.0706). The paired "−0.4189 SEP" therefore compares
a **one-component** composite against a **two-component** one — ⛔ **exactly the defect the panel gate
exists to prevent, and it is not a model comparison.** The admissible reading of that control is its
components, where it goes **up** on progress and is n.s. on recovery.
⇒ **My degradation guard passes cleanly in the under-travel direction and only partially in the
over-travel direction.** Stated, not hidden.

### 5.3 The metric refuses a solved problem

`oracle_lon_straight` — a straight plan walked to the **true** logged distance — has `ego_progress`
spanning **0.9781 → 1.0000**: `observed_range = 0.0219 < RANGE_MIN = 0.05` ⇒ **`"range below
range_min"` ⇒ INADMISSIBLE**, so the arm cannot be scored on the panel at all.
⚠️ **An arm that gets the axis right is refused by the very gate that exists to keep the axis
meaningful.** Its informative numbers are its raw axes (§4): along `|err|` **0.331 m**, the best in the
study, on a 25 m displacement.

### 5.4 ⛔ The per-arm gate would have manufactured a CONFIRM

| arm | `comfort` mean |
|---|---:|
| `v1_tactical_follow` | **0.0004** |
| ⭐ `v1_ego_v0` | **0.2882 — 720× higher** |
| `v1_ego_oracle_lon` | 0.2453 |
| `v1_lat_straight` | 0.0099 |
| `cv_holdv0` | 1.0 (saturated ⇒ inadmissible) · `v4_oracle` 0.0 (floored ⇒ inadmissible) |

Re-timing a plan at constant speed **smooths it** — the jerk that trips the comfort bound is an
artefact of independent per-horizon waypoint regression, and constant-speed resampling removes it.
Under the per-arm gate that 720× comfort jump enters the composite for both arms and turns
**+0.0078 n.s.** into **+0.0836 SEP**. ⛔ **A verdict flip, on my own primary result, driven by a
smoothing artefact and not by planning.** The panel-wide gate is used throughout and this is the
sharpest available demonstration of why it must be.

---

## 6. Block A — ablating a TRAINED speed input inside one checkpoint

*(REF-C is the only panel family whose planner already consumes `v0`; see §1.3. Status and results in
`artifacts/blockA_*`; see §9 for anything still running at hand-off.)*

**BLOCK_A_PLACEHOLDER**

---

## 7. Everything that is wrong with this study, stated by me

| # | limitation | status |
|:--:|---|---|
| 1 | ⛔ **Block B's arms are PLAN TRANSFORMS, not trained planners.** They bound what an ego input could buy v1's *existing* curve; a *trained* ego input could also change the curve. **The `REFUTE` is therefore about the longitudinal schedule, which is what the mechanism claim was about — but it is not proof that no trained ego-conditioned tactical head can beat CV.** | ⚠️ **the single most important caveat** |
| 2 | **No trained ego-on flagship arm exists in this study**, for the host/quality reasons in §2. | disclosed, §2 |
| 3 | ⚠️ **Two arms are ORACLES** (`v1_ego_oracle_lon`, `oracle_lon_straight`) — they read the ego's own future. Upper bounds, never deployable, stamped on every node. | disclosed |
| 4 | ⚠️ **`v1_ego_v0` extrapolates past the end of v1's own path on 37.34 % of rows** (v1's emitted path is often shorter than `v0·2 s`); the oracle arm on 40.65 %. Extrapolation is a straight continuation along the terminal tangent, so those rows carry **less** of v1's shape than the others. Reported per arm in `blockB_build.json`. | disclosed, quantified |
| 5 | ⚠️ **`v1_ego_v0`'s `recovery` defined-fraction is 0.8196 vs the control's 0.8423 — a 2.27 pp drop, just over my pre-registered 2 pp guard.** Its `recovery` (and hence part of its composite) is therefore **not a strict like-for-like** and I do not quote the recovery move as a clean loss. The `ego_progress` half, which carries the result, is unaffected (identical NaN mask, 15,442 rows both sides). | ⛔ **guard tripped, disclosed** |
| 6 | **`comfort` is dropped, not fixed** — inherited from the panel, deliberately not retuned. §5.4 shows the cost of the alternative. | inherited, deliberate |
| 7 | **No collision, no TTC** — the val cache has no cuboids. ⇒ `PSS` **is not a Driving Score**. | inherited blocker |
| 8 | **2 s horizon, non-reactive log replay, lateral axis refused** — all inherited from the panel and all bounding. The 2 s horizon in particular is why CV is a strong baseline by construction. | inherited |
| 9 | **`v1_ego_double`'s composite contrast is inadmissible** (§5.2), so my degradation guard is only a partial pass in the over-travel direction. | ⛔ disclosed |
| 10 | **I did not test a strategic-level ego input**, though §1 shows the gap is identical there. Out of budget. | not done, flagged |
| 11 | **The `ego_plan` graft is tested but never trained.** `attach_ego_input` is validated for identity-at-init, for state-dict compatibility with a `--v2` run, and for responding once non-zero — but no arm in this report was produced by it. | ⚠️ **UNVERIFIED as a training recipe** |

---

## 8. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **E1** | ⛔⛔ **LIVE BUG: every eval harness drops `ego=`.** `flagship-v2corpus-30k` is training on pod1 **with `v2_ego_to_planners = true`** and will be scored **ego-blind** by `closedloop.py`, `planner_p2.py`, `planning.py`, `corpus_overlay.py`, `blindimag.py` and `panel_run.py` — silently, with no error. **Its ego weights will simply not be exercised, and the arm will be mis-attributed as "the ego lever does nothing".** The fix is one keyword argument per call site plus an assertion that refuses to evaluate a policy with `ego_emb is not None` and `ego is None`. **This will not happen by itself.** | **`taniteval` maintainer / Benchmarks & Eval — BEFORE the v2corpus gate** |
| **E2** | ⛔⛔ **`PSS_recovery_progress` MUST NOT BE v5's SOLE PRIMARY.** It returned **n.s.** on a **5.65×** along-track RMS improvement (§4.1). Two measured causes: `ego_progress` is **one-sided** (a 2× over-travelling plan scores **higher** than a perfect one, +0.0785 SEP) and **56 % of the control's rows are already at its ceiling**; and an arm that solves the axis is **refused** by the range gate (§5.3). ⇒ **A two-sided progress term** (e.g. `1 − |1 − ratio|`) **and a longitudinal-accuracy component are needed before this surface can adjudicate a longitudinal lever.** ⚠️ This also bounds the panel's own headline: *"nothing beats holding v₀"* was measured on a composite that cannot resolve longitudinal accuracy. | **`taniteval` maintainer / PI** |
| **E3** | ⭐ **v5's tactical brain SHOULD take an ego input — but the decision-grade reason is the axis, not the composite.** The input fixes 76.9 % → 8.9 % of squared endpoint error at zero deployment cost, and the seam already exists (`v2_ego_to_planners`, plus `tanitad.ego_plan.attach_ego_input` for grafting onto a trained brain). ⛔ **But it will not, by itself, beat constant velocity** — measured, −0.0100 SEP below. | **Architecture & Inference / v5** |
| **E4** | ⭐⭐ **THE LEVER IS THE 2 s FUTURE DISPLACEMENT, NOT `v0`.** `v0` closes **89.3 %** of the achievable longitudinal error but only **19.2 %** of the composite's oracle move; the oracle **beats CV, separated (+0.0228)**. This is the **same axis** `V5_PLAN` §8's E-GOAL stream reached independently (+83.7 % along-track vs +2.9 % cross-track). ⇒ **Fund a longitudinal-displacement predictor** (target: beat `v0·t`), **not** a lateral/route one. ⚠️ E-GOAL also measured that a *realisable* head missed its break-even (σ₀ = 0.955 m, achieved 1.330 m) — so this is a hard target with a known bar, not a free win. | **PI / Architecture — v5 scoping** |
| **E5** | ⛔ **v1's TACTICAL STEERING IS WORTH NOTHING (+0.0006 n.s.) AND IS HARMFUL ONCE TIMED (+0.0100 SEP against it).** A straight line matches the flagship's tactical head laterally. Before v5 spends capacity on a lateral/multimodal tactical decoder, this needs an answer. | **PI / Architecture** |
| **E6** | ⚠️ **`FiLM.to_scale_shift` is zero-init** (`predictor.py:25-26`), so on a from-scratch brain the whole `ctx`/ego path has **no gradient at init**. Grafting onto a *trained* brain avoids this; training a new 4-brain with an ego input should not assume the seam is live at step 0. | **Architecture & Inference** |
| **E7** | ⚠️ **`MODEL_REGISTRY` should stamp that `ego_input_on_planners = False` on v1, the no-speed control and v4.** The panel measured it; the registry does not carry it, and "the speed fix" is routinely quoted without noting that it never reached the planner. | **Model-registry agent** |

---

## 9. Deliverable manifest

Repo dir: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-28-tactical-action-input/`
Everything `git add`-ed into the working tree. ⛔ **NOT committed, NOT pushed.** ⚠️ marks anything in only ONE place.

| artifact | where it lives | what it is |
|---|---|---|
| `TACTICAL_ACTION_INPUT.md` | repo (this dir) | this report |
| `PRE_REGISTRATION.md` | repo | frozen before any new number existed; carries the verified code reading |
| `scripts/make_retimed_arms.py` | repo | Block B builder — the factorisation, the two self-validating identities, the probes |
| `scripts/decompose.py` | repo | lateral/longitudinal decomposition (imports `pseudosim._cross_and_along`) |
| `scripts/refc_v0_ablate.py` | repo · `tanitad-eval:/workspace/_egoin/scripts/` | Block A — the 3-line `v0` ablation over the published `panel_run` adapters |
| `artifacts/blockB_panel.json` | repo · scratchpad | the full 16-arm panel: gate, composites, **105 paired blocks**, per-arm-gate sensitivity |
| `artifacts/blockB_build.json` | repo | build report incl. the two self-validation identities and per-arm extrapolation fractions |
| `artifacts/decomposition.json` | repo | per-arm along/cross endpoint error + longitudinal share |
| `artifacts/pw/pw_*.npz` × 6 | repo | ⭐ **per-window dumps for every derived arm** — every number here recomputes from these with **no GPU, no checkpoint, no corpus** |
| `artifacts/blockB_combine.log` | repo | the combine run log |
| `stack/tanitad/ego_plan.py` | repo | the geometry + `attach_ego_input` (the v5 seam), **a NEW module — no existing file's bytes changed** |
| `stack/tests/test_ego_plan.py` | repo | **31 tests**, incl. the two identity tests, both-direction graft validation, and the two premise-pinning tests |
| Block A artifacts | see §6 | — |

**Suites green, both, after the change:** `stack/` **1379 passed, 12 skipped** (2:05) and `taniteval/`
**565 passed** (1:16). ⚠️ Both are *above* the counts in the brief (1324 / 559) — 31 of the extra 55 in
`stack/` are mine; the remainder are other agents' work already in the tree. ⛔ **No existing file's
bytes were changed**: `ego_plan.py` and `test_ego_plan.py` are new files, and `fourbrain.py`,
`config.py`, `predictor.py` and `train_flagship4b.py` are untouched — pod1's running arm is unaffected.

**Nothing that took real effort exists in only one place.** No checkpoint was modified; all were read.
🔒 No clip UUID or raw PhysicalAI content appears in any artifact. **Parity untouched** — no episode was
re-selected; every arm uses the identical 15,981 rows and row identity is *asserted*, not assumed.

⭐ **The no-GPU recompute path is VERIFIED for this study too:**

```
CUDA_VISIBLE_DEVICES="" python3 <panel>/scripts/panel_combine.py \
    --in-dir <the 10 published pw_*.npz + artifacts/pw/*.npz> \
    --out repro.json --reference v1_tactical_follow \
    --adversary stand_still,v1_ego_half,v1_ego_double,oracle_lon_straight
```

**BLOCK_A_MANIFEST_PLACEHOLDER**

---

## 10. Self-refutations, and what was deliberately NOT done

| # | what | status |
|:--:|---|---|
| 1 | ⛔ **I nearly reported the brief's premise as verified.** `TacticalPolicy.forward` **does** have an `ego` parameter. Reading the file rather than the brief changed the claim from "no input" to "a disabled input **plus** a harness that drops it" — and the second half (E1) is a live bug the brief, the panel and `V5_PLAN` all missed. | corrected, §1 |
| 2 | ⛔ **My first combine let a probe into the panel gate and it deleted `ego_progress` for all 16 arms.** The published composites changed. Caught because the published arms no longer reproduced their published values — which is exactly what the reproduction gate is for. Fixed by treating all four probes as validation probes, per the panel's own `stand_still` rule. | corrected, §3.1 |
| 3 | ⛔ **My first Block-A launch failed all four arms in 1 s each** — `panel_run` inserts stale pod paths at `sys.path[0]` and shadowed the shipped `taniteval`. The runner also reported `rc=0` for every failure because `$?` captured an `echo`. Both fixed; the driver now **asserts** the packages resolved to the shipped copy. | corrected, §6 |
| 4 | ⚠️ **Two of my own tests failed first and both were right to.** The half-speed control is not *exactly* half on a curved path (chord vs arc), and a freshly built policy ignores `ctx` entirely (§1.4). I changed the tests to state the true facts rather than loosening them into vacuity. | corrected |
| 5 | ⚠️ **My `recovery` defined-fraction guard tripped** (−2.27 pp vs the 2 pp bound) and I report the arm as not strictly like-for-like on that component instead of quietly quoting it. | disclosed, §7.5 |
| 6 | ⛔ **The per-arm gate flips my primary verdict to CONFIRM.** I could have quoted it. It is a comfort/smoothing artefact and I use the panel-wide gate. Both are published. | disclosed, §5.4 |
| 7 | **I did not train anything.** No ego-conditioned tactical head was fitted, so limitation §7.1 is real and the `REFUTE` is scoped to the longitudinal *schedule*, which is what the mechanism claim was about. | deliberate, scoped |
| 8 | **I did not touch pod1 or pod2**, so `flagship-v2corpus-30k` — the one arm that would answer §7.1 directly — is absent. | by rule |
| 9 | ⛔ **I did not hold anything to v1's 0.4271.** Verified: `taniteval/rollout.py:170` sets `actions_source="expert_future"`; it is `wm_fidelity_ade_2s` and is not a planning bar. | correct by construction |
