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
| **1** | ⛔ **THE BRIEF'S PREMISE IS PARTLY FALSE, AND THE TRUE VERSION IS WORSE.** `TacticalPolicy.forward(states, ctx, ego=None)` **does** take an ego vector (`fourbrain.py:328`). It is inert for **two independent reasons**: the `v2_ego_to_planners` lever is off by default (`config.py:196`, and `ego_input_on_planners = False` on every panel arm), **and — not named in the brief, the panel or `V5_PLAN` — `ego=` is passed at exactly THREE call sites in the whole repo, all three in the TRAINER. Not one evaluation path passes it.** An ego-**trained** checkpoint is evaluated ego-**blind**, silently, with no error. | `MEASURED` **tier 1** |
| **2** | ⛔⛔ **PRE-REGISTERED VERDICT: `REFUTE`.** Handing v1's tactical plan the speed it never saw moves the pseudo-simulation composite **+0.0078 [−0.0110, +0.0260] — NOT SEPARATED** from its own no-input control. It also stays **separated BELOW `cv_holdv0`** (−0.0100 [−0.0170, −0.0033] SEP). **The missing action input is not the mechanism, and I do not re-scope.** | `MEASURED` **tier 2** |
| **3** | ⭐⭐ **AND THE REASON IS THE FINDING: THE MECHANISM IS REAL AND HUGE, AND THE COMPOSITE CANNOT SEE IT.** The same intervention cuts v1's along-track endpoint RMS from **8.799 m → 1.557 m (5.65×)**, its longitudinal share of squared error from **76.9 % → 8.9 %**, and the fraction of windows where the plan travels the right distance (±5 %) from **15.00 % → 58.93 % (3.93×)**. **All of that returns `n.s.` on `PSS_recovery_progress`.** | `MEASURED` **tier 1** (arithmetic) |
| **3b** | ⛔⛔ **AND THE EXACT REASON, IN ONE NUMBER: v1 OVER-TRAVELS ON 48.80 % OF WINDOWS (p95 ratio 2.430×) AND `ego_progress` CLAMPS TO 1.0, SO IT CHARGES NOTHING FOR ANY OF IT.** Most of what a speed input fixes is worth **exactly zero** to the composite by construction; only the 36.21 % under-travelling rows can pay. | `MEASURED` **tier 1** |
| **3c** | ⛔⛔ **AND BLOCK A PROVES IT IS BLINDNESS, NOT LOW POWER.** Ablating REF-C-XL's **trained** `v0` inside one checkpoint degrades along-track \|err\| **1.199 → 4.028 m (×3.36)** and the composite **separates: −0.0332 [−0.0433, −0.0243] SEP, under BOTH gates** (**replicated at base scale: 1.276 → 4.780 m, ×3.75, composite +0.0461 [+0.0354, +0.0579] SEP**). ⇒ **The same estimator on the same rows separates a 3.36× DEGRADATION and cannot separate a 5.41× IMPROVEMENT** *(both ×'s on mean along-track \|err\|; the improvement is 5.65× on RMS)* — because v1's error is 48.80 % *over*-travel, which the clamp does not charge for. **`PSS` punishes going too slow and is blind to going too fast.** | `MEASURED` **tier 2** |
| **4** | ⭐⭐ **A PERFECT LONGITUDINAL INPUT *DOES* BEAT CONSTANT VELOCITY — separated.** `v1_ego_oracle_lon` (v1's own curve, walked to the **true** logged distance) scores **0.5946 [0.5868, 0.6033]** vs `cv_holdv0` **0.5705**: paired **+0.0228 [+0.0064, +0.0412] SEP**, and **+0.0407 [+0.0308, +0.0512] SEP** over its own control. ⚠️ **ORACLE — an upper bound, not deployable.** *To my knowledge the first thing in this program to clear CV on this surface, and it needs the future to do it.* | `MEASURED` **tier 2**, ⚠️ oracle |
| **5** | ⭐⭐⭐ **THE REALISABLE/ORACLE GAP LOCATES THE ACTUAL MISSING QUANTITY, AND IT IS NOT `v0`.** `v0·t` recovers **89.3 %** of the achievable along-track RMS reduction but only **19.2 %** of the composite's oracle move. **The missing input is the 2 s FUTURE DISPLACEMENT, which `v0` does not determine** — independently converging with `V5_PLAN` §8's E-GOAL result (*"the 88 % lives on how far the car will travel in 2 s"*). | `MEASURED` **tier 2** |
| **6** | ⛔ **v1's TACTICAL STEERING IS WORTH NOTHING, AND IS HARMFUL ONCE THE TIMING IS FIXED.** Replacing v1's entire curve with a straight line on its own schedule: **+0.0006 [−0.0065, +0.0072] n.s.** (the tightest contrast in this study). On the `v0·t` schedule the straight line is **separated BETTER**: **+0.0100 [+0.0033, +0.0170] SEP**. Cross-track: v1 **3.604 m** vs a straight line **3.448 m**. | `MEASURED` **tier 2** |
| **7** | ⛔ **THE NAMED GATE TRAP FIRED ON MY OWN PRIMARY RESULT.** Under the shipped **per-arm** gate the same contrast reads **+0.0836 SEPARATED** — a **verdict flip**. Cause, measured: re-timing at constant speed *smooths* the plan, so `comfort` jumps **0.0004 → 0.2882 (720×)** and enters the composite for both arms. **The per-arm gate would have manufactured a CONFIRM out of a smoothing artefact.** I use the panel-wide gate throughout and report both. | `MEASURED` **tier 1** |
| **8** | ⚠️ **`ego_progress` IS ONE-SIDED AND ALREADY SATURATED, WHICH IS WHY (3) HAPPENS.** It is `clamp(along/human, 0, 1)`: **55.91 %** of v1's rows are *already* at the ceiling, so there is no room to pay for an improvement; and a plan travelling **1.97×** as far scores `ego_progress` **+0.0785 [+0.0565, +0.1006] SEP HIGHER** than the control. ⛔ **The component cannot punish over-travel at all.** | `MEASURED` **tier 1** |
| **9** | ⚠️ **THE METRIC REFUSES A SOLVED PROBLEM.** A straight plan walked to the true distance (`oracle_lon_straight`) has `ego_progress` spanning **0.9781–1.0000**, `observed_range = 0.0219 < RANGE_MIN = 0.05` ⇒ **`range below range_min` ⇒ INADMISSIBLE**. An arm that gets the axis right is refused by the gate that exists to keep the axis meaningful. | `MEASURED` **tier 1** |
| **10** | ✅ **THE DEGRADATION GUARD HELD — the composite did NOT rise for a degraded planner.** `v1_ego_half` **−0.2421 [−0.2565, −0.2285] SEP worse**. ⚠️ The over-travel probe is a **partial** pass: its composite falls only because `ego_progress` becomes inadmissible for it; on the admissible components it goes **UP**. Reported, not hidden. | `MEASURED` **tier 1** |
| **11** | ⭐ **A THIRD GAP, FOUND WHILE PINNING THE SECOND: `FiLM.to_scale_shift` IS ZERO-INITIALISED** (`predictor.py:25-26`). On a from-scratch policy the **entire `cond` path — `ctx` and any ego graft — has zero effect and zero gradient at init.** A v5 ego graft on a fresh brain must account for this; on a *trained* brain it does not apply. | `MEASURED` **tier 1** |

### 0.1 The verdict in one sentence

**The tactical brain really is speed-blind, the blindness really does cost 5.65× in along-track
accuracy, and fixing it with the ego speed really does not move the closed-loop composite — because
that composite punishes travelling too slow and is blind to travelling too fast (proven: it separates
a 3.36× degradation of the same axis but not a 5.41× improvement), and because the quantity that
*would* move it is not the current speed but the 2 s future displacement, which `v0` does not
determine.**
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

**Every** eval call site invokes the policy positionally with two arguments. Even a checkpoint trained
with `--v2` — which *has* `ego_emb` weights — is evaluated **ego-blind**, silently, with no error and
no log line. 8 files, 21 policy calls, zero `ego=`:

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

⭐ **The sharpest form of this, and it is a whole-repo grep, not a sample:** `ego=` is passed at
**exactly three call sites in the entire repository** — `flagship_losses.py:245, 246, 351` — **all
three in the TRAINER.** `fourbrain.run_hierarchy:386-387` forwards whatever its caller gives it, and
its only in-repo caller (`refs/refa.py:260`) gives it nothing. ⇒ **The training path feeds the ego
vector. Not one evaluation path in the repo does.**

⇒ ⛔ **`flagship-v2corpus-30k`, now training on pod1 with `v2_ego_to_planners = true`, will be
evaluated ego-blind by every harness in the repo unless this is fixed** — its trained `ego_emb`
weights will simply not be exercised, and the arm will be mis-attributed as *"the ego lever does
nothing"*. That is escalation **E1**, the single most actionable item in this report. The guard it
needs now exists (`tanitad.ego_plan.assert_ego_is_fed`, 6 tests); **it still has to be called.**
Pinned by `test_passing_ego_to_a_stock_policy_is_SILENTLY_IGNORED`.

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
⚠️ I discovered this the hard way, **twice**: letting `v1_ego_double` into the gate dropped
`ego_progress` for **every** arm (it is ceiling-saturated at 97.49 %), and so, later and for the
opposite reason, did `oracle_lon_straight` (its range is *too small*, §5.3). Both times the published
arms stopped reproducing their published values, which is exactly what the reproduction gate is for.
Recorded in §10.

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

### 4.4 ⭐⭐ The panel splits perfectly by "does this planner have ANY longitudinal input?"

Nothing in this section was designed; it falls out of the decomposition, and it is the cleanest
pattern in the study.

| arm | longitudinal input to the **planner** | along-track **\|err\|** |
|---|---|---:|
| `nospeed_tactical_oracle` | ⛔ none | **6.611 m** |
| `v1_tactical_follow` | ⛔ none | **6.246 m** |
| `v1_lat_straight` | ⛔ none | **6.240 m** |
| `v4_oracle` | ✅ **`vt_band`** — a speed band, in its 3-field oracle goal | **1.509 m** |
| `refc_xl_produced` | ✅ **`v0`**, `refc.py:786-787` | **1.199 m** |
| `v1_ego_v0` *(this study)* | ✅ `v0·t` | **1.154 m** |
| `cv_holdv0` | ✅ `v0` (it is nothing but `v0`) | **1.067 m** |

⛔ **The three arms with no longitudinal input sit at 6.2–6.6 m. Every arm with one sits at
1.07–1.51 m. The gap is 4.1–6.2× and there is no overlap.** Membership of the two groups is decided
purely by a code fact (§1), never by architecture, scale or training budget.

⭐ **And this independently reproduces the panel's §5.4 mechanism on the AXIS rather than the
composite.** Matched pair, both on oracle nav: `nospeed` **6.611 m** vs `v1_tactical_oracle`
**6.088 m** — the `--speed-input` ablation, the program's largest measured effect at **6.5×** on the
operative rollout surface, is worth **1.09×** on the tactical planner's own output. **The speed fix
improved the world model; it barely reached the plan.** That is exactly what §1 predicts from the code,
now measured on the quantity in dispute.

---

## 5. ⚠️ WHY THE COMPOSITE COULD NOT SEE A 5.65× FIX — three measured instrument facts

### 5.1 `ego_progress` is one-sided and already saturated

`ego_progress = clamp(along_end / human_dist, 0, 1)` (`pseudosim.py:496-501`); `CEIL_FRAC_MAX = 0.95`
and `RANGE_MIN = 0.05` at `:161-162`.

| arm | `ego_progress` mean | **frac. of rows already at the ceiling** |
|---|---:|---:|
| **`v1_tactical_follow`** | 0.9081 | ⛔ **55.91 %** |
| `cv_holdv0` | 0.9407 | 26.90 % |
| `v1_ego_v0` | 0.9324 | 24.49 % |
| `v1_ego_oracle_lon` | 0.9805 | 17.77 % |
| ⛔ `v1_ego_double` | **0.9866 — the highest in the panel** | 97.49 % |

⛔ **Two defects at once.** (a) **55.91 % of the control's rows are already scored 1.0**, so the metric
cannot pay for an improvement on them — an improvement can only be banked on the other 44 %.
(b) **The clamp is one-sided: over-travel is never punished.** A plan travelling **1.97×** as far as it
should (**50.066 m** against a **25.462 m** logged displacement — a ~90 km/h plan where the car did 46)
scores `ego_progress` **+0.0785 [+0.0565, +0.1006] SEP HIGHER** than the control, and **higher than the
perfectly-correct arm** (0.9866 vs `oracle_lon_straight`'s 0.9916 is within noise, and both sit above
`v1_ego_oracle_lon`'s 0.9805). ⇒ **`ego_progress` cannot distinguish "exactly right" from "twice too
far".**

#### ⭐⭐⭐ 5.1a The distribution that explains the whole result

`ego_progress` clamps `ratio = plan_along / human_along` to `[0, 1]`. So look at the **unclamped**
ratio — the quantity the metric throws away (`artifacts/ratio_distribution.json`, recomputed from the
committed dumps, no GPU):

| arm | ratio **< 0.95** (under-travels) | **within ±5 %** | ratio **> 1.05** (over-travels) | median | p05 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **`v1_tactical_follow`** | 36.21 % | ⛔ **15.00 %** | ⛔ **48.80 %** | 1.041 | 0.565 | ⛔ **2.430** |
| ⭐ **`v1_ego_v0`** | 28.36 % | ⭐ **58.93 %** | 12.71 % | 0.979 | 0.686 | 1.204 |
| `cv_holdv0` | 24.00 % | 62.76 % | 13.24 % | 0.983 | 0.719 | 1.217 |
| ⚠️ `v1_ego_oracle_lon` | 7.02 % | **92.95 %** | 0.03 % | 0.990 | 0.932 | 1.002 |
| ⚠️ `oracle_lon_straight` | 0.00 % | **99.75 %** | 0.25 % | 0.993 | 0.978 | 1.012 |
| ⛔ `v1_ego_double` | 2.38 % | 0.40 % | 97.22 % | 1.956 | 1.326 | 2.394 |

⛔⛔ **v1's tactical head gets the 2 s distance right (±5 %) on 15.00 % of windows and OVER-TRAVELS on
48.80 %, with a p95 of 2.43× — it routinely plans to go nearly two and a half times as far as the car
actually goes.** Handing it `v0·t` lifts the ±5 % hit-rate **15.00 % → 58.93 %, a 3.93× improvement**
— still just short of `cv_holdv0`'s **62.76 %**, which is consistent with `v1_ego_v0` sitting
separated-below CV, and far short of the perfect schedule's **99.75 %**.

⭐⭐⭐ **And here is the whole result in one line: `ego_progress` charges NOTHING for the 48.80 % of
windows where v1 over-travels, because they clamp to 1.0.** Fixing them — which is most of what the
speed input does — is worth **exactly zero** to the composite. Only the under-travelling rows can pay,
and they move just 36.21 % → 28.36 %. **A 3.93× improvement in distance accuracy is nearly invisible
by construction.**

⚠️ **Does this inflate my own `v1_ego_v0` result?** ⛔ **No — it DEFLATES it, and the table above is
the proof.** I checked in the direction that could have embarrassed me: the worry was that
`v1_ego_v0` might be harvesting the one-sided clamp by simply travelling further. It travels
**less** (**25.082 m** vs the control's **26.650 m** mean ego-frame endpoint `x`, against a
**25.462 m** logged human displacement), and **48.80 % of what it fixes — the over-travel — is worth
zero to `ego_progress` by construction.** ⇒ **The one-sidedness makes my `REFUTE` conservative: the
true effect on distance accuracy is larger than the composite reports, not smaller.**
*(The over-travel probe for comparison: `v1_ego_double` ends at **50.066 m** against the same 25.462 m
ground truth — a **1.97×** plan, 97.22 % of rows over-travelling — and is the panel's top
`ego_progress`.)*

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

REF-C is the only panel family whose planner already consumes `v0` (§1.3), so the question *"does a
trained speed input reach a planner's output at all?"* can be answered **causally, inside one
checkpoint, with no training**: same weights, same 15,981 rows, `v0` real vs `v0 = 0`.

⭐ **`v0 = 0` is IN-DISTRIBUTION, not OOD** — `refc_ego_dropout_at_train = 0.5`, **read from the
loaded checkpoint config at run time**, not from the source default. Half of REF-C's training steps
saw exactly this input. ⚠️ The same fact **bounds** the block: REF-C was deliberately trained *not*
to lean on `v0`, so a small effect here is partly by design and must not be generalised to
"speed inputs do nothing".

**Provenance, arm 1 (MEASURED, `artifacts/blockA/arm_refc_xl_v0on.json`):** ckpt md5
`966d4eff1ea5ddf86efba01b8344e198` — ✅ **identical to the panel's `refc_xl_produced` and to
`MODEL_REGISTRY` §4.1** — step 29,999, 256 anchors, 2 denoise steps, `nav_mode = produced`,
15,981 rows, `densify_endpoint_max_err = 0.0`, **G4 falsifier PASS**, envelope verdict
*"MEASUREMENT — every step stayed inside the MEASURED envelope"*, 846.6 s on `tanitad-eval`.

### 6.1 ⭐⭐ The port fidelity gate — PASSED, at millimetre level, across hosts

`refc_xl_v0on` is the **shipped** arm re-run on a different host. Against the published
`pw_refc_xl_produced.npz` (produced on pod2, different GPU, different torch):

| quantity | result |
|---|---|
| rows | **15,981 both**, and `ep_i` / `anchor` / `pt_dlat` / `pt_dyaw` / `pt_dlon` / `v0` / `ref_yaw` **bit-identical** |
| trajectory | **max \|diff\| = 2.177 × 10⁻³ m**, mean 1.92 × 10⁻⁵ m |
| 2 s endpoint (what the metric reads) | **max \|diff\| = 2.177 × 10⁻³ m**, mean 4.88 × 10⁻⁵ m |

⇒ **A 2.2 mm worst-case disagreement on a ~25 m plan, across two hosts and two torch versions**
(pod2 vs `tanitad-eval` torch 2.8.0+cu128) — float non-determinism in the anchored-diffusion decode,
not a difference in the arm. **The Block-A port is admissible.**

### 6.2 ⭐⭐⭐ The result — the SAME mechanism, in the OPPOSITE direction, on a TRAINED input

Block B *added* a speed schedule to a planner that never had one. Block A *removes* a **trained**
speed input from a planner that does. If the mechanism in §1 is real, the two must move the same axis
by a comparable amount — and they do:

| | along-track **\|err\|** | along RMS | **longitudinal share of sq. err** | cross **\|err\|** |
|---|---:|---:|---:|---:|
| `refc_xl_v0on` (shipped, has `v0`) | **1.199 m** | 1.654 | **6.7 %** | 4.264 |
| `refc_xl_v0off` (identical weights, `v0 = 0`) | **4.028 m** | 6.279 | ⛔ **57.2 %** | 3.861 |
| **effect of removing the trained speed input** | ⛔ **×3.36 worse** | ×3.80 | **6.7 % → 57.2 %** | *slightly better* |

⭐ **Removing a trained ego-speed input degrades along-track accuracy 3.36× and turns a
lateral-dominated error (6.7 % longitudinal) into a longitudinal-dominated one (57.2 %).** Block B's
mirror-image number was 6.246 m → 1.154 m (**5.41×**) for *adding* one. **Two independent directions,
one axis, the same magnitude class — measured inside single checkpoints, with nothing else changed.**

⇒ ⭐ **This upgrades §4.4 from a pattern to a CAUSAL statement.** The split between the 6.2–6.6 m
group and the 1.07–1.51 m group is *caused* by the presence of a longitudinal input, not merely
correlated with it: take the input away from a member of the good group and it moves **toward** the
bad group (4.028 m). ⚠️ It does not reach it (6.2 m), which is itself informative — REF-C trains with
`ego_dropout = 0.5`, so it has partially learned to read speed from the image.

The plan really does change: mean 2 s endpoint displacement between the two arms is **1.903 m**, and
mean endpoint `x` drops **25.399 m → 23.998 m**. ⚠️ The fed route command histogram is **identical**
(`follow` 14,414 · `left` 936 · `right` 631), so `resolve_nav` is unaffected and the contrast isolates
the decoder's speed port — **not** a route-selection side effect.

### 6.3 ⭐⭐ On the composite — A-CONFIRM fires, and the fidelity gate is EXACT

Same 15,981 rows, panel-wide gate, paired episode-cluster bootstrap B = 2000
(`artifacts/blockA/blockA_full_panel_20arm.json`):

| contrast | Δ PSS (panel gate) | Δ `ego_progress` | Δ `recovery` | per-arm gate |
|---|---|---|---|---|
| ✅ **fidelity** `refc_xl_v0on` − `refc_xl_produced` | **−0.0000 [−0.0000, +0.0000] n.s.** | −0.0000 n.s. | −0.0000 n.s. | −0.0000 n.s. |
| ⭐ **`refc_xl_v0on` − `refc_xl_v0off`** | **+0.0332 [+0.0243, +0.0433] ⭐ SEP** | **+0.0706 [+0.0543, +0.0884] SEP** | −0.0108 [−0.0201, −0.0034] SEP | +0.0263 **SEP** |
| `refc_xl_v0off` − `cv_holdv0` | −0.0535 [−0.0676, −0.0383] SEP | −0.0676 SEP | −0.0420 SEP | −0.1541 SEP |
| `refc_xl_v0off` − `v1_tactical_follow` | −0.0355 [−0.0483, −0.0235] SEP | −0.0349 SEP | −0.0403 SEP | −0.0290 SEP |

* ✅ **The fidelity precondition is met beyond what was asked**: the re-run arm scores
  **0.5499 [0.5421, 0.5566]** — *identical to the published `refc_xl_produced` to four decimals* — and
  the paired delta is **−0.0000** on every component. Block A is **not** quarantined.
* ⭐ **A-CONFIRM fires: a trained speed input measurably reaches a planner's output**
  (+0.0332 SEP), **and it survives BOTH gate choices** (+0.0263 SEP per-arm) — unlike Block B's
  primary, which flips. `recovery` defined-fraction moves only **0.21 pp** (0.8256 → 0.8235), well
  inside my 2 pp guard, so this is a clean like-for-like.
* ⚠️ **And it still does not rescue REF-C**: even with its speed input, `refc_xl_v0on` is
  **separated BELOW `cv_holdv0`** (−0.0203). Removing it makes things much worse (−0.0535). **A
  speed input is necessary-ish and nowhere near sufficient.**

### 6.4 ⛔⛔ THE ASYMMETRY — this is the PROOF of §5.1a, not an inference from it

Put the two blocks side by side. Same instrument, same estimator, same 15,981 rows, same axis:

| block | intervention | along-track **\|err\|** | **Δ PSS** |
|---|---|---|---|
| **A** | ⛔ *remove* REF-C's **trained** speed input | 1.199 → 4.028 m — **×3.36 WORSE** | **−0.0332 [−0.0433, −0.0243] ⭐ SEPARATED** |
| **B** | ⭐ *add* a speed schedule to v1's plan | 6.246 → 1.154 m — **×5.41 BETTER** | **+0.0078 [−0.0110, +0.0260] n.s.** |

⛔⛔ **THE COMPOSITE SEPARATES A 3.36× DEGRADATION AND CANNOT SEPARATE A 5.41× IMPROVEMENT** — both
factors on the *same* statistic (mean along-track `|err|`), so they are directly comparable.

⭐ **This kills the obvious objection to §5.1a — that the null is just low power.** It is not: the
identical estimator on the identical rows returns a **separated** result for a *smaller* effect on the
*same axis*, in the other direction. **The surface has the power; it lacks the sensitivity in the
improving direction.**

⇒ **The mechanism is exactly §5.1a.** REF-C-`v0on` already sits in the "right distance" band, so
removing its speed input pushes rows into **under**-travel, where `ego_progress` charges in full.
v1 starts with **48.80 %** of rows **over**-travelling, where the clamp charges **nothing** — so
repairing them earns nothing. ⇒ **`PSS_recovery_progress` is an asymmetric detector of longitudinal
error: it punishes going too slow and is blind to going too fast.** That is escalation **E2**, now
demonstrated rather than argued.

### 6.5 ⭐⭐ The base-scale replication — IT REPLICATES

All four arms completed clean (`BLOCK_A_ALL_DONE fail=0`; `artifacts/blockA/egoin_blockA.log`).

| | along-track **\|err\|** | along RMS | **longitudinal share of sq. err** |
|---|---:|---:|---:|
| `refc_base_v0on` (shipped, has `v0`) | **1.276 m** | 1.751 | **7.3 %** |
| `refc_base_v0off` (identical weights, `v0 = 0`) | **4.780 m** | 7.403 | ⛔ **67.1 %** |
| **effect of removing the trained speed input** | ⛔ **×3.75 worse** | ×4.23 | **7.3 % → 67.1 %** |
| *(XL, for comparison)* | *×3.36 worse* | *×3.80* | *6.7 % → 57.2 %* |

⭐⭐ **A-CONFIRM REPLICATES AT BOTH SCALES, BOTH SEPARATED** — on the axis (×3.36 XL / ×3.75 base) and
on the composite (**+0.0332 XL / +0.0461 base**, the base effect the *larger* of the two).
✅ And a second exact fidelity check falls out for free: `refc_base_v0on`'s along-track `|err|` is
**1.276 m**, *identical to the published `refc_base_produced`* in §4's table — the re-run reproduces
the shipped arm on this axis to the printed precision.

**On the composite** (panel-wide gate, final 20-arm recombination, 190 paired blocks):

| arm | PSS |
|---|---|
| `refc_base_v0on` | **0.5439 [0.5345, 0.5519]** — ✅ *identical to the published `refc_base_produced` 0.5439* |
| `refc_base_v0off` | **0.4980 [0.4838, 0.5106]** |
| ⭐ **`refc_base_v0on` − `refc_base_v0off`** | **+0.0461 [+0.0354, +0.0579] ⭐ SEPARATED**, per-arm gate **+0.0377 SEP** *(XL: +0.0332 [+0.0243, +0.0433] SEP, per-arm +0.0263 SEP)* |
| ✅ fidelity `refc_base_v0on` − `refc_base_produced` | **−0.0000 [−0.0000, +0.0000] n.s.** — exact, like XL |

⭐ **A second exact fidelity reproduction**: the base re-run scores **0.5439**, the published value to
four decimals on a different host, and the paired delta against the published arm is **−0.0000
[−0.0000, +0.0000]** — so **both** Block-A scales reproduce their shipped arms exactly before they are
ablated. ⭐ **A-CONFIRM therefore holds at two scales AND under both gate choices** (+0.0263 / +0.0377
per-arm), unlike Block B's primary, which flips.

⚠️ **The `v0off` arms do not fall all the way to the genuinely speed-blind group** (4.78 / 4.03 m vs
v1's 6.25 m and `nospeed`'s 6.61 m). That is expected and quantified: REF-C trains with
`ego_dropout = 0.5`, so it has partially learned to read speed from the image. **A model trained to
rely on the input would show a larger ablation effect, not a smaller one** — so this is a *lower*
bound on what the input is worth.

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
| 12 | ⚠️ **Block A's arm is REF-C, not the flagship.** It answers *"does a trained speed input reach a planner's output?"* (yes, +0.0332 SEP) — **not** *"would it reach the flagship's tactical head?"*. Different architecture, different training. | disclosed, §6 |
| 13 | ⚠️ **Block A's effect size is bounded from BELOW by design.** REF-C trains with `ego_dropout = 0.5` (verified from the loaded checkpoint, not the source default), i.e. it was deliberately trained not to lean on `v0`, and it has partially learned to read speed from the image — which is why `v0off` lands at 4.028 m rather than the 6.2–6.6 m of the genuinely speed-blind arms. **A model trained to USE the input would show a larger ablation effect, not a smaller one.** | disclosed, quantified |
| 14 | ⚠️ **`v0 = 0` is one particular ablation.** It is in-distribution here, but it is not the same as `ego=None` (which skips the bias entirely) and not the same as an untrained-without-speed model. The clean control for the latter would be a REF-C trained with no `v0` port at all, which does not exist. | disclosed |
| 11 | **The `ego_plan` graft is tested but never trained.** `attach_ego_input` is validated for identity-at-init, for state-dict compatibility with a `--v2` run, and for responding once non-zero — but no arm in this report was produced by it. | ⚠️ **UNVERIFIED as a training recipe** |

---

## 8. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **E1** | ⛔⛔ **LIVE BUG: every eval harness drops `ego=`.** `flagship-v2corpus-30k` is training on pod1 **with `v2_ego_to_planners = true`** and will be scored **ego-blind** by `closedloop.py`, `planner_p2.py`, `planning.py`, `corpus_overlay.py`, `blindimag.py` and `panel_run.py` — silently, with no error. **Its ego weights will simply not be exercised, and the arm will be mis-attributed as "the ego lever does nothing".** The fix is one keyword argument per call site plus an assertion that refuses to evaluate a policy with `ego_emb is not None` and `ego is None`. ⭐ **That assertion now EXISTS and is tested — `tanitad.ego_plan.assert_ego_is_fed` (6 tests, and it is a provable no-op for every arm in the published panel, so adding the call changes no published number). It still has to be CALLED.** **This will not happen by itself.** | **`taniteval` maintainer / Benchmarks & Eval — BEFORE the v2corpus gate** |
| **E2** | ⛔⛔ **`PSS_recovery_progress` MUST NOT BE v5's SOLE PRIMARY, AND THE FIX IS A ONE-LINE CHANGE.** It returned **n.s.** on a **5.65×** along-track RMS improvement and a **3.93×** improvement in distance hit-rate (§4.1, §5.1a). The measured cause is exact: **`ego_progress = clamp(ratio, 0, 1)` is ONE-SIDED, and v1 OVER-travels on 48.80 % of windows (p95 ratio 2.430×), so nearly half of what a longitudinal lever fixes is worth ZERO by construction.** ⇒ **Make the progress term two-sided — `1 − \|1 − ratio\|` clipped at 0** — which costs one line, keeps the published ranking's intent, and would let the surface see a longitudinal lever at all. ⭐ **This is now PROVEN, not argued (§6.4): the identical estimator on the identical rows SEPARATES a 3.36× degradation of the same axis (Block A, −0.0332 SEP under both gates) while returning n.s. on a 5.41× improvement (Block B). It is not low power — it is one-sided blindness.** Secondary: an arm that *solves* the axis is **refused** by the range gate (§5.3). ⚠️ **This bounds the panel's own headline**: *"nothing beats holding v₀"* was measured on a composite that cannot resolve longitudinal accuracy — and `cv_holdv0` is precisely the arm that gets the longitudinal axis right by construction. | **`taniteval` maintainer / PI** |
| **E3** | ⭐ **v5's tactical brain SHOULD take an ego input — but the decision-grade reason is the axis, not the composite.** ⭐ **Block A settles the "does it even reach the output?" question: on a TRAINED input, inside one checkpoint, ablating `v0` costs −0.0332 PSS at XL (SEPARATED UNDER BOTH GATES) and −0.0461 at base — replicated at two scales (§6.3, §6.5).** The input fixes 76.9 % → 8.9 % of squared endpoint error at zero deployment cost, and the seam already exists (`v2_ego_to_planners`, plus `tanitad.ego_plan.attach_ego_input` for grafting onto a trained brain). ⛔ **But it will not, by itself, beat constant velocity** — measured, −0.0100 SEP below. | **Architecture & Inference / v5** |
| **E4** | ⭐⭐ **THE LEVER IS THE 2 s FUTURE DISPLACEMENT, NOT `v0`.** `v0` closes **89.3 %** of the achievable longitudinal error but only **19.2 %** of the composite's oracle move; the oracle **beats CV, separated (+0.0228)**. This is the **same axis** `V5_PLAN` §8's E-GOAL stream reached independently (+83.7 % along-track vs +2.9 % cross-track). ⇒ **Fund a longitudinal-displacement predictor** (target: beat `v0·t`), **not** a lateral/route one. ⚠️ E-GOAL also measured that a *realisable* head missed its break-even (σ₀ = 0.955 m, achieved 1.330 m) — so this is a hard target with a known bar, not a free win. | **PI / Architecture — v5 scoping** |
| **E5** | ⛔ **v1's TACTICAL STEERING IS WORTH NOTHING (+0.0006 n.s.) AND IS HARMFUL ONCE TIMED (+0.0100 SEP against it).** A straight line matches the flagship's tactical head laterally. Before v5 spends capacity on a lateral/multimodal tactical decoder, this needs an answer. | **PI / Architecture** |
| **E6** | ⚠️ **`FiLM.to_scale_shift` is zero-init** (`predictor.py:25-26`), so on a from-scratch brain the whole `ctx`/ego path has **no gradient at init**. Grafting onto a *trained* brain avoids this; training a new 4-brain with an ego input should not assume the seam is live at step 0. | **Architecture & Inference** |
| **E8** | ⭐ **A NEW RETRACTION CLASS, offered for `RETRACTION_LOG.md` — I did not append it myself.** ⛔ **`TWO-CONDITION GATE AUDITED AT ONE CONDITION`.** `TacticalPolicy.forward` guards its ego term with `if self.ego_emb is not None and ego is not None` — a **build-time** flag AND a **call-time** argument. Every audit so far (the panel's escalation 4, `V5_PLAN`, the brief I was given) checked the *build flag* and concluded "the lever is off". It is — **and the call-site condition is independently, silently false in 100 % of eval paths**, so turning the build flag on changes nothing until someone also passes the argument. ⇒ **When a feature is gated by both a constructor flag and a forward-argument, flipping the flag is not enabling the feature. Grep the CALL SITES, not just the config.** *Detection heuristic: an `and` in the guard with one operand from `__init__` and one from `forward`.* Suggested owner text is verbatim above. | **PI / RETRACTION_LOG owner** |
| **E7** | ⚠️ **`MODEL_REGISTRY` should stamp that `ego_input_on_planners = False` on v1, the no-speed control and v4.** The panel measured it; the registry does not carry it, and "the speed fix" is routinely quoted without noting that it never reached the planner. | **Model-registry agent** |

---

## 9. Deliverable manifest

Repo dir: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-28-tactical-action-input/`
Everything `git add`-ed into the working tree. ⛔ **I did not commit and did not push.** ⚠️ marks
anything in only ONE place.

⚠️ **Disclosure:** the Block-B deliverables were staged incrementally, and the orchestrator's own
sweep committed them mid-session as **`2b0f166`** ("C2 shipped as a real option…") — a whole-index
commit under someone else's message, which is exactly the hazard `CLAUDE.md`'s git-hygiene section
describes. **I did not author that commit**; my later edits sit staged on top of it. Flagging it so
the lineage is not mis-read later.

| artifact | where it lives | what it is |
|---|---|---|
| `TACTICAL_ACTION_INPUT.md` | repo (this dir) | this report |
| `PRE_REGISTRATION.md` | repo | frozen before any new number existed; carries the verified code reading |
| `scripts/make_retimed_arms.py` | repo | Block B builder — the factorisation, the two self-validating identities, the probes |
| `scripts/decompose.py` | repo | lateral/longitudinal decomposition (imports `pseudosim._cross_and_along`) |
| `scripts/summarize.py` | repo | ⭐ regenerates **every table in this report** from the artifacts — the tables are generated, not hand-typed |
| `artifacts/tables.md` | repo | the generated tables, verbatim as transcribed into §3 and §4 |
| `scripts/refc_v0_ablate.py` | repo · `tanitad-eval:/workspace/_egoin/scripts/` | Block A — the 3-line `v0` ablation over the published `panel_run` adapters |
| `artifacts/blockB_panel.json` | repo · scratchpad | the full 16-arm panel: gate, composites, **120 paired blocks**, per-arm-gate sensitivity |
| `artifacts/blockB_build.json` | repo | build report incl. the two self-validation identities and per-arm extrapolation fractions |
| `artifacts/decomposition.json` | repo | per-arm along/cross endpoint error + longitudinal share |
| `artifacts/ratio_distribution.json` | repo | ⭐ the UNCLAMPED `ego_progress` ratio distribution — the artifact behind §5.1a, the sharpest instrument finding here |
| `artifacts/pw/pw_*.npz` × 6 | repo | ⭐ **per-window dumps for every derived arm** — every number here recomputes from these with **no GPU, no checkpoint, no corpus** |
| `artifacts/blockB_combine.log` | repo | the combine run log |
| `stack/tanitad/ego_plan.py` | repo | the geometry + `attach_ego_input` (the v5 seam), **a NEW module — no existing file's bytes changed** |
| `stack/tests/test_ego_plan.py` | repo | **37 tests**, incl. the two identity tests, both-direction graft validation, the two premise-pinning tests and the six E1-guard tests |
| Block A artifacts | see §6 | — |

**Suites green, both, after the change:** `stack/` **1379 passed, 12 skipped** (2:05, re-run **1385** after the E1 guard) and `taniteval/`
**565 passed** (1:16). ⚠️ Both are *above* the counts in the brief (1324 / 559) — 37 of the extra are mine; the remainder are other agents' work already in the tree. ⛔ **No existing file's
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

### 9.1 Block A artifacts and running processes

| artifact | where it lives | what it is |
|---|---|---|
| `artifacts/blockA/pw_refc_*.npz` | repo · `tanitad-eval:/workspace/_egoin/out/` | per-window dumps, one per Block-A arm |
| `artifacts/blockA/arm_refc_*.json` | repo · `tanitad-eval:/workspace/_egoin/out/` | per-arm nodes incl. ckpt md5, step, `refc_ego_dropout_at_train`, envelope proof, G4 falsifier |
| `artifacts/blockA/decomposition_blockA.json` | repo | the §6.2 / §6.5 lateral/longitudinal split, all four arms |
| `artifacts/blockA/blockA_full_panel_20arm.json` + `_combine.log` | repo | ⭐ the **final 20-arm panel: 190 paired blocks**, every §3/§6 number, both gate variants |
| `artifacts/blockA/egoin_blockA.log` | repo · `tanitad-eval:/tmp/egoin_blockA.log` | the run log for all four arms |
| shipped packages | ⚠️ `tanitad-eval:/workspace/_egoin/lib/{taniteval,tanitad}` **pod-only** | copies of the repo packages, shipped by `scp` because `origin/main` lacks the new code and the pod's `/root/taniteval` is stale. **Nothing original lives there** — they are copies of tracked repo files, verified by an in-driver assert that the import resolved to the shipped path. |

**Processes and PIDs.** Host `tanitad-eval` (A40, idle before this work):
`bash /workspace/_egoin/egoin_blockA.sh`, first arm **PID 1793447**, log **`/tmp/egoin_blockA.log`**,
output dir **`/workspace/_egoin/out/`**. Four arms run **sequentially** (one A40, `OMP_NUM_THREADS=6`),
846.6 / 849.1 / 519.7 / 513.2 s. ✅ **Finished: `BLOCK_A_ALL_DONE fail=0` at 12:40:35Z.
Nothing is left running on any host by this task.** All four dumps and arm nodes are pulled into the
repo (`artifacts/blockA/`), so nothing lives only on the pod. ⛔ **Nothing was left
running on pod1 or pod2, and neither was touched** — verified mid-run: pod1 PID 699286 still training
`flagship-v2corpus-30k` at GPU 100 % (2 d 09 h elapsed), pod2 running its own armed 120° val build.

---

## 10. Self-refutations, and what was deliberately NOT done

| # | what | status |
|:--:|---|---|
| 1 | ⛔ **I nearly reported the brief's premise as verified.** `TacticalPolicy.forward` **does** have an `ego` parameter. Reading the file rather than the brief changed the claim from "no input" to "a disabled input **plus** a harness that drops it" — and the second half (E1) is a live bug the brief, the panel and `V5_PLAN` all missed. | corrected, §1 |
| 2 | ⛔ **I let a probe into the panel gate TWICE and it deleted `ego_progress` for every arm** — once because `v1_ego_double` is ceiling-saturated (97.49 %) and once, for the *opposite* reason, because `oracle_lon_straight`'s range is too small (0.0219 < 0.05). Both times the published composites changed. Caught because the published arms no longer reproduced their published values — which is exactly what the reproduction gate is for. Fixed by treating all four probes as validation probes, per the panel's own `stand_still` rule. | corrected, §3.1 |
| 3 | ⛔ **My first Block-A launch failed all four arms in 1 s each** — `panel_run` inserts stale pod paths at `sys.path[0]` and shadowed the shipped `taniteval`. The runner also reported `rc=0` for every failure because `$?` captured an `echo`. Both fixed; the driver now **asserts** the packages resolved to the shipped copy. | corrected, §6 |
| 4 | ⚠️ **Two of my own tests failed first and both were right to.** The half-speed control is not *exactly* half on a curved path (chord vs arc), and a freshly built policy ignores `ctx` entirely (§1.4). I changed the tests to state the true facts rather than loosening them into vacuity. | corrected |
| 5 | ⚠️ **My `recovery` defined-fraction guard tripped** (−2.27 pp vs the 2 pp bound) and I report the arm as not strictly like-for-like on that component instead of quietly quoting it. | disclosed, §7.5 |
| 6 | ⛔ **The per-arm gate flips my primary verdict to CONFIRM.** I could have quoted it. It is a comfort/smoothing artefact and I use the panel-wide gate. Both are published. | disclosed, §5.4 |
| 7 | **I did not train anything.** No ego-conditioned tactical head was fitted, so limitation §7.1 is real and the `REFUTE` is scoped to the longitudinal *schedule*, which is what the mechanism claim was about. | deliberate, scoped |
| 8 | **I did not touch pod1 or pod2**, so `flagship-v2corpus-30k` — the one arm that would answer §7.1 directly — is absent. | by rule |
| 9 | ⛔ **I did not hold anything to v1's 0.4271.** Verified: `taniteval/rollout.py:170` sets `actions_source="expert_future"`; it is `wm_fidelity_ade_2s` and is not a planning bar. | correct by construction |
