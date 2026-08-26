# Plan revision — what E-DEC-57 changes, and what it does not

**Written** 2026-08-26 · **Author** Master Mind · **Trigger** the PI: *"What does
this mean for our plan?"* after E-DEC-57 established that our `action` channel is
the ego's measured motion in other units (closed-form r **0.9988**).

⛔ **This revises SEQUENCING and one product's ROLE. It does not revise the 8
products, the org, or the north star** — those are the constitution's and the PI's.

---

## 0. ⛔⛔ CORRECTION, 07:05 — §2 AND §3 ARE WITHDRAWN. The PI refuted them within the hour.

**The PI:** *"but the AV dataset must contain ego data, why do we need an IDM?"*
**Correct, and §2/§3 below do not survive it.**

1. ⛔ **PhysicalAI HAS ego data** — `curvature`, `ax`, `vx/vy`, quaternion yaw. **An
   IDM reconstructs actions for corpora that have NONE** (YouTube, dashcam,
   smartphone). That is P3's ORIGINAL scope and it remains right *there*. ⇒ **§2's
   "P3 moves onto the critical path" CONFLATED TWO DIFFERENT PROBLEMS and is
   withdrawn.** P3's design constraint (never regress the derived `steer`) is still
   worth keeping — but as a note on P3, not as a programme re-prioritisation.
2. ⛔ **E-DEC-57 establishes something NARROWER than §1–§3 assumed.** It shows our
   *"action → ego dynamics"* results were **CIRCULAR** — the action *is* the
   realised motion. It does **NOT** show the action channel is unusable for world
   modelling: **`(v, κ)` is a legitimate control input; a bicycle model takes
   exactly that.** "State vs command" was never the defect, and building a plan on
   that distinction was an over-reach.
3. ⇒ **§3's comma2k19 gate is NOT the decision it was billed as.** It would tell us
   whether command and realised motion differ (interesting for actuation lag), but
   it does not decide whether action-conditioning is possible.

⭐⭐⭐ **AND THE MISS THE PI'S QUESTION SURFACED, WHICH IS LARGER THAN THE ERROR IT
CORRECTED: every ego panel ran on 20 of 129 AVAILABLE held-out clips.** The measured
null reached \|t\| **3.49** largely *because* n = 20 — with 20 clip-level scores the
SE estimate is itself noisy, so the statistic is heavy-tailed. ⇒ **The whole chain
built on that null — E-DEC-53 (census null), E-DEC-54 (retracting "the encoder
carries ego state"), E-DEC-56 (two more retractions) — is PROVISIONAL ON AN
UNDERPOWERED PANEL, and I presented it as settled.**

At 129 clips the null **tightens** (more folds ⇒ a better SE estimate) while a real
effect **grows ~2.55×**: `rdw8p30k`'s yaw-rate would go 2.76 → ~7.0 *if it is real*.
**The discrimination improves from both sides.**

⇒ ▶ **RUNNING NOW:** the identical ego panel at **129 clips** (`rdw8p30k`,
`postrain30k`) and a **fresh null at 129 clips** — the banked constant is explicitly
scoped to ~20-clip panels and may not be reused. **No IDM, no simulator, no new
data.** ⛔ **Nothing in §1–§6 below should be acted on until that reads.**

⚠️ **The transferable lesson:** I measured a null, corrected three claims against it,
built a plan revision on the corrected picture — and never asked whether the panel
had the power to see the effect in the first place. **A null is not a finding when
n is a free parameter you left at 15 % of the corpus.**

---

## 1. ⭐ The one structural consequence: the programme was two tracks wearing one name

| | **TRACK A — representation & perception** | **TRACK B — world model & imagination** |
|---|---|---|
| status | ✅ **SOLVED, lever known** | ⛔ **BLOCKED, and not on compute** |
| the lever | `--init-from <DINOv3-distilled ckpt>` | — |
| evidence | drift 0.175–0.365 vs scratch 0.614–0.642 (no overlap); `n_agents` **+0.1220** > frozen DINOv3 **+0.0998** | ten objectives failed; the action channel has no causal content |
| needs actions? | ⛔ **NO** | ✅ **entirely** |
| next spend | **scale it** | ⛔ **spend nothing until §3's gate reads** |

⇒ **Track A can proceed to full scale now.** It feeds scene understanding,
occupancy, agent slots, the situation classifier — none of which depend on
action-conditioning. **Track B cannot be advanced by objective design on this
corpus, and ten attempts is enough evidence of that.**

⚠️ **The two must stop being reported as one result.** A v7-full trained on the S-W
recipe advances **A** and says nothing about **B** — and must not be presented as a
working world model.

---

## 2. ⭐⭐⭐ P3 moves from convenience to CRITICAL PATH — and gains a hard constraint

The constitution already names it:

> **P3 · TanitAD_DataReconstruction** — *"…**IDM reconstructing actions from
> observation**"*

**It was scoped as a data-VOLUME play** (turn YouTube/dashcam video into usable
training data). **E-DEC-57 makes it a CORRECTNESS prerequisite**: an inverse
dynamics model is the only mechanism in the plan that can produce an action channel
which is *not* a restatement of the trajectory we already observe. The literature
family is VPT (2206.11795), LAPO (2312.10812), Genie (2402.15391) — all banked
2026-08-26, all sha256-verified.

⛔⛔ **THE DESIGN CONSTRAINT E-DEC-57 IMPOSES ON P3, AND IT IS LOAD-BEARING:**

> **P3's IDM must regress a TRUE COMMAND — CAN steering-wheel angle, pedal
> position — and NEVER the derived `curvature`/`steer`.**

An IDM trained to predict our existing `steer` from observation pairs would
reconstruct **the same tautology at scale**, on far more data, and every downstream
result would inherit it. ⇒ **This constraint must be written into P3's work package
before any IDM work starts.** It is the difference between P3 unblocking Track B and
P3 industrialising the defect.

---

## 3. ⛔ THE GATE — one CPU hour decides which plan we are on

**Question:** does a genuine command separate from realised motion in any corpus we
hold?

**Test:** on **comma2k19**, `r(steering_wheel_angle, v·κ)` on the same held-out
protocol as E-DEC-57. ~1 h, CPU, no GPU, no new data.

| outcome | the plan we are on |
|---|---|
| **r ≈ 0.99**, as PhysicalAI | ⛔ **No observational corpus can settle action-conditioning.** P3's IDM cannot help — there is no command to reconstruct. Track B needs **interventional data** (AlpaSim / simulation), which is a **PI provisioning decision**, and the world-model thesis is parked until it lands. |
| **r clearly < 0.99** | ⭐ **We have a real action channel.** P3's IDM becomes buildable and load-bearing: train on comma2k19 CAN, pseudo-label PhysicalAI. **The ten failed objectives then deserve exactly ONE honest rerun** — the first ever on a channel with causal content. |

⇒ **Nothing about Track B should be scheduled before this reads.** It is the
cheapest decision in the programme and it changes the next three months.

---

## 4. Sequencing — what to spend GPU on, in order

| # | item | track | status |
|---|---|---|---|
| 1 | **`postrain30k_seed1`** — the first run-to-run variance estimate the programme has ever had | A | ▶ **running** (step ~7,600) |
| 2 | **comma2k19 CAN separability** | gate | ⏸ **1 CPU-hour, unblocked, run next** |
| 3 | **v7-full on the S-W recipe** (`--init-from` distilled) | A | ready once (1) confirms the lever replicates |
| 4 | **ACT-Bench adoption** | eval | banked, "queued" for days, still not done |
| 5 | P3 IDM — *only if the gate opens* | B | blocked on (2) |
| ⛔ | an eleventh objective term | B | **do not** |

⚠️ **Item 1 is not optional bookkeeping.** Track A's entire claim — the distilled-init
lever — is **single-seed**. If seed 1 does not reproduce the drift separation, the
one solved axis is not solved, and item 3 should not launch.

---

## 5. What must change in what we publish

- ⛔ **The hierarchy/imagination thesis is currently UNSUPPORTED.** §9 of the paper
  (goal-vocabulary planning *over the world model*) assumes imagination works.
  Nothing in this campaign supports that, and E-DEC-57 explains why it could not
  have been tested. **It must be labelled as a design, not a result.**
- ⛔ **T1 has NEVER been run on any v7 arm.** Every number in this campaign is T0 —
  a world-model diagnostic. **No capability claim is available at all** until T1 is.
- ✅ **Track A's result is publishable now**, and is a genuine one: a single
  initialisation change defeats collapse and beats a frozen DINOv3 teacher on the
  target that matters most.
- ⭐ **The negative is publishable too, and may be the more useful contribution:**
  *observational driving corpora can encode "actions" that are kinematic
  restatements of the ego trajectory, and world models trained on them cannot be
  action-conditioned no matter how the objective is designed.* Ten terms, a measured
  null, and a closed-form identity at r 0.9988 is a strong, reproducible case.

---

## 6. What does NOT change

The 8 products, the org, the north star, parity, the eval doctrine, the four metric
families. **This revision moves P3 onto the critical path and pauses Track B behind
a one-hour gate. Everything else stands.**
