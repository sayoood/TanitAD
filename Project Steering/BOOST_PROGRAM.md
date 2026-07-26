# BOOST PROGRAM — raising the program to its intended standard

**Commissioned by Sayed, 2026-07-26**, on the finding that we produce **too many statements that we
correct shortly afterwards**. This document accepts that finding, diagnoses it mechanically, and
defines the changes. It is a working instrument, not a memo.

**The measure that makes the criticism concrete:** `RETRACTION_LOG.md` holds **61 retractions**, of
which **18 were logged today**. That is the highest single day in the program's history. A retraction
log is supposed to be a learning mechanism; at 18/day it has become a **substitute for getting it
right**, and it lets us feel rigorous while still shipping errors to the PI.

---

## 1. The diagnosis — five mechanical causes, not a motivation problem

Each is stated with the specific failure that demonstrates it, from today.

### C-I. We publish at agent-completion speed, not at verification speed

The pipeline has been: agent finishes → I write a chat headline within minutes → a *later* agent
finds the error → retraction. **The verification is happening after the announcement.**

*Demonstrated:* the K=60/K=70 horizon recommendation was headlined at 12:4x and corrected by 17:0x
the same day — by the very agent commissioned to check it. The correction was already scheduled when
I published the claim.

### C-II. I inject unverified premises into agent briefs

*Demonstrated, three times today:*
- *"the gate recommends K ≈ 60–120"* — **I invented that range.** It is not in the gate report.
- *"130–472 lane polygons per scene"* — wrong at **both** ends (actual 40–702).
- *"λ_plan = 1.0 lets the planner loss starve the world model"* — **λ_plan is not a loss weight**;
  `lambda_plan == 1` is a documented no-op. I named a culprit without reading its implementation.

Where I *did* mark a number `INHERITED` and instruct the agent to verify it, **the agent caught the
error** (the lane-polygon range). The mechanism works when used. I did not use it consistently.

### C-III. The instrument layer is unstable and is being used while under construction

Every one of these was found *in the same session it was relied upon*:
`run_gate.py` refused every completed run (0-index off-by-one) · `lateral.py` mislabelled the horizon
**5×** · `corridor.py` was **absent** from the eval host · `ood.py`'s predecessor could not fail ·
`eval_flagship_v4.py` `SyntaxError`d on the designated eval host · `clhorizon.py::run_v4` raises on
its first step.

**A program cannot outrun its measurement layer.** Ours is younger than the claims built on it.

### C-IV. We measure many things and decide few

Ten-plus concurrent streams. The PI's stated model is *"if I define a problem, iterate until it is
solved at the highest possible quality"* — that is **depth**. We have been running **breadth**, and
breadth multiplies the surface on which a false positive can appear.

### C-V. The one number that matters most is not yet measurable

**No admissible closed-loop horizon is a measurement rather than an extrapolation** — including the
K=20 already in use. Until that is repaired, every closed-loop claim is provisional, and the D-A
mandate ("loop until significant closed-loop performance") cannot be honestly evaluated.

---

## 2. The changes — six measures

### M1 — Evidence TIERS, and only the top tier reaches the PI or a decision

Evidence *class* (MEASURED/PUBLISHED/INHERITED/…) says where a number came from. It does **not** say
whether anyone checked it. Add an orthogonal **tier**:

| tier | definition | may appear in |
|---|---|---|
| **PROVISIONAL** | one agent, one path, unreproduced | working notes only |
| **CONFIRMED** | reproduced by an **independent** path or agent | reports, chat, docs |
| **DECISION-GRADE** | CONFIRMED **+** pre-registered **+** estimator named **+** the falsifier stated | **required** to decide a GPU-day or enter `MODEL_REGISTRY` |

**Binding on me:** I stop putting PROVISIONAL numbers in chat headlines. If a result is not yet
confirmed, it is reported as *"one agent reports X; unconfirmed"* — or it waits.

### M2 — No unverified premise may enter a brief

Every number in an agent brief is either (a) verified by me at write time with its artifact path, or
(b) explicitly tagged **`INHERITED-UNVERIFIED — verify before building on this`**. There is no third
option. *(This already works: it is how the lane-polygon range was caught.)*

### M3 — Certify the instrument layer before it adjudicates anything

Every instrument that can decide something must carry:
1. a **self-test with a deliberately failing input** that asserts the failing verdict is rendered —
   the `e1c_selftest.py` pattern, which is the single practice in this program that has consistently
   worked;
2. an explicit **"what value would make this FAIL?"** answer, with proof the estimator can reach it
   (the C13 rule);
3. a version stamp and a host-compatibility check (Python version, module provenance, file hash).

**Uncertified instruments may be run for exploration but may not adjudicate.**

### M4 — Mid-run held-out gates, and stop throwing the diagnostics away

**~29.5 GPU-hours — half the v4 run — went into training past the best checkpoint**, and the four
diagnostics that would have caught it (`rank_acc`, `frac_sel_2x_worse_than_oracle`, `sel_gate`,
`sel_pen_span`) were **computed 601 times and discarded by the row-writer**.

Fix: emit them; add a **held-out probe on the deployable surface** at a fixed step cadence; stop the
run when the held-out primary is separated-worse for two consecutive probes. This converts a 59-hour
loss into a ~20-hour loss.

### M5 — Three active streams, not ten

Concentrate. Everything else is paused with its state banked, not abandoned. Proposed streams in §4.

### M6 — Repair the closed-loop measurement before making closed-loop claims

Until P1 re-validation lands, **all closed-loop numbers are labelled EXTRAPOLATION** and none enters
a kill conjunction. This is already in flight.

---

## 3. v4 — the honest answer to "why will the next run be the breakthrough"

**It will not, as currently specified — and saying otherwise would be the exact failure mode this
document exists to fix.** A selector-only restart fails the card by *arithmetic*, not by prediction.
What follows is what the evidence does and does not license.

### 3.1 What went wrong in the last run — mechanically

| fact | value | class |
|---|---|---|
| wallclock | 59.0 h (212,544.6 s) | MEASURED |
| every *training* term | improved monotonically to 30k | MEASURED |
| held-out *selection* | **separated-worse** (`ade_0_2s` +0.0584 [+0.0043, +0.1179]) | MEASURED, paired episode-cluster bootstrap |
| regression onset | `sel_gap` climbs from **~step 11,000**; level shift at **26,000** | MEASURED |
| wasted compute | **~29.5 GPU-h past the best checkpoint** | MEASURED |

This is **C11 in clean form** — a monotonically improving training loss licenses nothing about
held-out behaviour. We had no held-out early-stop signal, and the instrument that would have supplied
one was being discarded every step. **That is the whole explanation for the disappointment.** It is
an operations failure, not a science failure.

⚠️ **And a second false positive I must name:** we called this a *"formal 8-metric gate."* It was a
**6-metric gate**. `speed_benefit_recovered_frac` has **no emitter anywhere in the codebase**, and
`deploy_tick_p99_ms` was never measured. Both are recorded `null`. Calling it 8 was wrong.

### 3.2 The genuinely strong result, and it survives the obvious objection

| surface | fan's best | selected | **selector waste** |
|---|---:|---:|---:|
| goal-oracle | 0.2330 | 0.6423 | **0.4093 m** |
| **produced (deployable)** | **0.2505** | 0.8563 | **0.6058 m** |

**The objection to test was: "the fan only looks good because it is goal-oracle-fed."** It is not.
Removing the oracle degrades **the fan by +7.5 %** but **selection by +33.3 %**. Nearly the entire
oracle privilege is consumed by the *selector*, not by the world model.

⇒ **On the deployable surface, v4's proposals are 0.2505 m against v1's 0.4271 — 41.3 % better than
our deployed best model.** The world model is not the problem. This is the strongest positive
result in the program, and it is CONFIRMED (two independent goal modes, same conclusion).

### 3.3 The two bars a restart must clear — both, not either

**Bar A — the selector must recover ≥ 70.8 % of its waste merely to TIE v1.**
Deployable waste is 0.6058 m; closing 0.4292 m of it reaches v1's 0.4271. Anything less and v4 is
still behind the model we already have. **This is the number to pre-register, and it is demanding.**

**Bar B — `wm_canary_ade_2s` must fall 2.07× (1.1409 → ≤ 0.55).**
It is **identical in both goal modes**, so it is a genuine world-model deficiency, and **nothing in
the selector recommendation touches it.** Its observed descent is −21.6 % per 20k steps. **No lever
for Bar B has been identified.** That is the real risk, and it is currently unowned.

### 3.4 Therefore — the decision I recommend

**Do not spend 59 GPU-hours yet.** Spend **1–2 GPU-hours** first:

> **The discriminating experiment:** head-only fine-tune of the selector on a **frozen fan**, with the
> cost-sensitive expected-regret listwise loss. Pre-register: **CONFIRM** if it recovers ≥ 70.8 % of
> the 0.6058 m waste · **PARTIAL** if 30–70 % · **REFUTE** if < 30 %.

That costs **0.03 %** of a full run and settles Bar A outright. **Bar B needs a separate answer
before any full restart is justified** — and if we cannot name a Bar-B lever, the correct decision is
to *not* restart v4 and instead carry the fan into the v2-corpus line, which finishes in ~54 h.

---

## 4. Proposed concentration — three streams

| # | stream | why it earns a slot |
|---|---|---|
| **S-1** | **Closed-loop measurability** (P1 envelope → certified instrument → a horizon that is a genuine measurement) | Nothing in D-A can be honestly claimed until this lands. Blocks the most valuable claim we want to make. |
| **S-2** | **The selector** (Bar A test → then Bar B lever or an explicit decision not to restart) | We have a world model whose proposals beat our deployed model by 41 %. Converting that is the highest-value engineering task in the program. |
| **S-3** | **v2-corpus arm to completion** (~54 h) + its gate, with M4's mid-run held-out probe attached | Already running, single-variable, and it is the clean test of the corpus hypothesis. |

Paused with state banked: H2 (finishing its first training run), IDM/YouTube, 4-brain HP-2…HP-8,
datasets/AV2, Orin/Thor, AlpaSim consolidation.

---

## 5. Decisions needed from Sayed

1. **v4:** approve the 1–2 GPU-hour Bar-A test *before* any restart? **(Recommended: yes.)** And is
   there an intended Bar-B lever, or do we decline the restart?
2. **Concentration:** accept the three streams in §4, or choose a different three?
3. **ZOD access application** — the only *commercially publishable* corpus found; one human action.
4. **Corridor threshold:** add the measured **1.391 m** as a second grid row (additive, reversible).
5. **AV2:** approve the ~147 MiB lane-graph pull.
6. Standing: wheelbase **B** · **+17 scenes** for HP-4 · **~103 scenes** for the strategic proof.

*Blocked on his machine, indefinitely: nuScenes Terms, HF cleanup.*

---

## 6. Status of this program

M1, M2, M3 are binding from this commit. M4's row-writer fix and instrument certification are being
implemented now. M5 awaits Sayed's answer on §5.2. M6 is in flight on the eval pod.
