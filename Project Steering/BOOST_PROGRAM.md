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

### M5 — ⛔ OVERRULED BY THE PI, 2026-07-26: **at least FIVE streams, not three**

> *"no concentration on only three stream, at least five streams"*

**This is the PI's call and it is implemented as stated.** But it removes the mitigation I had
proposed for **C-IV** (breadth multiplies the surface on which a false positive can appear), so
C-IV needs a different answer rather than none. **The replacement: if breadth stays, the per-stream
verification bar rises to carry the load.** Concretely, and binding:

1. **M1's tiers do the work M5 would have done.** With ten streams, PROVISIONAL results are ten
   times as likely to reach the PI. So: a stream may run at any breadth, but **nothing leaves it as
   CONFIRMED without an independent reproduction path**, and DECISION-GRADE additionally needs the
   pre-registration and the stated falsifier.
2. **Every stream carries its own falsifier before it starts.** A stream that cannot say what result
   would end it is not a stream, it is an activity.
3. **Cross-stream premise contamination is the specific breadth risk** — today's three brief errors
   (C-II) all came from carrying one stream's number into another stream's brief. **M2 is therefore
   load-bearing under breadth, not optional.**

~~Original proposal: three active streams, everything else paused with state banked.~~
**Superseded.**

### M5-bis (retired) — the original concentration proposal

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

> ## ⛔ CORRECTION, 2026-07-26 — the "selector waste" above is an ORACLE BOUND, not a recoverable budget
>
> **Bar A ran and returned `REFUTE` on both goal surfaces.** The framing in this section — that the
> selector "throws away 0.4093 m / 0.6058 m", with the implication that the headroom was
> *recoverable* — is **retired**. It is retained above only so the reasoning that led to Bar A stays
> legible.
>
> **The diagnosis's own falsifier fired.** `V4_RESTART_LEVER.md` §6 had committed, in advance: *"if a
> re-scored frozen fan cannot get below ~0.43 m, the fan's v1-beating content is true only in an
> oracle sense."* Fitting a re-scorer **in-sample** — same 6,844 windows it is then scored on, 796k
> params, **zero generalization gap** — bottoms out at **0.4907**, still separated-worse than v1's
> **0.4271** and **1.96×** the fan's own best. Out-of-fold both arms were on the **wrong side of
> zero**: CE_CONTROL **−11.03 %**, REGRET **−4.20 %**, against a ≥ 70.8 % bar.
>
> ⇒ **`oracle_in_fan` = 0.2505 remains MEASURED and true — but it is an ORACLE quantity. It is not a
> budget any scorer over these features can draw on.**
>
> ⭐ **AND THE SAME RUN GAVE THE NUMBER THAT REDIRECTS THE PROGRAM.** The in-sample ceiling moves
> **0.4907 → 0.4138** when the score is given goal **INFORMATION**, while changing the **OBJECTIVE**
> moved it by only **0.0317 / 0.0089 — and in the wrong direction**:
>
> > **INFORMATION BEATS OBJECTIVE BY 2.4× – 8.6×.**
>
> (0.4138 is *below* v1's 0.4271, though in-sample and therefore not deployable.) This is the
> pre-committed §0.8 privileged-input reading with a number attached, and it is why v5's ladder is
> **conditioning-first, not loss-first** — see `V5_PLAN.md`.
>
> **Decomposition, and it is damning for the objective route:** the regret arm left the
> **longitudinal** axis flat (+0.0038, p = 0.63) — *the axis carrying 100 % of v4's regression* — and
> made **lateral separated-WORSE** (+0.0222 [+0.0043, +0.0522]). **It did not act on the failing axis
> at all.**
>
> **Two mechanism findings worth keeping:** fine-tuning drove the factorised grafts past the head's
> **own** `seam_fail = 1.5` guard (**1.652** observed vs 0.1204 as-trained) — *the architecture
> actively refuses the direction a ranking objective pushes it*. And **fp16 caching is unsafe on this
> selector**: 256 candidates are separated by less than fp16's ULP, and a self-test caught a
> **0.0028 m** bias and aborted **before** training.
>
> **Tier: CONFIRMED, and DECISION-GRADE for the negative decision** — pre-registered, estimator
> named, falsifier stated, **five independent reproduction paths** (the committed 30k numbers
> reproduced to max abs diff 5e-5). **Not** decision-grade for any positive claim. Cost: **1.88
> GPU-h** of the 1–2 h authorised.

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

---

## 7. HARVEST — the other half of the recovery (added 2026-07-26 on the PI's question)

> *"look on all our program agents how we can leverage their results and incorporate them in the
> recovery plan."*

**This reframes the plan, and the reframe is correct.** §1–§6 attack the **false-positive rate** — the
claims we make and retract. But the program has a second, quieter failure: a **low harvest rate.**

**There are 134 agent deliverable directories.** We have been **generating results faster than we
have been harvesting them.** The retractions got attention precisely because they were loud; the
*unexploited true positives* are silent, and there are almost certainly more of them. The program's
stranding history is the same disease in an earlier form — an orthogonality instrument unmerged
**10 days**, LAL-v2 anticipation **12 days**, TanitEval and REF-B v2's architecture each on a single
disk.

⇒ **M7 — harvest is a standing obligation, not an occasional audit.**

### 7.1 The highest-leverage item, available now for near-zero compute

`MODEL_REGISTRY.md` §1.2a, MEASURED: **"any verdict resting on a 40-episode 'not separated' is
UNPOWERED, not refuted."** Going 40 → 600 episodes shrinks CI half-widths **×2.8–3.9 (mean 3.4)**.
**One verdict has already flipped on power alone** — `along_track_vs_cv`, tie → "model wins", with the
point estimate moving **0.7 %**.

**We have been treating "not separated" as "no effect" across a program's worth of reports.** Every
one of those was a **statement about our sample size**, not about the world.

The corpus exists (600 episodes on pod2, parity-verified as an order-preserving superset of the 40).
The harness works. The checkpoints exist. **Re-scoring is cheap, and some fraction of our nulls are
real effects we discarded.** Best value-per-GPU-hour available, and it invents nothing — it is pure
recovery.

⚠️ **The discipline that stops it becoming a fishing expedition:** the list of claims to re-adjudicate
is **frozen before any re-scoring runs**, ranked by |effect| / half-width. A null that separates is a
**rediscovery**; a null that stays null is now a **powered** null and is worth more than it was. Both
outcomes recorded. **No claim is added to the list after seeing a result.**

*(H2 supplied a worked example the same day: its primary read is UNDERPOWERED — and it knew this
because it measured **the label's own lift on its own held-out subset**, 2.171× [0.645, 4.469], not
separated, **before** reading the classifier. A power ceiling measured first turns an ambiguous null
into a known-unanswerable question.)*

### 7.2 The four other harvest inventories

| # | inventory | why it pays |
|---|---|---|
| **H2** | **capabilities built but never called** | `ood.py`, `clhorizon.py`, `corridor.py`, `strategic_probes.py`, `blind_baseline.py`, `hierarchy_guard.py`, `argoverse2.py`, `parity.py`, `registry_lint.py` — each cost real effort; several are wired into nothing |
| **H3** | **stranded integrations** | the 10-day-README failure, still live: AV2 has **no ingest driver**, `clhorizon::run_v4` **raises**, Overture's registry entry is **written but not applied**, three IDM fixes unowned |
| **H4** | **unresolved cross-agent contradictions** | two agents disagreeing with nobody adjudicating is a false positive waiting to be quoted |
| **H5** | **levers nobody connected** | *the* pattern to hunt. Demonstrated today: P1 established the closed-loop envelope is **not** a renderer limit (the yaw warp is geometrically exact), so half of it is our own arm's OOD sensitivity ⇒ **training-time off-path augmentation** becomes a candidate lever for v4's `wm_canary` — **a bar that otherwise has none** |

### 7.3 A third failure mode, found today: the unfalsifiable *benefit* claim

C13 said a **guard** that cannot fail is not a guard. H2 found the mirror image on the **benefit** side.

Its efficiency claim — *"selective camera activation saves ~85 % versus always-on-7"* — is true,
measured, and **information-free**: against always-on-7, **never escalating saves 85.7 %**, a **perfect
oracle saves 85.6 %**, and the measured operating point sits at **84.8 % [84.5, 85.1]**. **The entire
span between a useless gate and a perfect one is 0.1 percentage points.**

⇒ **No compute-saving number can distinguish a good gate from a useless one.** The claim could never
have failed, so it never carried evidence. **Standing rule: state what value of a BENEFIT metric would
be disappointing before quoting it** — the same question C13 forces for guards. The informative axis
here was recall at a fixed budget, and it had to be constructed.

### 7.4 What this changes about how streams are run

Under the PI's five-plus-stream directive (§M5), harvest is what stops breadth becoming accumulation.
**A stream whose result nobody reads has the same value as a stream that produced nothing, at higher
cost.** Binding:

1. **Every stream's closing report names what OTHER stream its result unblocks** — or states plainly
   that it unblocks nothing. H5 stops being luck and becomes a required field.
2. **A result is done when it is USED or explicitly SHELVED WITH A REASON**, not when it is measured.
   The "finish before you start" rule, applied to findings rather than artifacts.
3. **The harvest index is re-run and diffed**, so new stranding surfaces in days, not weeks.
