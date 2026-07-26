# v5 — what Bar A actually means, and the two directions that follow

**Written 2026-07-26 on the PI's instruction**, immediately after Bar A returned `REFUTE`.
> *"give up and do nothing is not an option… The two main directions remain and must be:
> (1) Combine our both most encouraging results: v1 WM results and the REF-C good results of the
> planner. (2) Leverage performance and efficiency dominance based hierarchical planning,
> prediction, thinking and goal setting, e.g. in picking the best trajectory or trajectory plan."*

**The PI is right, and Bar A supports both directions rather than blocking them.** This document
states precisely what was refuted, what was not, and the experiment ladder that follows.

---

## 1. What Bar A killed — stated narrowly, because the narrowness is the point

MEASURED, `…/2026-07-26-bar-a-selector/raw/bar_a_produced.json`, pre-registered before the run:

**Re-scoring the frozen v4 fan with a learned head over the existing latent features cannot reach
v1.** Even fitting *and* scoring on the **same windows** — maximum possible overfit, not a
generalization number — the ceiling is **0.4907 (CE) / 0.5224 (regret)** against v1's **0.4271**.
Out-of-fold, both arms were *worse* than as-trained (recovered **−4.2 %** and **−11.0 %**).

**So one thing is settled: no loss function over these features fixes the picker.** The waste is
real and it is **not recoverable by re-scoring**. *(That corrects my own earlier framing to the PI,
which implied the 0.4093 / 0.6058 m headroom was recoverable.)*

## 2. What Bar A did **not** kill — four openings, each untested

| # | not refuted | why it matters |
|---|---|---|
| **1** | **The fan is good.** `oracle_in_fan` = **0.2505** deployable, vs v1's **0.4271** | the proposals already beat our best deployed model by **41 %**. The generator is not the problem. |
| **2** | A **different selector INPUT** — only the existing feature set was tested | the information may exist; it may just not be in what the head reads |
| **3** | **Simulative** selection — Bar A tested a **discriminative** scorer only | ⭐ the world model can *roll each candidate forward and score what happens*. This is the direct answer. |
| **4** | **Hierarchical** selection — a flat 1-of-256 pick was tested, with no strategic/tactical structure | ⭐ we claim a hierarchy and then select flat. That claim has never been used where it would pay. |

**Openings 3 and 4 are exactly the PI's two directions.** They are not a consolation prize; they are
the hypotheses Bar A was never designed to test.

## 3. The reframe: the question moved from **ranking** to **representation**

The fan contains a 0.2505 m trajectory. A learned head over the latent cannot find it **even when
allowed to memorise the answer.** So the information that identifies the good candidate is **not
exposed in the features the selector reads**.

Two ways to get at information a feature vector does not expose:
- **simulate it** — roll the candidate forward and observe the consequence (direction 2);
- **condition on structure it lacks** — a goal and a manoeuvre class the flat scorer never saw
  (direction 2, hierarchical form);
- **swap in a better representation** — a different world model, or a different planner (direction 1).

---

## 4. Direction 1 — v1's world model + REF-C's planner

**The asymmetry that motivates it, MEASURED:** v4's own `wm_canary_ade_2s` is **1.1409** (bar ≤0.55)
while the v1 line is cited at **~0.452** *(INHERITED — to be established, not assumed)*. **v1 imagines
far better than v4. v4 proposes far better than v1.** Nothing has combined them.

⚠️ **And a scoring rule is only as good as its simulator.** If we score candidates by imagining
consequences, we should imagine with **the best world model we have**, which is not v4's. Whether v1's
WM can score v4's fan is an **architectural-feasibility question that must be established, not
assumed** — and a clean negative there is itself a real finding about arm compatibility.

⚠️ **The REF-C confound that must not be inherited:** REF-C evaluates with **`nav_cmd=None`**, so its
decoder never had a working route input. This is the documented reason the *"strategic choice is a
~2 % lever"* reading was confounded. **A planner with a working goal input has never been compared to
one without.** That comparison is an arm, not a footnote.

## 5. Direction 2 — hierarchical, imagination-based selection

**The mechanism:** for each candidate, roll the world model forward under its action sequence and
score the **imagined outcome** — corridor departure, along/cross-track deviation from goal, kinematic
plausibility, terminal state. Then select. Structure the selection strategically → tactically →
operatively rather than flat.

**The instrument already exists and was verified in source**: `metric_dynamics.rollout_decode`
advances its window by appending the model's **own predicted latent** — no frame is ever re-encoded,
and `k` is free. It is a per-candidate imagination roll-out, ready to use.

**Bars, pre-registered:**
- **CONFIRM** — beats **0.4907**, the in-sample ceiling of *any* re-scoring of this fan ⇒ imagination
  does what discrimination provably cannot.
- **STRONG** — additionally beats **v1's 0.4271** ⇒ the synthesis is real and v5 has a spine.
- **REFUTE** — fails to beat 0.4907 ⇒ the limit is the fan's own action-sequence information, and we
  say so.

**Stratify by simulator quality.** Bar B measured that **22.7 % of windows are already under the
canary bar** with a heavy tail (p50 0.9788, p95 2.6356) — **the WM failure is concentrated, not
uniform.** If imagination-scoring works where the WM is good, that is a **deployable gate**, not a
failure — and it is the same shape as the tactical brain deciding when to think harder.

## 6. Why this is also the efficiency argument

Imagination-scoring costs `n_candidates × k` predictor steps and **zero camera passes**. That makes
the efficiency claim structural rather than rhetorical: *think more, look less*.

It also joins the blind-imagination stream: **`T_blind`** — how long the model can drive without the
front camera — is the horizon over which imagination-based scoring is trustworthy. **One measurement
serves both.**

⚠️ **§7.3 binding:** state what value of the efficiency metric would be **disappointing** before
quoting it. H2 MEASURED that the naive framing is information-free — never-escalate saves 85.7 %, a
perfect oracle 85.6 %, the whole span 0.1 pp. Build an axis with real dynamic range or report that
you could not.

---

## 7. The ladder, in order, with what each costs

| step | question | cost | bar |
|---|---|---|---|
| **E-V5-1** | does imagination-scoring beat the re-scoring ceiling? | **hours** — reuses Bar A's cache and frozen fan | 0.4907, then 0.4271 |
| **E-V5-1b** | can v1's WM score v4's fan? | hours (feasibility first) | architectural verdict |
| **E-V5-2** | does hierarchical selection beat flat, with a **working** goal input? | hours | flat pick on the same fan |
| **E-V5-3** | cost-vs-quality of candidates × rollout depth | hours | an axis with dynamic range |
| **v5 train** | only if the above CONFIRM | GPU-week | pre-registered from the results |

**No GPU-week is committed until the hours-scale ladder returns.** That discipline is what turned a
59-hour v4 restart into a 2-hour refutation — and it is why declining *that* restart is not
inaction: **it is what makes these four experiments affordable.**

## 8. Status

E-V5-1 through E-V5-3 are **running on the eval pod** as of 2026-07-26. The blind-imagination sweep
that supplies `T_blind` is running on pod2. v4's restart budget stays **0/2** — unspent, not
forfeited.
