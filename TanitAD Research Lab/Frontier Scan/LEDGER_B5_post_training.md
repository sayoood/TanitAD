<title>LEDGER B5 - post-training, RL, finetuning</title>

# LEDGER B5 — Post-training, RL, finetuning

`Frontier Scan track B5. APPEND-ONLY (charter §3). The running state of the art in this track, WITH OUR`
`POSITION IN IT. Never rewrite history — correct with a dated correction entry.`
`Transfer question for this track: can a trained WM be improved without a retrain?`

**Track opened 2026-09-05.** Prior status: SCANNED 2026-08-31 (AtomVLA + RLVR-World banked
**abstract-only**, full text owed since); not scanned 2026-09-01 or 2026-09-02.

---

## Entry 2026-09-05-01 — ⭐ Today's measured RL regression is a KNOWN FIELD FAILURE MODE

`Post-Training in End-to-End Autonomous Driving: A Unified View, arXiv 2607.08072. PUBLISHED,`
`ABSTRACT/SUMMARY-LEVEL READ — declared, not a full text. Banked this pass.`
`Cited by: Frontier Scan/Daily/2026-09-05/RESULT.md F6.`

### The claim

Most RL post-training for driving uses **GRPO-style optimisation borrowed from LLM alignment**, while
driving quality is *multi-dimensional, long-horizon and highly context-dependent* — trading off collision
avoidance, drivable-area compliance, comfort and progress. Consequently **improvements in one dimension
may hide degradation in another**. The recommended direction is **localized feedback around key moments**
(lane changes, yielding, braking, near-collisions) instead of one scalar reward over a whole rollout.

### ⭐ COMBINATION — this is our own result, published independently

MEASURED by TanitAD **today** (2026-09-05, commits `0762b0d` / `7237cb4`):
*"the V2-faithful RL stage makes refcv3's fan LESS safe, and the veto-only control proves it is the REWARD
that does it"*, and *"my own pre-registration was wrong about why"*.

⇒ **`H-RL-MIN-1`'s fan-safety regression is a mechanism the field has already named, not a defect in our
implementation.** ⭐ **That changes the response**: from *debug the arm* to **change the reward's
granularity** — event-localised credit rather than a scalar rollout return.

⚠️ **Evidence discipline:** this is a survey read at abstract/summary level. It corroborates a
**mechanism**; it supplies **no number in our units**, and it may not on its own decide a GPU-day
(charter §5.2). The MEASURED half is ours; the PUBLISHED half only says we are not alone.

### CHANCES / RISKS

*Chance:* a cheap re-specification of an existing arm — event-localised reward on the four families we
already compute — rather than abandoning the RL stage.
*Risks:* **(a)** event-localised credit needs an **event detector**, and a mis-specified detector moves the
failure rather than fixing it; **(b)** our four-families rule already decomposes the objective, so part of
the survey's advice may already be satisfied and the true defect may be the **aggregation**, not the
granularity — these are different fixes and must not be conflated; **(c)** abstract-level read.

### EXPERIMENT (pre-registerable, 0 GPU to design)

Re-score the **existing** RL and veto-only arms with the reward decomposed per family and per event band
instead of aggregated. **Committed in advance:** if the safety regression is visible in one family while
another improves, the survey's mechanism is CONFIRMED on our data and the fix is granularity; **if every
family moves the same way, the mechanism is REFUTED for us** and the reward's *sign or shape* is the
defect, not its resolution.

### ⚠️ Adjacent, and it becomes load-bearing after today's B7 entry

`2606.30807` — *"Off the Rails: Hijacking the Scoring Head in Generative End-to-End Driving Planners with
Safety-Violating Adversarial Perturbations"*. **Unread, not banked.** Relevance rose sharply this pass:
B7's GuideFlow entry measures the **scorer** as the dominant component (+15.9 EPDMS), so a published attack
on scoring heads is a direct robustness question for the component we are about to make load-bearing.

`Track debts carried: AtomVLA and RLVR-World full text (owed since 2026-08-31); 2606.30807 unbanked;`
`DriveDPO 2509.17940 and PlanRL 2606.26858 surfaced and untriaged.`

---

## 2026-09-09-01 - RLIR: an inverse-dynamics reward, and why we must NOT adopt it yet

`lib 2509.23958 "Reinforcement Learning with Inverse Rewards for World Model Post-training". ABSTRACT-ONLY, declared. Class PUBLISHED abstract-only. Retrieved 2026-09-09.`

RLIR *"derives verifiable reward signals by **recovering input actions from generated videos using an
Inverse Dynamics Model**"*, mapping video to a low-dimensional action space so GRPO has an objective,
verifiable reward. Reported: **5-10 % gains in action-following**, up to 10 % on visual quality, higher
human preference. They frame it as the *"first post-training method specifically designed to enhance
action-following in video world models"*.

**It is exactly the shape our RL fan-safety regression wants - and it is exactly the shape GS-1 warns
about.**

GS-1 (from Delta-JEPA, 2026-09-02): an inverse decoder can succeed on *"action-correlated cues"*
absorbed into the endpoint *"without requiring the model to represent the actual transition"*. **RLIR's
reward IS an inverse decoder scored on the model's own output.** On our action channel - realised
motion, `r = 0.9988` - that reward is plausibly satisfiable by the leak rather than by action-following.

**Position: RLIR is BLOCKED FOR TANITAD PENDING THE GS-1 INSTRUMENT AUDIT.** Adopting it first would
spend RL compute optimising our best-documented confound. This is not a refutation of RLIR; it is an
ordering constraint that GS-1 already implies and that nobody had connected to this method.

**The contrast that sharpens both:** today's A2 item (ACPC) compares **two rollouts under identical
actions** and never reads the real `z_{t+1}` endpoint, so it does **not** carry the GS-1 leak. **Two
action-sensitivity instruments arrived on the same day with opposite leak exposure.** Prefer the
leak-resistant one for measurement; gate the leak-prone one behind the audit.

**Carried forward from 2026-09-05 and still standing:** `2607.08072` - a scalar rollout reward hides
cross-dimension degradation; the fix is event-localised credit (work item FS5-5).

## 2026-09-10-01 — Closed-loop post-training: right diagnosis, unmeasured prescription

`Band-D counter-search (query Q10).` Sources: Sim2Real-AD `2604.03497` · AD-R1 (CVPR 2026) · RAD-2 `2604.15308` · `2607.08072`.

The **diagnosis** — open-loop is insufficient because actions change future states and errors compound — is confirmed at ≥3 independent sources and by our own three measurements (action echo 97.9 % vs 0.0 % hold-action; Alpamayo's open-loop metric flat across 3.3× params; Waymo L5).

⛔ The **prescription** — post-train on simulator rollouts — is contested. **Sim2Real-AD verbatim:** *"Most simulator-trained policies are tightly coupled to their training environment… **direct deployment fails even when the policy performs well in simulation**."* **AD-R1 verbatim:** *"Conventional RL Post-training relies on external simulators, suffering from a sim-to-real gap and heuristic rewards"* — and it **replaces** the external simulator with an internal world model **for that reason**, i.e. an independent group arguing against the exact mechanism NVIDIA recommends.

⭐ Pairs with `2607.08072`'s scalar-reward masking (ledger 2026-09-05): closed-loop post-training can lift an aggregate while regressing a family — which is what we measured on refcv3's RL stage.

⇒ Guideline **T-7**: any TanitAD closed-loop post-training arm reports **per-family reward decomposition from the first run**, not after a regression appears.

Full adjudication: `../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md` rows **A16-1 … A16-3**. New debt **D-12** — no AlpaSim fidelity characterisation exists in any NVIDIA source we have read, **and we run AlpaSim ourselves, so it is answerable by measurement rather than only by reading.**
