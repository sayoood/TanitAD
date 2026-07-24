# The planner, not the world model, is the bottleneck — a grounded synthesis

**Date:** 2026-07-23 (Europe/Berlin, UTC+2) · **Type:** design synthesis, reading over in-repo artifacts
only — **no GPU launched, no pod written to.** · **Author:** architecture subagent.

**Number discipline (CLAUDE.md).** Every claim carries an evidence class: **MEASURED** (ours + artifact
path) · **PUBLISHED** (cited) · **INHERITED** (another doc/agent, not re-verified here) · **ESTIMATED** ·
**HYPOTHESIS**. A claim that would decide a GPU-day is MEASURED or PUBLISHED. Intervals carry their
estimator. RETRACTION_LOG classes C1–C6 were read before asserting; the C5/C6 closed-loop entries of
07-22 are load-bearing and are honoured throughout (every AlpaSim number below is flagged
WITHIN-SIM-RELATIVE and ~3.2× reconstruction-OOD).

---

## 0. The thesis in one paragraph

Three MEASURED results this run converge on one conclusion: **our world model is not what is failing —
our planners are.** v1's tactical head is an offroad-prone wide-swing planner (closed-loop plan-deviation
**1.12 vs REF-C's 0.34**, MEASURED). v4.1's planner is gradient-**starved** (λ_plan clamped to 1.5e-5
since step ~2000, in-loop val ade@2s **0.705 > v1's 0.452**, MEASURED). Our strongest planner is **REF-C's
open-loop anchored diffusion** (plan-dev 0.34, pass 8/12) — yet it too departs/collides on
reconstruction-OOD input and is open-loop-trained. Meanwhile imagination-in-the-loop is a **real but weak**
lever (paired Δ ade@2s −0.213 m, MEASURED). **v4's operative planner already IS REF-C's anchored diffusion,
so the design already dodges v1-tactical's measured failure mode** — the open questions are (a) whether
v4.1's planner is bad by *design* or merely *starved*, and (b) whether open-loop training is the real
ceiling, both of which have cheap discriminating experiments below.

---

## 1. The three anchor findings — VERIFIED against the artifacts

### 1.1 AlpaSim n=12 paired closed-loop — flagship v1's tactical head LOSES to REF-C base (MEASURED)

Artifact: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-22-alpasim-closedloop-evalpod/flagship_vs_refc_suite_{NOTE.md,results.json}`.
On the same 12 `public_2601` scenes, both models fed identical NuRec renders (f-theta canon verified per
run), one rollout each, only `policy.plan()` differs:

| metric (n=12, 480×854) | flagship v1 (WM + tactical head) | REF-C base (open-loop anchored diffusion) | Δ |
|---|---|---|---|
| pass rate | **2/12 (0.167)** | **8/12 (0.667)** | −0.500 |
| mean score | **0.066** | **0.496** | paired Δ **−0.430 [−0.646, −0.215]** (excl. 0) |
| **at-fault collision** | 2/12 | 2/12 | **0.000 — TIED** (McNemar p=1.0) |
| offroad rate | **8/12 (0.667)** | 2/12 (0.167) | +0.500 (flagship worse) |
| **mean plan_deviation** | **1.125** | **0.342** | **+0.783 (3.3× wider)** |
| mean dist_traveled (m) | 71.3 | 115.2 | departs ~44 m earlier |

Sign test REF-C strictly better **8/12, flagship 0/12** (p=0.0078); pass McNemar **6 vs 0** (p=0.031).
**The mechanism is MEASURED, not a story:** flagship's tactical policy is a high-deviation planner
(plan_dev 1.12 vs 0.34) — it swings wide, and across a representative suite that drives it **off-road,
not into collisions** (collisions tied). The n=1 "flagship beats REF-C" (rollout `71f9740c`) was a lucky
wide-highway scene (C5), and it is **not** in this suite → clean out-of-sample reversal.

⚠️ **Framing (mandatory).** WITHIN-SIM RELATIVE. Both fed NuRec input ~3.2× more OOD than REF-C's
real-footage training (open-loop ADE **1.47 on these reconstructions** vs 0.4728 real; RETRACTION_LOG C6).
The paired design controls scene + reconstruction fidelity, so the delta **isolates the planner** — it is
**not** a real-world rate for either model. **Residual confound (top):** the suite is 480×854; the n=1
flagship *pass* was native 1080×1920. Mitigated but not eliminated (flagship's plan_dev is 0.94 already at
native res, §14 of the NOTE; the pairing controls resolution). This is the one unresolved thing between
"flagship is a worse closed-loop planner" and "flagship's frozen ViT is resolution-sensitive," and it has
a ~1.5 h GPU discriminator (§6, exp #1).

### 1.2 v4.1 10k in-loop — the planner is gradient-STARVED, and the gate is BLOCKED not FAILED (MEASURED)

Artifacts: `…/incoming/2026-07-23-v41-10k-gate/{STATUS_BLOCKED.md,v41_step10000_inloop_health.json,v4.1_config.json}`.
**The 10k KILL-gate returned `BLOCKED`, not a verdict** — no v4 held-out eval driver exists, ≥3 of 8 KILL
secondaries have no v4 producer, and pod1 is unprovisioned. It **must not** be forced (GATE_PROTOCOL /
V4_DESIGN §17 O-03). The numbers below are **in-loop trainer diagnostics, NOT the gate primary** (C1:
"only eval output is quotable" — do not quote val.ade as a gate result):

| in-loop @ step 10000 | value | card bar (held-out, different metric) |
|---|---|---|
| `canary_ade@2s` (plan-free WM) | **0.460** ("ok") | wm_canary ≤ 0.55 ✓-direction |
| controller `lam_mult` | **1.53e-5** (planner grad ≈ OFF since ~step 2000) | — |
| `val.ade@2s` | **0.705** | primary ≤ 0.60 · v1 = 0.452 |
| `val.oracle_ade@2s` | **0.360** | oracle_in_fan ≤ 0.30 |
| `val.miss@2m` | **0.249** | miss_at_2m ≤ 0.10 |

**The mechanism (MEASURED).** v4.1 lowered `lr_trunk` to **3e-5** (10× below v4's 3e-4;
`v4.1_config.json:optimizer`) to cure v4's hot-trunk WM degradation. It worked — but the WM is now healthy
*partly because the planner is off*. The λ_plan controller is **down-only** by design (V4_DESIGN §14.4
O-14: *"may only move it down"*) — it halves λ_plan on any canary breach and can never restore it. Because
any meaningful planner gradient breaches the canary (neighbours 0.603@9k, 0.633@11k), it halved on nearly
every eval: `lam_mult` decays monotonically 4.9e-4@7k → 1.5e-5@10k → 3.8e-6@11k. **The WM/planner tension
was resolved by sacrificing the planner.** This is a *different* failure from v4 (hot-trunk WM collapse) —
v4.1 over-corrected. Note: the design's own R1 fallback intended *"λ_plan capped at its last healthy value"*
(cap-and-hold), but what ran was naive halving-to-zero — an implementation/design gap.

### 1.3 Imagination-in-the-loop — real but weak, and a DIFFERENT comparison (MEASURED)

Artifact: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-22-imagination-closedloop-proof/`.
Flagship v1 WM, n=265 windows / 12 held-out val eps, local 4060, wall **32.4 s**. Paired, byte-identical at
tick 0, only what happens after differs:

- (B) imagination-in-the-loop (re-plan every 0.1 s on the imagined latent) vs (A) single-shot open-loop:
  **paired Δ ade@2s −0.213 m [−0.341, −0.053], separated, P(B>A)=0.0045**; divergence >5 m **39.2 %→22.3 %**.
  Verdict IMAGINATION_HELPS.

⚠️ This is the **weak** form (re-plan, NOT CEM imagine-and-select); FDE@2s not separated; self-referential
WM. Crucially it is a **different comparison** from §1.1 — re-plan-vs-single-shot *on v1's own WM*, not
flagship-tactical-vs-REF-C — so it does **not** contradict §1.1. It says: given a WM, closing the imagination
loop helps ~11 %; it does **not** say v1's planner is good.

---

## 2. Q1 — Is v4's operative planner (anchored diffusion) already the right base? **YES — a rare case where the design already dodges a measured failure.**

**MEASURED design fact.** v4's operative planner ③ is the in-repo `FlagshipV15Head` / `V15Decoder`
(`stack/tanitad/models/flagship_v15.py`), which subclasses REF-C's `AnchoredDiffusionDecoder` — d384×4L,
**256 anchors** — instantiated at 2 s dense (V4_DESIGN §2.4, §3.1, param count 9,778,604 MEASURED). It is
**the same decoder family as REF-C base**, the arm that posted plan_dev **0.342** in §1.1.

**Why the family inherits tight deviation (mechanism, well-grounded).** Anchored diffusion emits a
selection over a **fixed vocabulary of 256 FPS anchors fit to real GT trajectories** (a buffer, 0 params;
V4_DESIGN §3.3). Its output lives near the convex hull of *feasible, real* trajectories and cannot swing to
an arbitrary off-road point the way an unconstrained regression head can — which is exactly the property
that gives REF-C base plan_dev 0.34 and flagship-tactical 1.12. v4 additionally **repairs REF-C's selection
flaw** (ranks on `refined_logits` / supervises `sel_score`; [PM] §4.6 exonerates the decoder: n_modes
1→13, conf_norm 40→189, wta 0.253→0.034, man_acc at v1's level; V4_DESIGN §2.7).

**Verdict.** Plainly: **v4's operative head structurally avoids v1-tactical's wide-swing/offroad failure
mode.** Evidence class — the inheritance is a MEASURED design/source fact; REF-C base's 0.34 deviation is
MEASURED (§1.1); the forward claim "v4-operative will show REF-C-like deviation" is **ESTIMATED** (v4 not
yet evaluated closed-loop) but very well-grounded because it is literally the same decoder family.

**Two honest boundaries on this "yes":**
1. **The resolution confound (§1.1) is not yet cleared** — until the native-1080 re-run, part of flagship's
   1.12 could be frozen-ViT resolution-sensitivity rather than the head. This does not touch v4's *operative*
   head (it is REF-C-family regardless), but it does gate how cleanly we can say "v1-tactical is a bad planner."
2. **"Right base for deviation" ≠ "solves closed-loop."** v4-operative is **also open-loop-trained**, like
   REF-C — so it will likely share REF-C's OOD/closed-loop *departure* behavior (tight deviation, but still
   ~half-failing on 3.2×-OOD reconstructions — that failure is environmental, per RETRACTION_LOG C6, not the
   planner). The deviation failure mode is dodged; the **open-loop-training gap is not** (that is Q4).

---

## 3. Q2 — Why is v1's tactical head so high-deviation (1.12 vs 0.34)? **Head design, primarily — and anchored-diffusion is strictly better, not a tunable fix.**

**Primary cause: head design (MEASURED).** v1's `tactical_policy` is a **unimodal waypoint-regression head**
(`wp_heads` over 4 points {5,10,15,20}) + maneuver + intent (`v4.1_config.json:tactical_policy`;
V4_DESIGN §2.7). It has **no anchor vocabulary** — an unconstrained MLP regressor. Three MEASURED facts
explain the wide swing:

1. **It is a lossy readout.** Its own output scores **3.1501 open-loop vs the same model's rollout 0.4522**
   (MODEL_REGISTRY §5) — 7× worse than the path it reads out. A head that poor, fed ~3.2× OOD reconstruction
   input, **extrapolates off-manifold** → wide swings → offroad 8/12. Anchors clamp outputs to the real-GT
   convex hull; an MLP does not.
2. **Its intent seam is the program's ONLY MEASURED HARMFUL seam** (F3, cos **−0.238**, norm 31.4 vs 28.3;
   V4_DESIGN §2.7 / ARCHITECTURE_WIRING_COMPARISON §2.4). It actively injects a counterproductive signal.
3. **It emits a single deterministic trajectory** — no fan, no selection over candidates, no multimodal
   fallback. Whatever the regressor extrapolates is what ships. REF-C selects the best of 256 anchors.

**Objective / selection are secondary.** L2-to-GT regression normally *shrinks* deviation toward the mean
in-distribution — so the objective is not the wide-swing cause on its own; it is the **interaction of an
unconstrained head with OOD input** (mechanism #1) plus the harmful seam (#2).

**Fixable, or is anchored-diffusion strictly better? Strictly better.** The wide-swing property is intrinsic
to unconstrained regression + a harmful seam; you do not "tune" a regression head to 0.34 deviation. v4
**removes both**: it drops `tactical_policy` entirely (−22,736,141 params) *and* its HARMFUL intent seam,
and replaces it with the anchored vocabulary (V4_DESIGN §2.7). Evidence class MEASURED throughout.

**Honest caveat (resolution):** part of the 1.12 magnitude could be flagship's frozen ViT being more
resolution-sensitive at 480×854 than REF-C's ResNet. But native-res plan_dev is still 0.94 (wide), and the
pairing controls resolution — so the **head-design story survives; resolution is a magnitude modifier, not
the cause.** Exp #1 (§6) settles the residual.

---

## 4. Q3 — v4.1 planner-starvation: ranked fixes + a pre-registered discriminator

**The tension to resolve (MEASURED, §1.2):** with the lowered `lr_trunk`, the WM now needs *protection*;
any planner gradient (λ_plan) that reaches the trunk breaches the canary; the down-only controller then
starves the planner. You cannot get both a healthy WM and a well-trained planner **through the shared trunk
under a down-only controller**. The three fixes attack this differently:

| rank | fix | what it changes | keeps Sayed's "end2end at the same time" charter? | cost (ESTIMATE) |
|---|---|---|---|---|
| **1** | **(iii) planner on a FROZEN healthy WM latent** = `--lambda-plan 0` for the whole run | zero planner→trunk gradient. Trunk trains under WM losses only (keeps improving + stays rollable — *stronger* than REF-C's frozen ResNet); planner head + factorised heads + imagination-**conditioning** train fully on top | **No** — sidesteps joint training. It is the discriminator + the safe floor, not the destination | **~0.5 A40-day** to a 10k read (already priced as the R1 ultimate fallback, V4_DESIGN §11: *"planner-over-frozen-v1, v1.5 with the v4 head"*) |
| 2 | **(i) less-aggressive controller** — cap-and-hold at last healthy λ_plan (what R1 *intended*), or bidirectional recovery, or a λ_plan floor, or threshold +0.10 not +0.05 | accepts mild WM degradation to keep planner gradient alive | **Yes** | ~5 A40-days (controller code change + 30k joint run) |
| 3 | **(ii) hard phase-A freeze → unfreeze with a λ_plan FLOOR** | structured version of (i): trunk frozen through phase A, then a floored ramp | Yes | ~5 A40-days |

**Why (iii) is ranked first — it is both cheapest AND most discriminating.** It is an **existing switch**
(`--lambda-plan 0` = "frozen-trunk v1.5 regime, byte-identical," V4_DESIGN §16), carries **zero WM-collapse
risk**, and it cleanly splits the two live hypotheses that (i)/(ii) cannot separate:

- **Is v4.1's planner bad because it is STARVED, or bad by DESIGN?** Under (iii) the planner gets *full*
  head-gradient (lr_head 1e-4) over a healthy WM latent + imagination-conditioning, with no controller
  strangling it. If it reaches ~v1/REF-C quality → starvation was the culprit → (i)/(ii) are worth their
  ~5 A40-days. If it *still* can't → the planner **design** is the problem and **no controller fix helps** —
  redirect budget. Spending ~5 A40-days on a controller restart *before* this ~0.5-day test inverts the
  cost-of-information.

**Is (iii) the honest MVP? Yes — for the right question.** It is the honest MVP *for "is our planner any good
on a healthy WM"* and for de-risking a ship (it doubles as the R1 fallback AND produces the healthy-canary
checkpoint the closed-loop synthesis §5 AlpaSim test needs — three birds). It is **not** the honest MVP for
the joint-training charter, which it deliberately sidesteps; frame it as a **discriminator and floor**, not
the destination.

### 4.1 Pre-registered discriminating experiment for the top fix — `planner_on_frozen_wm`

- **Setup.** Warm-start trunk from `flagship4b-speedjerk-30k`; `--lambda-plan 0` for all steps; everything
  else v4.1 (dense operative anchored diffusion + factorised LAT×LON×DIST + produced goal + `--probe-grad
  one` imagination-conditioning). Requires building the v4 held-out eval driver (the STATUS_BLOCKED blocker)
  — **which is needed to score ANY v4 variant regardless**, so it is not extra cost attributable to this exp.
- **Read at 10k on the same card** (`Project Steering/Gates/flagship-v4.card.json`): primary `ade_0_2s`
  (held-out, produced-goal, `cluster_bootstrap`), `oracle_in_fan`, `miss_at_2m`. Canary is trivially healthy
  by construction (λ_plan=0) → it is a *control*, not a test.
- **Both outcomes committed IN ADVANCE:**
  - **Planner competitive** (ade_0_2s ≤ ~0.50 AND oracle ≤ 0.30 AND miss ≤ 0.10) ⇒ the planner design is
    **sound**; v4.1's 0.705 was **starvation**. Green-light (i) cap-and-hold to recover the joint charter,
    and ship (iii) as the safe floor if (i) fails.
  - **Planner still weak** (ade_0_2s ≥ ~0.60, or oracle > 0.30, or miss > 0.10, even decoupled) ⇒ the planner
    **design** is the problem, not starvation. **Stop tuning the controller.** The lever moves to the planner
    head/objective and to closed-loop-aware training (Q4). This *refutes* "starvation is the whole story."

---

## 5. Q4 — Closed-loop-aware training: is DAgger / rollout-in-the-loop the real lever?

**The shared root cause (MEASURED).** Both v1-tactical and REF-C are **open-loop-trained**, and both fail
closed-loop differently (v1 offroad; REF-C ~half on OOD recon). **Open-loop ADE does not predict closed-loop**
(REF-C base open-loop ade 0.47 yet ~half-fails; MODEL_REGISTRY §4.4; closed-loop synthesis §1). Imagination
is a real but **weak** lever (−0.213 m re-plan; the strong CEM imagine-and-select form is untested) and the
closed-loop synthesis §6 already pre-commits: if imagination ties REF-C, the lever moves to closed-loop-aware
training. So the hypothesis "DAgger/rollout-in-the-loop is the real lever" is **live and well-motivated**.

**The cheap proving ground exists.** The no-renderer kinematic harness (task #21,
`taniteval/taniteval/closedloop.py`) uses the WM as its own neural simulator — the imagination proof ran in
**32.4 s** on a local 4060 (§1.3). A DAgger fine-tune over it is cheap.

### 5.1 Pre-registered cheapest closed-loop-aware fine-tune — `dagger_planner_ft`

- **Setup.** Take a healthy-WM checkpoint (v1, or the exp-#3 `planner_on_frozen_wm` output). **Freeze the WM**
  (isolate the planner). For N inner rounds: roll the planner closed-loop in the kinematic harness (WM imagines
  the latent under executed actions), **collect the on-policy states it actually visits** (the compounding-error
  distribution open-loop training omits), label each with the GT-nearest anchor / expert waypoint from the true
  future, aggregate, and fine-tune the planner head on the union. Pure DAgger: expert = open-loop GT-nearest
  anchor; states = on-policy closed-loop visited states. ESTIMATE ~0.1–0.3 A40-day (harness eval is seconds;
  the aggregate+finetune loop dominates).
- **Read (pre-registered):** paired closed-loop ade@2s and divergence>5 m of DAgger-FT vs the same planner
  open-loop-trained, held-out val eps, **paired episode-cluster bootstrap** (`taniteval/ci.py`).
- **Both outcomes committed:**
  - **DAgger-FT materially reduces drift** (paired Δ CI-separated, ideally beyond the −0.213 m imagination
    lever) ⇒ closed-loop-aware training **is** the real lever; promote it as a v4 curriculum Phase D and run
    the decision-grade AlpaSim confirmation.
  - **DAgger-FT ties open-loop** (CI includes 0) ⇒ on-policy state coverage is **not** the bottleneck at this
    fidelity; the residual is WM faithfulness under self-rollout or the *selection objective*. Redirect to a
    **consequence-aware objective** (imagined-offroad/collision penalty in selection), and flag that the
    no-renderer harness has hit its ceiling (needs external AlpaSim/photoreal sim to go further).
- **Caveat (binding).** The harness is **self-referential** (WM is both simulator and state estimator) and has
  **no collision/drivable-area** (drift/stability loop only). It can prove/refute the DAgger *mechanism*
  cheaply; the offroad/collision *rate* must be confirmed on AlpaSim (exp #4). Honest MVP: mechanism cheap,
  safety metric expensive.

---

## 6. RANKED experiment table — cheapest-discriminating first, both outcomes pre-committed

| # | experiment | GPU cost (ESTIMATE) | discriminates | outcome A → | outcome B → |
|---|---|---|---|---|---|
| **1** | **native-1080×1920 paired AlpaSim re-run** (flagship v1 vs REF-C base, same 12 scenes) | **~1.5 h ≈ 0.06 A40-day** (lock released, NOTE §caveat 2) | head-design vs frozen-ViT resolution-sensitivity — the one residual confound on §1.1 | flagship plan_dev still ≫ REF-C at native res ⇒ **§1.1/Q2 verdict is clean**, resolution exonerated | plan_dev collapses to ~REF-C ⇒ the loss was resolution-sensitivity, **re-open Q2**; v1-tactical is less indicted |
| **2 ⭐** | **`planner_on_frozen_wm`** (`--lambda-plan 0`, whole run, 10k read) — *highest decision-value* | **~0.5 A40-day** (needs the v4 eval driver, required for any v4 gate regardless) | v4.1 planner: **STARVED vs bad-by-DESIGN** (the central Q3 fork) | ade≤~0.50 & oracle≤0.30 & miss≤0.10 ⇒ **starvation**; green-light controller fix (i); ship (iii) as floor | still weak decoupled ⇒ **planner design**; stop tuning the controller, move to Q4 / head objective |
| **3** | **`dagger_planner_ft`** (no-renderer, frozen WM, on-policy) | **~0.1–0.3 A40-day** | is **closed-loop-aware training** the real lever above open-loop? | paired Δ CI-separated ⇒ promote a curriculum Phase D + AlpaSim confirm | ties open-loop ⇒ coverage isn't the bottleneck; move to consequence-aware objective; harness at its ceiling |
| **4** | **AlpaSim `public_2601` suite on a healthy-canary planner vs REF-C** (decision-grade closed-loop, OOD caveat) | **~0.2 eval-day** (gated on #2/#3 producing a healthy planner) | does imagination-over-a-healthy-WM beat open-loop diffusion **in closed loop**? (closed-loop synthesis §5) | materially lower at-fault/offroad at ≥ matched open-loop ADE ⇒ **v4 hierarchy earns its complexity** | ties REF-C ⇒ imagination-as-implemented doesn't buy closed-loop robustness; lever is #3/objective |
| **5** | **v4.2 cap-and-hold / floored-λ_plan joint 30k** (recover the charter) | **~5 A40-days** (gated on #2 = "starved") | is jointly training WM+planner on this trunk **recoverable**? | reaches ≤0.60 with canary ≤0.55 ⇒ **charter intact**, v4.2 is the arm | canary breaches whenever λ_plan>floor ⇒ approach the honest `REFUTE_LEVER_FAMILY` question for `joint-planner-wm` |

**Order rationale.** Strict cheapest-first among experiments that each discriminate something real; #2 is
flagged as highest decision-value despite #1 being cheaper, because #2 resolves the central Q3 fork and
gates the expensive #5. #4–#5 are dependency-gated on #2/#3.

---

## 7. ESCALATION — a recommendation for Sayed, not a decision I make

**Which of {continue v4.1, restart as v4.2 with a looser controller, change the planner design} does today's
evidence most support?** — **None of the three directly. The evidence supports running the ~0.5 A40-day
`planner_on_frozen_wm` discriminator (exp #2) FIRST, because its outcome pre-commits the fork:**

- **Continue v4.1 unchanged — weakly supported.** Its planner is gradient-starved (MEASURED) and its 10k gate
  is **BLOCKED, not passed** (no eval driver, pod1 unprovisioned). Continuing burns budget on a run whose
  planner is off. At minimum, the **v4 held-out eval driver must be built** — it is the true blocker and is
  needed for every path forward.
- **Restart v4.2 with a looser controller — premature.** We do not yet know whether the planner is good when
  fed gradient (→ controller is the lever) or bad by design (→ controller is irrelevant). Spending ~5 A40-days
  before the ~0.5-day discriminator inverts the cost-of-information (CLAUDE operating standard #5: cheapest
  discriminating experiment, both outcomes pre-committed).
- **Change the planner (operative) design — NOT supported.** The operative planner already **is** REF-C's
  anchored diffusion — our best planner, with the exact tight-deviation behavior we want (Q1). Changing it
  would solve a problem v4 does not have. (The *tactical/regression* head is a different object and v4 already
  removed it.)

**Recommended next step (my recommendation, Sayed decides):** build the v4 eval driver (unblocks the gate on
every path), then run **exp #2 `planner_on_frozen_wm`**. It is the ~0.5 A40-day experiment that tells us
which of the three to do, and it simultaneously (a) yields the R1 ship-fallback and (b) produces the
healthy-canary checkpoint needed for the decision-grade AlpaSim closed-loop test. Then follow the exp-#2
branch pre-committed in §4.1.

**Two integration escalations (do not let these sit):**
1. **Stale live-state headline.** The 2026-07-22 closed-loop synthesis §8 (*"flagship v1 drives collision-free,
   n=1"*) and `LEADERBOARD.md` §5.5 are **reversed** by §1.1 (REF-C base beats flagship v1, pass 8/12 vs 2/12,
   score 0.496 vs 0.066; collisions tied; n=1 was a lucky scene). The NOTE.md already flags a **C5**
   RETRACTION_LOG entry is owed. This needs the registry/leaderboard owner to update — I do not edit those.
2. **v4.1 controller bug worth fixing before any v4.2.** What ran was naive halve-to-zero; the design's R1
   intent was cap-and-hold at last-healthy λ_plan. If v4.2 proceeds, the controller must implement cap-and-hold
   (or a floor) — otherwise it will re-starve the planner identically.

---

## 8. Framing caveats carried through the whole doc (so no reader over-reads)

1. **AlpaSim numbers are WITHIN-SIM RELATIVE and ~3.2× reconstruction-OOD** (RETRACTION_LOG C6). They isolate
   the planner via pairing; they are not real-world rates. "REF-C beats flagship closed-loop" means *on these
   reconstructions, as planners* — not "REF-C is the better driver in the world."
2. **v4.1 step-10000 numbers are in-loop trainer diagnostics, not the gate primary** (C1). The gate is BLOCKED;
   the directional read (planner under-trained) is HYPOTHESIS until the held-out eval runs.
3. **The imagination proof is a different comparison** (re-plan vs single-shot on v1's WM) and the weak form
   (no CEM select); it supports the imagination *thesis*, it does not rank v1's planner.
4. **Every "v4 will…" is ESTIMATED** — v4/v4.1 has no held-out eval yet. The MEASURED anchors are v1-tactical,
   REF-C base, and the v4.1 in-loop curve; the design claims are source-read facts.

---

## Deliverable manifest

| artifact | where | status |
|---|---|---|
| this synthesis | `TanitAD Research Hub/Architecture & Inference/Research/2026-07-23-planner-is-the-bottleneck.md` | **STAGED** (git add, not committed, not pushed) |

**Inputs read (all in-repo, not modified):** `Project Steering/RETRACTION_LOG.md`;
`…/incoming/2026-07-22-alpasim-closedloop-evalpod/flagship_vs_refc_suite_{NOTE.md,results.json}`;
`…/incoming/2026-07-23-v41-10k-gate/{STATUS_BLOCKED.md,v41_step10000_inloop_health.json,v4.1_config.json}`;
`…/incoming/2026-07-22-imagination-closedloop-proof/{README.md,closedloop_flagship-30k_imagination-proof.json}`;
`Architecture & Inference/V4_FLAGSHIP_DESIGN.md`;
`Architecture & Inference/Research/2026-07-22-{closed-loop-robustness-and-imagination,encoder-strategy-and-vjepa2ac}.md`;
`Project Steering/MODEL_REGISTRY.md` §4.2/§4.3/§4.4/§5.

**Escalations (need a decision/owner, not a paragraph):** (1) update closed-loop synthesis §8 + LEADERBOARD
§5.5 for the n=12 reversal + append the owed C5 retraction — registry/leaderboard owner; (2) build the v4
held-out eval driver (true blocker on every path) — eval owner; (3) fix the λ_plan controller to cap-and-hold
before any v4.2 — trainer owner.

**No GPU launched. No pod written to. No commit, no push.**
