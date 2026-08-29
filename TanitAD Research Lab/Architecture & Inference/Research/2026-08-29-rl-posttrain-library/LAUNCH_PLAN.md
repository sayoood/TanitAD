# LAUNCH PLAN — RL post-training of the refcv3 planner (D-RL-REFCV3)

`Owner: TanitAD_TrainingFlyWheel · 2026-08-29 · PREREG-READY, NOT LAUNCHED.
Library: stack/tanitad/rl/ (111 tests green). Commission: PI via Master Mind —
"develop a library of RL methods to post-train the planner of refcv3 by RL …
implement it and validate it, to be prepared for training."`

⛔ **NOTHING IN THIS PLAN HAS RUN.** Every number below is `ESTIMATED` or
`UNVERIFIED` unless it carries `MEASURED (ours)`. The plan's whole purpose is
that a launch decision is made against committed outcomes rather than against a
result someone already saw.

---

## 0. ⛔ THE ARM THAT MUST RUN FIRST — AND IT IS NOT A TRAINING RUN

**A0 — REWARD COVERAGE ON REAL refcv3 WINDOWS. 0 training steps.**

Our own audit already refuses to bless the reward without it. Running
`audit_reward` with no scene context returns **INCONCLUSIVE**, naming
`collision`, `headway` and `gt_similarity` as CONSTANT across the whole
degenerate panel — because with no obstacles, no lead vehicle and no GT in the
context those components are identically zero. *(MEASURED 2026-08-29, and it is
why `AuditReport` has three states rather than two.)*

⇒ **A reward whose safety terms never fire is not a safe reward — it is an
absent measurement wearing safety's name.** Before any RL step:

```
for each real refcv3 window batch:
    report_component_coverage(spec, fan, ctx)   ->  fired / spread / n
```

| outcome | what it means | what we do |
|---|---|---|
| all six components fire with non-zero spread over the fan | the reward can rank candidates | proceed to A1 |
| `collision` / `headway` never fire (no obstacle or lead join on these windows) | the reward is effectively `progress + feasibility + comfort + gt_similarity` — i.e. **much closer to the hackable arm than intended** | ⛔ **DO NOT LAUNCH A1.** Fix the scene join first (`obstacle.offline`, `build_obstacle_join.py`) — this becomes a DataFlyWheel dependency, escalated, not worked around |
| components fire but with negligible spread across the fan | the fan's candidates are too similar for a group-relative advantage to have signal | raise exploration `noise_scale`, or the fan is degenerate — report either way |

**Cost:** minutes, CPU or 4060. **This is the cheapest possible way to discover
the campaign is not fundable, and it costs no GPU-hours to find out.**

### ✅ A0 HAS RUN — 2026-08-29 — **PASS**. (An earlier FAIL is RETRACTED: TRAIN-C2.)

`raw/a0_coverage.json` · instrument `stack/scripts/rl_a0_coverage.py` ·
**240 windows** over the **held-out** `physicalai-val130-heldout` corpus (129
episodes, 124 carrying the `obstacle.offline` join) · 184 windows with obstacles,
68 with a lead · fan = the anchor vocabulary (checkpoint-independent; refcv3 has
no trained checkpoint yet). MEASURED (ours).

| component | weight | applies | spread \| applicable | spread \| ALL | mean |
|---|---|---|---|---|---|
| feasibility | 0.50 | 100 % | 1.0000 | 1.0000 | 0.9783 |
| comfort | 0.20 | 100 % | 0.9896 | 0.9896 | 0.4172 |
| progress | 0.30 | 100 % | 1.5000 | 1.5000 | 1.1478 |
| collision | 1.00 | 54.2 % | 1.0000 | 1.0000 | −0.1902 |
| headway | 0.30 | 28.3 % | **0.9209** | 0.0000 | 0.7454 |

**VERDICT: PASS — every weighted component ranks the fan wherever it applies.**
⚠️ `headway` (28.3 %) and `collision` (54.2 %) apply on fewer than all windows.
That is the **BASE RATE of the situations they score** — most windows have no lead
vehicle and open road has no obstacle in the corridor — not a defect. It is
recorded here so absence is never later read as safety.

#### ⛔ RETRACTED: the first A0 run reported FAIL, and it was MY METRIC, not the reward

The first version of this probe pooled the spread over **all 240 windows**.
`headway` is UNDEFINED where there is no lead (72 % of them) and returns a
constant there by design, so the pooled median read **0.0000** and I reported the
component INERT. Measured A/B on the identical 68 lead-present windows
(`code/ab_headway.py`):

| headway shape | ranks on | median spread when ranking |
|---|---|---|
| OLD (saturating) | **68/68 = 100 %** | **1.0000** |
| NEW (graded) | 68/68 = 100 % | 0.9209 |

⇒ **The component was never inert, and the redesign did not fix an inertness —
it slightly narrowed the raw-fan spread.** Full entry: `RETRACTION_LOG.md`
**TRAIN-C2**. ⭐ *A statistic computed over a population where the quantity is
undefined is not a weak measurement; it is a different measurement.*

#### ✅ DECIDED (Master Mind, 2026-08-29): GRADED STAYS — on independent merit

Ruling, verbatim in substance: *the old saturating shape genuinely carries no
ordering above T*, and ordering both regimes is worth 8 % raw spread — spread was
never the goal, INFORMATIVE ORDERING is.* ⭐ That is the pooled-median lesson
applied one level up: a wider spread over a range where the term cannot order
anything is not more signal. The TTC-veto separation stands.

#### Why it is kept — design merit, NOT my retracted premise
* **Graded asymmetric headway** — the old shape saturates at 1.0 for every gap
  ≥ T*, so it cannot tell a 2 s gap from a 6 s one and carries **no dawdling
  signal**, which the four-families longitudinal rule exists to protect. The
  graded shape ranks both sides of T*, with tailgating penalised exactly 3×
  dawdling (analytic, pinned by test).
* **TTC veto separated from the ranking term** — a constraint pins a candidate,
  a ranking signal orders them; fusing them produces something that does neither.

#### ✅ `progress` — PER-WINDOW REFERENCE ADOPTED AND MEASURED (2026-08-29)

Decision (Master Mind): reference = `max(v0 × horizon, 5 m)`, where `v0` is the
window's own current ego speed — inference-admissible ego state, ⛔ **never
fan-derived and never expert-derived** (either would be the tune-on-what-you-score
class). Semantics: **1.0 = "kept the current speed"**, and that zero point is a
driving fact rather than an arbitrary metre count.

**The prediction was: saturation disappears on fast windows and appears honestly
on stopped ones. MEASURED, 240 windows, fraction of the fan pinned at the clamp:**

| v0 band | n | clamped, PER-WINDOW ref | clamped, OLD fixed 30 m ref |
|---|---|---|---|
| stopped < 2 m/s | 39 | 95.3 % | 37.5 % |
| slow 2–8 | 81 | 89.3 % | 37.5 % |
| mid 8–15 | 82 | 63.8 % | 37.5 % |
| **fast > 15** | 38 | **19.9 %** | 37.5 % |

⭐ Confirmed on both halves: saturation on fast windows nearly **halves**
(37.5 % → 19.9 %), and rises on stopped ones where the quantity is genuinely
ill-posed. ⭐ **The tell is the OLD column: 37.5 % in EVERY band.** A fixed
reference is *blind to the window* — it applies one yardstick to a stopped car
and a highway cruise, which is precisely the speed prior the change removes.

⚠️ **The pooled mean moved the WRONG WAY and would have misled** (1.1478 →
1.3192), because this corpus is dominated by slow windows (median v0 **8.1 m/s**,
p10 0.7, p90 17.3) where saturation is now intentionally high. That is TRAIN-C2's
lesson recurring within a day: **a pooled average over a heterogeneous population
answers a different question than the one asked.** The band split is the answer.

⚠️ **HONEST RESIDUAL:** at 95.3 % clamped, `progress` carries almost no ranking
on stopped windows — "weakly" is generous. There the reward is effectively
`feasibility + comfort + collision`. Defensible (maintaining 0.7 m/s is not a
goal worth ranking toward) but it must not be discovered later: **on near-stopped
windows the progress term is inert by construction**, and A1's per-band reporting
should carry it.

⚠️ **AND THIS IS AN UPPER BOUND ON SATURATION.** The fan here is the raw anchor
vocabulary, which samples 0–30 m/s uniformly, so most candidates are implausible
for any given window. A trained decoder's fan concentrates near plausible speeds
⇒ real saturation will be LOWER in every band. Re-measure once a cold start exists.

---

## 0c. ⛔ A1 IS BLOCKED, AND NOT ON THE EMA READ — refcv3 HAS NO IL COLD START

**Verified by content 2026-08-29** (two probes, per the absence rule):
* `Project Steering/MODEL_REGISTRY.md` — **no refcv3 row.** Its REF-C rows are the
  older v2.1 lineage (`refc-diffusion-{small,base,xl}-v21-30k`), not v3.
* **No refcv3 checkpoint on disk** anywhere in the repo or the session scratchpads.
  The nearest lineage artifact, REF-C-XL's `ckpt.pt` (md5
  `966d4eff1ea5ddf86efba01b8344e198`), is recorded at
  `tanitad-eval:/root/models/refc-xl-30k/` — and **`tanitad-eval` is terminated**.

A1 is *RL post-training from an IL cold start* (DDv2's protocol, and the whole
point of the method). **There is no cold start to post-train from.** Running the
loop from a random init would not be A1; it would be RL-from-scratch, which is
neither what DDv2 did nor what the commission asked for, and any number from it
would be uninterpretable.

⇒ **Sequencing: B1 corpus → refcv3 IL cold start → A1.** The IL arm is the
prerequisite, it is owned outside this FlyWheel, and its cost dominates §4.

**What is runnable NOW, and is deliberately NOT an arm:** a MECHANISM smoke of the
full loop on a randomly-initialised refcv3 at real scale — it proves the pipeline
runs end-to-end, exercises the veto and both advantage halves on real windows, and
produced the §4 cost numbers. ⛔ It is labelled a mechanism check and may never be
quoted as an A1 result.

---

## 1. The arms

All arms: refcv3, **frozen trunk** (`freeze_trunk=True`), planner head only,
IL cold start from the existing refcv3 checkpoint, parity corpus
`physicalai-train-e438721ae894` (skip-hash `f09e44db`), seed 0.

| arm | method | reward | purpose |
|---|---|---|---|
| **A0** | — | — | coverage probe (§0). Gate for everything below |
| **A1** | `grpo` | `DEFAULT_WEIGHTS` | ⭐ the headline arm: DDv2's intra-anchor GRPO + inter-anchor truncated advantage |
| **A2** | `grpo` | `HACKABLE_WEIGHTS` (progress-only) | ⛔ **DELIBERATE REGRESSION.** The audit must FLAG it, and the trained policy must visibly degrade (drive faster, collide more). If A2 looks *good*, our metrics are measuring the wrong thing |
| **A3** | `awr` | `DEFAULT_WEIGHTS` | advantage-weighted regression: no new module, no state-dict key. The cheap alternative that must be beaten before A1's complexity is justified |
| **A4** | `grpo`, `freeze_trunk=False` | `DEFAULT_WEIGHTS` | ⚠️ **CONTROL ONLY, run last if at all.** Tests whether the frozen-trunk constraint costs anything. A trunk that moves under a rule-based reward invalidates every representation number measured on it |

⛔ **`dpo` IS NOT AN ARM.** `PostTrainConfig(method="dpo").validate()` raises.
A demonstration corpus carries one policy's output per state — there is no
negative class. The manufacturable negative (thresholding cardinal GT distance
into a binary) was MEASURED to cost **+0.0974 m (base) / +0.1670 m (XL),
separated**. DPO becomes available only when a preference-collection instrument
exists (P4 backlog **M-B4**).

---

## 2. Committed outcomes — written before any arm runs

### A1 (the headline)
| result | reading | what we do |
|---|---|---|
| A1 beats the IL cold start on the four families **and** raw-fan top-10 quality improves | the DDv2 transfer works on our stack | promote to a registry row; plan the scaled run |
| A1 improves the **selected** trajectory but the raw fan is unchanged | we trained the selector's taste, not the generator — ⛔ **the exact defect DDv2 exists to fix** (raw-fan PDMS 93.5→75.3 top-1→top-10) | REJECT the arm. Report the fan floor, not the top-1 |
| A1 improves ADE but degrades collision/headway | reward hacking that our composition did not prevent | REJECT, and add the winning degenerate to the audit panel — the panel is fixed and finite, so every real hack found must be added to it |
| A1 flat vs cold start | the rule-based reward carries too little signal at this fan diversity | report; do not scale. Next lever is the reward, not more steps |

### A2 (the regression arm)
| result | reading |
|---|---|
| audit FLAGS it **and** trained behaviour degrades | ✅ the instrument chain works end-to-end. **This is the pass condition** |
| audit flags it but behaviour does not degrade | our behaviour metrics cannot see the failure the audit predicts — ⛔ **the metrics are the problem**, and A1's result is not trustworthy either |
| audit does NOT flag it | ⛔ the audit is broken; every clean verdict it has produced is void |

### A3 vs A1
If **A3 (AWR) matches A1**, ship A3: it adds no module, no state-dict key and no
`STAGE_MAY_INTRODUCE` entry. Complexity must earn its place.

---

## 3. Preflights (all must pass; each has a failure that earned it)

| check | why |
|---|---|
| `PostTrainConfig.validate()` | refuses `dpo`, `group_size<2`, bad `normalize`/`noise_mode` |
| `select_trainable()` report shows `frozen_params > 0` **and** `trainable_params > 0` | an optimiser over an empty parameter list steps happily and changes nothing |
| `assert_selector_disjoint(ctx)` on the real context | ⛔ the winner's-curse firewall. The reward may never be the selector's own score |
| `audit_reward(spec, real_ctx).verdict == "clean"` | an INCONCLUSIVE audit means the reward cannot be exercised — see §0 |
| A0 coverage: every weighted component `fired: true` | a component that never fires cancels in the group-relative advantage |
| md5 code-freshness on the target machine | Thor has no git credentials; ship files and verify by content |
| `git rev-parse HEAD` + dirty flag recorded | P4 SPEC R3 — run provenance |

---

## 4. Cost — MEASURED 2026-08-29, and it is not the bottleneck

⭐ **RL post-training is compute-TRIVIAL next to the IL cold start it needs.**
Measured on the real-scale model (`refc_v3_sized_config('small')`, **62.94 M
params**, `core.decoder` trainable **9.06 M = 14.4 %**) on the dev-box RTX 4060,
fan `[B=2, N=128, G=4]`:

| quantity | value | class |
|---|---|---|
| step time | **0.054 s/step** | MEASURED (ours) |
| 1,000 steps | **0.9 min** | derived |
| 10,000 steps (DDv2 used 10 RL epochs) | **~0.2 h** | derived |
| peak GPU | **0.56 GB** of 8.19 GB | MEASURED (ours) |

⚠️ **STATED LIMIT — this excludes DATA LOADING.** Frames were synthetic; the real
loop adds PNG/JPEG decode per window, which on this stack has no DataLoader
workers and is the suspected wall-clock driver elsewhere (P4-6). ⇒ Treat 0.054 s
as the COMPUTE floor, not the run time. The honest reading is a *ratio*: the RL
update is cheap enough that the campaign's cost is dominated by (a) the cold
start and (b) data loading — not by the policy gradient.

⛔ **THE REAL BLOCKER IS NOT COMPUTE.** See §0c.

**Machines.** ⛔ Thor is off-limits (EMA arm, then the B1 epcache build overnight).
The **dev-box RTX 4060 is sufficient for every arm in this plan** — 0.56 GB peak
leaves the box free — and is to be used only when `nvidia-smi` shows no python
compute.

## 5. What this plan deliberately does not claim

* **Everything here is T0 training-side.** A rule-based reward improving is not
  a driving result. A capability claim needs **T1** (`EVAL_DOCTRINE.md`), and any
  eval reports **all four metric families**, never ADE alone.
* **The audit panel is fixed and finite.** "No degenerate policy beats the
  reference" is *necessary, not sufficient* — it bounds the claim to the five
  policies in the panel.
* **A linear reading of a reward is not a capability.** The reward ranks
  candidates; it does not tell us the car drives.
* ⚠️ **The published +3.1 PDMS is DDv2's, on NAVSIM, with a PDMS oracle in the
  loop.** We have no PDMS oracle. Our rule-based composition is a *proxy for a
  proxy*, and the transfer is a hypothesis this plan tests — not a result we
  inherit.
