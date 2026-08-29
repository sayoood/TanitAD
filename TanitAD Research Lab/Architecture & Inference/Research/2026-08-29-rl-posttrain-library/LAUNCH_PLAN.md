# LAUNCH PLAN — RL post-training of the refcv3 planner (D-RL-REFCV3)

`Owner: TanitAD_TrainingFlyWheel · 2026-08-29 · PREREG-READY, NOT LAUNCHED.
Library: stack/tanitad/rl/ (85 tests green). Commission: PI via Master Mind —
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

#### What is kept anyway, on design merit rather than on my false premise
* **Graded asymmetric headway** — the old shape saturates at 1.0 for every gap
  ≥ T*, so it cannot tell a 2 s gap from a 6 s one and carries **no dawdling
  signal**, which the four-families longitudinal rule exists to protect. The
  graded shape ranks both sides of T*, with tailgating penalised exactly 3×
  dawdling (analytic, pinned by test).
* **TTC veto separated from the ranking term** — a constraint pins a candidate,
  a ranking signal orders them; fusing them produces something that does neither.

⚠️ **Second, milder finding (stands):** `progress` has spread **1.5000**,
exactly its clamp width, mean 1.1478 — saturating at its `hi` bound, compressing
ranking among the fastest candidates. Not blocking; recorded.

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

## 4. Cost

⚠️ **`UNVERIFIED` — refcv3 post-training step time has never been measured**, and
I will not invent one. Step 0 of A1 is to log `step_s_interval` for 200 steps and
put the real number here. What IS known:

| fact | value | class |
|---|---|---|
| RL post-training is a **fine-tune from a cold start**, not a from-scratch run | DDv2 used **10 RL epochs** on 8×L20 after full IL pretraining | PUBLISHED |
| our comparable from-scratch run (v7-tiny scale1, 23.87 M, 30k steps, Thor) | **16,323.9 s = 4.5 h**, `step_s_interval` ≈ 0.55 s | MEASURED (ours) |
| the reward adds a per-candidate geometric pass over `[B, N, G, S, 2]` | pure tensor ops, no model forward — expected small vs the decoder | ESTIMATED |
| `group_size` multiplies decoder sampling cost by G | G=4 default | — |

**Machines.** ⛔ **Thor is off-limits while the EMA arm runs** (`emao14_30k`,
PID 1640883, 6h54m elapsed at 16:20 CEST) and the B1 epcache build has the box
overnight. ⇒ **A0 and the A2 regression arm run on the dev-box RTX 4060** (they
are small); **A1/A3 wait for Thor**, or run on the 4060 at reduced `--size` as a
mechanism check that is explicitly NOT the headline number.
⛔ 4060 only when `nvidia-smi` shows no python compute.

---

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
