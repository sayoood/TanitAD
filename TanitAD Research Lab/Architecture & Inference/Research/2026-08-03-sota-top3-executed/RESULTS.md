# SOTA_SCAN §11 items 1–3, EXECUTED — and the one that matters came out the other way

**Stream E · 2026-08-03 · 0 GPU · $0 · dev-box CPU + one 90-second Thor probe.**
Pre-registration: `PRE_REGISTRATION.md` (written before any runner executed).
Evidence class on every number below: **MEASURED (ours)** unless tagged `PUBLISHED` / `INHERITED`.

---

## 0. The one-paragraph verdict

The scan's three 0-GPU items are done. Item 1 was **already built by a sibling stream hours earlier**
and I verified it rather than rebuilt it. Item 2 was executed in full and **its headline proposal was
refuted on our own data**: adding `progress-ratio` to the LONGITUDINAL family is *admissible* (it
discriminates, and it now ships) but it must **not** become a gate emitter — its partial correlation
with closed-loop failure **flips sign between two sibling arms**, both separated. What the same
experiment found instead is stronger than what it was looking for: on 881 windows / 40 episodes, with
the paired episode-cluster bootstrap, **an arm's open-loop ADE carries almost no information about
its own closed-loop failure once window difficulty is controlled (separated on 1 of 4 arms), while
its LATERAL cross-track error does — separated on 4 of 4.** A hold-`v0` baseline that knows nothing
about the model predicts the model's closed-loop failure *better than the model's own ADE does* on 3
of 4 arms. Item 3 shipped the ego-frame lane-graph raster and it **passed** its discrimination
control (route readout 0.9294 vs a 0.7509 majority) with the shuffled-raster negative control
collapsing as required — but on **one scene**, so it is DIRECTIONAL, exactly as pre-registered.

---

## 1. Item 1 — `obstacle.offline` lead-agent ingest: **NOT MINE TO BUILD. VERIFIED INSTEAD.**

**Branch taken: V-PASS** (`PRE_REGISTRATION.md` §1).

A sibling stream published `Research/2026-08-03-longitudinal-distance-keeping.md` earlier today with
`taniteval/lead_metrics.py` **landed**. Duplicating it would have been waste; quoting it would have
been `INHERITED` on a claim that decides a gate. So I verified it:

| check | result |
|---|---|
| `pytest tests/test_lead_metrics.py` | **14 passed** |
| D-LEAD-1 numbers vs the artifact JSON, not the prose | Δ min-TTC **+1.7474** [1.5813, 1.9218] · Δ headway **+0.9769** [0.8830, 1.0758] · Δ time-gap **+0.1641** [0.1499, 0.1786] — **reproduce exactly** from `…/incoming/2026-08-03-longitudinal-distance-keeping/raw/dlead1_discrimination.json` |
| coverage in the artifact | 14,027 paired windows / 1,431 clusters, `paired_episode_cluster_bootstrap`, B=2000, seed 0 |

⇒ **the instrument is real and its published numbers are not prose-drift.** I did not touch a line of
their code.

### 1.1 The V-GAP, measured rather than repeated

Their own §6 names the gap: arm evals still report the family UNAVAILABLE because `win["lead"]` is
not built for the 40 val episodes. I measured **where** that gap actually is, which their note could
not:

* the canonical clean val cache **`physicalai-val-0c5f7dac3b11` is not on this disk** — two probes:
  `C:/Users/Admin/tanitad-data/physicalai/_epcache/` holds only `physicalai-val-bb543bdf7836`
  (100 episodes), and `…/tanitad-data/eval/` holds no PhysicalAI val at all;
* the cached episodes carry `episode_id` as an **int hash** (e.g. `808608309`), not a clip id, so the
  local `obstacle.offline` chunks (26 on this disk) **cannot be joined to the val-40 without the
  split manifest**;
* **`tanitad-eval` is dead** — `Connection refused` on my probe, and independently on theirs. RunPod
  has reassigned that IP to `tanitad-new` on a different port.

⇒ **P0 L1 is not a scripting task, it is a data-availability task**: it needs the val-40 split
manifest plus the matching `obstacle.offline` chunks on a live host. Stated here so the next agent
does not rediscover it by writing a runner that cannot find its input.

---

## 2. Item 2 — progress-ratio, four families over the bank, and open→closed predictivity

### 2.0 ⚠️ The scan's premise was partly already true — probed before writing code

The scan says progress-ratio *"is a few lines on the same tensors"*. Before writing those lines I
probed (CLAUDE.md rule 2) and found **the ratio already exists twice**:

| where | what it is | why it was still not enough |
|---|---|---|
| `taniteval/pseudosim.py` | an **audited, versioned** `ego_progress` PDM sub-score (`progress_ratios_per_step`, `PROGRESS_TERMS`, `PROGRESS_HUMAN_MIN_M = 0.5`) — and it **already cites 2605.00066's ρ = 0.83 / −0.36** | a bounded *scoring term*, not a reportable metric; and it uses the **t0-axis convention** (§2.1) |
| `taniteval/driving.py` | `progress_abs_err_m` = `|arclength(pred) − arclength(gt)|` | metres, unprojected, unnormalised — not a ratio |
| `taniteval/four_families.longitudinal` | **neither** | this is the actual gap |

⇒ what I added is the **reporting form** of a quantity the programme already owned, reusing
pseudosim's constant verbatim (a test pins the two equal so a future edit cannot fork the
definition). That is a smaller delta than the scan implies, and saying so is part of the result.

### 2.1 ⭐ A convention defect found on the way in, and its size measured

`pseudosim`'s published reading is `ratio = plan_x_in_t0_frame / human_chord`. Therefore **the human
scores `cos θ` against itself**, where θ is the angle between its own chord and the t0 heading — it
is charged for under-progress it did not commit, and *an arm that drives straighter than the human is
rewarded*. `tests/test_progress.py` pins this at θ ∈ {0, 15, 30, 45}° and pins the consequence: a
straight arm scores a perfect 1.0 on a 30° window where the human scores 0.866.

⚠️ **Magnitude on our corpus, so this is calibrated rather than alarming: `t0_axis_gt_self_ratio` =
0.9919** over the 850 scored val windows — a **0.81 % mean** shortfall on 2 s windows. **Real,
provable, and small here**; it grows with window curvature and horizon. My module uses the
`human_dir` projection instead, under which **GT scores exactly 1.0 by construction** — the property
the whole GT-vs-CV control depends on. ⛔ Do not compare a `human_dir` number to a published PSS
`t0_axis` number.

### 2.2 Controls FIRST (the order the standard requires)

| control | result | verdict |
|---|---|---|
| **D-PROG-2** self-consistency: module vs an independent numpy reducer sharing no code | `max|Δ| = 4.44e-16` over 850 windows, nan-masks identical | **PASS** |
| **D-PROG-1** discrimination: CV `progress_error` − GT `progress_error`, paired episode-cluster bootstrap | **+0.1076 [0.0686, 0.1525]**, p(Δ>0) = 1.0, 850 windows / **40 episodes**, 31 excluded for a stopped human | **PASS — ADMISSIBLE** |

GT's own progress error is **0.000000** (as it must be under `human_dir`); hold-`v0` CV's is 0.1076.

### 2.3 The four families over the banked bank — and what is still missing

**27 banked arms** loaded from `taniteval/results/windows_*.pt` (881 windows / 40 episodes each; two
88-window smoke arms). **22 are cross-comparable** — identical `gt` *and* identical `eid` to the v1
reference. ⚠️ Three arms (`flagship-v16-ab-ft`, `flagship-v4.1-10k`, `flagship-v4.2-step4000`) have
**identical `gt` but a different `eid` ordering**, so they were excluded from the ranking rather than
silently ranked; two smoke arms have a different `gt` entirely.

Selected rows (full table in `raw/item2_progress_and_predictivity.json`), each with an
episode-cluster bootstrap:

| arm | `ade_0_2s` | `progress_error` | ratio | under-progress rate |
|---|---:|---:|---:|---:|
| flagship-30k (**v1**) | 0.4271 | 0.0663 | 0.9970 | 0.293 |
| refc-xl-30k | 0.4714 | 0.0918 | 1.0134 | 0.536 |
| refc-base-30k | 0.4728 | 0.0964 | 1.0119 | 0.486 |
| **flagship-speed** | **0.6152** | **0.0893** | 0.9912 | 0.308 |
| refb-v2-30k | 0.5913 | 0.1252 | 1.0175 | 0.460 |
| flagship-nospeed | 3.0175 | 0.3603 | 1.0331 | 0.434 |
| refa-dynin-30k | 3.0471 | 0.4453 | **1.3086** | 0.451 |
| flagship-v2-6k | 5.9396 | 0.5785 | 1.3399 | 0.236 |

⭐ **`flagship-speed` is the rank disagreement**: 30 % *worse* ADE than the REF-C arms while
*better* on progress. ⭐ **REF-A's 1.3086 ratio** — it over-drives by 31 % — is the speed-blindness
the programme already knew about, now visible as a single scalar instead of an inference.

⛔ **Binding-rule compliance, stated plainly: TACTICAL and STRATEGIC are `UNAVAILABLE` on all 27
arms**, because these dumps carry waypoints only and a fidelity pass does not traverse the hierarchy.
`_complete = false` for every arm. **That is a work item, not a pass** — it needs a hierarchy-traversing
re-eval, which is not 0-GPU and is therefore out of this item's scope. The LONGITUDINAL family is also
incomplete pending item 1's `win["lead"]` (§1.1).

### 2.4 E2c — the scan's registered ranking test

**Kendall τ_b = 0.7991** over the 22 comparable arms (Spearman ρ of the same ranking 0.9209).

That lands in the **0.7 ≤ τ ≤ 0.9 band the scan left unspecified and my pre-registration closed in
advance**: ⇒ **INDETERMINATE on ranking alone. No promotion on 2c.** The decision goes to E2d, which
is the question the ranking was only ever a proxy for.

⚠️ **Two caveats on that τ, both MEASURED, both inflating it.** (a) The arms are **not independent** —
many are checkpoints of one lineage, registered in advance as unable to promote a metric alone.
(b) Hashing every dump's predictions shows **26 unique prediction sets among 27 files**: `refa-dynin-30k`
and `overfit_refa-dynin-30k` are **bit-identical**, so one of those two names is wrong or a dump was
duplicated. It contributes one spurious perfectly-concordant pair to τ. *(Flagged for whoever owns the
bank; I did not rename anything.)*

### 2.5 ⭐⭐ E2d — the decision-grade test, and every registered branch failed to fire

Three arms plus the imagination-proof arm carry **per-window open-loop AND per-window closed-loop
trajectories on the same windows** (`…/2026-07-26-closedloop-artifact-rerun/raw_windows/clwin_*.pt`).
So "does the open-loop metric predict closed-loop failure" is askable **within one model, at n =
hundreds of windows, with an episode-cluster bootstrap on ρ itself** — rather than as a rank
correlation over 8 published scalars.

Spearman ρ(open-loop metric, closed-loop ADE@2s), `open_grnd` predictor, `*` = CI excludes 0:

| metric | flagship-30k | flagship-nospeed | flagship-speed | imagination-proof |
|---|---|---|---|---|
| `ade_0_2s` | +0.168 [0.036, 0.297]\* | +0.397\* | +0.147 [−0.001, 0.282] | +0.050 [−0.134, 0.218] |
| `progress_error` | +0.021 [−0.082, 0.123] | +0.317\* | +0.029 [−0.092, 0.150] | +0.037 [−0.129, 0.198] |
| **`cross_mae`** | **+0.233 [0.096, 0.366]\*** | **+0.339\*** | **+0.232 [0.085, 0.367]\*** | **+0.320 [0.102, 0.479]\*** |
| `heading_mae` | +0.110 | +0.234\* | +0.175\* | +0.111 |
| `yaw_rate_mae` | +0.150 | +0.256\* | +0.148 | +0.252 |
| `speed_bias_abs` | +0.093 | +0.382\* | +0.085 | +0.053 |

**Adjudication against `PRE_REGISTRATION.md` §2d, honestly:**

* **"PASS for progress"** required ρ(progress) separated **and** above ρ(ADE) on ≥ 2 of 3 arms.
  It is **below** ρ(ADE) on **all four** arms and separated on **one**. → **does not fire.**
* **"PASS for ADE"** required ρ(ADE) separated **and the largest**. It is separated on 2 of 4 and is
  **not the largest on any arm**. → **does not fire.**
* **"BOTH FAIL"** required *no* metric to separate. `cross_mae` separates on **4 of 4**. → **does not fire.**

⇒ **My registered branches are exhausted without a match, and I am recording that rather than
retro-fitting a branch to the data.** The observation that the LATERAL family is the universal
predictor is a **HYPOTHESIS generated by this run**; it needs its own pre-registered replication on
arms I did not look at first.

### 2.6 ⭐⭐⭐ E2e — the confound control E2d's own output demanded (POST-HOC, labelled)

The `cv` row of E2d forced a question I had not registered: **the hold-`v0` baseline's open-loop
error predicts the MODEL's closed-loop failure at ρ = 0.363 / 0.304 / 0.365 / 0.438 — higher than the
model's own ADE on 3 of 4 arms.** A baseline that knows nothing about the model cannot be measuring
the model. It is measuring **window difficulty**.

So: rank partial correlation, residualising both predictor and outcome on the difficulty proxy, same
episode-cluster bootstrap on the statistic.

| partial ρ given difficulty | flagship-30k | flagship-nospeed | flagship-speed | imagination-proof |
|---|---|---|---|---|
| `ade_0_2s` | +0.098 [−0.05, 0.22] | +0.429 [0.32, 0.51]\* | +0.065 [−0.09, 0.19] | +0.032 [−0.22, 0.21] |
| `progress_error` | −0.106 [−0.21, 0.01] | **+0.295 [0.18, 0.41]\*** | **−0.126 [−0.23, −0.01]\*** | −0.110 [−0.22, 0.07] |
| **`cross_mae`** | **+0.163 [0.04, 0.28]\*** | **+0.254 [0.12, 0.37]\*** | **+0.152 [0.02, 0.28]\*** | **+0.231 [0.02, 0.38]\*** |
| `heading_mae` | +0.027 | +0.145 | +0.102 | −0.033 |
| `yaw_rate_mae` | +0.043 | +0.159\* | +0.039 | +0.066 |
| `speed_bias_abs` | +0.035 | +0.414\* | −0.010 | +0.043 |

Three readings, in order of how much they should change what we do:

1. ⛔ **`progress_error` FLIPS SIGN between two sibling arms, both separated**: **+0.295** on
   `flagship-nospeed` and **−0.126** on `flagship-speed`. A gate emitter whose sign depends on which
   arm you point it at is not a gate emitter. ⇒ **the scan's item-2 recommendation is REFUTED as a
   gate proposal**, while the metric itself stays admissible as a *reported* LONGITUDINAL scalar.
2. **ADE's partial ρ separates on 1 of 4 arms.** On our corpus, at window level, once you account for
   the fact that hard windows are hard for everyone, **an arm's open-loop ADE says almost nothing
   about its own closed-loop failure.** That is our own, within-model, CI-bearing corroboration of
   2605.00066's direction — and it is stronger evidence than theirs (§4).
3. ⭐ **`cross_mae` separates on 4 of 4, after the control.** The LATERAL family is where closed-loop
   failure is predictable from open-loop scoring. **This converges with the independent Thor
   closed-loop result** — REF-C beats flagship v1 with the separation *entirely lateral* while ADE is
   not separated (`INHERITED`, brief STATE). Two different instruments, same answer.

⚠️ `flagship-nospeed` is the outlier where everything correlates (ADE +0.429, speed +0.414). It is
the arm with a 3.0175 ADE — when an arm is broken enough, every metric tracks the breakage. **That is
an argument against selecting metrics on broken arms**, and it is why the near-converged arms
(`flagship-30k`, `flagship-speed`) carry the signal here.

---

## 3. Item 3 — ego-frame lane-graph raster from `map.xodr`

`taniteval/taniteval/lane_raster.py` renders the OpenDRIVE lane graph into an ego-frame BEV:
**[3, 128, 128]** — `lane_presence`, `lane_dir_cos`, `lane_dir_sin` — over −8…+56 m forward and
±32 m lateral at 0.5 m/px, in the same `x` forward / `y` left metres as a waypoint, so a pixel and a
metre mean the same thing without a second transform. **18–24 ms/frame on dev-box CPU**, 299 poses,
mean lane occupancy 0.0520 (min 0.0312, max 0.0933), **0 empty frames**.

Encoding the lane **heading relative to the ego** is what makes it a *directed* graph: a lane going
the other way is not a route option, and a presence-only raster cannot say so.

### 3.1 Controls

| control | result |
|---|---|
| **D-MAP-1** raster-only route readout (right/straight/left at 3 s), logistic regression on 8×8-pooled channels, **8 contiguous time blocks** (never a random split — 10 Hz neighbours would leak) | **0.9294 [0.8513, 0.9889]** vs majority **0.7509**, margin **+0.1785**, lower bound above the baseline → **PASS** |
| **D-MAP-2** the same readout on a shuffled raster | **0.6357**, i.e. *below* the majority rate → **PASS**: D-MAP-1 is reading the map, not a constant |
| **D-MAP-3** anticipation stratum (frames where the ego is **not yet turning**, \|Δyaw\| < 1° over 0.5 s) | **NOT-APPLICABLE** — the stratum holds **2** turn examples. Under-powered, **not** negative |

⚠️ **D-MAP-3 caught a defect in my own first adjudication.** The first version scored the stratum and
returned "FAIL — the raster only reports a turn already in progress" against a **98.88 % majority
baseline built from 2 turn rows**. That would have published a false negative about the map. The gate
is now explicit: minority class < 20 ⇒ NOT-APPLICABLE with its n, never a verdict. *(A second, silent
defect surfaced in the same block: my reducer's `n` key overwrote the stratum's `n` in the output
dict, so the record said `n = 149` while the counts summed to 178. Both fixed; the class is
"a reducer that quietly renames the caller's field".)*

### 3.2 ⛔ What this may not conclude, per pre-registration §3b/§3c

* **ONE scene.** One scene is one cluster, so the interval above is a **contiguous-block bootstrap,
  explicitly NOT the programme's decision-grade episode-cluster bootstrap.** The finding is
  **DIRECTIONAL**. It may not promote anything to a trained config.
* **Two classes only** — 67 right / 202 straight / **0 left** in this 30 s clip.
* The second NuRec bundle on Thor was **mid-download by another stream at run time** (776 MB of
  ~2 GB, zip central directory absent — `BadZipFile`), so the 2-scene cross-scene generalisation test
  is a follow-up, not a gap in this run.
* It says **nothing about our model**. The trained frozen-trunk probe is SOTA §11 row 4 (~1 GPU-h).
* ⛔ The xodr's `<speed max>` must not become a LONGITUDINAL target — `INHERITED` from the sibling
  probe: the values are km/h under an `mph` tag *and* inconsistent with the observed driving.

---

## 4. Source scepticism — every external claim, with its protocol and whether it is comparable

| source | protocol | seeds / CI | lat/lon split | comparable to ours? |
|---|---|---|---|---|
| **2605.00066** (open- vs closed-loop) | Spearman over **8 methods** with paired NAVSIM + Bench2Drive results | **p-values, NO CI** | **no** | **direction only.** Their unit is a *method*; ours is a *window*. A method-level ρ = −0.36 and our window-level partial ρ are different quantities, and our result does **not** refute theirs. What our result does is decide **our** gate design — and it says the same thing about ADE by a stronger route. |
| **2606.03159** (Cosmos-Dreams) | 574 PhysicalAI-AV NuRec scenes, closed loop, 20 s rollouts | **none, single run, self-evaluated** | front/lat/rear collisions only | only the **conditioning format** transfers — which is precisely what item 3 built. The 68 FPS is a **GB300** number; ⛔ not a Thor plan. |
| **2603.28029** (criticality metrics) | definitional survey | n/a | n/a | definitions only |
| pseudosim's own citation of 2605.00066 | — | — | — | ⚠️ the programme had already imported these numbers into a **weight-5.0 scoring term** before anyone tested them on our data. This run is that test. |

⛔ **Not one external number here is a baseline we may claim to beat.** They are directions and
mechanisms on different corpora with different estimators — several with no estimator at all.

---

## 5. What changes, and what does not

1. ✅ **`ego_progress` now ships in the LONGITUDINAL family** (`four_families.longitudinal`),
   additive, with its convention, its exclusion count and its admission control in the record.
2. ⛔ **It does NOT become a gate emitter.** Its partial correlation with closed-loop failure flips
   sign across two sibling arms, both separated. The scan's §11-row-2 recommendation is **refuted as
   a gate proposal** on our own data. *(This is the escalation the scan's own §11 flagged as touching
   `GATE_PROTOCOL.md` — the answer is "report it, do not gate on it".)*
3. ⭐ **The gate's closed-loop-predictive slot belongs to the LATERAL family, not to ADE and not to
   progress** — `cross_mae` is the only metric separated on 4 of 4 arms after the difficulty control.
   This needs a **pre-registered replication** before it becomes a gate rule; it is a hypothesis from
   this run, not a registered outcome.
4. ⚠️ **Any future open-vs-closed correlation must carry the difficulty control.** Without it, a
   hold-`v0` baseline outperforms the model's own ADE as a predictor of the model's closed-loop
   failure — and a report that omitted the control would have published "open-loop ADE predicts
   closed-loop, ρ ≈ 0.17–0.40" and been wrong about what it measured.
5. ✅ **The lane raster exists and discriminates**; the strategic-input wiring remains a **D-018
   escalate** (the scan said so and I did not exceed it).

---

## 6. Deliverable manifest — every artifact and where it lives

| artifact | where it lives | state |
|---|---|---|
| pre-registration | `…/Research/2026-08-03-sota-top3-executed/PRE_REGISTRATION.md` | repo, **staged** |
| this report | `…/Research/2026-08-03-sota-top3-executed/RESULTS.md` | repo, **staged** |
| **ego-progress metric** | `taniteval/taniteval/progress.py` | repo, **landed** |
| **lane-graph raster** | `taniteval/taniteval/lane_raster.py` | repo, **landed** |
| tests (15 functions / **18 collected**, one parametrised ×4) | `taniteval/tests/test_progress.py` | repo, **landed** |
| family wiring (additive, 2 hunks) | `taniteval/taniteval/four_families.py` | repo, **landed** |
| item-2 runner | `…/2026-08-03-sota-top3-executed/code/run_item2_progress_and_predictivity.py` | repo, **staged** |
| item-2 confound control | `…/code/run_item2b_difficulty_control.py` | repo, **staged** |
| item-3 runner | `…/code/run_item3_lane_raster.py` | repo, **staged** |
| **result JSON** (27 arms × 4 families, E2c, E2d) | `…/raw/item2_progress_and_predictivity.json` | repo, **staged** |
| **result JSON** (E2e partial correlations) | `…/raw/item2b_difficulty_control.json` | repo, **staged** |
| **result JSON** (raster + D-MAP-1/2/3) | `…/raw/item3_lane_raster.json` | repo, **staged** |
| rasters, 299 × [3,128,128] | `…/raw/lane_raster_scene00040136.npz` (1.7 MB compressed) | repo, **staged** |
| visual strip, human-checkable | `…/raw/lane_raster_strip.png` | repo, **staged** |

**Nothing is stranded on a pod or in a worktree.** Hosts touched: dev box (CPU) and **one 90-second
read-only `ls`/`zipfile` probe on tanitad-thor**. `tanitad-new`, `tanitad-pod4` untouched. GPU: **0**.

**Suite green after the change:** `taniteval` **828 passed** (810 before + 18 new), `stack`
**1808 passed / 12 skipped / 2 xfailed**.

---

## 7. Escalations — named here, not left in a README

1. ⭐ **The LATERAL-predicts-closed-loop hypothesis needs a pre-registered replication.** It is the
   most decision-relevant thing in this report and it arrived through an unregistered branch. It
   should be run on arms not used to generate it, ideally with the Thor closed-loop harness rather
   than the imagination-in-the-loop one, so the two instruments are genuinely independent.
2. **`GATE_PROTOCOL.md` should record that `progress_ratio` is REPORTED but NOT GATED**, with the
   sign-flip as the reason — otherwise the scan's recommendation will be re-proposed.
3. **P0 L1 (`win["lead"]` for the val-40) is blocked on data, not code** (§1.1) — the split manifest
   and matching `obstacle.offline` chunks must be on a live host; `tanitad-eval` is gone.
4. **The pseudosim `t0_axis` convention** biases GT's own ego-progress by 0.81 % on our 2 s windows
   and more on curves. Not urgent, but it should be recorded beside the PSS numbers rather than
   discovered again.
