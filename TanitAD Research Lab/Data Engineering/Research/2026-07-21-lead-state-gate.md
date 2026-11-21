# The lead-state gate — does agent state predict the ego's future 2 s longitudinal displacement?

**Author:** Data Engineering agent · **Date:** 2026-07-21 (local, Europe/Berlin) · **Status:** PRE-REGISTERED, results below
**Authorised by:** Sayed, 2026-07-21 — "run the gate first; only ingest on a pass".
**Code:** `stack/scripts/lead_state_gate.py` (staged) · **Result JSON:** see §5.

---

## 1. What is being tested, and why this and not something else

`obstacle.offline` — full 3D agent tracks, present on **96.90 %** of our corpus — was never ingested;
our PhysicalAI loader reads **2 of 36** available features (MEASURED, ours, 2026-07-21,
`DATA_STRATEGY_FOR_HIERARCHY.md` §1–2). Consequences that all trace to the same hole:
`lake/enrich.py:vlm_pending_lead_state()` returns a `None` stub; `TANITEVAL_V2_METRIC_SUITE.md`
**refused** headway/TTC/distance-keeping metrics; "wait for the vehicles to pass" was scoped out of the
tactical layer.

The ingest costs **12.4 GB and 2–3 eng-days**. It rests on one premise:

> **"agent state unblocks tactics"** ⟹ knowing the lead vehicle must **measurably** predict how far the
> ego travels along-track over the next 2 s.

This gate tests **the premise, not an implementation**, so a negative result cannot be blamed on a
training bug. It targets the program's MEASURED dominant residual — **83 % of the 2 s error is
along-track** (`flagship-longitudinal-lever`, prior).

## 2. Pre-registration — both outcomes committed in advance

**Primary statistic.** Relative reduction in **MAE of the 2 s along-track displacement** on **held-out
episodes**, arm **A (ego state only)** → arm **B (ego state + lead state)**, interval from the
**paired episode-cluster bootstrap** over held-out episodes (`taniteval/taniteval/ci.py`,
B = 2000, seed 0). The legacy `overlapping_holdout_se` is 1.28–2.06× too narrow and **may not decide
this**.

| outcome | rule | consequence |
|---|---|---|
| **PASS** | ≥ **15 %** reduction, CI excluding 0 | premise holds → ingest all 197 chunks |
| **FAIL** | ≤ **5 %**, **or** the CI spans 0 | **premise FALSIFIED** → do **not** ingest; hand back 12.4 GB and 2–3 eng-days |
| **AMBIGUOUS** | 5–15 % | report the number, do **not** round it to either story, escalate |

**Do not tune the experiment until it agrees.**

## 3. Design — and the four ways this could have lied

| # | risk | what the design does about it |
|---|---|---|
| 1 | **Future leakage through the lead track** (class C6). Interpolating a track *across* t imports the lead's future position, which correlates with the ego's future displacement through shared traffic dynamics. | **Strictly causal.** A window at time `t` sees only obstacle rows with `timestamp_us ≤ t`: nearest sample, staleness ≤ 0.5 s; closing rate from a **backward** ~0.5 s baseline. Nothing after `t` is ever read except the target itself. |
| 2 | **A leak somewhere else in the pipeline.** | **Negative control arm `B_shuf`** — identical lead columns, permuted **across episodes**. If the shuffled arm also improves, the result is void. |
| 3 | **Capacity, not information.** Arm B has 7 more columns. | Identical regressor, identical hyper-parameters, evaluation on **held-out episodes** — extra columns can only help through generalisation. Two model families reported (`HistGradientBoostingRegressor`, ridge) so the verdict is not a property of one learner. |
| 4 | **Split-selection noise / an interval that decides nothing** (classes C5, "never quote an interval without its estimator"). | Episode-disjoint splits; **episode-cluster bootstrap**, never the deprecated overlapping-holdout SE. Two designs: the program's own canonical split (primary) and 5-fold episode-level out-of-fold (power check). |

**Geometry note.** `obstacle.offline` boxes carry `reference_frame = rig` at their **own** timestamp, so a
backward difference of `center_x` **is** the relative closing rate — no ego-motion compensation is needed
or wanted.

**One pre-result correction, on physical grounds.** The first smoke build differenced the two most recent
samples of a track, with `dt` as small as 20 ms; that differentiates box noise, not motion, and produced a
**28 m/s "closing rate"**. Replaced with a ~0.5 s backward baseline (`DIFF_BASE_S = 0.5`,
`MIN_DIFF_S = 0.15`) **before any gate number was computed**. A noisy closing feature biases *toward* FAIL,
so this correction is against the null, not for it.

**Features.**
`A` = `v, ax, ay, curvature, |curvature|, yaw-rate, Δv@0.5 s, Δv@1.0 s, v(t−0.5), v(t−1.0)` — all causal.
`B` = `A` + `lead_present, gap_m, closing_ms, ttc_s, 1/ttc, lead_lat_m, lead_is_big`.
`B+` (exploratory, not the gate) = `B` + `n_vehicles_ahead_50m, n_VRU_near`.
Target `y_long(t) = (p(t+2 s) − p(t)) · [cos yaw(t), sin yaw(t)]` in metres.

**Corpus.** Windows at 10 Hz, `t ∈ [1.0 s, 18.0 s]` (so both the 1 s history and the 2 s future are inside
the clip) → 171 windows/clip. Clips are drawn **only from the parity corpus**
(`r0/phase0_selection.parquet`, 3000 clips), split by the program's own rule
(sorted ids → `torch.randperm(seed 0)` → first 20 % val). **Read-only: no episode is re-selected,
re-ordered, added or dropped; `_epcache` is never touched.**

---

## 4. Sample

26 `obstacle.offline` chunks, chosen as the **largest chunk per country** over all 25 countries in the
corpus (plus chunk 0036, the chunk the prior measurement used) — 1.1 GB downloaded, MEASURED.

## 5. Result — **FAIL. The premise is falsified at the pre-registered horizon.**

All numbers **MEASURED (ours, 2026-07-21)**, artifact
`Implementation/incoming/2026-07-21-lead-state-gate/lead_gate_result.json`
(regenerate: `python stack/scripts/lead_state_gate.py build|gate --out <dir>`).

**Corpus scored:** 104,994 windows · **614 clips** · 26 chunks · **25 countries**, all inside the parity
corpus. Target `y_long`: mean 27.02 m, sd 19.60 m.

| cell | test windows / episodes | MAE **A** (ego) | MAE **B** (ego+lead) | **relative reduction** (95 % episode-cluster bootstrap) | CI excludes 0 |
|---|---|---|---|---|---|
| **`gbm \| canonical` (PRIMARY)** | 21,546 / **126** | **0.4435 m** | **0.4383 m** | **+1.16 %  [−0.92, +3.19]** | **no** |
| `gbm \| oof5` (power check) | 104,994 / **614** | 0.4916 | 0.4831 | **+1.73 %  [+0.25, +3.67]** | yes |
| `ridge \| canonical` | 21,546 / 126 | 0.4298 | 0.4281 | +0.41 %  [−0.51, +1.23] | no |
| `ridge \| oof5` | 104,994 / 614 | 0.4412 | 0.4407 | +0.13 %  [−0.30, +0.53] | no |

**VERDICT (pre-registered rule, primary cell): FAIL** — `+1.16 % [−0.92, +3.19]`, which is **≤ 5 % *and*
an interval spanning 0**, i.e. it trips *both* fail conditions. The most powerful cell available
(`gbm|oof5`, every one of the 614 episodes held out once) separates from zero but lands at **+1.73 %** —
**one ninth of the 15 % pass bar** and still inside the fail band. There is no cell, model family or
design in which the premise survives.

**In absolute terms the entire lead-vehicle channel is worth 5.1 mm of 2 s along-track MAE.**

### 5.1 The controls say the null is real, not an artefact

| control | reading | what it rules out |
|---|---|---|
| **`B_shuf`** — same lead columns, permuted across episodes | −1.16 % / −1.27 % / −0.32 % / −0.18 % (i.e. *worse* in every cell) | a leak or a capacity artefact. Real lead features beat their own shuffle by ~2.3 pp — the signal exists, it is just **tiny** |
| **`B+`** — B plus agent-density counts (`n_vehicles_ahead_50m`, `n_VRU_near`) | +0.74 % / +1.33 % / +0.36 % / +0.10 % | "we picked the wrong agent summary". More agent information does **not** help more |
| **lead-present subgroup only** (7,823 windows / 80 episodes, primary cell) | **+4.22 % [−0.71, +8.95]** | "it is diluted by the 61.5 % of windows with no lead". Even where a lead *exists*, the reduction is below the 5 % fail bar and spans 0 |
| **ridge beats GBM** on the canonical split (0.4298 vs 0.4435) | — | the target is near-**linear** in ego state; the flexible learner's extra capacity buys nothing, so arm B did not fail for lack of expressive power |

### 5.2 Why — the mechanism, MEASURED not asserted

The 2 s along-track displacement is a **near-deterministic function of ego kinematics**:

| predictor | R² | MAE |
|---|---|---|
| constant velocity (`2·v`) | — | 0.9133 m |
| speed alone, OLS | **0.995038** | 0.9173 m |
| full ego state (10 causal features, GBM) | — | **0.4435 m** |
| + lead state | — | 0.4383 m |

Two seconds is simply **too short for another vehicle to move the ego**. Whatever the lead is doing, the
ego's own speed and acceleration have already committed the next 2 s of along-track motion; the lead's
influence has to arrive through a driver reaction that mostly lands *after* the window closes. The
information is in `obstacle.offline` — it just is not information *about this quantity*.

### 5.3 A second-order finding worth its own line

A **10-feature causal ego-kinematics GBM predicts 2 s along-track displacement to 0.4435 m MAE** on
held-out episodes of the parity corpus. That is the same order as the deployed flagship v1's full 2 s
ADE (0.452 m, `MODEL_REGISTRY.md`, `flagship4b-speedjerk-30k`). ⚠️ **These are not the same statistic**
(ADE is a two-axis average over a trajectory; this is a single-axis endpoint MAE) so it is **not** a
like-for-like comparison and must not be quoted as one — but it does say that most of the along-track
budget is reachable from ego kinematics alone, which is consistent with the program's standing
"no arm beats hold-`v0` at cruising" finding.

---

## 6. Decision

**Do NOT ingest `obstacle.offline` on the strength of this premise.** The 12.4 GB and 2–3 eng-days are
handed back. What was actually spent: **1.1 GB** (26 chunks), **0 GPU-hours**, **$0**, one agent session,
on CPU only — no pod was touched.

**This does not say `obstacle.offline` is worthless.** It says one specific claim is false: *that agent
state measurably improves the 2 s longitudinal prediction that dominates our residual*. §7 separates what
the gate did and did not rule out.

---

## 7. POST-HOC — ⚠️ EXPLORATORY, run AFTER the verdict, and it CANNOT make this a PASS

**Read this section with the rule in mind: a subgroup chosen after seeing a null, scored with a model
trained on that subgroup, is a HYPOTHESIS.** It is recorded here because it changes what the *next*
pre-registration should ask, not because it changes this one. Artifact:
`Implementation/incoming/2026-07-21-lead-state-gate/lead_gate_posthoc.json`
(`stack/scripts/lead_state_gate_posthoc.py`). All cells: GBM, canonical split, same estimator.

### 7.1 The horizon is **not** the excuse — this was the obvious defence and it fails

The natural objection is "2 s is too short for a lead to move the ego". Tested directly:

| horizon | 2 s | 3 s | 4 s | 5 s | 6 s |
|---|---|---|---|---|---|
| **displacement**, rel. reduction | +0.50 % | +0.86 % | +1.51 % | +1.83 % | +1.63 % |
| CI | [−1.53,+2.44] | [−1.61,+3.19] | [−0.85,+3.79] | [−0.68,+4.12] | [−1.40,+4.35] |
| **speed change**, rel. reduction | +1.47 % | +1.57 % | +1.28 % | +0.90 % | +0.55 % |

Every corpus-wide cell spans zero, out to **6 s** — three times the pre-registered horizon — and on a
target (`Δv`) that deliberately strips the near-deterministic `v·H` term. **Extending the horizon does not
rescue the premise.** Shuffle controls: −0.5 % to −1.8 % throughout.

### 7.2 The one place a real effect lives: *conditional on a lead existing*

| slice (trained AND scored on the slice) | n windows / episodes | MAE A → B | rel. reduction | shuffle control |
|---|---|---|---|---|
| **lead present, 2 s** | 6,106 / **78** | 0.5171 → 0.4659 | **+9.91 %  [+2.19, +17.24]** | −0.90 % |
| lead present, 3 s | 6,106 / 78 | 1.2486 → 1.1470 | +8.13 % [+1.58, +14.43] | −0.07 % |
| lead present, 4 s | 6,106 / 78 | 2.3669 → 2.1623 | +8.64 % [+2.80, +14.25] | −0.39 % |
| lead present, 5 s | 6,106 / 78 | 3.8560 → 3.5471 | +8.01 % [+2.45, +13.40] | +0.51 % |
| lead present, 6 s | 6,059 / 77 | 5.6894 → 5.3274 | +6.36 % [+0.37, +12.32] | −0.56 % |
| lead gap < 25 m, 2 s | 2,443 / 48 | 0.5721 → 0.5347 | +6.54 % [−2.28, +14.27] | +0.45 % |
| lead TTC < 6 s, 2 s | 868 / 32 | 0.7586 → 0.7407 | +2.35 % [−7.90, +9.42] | −0.04 % |
| closing > 0.5 m/s, 2 s | 2,414 / 58 | 0.6735 → 0.6841 | **−1.58 %** [−11.44, +7.22] | −2.87 % |
| ego decelerating (`ax` < −0.5), 2 s | 3,237 / 93 | 0.7822 → 0.7762 | +0.77 % [−2.17, +3.29] | −1.10 % |

**What this is:** a **lead-conditioned specialist** — a model that only ever sees windows with a lead —
gains **~8–10 %**, stably across every horizon, with a clean shuffle control. That is a real effect and it
sits in the pre-registered **5–15 % escalation band**.

**What this is NOT:** the pre-registered statistic. Three reasons it may not be quoted as a pass.
1. The subgroup was selected **after** seeing the null (post-hoc selection).
2. The specialist is **trained on the subgroup**, so it is a different model, not a different feature set.
   The corpus-wide model *evaluated* on the same slice gains only **+4.22 % [−0.71, +8.95]** — spans zero.
   **The gain comes as much from specialising the model as from the lead features.**
3. `+9.91 %` is still short of the 15 % bar, on **78 episodes**, with a CI reaching down to +2.19 %.

**The honest reading:** *concatenating lead columns into one global regressor buys nothing (1.16 %);
routing lead-present windows to their own predictor buys ~8–10 % on the 38.5 % of windows that have a
lead — worth ~3–4 % corpus-wide at best.* Note also that the two slices where the lead should matter MOST
(low TTC, actively closing) show **nothing** — which argues the effect is a **regime** signal
(car-following vs free-flow) rather than a **dynamics** signal (this specific lead's kinematics).

---

## 8. What this does and does not rule out

| claim | status after this gate |
|---|---|
| "Agent state measurably improves the 2 s longitudinal prediction that dominates our residual" | ❌ **FALSIFIED** — +1.16 % [−0.92, +3.19], and ≤ +1.83 % at every horizon out to 6 s |
| "The null is because 2 s is too short" | ❌ **refuted directly** (§7.1) |
| "The null is because we summarised agents badly" | ❌ arm **B+** with density counts is no better |
| "The null is a leak or a capacity artefact" | ❌ the shuffle control is **negative in every cell** |
| "A lead-conditioned *specialist* predictor gains ~8–10 % where a lead exists" | 🟡 **post-hoc HYPOTHESIS**, needs its own pre-registration (§7.2) |
| `lead_state` unblocks **headway / TTC / distance-keeping METRICS** | 🟢 **untouched by this gate.** The gate asked whether agent state *predicts ego motion*, not whether it is *measurable*. It is: gap, closing rate and TTC computed cleanly on 38.5 % of windows |
| `lead_state` unblocks **scenario stratification** by lead presence | 🟢 **untouched, and partly delivered** — see §9 |
| Agent state helps the **v4 tactical layer** | ⚠️ **not tested here and must not be assumed.** This gate is about a *kinematic* target. Flagged for the v4 designer |
| Other agents' **indicators / light state** | ❌ still absent from the schema — unchanged |

---

## 9. Delivered anyway: the corpus-wide lead-presence statistic

The prior estimate came from **271 clips across 3 of 197 chunks** and was explicitly flagged
*"enough to decide whether to ingest; not a corpus statistic"*. This run replaces it with a
**26-chunk, 25-country, 614-clip** measurement on the parity corpus
(`Implementation/incoming/2026-07-21-lead-state-gate/lead_presence_by_country.csv`).

| quantity | prior (3 chunks) | **this run (26 chunks, 25 countries, 614 clips)** |
|---|---|---|
| frames with a lead vehicle | 28.0–53.4 % by region | **38.51 %** of windows |
| clips with a lead at some point | 87 % (chunk 0036 only) | **66.1 %** |
| vehicles ahead < 50 m | — | **1.91 / frame** |

Regional spread is **4×**, not 2×: Italy **68.6 %** · Portugal 54.4 % · Czechia 51.1 % · United States
48.0 % … Germany 23.0 % · Latvia 20.6 % · Estonia 18.3 % · **Croatia 16.6 %**. ⚠️ These are **not**
directly comparable to the prior numbers — this run uses a strictly causal, ≤ 0.5 s-stale, bumper-gap
definition, where the prior used centre distance with no causality constraint. Compare within a column,
never across.

**This is a usable stratification axis today** — it is exactly the "lead presence/behaviour" stratum the
metric suite wanted and the VLM could not supply — and it cost 1.1 GB.

---

## 10. Provenance and parity

* **Nothing was ingested.** Step 2 was not reached; the pre-registered rule says stop and it was obeyed.
* **Read-only against the corpus.** No episode was re-selected, re-ordered, added or dropped; no
  `_epcache` entry was written. Verified after the run (MEASURED):
  `r0/phase0_selection.parquet` sha256[:16] `81ea25bed7109f43`, mtime **2026-07-12 21:20**;
  `r0/r0_selection.parquet` `c0af506fb3eed672`, mtime **2026-07-06 22:15**;
  `_epcache/physicalai-train-14231cd29c74` **400** `ep_*.pt`, dir mtime **2026-07-12 21:10**;
  `_epcache/physicalai-val-bb543bdf7836` **100**, mtime **2026-07-12 21:20** — all pre-dating this
  session. The only writes were the 26 new `labels/obstacle.offline/*.zip` and a derived
  `lead_gate/` folder.
* **`gated-confidential` discipline.** Nothing derived from PhysicalAI-AV was fed to the lake, exported,
  published or pushed to HF. The `assemble_lake_record` `PermissionError` guard was never approached
  because the lake was never touched. These artifacts are **internal-dev-only** and belong to neither
  TanitDataSet-C nor -R.
* **No GPU was used.** No pod was touched. CPU + 1.1 GB of download.
