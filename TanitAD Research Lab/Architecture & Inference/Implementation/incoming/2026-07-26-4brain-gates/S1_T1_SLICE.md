# S1 and T1 — the first working slice

**Date:** 2026-07-26 (Europe/Berlin) · **Author:** 4-brain critical-path agent
**Depends on:** `GATE_RESULTS.md` (Gate 1 ✅ PASS · Gate 2 see §3)
**Compute:** `tanitad-eval` only, CPU. **Nothing staged, committed or pushed.**

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` · `INHERITED` · `ESTIMATED` ·
`HYPOTHESIS`.

---

## 0. What this slice does and does not deliver

**Delivered** (`MEASURED`, all artifacts in this directory):

| Component | S1 | T1 |
|---|---|---|
| decision-point miner | ✅ `gate1_connectivity_probe.py` | ✅ `t1_conflict_miner.py` |
| option-set constructor | ✅ `succ(lane)`, variable arity | ✅ `{yield, proceed}` |
| **non-circular target** | ✅ map ∩ realised path | ⛔ `Y_expert` only — **`Y_outcome` blocked: Gate 2 FAILED** (§3.4) |
| metric | ✅ `branch_accuracy` (open-loop-choice) | ✅ defined; not computable without `Y_outcome` |
| estimator | ✅ paired episode-cluster bootstrap, unit = scene | ✅ same |
| coverage + class balance | ✅ §1.2 | ✅ §3.2 |
| **blind-conditioning firewall** | ✅ **run, 3 variants** | ✅ run |
| power statement | ✅ §1.4 | ✅ §3.3 |

**NOT delivered, and deliberately not faked:** no model has been *scored* on either metric. Scoring
needs the A0-FLAT / A2-HIER arms of `4BRAIN_DOMINANCE_PROGRAM.md` §3.1, which do not exist. What this
slice establishes is the **instrument and its baselines** — chance, majority, and blind — which is
precisely what must exist *before* an arm is trained, and is what would have caught the v1 route label.

---

## 1. S1 — branch selection at a multi-option junction

### 1.1 The target, and why it is not circular

```
target_branch = index of the successor lane that the ego's REALISED future path first enters,
                by nearest-centreline matching over the option set succ(lane(ego, t_dp))
```

`MEASURED`: it is a **map fact intersected with a trajectory**. The map supplies the option set; the
recorded drive supplies which one was taken. **Neither is a model input.** This is structurally unlike
`route_target(nav_cmd) → _NAV_TO_ROUTE[nav_cmd]`, where the target was a fixed function of an input.

Decision point (spec §2.1): the last frame at which the ego is on a lane whose forward lane-graph walk
reaches a lane with |succ| ≥ 2 at a distance in **[15, 60] m**. ⚠️ The walk is the load-bearing detail —
see `GATE_RESULTS.md` §1.3(b); measuring to the end of the *current* lane under-counts by **3.3×**
because MADS lane segments are only ~20.65 m.

### 1.2 Coverage and class balance (`MEASURED`, `S1_RESULTS.json`, 51 scenes)

| quantity | value |
|---|---|
| decision points mined | **43** |
| …with a resolved `target_branch` | **30** |
| scenes carrying ≥1 resolved DP | **20** |
| DPs per contributing scene | 1.5 |
| option-set arity | **27 binary · 3 ternary** |
| **class balance** (target index) | **0 : 21 · 1 : 8 · 2 : 1** |
| **majority-class rate** | **0.7000** |
| mean chance rate (1/K) | 0.4833 |
| median branch heading separation | **45.0°** |
| median time from `t_dp` to branch entry | **1.70 s** |
| median remaining ego path from `t_dp` | **111.8 m** |

⚠️ **A horizon caveat that must travel with S1-on-AlpaSim.** The spec scores S1 over `t_dp → +20 s`.
AlpaSim NuRec scenes are **20.0 s clips** (`MEASURED`, median), and the median remaining path from a
decision point is **111.8 m ≈ 12 s**. So the *branch label* is well defined (it resolves in 1.70 s
median), but **`route_compliance_rate` over a 20 s window is not measurable on most of these scenes.**
S1's open-loop-choice metric is available; its closed-loop 20 s compliance metric is not, on this corpus.

### 1.3 ⭐ The firewall result — `blind_conditioning_baseline`, run on 3 variants

Method: a scene-blind predictor sees **only the conditioning channels** (ego speed, goal, and the
map-derived option geometry) — **no pixels, no history** — under **leave-one-scene-out** CV. Two
independent attacks are run and **the STRONGER is reported**, because a weak blind test fails *unsafe*.

| variant | goal given to the model | n | clusters | `acc_blind` | `acc_major` | chance | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **E** easy | route polyline, ego frame, truncated **30 m** | 26 | 20 | **0.7692** | 0.6923 | 0.4872 | ✅ **ADMITTED** |
| **H** hard | single goal point at **150–200 m** | 8 | 6 | **0.5000** | 0.7500 | 0.5000 | ✅ **ADMITTED** |
| **CONTROL** | *no goal at all* | 30 | 20 | 0.6333 | 0.7000 | 0.4833 | ✅ ADMITTED |

Refusal threshold is `acc_blind ≥ 0.98 × ceiling = 0.98`. **No variant comes close.**
⇒ **The S1 target is NOT recoverable from the conditioning channels. It is admissible.**

**The self-test was executed and rendered its failing verdict** (07-26 C10 rule — *a guardrail that has
never failed is a comment*): a synthetic echo label gives `acc_blind = 1.0000 → REFUSED`; an
uninformative label gives `0.4000 → ADMITTED`. `python blind_conditioning_baseline.py --self-test`.

⚠️ **Three honest qualifications.**
1. `acc_blind` (0.7692) **exceeds** `acc_major` (0.6923), so the operative bar for any model is
   **`skill = acc_model − 0.7692`**, not `acc_model − 0.70`, and not `acc_model` alone.
2. Blind-minus-majority is **+0.0385 [−0.2222, +0.2692]**, paired episode-cluster bootstrap
   (`taniteval.ci`, B=2000, unit = scene) — **not separated**. At n=26/20 clusters this test cannot
   establish that the goal channel helps *or* that it leaks. **ADMITTED here means "not refused",
   not "certified clean."** It should be re-run at the ≥40-cluster corpus.
3. Variant **H is only available on 8 of 30 decision points**, because a 150–200 m goal needs 150–200 m
   of remaining path and the median is 111.8 m. The spec's strongest anti-echo variant is the one this
   corpus can least support.

### 1.4 Power (`MEASURED`)

Resampling unit = **AlpaSim scene**, per spec.

| | available | bar | shortfall |
|---|---:|---:|---:|
| single-arm | **20** | 40 | **2.0×** |
| two-arm | **20** | 200 | **10.0×** |

**Neither bar is met.** Yield is **0.39 resolved-target clusters per scene** (20/51) ⇒
≥40 needs **~103 scenes**, ≥200 needs **~513 scenes** (`ESTIMATED` from the measured yield).

---

## 2. What a model would be scored on

```
branch_accuracy      = mean[ argmax_k score(option_k | image, ego, goal) == target_branch ]
surface              = open-loop-choice        horizon = t_dp -> branch entry (median 1.70 s)
estimator            = paired_episode_cluster_bootstrap(a, b, eid=scene, B=2000)   [taniteval/ci.py]
reported capability  = skill = acc_model - acc_blind      (NEVER acc_model alone)
```

The option scorer in `blind_conditioning_baseline.py` handles **variable arity natively** (softmax over
a K-row feature matrix). That is the property `STRATEGIC_TACTICAL_PROBLEM_SPEC.md` §2.1 names as
structurally unavailable to a fixed `{L,S,R}` head — here it is the baseline's actual implementation,
so the flat-vs-hierarchical comparison can be made on identical machinery.

---

## 3. T1 — yield vs proceed at an unprotected conflict

### 3.1 What was built

`t1_conflict_miner.py` mines genuine right-of-way conflicts from map + logged agent tracks:
a spatial crossing of the ego's and an agent's paths, with **crossing angle ≥ 20°** (a crossing, not a
follow), **both arrival times within the 15 s horizon**, and **|arrival gap| ≤ 4 s** (a real conflict —
if both parties are 10 s apart, nobody had to yield). `Y_expert` = did the ego reach the conflict
point first.

### 3.2 Result — ⚠️ the corpus does not carry T1 at a scorable n

`MEASURED` (`t1_conflict_points.json`, 51 scenes):

| quantity | value |
|---|---|
| conflict decision points | **4** |
| scenes carrying one | **4 / 51** |
| `Y_expert` class balance | **2 ego-first / 2 agent-first** (majority rate 0.500 — perfectly balanced) |
| median \|arrival gap\| | 2.95 s |
| median TTC_ego | 11.05 s |
| median crossing angle | **67.3°** |
| agent classes | automobile 2 · person 1 · rider 1 |

The four are *genuine* — crossing angles 21.3° / 46.9° / 87.7° / 114.1°, gaps 2.2–3.5 s, and they
include a pedestrian and a rider. The problem is n.

**This is a corpus property, not a threshold artefact** — `MEASURED` sensitivity sweep
(`t1_sensitivity_sweep.txt`), 159 raw spatial crossings over 39 scenes:

| \|gap\| ≤ | angle ≥ | conflicts | scenes | clusters/scene |
|---:|---:|---:|---:|---:|
| 4 s | 20° | 4 | 4 | 0.078 |
| 8 s | 20° | 7 | 6 | 0.118 |
| 12 s | 20° | 22 | 13 | 0.255 |
| 4 s | 0° | 52 | 25 | 0.490 |

Loosening the *angle* to 0° is what buys volume — but a 0° "crossing" is a car-following event, not a
right-of-way conflict, so that column **buys n by dissolving the problem**. Loosening the *gap* to 12 s
means neither party had to yield. **The strict criteria are the correct ones, and they yield 4.**

### 3.3 Power, and a hard consequence

Yield **0.078 conflict-clusters per scene** (4/51):

| | available | bar | scenes needed at measured yield |
|---|---:|---:|---:|
| single-arm | **4** | 40 | **~510** |
| two-arm | **4** | 200 | **~2,550** |

⚠️ **The two-arm figure exceeds the entire NuRec `public_2604` pool (1,606 scenes,** `INHERITED`,
`ALPASIM_STATE.md`**).** ⇒ **T1 two-arm power is NOT reachable on AlpaSim NuRec scenes by scene
acquisition alone.** That is a PI-level finding: it needs longer scenes, a conflict-enriched selection
pass (the balanced-suite builder screens on scene *type*, not on conflict presence), or another corpus.

A second structural limit: all 4 conflicts have `t_dp_idx = 0` — the 15 s tactical lead time runs off
the front of the clip. **20 s scenes cannot carry a 15 s decision horizon plus the conflict itself.**

### 3.4 `Y_outcome` — ⛔ blocked by Gate 2, and NOT worked around

`Y_outcome` is T1's *entire* admissibility argument: *"it is not a label; it is a simulated consequence,
so it cannot be circular with any input by construction."* That argument holds only if the simulated
consequence **is a function of the policy's choice**.

**Gate 2 FAILS on exactly that clause** (`GATE_RESULTS.md` §2.4). `MEASURED` over 4 scenes, 5 repeats
per arm, paired episode-cluster bootstrap with unit = agent: the non-ego agents **do** depart from
their logged tracks (8.4–78.2 m — *not* replay), but the departure is **not attributable to the ego**.
In the best-powered scene (59 dynamic agents) the ego-induced effect is bounded at
**[−0.21, +0.14] m** against a **4.5 m** sampling-noise floor, and the near-ego stratum is null in
**4/4** scenes — while the ego differed between arms by up to **149 m**.

⇒ **`Y_outcome` is not computed, and T1 is not built on it.** Per the brief, a failed gate is reported,
not circumvented.

Two consequences worth carrying:

1. **Even if reactivity were demonstrated, `Y_outcome` needs many rollouts per scene.** Run-to-run
   spread between *identical* repeats is **4.2–23.2 m** over 17.5 s. A single-rollout `Y_outcome` would
   be dominated by traffic-model noise rather than by the decision under test.
2. **The corpus blocks T1 independently** (§3.3). Even a passing Gate 2 would leave T1 at 4 clusters.

The firewall on `Y_expert` is reported for completeness but is **not informative at n=4** (4 items over
4 clusters, a 2/2 split). It is recorded as `INSUFFICIENT-N`, not as a pass. Note that per spec T1 feeds
**no agent boxes** to the model, so the conditioning channels are ego state + chosen branch only — a
blind predictor has almost nothing to work with by construction, which is the design intent.

The firewall on `Y_expert` is reported for completeness but is **not informative at n=4** (4 items over
4 clusters; leave-one-cluster-out on a 2/2 split). It is recorded as `INSUFFICIENT-N`, not as a pass.
Note that per spec T1 feeds **no agent boxes** to the model, so the conditioning channels are ego state
+ chosen branch only — a blind predictor has almost nothing to work with by construction, which is the
design intent.

---

## 4. Recommendation

1. **S1 is buildable and admissible today; it is underpowered by 2×.** The cheapest unblock is
   **~103 scenes** (≈155 GB) for the single-arm bar, using the committed balanced-suite downloader.
2. **T1 is blocked twice over and should be escalated, not scaled.** Its target machinery fails Gate 2,
   and its corpus yield puts the two-arm bar beyond the entire NuRec pool. Neither is fixed by
   downloading more scenes.
3. **Do not spend GPU on an S1/T1 arm comparison yet.** The instrument now exists and its baselines are
   measured; the binding constraint for S1 is corpus (a download decision, not a training decision),
   and for T1 it is the target itself.
4. **The one cheap experiment that could reopen T1:** a single full closed-loop rollout with
   `trafficsim=catk` inside the runtime, to test whether reactivity appears under the real integration
   rather than the direct-gRPC contract (`GATE_RESULTS.md` §2.4, boxed caveat). Every other blocker is
   now removed, so this costs renderer time only. **It is the correct next action before any tactical
   work is planned or descoped.**
