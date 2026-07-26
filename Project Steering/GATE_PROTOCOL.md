# TanitAD Restart/Continue Gate Protocol — STANDING

**Status:** binding default from 2026-07-20. Replaces the learning-curve power-law exponent gate
(D-031 / D-A7). Origin: 360° review W2 / P1. Tool: `stack/scripts/run_gate.py`.
**Amended 2026-07-26** — the horizon correction (§0), Tier-1 #1 of the independent chief-scientist
review, *"the single highest-leverage correction in the review"* (`01_EXECUTION_PLAN.md` B.2 T1-1).

---

## 0. THE HORIZON RULE — added 2026-07-26

> **A gate verdict must NAME its metric's horizon and n, or it is not admissible.**
> This is the same rule as *"never quote a learning-curve exponent bare"* and *"never quote an
> interval without its estimator"*, applied to the axis those two missed.

### 0.1 Why — MEASURED, not argued

E1a (`…/incoming/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44_K185.json`,
paired common-start, **43 IDENTICAL windows**, `episode_cluster_bootstrap` B=2000, REF-C base
`refc-diffusion-base-v21-30k` @ 29999, real-footage low-OOD closed loop):

| stratum | CDR@1.75 m, K=20 (2.0 s) | CDR@1.75 m, K=185 (18.5 s) | peak XTE 2 s → 18.5 s |
|---|---|---|---|
| overall | 0.0035 | **0.5877** [0.5107, 0.6622] | 0.35 m → **38.94 m** |
| junction (n=6) | 0.0250 | **0.8414** [0.8144, 0.8667] | 1.23 m → **46.25 m** |

Paired Δ (K=185 − K=20), overall: **+0.5842 [0.5071, 0.6565]**, `separated: true`,
`p_delta_gt0 = 1.0`, `paired_episode_cluster_bootstrap`. The OOD-envelope ratio stays **≤ 1.30**, so
this is genuine **in-distribution** failure, not extrapolation.

**The 2 s instrument hid the dominant failure mode by ~168×.** On the *same* 43 windows the paired
ADE@2s delta is `0.0109 [−0.0, 0.0312]`, **not separated** — ADE@2s records essentially nothing while
corridor departure goes 0.0035 → 0.5877. And ADE is **98.6 % longitudinal by squared-error energy**
(`LATERAL_VS_LONGITUDINAL_ANALYSIS.md`), while the axis that ends a drive is lateral.

### 0.2 What changed

| | before 2026-07-26 | from 2026-07-26 |
|---|---|---|
| primary | `ade_0_2s` ≤ T, decides alone | **co-primary** `corridor_departure_rate` @ pre-registered K |
| `ade_0_2s` | the gate | **proposal-quality diagnostic** (`primary_role: "diagnostic"`) — read, recorded, informative about trajectory quality; **does not adjudicate** |
| horizon | implicit (K=20) | **explicit, pre-registered, and printed in the verdict** |
| junction | not separated | **reported separately, always**; optionally adjudicated |
| interval | cluster bootstrap | unchanged, and now **enforced on the co-primary too** |

### 0.3 The horizon is pre-registered and bounded

`register` **refuses**:

- **K ≤ 20** — that *is* `ade_0_2s`' own horizon (2.0 s). Registering a co-primary there buys the
  name of the fix without the fix (0.0035 vs 0.5877 on the same windows).
- **K > 190** — **structurally impossible** on this corpus. PhysicalAI clips are 190–199 frames and a
  window exists only if `T − W − K ≥ 1`, so the ceiling is **K = 190 (19.0 s)**; K = 200 cannot be
  produced by any episode. E1a used **K = 185 (18.5 s) = 97 % of the ceiling**.
- **a card with no co-primary at all** — unless `--no-co-primary "<reason>"` puts the exception on the
  record. `check` then stamps the verdict `horizon_honest: false` and prints why.

`check` **refuses** a corridor block whose `horizon_K` differs from the registered one, or whose
corridor half-width differs from the registered one. Re-rendering at a horizon chosen after seeing the
number is a garden of forking paths with extra steps.

### 0.4 The junction stratum is reported separately — always

`junction` = **|net heading change over the FIRST 2 s| ≥ 10°** (E1a's definition, held FIXED across
horizons so strata stay comparable as K varies). It is **a kinematic signature, never a topology** —
there is no map, no lane graph and no junction annotation in this corpus, and renaming it
"intersection" is the specific error `driving.py` §6.3 refuses.

It is never folded into the overall number, because that is where the failure concentrates
(**0.8414 vs 0.5877**). A stratum too small to bootstrap returns `None` from `taniteval.corridor` and
the gate records it as **NOT MEASURED** — never as a pass.

### 0.5 Estimator, on the co-primary too

Every corridor interval is the **episode-cluster bootstrap** over val episodes (`taniteval/ci.py`);
every arm-vs-arm delta is the **paired** form. `run_gate` **raises** — never warns — on
`overlapping_holdout_se`, on an unnamed estimator, and on a bare corridor number with no interval.
Two independent reasons, both MEASURED: it is **1.28–2.06× too narrow** across 10 arms
(`CI_RECOMPUTE_2026-07-20.json`) *and* it biases the **point estimate** (up to **×4.29**, with sign
flips, measured 2026-07-25). A quadrature combination of two single-arm intervals is not merely weaker
than the paired form — on shared windows it is **invalid**.

> ⚠️ **Live tripwire, do not "clean up".** `run_gate._deprecated_present` keys on the literal block
> name **`heldout`**, which `bench.py` deliberately retains as a back-compat alias for the legacy
> block. Renaming that key silently disarms the gate's own refusal.

### 0.6 The honest limits, stated

- **The corridor is not a lane.** No lane geometry exists in this corpus; the corridor is a half-width
  about the reference path and **1.75 m is PROPOSED, not measured**. That is why the whole threshold
  grid (1.0 / 1.75 / 2.5 m) is always emitted — a verdict that survives only at one half-width is a
  knife-edge, not a result.
- **Two surfaces, never pooled.** Closed-loop corridor departure accumulates *control* error;
  open-loop dense corridor departure is a *prediction residual* against the expert's own path. Every
  block names its `surface` and the gate records it.
- **Open loop cannot reach a long horizon.** `rollout.collect`'s dense path runs to `fwd_k = 20`, i.e.
  **K = 20 (2.0 s)** — the blind horizon. **A K ≥ 100 co-primary requires a closed-loop rollout**
  (E1a's surface), which requires GPU.

### 0.7 🟥 VOID SECONDARIES — a metric that cannot vary with the model may not kill a run

> **A gate secondary whose value is fixed by a LABEL or HARNESS defect carries zero information about
> the model. It must be adjudicated INSTRUMENT-FAIL, never MODEL-FAIL.** Refusing to decide on a broken
> measurement is not moving the bar — deciding on it is.

**Standing instance, effective immediately — `nonav_route_beats_majority >= 1` is VOID.**

`Project Steering/Gates/flagship-v4.card.json` lists it among the **KILL** secondaries. It **cannot pass**,
for a reason that has nothing to do with the checkpoint: the strategic route **target is a lookup of the
route input** (`refb_labels.route_target = _NAV_TO_ROUTE[nav_cmd]`), so training route-CE reaches
**exactly 0.0** by ~step 14.5 k and **`route_skill` is 0.0 BY CONSTRUCTION**; the follow-head answers
`straight` **240/240**. The metric measures **the label bug**, not the model. *(MEASURED —
`…/incoming/2026-07-25-hpp0-confound-audit/HPP0_CONFOUND_AUDIT.md`.)*

**Adjudication for the flagship-v4 30 k gate (and any gate carrying this secondary):**

1. Record `nonav_route_beats_majority` as **`INSTRUMENT-FAIL (VOID)`** with this section as its citation.
   **It does NOT contribute to the kill conjunction.**
2. Render the verdict from the remaining KILL secondaries plus the co-primary. *(The committed dry-run
   already separates these: `v1_g1_dryrun_gate_FIXED.json` → `split_8_KILL_5_REPORT = {kill_adjudicated: 8,
   report_only: 5, verdict_from_kill_only: "CONTINUE"}`.)*
3. **State the void secondary explicitly in the verdict output** — a suppressed criterion that is not
   printed is indistinguishable from one that passed.
4. **The metric is re-armed only after the label fix lands** — the two config flags that already exist,
   `--labels-v2` (⇒ `route_target_v21` / `route_from_future_v3`, coverage 27 % → 80.4 %) and
   `--v2-route-from-vision --route-vis-weight 0.3`. Once a run trains with real route supervision, this
   secondary becomes meaningful again and returns to the kill set.

⚠️ **Why this rule exists.** Without it, the 30 k gate would **kill a healthy arm for a label bug** — and
the fix is two flags we already ship. That is the most expensive possible way to be wrong: a
correctly-behaving model discarded on a measurement that could never have said anything else.

⚠️ **The general rule, so this is not a one-off exemption.** Before a gate renders, every secondary must
answer: *could this metric have taken a different value if the model were better?* If not — because a
label is degenerate, an emitter is missing, a field is declared void, or the harness never exercises the
path — it is **VOID and report-only**. Precedent in this program: `reached_at_step: 450` (declared void by
the registry, yet published in this very protocol for four days, §4b) and the `decorr`-never-on window that
made the entire v3enc gate un-interpretable.

### 0.8 🔴 PRIVILEGED-INPUT PRIMARIES — a goal-oracle number is not a deployed capability

> **A gate metric computed with inputs derived from the ego's own FUTURE measures a different
> quantity than the deployed capability. It may be reported — it may NOT be quoted as the model's
> performance, and it may not alone carry a CONTINUE/KILL verdict.**

**Standing instance, and it is time-critical — the flagship-v4 30 k gate.**

`eval_flagship_v4.py:322` feeds **`route`, `route_graded`, `vt_band`, `vt_speed`** minted from the ego's
own future poses. **MODE B structurally requires them** — `eval_flagship_v4.py:140-141` states the head's
`_goal_inputs` reads them off the batch — so **every MODE B number, including the 30 k gate's primary,
is a goal-oracle number.** This violates `V4_FLAGSHIP_DESIGN.md:558-560`, the design's own rule 3.

**The magnitude is measured, not assumed.** The S3 blind-conditioning firewall found that adding
`route`/`route_graded` **alone** lifts a **no-image** baseline from QWK **0.1128 → 0.3381**, paired
**Δ +0.2254 [+0.0631, +0.3674], CI-separated.** A model with no pixels gets a large, separated lift from
these channels. That is the size of the privilege being handed to the arm under test.

**Adjudication for the 30 k gate:**

1. **Render the gate.** Do not block on this — but **stamp the primary `goal_provenance: ORACLE`**
   (`stack/scripts/goal_provenance.py` exists for exactly this) and print it in the verdict.
2. **The verdict may not be worded as a deployed-capability claim.** Admissible: *"MODE B, goal-oracle
   inputs, ADE@2s = X."* **Inadmissible:** *"the flagship achieves X."*
3. **Weight the horizon-honest co-primary** (`corridor_departure_rate` @ pre-registered K, §0.7) — it is
   closed-loop and does not consume the oracle channels.
4. **Do not compare this number to REF-C's 0.4728/0.4714.** Those were collected with the route input
   **never exercised** (`follow_constant`). Oracle-vs-withheld is not a comparison; it is two different
   experiments.
5. **The un-privileged number is a follow-up, not a blocker** — a produced-goal path (the model's own
   `route_head`, two-pass, as now implemented for REF-C via `nav_mode`) does not yet exist for v4. Build
   it, then re-render.

**Also affected and now disclosed in place** (values untouched — the registry owner corrects them):
**flagship v1.6's headline ADE@2s 0.4375 m** (`eval_flagship_v16.py:135-143`) and **every flagship v1.5
anchored-fan number** (`eval_flagship_v15.py:92-103`).

⚠️ **The general rule.** This is the same class as the retracted *"CEM search = 0.132, a 4.5× headroom"*
(C6, 07-24) — a **hindsight-privileged** arm quoted as though it were deployable. **Before quoting any
number, ask what the model was given that it will not have at deployment, and name it.** A privileged
input does not invalidate a measurement; it changes what the measurement is *of*.

---

## 1. The rule

**No run is killed or continued except at a pre-registered gate step, on a held-out metric measured at
a pre-registered horizon, against a threshold written down before launch.**

Concretely, before any multi-GPU-day launch:

```bash
python stack/scripts/run_gate.py register --run <arm> --gate-step <S> \
    --primary-metric ade_0_2s --primary-threshold <T> \
    --co-primary-threshold <C> --co-primary-horizon-K 185 \
    --co-primary-junction-threshold <J> \
    --secondary "<mechanism>>=<v>" ... \
    --reference-run <ref> --reference-log <ref train_log.jsonl> \
    --compare-metric g_op_fwd_ade_m --tau 1.5 \
    --lever-family <family> --restarts-used <n> \
    --card "Project Steering/Gates/<arm>.card.json"
```

and at step `S`:

```bash
# emit the co-primary panel (and the exact --corridor-json argument)
python stack/scripts/gate_emitters.py corridor --windows results/windows_<arm>.pt \
    --out-corridor results/corridor_<arm>.json

python stack/scripts/run_gate.py check --card "Project Steering/Gates/<arm>.card.json" \
    --log <run log> --eval-json <held-out taniteval result JSON> \
    --corridor-json results/corridor_<arm>.json \
    --secondary-value <name>=<measured> ...
```

`check` refuses to return a verdict before the registered step (`NOT_YET`), refuses to decide from a
train-log slope (`BLOCKED`), and — since 2026-07-26 — renders **`INCOMPLETE`** rather than deciding on
`ade_0_2s` alone when a registered co-primary was not measured. All three refusals are the point.

## 2. What counts as evidence

| Element | Prescription | Enforced by |
|---|---|---|
| **Co-primary** ⭐ | **`corridor_departure_rate` @ a pre-registered K**, from `taniteval.corridor` on held-out windows, junction stratum reported separately. K explicit, 20 < K ≤ 190. | `check` → `INCOMPLETE` without `--corridor-json`; `validate_horizon_K` |
| **Primary (demoted)** | **Held-out** val ADE@2s at an archived milestone (D-032 archives 5k/15k/20k/30k). Never a train-log slope. **Since 2026-07-26 a diagnostic**: reported with the verdict, does not adjudicate where a co-primary exists. | `check` exits `BLOCKED` without `--eval-json`; `primary_role` |
| **Horizon** ⭐ | Every verdict **names its horizon (K and seconds) and its n** (windows / episodes / junction windows) and its surface. | `check` writes the `horizon` block; `horizon_honest` flag |
| **Comparative** | **Matched-step ratio** r(s) = M_new(s)/M_ref(s) at *identical* s, plus the assumption-free "the reference reached the new run's current value at step X". No power law, no extrapolation. A **diagnostic** — since 2026-07-26 it can no longer abort a verdict when the two logs share no metric. | `run_gate.py ratio`; `matched_step_ratio.available` |
| **Interval** | Bootstrap CI on every slope and every ratio. Decision-grade single-arm and paired intervals come from the **episode-cluster bootstrap** (`taniteval/ci.py`), not the deprecated overlapping-holdout SE — which is both too narrow *and* point-biased. | `taniteval/ci.py`, `SlopeFit`, `_corridor_stratum_value` |
| **Budget** | Compare at equal **GPU-hours**, not equal steps. `check` prints s/step and steps/GPU-hour for both arms. | `gpu_hours()` / `s_per_step()` |
| **Multiplicity** | **One** pre-registered gate step. Not "look at every milestone and decide". | card + `NOT_YET` |
| **Anti-regress** | **Two** restarts per lever family. A third failure **refutes the lever family**; it does not license more schedule tuning. | `restart_cap`, verdict `REFUTE_LEVER_FAMILY` |

## 3. The exponent is a diagnostic, and it cannot be quoted bare

An exponent may be logged. It may never decide a restart. Whenever one is printed it carries its
**fit window, R², n and bootstrap CI** — there is no code path in `run_gate.py` that returns a bare
float (`SlopeFit.exponent` raises below the R² floor; `SlopeFit.render()` always carries provenance).

- **R² ≥ 0.80** required before an exponent may be quoted at all. Below it: "power law unsupported",
  fall back to the ratio.
- **Extrapolation capped at 2×** the fitted range. `SlopeFit.project()` refuses beyond it.
- All arms compared must be **refit over identical step windows**.

### Why — measured on the actual logs, 2026-07-20

Same metric (`g_op_fwd_ade_m`), same two runs, different windows:

| fit window | flagship v1 | v3enc |
|---|---|---|
| 50–5350 | −0.421 (R² 0.566) [−0.507, −0.324] | −0.387 (R² 0.579) [−0.461, −0.332] |
| 1500–5350 | −0.663 (R² 0.375) [−0.851, −0.483] | −0.505 (R² 0.238) [−0.715, −0.316] |
| 2000–5350 | — | −0.738 (R² 0.299) [−1.022, −0.461] |
| 3000–5350 | — | −0.621 (R² 0.091) [−1.125, −0.138] |
| **1500–7500** | **−0.839** (R² 0.541) [−0.990, −0.689] | n/a |
| 50–29999 | −0.836 (R² **0.853**) | n/a |
| 1500–29999 | −1.021 (R² **0.877**) | n/a |

Three things follow, and they are the whole case:

1. **v1's famous −0.84 is the 1500–7500 window at R² 0.541.** It is below the floor and should never
   have been quotable. Its agreement with the full-run −0.836 is a coincidence of that window.
2. **On matched windows v1 and v3enc are statistically indistinguishable** (50–5350: −0.421 vs
   −0.387, CIs overlapping heavily). The claim "v3enc sits at the level that killed v2, far from
   v1's −0.84" is an artefact of comparing *unmatched windows*.
3. **The only R² ≥ 0.8 fits are full-run fits**, which by construction cannot gate an early decision.
   At every early window the power law does not describe the data — on *either* run.

## 4. What the corrected gate says about v3enc (2026-07-20, step 5350)

**VERDICT: `NOT_YET`.** The old gate had no basis to judge. The pre-registered 10k criterion stands
in its place, and it is `Project Steering/Gates/flagship-v3enc.card.json`:

- **primary** held-out ADE@2s ≤ 2.5 m at step 10 000
- **secondary** encoder speed-probe R² ≥ 0.55 · high-speed long overshoot ≤ 8 m
- (thresholds are the val-side gates already written down at
  `2026-07-19-flagshipv2-6k-diagnostic.md:196-199` — this protocol adopts them, it does not invent them)

Comparative diagnostics at step 5350, for information only:

- matched-step ratio v3enc/v1 = **1.834**, CI [1.759, 1.909], first 0.757 → last 2.224 (**widening**).
  v2's was 1.51 → **4.33**: v3enc is roughly **half as far behind** as v2 was.
- ~~v1 reached v3enc's current `g_op_fwd_ade_m` (0.422) at step **450** → ~**12×** slower. v2's figure
  was ~30×.~~ 🔴 **RETRACTED / VOID — do not quote (swept 2026-07-25).** `MODEL_REGISTRY.md:424-426`
  declares this exact field void: the gate JSONs (`Gates/flagship-v3enc-gate-2026-07-20.json`,
  `…-10k-2026-07-21.json`) *"retain `"reached_at_step": 450` under `"smoothing": "3-point rolling
  median"` — **that field is void in both**; they were not re-run (no GPU)."* A retracted number
  survived here for four days **in a BINDING protocol**, i.e. in the document that decides
  restart/continue — the highest-consequence place for a void figure to sit. The admissible
  comparative reading is the **matched-step ratio** in the bullet above (1.834, CI [1.759, 1.909]),
  which is exactly what §"Never quote a learning-curve exponent bare" prescribes when the curve-fit
  quantity is inadmissible. *(Found by `tools/registry_lint.py` on the first run after its scan set
  was extended beyond `MODEL_REGISTRY.md` — this file was previously never linted.)*
- **Budget:** v3enc 10.22 s/step vs v1 10.89 s/step → 352 vs 331 steps/GPU-hour. Equal-step is
  ~equal-cost here (within 6 %); the concern that budget normalization would flatter the new arm does
  **not** bite for this pair. v3enc has spent **15.2 GPU-hours** to step 5350.
- restart budget: **1 / 2** for lever family `encoder-grounding`. One more failure exhausts it and
  refutes the family.

**Do not kill v3enc on its pre-5k exponent.** There is no admissible exponent to kill it on.

## 4b. Re-gate status under the horizon correction (2026-07-26)

Both standing verdicts were rendered on the horizon-blind primary, so both were re-rendered. The
**estimator** blast-radius sweep (`…/incoming/2026-07-25-jack-blast-radius/`) found **no verdict flips**
— but it did **not** test the horizon change, which is a different and larger effect.

| arm | historical verdict | re-render on its ORIGINAL card | re-gate on a horizon-honest card |
|---|---|---|---|
| **flagship-v3enc** @ 10k | `RESTART` (2026-07-21) | **`RESTART`** — unchanged, now stamped `horizon_honest: false` | **`INCOMPLETE`** — co-primary not measured |
| **flagship-v4.1** @ 10k | `INCOMPLETE`, substantively FAIL (2026-07-23) | **`INCOMPLETE`** — unchanged, stamped `horizon_honest: false` | **`INCOMPLETE`** — co-primary not measured |

**Neither verdict is currently admissible as horizon-honest, and neither is currently overturned.**
Artifacts: `…/incoming/2026-07-26-gate-primary-change/regate/`.

**The blocker is MEASURED, and it is data, not code.** `0 of 30` committed `windows_*.pt` dumps carry
`pred_dense`/`gt_dense`; all 30 are the 4-waypoint sparse view (`wp_steps = [5, 10, 15, 20]`). And even
once the dense keys land, the **open-loop** dense surface caps at **K = 20 (2.0 s)** — the blind horizon
itself. **A K ≥ 100 corridor read requires a closed-loop rollout (E1a's surface) on GPU.**

> ⚠️ **v3enc carries a second, independent defect and a corridor number will not repair it.** Its whole
> 0–9,999 gate window ran with **`decorr` NEVER ON** (`train_flagship4b.py:92` sets `decorr_w = 0.0` for
> `step < 10000`; `MODEL_REGISTRY.md` §"the finding that reframes the failure"). The gate measured the
> arm *before* the staged lever under test was applied, so the 10k gate does not test D-A7's hypothesis
> **at any horizon**. Re-running the corridor metric on `ckpt_step10000.pt` would settle only "is this
> checkpoint corridor-safe", never "is `encoder-grounding` refuted". Settling the latter needs a gate
> step **after** decorr engages (≥ ~12–15k) or a restart with the lever on from step 0 — a PI decision,
> not an eval.

## 5. Relationship to the interval estimator

The comparative ratio's CI is a bootstrap over log points and is labelled `log_point_bootstrap` —
**diagnostic-grade**. Decision-grade intervals come from `taniteval/ci.py`
(`episode_cluster_bootstrap`, `paired_episode_cluster_bootstrap`) on held-out windows. Milestone
evals must therefore persist `windows_<key>.pt`, or the paired comparison the gate wants is not
computable afterwards.

**Amended 2026-07-26:** persisting `windows_<key>.pt` is necessary but **not sufficient**. The
co-primary additionally needs the **dense path** (`pred_dense`/`gt_dense`, persisted by
`rollout.collect` since 2026-07-25) for an open-loop read, and a **closed-loop rollout** for any
horizon beyond K = 20. A milestone eval that persists only the 4 waypoints leaves the gate's own
primary uncomputable after the fact — which is exactly the state all 30 committed dumps are in.

---

## 6. Backward compatibility, and its boundary

A card written before 2026-07-26 carries no co-primary. `check` still renders its verdict **exactly as
before** — the committed dry-run pin (`taniteval/results/v1_g1_dryrun_gate_FIXED.json`:
`kill_adjudicated: 8, report_only: 5, verdict_from_kill_only: CONTINUE`) reproduces, and is pinned by
`stack/tests/test_run_gate_corridor.py::test_the_committed_dryrun_split_survives_the_change`.

The boundary is deliberate and narrow:

1. such a verdict is stamped **`horizon_honest: false`** and carries the reason, so it can never again
   be quoted as if it saw the whole failure;
2. `register` will **not write a new blind card** without `--no-co-primary "<reason>"`;
3. the demoted `ade_0_2s` still adjudicates on those old cards **only because nothing else can** —
   `primary_role` says so explicitly in the JSON.

## 7. Standing rules this protocol now carries

1. **Never quote a learning-curve exponent bare** — window, R², n, CI, or it is inadmissible (§3).
2. **Never quote an interval without its estimator** — the decision-grade one is the episode-cluster
   bootstrap; the paired form for any two arms on shared windows (§0.5).
3. **⭐ Never quote a gate verdict without its horizon and n** — K, seconds, windows, episodes,
   junction windows, surface. A verdict that does not name its horizon is not admissible, for the same
   reason a bare exponent is not: *the number is a function of a choice that was not disclosed* (§0).

*Why rule 3 is written down here and not merely implemented: this file has already carried an
inadmissible number in a binding protocol. `reached_at_step: 450` — declared **VOID** by the registry —
sat in §4 for **four days** and was struck only on 2026-07-25, by a lint whose scan set had just been
widened past `MODEL_REGISTRY.md`. A protocol that decides GPU-days is the highest-consequence place for
an undisclosed choice to hide.*
